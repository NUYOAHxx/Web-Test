#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
moveit_bridge.py
================
MoveIt 2 (ROS 2) 实时机械臂模型与空间重力矩桥接服务 (纯动态读取，无内置假模型)

使用步骤 (极简两步)：
-------------------
1. 启动您的 MoveIt 2 demo 节点 (以您的机械臂配置为准)：
   ros2 launch <your_robot>_moveit_config demo.launch.py

2. 运行本脚本：
   python3 moveit_bridge.py

然后在 Web 浏览器打开 index.html 点击【连接 MoveIt 2】即可在 3D 场景中加载正在运行的机器人模型！
*注：本脚本完全动态订阅 /robot_description 和 /joint_states，不预设或内置任何假模型。*
"""

import sys
import json
import time
import math
import socket
import struct
import base64
import hashlib
import threading
import xml.etree.ElementTree as ET
import http.server
import urllib.parse
import re
import os

# 检查 ROS 2 运行库环境
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, qos_profile_sensor_data
    from std_msgs.msg import String
    from sensor_msgs.msg import JointState
    HAS_ROS2 = True
except ImportError:
    HAS_ROS2 = False

# Pinocchio 动力学库 (环境核心)
try:
    import pinocchio as pin
    HAS_PINOCCHIO = True
except ImportError:
    HAS_PINOCCHIO = False
    print("[提示] 未检测到 pinocchio 动力学库。请在已安装好的 conda 环境中运行: conda activate Deeplearning")


# ==========================================================
# 1. 动态 URDF 动力学与正向运动学解析器 (纯原生 Python)
# ==========================================================
class URDFKinematicParser:
    def __init__(self):
        self.robot_name = "未获取"
        self.links = {}
        self.joints = {}
        self.root_link = None
        self.total_mass = 0.0
        self.is_parsed = False
        self.pin_model = None
        self.pin_data = None
        self.raw_urdf_xml = None

    @staticmethod
    def rpy_to_mat(r, p, y):
        cr, sr = math.cos(r), math.sin(r)
        cp, sp = math.cos(p), math.sin(p)
        cy, sy = math.cos(y), math.sin(y)
        return [
            [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
            [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
            [-sp,   cp*sr,            cp*cr           ]
        ]

    @staticmethod
    def axis_angle_to_mat(axis, angle):
        x, y, z = axis
        norm = math.sqrt(x*x + y*y + z*z)
        if norm < 1e-6:
            return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        x, y, z = x/norm, y/norm, z/norm
        c, s = math.cos(angle), math.sin(angle)
        C = 1.0 - c
        return [
            [x*x*C + c,   x*y*C - z*s, x*z*C + y*s],
            [y*x*C + z*s, y*y*C + c,   y*z*C - x*s],
            [z*x*C - y*s, z*y*C + x*s, z*z*C + c  ]
        ]

    @staticmethod
    def multiply_transforms(T_a, T_b):
        R_a, p_a = T_a
        R_b, p_b = T_b
        R_new = [
            [sum(R_a[i][k] * R_b[k][j] for k in range(3)) for j in range(3)]
            for i in range(3)
        ]
        p_new = [
            sum(R_a[i][k] * p_b[k] for k in range(3)) + p_a[i]
            for i in range(3)
        ]
        return (R_new, p_new)

    @staticmethod
    def mat_to_quat(m):
        tr = m[0][0] + m[1][1] + m[2][2]
        if tr > 0:
            s = math.sqrt(max(0.0, tr + 1.0)) * 2
            s = max(s, 1e-9)
            qw = 0.25 * s
            qx = (m[2][1] - m[1][2]) / s
            qy = (m[0][2] - m[2][0]) / s
            qz = (m[1][0] - m[0][1]) / s
        elif (m[0][0] > m[1][1]) and (m[0][0] > m[2][2]):
            s = math.sqrt(max(0.0, 1.0 + m[0][0] - m[1][1] - m[2][2])) * 2
            s = max(s, 1e-9)
            qw = (m[2][1] - m[1][2]) / s
            qx = 0.25 * s
            qy = (m[0][1] + m[1][0]) / s
            qz = (m[0][2] + m[2][0]) / s
        elif m[1][1] > m[2][2]:
            s = math.sqrt(max(0.0, 1.0 + m[1][1] - m[0][0] - m[2][2])) * 2
            s = max(s, 1e-9)
            qw = (m[0][2] - m[2][0]) / s
            qx = (m[0][1] + m[1][0]) / s
            qy = 0.25 * s
            qz = (m[1][2] + m[2][1]) / s
        else:
            s = math.sqrt(max(0.0, 1.0 + m[2][2] - m[0][0] - m[1][1])) * 2
            s = max(s, 1e-9)
            qw = (m[1][0] - m[0][1]) / s
            qx = (m[0][2] + m[2][0]) / s
            qy = (m[1][2] + m[2][1]) / s
            qz = 0.25 * s
        return [round(qx, 5), round(qy, 5), round(qz, 5), round(qw, 5)]

    def parse_urdf_string(self, xml_string):
        """完全从 ROS 2 的 /robot_description 或 URDF 字符串解析实际机器人模型与动力学参数"""
        try:
            self.raw_urdf_xml = xml_string
            root = ET.fromstring(xml_string)
            self.robot_name = root.attrib.get('name', 'moveit_robot')
            self.links.clear()
            self.joints.clear()
            total_m = 0.0

            # 1. 尝试构建 Pinocchio 动力学模型
            self.pin_model = None
            self.pin_data = None
            if HAS_PINOCCHIO:
                try:
                    self.pin_model = pin.buildModelFromXML(xml_string)
                    self.pin_data = self.pin_model.createData()
                    print(f"[Pinocchio 动力学] 成功加载动力学模型: 关节数={self.pin_model.njoints}, 广义坐标数={self.pin_model.nq}")
                except Exception as e:
                    self.pin_model = None
                    print(f"[Pinocchio] 模型解析提示: {e}")

            for link_el in root.findall('link'):
                lname = link_el.attrib.get('name')
                if not lname: continue
                mass = 0.0
                com_xyz = [0.0, 0.0, 0.0]
                com_rpy = [0.0, 0.0, 0.0]
                inertial_el = link_el.find('inertial')
                if inertial_el is not None:
                    mass_el = inertial_el.find('mass')
                    if mass_el is not None:
                        try:
                            mass = float(mass_el.attrib.get('value', 0.0))
                        except (ValueError, TypeError):
                            mass = 0.0
                    origin_el = inertial_el.find('origin')
                    if origin_el is not None:
                        if 'xyz' in origin_el.attrib:
                            try:
                                com_xyz = [float(v) for v in origin_el.attrib['xyz'].split()]
                            except (ValueError, TypeError):
                                com_xyz = [0.0, 0.0, 0.0]
                        if 'rpy' in origin_el.attrib:
                            try:
                                com_rpy = [float(v) for v in origin_el.attrib['rpy'].split()]
                            except (ValueError, TypeError):
                                com_rpy = [0.0, 0.0, 0.0]

                visuals = []
                for vis_el in link_el.findall('visual'):
                    v_orig_xyz = [0.0, 0.0, 0.0]
                    v_orig_rpy = [0.0, 0.0, 0.0]
                    v_origin_el = vis_el.find('origin')
                    if v_origin_el is not None:
                        if 'xyz' in v_origin_el.attrib:
                            v_orig_xyz = [float(v) for v in v_origin_el.attrib['xyz'].split()]
                        if 'rpy' in v_origin_el.attrib:
                            v_orig_rpy = [float(v) for v in v_origin_el.attrib['rpy'].split()]
                    geo_el = vis_el.find('geometry')
                    if geo_el is not None:
                        for child in geo_el:
                            tag = child.tag.lower()
                            if tag == 'box' and 'size' in child.attrib:
                                visuals.append({
                                    'type': 'box',
                                    'size': [float(v) for v in child.attrib['size'].split()],
                                    'origin_xyz': v_orig_xyz,
                                    'origin_rpy': v_orig_rpy
                                })
                            elif tag == 'cylinder':
                                visuals.append({
                                    'type': 'cylinder',
                                    'radius': float(child.attrib.get('radius', 0.05)),
                                    'length': float(child.attrib.get('length', 0.1)),
                                    'origin_xyz': v_orig_xyz,
                                    'origin_rpy': v_orig_rpy
                                })
                            elif tag == 'sphere':
                                visuals.append({
                                    'type': 'sphere',
                                    'radius': float(child.attrib.get('radius', 0.05)),
                                    'origin_xyz': v_orig_xyz,
                                    'origin_rpy': v_orig_rpy
                                })
                            elif tag == 'mesh' and 'filename' in child.attrib:
                                scale = [1.0, 1.0, 1.0]
                                if 'scale' in child.attrib:
                                    scale = [float(v) for v in child.attrib['scale'].split()]
                                visuals.append({
                                    'type': 'mesh',
                                    'filename': child.attrib['filename'],
                                    'scale': scale,
                                    'origin_xyz': v_orig_xyz,
                                    'origin_rpy': v_orig_rpy
                                })

                self.links[lname] = {
                    'mass': mass,
                    'com_local': com_xyz,
                    'com_rpy': com_rpy,
                    'visuals': visuals
                }
                total_m += mass

            child_links = set()
            for joint_el in root.findall('joint'):
                jname = joint_el.attrib.get('name')
                jtype = joint_el.attrib.get('type', 'revolute')
                pel = joint_el.find('parent')
                cel = joint_el.find('child')
                if pel is None or cel is None: continue
                pname = pel.attrib.get('link')
                cname = cel.attrib.get('link')
                if not pname or not cname: continue
                child_links.add(cname)

                orig_xyz = [0.0, 0.0, 0.0]
                orig_rpy = [0.0, 0.0, 0.0]
                orig_el = joint_el.find('origin')
                if orig_el is not None:
                    if 'xyz' in orig_el.attrib:
                        orig_xyz = [float(v) for v in orig_el.attrib['xyz'].split()]
                    if 'rpy' in orig_el.attrib:
                        orig_rpy = [float(v) for v in orig_el.attrib['rpy'].split()]

                axis = [0.0, 0.0, 1.0]
                axis_el = joint_el.find('axis')
                if axis_el is not None and 'xyz' in axis_el.attrib:
                    axis = [float(v) for v in axis_el.attrib['xyz'].split()]

                effort_limit = 0.0
                limit_el = joint_el.find('limit')
                if limit_el is not None and 'effort' in limit_el.attrib:
                    try:
                        effort_limit = float(limit_el.attrib['effort'])
                    except ValueError:
                        effort_limit = 0.0

                self.joints[jname] = {
                    'parent': pname,
                    'child': cname,
                    'type': jtype,
                    'origin_xyz': orig_xyz,
                    'origin_rpy': orig_rpy,
                    'axis': axis,
                    'effort_limit': effort_limit
                }

            all_links = set(self.links.keys())
            root_candidates = all_links - child_links
            self.root_link = next(iter(root_candidates)) if root_candidates else (list(self.links.keys())[0] if self.links else None)

            # 若 URDF 未填写惯性质量，按每连杆保底 1.0kg 估算总重
            if total_m <= 0.01:
                total_m = 0.0
                for lname, ldata in self.links.items():
                    ldata['mass'] = 1.0
                    total_m += 1.0

            self.total_mass = total_m
            self.is_parsed = True
            return True
        except Exception as e:
            print(f"[URDF] 解析异常: {e}")
            return False

    def compute_forward_kinematics(self, joint_angles):
        """基于真实 URDF 与关节角实时正解每根连杆位姿、骨架段与整机重心"""
        if not self.is_parsed or not self.root_link:
            return None

        link_transforms = {}
        identity = ([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], [0.0, 0.0, 0.0])
        link_transforms[self.root_link] = identity

        queue = [self.root_link]
        visited = {self.root_link}

        while queue:
            curr_link = queue.pop(0)
            T_curr = link_transforms[curr_link]

            for jname, j in self.joints.items():
                if j['parent'] == curr_link and j['child'] not in visited:
                    child = j['child']
                    q = joint_angles.get(jname, 0.0)

                    R_orig = self.rpy_to_mat(*j['origin_rpy'])
                    T_orig = (R_orig, j['origin_xyz'])

                    if j['type'] in ('revolute', 'continuous'):
                        R_motion = self.axis_angle_to_mat(j['axis'], q)
                        T_motion = (R_motion, [0.0, 0.0, 0.0])
                    elif j['type'] == 'prismatic':
                        T_motion = ([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], [j['axis'][k] * q for k in range(3)])
                    else:
                        T_motion = ([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], [0.0, 0.0, 0.0])

                    T_rel = self.multiply_transforms(T_orig, T_motion)
                    T_child = self.multiply_transforms(T_curr, T_rel)
                    link_transforms[child] = T_child
                    visited.add(child)
                    queue.append(child)

        # 基于 Pinocchio 原生动力学精确解算全机质心与总质量
        if HAS_PINOCCHIO and self.pin_model is not None and self.pin_data is not None:
            q_pin = pin.neutral(self.pin_model)
            for jname, val in joint_angles.items():
                if self.pin_model.existJointName(jname):
                    jid = self.pin_model.getJointId(jname)
                    jobj = self.pin_model.joints[jid]
                    idx_q = jobj.idx_q
                    if jobj.nq == 1:
                        q_pin[idx_q] = float(val)
                    elif jobj.nq == 2 and jobj.nv == 1:
                        q_pin[idx_q] = math.cos(float(val))
                        q_pin[idx_q + 1] = math.sin(float(val))
            pin.forwardKinematics(self.pin_model, self.pin_data, q_pin)
            pin.updateFramePlacements(self.pin_model, self.pin_data)
            com_vec = pin.centerOfMass(self.pin_model, self.pin_data, q_pin)
            com = [float(com_vec[0]), float(com_vec[1]), float(com_vec[2])]
            total_m = sum(m.mass for m in self.pin_model.inertias)
        else:
            com_weighted = [0.0, 0.0, 0.0]
            total_m = 0.0
            for lname, ldata in self.links.items():
                if lname in link_transforms:
                    m = ldata['mass']
                    T_l = link_transforms[lname]
                    local_com = ldata['com_local']
                    R_l, p_l = T_l
                    global_com = [
                        sum(R_l[i][k] * local_com[k] for k in range(3)) + p_l[i]
                        for i in range(3)
                    ]
                    for i in range(3):
                        com_weighted[i] += m * global_com[i]
                    total_m += m
            com = [com_weighted[i] / total_m for i in range(3)] if total_m > 0 else [0.0, 0.0, 0.0]

        # 提取真实关节连线段 (Joint Segments, 仅保留长度 > 2cm 的有效肢体段，过滤零位虚关节)
        segments = []
        for idx, (jname, j) in enumerate(self.joints.items()):
            p_link = j['parent']
            c_link = j['child']
            if p_link in link_transforms and c_link in link_transforms:
                p1 = link_transforms[p_link][1]
                p2 = link_transforms[c_link][1]
                dx, dy, dz = p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]
                dist = math.sqrt(dx*dx + dy*dy + dz*dz)
                if dist > 0.02:
                    rad = max(0.015, min(0.04, dist * 0.12))
                    segments.append({
                        'joint': jname,
                        'parent': p_link,
                        'child': c_link,
                        'from': [round(v, 4) for v in p1],
                        'to': [round(v, 4) for v in p2],
                        'radius': round(rad, 4),
                        'length': round(dist, 4)
                    })

        # 提取 URDF 原生几何图元与 Mesh 网格，并解算全机空间包围盒高度
        visuals = []
        min_z = float('inf')
        max_z = float('-inf')

        for lname, T_l in link_transforms.items():
            pz = T_l[1][2]
            if pz < min_z: min_z = pz
            if pz > max_z: max_z = pz

        for lname, ldata in self.links.items():
            if lname in link_transforms:
                T_l = link_transforms[lname]
                for v in ldata.get('visuals', []):
                    T_v_orig = (self.rpy_to_mat(*v['origin_rpy']), v['origin_xyz'])
                    T_v_world = self.multiply_transforms(T_l, T_v_orig)
                    pz = T_v_world[1][2]
                    if pz < min_z: min_z = pz
                    if pz > max_z: max_z = pz

                    v_item = {
                        'type': v['type'],
                        'link': lname,
                        'pos': [round(x, 4) for x in T_v_world[1]],
                        'rot': self.mat_to_quat(T_v_world[0])
                    }
                    if v['type'] == 'mesh':
                        v_item['filename'] = v['filename']
                        v_item['scale'] = v.get('scale', [1.0, 1.0, 1.0])
                    elif v['type'] == 'box':
                        v_item['size'] = v['size']
                    elif v['type'] == 'cylinder':
                        v_item['radius'] = v['radius']
                        v_item['length'] = v['length']
                    elif v['type'] == 'sphere':
                        v_item['radius'] = v['radius']
                    visuals.append(v_item)

        ground_offset = round(-min_z, 4) if min_z < -0.05 else 0.0

        transforms_dict = {}
        for lname, T_l in link_transforms.items():
            transforms_dict[lname] = {
                'pos': [round(x, 4) for x in T_l[1]],
                'rot': self.mat_to_quat(T_l[0])
            }

        # 提取每个连杆的独立质心世界坐标 (Link CoM)
        link_coms = []
        for lname, ldata in self.links.items():
            if lname in link_transforms:
                m = ldata.get('mass', 0.0)
                if m > 1e-4:
                    T_l = link_transforms[lname]
                    local_com = ldata.get('com_local', [0.0, 0.0, 0.0])
                    R_l, p_l = T_l
                    global_com = [
                        sum(R_l[i][k] * local_com[k] for k in range(3)) + p_l[i]
                        for i in range(3)
                    ]
                    link_coms.append({
                        'link': lname,
                        'mass': round(m, 4),
                        'pos': [round(x, 4) for x in global_com]
                    })

        # 动力学与关节静载负荷谱解算 (Pinocchio 优先)
        joint_loads, dynamics = self.compute_joint_loads(joint_angles, link_transforms, com, total_m, g=9.81)

        return {
            "type": "moveit_robot_data",
            "robot_name": self.robot_name,
            "mass": round(total_m, 2),
            "links_count": len(self.links),
            "pivot": {"x": 0.0, "y": 0.0, "z": 0.0},
            "com": {
                "x": round(com[0], 4),
                "y": round(com[1], 4),
                "z": round(com[2], 4)
            },
            "link_coms": link_coms,
            "min_z": round(min_z, 4) if min_z != float('inf') else 0.0,
            "max_z": round(max_z, 4) if max_z != float('-inf') else 1.0,
            "ground_offset": ground_offset,
            "segments": segments,
            "link_transforms": transforms_dict,
            "visuals": visuals,
            "joint_positions": {k: round(v, 4) for k, v in joint_angles.items()},
            "joint_loads": joint_loads,
            "dynamics": dynamics,
            "timestamp": time.time()
        }

    def compute_joint_loads(self, joint_angles, link_transforms, com, total_m, g=9.81):
        """基于 Pinocchio 动力学精确计算各关节广义重力矩负荷谱、利用率及整机动力学指标"""
        joint_loads = []
        engine_name = "Pinocchio 动力学"

        if HAS_PINOCCHIO and self.pin_model is not None and self.pin_data is not None:
            try:
                q = pin.neutral(self.pin_model)
                for jname, val in joint_angles.items():
                    if self.pin_model.existJointName(jname):
                        jid = self.pin_model.getJointId(jname)
                        joint_obj = self.pin_model.joints[jid]
                        idx_q = joint_obj.idx_q
                        if joint_obj.nq == 1:
                            q[idx_q] = float(val)
                        elif joint_obj.nq == 2 and joint_obj.nv == 1:
                            q[idx_q] = math.cos(float(val))
                            q[idx_q + 1] = math.sin(float(val))

                tau_vec = pin.computeGeneralizedGravity(self.pin_model, self.pin_data, q)

                for i in range(1, self.pin_model.njoints):
                    jname = self.pin_model.names[i]
                    joint_obj = self.pin_model.joints[i]
                    idx_v = joint_obj.idx_v
                    if joint_obj.nv == 1 and idx_v < len(tau_vec):
                        tau_val = float(tau_vec[idx_v])
                        limit = float(self.pin_model.effortLimit[idx_v]) if idx_v < len(self.pin_model.effortLimit) else 0.0
                        if limit <= 0.0 and jname in self.joints:
                            limit = float(self.joints[jname].get('effort_limit', 0.0))

                        pct = (abs(tau_val) / limit * 100.0) if limit > 0 else 0.0
                        status = "safe" if pct < 50.0 else ("warning" if pct < 80.0 else "danger")
                        joint_loads.append({
                            "name": jname,
                            "torque": round(tau_val, 3),
                            "limit": round(limit, 1),
                            "load_pct": round(pct, 1),
                            "status": status
                        })
            except Exception as e:
                print(f"[Pinocchio] 动力学解算异常: {e}")

        sum_tau_sq = sum(jl['torque'] ** 2 for jl in joint_loads)
        tau_rms = math.sqrt(sum_tau_sq / len(joint_loads)) if joint_loads else 0.0
        peak_joint = max(joint_loads, key=lambda x: x['load_pct']) if joint_loads else None
        peak_torque = max(joint_loads, key=lambda x: abs(x['torque'])) if joint_loads else None

        base_mx = -total_m * g * com[1]
        base_my = total_m * g * com[0]
        base_mag = math.sqrt(base_mx**2 + base_my**2)

        dynamics = {
            "engine": engine_name,
            "total_mass": round(total_m, 2),
            "gravity_constant": g,
            "base_wrench": {
                "fz": round(total_m * g, 2),
                "mx": round(base_mx, 2),
                "my": round(base_my, 2),
                "mag": round(base_mag, 2)
            },
            "sum_tau_sq": round(sum_tau_sq, 3),
            "tau_rms": round(tau_rms, 3),
            "peak_joint": peak_joint,
            "peak_torque": peak_torque,
            "stability_margin": round(math.sqrt(com[0]**2 + com[1]**2), 4)
        }

        return joint_loads, dynamics


# ==========================================================
# 2. 纯原生 Python WebSocket 服务端 (RFC 6455 零依赖实现)
# ==========================================================
class StandaloneWebSocketServer:
    def __init__(self, host='0.0.0.0', port=9090):
        self.host = host
        self.port = port
        self.clients = set()
        self.lock = threading.Lock()
        self.running = True
        self.last_payload = None

    def start(self):
        t = threading.Thread(target=self._server_loop, daemon=True)
        t.start()
        print(f"[WebSocket] 服务已就绪: ws://127.0.0.1:{self.port} (等待 MoveIt 2 连接)")

    def _server_loop(self):
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            server_sock.bind((self.host, self.port))
            server_sock.listen(5)
        except Exception as e:
            print(f"[WebSocket] 端口 {self.port} 绑定失败: {e}")
            return

        while self.running:
            try:
                client_sock, addr = server_sock.accept()
                threading.Thread(target=self._handle_client, args=(client_sock, addr), daemon=True).start()
            except Exception:
                break

    def _handle_client(self, sock, addr):
        try:
            req = sock.recv(2048).decode('utf-8', errors='ignore')
            key = None
            for line in req.split('\r\n'):
                if line.lower().startswith('sec-websocket-key:'):
                    key = line.split(':', 1)[1].strip()
                    break
            if not key:
                sock.close()
                return

            guid = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
            sha1 = hashlib.sha1((key + guid).encode('utf-8')).digest()
            accept_val = base64.b64encode(sha1).decode('utf-8')
            resp = (
                "HTTP/1.1 101 Switching Protocols\r\n"
                "Upgrade: websocket\r\n"
                "Connection: Upgrade\r\n"
                f"Sec-WebSocket-Accept: {accept_val}\r\n\r\n"
            )
            sock.sendall(resp.encode('utf-8'))

            with self.lock:
                self.clients.add(sock)
                # 新客户端连接后，若已有解析好的真实机器人数据，立即发送
                if self.last_payload:
                    packet = json.dumps({
                        "op": "publish",
                        "topic": "/moveit_gravity_data",
                        "msg": { "data": json.dumps(self.last_payload) }
                    })
                    try:
                        sock.sendall(self._encode_frame(packet))
                    except Exception:
                        pass

            print(f"[WebSocket] 网页可视化前端已连接 ({addr[0]}:{addr[1]})")

            while self.running:
                data = sock.recv(1024)
                if not data:
                    break
        except Exception:
            pass
        finally:
            with self.lock:
                self.clients.discard(sock)
            try:
                sock.close()
            except Exception:
                pass
            print(f"[WebSocket] 网页前端已断开 ({addr[0]})")

    def broadcast_json(self, payload):
        self.last_payload = payload
        with self.lock:
            if not self.clients:
                return
            packet = json.dumps({
                "op": "publish",
                "topic": "/moveit_gravity_data",
                "msg": { "data": json.dumps(payload) }
            })
            frame = self._encode_frame(packet)
            dead_clients = []
            for c in self.clients:
                try:
                    c.sendall(frame)
                except Exception:
                    dead_clients.append(c)
            for dc in dead_clients:
                self.clients.discard(dc)

    def _encode_frame(self, text):
        data = text.encode('utf-8')
        length = len(data)
        if length < 126:
            header = struct.pack('!BB', 0x81, length)
        elif length <= 0xFFFF:
            header = struct.pack('!BBH', 0x81, 126, length)
        else:
            header = struct.pack('!BBQ', 0x81, 127, length)
        return header + data


# ==========================================================
# 3. 本地 3D Mesh 网格轻量 HTTP 伺服服务 (零依赖)
# ==========================================================
class MeshHTTPHandler(http.server.BaseHTTPRequestHandler):
    search_dirs = [os.path.abspath('.')]

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path.startswith('/mesh'):
            qs = urllib.parse.parse_qs(parsed.query)
            fname = qs.get('file', [''])[0] or qs.get('path', [''])[0]
            clean = fname.replace('\\', '/')
            clean = re.sub(r'^package:\/\/[^\/]+\/', '', clean)
            clean = re.sub(r'^file:\/\/', '', clean)
            base = os.path.basename(clean)

            target_path = None
            clean_norm = os.path.normpath(clean).lstrip('/\\')
            for sdir in MeshHTTPHandler.search_dirs:
                sdir_real = os.path.realpath(sdir)
                cands = [
                    os.path.join(sdir, clean_norm),
                    os.path.join(sdir, 'meshes', base),
                    os.path.join(sdir, 'visual', base),
                    os.path.join(sdir, base)
                ]
                for c in cands:
                    c_real = os.path.realpath(c)
                    if (c_real.startswith(sdir_real + os.sep) or c_real == sdir_real) and os.path.isfile(c_real):
                        target_path = c_real
                        break
                if target_path:
                    break

            if target_path and os.path.isfile(target_path):
                try:
                    with open(target_path, 'rb') as f:
                        data = f.read()
                    self.send_response(200)
                    if target_path.lower().endswith('.stl'):
                        self.send_header('Content-Type', 'model/stl')
                    elif target_path.lower().endswith('.dae'):
                        self.send_header('Content-Type', 'model/vnd.collada+xml')
                    else:
                        self.send_header('Content-Type', 'application/octet-stream')
                    self.send_header('Content-Length', str(len(data)))
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(data)
                    return
                except Exception:
                    pass

            self.send_response(404)
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(b"Mesh not found")
        else:
            self.send_response(200)
            self.send_header('Content-Type', 'text/plain; charset=utf-8')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(b"MoveIt 2 Mesh Server Ready")

    def log_message(self, format, *args):
        pass  # 静默运行避免干扰终端日志


def start_mesh_http_server(port=9091, search_dirs=None):
    if search_dirs:
        MeshHTTPHandler.search_dirs = [os.path.abspath(d) for d in search_dirs if os.path.isdir(d)]
    try:
        server = http.server.ThreadingHTTPServer(('0.0.0.0', port), MeshHTTPHandler)
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        print(f"[Mesh 服务] 已在 http://127.0.0.1:{port}/mesh 启动 (自动供前端加载 3D 几何文件)")
        return server
    except Exception as e:
        print(f"[Mesh 服务] 端口 {port} 绑定提示: {e}")
        return None


# ==========================================================
# 4. ROS 2 监听节点
# ==========================================================
if HAS_ROS2:
    class MoveIt2LiveBridgeNode(Node):
        def __init__(self, ws_server):
            super().__init__('moveit2_gravity_bridge')
            self.ws_server = ws_server
            self.parser = URDFKinematicParser()
            self.current_joints = {}

            # 双 QoS 订阅：捕获 demo.launch.py 中发布的所有 /robot_description
            latched_qos = QoSProfile(
                depth=1,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
                reliability=ReliabilityPolicy.RELIABLE
            )
            volatile_qos = QoSProfile(
                depth=1,
                durability=DurabilityPolicy.VOLATILE,
                reliability=ReliabilityPolicy.RELIABLE
            )
            self.sub_desc_latched = self.create_subscription(
                String,
                '/robot_description',
                self.robot_description_callback,
                latched_qos
            )
            self.sub_desc_volatile = self.create_subscription(
                String,
                '/robot_description',
                self.robot_description_callback,
                volatile_qos
            )

            # 使用 qos_profile_sensor_data (BEST_EFFORT) 监听实时关节角
            self.sub_joints = self.create_subscription(
                JointState,
                '/joint_states',
                self.joint_states_callback,
                qos_profile_sensor_data
            )

            # 20 Hz 刷新率
            self.timer = self.create_timer(0.05, self.timer_callback)

            print("=" * 65)
            print("MoveIt 2 实时模型与数据桥接节点已就绪")
            print("   正在自动监听: /robot_description (机器人模型)")
            print("   正在自动监听: /joint_states (实时关节角)")
            print("=" * 65)

        def robot_description_callback(self, msg: String):
            if not self.parser.is_parsed:
                if self.parser.parse_urdf_string(msg.data):
                    print(f"成功从 demo.launch 接收实际机器人模型: {self.parser.robot_name} (连杆: {len(self.parser.links)} 节, 总质量: {self.parser.total_mass:.2f} kg)")

        def joint_states_callback(self, msg: JointState):
            for name, pos in zip(msg.name, msg.position):
                self.current_joints[name] = pos

        def timer_callback(self):
            if not self.parser.is_parsed:
                return  # 尚未收到真实 MoveIt URDF 前不广播
            payload = self.parser.compute_forward_kinematics(self.current_joints)
            if payload:
                self.ws_server.broadcast_json(payload)


def main():
    import argparse
    import os

    parser_arg = argparse.ArgumentParser(description="MoveIt 2 & URDF 通用机器人重力矩桥接服务")
    parser_arg.add_argument('--urdf', type=str, default=None, help="本地 URDF 文件路径 (支持任意机器人模型)")
    parser_arg.add_argument('--port', type=int, default=9090, help="WebSocket 端口 (默认: 9090)")
    parser_arg.add_argument('--mesh-port', type=int, default=9091, help="Mesh 几何文件 HTTP 端口 (默认: 9091)")
    parser_arg.add_argument('--test', action='store_true', help="测试模式: 解析后打印并启动服务")
    args, _ = parser_arg.parse_known_args()

    search_dirs = [os.path.abspath('.')]
    if args.urdf:
        urdf_dir = os.path.dirname(os.path.abspath(args.urdf))
        if urdf_dir not in search_dirs:
            search_dirs.insert(0, urdf_dir)

    # 启动 3D Mesh 伺服服务
    start_mesh_http_server(port=args.mesh_port, search_dirs=search_dirs)

    ws_server = StandaloneWebSocketServer(port=args.port)
    ws_server.start()

    # 如果用户指定了本地 URDF 模型文件，直接加载并广播
    target_urdf = args.urdf
    if target_urdf and os.path.exists(target_urdf):
        print(f"正在加载通用机器人 URDF 模型: {target_urdf}")
        with open(target_urdf, 'r', encoding='utf-8') as f:
            urdf_str = f.read()
        file_parser = URDFKinematicParser()
        if file_parser.parse_urdf_string(urdf_str):
            print(f"成功加载机器人模型: {file_parser.robot_name} (连杆: {len(file_parser.links)} 节, 总质量: {file_parser.total_mass:.2f} kg)")
            payload = file_parser.compute_forward_kinematics({})
            ws_server.broadcast_json(payload)
            print(f"机器人 {file_parser.robot_name} 3D 模型已在 ws://127.0.0.1:{args.port} 广播！打开 index.html 点击【连接 MoveIt 2】即可一键呈现！")

            if not HAS_ROS2 or args.test:
                try:
                    while True:
                        time.sleep(1)
                        ws_server.broadcast_json(payload)
                except KeyboardInterrupt:
                    pass
                return

    if not HAS_ROS2:
        print("[提示] 当前主机未检测到 ROS 2 环境。请通过 --urdf <路径> 加载本地模型，或在运行 MoveIt 2 (demo.launch.py) 的环境运行此脚本。")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        return

    rclpy.init()
    node = MoveIt2LiveBridgeNode(ws_server)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
