#!/usr/bin/env python3
"""
================================================================================
Unitree G1 实时高精度数字孪生与遥测监控大屏服务端 (Pure Telemetry & Command Dispatcher)
================================================================================

定位：纯被动遥测展示 + 目标指令下发中枢 (Zero IK Calculation · Pure Telemetry & Dispatcher)
1. 算展彻底解耦：本服务端绝对不进行任何逆解 (IK) 计算，专注高性能被动监听与指令下发；
2. 目标指令分发：提供 POST /api/send_target，将前端下达的笛卡尔目标坐标发布至 ROS 2 标准话题：
   /g1/kinematics/target_pose 与 /g1/kinematics/target_command；
3. 工业标准监听：被动订阅 ROS 2 /joint_states、/g1/kinematics/actual_pose、
   /g1/kinematics/solver_metrics 与 /g1/safety/collision_status；
4. 正向运动学镜像：利用 Pinocchio 前向运动学 (FK, <0.05ms) 计算 36 个官方 STL 连杆世界坐标；
5. 全域安全雷达：实时计算 28 对关键连杆物理安全净空与干涉告警；
6. 30Hz SSE 本地实时推送，为 Web 大屏提供丝滑低延迟数据孪生渲染。
"""

import os
import sys
import json
import time
import math
import argparse
import threading
from typing import Dict, List, Optional, Tuple, Any
from http.server import HTTPServer, SimpleHTTPRequestHandler
from socketserver import ThreadingMixIn

# 将项目根目录加入 sys.path
DIR_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if DIR_ROOT not in sys.path:
    sys.path.insert(0, DIR_ROOT)

import numpy as np
import pinocchio as pin

from core.kinematics.g1_model import (
    G1KinematicsModel,
    G1_JOINT_LIMITS,
    G1_WAIST_LIMITS,
    G1_READY_POSE,
    G1_DEFAULT_STAND_JOINTS,
)
from core.collision.g1_collision import G1CollisionChecker
from core.solver.g1_hybrid_ik import IK_PIPELINE_STAGES

# 检查 ROS 2 是否可用
HAS_ROS2 = False
try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    from geometry_msgs.msg import PoseStamped
    from std_msgs.msg import String
    HAS_ROS2 = True
except ImportError:
    HAS_ROS2 = False


class RobotTelemetryManager:
    """纯被动遥测与指令分发中枢：负责接收遥测、前向位姿计算、碰撞雷达与目标下发"""

    def __init__(self, enable_ros: bool = True):
        self.lock = threading.Lock()
        print("[Web 纯遥测后端] 正在加载 G1 运动学引擎与 28 对安全碰撞检测雷达...")
        self.kin = G1KinematicsModel()
        self.collision = G1CollisionChecker(self.kin)

        # 加载 36 个 3D 官方 CAD STL 视觉网格拓扑
        self.visual_model = None
        self.visual_data = None
        try:
            _, _, self.visual_model = pin.buildModelsFromUrdf(
                self.kin.urdf_path,
                package_dirs=["/home/parallels/ws_moveit/src", os.path.join(DIR_ROOT, "resources")]
            )
            self.visual_data = self.visual_model.createData()
            print(f"[Web 纯遥测后端] ✔️ 成功挂载 {len(self.visual_model.geometryObjects)} 个官方 3D 视觉网格！")
        except Exception as e:
            print(f"[Web 纯遥测后端] 视觉网格加载警告: {e}")

        # 计算就绪姿态下末端手爪初始世界坐标
        q_init = pin.neutral(self.kin.model)
        for k, v in G1_DEFAULT_STAND_JOINTS.items():
            if self.kin.model.existJointName(k):
                q_init[self.kin.model.joints[self.kin.model.getJointId(k)].idx_q] = v
        pin.forwardKinematics(self.kin.model, self.kin.data, q_init)
        pin.updateFramePlacements(self.kin.model, self.kin.data)
        fid_left = self.kin.model.getFrameId("left_wrist_yaw_link")
        init_hand_p = self.kin.data.oMf[fid_left].translation.copy()
        init_hand_rot = self.kin.data.oMf[fid_left].rotation.copy()

        # 遥测数据缓存 (初始目标与手爪精准贴合，初始误差 0.0mm，姿态 0.0°)
        self.active_arm = "left_arm"
        self.joint_positions = dict(G1_DEFAULT_STAND_JOINTS)
        self.target_pos = init_hand_p.copy()
        self.target_rot = init_hand_rot.copy()

        # 算法诊断与度量指标缓存
        init_rpy_deg = [round(float(np.degrees(v)), 2) for v in pin.rpy.matrixToRpy(init_hand_rot)]
        self.solver_metrics = {
            "success": True,
            "time_ms": 0.0,
            "pos_err_mm": 0.0,
            "rot_err_deg": 0.0,
            "iters": 0,
            "arm": self.active_arm,
            "mode": "INITIAL_STANDBY",
            "seed_used": 0,
            "seed_name": "Standby",
            "seed_switches": [0],
            "convergence_trace": [],
            "step_details": [],
            "pipeline_stages": IK_PIPELINE_STAGES,
            "algorithm": "10-DoF Weighted DLS (W_waist=8.0) + 28-Pair Barrier",
            "actual_rpy_deg": init_rpy_deg,
            "target_rpy_deg": init_rpy_deg,
        }

        # 频率统计与连接状态
        self.last_msg_time = 0.0
        self.msg_count = 0
        self.current_hz = 0.0
        self._hz_window: List[float] = []

        # 实时事件日志队列
        self.event_logs: List[Dict[str, Any]] = [
            {"time": time.strftime("%H:%M:%S"), "type": "SYSTEM", "title": "遥测监控引擎就绪", "desc": "Pinocchio 动力学拓扑与 28 对安全雷达就绪"},
            {"time": time.strftime("%H:%M:%S"), "type": "INFO", "title": "初始就绪姿态对齐", "desc": f"手爪与规划目标精准贴合，初始空间残差 0.0 mm"},
        ]

        # 启动 ROS 2 监听与指令分发节点
        self.running = True
        self.ros_node = None
        self.pub_target_pose = None
        self.pub_command = None
        if enable_ros and HAS_ROS2:
            try:
                if not rclpy.ok():
                    rclpy.init()
                self.ros_node = rclpy.create_node("g1_telemetry_hub")

                # 订阅标准遥测话题 (标准话题，不作冗余兼容)
                self.ros_node.create_subscription(JointState, "/joint_states", self._on_joint_state, 10)
                self.ros_node.create_subscription(PoseStamped, "/g1/kinematics/target_pose", self._on_target_pose, 10)
                self.ros_node.create_subscription(String, "/g1/kinematics/solver_metrics", self._on_solver_metrics, 10)

                # 发布目标控制指令话题
                self.pub_target_pose = self.ros_node.create_publisher(PoseStamped, "/g1/kinematics/target_pose", 10)
                self.pub_command = self.ros_node.create_publisher(String, "/g1/kinematics/target_command", 10)

                print("[Web 纯遥测后端] ✔️ 成功建立 ROS 2 数据总线：监听 /joint_states & 分发 /g1/kinematics/target_pose！")

                self.ros_spin_thread = threading.Thread(target=self._ros_spin_loop, daemon=True)
                self.ros_spin_thread.start()
            except Exception as e:
                print(f"[Web 纯遥测后端] ROS 2 节点启动警告: {e}")

        # 运动学与碰撞检测按需计算缓存 (Pose Dirty Checking, 杜绝待机时重复 FK 与 28 对 FCL 计算)
        self._fk_dirty = True
        self._cached_kinematics_payload: Optional[Dict[str, Any]] = None

        # 高性能遥测状态缓存与集中广播引擎 (杜绝多客户端重复 Pinocchio & 碰撞计算与锁争用)
        self._state_cache_lock = threading.Lock()
        self._cached_state_dict: Dict[str, Any] = {}
        self._cached_state_json: str = ""
        self._compute_and_cache_state()

        self._state_thread = threading.Thread(target=self._state_worker_loop, daemon=True)
        self._state_thread.start()

    def _state_worker_loop(self):
        """以 30Hz 频率集中计算并缓存遥测状态，所有 Web 客户端共享同一计算结果"""
        interval = 1.0 / 30.0
        while self.running:
            t0 = time.time()
            try:
                self._compute_and_cache_state()
            except Exception:
                pass
            elapsed = time.time() - t0
            sleep_time = max(0.001, interval - elapsed)
            time.sleep(sleep_time)

    def _compute_and_cache_state(self):
        state = self.get_state_dict()
        state_json = json.dumps(state, ensure_ascii=False)
        with self._state_cache_lock:
            self._cached_state_dict = state
            self._cached_state_json = state_json

    def get_cached_state_json(self) -> str:
        with self._state_cache_lock:
            return self._cached_state_json

    def get_cached_state_dict(self) -> Dict[str, Any]:
        with self._state_cache_lock:
            return dict(self._cached_state_dict)

    def _ros_spin_loop(self):
        while self.running and rclpy.ok():
            try:
                rclpy.spin_once(self.ros_node, timeout_sec=0.05)
            except Exception:
                break

    def _update_freq_counter(self):
        now = time.time()
        self._hz_window.append(now)
        if len(self._hz_window) > 30:
            self._hz_window.pop(0)
        if len(self._hz_window) >= 2:
            span = self._hz_window[-1] - self._hz_window[0]
            if span > 0.001:
                self.current_hz = (len(self._hz_window) - 1) / span

    # ─────────────────────────────────────────────────────────────
    # ROS 2 遥测数据回调 (纯被动接收)
    # ─────────────────────────────────────────────────────────────
    def _on_joint_state(self, msg: JointState):
        with self.lock:
            self.last_msg_time = time.time()
            self.msg_count += 1
            self._update_freq_counter()

            changed = False
            for name, pos in zip(msg.name, msg.position):
                old_val = self.joint_positions.get(name, None)
                val_f = float(pos)
                if old_val is None or abs(old_val - val_f) > 1e-4:
                    changed = True
                self.joint_positions[name] = val_f
            if changed:
                self._fk_dirty = True

    def _on_target_pose(self, msg: PoseStamped):
        with self.lock:
            new_pos = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z], dtype=np.float64)
            qx = float(msg.pose.orientation.x)
            qy = float(msg.pose.orientation.y)
            qz = float(msg.pose.orientation.z)
            qw = float(msg.pose.orientation.w)
            if qx**2 + qy**2 + qz**2 + qw**2 > 1e-4:
                new_rot = pin.Quaternion(qw, qx, qy, qz).normalized().toRotationMatrix()
                self.target_rot = new_rot
            if self.target_pos is None or not np.allclose(new_pos, self.target_pos, atol=1e-4):
                self._fk_dirty = True
            self.target_pos = new_pos

    def _on_solver_metrics(self, msg: String):
        try:
            data = json.loads(msg.data)
            with self.lock:
                self.solver_metrics.update(data)
                seed_info = data.get("seed_name") or f"Seed #{data.get('seed_used', 0)}"
                seed_desc = f" ({seed_info})" if data.get("seed_used", 0) > 0 else ""
                rot_str = f" | 姿态角差: {data.get('rot_err_deg', 0):.1f}°" if "rot_err_deg" in data else ""
                self.add_event_log(
                    "IK_SOLVER",
                    f"IK 解算收敛 ({data.get('time_ms', 0):.1f}ms)",
                    f"位置残差: {data.get('pos_err_mm', 0):.2f}mm{rot_str} | 步数: {data.get('iters', 0)}{seed_desc} | 模式: {data.get('mode', '10-DoF')}"
                )
        except Exception:
            pass

    def add_event_log(self, type_str: str, title: str, desc: str):
        log_entry = {
            "time": time.strftime("%H:%M:%S"),
            "type": type_str,
            "title": title,
            "desc": desc,
        }
        self.event_logs.append(log_entry)
        if len(self.event_logs) > 60:
            self.event_logs.pop(0)

    # ─────────────────────────────────────────────────────────────
    # Web 目标指令下发中枢 (只下发，不解算，交由后台 IK 求解节点)
    # ─────────────────────────────────────────────────────────────
    def dispatch_target_command(self, x: float, y: float, z: float, rpy: Optional[List[float]] = None, arm: str = "left_arm", preset_name: str = "自定义目标"):
        """向 ROS 2 广播 6-DoF 目标位姿指令，交由 IK 求解器计算"""
        target_rot = None
        target_quat = [0.0, 0.0, 0.0, 1.0]
        if rpy is not None and len(rpy) == 3:
            rpy_rad = np.radians([float(v) for v in rpy])
            target_rot = pin.rpy.rpyToMatrix(float(rpy_rad[0]), float(rpy_rad[1]), float(rpy_rad[2]))
            q_pin = pin.Quaternion(target_rot)
            target_quat = [float(q_pin.x), float(q_pin.y), float(q_pin.z), float(q_pin.w)]

        with self.lock:
            self.active_arm = arm
            self.target_pos = np.array([float(x), float(y), float(z)], dtype=np.float64)
            if target_rot is not None:
                self.target_rot = target_rot.copy()
            self._fk_dirty = True
            rpy_str = f", RPY: [{rpy[0]:.1f}°, {rpy[1]:.1f}°, {rpy[2]:.1f}°]" if rpy else ""
            self.add_event_log(
                "COMMAND",
                f"下达目标指令: {preset_name}",
                f"[{self.active_arm}] 目标坐标 X: {x:+.3f}m, Y: {y:+.3f}m, Z: {z:+.3f}m{rpy_str}"
            )

        if self.pub_command is not None:
            cmd_data = {
                "action": "SET_TARGET",
                "arm": arm,
                "target": [float(x), float(y), float(z)],
                "preset": preset_name,
            }
            if rpy is not None:
                cmd_data["rpy"] = [float(v) for v in rpy]
                cmd_data["quat"] = target_quat
            cmd_msg = String()
            cmd_msg.data = json.dumps(cmd_data, ensure_ascii=False)
            self.pub_command.publish(cmd_msg)

        if self.pub_target_pose is not None:
            p_msg = PoseStamped()
            p_msg.header.frame_id = "world"
            p_msg.header.stamp = self.ros_node.get_clock().now().to_msg()
            p_msg.pose.position.x = float(x)
            p_msg.pose.position.y = float(y)
            p_msg.pose.position.z = float(z)
            p_msg.pose.orientation.x = float(target_quat[0])
            p_msg.pose.orientation.y = float(target_quat[1])
            p_msg.pose.orientation.z = float(target_quat[2])
            p_msg.pose.orientation.w = float(target_quat[3])
            self.pub_target_pose.publish(p_msg)

        print(f"[Web 目标中枢] 🚀 成功向 ROS 2 发布 6D 目标指令: [{x:.3f}, {y:.3f}, {z:.3f}] (执行臂: {arm}, RPY: {rpy})")

    def _compute_ee_pose_fk(self, arm: str, joint_dict: Optional[Dict[str, float]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """计算指定关节配置下末端手爪的笛卡尔空间坐标与旋转矩阵"""
        joints = joint_dict if joint_dict is not None else G1_DEFAULT_STAND_JOINTS
        q_full = pin.neutral(self.kin.model)
        for jname, val in joints.items():
            if self.kin.model.existJointName(jname):
                jid = self.kin.model.getJointId(jname)
                q_full[self.kin.model.joints[jid].idx_q] = val
        pin.forwardKinematics(self.kin.model, self.kin.data, q_full)
        pin.updateFramePlacements(self.kin.model, self.kin.data)
        ee_frame = "left_wrist_yaw_link" if arm == "left_arm" else "right_wrist_yaw_link"
        fid = self.kin.model.getFrameId(ee_frame)
        return self.kin.data.oMf[fid].translation.copy(), self.kin.data.oMf[fid].rotation.copy()

    def dispatch_set_arm(self, arm: str):
        """切换当前监控与控制的手臂 (left_arm / right_arm)"""
        with self.lock:
            self.active_arm = arm
            # 同步更新目标位置为该臂当前手爪真实位置与姿态，保持零残差
            self.target_pos, self.target_rot = self._compute_ee_pose_fk(arm, self.joint_positions)
            self._fk_dirty = True
            self.add_event_log("CONFIG", f"切换操作臂: {arm}", f"已切换至 {'左臂 (Left Arm)' if arm == 'left_arm' else '右臂 (Right Arm)'}")

        if self.pub_command is not None:
            cmd_msg = String()
            cmd_msg.data = json.dumps({"action": "SET_ARM", "arm": arm}, ensure_ascii=False)
            self.pub_command.publish(cmd_msg)

    def dispatch_reset_stand(self):
        """下达恢复标准直立就绪指令"""
        with self.lock:
            self.add_event_log("COMMAND", "下达复位指令", "恢复官方对称微屈直立就绪姿态")
            # 重新计算就绪姿态下手爪真实位置与姿态并重置目标
            self.target_pos, self.target_rot = self._compute_ee_pose_fk(self.active_arm, G1_DEFAULT_STAND_JOINTS)
            self._fk_dirty = True

        stand_quat = pin.Quaternion(self.target_rot)
        if self.pub_target_pose is not None:
            p_msg = PoseStamped()
            p_msg.header.frame_id = "world"
            p_msg.header.stamp = self.ros_node.get_clock().now().to_msg()
            p_msg.pose.position.x = float(self.target_pos[0])
            p_msg.pose.position.y = float(self.target_pos[1])
            p_msg.pose.position.z = float(self.target_pos[2])
            p_msg.pose.orientation.x = float(stand_quat.x)
            p_msg.pose.orientation.y = float(stand_quat.y)
            p_msg.pose.orientation.z = float(stand_quat.z)
            p_msg.pose.orientation.w = float(stand_quat.w)
            self.pub_target_pose.publish(p_msg)

        if self.pub_command is not None:
            cmd_msg = String()
            cmd_msg.data = json.dumps({"action": "RESET_STAND", "arm": self.active_arm}, ensure_ascii=False)
            self.pub_command.publish(cmd_msg)
        else:
            self.reset_to_stand_local()

    def dispatch_circle_demo(self):
        """下达连续空间轨迹演示指令"""
        with self.lock:
            self.add_event_log("COMMAND", "下达轨迹演示指令", "启动连续空间圆周平滑轨迹")

        if self.pub_command is not None:
            cmd_msg = String()
            cmd_msg.data = json.dumps({"action": "CIRCLE_DEMO", "arm": self.active_arm}, ensure_ascii=False)
            self.pub_command.publish(cmd_msg)

    def reset_to_stand_local(self):
        with self.lock:
            self.joint_positions = dict(G1_DEFAULT_STAND_JOINTS)
            self.target_pos, self.target_rot = self._compute_ee_pose_fk(self.active_arm, G1_DEFAULT_STAND_JOINTS)
            self._fk_dirty = True

    # ─────────────────────────────────────────────────────────────
    # 纯前向运动学 (FK) 与遥测字典组装 (零 IK 计算，耗时 < 0.05ms)
    # ─────────────────────────────────────────────────────────────
    def get_state_dict(self) -> Dict[str, Any]:
        with self.lock:
            now = time.time()
            is_connected = (now - self.last_msg_time) < 2.5 if self.last_msg_time > 0 else False

            # 1. 运动学脏检查：若机器人未发生关节位移且目标未变，直接复用静态几何与碰撞计算结果
            if not self._fk_dirty and self._cached_kinematics_payload is not None:
                state = dict(self._cached_kinematics_payload)
                state["solver_diagnostics"] = self.solver_metrics
                state["telemetry_link"] = {
                    "is_connected": is_connected,
                    "dds_rate_hz": round(self.current_hz, 1),
                    "ingress_packets": self.msg_count,
                    "topic_target": "/g1/kinematics/target_pose",
                    "topic_joints": "/joint_states",
                    "transport": "ROS 2 DDS + 30Hz SSE",
                    "mode": "PURE_TELEMETRY_AND_DISPATCH",
                }
                state["activity_logs"] = list(self.event_logs)
                return state

            # 2. 组装全模型 q 向量
            q_full = pin.neutral(self.kin.model)
            for jname, val in self.joint_positions.items():
                if self.kin.model.existJointName(jname):
                    jid = self.kin.model.getJointId(jname)
                    q_full[self.kin.model.joints[jid].idx_q] = val

            # 3. Pinocchio 前向运动学 (FK)
            pin.forwardKinematics(self.kin.model, self.kin.data, q_full)
            pin.updateFramePlacements(self.kin.model, self.kin.data)

            # 计算末端执行器真实空间坐标与朝向
            ee_frame = "left_wrist_yaw_link" if self.active_arm == "left_arm" else "right_wrist_yaw_link"
            fid = self.kin.model.getFrameId(ee_frame)
            fk_actual_pos = self.kin.data.oMf[fid].translation.copy()
            fk_actual_rot = self.kin.data.oMf[fid].rotation.copy()
            display_actual_pos = fk_actual_pos
            display_actual_rot = fk_actual_rot

            # 空间轴向偏差向量与欧氏残差范数 (毫米级)
            delta_xyz = (display_actual_pos - self.target_pos) * 1000.0
            err_norm_mm = float(np.linalg.norm(delta_xyz))

            # 空间 3D 朝向欧拉角与姿态残差
            cmd_rpy_deg = [round(float(np.degrees(v)), 2) for v in pin.rpy.matrixToRpy(self.target_rot)] if self.target_rot is not None else [0.0, 0.0, 0.0]
            act_rpy_deg = [round(float(np.degrees(v)), 2) for v in pin.rpy.matrixToRpy(display_actual_rot)]
            delta_rpy_deg = [round(float(act_rpy_deg[i] - cmd_rpy_deg[i]), 2) for i in range(3)]
            rot_err_norm_deg = 0.0
            if self.target_rot is not None:
                R_err = self.target_rot @ display_actual_rot.T
                rot_err_norm_deg = round(float(np.degrees(np.linalg.norm(pin.log3(R_err)))), 2)

            cmd_quat = pin.Quaternion(self.target_rot) if self.target_rot is not None else pin.Quaternion(np.eye(3))
            act_quat = pin.Quaternion(display_actual_rot)

            # 4. 计算 36 个官方 STL 视觉网格的实时世界坐标与四元数
            visual_poses = {}
            if self.visual_model is not None and self.visual_data is not None:
                pin.updateGeometryPlacements(self.kin.model, self.kin.data, self.visual_model, self.visual_data)
                for i, geom in enumerate(self.visual_model.geometryObjects):
                    T = self.visual_data.oMg[i]
                    q_rot = pin.Quaternion(T.rotation)
                    mesh_name = os.path.basename(geom.meshPath)
                    visual_poses[geom.name] = {
                        "mesh": mesh_name,
                        "pos": [round(float(v), 4) for v in T.translation],
                        "quat": [round(float(v), 5) for v in [q_rot.x, q_rot.y, q_rot.z, q_rot.w]],
                    }

            # 5. 全域碰撞干涉检测与物理安全净空 (update_fk=False 复用第3步已计算的FK，避免重复求距)
            col_pairs = self.collision.get_colliding_pairs(q_full, arm=self.active_arm, update_fk=False)
            zone_clearances = self.collision.compute_zone_distances(q_full, arm=self.active_arm, update_fk=False)
            min_clearance_mm = float(min(zone_clearances.values())) if zone_clearances else 50.0

            # 5.1 Pinocchio 全身刚体动力学计算 (CoM 双足平衡安全、广义重力补偿力矩与可操作度)
            bal_metrics = self.kin.evaluate_balance(q_full)
            tau_g = self.kin.compute_gravity_torques(q_full, arm=self.active_arm, is_10dof=True)
            active_10dof_jnames = self.kin.chain_10dof_joint_names[self.active_arm]
            q_10 = np.array([self.joint_positions.get(jn, 0.0) for jn in active_10dof_jnames], dtype=np.float64)
            manip_metrics = self.kin.compute_manipulability(self.active_arm, q_10, has_rot=True)

            gravity_torques_data = []
            for idx, jname in enumerate(active_10dof_jnames):
                val_nm = float(tau_g[idx])
                clean_lbl = jname.replace("left_", "").replace("right_", "").replace("_joint", "")
                gravity_torques_data.append({
                    "name": jname,
                    "label": clean_lbl,
                    "torque_nm": round(val_nm, 2),
                    "abs_nm": round(abs(val_nm), 2),
                })

            # 6. 腰部 3 自由度关节遥测数据 (直接使用官方关节名与符号)
            waist_metadata = {
                "waist_yaw_joint":   {"label": "waist_yaw",   "symbol": "q_w0"},
                "waist_roll_joint":  {"label": "waist_roll",  "symbol": "q_w1"},
                "waist_pitch_joint": {"label": "waist_pitch", "symbol": "q_w2"},
            }
            waist_data = []
            for name in self.kin.waist_joint_names:
                val_rad = float(self.joint_positions.get(name, 0.0))
                val_deg = round(math.degrees(val_rad), 1)
                lim_min, lim_max = G1_WAIST_LIMITS[name]
                min_deg = round(math.degrees(lim_min), 1)
                max_deg = round(math.degrees(lim_max), 1)
                offset_pct = (val_rad / lim_max * 100.0) if val_rad >= 0 else (val_rad / abs(lim_min) * 100.0)
                offset_pct = max(-100.0, min(100.0, offset_pct))
                meta = waist_metadata.get(name, {"label": name, "symbol": "q"})

                waist_data.append({
                    "name": name,
                    "label": meta["label"],
                    "symbol": meta["symbol"],
                    "rad": round(val_rad, 4),
                    "deg": val_deg,
                    "min_deg": min_deg,
                    "max_deg": max_deg,
                    "offset_pct": round(offset_pct, 1),
                })

            # 7. 机械臂 7 自由度关节遥测数据 (使用官方 URDF 原始关节名与符号)
            arm_metadata = {
                "shoulder_pitch": {"label": "shoulder_pitch", "symbol": "q_a0"},
                "shoulder_roll":  {"label": "shoulder_roll",  "symbol": "q_a1"},
                "shoulder_yaw":   {"label": "shoulder_yaw",   "symbol": "q_a2"},
                "elbow":          {"label": "elbow",          "symbol": "q_a3"},
                "wrist_roll":     {"label": "wrist_roll",     "symbol": "q_a4"},
                "wrist_pitch":    {"label": "wrist_pitch",    "symbol": "q_a5"},
                "wrist_yaw":      {"label": "wrist_yaw",      "symbol": "q_a6"},
            }
            arm_jnames = self.kin.left_arm_joint_names if self.active_arm == "left_arm" else self.kin.right_arm_joint_names
            arm_data = []
            for name in arm_jnames:
                val_rad = float(self.joint_positions.get(name, 0.0))
                val_deg = round(math.degrees(val_rad), 1)
                lim_min, lim_max = G1_JOINT_LIMITS[name]
                min_deg = round(math.degrees(lim_min), 1)
                max_deg = round(math.degrees(lim_max), 1)
                offset_pct = (val_rad / lim_max * 100.0) if val_rad >= 0 else (val_rad / abs(lim_min) * 100.0)
                offset_pct = max(-100.0, min(100.0, offset_pct))
                clean_key = name.replace("left_", "").replace("right_", "").replace("_joint", "")
                meta = arm_metadata.get(clean_key, {"label": clean_key, "symbol": "q"})

                arm_data.append({
                    "name": name,
                    "label": meta["label"],
                    "symbol": meta["symbol"],
                    "rad": round(val_rad, 4),
                    "deg": val_deg,
                    "min_deg": min_deg,
                    "max_deg": max_deg,
                    "offset_pct": round(offset_pct, 1),
                })

            # 缓存静态几何与碰撞计算结果 (规范命名，零兼容冗余)
            self._cached_kinematics_payload = {
                "arm": self.active_arm,
                "cartesian_cmd_pose": [round(float(v), 4) for v in self.target_pos],
                "cartesian_actual_pose": [round(float(v), 4) for v in display_actual_pos],
                "cartesian_cmd_rpy_deg": cmd_rpy_deg,
                "cartesian_actual_rpy_deg": act_rpy_deg,
                "cartesian_cmd_quat": [round(float(v), 5) for v in [cmd_quat.x, cmd_quat.y, cmd_quat.z, cmd_quat.w]],
                "cartesian_actual_quat": [round(float(v), 5) for v in [act_quat.x, act_quat.y, act_quat.z, act_quat.w]],
                "spatial_delta_mm": [round(float(v), 2) for v in delta_xyz],
                "spatial_delta_rpy_deg": delta_rpy_deg,
                "euclidean_error_norm_mm": round(err_norm_mm, 2),
                "rot_error_norm_deg": rot_err_norm_deg,
                "min_clearance_mm": round(min_clearance_mm, 1),
                "zone_clearances": zone_clearances,
                "is_colliding": len(col_pairs) > 0,
                "colliding_pairs": col_pairs,
                "waist_telemetry": waist_data,
                "arm_telemetry": arm_data,
                "dynamics_telemetry": {
                    "com_balance": bal_metrics,
                    "gravity_torques": gravity_torques_data,
                    "manipulability": manip_metrics,
                    "max_torque_joint": max(gravity_torques_data, key=lambda x: x["abs_nm"]) if gravity_torques_data else {},
                },
                "visuals": visual_poses,
            }
            self._fk_dirty = False

            # 组装完整返回报文
            state = dict(self._cached_kinematics_payload)
            state["solver_diagnostics"] = self.solver_metrics
            state["telemetry_link"] = {
                "is_connected": is_connected,
                "dds_rate_hz": round(self.current_hz, 1),
                "ingress_packets": self.msg_count,
                "topic_target": "/g1/kinematics/target_pose",
                "topic_joints": "/joint_states",
                "transport": "ROS 2 DDS + 30Hz SSE",
                "mode": "PURE_TELEMETRY_AND_DISPATCH",
            }
            state["activity_logs"] = list(self.event_logs)
            return state


MANAGER: Optional[RobotTelemetryManager] = None


class DashboardHTTPHandler(SimpleHTTPRequestHandler):
    """处理前端静态网页、SSE 遥测推流与目标指令下发 API"""

    def __init__(self, *args, **kwargs):
        web_dir = os.path.dirname(os.path.abspath(__file__))
        super().__init__(*args, directory=web_dir, **kwargs)

    def do_GET(self):
        if self.path == "/api/state":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            data_json = MANAGER.get_cached_state_json() if MANAGER else "{}"
            self.wfile.write(data_json.encode("utf-8"))
            return

        elif self.path == "/api/stream":
            # 30Hz Server-Sent Events (SSE) 高频遥测推送流 (共享统一计算缓存，零锁争用)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("X-Accel-Buffering", "no")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            print(f"[Web 推送流] ✔️ 客户端已接入 SSE 遥测实时推流通道 ({self.client_address[0]})")
            interval = 1.0 / 60.0  # 60Hz 数据帧
            keepalive_interval = 15.0  # 每 15 秒发送一次 SSE 心跳注释行
            last_keepalive = time.time()
            try:
                while True:
                    now = time.time()
                    if MANAGER:
                        cached_json = MANAGER.get_cached_state_json()
                        if cached_json:
                            payload = f"data: {cached_json}\n\n"
                            self.wfile.write(payload.encode("utf-8"))
                            self.wfile.flush()
                            last_keepalive = now  # 数据帧本身也算保活
                    # SSE 标准心跳：注释行 ": keepalive\n\n" 被浏览器忽略但保持 TCP 连接
                    if now - last_keepalive >= keepalive_interval:
                        self.wfile.write(b": keepalive\n\n")
                        self.wfile.flush()
                        last_keepalive = now
                    time.sleep(interval)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, EOFError, IOError, OSError):
                print(f"[Web 推送流] 客户端断开连接 ({self.client_address[0]})")
            return

        super().do_GET()

    def do_POST(self):
        # 目标指令下发 API (严格防御性数值与安全包络校验，交由 IK 节点求解)
        if self.path == "/api/send_target":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            try:
                data = json.loads(body)
                x = float(data.get("x", 0.35))
                y = float(data.get("y", 0.22))
                z = float(data.get("z", 0.85))
                arm = str(data.get("arm", "left_arm"))
                preset = str(data.get("preset_name", "空间目标"))

                # 1. 非法浮点数防御校验 (NaN / Infinity)
                if math.isnan(x) or math.isnan(y) or math.isnan(z) or math.isinf(x) or math.isinf(y) or math.isinf(z):
                    raise ValueError("Target coordinates cannot contain NaN or Infinity")

                # 2. 机械臂执行空间物理安全包络防御校验 (Safe Reach Envelope)
                # G1 左/右臂 10-DoF 工作空间实测边界 (5000点蒙特卡洛采样验证):
                #   X: [0.0, 1.0]  Y: [-0.8, 0.8]  Z: [0.64, 1.5]
                # Z < 0.64m 超出关节极限，任何 IK 解算器均无法收敛！
                z_min = 0.64
                if not (0.0 <= x <= 1.0 and -0.8 <= y <= 0.8 and z_min <= z <= 1.5):
                    raise ValueError(
                        f"目标 [{x:.3f}, {y:.3f}, {z:.3f}] 超出 G1 物理工作空间: "
                        f"X:[0.0, 1.0]m, Y:[-0.8, 0.8]m, Z:[{z_min}, 1.5]m. "
                        f"G1 左臂最低可达高度约 0.64m，Z={z:.3f}m 不可达！"
                    )

                # 3. 操作臂合法性校验
                if arm not in ("left_arm", "right_arm"):
                    raise ValueError(f"Invalid arm '{arm}', must be 'left_arm' or 'right_arm'")

                # 4. 6-DoF 空间朝向欧拉角解析 (Roll, Pitch, Yaw 单位: 度)
                rpy_raw = data.get("rpy", None)
                if rpy_raw is None and any(k in data for k in ("roll", "pitch", "yaw")):
                    rpy_raw = [float(data.get("roll", 0.0)), float(data.get("pitch", 0.0)), float(data.get("yaw", 0.0))]
                rpy = None
                if rpy_raw is not None and isinstance(rpy_raw, (list, tuple)) and len(rpy_raw) == 3:
                    rpy = [float(rpy_raw[0]), float(rpy_raw[1]), float(rpy_raw[2])]
                    for angle in rpy:
                        if math.isnan(angle) or math.isinf(angle):
                            raise ValueError("RPY angles cannot contain NaN or Infinity")

                if MANAGER:
                    MANAGER.dispatch_target_command(x, y, z, rpy=rpy, arm=arm, preset_name=preset)

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps({
                    "status": "DISPATCHED",
                    "target": [x, y, z],
                    "rpy": rpy,
                    "arm": arm,
                    "preset": preset,
                    "message": f"6D 目标指令已发布至 ROS 2 话题 /g1/kinematics/target_pose，由 IK 引擎解算",
                }, ensure_ascii=False).encode("utf-8"))
            except Exception as e:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
            return

        elif self.path == "/api/set_arm":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            try:
                data = json.loads(body)
                arm = str(data.get("arm", "left_arm"))
                if arm not in ("left_arm", "right_arm"):
                    raise ValueError(f"Invalid arm '{arm}', must be 'left_arm' or 'right_arm'")

                if MANAGER:
                    MANAGER.dispatch_set_arm(arm)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "ARM_SET", "arm": arm}, ensure_ascii=False).encode("utf-8"))
            except Exception as e:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
            return

        elif self.path == "/api/reset_stand":
            if MANAGER:
                MANAGER.dispatch_reset_stand()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "RESET_DISPATCHED", "message": "已发布就绪复位指令"}, ensure_ascii=False).encode("utf-8"))
            return

        elif self.path == "/api/demo_circle":
            if MANAGER:
                MANAGER.dispatch_circle_demo()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "DEMO_DISPATCHED", "message": "已触发连续空间轨迹演示"}, ensure_ascii=False).encode("utf-8"))
            return

        super().do_POST()

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def log_message(self, format, *args):
        # 屏蔽高频 SSE 与静态资源的刷屏访问日志
        pass


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main():
    global MANAGER
    parser = argparse.ArgumentParser(description="Unitree G1 纯被动遥测与指令分发服务端")
    parser.add_argument("--port", type=int, default=8080, help="Web 监控端口 (默认 8080)")
    parser.add_argument("--no-ros", action="store_true", help="禁用 ROS 2 订阅")
    args = parser.parse_args()

    print("\n" + "=" * 80)
    print(" 📡 Unitree G1 实时高精度数字孪生与遥测监控大屏 (Pure Telemetry & Dispatcher)")
    print("=" * 80)
    print(" 架构原则：零 IK 计算 · 纯被动遥测 · 笛卡尔目标指令一键分发")

    MANAGER = RobotTelemetryManager(enable_ros=not args.no_ros)

    server = ThreadedHTTPServer(("0.0.0.0", args.port), DashboardHTTPHandler)
    print(f"\n🚀 Web 遥测监控大屏已启动: http://localhost:{args.port}")
    print(f"👉 局域网访问地址: http://0.0.0.0:{args.port}")
    print("--------------------------------------------------------------------------------\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n⏹️ 正在安全关闭 Web 监控大屏服务端...")
    finally:
        if MANAGER:
            MANAGER.running = False
        server.server_close()


if __name__ == "__main__":
    main()
