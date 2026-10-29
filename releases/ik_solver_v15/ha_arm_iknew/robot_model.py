#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""URDF 串联机器人运动学链模型及人形臂配置。"""

from collections import defaultdict, deque
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation as R

from ha_arm_ik.ik_common import (
    T_origin,
    T_prismatic,
    T_revolute,
    make_T,
    normalize,
    sew_angle_from_points,
)

# 臂角 Shoulder(肩) / Elbow(肘) / Wrist(腕) 关节的默认兜底名（与左臂一致）
S_JOINT = "LHand_Link2"
E_JOINT = "LHand_Link4"
W_JOINT = "LHand_Link6"
REF_DIR = np.array([0.0, 0.0, 1.0], dtype=float)
WAIST_JOINT_NAMES = ["Hip_Link1", "Body_Link"]

ARM_CONFIGS = {
    "left": {
        "base_link": "Body_Link",
        "ee_link": "left_l_dg_tcp",
        "s_joint": "LHand_Link2",
        "e_joint": "LHand_Link4",
        "w_joint": "LHand_Link6",
        "mount_xyz": np.array([0.0, 0.0, 0.5087]),
        "mount_rpy": np.array([0.0, 0.0, 0.0]),
        "q_ready": np.array([-0.523599, -0.785398, 0.0, -0.698132, 0.0, 1.570796, 0.0]),
    },
    "right": {
        "base_link": "Body_Link",
        "ee_link": "right_l_dg_tcp",
        "s_joint": "RHand_Link2",
        "e_joint": "RHand_Link4",
        "w_joint": "RHand_Link6",
        "mount_xyz": np.array([0.0, 0.0, 0.5087]),
        "mount_rpy": np.array([0.0, 0.0, 0.0]),
        # 左臂的镜像（真机约定）：奇数关节 1/3/5/7 反号，偶数关节 2/4/6 同号
        "q_ready": np.array([0.523599, -0.785398, 0.0, -0.698132, 0.0, 1.570796, 0.0]),
    },
}


def get_sew_joints(chain=None, s_joint=None, e_joint=None, w_joint=None):
    """获取肩、肘、腕关节点名称，未指定时按链属性或全局常量兜底。"""
    s = s_joint or getattr(chain, "s_joint", None) or S_JOINT
    e = e_joint or getattr(chain, "e_joint", None) or E_JOINT
    w = w_joint or getattr(chain, "w_joint", None) or W_JOINT
    return s, e, w


def parse_vec(s):
    """解析空格分隔的浮点数字符串为一维数组。"""
    return np.array([float(x) for x in s.split()], dtype=float)


def get_fixed_joint_origin(urdf_path, parent_link, child_link):
    """从 URDF 检索连接指定父子连杆的 fixed 关节原点 (xyz, rpy)。"""
    root = ET.parse(urdf_path).getroot()

    for joint in root.findall("joint"):
        if joint.attrib.get("type", "") != "fixed":
            continue

        parent_elem = joint.find("parent")
        child_elem = joint.find("child")
        if parent_elem is None or child_elem is None:
            continue

        parent = parent_elem.attrib["link"]
        child = child_elem.attrib["link"]
        if parent != parent_link or child != child_link:
            continue

        origin_elem = joint.find("origin")
        if origin_elem is None:
            return np.zeros(3), np.zeros(3)

        return (
            parse_vec(origin_elem.attrib.get("xyz", "0 0 0")),
            parse_vec(origin_elem.attrib.get("rpy", "0 0 0")),
        )

    return None


def get_urdf_root_link(urdf_path):
    """检索 URDF 拓扑树中唯一的根连杆。"""
    root = ET.parse(urdf_path).getroot()

    links = {
        link.attrib.get("name")
        for link in root.findall("link")
        if link.attrib.get("name")
    }

    child_links = {
        joint.find("child").attrib["link"]
        for joint in root.findall("joint")
        if joint.find("child") is not None
    }

    roots = sorted(list(links - child_links))
    if len(roots) != 1:
        raise RuntimeError(f"Expected exactly one URDF root link, got {roots}")

    return roots[0]


def find_chain(urdf_path, base_link, ee_link):
    """使用 BFS 从 URDF 中提取 base_link 到 ee_link 的串联关节链。"""
    root = ET.parse(urdf_path).getroot()

    children = defaultdict(list)
    joint_by_edge = {}

    for joint in root.findall("joint"):
        name = joint.attrib["name"]
        joint_type = joint.attrib.get("type", "")
        parent = joint.find("parent").attrib["link"]
        child = joint.find("child").attrib["link"]

        origin_elem = joint.find("origin")
        xyz = (
            parse_vec(origin_elem.attrib.get("xyz", "0 0 0"))
            if origin_elem is not None
            else np.zeros(3)
        )
        rpy = (
            parse_vec(origin_elem.attrib.get("rpy", "0 0 0"))
            if origin_elem is not None
            else np.zeros(3)
        )

        axis_elem = joint.find("axis")
        axis = (
            parse_vec(axis_elem.attrib.get("xyz", "0 0 1"))
            if axis_elem is not None
            else np.array([0.0, 0.0, 1.0])
        )

        limit_elem = joint.find("limit")
        if limit_elem is not None:
            lower = float(limit_elem.attrib.get("lower", "-inf"))
            upper = float(limit_elem.attrib.get("upper", "inf"))
        else:
            lower = -np.inf
            upper = np.inf

        joint_data = {
            "name": name,
            "type": joint_type,
            "parent": parent,
            "child": child,
            "xyz": xyz,
            "rpy": rpy,
            "axis": axis,
            "lower": lower,
            "upper": upper,
        }
        children[parent].append(child)
        joint_by_edge[(parent, child)] = joint_data

    queue = deque([(base_link, [])])
    visited = set()

    while queue:
        link, path = queue.popleft()
        if link == ee_link:
            return path
        if link in visited:
            continue

        visited.add(link)
        for child in children[link]:
            queue.append((child, path + [joint_by_edge[(link, child)]]))

    raise RuntimeError(f"Cannot find chain from {base_link} to {ee_link}")


class SerialChain:
    """URDF 串联机器人运动学链。"""

    def __init__(
        self,
        urdf_path,
        base_link,
        ee_link,
        s_joint=None,
        e_joint=None,
        w_joint=None,
    ):
        self.urdf_path = urdf_path
        self.base_link = base_link
        self.ee_link = ee_link
        self.s_joint = s_joint
        self.e_joint = e_joint
        self.w_joint = w_joint
        self.joints = find_chain(urdf_path, base_link, ee_link)
        self.active_joints = [
            joint
            for joint in self.joints
            if joint["type"] in ["revolute", "continuous", "prismatic"]
        ]

        if len(self.active_joints) not in (7, 9):
            print(
                "WARNING: active joint count is not 7 (arm) or 9 "
                "(waist+arm):",
                len(self.active_joints),
            )

    def bounds(self):
        """返回所有活动关节的限位上下界数组 (lower, upper)。"""
        lower = []
        upper = []

        for joint in self.active_joints:
            if joint["type"] == "continuous":
                lower.append(-np.pi)
                upper.append(np.pi)
            else:
                lower.append(joint["lower"])
                upper.append(joint["upper"])

        return np.array(lower, dtype=float), np.array(upper, dtype=float)

    def fk_all(self, q):
        """计算所有连杆及关节原点相对于基座的 4x4 齐次变换矩阵。"""
        q = np.asarray(q, dtype=float)
        if len(q) != len(self.active_joints):
            raise ValueError(
                f"q size mismatch: got {len(q)}, "
                f"expected {len(self.active_joints)}"
            )

        T = np.eye(4)
        motions = self._joint_motions(q)

        link_T = {self.base_link: T.copy()}
        joint_origin_T = {}

        for joint, Tmotion in zip(self.joints, motions):
            Tj_origin = T @ self._origin_T(joint)
            joint_origin_T[joint["name"]] = Tj_origin.copy()

            if Tmotion is None:  # fixed
                Tmotion = np.eye(4)

            T = Tj_origin @ Tmotion
            link_T[joint["child"]] = T.copy()

        return link_T, joint_origin_T

    def _origin_T(self, joint):
        """按 (xyz, rpy) 字节缓存关节原点变换矩阵。"""
        key = (
            np.asarray(joint["xyz"], dtype=float).tobytes(),
            np.asarray(joint["rpy"], dtype=float).tobytes(),
        )
        cache = self.__dict__.setdefault("_origin_cache", {})
        T = cache.get(key)
        if T is None:
            T = T_origin(joint["xyz"], joint["rpy"])
            T.flags.writeable = False
            cache[key] = T
        return T

    def _unit_axis(self, axis):
        """按原始转轴字节缓存归一化轴。"""
        key = np.asarray(axis, dtype=float).tobytes()
        cache = self.__dict__.setdefault("_axis_cache", {})
        unit = cache.get(key)
        if unit is None:
            unit = normalize(axis)
            unit.flags.writeable = False
            cache[key] = unit
        return unit

    def _joint_motions(self, q):
        """批量计算与 self.joints 对齐的局部运动变换。"""
        motions = [None] * len(self.joints)
        rows, rotvecs = [], []
        active_idx = 0
        for k, joint in enumerate(self.joints):
            if joint["type"] in ("revolute", "continuous"):
                rotvecs.append(self._unit_axis(joint["axis"]) * q[active_idx])
                rows.append(k)
                active_idx += 1
            elif joint["type"] == "prismatic":
                motions[k] = T_prismatic(joint["axis"], q[active_idx])
                active_idx += 1
            elif joint["type"] != "fixed":
                raise NotImplementedError(
                    f"Unsupported joint type: {joint['type']}"
                )
        if rotvecs:
            mats = R.from_rotvec(np.array(rotvecs)).as_matrix()
            zero = np.zeros(3)
            for k, Rm in zip(rows, mats):
                motions[k] = make_T(Rm, zero)
        return motions

    def fk(self, q):
        """返回末端执行器（ee_link）的 4x4 位姿变换。"""
        link_T, _ = self.fk_all(q)
        return link_T[self.ee_link]

    def fk_pose_and_joints(self, q, joint_names):
        """单次前向递推同时获取末端位姿与指定关节点原点位姿。"""
        q = np.asarray(q, dtype=float)
        if len(q) != len(self.active_joints):
            raise ValueError(
                f"q size mismatch: got {len(q)}, "
                f"expected {len(self.active_joints)}"
            )

        wanted = set(joint_names)
        joint_T = {}
        T = np.eye(4)
        motions = self._joint_motions(q)

        for joint, Tmotion in zip(self.joints, motions):
            T = T @ self._origin_T(joint)
            if joint["name"] in wanted:
                joint_T[joint["name"]] = T.copy()

            if Tmotion is not None:
                T = T @ Tmotion

        missing = wanted.difference(joint_T)
        if missing:
            raise KeyError(f"Joints not found in chain: {sorted(missing)}")
        return T, joint_T

    def joint_position(self, q, joint_name):
        """获取指定关节原点在基座系下的三维坐标。"""
        _, joint_origin_T = self.fk_all(q)
        if joint_name not in joint_origin_T:
            raise KeyError(f"Joint {joint_name} not found")
        return joint_origin_T[joint_name][:3, 3]

    def psi(
        self,
        q,
        S_joint=None,
        E_joint=None,
        W_joint=None,
        ref_dir=REF_DIR,
    ):
        """计算给定关节角下的 Shoulder-Elbow-Wrist (SEW) 臂角 ψ。"""
        S_joint, E_joint, W_joint = get_sew_joints(self, S_joint, E_joint, W_joint)
        _, joint_origin_T = self.fk_all(q)
        for name in (S_joint, E_joint, W_joint):
            if name not in joint_origin_T:
                raise KeyError(f"Joint {name} not found")
        S = joint_origin_T[S_joint][:3, 3]
        E = joint_origin_T[E_joint][:3, 3]
        W = joint_origin_T[W_joint][:3, 3]
        return sew_angle_from_points(S, E, W, ref_dir)


def estimate_chain_max_reach_from_joint(chain, anchor_joint_name):
    """估算串联链从指定锚点关节原点到末端的最大理论伸展距离。"""
    reach = 0.0
    found_anchor = False

    for joint in chain.joints:
        if joint["name"] == anchor_joint_name:
            found_anchor = True
            continue
        if found_anchor:
            xyz = np.asarray(joint.get("xyz", np.zeros(3)), dtype=float)
            reach += float(np.linalg.norm(xyz))

    if not found_anchor:
        raise KeyError(f"Anchor joint {anchor_joint_name} not found in chain.")

    return reach


def joint_origin_position_at_zero(chain, joint_name):
    """返回零位姿下指定关节原点在基座系下的三维坐标。"""
    q0 = np.zeros(len(chain.active_joints), dtype=float)
    return chain.joint_position(q0, joint_name).copy()
