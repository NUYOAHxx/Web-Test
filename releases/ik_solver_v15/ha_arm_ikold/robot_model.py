#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""URDF serial-chain model and the built-in humanoid arm configurations."""

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

# 臂角 S / E / W 关节的兜底名（SerialChain 未给 s_joint / e_joint / w_joint 时使用）。
# 与 ARM_CONFIGS["left"] 的命名一致；旧值 "Left_J2/4/6" 与 ARM_CONFIGS 的 LHand_Link* 命名不符，
# 兜底生效时找不到关节，臂角恒为 NaN。
S_JOINT = "LHand_Link2"
E_JOINT = "LHand_Link4"
W_JOINT = "LHand_Link6"
REF_DIR = np.array([0.0, 0.0, 1.0], dtype=float)
WAIST_JOINT_NAMES = ["Hip_Link1", "Body_Link"]

ARM_CONFIGS = {
    "left": {
        # Arm IK root. Body_Link is intentional so Hip_Link1 / Body_Link are
        # not part of the 7-DOF arm chain.
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


def parse_vec(s):
    return np.array([float(x) for x in s.split()], dtype=float)


def get_fixed_joint_origin(urdf_path, parent_link, child_link):
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
    """Return the single root link of a URDF tree."""
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

    # v15 提速（结果逐位不变）：
    #   - 关节原点变换只取决于 (xyz, rpy) 的数值，按其 float64 字节缓存，命中时与重算 T_origin 完全相同；
    #   - 转动关节的旋转矩阵整链一次批量调用 scipy from_rotvec（逐行同一例程，与逐关节 rot_axis 逐位相同）。
    def _origin_T(self, joint):
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
        key = np.asarray(axis, dtype=float).tobytes()
        cache = self.__dict__.setdefault("_axis_cache", {})
        unit = cache.get(key)
        if unit is None:
            unit = normalize(axis)
            unit.flags.writeable = False
            cache[key] = unit
        return unit

    def _joint_motions(self, q):
        """与 self.joints 对齐的关节运动变换（fixed 为 None）；数值与逐个 T_revolute / T_prismatic 相同。"""
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
        link_T, _ = self.fk_all(q)
        return link_T[self.ee_link]

    def fk_pose_and_joints(self, q, joint_names):
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
        if S_joint is None:
            S_joint = self.s_joint if self.s_joint else S_JOINT
        if E_joint is None:
            E_joint = self.e_joint if self.e_joint else E_JOINT
        if W_joint is None:
            W_joint = self.w_joint if self.w_joint else W_JOINT

        S = self.joint_position(q, S_joint)
        E = self.joint_position(q, E_joint)
        W = self.joint_position(q, W_joint)
        return sew_angle_from_points(S, E, W, ref_dir)


def estimate_chain_max_reach_from_joint(chain, anchor_joint_name):
    """Estimate max EE reach from a given joint origin.

    Sums all joint-origin translation lengths after ``anchor_joint_name`` in
    the serial chain.
    """
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
    """Return a joint-origin position in the chain/base frame at zero pose."""
    q0 = np.zeros(len(chain.active_joints), dtype=float)
    _, joint_origin_T = chain.fk_all(q0)

    if joint_name not in joint_origin_T:
        raise KeyError(f"Joint {joint_name} not found in chain.")

    return joint_origin_T[joint_name][:3, 3].copy()