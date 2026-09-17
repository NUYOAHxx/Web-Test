#!/usr/bin/env python3
"""
================================================================================
Unitree G1 机械臂动力学与运动学模型引擎 (Kinematics & Dynamics Model Engine)
================================================================================

职责：
1. 管理机器人 URDF 几何拓扑、刚体动力学模型 (Pinocchio Model/Data)；
2. 维护机械臂 (7-DoF) 与腰部 (3-DoF) 的关节名称映射、索引字典与物理硬件极限；
3. 提供高性能前向运动学 (FK) 与空间解析雅可比 (Analytical Jacobian) 计算服务。
"""

import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import pinocchio as pin


# ==============================================================================
# 硬件物理规格与参考基准姿态 (Hardware Limits & Reference Configurations)
# ==============================================================================

# 宇树 G1 机械臂 7 自由度硬件限位 (单位: 弧度)
G1_JOINT_LIMITS: Dict[str, Tuple[float, float]] = {
    # 左臂 7-DoF
    "left_shoulder_pitch_joint": (-3.0892, 2.6704),
    "left_shoulder_roll_joint":  (-1.5882, 2.2515),
    "left_shoulder_yaw_joint":   (-2.6180, 2.6180),
    "left_elbow_joint":          (-1.0472, 2.0944),
    "left_wrist_roll_joint":     (-1.9722, 1.9722),
    "left_wrist_pitch_joint":    (-1.6144, 1.6144),
    "left_wrist_yaw_joint":      (-1.6144, 1.6144),
    # 右臂 7-DoF
    "right_shoulder_pitch_joint": (-3.0892, 2.6704),
    "right_shoulder_roll_joint":  (-2.2515, 1.5882),
    "right_shoulder_yaw_joint":   (-2.6180, 2.6180),
    "right_elbow_joint":          (-1.0472, 2.0944),
    "right_wrist_roll_joint":     (-1.9722, 1.9722),
    "right_wrist_pitch_joint":    (-1.6144, 1.6144),
    "right_wrist_yaw_joint":      (-1.6144, 1.6144),
}

# 宇树 G1 腰部 3 自由度物理限位 (单位: 弧度)
G1_WAIST_LIMITS: Dict[str, Tuple[float, float]] = {
    "waist_yaw_joint":   (-2.6180, 2.6180),   # 左右旋转 ±150 度
    "waist_roll_joint":  (-0.5200, 0.5200),   # 左右侧倾 ±30 度
    "waist_pitch_joint": (-0.5200, 0.5200),   # 前后俯仰 ±30 度
}

# 人体仿生预备就绪姿态 (肘部自然朝下微弯、手腕平直)
G1_READY_POSE: Dict[str, np.ndarray] = {
    "left_arm":  np.array([0.2,  0.2, 0.0, 0.5, 0.0, 0.0, 0.0], dtype=np.float64),
    "right_arm": np.array([0.2, -0.2, 0.0, 0.5, 0.0, 0.0, 0.0], dtype=np.float64),
}


class G1KinematicsModel:
    """
    Unitree G1 运动学引擎：封装 Pinocchio 模型与动力学计算图
    """

    def __init__(self, urdf_path: Optional[str] = None):
        if urdf_path is None:
            default_paths = [
                "/home/parallels/ws_moveit/src/g1_description/urdf/g1_29dof.urdf",
                os.path.join(os.path.dirname(__file__), "../../resources/g1/g1_29dof.urdf"),
            ]
            for p in default_paths:
                if os.path.exists(p):
                    urdf_path = os.path.abspath(p)
                    break

        if not urdf_path or not os.path.exists(urdf_path):
            raise FileNotFoundError(f"无法找到 G1 URDF 模型文件: {urdf_path}")

        self.urdf_path: str = urdf_path
        self.model: pin.Model = pin.buildModelFromUrdf(self.urdf_path)
        self.data: pin.Data = self.model.createData()

        # ── 1. 关节定义与末端 Frame ──
        self.left_arm_joint_names: List[str] = [
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
        ]

        self.right_arm_joint_names: List[str] = [
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        ]

        self.ee_frame_names: Dict[str, str] = {
            "left_arm":  "left_wrist_yaw_link",
            "right_arm": "right_wrist_yaw_link",
        }

        self.ee_frame_ids: Dict[str, int] = {
            "left_arm":  self.model.getFrameId(self.ee_frame_names["left_arm"]),
            "right_arm": self.model.getFrameId(self.ee_frame_names["right_arm"]),
        }

        # 缓存单臂各关节在全模型 q 向量中的全局索引
        self.arm_q_indices: Dict[str, List[int]] = {
            "left_arm":  [self.model.getJointId(name) - 1 for name in self.left_arm_joint_names],
            "right_arm": [self.model.getJointId(name) - 1 for name in self.right_arm_joint_names],
        }

        # 提取 7-DoF 关节物理限位数组 (lower, upper)
        self.limits: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for arm, jnames in [("left_arm", self.left_arm_joint_names), ("right_arm", self.right_arm_joint_names)]:
            lower = np.array([G1_JOINT_LIMITS[name][0] for name in jnames], dtype=np.float64)
            upper = np.array([G1_JOINT_LIMITS[name][1] for name in jnames], dtype=np.float64)
            self.limits[arm] = (lower, upper)

        # ── 2. 腰部 3 自由度关节与 10-DoF 协同链 ──
        self.waist_joint_names: List[str] = [
            "waist_yaw_joint",
            "waist_roll_joint",
            "waist_pitch_joint",
        ]
        self.waist_q_indices: List[int] = [self.model.getJointId(name) - 1 for name in self.waist_joint_names]
        waist_lower = np.array([G1_WAIST_LIMITS[name][0] for name in self.waist_joint_names], dtype=np.float64)
        waist_upper = np.array([G1_WAIST_LIMITS[name][1] for name in self.waist_joint_names], dtype=np.float64)
        self.waist_limits: Tuple[np.ndarray, np.ndarray] = (waist_lower, waist_upper)

        self.chain_10dof_joint_names: Dict[str, List[str]] = {
            "left_arm":  self.waist_joint_names + self.left_arm_joint_names,
            "right_arm": self.waist_joint_names + self.right_arm_joint_names,
        }
        self.chain_10dof_indices: Dict[str, List[int]] = {
            "left_arm":  self.waist_q_indices + self.arm_q_indices["left_arm"],
            "right_arm": self.waist_q_indices + self.arm_q_indices["right_arm"],
        }
        self.limits_10dof: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for arm in ["left_arm", "right_arm"]:
            l_10 = np.concatenate([waist_lower, self.limits[arm][0]])
            u_10 = np.concatenate([waist_upper, self.limits[arm][1]])
            self.limits_10dof[arm] = (l_10, u_10)

    # --------------------------------------------------------------------------
    # 正向运动学 (Forward Kinematics)
    # --------------------------------------------------------------------------

    def forward_kinematics(
        self, arm: str, q_arm: np.ndarray, q_full_base: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算 7-DoF 单臂正向运动学 (FK)
        :param arm: "left_arm" 或 "right_arm"
        :param q_arm: (7,) 单臂关节角
        :param q_full_base: (29,) 全身基础姿势 (可选，默认 neutral)
        :return: (pos: np.ndarray(3,), rot_matrix: np.ndarray(3,3))
        """
        if q_full_base is None:
            q_full = pin.neutral(self.model)
        else:
            q_full = q_full_base.copy()

        indices = self.arm_q_indices[arm]
        for i, idx in enumerate(indices):
            q_full[idx] = q_arm[i]

        pin.forwardKinematics(self.model, self.data, q_full)
        pin.updateFramePlacements(self.model, self.data)

        frame_id = self.ee_frame_ids[arm]
        oMf = self.data.oMf[frame_id]
        return oMf.translation.copy(), oMf.rotation.copy()

    def forward_kinematics_10dof(
        self, arm: str, q_waist: np.ndarray, q_arm: np.ndarray, q_full_base: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算 10-DoF (3腰 + 7臂) 协同正向运动学 (FK)
        :param arm: "left_arm" 或 "right_arm"
        :param q_waist: (3,) 腰部关节角 [yaw, roll, pitch]
        :param q_arm: (7,) 单臂关节角
        :param q_full_base: (29,) 全身基础姿势 (可选)
        :return: (pos: np.ndarray(3,), rot_matrix: np.ndarray(3,3))
        """
        if q_full_base is None:
            q_full = pin.neutral(self.model)
        else:
            q_full = q_full_base.copy()

        indices = self.chain_10dof_indices[arm]
        q_10dof = np.concatenate([np.asarray(q_waist, dtype=np.float64), np.asarray(q_arm, dtype=np.float64)])
        for i, idx in enumerate(indices):
            q_full[idx] = q_10dof[i]

        pin.forwardKinematics(self.model, self.data, q_full)
        pin.updateFramePlacements(self.model, self.data)

        frame_id = self.ee_frame_ids[arm]
        oMf = self.data.oMf[frame_id]
        return oMf.translation.copy(), oMf.rotation.copy()

    def compute_subspace_jacobian(
        self, q_full: np.ndarray, arm: str, is_10dof: bool = False, has_rot: bool = False
    ) -> np.ndarray:
        """
        计算指定手臂/协同链在基坐标系下的解析雅可比子矩阵
        :param q_full: (29,) 当前全身关节角向量
        :param arm: "left_arm" 或 "right_arm"
        :param is_10dof: 是否提取 10-DoF 子空间 (True 为 10 维，False 为 7 维)
        :param has_rot: 是否包含旋转行 (True: 6维空间位姿，False: 仅3维线速度)
        :return: 提取后的任务雅可比矩阵
        """
        frame_id = self.ee_frame_ids[arm]
        indices = self.chain_10dof_indices[arm] if is_10dof else self.arm_q_indices[arm]

        J_full = pin.computeFrameJacobian(
            self.model, self.data, q_full, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )
        if has_rot:
            return J_full[:, indices]
        return J_full[:3, indices]
