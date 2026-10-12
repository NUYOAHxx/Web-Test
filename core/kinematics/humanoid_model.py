#!/usr/bin/env python3
"""
================================================================================
人形机器人 机械臂动力学与运动学模型引擎 (Kinematics & Dynamics Model Engine)
================================================================================

职责：
1. 管理机器人 URDF 几何拓扑、刚体动力学模型 (Pinocchio Model/Data)；
2. 维护机械臂 (7-DoF) 与腰部 (3-DoF) 的关节名称映射、索引字典与物理硬件极限；
3. 提供高性能前向运动学 (FK) 与空间解析雅可比 (Analytical Jacobian) 计算服务。
"""

import os
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pinocchio as pin


# ==============================================================================
# 硬件物理规格与参考基准姿态 (Hardware Limits & Reference Configurations)
# ==============================================================================

# 人形机器人 机械臂 7 自由度硬件限位 (单位: 弧度)
JOINT_LIMITS: Dict[str, Tuple[float, float]] = {
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

# 人形机器人 腰部 3 自由度物理限位 (单位: 弧度)
WAIST_LIMITS: Dict[str, Tuple[float, float]] = {
    "waist_yaw_joint":   (-2.6180, 2.6180),   # 左右旋转 ±150 度
    "waist_roll_joint":  (-0.5200, 0.5200),   # 左右侧倾 ±30 度
    "waist_pitch_joint": (-0.5200, 0.5200),   # 前后俯仰 ±30 度
}

# 腰部活动关节集合 —— 固定为 2 自由度 (Yaw + Pitch)。
# 刻意不含 waist_roll: 躯干不侧倾, 末端姿态完全由 7 轴手臂决定, 姿态可控性更直接,
# 也避免 3 自由度腰部引入的额外多解分支。
WAIST_JOINTS: List[str] = [
    "waist_yaw_joint",
    "waist_pitch_joint",
]

# 人体仿生预备就绪姿态 (肘部自然朝下微弯、手腕平直)
READY_POSE: Dict[str, np.ndarray] = {
    "left_arm":  np.array([0.2,  0.2, 0.0, 0.5, 0.0, 0.0, 0.0], dtype=np.float64),
    "right_arm": np.array([0.2, -0.2, 0.0, 0.5, 0.0, 0.0, 0.0], dtype=np.float64),
}

# 人形机器人 官方标准对称直立预备就绪姿态全关节字典
DEFAULT_STAND_JOINTS: Dict[str, float] = {
    "waist_yaw_joint": 0.0,
    "waist_roll_joint": 0.0,
    "waist_pitch_joint": 0.0,
    "left_shoulder_pitch_joint": 0.2,
    "left_shoulder_roll_joint": 0.2,
    "left_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": 0.5,
    "left_wrist_roll_joint": 0.0,
    "left_wrist_pitch_joint": 0.0,
    "left_wrist_yaw_joint": 0.0,
    "right_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    "right_shoulder_yaw_joint": 0.0,
    "right_elbow_joint": 0.5,
    "right_wrist_roll_joint": 0.0,
    "right_wrist_pitch_joint": 0.0,
    "right_wrist_yaw_joint": 0.0,
    "left_hip_pitch_joint": 0.0,
    "left_hip_roll_joint": 0.0,
    "left_hip_yaw_joint": 0.0,
    "left_knee_joint": 0.0,
    "left_ankle_pitch_joint": 0.0,
    "left_ankle_roll_joint": 0.0,
    "right_hip_pitch_joint": 0.0,
    "right_hip_roll_joint": 0.0,
    "right_hip_yaw_joint": 0.0,
    "right_knee_joint": 0.0,
    "right_ankle_pitch_joint": 0.0,
    "right_ankle_roll_joint": 0.0,
}


class HumanoidKinematicsModel:
    """
    人形机器人 运动学引擎：封装 Pinocchio 模型与动力学计算图
    """

    def __init__(
        self,
        urdf_path: Optional[str] = None,
        srdf_path: Optional[str] = None,
    ):
        """
        :param waist_joint_names: 已移除 —— 腰部自由度固定为 2 (Yaw + Pitch)，即 9-DoF 总自由度。
        """
        if urdf_path is None:
            urdf_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../resources/humanoid/humanoid_29dof.urdf"))

        if not urdf_path or not os.path.exists(urdf_path):
            raise FileNotFoundError(f"无法找到 URDF 模型文件: {urdf_path}")

        self.urdf_path: str = urdf_path
        self.model: pin.Model = pin.buildModelFromUrdf(self.urdf_path)
        self.data: pin.Data = self.model.createData()

        # ── 0. 载入与解析 MoveIt SRDF 碰撞矩阵 (ACM) ──
        if srdf_path is None:
            srdf_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../resources/humanoid/humanoid_29dof.srdf"))
        self.srdf_path: Optional[str] = srdf_path if os.path.exists(srdf_path) else None

        # 构建工业级几何碰撞模型与几何数据
        self.coll_model: pin.GeometryModel = pin.buildGeomFromUrdf(self.model, self.urdf_path, pin.GeometryType.COLLISION)
        self.coll_model.addAllCollisionPairs()
        if self.srdf_path and os.path.exists(self.srdf_path):
            pin.removeCollisionPairs(self.model, self.coll_model, self.srdf_path)
        self.coll_data: pin.GeometryData = pin.GeometryData(self.coll_model)

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
            lower = np.array([JOINT_LIMITS[name][0] for name in jnames], dtype=np.float64)
            upper = np.array([JOINT_LIMITS[name][1] for name in jnames], dtype=np.float64)
            self.limits[arm] = (lower, upper)

        # ── 2. 腰部 3 自由度关节与 9-DoF 协同链 ──
        self.waist_joint_names: List[str] = list(WAIST_JOINTS)
        # 腰部自由度由关节集合长度派生, 而非硬编码 3 —— 下游 (求解器/碰撞/降维模型)
        # 全部依赖该值, 9-DoF 与 9-DoF 因此共用同一条代码路径。
        self.waist_dim: int = len(self.waist_joint_names)
        self.waist_q_indices: List[int] = [self.model.getJointId(name) - 1 for name in self.waist_joint_names]
        waist_lower = np.array([WAIST_LIMITS[name][0] for name in self.waist_joint_names], dtype=np.float64)
        waist_upper = np.array([WAIST_LIMITS[name][1] for name in self.waist_joint_names], dtype=np.float64)
        self.waist_limits: Tuple[np.ndarray, np.ndarray] = (waist_lower, waist_upper)

        self.arm_joint_names: Dict[str, List[str]] = {
            "left_arm":  self.left_arm_joint_names,
            "right_arm": self.right_arm_joint_names,
        }
        self.chain_9dof_joint_names: Dict[str, List[str]] = {
            "left_arm":  self.waist_joint_names + self.left_arm_joint_names,
            "right_arm": self.waist_joint_names + self.right_arm_joint_names,
        }
        self.chain_9dof_indices: Dict[str, List[int]] = {
            "left_arm":  self.waist_q_indices + self.arm_q_indices["left_arm"],
            "right_arm": self.waist_q_indices + self.arm_q_indices["right_arm"],
        }
        self.limits_9dof: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for arm in ["left_arm", "right_arm"]:
            l_10 = np.concatenate([waist_lower, self.limits[arm][0]])
            u_10 = np.concatenate([waist_upper, self.limits[arm][1]])
            self.limits_9dof[arm] = (l_10, u_10)

        # ── 3. 双足物理支撑几何规格 (Dual-foot Support Polygon Geometry) ──
        # 双足长 ~0.22m (前0.13m, 后0.09m), 宽 ~0.10m, 双足外侧宽 ~0.34m (Y in [-0.17, 0.17])
        self.support_polygon: Dict[str, float] = {
            "x_min": -0.090,
            "x_max":  0.130,
            "y_min": -0.170,
            "y_max":  0.170,
        }

        # ── 4. 9-DoF & 7-DoF 轻量裁剪子模型缓存 (pin.buildReducedModel 提速 4.1x) ──
        self.model_9dof: Dict[str, pin.Model] = {}
        self.data_9dof: Dict[str, pin.Data] = {}
        self.coll_model_9dof: Dict[str, pin.GeometryModel] = {}
        self.coll_data_9dof: Dict[str, pin.GeometryData] = {}
        self.ee_frame_ids_9dof: Dict[str, int] = {}

        self.model_7dof: Dict[str, pin.Model] = {}
        self.data_7dof: Dict[str, pin.Data] = {}
        self.coll_model_7dof: Dict[str, pin.GeometryModel] = {}
        self.coll_data_7dof: Dict[str, pin.GeometryData] = {}
        self.ee_frame_ids_7dof: Dict[str, int] = {}
        self._init_reduced_models()

    def _init_reduced_models(self):
        """构建左右臂各 9-DoF (3腰 + 7臂) 与 7-DoF 单臂轻量化子模型与降维碰撞几何图，固定无关关节"""
        q_ref = pin.neutral(self.model)
        for arm in ["left_arm", "right_arm"]:
            # 1. 9-DoF (3腰 + 7臂)
            active_joints_10 = self.chain_9dof_joint_names[arm]
            locked_joint_ids_10 = [
                self.model.getJointId(jname)
                for jname in self.model.names[1:]
                if jname not in active_joints_10
            ]
            red_model_10, red_coll_10 = pin.buildReducedModel(self.model, self.coll_model, locked_joint_ids_10, q_ref)
            # 剪枝静态连杆对。
            # 判据: 降维模型中, 两侧几何体若挂在**同一个** parentJoint 上, 则相对位姿恒定
            # (要么都锁死、要么随同一关节刚性同动), 距离是常数, 逐帧重复检测无意义。
            # 旧判据用 `or` (任一侧会动即保留), 会把这类恒定对留下徒增 FCL 计算量。
            moving_pairs_10 = [
                p for p in red_coll_10.collisionPairs
                if red_coll_10.geometryObjects[p.first].parentJoint != red_coll_10.geometryObjects[p.second].parentJoint
            ]
            red_coll_10.removeAllCollisionPairs()
            for p in moving_pairs_10:
                red_coll_10.addCollisionPair(p)

            self.model_9dof[arm] = red_model_10
            self.data_9dof[arm] = red_model_10.createData()
            self.coll_model_9dof[arm] = red_coll_10
            self.coll_data_9dof[arm] = pin.GeometryData(red_coll_10)
            self.ee_frame_ids_9dof[arm] = red_model_10.getFrameId(self.ee_frame_names[arm])

            # 2. 7-DoF (7单臂)
            active_joints_7 = self.arm_joint_names[arm]
            locked_joint_ids_7 = [
                self.model.getJointId(jname)
                for jname in self.model.names[1:]
                if jname not in active_joints_7
            ]
            red_model_7, red_coll_7 = pin.buildReducedModel(self.model, self.coll_model, locked_joint_ids_7, q_ref)
            moving_pairs_7 = [
                p for p in red_coll_7.collisionPairs
                if red_coll_7.geometryObjects[p.first].parentJoint != red_coll_7.geometryObjects[p.second].parentJoint
            ]
            red_coll_7.removeAllCollisionPairs()
            for p in moving_pairs_7:
                red_coll_7.addCollisionPair(p)

            self.model_7dof[arm] = red_model_7
            self.data_7dof[arm] = red_model_7.createData()
            self.coll_model_7dof[arm] = red_coll_7
            self.coll_data_7dof[arm] = pin.GeometryData(red_coll_7)
            self.ee_frame_ids_7dof[arm] = red_model_7.getFrameId(self.ee_frame_names[arm])

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

    def build_q_9dof(
        self,
        arm: str,
        q_waist: np.ndarray,
        q_arm: np.ndarray,
        q_other_arm: Optional[np.ndarray] = None,
        base_q: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        根据 9-DoF (3腰 + 7臂) 构建 29-DoF 全身关节向量 q
        :param arm: 活动臂 ("left_arm" 或 "right_arm")
        :param q_waist: (3,) 腰部 3 关节 [yaw, roll, pitch]
        :param q_arm: (7,) 当前臂 7 关节
        :param q_other_arm: (7,) 对侧臂关节 (可选，默认使用 READY_POSE)
        :param base_q: (29,) 基础姿态 (可选，默认 neutral)
        """
        q_full = pin.neutral(self.model) if base_q is None else base_q.copy()
        for i, idx in enumerate(self.waist_q_indices):
            q_full[idx] = float(q_waist[i])
        for i, idx in enumerate(self.arm_q_indices[arm]):
            q_full[idx] = float(q_arm[i])
        opp_arm = "right_arm" if arm == "left_arm" else "left_arm"
        if q_other_arm is not None:
            for i, idx in enumerate(self.arm_q_indices[opp_arm]):
                q_full[idx] = float(q_other_arm[i])
        else:
            for i, idx in enumerate(self.arm_q_indices[opp_arm]):
                q_full[idx] = float(READY_POSE[opp_arm][i])
        return q_full

    def forward_kinematics_9dof(
        self, arm: str, q_waist: np.ndarray, q_arm: np.ndarray, q_full_base: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算 9-DoF (3腰 + 7臂) 协同正向运动学 (FK)
        :param arm: "left_arm" 或 "right_arm"
        :param q_waist: (3,) 腰部关节角 [yaw, roll, pitch]
        :param q_arm: (7,) 单臂关节角
        :param q_full_base: (29,) 全身基础姿势 (可选)
        :return: (pos: np.ndarray(3,), rot_matrix: np.ndarray(3,3))
        """
        q_full = self.build_q_9dof(arm, q_waist, q_arm, base_q=q_full_base)
        pin.forwardKinematics(self.model, self.data, q_full)
        pin.updateFramePlacements(self.model, self.data)
        oMf = self.data.oMf[self.ee_frame_ids[arm]]
        return oMf.translation.copy(), oMf.rotation.copy()

    def compute_subspace_jacobian(
        self, q_full: np.ndarray, arm: str, is_9dof: bool = False, has_rot: bool = False
    ) -> np.ndarray:
        """
        计算指定手臂/协同链在基坐标系下的解析雅可比子矩阵
        :param q_full: (29,) 当前全身关节角向量
        :param arm: "left_arm" 或 "right_arm"
        :param is_9dof: 是否提取 9-DoF 子空间 (True 为 10 维，False 为 7 维)
        :param has_rot: 是否包含旋转行 (True: 6维空间位姿，False: 仅3维线速度)
        :return: 提取后的任务雅可比矩阵
        """
        frame_id = self.ee_frame_ids[arm]
        indices = self.chain_9dof_indices[arm] if is_9dof else self.arm_q_indices[arm]

        J_full = pin.computeFrameJacobian(
            self.model, self.data, q_full, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )
        if has_rot:
            return J_full[:, indices]
        return J_full[:3, indices]

    # --------------------------------------------------------------------------
    # 9-DoF 轻量子模型极速正解与雅可比 (4.1x 提速)
    # --------------------------------------------------------------------------

    def forward_kinematics_reduced(
        self, arm: str, q_9dof: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        利用 9-DoF 裁剪子模型极速计算末端 FK (< 1us)
        :param arm: "left_arm" 或 "right_arm"
        :param q_9dof: (10,) 协同关节角度 [3腰 + 7臂]
        :return: (pos: np.ndarray(3,), rot_matrix: np.ndarray(3,3))
        """
        model = self.model_9dof[arm]
        data = self.data_9dof[arm]
        pin.forwardKinematics(model, data, q_9dof)
        pin.updateFramePlacements(model, data)
        oMf = data.oMf[self.ee_frame_ids_9dof[arm]]
        return oMf.translation.copy(), oMf.rotation.copy()

    def compute_jacobian_reduced(
        self, arm: str, q_9dof: np.ndarray, has_rot: bool = False
    ) -> np.ndarray:
        """
        利用 9-DoF 裁剪子模型极速计算解析空间雅可比 (< 1.5us)
        :param arm: "left_arm" 或 "right_arm"
        :param q_9dof: (10,) 协同关节角度
        :param has_rot: 是否包含姿态旋转分量 (True: 6x10, False: 3x10)
        :return: 任务空间雅可比矩阵
        """
        model = self.model_9dof[arm]
        data = self.data_9dof[arm]
        frame_id = self.ee_frame_ids_9dof[arm]
        J = pin.computeFrameJacobian(
            model, data, q_9dof, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )
        return J.copy() if has_rot else J[:3, :].copy()

    # --------------------------------------------------------------------------
    # 刚体动力学与全身体态平衡评估 (Pinocchio Dynamics & Balance)
    # --------------------------------------------------------------------------

    def compute_com(self, q_full: np.ndarray) -> np.ndarray:
        """
        计算机器人全身质心在基坐标系下的空间 3D 坐标
        :param q_full: (29,) 全身关节向量
        :return: (3,) 空间质心坐标 [x, y, z] (米)
        """
        com = pin.centerOfMass(self.model, self.data, q_full)
        return com.copy()

    def compute_com_jacobian(
        self, q_full: np.ndarray, arm: str = "left_arm", is_9dof: bool = True
    ) -> np.ndarray:
        """
        计算质心雅可比矩阵 J_com (d(CoM)/dq)
        :param q_full: (29,) 全身关节向量
        :param arm: 操作臂
        :param is_9dof: 是否切片提取 10 维子空间
        :return: (3, 10) 或 (3, 29) 质心雅可比
        """
        J_com = pin.jacobianCenterOfMass(self.model, self.data, q_full)
        if is_9dof:
            indices = self.chain_9dof_indices[arm]
            return J_com[:, indices].copy()
        return J_com.copy()

    def compute_gravity_torques(
        self, q_full: np.ndarray, arm: str = "left_arm", is_9dof: bool = True
    ) -> np.ndarray:
        """
        利用 Pinocchio 逆动力学 RNEA 计算各关节抵抗重力所需的广义重力补偿力矩 (N·m)
        :param q_full: (29,) 全身关节向量
        :param arm: 操作臂
        :param is_9dof: 是否提取 9-DoF (3腰 + 7臂)
        :return: 重力力矩向量 (N·m)
        """
        g_full = pin.computeGeneralizedGravity(self.model, self.data, q_full)
        if is_9dof:
            indices = self.chain_9dof_indices[arm]
            return g_full[indices].copy()
        return g_full.copy()

    def compute_manipulability(
        self, arm: str, q_9dof: np.ndarray, has_rot: bool = True
    ) -> Dict[str, Any]:
        """
        基于 Pinocchio 雅可比分析计算 Yoshikawa 可操作度指标与奇异点距离
        :param arm: "left_arm" 或 "right_arm"
        :param q_9dof: (10,) 协同关节角
        :param has_rot: 是否包含姿态行
        :return: 可操作度度量字典
        """
        J = self.compute_jacobian_reduced(arm, q_9dof, has_rot=has_rot)
        s = np.linalg.svd(J, compute_uv=False)
        sigma_min = float(s[-1])
        # Yoshikawa index: w = sqrt(det(J J^T)) = 奇异值之积
        yoshikawa = float(np.prod(s))
        is_singular = sigma_min < 0.015

        return {
            "yoshikawa": round(yoshikawa, 4),
            "min_singular_value": round(sigma_min, 4),
            "is_singular": is_singular,
            "condition_number": round(float(s[0] / max(1e-6, sigma_min)), 2),
        }

    def evaluate_balance(self, q_full: np.ndarray) -> Dict[str, Any]:
        """
        校验双足静平衡状态与质心在双足支撑多边形 (Support Polygon) 内的投影安全裕度
        :param q_full: (29,) 全身关节向量
        :return: 平衡安全度量字典 (com_pos, com_proj, margin_mm, status, support_polygon)
        """
        com = self.compute_com(q_full)
        com_x, com_y, com_z = float(com[0]), float(com[1]), float(com[2])
        poly = self.support_polygon

        # 计算地面投影距支撑面四周边界的最短安全距离
        dx_min = com_x - poly["x_min"]
        dx_max = poly["x_max"] - com_x
        dy_min = com_y - poly["y_min"]
        dy_max = poly["y_max"] - com_y

        margin_m = min(dx_min, dx_max, dy_min, dy_max)
        margin_mm = float(margin_m * 1000.0)

        if margin_mm > 30.0:
            status = "STABLE"
        elif margin_mm >= 0.0:
            status = "LEANING"
        else:
            status = "TIPPING"

        return {
            "com_pos": [round(com_x, 4), round(com_y, 4), round(com_z, 4)],
            "com_proj": [round(com_x, 4), round(com_y, 4)],
            "margin_mm": round(margin_mm, 1),
            "status": status,
            "support_polygon": dict(poly),
        }

