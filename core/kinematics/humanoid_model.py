#!/usr/bin/env python3
"""
================================================================================
人形机器人 机械臂动力学与运动学模型引擎 (Kinematics & Dynamics Model Engine)
================================================================================

职责：
1. 管理机器人 URDF 几何拓扑、刚体动力学模型 (Pinocchio Model/Data)；
2. 依据机器人描述档案 (RobotProfile) 维护腰部与机械臂的关节名称、索引与限位；
3. 提供高性能前向运动学 (FK) 与空间解析雅可比 (Analytical Jacobian) 计算服务。

关节链语义 (哪条链是腰、哪条是臂、哪两个连杆构成支撑足) 由
``core/kinematics/robot_profile.py`` 提供并在校验期强制与 URDF 对齐,
换用其他 URDF 时可通过同目录的 ``<名字>.profile.json`` 声明, 无需改码。
"""

import os
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pinocchio as pin

from core.kinematics import robot_profile as rp
from core.kinematics.robot_profile import RobotProfile, RobotProfileError


# ==============================================================================
# 硬件物理规格与参考基准姿态 (Hardware Limits & Reference Configurations)
# ==============================================================================

# 关节限位的唯一真源是 URDF 中的 <limit>, 由 :func:`read_joint_limits` 读取。
#
# 早期版本在此另存一份 JOINT_LIMITS / WAIST_LIMITS 副本字典, 供运动学引擎与
# 大屏遥测读取 —— 于是同一时刻存在两套限位来源, 二者分叉时不会报错, 只会
# 显示错误的限位条并放行越界姿态。现已全部改为从 URDF 读取, 副本随之删除;
# 需要覆盖某关节限位时, 在该 URDF 同目录的 <名字>.profile.json 中写
# {"limit_overrides": {"left_elbow_joint": [-1.0, 2.0]}}。

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


#: 力矩包络采样的样本数与随机种子。
#: 种子固定 ⇒ 结果可复现, 换模型重跑得到同样的阈值, 便于对照排查。
_ENVELOPE_SAMPLES = 400
_ENVELOPE_SEED = 20240501


class HumanoidKinematicsModel:
    """
    人形机器人 运动学引擎：封装 Pinocchio 模型与动力学计算图
    """

    def __init__(
        self,
        urdf_path: Optional[str] = None,
        srdf_path: Optional[str] = None,
        profile: Optional["RobotProfile"] = None,
        require_support: bool = True,
    ):
        """
        :param urdf_path: 机器人 URDF 路径, 为空时用仓库内置模型
        :param srdf_path: MoveIt SRDF 路径 (允许碰撞矩阵), 为空时按 URDF 同名推导
        :param profile: 机器人描述档案 (关节链语义)。为 None 时读取 URDF 同目录的
            ``<basename>.profile.json``, 不存在则用默认档案。
        :param require_support: 是否要求支撑连杆可解析 (仅平衡评估需要)

        :raises RobotProfileError: 档案声明的关节 / Frame / 连杆在 URDF 中不存在。
            这是刻意的: 旧实现对不存在的名称调用 ``getJointId()`` 会得到
            ``model.njoints``, 减一后恰好是合法下标, 于是静默指向另一个关节,
            求解器会对着一条不存在的链算出看似合理实则全错的结果。
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

        # ── 1. 关节链语义: 校验 + 自动补全 ──
        # 校验必须在任何 getJointId / getFrameId 调用**之前**完成, 否则错误索引
        # 已经混进后续全部计算, 报错位置也会远离真正的错误来源。
        self.profile: RobotProfile = rp.resolve(
            self.model,
            profile if profile is not None else rp.RobotProfile.load_for_urdf(self.urdf_path),
            require_support=require_support,
        )

        self.left_arm_joint_names: List[str] = list(self.profile.left_arm_joint_names)
        self.right_arm_joint_names: List[str] = list(self.profile.right_arm_joint_names)

        self.ee_frame_names: Dict[str, str] = {
            "left_arm":  self.profile.left_ee_frame,
            "right_arm": self.profile.right_ee_frame,
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

        # 关节限位: 唯一真源为 URDF 的 <limit>, 档案覆盖仅在 URDF 缺省时生效
        overrides = self.profile.limit_overrides
        self.limits: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for arm, jnames in [("left_arm", self.left_arm_joint_names), ("right_arm", self.right_arm_joint_names)]:
            bounds = rp.read_joint_limits(self.model, jnames, overrides)
            self.limits[arm] = (bounds[:, 0].copy(), bounds[:, 1].copy())

        # ── 2. 腰部关节与 9-DoF 协同链 ──
        self.waist_joint_names: List[str] = list(self.profile.waist_joint_names)
        # 腰部自由度由关节集合长度派生, 而非硬编码 3 —— 下游 (求解器/碰撞/降维模型)
        # 全部依赖该值, 9-DoF 与 7-DoF 因此共用同一条代码路径。
        self.waist_dim: int = len(self.waist_joint_names)
        self.waist_q_indices: List[int] = [self.model.getJointId(name) - 1 for name in self.waist_joint_names]
        waist_bounds = rp.read_joint_limits(self.model, self.waist_joint_names, overrides)
        waist_lower, waist_upper = waist_bounds[:, 0].copy(), waist_bounds[:, 1].copy()
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

        # ── 3. 双足支撑几何 (Dual-foot Support Polygon) ──
        # 由支撑连杆的碰撞几何在参考姿态下的 AABB 并集推导, 而非写死某型号尺寸 ——
        # 脚掌尺寸不同的模型, 写死的矩形会让平衡裕度系统性失真且无任何提示。
        # 档案可用 support_polygon 字段显式覆盖。
        self.support_polygon_source: str
        if self.profile.support_polygon:
            self.support_polygon = dict(self.profile.support_polygon)
            self.support_polygon_source = "档案显式指定"
        else:
            self.support_polygon = rp.derive_support_polygon(
                self.model, self.data, self.coll_model, self.profile.support_links,
            )
            self.support_polygon_source = f"由支撑几何 AABB 推导 {list(self.profile.support_links)}"

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

    def compute_orientation_envelope(
        self, arm: str = "left_arm", samples: int = 4000, seed: int = 20240501,
        inset: float = 0.85,
    ) -> Dict[str, Dict[str, float]]:
        """仅计算末端姿态 (RPY) 的可达范围。

        位置范围由可达性索引 (:mod:`core.kinematics.reachability`) 给出 ——
        它用更密的采样建 KD 树, 精度更高; 此处只补索引不覆盖的朝向维度。

        :return: ``{"roll": {...}, "pitch": {...}, "yaw": {...}}``, 单位度
        """
        return {k: v for k, v in
                self.compute_reach_envelope(arm, samples=samples, seed=seed,
                                            inset=inset).items()
                if k in ("roll", "pitch", "yaw")}

    def compute_reach_envelope(
        self,
        arm: str = "left_arm",
        samples: int = 6000,
        seed: int = 20240501,
        inset: float = 0.85,
        front_only: bool = True,
    ) -> Dict[str, Dict[str, float]]:
        """实测末端工作空间包络 (含姿态), 作为指令入口与前端滑块的**唯一真源**。

        早期实现把范围 (X:[0.05,0.45] Y:±0.30 Z:[0.70,1.30] deg:±180) 同时
        写在前端 HTML 的 min/max 与服务端的校验字典里, 两处无任何关联:

        - 改一边不改另一边, 就会出现"滑块能选但后端拒绝"或反之, 用户只看到
          一句报错, 不知道是界面给的越界;
        - 换 URDF 后数值不变, 而实际可达范围已完全改变;
        - 姿态根本没有范围校验 (后端只查 NaN/Inf), 滑到 (180,180,180) 会被
          原样下发, 表现为"校验通过但残差几百毫米" —— 正是当初收紧位置包络
          要避免的情况。

        本方法在限位盒内采样, 剔除自碰撞位姿后取各轴极值:

        - ``inset``   各侧按比例向可行域中心内缩, 避免用户恰好选在边界上
                       (边界处 IK 收敛率显著下降);
        - ``front_only`` X 只取前伸方向。身后可达 (实测 X 下界约 -0.5m) 属于
                       贴近躯干的自碰撞奇异区, SRDF 相邻对过滤捕捉不到。

        采样为确定性伪随机 (固定种子), 结果可复现, 便于换模型后对照排查。

        :return: ``{"x": {"min","max"}, "y": {...}, "z": {...},
                   "roll": {...}, "pitch": {...}, "yaw": {...}}``, 角度单位度
        """
        import pinocchio as pin

        lo = np.concatenate([self.waist_limits[0], self.limits[arm][0]])
        hi = np.concatenate([self.waist_limits[1], self.limits[arm][1]])
        names = self.chain_9dof_joint_names[arm]
        q_idx = [self.model.getJointId(n) - 1 for n in names]
        fid = self.ee_frame_ids[arm]

        # 自碰撞过滤需要完整全身模型状态, 此处用未降维的全模型判定
        collider = self._build_reduced_collider(arm)

        p_lo = np.array([np.inf] * 3)
        p_hi = np.array([-np.inf] * 3)
        r_lo = np.array([np.inf] * 3)
        r_hi = np.array([-np.inf] * 3)
        hit = 0
        rng = np.random.default_rng(seed)

        for _ in range(samples):
            q = pin.neutral(self.model)
            q_red = np.empty(len(names))
            for i, j in enumerate(q_idx):
                v = float(rng.uniform(lo[i], hi[i]))
                q[j] = v
                q_red[i] = v
            if collider is not None and collider(q_red):
                continue

            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            p = self.data.oMf[fid].translation
            p_lo = np.minimum(p_lo, p)
            p_hi = np.maximum(p_hi, p)
            rpy = pin.rpy.matrixToRpy(self.data.oMf[fid].rotation)
            r_lo = np.minimum(r_lo, rpy)
            r_hi = np.maximum(r_hi, rpy)
            hit += 1

        if hit == 0:
            raise RobotProfileError(
                "工作空间采样全部被自碰撞剔除, 无法推导可达包络。"
                "请检查 URDF 的碰撞几何与 SRDF 允许碰撞矩阵。")

        if front_only:
            # 身后为贴近躯干的奇异区, 不作为可用工作空间
            p_lo[0] = max(p_lo[0], 0.0)

        def _box(lo_v: float, hi_v: float) -> Dict[str, float]:
            c = (lo_v + hi_v) * 0.5
            half = (hi_v - lo_v) * 0.5 * inset
            return {"min": round(c - half, 4), "max": round(c + half, 4)}

        out: Dict[str, Dict[str, float]] = {
            "x": _box(p_lo[0], p_hi[0]),
            "y": _box(p_lo[1], p_hi[1]),
            "z": _box(p_lo[2], p_hi[2]),
        }
        for i, key in enumerate(("roll", "pitch", "yaw")):
            out[key] = _box(float(np.degrees(r_lo[i])), float(np.degrees(r_hi[i])))
        out["_samples"] = {"total": samples, "collision_free": hit}
        return out

    def _build_reduced_collider(self, arm: str):
        """构造该臂降维链上的自碰撞判据; 依赖碰撞引擎时返回 None (跳过过滤)。"""
        try:
            from core.collision.humanoid_collision import HumanoidCollisionChecker
            checker = HumanoidCollisionChecker(self)
        except Exception:
            return None
        return lambda q_red: checker.is_colliding_reduced(arm, q_red, is_9dof=True)

    def compute_torque_envelope(
        self, arm: str = "left_arm", is_9dof: bool = True,
    ) -> Dict[str, float]:
        """各关节在自身限位范围内可达到的最大静态重力力矩。

        这是**结构性负载**, 与"是否异常"无关: 腰部俯仰关节承载整个躯干,
        合法上限可达 12+ N·m; 腕关节则始终在 0.5 N·m 以内。

        用一条全局阈值判断全部关节必然误报 —— 躯干关节常年报警, 腕关节永远
        正常, 阈值失去意义。此处逐关节给出结构上限, 遥测按 ``|τ|/上限`` 的
        **利用率**着色, 换模型时随该模型自身的质量分布与限位自动重新标定。

        采样: 零位 + 每个关节各自的限位上下端点。重力力矩只依赖位形, 沿某
        单轴推到端点即覆盖该轴贡献的极值; 轴间的交叉项在多轴同时偏置时才会
        出现, 因此上限取各单轴极值的较小安全系数由调用方按利用率呈现。

        :return: ``{关节名: 最大 |重力力矩| (N·m)}``
        """
        names = self.chain_9dof_joint_names[arm] if is_9dof else list(self.model.names[1:])
        lower = (np.concatenate([self.waist_limits[0], self.limits[arm][0]])
                 if is_9dof else np.array([self.model.lowerPositionLimit[
                     self.model.getJointId(n) - 1] for n in names]))
        upper = (np.concatenate([self.waist_limits[1], self.limits[arm][1]])
                 if is_9dof else np.array([self.model.upperPositionLimit[
                     self.model.getJointId(n) - 1] for n in names]))

        q_idx = [self.model.getJointId(n) - 1 for n in names]
        q = pin.neutral(self.model)
        envelope = np.zeros(len(names), dtype=np.float64)

        def _sweep() -> None:
            tau = self.compute_gravity_torques(q, arm=arm, is_9dof=is_9dof)
            for i, v in enumerate(tau):
                envelope[i] = max(envelope[i], abs(float(v)))

        # (1) 零位与各关节单独推到限位端点 —— 覆盖单轴贡献的极值
        _sweep()
        for i, idx in enumerate(q_idx):
            saved = q[idx]
            for val in (float(lower[i]), float(upper[i])):
                q[idx] = val
                _sweep()
            q[idx] = saved

        # (2) 限位盒内的确定性伪随机采样 —— 覆盖多轴同时偏置时的交叉贡献。
        #     只取单轴极值会显著低估: 手臂前伸时肩关节负载高于任何单轴姿态,
        #     导致真实姿态的利用率超过 100%, 误报为"重载"。
        rng = np.random.default_rng(_ENVELOPE_SEED)
        for _ in range(_ENVELOPE_SAMPLES):
            for i, idx in enumerate(q_idx):
                q[idx] = float(rng.uniform(lower[i], upper[i]))
            _sweep()

        # 下限兜底: 完全无负载的关节 (如 yaw) 会得到 0, 使利用率除零
        return {n: float(max(v, 0.5)) for n, v in zip(names, envelope)}

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

