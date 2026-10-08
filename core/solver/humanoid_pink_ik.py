#!/usr/bin/env python3
"""
================================================================================
通用人形机器人躯干-机械臂运动学逆解优化求解器 (Humanoid Pink + ProxQP IK Engine)
================================================================================

本模块基于 Inria Pink 运动学库与 ProxQP / OSQP 凸优化内核，
为双足人形机器人提供高精度、强鲁棒、满足物理硬约束的笛卡尔末端逆运动学解算。

拓扑架构与自由度规范 (Topology & DoF Specification):
----------------------------------------------------
- 腰部驱动自由度 (2-DoF Waist): 仅包含旋转/偏航 (Yaw) 与 俯仰 (Pitch) 两个主动自由度，无侧倾 (Roll)；
- 手臂驱动自由度 (7-DoF Arm): 单臂 7 自由度高灵巧构型 (肩3 + 肘1 + 腕3)；
- 协同运动学链 (9-DoF Coordinated Chain): 2-DoF 腰部 + 7-DoF 机械臂。

================================================================================
腰部协同与介入控制逻辑 (Waist Coordination & Activation Control Logic):
================================================================================
在双足人形机器人全身运动学中，腰部是连接双腿基座与双臂的枢纽。若腰部无节制运动，
不仅会导致身姿怪异、能耗浪费，更会引发身体质心 (CoM) 剧烈偏移而导致双足失稳倾覆。
因此，本求解器设计了严格且精细的「腰臂自适应空间协调机制」：

1. 拓扑约束与防侧倾锁定 (Topology & Balance Protection):
   - 腰部严格仅开放 2 个主动自由度：偏航/旋转 (Yaw) 与 俯仰 (Pitch)；
   - 彻底去除侧倾 (Roll) 自由度（或将其刚性锁定为 0），杜绝躯干向左右大幅侧倾
     破坏双足支撑多边形的对称性与质心平衡。

2. 运动学收敛驱动的分层级联控制架构 (Convergence-Driven Hierarchical Cascade):
   彻底摒弃基于人工经验标量球形距离 (如 ||p - p_sh|| <= 30cm) 的粗糙启发式判据，
   全面采用机器人机构学通用的物理收敛性分层与零空间姿态惩罚：
   ┌───────────────────────┬─────────────────────────────────────────────────┐
   │ 级联阶段              │ 协同与介入控制行为                              │
   ├───────────────────────┼─────────────────────────────────────────────────┤
   │ 阶段 1: 单臂直立优先  │ 触发级联优先探测：单臂 7-DoF 尝试以腰部直立零位 │
   │ (Arm-First Priority)  │ 闭环求解。只要目标位于单臂真实几何与限位流形内，│
   │                       │ 腰部 100% 保持直立零位 [0.0, 0.0]，毫秒级立即返回│
   │                       │ （零腰动、零多余能耗、绝对维持双足质心稳定）。  │
   ├───────────────────────┼─────────────────────────────────────────────────┤
   │ 阶段 2: 躯干协同扩展  │ 当且仅当单臂物理不可达（或奇异）时，自动无缝激活│
   │ (Coordinated Whole-   │ 9-DoF 协同二次规划 (QP)。在李群流形上通过零空间 │
   │  Body QP)             │ 姿态任务对腰部施加高惩罚阻尼，驱动腰部仅做出    │
   │                       │ 满足末端闭环所需的「最小必要躯干移动」。        │
   └───────────────────────┴─────────────────────────────────────────────────┘

3. 各向异性动态阻尼刚度配置 (Anisotropic Dynamic Damping):
   腰部两轴具有明确的仿生优先级与安全性差异加权 (基准阻尼 [0.080, 0.120] 随 waist_weight 比例缩放)：
   - 偏航优先 (Yaw Priority, 基准阻尼 0.080)：
     旋转轴阻尼较小，优先驱动躯干迎向目标作业面，符合人体工效学；
   - 俯仰抑制 (Pitch Damping, 基准阻尼 0.120)：
     俯仰轴阻尼为偏航轴的 1.5 倍，强力抑制过大幅度的躬身与后仰，
     确保整机重心投影稳定在足底安全支撑区域内。

4. 初猜空间腰部阻尼探索 (Seed Damping in Null-Space):
   在多起点确定性准蒙特卡洛 (QMC) 探索阶段，对腰部候选种子施加 0.30 倍阻尼衰减，
   引导二次规划优化器优先在手臂构型流形中搜索可行解，避免陷入腰部扭曲的局部极值。

数学优化建模 (Mathematical Formulation):
----------------------------------------
在 SE(3) 李群流形上，逆运动学解算被建模为有约束凸二次规划 (Convex QP)：

    min_{v}   (1/2) * || J_ee(q) * v - e_se3 ||_{W_task}^2
            + (1/2) * || v - v_ref ||_{W_posture}^2
    s.t.      (q_min - q) / dt <= v <= (q_max - q) / dt     (硬关节物理限位约束)
              ||v * dt||_inf <= Δq_max                      (信赖域步长截断约束)
"""

import os
import time
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pinocchio as pin
import pink
from pink.limits import ConfigurationLimit
from pink.tasks import FrameTask, PostureTask


# ==============================================================================
# 通用参考基准与默认硬件限位配置
# ==============================================================================

# 标准通用双足人形机器人腰部 2 自由度物理限位 (单位: 弧度) - 仅旋转 (Yaw) 与 俯仰 (Pitch)
DEFAULT_HUMANOID_WAIST_LIMITS: Dict[str, Tuple[float, float]] = {
    "waist_yaw_joint": (-2.6180, 2.6180),    # 偏航 ±150°
    "waist_pitch_joint": (-0.5200, 0.5200),  # 俯仰 ±30°
}

# 标准 7-DoF 机械臂通用硬件限位参考 (单位: 弧度)
DEFAULT_HUMANOID_ARM_LIMITS: Dict[str, Tuple[float, float]] = {
    # 左臂 7-DoF
    "left_shoulder_pitch_joint": (-3.0892, 2.6704),
    "left_shoulder_roll_joint": (-1.5882, 2.2515),
    "left_shoulder_yaw_joint": (-2.6180, 2.6180),
    "left_elbow_joint": (-1.0472, 2.0944),
    "left_wrist_roll_joint": (-1.9722, 1.9722),
    "left_wrist_pitch_joint": (-1.6144, 1.6144),
    "left_wrist_yaw_joint": (-1.6144, 1.6144),
    # 右臂 7-DoF
    "right_shoulder_pitch_joint": (-3.0892, 2.6704),
    "right_shoulder_roll_joint": (-2.2515, 1.5882),
    "right_shoulder_yaw_joint": (-2.6180, 2.6180),
    "right_elbow_joint": (-1.0472, 2.0944),
    "right_wrist_roll_joint": (-1.9722, 1.9722),
    "right_wrist_pitch_joint": (-1.6144, 1.6144),
    "right_wrist_yaw_joint": (-1.6144, 1.6144),
}

# 通用仿生预备就绪姿态 (肘部自然微弯、手腕平直)
DEFAULT_HUMANOID_READY_POSE: Dict[str, np.ndarray] = {
    "left_arm": np.array([0.2, 0.2, 0.0, 0.5, 0.0, 0.0, 0.0], dtype=np.float64),
    "right_arm": np.array([0.2, -0.2, 0.0, 0.5, 0.0, 0.0, 0.0], dtype=np.float64),
}

# 姿态误差对位置误差的综合评估量纲折算权重 (1 rad 姿态误差折合约 0.25 m 位置误差)
WEIGHT_ROT_EVAL: float = 0.25

# 逆运动学算法五大计算流水线阶段定义
IK_PIPELINE_STAGES = [
    {
        "id": 1,
        "name": "ConfigurationLimit",
        "fullName": "Joint Limit Hard Bounds",
        "desc": "在 QP 优化空间绑定硬关节限位凸多面体不等式约束，严格保证物理不越界",
    },
    {
        "id": 2,
        "name": "FrameTask",
        "fullName": "SE(3) Geodesic Task",
        "desc": "末端执行器笛卡尔位置与 SO(3) 测地线流形误差追踪",
    },
    {
        "id": 3,
        "name": "PostureTask",
        "fullName": "Waist-Arm Posture Damping",
        "desc": "各关节运动加权惩罚（2-DoF 腰部自适应动态阻尼，手臂优先伸展）",
    },
    {
        "id": 4,
        "name": "QP Solver",
        "fullName": "ProxQP / OSQP Convex Solver",
        "desc": "原对偶二次规划求解器执行信赖域微秒级迭代求解",
    },
    {
        "id": 5,
        "name": "Gauss-Newton Polish",
        "fullName": "Two-Stage Geodesic Polish",
        "desc": "二阶段李群测地线高斯-牛顿微调，消除正则化阻尼残差",
    },
]


# ==============================================================================
# 状态枚举与诊断结果容器
# ==============================================================================


class IKSolveStatus(str, Enum):
    """逆运动学求解诊断状态枚举。"""

    CONVERGED = "CONVERGED"  # 双重收敛 (位置与姿态均达标，关节未越界)
    POSITION_REACHED_ONLY = "POSITION_REACHED_ONLY"  # 仅位置达标，姿态未达标
    RELAXED_ORIENTATION = "RELAXED_ORIENTATION"  # 姿态受控微松弛容差下收敛 (<=2.0°)
    MAX_ITERATIONS_EXCEEDED = "MAX_ITERATIONS_EXCEEDED"  # 达到最大迭代步数仍未达标
    JOINT_LIMIT_VIOLATION = "JOINT_LIMIT_VIOLATION"  # 撞击关节物理硬限位
    QP_INFEASIBLE = "QP_INFEASIBLE"  # QP 优化器无可行域
    INVALID_INPUT = "INVALID_INPUT"  # 输入参数异常 (NaN 或 Inf)


class IKResult(dict):
    """通用人形机器人逆运动学求解诊断结果对象。

    同时继承 dict 与提供对象属性访问，兼容字典索引 (res['pos_err_mm'])
    与类型安全的属性自省 (res.pos_err_mm, res.status, res.within_limits)。
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.__dict__ = self

    @property
    def is_converged(self) -> bool:
        """是否成功收敛。"""
        return bool(self.get("success", False))

    @property
    def within_joint_limits(self) -> bool:
        """是否严格处于关节物理安全限位内。"""
        return bool(self.get("within_limits", True))

    def __repr__(self) -> str:
        status_val = self.get("status", "UNKNOWN")
        status_str = status_val.value if hasattr(status_val, "value") else str(status_val)
        return (
            f"<IKResult success={self.get('success', False)} status={status_str} "
            f"pos_err={self.get('pos_err_mm', 0.0):.2f}mm rot_err={self.get('rot_err_deg', 0.0):.2f}° "
            f"iters={self.get('iters', 0)} time={self.get('time_ms', 0.0):.2f}ms>"
        )


# ==============================================================================
# 通用运动学模型适配器 (HumanoidKinematicsAdapter)
# ==============================================================================


class HumanoidKinematicsAdapter:
    """通用双足人形机器人运动学模型引擎。

    架构特性：
    - 严格绑定 2-DoF 腰部 (偏航 Yaw, 俯仰 Pitch) 与 7-DoF 机械臂；
    - 基于 pin.buildReducedModel 裁剪生成 9-DoF (2腰+7臂) 与 7-DoF 单臂子模型。
    """

    def __init__(
        self,
        urdf_path: Optional[str] = None,
        waist_joint_names: Optional[List[str]] = None,
        arm_joint_names: Optional[Dict[str, List[str]]] = None,
        ee_frame_names: Optional[Dict[str, str]] = None,
    ) -> None:
        if urdf_path is None:
            default_candidate = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "../../resources/g1/g1_29dof.urdf")
            )
            if os.path.exists(default_candidate):
                urdf_path = default_candidate
            else:
                raise FileNotFoundError("未指定 urdf_path 且默认模型路径不存在，请提供机器人 URDF 路径。")

        self.urdf_path: str = os.path.abspath(urdf_path)
        if not os.path.exists(self.urdf_path):
            raise FileNotFoundError(f"无法找到 URDF 模型文件: {self.urdf_path}")

        self.model: pin.Model = pin.buildModelFromUrdf(self.urdf_path)
        self.data: pin.Data = self.model.createData()

        # ── 1. 2-DoF 腰部关节配置 (严格旋转与俯仰两个自由度) ──
        raw_waist_names = list(waist_joint_names) if waist_joint_names else ["waist_yaw_joint", "waist_pitch_joint"]
        resolved_waist_names: List[str] = []
        for waist_joint_name in raw_waist_names:
            if self.model.existJointName(waist_joint_name):
                resolved_waist_names.append(waist_joint_name)
            else:
                is_yaw = "yaw" in waist_joint_name.lower()
                found_name = next(
                    (
                        name for name in self.model.names
                        if ("waist" in name.lower() or "torso" in name.lower())
                        and ("yaw" in name.lower() if is_yaw else "pitch" in name.lower())
                    ),
                    None,
                )
                if found_name:
                    resolved_waist_names.append(found_name)
                else:
                    raise KeyError(f"模型中未找到指定的腰部关节: {waist_joint_name} (全模型关节: {list(self.model.names)})")

        self.waist_joint_names: List[str] = resolved_waist_names
        self.waist_dim: int = len(self.waist_joint_names)
        if self.waist_dim != 2:
            raise ValueError(f"腰部驱动轴必须严格为 2 自由度 (Yaw, Pitch)，当前检测到 {self.waist_dim} 轴")

        self.waist_q_indices: List[int] = [
            self.model.joints[self.model.getJointId(name)].idx_q
            for name in self.waist_joint_names
        ]

        # ── 2. 手臂关节定义 (左右臂各 7-DoF) ──
        if arm_joint_names is not None:
            self.arm_joint_names = dict(arm_joint_names)
        else:
            self.arm_joint_names = {
                "left_arm": [
                    "left_shoulder_pitch_joint",
                    "left_shoulder_roll_joint",
                    "left_shoulder_yaw_joint",
                    "left_elbow_joint",
                    "left_wrist_roll_joint",
                    "left_wrist_pitch_joint",
                    "left_wrist_yaw_joint",
                ],
                "right_arm": [
                    "right_shoulder_pitch_joint",
                    "right_shoulder_roll_joint",
                    "right_shoulder_yaw_joint",
                    "right_elbow_joint",
                    "right_wrist_roll_joint",
                    "right_wrist_pitch_joint",
                    "right_wrist_yaw_joint",
                ],
            }

        for arm, joint_names in self.arm_joint_names.items():
            if len(joint_names) != 7:
                raise ValueError(f"{arm} 关节数必须为 7 (当前: {len(joint_names)})")
            for joint_name in joint_names:
                if not self.model.existJointName(joint_name):
                    raise KeyError(f"模型中未找到指定的机械臂关节: {joint_name} (臂: {arm})")

        # ── 3. 末端 Frame 定义 ──
        if ee_frame_names is not None:
            self.ee_frame_names = dict(ee_frame_names)
        else:
            self.ee_frame_names = {
                "left_arm": (
                    "left_wrist_yaw_link"
                    if self.model.existFrame("left_wrist_yaw_link")
                    else "left_wrist_link"
                ),
                "right_arm": (
                    "right_wrist_yaw_link"
                    if self.model.existFrame("right_wrist_yaw_link")
                    else "right_wrist_link"
                ),
            }

        for arm, frame_name in self.ee_frame_names.items():
            if not self.model.existFrame(frame_name):
                raise KeyError(f"模型中未找到指定的末端坐标系: {frame_name} (臂: {arm})")

        self.ee_frame_ids: Dict[str, int] = {
            arm: self.model.getFrameId(frame_name) for arm, frame_name in self.ee_frame_names.items()
        }

        # ── 4. 提取物理限位 ──
        self.waist_limits: Tuple[np.ndarray, np.ndarray] = (
            np.array([self.model.lowerPositionLimit[idx] for idx in self.waist_q_indices], dtype=np.float64),
            np.array([self.model.upperPositionLimit[idx] for idx in self.waist_q_indices], dtype=np.float64),
        )

        self.arm_q_indices: Dict[str, List[int]] = {
            arm: [self.model.joints[self.model.getJointId(name)].idx_q for name in jnames]
            for arm, jnames in self.arm_joint_names.items()
        }

        self.limits_arm: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for arm, indices in self.arm_q_indices.items():
            lower = np.array([self.model.lowerPositionLimit[i] for i in indices], dtype=np.float64)
            upper = np.array([self.model.upperPositionLimit[i] for i in indices], dtype=np.float64)
            self.limits_arm[arm] = (lower, upper)

        # ── 5. 9-DoF 协同链 (2-DoF 腰部 + 7-DoF 机械臂) ──
        self.chain_coord_joint_names: Dict[str, List[str]] = {
            arm: self.waist_joint_names + self.arm_joint_names[arm]
            for arm in self.arm_joint_names
        }
        self.limits_coord: Dict[str, Tuple[np.ndarray, np.ndarray]] = {
            arm: (
                np.concatenate([self.waist_limits[0], self.limits_arm[arm][0]]),
                np.concatenate([self.waist_limits[1], self.limits_arm[arm][1]]),
            )
            for arm in self.arm_joint_names
        }

        # ── 6. 构建轻量化裁剪子模型 (9-DoF 与 7-DoF) ──
        self.model_coord: Dict[str, pin.Model] = {}
        self.data_coord: Dict[str, pin.Data] = {}
        self.ee_frame_ids_coord: Dict[str, int] = {}

        self.model_7dof: Dict[str, pin.Model] = {}
        self.data_7dof: Dict[str, pin.Data] = {}
        self.ee_frame_ids_7dof: Dict[str, int] = {}

        q_neutral = pin.neutral(self.model)
        for arm in self.arm_joint_names:
            ee_name = self.ee_frame_names[arm]
            # 1. 9-DoF 协同链子模型 (2-DoF 腰部 + 7-DoF 机械臂)
            self.model_coord[arm], self.data_coord[arm], self.ee_frame_ids_coord[arm] = (
                self._reduce_model(self.chain_coord_joint_names[arm], ee_name, q_neutral)
            )
            # 2. 7-DoF 单臂子模型
            self.model_7dof[arm], self.data_7dof[arm], self.ee_frame_ids_7dof[arm] = (
                self._reduce_model(self.arm_joint_names[arm], ee_name, q_neutral)
            )

    def _reduce_model(
        self, active_joints: List[str], ee_frame_name: str, q_ref: np.ndarray
    ) -> Tuple[pin.Model, pin.Data, int]:
        """按指定活动关节列表裁剪抽取子模型，并创建对应 Data 与末端 Frame ID。"""
        locked_joint_ids = [
            self.model.getJointId(joint_name)
            for joint_name in self.model.names[1:]
            if joint_name not in active_joints
        ]
        reduced_model = pin.buildReducedModel(self.model, locked_joint_ids, q_ref)
        reduced_data = reduced_model.createData()
        ee_frame_id = reduced_model.getFrameId(ee_frame_name)
        return reduced_model, reduced_data, ee_frame_id

    # --------------------------------------------------------------------------
    # 正向运动学推演接口 (Forward Kinematics Queries)
    # --------------------------------------------------------------------------

    def forward_kinematics(
        self, arm: str, q: np.ndarray, is_coord: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """通用正向运动学推演 (返回末端位置平移向量与 3x3 旋转矩阵)。

        :param arm: 'left_arm' 或 'right_arm'
        :param q: 关节角度向量 (9-DoF 协同链或 7-DoF 单臂)
        :param is_coord: 是否为 9-DoF 协同模式 (True: 9-DoF, False: 7-DoF)
        """
        model = self.model_coord[arm] if is_coord else self.model_7dof[arm]
        data = self.data_coord[arm] if is_coord else self.data_7dof[arm]
        ee_frame_id = self.ee_frame_ids_coord[arm] if is_coord else self.ee_frame_ids_7dof[arm]
        pin.forwardKinematics(model, data, np.asarray(q, dtype=np.float64))
        pin.updateFramePlacements(model, data)
        placement = data.oMf[ee_frame_id]
        return placement.translation.copy(), placement.rotation.copy()

    def forward_kinematics_coord(
        self,
        arm: str,
        waist_q: np.ndarray,
        arm_q: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """计算 9-DoF (2腰 + 7臂) 协同正向运动学 (委托至通用 forward_kinematics)。"""
        coordinated_q = np.concatenate([
            np.asarray(waist_q, dtype=np.float64),
            np.asarray(arm_q, dtype=np.float64),
        ])
        return self.forward_kinematics(arm, coordinated_q, is_coord=True)

    def forward_kinematics_arm(
        self,
        arm: str,
        arm_q: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """计算 7-DoF 单臂正向运动学 (委托至通用 forward_kinematics)。"""
        return self.forward_kinematics(arm, arm_q, is_coord=False)


# ==============================================================================
# 通用双足人形机器人逆运动学求解器 (HumanoidPinkIKSolver)
# ==============================================================================


class HumanoidPinkIKSolver:
    """通用人形机器人 2-DoF 腰部 (旋转 + 俯仰) + 7-DoF 机械臂逆解优化求解器。

    特性：
    1. 腰部严格配置为 2 自由度: 旋转/偏航 (Yaw) 与 俯仰 (Pitch)；
    2. 机械臂配置为 7 自由度高灵巧单臂构型；
    3. 全面基于 Pink 4.4.0 与 ProxQP 凸二次规划内核；
    4. 显式绑定 ConfigurationLimit，从优化数学源头保证物理硬限位；
    5. 运动学收敛驱动的分层级联控制：单臂直立优先闭环快速探测与李群零空间各向异性动态阻尼；
    6. 二阶段李群高斯-牛顿微调抛光 (消除 QP 正则化残差)。
    """

    # ==========================================================================
    # 模块 1: 初始化与正向运动学接口 (Initialization & Forward Kinematics)
    # ==========================================================================

    def __init__(
        self,
        urdf_path: Optional[str] = None,
        kinematics: Optional[HumanoidKinematicsAdapter] = None,
        waist_joint_names: Optional[List[str]] = None,
        ready_pose: Optional[Dict[str, np.ndarray]] = None,
    ) -> None:
        """初始化通用人形机器人逆解求解器。

        :param urdf_path: 机器人 URDF 文件路径 (若 kinematics 为 None 时使用)
        :param kinematics: 可选的预建运动学引擎适配器
        :param waist_joint_names: 自定义 2-DoF 腰部关节名 (默认: ['waist_yaw_joint', 'waist_pitch_joint'])
        :param ready_pose: 自定义就绪姿态字典 {"left_arm": array(7), "right_arm": array(7)}
        """
        if kinematics is None:
            self.kin = HumanoidKinematicsAdapter(
                urdf_path=urdf_path,
                waist_joint_names=waist_joint_names,
            )
        else:
            self.kin = kinematics

        self.ready_pose = ready_pose or DEFAULT_HUMANOID_READY_POSE

        # 核心运动学属性委托
        self.urdf_path: str = self.kin.urdf_path
        self.model: pin.Model = self.kin.model
        self.data: pin.Data = self.kin.data
        self.waist_joint_names: List[str] = self.kin.waist_joint_names
        self.waist_dim: int = self.kin.waist_dim  # 严格为 2
        self.waist_limits: Tuple[np.ndarray, np.ndarray] = self.kin.waist_limits

        self.arm_joint_names: Dict[str, List[str]] = self.kin.arm_joint_names
        self.ee_frame_names: Dict[str, str] = self.kin.ee_frame_names
        self.chain_coord_joint_names: Dict[str, List[str]] = self.kin.chain_coord_joint_names
        self.limits_arm: Dict[str, Tuple[np.ndarray, np.ndarray]] = self.kin.limits_arm
        self.limits_coord: Dict[str, Tuple[np.ndarray, np.ndarray]] = self.kin.limits_coord

    def forward_kinematics_arm(self, arm: str, arm_q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """计算 7-DoF 单臂正向运动学。"""
        return self.kin.forward_kinematics_arm(arm, arm_q)

    def forward_kinematics_coord(
        self, arm: str, waist_q: np.ndarray, arm_q: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """计算 9-DoF (2腰 + 7臂) 协同正向运动学。"""
        return self.kin.forward_kinematics_coord(arm, waist_q, arm_q)

    # ==========================================================================
    # 模块 2: 输入校验、兜底响应与求解上下文预处理 (Validation & Preparation)
    # ==========================================================================

    @staticmethod
    def _normalize_target_pose(
        target_pos: Any,
        target_rot: Optional[np.ndarray] = None,
        target_rpy: Optional[Union[List[float], np.ndarray]] = None,
    ) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray], Optional[str]]:
        """统一解析并校验笛卡尔末端目标位姿 (位置、RPY/旋转矩阵)。

        :return: (is_valid, pos_array(3,), rot_matrix(3,3) or None, error_message)
        """
        # 1. 校验位置
        position_array = np.asarray(target_pos, dtype=np.float64)
        if position_array.shape != (3,) or not np.all(np.isfinite(position_array)):
            return False, None, None, "Invalid target_pos: must be 3-element finite float array"

        # 2. 校验与转换姿态
        rotation_matrix: Optional[np.ndarray] = None
        if target_rot is None and target_rpy is not None:
            rpy_array = np.asarray(target_rpy, dtype=np.float64)
            if rpy_array.shape != (3,) or not np.all(np.isfinite(rpy_array)):
                return False, None, None, "Invalid target_rpy: must be 3-element finite float array"
            rotation_matrix = pin.rpy.rpyToMatrix(float(rpy_array[0]), float(rpy_array[1]), float(rpy_array[2]))
        elif target_rot is not None:
            rotation_matrix = np.asarray(target_rot, dtype=np.float64)
            if rotation_matrix.shape != (3, 3) or not np.all(np.isfinite(rotation_matrix)):
                return False, None, None, "Invalid target_rot: must be 3x3 finite float matrix"
            det_rot = float(np.linalg.det(rotation_matrix))
            if abs(det_rot - 1.0) > 0.05:
                return False, None, None, f"Invalid target_rot: determinant={det_rot:.3f} deviates from SO(3)"

        return True, position_array, rotation_matrix, None

    def _create_invalid_input_result(
        self,
        arm: str,
        is_coord: bool,
        error: str,
    ) -> IKResult:
        """构造非法输入时的默认失败结果结构体。"""
        ready_arm = self.ready_pose[arm]
        waist_zero = np.zeros(self.waist_dim)
        default_solution_q = np.concatenate([waist_zero, ready_arm]) if is_coord else ready_arm.copy()
        return IKResult(
            success=False,
            status=IKSolveStatus.INVALID_INPUT,
            q_solution=default_solution_q,
            waist_solution=waist_zero.copy() if is_coord else None,
            arm_solution=ready_arm.copy(),
            pos_err_mm=999.0,
            rot_err_deg=180.0,
            within_limits=True,
            time_ms=0.0,
            iters=0,
            error=error,
        )

    def _prepare_solver_context(
        self,
        arm: str,
        is_coord: bool,
        custom_limits: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        waist_weights: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """统一构建求解限位 (lower_limit, upper_limit)、就绪姿态 ready_q 与关节代价权重 cost_posture。"""
        if is_coord:
            lower_limit, upper_limit = custom_limits if custom_limits is not None else self.limits_coord[arm]
            ready_q = np.concatenate([np.zeros(self.waist_dim), self.ready_pose[arm]])
            cost_posture = np.ones(len(ready_q)) * 1e-4
            if waist_weights is not None:
                cost_posture[:self.waist_dim] = waist_weights
        else:
            lower_limit, upper_limit = custom_limits if custom_limits is not None else self.limits_arm[arm]
            ready_q = self.ready_pose[arm]
            cost_posture = np.ones(len(ready_q)) * 1e-4
        return lower_limit, upper_limit, ready_q, cost_posture

    def _build_seed_chain(
        self,
        is_coord: bool,
        arm: str,
        ready_q: np.ndarray,
        lower_limit: np.ndarray,
        upper_limit: np.ndarray,
        seed_q: Optional[np.ndarray] = None,
        seed_waist: Optional[np.ndarray] = None,
        seed_arm: Optional[np.ndarray] = None,
    ) -> Tuple[List[np.ndarray], List[str]]:
        """构建确定性启发式多种子链 (Warm Start -> Ready Pose -> 4组 QMC 空间网格)。"""
        seed_chain: List[np.ndarray] = []
        seed_names: List[str] = []

        # 1. Warm Start
        if is_coord:
            if seed_waist is not None or seed_arm is not None:
                ready_arm = self.ready_pose[arm]
                seed_waist_clamped = np.clip(
                    np.asarray(
                        seed_waist if seed_waist is not None else np.zeros(self.waist_dim),
                        dtype=np.float64,
                    ),
                    self.waist_limits[0],
                    self.waist_limits[1],
                )
                seed_arm_clamped = np.clip(
                    np.asarray(
                        seed_arm if seed_arm is not None else ready_arm,
                        dtype=np.float64,
                    ),
                    self.limits_arm[arm][0],
                    self.limits_arm[arm][1],
                )
                seed_chain.append(np.concatenate([seed_waist_clamped, seed_arm_clamped]))
                seed_names.append("Warm Start")
        else:
            if seed_q is not None:
                seed_chain.append(
                    np.clip(np.asarray(seed_q, dtype=np.float64), lower_limit, upper_limit)
                )
                seed_names.append("Warm Start")

        # 2. Ready Pose
        seed_chain.append(np.clip(ready_q.copy(), lower_limit, upper_limit))
        seed_names.append("Ready Pose")

        # 3. 确定性低差异准蒙特卡洛多起点探索 (4 组均匀分位网格)
        for grid_idx in range(4):
            grid_ratio = (grid_idx + 1) / 5.0
            qmc_seed = lower_limit + grid_ratio * (upper_limit - lower_limit)
            if is_coord:
                qmc_seed[:self.waist_dim] *= 0.30  # 对腰部初猜施加阻尼，优先探索手臂空间
            seed_chain.append(np.clip(qmc_seed, lower_limit, upper_limit))
            seed_names.append(f"QMC Grid #{grid_idx + 1}")

        return seed_chain, seed_names

    # ==========================================================================
    # 模块 3: 二次规划求解内核、微调抛光与解算遥测 (Numerical QP Core, Polish & Telemetry)
    # ==========================================================================

    @staticmethod
    def _polish_gauss_newton(
        model: pin.Model,
        data: pin.Data,
        ee_frame_id: int,
        initial_q: np.ndarray,
        target_pos: np.ndarray,
        target_rot: Optional[np.ndarray],
        lower_limit: np.ndarray,
        upper_limit: np.ndarray,
        max_polish_steps: int = 2,
    ) -> Tuple[np.ndarray, float, float]:
        """二阶段李群高斯-牛顿微调抛光器，消除 QP 正则化残差。"""
        has_rot = target_rot is not None
        pin.forwardKinematics(model, data, initial_q)
        pin.updateFramePlacements(model, data)
        current_pose = data.oMf[ee_frame_id]
        pos_err = float(np.linalg.norm(target_pos - current_pose.translation))
        rot_err = float(np.linalg.norm(pin.log3(target_rot @ current_pose.rotation.T))) if has_rot else 0.0

        best_q = initial_q.copy()
        best_err = pos_err + rot_err * WEIGHT_ROT_EVAL
        best_rot_err = rot_err

        if max_polish_steps <= 0:
            return best_q, best_err, best_rot_err

        task_dim = 6 if has_rot else 3
        damping_matrix = 1e-6 * np.eye(task_dim)
        current_q = initial_q.copy()

        for _ in range(max_polish_steps):
            # data 中已保存有上一轮（或初始）current_q 对应的正运动学，无需重复调用 forwardKinematics
            current_pose = data.oMf[ee_frame_id]
            pos_error_vec = target_pos - current_pose.translation
            if has_rot:
                rot_error_vec = pin.log3(target_rot @ current_pose.rotation.T)
                spatial_error_vec = np.concatenate([pos_error_vec, rot_error_vec])
                jacobian = pin.computeFrameJacobian(
                    model, data, current_q, ee_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
                )
            else:
                spatial_error_vec = pos_error_vec
                jacobian = pin.computeFrameJacobian(
                    model, data, current_q, ee_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
                )[:3, :]

            # 阻尼最小二乘求解，安全捕获病态奇异代数异常
            try:
                joint_delta_q = jacobian.T @ np.linalg.solve(
                    jacobian @ jacobian.T + damping_matrix, spatial_error_vec
                )
            except (np.linalg.LinAlgError, ValueError):
                break

            if not np.all(np.isfinite(joint_delta_q)):
                break

            joint_delta_q = np.clip(joint_delta_q, -0.04, 0.04)
            candidate_q = np.clip(current_q + joint_delta_q, lower_limit, upper_limit)

            pin.forwardKinematics(model, data, candidate_q)
            pin.updateFramePlacements(model, data)
            candidate_pose = data.oMf[ee_frame_id]
            candidate_pos_err = float(np.linalg.norm(target_pos - candidate_pose.translation))
            candidate_rot_err = (
                float(np.linalg.norm(pin.log3(target_rot @ candidate_pose.rotation.T)))
                if has_rot
                else 0.0
            )
            candidate_total_err = candidate_pos_err + candidate_rot_err * WEIGHT_ROT_EVAL

            if candidate_total_err < best_err:
                current_q = candidate_q
                best_q = candidate_q.copy()
                best_err = candidate_total_err
                best_rot_err = candidate_rot_err
            else:
                break  # 目标未继续改善，提前剪枝退出

        return best_q, best_err, best_rot_err

    def _extract_solution_telemetry(
        self,
        arm: str,
        waist_q: Optional[np.ndarray],
        arm_q: np.ndarray,
        target_rot: Optional[np.ndarray] = None,
        is_coord: bool = True,
        fk_pos: Optional[np.ndarray] = None,
        fk_rot: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """提取解算构型下的空间位姿误差与关节限位裕度 (支持复用已计算的 FK 结果)。"""
        if is_coord:
            waist_vector = np.asarray(
                waist_q if waist_q is not None else np.zeros(self.waist_dim),
                dtype=np.float64,
            )
            active_q = np.concatenate([waist_vector, np.asarray(arm_q, dtype=np.float64)])
            lower_limits, upper_limits = self.limits_coord[arm]
            active_joint_names = self.chain_coord_joint_names[arm]
            if fk_pos is None or fk_rot is None:
                fk_pos, fk_rot = self.kin.forward_kinematics(arm, active_q, is_coord=True)
        else:
            active_q = np.asarray(arm_q, dtype=np.float64)
            lower_limits, upper_limits = self.limits_arm[arm]
            active_joint_names = list(self.arm_joint_names[arm])
            if fk_pos is None or fk_rot is None:
                fk_pos, fk_rot = self.kin.forward_kinematics(arm, active_q, is_coord=False)

        actual_rpy_rad = pin.rpy.matrixToRpy(fk_rot)
        actual_rpy_deg = [round(float(np.degrees(v)), 2) for v in actual_rpy_rad]
        actual_quat = pin.Quaternion(fk_rot)

        target_rpy_deg = None
        rot_err_deg = 0.0
        if target_rot is not None:
            target_rpy_rad = pin.rpy.matrixToRpy(target_rot)
            target_rpy_deg = [round(float(np.degrees(v)), 2) for v in target_rpy_rad]
            relative_rot = target_rot @ fk_rot.T
            rot_err_deg = round(float(np.degrees(np.linalg.norm(pin.log3(relative_rot)))), 2)

        # 计算当前构型距物理限位的安全裕度及最危险关节
        joint_margins = np.minimum(active_q - lower_limits, upper_limits - active_q)
        critical_idx = int(np.argmin(joint_margins))
        margin_rad = float(joint_margins[critical_idx])
        margin_deg = round(float(np.degrees(margin_rad)), 2)
        within_limits = bool(
            np.all(active_q >= lower_limits - 1e-4) and np.all(active_q <= upper_limits + 1e-4)
        )

        return {
            "within_limits": within_limits,
            "joint_limit_margin": round(margin_rad, 4),
            "joint_limit_margin_deg": margin_deg,
            "critical_joint": active_joint_names[critical_idx],
            "joint_limits_bounds": {
                "lower": [round(float(v), 4) for v in lower_limits],
                "upper": [round(float(v), 4) for v in upper_limits],
            },
            "actual_pos": [round(float(v), 5) for v in fk_pos],
            "actual_rpy_deg": actual_rpy_deg,
            "actual_quat": [
                round(float(v), 5) for v in [actual_quat.x, actual_quat.y, actual_quat.z, actual_quat.w]
            ],
            "target_rpy_deg": target_rpy_deg,
            "rot_err_deg": rot_err_deg,
        }

    def _solve_qp_core(
        self,
        arm: str,
        is_coord: bool,
        target_pos: np.ndarray,
        target_rot: Optional[np.ndarray],
        seed_chain: List[np.ndarray],
        seed_names: List[str],
        ready_q: np.ndarray,
        cost_posture: np.ndarray,
        lower_limit: np.ndarray,
        upper_limit: np.ndarray,
        pos_tol: float,
        rot_tol: float,
        max_iters: int,
        max_step: float,
        start_time: float,
        allow_relaxation: bool = False,
    ) -> Tuple[bool, np.ndarray, IKResult]:
        """通用 SE(3) 流形凸二次规划优化内核 (支持 7-DoF 与 9-DoF 协同链)。"""
        has_rot = target_rot is not None
        model = self.kin.model_coord[arm] if is_coord else self.kin.model_7dof[arm]
        data = self.kin.data_coord[arm] if is_coord else self.kin.data_7dof[arm]
        ee_frame = self.ee_frame_names[arm]
        ee_frame_id = self.kin.ee_frame_ids_coord[arm] if is_coord else self.kin.ee_frame_ids_7dof[arm]
        target_pose = pin.SE3(target_rot if has_rot else np.eye(3), target_pos)

        # 边界入参清洗与稳健防护
        pos_tol = max(1e-6, float(pos_tol))
        rot_tol = max(1e-5, float(rot_tol))
        max_iters = max(1, int(max_iters))
        max_step = max(1e-3, float(max_step)) if max_step and max_step > 0 else 0.20

        # ── Pink 任务栈 ──
        task_ee = FrameTask(
            ee_frame,
            position_cost=10.0,
            orientation_cost=5.0 if has_rot else 0.0,
            lm_damping=1e-4,
        )
        task_ee.set_target(target_pose)

        task_posture = PostureTask(cost=cost_posture)
        task_posture.set_target(ready_q)

        tasks = [task_ee, task_posture]

        # 绑定硬件硬关节限位凸多面体不等式 (ConfigurationLimit)
        joint_limits = [ConfigurationLimit(model, config_limit_gain=0.5)]

        best_q: Optional[np.ndarray] = None
        best_err = float("inf")
        total_iters = 0
        dt = 0.25

        config = pink.Configuration(model, data, seed_chain[0].copy())

        for seed_idx, seed in enumerate(seed_chain):
            if total_iters >= max_iters:
                break

            clamped_seed = np.clip(seed.copy(), lower_limit, upper_limit)
            config.update(clamped_seed)
            prev_pos_err: Optional[float] = None
            stall_count = 0

            # 种子预算自适应分配：主种子 (初始解/就绪位) 分配充裕预算 (<=70%)，其余探索种子快速试探 (<=35%)
            is_primary_seed = (seed_idx == 0) or (seed_idx == 1 and seed_names[0] == "Warm Start")
            remaining_total = max_iters - total_iters
            if is_primary_seed:
                seed_iter_budget = min(remaining_total, max(22, int(max_iters * 0.70)))
            else:
                seed_iter_budget = min(remaining_total, max(10, int(max_iters * 0.35)))

            for _ in range(seed_iter_budget):
                total_iters += 1

                # 关节就绪态引力退火：末端逼近目标 (<6mm) 时衰减 PostureTask 权重，消除末端稳态残差
                if prev_pos_err is not None and prev_pos_err < 0.006:
                    alpha_posture = max(0.01, prev_pos_err / 0.006)
                    task_posture.cost = cost_posture * alpha_posture
                else:
                    task_posture.cost = cost_posture.copy()

                # 1. QP 求解：扁平遍历多求解器降级容错
                joint_velocity = None
                for qp_solver in ("proxqp", "osqp"):
                    try:
                        joint_velocity = pink.solve_ik(config, tasks, dt, solver=qp_solver, limits=joint_limits)
                        break
                    except Exception:
                        continue
                if joint_velocity is None:
                    break

                # 2. 信赖域步长截断 (L-inf 范数等比限幅)
                step_max = float(np.max(np.abs(joint_velocity))) * dt
                if max_step and step_max > max_step:
                    joint_velocity *= (max_step / step_max)

                # 3. 李群积分并确保硬限位安全 (按需更新)
                config.integrate_inplace(joint_velocity, dt)
                current_q = np.clip(config.q, lower_limit, upper_limit)
                if (config.q != current_q).any():
                    config.update(current_q)

                # 4. 误差度量 (统一国际单位米)
                current_pose = config.get_transform_frame_to_world(ee_frame)
                pos_err = float(np.linalg.norm(target_pos - current_pose.translation))
                rot_err = (
                    float(np.linalg.norm(pin.log3(target_rot @ current_pose.rotation.T)))
                    if has_rot
                    else 0.0
                )

                # 5. 记录最优解 (WEIGHT_ROT_EVAL 为姿态-位置量纲折算系数)
                total_err = pos_err + rot_err * WEIGHT_ROT_EVAL
                if total_err < best_err:
                    best_err = total_err
                    best_q = current_q.copy()

                # 6. 收敛判定
                if pos_err < pos_tol and (not has_rot or rot_err < rot_tol):
                    break

                # 7. 停滞阻断判定 (位移改善不足 0.05mm 累计 4 次跳出)
                delta_pos = (prev_pos_err - pos_err) if prev_pos_err is not None else 0.0
                prev_pos_err = pos_err
                stall_count = (stall_count + 1) if delta_pos < 5e-5 else 0
                if stall_count >= 4:
                    break

            if best_q is not None and best_err < pos_tol + (rot_tol * WEIGHT_ROT_EVAL if has_rot else 0.0):
                break

        # ── 二阶段微调抛光 ──
        target_polish_q = best_q if best_q is not None else ready_q.copy()
        if best_err < 0.05:
            best_q, best_err, _ = self._polish_gauss_newton(
                model=model,
                data=data,
                ee_frame_id=ee_frame_id,
                initial_q=target_polish_q,
                target_pos=target_pos,
                target_rot=target_rot,
                lower_limit=lower_limit,
                upper_limit=upper_limit,
            )

        if best_q is None:
            best_q = ready_q.copy()

        best_q = np.clip(best_q, lower_limit, upper_limit)

        pin.forwardKinematics(model, data, best_q)
        pin.updateFramePlacements(model, data)
        final_pose = data.oMf[ee_frame_id]
        final_pos_err = float(np.linalg.norm(target_pos - final_pose.translation))
        final_rot_err = (
            float(np.linalg.norm(pin.log3(target_rot @ final_pose.rotation.T)))
            if has_rot
            else 0.0
        )
        is_within_limits = bool(
            np.all(best_q >= lower_limit - 1e-4) and np.all(best_q <= upper_limit + 1e-4)
        )

        relaxed_ok = allow_relaxation and has_rot and (final_pos_err < pos_tol) and (final_rot_err < 3.5e-2)
        is_success = (
            ((final_pos_err < pos_tol and (not has_rot or final_rot_err < rot_tol)) or relaxed_ok)
            and is_within_limits
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        # 分割腰部解与手臂解
        waist_solution = best_q[:self.waist_dim].copy() if is_coord else None
        arm_solution = best_q[self.waist_dim:].copy() if is_coord else best_q.copy()

        # 复用已计算的 final_pose，消除 telemetry 内的重复 FK 计算
        telemetry = self._extract_solution_telemetry(
            arm=arm,
            waist_q=waist_solution,
            arm_q=arm_solution,
            target_rot=target_rot if has_rot else None,
            is_coord=is_coord,
            fk_pos=final_pose.translation,
            fk_rot=final_pose.rotation,
        )

        # 判定最终解算状态分类
        if not is_within_limits:
            solve_status = IKSolveStatus.JOINT_LIMIT_VIOLATION
        elif is_success:
            solve_status = IKSolveStatus.CONVERGED
        elif allow_relaxation and has_rot and final_pos_err < pos_tol and final_rot_err < 3.5e-2:
            solve_status = IKSolveStatus.RELAXED_ORIENTATION
        elif final_pos_err < pos_tol and has_rot and final_rot_err >= rot_tol:
            solve_status = IKSolveStatus.POSITION_REACHED_ONLY
        else:
            solve_status = IKSolveStatus.MAX_ITERATIONS_EXCEEDED

        if is_coord:
            mode_str = "9DOF_6DOF_POSE" if has_rot else "9DOF_3DOF_POS"
        else:
            mode_str = "7DOF_6DOF_POSE" if has_rot else "7DOF_3DOF_POS"

        ik_res = IKResult(
            success=bool(is_success),
            status=solve_status,
            q_solution=best_q.copy(),
            waist_solution=waist_solution,
            arm_solution=arm_solution,
            iters=total_iters,
            time_ms=elapsed_ms,
            pos_err_mm=round(final_pos_err * 1000.0, 4),
            pipeline_stages=IK_PIPELINE_STAGES,
            mode=mode_str,
            **telemetry,
        )
        return is_success, best_q, ik_res

    # ==========================================================================
    # 模块 4: 对外高阶逆运动学求解接口 (Public Inverse Kinematics APIs)
    # ==========================================================================

    def solve_ik(
        self,
        arm: str,
        target_pos: Union[List[float], np.ndarray],
        target_rot: Optional[np.ndarray] = None,
        target_rpy: Optional[Union[List[float], np.ndarray]] = None,
        seed_q: Optional[np.ndarray] = None,
        pos_tol: float = 1e-3,  # 1.0 mm
        rot_tol: float = 2e-2,  # ~1.15°
        max_iters: int = 35,
        max_step: float = 0.20,
        custom_limits: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        allow_relaxation: bool = False,
        **kwargs: Any,
    ) -> Tuple[bool, np.ndarray, IKResult]:
        """7-DoF 单臂逆运动学求解。

        :param arm: 操作臂 ("left_arm" 或 "right_arm")
        :param target_pos: 目标空间坐标 [x, y, z] (米)
        :param target_rot: 可选的 3x3 目标旋转矩阵
        :param target_rpy: 可选的目标欧拉角 [roll, pitch, yaw] (弧度)
        :param seed_q: 初始猜测关节角 (7-DoF)
        :param pos_tol: 位置收敛容差 (米, 默认 1e-3, 即 1.0mm)
        :param rot_tol: 姿态收敛容差 (弧度, 默认 2e-2, 约 1.15°)
        :param max_iters: 最大优化迭代步数
        :param max_step: 单步最大关节步长截断 (rad)
        :param custom_limits: 可选的自定义关节限位 (lower_limits, upper_limits)
        :param allow_relaxation: 是否允许姿态微松弛收敛 (<=2.0°)
        :return: (is_success, q_solution, ik_result_info)
        """
        start_time = time.perf_counter()
        is_valid, normalized_pos, normalized_rot, error_message = self._normalize_target_pose(
            target_pos=target_pos, target_rot=target_rot, target_rpy=target_rpy
        )
        if not is_valid:
            failure_result = self._create_invalid_input_result(
                arm=arm, is_coord=False, error=error_message or "Invalid target pose"
            )
            return False, self.ready_pose[arm].copy(), failure_result

        lower_limit, upper_limit, ready_q, cost_posture = self._prepare_solver_context(
            arm=arm, is_coord=False, custom_limits=custom_limits
        )

        seed_chain, seed_names = self._build_seed_chain(
            is_coord=False,
            arm=arm,
            ready_q=ready_q,
            lower_limit=lower_limit,
            upper_limit=upper_limit,
            seed_q=seed_q,
        )

        return self._solve_qp_core(
            arm=arm,
            is_coord=False,
            target_pos=normalized_pos,
            target_rot=normalized_rot,
            seed_chain=seed_chain,
            seed_names=seed_names,
            ready_q=ready_q,
            cost_posture=cost_posture,
            lower_limit=lower_limit,
            upper_limit=upper_limit,
            pos_tol=pos_tol,
            rot_tol=rot_tol,
            max_iters=max_iters,
            max_step=max_step,
            start_time=start_time,
            allow_relaxation=allow_relaxation,
        )

    # ── 9-DoF (2-DoF 腰部 + 7-DoF 手臂) 协同逆运动学求解 (solve_coordinated_ik) ──

    def solve_coordinated_ik(
        self,
        arm: str,
        target_pos: Union[List[float], np.ndarray],
        target_rot: Optional[np.ndarray] = None,
        target_rpy: Optional[Union[List[float], np.ndarray]] = None,
        seed_waist: Optional[np.ndarray] = None,
        seed_arm: Optional[np.ndarray] = None,
        pos_tol: float = 1e-3,  # 1.0 mm
        rot_tol: float = 2.0e-2,  # ~1.15°
        max_iters: int = 35,
        waist_weight: float = 10.0,
        max_step: float = 0.18,
        cascade_upright: bool = True,
        custom_limits: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        allow_relaxation: bool = False,
        **kwargs: Any,
    ) -> Tuple[bool, np.ndarray, np.ndarray, IKResult]:
        """9-DoF (2-DoF 腰部 + 7-DoF 机械臂) 躯干-手臂自适应协同逆运动学求解。

        腰部自由度特性：
        - 仅包含 2 自由度：旋转 (Yaw) 与 俯仰 (Pitch)；
        - 无侧倾 (Roll) 自由度；
        - 解算返回的 waist_solution 为长度为 2 的浮点数组 [waist_yaw, waist_pitch]。

        :param arm: 操作臂 ("left_arm" 或 "right_arm")
        :param target_pos: 目标末端空间坐标 [x, y, z] (米)
        :param target_rot: 可选的 3x3 目标旋转矩阵
        :param target_rpy: 可选的目标欧拉角 [roll, pitch, yaw] (弧度)
        :param seed_waist: 腰部 2-DoF 初始种子 [yaw, pitch]
        :param seed_arm: 手臂 7-DoF 初始种子
        :param pos_tol: 位置收敛容差 (米, 默认 1e-3, 即 1.0mm)
        :param rot_tol: 姿态收敛容差 (弧度, 默认 2.0e-2, 约 1.15°)
        :param max_iters: 最大优化迭代步数 (默认 35)
        :param waist_weight: 腰部惩罚基准 (默认 10.0，数值越大越抑制腰部运动)
        :param max_step: 单步最大关节步长截断 (rad, 默认 0.18)
        :param cascade_upright: 是否启用单臂直立优先探测快速通道 (若单臂物理可达则腰部严格保持直立零位)
        :param custom_limits: 可选的自定义联合限位 (lower_limits, upper_limits)
        :param allow_relaxation: 是否允许姿态微松弛收敛 (<=2.0°)
        :return: (is_success, waist_solution(2,), arm_solution(7,), ik_result_info)
        """
        start_time = time.perf_counter()
        is_valid, normalized_pos, normalized_rot, error_message = self._normalize_target_pose(
            target_pos=target_pos, target_rot=target_rot, target_rpy=target_rpy
        )
        if not is_valid:
            failure_result = self._create_invalid_input_result(
                arm=arm, is_coord=True, error=error_message or "Invalid target pose"
            )
            return False, np.zeros(self.waist_dim), self.ready_pose[arm].copy(), failure_result

        # ── 1. 级联分层直立优先探测 (Arm-First Priority by Physical Convergence) ──
        # 若腰部初猜接近零位，优先尝试单臂 7-DoF 单独闭环求解。
        # 只要目标位于单臂真实几何与限位流形内，腰部 100% 保持直立零位，毫秒级快速返回！
        # 绝不使用任何人工经验标量球形距离截断，纯靠运动学真实收敛性裁决。
        seed_waist_norm = float(np.linalg.norm(seed_waist)) if seed_waist is not None else 0.0
        if cascade_upright and seed_waist_norm < 0.02:
            success_single_arm, arm_solution_single, result_single_arm = self.solve_ik(
                arm=arm,
                target_pos=normalized_pos,
                target_rot=normalized_rot,
                seed_q=seed_arm,
                pos_tol=pos_tol,
                rot_tol=rot_tol,
                max_iters=min(20, max_iters),
                max_step=max_step,
                allow_relaxation=allow_relaxation,
            )
            if success_single_arm:
                waist_zero_solution = np.zeros(self.waist_dim)
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                telemetry = self._extract_solution_telemetry(
                    arm=arm,
                    waist_q=waist_zero_solution,
                    arm_q=arm_solution_single,
                    target_rot=normalized_rot,
                    is_coord=True,
                )
                result_single_arm.update({
                    "time_ms": elapsed_ms,
                    "cascade_stage": "ARM_UPRIGHT_PRIORITY",
                    "mode": "9DOF_6DOF_POSE" if normalized_rot is not None else "9DOF_3DOF_POS",
                    "waist_solution": waist_zero_solution.copy(),
                    "arm_solution": arm_solution_single.copy(),
                    "q_solution": np.concatenate([waist_zero_solution, arm_solution_single]),
                    **telemetry,
                })
                return True, waist_zero_solution, arm_solution_single, result_single_arm

        # ── 2. 9-DoF 躯干协同二次规划 (Coordinated Whole-Body QP) ──
        # 当且仅当单臂物理不可达时，自动激活协同链。
        # 在二次规划目标中，通过李群零空间姿态惩罚对腰部施加各向异性高刚度阻尼：
        # 偏航轴阻尼优先 (0.080)，俯仰轴强阻尼抑制躬身失稳 (0.120)，手臂维持灵巧轻量阻尼 (1e-4)。
        # QP 优化器在数学上会自动寻找「能够达成末端闭环所需的最小必要躯干位移」。
        stage_name = "9DOF_COORDINATED"
        waist_weights = np.array([0.080, 0.120]) * (float(waist_weight) / 10.0)

        lower_limit, upper_limit, ready_coordinated_q, cost_posture = self._prepare_solver_context(
            arm=arm,
            is_coord=True,
            custom_limits=custom_limits,
            waist_weights=waist_weights,
        )

        seed_chain, seed_names = self._build_seed_chain(
            is_coord=True,
            arm=arm,
            ready_q=ready_coordinated_q,
            lower_limit=lower_limit,
            upper_limit=upper_limit,
            seed_waist=seed_waist,
            seed_arm=seed_arm,
        )

        success_coord, coordinated_solution, result_coord = self._solve_qp_core(
            arm=arm,
            is_coord=True,
            target_pos=normalized_pos,
            target_rot=normalized_rot,
            seed_chain=seed_chain,
            seed_names=seed_names,
            ready_q=ready_coordinated_q,
            cost_posture=cost_posture,
            lower_limit=lower_limit,
            upper_limit=upper_limit,
            pos_tol=pos_tol,
            rot_tol=rot_tol,
            max_iters=max_iters,
            max_step=max_step,
            start_time=start_time,
            allow_relaxation=allow_relaxation,
        )

        waist_solution = coordinated_solution[:self.waist_dim].copy()
        arm_solution = coordinated_solution[self.waist_dim:].copy()
        result_coord.cascade_stage = stage_name
        result_coord.waist_solution = waist_solution
        result_coord.arm_solution = arm_solution

        return success_coord, waist_solution, arm_solution, result_coord


__all__ = [
    "HumanoidPinkIKSolver",             # 核心 IK 求解器类
    "HumanoidKinematicsAdapter",      # 机械臂正逆运动学计算类
    "IKSolveStatus",                  # IK 求解状态枚举
    "IKResult",                         # IK 求解结果类
    "IK_PIPELINE_STAGES",             # IK 求解流程阶段常量
    "DEFAULT_HUMANOID_WAIST_LIMITS",    # 腰部 2-DoF 默认限位 (Yaw/Pitch)
    "DEFAULT_HUMANOID_ARM_LIMITS",      # 标准 7-DoF 机械臂通用硬件限位参考
    "DEFAULT_HUMANOID_READY_POSE",      # 人形机器人默认待机/准备姿态
]
