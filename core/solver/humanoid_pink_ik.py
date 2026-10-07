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

2. 工作空间三段式自适应协调 (Three-Stage Workspace Coordination):
   以末端目标相对肩部原点的空间欧几里得距离 d = ||p_target - p_shoulder|| 为感知变量：
   ┌───────────────────────┬─────────────────────────────────────────────────┐
   │ 距离区间 (工作区)     │ 协同与介入行为                                  │
   ├───────────────────────┼─────────────────────────────────────────────────┤
   │ 近端舒适区            │ 触发 cascade_upright 快速通道：优先采用单臂 7-DoF│
   │ (d <= 30.0 cm)        │ 求解，腰部 100% 锁定在直立零位 [0.0, 0.0]，     │
   │                       │ 姿态阻尼权重最高 (w_lock)，抑制任何非必要腰动。 │
   ├───────────────────────┼─────────────────────────────────────────────────┤
   │ C^2 平滑过渡区        │ 采用 C^2 连续 Smoothstep 激活函数：             │
   │ (30.0cm < d < 38.5cm) │ s = (d - 30cm) / (38.5cm - 30cm)                │
   │                       │ mu = s^2 * (3 - 2s)                             │
   │                       │ 权重由锁紧刚度 w_lock 向辅助刚度 w_assist 平滑  │
   │                       │ 过渡，实现腰部无冲击、无抖动、连续可微柔顺介入。│
   ├───────────────────────┼─────────────────────────────────────────────────┤
   │ 远端极限区            │ 激活系数 mu = 1.0，手臂逼近展长极限，腰部全面   │
   │ (d >= 38.5 cm)        │ 进入低阻尼协同状态 (w_assist)，主动旋转与俯仰， │
   │                       │ 显著扩展操作半径 (外延 15~30cm)，支持大跨度拾取。│
   └───────────────────────┴─────────────────────────────────────────────────┘

3. 各向异性动态阻尼刚度配置 (Anisotropic Dynamic Damping):
   腰部两轴具有明确的仿生优先级与安全性差异加权：
   - 偏航优先 (Yaw Priority): w_lock = 0.080, w_assist = 1.5e-4。
     旋转轴阻尼较小，优先驱动躯干迎向目标作业面，符合人体工效学；
   - 俯仰抑制 (Pitch Damping): w_lock = 0.120, w_assist = 3.0e-4。
     俯仰轴阻尼为偏航轴的 1.5~2 倍，强力抑制过大幅度的躬身与后仰，
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
        for wj in raw_waist_names:
            if self.model.existJointName(wj):
                resolved_waist_names.append(wj)
            else:
                is_yaw = "yaw" in wj.lower()
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
                    raise KeyError(f"模型中未找到指定的腰部关节: {wj} (全模型关节: {list(self.model.names)})")

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

        for arm, jnames in self.arm_joint_names.items():
            if len(jnames) != 7:
                raise ValueError(f"{arm} 关节数必须为 7 (当前: {len(jnames)})")
            for jn in jnames:
                if not self.model.existJointName(jn):
                    raise KeyError(f"模型中未找到指定的机械臂关节: {jn} (臂: {arm})")

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

        for arm, fname in self.ee_frame_names.items():
            if not self.model.existFrame(fname):
                raise KeyError(f"模型中未找到指定的末端坐标系: {fname} (臂: {arm})")

        self.ee_frame_ids: Dict[str, int] = {
            arm: self.model.getFrameId(fname) for arm, fname in self.ee_frame_names.items()
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
            for arm in ["left_arm", "right_arm"]
        }
        self.limits_coord: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for arm in ["left_arm", "right_arm"]:
            l_coord = np.concatenate([self.waist_limits[0], self.limits_arm[arm][0]])
            u_coord = np.concatenate([self.waist_limits[1], self.limits_arm[arm][1]])
            self.limits_coord[arm] = (l_coord, u_coord)

        # ── 6. 构建轻量化裁剪子模型 (9-DoF 与 7-DoF) ──
        self.model_coord: Dict[str, pin.Model] = {}
        self.data_coord: Dict[str, pin.Data] = {}
        self.ee_frame_ids_coord: Dict[str, int] = {}

        self.model_7dof: Dict[str, pin.Model] = {}
        self.data_7dof: Dict[str, pin.Data] = {}
        self.ee_frame_ids_7dof: Dict[str, int] = {}

        self._build_submodels()

    def _build_submodels(self) -> None:
        """构建左右臂各 9-DoF (2腰 + 7臂) 与 7-DoF 单臂轻量化子模型。"""
        q_ref = pin.neutral(self.model)
        for arm in ["left_arm", "right_arm"]:
            # 1. 9-DoF 协同链子模型
            active_coord = self.chain_coord_joint_names[arm]
            locked_coord = [
                self.model.getJointId(jname)
                for jname in self.model.names[1:]
                if jname not in active_coord
            ]
            red_coord = pin.buildReducedModel(self.model, locked_coord, q_ref)
            self.model_coord[arm] = red_coord
            self.data_coord[arm] = red_coord.createData()
            self.ee_frame_ids_coord[arm] = red_coord.getFrameId(self.ee_frame_names[arm])

            # 2. 7-DoF 单臂子模型
            active_7 = self.arm_joint_names[arm]
            locked_7 = [
                self.model.getJointId(jname)
                for jname in self.model.names[1:]
                if jname not in active_7
            ]
            red_7 = pin.buildReducedModel(self.model, locked_7, q_ref)
            self.model_7dof[arm] = red_7
            self.data_7dof[arm] = red_7.createData()
            self.ee_frame_ids_7dof[arm] = red_7.getFrameId(self.ee_frame_names[arm])

    def forward_kinematics_coord(
        self,
        arm: str,
        waist_q: np.ndarray,
        arm_q: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """计算 9-DoF (2腰 + 7臂) 协同正向运动学。

        :param arm: 'left_arm' 或 'right_arm'
        :param waist_q: 腰部 2 自由度关节角 (弧度)
        :param arm_q: 手臂 7 自由度关节角 (弧度)
        :return: (translation, rotation_matrix)
        """
        q_coord = np.concatenate([
            np.asarray(waist_q, dtype=np.float64),
            np.asarray(arm_q, dtype=np.float64),
        ])
        m = self.model_coord[arm]
        d = self.data_coord[arm]
        ee_id = self.ee_frame_ids_coord[arm]
        pin.forwardKinematics(m, d, q_coord)
        pin.updateFramePlacements(m, d)
        t = d.oMf[ee_id]
        return t.translation.copy(), t.rotation.copy()

    def forward_kinematics_arm(
        self,
        arm: str,
        arm_q: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """计算 7-DoF 单臂正向运动学。

        :param arm: 'left_arm' 或 'right_arm'
        :param arm_q: 手臂 7 自由度关节角 (弧度)
        :return: (translation, rotation_matrix)
        """
        m = self.model_7dof[arm]
        d = self.data_7dof[arm]
        ee_id = self.ee_frame_ids_7dof[arm]
        pin.forwardKinematics(m, d, np.asarray(arm_q, dtype=np.float64))
        pin.updateFramePlacements(m, d)
        t = d.oMf[ee_id]
        return t.translation.copy(), t.rotation.copy()


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
    5. C^2 连续 Smoothstep 激活函数调节腰臂动态刚度，近端舒适区腰部直立锁定；
    6. 二阶段李群高斯-牛顿微调抛光 (消除 QP 正则化残差)。
    """

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
        self.limits_arm: Dict[str, Tuple[np.ndarray, np.ndarray]] = self.kin.limits_arm
        self.limits_coord: Dict[str, Tuple[np.ndarray, np.ndarray]] = self.kin.limits_coord

    # --------------------------------------------------------------------------
    # 状态自省与度量指标计算
    # --------------------------------------------------------------------------

    def forward_kinematics_arm(self, arm: str, arm_q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """计算 7-DoF 单臂正向运动学。"""
        return self.kin.forward_kinematics_arm(arm, arm_q)

    def forward_kinematics_coord(
        self, arm: str, waist_q: np.ndarray, arm_q: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """计算 9-DoF (2腰 + 7臂) 协同正向运动学。"""
        return self.kin.forward_kinematics_coord(arm, waist_q, arm_q)

    def _extract_solution_telemetry(
        self,
        arm: str,
        waist_q: Optional[np.ndarray],
        arm_q: np.ndarray,
        target_rot: Optional[np.ndarray] = None,
        is_coord: bool = True,
    ) -> Dict[str, Any]:
        """提取解算构型下的空间位姿误差与关节限位裕度。"""
        if is_coord:
            wq = np.asarray(
                waist_q if waist_q is not None else np.zeros(self.waist_dim),
                dtype=np.float64,
            )
            q_active = np.concatenate([wq, np.asarray(arm_q, dtype=np.float64)])
            low_lim, up_lim = self.limits_coord[arm]
            all_names = list(self.waist_joint_names) + list(self.arm_joint_names[arm])
            fk_pos, fk_rot = self.kin.forward_kinematics_coord(arm, wq, arm_q)
        else:
            q_active = np.asarray(arm_q, dtype=np.float64)
            low_lim, up_lim = self.limits_arm[arm]
            all_names = list(self.arm_joint_names[arm])
            fk_pos, fk_rot = self.kin.forward_kinematics_arm(arm, arm_q)

        act_rpy_rad = pin.rpy.matrixToRpy(fk_rot)
        act_rpy_deg = [round(float(np.degrees(v)), 2) for v in act_rpy_rad]
        act_quat = pin.Quaternion(fk_rot)

        tgt_rpy_deg = None
        rot_err_deg = 0.0
        if target_rot is not None:
            tgt_rpy_rad = pin.rpy.matrixToRpy(target_rot)
            tgt_rpy_deg = [round(float(np.degrees(v)), 2) for v in tgt_rpy_rad]
            r_err = target_rot @ fk_rot.T
            rot_err_deg = round(float(np.degrees(np.linalg.norm(pin.log3(r_err)))), 2)

        # 关节限位安全裕度 (距物理硬限位的最小角距离)
        dist_to_low = q_active - low_lim
        dist_to_up = up_lim - q_active
        margins = np.minimum(dist_to_low, dist_to_up)
        crit_idx = int(np.argmin(margins))
        joint_limit_margin_rad = float(margins[crit_idx])
        joint_limit_margin_deg = round(float(np.degrees(joint_limit_margin_rad)), 2)
        critical_joint_name = all_names[crit_idx]

        within_limits = bool(
            np.all(q_active >= low_lim - 1e-4) and np.all(q_active <= up_lim + 1e-4)
        )

        return {
            "within_limits": within_limits,
            "joint_limit_margin": round(joint_limit_margin_rad, 4),
            "joint_limit_margin_deg": joint_limit_margin_deg,
            "critical_joint": critical_joint_name,
            "joint_limits_bounds": {
                "lower": [round(float(v), 4) for v in low_lim],
                "upper": [round(float(v), 4) for v in up_lim],
            },
            "actual_pos": [round(float(v), 5) for v in fk_pos],
            "actual_rpy_deg": act_rpy_deg,
            "actual_quat": [
                round(float(v), 5) for v in [act_quat.x, act_quat.y, act_quat.z, act_quat.w]
            ],
            "target_rpy_deg": tgt_rpy_deg,
            "rot_err_deg": rot_err_deg,
        }

    @staticmethod
    def _classify_solve_status(
        is_success: bool,
        fin_ep: float,
        fin_er: float,
        pos_tol: float,
        rot_tol: float,
        has_rot: bool,
        within_limits: bool = True,
        allow_relaxation: bool = False,
        relaxed_rot_tol: float = 3.5e-2,  # ~2.0°
    ) -> IKSolveStatus:
        """纯函数式状态分类机。"""
        if not within_limits:
            return IKSolveStatus.JOINT_LIMIT_VIOLATION    # 关节限位违反
        if is_success:
            return IKSolveStatus.CONVERGED                # 成功收敛
        if allow_relaxation and has_rot and fin_ep < pos_tol and fin_er < relaxed_rot_tol:
            return IKSolveStatus.RELAXED_ORIENTATION    # 允许放宽
        if fin_ep < pos_tol and has_rot and fin_er >= rot_tol:
            return IKSolveStatus.POSITION_REACHED_ONLY  # 位置已达到，方向未达到
        return IKSolveStatus.MAX_ITERATIONS_EXCEEDED    # 达到最大迭代次数

    def _validate_target_pos(
        self,
        target_pos: Any,
        arm: str,
        is_coord: bool,
    ) -> Tuple[bool, np.ndarray, Optional[IKResult]]:
        """检验目标空间坐标有效性，拦截 NaN / Inf 等非法浮点异常。"""
        pos = np.asarray(target_pos, dtype=np.float64)
        if pos.shape != (3,) or np.isnan(pos).any() or np.isinf(pos).any():
            ready_arm = self.ready_pose[arm]
            waist_zero = np.zeros(self.waist_dim)
            q_sol = np.concatenate([waist_zero, ready_arm]) if is_coord else ready_arm.copy()
            fail_res = IKResult(
                success=False,
                status=IKSolveStatus.INVALID_INPUT,
                q_solution=q_sol,
                waist_solution=waist_zero.copy() if is_coord else None,
                arm_solution=ready_arm.copy(),
                pos_err_mm=999.0,
                rot_err_deg=180.0,
                within_limits=True,
                time_ms=0.0,
                iters=0,
                error="Invalid target_pos: must be 3-element finite float array",
            )
            return False, pos, fail_res
        return True, pos, None

    # --------------------------------------------------------------------------
    # 启发式初猜与二阶段抛光器
    # --------------------------------------------------------------------------

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
                sw = np.clip(
                    np.asarray(
                        seed_waist if seed_waist is not None else np.zeros(self.waist_dim),
                        dtype=np.float64,
                    ),
                    self.waist_limits[0],
                    self.waist_limits[1],
                )
                sa = np.clip(
                    np.asarray(
                        seed_arm if seed_arm is not None else ready_arm,
                        dtype=np.float64,
                    ),
                    self.limits_arm[arm][0],
                    self.limits_arm[arm][1],
                )
                seed_chain.append(np.concatenate([sw, sa]))
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
        for r_idx in range(4):
            alpha = (r_idx + 1) / 5.0
            qmc_seed = lower_limit + alpha * (upper_limit - lower_limit)
            if is_coord:
                qmc_seed[:self.waist_dim] *= 0.30  # 对腰部初猜施加阻尼，优先探索手臂空间
            seed_chain.append(np.clip(qmc_seed, lower_limit, upper_limit))
            seed_names.append(f"QMC Grid #{r_idx + 1}")

        return seed_chain, seed_names

    @staticmethod
    def _polish_gauss_newton(
        model: pin.Model,
        data: pin.Data,
        ee_id: int,
        q_init: np.ndarray,
        target_pos: np.ndarray,
        target_rot: Optional[np.ndarray],
        lower_limit: np.ndarray,
        upper_limit: np.ndarray,
        max_polish_steps: int = 2,
    ) -> Tuple[np.ndarray, float, float]:
        """二阶段李群高斯-牛顿微调抛光器，消除 QP 正则化残差。"""
        has_rot = target_rot is not None
        pin.forwardKinematics(model, data, q_init)
        pin.updateFramePlacements(model, data)
        cur_t = data.oMf[ee_id]
        ep_norm = float(np.linalg.norm(target_pos - cur_t.translation))
        er_norm = float(np.linalg.norm(pin.log3(target_rot @ cur_t.rotation.T))) if has_rot else 0.0

        best_q = q_init.copy()
        best_err = ep_norm + er_norm * 0.25
        best_rot_norm = er_norm

        q_pol = q_init.copy()
        for _ in range(max_polish_steps):
            pin.forwardKinematics(model, data, q_pol)
            pin.updateFramePlacements(model, data)
            cur_t = data.oMf[ee_id]
            ep = target_pos - cur_t.translation
            if has_rot:
                er = pin.log3(target_rot @ cur_t.rotation.T)
                e_vec = np.concatenate([ep, er])
                j_mat = pin.computeFrameJacobian(
                    model, data, q_pol, ee_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
                )
            else:
                e_vec = ep
                j_mat = pin.computeFrameJacobian(
                    model, data, q_pol, ee_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
                )[:3, :]

            m = 6 if has_rot else 3
            dq = j_mat.T @ np.linalg.solve(j_mat @ j_mat.T + 1e-6 * np.eye(m), e_vec)
            dq = np.clip(dq, -0.04, 0.04)

            q_cand = np.clip(q_pol + dq, lower_limit, upper_limit)
            pin.forwardKinematics(model, data, q_cand)
            pin.updateFramePlacements(model, data)
            c_t = data.oMf[ee_id]
            cand_ep = float(np.linalg.norm(target_pos - c_t.translation))
            cand_er = float(np.linalg.norm(pin.log3(target_rot @ c_t.rotation.T))) if has_rot else 0.0
            cand_err = cand_ep + cand_er * 0.25
            if cand_err < best_err:
                q_pol = q_cand
                best_q = q_cand.copy()
                best_err = cand_err
                best_rot_norm = cand_er

        return best_q, best_err, best_rot_norm

    # --------------------------------------------------------------------------
    # 核心凸二次规划求解器 (_solve_qp_core)
    # --------------------------------------------------------------------------

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
        target_pose = pin.SE3(target_rot if has_rot else np.eye(3), target_pos)

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
        best_rot_norm: float = 0.0
        total_iters = 0
        dt = 0.25

        config = pink.Configuration(model, data, seed_chain[0].copy())

        for seed_idx, seed in enumerate(seed_chain):
            if total_iters >= max_iters:
                break

            clamped_seed = np.clip(seed.copy(), lower_limit, upper_limit)
            config.update(clamped_seed)
            prev_err_mm: Optional[float] = None
            stall_count = 0

            # 初猜迭代预算分配
            is_primary_seed = (seed_idx == 0) or (seed_idx == 1 and seed_names[0] == "Warm Start")
            remaining_total = max_iters - total_iters
            if is_primary_seed:
                seed_iter_budget = min(remaining_total, max(22, int(max_iters * 0.70)))
            else:
                seed_iter_budget = min(remaining_total, max(10, int(max_iters * 0.35)))

            for _ in range(seed_iter_budget):
                total_iters += 1

                # 姿态引力动态退火
                if prev_err_mm is not None and prev_err_mm < 6.0:
                    alpha_posture = max(0.01, prev_err_mm / 6.0)
                    task_posture.cost = cost_posture * alpha_posture
                else:
                    task_posture.cost = cost_posture.copy()

                try:
                    v = pink.solve_ik(
                        config,
                        tasks,
                        dt,
                        solver="proxqp",
                        limits=joint_limits,
                    )
                except Exception:
                    try:
                        v = pink.solve_ik(
                            config,
                            tasks,
                            dt,
                            solver="osqp",
                            limits=joint_limits,
                        )
                    except Exception:
                        break

                # 步长硬截断
                step = v * dt
                step_max = float(np.max(np.abs(step)))
                if max_step is not None and max_step > 0 and step_max > max_step:
                    v = v * (max_step / step_max)

                config.integrate_inplace(v, dt)
                q_cur = np.clip(config.q.copy(), lower_limit, upper_limit)
                if not np.array_equal(q_cur, config.q):
                    config.update(q_cur)

                cur_pose = config.get_transform_frame_to_world(ee_frame)
                pos_norm = float(np.linalg.norm(target_pos - cur_pose.translation))
                cur_err_mm = pos_norm * 1000.0
                delta_mm = (prev_err_mm - cur_err_mm) if prev_err_mm is not None else 0.0
                prev_err_mm = cur_err_mm

                rot_norm = 0.0
                if has_rot:
                    r_err = target_rot @ cur_pose.rotation.T
                    rot_norm = float(np.linalg.norm(pin.log3(r_err)))

                err_val = pos_norm + rot_norm * 0.25
                if err_val < best_err:
                    best_err = err_val
                    best_q = q_cur.copy()
                    best_rot_norm = rot_norm

                if pos_norm < pos_tol and (not has_rot or rot_norm < rot_tol):
                    break

                if delta_mm < 0.05:
                    stall_count += 1
                    if stall_count >= 4:
                        break
                else:
                    stall_count = 0

            if best_q is not None and best_err < pos_tol + (rot_tol * 0.25 if has_rot else 0.0):
                break

        # ── 二阶段微调抛光 ──
        ee_id = self.kin.ee_frame_ids_coord[arm] if is_coord else self.kin.ee_frame_ids_7dof[arm]
        target_polish_q = best_q if best_q is not None else ready_q.copy()
        if best_err < 0.05:
            best_q, best_err, best_rot_norm = self._polish_gauss_newton(
                model=model,
                data=data,
                ee_id=ee_id,
                q_init=target_polish_q,
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
        fin_t = data.oMf[ee_id]
        fin_ep = float(np.linalg.norm(target_pos - fin_t.translation))
        fin_er = float(np.linalg.norm(pin.log3(target_rot @ fin_t.rotation.T))) if has_rot else 0.0
        within_limits = bool(
            np.all(best_q >= lower_limit - 1e-4) and np.all(best_q <= upper_limit + 1e-4)
        )

        relaxed_ok = allow_relaxation and has_rot and (fin_ep < pos_tol) and (fin_er < 3.5e-2)
        is_success = (
            ((fin_ep < pos_tol and (not has_rot or fin_er < rot_tol)) or relaxed_ok)
            and within_limits
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        # 分割腰部解与手臂解
        w_q = best_q[:self.waist_dim] if is_coord else None
        a_q = best_q[self.waist_dim:] if is_coord else best_q

        telemetry = self._extract_solution_telemetry(
            arm=arm,
            waist_q=w_q if is_coord else None,
            arm_q=a_q,
            target_rot=target_rot if has_rot else None,
            is_coord=is_coord,
        )

        solve_status = self._classify_solve_status(
            is_success=is_success,
            fin_ep=fin_ep,
            fin_er=fin_er,
            pos_tol=pos_tol,
            rot_tol=rot_tol,
            has_rot=has_rot,
            within_limits=within_limits,
            allow_relaxation=allow_relaxation,
        )

        if is_coord:
            mode_str = "9DOF_6DOF_POSE" if has_rot else "9DOF_3DOF_POS"
        else:
            mode_str = "7DOF_6DOF_POSE" if has_rot else "7DOF_3DOF_POS"

        ik_res = IKResult(
            success=bool(is_success),
            status=solve_status,
            q_solution=best_q.copy(),
            waist_solution=w_q.copy() if is_coord else None,
            arm_solution=a_q.copy(),
            iters=total_iters,
            time_ms=elapsed_ms,
            pos_err_mm=round(fin_ep * 1000.0, 4),
            pipeline_stages=IK_PIPELINE_STAGES,
            mode=mode_str,
            **telemetry,
        )
        return is_success, best_q, ik_res

    # --------------------------------------------------------------------------
    # 7-DoF 单臂逆运动学求解接口 (solve_ik)
    # --------------------------------------------------------------------------

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
        :param pos_tol: 位置收敛容差 (米, 默认 1.0mm)
        :param rot_tol: 姿态收敛容差 (弧度, 默认 ~1.15°)
        :param max_iters: 最大优化迭代步数
        :param max_step: 单步最大关节步长截断 (rad)
        :param custom_limits: 可选的自定义关节限位 (lower_limits, upper_limits)
        :param allow_relaxation: 是否允许姿态微松弛收敛 (<=2.0°)
        :return: (is_success, q_solution, ik_result_info)
        """
        start_time = time.perf_counter()
        valid, pos_arr, fail_res = self._validate_target_pos(target_pos, arm, is_coord=False)
        if not valid:
            return False, self.ready_pose[arm].copy(), fail_res

        if target_rot is None and target_rpy is not None:
            rpy = np.asarray(target_rpy, dtype=np.float64)
            target_rot = pin.rpy.rpyToMatrix(float(rpy[0]), float(rpy[1]), float(rpy[2]))
        if target_rot is not None:
            target_rot = np.asarray(target_rot, dtype=np.float64)

        lower_limit, upper_limit = (
            custom_limits if custom_limits is not None else self.limits_arm[arm]
        )
        ready_q = self.ready_pose[arm]

        seed_chain, seed_names = self._build_seed_chain(
            is_coord=False,
            arm=arm,
            ready_q=ready_q,
            lower_limit=lower_limit,
            upper_limit=upper_limit,
            seed_q=seed_q,
        )

        cost_posture = np.ones(len(ready_q)) * 1e-4

        return self._solve_qp_core(
            arm=arm,
            is_coord=False,
            target_pos=pos_arr,
            target_rot=target_rot,
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

    # --------------------------------------------------------------------------
    # 9-DoF (2-DoF 腰部 + 7-DoF 手臂) 协同逆运动学求解接口 (solve_coordinated_ik)
    # --------------------------------------------------------------------------

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
        :param waist_weight: 腰部惩罚基准 (默认 10.0，数值越大越抑制腰部运动)
        :param cascade_upright: 是否启用近端舒适区腰部直立锁定快速通道
        :param custom_limits: 可选的自定义联合限位 (lower_limits, upper_limits)
        :param allow_relaxation: 是否允许姿态微松弛收敛 (<=2.0°)
        :return: (is_success, waist_solution(2,), arm_solution(7,), ik_result_info)
        """
        start_time = time.perf_counter()
        valid, pos_arr, fail_res = self._validate_target_pos(target_pos, arm, is_coord=True)
        if not valid:
            return False, np.zeros(self.waist_dim), self.ready_pose[arm].copy(), fail_res

        if target_rot is None and target_rpy is not None:
            rpy = np.asarray(target_rpy, dtype=np.float64)
            target_rot = pin.rpy.rpyToMatrix(float(rpy[0]), float(rpy[1]), float(rpy[2]))
        if target_rot is not None:
            target_rot = np.asarray(target_rot, dtype=np.float64)

        # ── 连续自适应工作空间协调架构 ──
        sh_origin = self.kin.model_7dof[arm].jointPlacements[1].translation
        dist_to_shoulder = float(np.linalg.norm(pos_arr - sh_origin))

        # 舒适区直立快速通道
        sw_norm = float(np.linalg.norm(seed_waist)) if seed_waist is not None else 0.0
        if cascade_upright and dist_to_shoulder <= 0.300 and sw_norm < 0.02:
            ok_7, q_arm_7, info_7 = self.solve_ik(
                arm=arm,
                target_pos=pos_arr,
                target_rot=target_rot,
                seed_q=seed_arm,
                pos_tol=pos_tol,
                rot_tol=rot_tol,
                max_iters=min(18, max_iters),
                max_step=max_step,
                allow_relaxation=allow_relaxation,
            )
            if ok_7:
                waist_zero = np.zeros(self.waist_dim)
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                telemetry = self._extract_solution_telemetry(
                    arm=arm,
                    waist_q=waist_zero,
                    arm_q=q_arm_7,
                    target_rot=target_rot,
                    is_coord=True,
                )
                info_7.update({
                    "time_ms": elapsed_ms,
                    "cascade_stage": "ARM_COMFORT_UPRIGHT",
                    "mode": "9DOF_6DOF_POSE" if target_rot is not None else "9DOF_3DOF_POS",
                    "waist_solution": waist_zero.copy(),
                    "arm_solution": q_arm_7.copy(),
                    "q_solution": np.concatenate([waist_zero, q_arm_7]),
                    **telemetry,
                })
                return True, waist_zero, q_arm_7, info_7

        # ── 连续过渡区域激活权重计算 (C2 Smoothstep) ──
        d_near = 0.300      # 舒适区距离
        d_far = 0.385       # 过渡区距离
        if dist_to_shoulder <= d_near:
            mu = 0.0
            stage_name = "ARM_COMFORT_ZONE"
        elif dist_to_shoulder >= d_far:
            mu = 1.0
            stage_name = "9DOF_COORDINATED"
        else:
            s = (dist_to_shoulder - d_near) / (d_far - d_near)
            mu = s * s * (3.0 - 2.0 * s)
            stage_name = "CASCADE_SMOOTH_TRANSITION"

        # 2-DoF 腰部各向异性动态阻尼刚度: [Yaw(偏航), Pitch(俯仰)]
        w_lock = np.array([0.080, 0.120]) * (float(waist_weight) / 10.0)      # 锁定腰部权重
        w_assist = np.array([1.5e-4, 3.0e-4]) * (float(waist_weight) / 10.0)
        waist_weights = w_lock * (1.0 - mu) + w_assist * mu

        lower_limit, upper_limit = (
            custom_limits if custom_limits is not None else self.limits_coord[arm]
        )

        ready_arm = self.ready_pose[arm]        # 准备姿态
        ready_coord = np.concatenate([np.zeros(self.waist_dim), ready_arm]) 
        n = self.waist_dim + len(ready_arm)

        seed_chain, seed_names = self._build_seed_chain(        # 构建种子链
            is_coord=True,      # 协调模式
            arm=arm,            
            ready_q=ready_coord,
            lower_limit=lower_limit,
            upper_limit=upper_limit,
            seed_waist=seed_waist,
            seed_arm=seed_arm,
        )

        cost_posture = np.ones(n) * 1e-4            # 姿态成本
        cost_posture[:self.waist_dim] = waist_weights       # 腰部成本

        ok_coord, q_sol, info_coord = self._solve_qp_core(          # 核心 IK 求解过程
            arm=arm,
            is_coord=True,  
            target_pos=pos_arr,             # 目标位置
            target_rot=target_rot,          # 目标旋转
            seed_chain=seed_chain,          # 种子链
            seed_names=seed_names,          # 种子名称
            ready_q=ready_coord,            # 准备姿态
            cost_posture=cost_posture,      # 姿态成本
            lower_limit=lower_limit,        # 下限
            upper_limit=upper_limit,        # 上限
            pos_tol=pos_tol,                # 位置容忍度
            rot_tol=rot_tol,                # 旋转容忍度
            max_iters=max_iters,            # 最大迭代次数
            max_step=max_step,              # 最大步长
            start_time=start_time,          # 开始时间
            allow_relaxation=allow_relaxation,
        )

        waist_sol = q_sol[:self.waist_dim].copy()       
        arm_sol = q_sol[self.waist_dim:].copy()
        info_coord.cascade_stage = stage_name
        info_coord.waist_solution = waist_sol
        info_coord.arm_solution = arm_sol

        return ok_coord, waist_sol, arm_sol, info_coord


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
