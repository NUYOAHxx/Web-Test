"""
Unitree G1 / 通用人形机器人运动学逆解求解器模块 (core.solver)
==============================================================
本模块提供分层的 Pink + ProxQP 凸优化逆运动学解算体系：

1. 通用算法层 (Generic Humanoid Layer):
   - HumanoidPinkIKSolver: 通用标准人形机器人独立求解器
     * 架构: 2-DoF 腰部 (Yaw, Pitch 防侧倾锁定) + 7-DoF 机械臂 (9-DoF 协同)
     * 特性: 纯净零 ROS/MoveIt 依赖、自包含、支持跨机型 URDF
     * 适用: 跨人形机型移植、单双臂/通用双足算法基准与学术验证

2. G1 机型特化层 (Unitree G1 Tailored Layer):
   - G1PinkIKSolver: Unitree G1 专用 10-DoF 全局协同求解器
     * 架构: 3-DoF 腰部 (Yaw, Roll, Pitch 全轴开放) + 7-DoF 机械臂
     * 特性: 深度释放 G1 侧向倾角工作空间，深度绑定 MoveIt 2、RViz 与实机控制器
     * 适用: G1 避障轨迹规划、大跨度拾取工况与实机动态伺服
"""

from core.solver.g1_pink_ik import (
    G1PinkIKSolver,
    IKSolveStatus,
    IKResult,
    IK_PIPELINE_STAGES,
    G1_READY_POSE,
    G1_DEFAULT_STAND_JOINTS,
)
from core.solver.humanoid_pink_ik import (
    HumanoidPinkIKSolver,
    HumanoidKinematicsAdapter,
    DEFAULT_HUMANOID_WAIST_LIMITS,
    DEFAULT_HUMANOID_ARM_LIMITS,
    DEFAULT_HUMANOID_READY_POSE,
)

__all__ = [
    "G1PinkIKSolver",
    "HumanoidPinkIKSolver",
    "HumanoidKinematicsAdapter",
    "IKSolveStatus",
    "IKResult",
    "IK_PIPELINE_STAGES",
    "G1_READY_POSE",
    "G1_DEFAULT_STAND_JOINTS",
    "DEFAULT_HUMANOID_WAIST_LIMITS",
    "DEFAULT_HUMANOID_ARM_LIMITS",
    "DEFAULT_HUMANOID_READY_POSE",
]
