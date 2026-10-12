"""
通用人形机器人逆运动学求解器 (core.solver)
==============================================================
**IK 解算实现完整地集中在单一文件 `humanoid_pink_ik.py`**，不再拆分。

HumanoidPinkIKSolver —— 通用人形机器人求解器
    * 架构: 2-DoF 腰部 (waist_yaw, waist_pitch, 不含侧倾) + 7-DoF 机械臂 = 9-DoF
    * 求解任务: 6D 位姿 (位置 + 姿态, SE(3))，亦支持纯位置退化模式
    * 底层: Inria Pink 流形优化器 + ProxQP 凸二次规划
    * 可移植: HumanoidKinematicsAdapter 接受任意 URDF/SRDF，默认使用本仓参考模型
    * 安全层: 自碰撞二值门禁 (惰性求值)，可选 SelfCollisionBarrier

对外接口:
    solve_ik()            —— 7-DoF 单臂解算 (腰部锁定直立)
    solve_coordinated_ik() —— 9-DoF 腰臂协同解算
"""

from core.solver.humanoid_pink_ik import (
    HumanoidPinkIKSolver,
    HumanoidKinematicsAdapter,
    IKSolveStatus,
    IKResult,
    IK_PIPELINE_STAGES,
    DEFAULT_HUMANOID_WAIST_LIMITS,
    DEFAULT_HUMANOID_ARM_LIMITS,
    DEFAULT_HUMANOID_READY_POSE,
)

__all__ = [
    "HumanoidPinkIKSolver",
    "HumanoidKinematicsAdapter",
    "IKSolveStatus",
    "IKResult",
    "IK_PIPELINE_STAGES",
    "DEFAULT_HUMANOID_WAIST_LIMITS",
    "DEFAULT_HUMANOID_ARM_LIMITS",
    "DEFAULT_HUMANOID_READY_POSE",
    "DualArmIKSolver",
]

# 双臂 16-DoF 求解器使用独立算法 (分阶段 QP 精修, 非 pink 路线), 单独成文件。
# 依赖 qpsolvers / scipy 稀疏运算, 为可选依赖: 缺失时静默跳过而非阻断主求解器导入。
try:  # pragma: no cover - 取决于本地可选依赖是否安装
    from core.solver.dual_arm_ik import DualArmIKSolver
except ImportError:  # pragma: no cover
    DualArmIKSolver = None  # type: ignore[assignment,misc]