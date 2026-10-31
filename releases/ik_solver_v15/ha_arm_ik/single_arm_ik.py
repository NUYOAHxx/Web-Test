#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""ha_arm_ik.single_arm_ik —— 腰 2 + 臂 7（9 维 / 7 维）单臂逆运动学高内聚求解器。

本文件整合了单臂数值逆运动学求解器的全部算法逻辑、数学模型与面向对象接口，
与双臂逆运动学模块 (dual_arm_ik.py) 共同构成标准统一的机械臂逆运动学套件。

十层分层架构（由底层至高层，单向依赖）：
  [1] 模块元数据与统一依赖导入 (Module Metadata & Unified Imports)
  [2] 统一配置超参数与策略常量 (Unified Configuration & Global Constants)
  [3] 核心数据结构与门控策略模型 (Core Data Structures: JointStateSnapshot, LoadPolicy)
  [4] 几何变换、配置校验与工具函数 (Geometry & Config Helpers: validate_transform 等)
  [5] Pinocchio 关节映射、元数据审计与限位窗口 (Pinocchio Mappings & Effective Bounds)
  [6] 物理场、重力矩计算与静态载荷上下文 (Physics, Gravity Torque & Load Context)
  [7] 任务空间种子库与腰部舒适窗推导 (Task-Space Seed Library & Waist Comfort Bounds)
  [8] 数值 DLS/SNS 单种子迭代核心 (Numerical DLS/SNS Engine: _SecondaryObjective, _sns_clamped_core)
  [9] 备用与历史求解算法 (Legacy Solvers: DirectPsiProblem & SSIK 兼容包装)
  [10] 多阶段候选解流水线与 SingleArmIKSolver / HAArmIKSolver 主类 (Multi-Stage Pipeline & Unified OOP Solver)
"""

from collections import deque
from collections.abc import Mapping
import copy
from dataclasses import dataclass, field, replace
import hashlib
import math
from pathlib import Path
import threading
import time
from types import MappingProxyType
from typing import Mapping as TypingMapping, Sequence
import warnings
import xml.etree.ElementTree as ET

import numpy as np
from scipy.linalg import lapack as _lapack
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

try:
    import pinocchio as pin
except Exception:
    pin = None

try:
    import ssik
except Exception:
    ssik = None

from ha_arm_ik.ik_base import (
    ARM_CONFIGS,
    E_JOINT,
    FailureReason,
    IKResult,
    ORI_TOL,
    POS_TOL,
    REF_DIR,
    SAFE_MARGIN,
    S_JOINT,
    SerialChain,
    Status,
    TOOL_AXIS_X,
    TOOL_CONE_HALF_ANGLE,
    WAIST_JOINT_NAMES,
    W_JOINT,
    W_TORQUE,
    bounded_joint_difference,
    candidate_sort_key,
    clip_to_bounds,
    closest_direction_on_cone,
    estimate_chain_max_reach_from_joint,
    get_fixed_joint_origin,
    get_urdf_root_link,
    invert_T,
    is_valid_status,
    joint_center_from_bounds,
    joint_difference,
    joint_dist,
    joint_margin,
    joint_origin_position_at_zero,
    joint_range_scale,
    make_result,
    T_origin,
    make_translation_T,
    multi_turn_mask_from_chain,
    normalize,
    normalize_cone_half_angle,
    pose_error,
    rpy_to_R,
    score_candidate,
    select_best_candidate,
    select_closest_bounded_branch,
    sew_angle_from_points,
    skew,
    target_frames,
    tool_cone_angle_error,
    tool_cone_residual,
    tool_direction_angle_error,
    tool_direction_from_T,
    tool_direction_residual,
    transform_point,
    unique_seed_list,
    update_candidate_final_cost_with_torque,
    validate_orientation_mode,
    wrap_to_pi,
)

# 公共导出接口
__all__ = [
    # 核心求解器与公共结果
    "SingleArmIKSolver",
    "HAArmIKSolver",
    "IKResult",
    "Status",
    "FailureReason",
    "make_result",
    # 状态快照与负荷策略
    "JointStateSnapshot",
    "LoadPolicy",
    # DLS 数值核心与种子管理 (模块级兼容入口)
    "pinocchio_ik_one_seed",
    "dls_task_terms",
    "build_seed_library",
    "select_library_seeds",
    "seed_library_target",
    "make_scatter_seeds",
    "derive_waist_comfort_bounds",
    "build_pinocchio_joint_maps",
    "configure_model_metadata",
    "active_q_to_pinocchio_q",
    "pinocchio_q_to_active_q",
    "active_periodic_mask",
    "active_multi_turn_mask",
    "evaluate_pose_physics",
    "prepare_load_context",
    "load_physics",
    "estimate_arm_max_reach",
    "validate_transform",
    "finite_vector",
    # Direct-Psi 与候选搜集
    "DirectPsiProblem",
    "solve_direct_psi",
    "collect_ssik_candidates_and_seeds",
    "collect_pinocchio_candidates_and_seeds",
]

SOLVER_VERSION = "merged-v15"   # 写入 last_load_diagnostics["version"]

# =============================================================================
# 2. 统一配置超参数与策略常量 (Unified Configuration & Global Constants)
# =============================================================================

# --- 2.1 Pinocchio DLS 基础数值与收敛容差参数 (来源于 solver_dls.py) ---
PIN_IK_MAX_ITER = 500
PIN_IK_DAMPING = 1e-6
PIN_IK_POS_TOL = 5e-4             # DLS 内部位姿容差上限；solver.py 实际取 min(本值, ik_common.POS_TOL)
PIN_IK_ORI_TOL = 5e-4             # 同上，实际取 min(本值, ik_common.ORI_TOL)
PIN_IK_MAX_STEP = 0.5             # 单次迭代关节步长 2-范数上限（弧度）
PIN_IK_STALL_ITERS = 12           # 停滞检测窗口：位姿未达标时每 12 次迭代检查一次进展（v6 为 25）
PIN_IK_STALL_RTOL = 0.025         # 窗口内主任务误差范数相对下降 < 2.5% -> 判定停滞，换下一个种子
                                  # （v6 为每 25 次迭代 5%；v7 窗口减半、阈值减半，判定速率与 v6 相当但检查更频繁。
                                  #   窗口 12 + 5% 会误杀收敛较慢的锁腰种子，tool 模式少数目标丢掉腰部回中解）
BOUND_SNAP_TOL = 1e-9             # 试探点越界不超过此值（浮点舍入）时吸附回边界，不整步拒绝
N_NOISE_PER_BASE = 16
# 扰动种子标准差（rad），从大到小：前几个扰动种子先探索不同解族（提高成功率、找低负荷解），
# 后面的小扰动做局部细化。v5 为 [0.02, 0.05, 0.10, 0.20] 且先排完 q_prev 的 16 个小扰动，
# max_seeds=10 时扰动种子几乎与 q_prev 重合，远目标成功率 97% -> v6 100%（合成基准 100 例）。
NOISE_SCALES = [0.60, 0.30, 0.15, 0.05]

# =============================================================================
# 2. 零空间防漂移与流形稳态收敛参数

# --- 2.2 零空间防漂移与流形稳态收敛参数 ---
# =============================================================================
NULLSPACE_GAIN = 0.30             # 手臂冗余关节拉向 q_ref 物理坐标的次任务增益
WAIST_NULL_GAIN = 0.50            # 腰部关节拉向 q_ref 腰部角（默认 q_ready）的次任务增益
NULLSPACE_ERR_SCALE = 0.06        # 零空间权重 = (达标 1.5 / 未达标 1.0) / (1 + (|e| / 0.06)^2)，
                                  # e 为位置按 characteristic_length(0.5 m) 缩放后的主任务误差：
                                  # |e| = 0.06 时权重减半（约 30 mm 或 3.4°），误差越小越接近全功率
POSTURE_SETTLE_TOL = 1e-4         # 末端收敛后，零空间步长小于此阈值(rad)才算稳态收敛
MAX_SETTLE_EXTRA_ITERS = 50       # 末端达标后继续在零空间整理姿态的安全上限（v8 恢复 v6 取值；v7 的 15
                                  # 在收敛前截断：tool_dir 的 SETTLE_LIMIT 次数 2 -> 8、次任务代价平均 +1.9%）
SETTLE_COST_RTOL = 1e-3           # 姿态整理阶段：单步次任务代价相对下降 < 0.1% 记为“慢步”
SETTLE_PATIENCE = 3               # 连续 3 个慢步即视为已收敛（收敛判据，v8 恢复 v6 取值；v7 为 2）
POSE_PEAK_BAND_FRACTION = 0.25    # 单种子内择优的峰值容差带 = 0.25 * LoadPolicy.peak_tie（默认 0.005）
NO_SETTLE_NULL_STEP_RATIO = 2.0   # v15：posture_settle=False（续算）时未达标的零空间原始步 |w·D z| <= 2 × |e|
                                  # （e 为缩放后的主任务误差，与双臂 NULL_STEP_RATIO 相同）：回中幅度与手的
                                  # 移动量成正比；不限幅时 1 mm 的目标移动也会让关节在收敛过程中变 0.1 rad
LOAD_EARLY_STOP_RATIO = 0.70      # 低负荷早停阈值默认值，实际取 min(本值, soft_ratio)（v6 默认 0.60）

# =============================================================================
# 3. 无状态腰部自适应释放参数（路径无关，去返程完全对称）

# --- 2.3 无状态腰部自适应释放参数（去返程完全对称） ---
# =============================================================================
# 逆度量 D 的腰部对角元 = (1-α)/WAIST_TASK_WEIGHT + α/WAIST_TASK_WEIGHT_FAR（手臂 = 1），
# α 为释放系数（0 = 近目标，1 = 远目标或负荷释放）。D 越小，腰部单位步长在 DLS 中越“贵”。
WAIST_TASK_WEIGHT = 6.0           # 近目标：腰部步长代价为手臂的 6 倍（D = 1/6，不是锁死）
WAIST_TASK_WEIGHT_FAR = 1.0       # 远目标完全释放：腰部与手臂同价（D = 1），腰部按需参与够远
WAIST_RELEASE_START = 0.75        # 肩->目标距离 / 最大臂展 达到此比例开始放腰
WAIST_RELEASE_END = 0.94          # 达到此比例腰完全释放
WAIST_RETURN_RELEASE_RAD = 0.20   # 返程解锁尺度(rad)：腰偏离 q_ref 时保持释放通道以便退回

# =============================================================================
# 4. 电机重力矩卸载、重心稳定与限位保护参数

# --- 2.4 电机重力矩卸载、重心稳定与限位保护参数 ---
# =============================================================================
GRAVITY_TORQUE_GAIN = 0.04        # 默认绝对归一化重力矩卸载增益
RELATIVE_GRAVITY_GAIN = 0.0       # 可选的相对参考姿态重力矩保持增益
COM_XY_GAIN = 0.12                # 零空间整机质心(CoM)水平投影稳定增益（代价 0.5 * 0.12 * |Δc|^2，Δc 单位 m）
JOINT_LIMIT_GAIN = 0.03           # 限位排斥增益；排斥区宽度 = max(SAFE_MARGIN, 0.02) rad
WAIST_YAW_VIRTUAL_LIM = np.deg2rad(90.0)   # v8：重力无关腰关节（偏航类）的舒适半窗，见 derive_waist_comfort_bounds

# =============================================================================
# 5. 腰部舒适窗（v8，详见 derive_waist_comfort_bounds）

# --- 2.5 腰部舒适窗自适应参数 ---
# =============================================================================
WAIST_COMFORT_SOFT_RATIO = 0.70   # 未配置 load_soft_ratio 时的默认值（与 LoadPolicy.soft_ratio 默认一致）
WAIST_COMFORT_MAX_HALF = 0.5 * np.pi       # 任何腰关节的舒适半窗不超过 90°（躯干最多转四分之一圈）
WAIST_GRAVITY_FREE_COS = math.cos(math.radians(30.0))  # 关节轴与重力夹角 < 30° 视为重力无关（偏航类）
WAIST_COMFORT_EDGE_SEEDS = 1      # 舒适窗内无合格解时，取窗内主任务误差最小的几个迭代终点作为放开后的续算种子
WAIST_COMFORT_CONTINUATION_STEP = 0.10   # 续算判据：目标相对 FK(q_prev) 位置差 <= 0.1 × 臂展且姿态差 <= 0.1 rad
                                         # （约 0.1 rad 关节运动，雅可比线性化仍有效）时不加舒适窗

# =============================================================================

# --- 2.6 任务空间种子库参数 ---
# v13：任务空间种子库（最近邻种子）
# =============================================================================
SEED_LIBRARY_SIZE = 16384         # 初始化时在有效限位内均匀采样的关节角个数（建库约 150 ms、2.8 MB）；0 = 关闭
SEED_LIBRARY_SEEDS = 3            # 每个常规搜索阶段插入的种子库种子个数（排在 q_prev / 锁腰 / q_ready 之后）
SEED_LIBRARY_MIN_SEPARATION = 0.5 # 入选种子之间、与本阶段已排队种子之间的关节差 ∞-范数下限（rad）
SEED_LIBRARY_POOL = 64            # 按任务空间距离预选的候选个数（再按关节差去重）
SEED_LIBRARY_RNG_SEED = 13        # 采样随机数种子（与求解用的 RNG_SEED 无关，库内容可复现）


# =============================================================================

# --- 2.7 全局物理环境与主求解器配置 (来源于 solver.py) ---
G_WORLD = np.array([0.0, 0.0, -9.81], dtype=float)

# ---------------- direct-psi solve config ----------------
PORTFOLIO_DIRECT_MAX_NFEV = 150
PORTFOLIO_DIRECT_POS_TOL = 5e-4
PORTFOLIO_DIRECT_ORI_TOL = 5e-4
PORTFOLIO_DIRECT_PSI_TOL = 1.5

PORTFOLIO_Q_REF_MODE = "ready"  # "ready" or "prev"
DIRECT_LOCAL_NOISE_SCALES = (0.002, 0.005, 0.01, 0.02, 0.04)

# ---------------- Optional modules ----------------
USE_DIRECT_PSI = False
USE_LOCAL_SEEDS = False
USE_SSIK = False

USE_PINOCCHIO_IK = True
USE_TORQUE_COST = True

# SSIK config
SSIK_RESPECT_LIMITS = True
SSIK_ALLOW_REFINEMENT = True
SSIK_REFINEMENT_MAX_ITERS = 30
SSIK_MAX_SOLUTIONS = True
SSIK_USE_READY_AS_SEED = True
SSIK_TOP_K_FOR_DIRECT = 10

PIN_TOP_K_FOR_DIRECT = 15
RNG_SEED = 120
MAX_JOINT_JUMP_RAD = np.pi  # 默认“绕大圈”阈值：单关节相对 q_prev 超过半圈即视为绕大圈
# v7：q_prev 的解已合格且腰部偏离 q_ready 不超过该值（rad，逐关节最大值）时跳过锁腰种子；None = v6 顺序
LOCKED_SEED_SKIP_TOL = 0.02
JOINT_JUMP_POLICIES = ("reject", "prefer")
# v9：续算（目标相对 FK(q_prev) 的主任务误差在 continuation_tol 以内，即路点跟踪）时先跑续算阶段，
# 只用与 q_prev 连续的候选（见 collect_pinocchio_candidates_and_seeds）。原因：ik_candidate 的
# W_WAIST_RETURN（2000）远大于 W_SMOOTH（8），逐帧在全部种子里择优会在“锁腰（腰回 q_ready）”与“随动”
# 两个分支间来回切换，造成 0.8~2.5 rad 的单步跳变；设了 max_joint_jump_rad 时切换点的跳变超限，
# q_prev 不再更新，后续路点全部失败。False = v8 行为。
CONTINUATION_LOCAL = True
# v9：续算阶段的单关节步长上限（rad，相对 q_prev，与 joint_jump_metrics 同一度量）：随动续算的搜索
# 范围是 q_prev ± 该值；步长更大的候选（典型：q_prev 腰部远离 q_ready 时的锁腰解）不算局部解。
# 取值依据（bench/v9，真实 ik_candidate）：0.1~0.2 普通跟踪逐帧步长 <= 0.020、腰部不漂；0.3 时随动解
# 在负荷容差带边界与锁腰解来回切换（单步 0.198，与 v8 相同）；0.1 时腰部斜坡跟踪在框边上反复切换
# （单步均值 0.081，0.15 时 0.034）；腰部大幅跟踪单步上限即该值，0.15 无回退。
CONTINUATION_MAX_STEP = 0.15
# v15：续算阶段不做达标后的姿态整理（DLS posture_settle=False：位姿达标、负荷合格且离限位 >= SAFE_MARGIN 的
# 第一个迭代点即返回），q_prev 本身已是可接受解时原样返回、不再跑锁腰续算。v14 在续算框内照常整理：目标不动、
# q_prev 已是解时每次调用关节仍变 0.10~0.15 rad，伺服循环里等于机器人自己在动。回中只来自收敛过程中的
# 零空间步，与手的移动量成正比（与双臂 continuation 相同）。True = v14 行为（等价回放用）。
CONTINUATION_POSTURE_SETTLE = False
# v9：续算阶段 DLS 单步（2-范数）上限 = 该比例 × CONTINUATION_MAX_STEP：从框中心出发的首步不会越出续算框
# （越框会被 SNS 钳制、线搜索停滞）
CONTINUATION_DLS_STEP_FRACTION = 0.5
# v13：舒适阶段提前回退（dls_waist_comfort_early_fallback，默认 True）：q_prev / 锁腰 / q_ready / 种子库种子
# 都跑完仍没有可接受候选（满足全部硬约束）时结束舒适阶段、回退到有效限位（目标需要舒适窗外的腰部时，
# v12 要在窗内白跑满 max_seeds 个种子）。没有种子库时检查点放在第 WAIST_COMFORT_PATIENCE 个名额之后。
WAIST_COMFORT_EARLY_FALLBACK = True
WAIST_COMFORT_PATIENCE = 4
LIBRARY_PENDING = "library_pending"   # 搜索队列中的种子库占位（排到时展开，见 collect_pinocchio_candidates_and_seeds）
COMFORT_CHECKPOINT = "comfort_checkpoint"   # 舒适阶段提前回退的检查点（不占名额）
# v13：常规搜索只找到“跳变超限”的位姿解（其余硬约束都满足）时，在跳变框 q_prev ± max_joint_jump_rad 内
# 重搜一次（阶段 "jump_box"）。v9 的跳变框只对 < π 的上限生效，默认 π 时 DLS 不知道跳变上限，
# 会收敛到要转过半圈以上的解、整批被拒（JOINT_JUMP_LIMIT）。False = v12 行为。
JUMP_LIMIT_RESCUE = True
# v13：锁腰种子可达性检查（见 locked_arm_reach_bound）：目标超出锁腰时手臂的严格可达范围就跳过锁腰种子
# （v12 会让它迭代到停滞，平均 37 次迭代）。False = v12 行为。
LOCKED_REACH_CHECK = True
# v13：续算类求解（目标就在 FK(q_prev) 附近，即路点跟踪）不排种子库种子。库样本是全局的，收敛到的多是离
# q_prev 很远的分支；q_prev 附近的解只差限位裕度（NEAR_LIMIT）时，择优会换成这类远处的合格解（跟踪基准、
# 未设跳变上限：单步 > 0.3 rad 由 v12 的 1 次增至 5 次，最大 3.05 rad）。续算类求解按 v12 的种子顺序搜索。
# True = 续算类求解也用种子库。
SEED_LIBRARY_CONTINUATION = False
# v13：种子库只取决于运动学模型、活动关节、末端帧、有效限位、q_ready、舒适窗与库大小，同一模型的多个
# 求解器实例（如左右臂各建多个实例、测试反复构造）共享同一份只读库，不重复建库
_SEED_LIBRARY_CACHE = {}
_SEED_LIBRARY_CACHE_MAX = 8
_SEED_LIBRARY_CACHE_LOCK = threading.Lock()


# =============================================================================
# 3. 核心数据结构与门控策略模型 (Core Data Structures & Policy Models)
# =============================================================================


@dataclass(frozen=True)
class JointStateSnapshot:
    """不可变完整关节状态快照。严禁将旧缓存状态重新打时间戳冒充新鲜状态。"""

    positions: Mapping[str, float]
    received_monotonic: float = field(default_factory=time.monotonic)
    source_age_at_receipt_s: float | None = None

    def __post_init__(self):
        values = {str(k): float(v) for k, v in self.positions.items()}
        if not values or not all(math.isfinite(v) for v in values.values()):
            raise ValueError("joint state must be nonempty and finite")
        if not math.isfinite(self.received_monotonic):
            raise ValueError("invalid monotonic receipt time")
        age = self.source_age_at_receipt_s
        if age is not None and not math.isfinite(age):
            raise ValueError("invalid source age")
        object.__setattr__(self, "positions", MappingProxyType(values))

    @classmethod
    def from_joint_state(cls, msg, *, now_ros_ns=None, received_monotonic=None):
        names, positions = list(msg.name), list(msg.position)
        if (
            not names
            or len(names) != len(positions)
            or len(set(names)) != len(names)
        ):
            raise ValueError(
                "JointState names/positions are empty, duplicated or mismatched"
            )
        stamp = getattr(getattr(msg, "header", None), "stamp", None)
        source_age = None
        if stamp is not None and now_ros_ns is not None:
            sec = int(getattr(stamp, "sec", getattr(stamp, "secs", 0)))
            nsec = int(getattr(stamp, "nanosec", getattr(stamp, "nsecs", 0)))
            stamp_ns = sec * 1_000_000_000 + nsec
            if stamp_ns != 0:
                source_age = (int(now_ros_ns) - stamp_ns) / 1e9
        return cls(
            dict(zip(names, positions)),
            time.monotonic()
            if received_monotonic is None
            else received_monotonic,
            source_age,
        )

    def freshness_error(self, max_age_s, *, require_source_age=False, now=None):
        now = time.monotonic() if now is None else now
        receipt_age = now - self.received_monotonic
        if receipt_age < -0.05:
            return "receipt_time_in_future"
        if receipt_age > max_age_s:
            return "receipt_state_stale"
        if self.source_age_at_receipt_s is None:
            return "source_timestamp_unverified" if require_source_age else None
        age = self.source_age_at_receipt_s + receipt_age
        if age < -0.05:
            return "source_timestamp_in_future"
        if age > max_age_s:
            return "source_state_stale"
        return None


@dataclass(frozen=True)
class LoadPolicy:
    """静态负荷优化与硬门控策略。allowable_efforts 必须显式提供经确认的正有限值。"""

    allowable_efforts: Mapping[str, float]
    guard_joint_names: Sequence[str] | None = None
    joint_weights: Mapping[str, float] = field(default_factory=dict)
    absolute_weight: float = 0.05
    excess_weight: float = 0.50
    relative_weight: float = 0.0
    soft_ratio: float = 0.70
    # 低负荷早停阈值；None（默认）= min(LOAD_EARLY_STOP_RATIO, soft_ratio)。v6 默认固定 0.60
    early_stop_ratio: float | None = None
    peak_tie: float = 0.02  # 峰值负荷率容差带：差距在此以内视为同档，再比较姿态/连续性代价

    def __post_init__(self):
        capacities = {str(k): float(v) for k, v in self.allowable_efforts.items()}
        weights = {str(k): float(v) for k, v in self.joint_weights.items()}
        if not capacities or any(
            not math.isfinite(v) or v <= 0.0 for v in capacities.values()
        ):
            raise ValueError(
                "explicit positive finite allowable_efforts are required; no fallback"
            )
        if any(not math.isfinite(v) or v <= 0.0 for v in weights.values()):
            raise ValueError("joint load weights must be positive and finite")
        for name in (
            "absolute_weight",
            "excess_weight",
            "relative_weight",
            "peak_tie",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be nonnegative and finite")
        if not 0.0 < self.soft_ratio < 1.0:
            raise ValueError("soft_ratio must lie strictly between 0 and 1")
        if self.early_stop_ratio is None:
            object.__setattr__(
                self,
                "early_stop_ratio",
                min(LOAD_EARLY_STOP_RATIO, float(self.soft_ratio)),
            )
        if not 0.0 < self.early_stop_ratio <= self.soft_ratio:
            raise ValueError(
                "early_stop_ratio must be positive and <= soft_ratio"
            )
        guards = (
            None
            if self.guard_joint_names is None
            else tuple(self.guard_joint_names)
        )
        if guards is not None and (not guards or len(set(guards)) != len(guards)):
            raise ValueError("guard_joint_names must be nonempty and unique")
        object.__setattr__(
            self, "allowable_efforts", MappingProxyType(capacities)
        )
        object.__setattr__(self, "joint_weights", MappingProxyType(weights))
        object.__setattr__(self, "guard_joint_names", guards)
        # 非字段缓存：按关节名元组缓存 (容许力矩, 权重) 向量，避免迭代内重复查字典
        object.__setattr__(self, "_vector_cache", {})

    def vectors(self, names):
        key = tuple(names)
        cached = self._vector_cache.get(key)
        if cached is not None:
            return cached
        missing = [n for n in key if n not in self.allowable_efforts]
        if missing:
            raise ValueError(f"missing confirmed allowable effort for: {missing}")
        caps = np.array([self.allowable_efforts[n] for n in key], dtype=float)
        weights = np.array(
            [self.joint_weights.get(n, 1.0) for n in key], dtype=float
        )
        caps.setflags(write=False)
        weights.setflags(write=False)
        self._vector_cache[key] = (caps, weights)
        return caps, weights

    def _cost_terms(self, tau, caps, weights, tau_reference, with_gradient):
        """cost_gradient 与单种子快速次任务共用的负荷代价运算（同一份代码，结果逐位一致）。

        tau / tau_reference 须已校验为有限向量；with_gradient=False 时不计算 d cost / d tau。
        返回 (absolute, excess, relative, dcost_dtau 或 None)。
        """
        r = tau / caps
        excess = np.maximum(np.abs(r) - self.soft_ratio, 0.0)
        n = len(tau)

        abs_cost = 0.5 * self.absolute_weight * float(np.sum(weights * r * r)) / n
        excess_cost = (
            0.5 * self.excess_weight * float(np.sum(weights * excess * excess)) / n
        )
        dcost_dtau = None
        if with_gradient:
            dcost_dtau = (
                self.absolute_weight * weights * r / caps
                + self.excess_weight * weights * excess * np.sign(r) / caps
            ) / n

        rel_cost = 0.0
        if self.relative_weight > 0.0:
            diff = (tau - tau_reference) / caps
            rel_cost = (
                0.5 * self.relative_weight * float(np.sum(weights * diff * diff)) / n
            )
            if with_gradient:
                dcost_dtau += self.relative_weight * weights * diff / caps / n
        return abs_cost, excess_cost, rel_cost, dcost_dtau

    def cost_gradient(self, tau, d_tau_d_active, names, *, tau_reference=None):
        tau = finite_vector(tau, len(names), "gravity efforts")
        J = np.asarray(d_tau_d_active, dtype=float)
        if J.ndim != 2 or J.shape[0] != len(names) or not np.all(np.isfinite(J)):
            raise ValueError("invalid gravity effort derivative")
        caps, weights = self.vectors(names)
        ref = None
        if self.relative_weight > 0.0:
            ref = finite_vector(
                tau_reference, len(names), "reference gravity efforts"
            )
        abs_cost, excess_cost, rel_cost, dcost_dtau = self._cost_terms(
            tau, caps, weights, ref, True
        )

        gradient = J.T @ dcost_dtau
        cost = float(abs_cost + excess_cost + rel_cost)
        if not np.all(np.isfinite(gradient)) or not math.isfinite(cost):
            raise ValueError("non-finite load objective")
        return (
            cost,
            gradient,
            {
                "absolute": float(abs_cost),
                "excess": float(excess_cost),
                "relative": float(rel_cost),
            },
        )

    def evaluate(self, tau, names):
        tau = finite_vector(tau, len(names), "gravity efforts")
        caps, _ = self.vectors(names)
        with np.errstate(over="ignore", invalid="ignore"):
            ratios = np.abs(tau) / caps
        if not np.all(np.isfinite(ratios)):
            raise ValueError("non-finite normalized load; check capacities/model")
        i = int(np.argmax(ratios))
        return {
            "ratios": ratios,
            "peak_ratio": float(ratios[i]),
            "mean_square_ratio": float(np.mean(ratios ** 2)),
            "peak_joint": names[i],
            "load_ok": bool(np.all(ratios <= 1.0)),
            "overloaded_joints": [
                n for n, r in zip(names, ratios) if r > 1.0
            ],
        }


# =============================================================================
# 4. 几何变换、配置校验与工具函数 (Geometry & Configuration Helpers)
# =============================================================================
# 基础数值校验与数据结构（2πk 分支对齐 / 去重用 ik_common 的 select_closest_bounded_branch、
# bounded_joint_difference、joint_difference）
# =============================================================================
def finite_vector(value, size=None, name="vector"):
    result = np.asarray(value, dtype=float)
    if result.ndim != 1 or (size is not None and result.size != size):
        raise ValueError(f"{name}: expected a 1-D vector of size {size}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name}: non-finite values")
    return result.copy()


def _vec_norm(x):
    """1-D float 向量的 2-范数，返回 Python float。

    与 float(np.linalg.norm(x)) 逐位一致（numpy 对 1-D 向量的实现就是 sqrt(x.dot(x))），
    但省去了 norm 的参数分派开销；只用于内部热路径上已知为 1-D float64 的数组。
    """
    return math.sqrt(x.dot(x))


_RIGID_BOTTOM_ROW = np.array([0.0, 0.0, 0.0, 1.0])
_EYE3 = np.eye(3)


def validate_transform(value):
    T = np.asarray(value, dtype=float)
    if T.shape != (4, 4) or not np.isfinite(T).all():
        raise ValueError("target must be a finite 4x4 transform")
    rot = T[:3, :3]
    # 判定与 np.allclose / np.isclose(rtol=0) 逐位等价：参考值有限时 isclose(x, y) ⇔ |x - y| <= atol
    # （含 inf / NaN 情形均为 False）；省去 isclose 的通用分派开销
    if (
        not (np.abs(T[3] - _RIGID_BOTTOM_ROW) <= 1e-9).all()
        or not (np.abs(rot.T @ rot - _EYE3) <= 1e-6).all()
        or not abs(np.linalg.det(rot) - 1.0) <= 1e-6
    ):
        raise ValueError("target rotation/bottom row is not a rigid transform")
    return T.copy()


def _config_flag(cfg, key, default):
    """布尔开关：只接受 True / False（含 numpy.bool_），"yes" / 1 / None 之类一律报错。"""
    value = cfg.get(key, default)
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{key} must be True or False")
    return bool(value)


def _finite_angle(value, message, *, positive):
    """有限角度（rad）：bool / 非数值 / NaN / inf / 负值报错；positive=True 时 0 也报错。"""
    try:
        value = float("nan") if isinstance(value, (bool, np.bool_)) else float(value)
    except (TypeError, ValueError):
        value = float("nan")
    if not math.isfinite(value) or value < 0.0 or (positive and value == 0.0):
        raise ValueError(message)
    return value


def _int_at_least(value, minimum, message):
    """整数（含 numpy 整数；bool / float / 字符串报错）且 >= minimum。"""
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < minimum
    ):
        raise ValueError(message)
    return int(value)


def joint_jump_metrics(q, q_prev, periodic_mask, jump_limits):
    """单关节相对 q_prev 的位移（物理坐标差，仅无界 continuous 关节折叠到 [-π, π)）。

    候选在此之前已对齐到最接近 q_prev 的合法 2πk 分支，所以这里的差值就是执行时的真实转动量。
    返回 (最大位移 rad, 最大位移所在关节下标, 是否全部 <= jump_limits)。jump_limits 为 None 表示不限。
    """
    d = np.abs(joint_difference(q, q_prev, periodic_mask))
    i = int(np.argmax(d)) if d.size else 0
    ok = True if jump_limits is None else bool(np.all(d <= jump_limits + 1e-9))
    return (float(d[i]) if d.size else 0.0), i, ok


def unique_add(solutions, q, tol=1e-4):
    q = np.asarray(q, dtype=float)

    for old in solutions:
        if np.linalg.norm(q - old) < tol:
            return False

    solutions.append(q.copy())
    return True



def json_safe(value):
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    return value



# =============================================================================
# 5. Pinocchio 关节映射、元数据审计与限位窗口 (Pinocchio Mappings & Bounds)
# =============================================================================

# =============================================================================
# Pinocchio helpers
# =============================================================================
def require_pinocchio():
    if pin is None:
        raise RuntimeError(
            "pinocchio is not installed but USE_PINOCCHIO_IK or "
            "USE_TORQUE_COST is enabled.  Install with:  pip install pin"
        )


def load_pinocchio_model(urdf_path):
    """Load URDF into Pinocchio model + data."""
    model = pin.buildModelFromUrdf(str(urdf_path))
    data = model.createData()
    return model, data


def get_pinocchio_frame_id(model, frame_name):
    """按名字取 Pinocchio 帧 id，优先取连杆帧（pin.FrameType.BODY）。

    URDF 中关节与连杆同名时（例如关节按子连杆命名），同一个名字会对应 JOINT 和 BODY 两个帧，
    Pinocchio 3.x 起不带类型的 getFrameId 会抛 ValueError("Several frames match the filter")。
    ee_link / base_link 都是连杆，因此按 BODY 查询；没有同名 BODY 帧时再按名字唯一匹配其它类型的帧
    （例如手工添加的 OP_FRAME；同名多帧时仍抛 ValueError）。找不到时抛 KeyError。
    """
    body = pin.FrameType.BODY
    if model.existFrame(frame_name, body):
        return model.getFrameId(frame_name, body)
    if not model.existFrame(frame_name):
        available = [model.frames[i].name for i in range(model.nframes)]
        raise KeyError(
            f"Frame '{frame_name}' not found in Pinocchio model. "
            f"Available frames ({len(available)}): {available[:30]} ..."
        )

    return model.getFrameId(frame_name)


# =============================================================================
# Pinocchio 映射与模型物理审计
# =============================================================================
def build_pinocchio_joint_maps(model, active_joint_names):
    """建立标量关节映射（含 continuous nq=2/nv=1 关节）。"""
    all_specs = {}
    for jid in range(1, model.njoints):
        joint = model.joints[jid]
        continuous = (
            joint.nq == 2 and joint.nv == 1 and "RUB" in joint.shortname()
        )
        if joint.nv != 1 or (joint.nq != 1 and not continuous):
            raise ValueError(
                f"{model.names[jid]}: unsupported nq/nv {joint.nq}/{joint.nv}"
            )
        raw = float(model.effortLimit[joint.idx_v])
        all_specs[model.names[jid]] = {
            "joint_id": jid,
            "idx_q": joint.idx_q,
            "idx_v": joint.idx_v,
            "nq": joint.nq,
            "continuous": continuous,
            "raw_effort": raw if (np.isfinite(raw) and raw > 0.0) else np.nan,
        }

    names = list(active_joint_names)
    if len(names) != len(set(names)) or (set(names) - set(all_specs)):
        raise ValueError("invalid or missing active joint names")

    efforts = np.array(
        [all_specs[n]["raw_effort"] for n in names], dtype=float
    )
    return {
        "all_specs": all_specs,
        "all_names": list(all_specs),
        "active_names": names,
        "name_to_joint_id": {n: s["joint_id"] for n, s in all_specs.items()},
        "name_to_q_idx": {n: s["idx_q"] for n, s in all_specs.items()},
        "name_to_v_idx": {n: s["idx_v"] for n, s in all_specs.items()},
        "active_q_indices": [all_specs[n]["idx_q"] for n in names],
        "active_v_indices": [all_specs[n]["idx_v"] for n in names],
        "active_continuous": np.array(
            [all_specs[n]["continuous"] for n in names], dtype=bool
        ),
        "effort_limits": efforts.copy(),
        "urdf_effort_limits": efforts.copy(),
    }


def active_periodic_mask(pin_maps):
    """无界 continuous 活动关节（角度差折叠到 [-π, π)）。configure_model_metadata 之前退回 active_continuous。"""
    return pin_maps.get("active_periodic", pin_maps["active_continuous"])


def active_multi_turn_mask(pin_maps):
    """可做 2πk 分支对齐的活动关节（continuous 或限位跨度 >= 2π 的 revolute）；未配置时依次退回
    active_periodic / active_continuous。"""
    return pin_maps.get("active_multi_turn", active_periodic_mask(pin_maps))


def _active_q_layout(pin_maps):
    """活跃关节在 Pinocchio q 向量中的索引布局（首次调用时构建并缓存在 pin_maps 中）。

    返回 (标量关节在活跃向量中的位置, 其 idx_q, continuous 关节位置, 其 idx_q)；
    continuous 关节在 q 中占 (cos, sin) 两个分量。
    """
    layout = pin_maps.get("_active_q_layout")
    if layout is None or layout[0] != tuple(pin_maps["active_names"]):
        specs = [pin_maps["all_specs"][n] for n in pin_maps["active_names"]]
        cont = np.array([s["continuous"] for s in specs], dtype=bool)
        idx_q = np.array([s["idx_q"] for s in specs], dtype=np.intp)
        pos = np.arange(len(specs), dtype=np.intp)
        layout = (
            tuple(pin_maps["active_names"]),
            pos[~cont],
            idx_q[~cont],
            pos[cont],
            idx_q[cont],
        )
        pin_maps["_active_q_layout"] = layout
    return layout[1:]


def active_q_to_pinocchio_q(model, pin_maps, q_active):
    """活跃关节角 -> Pinocchio 全身 q。

    非活跃关节取冻结的全身状态模板 q_template（strict 模式由实测快照构建），没有模板时取 neutral；
    continuous 关节写入 (cos, sin)。迭代热路径，索引布局预先缓存、整体向量化赋值。
    """
    values = np.asarray(q_active, dtype=float)
    if values.shape != (len(pin_maps["active_names"]),) or not np.all(
        np.isfinite(values)
    ):
        raise ValueError(
            f"active q: expected a finite 1-D vector of size {len(pin_maps['active_names'])}"
        )
    template = pin_maps.get("q_template")
    q = (
        pin.neutral(model)
        if template is None
        else np.array(template, dtype=float, copy=True)
    )
    scalar_pos, scalar_idx, cont_pos, cont_idx = _active_q_layout(pin_maps)
    q[scalar_idx] = values[scalar_pos]
    if cont_pos.size:
        q[cont_idx] = np.cos(values[cont_pos])
        q[cont_idx + 1] = np.sin(values[cont_pos])
    return q


def _make_q_builder(model, pin_maps):
    """单种子内复用的 active_q_to_pinocchio_q：模板与索引布局只取一次，结果与公共函数逐位一致。

    种子求解期间 pin_maps["q_template"] 不变（求解入口设置、结束后移除），因此可以安全缓存。
    """
    n_active = len(pin_maps["active_names"])
    template = pin_maps.get("q_template")
    base = (
        pin.neutral(model)
        if template is None
        else np.array(template, dtype=float, copy=True)
    )
    scalar_pos, scalar_idx, cont_pos, cont_idx = _active_q_layout(pin_maps)
    cont_idx_sin = cont_idx + 1
    has_cont = bool(cont_pos.size)
    shape = (n_active,)
    # v15：全部为标量关节、按活跃顺序连续存放时用切片赋值（纯拷贝，与花式索引逐位相同）
    scalar_range = _contiguous_range(scalar_idx)
    direct = (
        not has_cont
        and scalar_range is not None
        and np.array_equal(scalar_pos, np.arange(n_active))
    )
    q_slice = slice(*scalar_range) if direct else None

    def build(q_active):
        values = np.asarray(q_active, dtype=float)
        # 有限性：x·x 有限 => 全部有限；为 inf/NaN（或极端值溢出）时再逐元素判定，异常与原实现相同
        if values.shape != shape or (
            not math.isfinite(values.dot(values)) and not np.isfinite(values).all()
        ):
            raise ValueError(
                f"active q: expected a finite 1-D vector of size {n_active}"
            )
        q = base.copy()
        if direct:
            q[q_slice] = values
            return q
        q[scalar_idx] = values[scalar_pos]
        if has_cont:
            angles = values[cont_pos]
            q[cont_idx] = np.cos(angles)
            q[cont_idx_sin] = np.sin(angles)
        return q

    return build


def pinocchio_q_to_active_q(pin_maps, q_full, reference=None):
    """提取活跃关节标量角度；若提供 reference，则在合法限位内对齐至最接近 reference 的 2πk 分支。"""
    values = []
    for name in pin_maps["active_names"]:
        s = pin_maps["all_specs"][name]
        if s["continuous"]:
            values.append(
                math.atan2(q_full[s["idx_q"] + 1], q_full[s["idx_q"]])
            )
        else:
            values.append(float(q_full[s["idx_q"]]))
    q = np.array(values, dtype=float)

    if reference is not None:
        ref = finite_vector(reference, len(q), "angle branch reference")
        if "active_lower" in pin_maps and "active_upper" in pin_maps:
            q = select_closest_bounded_branch(
                q,
                ref,
                pin_maps["active_lower"],
                pin_maps["active_upper"],
                active_multi_turn_mask(pin_maps),
            )
        else:
            mask = pin_maps["active_continuous"]
            q[mask] = ref[mask] + joint_difference(q, ref, mask)[mask]
    return q


def _legacy_shoulder_joint(pin_maps):
    """v7 及以前的肩关节约定：第一个名字以 "_Link2" 结尾的非腰部活跃关节；没有则返回 None。"""
    return next(
        (
            n
            for n in pin_maps["active_names"]
            if n not in WAIST_JOINT_NAMES and n.endswith("_Link2")
        ),
        None,
    )


def estimate_arm_max_reach(pin_model, pin_maps, ee_frame_id, shoulder_joint=None):
    """肩关节原点 -> TCP 距离的上界：沿运动链逐段累加关节原点偏移长度 + TCP 帧偏移长度。

    v8：从 TCP 帧所在关节沿父关节链回溯到肩关节，累加肩关节之后每个关节原点的平移长度，
    再加 TCP 帧相对其父关节的平移长度（与 robot_model.estimate_chain_max_reach_from_joint 同一定义；
    各段能同时拉直时等于真实最大臂展，否则是上界）。v7 只取 Link3 / Link5 两段 + TCP 偏移，
    偏移分布在其它关节上的模型会低估（合成模型 0.62 m，实际 1.15 m），导致腰部过早释放。

    shoulder_joint=None 时沿用旧约定（第一个以 "_Link2" 结尾的非腰部活跃关节）。
    找不到肩关节、肩关节不在 TCP 的祖先链上或臂展为 0 时返回 None（腰部按距离释放随之关闭）。
    """
    name = _legacy_shoulder_joint(pin_maps) if shoulder_joint is None else shoulder_joint
    if name is None or name not in pin_maps["name_to_joint_id"]:
        return None
    shoulder_id = pin_maps["name_to_joint_id"][name]
    frame = pin_model.frames[ee_frame_id]
    reach = float(np.linalg.norm(frame.placement.translation))
    jid = int(frame.parentJoint if hasattr(frame, "parentJoint") else frame.parent)
    while jid > 0 and jid != shoulder_id:
        reach += float(np.linalg.norm(pin_model.jointPlacements[jid].translation))
        jid = int(pin_model.parents[jid])
    if jid != shoulder_id or reach <= 1e-9:
        return None
    return reach



# =============================================================================
# Pinocchio DLS candidate orchestration
# =============================================================================
def locked_arm_reach_bound(pin_model, pin_maps, ee_frame_id):
    """v13：(手臂首关节编号, 首关节原点 -> TCP 距离的严格上界)；腰部以外第一个活动关节为手臂首关节。

    上界 = 首关节之后各关节原点偏移长度 + TCP 帧偏移长度（三角不等式；转动关节不改变到其原点的距离）。
    路径上有非转动关节（如移动关节）或 TCP 不在首关节的子树上时返回 None（不做检查）。
    """
    names = pin_maps["active_names"]
    arm_ids = [
        int(pin_maps["name_to_joint_id"][n]) for n in names if n not in WAIST_JOINT_NAMES
    ]
    if not arm_ids or len(arm_ids) == len(names):
        return None
    first = min(arm_ids)
    frame = pin_model.frames[int(ee_frame_id)]
    bound = float(np.linalg.norm(frame.placement.translation))
    joint = int(frame.parentJoint)
    while joint != first:
        if joint <= 0:
            return None
        if not pin_model.joints[joint].shortname().startswith("JointModelR"):
            return None
        bound += float(np.linalg.norm(pin_model.jointPlacements[joint].translation))
        joint = int(pin_model.parents[joint])
    if not pin_model.joints[first].shortname().startswith("JointModelR"):
        return None
    return first, bound




def _resolve_virtual_windows(
    virtual_half_windows, active_names, n_active
):
    """解析可选的逐关节虚拟搜索半窗（支持 None、1D 数组或 {joint_name: rad} 字典）。"""
    if virtual_half_windows is None:
        return None
    if isinstance(virtual_half_windows, Mapping):
        win = np.full(n_active, np.inf, dtype=float)
        unknown = set(virtual_half_windows) - set(active_names)
        if unknown:
            raise ValueError(
                f"virtual_half_windows contains unknown joints: {sorted(unknown)}"
            )
        for i, name in enumerate(active_names):
            if name in virtual_half_windows and virtual_half_windows[name] is not None:
                val = float(virtual_half_windows[name])
                if not math.isfinite(val) or val <= 0.0:
                    raise ValueError(
                        f"virtual_half_windows[{name}] must be positive and finite"
                    )
                win[i] = val
        return win
    win = finite_vector(
        virtual_half_windows, n_active, "explicit search half-windows"
    )
    if np.any(win <= 0.0):
        raise ValueError("search half-windows must be positive")
    return win


def _build_effective_bounds(lower, upper, q_ref, *, virtual_half_windows=None, active_names=None):
    """构建有效关节限位：显式配置 virtual_half_windows 时作为硬搜索窗口与真实限位求交。

    窗口只由 virtual_half_windows 显式决定，绝不按关节类型或参考角隐式生成。
    """
    lo = np.asarray(lower, dtype=float).copy()
    hi = np.asarray(upper, dtype=float).copy()
    if np.any(np.isnan(lo)) or np.any(np.isnan(hi)) or np.any(lo >= hi):
        raise ValueError("invalid actual/planning bounds")

    windows = _resolve_virtual_windows(
        virtual_half_windows, active_names or (), len(lo)
    )
    if windows is not None:
        finite_win = np.isfinite(windows)
        lo[finite_win] = np.maximum(
            lo[finite_win], q_ref[finite_win] - windows[finite_win]
        )
        hi[finite_win] = np.minimum(
            hi[finite_win], q_ref[finite_win] + windows[finite_win]
        )
        if np.any(lo >= hi):
            raise ValueError("search windows do not intersect actual bounds")

    return lo, hi



def _contiguous_range(indices):
    """indices 为连续递增整数序列时返回 (start, stop)，否则 None（用于把花式索引换成切片）。"""
    if indices is None:
        return None
    values = np.asarray(indices).ravel()
    if values.size == 0 or values.dtype.kind not in "iu":
        return None
    seq = values.tolist()
    start = seq[0]
    if start < 0 or seq != list(range(start, start + len(seq))):
        return None
    return start, start + len(seq)



def configure_model_metadata(
    model, pin_maps, urdf_xml, *, massless_links=(), position_bounds=None
):
    """启动时对运行时 URDF 执行一次完整审计；任何物理模型缺陷保持可见并阻断 strict 求解。"""
    root = ET.fromstring(urdf_xml)
    errors, warnings = [], []
    massless = set(massless_links)
    links = {e.get("name"): e for e in root.findall("link")}
    if massless - set(links):
        errors.append(
            f"unknown massless links: {sorted(massless - set(links))}"
        )
    source_joints = {e.get("name"): e for e in root.findall("joint")}
    overrides = dict(position_bounds or {})
    if set(overrides) - set(pin_maps["all_names"]):
        errors.append("position_bounds contains unknown/fixed joints")

    points = []
    for name, link in links.items():
        inertial = link.find("inertial")
        mass_el = None if inertial is None else inertial.find("mass")
        mass = 0.0 if mass_el is None else float(mass_el.get("value", "nan"))
        if not np.isfinite(mass) or mass < 0.0:
            errors.append(f"{name}: invalid mass")
            continue
        if mass == 0.0:
            if name not in massless:
                errors.append(
                    f"{name}: missing/zero mass; explicit massless frame confirmation required"
                )
            else:
                warnings.append(f"confirmed_massless_frame:{name}")
            continue
        if name in massless:
            errors.append(f"{name}: positive mass cannot be discarded")

        origin = inertial.find("origin")
        xyz = finite_vector(
            [0.0, 0.0, 0.0]
            if origin is None
            else [float(v) for v in origin.get("xyz", "0 0 0").split()],
            3,
            "local COM",
        )
        if origin is not None:
            finite_vector(
                [float(v) for v in origin.get("rpy", "0 0 0").split()],
                3,
                "inertial RPY",
            )
        tensor = inertial.find("inertia")
        if tensor is None:
            errors.append(f"{name}: missing inertia tensor")
            continue
        a = tensor.attrib
        I_mat = np.array(
            [
                [float(a["ixx"]), float(a["ixy"]), float(a["ixz"])],
                [float(a["ixy"]), float(a["iyy"]), float(a["iyz"])],
                [float(a["ixz"]), float(a["iyz"]), float(a["izz"])],
            ],
            dtype=float,
        )
        if not np.all(np.isfinite(I_mat)):
            errors.append(f"{name}: non-finite inertia")
            continue
        eig = np.linalg.eigvalsh(I_mat)
        if eig[0] < -1e-10 or eig[-1] > eig[:2].sum() + 1e-8:
            errors.append(f"{name}: nonphysical inertia")
        # 连杆帧按 BODY 类型查询：关节与连杆同名时，不带类型的查询在 Pinocchio 3.x 起会抛 ValueError
        if not model.existFrame(name, pin.FrameType.BODY):
            errors.append(f"{name}: model frame missing")
            continue
        points.append((model.getFrameId(name, pin.FrameType.BODY), mass, xyz))

    for name, s in pin_maps["all_specs"].items():
        e = source_joints.get(name)
        if e is None:
            errors.append(f"{name}: not found in URDF joints")
            continue
        kind = e.get("type")
        s["kind"] = kind
        if e.find("mimic") is not None:
            errors.append(f"{name}: mimic is not supported in this version")
        if kind not in ("continuous", "revolute", "prismatic"):
            errors.append(f"{name}: unsupported joint type {kind}")
        if s["continuous"] != (kind == "continuous"):
            errors.append(f"{name}: URDF/Pinocchio continuous type mismatch")
        axis = e.find("axis")
        v = finite_vector(
            [1.0, 0.0, 0.0]
            if axis is None
            else [float(x) for x in axis.get("xyz", "1 0 0").split()],
            3,
            "joint axis",
        )
        if np.linalg.norm(v) < 1e-12:
            errors.append(f"{name}: zero axis")
        if kind == "continuous":
            lo, hi = -np.inf, np.inf
        else:
            limit = e.find("limit")
            if (
                limit is None
                or "lower" not in limit.attrib
                or "upper" not in limit.attrib
            ):
                errors.append(f"{name}: missing position limits")
                lo, hi = -np.inf, np.inf
            else:
                lo, hi = float(limit.get("lower")), float(limit.get("upper"))
                if not np.isfinite(lo + hi) or lo >= hi:
                    errors.append(f"{name}: invalid position limits")
        if name in overrides:
            bound = finite_vector(overrides[name], 2, f"{name} override")
            if bound[0] >= bound[1] or bound[0] < lo or bound[1] > hi:
                errors.append(
                    f"{name}: override must narrow actual URDF limits"
                )
            else:
                lo, hi = bound
        s["lower"], s["upper"] = float(lo), float(hi)

    pin_maps["mass_points"] = points
    pin_maps["model_mass_kg"] = float(sum(p[1] for p in points))
    if pin_maps["model_mass_kg"] <= 0.0:
        errors.append("no positive model mass")
    pin_maps["model_audit_errors"] = errors
    pin_maps["model_audit_warnings"] = warnings
    pin_maps["active_lower"] = np.array(
        [pin_maps["all_specs"][n]["lower"] for n in pin_maps["active_names"]],
        dtype=float,
    )
    pin_maps["active_upper"] = np.array(
        [pin_maps["all_specs"][n]["upper"] for n in pin_maps["active_names"]],
        dtype=float,
    )
    pin_maps["active_periodic"] = np.array(
        [
            pin_maps["all_specs"][n]["continuous"]
            and not np.isfinite(pin_maps["all_specs"][n]["lower"])
            and not np.isfinite(pin_maps["all_specs"][n]["upper"])
            for n in pin_maps["active_names"]
        ],
        dtype=bool,
    )
    pin_maps["active_multi_turn"] = np.array(
        [
            pin_maps["all_specs"][n]["continuous"]
            or (
                pin_maps["all_specs"][n].get("kind") == "revolute"
                and np.isfinite(pin_maps["all_specs"][n]["lower"])
                and np.isfinite(pin_maps["all_specs"][n]["upper"])
                and (
                    pin_maps["all_specs"][n]["upper"]
                    - pin_maps["all_specs"][n]["lower"]
                )
                >= 2.0 * np.pi - 1e-6
            )
            for n in pin_maps["active_names"]
        ],
        dtype=bool,
    )
    return pin_maps


# =============================================================================
# 6. 物理场、重力矩计算与静态载荷上下文 (Physics, Gravity & Load Context)
# =============================================================================

def model_com_with_jacobian(model, data, q_full, *, active_v=None, derivatives=True):
    """整机质心（URDF 根坐标系）及其对活跃关节的雅可比，含固定在根上的质量。

    Pinocchio 的 centerOfMass / jacobianCenterOfMass 只累加 joint 1..n 的子树质量，
    不含 model.inertias[0]（URDF 根 link 以及经 fixed joint 挂在根上的 link）。
    这部分质量 m0 不随 q 运动，质心为 c0（根坐标系），因此：
        com   = (M_mov * c_mov + m0 * c0) / (M_mov + m0)
        J_com = M_mov / (M_mov + m0) * J_mov
    与 configure_model_metadata 统计的 model_mass_kg（全部 link 质量）保持一致。
    """
    nv = model.nv
    if derivatives:
        # nv = 1 时 Pinocchio 返回形状 (3,) 的数组，统一 reshape 为 (3, nv)
        J_mov = np.asarray(
            pin.jacobianCenterOfMass(model, data, q_full, False), dtype=float
        ).reshape(3, nv)
    else:
        pin.centerOfMass(model, data, q_full, False)
        J_mov = None

    m_mov = float(data.mass[0])
    c_mov = np.asarray(data.com[0], dtype=float)
    root_inertia = model.inertias[0]
    m_root = float(root_inertia.mass)
    m_total = m_mov + m_root
    if not math.isfinite(m_total) or m_total <= 0.0:
        raise ValueError("model has no positive mass for COM")
    com = (m_mov * c_mov + m_root * np.asarray(root_inertia.lever, dtype=float)) / m_total

    n_cols = len(active_v) if active_v is not None else nv
    if J_mov is None:
        return com, np.zeros((3, n_cols), dtype=float)
    J_cols = J_mov if active_v is None else J_mov[:, active_v]
    return com, (m_mov / m_total) * J_cols


def _reference_com(model, data, q_full):
    """质心参考值：与 _SecondaryObjective 代价相同的整机质心（含根 link 及固定在根上的质量）。"""
    com, _ = model_com_with_jacobian(model, data, q_full, derivatives=False)
    return np.array(com, dtype=float, copy=True)


def evaluate_pose_physics(pin_model, pin_data, pin_maps, q_active, q_ref=None):
    """计算给定关节构型下的重力矩、电机负载率、整机质心偏移与物理坐标回中偏差。"""
    q_active = finite_vector(
        q_active, len(pin_maps["active_names"]), "active q"
    )
    full = active_q_to_pinocchio_q(pin_model, pin_maps, q_active)
    active_v = pin_maps["active_v_indices"]

    tau_full = np.array(
        pin.computeGeneralizedGravity(pin_model, pin_data, full), copy=True
    )
    tau = tau_full[active_v]
    effort = pin_maps["effort_limits"]
    ratio = np.full(tau.shape, np.nan, dtype=float)
    valid = np.isfinite(effort) & (effort > 0.0)
    ratio[valid] = 100.0 * np.abs(tau[valid]) / effort[valid]

    com, _ = model_com_with_jacobian(
        pin_model, pin_data, full, active_v=active_v, derivatives=False
    )
    mass = float(
        pin_maps.get(
            "model_mass_kg",
            float(pin_data.mass[0]) + float(pin_model.inertias[0].mass),
        )
    )

    g = np.asarray(pin_model.gravity.linear, dtype=float)
    ghat = g / max(float(np.linalg.norm(g)), 1e-30)
    horizontal = (np.eye(3) - np.outer(ghat, ghat)) @ com

    # 姿态回中漂移按真实关节坐标差评估（仅对无界 continuous 关节做周期折叠）
    diff = (
        np.zeros(len(q_active), dtype=float)
        if q_ref is None
        else joint_difference(
            q_active,
            q_ref,
            active_periodic_mask(pin_maps),
        )
    )

    return {
        "tau_g_nm": tau,
        "tau_norm_nm": float(np.linalg.norm(tau)),
        "motor_load_pct": ratio,
        "max_motor_load_pct": (
            float(np.max(ratio)) if np.all(valid) else np.nan
        ),
        "effort_basis": pin_maps.get(
            "effort_basis",
            "URDF reference, not confirmed continuous rating",
        ),
        "com_xyz_m": com,
        "model_mass_kg": mass,
        "com_xy_offset_mm": float(np.linalg.norm(horizontal) * 1000.0),
        "overturning_moment_nm": float(np.linalg.norm(np.cross(com, mass * g))),
        "moment_definition": "static gravity moment about model root, NOT support-edge safety margin",
        "posture_drift_deg": float(np.rad2deg(np.linalg.norm(diff))),
        "full_state_available": "q_template" in pin_maps,
        "model_audit_errors": pin_maps.get(
            "model_audit_errors", ["not audited"]
        ),
    }


def prepare_load_context(model, pin_maps, config, snapshot):
    """构建严格、不可变的全身静态负荷上下文；绝不虚构关节状态或默认力矩上限。"""
    if pin_maps.get("model_audit_errors"):
        raise ValueError(
            "model audit: " + "; ".join(pin_maps["model_audit_errors"])
        )
    if not (isinstance(snapshot, JointStateSnapshot) or type(snapshot).__name__ == "JointStateSnapshot"):
        raise ValueError("fresh complete JointStateSnapshot is required")

    max_age = float(config.get("load_state_max_age_s", 1.0))
    if not np.isfinite(max_age) or max_age <= 0.0:
        raise ValueError("invalid load_state_max_age_s")
    require_source = bool(config.get("load_require_source_age", False))
    error = snapshot.freshness_error(max_age, require_source_age=require_source)
    if error:
        raise ValueError(error)

    missing = set(pin_maps["all_names"]) - set(snapshot.positions)
    if missing:
        raise ValueError(
            f"missing whole-model joint state: {sorted(missing)}"
        )

    policy = LoadPolicy(
        allowable_efforts=config.get("load_allowable_efforts", {}),
        guard_joint_names=config.get("load_guard_joint_names"),
        joint_weights=config.get("load_joint_weights", {}),
        absolute_weight=float(
            config.get(
                "load_absolute_gain",
                config.get("dls_gravity_torque_gain")
                if config.get("dls_gravity_torque_gain") is not None
                else GRAVITY_TORQUE_GAIN,
            )
        ),
        excess_weight=float(config.get("load_excess_gain", 0.50)),
        relative_weight=float(config.get("load_relative_gain", 0.0)),
        soft_ratio=float(config.get("load_soft_ratio", 0.70)),
        # v7：未配置时取 min(LOAD_EARLY_STOP_RATIO=0.70, soft_ratio)（v6 默认 0.60）
        early_stop_ratio=(
            None
            if config.get("load_early_stop_ratio") is None
            else float(config["load_early_stop_ratio"])
        ),
        peak_tie=float(config.get("load_peak_tie", 0.02)),
    )
    guards = list(policy.guard_joint_names or pin_maps["all_names"])
    if set(guards) - set(pin_maps["all_names"]):
        raise ValueError("unknown guarded joint")
    if set(pin_maps["active_names"]) - set(guards):
        raise ValueError("all IK-active joints must be guarded")
    policy.vectors(guards)

    q = pin.neutral(model)
    for name in pin_maps["all_names"]:
        s = pin_maps["all_specs"][name]
        value = snapshot.positions[name]
        if value < s["lower"] or value > s["upper"]:
            raise ValueError(
                f"observed {name} outside configured bounds; not clipped"
            )
        if s["continuous"]:
            q[s["idx_q"] : s["idx_q"] + 2] = (
                math.cos(value),
                math.sin(value),
            )
        else:
            q[s["idx_q"]] = value

    return {
        "policy": policy,
        "snapshot": snapshot,
        "max_age_s": max_age,
        "require_source_age": require_source,
        "q_template": q,
        "guard_names": guards,
        "guard_v": np.array(
            [pin_maps["all_specs"][n]["idx_v"] for n in guards], dtype=np.intp
        ),
        "active_v": np.array(pin_maps["active_v_indices"], dtype=np.intp),
        "periodic": pin_maps["active_periodic"].copy(),
        "model_mass_kg": pin_maps["model_mass_kg"],
        "unchecked_joints": sorted(set(pin_maps["all_names"]) - set(guards)),
    }


def load_physics(model, data, pin_maps, q_active, context, derivatives=False):
    """计算守护关节的静态重力矩、导数、归一化负载率与整机质心。"""
    q_full = active_q_to_pinocchio_q(model, pin_maps, q_active)
    tau_full = np.array(
        pin.computeGeneralizedGravity(model, data, q_full), copy=True
    )
    tau = tau_full[context["guard_v"]]
    derivative = np.zeros(
        (len(tau), len(context["active_v"])), dtype=float
    )
    if derivatives:
        full_deriv = np.asarray(
            pin.computeGeneralizedGravityDerivatives(model, data, q_full)
        ).reshape(model.nv, model.nv)
        derivative = full_deriv[
            np.ix_(context["guard_v"], context["active_v"])
        ].copy()

    com, Jcom = model_com_with_jacobian(
        model, data, q_full, active_v=context["active_v"], derivatives=derivatives
    )
    if not np.all(np.isfinite(com)) or not np.all(np.isfinite(Jcom)):
        raise ValueError("non-finite model COM")
    result = context["policy"].evaluate(tau, context["guard_names"])
    result.update(
        tau=tau,
        d_tau_d_active=derivative,
        com=com,
        Jcom=Jcom,
        q_full=q_full,
    )
    return result

# =============================================================================
# Torque cost helpers (Pinocchio)
# =============================================================================
def gravity_torque_for_active_q(
    pin_model,
    pin_data,
    pin_maps,
    active_joint_names,
    q_active,
):
    """Compute gravity compensation torques for active joints using Pinocchio."""
    q_active = np.asarray(q_active, dtype=float)
    q_full = active_q_to_pinocchio_q(pin_model, pin_maps, q_active)

    tau_gravity_full = pin.computeGeneralizedGravity(
        pin_model, pin_data, q_full
    )
    tau_active = np.array(
        [float(tau_gravity_full[idx]) for idx in pin_maps["active_v_indices"]],
        dtype=float,
    )

    return tau_active


def get_active_joint_efforts(pin_maps):
    """Return effort limits for active joints from Pinocchio maps."""
    return pin_maps["effort_limits"].copy()


def add_torque_metrics_to_candidate(
    candidate,
    pin_model,
    pin_data,
    pin_maps,
    active_joint_names,
    joint_efforts_active,
):
    if candidate is None or "q" not in candidate:
        return

    guard_efforts = candidate.get("load_guard_efforts")
    q_arr = np.asarray(candidate["q"], dtype=float)
    if (
        isinstance(guard_efforts, dict)
        and candidate.get("_load_checked_q_bytes") == q_arr.tobytes()
        and all(n in guard_efforts for n in active_joint_names)
    ):
        # strict 模式下 _check_load_candidate 已在同一 q_template 下算过全身重力矩，直接复用（结果逐位相同）
        tau = np.array([float(guard_efforts[n]) for n in active_joint_names], dtype=float)
    else:
        tau = gravity_torque_for_active_q(
            pin_model=pin_model,
            pin_data=pin_data,
            pin_maps=pin_maps,
            active_joint_names=active_joint_names,
            q_active=q_arr,
        )

    if not np.all(np.isfinite(joint_efforts_active)) or np.any(
        joint_efforts_active <= 0.0
    ):
        raise ValueError(
            "Unknown effort limits; no 1.0 fallback or fake utilization"
        )
    ratio = np.abs(tau) / joint_efforts_active
    max_idx = int(np.argmax(ratio))

    candidate["tau_gravity"] = tau
    candidate["torque_ratio"] = ratio
    candidate["torque_cost"] = float(np.mean(ratio ** 2))
    candidate["torque_l2"] = float(np.linalg.norm(tau))
    candidate["max_abs_tau"] = float(np.max(np.abs(tau)))
    candidate["max_usage"] = float(np.max(ratio))
    candidate["max_usage_joint_idx"] = max_idx
    candidate["max_usage_joint_name"] = active_joint_names[max_idx]

    update_candidate_final_cost_with_torque(candidate)


def add_torque_metrics_to_candidates(
    candidates,
    pin_model,
    pin_data,
    pin_maps,
    active_joint_names,
    joint_efforts_active,
):
    for c in candidates:
        if c is None:
            continue
        try:
            add_torque_metrics_to_candidate(
                candidate=c,
                pin_model=pin_model,
                pin_data=pin_data,
                pin_maps=pin_maps,
                active_joint_names=active_joint_names,
                joint_efforts_active=joint_efforts_active,
            )
        except Exception as e:
            c["torque_error"] = repr(e)
            update_candidate_final_cost_with_torque(c)


# =============================================================================
# 7. 任务空间种子库与腰部舒适窗推导 (Seed Management & Comfort Bounds)
# =============================================================================

def _comfort_edge(ok, limit, step=1e-3):
    """从 δ = 0（满足）沿正方向找 ok(δ) 首次不满足的边界，至多到 limit；返回 (边界, 是否被判据截断)。"""
    grid = np.arange(step, limit + 0.5 * step, step)
    bad = np.nonzero(~ok(grid))[0]
    if bad.size == 0:
        return float(limit), False
    hi = float(grid[bad[0]])
    lo = float(grid[bad[0] - 1]) if bad[0] > 0 else 0.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if ok(np.array([mid]))[0]:
            lo = mid
        else:
            hi = mid
    return lo, True


def derive_waist_comfort_bounds(
    model,
    data,
    pin_maps,
    q_ready,
    lower,
    upper,
    *,
    allowable_efforts=None,
    soft_ratio=None,
    gravity_free_half=None,
    max_half=None,
):
    """由静态物理量推导腰部舒适窗（v8）。返回 (comfort_lower, comfort_upper, info)。

    舒适窗是偏好而不是硬约束：solver.py 先只在窗内搜索，窗内没有满足全部硬约束的解时
    再放开到真实限位（∩ 显式 virtual_half_windows）重试，因此不会减少可解目标。

    推导依据（第一性原理）：腰部承载整个上半身，偏转的代价不由关节角本身决定，而由
      (1) 静态重力矩：其它关节固定时，转动腰关节 j 只让它的子树绕轴刚性转动，所以该关节重力矩
          是精确正弦 τ(θ) = C·sin(θ - φ)。C = 子树质量 × g × 质心到轴的垂距 × sin(轴与重力夹角)，
          即“上半身放平”时的最大重力矩；φ 为子树质心正好位于轴上方的平衡角。
          τ 同时就是上半身质心相对关节轴的水平力臂 × 重量，所以它既度量电机负荷，也度量质心偏移。
      (2) 躯干朝向：轴与重力夹角 < 30° 的腰关节（偏航类）几乎不产生重力矩，但转过 90° 后胸前
          （另一只手、头部相机）背离前方工作区，这种目标应由底盘转向而不是腰部完成。
    窗口定义（s = soft_ratio，默认 0.7，与 LoadPolicy 的 soft_ratio 同义：“只用 70% 的能力”）：
      重力类    包含 q_ready 的连通区间内同时满足
                  负荷  |τ(θ)| <= max(s × min(C, A), |τ(q_ready)|)，A = 容许力矩
                        （allowable_efforts，没有则用 URDF effort，再没有则不限）；
                  质心  |τ(θ) - τ(q_ready)| <= s × C，即上半身质心相对 q_ready 的水平位移
                        不超过 s × 质心到轴的垂距（支撑面按 q_ready 设计，不让质心大幅移出）。
                直立且前后平衡的上半身两条都给出 φ ± asin(s) = φ ± 44.4°；前重的上半身向前更窄、
                向后由质心条件限住；A < C 时负荷条件更窄（该关节负载率 <= s）。
      重力无关类 q_ready ± gravity_free_half（默认 WAIST_YAW_VIRTUAL_LIM = 90°）。
    两类都再与 q_ready ± max_half（默认 90°）及 [lower, upper] 求交；prismatic 腰关节不设窗。
    计算构型：活跃关节取 q_ready，其它关节取 pin_maps["q_template"]（没有则取 URDF 零位）。
    非腰部关节的窗口为 (-inf, inf)。窗口退化（宽度 < 1e-9）或重力矩不满足正弦律（kind="gravity_check_failed"）
    的关节不设窗并在 info 中注明。
    """
    names = pin_maps["active_names"]
    n = len(names)
    q0 = finite_vector(q_ready, n, "q_ready")
    lo_b = np.asarray(lower, dtype=float)
    hi_b = np.asarray(upper, dtype=float)
    soft = WAIST_COMFORT_SOFT_RATIO if soft_ratio is None else float(soft_ratio)
    free_half = WAIST_YAW_VIRTUAL_LIM if gravity_free_half is None else float(gravity_free_half)
    cap_half = WAIST_COMFORT_MAX_HALF if max_half is None else float(max_half)
    if not (0.0 < soft <= 1.0) or not (free_half > 0.0) or not (cap_half > 0.0):
        raise ValueError("invalid waist comfort parameters")
    allowable = dict(allowable_efforts or {})

    comfort_lo = np.full(n, -np.inf, dtype=float)
    comfort_hi = np.full(n, np.inf, dtype=float)
    info = {}
    g = np.asarray(model.gravity.linear, dtype=float)
    g_norm = float(np.linalg.norm(g))
    ghat = g / g_norm if g_norm > 1e-12 else np.zeros(3)

    q_full = active_q_to_pinocchio_q(model, pin_maps, q0)
    J_world = np.array(pin.computeJointJacobians(model, data, q_full), copy=True)

    def tau_at(i, theta):
        q = q0.copy()
        q[i] = theta
        full = active_q_to_pinocchio_q(model, pin_maps, q)
        iv = pin_maps["active_v_indices"][i]
        return float(pin.computeGeneralizedGravity(model, data, full)[iv])

    for i, name in enumerate(names):
        if name not in WAIST_JOINT_NAMES:
            continue
        spec = pin_maps["all_specs"][name]
        theta0 = float(q0[i])
        if spec.get("kind") == "prismatic":
            info[name] = {"kind": "prismatic_no_window"}
            continue
        axis = J_world[3:6, spec["idx_v"]]
        axis_norm = float(np.linalg.norm(axis))
        cos_g = abs(float(axis @ ghat)) / axis_norm if axis_norm > 1e-12 else 1.0
        entry = {"axis_gravity_cos": cos_g}

        if g_norm <= 1e-12 or cos_g >= WAIST_GRAVITY_FREE_COS:
            half = min(free_half, cap_half)
            lo, hi = theta0 - half, theta0 + half
            entry.update(kind="gravity_free", half_window=float(half))
        else:
            tau0 = tau_at(i, theta0)
            tau90 = tau_at(i, theta0 + 0.5 * np.pi)
            c_max = math.hypot(tau0, tau90)
            x0 = math.atan2(tau0, tau90)           # τ(θ0 + δ) = C·sin(x0 + δ)
            check = tau_at(i, theta0 + 0.25 * np.pi)
            if abs(check - c_max * math.sin(x0 + 0.25 * np.pi)) > 1e-6 * (1.0 + c_max):
                # 刚体子树绕轴转动时不会发生；万一模型不满足该假设，此关节不设舒适窗（= v7 行为），不中断初始化
                entry.update(kind="gravity_check_failed", lower=-np.inf, upper=np.inf)
                info[name] = entry
                continue
            raw = allowable.get(name, spec.get("raw_effort", np.nan))
            a_cap = float(raw) if raw is not None and np.isfinite(raw) and raw > 0.0 else np.inf
            cap = min(c_max, a_cap)
            t_load = max(soft * cap, abs(tau0))
            t_com = soft * c_max
            if a_cap < c_max:
                basis = "allowable_effort" if name in allowable else "urdf_effort"
            else:
                basis = "max_gravity_torque"
            # 平衡角：|sin x| 在 x = 0 或 ±π 处为零，取离 q_ready（x0）最近的那一个
            xc = 0.0 if abs(x0) <= 0.5 * np.pi else math.copysign(np.pi, x0)

            def ok_dir(sign):
                def ok(delta):
                    tau = c_max * np.sin(x0 + sign * delta)
                    return (np.abs(tau) <= t_load * (1.0 + 1e-12)) & (
                        np.abs(tau - tau0) <= t_com * (1.0 + 1e-12)
                    )
                return ok

            if c_max <= 1e-12:
                d_hi = d_lo = cap_half
                cut_hi = cut_lo = False
            else:
                d_hi, cut_hi = _comfort_edge(ok_dir(1.0), cap_half)
                d_lo, cut_lo = _comfort_edge(ok_dir(-1.0), cap_half)
            lo, hi = theta0 - d_lo, theta0 + d_hi
            entry.update(
                kind="gravity",
                tau_ready_nm=tau0,
                tau_max_nm=c_max,
                balance_angle=theta0 + (xc - x0),
                cap_nm=cap,
                cap_basis=basis,
                load_threshold_nm=t_load,
                com_threshold_nm=t_com,
                limited_by_gravity={"lower": bool(cut_lo), "upper": bool(cut_hi)},
            )
        lo = max(lo, theta0 - cap_half, float(lo_b[i]))
        hi = min(hi, theta0 + cap_half, float(hi_b[i]))
        if not (hi - lo >= 1e-9 and lo <= theta0 <= hi):
            entry.update(degenerate=True, lower=-np.inf, upper=np.inf)
            info[name] = entry
            continue
        comfort_lo[i], comfort_hi[i] = lo, hi
        entry.update(lower=float(lo), upper=float(hi))
        info[name] = entry
    return comfort_lo, comfort_hi, info



def make_scatter_seeds(q_prev, q_ready, lower, upper, rng, *, max_count=None):
    """生成种子序列：q_prev（就近、连续性最好）-> q_ready（标准姿态）-> 两者附近的扰动 -> 限位中点。

    q_prev 放在第一位：从当前构型出发的 DLS 天然收敛到最近的解，最不容易“绕大圈”；
    q_ready 作为第二个基准种子提供“姿态合理”的另一族解。扰动按 (轮次 -> 尺度由大到小 ->
    q_prev / q_ready) 交错排列，保证前 max_seeds 个种子足够分散。对无界 continuous 关节不产生 NaN 中点。

    max_count（v7）：调用方最多只用前 max_count 个种子时传入。仍生成全部原始种子（随机数消耗与
    顺序不变），但只对足以凑够 max_count 个不重复种子的前缀去重（unique_seed_list 是保序、
    保留首次出现的逐个去重，前缀去重结果 == 全量去重结果的前 max_count 个）；None 时与 v6 相同。
    """
    seeds = [
        clip_to_bounds(q_prev, lower, upper),
        clip_to_bounds(q_ready, lower, upper),
    ]
    for _ in range(max(1, N_NOISE_PER_BASE // len(NOISE_SCALES))):
        for scale in NOISE_SCALES:
            for base in (q_prev, q_ready):
                seeds.append(
                    clip_to_bounds(
                        base + rng.normal(scale=scale, size=len(base)),
                        lower,
                        upper,
                    )
                )
    midpoint = np.asarray(q_ready, dtype=float).copy()
    finite = np.isfinite(lower) & np.isfinite(upper)
    midpoint[finite] = 0.5 * (lower[finite] + upper[finite])
    seeds.append(midpoint)
    if max_count is None:
        return unique_seed_list(seeds)

    want = int(max_count)
    if want < 1:
        raise ValueError("max_count must be >= 1")
    n = min(len(seeds), want)
    while True:
        result = unique_seed_list(seeds[:n])
        if len(result) >= want or n >= len(seeds):
            return result[:want]
        # 前缀里有重复：按缺口扩大前缀（至多扩到全部种子）
        n = min(len(seeds), n + (want - len(result)))


def chain_base_joint(pin_model, pin_maps):
    """运动链基座：最靠近根的活动关节的父关节（Pinocchio 关节编号按拓扑序，父编号小于子编号）。"""
    ids = [int(pin_maps["name_to_joint_id"][n]) for n in pin_maps["active_names"]]
    return int(pin_model.parents[min(ids)])


def build_seed_library(
    pin_model,
    pin_data,
    pin_maps,
    ee_frame_id,
    lower,
    upper,
    q_ready,
    size=SEED_LIBRARY_SIZE,
    *,
    comfort_bounds=None,
    rng_seed=SEED_LIBRARY_RNG_SEED,
):
    """v13 种子库：在有效限位内均匀采样关节角，记录末端相对运动链基座的位姿（None = 关闭）。

    位姿取相对 chain_base_joint 坐标系的值：基座之前的非活动关节（strict 模式由实测快照决定）
    不影响库内容，求解时把目标换到同一坐标系即可比较。无界 continuous 关节在 q_ready ± π 内采样。
    comfort_bounds=(lower, upper) 时一半样本落在舒适窗内（舒适阶段只能用窗内样本）。
    种子库只决定“从哪里开始迭代”，不影响任何约束与择优；库位姿不准只会让种子变差。
    """
    size = int(size or 0)
    if size <= 0:
        return None
    q_ready = np.asarray(q_ready, dtype=float)
    lo = np.where(np.isfinite(lower), lower, q_ready - np.pi)
    hi = np.where(np.isfinite(upper), upper, q_ready + np.pi)
    if not np.all(lo <= hi):
        raise ValueError("seed library: empty bounds")
    rng = np.random.default_rng(rng_seed)
    parts = []
    if comfort_bounds is not None:
        c_lo = np.maximum(lo, np.asarray(comfort_bounds[0], dtype=float))
        c_hi = np.minimum(hi, np.asarray(comfort_bounds[1], dtype=float))
        if np.all(c_lo < c_hi):
            parts.append(rng.uniform(c_lo, c_hi, size=(size // 2, len(lo))))
    parts.append(rng.uniform(lo, hi, size=(size - sum(len(x) for x in parts), len(lo))))
    Q = np.vstack(parts)
    build = _make_q_builder(pin_model, pin_maps)
    base = chain_base_joint(pin_model, pin_maps)
    P = np.empty((size, 3), dtype=float)
    Rm = np.empty((size, 3, 3), dtype=float)
    for i in range(size):
        pin.forwardKinematics(pin_model, pin_data, build(Q[i]))
        M = pin_data.oMi[base].actInv(
            pin.updateFramePlacement(pin_model, pin_data, ee_frame_id)
        )
        P[i] = M.translation
        Rm[i] = M.rotation
    q_min = Q.min(axis=0)
    q_max = Q.max(axis=0)
    QT = np.ascontiguousarray(Q.T)
    for arr in (Q, P, Rm, q_min, q_max, QT):
        arr.setflags(write=False)   # 同一模型的多个求解器实例共享同一份库（见 solver._shared_seed_library）
    return {
        "q": Q, "p": P, "R": Rm, "base_joint": base, "size": size,
        "_q_min": q_min, "_q_max": q_max, "_qt": QT,
    }


def seed_library_target(pin_model, pin_data, pin_maps, library, q_active, T_target):
    """把目标位姿换到种子库的基座坐标系（基座位姿按当前 q 模板计算，与活动关节取值无关）。"""
    pin.forwardKinematics(
        pin_model, pin_data, active_q_to_pinocchio_q(pin_model, pin_maps, q_active)
    )
    base = pin_data.oMi[library["base_joint"]]
    target = np.asarray(T_target, dtype=float)
    return np.array(
        base.actInv(pin.SE3(target[:3, :3], target[:3, 3])).homogeneous, copy=True
    )


def select_library_seeds(
    library,
    T_target_base,
    orientation_mode,
    tool_axis,
    cone,
    characteristic_length,
    lower,
    upper,
    count,
    *,
    exclude=(),
    min_separation=SEED_LIBRARY_MIN_SEPARATION,
    pool=SEED_LIBRARY_POOL,
):
    """按任务空间距离选种子：sqrt((位置差 / 特征长度)^2 + 姿态差^2)，姿态差口径与 DLS 主任务相同
    （full：转角；tool_dir：工具轴夹角；tool_cone：夹角超出锥角的部分）。

    只用完全落在 [lower, upper] 内的样本（舒适窗 / 跳变框同样适用，不做截断）；入选种子之间、
    与 exclude（本阶段已排队的种子）之间的关节差 ∞-范数 >= min_separation，避免几个种子收敛到同一处。
    """
    count = int(count)
    if library is None or count <= 0:
        return []
    Q = library["q"]
    lo_b = np.asarray(lower, dtype=float)
    hi_b = np.asarray(upper, dtype=float)
    q_min = library.get("_q_min")
    q_max = library.get("_q_max")
    QT = library.get("_qt")
    idx = None
    if q_min is not None and q_max is not None and QT is not None:
        if np.isnan(lo_b).any() or np.isnan(hi_b).any() or (lo_b > hi_b).any():
            return []
        if (lo_b > q_max).any() or (hi_b < q_min).any():
            return []
        tight = np.flatnonzero((lo_b > q_min) | (hi_b < q_max))
        if tight.size == 0:
            n_in = len(Q)
            P_sub = library["p"]
            Rm = library["R"]
        else:
            j0 = int(tight[0])
            inside = (QT[j0] >= lo_b[j0]) & (QT[j0] <= hi_b[j0])
            for j in tight[1:]:
                jj = int(j)
                inside &= (QT[jj] >= lo_b[jj]) & (QT[jj] <= hi_b[jj])
            idx = np.flatnonzero(inside)
            n_in = idx.size
            if n_in == 0:
                return []
            P_sub = library["p"] if n_in == len(Q) else library["p"][idx]
            Rm = library["R"] if n_in == len(Q) else library["R"][idx]
            if n_in == len(Q):
                idx = None
    else:
        inside = np.all((Q >= lo_b) & (Q <= hi_b), axis=1)
        idx = np.flatnonzero(inside)
        n_in = idx.size
        if n_in == 0:
            return []
        P_sub = library["p"][idx]
        Rm = library["R"][idx]
    T = np.asarray(T_target_base, dtype=float)
    dist = np.linalg.norm(P_sub - T[:3, 3], axis=1) / float(
        characteristic_length
    )
    if orientation_mode == "full":
        cosine = 0.5 * (np.einsum("nij,ij->n", Rm, T[:3, :3]) - 1.0)
        angle = np.arccos(np.clip(cosine, -1.0, 1.0))
    else:
        axis = np.asarray(TOOL_AXIS_X if tool_axis is None else tool_axis, dtype=float)
        axis = axis / np.linalg.norm(axis)
        angle = np.arccos(np.clip((Rm @ axis) @ (T[:3, :3] @ axis), -1.0, 1.0))
        if orientation_mode == "tool_cone":
            angle = np.maximum(0.0, angle - float(cone))
    score = np.hypot(dist, angle)
    k = min(int(pool), n_in)
    near = np.argpartition(score, k - 1)[:k] if k < n_in else np.arange(n_in)
    near = near[np.argsort(score[near], kind="stable")]
    chosen = []
    taken = [np.asarray(e, dtype=float) for e in exclude]
    for j in near:
        q = Q[j] if idx is None else Q[idx[j]]
        if all(float(np.max(np.abs(q - e))) >= min_separation for e in taken):
            chosen.append(q.copy())
            taken.append(q)
            if len(chosen) >= count:
                break
    return chosen


# =============================================================================
# 8. 数值 DLS/SNS 单种子迭代核心 (Numerical DLS/SNS Iteration Core)
# =============================================================================

class _SecondaryObjective:
    """零空间次任务 H(q) 与负梯度 z = -dH/dq 的求值器：每个种子构造一次，迭代内复用。

    与 v6 的 _compute_nullspace_gradient_and_cost 逐项同序运算（v14 起该 v6 形式入口只保留在
    tests/dls_compat.py 供数值测试使用）。提速只来自：
      - 每次调用都相同的量提前算好：周期掩码、限位缓冲边界、质心水平投影 P、参考质心、
        容许力矩/权重向量、雅可比索引、根 link 质量项；
      - 直接复用主任务已算好的 Pinocchio q（单种子循环内两者来自同一组关节角，数值完全相同）；
      - need_z=False（只比较代价的试探点）时不计算梯度与重力矩导数。
    """

    def __init__(
        self,
        pin_model,
        pin_data,
        pin_maps,
        q_ref,
        lower,
        upper,
        ns_gain,
        active_v_indices,
        effort_limits,
        movable_mask,
        include_waist,
        tau_g_ref,
        com_xy_ref,
        gravity_gain,
        com_gain,
        limit_gain,
        load_context=None,
        relative_gravity_gain=0.0,
        posture_use_equivalent_branch=False,
    ):
        self.model = pin_model
        self.data = pin_data
        self.nv = pin_model.nv
        self.q_ref = q_ref
        self.lower = lower
        self.upper = upper
        self.ns_gain = ns_gain
        self.neg_ns_gain = -ns_gain
        self.movable = movable_mask

        # 姿态回中：默认真实编码器坐标差（仅无界 continuous 关节做周期折叠）
        self.multi_turn = None
        if (
            posture_use_equivalent_branch
            and pin_maps is not None
            and "active_multi_turn" in pin_maps
        ):
            self.multi_turn = pin_maps["active_multi_turn"]
        else:
            periodic = (
                (
                    active_periodic_mask(pin_maps)
                    if pin_maps is not None
                    else np.zeros(len(q_ref), dtype=bool)
                )
                if load_context is None
                else load_context["periodic"]
            )
            self.periodic = np.asarray(periodic, dtype=bool)
            self.periodic_any = bool(np.any(self.periodic))

        # 单侧安全裕度排斥项的缓冲边界
        self.margin = max(float(SAFE_MARGIN), 0.02)
        self.limit_gain = limit_gain
        if limit_gain > 0.0:
            self.b_low = np.minimum(q_ref, lower + self.margin)
            self.b_high = np.maximum(q_ref, upper - self.margin)

        # 静态重力矩与质心
        self.context = load_context
        self.com_gain = com_gain
        self.use_com = bool(include_waist and com_gain > 0.0 and com_xy_ref is not None)
        if load_context is not None:
            self.policy = load_context["policy"]
            self.guard_v = load_context["guard_v"]
            self.caps, self.weights = self.policy.vectors(load_context["guard_names"])
            self.com_cols = load_context["active_v"]
            self.dtau_index = np.ix_(load_context["guard_v"], load_context["active_v"])
            self.tau_ref = None
            if self.policy.relative_weight > 0.0:
                self.tau_ref = finite_vector(
                    tau_g_ref,
                    len(load_context["guard_names"]),
                    "reference gravity efforts",
                )
        else:
            valid_eff = (
                effort_limits is not None
                and np.all(np.isfinite(effort_limits))
                and np.all(effort_limits > 0.0)
            )
            self.gravity_gain = gravity_gain
            self.rel_gain = relative_gravity_gain
            self.use_gravity = bool(
                valid_eff and (gravity_gain > 0.0 or relative_gravity_gain > 0.0)
            )
            self.use_rel = relative_gravity_gain > 0.0 and tau_g_ref is not None
            self.tau_ref = tau_g_ref
            self.com_cols = active_v_indices
            if self.use_gravity:
                self.active_v = active_v_indices
                self.inv_eff_sq = 1.0 / (effort_limits ** 2)
                self.n_act = max(1, len(active_v_indices))
                self.dtau_index = np.ix_(active_v_indices, active_v_indices)
        if load_context is not None or self.use_com:
            root_inertia = pin_model.inertias[0]
            self.m_root = float(root_inertia.mass)
            self.root_term = self.m_root * np.asarray(root_inertia.lever, dtype=float)
        if self.use_com:
            self.ref_com = finite_vector(com_xy_ref, 3, "reference COM")
            ghat = np.asarray(pin_model.gravity.linear, dtype=float)
            ghat = ghat / max(float(np.linalg.norm(ghat)), 1e-30)
            self.P = np.eye(3) - np.outer(ghat, ghat)

        # v15 热路径预计算（只改写法，不改任何参与结果的浮点运算）
        self.movable_f = np.asarray(movable_mask, dtype=float).copy()
        self.movable_all = bool(np.all(np.asarray(movable_mask, dtype=bool)))
        if limit_gain > 0.0:
            self.b_low_list = np.asarray(self.b_low, dtype=float).tolist()
            self.b_high_list = np.asarray(self.b_high, dtype=float).tolist()
        col_range = _contiguous_range(self.com_cols)
        self.com_col_slice = None if col_range is None else slice(*col_range)
        self.dtau_slices = None
        dtau_rows = dtau_cols = None
        if load_context is not None:
            dtau_rows, dtau_cols = load_context["guard_v"], load_context["active_v"]
            policy = self.policy
            self.n_guard = len(load_context["guard_v"])   # == len(tau)，与 _cost_terms 的 n 相同
            self.soft_ratio = float(policy.soft_ratio)
            self.half_aw = 0.5 * policy.absolute_weight
            self.aw_w = policy.absolute_weight * self.weights
            # 快路径（所有关节负荷率 <= soft_ratio 时 excess 恒为 +0.0）：见 __call__ 注释
            self.fast_load_cost = bool(
                policy.relative_weight == 0.0 and policy.absolute_weight > 0.0
            )
        elif getattr(self, "use_gravity", False):
            dtau_rows = dtau_cols = active_v_indices
        if dtau_rows is not None:
            r_rng, c_rng = _contiguous_range(dtau_rows), _contiguous_range(dtau_cols)
            if r_rng is not None and c_rng is not None:
                self.dtau_slices = (slice(*r_rng), slice(*c_rng))

    def _dtau(self, raw):
        """重力矩导数子块（与 reshape(nv, nv)[np.ix_(rows, cols)] 同值、同为行主序连续）。"""
        full = np.asarray(raw).reshape(self.nv, self.nv)
        if self.dtau_slices is None:
            return full[self.dtau_index]
        return np.ascontiguousarray(full[self.dtau_slices])

    def _com(self, q_full, derivatives):
        """model_com_with_jacobian 的同序运算（根 link 质量项已预计算）。导数关闭时雅可比返回 None。"""
        model, data = self.model, self.data
        if derivatives:
            J_mov = np.asarray(
                pin.jacobianCenterOfMass(model, data, q_full, False), dtype=float
            ).reshape(3, self.nv)
        else:
            pin.centerOfMass(model, data, q_full, False)
            J_mov = None
        m_mov = float(data.mass[0])
        c_mov = np.asarray(data.com[0], dtype=float)
        m_total = m_mov + self.m_root
        if not math.isfinite(m_total) or m_total <= 0.0:
            raise ValueError("model has no positive mass for COM")
        com = (m_mov * c_mov + self.root_term) / m_total
        if J_mov is None:
            return com, None
        if self.com_cols is None:
            J_cols = J_mov
        elif self.com_col_slice is not None:
            # 与 J_mov[:, cols]（花式索引结果为列主序）同值同布局
            J_cols = np.asfortranarray(J_mov[:, self.com_col_slice])
        else:
            J_cols = J_mov[:, self.com_cols]
        return com, (m_mov / m_total) * J_cols

    def _limit_terms(self, q_active, need_z):
        """单侧安全裕度排斥项：返回 (z 增量或 None, c_lim)。

        与原 numpy 写法逐位一致：under = max(0, b_low - q) / margin，over = max(0, q - b_high) / margin，
        z 增量 = limit_gain * (under - over) / margin，c_lim = 0.5 * limit_gain * dot(under^2 + over^2, movable)。
        q 全部处于缓冲区外（常见情形）时 under = over = +0.0：z 增量恰为 +0.0（返回标量 0.0，
        与加零向量逐位相同），c_lim = 0.0。含 NaN 时回到 numpy 原写法。
        """
        margin, gain = self.margin, self.limit_gain
        q_list = q_active.tolist()
        active = False
        for qi, lo_i, hi_i in zip(q_list, self.b_low_list, self.b_high_list):
            if qi < lo_i or qi > hi_i:
                active = True
            elif qi != qi:
                under = np.maximum(0.0, self.b_low - q_active) / margin
                over = np.maximum(0.0, q_active - self.b_high) / margin
                dz = gain * (under - over) / margin if need_z else None
                c_lim = (
                    0.5 * gain * float(np.dot(under ** 2 + over ** 2, self.movable))
                )
                return dz, c_lim
        if not active:
            return (0.0 if need_z else None), 0.5 * gain * 0.0
        dz_list, sq_list = [], []
        for qi, lo_i, hi_i in zip(q_list, self.b_low_list, self.b_high_list):
            u = lo_i - qi
            u = (u if u > 0.0 else 0.0) / margin
            o = qi - hi_i
            o = (o if o > 0.0 else 0.0) / margin
            dz_list.append(gain * (u - o) / margin)
            sq_list.append(u * u + o * o)
        c_lim = 0.5 * gain * float(np.dot(np.array(sq_list), self.movable_f))
        return (np.array(dz_list) if need_z else None), c_lim

    def _raise_nonfinite(self, com, Jcom, tau, ratios, dtau, g_load, c_load, cost, z, derivatives, need_z):
        """v14 的逐项有限性检查（原顺序、原异常消息）；v15 仅在末尾合并检查失败时调用。"""
        if self.context is not None:
            if not np.isfinite(com).all() or (
                Jcom is not None and not np.isfinite(Jcom).all()
            ):
                raise ValueError("non-finite model COM")
            if not np.isfinite(tau).all():
                raise ValueError("gravity efforts: non-finite values")
            if not np.isfinite(ratios).all():
                raise ValueError("non-finite normalized load; check capacities/model")
            if derivatives and not np.isfinite(dtau).all():
                raise ValueError("invalid gravity effort derivative")
            if derivatives:
                if not np.isfinite(g_load).all() or not math.isfinite(c_load):
                    raise ValueError("non-finite load objective")
            elif not math.isfinite(c_load):
                raise ValueError("non-finite load objective")
        if not math.isfinite(cost) or (need_z and not np.isfinite(z).all()):
            raise ValueError("non-finite secondary objective")

    def __call__(self, q_active, q_full, derivatives=True, components=None, need_z=True):
        """返回 (z * movable_mask, H)；need_z=False 且 derivatives=False 时返回 (None, H)。

        q_full 必须是 q_active 对应的 Pinocchio q（active_q_to_pinocchio_q 的结果）。
        v15：与 v14 逐位一致的改写——限位项按关节标量运算（缓冲区外直接为零）、严格负荷代价在
        全部负荷率 <= soft_ratio 时跳过恒为零的 excess 项、有限性检查合并到末尾（失败时按原顺序
        复核并抛出原异常）、少做冗余拷贝与布尔转换。
        """
        need_z = need_z or derivatives
        if self.multi_turn is not None:
            diff = bounded_joint_difference(
                q_active, self.q_ref, self.lower, self.upper, self.multi_turn
            )
        else:
            diff = q_active - self.q_ref
            if self.periodic_any:
                mask = self.periodic
                diff[mask] = (diff[mask] + np.pi) % (2.0 * np.pi) - np.pi

        z = self.neg_ns_gain * diff if need_z else None
        posture = 0.5 * float(np.dot(self.ns_gain * (diff ** 2), self.movable_f))
        cost = posture
        c_lim = 0.0
        abs_load = excess_load = rel_load = 0.0
        c_com = 0.0
        peak = load_ok = None
        com = Jcom = tau = ratios = dtau = g_load = None
        c_load = 0.0

        # 1. 单侧安全裕度排斥项
        if self.limit_gain > 0.0:
            dz, c_lim = self._limit_terms(q_active, need_z)
            if need_z:
                z += dz
            cost += c_lim

        model, data = self.model, self.data
        if self.context is not None:
            # 2a. 严格负荷模式：与 load_physics + LoadPolicy.evaluate/cost_gradient 同序
            tau = pin.computeGeneralizedGravity(model, data, q_full)[self.guard_v]
            if derivatives:
                dtau = self._dtau(
                    pin.computeGeneralizedGravityDerivatives(model, data, q_full)
                )
            com, Jcom = self._com(q_full, derivatives)
            ratios = np.abs(tau) / self.caps
            peak = float(ratios.max())
            load_ok = peak <= 1.0
            if self.fast_load_cost and peak <= self.soft_ratio:
                # 全部 |r| <= soft_ratio：excess = max(|r| - soft, 0) 恒为 +0.0，故
                #   excess 代价 = 0.5*w_e*0.0/n = 0.0；梯度第二项为 ±0.0（r<0 时 -0.0），
                #   x + (-0.0) == x，x + (+0.0) 只把 -0.0 变 +0.0（r == -0.0 时第二项为 +0.0），
                #   所以 first + 0.0 与原式逐位相同（absolute_weight > 0 保证 r<0 时 first 非零）。
                r = tau / self.caps
                n = self.n_guard
                abs_load = self.half_aw * float((self.weights * r * r).sum()) / n
                excess_load = 0.5 * self.policy.excess_weight * 0.0 / n
                rel_load = 0.0
                dcost_dtau = (
                    (self.aw_w * r / self.caps + 0.0) / n if derivatives else None
                )
            else:
                abs_load, excess_load, rel_load, dcost_dtau = self.policy._cost_terms(
                    tau, self.caps, self.weights, self.tau_ref, derivatives
                )
            c_load = float(abs_load + excess_load + rel_load)
            if derivatives:
                g_load = dtau.T @ dcost_dtau
                z -= g_load
            cost += c_load
            if self.use_com:
                dcom = self.P @ (com - self.ref_com)
                if derivatives:
                    z -= self.com_gain * (Jcom.T @ dcom)
                c_com = 0.5 * self.com_gain * float(dcom @ dcom)
                cost += c_com
            finite = (
                math.isfinite(cost)
                and math.isfinite(float(com @ com))
                and (
                    not derivatives
                    or (
                        math.isfinite(float(dtau.sum()))
                        and math.isfinite(float(Jcom.sum()))
                    )
                )
            )
        else:
            # 2b. 未配置负荷上下文：可选的 URDF 力矩归一化项（与 v6 同序）
            if self.use_gravity:
                tau_cur = pin.computeGeneralizedGravity(model, data, q_full)[
                    self.active_v
                ]
                inv_eff_sq, n_act = self.inv_eff_sq, self.n_act
                dcost_dtau = np.zeros_like(tau_cur) if derivatives else None
                if self.gravity_gain > 0.0:
                    abs_load = (
                        0.5
                        * self.gravity_gain
                        * float(((tau_cur ** 2) * inv_eff_sq).sum())
                        / n_act
                    )
                    cost += abs_load
                    if derivatives:
                        dcost_dtau += self.gravity_gain * (tau_cur * inv_eff_sq) / n_act
                if self.use_rel:
                    tau_diff = tau_cur - self.tau_ref
                    rel_load = (
                        0.5
                        * self.rel_gain
                        * float(((tau_diff ** 2) * inv_eff_sq).sum())
                        / n_act
                    )
                    cost += rel_load
                    if derivatives:
                        dcost_dtau += self.rel_gain * (tau_diff * inv_eff_sq) / n_act
                if derivatives:
                    dtau_act = self._dtau(
                        pin.computeGeneralizedGravityDerivatives(model, data, q_full)
                    )
                    z -= dtau_act.T @ dcost_dtau
            if self.use_com:
                com_cur, J_com = self._com(q_full, derivatives)
                dcom = self.P @ (com_cur - self.ref_com)
                c_com = 0.5 * self.com_gain * float(dcom @ dcom)
                cost += c_com
                if derivatives:
                    z -= self.com_gain * (J_com.T @ dcom)
            finite = math.isfinite(cost)

        if not finite or (need_z and not math.isfinite(float(z @ z))):
            self._raise_nonfinite(
                com, Jcom, tau, ratios, dtau, g_load, c_load, cost, z, derivatives, need_z
            )
        if components is not None:
            components["posture"] = posture
            components["joint_limit"] = c_lim
            components["absolute_load"] = abs_load
            components["excess_load"] = excess_load
            components["relative_load"] = rel_load
            components["com"] = c_com
            if self.context is not None:
                components["peak_ratio"] = peak
                components["load_ok"] = load_ok
        if not need_z:
            return None, cost
        return (z if self.movable_all else z * self.movable_f), cost


def _metric_svd_step(J, error, z, D, damping, rank_rtol=1e-9):
    """基于加权度量 SVD 的主任务阻尼步长与精确数值度量零空间下降步。"""
    J = np.asarray(J, dtype=float)
    D = np.asarray(D, dtype=float)
    if not np.all(np.isfinite(J)) or not np.all(np.isfinite(D)):
        raise ValueError("non-finite Jacobian/metric")
    if not np.allclose(D, D.T, atol=1e-12, rtol=0.0):
        raise ValueError("inverse metric must be symmetric")
    eig, V = np.linalg.eigh(D)
    if np.any(eig <= 0.0):
        raise ValueError("inverse metric must be positive definite")
    S = (V * np.sqrt(eig)) @ V.T
    U, s, Vt = np.linalg.svd(J @ S, full_matrices=True)
    rank = int(
        np.sum(s > rank_rtol * max(1.0, float(s[0]) if s.size else 1.0))
    )
    B = Vt.T
    primary = np.zeros(D.shape[0], dtype=float)
    if rank:
        primary = (
            S
            @ B[:, :rank]
            @ ((s[:rank] / (s[:rank] ** 2 + damping)) * (U[:, :rank].T @ error))
        )
    Z = B[:, rank:]
    secondary = (
        np.zeros_like(primary)
        if z is None
        else S @ Z @ (Z.T @ (S @ z))
    )
    return primary, secondary


def _chol_factor(A):
    """上三角 Cholesky 分解 A = U^T U（LAPACK dpotrf，与 cho_factor(lower=False) 同一例程）。

    非正定时抛 LinAlgError，由调用方转入 SVD 回退。v7 直接调用 LAPACK，绕过 scipy 封装
    （约 25 µs/次）；scipy 1.18 的 cho_factor/cho_solve 与直接调用相差 1e-12 量级、不逐位一致。
    """
    c, info = _lapack.dpotrf(A, lower=0, clean=0, overwrite_a=0)
    if info > 0:
        raise np.linalg.LinAlgError(
            f"{info}-th leading minor not positive definite"
        )
    if info < 0:
        raise ValueError(f"illegal value in {-info}-th argument of internal potrf")
    return c


def _chol_solve(c, b):
    """用 _chol_factor 的结果解 A x = b（LAPACK dpotrs）。"""
    x, info = _lapack.dpotrs(c, b, lower=0, overwrite_b=0)
    if info != 0:
        raise ValueError(f"illegal value in {-info}-th argument of internal potrs")
    return x


def _sns_clamped_core(
    J_active,
    err,
    z_grad,
    null_weight,
    d_inv,
    W_inv,
    movable_mask,
    q_active_cur,
    eff_lower,
    eff_upper,
    damping,
    max_step,
):
    """主动集限位钳制（SNS）+ 正交信赖域：求一步既不越界、步长 <= max_step 的关节增量。

    数学与返回值见 _sns_clamped_core_reference（v14 原实现）。v15 只改对角逆度量热路径的写法，
    每一个参与结果的浮点运算（矩阵乘、Cholesky、逐元素运算）与原实现同序、同内存布局，结果逐位一致：
      - 第一轮（尚无钳制）e_rem = e - J @ 0 恰为 e，||dq_clamped||^2 = 0，直接取用；
      - 全部关节自由时 J_free 用 np.asfortranarray(J)（与 J[:, mask] 同值同为列主序），
        d / z 不再做布尔索引；dq_total / dq_null 不再经零向量拷贝赋值；
      - 无零空间项时不建零向量，dq_task + 0.0 与 dq_task + zeros 逐位相同；
      - 越界检查用 Python 标量逐关节比较（与 numpy 逐元素减法同一 IEEE 运算），只在出现 NaN 时
        回到 numpy 原写法，保证最坏关节的选择与原实现完全相同。
    W_inv（一般矩阵度量）仍走原实现。
    """
    if damping < 0.0 or max_step <= 0.0:
        raise ValueError("invalid damping/max_step")
    if d_inv is None:
        return _sns_clamped_core_reference(
            J_active,
            err,
            z_grad,
            null_weight,
            None,
            W_inv,
            movable_mask,
            q_active_cur,
            eff_lower,
            eff_upper,
            damping,
            max_step,
        )

    n_active = len(q_active_cur)
    free_mask = np.array(movable_mask, dtype=bool)
    free_list = free_mask.tolist()
    n_free = free_list.count(True)
    all_free = n_free == n_active
    dq_clamped = None
    task_dim = err.size
    err_norm_sq = float(err @ err)
    if not math.isfinite(err_norm_sq):
        raise ValueError("non-finite task residual")
    eff_damping = damping + min(1e-3, 1e-4 * err_norm_sq)
    use_null = z_grad is not None and null_weight > 0.0
    clamped_converged = False
    dq_total = dq_null_out = None
    max_step_sq = float(max_step * max_step)
    q_list = lo_list = hi_list = None

    for _ in range(n_free + 1):
        if n_free == 0:
            dq_total = (
                np.zeros(n_active, dtype=float)
                if dq_clamped is None
                else dq_clamped.copy()
            )
            dq_null_out = np.zeros(n_active, dtype=float)
            clamped_converged = True
            break

        if dq_clamped is None:
            rem_step_sq = max_step_sq - 0.0
        else:
            rem_step_sq = max_step_sq - float(dq_clamped @ dq_clamped)
        if rem_step_sq <= 1e-18:
            # 信赖域预算已完全由边界钳制位移占满
            dq_total = (
                np.zeros(n_active, dtype=float)
                if dq_clamped is None
                else dq_clamped.copy()
            )
            dq_null_out = np.zeros(n_active, dtype=float)
            clamped_converged = True
            break

        err_rem = err if dq_clamped is None else err - J_active @ dq_clamped
        if all_free:
            J_free = np.asfortranarray(J_active)
            d_free = d_inv
        else:
            J_free = J_active[:, free_mask]
            d_free = d_inv[free_mask]
        A = (J_free * d_free) @ J_free.T
        A.flat[:: task_dim + 1] += eff_damping

        dq_null_free = None
        try:
            chol_fac = _chol_factor(A)
            rhs = _chol_solve(chol_fac, err_rem)
            dq_task_free = d_free * (J_free.T @ rhs)
            if use_null:
                # 度量一致的零空间下降步：w (D z - D J^T A^-1 J D z)
                Dz = d_free * (z_grad if all_free else z_grad[free_mask])
                JT_inv = J_free.T @ _chol_solve(chol_fac, J_free @ Dz)
                dq_null_free = null_weight * (Dz - d_free * JT_inv)
        except (np.linalg.LinAlgError, ValueError):
            dq_task_free, sec_free = _metric_svd_step(
                J_free,
                err_rem,
                None if z_grad is None else z_grad[free_mask],
                np.diag(d_free),
                eff_damping,
            )
            dq_null_free = null_weight * sec_free

        dq_free = dq_task_free + (0.0 if dq_null_free is None else dq_null_free)
        free_norm = math.sqrt(float(dq_free @ dq_free))
        max_free_norm = math.sqrt(rem_step_sq)

        # 仅缩放自由关节子空间步长，绝不改动已钳制在边界的 dq_clamped
        if free_norm > max_free_norm:
            scale_free = max_free_norm / free_norm
            dq_free *= scale_free
            if dq_null_free is not None:
                dq_null_free *= scale_free

        if all_free:
            dq_total = dq_free
            dq_null_out = (
                np.zeros(n_active, dtype=float)
                if dq_null_free is None
                else dq_null_free
            )
        else:
            dq_total = (
                np.zeros(n_active, dtype=float)
                if dq_clamped is None
                else dq_clamped.copy()
            )
            dq_total[free_mask] = dq_free
            dq_null_out = np.zeros(n_active, dtype=float)
            if dq_null_free is not None:
                dq_null_out[free_mask] = dq_null_free

        # 越界检查：viol_i = max(0, lo_i - q_i', q_i' - hi_i)（仅自由关节），取首个最大者钳制
        if q_list is None:
            q_list = q_active_cur.tolist()
            lo_list = eff_lower.tolist()
            hi_list = eff_upper.tolist()
        worst_idx, worst_val, worst_under = -1, 0.0, False
        has_nan = False
        for i, (qi, di) in enumerate(zip(q_list, dq_total.tolist())):
            if not free_list[i]:
                continue
            qt = qi + di
            under = lo_list[i] - qt
            over = qt - hi_list[i]
            if under != under or over != over:
                has_nan = True
                break
            if under > worst_val:
                worst_idx, worst_val, worst_under = i, under, True
            if over > worst_val:
                worst_idx, worst_val, worst_under = i, over, False
        if has_nan:
            # NaN：按原实现的 numpy 写法确定（max 为 NaN -> 未收敛，argmax 取首个 NaN）
            q_trial = q_active_cur + dq_total
            under_amt = np.where(
                free_mask, np.maximum(0.0, eff_lower - q_trial), 0.0
            )
            over_amt = np.where(
                free_mask, np.maximum(0.0, q_trial - eff_upper), 0.0
            )
            viol_amt = np.maximum(under_amt, over_amt)
            if viol_amt.max() <= 1e-12:
                clamped_converged = True
                break
            worst_idx = int(np.argmax(viol_amt))
            worst_under = bool(under_amt[worst_idx] >= over_amt[worst_idx])
        elif worst_val <= 1e-12:
            clamped_converged = True
            break

        # 钳制本轮越界最严重的单个自由关节；由于边界距离严格小于 |dq_free[worst_idx]|，
        # 钳制后的 ||dq_clamped|| 必然严格小于 max_step
        if dq_clamped is None:
            dq_clamped = np.zeros(n_active, dtype=float)
        if worst_under:
            dq_clamped[worst_idx] = eff_lower[worst_idx] - q_active_cur[worst_idx]
        else:
            dq_clamped[worst_idx] = eff_upper[worst_idx] - q_active_cur[worst_idx]
        free_mask[worst_idx] = False
        free_list[worst_idx] = False
        n_free -= 1
        all_free = False

    return dq_total, dq_null_out, clamped_converged


def _sns_clamped_core_reference(
    J_active,
    err,
    z_grad,
    null_weight,
    d_inv,
    W_inv,
    movable_mask,
    q_active_cur,
    eff_lower,
    eff_upper,
    damping,
    max_step,
):
    """v14 原实现（逐行未改），v15 起只服务一般矩阵 W_inv 路径与逐位对照测试。

    主动集限位钳制（SNS）+ 正交信赖域：求一步既不越界、步长 <= max_step 的关节增量。

    返回 (dq_total, dq_null, clamped_converged)。记 D 为逆度量（关节权重越大 D 越小、该关节越“贵”），
    A = J_f D_f J_f^T + λI：
      任务步   dq_task = D_f J_f^T A^-1 e_rem
      零空间步 dq_null = w · (D_f z - D_f J_f^T A^-1 J_f D_f z)
    其中 e_rem = e - J dq_clamped 是扣除已钳制关节位移后的剩余误差，z = -∇H 为次任务负梯度。
    零空间投影 D - D J^T A^-1 J D 对称半正定，所以 z^T dq_null >= 0，次任务一定是下降方向
    （与 SVD 回退 _metric_svd_step 的 S Z Z^T S z 一致）。
    循环：每轮把越界最严重的一个自由关节钳制到边界（位移 = 边界 - 当前值，此后不再缩放），
    其余自由关节按剩余信赖域半径 sqrt(max_step^2 - ||dq_clamped||^2) 缩放；被钳制关节的越界距离
    严格小于其本轮位移，故 ||dq_clamped|| < max_step 恒成立。至多 n_free + 1 轮；
    clamped_converged=False 表示轮数用尽仍有越界（调用方记诊断）。

    d_inv 不为 None 时为对角逆度量的对角元（单种子迭代的热路径，向量运算）；否则用一般矩阵 W_inv。
    对角情形下 J D J^T == (J * d) J^T、D v == d * v 逐位一致。
    """
    if damping < 0.0 or max_step <= 0.0:
        raise ValueError("invalid damping/max_step")

    diagonal = d_inv is not None
    n_active = len(q_active_cur)
    free_mask = np.asarray(movable_mask, dtype=bool).copy()
    dq_clamped = np.zeros(n_active, dtype=float)
    dq_null_out = np.zeros(n_active, dtype=float)
    task_dim = err.size
    err_norm_sq = float(err @ err)
    if not math.isfinite(err_norm_sq):
        raise ValueError("non-finite task residual")
    eff_damping = damping + min(1e-3, 1e-4 * err_norm_sq)

    max_passes = int(np.sum(free_mask)) + 1
    clamped_converged = False
    dq_total = np.zeros(n_active, dtype=float)
    max_step_sq = float(max_step * max_step)

    for _ in range(max_passes):
        if not np.any(free_mask):
            dq_total = dq_clamped.copy()
            dq_null_out.fill(0.0)
            clamped_converged = True
            break

        clamped_norm_sq = float(dq_clamped @ dq_clamped)
        rem_step_sq = max_step_sq - clamped_norm_sq
        if rem_step_sq <= 1e-18:
            # 信赖域预算已完全由边界钳制位移占满
            dq_total = dq_clamped.copy()
            dq_null_out.fill(0.0)
            clamped_converged = True
            break

        err_rem = err - J_active @ dq_clamped
        J_free = J_active[:, free_mask]
        if diagonal:
            d_free = d_inv[free_mask]
            A = (J_free * d_free) @ J_free.T
        else:
            W_inv_free = W_inv[np.ix_(free_mask, free_mask)]
            A = J_free @ W_inv_free @ J_free.T
        A.flat[:: task_dim + 1] += eff_damping

        try:
            chol_fac = _chol_factor(A)
            rhs = _chol_solve(chol_fac, err_rem)
            JT_rhs = J_free.T @ rhs
            dq_task_free = d_free * JT_rhs if diagonal else W_inv_free @ JT_rhs
            dq_null_free = np.zeros_like(dq_task_free)
            if z_grad is not None and null_weight > 0.0:
                # 度量一致的零空间下降步：w (D z - D J^T A^-1 J D z)
                z_free = z_grad[free_mask]
                Dz = d_free * z_free if diagonal else W_inv_free @ z_free
                invA_JDz = _chol_solve(chol_fac, J_free @ Dz)
                JT_inv = J_free.T @ invA_JDz
                dq_null_free = null_weight * (
                    Dz - (d_free * JT_inv if diagonal else W_inv_free @ JT_inv)
                )
        except (np.linalg.LinAlgError, ValueError):
            dq_task_free, sec_free = _metric_svd_step(
                J_free,
                err_rem,
                None if z_grad is None else z_grad[free_mask],
                np.diag(d_free) if diagonal else W_inv_free,
                eff_damping,
            )
            dq_null_free = null_weight * sec_free

        dq_free = dq_task_free + dq_null_free
        free_norm = math.sqrt(float(dq_free @ dq_free))
        max_free_norm = math.sqrt(rem_step_sq)

        # 仅缩放自由关节子空间步长，绝不改动已钳制在边界的 dq_clamped
        if free_norm > max_free_norm:
            scale_free = max_free_norm / free_norm
            dq_free *= scale_free
            dq_null_free *= scale_free

        dq_total = dq_clamped.copy()
        dq_total[free_mask] = dq_free

        dq_null_out.fill(0.0)
        dq_null_out[free_mask] = dq_null_free

        q_trial = q_active_cur + dq_total
        under_amt = np.where(
            free_mask, np.maximum(0.0, eff_lower - q_trial), 0.0
        )
        over_amt = np.where(
            free_mask, np.maximum(0.0, q_trial - eff_upper), 0.0
        )
        viol_amt = np.maximum(under_amt, over_amt)

        if viol_amt.max() <= 1e-12:
            clamped_converged = True
            break

        # 钳制本轮越界最严重的单个自由关节；由于边界距离严格小于 |dq_free[worst_idx]|，
        # 钳制后的 ||dq_clamped|| 必然严格小于 max_step
        worst_idx = int(np.argmax(viol_amt))
        if under_amt[worst_idx] >= over_amt[worst_idx]:
            dq_clamped[worst_idx] = (
                eff_lower[worst_idx] - q_active_cur[worst_idx]
            )
        else:
            dq_clamped[worst_idx] = (
                eff_upper[worst_idx] - q_active_cur[worst_idx]
            )
        free_mask[worst_idx] = False

    return dq_total, dq_null_out, clamped_converged


def _antipodal_tangent(a, motion=None):
    """180° 对径死点（a 与目标方向相反、叉积为零）时选取切向，返回垂直于 a 的单位向量。

    motion 为 3 x n 矩阵，列是各关节能产生的“方向变化 / 转轴”；取其在 a 切平面内投影的
    主左奇异向量，即当前构型下实际最容易转动的方向。旧版用“与 a 最不平行的坐标轴叉乘”，
    少轴或特定构型下该方向可能恰好不可达（雅可比行为零）而卡死在对径点。
    motion 为 None（无雅可比的试探点评估）或不可达时退回坐标轴叉乘；
    此时误差范数只依赖转角而与切向无关，不影响线搜索判据。
    """
    if motion is not None and motion.size:
        M = motion - np.outer(a, a @ motion)
        U, sv, _ = np.linalg.svd(M, full_matrices=False)
        if sv.size and sv[0] > 1e-9:
            t = U[:, 0] - a * float(a @ U[:, 0])
            n_t = float(np.linalg.norm(t))
            if n_t > 1e-12:
                return t / n_t
    t = np.cross(a, np.eye(3)[int(np.argmin(np.abs(a)))])
    return t / np.linalg.norm(t)


def _make_task_terms(
    model,
    data,
    pin_maps,
    fid,
    target,
    mode,
    axis,
    cone,
    characteristic_length=0.5,
):
    """构造主任务求值器 terms(q_active, jacobian=True, all_frames=False, need_A=None)。

    与 dls_task_terms 是同一份实现（公共函数即“构造 + 求值一次”）。每个种子构造一次时，
    目标 SE3 / 目标工具轴 b = R_target @ axis / 索引 / q 模板只算一次；求值时：
      - all_frames=False 只更新末端单帧位姿（迭代内其它帧不被读取），True 与 v6 一样更新全部帧；
      - need_A=False（默认随 jacobian）不构造任务雅可比 A，返回的 "J" 为 None。
    叉积、截断、拼接、范数改为与 numpy 原实现逐位一致的等价写法，以减少 Python 调用开销。
    """
    to_q = _make_q_builder(model, pin_maps)
    v_idx = pin_maps["active_v_indices"]
    n_v = len(v_idx)
    nv = model.nv
    v_range = _contiguous_range(v_idx)
    v_slice = None if v_range is None else slice(*v_range)
    cl = characteristic_length
    target_p = np.array(target[:3, 3], dtype=float)
    if mode == "full":
        target_se3 = pin.SE3(target[:3, :3], target[:3, 3])
        ref_frame = pin.LOCAL
        b = None
    elif mode in ("tool_dir", "tool_cone"):
        target_se3 = None
        ref_frame = pin.LOCAL_WORLD_ALIGNED
        b = target[:3, :3] @ axis
        b0, b1, b2 = b.tolist()
    else:
        raise ValueError("invalid orientation mode")

    def terms(q_active, jacobian=True, all_frames=False, need_A=None):
        build_A = jacobian if need_A is None else bool(need_A)
        q_full = to_q(q_active)
        pin.forwardKinematics(model, data, q_full)
        if all_frames:
            pin.updateFramePlacements(model, data)
        else:
            pin.updateFramePlacement(model, data, fid)
        # v15：不再拷贝 SE3。computeFrameJacobian 会按同一公式重写 data.oMf[fid]（数值逐位相同），
        # 其余读取都在下一次 Pinocchio 调用之前完成
        current = data.oMf[fid]
        pos_error = target_p - current.translation
        pos = _vec_norm(pos_error)

        if jacobian:
            J_all = np.asarray(
                pin.computeFrameJacobian(model, data, q_full, fid, ref_frame)
            ).reshape(6, nv)
            # 与 J_all[:, v_idx]（花式索引结果为列主序）同值同布局
            J = (
                np.asfortranarray(J_all[:, v_slice])
                if v_slice is not None
                else J_all[:, v_idx]
            )
        elif build_A:
            J = np.zeros((6, n_v), dtype=float)
        else:
            J = None

        antipodal = False
        A = None
        if mode == "full":
            delta = current.inverse() * target_se3
            e = np.array(pin.log6(delta).vector, copy=True)
            angle = _vec_norm(e[3:])
            if build_A:
                # 矩阵乘结果本身就是新数组（v14 多一次 np.array 拷贝）
                A = pin.Jlog6(delta.inverse()) @ J if jacobian else J.copy()
                A[:3] /= cl
            e[:3] /= cl
            violation = angle
        else:
            a = np.asarray(current.rotation) @ axis
            a0, a1, a2 = a.tolist()
            # 与 np.cross 同序的标量公式（逐位一致）
            cross = np.array(
                (a1 * b2 - a2 * b1, a2 * b0 - a0 * b2, a0 * b1 - a1 * b0)
            )
            sine = _vec_norm(cross)
            cosine = min(max(float(a @ b), -1.0), 1.0)
            angle = math.atan2(sine, cosine)

            if mode == "tool_dir":
                edir = b - a
                Jdir = -skew(a) @ J[3:] if build_A else None  # 行：工具轴方向 a 的变化率 da/dq
                if cosine < -1.0 + 1e-8:
                    tangent = _antipodal_tangent(a, Jdir if jacobian else None)
                    edir = angle * tangent
                    antipodal = True
                e = np.concatenate((pos_error / cl, edir))
                if build_A:
                    A = np.vstack([J[:3] / cl, Jdir])
                violation = angle
            else:
                violation = max(0.0, angle - cone)
                row = np.zeros(n_v, dtype=float) if build_A else None
                if violation > 0.0:
                    if sine > 1e-9:
                        tangent = cross / sine       # 把 a 转向 b 的转轴
                    else:
                        # 对径：任意垂直于 a 的转轴都能减小夹角，取雅可比上最容易转动的那一个
                        tangent = (
                            _antipodal_tangent(a, J[3:] if jacobian else None)
                            if build_A
                            else None
                        )
                        antipodal = True
                    if build_A:
                        row = tangent @ J[3:]
                e = np.concatenate((pos_error / cl, (violation,)))
                if build_A:
                    A = np.vstack([J[:3] / cl, row])

        return {
            "error": e,
            "J": A,
            "pos_err": pos,
            "ori_err": angle,
            "ori_violation": violation,
            "merit": 0.5 * float(e @ e),
            "antipodal": antipodal,
            "q_full": q_full,
        }

    return terms


def dls_task_terms(
    model,
    data,
    pin_maps,
    fid,
    q_active,
    target,
    mode,
    axis,
    cone,
    characteristic_length=0.5,
    jacobian=True,
):
    """计算主任务残差与解析雅可比（含一致的 tool_dir 180° 对径点方向导数 -skew(a) @ J[3:]）。

    与 v6 行为一致：更新全部帧位姿；jacobian=False 时 "J" 为按零雅可比构造的同形状矩阵。
    """
    terms = _make_task_terms(
        model, data, pin_maps, fid, target, mode, axis, cone, characteristic_length
    )
    return terms(q_active, jacobian=jacobian, all_frames=True, need_A=True)


def pinocchio_ik_one_seed(
    pin_model,
    pin_data,
    pin_maps,
    ee_frame_id,
    T_target,
    seed,
    lower,
    upper,
    max_iter=PIN_IK_MAX_ITER,
    damping=PIN_IK_DAMPING,
    pos_tol=PIN_IK_POS_TOL,
    ori_tol=PIN_IK_ORI_TOL,
    max_step=PIN_IK_MAX_STEP,
    orientation_mode="full",
    tool_axis=None,
    tool_cone_half_angle=None,
    joint_weights=None,
    q_ref=None,
    include_waist=False,
    fixed_joint_mask=None,
    waist_release=False,
    waist_max_reach=None,
    nullspace_gain=None,
    waist_null_gain=None,
    waist_task_weight=None,
    waist_task_weight_far=None,
    gravity_torque_gain=None,
    com_xy_gain=None,
    joint_limit_gain=None,
    *,
    load_context=None,
    deadline=None,
    waist_load_release=False,
    characteristic_length=0.5,
    backtrack_steps=12,
    virtual_half_windows=None,
    relative_gravity_gain=None,
    posture_use_equivalent_branch=False,
    shoulder_joint=None,
    posture_settle=True,
    margin_bounds=None,
):
    """单种子 DLS IK：主任务为末端位姿，零空间次任务为回中 / 限位裕度 / 负荷 / 质心。

    返回 (q, diag)。q 优先取“位姿达标点”中按负荷峰值容差带 + 次任务代价择优的那个；
    没有达标点时返回主任务误差最小的迭代点（diag["converged"] = False）。
    迭代全程满足有效限位（URDF 限位 ∩ virtual_half_windows），由 SNS 钳制 + 边界吸附保证。
    diag["stop_reason"]：SETTLED / SETTLE_LIMIT / SMALL_STEP / STALLED_NO_PROGRESS /
    LINE_SEARCH_STALLED / TIME_BUDGET / ITERATION_LIMIT / POSE_REACHED。
    posture_settle=False（v15，续算用，与双臂 continuation 相同）：未达标时零空间原始步长限幅为
    NO_SETTLE_NULL_STEP_RATIO × |e|；位姿达标、负荷合格（strict）且可动关节离限位 >= SAFE_MARGIN 的
    第一个迭代点即返回（stop_reason = POSE_REACHED，不做达标后的姿态整理）；种子本身满足时原样返回。
    达标点负荷超限或贴限位时照常整理，直到满足上述条件或按原判据停止。margin_bounds = (lower, upper)
    给出该裕度所用的限位（调用方的有效限位；续算框边不是物理限位），None = 本次迭代限位。
    实现要点：每个种子预计算主任务 / 次任务求值器（_make_task_terms / _SecondaryObjective）与对角逆度量
    向量；SNS 的 Cholesky 直接调用 LAPACK。shoulder_joint 显式指定腰部释放所用的肩关节（None = 旧约定
    "_Link2" 后缀）；未传 waist_max_reach 时按 estimate_arm_max_reach 沿运动链逐段累加估计；
    load_context=None 时质心参考与代价使用同一整机质心（含根 link 质量）。
    """
    mode = validate_orientation_mode(orientation_mode)
    target = validate_transform(T_target)
    axis = finite_vector(
        TOOL_AXIS_X if tool_axis is None else tool_axis, 3, "tool axis"
    )
    if np.linalg.norm(axis) < 1e-12:
        raise ValueError("zero tool axis")
    axis /= np.linalg.norm(axis)
    cone = normalize_cone_half_angle(tool_cone_half_angle)

    q = finite_vector(seed, len(pin_maps["active_names"]), "seed")
    ref = q.copy() if q_ref is None else finite_vector(q_ref, len(q), "q_ref")
    if (
        max_iter < 1
        or pos_tol <= 0.0
        or ori_tol <= 0.0
        or not np.isfinite(characteristic_length)
        or characteristic_length <= 0.0
    ):
        raise ValueError("invalid DLS iteration/tolerances/scale")

    waist = np.array(
        [n in WAIST_JOINT_NAMES for n in pin_maps["active_names"]], dtype=bool
    )
    movable = (
        np.ones(len(q), dtype=bool)
        if fixed_joint_mask is None
        else ~np.asarray(fixed_joint_mask, dtype=bool)
    )
    if movable.shape != q.shape:
        raise ValueError("fixed mask shape mismatch")

    lo, hi = _build_effective_bounds(
        lower,
        upper,
        ref,
        virtual_half_windows=virtual_half_windows,
        active_names=pin_maps["active_names"],
    )
    q = np.minimum(np.maximum(q, lo), hi)

    gain = NULLSPACE_GAIN if nullspace_gain is None else float(nullspace_gain)
    waist_gain = (
        WAIST_NULL_GAIN if waist_null_gain is None else float(waist_null_gain)
    )
    ns_gain = np.where(waist, waist_gain, gain)
    com_gain = COM_XY_GAIN if com_xy_gain is None else float(com_xy_gain)
    lim_gain = (
        JOINT_LIMIT_GAIN
        if joint_limit_gain is None
        else float(joint_limit_gain)
    )
    grav_gain = (
        GRAVITY_TORQUE_GAIN
        if gravity_torque_gain is None
        else float(gravity_torque_gain)
    )
    rel_grav_gain = (
        RELATIVE_GRAVITY_GAIN
        if relative_gravity_gain is None
        else float(relative_gravity_gain)
    )
    near_weight = (
        WAIST_TASK_WEIGHT
        if waist_task_weight is None
        else float(waist_task_weight)
    )
    far_weight = (
        WAIST_TASK_WEIGHT_FAR
        if waist_task_weight_far is None
        else float(waist_task_weight_far)
    )
    weights = (
        np.ones(len(q), dtype=float)
        if joint_weights is None
        else finite_vector(joint_weights, len(q), "joint weights")
    )
    if (
        np.any(weights <= 0.0)
        or min(gain, waist_gain, com_gain, lim_gain, grav_gain, rel_grav_gain)
        < 0.0
        or min(near_weight, far_weight) <= 0.0
    ):
        raise ValueError("invalid DLS gain/weight")

    active_v_indices = np.array(pin_maps["active_v_indices"], dtype=np.intp)
    context = None if load_context is None else dict(load_context)
    if context is not None and gravity_torque_gain is not None:
        context["policy"] = replace(
            context["policy"], absolute_weight=grav_gain
        )

    ref_full = active_q_to_pinocchio_q(pin_model, pin_maps, ref)
    pin.forwardKinematics(pin_model, pin_data, ref_full)
    pin.updateFramePlacements(pin_model, pin_data)

    ref_tau, ref_com = None, None
    if context is not None:
        ref_physics = load_physics(
            pin_model, pin_data, pin_maps, ref, context, False
        )
        ref_tau, ref_com = ref_physics["tau"], ref_physics["com"]
    else:
        if rel_grav_gain > 0.0:
            tau_ref_full = np.array(
                pin.computeGeneralizedGravity(pin_model, pin_data, ref_full),
                copy=True,
            )
            ref_tau = tau_ref_full[active_v_indices].copy()
        if include_waist and com_gain > 0.0:
            # v8：与次任务代价同一个整机质心（含根 link 质量）；v7 用 pin.centerOfMass（不含根质量），
            # 参考与代价口径不一致，q_ready 处质心项梯度不为零
            ref_com = _reference_com(pin_model, pin_data, ref_full)

    shoulder_name = (
        _legacy_shoulder_joint(pin_maps) if shoulder_joint is None else shoulder_joint
    )
    shoulder_id = (
        pin_maps["name_to_joint_id"].get(shoulder_name)
        if shoulder_name is not None
        else None
    )
    shoulder_ref = (
        None
        if shoulder_id is None
        else np.array(pin_data.oMi[shoulder_id].translation, copy=True)
    )
    if waist_release and waist_max_reach is None:
        waist_max_reach = estimate_arm_max_reach(
            pin_model, pin_maps, ee_frame_id, shoulder_name
        )
    release_enabled = bool(
        waist_release
        and include_waist
        and np.any(waist & movable)
        and waist_max_reach
        and shoulder_ref is not None
    )

    def release_fraction(distance):
        if not release_enabled:
            return 0.0
        t_val = (distance - WAIST_RELEASE_START * waist_max_reach) / max(
            (WAIST_RELEASE_END - WAIST_RELEASE_START) * waist_max_reach,
            1e-6,
        )
        t_val = min(max(t_val, 0.0), 1.0)   # 标量截断，与 np.clip 结果相同
        return float(t_val * t_val * (3.0 - 2.0 * t_val))

    ref_release = (
        release_fraction(float(np.linalg.norm(target[:3, 3] - shoulder_ref)))
        if release_enabled
        else 0.0
    )

    diag = {
        "converged": False,
        "iters": 0,
        "final_pos_err": np.nan,
        "final_ori_err": np.nan,
        "stalled": False,
        "error_history": [],
        "stop_reason": "ITERATION_LIMIT",
        "load_configured": context is not None,
        "waist_released_for_load": bool(waist_load_release),
        "antipodal_escape_used": False,
        "antipodal_escape_steps": 0,
        "sns_clamp_incomplete": False,
        "bound_snaps": 0,
    }

    best_q, best_merit = q.copy(), math.inf
    settle = 0
    periodic_unbounded = np.asarray(
        active_periodic_mask(pin_maps), dtype=bool
    )
    periodic_any = bool(np.any(periodic_unbounded))
    # 逆度量 D 为对角阵：只保存对角元向量（v6 为 np.diag(1.0 / weights)，SNS 结果逐位一致）
    d_base = 1.0 / weights
    waist_idx = np.nonzero(waist)[0]
    waist_movable = waist & movable
    has_waist_dof = bool(np.any(waist_movable))
    target_p = np.array(target[:3, 3], dtype=float)

    # 每种子构造一次的求值器（与 dls_task_terms / tests/dls_compat 的 v6 形式入口同一实现）
    objective = _SecondaryObjective(
        pin_model,
        pin_data,
        pin_maps,
        ref,
        lo,
        hi,
        ns_gain,
        active_v_indices,
        pin_maps["effort_limits"],
        movable,
        include_waist,
        ref_tau,
        ref_com,
        grav_gain,
        com_gain,
        lim_gain,
        load_context=context,
        relative_gravity_gain=rel_grav_gain,
        posture_use_equivalent_branch=posture_use_equivalent_branch,
    )
    terms = _make_task_terms(
        pin_model,
        pin_data,
        pin_maps,
        ee_frame_id,
        target,
        mode,
        axis,
        cone,
        characteristic_length,
    )

    def is_pose(t_dict):
        return t_dict["pos_err"] <= pos_tol and t_dict["ori_violation"] <= ori_tol

    lo_list, hi_list = lo.tolist(), hi.tolist()

    def snap(trial):
        """试探点只允许浮点舍入级越界（<= BOUND_SNAP_TOL），吸附回边界；真实越界返回 None。

        v15：越界量按关节标量计算（与 numpy 逐元素减法同一 IEEE 运算），NaN 传播与
        max(float((lo - trial).max()), float((trial - hi).max())) 相同。
        """
        under_max = over_max = -math.inf
        nan_under = nan_over = False
        for t_i, lo_i, hi_i in zip(trial.tolist(), lo_list, hi_list):
            a = lo_i - t_i
            b = t_i - hi_i
            if a > under_max:
                under_max = a
            elif a != a:
                nan_under = True
            if b > over_max:
                over_max = b
            elif b != b:
                nan_over = True
        if nan_under:
            under_max = math.nan
        if nan_over:
            over_max = math.nan
        viol = max(under_max, over_max)
        if viol <= 0.0:
            return trial
        if viol > BOUND_SNAP_TOL:
            return None
        diag["bound_snaps"] += 1
        return np.minimum(np.maximum(trial, lo), hi)

    # ---- 位姿达标迭代点择优 ----------------------------------------------------
    # 负荷合格（load_ok）优先；同为合格时，峰值落在 [floor, floor + band] 容差带内的点里取
    # 次任务代价最小者（floor = 已见合格点的最低峰值，单调不增，所以峰值不会被逐步“蚕食”）；
    # 同为超载时峰值越低越好。旧版严格字典序 (超载, 峰值, 代价) 会因峰值差 0.001 丢掉姿态改善。
    # band 取 peak_tie 的一部分：种子内容差 + 候选间容差 (peak_tie) 合计仍接近 peak_tie。
    peak_band = (
        0.0
        if context is None
        else POSE_PEAK_BAND_FRACTION * float(context["policy"].peak_tie)
    )
    pose_best = {"q": None, "overloaded": False, "peak": math.inf, "cost": math.inf}
    peak_floor = math.inf

    def offer_pose(values, cost_val, parts_val):
        nonlocal peak_floor
        overloaded = context is not None and not parts_val.get("load_ok", True)
        peak = 0.0 if context is None else float(parts_val.get("peak_ratio", 0.0))
        cur = pose_best
        if cur["q"] is None or (cur["overloaded"] and not overloaded):
            take = True
        elif overloaded != cur["overloaded"]:
            take = False
        elif overloaded:
            take = peak < cur["peak"] - 1e-12 or (
                peak <= cur["peak"] + 1e-12 and cost_val < cur["cost"]
            )
        else:
            floor = min(peak_floor, peak)
            take = peak <= floor + peak_band and (
                cur["peak"] > floor + peak_band or cost_val < cur["cost"]
            )
        if not overloaded:
            peak_floor = min(peak_floor, peak)
        if take:
            cur.update(
                q=values.copy(), overloaded=overloaded, peak=peak, cost=cost_val
            )
        return take

    stall_ref, stall_it = None, 0
    settle_prev_cost, settle_slow = None, 0
    timed_out = False
    pose_reached = False
    if not posture_settle:
        if margin_bounds is None:
            margin_lo, margin_hi = lo, hi
        else:
            margin_lo = np.asarray(margin_bounds[0], dtype=float)
            margin_hi = np.asarray(margin_bounds[1], dtype=float)
            if (
                margin_lo.shape != q.shape
                or margin_hi.shape != q.shape
                or np.isnan(margin_lo).any()
                or np.isnan(margin_hi).any()
            ):
                raise ValueError("margin_bounds must be two vectors of joint bounds")

    def settle_free(values, parts_val):
        """posture_settle=False 的停止条件：负荷合格且可动关节离限位 >= SAFE_MARGIN（无界关节不计）。"""
        if context is not None and not parts_val.get("load_ok", True):
            return False
        gap = np.minimum(values - margin_lo, margin_hi - values)[movable]
        gap = gap[np.isfinite(gap)]
        return gap.size == 0 or float(gap.min()) >= SAFE_MARGIN

    for it in range(int(max_iter)):
        if deadline is not None and time.monotonic() >= deadline:
            diag["stop_reason"] = "TIME_BUDGET"
            break

        t = terms(q, jacobian=True)
        # 目标 - 当前肩位置（v14 先拷贝肩位置、稍后再相减；同一减法，提前做省一次拷贝）
        shoulder_gap = (
            target_p - pin_data.oMi[shoulder_id].translation
            if release_enabled
            else None
        )

        parts = {}
        z, cost = objective(q, t["q_full"], True, parts)
        pose_ok = is_pose(t)

        diag["iters"] = it + 1
        diag["antipodal_escape_used"] |= t["antipodal"]
        diag["error_history"].append(
            {
                "iteration": it + 1,
                "pos_err": t["pos_err"],
                "ori_err": t["ori_err"],
                "ori_violation": t["ori_violation"],
            }
        )

        if t["merit"] < best_merit:
            best_q, best_merit = q.copy(), t["merit"]

        if pose_ok:
            if not posture_settle and settle_free(q, parts):
                # v15 续算：不做姿态整理，直接返回本点（不经 offer_pose 择优：之前的达标点都不满足停止条件）
                pose_best.update(
                    q=q.copy(),
                    overloaded=False,
                    peak=0.0 if context is None else float(parts.get("peak_ratio", 0.0)),
                    cost=cost,
                )
                pose_reached = True
                diag["stop_reason"] = "POSE_REACHED"
                break
            offer_pose(q, cost, parts)
            settle += 1
            if settle > MAX_SETTLE_EXTRA_ITERS:
                diag["stop_reason"] = "SETTLE_LIMIT"
                break
            # 姿态整理的边际收益判据：连续 SETTLE_PATIENCE 步代价相对下降都 < SETTLE_COST_RTOL 即停
            if settle_prev_cost is not None:
                gain = settle_prev_cost - cost
                slow = gain < SETTLE_COST_RTOL * max(abs(settle_prev_cost), 1e-12)
                settle_slow = settle_slow + 1 if slow else 0
                if settle_slow >= SETTLE_PATIENCE:
                    diag["stop_reason"] = "SETTLED"
                    break
            settle_prev_cost = cost
        else:
            # 停滞检测：每 PIN_IK_STALL_ITERS 次迭代，主任务误差范数下降不足 PIN_IK_STALL_RTOL
            # 即放弃本种子（典型场景：锁腰种子目标超出手臂可达范围，误差停在几十毫米处缓慢爬行）
            err_norm = math.sqrt(2.0 * best_merit)
            if stall_ref is None:
                stall_ref, stall_it = err_norm, it
            elif it - stall_it >= PIN_IK_STALL_ITERS:
                if err_norm > (1.0 - PIN_IK_STALL_RTOL) * stall_ref:
                    diag["stalled"] = True
                    diag["stop_reason"] = "STALLED_NO_PROGRESS"
                    break
                stall_ref, stall_it = err_norm, it

        # 逆度量 D（对角元向量）：手臂 = 1；腰部在近距离目标时“贵”(1/near_weight)，远目标或回程时逐步释放
        d_inv = d_base.copy()
        if has_waist_dof:
            alpha = 1.0 if waist_load_release else 0.0
            if release_enabled and shoulder_gap is not None:
                current_distance = _vec_norm(shoulder_gap)
                waist_diff = q - ref
                if periodic_any:
                    waist_diff[periodic_unbounded] = (
                        waist_diff[periodic_unbounded] + np.pi
                    ) % (2.0 * np.pi) - np.pi
                waist_diff = waist_diff[waist_movable]
                ret = _vec_norm(waist_diff) / max(WAIST_RETURN_RELEASE_RAD, 1e-6)
                alpha = max(
                    alpha,
                    ref_release,
                    release_fraction(current_distance),
                    min(max(ret, 0.0), 1.0),
                )
            d_inv[waist_idx] = (
                1.0 - alpha
            ) / near_weight + alpha / far_weight

        # 主任务误差越小，零空间次任务权重越大（达标后再放大 1.5 倍做姿态整理）
        null_weight = (1.5 if pose_ok else 1.0) / (
            1.0
            + (
                _vec_norm(t["error"])
                / max(NULLSPACE_ERR_SCALE, 1e-6)
            )
            ** 2
        )

        if not posture_settle and not pose_ok:
            # v15 续算：零空间原始步 w·D z（可动关节）限幅到 NO_SETTLE_NULL_STEP_RATIO × |e|
            raw = null_weight * _vec_norm((d_inv * z)[movable])
            cap = NO_SETTLE_NULL_STEP_RATIO * _vec_norm(t["error"])
            if raw > cap:
                null_weight *= cap / raw

        dq, dq_null, clamped_ok = _sns_clamped_core(
            t["J"],
            t["error"],
            z,
            null_weight,
            d_inv,
            None,
            movable,
            q,
            lo,
            hi,
            damping,
            max_step,
        )
        diag["sns_clamp_incomplete"] |= not clamped_ok
        dq_null_norm = _vec_norm(dq_null)

        # 末端已达标且零空间步长收敛：最后试一次整步，仍达标且择优通过才采纳
        if pose_ok and dq_null_norm < POSTURE_SETTLE_TOL:
            trial = snap(q + dq)
            if trial is not None:
                tt = terms(trial, jacobian=False)
                if is_pose(tt):
                    tparts = {}
                    trial_cost = objective(
                        trial, tt["q_full"], False, tparts, need_z=False
                    )[1]
                    if offer_pose(trial, trial_cost, tparts):
                        q = trial
            diag["stop_reason"] = "SETTLED"
            break

        if _vec_norm(dq) < 1e-10:
            diag["stop_reason"] = "SMALL_STEP"
            break

        # 回溯线搜索（严格单调）：先试“主任务 + 零空间”合成步，失败再试纯主任务步（按需才求解）
        directions = [dq]
        need_pure = (not pose_ok) and dq_null_norm > 1e-12
        accepted = False
        d_idx = 0
        while d_idx < len(directions) and not accepted and not timed_out:
            direction = directions[d_idx]
            for backtrack in range(int(backtrack_steps)):
                if deadline is not None and time.monotonic() >= deadline:
                    timed_out = True
                    break
                # 1.0 * direction 与 direction 逐位相同，首个试探点省一次乘法
                trial = snap(
                    q + direction if backtrack == 0 else q + (0.5 ** backtrack) * direction
                )
                if trial is None:
                    continue
                tt = terms(trial, jacobian=False)
                if pose_ok:
                    # 姿态整理阶段：必须保持位姿达标，且次任务代价严格下降
                    progress = (
                        is_pose(tt)
                        and objective(trial, tt["q_full"], False, need_z=False)[1]
                        < cost - 1e-13
                    )
                elif t["antipodal"]:
                    # 180° 对径死点逃逸：方向误差严格减小、位置误差增量不超过一个容差即可
                    progress = (
                        tt["ori_violation"] < t["ori_violation"] - 1e-8
                        and tt["pos_err"] <= t["pos_err"] + pos_tol
                    )
                else:
                    progress = is_pose(tt) or (
                        tt["merit"]
                        < t["merit"] - max(1e-14, 1e-6 * t["merit"])
                    )
                if progress:
                    q = trial
                    accepted = True
                    if not pose_ok and t["antipodal"]:
                        diag["antipodal_escape_steps"] += 1
                    break
            d_idx += 1
            if d_idx == 1 and need_pure and not accepted and not timed_out:
                pure_dq, _, _ = _sns_clamped_core(
                    t["J"],
                    t["error"],
                    None,
                    0.0,
                    d_inv,
                    None,
                    movable,
                    q,
                    lo,
                    hi,
                    damping,
                    max_step,
                )
                directions.append(pure_dq)

        if not accepted:
            if timed_out:
                diag["stop_reason"] = "TIME_BUDGET"
            elif pose_ok:
                diag["stop_reason"] = "SETTLED"
            else:
                diag["stalled"] = True
                diag["stop_reason"] = "LINE_SEARCH_STALLED"
            break

    if not pose_reached:
        last = terms(q, jacobian=False)
        if is_pose(last):
            lparts = {}
            offer_pose(
                q, objective(q, last["q_full"], False, lparts, need_z=False)[1], lparts
            )
        elif last["merit"] < best_merit:
            best_q = q.copy()

    result = pose_best["q"] if pose_best["q"] is not None else best_q
    # 严格在有效限位 [lo, hi]（包含 virtual_half_windows 约束）内选择最接近起始 seed 的合法 2πk 分支
    multi_turn = active_multi_turn_mask(pin_maps)
    result = select_closest_bounded_branch(
        result, seed, lo, hi, multi_turn_mask=multi_turn
    )

    # 最终评估与 v6 一样更新全部帧位姿（调用方之后读取 pin_data 时状态与 v6 相同）
    final = terms(result, jacobian=False, all_frames=True)
    parts = {}
    objective(result, final["q_full"], False, parts, need_z=False)
    diag.update(
        converged=is_pose(final),
        final_pos_err=final["pos_err"],
        final_ori_err=final["ori_err"],
        final_ori_violation=final["ori_violation"],
        secondary_components=parts,
    )
    if context is not None:
        diag.update(
            final_peak_ratio=float(parts["peak_ratio"]),
            final_load_ok=bool(parts["load_ok"]),
        )
    return result.copy(), diag


# =============================================================================
# 9. 备用与历史求解算法 (Legacy Solvers: Direct-Psi & SSIK)
# =============================================================================

# =============================================================================
# Analytical Jacobian direct-psi IK
# =============================================================================
def geometric_jacobian(chain, q, fk_result=None):
    if fk_result is None:
        fk_result = chain.fk_all(q)

    link_T, joint_origin_T = fk_result

    T_ee = link_T[chain.ee_link]
    p_ee = T_ee[:3, 3]

    n = len(chain.active_joints)
    J = np.zeros((6, n))
    for i, aj in enumerate(chain.active_joints):
        T_j = joint_origin_T[aj["name"]]
        z_i = T_j[:3, :3] @ np.asarray(aj["axis"], dtype=float)
        p_i = T_j[:3, 3]
        if aj["type"] in ("revolute", "continuous"):
            J[:3, i] = np.cross(z_i, p_ee - p_i)
            J[3:, i] = z_i
        elif aj["type"] == "prismatic":
            J[:3, i] = z_i

    return J


def joint_position_jacobian(chain, joint_name, joint_origin_T):
    n = len(chain.active_joints)
    J = np.zeros((3, n))

    p_target = joint_origin_T[joint_name][:3, 3]

    before = set()
    for j in chain.joints:
        if j["name"] == joint_name:
            break
        if j["type"] in ("revolute", "continuous", "prismatic"):
            before.add(j["name"])

    for i, aj in enumerate(chain.active_joints):
        if aj["name"] not in before:
            continue
        T_j = joint_origin_T[aj["name"]]
        z_i = T_j[:3, :3] @ np.asarray(aj["axis"], dtype=float)
        p_i = T_j[:3, 3]

        if aj["type"] in ("revolute", "continuous"):
            J[:, i] = np.cross(z_i, p_target - p_i)
        elif aj["type"] == "prismatic":
            J[:, i] = z_i

    return J


def dpsi_dq(
    chain,
    q,
    fk_result=None,
    S_joint=S_JOINT,
    E_joint=E_JOINT,
    W_joint=W_JOINT,
    ref_dir=REF_DIR,
    eps=1e-7,
):
    if fk_result is None:
        fk_result = chain.fk_all(q)

    _, joint_origin_T = fk_result
    n = len(chain.active_joints)
    S = joint_origin_T[S_joint][:3, 3].copy()
    E = joint_origin_T[E_joint][:3, 3].copy()
    W = joint_origin_T[W_joint][:3, 3].copy()
    J_S = joint_position_jacobian(chain, S_joint, joint_origin_T)
    J_E = joint_position_jacobian(chain, E_joint, joint_origin_T)
    J_W = joint_position_jacobian(chain, W_joint, joint_origin_T)

    try:
        psi0 = sew_angle_from_points(S, E, W, ref_dir)
    except Exception:
        return np.zeros((1, n))

    dpsi_dS = np.zeros(3)
    dpsi_dE = np.zeros(3)
    dpsi_dW = np.zeros(3)
    for k in range(3):
        try:
            Sp = S.copy()
            Sp[k] += eps
            dpsi_dS[k] = (
                wrap_to_pi(sew_angle_from_points(Sp, E, W, ref_dir) - psi0)
                / eps
            )
        except Exception:
            pass
        try:
            Ep = E.copy()
            Ep[k] += eps
            dpsi_dE[k] = (
                wrap_to_pi(sew_angle_from_points(S, Ep, W, ref_dir) - psi0)
                / eps
            )
        except Exception:
            pass
        try:
            Wp = W.copy()
            Wp[k] += eps
            dpsi_dW[k] = (
                wrap_to_pi(sew_angle_from_points(S, E, Wp, ref_dir) - psi0)
                / eps
            )
        except Exception:
            pass

    return (dpsi_dS @ J_S + dpsi_dE @ J_E + dpsi_dW @ J_W).reshape(1, n)


def evaluate_direct_psi_solution(
    chain,
    T_target,
    psi_des,
    q,
    q_ref=None,
    S_joint=S_JOINT,
    E_joint=E_JOINT,
    W_joint=W_JOINT,
    ref_dir=REF_DIR,
    orientation_mode="full",
    tool_axis=None,
    tool_cone_half_angle=None,
    bounds=None,
):
    if bounds is None:
        lower, upper = chain.bounds()
    else:
        lower, upper = bounds

    q = np.asarray(q, dtype=float)
    orientation_mode = validate_orientation_mode(orientation_mode)
    tool_cone_half_angle = normalize_cone_half_angle(tool_cone_half_angle)

    if tool_axis is None:
        tool_axis = TOOL_AXIS_X

    tool_axis = normalize(tool_axis)
    Tq = chain.fk(q)

    dp = Tq[:3, 3] - T_target[:3, 3]
    pos_err = float(np.linalg.norm(dp))

    tool_dir_angle_err = np.nan
    tool_cone_violation = np.nan
    if orientation_mode == "full":
        _, drot = pose_error(Tq, T_target)
        ori_err = float(np.linalg.norm(drot))
    elif orientation_mode == "tool_dir":
        tool_dir_angle_err = tool_direction_angle_error(
            T_current=Tq,
            T_target=T_target,
            tool_axis=tool_axis,
        )
        ori_err = tool_dir_angle_err
    elif orientation_mode == "tool_cone":
        tool_cone_violation, tool_dir_angle_err = tool_cone_angle_error(
            T_current=Tq,
            T_target=T_target,
            tool_axis=tool_axis,
            cone_half_angle=tool_cone_half_angle,
        )
        ori_err = tool_cone_violation

    psi_q = chain.psi(
        q,
        s_joint=S_joint,
        e_joint=E_joint,
        w_joint=W_joint,
        ref_dir=ref_dir,
    )
    psi_err = float(abs(wrap_to_pi(psi_q - psi_des)))

    margin, margin_idx, _ = joint_margin(q, lower, upper)

    if q_ref is None:
        q_ref_dq = np.nan
    else:
        q_ref_dq = joint_dist(q, q_ref)

    return {
        "q": q,
        "pos_err": pos_err,
        "ori_err": ori_err,
        "orientation_mode": orientation_mode,
        "tool_axis": tool_axis.copy(),
        "psi": float(psi_q),
        "psi_err": psi_err,
        "margin": margin,
        "margin_idx": margin_idx,
        "q_ref_dq": q_ref_dq,
        "tool_dir_angle_err": float(tool_dir_angle_err),
        "tool_dir_angle_err_deg": (
            float(np.rad2deg(tool_dir_angle_err))
            if np.isfinite(tool_dir_angle_err)
            else np.nan
        ),
        "tool_cone_half_angle": float(tool_cone_half_angle),
        "tool_cone_half_angle_deg": float(np.rad2deg(tool_cone_half_angle)),
        "tool_cone_violation": float(tool_cone_violation),
        "tool_cone_violation_deg": (
            float(np.rad2deg(tool_cone_violation))
            if np.isfinite(tool_cone_violation)
            else np.nan
        ),
    }


class DirectPsiProblem:
    def __init__(
        self,
        chain,
        T_target,
        psi_des,
        q_ref,
        q_bias,
        q_center,
        q_scale,
        lower,
        upper,
        w_pos,
        w_ori,
        w_psi,
        w_seed,
        w_bias,
        w_center,
        w_limit,
        limit_margin,
        S_joint,
        E_joint,
        W_joint,
        ref_dir,
        orientation_mode="full",
        tool_axis=None,
        tool_cone_half_angle=None,
    ):
        self.chain = chain
        self.T_target = np.asarray(T_target, dtype=float)
        self.R_target = self.T_target[:3, :3]
        self.psi_des = float(psi_des)

        self.q_ref = q_ref
        self.q_bias = q_bias
        self.q_center = q_center
        self.q_scale = q_scale
        self.lower = lower
        self.upper = upper

        self.w_pos = w_pos
        self.w_ori = w_ori
        self.w_psi = w_psi
        self.w_seed = w_seed
        self.w_bias = w_bias
        self.w_center = w_center
        # w_limit / limit_margin 保留以兼容旧版构造签名；关节限位在 solve_direct_psi 中由 least_squares bounds 约束
        self.w_limit = w_limit
        self.limit_margin = limit_margin

        self.S_joint = S_joint
        self.E_joint = E_joint
        self.W_joint = W_joint
        self.ref_dir = ref_dir
        self.orientation_mode = validate_orientation_mode(orientation_mode)
        self.tool_cone_half_angle = normalize_cone_half_angle(
            tool_cone_half_angle
        )

        if tool_axis is None:
            tool_axis = TOOL_AXIS_X

        self.tool_axis = normalize(tool_axis)
        self.n = len(chain.active_joints)

        self._cache_q = None
        self._cache_fk = None

    def _fk(self, q):
        q = np.asarray(q, dtype=float)

        if self._cache_q is not None and np.array_equal(q, self._cache_q):
            return self._cache_fk

        self._cache_q = q.copy()
        self._cache_fk = self.chain.fk_all(q)
        return self._cache_fk

    def residual_task(self, q):
        try:
            fk_result = self._fk(q)
            link_T, joint_origin_T = fk_result

            T_ee = link_T[self.chain.ee_link]
            dp = T_ee[:3, 3] - self.T_target[:3, 3]

            if self.orientation_mode == "full":
                _, e_ori = pose_error(T_ee, self.T_target)
            elif self.orientation_mode == "tool_dir":
                e_ori, _, _ = tool_direction_residual(
                    T_current=T_ee,
                    T_target=self.T_target,
                    tool_axis=self.tool_axis,
                )
            elif self.orientation_mode == "tool_cone":
                e_ori, _, _, _, _, _, _ = tool_cone_residual(
                    T_current=T_ee,
                    T_target=self.T_target,
                    tool_axis=self.tool_axis,
                    cone_half_angle=self.tool_cone_half_angle,
                )

            S = joint_origin_T[self.S_joint][:3, 3]
            E = joint_origin_T[self.E_joint][:3, 3]
            W = joint_origin_T[self.W_joint][:3, 3]

            psi_q = sew_angle_from_points(S, E, W, self.ref_dir)
            e_psi = wrap_to_pi(psi_q - self.psi_des)

            return np.concatenate(
                [
                    self.w_pos * dp,
                    self.w_ori * e_ori,
                    [self.w_psi * e_psi],
                ]
            )

        except Exception:
            return np.ones(7) * 1e3

    def jacobian_task(self, q):
        try:
            fk_result = self._fk(q)
            J_geom = geometric_jacobian(self.chain, q, fk_result)
            Jv = J_geom[:3, :]
            Jw = J_geom[3:, :]

            if self.orientation_mode == "full":
                J_ori = self.R_target.T @ Jw
            elif self.orientation_mode == "tool_dir":
                link_T, _ = fk_result
                T_ee = link_T[self.chain.ee_link]
                d_current = tool_direction_from_T(
                    T_ee,
                    tool_axis=self.tool_axis,
                )
                J_ori = -skew(d_current) @ Jw
            elif self.orientation_mode == "tool_cone":
                link_T, _ = fk_result
                T_ee = link_T[self.chain.ee_link]

                d_current = tool_direction_from_T(
                    T_ee,
                    tool_axis=self.tool_axis,
                )
                d_target = tool_direction_from_T(
                    self.T_target,
                    tool_axis=self.tool_axis,
                )

                _, _, violation, inside = closest_direction_on_cone(
                    d_current=d_current,
                    d_target=d_target,
                    cone_half_angle=self.tool_cone_half_angle,
                )

                if inside or violation <= 0.0:
                    J_ori = np.zeros_like(Jw)
                else:
                    J_ori = -skew(d_current) @ Jw

            J_psi = dpsi_dq(
                self.chain,
                q,
                fk_result,
                self.S_joint,
                self.E_joint,
                self.W_joint,
                self.ref_dir,
            )

            return np.vstack(
                [
                    self.w_pos * Jv,
                    self.w_ori * J_ori,
                    self.w_psi * J_psi,
                ]
            )

        except Exception:
            return np.eye(7, self.n) * 1e-6

    def residual_reg(self, q):
        try:
            base = self.residual_task(q)
            n = self.n

            e_seed = (
                (q - self.q_ref) / self.q_scale
                if self.q_ref is not None
                else np.zeros(n)
            )
            e_bias = (
                (q - self.q_bias) / self.q_scale
                if self.q_bias is not None
                else np.zeros(n)
            )
            e_center = (q - self.q_center) / self.q_scale
            return np.concatenate(
                [
                    base,
                    self.w_seed * e_seed,
                    self.w_bias * e_bias,
                    self.w_center * e_center,
                ]
            )

        except Exception:
            return np.ones(7 + 3 * self.n) * 1e3

    def jacobian_reg(self, q):
        try:
            J_task = self.jacobian_task(q)
            n = self.n

            inv_scale = np.diag(1.0 / self.q_scale)
            J_seed = (
                self.w_seed * inv_scale
                if self.q_ref is not None
                else np.zeros((n, n))
            )
            J_bias = (
                self.w_bias * inv_scale
                if self.q_bias is not None
                else np.zeros((n, n))
            )
            J_center = self.w_center * inv_scale
            return np.vstack(
                [
                    J_task,
                    J_seed,
                    J_bias,
                    J_center,
                ]
            )

        except Exception:
            m = 7 + 3 * self.n
            return np.eye(m, self.n) * 1e-6


def solve_direct_psi(
    chain,
    T_target,
    psi_des,
    seeds,
    S_joint=S_JOINT,
    E_joint=E_JOINT,
    W_joint=W_JOINT,
    ref_dir=REF_DIR,
    q_ref=None,
    q_bias=None,
    orientation_mode="full",
    tool_axis=None,
    tool_cone_half_angle=None,
    verbose=False,
    w_pos=100.0,
    w_ori=20.0,
    w_psi=20.0,
    w_seed=0.03,
    w_bias=0.00,
    w_center=0.005,
    w_limit=0.05,
    limit_margin=0.10,
    max_nfev=150,
    refine_task=True,
    pos_tol=1e-5,
    ori_tol=1e-5,
    psi_tol=1.5,
    max_solutions=None,
    bounds=None,
):
    if bounds is None:
        lower, upper = chain.bounds()
    else:
        lower, upper = bounds

    orientation_mode = validate_orientation_mode(orientation_mode)
    tool_cone_half_angle = normalize_cone_half_angle(tool_cone_half_angle)
    if tool_axis is None:
        tool_axis = TOOL_AXIS_X

    tool_axis = normalize(tool_axis)
    T_target = np.asarray(T_target, dtype=float)
    psi_des = float(wrap_to_pi(psi_des))

    q_scale = joint_range_scale(lower, upper)
    q_center_arr = joint_center_from_bounds(lower, upper)

    q_ref_arr = (
        clip_to_bounds(q_ref, lower, upper) if q_ref is not None else None
    )
    q_bias_arr = (
        clip_to_bounds(q_bias, lower, upper) if q_bias is not None else None
    )
    prob = DirectPsiProblem(
        chain=chain,
        T_target=T_target,
        psi_des=psi_des,
        q_ref=q_ref_arr,
        q_bias=q_bias_arr,
        q_center=q_center_arr,
        q_scale=q_scale,
        lower=lower,
        upper=upper,
        w_pos=w_pos,
        w_ori=w_ori,
        w_psi=w_psi,
        w_seed=w_seed,
        w_bias=w_bias,
        w_center=w_center,
        w_limit=w_limit,
        limit_margin=limit_margin,
        S_joint=S_joint,
        E_joint=E_joint,
        W_joint=W_joint,
        ref_dir=ref_dir,
        orientation_mode=orientation_mode,
        tool_axis=tool_axis,
        tool_cone_half_angle=tool_cone_half_angle,
    )

    results = []
    unique_solutions = []
    clean_seeds = []
    for seed in seeds:
        seed = np.asarray(seed, dtype=float)
        if seed.shape != lower.shape:
            continue

        seed = clip_to_bounds(seed, lower, upper)
        dup = any(np.linalg.norm(seed - old) < 1e-8 for old in clean_seeds)

        if not dup:
            clean_seeds.append(seed)

    for si, seed in enumerate(clean_seeds):
        prob._cache_q = None

        res1 = least_squares(
            prob.residual_reg,
            seed,
            jac=prob.jacobian_reg,
            bounds=(lower, upper),
            method="trf",
            xtol=1e-12,
            ftol=1e-12,
            gtol=1e-12,
            max_nfev=max_nfev,
            x_scale="jac",
        )
        q = res1.x
        res_used = res1

        if refine_task:
            prob._cache_q = None

            res2 = least_squares(
                prob.residual_task,
                q,
                jac=prob.jacobian_task,
                bounds=(lower, upper),
                method="trf",
                xtol=1e-12,
                ftol=1e-12,
                gtol=1e-12,
                max_nfev=max(200, max_nfev // 2),
                x_scale="jac",
            )

            q = res2.x
            res_used = res2

        try:
            info = evaluate_direct_psi_solution(
                chain,
                T_target,
                psi_des,
                q,
                q_ref=q_ref_arr,
                S_joint=S_joint,
                E_joint=E_joint,
                W_joint=W_joint,
                ref_dir=ref_dir,
                orientation_mode=orientation_mode,
                tool_axis=tool_axis,
                tool_cone_half_angle=tool_cone_half_angle,
                bounds=(lower, upper),
            )
        except Exception as e:
            if verbose:
                print(f"seed {si}: eval failed: {repr(e)}")
            continue

        pos_err = info["pos_err"]
        ori_err = info["ori_err"]
        psi_err = info["psi_err"]

        ok = pos_err < pos_tol and ori_err < ori_tol and psi_err < psi_tol

        if verbose:
            print(
                f"seed {si}: ok={ok} "
                f"pos={pos_err:.3e} ori={ori_err:.3e} psi={psi_err:.3e} "
                f"margin={info['margin']:.3e} nfev={res_used.nfev}"
            )

        if ok and unique_add(unique_solutions, q):
            item = dict(info)
            item.update(
                {
                    "cost": float(res_used.cost),
                    "success": bool(res_used.success),
                    "message": res_used.message,
                    "nfev": int(res_used.nfev),
                }
            )
            results.append(item)

            if max_solutions is not None and len(results) >= max_solutions:
                break

    PSI_BUCKET = np.deg2rad(2.0)

    def sort_key(s):
        dq = s.get("q_ref_dq", np.nan)
        if not np.isfinite(dq):
            dq = 0.0

        psi_bucket = int(s["psi_err"] / PSI_BUCKET)
        return (
            psi_bucket,
            dq,
            -s.get("margin", -1e9),
            s["psi_err"],
            s["pos_err"],
            s["ori_err"],
            s.get("cost", 0.0),
        )

    if q_ref_arr is not None:
        results.sort(key=sort_key)
    else:
        results.sort(
            key=lambda x: (
                x["psi_err"],
                x["pos_err"],
                x["ori_err"],
                -x.get("margin", -1e9),
                x.get("cost", 0.0),
            )
        )

    return results


# =============================================================================
# Seeds / SSIK
# =============================================================================
def make_direct_local_seeds(q_prev, q_ready, lower, upper, rng):
    seeds = []

    q_prev = clip_to_bounds(q_prev, lower, upper)
    q_ready = clip_to_bounds(q_ready, lower, upper)
    seeds.append(q_prev.copy())

    for scale in DIRECT_LOCAL_NOISE_SCALES:
        q_noise = q_prev + rng.normal(scale=scale, size=len(q_prev))
        seeds.append(clip_to_bounds(q_noise, lower, upper))

    seeds.append(q_ready.copy())

    for scale in DIRECT_LOCAL_NOISE_SCALES:
        q_noise = q_ready + rng.normal(scale=scale, size=len(q_ready))
        seeds.append(clip_to_bounds(q_noise, lower, upper))

    q_mid = np.asarray(q_ready, dtype=float).copy()
    finite = np.isfinite(lower) & np.isfinite(upper)
    q_mid[finite] = 0.5 * (lower[finite] + upper[finite])
    seeds.append(clip_to_bounds(q_mid, lower, upper))
    return unique_seed_list(seeds)


def extract_q(sol):
    if hasattr(sol, "q"):
        return np.asarray(sol.q, dtype=float)
    if isinstance(sol, dict) and "q" in sol:
        return np.asarray(sol["q"], dtype=float)

    return np.asarray(sol, dtype=float)


def call_ssik_solve(arm, T_target, q_seed=None):
    kwargs = dict(
        respect_limits=SSIK_RESPECT_LIMITS,
        allow_refinement=SSIK_ALLOW_REFINEMENT,
        refinement_max_iters=SSIK_REFINEMENT_MAX_ITERS,
    )

    if SSIK_MAX_SOLUTIONS is not None:
        kwargs["max_solutions"] = SSIK_MAX_SOLUTIONS

    if q_seed is not None:
        kwargs["q_seed"] = np.asarray(q_seed, dtype=float)

    try:
        return arm.solve(T_target, **kwargs)
    except TypeError:
        kwargs.pop("max_solutions", None)
        try:
            return arm.solve(T_target, **kwargs)
        except TypeError:
            kwargs.pop("q_seed", None)
            try:
                return arm.solve(T_target, **kwargs)
            except TypeError:
                return arm.solve(T_target)


def collect_ssik_candidates_and_seeds(
    arm,
    chain,
    T_target,
    psi_ready,
    q_prev,
    q_ready,
    lower,
    upper,
    orientation_mode="full",
    tool_axis=None,
    tool_cone_half_angle=None,
    include_waist=False,
    multi_turn_mask=None,
):
    if arm is None:
        return [], []

    try:
        q_seed = q_ready if SSIK_USE_READY_AS_SEED else None
        sols = call_ssik_solve(arm, T_target, q_seed=q_seed)

        if sols is None:
            sols = []
    except Exception:
        return [], []

    ssik_candidates = []

    for sol in sols:
        try:
            q = extract_q(sol)
        except Exception:
            continue

        if q.ndim != 1 or len(q) != len(lower):
            continue
        q = select_closest_bounded_branch(
            q, q_prev, lower, upper, multi_turn_mask=multi_turn_mask
        )
        q = clip_to_bounds(q, lower, upper)

        info = score_candidate(
            chain,
            q,
            T_target,
            psi_ready,
            q_prev,
            q_ready,
            lower,
            upper,
            orientation_mode=orientation_mode,
            tool_axis=tool_axis,
            tool_cone_half_angle=tool_cone_half_angle,
            apply_waist_return=include_waist,
        )

        info["source"] = "ssik"
        info["solve_method"] = "ssik_pose"
        info["fallback_used"] = True

        ssik_candidates.append(info)

    ssik_candidates.sort(key=candidate_sort_key)
    ssik_seeds_for_direct = [
        c["q"] for c in ssik_candidates[:SSIK_TOP_K_FOR_DIRECT]
    ]

    return ssik_candidates, ssik_seeds_for_direct


# =============================================================================
# 10. 多阶段候选解流水线与 HAArmIKSolver 主类 (Multi-Stage Pipeline & Unified OOP Solver)
# =============================================================================

def collect_pinocchio_candidates_and_seeds(
    pin_model,
    pin_data,
    pin_maps,
    ee_frame_id,
    chain,
    T_target_pin,
    T_target_local,
    psi_ready,
    q_prev,
    q_ready,
    lower,
    upper,
    rng,
    orientation_mode="full",
    tool_axis=None,
    tool_cone_half_angle=None,
    include_waist=False,
    dls_params=None,
    *,
    load_search=None,
    comfort_bounds=None,
    comfort_continuation_tol=None,
    comfort_info=None,
    continuation_local=False,
    seed_library=None,
    locked_reach=None,
):
    """多阶段、多种子 DLS 搜索；返回 (候选列表, direct-psi 种子, per_seed 诊断)。

    求解类别（启用舒适窗或续算局部优先时判定）：目标相对 FK(q_prev) 的主任务误差（与 DLS 同一度量）
    位置 <= pos_tol 且姿态 <= ori_tol 为“续算”（路点跟踪），否则为“重新定位”；
    comfort_continuation_tol = (pos_tol, ori_tol)，默认 (0.1 × 臂展, 0.1 rad)。

    阶段（依次判定；候选池跨阶段累积，各阶段最多 max_seeds 个名额，共享时间预算）：
      continuation  续算且 continuation_local=True：随动续算（种子 q_prev，范围 = 有效限位 ∩ q_prev ±
                    continuation_max_step ∩ 跳变框，DLS 单步 <= CONTINUATION_DLS_STEP_FRACTION × 该值；
                    v15 默认 CONTINUATION_POSTURE_SETTLE=False 不做达标后的姿态整理，随动续算解等于 q_prev 且
                    可接受时直接记 continuation_hold 返回、不再跑锁腰续算）+ 锁腰续算（未触发 hold 且 q_prev 的
                    腰部偏离 q_ready <= 该值时才跑；锁腰解负荷超 soft_ratio 时释放腰部、重跑随动续算）。
                    有局部可接受候选（单关节步长 <= continuation_max_step）时候选池只保留它们、不再跑后面的阶段
                    （stop = CONTINUATION_LOCAL），否则候选留在池中继续常规搜索
      comfort       重新定位、舒适窗确实收窄限位且含 q_ready：有效限位 ∩ 舒适窗（∩ 跳变框）
      full          未启用舒适窗时的唯一常规阶段；或舒适阶段没有可接受候选时回退到有效限位：
                    种子 q_prev -> 舒适阶段主任务误差最小的未达标终点（comfort_edge）-> q_ready -> 种子库 -> 扰动，
                    不跑锁腰种子，随机扰动与未启用舒适窗时相同（预先复制随机数状态）
      jump_box      jump_rescue 且没有跳变框（上限都 >= π）：常规阶段只找到“只因跳变超限被拒”的解时，在
                    q_prev ± max_joint_jump 内重搜；舒适阶段已出现这类解时，回退阶段直接在框内搜索
      unboxed       跳变框（joint_jump_policy="reject" 且上限 < π 的关节，常规阶段都在 q_prev ± 上限内搜索）内没有
                    可接受候选时，按有效限位不加框重搜一次（只有超限解时照旧报 JOINT_JUMP_LIMIT）

    常规阶段的种子队列（regular_queue）：q_prev -> 锁腰（q_prev 且腰部锁在 q_ready）-> q_ready -> 种子库 ->
    随机扰动（make_scatter_seeds）。种子库种子排到时才选：只用本阶段范围内且满足跳变上限的样本，按任务空间
    距离由近到远，彼此及与本阶段已跑种子的关节差 ∞-范数 >= library_min_separation；续算类求解默认不用
    （library_continuation）。locked_seed_skip_tol=None 时为 v6 顺序（锁腰在 q_prev 之前，不跳过、不重跑）。
      - q_prev 的解已合格且腰部 |q - q_ready| <= locked_seed_skip_tol：锁腰种子与之等价，跳过（占名额、算已比较）；
      - 目标超出锁腰时手臂的严格可达范围（locked_reach，见 locked_arm_reach_bound）：同样跳过锁腰种子；
      - 锁腰解峰值 > soft_ratio：释放腰部（之后的种子 waist_load_release=True），紧接着用释放状态重跑 q_prev；
      - comfort_early_fallback：q_prev / 锁腰 / q_ready / 种子库都跑完仍没有可接受候选即结束舒适阶段
        （stop = COMFORT_PATIENCE；没有种子库时检查点在第 comfort_patience 个名额之后）。

    每个种子：pinocchio_ik_one_seed -> 在本阶段范围内对齐到最接近 q_prev 的 2πk 分支 -> score_candidate（限位
    裕度按有效限位计，舒适窗不是物理限位）-> 单关节跳变 -> strict 负荷 -> 按 2πk 等价关节差去重入池。
    种子异常计数并写入 per_seed，不中断搜索。

    候选判据：可接受 = 位姿达标、跳变不超限、限位裕度 >= SAFE_MARGIN，strict 下负荷合格；
              合格 = 可接受且（strict 下）负荷峰值 <= early_stop_ratio。
    早停：strict —— 已用 >= min_seeds 个名额（含腰时还需 >= min_unlocked 个不锁腰种子且已完成锁腰比较）且已有
          合格候选；非 strict —— 已用 >= 2 个名额、已完成锁腰比较且当前种子的解合格（锁腰解本身不触发）。

    诊断：per_seed 每个种子一行（seed_kind / search_stage / 迭代诊断 / 跳变 / 去重）；comfort_info 写入求解类别、
    舒适窗是否生效与回退；strict 下 load_context["search_diagnostics"] 记录停止原因、名额、种子库与重搜情况。
    """
    params = dict(dls_params or {})
    continuation_reach = params.pop("continuation_reach", None)  # 只用于求解类别判据，不传给 DLS
    q_prev = np.asarray(q_prev, dtype=float)
    q_ready = np.asarray(q_ready, dtype=float)
    search = dict(load_search or {})
    context = params.get("load_context")
    strict = context is not None
    jump_limits = search.get("max_joint_jump")
    periodic = active_periodic_mask(pin_maps)

    # 早停规则：至少用掉 min_seeds 个名额（其中 >= min_unlocked 个不锁腰、且已完成锁腰比较）且已有
    # “低负荷”候选（峰值 <= early_stop_ratio、跳变不超限、限位裕度足够）即停止；否则继续找更低负荷的解。
    max_seeds = int(search.get("max_seeds", 10))
    min_seeds = int(search.get("min_seeds", 2))
    min_unlocked = int(search.get("min_unlocked", 1))
    if min(max_seeds, min_seeds, min_unlocked) < 1 or min_seeds > max_seeds:
        raise ValueError("invalid load search seed counts")
    if include_waist and max_seeds < min_unlocked + 1:
        raise ValueError(
            "max_seeds must include locked and unlocked comparisons"
        )
    skip_tol = search.get("locked_seed_skip_tol", LOCKED_SEED_SKIP_TOL)
    if skip_tol is not None:
        skip_tol = float(skip_tol)
        if not math.isfinite(skip_tol) or skip_tol < 0.0:
            raise ValueError(
                "locked_seed_skip_tol must be None or a finite nonnegative angle"
            )
    q_prev_first = bool(include_waist and skip_tol is not None)

    waist_mask = np.array(
        [n in WAIST_JOINT_NAMES for n in pin_maps["active_names"]],
        dtype=bool,
    )
    # 统一计算有效搜索边界（含显式配置的 virtual_half_windows）
    eff_lower, eff_upper = _build_effective_bounds(
        lower,
        upper,
        q_ready,
        virtual_half_windows=params.get("virtual_half_windows"),
        active_names=pin_maps["active_names"],
    )

    # v8：舒适窗与有效限位求交；不收窄任何关节、或不含 q_ready 时常规搜索退化为单一 full 阶段
    comfort = None
    regime = None
    comfort_enabled = bool(include_waist and comfort_bounds is not None)
    if comfort_enabled or continuation_local:
        # 求解类别：目标相对 FK(q_prev) 的主任务误差（与 DLS 同一度量）
        reach = (
            continuation_reach
            or params.get("waist_max_reach")
            or params.get("characteristic_length", 0.5)
        )
        tol_pos, tol_ori = (
            (WAIST_COMFORT_CONTINUATION_STEP * float(reach), WAIST_COMFORT_CONTINUATION_STEP)
            if comfort_continuation_tol is None
            else (float(comfort_continuation_tol[0]), float(comfort_continuation_tol[1]))
        )
        start = dls_task_terms(
            pin_model,
            pin_data,
            pin_maps,
            ee_frame_id,
            np.asarray(q_prev, dtype=float),
            T_target_pin,
            orientation_mode,
            TOOL_AXIS_X if tool_axis is None else tool_axis,
            normalize_cone_half_angle(tool_cone_half_angle),
            characteristic_length=params.get("characteristic_length", 0.5),
            jacobian=False,
        )
        regime = (
            "continuation"
            if start["pos_err"] <= tol_pos and start["ori_violation"] <= tol_ori
            else "reposition"
        )
    local_first = bool(continuation_local and regime == "continuation")
    cont_step = float(search.get("continuation_max_step", CONTINUATION_MAX_STEP))
    if not math.isfinite(cont_step) or cont_step <= 0.0:
        raise ValueError("continuation_max_step must be a positive finite angle")
    local_used = False
    local_pick = None
    if comfort_enabled and regime == "reposition":
        c_lo = np.maximum(eff_lower, np.asarray(comfort_bounds[0], dtype=float))
        c_hi = np.minimum(eff_upper, np.asarray(comfort_bounds[1], dtype=float))
        narrows = bool(np.any(c_lo > eff_lower) or np.any(c_hi < eff_upper))
        if (
            narrows
            and np.all(c_lo < c_hi)
            and np.all(q_ready >= c_lo)
            and np.all(q_ready <= c_hi)
        ):
            comfort = (c_lo, c_hi)
    # full 回退阶段使用与 v7 完全相同的随机扰动：在 comfort 阶段消耗随机数之前复制生成器状态
    rng_full = copy.deepcopy(rng) if comfort is not None else None

    near = (
        WAIST_TASK_WEIGHT
        if params.get("waist_task_weight") is None
        else float(params["waist_task_weight"])
    )
    weights = np.array(
        [near if n in WAIST_JOINT_NAMES else 1.0 for n in pin_maps["active_names"]],
        dtype=float,
    )
    char_len = float(params.get("characteristic_length", 0.5))

    candidates, per_seed, unique = [], [], []
    seed_errors, first_seed_error = 0, None
    unlocked = 0
    released = False
    multi_turn = active_multi_turn_mask(pin_maps)
    early_stop_ratio = context["policy"].early_stop_ratio if strict else None
    edge_pool = []                  # comfort 阶段未达标种子的 (主任务 merit, seed_idx, q)

    # v13：种子库种子（延迟计算：只有排到时才把目标换到库基座坐标系并选样本）
    lib_k = 0 if seed_library is None else int(search.get("library_seeds", SEED_LIBRARY_SEEDS) or 0)
    # 续算类求解默认不用种子库（见 SEED_LIBRARY_CONTINUATION）
    lib_skip_continuation = bool(
        lib_k > 0
        and regime == "continuation"
        and not search.get("library_continuation", SEED_LIBRARY_CONTINUATION)
    )
    if lib_skip_continuation:
        lib_k = 0
    lib_sep = float(search.get("library_min_separation", SEED_LIBRARY_MIN_SEPARATION))
    lib_target = []
    lib_used = 0
    reach_skips = 0
    pos_tol_reach = float(params.get("pos_tol", PIN_IK_POS_TOL))

    def locked_unreachable(seed):
        """腰锁在 q_ready 的种子：目标位置是否超出手臂严格可达范围（三角不等式上界）。"""
        if locked_reach is None:
            return False
        joint_id, bound = locked_reach
        pin.forwardKinematics(
            pin_model, pin_data, active_q_to_pinocchio_q(pin_model, pin_maps, seed)
        )
        origin = np.asarray(pin_data.oMi[int(joint_id)].translation, dtype=float)
        target_p = np.asarray(T_target_pin, dtype=float)[:3, 3]
        return float(np.linalg.norm(target_p - origin)) > float(bound) + pos_tol_reach

    early_fallback = bool(search.get("comfort_early_fallback", False))
    patience = int(search.get("comfort_patience", WAIST_COMFORT_PATIENCE))
    if patience < 1:
        raise ValueError("comfort_patience must be >= 1")

    # ---- 候选判据（共用“位姿达标 + 限位裕度足够 + strict 下负荷合格”，只在跳变 / 峰值上区分）----
    def feasible_but_jump(cand):
        """H1-H3：位姿达标、限位裕度 >= SAFE_MARGIN；strict 下静态负荷合格。不看跳变。"""
        return bool(
            cand.get("pose_ok")
            and cand.get("margin", -np.inf) >= SAFE_MARGIN
            and (not strict or cand.get("load_ok"))
        )

    def acceptable(cand):
        """可接受 = 满足全部硬约束 H1-H4（不要求低于早停阈值）：阶段成功、回退与重搜的判据。"""
        return bool(cand.get("joint_jump_ok", True)) and feasible_but_jump(cand)

    def qualified(cand):
        """早停认可：可接受，且 strict 下负荷峰值 <= early_stop_ratio（低负荷解）。"""
        return bool(
            cand is not None
            and acceptable(cand)
            and (not strict or cand["load_peak_ratio"] <= early_stop_ratio)
        )

    def jump_only(cand):
        """只因跳变超限被拒（其余硬约束都满足）：跳变重搜 jump_box 的触发条件。"""
        return not cand.get("joint_jump_ok", True) and feasible_but_jump(cand)

    def library_entries(lo_s, hi_s, ran):
        """本阶段的种子库种子（队列元素）；ran = 本阶段已运行的种子（参与关节差去重）。"""
        nonlocal lib_used
        if lib_k <= 0:
            return []
        if not lib_target:
            lib_target.append(
                seed_library_target(
                    pin_model, pin_data, pin_maps, seed_library, q_prev, T_target_pin
                )
            )
        if jump_limits is not None:
            # 跳变上限内的样本：|q - q_prev| <= 上限（多圈关节在库里只采了一圈，按 2π 折叠后比较）
            lim = np.asarray(jump_limits, dtype=float)
            lo_s = np.maximum(lo_s, np.where(multi_turn, -np.inf, q_prev - lim))
            hi_s = np.minimum(hi_s, np.where(multi_turn, np.inf, q_prev + lim))
        picked = select_library_seeds(
            seed_library,
            lib_target[0],
            orientation_mode,
            TOOL_AXIS_X if tool_axis is None else tool_axis,
            normalize_cone_half_angle(tool_cone_half_angle),
            char_len,
            lo_s,
            hi_s,
            lib_k,
            exclude=ran,
            min_separation=lib_sep,
        )
        lib_used += len(picked)
        return [(q, None, "library") for q in picked]

    def run_stage(
        stage, queue, lo_s, hi_s, q_prev_seed, locked_done, max_step=None, release_rerun=True,
        posture_settle=True,
    ):
        """按队列运行一个搜索阶段；返回该阶段的停止原因与名额统计。

        max_step 覆盖本阶段 DLS 的单步上限；release_rerun=False 时锁腰种子触发腰部释放后不在本阶段重跑 q_prev；
        posture_settle=False 时本阶段 DLS 不做达标后的姿态整理（v15 续算，见 CONTINUATION_POSTURE_SETTLE）。
        v13：队列里的 LIBRARY_PENDING 占位在排到时展开为种子库种子、COMFORT_CHECKPOINT 为舒适阶段提前
        回退检查点（本阶段还没有可接受候选即结束），两者都不占名额。
        """
        nonlocal unlocked, released, seed_errors, first_seed_error, reach_skips
        stop = "SEED_LIMIT"
        slots = 0                   # 已用名额：运行过的种子 + 被跳过的锁腰种子
        locked_skipped = False
        prev_rerun_queued = False
        prev_info = None            # q_prev 种子的候选信息（v7 顺序下用于跳过锁腰种子 / 旧版早停）
        stage_params = params if max_step is None else {**params, "max_step": float(max_step)}
        if not posture_settle:
            # 停止条件的限位裕度按有效限位计（与候选判据 SAFE_MARGIN 同一组限位；续算框边不是物理限位）
            stage_params = {
                **stage_params, "posture_settle": False, "margin_bounds": (eff_lower, eff_upper)
            }
        n_stage = len(candidates)
        ran = []

        while queue and slots < max_seeds:
            deadline = params.get("deadline")
            if deadline is not None and time.monotonic() >= deadline:
                stop = "TIME_BUDGET"
                break
            seed, fixed, kind = queue.popleft()
            if kind == LIBRARY_PENDING:
                queue.extendleft(reversed(library_entries(lo_s, hi_s, ran)))
                continue
            if kind == COMFORT_CHECKPOINT:
                if not any(acceptable(c) for c in candidates[n_stage:]):
                    stop = "COMFORT_PATIENCE"
                    break
                continue
            if kind == "locked" and locked_unreachable(seed):
                # v13：目标超出锁腰时的手臂可达范围 -> 与 v7 的等价跳过相同处理（占名额、完成锁腰比较）
                slots += 1
                locked_done = True
                locked_skipped = True
                reach_skips += 1
                continue
            ran.append(seed)
            seed_idx = len(per_seed)
            slots += 1
            if fixed is None:
                unlocked += 1
            else:
                locked_done = True

            started = time.monotonic()
            try:
                q, diag = pinocchio_ik_one_seed(
                    pin_model,
                    pin_data,
                    pin_maps,
                    ee_frame_id,
                    T_target_pin,
                    seed,
                    lo_s,
                    hi_s,
                    orientation_mode=orientation_mode,
                    tool_axis=tool_axis,
                    tool_cone_half_angle=tool_cone_half_angle,
                    joint_weights=weights,
                    q_ref=q_ready,
                    include_waist=include_waist,
                    fixed_joint_mask=fixed,
                    waist_release=include_waist,
                    waist_load_release=released,
                    **stage_params,
                )
                q_seed_out = q
                # 在本阶段搜索限位内对齐至最接近当前关节角 q_prev 的合法 2πk 分支
                q = select_closest_bounded_branch(
                    q, q_prev, lo_s, hi_s, multi_turn_mask=multi_turn
                )

                # 限位裕度按有效限位评估（舒适窗不是物理限位）
                info = score_candidate(
                    chain,
                    q,
                    T_target_local,
                    psi_ready,
                    q_prev,
                    q_ready,
                    eff_lower,
                    eff_upper,
                    orientation_mode=orientation_mode,
                    tool_axis=tool_axis,
                    tool_cone_half_angle=tool_cone_half_angle,
                    apply_waist_return=include_waist,
                )
                info.update(
                    source="pinocchio_ik",
                    solve_method="pinocchio_damped_ls_ik",
                    fallback_used=True,
                    seed_used=seed_idx,
                    converged=bool(diag["converged"]),
                )
                jump_max, jump_idx, jump_ok = joint_jump_metrics(
                    q, q_prev, periodic, jump_limits
                )
                info.update(
                    joint_jump_max_rad=jump_max,
                    joint_jump_joint=pin_maps["active_names"][jump_idx],
                    joint_jump_ok=jump_ok,
                )

                if strict:
                    physical = load_physics(
                        pin_model, pin_data, pin_maps, q, context, False
                    )
                    info.update(
                        load_ok=physical["load_ok"],
                        load_peak_ratio=physical["peak_ratio"],
                        load_peak_joint=physical["peak_joint"],
                    )
                    q_bytes = q.tobytes()
                    context.setdefault("_physics_cache", {})[q_bytes] = physical
                    if np.array_equal(q, q_seed_out):
                        context.setdefault("_task_cache", {})[(q_bytes, id(T_target_pin))] = (
                            float(diag["final_pos_err"]),
                            float(diag["final_ori_violation"]),
                        )
                    if (
                        fixed is not None
                        and physical["peak_ratio"] > context["policy"].soft_ratio
                    ):
                        released = True

                # 候选去重使用考虑合法 2πk 等价分支的关节差
                duplicate = any(
                    np.linalg.norm(
                        bounded_joint_difference(
                            q, old, eff_lower, eff_upper, multi_turn_mask=multi_turn
                        )
                    )
                    < 1e-5
                    for old in unique
                )
                if not duplicate:
                    unique.append(q.copy())
                    candidates.append(info)

                per_seed.append(
                    {
                        "seed_idx": seed_idx,
                        "seed_kind": kind,
                        "search_stage": stage,
                        **diag,
                        "fixed_waist": fixed is not None,
                        "pose_ok": bool(info["pose_ok"]),
                        "margin": float(info["margin"]),
                        "joint_jump_max_rad": jump_max,
                        "joint_jump_ok": jump_ok,
                        "duplicate": duplicate,
                        "elapsed_ms": (time.monotonic() - started) * 1000.0,
                    }
                )
                if stage == "comfort" and not diag["converged"]:
                    merit = 0.5 * (
                        (float(diag["final_pos_err"]) / char_len) ** 2
                        + float(diag["final_ori_violation"]) ** 2
                    )
                    if math.isfinite(merit):
                        edge_pool.append((merit, seed_idx, q.copy()))

                if kind == "q_prev":
                    prev_info = info
                    # v7：q_prev 的解已合格且腰部基本在 q_ready -> 锁腰种子与之等价，跳过
                    if q_prev_first and not locked_done and qualified(info):
                        waist_dev = float(
                            np.max(np.abs(q[waist_mask] - q_ready[waist_mask]))
                        )
                        if waist_dev <= skip_tol:
                            queue = deque(e for e in queue if e[2] != "locked")
                            slots += 1
                            locked_done = True
                            locked_skipped = True
                elif (
                    kind == "locked"
                    and q_prev_first
                    and released
                    and release_rerun
                    and not prev_rerun_queued
                ):
                    # v7：锁腰种子触发腰部释放 -> 下一个种子用释放状态重跑 q_prev（即 v6 的第二个种子）
                    queue.appendleft((q_prev_seed, None, "q_prev_release"))
                    prev_rerun_queued = True

                if strict:
                    low = [c for c in candidates if qualified(c)]
                    if (
                        slots >= min_seeds
                        and (
                            not include_waist
                            or (unlocked >= min_unlocked and locked_done)
                        )
                        and low
                        and min(c["load_peak_ratio"] for c in low)
                        <= early_stop_ratio
                    ):
                        stop = "LOW_LOAD_AFTER_REQUIRED_COMPARISON"
                        break
                elif (
                    slots >= 2
                    and locked_done
                    and qualified(prev_info if kind == "locked" else info)
                ):
                    # 旧版规则：至少比较 2 个种子后，以 q_prev 及之后种子的合格解早停（锁腰解本身不触发）
                    stop = "LEGACY_POSE_EARLY_STOP"
                    break
            except Exception as exc:
                # 种子异常不中断搜索，但必须可见：计数并保留首个异常，失败时写入失败码详情
                seed_errors += 1
                if first_seed_error is None:
                    first_seed_error = f"{type(exc).__name__}: {exc}"
                per_seed.append(
                    {
                        "seed_idx": seed_idx,
                        "seed_kind": kind,
                        "search_stage": stage,
                        "converged": False,
                        "fixed_waist": fixed is not None,
                        "exception": f"{type(exc).__name__}: {exc}",
                        "stop_reason": "VISIBLE_SEED_ERROR",
                        "elapsed_ms": (time.monotonic() - started) * 1000.0,
                    }
                )
        return {
            "name": stage,
            "stop": stop,
            "slots": slots,
            "locked_skipped": locked_skipped,
        }

    # ---- v9：跳变上限前移到搜索 ------------------------------------------------------
    # joint_jump_policy="reject" 且显式上限 < π 的关节：种子与迭代都限制在 q_prev ± 上限内（与有效限位 /
    # 舒适窗求交；交集为空的关节不加）。否则 DLS 会在零空间里漂到上限之外、再被跳变过滤整批剔除，
    # q_prev 不再更新，后续路点全部失败。框内没有可接受解时，按 v8 不加框重搜一次（见下方 "unboxed"）。
    box_half = None
    if jump_limits is not None and search.get("joint_jump_policy", "reject") == "reject":
        limits = np.asarray(jump_limits, dtype=float)
        limits = np.where(limits < np.pi, limits, np.inf)
        if np.any(np.isfinite(limits)):
            box_half = limits
    rng_unboxed = copy.deepcopy(rng) if box_half is not None else None
    # v13 跳变重搜与 unboxed 一样使用常规阶段之前的随机数状态（两者互斥：一个要求有框、一个要求无框）
    rng_rescue = (
        copy.deepcopy(rng)
        if box_half is None and jump_limits is not None and search.get("jump_rescue", False)
        else None
    )

    def boxed(lo, hi, half):
        if half is None:
            return lo, hi
        b_lo = np.maximum(lo, q_prev - half)
        b_hi = np.minimum(hi, q_prev + half)
        empty = b_lo >= b_hi
        return np.where(empty, lo, b_lo), np.where(empty, hi, b_hi)

    def with_library(entries, n_lead, seeds_s, lo_s, hi_s, checkpoint=False):
        """v13：在前 n_lead 个固定种子（q_prev / 锁腰 / comfort_edge）与 q_ready（若与 q_prev 不同）之后
        插入种子库占位（make_scatter_seeds 去重后 q_ready 若存在必在第 2 位）；checkpoint=True 时其后
        再放舒适阶段提前回退检查点（没有种子库时放在第 patience 个名额之后）。"""
        pos = n_lead
        if len(seeds_s) > 1 and np.array_equal(
            seeds_s[1], clip_to_bounds(q_ready, lo_s, hi_s)
        ):
            pos += 1
        marks = [(None, None, LIBRARY_PENDING)] if lib_k > 0 else []
        if checkpoint:
            if lib_k <= 0:
                pos = patience
            marks.append((None, None, COMFORT_CHECKPOINT))
        return entries[:pos] + marks + entries[pos:]

    def regular_queue(lo_s, hi_s, rng_s, *, jump_box=False, checkpoint=False):
        """常规阶段的种子队列 q_prev -> 锁腰 -> q_ready -> [种子库] -> 随机扰动；返回 (q_prev 种子, 队列)。

        队列元素 (种子, 固定关节掩码, 种子类别)；make_scatter_seeds 保证第一个是 q_prev。锁腰种子 = q_prev 且
        腰部锁在 q_ready：v6 顺序（q_prev_first=False）排在最前。jump_box=True（跳变重搜）时锁腰种子恒排第 2、
        腰部取值钳到框内（q_prev ± 跳变上限不一定含 q_ready 的腰部）。
        """
        seeds_s = make_scatter_seeds(
            q_prev, q_ready, lo_s, hi_s, rng_s, max_count=max_seeds
        )
        entries_s = [
            (seed, None, "q_prev" if i == 0 else "scatter")
            for i, seed in enumerate(seeds_s)
        ]
        if include_waist:
            locked_s = clip_to_bounds(q_prev.copy(), lo_s, hi_s)
            if jump_box:
                locked_s[waist_mask] = np.minimum(
                    np.maximum(q_ready[waist_mask], lo_s[waist_mask]), hi_s[waist_mask]
                )
                pos = 1
            else:
                locked_s[waist_mask] = q_ready[waist_mask]
                pos = 1 if q_prev_first else 0
            entries_s.insert(pos, (locked_s, waist_mask, "locked"))
        entries_s = with_library(
            entries_s, 2 if include_waist else 1, seeds_s, lo_s, hi_s, checkpoint=checkpoint
        )
        return seeds_s[0], deque(entries_s)

    stages = []
    # ---- v9：续算阶段（局部优先）----------------------------------------------------
    # 只用与 q_prev 连续的候选，避免在“锁腰（腰回 q_ready）”与“随动”两个分支间来回切换：
    #   随动续算：种子 q_prev，DLS 限制在 q_prev ± continuation_max_step（∩ 有效限位 ∩ 跳变框）内，
    #            零空间整理不会一步漂到远处的姿态；单步上限取 min(max_step, 0.5 × continuation_max_step)；
    #   锁腰续算：腰部锁在 q_ready、其余从 q_prev 出发（与常规锁腰种子相同），只有单关节步长
    #            <= continuation_max_step 时才算局部（q_prev 腰部离 q_ready 远时它就是分支切换）；
    #            锁腰解负荷超 soft_ratio 时释放腰部、在续算框内重跑随动续算。
    # 局部可接受候选（满足全部硬约束且步长 <= continuation_max_step）按常规择优规则比较；
    # 没有时进入常规搜索（与 v8 相同），续算阶段的候选留在候选池中。
    # v15（CONTINUATION_POSTURE_SETTLE = False）：续算 DLS 不做达标后的姿态整理；随动续算的解就是 q_prev 本身
    # 且可接受（目标不动、q_prev 已是解）时原样返回，不跑锁腰续算（continuation_hold）。
    hold_used = False
    if local_first:
        n_before = len(candidates)
        released_before = released
        settle_c = bool(CONTINUATION_POSTURE_SETTLE)
        half_a = np.full(len(q_prev), cont_step)
        if box_half is not None:
            half_a = np.minimum(half_a, box_half)
        lo_a, hi_a = boxed(eff_lower, eff_upper, half_a)
        seed_a = clip_to_bounds(q_prev, lo_a, hi_a)
        step_a = min(
            float(params.get("max_step", PIN_IK_MAX_STEP)),
            CONTINUATION_DLS_STEP_FRACTION * cont_step,
        )
        stages.append(
            run_stage(
                "continuation", deque([(seed_a, None, "q_prev")]), lo_a, hi_a,
                seed_a, True, max_step=step_a, posture_settle=settle_c,
            )
        )
        if not settle_c:
            fresh = candidates[n_before:]
            hold_used = bool(
                fresh and acceptable(fresh[0]) and np.array_equal(fresh[0]["q"], q_prev)
            )
        # 锁腰续算：腰部离 q_ready 不超过续算步长时才可能是局部解（否则只多花一次 DLS，常规搜索也会跑锁腰种子）。
        # 不按 v7 规则用“随动解腰部 ≈ q_ready”跳过：腰部等价不代表手臂等价（零空间整理可能已把手臂推到
        # 续算框边上，而锁腰解只需跟着目标小步移动），跳过会让轨迹离开锁腰分支、之后再也回不来
        if not hold_used and include_waist and np.any(waist_mask) and (
            float(np.max(np.abs(q_prev[waist_mask] - q_ready[waist_mask])))
            <= cont_step + 1e-9
        ):
            lo_b, hi_b = boxed(eff_lower, eff_upper, box_half)
            locked_b = clip_to_bounds(q_prev.copy(), lo_b, hi_b)
            locked_b[waist_mask] = q_ready[waist_mask]
            stages.append(
                run_stage(
                    "continuation", deque([(locked_b, waist_mask, "locked")]),
                    lo_b, hi_b, seed_a, True, release_rerun=False, posture_settle=settle_c,
                )
            )
            if released and not released_before:
                # 锁腰续算负荷超 soft_ratio -> 释放腰部，在续算框内重跑随动续算（对应常规的 q_prev_release）
                stages.append(
                    run_stage(
                        "continuation", deque([(seed_a, None, "q_prev_release")]),
                        lo_a, hi_a, seed_a, True, max_step=step_a, posture_settle=settle_c,
                    )
                )
        local = [
            c
            for c in candidates[n_before:]
            # 与 joint_jump_metrics 相同的 1e-9 容差：贴在续算框边上的解（q - q_prev 舍入后略大于框）仍算局部
            if acceptable(c) and c["joint_jump_max_rad"] <= cont_step + 1e-9
        ]
        if local:
            local_used, local_pick = True, local
            stages[-1]["stop"] = "CONTINUATION_LOCAL"
        else:
            released = released_before  # 回到常规搜索：腰部释放状态从头判定（与 v8 相同）

    # ---- 常规搜索阶段 1（comfort 舒适窗；未启用舒适窗时直接跑 full 阶段）-----------
    lo1, hi1 = comfort if comfort is not None else (eff_lower, eff_upper)
    lo1, hi1 = boxed(lo1, hi1, box_half)
    # 续算局部解已采用时也生成队列：随机数消耗与之前的版本一致
    first1, queue1 = regular_queue(
        lo1, hi1, rng, checkpoint=early_fallback and comfort is not None
    )
    if not local_used:
        stages.append(
            run_stage(
                "comfort" if comfort is not None else "full",
                queue1, lo1, hi1, first1, not include_waist,
            )
        )

    rescue_allowed = bool(
        search.get("jump_rescue", False)
        and box_half is None
        and jump_limits is not None
        and search.get("joint_jump_policy", "reject") == "reject"
        and not local_used
    )

    # ---- 常规搜索阶段 2（full / jump_box：舒适窗内无可接受解时放开到有效限位）------
    comfort_ok = comfort is not None and any(acceptable(c) for c in candidates)
    fallback = comfort is not None and not comfort_ok and not local_used
    # v13：舒适阶段已出现“只因跳变超限被拒”的解 -> 放开阶段直接在跳变框内搜索（阶段名 "jump_box"）。
    # 框外的解必被跳变过滤拒绝，不加框的放开阶段多半只会再收敛到一批超限解（回放基准里这类目标
    # 先跑满 10 个放开种子、再跳变重搜，单目标 300~500 ms）
    early_rescue = bool(
        fallback and rescue_allowed and any(jump_only(c) for c in candidates)
    )
    if fallback:
        half2 = np.asarray(jump_limits, dtype=float) if early_rescue else box_half
        lo2, hi2 = boxed(eff_lower, eff_upper, half2)
        seeds_full = make_scatter_seeds(
            q_prev, q_ready, lo2, hi2, rng_full, max_count=max_seeds
        )
        entries_full = [(seeds_full[0], None, "q_prev")]
        for _, _, q_edge in sorted(edge_pool, key=lambda e: (e[0], e[1]))[
            :WAIST_COMFORT_EDGE_SEEDS
        ]:
            if early_rescue:
                q_edge = clip_to_bounds(q_edge, lo2, hi2)
            entries_full.append((q_edge, None, "comfort_edge"))
        n_lead = len(entries_full)
        entries_full.extend((seed, None, "scatter") for seed in seeds_full[1:])
        entries_full = with_library(entries_full, n_lead, seeds_full, lo2, hi2)
        stages.append(
            run_stage(
                "jump_box" if early_rescue else "full",
                deque(entries_full), lo2, hi2, seeds_full[0], True,
            )
        )
    # ---- v9：跳变框内没有可接受解 -> 按 v8 不加框重搜一次 -------------------------------
    # 既保证 v8 能解的目标 v9 仍能解，也让“只有超限解”的目标照旧报 JOINT_JUMP_LIMIT（而不是位姿不可达）
    unboxed = bool(
        box_half is not None
        and not local_used
        and not any(acceptable(c) for c in candidates)
    )
    if unboxed:
        first_u, queue_u = regular_queue(eff_lower, eff_upper, rng_unboxed)
        stages.append(
            run_stage(
                "unboxed", queue_u, eff_lower, eff_upper, first_u, not include_waist
            )
        )
    # ---- v13：只因跳变超限而失败 -> 在跳变框内重搜一次（放开阶段已在框内搜过则不再重搜）------------
    rescue = bool(
        rescue_allowed
        and not early_rescue
        and not any(acceptable(c) for c in candidates)
        and any(jump_only(c) for c in candidates)
    )
    if rescue:
        lo_r, hi_r = boxed(eff_lower, eff_upper, np.asarray(jump_limits, dtype=float))
        first_r, queue_r = regular_queue(lo_r, hi_r, rng_rescue, jump_box=True)
        stages.append(
            run_stage(
                "jump_box", queue_r, lo_r, hi_r, first_r, not include_waist
            )
        )
    if local_used:
        # 续算局部解：候选池只保留局部可接受候选（离 q_prev 远的候选不参与择优，避免分支切换）
        candidates[:] = local_pick
    main_stages = [
        st for st in stages if st["name"] not in ("continuation", "unboxed", "jump_box")
    ]

    if comfort_info is not None:
        comfort_info.update(
            regime=regime if comfort_enabled else None,
            applied=comfort is not None,
            fallback_to_full_bounds=bool(fallback),
            continuation_regime=regime,
            continuation_local_used=local_used,
            continuation_hold=hold_used,
        )

    if strict:
        context["search_diagnostics"] = {
            "stop_reason": stages[-1]["stop"],
            "unlocked_seeds_attempted": unlocked,
            "waist_released_for_load": released,
            "seeds_attempted": len(per_seed),
            "seed_slots_used": sum(st["slots"] for st in stages),
            "seed_order": (
                "q_prev_first"
                if q_prev_first
                else ("locked_first" if include_waist else "no_waist")
            ),
            "locked_seed_skipped": bool(main_stages and main_stages[0]["locked_skipped"]),
            "continuation_local_used": local_used,
            "continuation_hold": hold_used,
            "jump_limit_search_box": box_half is not None,
            "jump_limit_unboxed_retry": unboxed,
            "q_prev_release_rerun": any(
                p.get("seed_kind") == "q_prev_release" for p in per_seed
            ),
            "seed_errors": seed_errors,
            "first_seed_error": first_seed_error,
            "library_seeds_queued": lib_used,
            "library_skipped_continuation": lib_skip_continuation,
            "locked_seed_unreachable": reach_skips,
            "jump_limit_rescue": rescue or early_rescue,
            "jump_limit_rescue_early": early_rescue,
            "comfort_patience_stop": any(st["stop"] == "COMFORT_PATIENCE" for st in stages),
            "candidates": len(candidates),
            "waist_comfort": {
                "enabled": comfort is not None,
                "regime": regime,
                "comfort_stop_reason": main_stages[0]["stop"] if comfort is not None else None,
                "comfort_solution_found": bool(comfort_ok),
                "fallback_to_full_bounds": bool(fallback),
            },
        }

    sorted_candidates = sorted(candidates, key=candidate_sort_key)
    return (
        candidates,
        [c["q"] for c in sorted_candidates[:PIN_TOP_K_FOR_DIRECT]],
        per_seed,
    )



# =============================================================================
# Standalone solver class
# =============================================================================
class HAArmIKSolver:
    def __init__(
        self,
        urdf,
        side,
        use_ssik=USE_SSIK,
        use_pinocchio_ik=USE_PINOCCHIO_IK,
        use_torque_cost=USE_TORQUE_COST,
        use_direct_psi=USE_DIRECT_PSI,
        include_waist=False,
        verbose=True,
        config=None,
    ):
        self.verbose = verbose
        self.include_waist = bool(include_waist)

        cfg_tune = dict(config) if config else {}
        self._load_config = cfg_tune.copy()
        self.load_mode = str(cfg_tune.get("load_mode", "strict"))
        if self.load_mode not in ("strict", "off"):
            raise ValueError("load_mode must be strict or explicitly off")

        self._solve_lock = threading.Lock()
        self._latest_joint_state = None
        self._load_context = None
        self._solve_T_root_base = None
        self._load_deadline = None
        self.last_load_diagnostics = {
            "version": SOLVER_VERSION,
            "accepted": False,
            "mode": self.load_mode,
            "code": "NOT_RUN",
        }

        self.max_direct_nfev = cfg_tune.get(
            "max_direct_nfev", PORTFOLIO_DIRECT_MAX_NFEV
        )
        self.direct_pos_tol = cfg_tune.get(
            "direct_pos_tol", PORTFOLIO_DIRECT_POS_TOL
        )
        self.direct_ori_tol = cfg_tune.get(
            "direct_ori_tol", PORTFOLIO_DIRECT_ORI_TOL
        )
        self.direct_psi_tol = cfg_tune.get(
            "direct_psi_tol", PORTFOLIO_DIRECT_PSI_TOL
        )
        self.portfolio_q_ref_mode = cfg_tune.get(
            "portfolio_q_ref_mode", PORTFOLIO_Q_REF_MODE
        )
        self.use_local_seeds = cfg_tune.get("use_local_seeds", USE_LOCAL_SEEDS)
        self.dls_max_iter = cfg_tune.get("dls_max_iter", PIN_IK_MAX_ITER)
        self.dls_damping = cfg_tune.get("dls_damping", PIN_IK_DAMPING)
        self.dls_max_step = cfg_tune.get("dls_max_step", PIN_IK_MAX_STEP)

        # 可选的零空间、物理负载与逐关节虚拟搜索窗参数
        self.dls_nullspace_gain = cfg_tune.get("dls_nullspace_gain", None)
        self.dls_waist_null_gain = cfg_tune.get("dls_waist_null_gain", None)
        self.dls_waist_task_weight = cfg_tune.get(
            "dls_waist_task_weight", None
        )
        self.dls_waist_task_weight_far = cfg_tune.get(
            "dls_waist_task_weight_far", None
        )
        self.dls_gravity_torque_gain = cfg_tune.get(
            "dls_gravity_torque_gain", None
        )
        self.dls_relative_gravity_gain = cfg_tune.get(
            "dls_relative_gravity_gain", None
        )
        self.dls_com_xy_gain = cfg_tune.get("dls_com_xy_gain", None)
        self.dls_joint_limit_gain = cfg_tune.get("dls_joint_limit_gain", None)
        self.dls_virtual_half_windows = cfg_tune.get(
            "dls_virtual_half_windows", None
        )
        self.dls_posture_use_equivalent_branch = bool(
            cfg_tune.get("dls_posture_use_equivalent_branch", False)
        )
        # v8：腰部舒适窗（偏好，非硬约束）。"auto" = 由 q_ready 处静态重力矩推导（默认）；
        # None / False / "off" = 关闭（v7 行为）；{关节名: 半窗 rad} = 以 q_ready 为中心显式指定
        self.dls_waist_comfort = cfg_tune.get("dls_waist_comfort", "auto")
        # 舒适窗只用于“重新定位”：目标相对 FK(q_prev) 的位置差 > pos_m 或姿态差 > ori_rad；
        # None = 默认 (0.1 × 臂展, 0.1 rad)。小位移续算（路点跟踪）不加舒适窗，保持 v7 的连续性
        tol = cfg_tune.get("dls_waist_comfort_continuation_tol")
        if tol is not None:
            try:
                tol = tuple(float(v) for v in tol)
            except (TypeError, ValueError):
                tol = ()
            if len(tol) != 2 or not all(math.isfinite(v) and v >= 0.0 for v in tol):
                raise ValueError(
                    "dls_waist_comfort_continuation_tol must be (pos_m, ori_rad) >= 0"
                )
        self.dls_waist_comfort_continuation_tol = tol
        # 续算局部优先（见 CONTINUATION_LOCAL）；续算判据与舒适窗共用 dls_waist_comfort_continuation_tol
        self.dls_continuation_local = _config_flag(
            cfg_tune, "dls_continuation_local", CONTINUATION_LOCAL
        )
        self.dls_continuation_max_step = _finite_angle(
            cfg_tune.get("dls_continuation_max_step", CONTINUATION_MAX_STEP),
            "dls_continuation_max_step must be a positive finite angle",
            positive=True,
        )
        # 任务空间种子库（初始化时建库，见 solver_dls.build_seed_library；None / False / 0 = 关闭）
        size = cfg_tune.get("dls_seed_library_size", SEED_LIBRARY_SIZE)
        self.dls_seed_library_size = _int_at_least(
            0 if size is None or size is False else size,
            0,
            "dls_seed_library_size must be a nonnegative integer (0 = off)",
        )
        self.dls_seed_library_seeds = _int_at_least(
            cfg_tune.get("dls_seed_library_seeds", SEED_LIBRARY_SEEDS),
            0,
            "dls_seed_library_seeds must be a nonnegative integer",
        )
        self.dls_seed_library_min_separation = _finite_angle(
            cfg_tune.get("dls_seed_library_min_separation", SEED_LIBRARY_MIN_SEPARATION),
            "dls_seed_library_min_separation must be a finite angle >= 0",
            positive=False,
        )
        # 舒适阶段提前回退与耐心、跳变重搜、锁腰种子可达性检查、续算类求解是否用种子库
        # （校验顺序与之前的版本相同：多项同时出错时报同一项）
        self.dls_waist_comfort_early_fallback = _config_flag(
            cfg_tune, "dls_waist_comfort_early_fallback", WAIST_COMFORT_EARLY_FALLBACK
        )
        self.dls_waist_comfort_patience = _int_at_least(
            cfg_tune.get("dls_waist_comfort_patience", WAIST_COMFORT_PATIENCE),
            1,
            "dls_waist_comfort_patience must be an integer >= 1",
        )
        self.dls_jump_limit_rescue = _config_flag(
            cfg_tune, "dls_jump_limit_rescue", JUMP_LIMIT_RESCUE
        )
        self.dls_locked_reach_check = _config_flag(
            cfg_tune, "dls_locked_reach_check", LOCKED_REACH_CHECK
        )
        self.dls_seed_library_continuation = _config_flag(
            cfg_tune, "dls_seed_library_continuation", SEED_LIBRARY_CONTINUATION
        )
        # 主任务位置误差的特征长度（m）：DLS 与 strict 最终复核同一口径
        self.dls_characteristic_length = cfg_tune.get("dls_characteristic_length", 0.5)
        self.seed_library = None
        self.dls_locked_reach = None

        self.urdf = str(Path(urdf).expanduser().resolve())
        if not Path(self.urdf).is_file():
            raise FileNotFoundError(f"URDF not found: {self.urdf}")
        self.urdf_root_link = get_urdf_root_link(self.urdf)
        self.side = side.lower()
        if self.side not in ARM_CONFIGS:
            raise ValueError("side must be 'left' or 'right'")

        cfg = ARM_CONFIGS[self.side]

        if self.include_waist:
            self.base_link = self.urdf_root_link
            use_ssik = False
            use_direct_psi = False
            if verbose:
                print(
                    "[waist-ik] include_waist=True: 9-DOF chain, "
                    "SSIK and direct_psi disabled."
                )
        else:
            self.base_link = cfg["base_link"]

        self.ee_link = cfg["ee_link"]
        self.s_joint = cfg["s_joint"]
        self.e_joint = cfg["e_joint"]
        self.w_joint = cfg["w_joint"]
        self.ref_dir = REF_DIR.copy()

        mount_from_urdf = get_fixed_joint_origin(
            self.urdf,
            parent_link="world",
            child_link=self.base_link,
        )

        if mount_from_urdf is not None:
            self.urdf_has_world_mount = True
            self.mount_xyz, self.mount_rpy = mount_from_urdf
        else:
            self.urdf_has_world_mount = False
            self.mount_xyz = cfg["mount_xyz"]
            self.mount_rpy = cfg["mount_rpy"]
        # Pinocchio 模型的根是 URDF 根。URDF 带 world 安装关节、或求解基座不是 URDF 根时，
        # T_world_from_base 取安装位姿，DLS 的目标与重力都按 URDF 根（"world"）坐标系给出；
        # 否则求解基座就是根，按 local 坐标系给出
        self._pin_uses_root_frame = bool(
            self.urdf_has_world_mount or self.urdf_root_link != self.base_link
        )

        self.use_ssik = bool(use_ssik and ssik is not None)
        self.use_pinocchio_ik = bool(use_pinocchio_ik)
        self.use_torque_cost = bool(use_torque_cost)
        self.use_direct_psi = bool(use_direct_psi)

        if use_ssik and ssik is None:
            print("[WARN] ssik import failed. SSIK disabled.")

        if (self.use_pinocchio_ik or self.use_torque_cost) and pin is None:
            print(
                "[WARN] pinocchio import failed. "
                "Pinocchio IK and torque cost disabled."
            )
            self.use_pinocchio_ik = False
            self.use_torque_cost = False

        self.mount_quat = tuple(
            float(v) for v in R.from_matrix(rpy_to_R(self.mount_rpy)).as_quat()
        )
        if self._pin_uses_root_frame:
            self.T_world_from_base = T_origin(
                self.mount_xyz,
                self.mount_rpy,
            )
        else:
            self.T_world_from_base = np.eye(4)

        self.T_base_from_world = invert_T(self.T_world_from_base)

        R_world_from_base = rpy_to_R(self.mount_rpy)
        self.g_mounted_base = R_world_from_base.T @ G_WORLD
        self.chain = SerialChain(
            self.urdf,
            self.base_link,
            self.ee_link,
            s_joint=self.s_joint,
            e_joint=self.e_joint,
            w_joint=self.w_joint,
        )

        self.workspace_anchor_joint = self.s_joint
        self.workspace_anchor_local_xyz = joint_origin_position_at_zero(
            self.chain,
            self.workspace_anchor_joint,
        )
        self.workspace_max_j7_radius_from_anchor = (
            estimate_chain_max_reach_from_joint(
                self.chain,
                self.workspace_anchor_joint,
            )
        )
        self.workspace_max_radius = self.workspace_max_j7_radius_from_anchor
        # 基座到末端的理论最大伸展长度（所有关节平移绝对模长之和，三角不等式绝对上界）
        self.chain_max_reach = float(
            sum(np.linalg.norm(j.get("xyz", [0.0, 0.0, 0.0])) for j in self.chain.joints)
        )

        self.lower, self.upper = self.chain.bounds()
        bounds_override = dict(cfg_tune.get("position_bounds", {}))
        active_names = [j["name"] for j in self.chain.active_joints]
        for i, joint in enumerate(self.chain.active_joints):
            if joint["type"] == "continuous":
                self.lower[i], self.upper[i] = -np.inf, np.inf
            if joint["name"] in bounds_override:
                b = finite_vector(
                    bounds_override[joint["name"]], 2, "position override"
                )
                if (
                    b[0] >= b[1]
                    or b[0] < self.lower[i]
                    or b[1] > self.upper[i]
                ):
                    raise ValueError(
                        "position override must narrow actual URDF bounds"
                    )
                self.lower[i], self.upper[i] = b

        raw_ready = (
            np.concatenate([np.zeros(2, dtype=float), cfg["q_ready"]])
            if self.include_waist
            else np.asarray(cfg["q_ready"], dtype=float)
        )
        self.q_ready = finite_vector(
            cfg_tune.get("q_ready", raw_ready), len(active_names), "q_ready"
        )
        if self.load_mode == "strict":
            if np.any(self.q_ready < self.lower) or np.any(
                self.q_ready > self.upper
            ):
                raise ValueError(
                    "q_ready outside actual/planning bounds; not clipped in strict mode"
                )
        else:
            self.q_ready = clip_to_bounds(self.q_ready, self.lower, self.upper)
        self.q_prev = self.q_ready.copy()

        self.active_joint_names = active_names
        # 基于 URDF 关节类型显式构造多圈旋转掩码（continuous，或跨度 >= 2π 的 revolute；prismatic 一律 False）。
        # 有 Pinocchio 模型时下面改用 pin_maps["active_multi_turn"]（同一判定，限位含 position_bounds）
        self.multi_turn_mask = multi_turn_mask_from_chain(self.chain, self.lower, self.upper)

        # H4 不绕大圈：单关节相对 q_prev 的最大允许转动量（详见 _resolve_joint_jump_limits）
        self.joint_jump_policy = str(cfg_tune.get("joint_jump_policy", "reject"))
        if self.joint_jump_policy not in JOINT_JUMP_POLICIES:
            raise ValueError(f"joint_jump_policy must be one of {JOINT_JUMP_POLICIES}")
        self.max_joint_jump = self._resolve_joint_jump_limits(
            cfg_tune.get("max_joint_jump_rad", MAX_JOINT_JUMP_RAD)
        )
        # v7：q_prev 先行 + 等价锁腰种子跳过的腰部容差（rad）；None = v6 种子顺序
        skip_tol = cfg_tune.get("locked_seed_skip_tol", LOCKED_SEED_SKIP_TOL)
        if skip_tol is not None:
            skip_tol = float(skip_tol)
            if not math.isfinite(skip_tol) or skip_tol < 0.0:
                raise ValueError(
                    "locked_seed_skip_tol must be None or a finite nonnegative angle"
                )
        self.locked_seed_skip_tol = skip_tol

        # v9：q_ready 处臂角无定义（肘部伸直 / S->W 与参考方向平行 / S、E、W 关节名不在链上）时
        # psi_ready = NaN：ik_candidate 的臂角项不参与打分（其余代价照算），并给出可见告警
        self.psi_ready_error = None
        try:
            self.psi_ready = float(self.chain.psi(self.q_ready))
        except Exception as exc:
            self.psi_ready = float("nan")
            self.psi_ready_error = f"{type(exc).__name__}: {exc}"
        if not math.isfinite(self.psi_ready):
            self.psi_ready = float("nan")
            self.psi_ready_error = self.psi_ready_error or "psi(q_ready) is not finite"
            warnings.warn(
                "psi(q_ready) undefined; the arm-angle term is excluded from candidate "
                f"scoring ({self.psi_ready_error})",
                RuntimeWarning,
                stacklevel=2,
            )

        # ---- SSIK ----
        self.arm = None
        if self.use_ssik:
            self.arm = ssik.Manipulator.from_urdf(
                self.urdf,
                base=self.base_link,
                ee=self.ee_link,
            )

        # ---- Pinocchio (shared for IK + torque) ----
        self.pin_model = None
        self.pin_data = None
        self.pin_maps = None
        self.pin_ee_frame_id = None
        self.joint_efforts_active = None
        if self.use_pinocchio_ik or self.use_torque_cost:
            require_pinocchio()

            self.pin_model, self.pin_data = load_pinocchio_model(self.urdf)
            g = G_WORLD if self._pin_uses_root_frame else self.g_mounted_base
            gravity_root = finite_vector(
                cfg_tune.get("gravity_root", g), 3, "gravity_root"
            )
            if np.linalg.norm(gravity_root) < 1e-9:
                raise ValueError(
                    "gravity_root must be nonzero; units m/s^2 in URDF-root frame"
                )
            self.pin_model.gravity.linear = gravity_root

            self.pin_maps = build_pinocchio_joint_maps(
                self.pin_model,
                self.active_joint_names,
            )
            configure_model_metadata(
                self.pin_model,
                self.pin_maps,
                Path(self.urdf).read_text(encoding="utf-8"),
                massless_links=cfg_tune.get("load_massless_links", ()),
                position_bounds=cfg_tune.get("position_bounds"),
            )
            self.lower = self.pin_maps["active_lower"].copy()
            self.upper = self.pin_maps["active_upper"].copy()
            self.multi_turn_mask = self.pin_maps["active_multi_turn"].copy()
            self.joint_efforts_active = get_active_joint_efforts(self.pin_maps)

        # 预计算有效搜索限位（若显式配置了 dls_virtual_half_windows，则全流程统一遵守该窗口）
        self.search_lower, self.search_upper = _build_effective_bounds(
            self.lower,
            self.upper,
            self.q_ready,
            virtual_half_windows=self.dls_virtual_half_windows,
            active_names=self.active_joint_names,
        )

        self.pin_base_frame_id = None
        self.dls_waist_max_reach = None
        self.dls_arm_max_reach = None
        self.waist_comfort_lower = None
        self.waist_comfort_upper = None
        self.waist_comfort_info = {}
        if self.pin_model is not None:
            self.pin_ee_frame_id = get_pinocchio_frame_id(
                self.pin_model,
                self.ee_link,
            )
            # strict 求解时读取 base_link 位姿用；初始化时缓存（按 BODY 类型查询，见 get_pinocchio_frame_id）
            try:
                self.pin_base_frame_id = get_pinocchio_frame_id(
                    self.pin_model, self.base_link
                )
            except Exception:
                self.pin_base_frame_id = None
            # 臂展：配置的 s_joint 原点 -> TCP，沿运动链逐段累加（续算判据的位置尺度；include_waist 时
            # 同时是腰部按距离释放所用的臂展）
            try:
                self.dls_arm_max_reach = estimate_arm_max_reach(
                    self.pin_model, self.pin_maps, self.pin_ee_frame_id, self.s_joint
                )
            except Exception:
                if self.include_waist:
                    raise
                self.dls_arm_max_reach = None
            if self.include_waist:
                self.dls_waist_max_reach = self.dls_arm_max_reach
                self._configure_waist_comfort(cfg_tune)
            if self.include_waist and self.dls_locked_reach_check:
                self.dls_locked_reach = locked_arm_reach_bound(
                    self.pin_model, self.pin_maps, self.pin_ee_frame_id
                )
            if self.use_pinocchio_ik and self.dls_seed_library_size > 0:
                self.seed_library = self._shared_seed_library()

        self._check_model()

        if self.verbose:
            self.print_info()

    def _shared_seed_library(self):
        """v13：建种子库；同一模型（URDF 内容、活动关节、末端帧、有效限位、q_ready、舒适窗、库大小）的实例共享。"""
        comfort = (
            None
            if self.waist_comfort_lower is None
            else (self.waist_comfort_lower, self.waist_comfort_upper)
        )
        key = (
            hashlib.sha256(Path(self.urdf).read_bytes()).hexdigest(),
            tuple(self.pin_maps["active_names"]),
            int(self.pin_ee_frame_id),
            np.asarray(self.search_lower, dtype=float).tobytes(),
            np.asarray(self.search_upper, dtype=float).tobytes(),
            np.asarray(self.q_ready, dtype=float).tobytes(),
            None if comfort is None else tuple(np.asarray(b, dtype=float).tobytes() for b in comfort),
            int(self.dls_seed_library_size),
            int(SEED_LIBRARY_RNG_SEED),
        )
        with _SEED_LIBRARY_CACHE_LOCK:
            library = _SEED_LIBRARY_CACHE.get(key)
        if library is not None:
            return library
        library = build_seed_library(
            self.pin_model,
            self.pin_data,
            self.pin_maps,
            self.pin_ee_frame_id,
            self.search_lower,
            self.search_upper,
            self.q_ready,
            self.dls_seed_library_size,
            comfort_bounds=comfort,
            rng_seed=SEED_LIBRARY_RNG_SEED,
        )
        with _SEED_LIBRARY_CACHE_LOCK:
            while len(_SEED_LIBRARY_CACHE) >= _SEED_LIBRARY_CACHE_MAX:
                _SEED_LIBRARY_CACHE.pop(next(iter(_SEED_LIBRARY_CACHE)))
            library = _SEED_LIBRARY_CACHE.setdefault(key, library)
        return library

    def _configure_waist_comfort(self, cfg_tune):
        """解析 dls_waist_comfort，计算舒适窗 [waist_comfort_lower, waist_comfort_upper]（不收窄时保持 None）。"""
        spec = self.dls_waist_comfort
        if spec is None or spec is False or (
            isinstance(spec, str) and spec.lower() in ("off", "none", "disabled")
        ):
            return
        waist_names = [n for n in self.active_joint_names if n in WAIST_JOINT_NAMES]
        if isinstance(spec, str):
            if spec.lower() != "auto":
                raise ValueError(
                    "dls_waist_comfort must be 'auto', None/False or {waist_joint: half_window_rad}"
                )
            lo, hi, info = derive_waist_comfort_bounds(
                self.pin_model,
                self.pin_data,
                self.pin_maps,
                self.q_ready,
                self.search_lower,
                self.search_upper,
                allowable_efforts=cfg_tune.get("load_allowable_efforts"),
                soft_ratio=cfg_tune.get("load_soft_ratio"),
            )
            for entry in info.values():
                entry["source"] = "auto"
        elif isinstance(spec, Mapping):
            unknown = set(spec) - set(waist_names)
            if unknown:
                raise ValueError(
                    f"dls_waist_comfort contains non-waist/unknown joints: {sorted(unknown)}"
                )
            lo, hi = _build_effective_bounds(
                self.search_lower,
                self.search_upper,
                self.q_ready,
                virtual_half_windows=spec,
                active_names=self.active_joint_names,
            )
            info = {
                name: {
                    "source": "explicit",
                    "half_window": float(spec[name]),
                    "lower": float(lo[self.active_joint_names.index(name)]),
                    "upper": float(hi[self.active_joint_names.index(name)]),
                }
                for name in spec
                if spec[name] is not None
            }
        else:
            raise ValueError(
                "dls_waist_comfort must be 'auto', None/False or {waist_joint: half_window_rad}"
            )
        lo = np.maximum(self.search_lower, lo)
        hi = np.minimum(self.search_upper, hi)
        self.waist_comfort_info = info
        if np.any(lo > self.search_lower) or np.any(hi < self.search_upper):
            self.waist_comfort_lower, self.waist_comfort_upper = lo, hi

    def _resolve_joint_jump_limits(self, value):
        """解析 max_joint_jump_rad：None = 不限；正标量 = 所有关节同一上限；
        {关节名: rad 或 None} = 逐关节设置，未列出的关节取默认 MAX_JOINT_JUMP_RAD（π）。"""
        if value is None:
            return None
        names = self.active_joint_names
        if isinstance(value, Mapping):
            unknown = set(value) - set(names)
            if unknown:
                raise ValueError(f"max_joint_jump_rad has unknown joints: {sorted(unknown)}")
            limits = np.array(
                [
                    np.inf if value.get(n, MAX_JOINT_JUMP_RAD) is None
                    else float(value.get(n, MAX_JOINT_JUMP_RAD))
                    for n in names
                ],
                dtype=float,
            )
        else:
            limits = np.full(len(names), float(value), dtype=float)
        if np.any(np.isnan(limits)) or np.any(limits <= 0.0):
            raise ValueError("max_joint_jump_rad must be positive (or None)")
        return limits

    def _periodic_mask(self):
        """无界 continuous 关节（仅这些关节的角度差折叠到 [-π, π)）。"""
        return ~np.isfinite(self.lower) & ~np.isfinite(self.upper)

    def _apply_joint_jump_policy(self, candidates, q_prev):
        """H4 不绕大圈：为所有候选（含 SSIK / direct-psi）补算单关节跳变并按策略过滤。

        reject（默认）：任一关节跳变超限的候选一律剔除；
        prefer：存在不超限的位姿达标候选时只用它们，否则保留全部并在诊断中标记。
        返回 (保留的候选, 因跳变被剔除的位姿达标候选数)。
        """
        periodic = self._periodic_mask()
        for c in candidates:
            if c is not None and "q" in c:
                jmax, jidx, ok = joint_jump_metrics(
                    c["q"], q_prev, periodic, self.max_joint_jump
                )
                c.update(
                    joint_jump_max_rad=jmax,
                    joint_jump_joint=self.active_joint_names[jidx],
                    joint_jump_ok=ok,
                )
        self.last_load_diagnostics.update(
            joint_jump_policy=self.joint_jump_policy,
            joint_jump_limit_rad=None
            if self.max_joint_jump is None
            else dict(zip(self.active_joint_names, self.max_joint_jump.tolist())),
        )
        if self.max_joint_jump is None:
            return candidates, 0
        kept = [c for c in candidates if c.get("joint_jump_ok", True)]
        rejected_pose = sum(
            1
            for c in candidates
            if not c.get("joint_jump_ok", True) and c.get("pose_ok")
        )
        if self.joint_jump_policy == "prefer" and not any(
            c.get("pose_ok") for c in kept
        ):
            kept = list(candidates)
        self.last_load_diagnostics["joint_jump_rejected"] = len(candidates) - len(kept)
        return kept, rejected_pose

    def _classify_empty_pool(self, n_raw, jump_rejected_pose, per_seed):
        """没有任何候选通过硬约束时给出准确失败码，避免把“没算出来”误报成“负荷超限”。"""
        errors = [s["exception"] for s in (per_seed or []) if "exception" in s]
        search = (self._load_context or {}).get("search_diagnostics", {})
        if n_raw == 0:
            if errors:
                return "SEED_ERRORS", f"all {len(errors)} seeds raised; first: {errors[0]}"
            if search.get("stop_reason") == "TIME_BUDGET":
                return "SEARCH_TIME_BUDGET", "budget exhausted before any seed produced a candidate"
            return "NO_CANDIDATE", "no seed produced a candidate"
        if self.last_load_diagnostics.get("pose_valid_count", 0) == 0:
            if jump_rejected_pose:
                return (
                    "JOINT_JUMP_LIMIT",
                    f"{jump_rejected_pose} pose-valid candidates exceed max_joint_jump_rad "
                    "relative to q_prev (would rotate a joint the long way round); "
                    "use an intermediate waypoint or raise the limit",
                )
            note = f"; {len(errors)} seeds raised, first: {errors[0]}" if errors else ""
            return (
                "NO_VALID_POSE_OR_PHYSICS",
                "no candidate reached the target pose within tolerance" + note,
            )
        return (
            "NO_LOAD_SAFE_SOLUTION",
            "No candidate passed bounds and confirmed static-load gates",
        )

    def _check_model(self):
        if self.pin_maps is not None:
            missing = [
                name
                for name in self.active_joint_names
                if name not in self.pin_maps["name_to_joint_id"]
            ]
            if missing:
                raise RuntimeError(
                    f"Missing active joints in Pinocchio model: {missing}"
                )

    def evaluate_physics(self, q_active, *, joint_state=None):
        """计算给定构型的物理指标；strict 模式强制要求真实全身状态快照与标定力矩。"""
        if not self._solve_lock.acquire(blocking=False):
            raise RuntimeError(
                "solver busy; do not share Pinocchio Data concurrently"
            )
        try:
            if self.pin_model is None:
                raise ValueError("Pinocchio model unavailable")
            if self.load_mode == "strict":
                snapshot = (
                    joint_state
                    if joint_state is not None
                    else self._latest_joint_state
                )
                context = prepare_load_context(
                    self.pin_model,
                    self.pin_maps,
                    self._load_config,
                    snapshot,
                )
                self._use_confirmed_efforts(context)
            return evaluate_pose_physics(
                self.pin_model,
                self.pin_data,
                self.pin_maps,
                q_active,
                q_ref=self.q_ready,
            )
        finally:
            if self.pin_maps is not None:
                self._use_urdf_efforts()
                self.joint_efforts_active = self.pin_maps["effort_limits"].copy()
            self._solve_lock.release()

    def _use_confirmed_efforts(self, context):
        """strict：求解 / 物理评估期间 pin_maps 使用冻结快照的 q 模板与确认的容许力矩。"""
        self.pin_maps["q_template"] = context["q_template"].copy()
        self.pin_maps["effort_limits"] = context["policy"].vectors(
            self.active_joint_names
        )[0]
        self.pin_maps["effort_basis"] = "confirmed_allowable_joint_side_efforts"

    def _use_urdf_efforts(self):
        """恢复 URDF 参考力矩、去掉 q 模板（求解 / 评估结束时无论成败都调用）。"""
        self.pin_maps.pop("q_template", None)
        self.pin_maps["effort_limits"] = self.pin_maps["urdf_effort_limits"].copy()
        self.pin_maps["effort_basis"] = "URDF reference, not confirmed continuous rating"

    def print_info(self):
        T_ready_local = self.chain.fk(self.q_ready)
        T_ready_world = self.T_world_from_base @ T_ready_local

        print("")
        print("=" * 80)
        print("HAArmIKSolver")
        print("=" * 80)
        print("URDF:", self.urdf)
        print("side:", self.side)
        print("include_waist:", self.include_waist)
        print("BASE_LINK:", self.base_link)
        print("EE_LINK:", self.ee_link)
        print("active joints:", self.active_joint_names)
        print(
            "Q_READY:",
            np.array2string(self.q_ready, precision=6, separator=", "),
        )
        print("psi_ready rad:", self.psi_ready)
        print("psi_ready deg:", np.rad2deg(self.psi_ready))
        print("EE ready local xyz:", T_ready_local[:3, 3])
        print("EE ready world xyz:", T_ready_world[:3, 3])
        print("workspace anchor joint:", self.workspace_anchor_joint)
        print("workspace anchor local xyz:", self.workspace_anchor_local_xyz)
        print(
            "workspace max anchor->EE radius:",
            self.workspace_max_j7_radius_from_anchor,
        )
        print("")
        print("USE_DIRECT_PSI:", self.use_direct_psi)
        print("USE_SSIK:", self.use_ssik)
        print("USE_PINOCCHIO_IK:", self.use_pinocchio_ik)
        print("USE_TORQUE_COST:", self.use_torque_cost)
        print("W_TORQUE:", W_TORQUE)
        print("G_MOUNTED_BASE:", np.round(self.g_mounted_base, 6))
        if self.pin_model is not None:
            print("Pinocchio model nq:", self.pin_model.nq)
            print("Pinocchio model nv:", self.pin_model.nv)
        if self.joint_efforts_active is not None:
            print("joint effort limits:", self.joint_efforts_active)
        if self.include_waist:
            print("waist release max reach (s_joint->TCP):", self.dls_waist_max_reach)
            if self.waist_comfort_lower is None:
                print("waist comfort window: disabled")
            else:
                for i, name in enumerate(self.active_joint_names):
                    if name in self.waist_comfort_info:
                        e = self.waist_comfort_info[name]
                        print(
                            f"waist comfort window {name}: "
                            f"[{self.waist_comfort_lower[i]:.4f}, {self.waist_comfort_upper[i]:.4f}] rad "
                            f"({e.get('source')}, {e.get('kind', 'explicit')}"
                            + (
                                f", tau_max={e['tau_max_nm']:.1f}Nm, cap={e['cap_nm']:.1f}Nm"
                                f" [{e['cap_basis']}]"
                                if e.get("kind") == "gravity"
                                else ""
                            )
                            + ")"
                        )
        print("=" * 80)
        print("URDF root link:", self.urdf_root_link)
        print("solver base link:", self.base_link)
        print(
            "Pinocchio target frame mode:",
            "world/root" if self._pin_uses_root_frame else "local/solver_base",
        )
        print("")

    def target_to_local(self, T_target, frame="local"):
        if self._solve_T_root_base is not None:
            T = validate_transform(T_target)
            if frame == "local":
                return T, self._solve_T_root_base @ T
            if frame == "world":
                return np.linalg.inv(self._solve_T_root_base) @ T, T
            raise ValueError(
                "frame must be local or world (URDF-root legacy convention)"
            )
        return target_frames(
            T_target,
            frame=frame,
            T_world_from_base=self.T_world_from_base,
            T_base_from_world=self.T_base_from_world,
        )

    def workspace_diagnostic(
        self,
        T_target,
        frame="local",
        target_is_tcp=False,
        tcp_offset_xyz=None,
        margin=None,
    ):
        """Simple max-radius workspace check from J1/J2 shoulder joint to target TCP."""
        T_target = np.asarray(T_target, dtype=float)

        if T_target.shape != (4, 4):
            raise ValueError("T_target must be 4x4 matrix.")

        if tcp_offset_xyz is None:
            tcp_offset_xyz = np.zeros(3)

        tcp_offset_xyz = np.asarray(tcp_offset_xyz, dtype=float)
        if tcp_offset_xyz.shape != (3,):
            raise ValueError(
                f"tcp_offset_xyz must be length 3, got shape {tcp_offset_xyz.shape}"
            )
        tcp_offset_norm = float(np.linalg.norm(tcp_offset_xyz))
        T_j7_to_tcp = make_translation_T(tcp_offset_xyz)

        if target_is_tcp:
            T_tcp_input = T_target
            T_j7_input = T_tcp_input @ invert_T(T_j7_to_tcp)
        else:
            T_j7_input = T_target
            T_tcp_input = T_j7_input @ T_j7_to_tcp

        T_j7_local, T_j7_world = self.target_to_local(T_j7_input, frame=frame)
        T_tcp_local, T_tcp_world = self.target_to_local(
            T_tcp_input, frame=frame
        )

        p_anchor_local = np.asarray(
            self.workspace_anchor_local_xyz, dtype=float
        )
        p_anchor_world = transform_point(
            self.T_world_from_base, p_anchor_local
        )

        p_j7_local = T_j7_local[:3, 3]
        p_tcp_local = T_tcp_local[:3, 3]
        p_j7_world = T_j7_world[:3, 3]
        p_tcp_world = T_tcp_world[:3, 3]

        dist_anchor_to_j7 = float(np.linalg.norm(p_j7_local - p_anchor_local))
        dist_anchor_to_tcp = float(
            np.linalg.norm(p_tcp_local - p_anchor_local)
        )
        max_j7_radius = float(self.workspace_max_j7_radius_from_anchor)
        max_tcp_radius = max_j7_radius + tcp_offset_norm

        ok = dist_anchor_to_tcp <= max_tcp_radius
        reason = "OK_MAX_TCP_RADIUS" if ok else "OUT_OF_MAX_TCP_RADIUS"

        return {
            "ok": bool(ok),
            "reason": reason,
            "frame": frame,
            "target_is_tcp": bool(target_is_tcp),
            "anchor_joint": self.workspace_anchor_joint,
            "anchor_local_xyz": p_anchor_local.copy(),
            "anchor_world_xyz": p_anchor_world.copy(),
            "tcp_offset_xyz": tcp_offset_xyz.copy(),
            "tcp_offset_norm": tcp_offset_norm,
            "T_j7_local": T_j7_local,
            "T_j7_world": T_j7_world,
            "T_tcp_local": T_tcp_local,
            "T_tcp_world": T_tcp_world,
            "required_j7_local_xyz": p_j7_local.copy(),
            "required_j7_world_xyz": p_j7_world.copy(),
            "tcp_local_xyz": p_tcp_local.copy(),
            "tcp_world_xyz": p_tcp_world.copy(),
            "dist_anchor_to_required_j7": dist_anchor_to_j7,
            "dist_anchor_to_tcp": dist_anchor_to_tcp,
            "workspace_max_j7_radius_from_anchor": max_j7_radius,
            "workspace_max_tcp_radius_from_anchor": max_tcp_radius,
        }

    def print_workspace_diagnostic(self, info, prefix="[WORKSPACE]"):
        ok = bool(info["ok"])
        reason = info["reason"]
        level = "OK" if ok else "WARN"

        anchor_joint = info.get("anchor_joint", "UNKNOWN")

        print("")
        print(f"{prefix} {level}: side={self.side}, reason={reason}")
        print(f"{prefix} frame                       : {info['frame']}")
        print(
            f"{prefix} target_is_tcp               : {info['target_is_tcp']}"
        )
        print(f"{prefix} anchor_joint                : {anchor_joint}")
        print(
            f"{prefix} anchor local xyz            : "
            f"{np.round(info['anchor_local_xyz'], 6)}"
        )
        print(
            f"{prefix} anchor world xyz            : "
            f"{np.round(info['anchor_world_xyz'], 6)}"
        )
        print(
            f"{prefix} tcp_offset_xyz              : "
            f"{np.round(info['tcp_offset_xyz'], 6)}"
        )
        print(
            f"{prefix} tcp_offset_norm             : "
            f"{info['tcp_offset_norm']:.6f} m"
        )
        print(
            f"{prefix} TCP local xyz               : "
            f"{np.round(info['tcp_local_xyz'], 6)}"
        )
        print(
            f"{prefix} required J7 local xyz       : "
            f"{np.round(info['required_j7_local_xyz'], 6)}"
        )
        print(
            f"{prefix} TCP world xyz               : "
            f"{np.round(info['tcp_world_xyz'], 6)}"
        )
        print(
            f"{prefix} required J7 world xyz       : "
            f"{np.round(info['required_j7_world_xyz'], 6)}"
        )
        print(
            f"{prefix} dist {anchor_joint}->TCP     : "
            f"{info['dist_anchor_to_tcp']:.6f} m"
        )
        print(
            f"{prefix} dist {anchor_joint}->J7      : "
            f"{info['dist_anchor_to_required_j7']:.6f} m"
        )
        print(
            f"{prefix} max {anchor_joint}->J7       : "
            f"{info['workspace_max_j7_radius_from_anchor']:.6f} m"
        )
        print(
            f"{prefix} max {anchor_joint}->TCP      : "
            f"{info['workspace_max_tcp_radius_from_anchor']:.6f} m"
        )

        if not ok:
            print(
                f"{prefix} WARNING: Target TCP is outside max shoulder-to-TCP radius."
            )
            print(
                f"{prefix}          This is likely a TCP target / tcp_offset / frame issue."
            )

        print("")

    def check_workspace(
        self,
        T_target,
        frame="local",
        target_is_tcp=False,
        tcp_offset_xyz=None,
        margin=None,
        verbose=True,
    ):
        info = self.workspace_diagnostic(
            T_target=T_target,
            frame=frame,
            target_is_tcp=target_is_tcp,
            tcp_offset_xyz=tcp_offset_xyz,
            margin=margin,
        )

        if verbose:
            self.print_workspace_diagnostic(info)

        return info["ok"], info

    def _collect_portfolio(
        self,
        T_target_local,
        T_target_world,
        q_prev,
        q_ready,
        lower,
        upper,
        rng,
        orientation_mode,
        tool_axis,
        tool_cone_half_angle,
    ):
        """Run the SSIK + Pinocchio-DLS stages."""
        candidates = []
        all_direct_seeds = []
        per_seed = []

        if self.use_ssik:
            ssik_candidates, ssik_seeds = collect_ssik_candidates_and_seeds(
                arm=self.arm,
                chain=self.chain,
                T_target=T_target_local,
                psi_ready=self.psi_ready,
                q_prev=q_prev,
                q_ready=q_ready,
                lower=lower,
                upper=upper,
                orientation_mode=orientation_mode,
                tool_axis=tool_axis,
                tool_cone_half_angle=tool_cone_half_angle,
                include_waist=self.include_waist,
                multi_turn_mask=self.multi_turn_mask,
            )
            candidates.extend(ssik_candidates)

            if self.use_direct_psi:
                all_direct_seeds.extend(ssik_seeds)

        if self.use_pinocchio_ik:
            T_target_pin = T_target_world if self._pin_uses_root_frame else T_target_local

            dls_params = {
                "max_iter": self.dls_max_iter,
                "damping": self.dls_damping,
                "max_step": self.dls_max_step,
                "nullspace_gain": self.dls_nullspace_gain,
                "waist_null_gain": self.dls_waist_null_gain,
                "waist_task_weight": self.dls_waist_task_weight,
                "waist_task_weight_far": self.dls_waist_task_weight_far,
                "gravity_torque_gain": self.dls_gravity_torque_gain,
                "relative_gravity_gain": self.dls_relative_gravity_gain,
                "com_xy_gain": self.dls_com_xy_gain,
                "joint_limit_gain": self.dls_joint_limit_gain,
                "virtual_half_windows": self.dls_virtual_half_windows,
                "posture_use_equivalent_branch": self.dls_posture_use_equivalent_branch,
                "load_context": self._load_context,
                "deadline": self._load_deadline,
                "characteristic_length": self.dls_characteristic_length,
                # v8：DLS 内部达标判据不宽于最终复核（ik_common.POS_TOL / ORI_TOL）
                "pos_tol": min(PIN_IK_POS_TOL, POS_TOL),
                "ori_tol": min(PIN_IK_ORI_TOL, ORI_TOL),
            }
            dls_params["continuation_reach"] = self.dls_arm_max_reach
            if self.include_waist:
                dls_params["shoulder_joint"] = self.s_joint
                dls_params["waist_max_reach"] = self.dls_waist_max_reach
            comfort_bounds = (
                None
                if self.waist_comfort_lower is None
                else (self.waist_comfort_lower, self.waist_comfort_upper)
            )
            comfort_info = {}

            pin_candidates, pin_seeds, pin_per_seed = (
                collect_pinocchio_candidates_and_seeds(
                    pin_model=self.pin_model,
                    pin_data=self.pin_data,
                    pin_maps=self.pin_maps,
                    ee_frame_id=self.pin_ee_frame_id,
                    chain=self.chain,
                    T_target_pin=T_target_pin,
                    T_target_local=T_target_local,
                    psi_ready=self.psi_ready,
                    q_prev=q_prev,
                    q_ready=q_ready,
                    lower=lower,
                    upper=upper,
                    rng=rng,
                    orientation_mode=orientation_mode,
                    tool_axis=tool_axis,
                    tool_cone_half_angle=tool_cone_half_angle,
                    include_waist=self.include_waist,
                    dls_params=dls_params,
                    load_search={
                        "max_seeds": self._load_config.get(
                            "load_max_seeds", 10
                        ),
                        "min_seeds": self._load_config.get(
                            "load_min_seeds", 2
                        ),
                        "min_unlocked": self._load_config.get(
                            "load_min_unlocked_seeds", 1
                        ),
                        "max_joint_jump": self.max_joint_jump,
                        "joint_jump_policy": self.joint_jump_policy,
                        "locked_seed_skip_tol": self.locked_seed_skip_tol,
                        "continuation_max_step": self.dls_continuation_max_step,
                        "library_seeds": self.dls_seed_library_seeds,
                        "library_min_separation": self.dls_seed_library_min_separation,
                        "library_continuation": self.dls_seed_library_continuation,
                        "comfort_early_fallback": self.dls_waist_comfort_early_fallback,
                        "jump_rescue": self.dls_jump_limit_rescue,
                        "comfort_patience": self.dls_waist_comfort_patience,
                    },
                    comfort_bounds=comfort_bounds,
                    comfort_continuation_tol=self.dls_waist_comfort_continuation_tol,
                    comfort_info=comfort_info,
                    continuation_local=self.dls_continuation_local,
                    seed_library=self.seed_library,
                    locked_reach=self.dls_locked_reach,
                )
            )
            local_used = bool(comfort_info.get("continuation_local_used"))
            self.last_load_diagnostics["continuation"] = {
                "local_enabled": self.dls_continuation_local,
                "regime": comfort_info.get("continuation_regime"),
                "local_used": local_used,
                "hold": bool(comfort_info.get("continuation_hold", False)),
            }
            if local_used:
                # 续算局部解：只保留局部候选（可选的 SSIK 候选不再参与择优，direct-psi 阶段也跳过）
                candidates = list(pin_candidates)
                all_direct_seeds = []
            else:
                candidates.extend(pin_candidates)
            per_seed.extend(pin_per_seed)
            stages = [p.get("search_stage") for p in pin_per_seed]
            self.last_load_diagnostics["waist_comfort"] = {
                "enabled": comfort_bounds is not None,
                "regime": comfort_info.get("regime"),
                "applied": bool(comfort_info.get("applied", False)),
                "fallback_to_full_bounds": bool(
                    comfort_info.get("fallback_to_full_bounds", False)
                ),
                "comfort_seeds": stages.count("comfort"),
                "full_bounds_seeds": stages.count("full") if comfort_info.get("applied") else 0,
                "jump_box_seeds": stages.count("jump_box"),
            }

            if self.use_direct_psi and not local_used:
                all_direct_seeds.extend(pin_seeds)

        return candidates, unique_seed_list(all_direct_seeds), per_seed

    def _solve_direct_psi_stage(
        self,
        T_target_local,
        q_prev,
        q_ready,
        lower,
        upper,
        direct_seeds,
        orientation_mode,
        tool_axis,
        tool_cone_half_angle,
        verbose,
    ):
        """direct-psi stage: solve from the collected seeds and score."""
        direct_sols = solve_direct_psi(
            self.chain,
            T_target_local,
            self.psi_ready,
            direct_seeds,
            S_joint=self.s_joint,
            E_joint=self.e_joint,
            W_joint=self.w_joint,
            ref_dir=self.ref_dir,
            q_ref=q_prev if self.portfolio_q_ref_mode == "prev" else q_ready,
            q_bias=None,
            orientation_mode=orientation_mode,
            tool_axis=tool_axis,
            tool_cone_half_angle=tool_cone_half_angle,
            verbose=verbose,
            w_pos=100.0,
            w_ori=20.0,
            w_psi=20.0,
            w_seed=0.03,
            w_bias=0.00,
            w_center=0.005,
            w_limit=0.05,
            limit_margin=0.10,
            max_nfev=self.max_direct_nfev,
            refine_task=(orientation_mode == "full"),
            pos_tol=self.direct_pos_tol,
            ori_tol=self.direct_ori_tol,
            psi_tol=self.direct_psi_tol,
            bounds=(lower, upper),
        )

        candidates = []
        for sol in direct_sols:
            q_aligned = select_closest_bounded_branch(
                sol["q"],
                q_prev,
                lower,
                upper,
                multi_turn_mask=self.multi_turn_mask,
            )
            info = score_candidate(
                self.chain,
                q_aligned,
                T_target_local,
                self.psi_ready,
                q_prev,
                q_ready,
                lower,
                upper,
                orientation_mode=orientation_mode,
                tool_axis=tool_axis,
                tool_cone_half_angle=tool_cone_half_angle,
            )
            info["source"] = "direct_psi"
            info["solve_method"] = "direct_psi_analytical_jacobian"
            info["fallback_used"] = False
            info["direct_seed_count"] = len(direct_seeds)
            if "psi_err" in sol:
                info["direct_psi_err"] = sol["psi_err"]
            if "q_ref_dq" in sol:
                info["q_ref_dq"] = sol["q_ref_dq"]
            candidates.append(info)

        return candidates

    def _finalize_result(
        self,
        candidates,
        per_seed,
        T_target_local,
        T_target_world,
        frame,
        orientation_mode,
        tool_axis,
        tool_cone_half_angle,
        q_prev,
        update_prev,
        allow_approximate,
    ):
        """硬约束过滤 -> 负荷峰值容差带 -> 原 select_best_candidate 择优 -> 受门控约束的近似回退。

        硬约束顺序：单关节跳变（不绕大圈）-> 有效限位 + 位姿复核 + 静态负荷（strict）。
        跳变过滤放在负荷容差带之前，否则容差带可能只剩“负荷略低但要绕大圈”的候选。
        """
        n_raw = len(candidates)
        candidates, jump_rejected_pose = self._apply_joint_jump_policy(
            candidates, q_prev
        )
        strict_load_safe_pool = []
        T_target_pin = (
            T_target_world if self._pin_uses_root_frame else T_target_local
        )
        if self.load_mode == "strict":
            stale = self._stale_context_failure(per_seed)
            if stale:
                return stale
            candidates, strict_load_safe_pool = self._filter_load_candidates(
                candidates,
                T_target_pin,
                orientation_mode,
                tool_axis,
                tool_cone_half_angle,
                return_load_safe_pool=True,
            )
            if not candidates:
                if allow_approximate and strict_load_safe_pool:
                    candidates = strict_load_safe_pool
                else:
                    code, detail = self._classify_empty_pool(
                        n_raw, jump_rejected_pose, per_seed
                    )
                    return self._load_failure(code, detail, per_seed)
        elif jump_rejected_pose and not any(c.get("pose_ok") for c in candidates):
            # load_mode="off"：位姿达标的候选全部因跳变超限被剔除
            code, detail = self._classify_empty_pool(
                n_raw, jump_rejected_pose, per_seed
            )
            return self._load_failure(code, detail, per_seed)

        if self.use_torque_cost and len(candidates) > 0:
            add_torque_metrics_to_candidates(
                candidates=candidates,
                pin_model=self.pin_model,
                pin_data=self.pin_data,
                pin_maps=self.pin_maps,
                active_joint_names=self.active_joint_names,
                joint_efforts_active=self.joint_efforts_active,
            )

        best, status = select_best_candidate(
            candidates,
            use_torque=self.use_torque_cost,
        )
        self._record_comfort_selection(best, per_seed)

        if best is not None and "joint_jump_max_rad" in best:
            self.last_load_diagnostics.update(
                joint_jump_max_rad=best["joint_jump_max_rad"],
                joint_jump_joint=best["joint_jump_joint"],
                joint_jump_ok=best["joint_jump_ok"],
            )

        if best is not None:
            best["solve_engine"] = "dls"
            best["target_frame"] = frame
            best["T_target_local"] = T_target_local
            best["T_target_world"] = T_target_world
            best["orientation_mode"] = orientation_mode
            best["tool_axis"] = tool_axis.copy()
            best["tool_cone_half_angle"] = float(tool_cone_half_angle)
            best["tool_cone_half_angle_deg"] = float(
                np.rad2deg(tool_cone_half_angle)
            )
            best["approximate"] = False
            best["approximate_accepted"] = False

        if best is not None and is_valid_status(status):
            if self.load_mode == "strict":
                if not self._check_load_candidate(
                    best,
                    T_target_pin,
                    orientation_mode,
                    tool_axis,
                    tool_cone_half_angle,
                ):
                    return self._load_failure(
                        "FINAL_RECHECK_FAILED",
                        "Selected candidate failed recomputation",
                        per_seed,
                    )
                stale = self._stale_context_failure(per_seed)
                if stale:
                    return stale
                code = "ACCEPTED"
            else:
                code = "ACCEPTED_WITH_LOAD_PROTECTION_OFF"
            return self._accept(best, status, code, per_seed, update_prev)

        if (
            allow_approximate
            and best is not None
            and status == "NO_VALID_POSE"
            and "q" in best
        ):
            if self.load_mode == "strict":
                self._check_load_candidate(
                    best,
                    T_target_pin,
                    orientation_mode,
                    tool_axis,
                    tool_cone_half_angle,
                )
                if not (
                    best.get("load_bounds_ok", False)
                    and best.get("load_ok", False)
                ):
                    return self._load_failure(
                        "NO_LOAD_SAFE_APPROXIMATE_SOLUTION",
                        "Approximate candidate violated bounds or confirmed static-load cap",
                        per_seed,
                    )
                stale = self._stale_context_failure(per_seed)
                if stale:
                    return stale
                code = "APPROXIMATE_ACCEPTED_LOAD_SAFE"
            else:
                code = "APPROXIMATE_WITH_LOAD_PROTECTION_OFF"

            best["approximate"] = True
            best["approximate_accepted"] = True
            best["original_status"] = status
            best["status"] = "APPROXIMATE"
            best["approximate_policy"] = (
                "allow_approximate=True: use best candidate by pose error "
                "(strict bounds and static-load gates enforced when load_mode='strict')"
            )
            return self._accept(best, "APPROXIMATE", code, per_seed, update_prev)

        failure_reason = None
        if best is None:
            failure_reason = FailureReason.SEEDS_EXHAUSTED
            errors = [s["exception"] for s in (per_seed or []) if "exception" in s]
            failure_detail = (
                "no candidate survived any solve pass "
                "(seed chain empty or all engines raised)"
                + (f"; {len(errors)} seeds raised, first: {errors[0]}" if errors else "")
            )
        else:
            failure_reason = FailureReason.NOT_CONVERGED
            n_stalled = sum(1 for s in per_seed if s.get("stalled"))
            best_pe = min(
                (
                    s["final_pos_err"]
                    for s in per_seed
                    if np.isfinite(s.get("final_pos_err", np.nan))
                ),
                default=np.nan,
            )
            failure_detail = (
                f"{len(candidates)} candidates, none within tolerance "
                f"(POS_TOL={POS_TOL:.1e}); "
                f"best pos_err={best.get('pos_err', np.nan):.3e} "
                f"ori_err={best.get('ori_err', np.nan):.3e}; "
                f"{n_stalled} seeds stalled; "
                f"best seed pos_err="
                f"{(f'{best_pe * 1000:.3f}mm' if np.isfinite(best_pe) else 'n/a')}"
            )

        return (
            None,
            status,
            make_result(
                best or {},
                status,
                failure_reason=failure_reason,
                failure_detail=failure_detail,
                per_seed=per_seed or None,
            ),
        )

    def _accept(self, best, status, code, per_seed, update_prev):
        """记录接受诊断（strict 附最终复核的负荷 / 位姿数值）、按需更新 q_prev，返回成功三元组。"""
        accepted = {"accepted": True, "code": code}
        if self.load_mode == "strict":
            accepted.update(
                peak_ratio=best["load_peak_ratio"],
                peak_joint=best["load_peak_joint"],
                ratios=best["load_guard_ratios"],
                gravity_efforts=best["load_guard_efforts"],
                position_error_m=best["load_final_pos_err"],
                orientation_violation_rad=best["load_final_ori_violation"],
            )
        self.last_load_diagnostics.update(accepted)
        if update_prev:
            self.q_prev = best["q"].copy()
        return (
            best["q"].copy(),
            status,
            make_result(
                {
                    k: v
                    for k, v in best.items()
                    if not k.startswith("load_") and k != "_load_checked_q_bytes"
                },
                status,
                per_seed=per_seed or None,
            ),
        )

    def _stale_context_failure(self, per_seed=None):
        """strict：冻结快照已过期或全身状态已变化时返回 STALE_OR_CHANGED_CONTEXT 失败三元组，否则 None。"""
        error = self._load_snapshot_error()
        if error:
            return self._load_failure("STALE_OR_CHANGED_CONTEXT", error, per_seed)
        return None

    def _record_comfort_selection(self, best, per_seed):
        """v8 诊断：被选中的候选来自哪个搜索阶段、腰部是否在舒适窗内。"""
        diag = self.last_load_diagnostics.get("waist_comfort")
        if diag is None or best is None or best.get("source") != "pinocchio_ik":
            return
        idx = best.get("seed_used")
        diag["selected_stage"] = next(
            (p.get("search_stage") for p in (per_seed or []) if p.get("seed_idx") == idx),
            None,
        )
        if self.waist_comfort_lower is not None and best.get("q") is not None:
            q = np.asarray(best["q"], dtype=float)
            diag["selected_in_comfort"] = bool(
                np.all(q >= self.waist_comfort_lower - 1e-9)
                and np.all(q <= self.waist_comfort_upper + 1e-9)
            )

    def _solve_impl(
        self,
        T_target,
        frame="local",
        q_prev=None,
        update_prev=True,
        verbose=False,
        allow_approximate=False,
        orientation_mode="full",
        tool_axis=None,
        tool_cone_half_angle=None,
    ):
        T_target_local, T_target_world = self.target_to_local(
            T_target, frame=frame
        )
        # 0. 工作空间绝对几何可达性预检 (< 0.001 ms)
        if (
            not allow_approximate
            and hasattr(self, "chain_max_reach")
            and self.chain_max_reach is not None
            and float(np.linalg.norm(T_target_local[:3, 3])) > self.chain_max_reach + POS_TOL
        ):
            target_dist = float(np.linalg.norm(T_target_local[:3, 3]))
            detail = f"target position {target_dist:.3f}m exceeds maximum chain reach {self.chain_max_reach:.3f}m"
            self.last_load_diagnostics.update(
                accepted=False,
                code="WORKSPACE_REJECTED",
                detail=detail,
            )
            return (
                None,
                "OUTSIDE_WORKSPACE",
                make_result(
                    {},
                    "OUTSIDE_WORKSPACE",
                    failure_reason=FailureReason.WORKSPACE_REJECTED,
                    failure_detail=f"[WORKSPACE_REJECTED] {detail}",
                ),
            )
        orientation_mode = validate_orientation_mode(orientation_mode)
        tool_cone_half_angle = normalize_cone_half_angle(tool_cone_half_angle)
        if tool_axis is None:
            tool_axis = TOOL_AXIS_X

        tool_axis = normalize(tool_axis)
        # 统一使用包含显式虚拟搜索窗的有效搜索限位
        lower, upper = self.search_lower, self.search_upper

        q_ready = self.q_ready.copy()

        if q_prev is None:
            q_prev = self.q_prev.copy()
        else:
            q_prev = finite_vector(
                q_prev, len(self.active_joint_names), "q_prev"
            )
            if self.load_mode == "strict":
                if np.any(q_prev < self.lower) or np.any(q_prev > self.upper):
                    raise ValueError(
                        "q_prev outside actual/planning bounds; not clipped"
                    )
            else:
                q_prev = clip_to_bounds(q_prev, lower, upper)

        rng = np.random.default_rng(RNG_SEED)

        candidates = []

        # 1. local seeds (direct_psi only)
        direct_seeds = []
        if self.use_direct_psi and self.use_local_seeds:
            direct_seeds.extend(
                make_direct_local_seeds(
                    q_prev=q_prev,
                    q_ready=q_ready,
                    lower=lower,
                    upper=upper,
                    rng=rng,
                )
            )

        # 2. SSIK + Pinocchio DLS portfolio
        portfolio_candidates, portfolio_seeds, per_seed = (
            self._collect_portfolio(
                T_target_local,
                T_target_world,
                q_prev,
                q_ready,
                lower,
                upper,
                rng,
                orientation_mode,
                tool_axis,
                tool_cone_half_angle,
            )
        )
        candidates.extend(portfolio_candidates)
        direct_seeds.extend(portfolio_seeds)

        # 3. direct-psi stage（v9：续算局部解已采用时跳过）
        if self.use_direct_psi and not (
            self.last_load_diagnostics.get("continuation") or {}
        ).get("local_used"):
            candidates.extend(
                self._solve_direct_psi_stage(
                    T_target_local,
                    q_prev,
                    q_ready,
                    lower,
                    upper,
                    direct_seeds,
                    orientation_mode,
                    tool_axis,
                    tool_cone_half_angle,
                    verbose,
                )
            )

        # 4. torque cost + select + result
        return self._finalize_result(
            candidates,
            per_seed,
            T_target_local,
            T_target_world,
            frame,
            orientation_mode,
            tool_axis,
            tool_cone_half_angle,
            q_prev,
            update_prev,
            allow_approximate,
        )

    def close(self):
        """Original close interface, protected against an in-flight solve."""
        if not self._solve_lock.acquire(blocking=False):
            raise RuntimeError("cannot close while solve/evaluate is in flight")
        try:
            self.pin_model = None
            self.pin_data = None
            self.pin_maps = None
            self.pin_ee_frame_id = None
        finally:
            self._solve_lock.release()

    def _load_failure(self, code, detail, per_seed=None):
        self.last_load_diagnostics.update(
            accepted=False, code=code, detail=str(detail)
        )
        status = "NO_VALID_POSE"
        if code == "JOINT_JUMP_LIMIT":
            reason = FailureReason.JOINT_JUMP_LIMIT
        elif code in ("SEARCH_TIME_BUDGET", "TIME_BUDGET", "TIME_BUDGET_EXCEEDED"):
            reason = FailureReason.TIME_BUDGET_EXCEEDED
        elif code in ("SOLVER_BUSY", "BUSY"):
            reason = FailureReason.SOLVER_BUSY
        elif code == "NO_LOAD_SAFE_SOLUTION":
            reason = FailureReason.NO_LOAD_SAFE_SOLUTION
        elif code in ("WORKSPACE_REJECTED", "OUTSIDE_WORKSPACE"):
            reason = FailureReason.WORKSPACE_REJECTED
            status = "OUTSIDE_WORKSPACE"
        elif code in FailureReason._value2member_map_:
            reason = FailureReason(code)
        else:
            reason = FailureReason.NOT_CONVERGED

        return (
            None,
            status,
            make_result(
                {},
                status,
                failure_reason=reason,
                failure_detail=f"[{code}] {detail}",
                per_seed=per_seed or None,
            ),
        )

    def update_joint_state(self, snapshot_or_msg, *, now_ros_ns=None):
        """Call in the existing subscriber; no ROS import/subscription is created."""
        if isinstance(snapshot_or_msg, JointStateSnapshot) or type(snapshot_or_msg).__name__ == "JointStateSnapshot":
            snapshot = snapshot_or_msg
        else:
            snapshot = JointStateSnapshot.from_joint_state(
                snapshot_or_msg, now_ros_ns=now_ros_ns
            )
        prev = self._latest_joint_state
        if (
            prev is None
            or snapshot.received_monotonic >= prev.received_monotonic
        ):
            self._latest_joint_state = snapshot
        return snapshot

    def _load_snapshot_error(self):
        context = self._load_context
        error = context["snapshot"].freshness_error(
            context["max_age_s"],
            require_source_age=context["require_source_age"],
        )
        if error:
            return error
        latest = self._latest_joint_state
        frozen = context["snapshot"]
        if (
            latest is not None
            and latest.received_monotonic > frozen.received_monotonic
        ):
            error = latest.freshness_error(
                context["max_age_s"],
                require_source_age=context["require_source_age"],
            )
            if error:
                return error
            if set(self.pin_maps["all_names"]) - set(latest.positions):
                return "latest_full_state_incomplete"
            for name in set(self.pin_maps["all_names"]) - set(
                self.active_joint_names
            ):
                s = self.pin_maps["all_specs"][name]
                value = latest.positions[name]
                if value < s["lower"] or value > s["upper"]:
                    return f"latest_context_out_of_bounds:{name}"
                delta = value - frozen.positions[name]
                if s["continuous"]:
                    delta = (delta + np.pi) % (2.0 * np.pi) - np.pi
                key = (
                    "load_context_tolerance_m"
                    if s["kind"] == "prismatic"
                    else "load_context_tolerance_rad"
                )
                tolerance = float(
                    self._load_config.get(
                        key, 0.001 if s["kind"] == "prismatic" else 0.01
                    )
                )
                if not np.isfinite(tolerance) or tolerance < 0.0:
                    return "invalid_context_change_tolerance"
                if abs(delta) > tolerance:
                    return f"CONTEXT_CHANGED:{name}"
        return None

    def _check_load_candidate(
        self, candidate, T_target_world, orientation_mode, tool_axis, cone
    ):
        q = finite_vector(
            candidate.get("q"), len(self.active_joint_names), "candidate q"
        )
        # 同时验证真实物理限位与显式配置的虚拟搜索窗口限位
        bounds_ok = bool(
            np.all(q >= self.search_lower) and np.all(q <= self.search_upper)
        )
        q_bytes = q.tobytes()
        filtering = bool(self._load_context is not None and self._load_context.get("_filtering"))
        task_cached = (
            self._load_context.get("_task_cache", {}).get((q_bytes, id(T_target_world)))
            if filtering
            else None
        )
        if task_cached is not None:
            pos_err, ori_violation = task_cached
        else:
            task = dls_task_terms(
                self.pin_model,
                self.pin_data,
                self.pin_maps,
                self.pin_ee_frame_id,
                q,
                T_target_world,
                orientation_mode,
                tool_axis,
                cone,
                float(self.dls_characteristic_length),
                False,
            )
            pos_err = task["pos_err"]
            ori_violation = task["ori_violation"]
        pose_ok = bool(
            candidate.get("pose_ok", False)
            and pos_err <= POS_TOL
            and ori_violation <= ORI_TOL
        )
        cache = self._load_context.get("_physics_cache") if filtering else None
        physical = None if cache is None else cache.get(q_bytes)
        if physical is None:
            physical = load_physics(
                self.pin_model,
                self.pin_data,
                self.pin_maps,
                q,
                self._load_context,
                False,
            )
        candidate.update(
            _load_checked_q_bytes=q_bytes,
            load_ok=physical["load_ok"],
            load_bounds_ok=bounds_ok,
            load_pose_ok=pose_ok,
            load_peak_ratio=physical["peak_ratio"],
            load_peak_joint=physical["peak_joint"],
            load_guard_ratios=dict(
                zip(self._load_context["guard_names"], physical["ratios"])
            ),
            load_guard_efforts=dict(
                zip(self._load_context["guard_names"], physical["tau"])
            ),
            load_overloaded_joints=physical["overloaded_joints"],
            load_final_pos_err=pos_err,
            load_final_ori_violation=ori_violation,
        )
        return bool(bounds_ok and pose_ok and physical["load_ok"])

    def _filter_load_candidates(
        self,
        candidates,
        T_target_world,
        orientation_mode,
        tool_axis,
        cone,
        *,
        return_load_safe_pool=False,
    ):
        eligible = []
        load_safe_pool = []
        rejected = []
        pose_count = 0
        errors = 0
        if self._load_context is not None:
            self._load_context["_filtering"] = True
        try:
            for index, c in enumerate(candidates):
                try:
                    safe = self._check_load_candidate(
                        c, T_target_world, orientation_mode, tool_axis, cone
                    )
                    pose_count += int(c["load_pose_ok"] and c["load_bounds_ok"])
                    if c["load_bounds_ok"] and c["load_ok"]:
                        load_safe_pool.append(c)
                    if safe:
                        eligible.append(c)
                    else:
                        rejected.append(
                            {
                                "index": index,
                                "pose_ok": c["load_pose_ok"],
                                "bounds_ok": c["load_bounds_ok"],
                                "peak_ratio": c["load_peak_ratio"],
                                "overloaded_joints": c["load_overloaded_joints"],
                            }
                        )
                except Exception as exc:
                    errors += 1
                    rejected.append(
                        {"index": index, "error": f"{type(exc).__name__}: {exc}"}
                    )
        finally:
            if self._load_context is not None:
                self._load_context["_filtering"] = False

        self.last_load_diagnostics.update(
            candidate_count=len(candidates),
            eligible_count=len(eligible),
            load_safe_approximate_count=len(load_safe_pool),
            rejected=rejected,
            pose_valid_count=pose_count,
            physics_errors=errors,
        )
        if not eligible:
            return ([], load_safe_pool) if return_load_safe_pool else []
        minimum = min(c["load_peak_ratio"] for c in eligible)
        policy = self._load_context["policy"]
        tie_limit = minimum + policy.peak_tie
        if self._load_config.get("load_relax_tie_below_early_stop", False):
            tie_limit = max(tie_limit, float(policy.early_stop_ratio))
        shortlist = [
            c
            for c in eligible
            if c["load_peak_ratio"] <= tie_limit
        ]
        self.last_load_diagnostics["peak_shortlist_count"] = len(shortlist)
        if return_load_safe_pool:
            return shortlist, load_safe_pool
        return shortlist

    def solve(
        self,
        T_target,
        frame="local",
        q_prev=None,
        update_prev=True,
        verbose=False,
        allow_approximate=False,
        orientation_mode="full",
        tool_axis=None,
        tool_cone_half_angle=None,
        *,
        joint_state=None,
    ):
        """Original public tuple/formatter retained; only a keyword state input is new."""
        start = time.monotonic()
        if not self._solve_lock.acquire(blocking=False):
            return (
                None,
                "NO_VALID_POSE",
                make_result(
                    {},
                    "NO_VALID_POSE",
                    failure_reason=FailureReason.SOLVER_BUSY,
                    failure_detail="[BUSY] one solve per instance; use a single worker",
                ),
            )
        old_prev = self.q_prev.copy()
        try:
            self.last_load_diagnostics = {
                "version": SOLVER_VERSION,
                "mode": self.load_mode,
                "accepted": False,
                "protection_enabled": self.load_mode == "strict",
                "code": "STARTED",
                "collision_check": "NOT_PERFORMED",
                "dynamic_load_check": "NOT_PERFORMED",
            }
            snapshot = (
                joint_state
                if joint_state is not None
                else self._latest_joint_state
            )
            self._load_context = None
            self._solve_T_root_base = None
            self._load_deadline = None

            if self.load_mode == "strict":
                if self.pin_model is None:
                    return self._load_failure(
                        "MODEL_UNAVAILABLE",
                        "Pinocchio model required for load checks",
                    )
                try:
                    context = prepare_load_context(
                        self.pin_model,
                        self.pin_maps,
                        self._load_config,
                        snapshot,
                    )
                    self._load_context = context
                    self._use_confirmed_efforts(context)
                    self.joint_efforts_active = self.pin_maps["effort_limits"].copy()
                    observed = np.array(
                        [
                            snapshot.positions[n]
                            for n in self.active_joint_names
                        ],
                        dtype=float,
                    )
                    # P1 修复：当调用方未显式指定 q_prev 时，严格模式默认使用实测关节角作为分支与连续性参考
                    if q_prev is None:
                        q_prev = observed.copy()

                    full = active_q_to_pinocchio_q(
                        self.pin_model, self.pin_maps, observed
                    )
                    pin.forwardKinematics(self.pin_model, self.pin_data, full)
                    pin.updateFramePlacements(self.pin_model, self.pin_data)
                    base_fid = (
                        self.pin_base_frame_id
                        if self.pin_base_frame_id is not None
                        else get_pinocchio_frame_id(self.pin_model, self.base_link)
                    )
                    self._solve_T_root_base = np.array(
                        self.pin_data.oMf[base_fid].homogeneous,
                        copy=True,
                    )
                    self.last_load_diagnostics.update(
                        guard_joint_names=context["guard_names"],
                        unchecked_joints=context["unchecked_joints"],
                        model_mass_kg=context["model_mass_kg"],
                        model_warnings=self.pin_maps["model_audit_warnings"],
                        source_timestamp_verified=snapshot.source_age_at_receipt_s
                        is not None,
                        frozen_snapshot_received_monotonic=snapshot.received_monotonic,
                    )
                except Exception as exc:
                    return self._load_failure(
                        "LOAD_CONFIG_OR_STATE_INVALID",
                        f"{type(exc).__name__}: {exc}",
                    )
                budget = float(self._load_config.get("load_budget_s", 0.5))
                if not np.isfinite(budget) or budget <= 0.0:
                    return self._load_failure(
                        "INVALID_BUDGET", "load_budget_s must be positive"
                    )
                self._load_deadline = start + budget
            else:
                self.last_load_diagnostics["code"] = (
                    "LOAD_PROTECTION_EXPLICITLY_DISABLED"
                )
                # P2 修复：load_mode="off" 时，仅在显式配置了 solve_budget_s 时才设置截止时间
                solve_budget = self._load_config.get("solve_budget_s", None)
                if solve_budget is not None:
                    budget = float(solve_budget)
                    if not np.isfinite(budget) or budget <= 0.0:
                        return self._load_failure(
                            "INVALID_BUDGET", "solve_budget_s must be positive"
                        )
                    self._load_deadline = start + budget

            result = self._solve_impl(
                T_target,
                frame,
                q_prev,
                update_prev,
                verbose,
                allow_approximate,
                orientation_mode,
                tool_axis,
                tool_cone_half_angle,
            )
            if (
                self._load_deadline is not None
                and time.monotonic() > self._load_deadline
            ):
                self.q_prev = old_prev
                per_seed = (
                    result[2].per_seed
                    if len(result) > 2 and hasattr(result[2], "per_seed")
                    else None
                )
                return self._load_failure(
                    "TIME_BUDGET_EXCEEDED",
                    "solve deadline exceeded before final acceptance",
                    per_seed=per_seed,
                )
            if self.load_mode == "strict" and result[0] is not None:
                stale = self._stale_context_failure(
                    per_seed=(
                        result[2].per_seed
                        if len(result) > 2 and hasattr(result[2], "per_seed")
                        else None
                    )
                )
                if stale:
                    self.q_prev = old_prev
                    return stale
            return result
        except Exception as exc:
            self.q_prev = old_prev
            return self._load_failure(
                "SOLVE_ERROR", f"{type(exc).__name__}: {exc}"
            )
        finally:
            self.last_load_diagnostics["elapsed_ms"] = (
                time.monotonic() - start
            ) * 1000.0
            self.last_load_diagnostics["budget_exceeded"] = bool(
                self._load_deadline is not None
                and time.monotonic() > self._load_deadline
            )
            if self._load_context is not None:
                self.last_load_diagnostics["search"] = self._load_context.get(
                    "search_diagnostics", {}
                )
            self._load_context = None
            self._solve_T_root_base = None
            if self.pin_maps is not None:
                self._use_urdf_efforts()
                self.joint_efforts_active = self.pin_maps["effort_limits"].copy()
            self._solve_lock.release()

    # ---------------- 面向对象高内聚扩展接口 (Unified OOP Methods) ----------------
    def pinocchio_ik_one_seed(
        self,
        target_T,
        q_init,
        *,
        lower=None,
        upper=None,
        q_ref=None,
        posture_settle=True,
        orientation_mode="full",
        tool_axis=None,
        cone_half_angle=TOOL_CONE_HALF_ANGLE,
        pos_tol=None,
        ori_tol=None,
        max_iter=None,
        damping=None,
        max_step=None,
        margin_bounds=None,
    ):
        """面向对象的单种子 DLS/SNS 迭代接口，自动注入当前求解器的 pin_model, pin_data, pin_maps 及 load_context。"""
        if not self._solve_lock.acquire(blocking=False):
            raise RuntimeError(
                "solver busy; do not share Pinocchio Data concurrently"
            )
        try:
            if self.pin_model is None:
                raise ValueError("Pinocchio model unavailable")
            eff_lower = self.search_lower if lower is None else lower
            eff_upper = self.search_upper if upper is None else upper
            ref = self.q_ready if q_ref is None else q_ref
            char_len = float(self.dls_characteristic_length or 0.5)
            return pinocchio_ik_one_seed(
                self.pin_model,
                self.pin_data,
                self.pin_maps,
                self.pin_ee_frame_id,
                target_T,
                q_init,
                eff_lower,
                eff_upper,
                max_iter=self.dls_max_iter if max_iter is None else max_iter,
                damping=self.dls_damping if damping is None else damping,
                pos_tol=min(PIN_IK_POS_TOL, POS_TOL) if pos_tol is None else pos_tol,
                ori_tol=min(PIN_IK_ORI_TOL, ORI_TOL) if ori_tol is None else ori_tol,
                max_step=self.dls_max_step if max_step is None else max_step,
                orientation_mode=orientation_mode,
                tool_axis=tool_axis,
                tool_cone_half_angle=cone_half_angle,
                q_ref=ref,
                include_waist=self.include_waist,
                waist_release=self.include_waist,
                waist_max_reach=self.dls_waist_max_reach,
                nullspace_gain=self.dls_nullspace_gain,
                waist_null_gain=self.dls_waist_null_gain,
                waist_task_weight=self.dls_waist_task_weight,
                waist_task_weight_far=self.dls_waist_task_weight_far,
                gravity_torque_gain=self.dls_gravity_torque_gain,
                com_xy_gain=self.dls_com_xy_gain,
                joint_limit_gain=self.dls_joint_limit_gain,
                load_context=self._load_context,
                deadline=self._load_deadline,
                characteristic_length=char_len,
                virtual_half_windows=self.dls_virtual_half_windows,
                relative_gravity_gain=self.dls_relative_gravity_gain,
                posture_use_equivalent_branch=self.dls_posture_use_equivalent_branch,
                shoulder_joint=self.s_joint if self.include_waist else None,
                posture_settle=posture_settle,
                margin_bounds=margin_bounds,
            )
        finally:
            self._solve_lock.release()

    def query_seed_library(
        self,
        target_T,
        max_seeds=None,
        bounds=None,
        *,
        orientation_mode="full",
        tool_axis=None,
        cone_half_angle=TOOL_CONE_HALF_ANGLE,
        exclude=(),
    ):
        """面向对象的任务空间种子库查询接口。"""
        if self.seed_library is None or self.pin_model is None:
            return []
        if not self._solve_lock.acquire(blocking=False):
            raise RuntimeError(
                "solver busy; do not share Pinocchio Data concurrently"
            )
        try:
            lo, hi = (
                (self.search_lower, self.search_upper)
                if bounds is None
                else bounds
            )
            count = (
                self.dls_seed_library_seeds
                if max_seeds is None
                else int(max_seeds)
            )
            char_len = float(self.dls_characteristic_length or 0.5)
            T_target_base = seed_library_target(
                self.pin_model,
                self.pin_data,
                self.pin_maps,
                self.seed_library,
                self.q_prev,
                target_T,
            )
            return select_library_seeds(
                self.seed_library,
                T_target_base,
                orientation_mode,
                TOOL_AXIS_X if tool_axis is None else tool_axis,
                normalize_cone_half_angle(cone_half_angle),
                char_len,
                lo,
                hi,
                count,
                exclude=exclude,
                min_separation=self.dls_seed_library_min_separation,
            )
        finally:
            self._solve_lock.release()

    def derive_waist_comfort_bounds(
        self,
        q_ref=None,
        soft_ratio=None,
        allowable_efforts=None,
        gravity_free_half=None,
        max_half=None,
    ):
        """面向对象的腰部舒适窗自适应推导接口。"""
        if not self._solve_lock.acquire(blocking=False):
            raise RuntimeError(
                "solver busy; do not share Pinocchio Data concurrently"
            )
        try:
            if self.pin_model is None:
                raise ValueError("Pinocchio model unavailable")
            ref = self.q_ready if q_ref is None else q_ref
            efforts = (
                self._load_config.get("load_allowable_efforts")
                if allowable_efforts is None
                else allowable_efforts
            )
            soft = (
                self._load_config.get("load_soft_ratio")
                if soft_ratio is None
                else soft_ratio
            )
            return derive_waist_comfort_bounds(
                self.pin_model,
                self.pin_data,
                self.pin_maps,
                ref,
                self.search_lower,
                self.search_upper,
                allowable_efforts=efforts,
                soft_ratio=soft,
                gravity_free_half=gravity_free_half,
                max_half=max_half,
            )
        finally:
            self._solve_lock.release()


# =============================================================================
# 别名映射：与 DualArmIKSolver 形成单臂/双臂对称 API 体系
# =============================================================================
SingleArmIKSolver = HAArmIKSolver

