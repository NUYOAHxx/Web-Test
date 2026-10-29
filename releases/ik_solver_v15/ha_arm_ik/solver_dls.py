#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""ha_arm_ik.solver_dls —— Pinocchio 数值核心：单种子 DLS/SNS 迭代、静态负荷模型、腰部舒适窗、种子库。

约束分层（硬约束不满足 = 不是解；软目标只在硬约束内部比较）：
  硬约束  H1 末端位姿误差 <= pos_tol / ori_tol（full / tool_dir / tool_cone 三种模式）
          H2 关节限位：URDF 限位 ∩ 可选虚拟搜索窗；迭代中由 SNS 钳制保证从不越界
          H3 静态负荷：每个守护关节 |tau_g| / allowable <= 1（strict 模式，solver.py 复核）
          H4 不绕大圈：相对 q_prev 的单关节位移 <= max_joint_jump（solver.py 候选过滤）
  偏好    P1 腰部舒适窗：先只在舒适窗内搜索，窗内没有满足 H1-H4 的解才放开到 H2 限位重试
  软目标  S1 负荷峰值最小（带 peak_tie 容差带），S2 次任务代价最小：
          q_ready 物理坐标回中 + 限位裕度 + 重力矩 + 整机质心水平偏移

单种子迭代 pinocchio_ik_one_seed（每次迭代）：
  1. 主任务 _make_task_terms：位姿残差 e（位置按 characteristic_length 归一）与解析雅可比 J
  2. 逆度量 D（对角）：手臂 = 1；腰部按“肩 -> 目标距离 / 臂展”与腰部偏离 q_ref 平滑释放（无状态，去返程对称）
  3. 次任务 _SecondaryObjective：H(q) 与 z = -∇H；零空间权重随 |e| 衰减。posture_settle=True（默认）时达标后
     放大权重做姿态整理；posture_settle=False（v15 续算）时未达标零空间步限幅为 NO_SETTLE_NULL_STEP_RATIO × |e|，
     第一个同时满足位姿达标、负荷合格、可动关节离限位 >= SAFE_MARGIN 的迭代点即直接返回（POSE_REACHED）
  4. _sns_clamped_core：主动集限位钳制 + 正交信赖域，一步内不越界且 ||dq|| <= max_step（_metric_svd_step 回退）
  5. 回溯线搜索（先合成步、再纯主任务步）、停滞检测、180° 对径死点逃逸、姿态整理与收敛判定
  返回 (q, diag)：q 为位姿达标点中按“峰值容差带 + 次任务代价”择优者（或 POSE_REACHED 首点）；无达标点时为主任务误差最小点

模块内容
  数据结构    JointStateSnapshot（不可变关节快照）、LoadPolicy（负荷门控策略）
  关节映射    build_pinocchio_joint_maps、configure_model_metadata（URDF 审计、掩码、限位）、
              active_q_to_pinocchio_q / pinocchio_q_to_active_q、active_periodic_mask / active_multi_turn_mask
  物理量      model_com_with_jacobian、evaluate_pose_physics、prepare_load_context、load_physics、
              estimate_arm_max_reach
  限位与窗口  _build_effective_bounds、derive_waist_comfort_bounds
  迭代核心    _SecondaryObjective、_sns_clamped_core、_metric_svd_step、_make_task_terms / dls_task_terms、
              pinocchio_ik_one_seed
  种子        make_scatter_seeds、build_seed_library / seed_library_target / select_library_seeds
2πk 分支对齐 / 去重直接用 ik_common 的 select_closest_bounded_branch、bounded_joint_difference、joint_difference。
测试专用的旧入口（整函数形式的次任务梯度与 SNS 步）在 tests/dls_compat.py。版本演进见 README_v14.md 与 README_v15.md。
"""

from dataclasses import dataclass, field, replace
import math
import time
from types import MappingProxyType
from typing import Mapping, Sequence
import xml.etree.ElementTree as ET

import numpy as np
from scipy.linalg import lapack as _lapack

try:
    import pinocchio as pin
except Exception:
    pin = None

from ha_arm_ik.ik_base import (
    SAFE_MARGIN,
    TOOL_AXIS_X,
    WAIST_JOINT_NAMES,
    bounded_joint_difference,
    clip_to_bounds,
    joint_difference,
    normalize_cone_half_angle,
    select_closest_bounded_branch,
    skew,
    unique_seed_list,
    validate_orientation_mode,
)

# =============================================================================
# 1. 基础数值与收敛容差参数
# =============================================================================
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
# =============================================================================
GRAVITY_TORQUE_GAIN = 0.04        # 默认绝对归一化重力矩卸载增益
RELATIVE_GRAVITY_GAIN = 0.0       # 可选的相对参考姿态重力矩保持增益
COM_XY_GAIN = 0.12                # 零空间整机质心(CoM)水平投影稳定增益（代价 0.5 * 0.12 * |Δc|^2，Δc 单位 m）
JOINT_LIMIT_GAIN = 0.03           # 限位排斥增益；排斥区宽度 = max(SAFE_MARGIN, 0.02) rad
WAIST_YAW_VIRTUAL_LIM = np.deg2rad(90.0)   # v8：重力无关腰关节（偏航类）的舒适半窗，见 derive_waist_comfort_bounds

# =============================================================================
# 5. 腰部舒适窗（v8，详见 derive_waist_comfort_bounds）
# =============================================================================
WAIST_COMFORT_SOFT_RATIO = 0.70   # 未配置 load_soft_ratio 时的默认值（与 LoadPolicy.soft_ratio 默认一致）
WAIST_COMFORT_MAX_HALF = 0.5 * np.pi       # 任何腰关节的舒适半窗不超过 90°（躯干最多转四分之一圈）
WAIST_GRAVITY_FREE_COS = math.cos(math.radians(30.0))  # 关节轴与重力夹角 < 30° 视为重力无关（偏航类）
WAIST_COMFORT_EDGE_SEEDS = 1      # 舒适窗内无合格解时，取窗内主任务误差最小的几个迭代终点作为放开后的续算种子
WAIST_COMFORT_CONTINUATION_STEP = 0.10   # 续算判据：目标相对 FK(q_prev) 位置差 <= 0.1 × 臂展且姿态差 <= 0.1 rad
                                         # （约 0.1 rad 关节运动，雅可比线性化仍有效）时不加舒适窗

# =============================================================================
# v13：任务空间种子库（最近邻种子）
# =============================================================================
SEED_LIBRARY_SIZE = 16384         # 初始化时在有效限位内均匀采样的关节角个数（建库约 150 ms、2.8 MB）；0 = 关闭
SEED_LIBRARY_SEEDS = 3            # 每个常规搜索阶段插入的种子库种子个数（排在 q_prev / 锁腰 / q_ready 之后）
SEED_LIBRARY_MIN_SEPARATION = 0.5 # 入选种子之间、与本阶段已排队种子之间的关节差 ∞-范数下限（rad）
SEED_LIBRARY_POOL = 64            # 按任务空间距离预选的候选个数（再按关节差去重）
SEED_LIBRARY_RNG_SEED = 13        # 采样随机数种子（与求解用的 RNG_SEED 无关，库内容可复现）


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
    eff_damping = damping + min(1e-3, 1e-4 * float(err @ err))
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
    eff_damping = damping + min(1e-3, 1e-4 * float(err @ err))

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


def prepare_load_context(model, pin_maps, config, snapshot):
    """构建严格、不可变的全身静态负荷上下文；绝不虚构关节状态或默认力矩上限。"""
    if pin_maps.get("model_audit_errors"):
        raise ValueError(
            "model audit: " + "; ".join(pin_maps["model_audit_errors"])
        )
    if not isinstance(snapshot, JointStateSnapshot):
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