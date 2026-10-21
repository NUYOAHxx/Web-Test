#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Pinocchio 阻尼最小二乘 IK 核心（merged-v6）。

约束分层（硬约束不满足 = 不是解；软目标只在硬约束内部比较）：
  硬约束  H1 末端位姿误差 <= pos_tol / ori_tol（full / tool_dir / tool_cone 三种模式）
          H2 关节限位：URDF 限位 ∩ 可选虚拟搜索窗；迭代中由 SNS 钳制保证从不越界
          H3 静态负荷：每个守护关节 |tau_g| / allowable <= 1（strict 模式，solver.py 复核）
          H4 不绕大圈：相对 q_prev 的单关节位移 <= max_joint_jump（solver.py 候选过滤）
  软目标  S1 负荷峰值最小（带 peak_tie 容差带），S2 次任务代价最小：
          q_ready 物理坐标回中 + 限位裕度 + 重力矩 + 整机质心水平偏移

v6 相对 v5 的修正：
  1. 整机质心计入 universe 惯量（URDF 根 link 及固定在根上的 link），nv=1 时雅可比形状正确
  2. SNS Cholesky 主路径的零空间步与 SVD 回退一致：D z - D J^T A^-1 J D z（保证下降方向）
  3. 位姿达标迭代点的择优改为“峰值容差带 + 代价”，不再因峰值差 0.001 丢弃姿态改善
  4. 180° 对径死点按雅可比实际可达方向选切向，单轴/少轴情况也能逃逸
  5. 主任务停滞检测：窗口内误差下降不足即换种子，避免不可达种子空转到迭代上限
  6. 线搜索边界舍入吸附、回溯内超时记为 TIME_BUDGET、负荷向量缓存与关节映射向量化
"""

from dataclasses import dataclass, field, replace
import math
import time
from types import MappingProxyType
from typing import Mapping, Sequence
import xml.etree.ElementTree as ET

import numpy as np
from scipy.linalg import cho_factor, cho_solve

try:
    import pinocchio as pin
except Exception:
    pin = None

from ha_arm_ik.ik_common import (
    SAFE_MARGIN,
    TOOL_AXIS_X,
    clip_to_bounds,
    closest_direction_on_cone,
    normalize,
    normalize_cone_half_angle,
    skew,
    unique_seed_list,
    validate_orientation_mode,
)
from ha_arm_ik.robot_model import WAIST_JOINT_NAMES

# =============================================================================
# 1. 基础数值与收敛容差参数
# =============================================================================
PIN_IK_MAX_ITER = 500
PIN_IK_DAMPING = 1e-6
PIN_IK_POS_TOL = 5e-4
PIN_IK_ORI_TOL = 5e-4
PIN_IK_MAX_STEP = 0.5             # 单次迭代关节步长 2-范数上限（弧度）
PIN_IK_STALL_ITERS = 25           # 停滞检测窗口：位姿未达标时每 25 次迭代检查一次进展
PIN_IK_STALL_RTOL = 0.05          # 窗口内主任务误差范数相对下降 < 5% -> 判定停滞，换下一个种子
PIN_IK_STALL_DELTA_M = 5e-5       # 兼容旧版导出的常量（v6 改用上面的相对窗口判据）
BOUND_SNAP_TOL = 1e-9             # 试探点越界不超过此值（浮点舍入）时吸附回边界，不整步拒绝
PIN_IK_MAX_VALID_CANDIDATES = 2
WAIST_RETURN_TOL = 0.10           # 外层早停的腰部回中阈值（弧度）
N_NOISE_PER_BASE = 16
# 扰动种子标准差（rad），从大到小：前几个扰动种子先探索不同解族（提高成功率、找低负荷解），
# 后面的小扰动做局部细化。v5 为 [0.02, 0.05, 0.10, 0.20] 且先排完 q_prev 的 16 个小扰动，
# max_seeds=10 时扰动种子几乎与 q_prev 重合，远目标成功率 97% -> v6 100%（合成基准 100 例）。
NOISE_SCALES = [0.60, 0.30, 0.15, 0.05]

# =============================================================================
# 2. 零空间防漂移与流形稳态收敛参数
# =============================================================================
NULLSPACE_GAIN = 0.30             # 手臂冗余关节拉向 q_ref 物理坐标的次任务增益
WAIST_NULL_GAIN = 0.50            # 腰部关节拉向 q_ref（直立零位）的次任务增益
NULLSPACE_ERR_SCALE = 0.06        # 主任务误差小于此值时零空间整理全功率激活
POSTURE_SETTLE_TOL = 1e-4         # 末端收敛后，零空间步长小于此阈值(rad)才算稳态收敛
MAX_SETTLE_EXTRA_ITERS = 50       # 末端达标后允许继续在零空间整理姿态的最大步数
SETTLE_COST_RTOL = 1e-3           # 姿态整理阶段：单步次任务代价相对下降 < 0.1% 记为“慢步”
SETTLE_PATIENCE = 3               # 连续 3 个慢步即视为已收敛（边际收益太小，不再耗时）
POSE_PEAK_BAND_FRACTION = 0.25    # 单种子内择优的峰值容差带 = 0.25 * LoadPolicy.peak_tie（默认 0.005）

# 兼容旧版外部导入的别名常量
WAIST_POST_GAIN = 0.30
WAIST_POST_ITERS = MAX_SETTLE_EXTRA_ITERS
WAIST_POST_STEP = 0.05
WAIST_POST_TOL = POSTURE_SETTLE_TOL

# =============================================================================
# 3. 无状态腰部自适应释放参数（路径无关，去返程完全对称）
# =============================================================================
WAIST_TASK_WEIGHT = 6.0           # 近距离时腰部关节空间代价（手臂 = 1.0，锁死稳重心）
WAIST_TASK_WEIGHT_FAR = 1.0       # 远目标释放后的腰部权重（与手臂平权协同，防过度弯腰）
WAIST_RELEASE_START = 0.75        # 肩->目标距离/最大臂展 达到此比例开始放腰
WAIST_RELEASE_END = 0.94          # 达到此比例腰完全释放
WAIST_RETURN_RELEASE_RAD = 0.20   # 返程解锁尺度(rad)：腰偏离 q_ref 时保持释放通道以便退回

# =============================================================================
# 4. 电机重力矩卸载、重心稳定与限位保护参数
# =============================================================================
GRAVITY_TORQUE_GAIN = 0.04        # 默认绝对归一化重力矩卸载增益
RELATIVE_GRAVITY_GAIN = 0.0       # 可选的相对参考姿态重力矩保持增益
COM_XY_GAIN = 0.12                # 零空间整机质心(CoM)水平投影稳定增益
JOINT_LIMIT_GAIN = 0.03           # 单侧安全裕度(SAFE_MARGIN)排斥增益
WAIST_YAW_VIRTUAL_LIM = np.deg2rad(90.0)   # 可选虚拟窗口参考常量：腰部偏航
ARM_ROLL_VIRTUAL_LIM = np.deg2rad(150.0)   # 可选虚拟窗口参考常量：手臂自转
MAX_SNS_CLAMP_PASSES = 10         # 主动集钳制默认上限（实际按 n_free + 1 动态决定）


# =============================================================================
# 基础数值校验、周期/多圈分支处理与数据结构
# =============================================================================
def finite_vector(value, size=None, name="vector"):
    result = np.asarray(value, dtype=float)
    if result.ndim != 1 or (size is not None and result.size != size):
        raise ValueError(f"{name}: expected a 1-D vector of size {size}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name}: non-finite values")
    return result.copy()


def validate_transform(value):
    T = np.asarray(value, dtype=float)
    if T.shape != (4, 4) or not np.all(np.isfinite(T)):
        raise ValueError("target must be a finite 4x4 transform")
    rot = T[:3, :3]
    if (
        not np.allclose(T[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9, rtol=0.0)
        or not np.allclose(rot.T @ rot, np.eye(3), atol=1e-6, rtol=0.0)
        or not np.isclose(np.linalg.det(rot), 1.0, atol=1e-6, rtol=0.0)
    ):
        raise ValueError("target rotation/bottom row is not a rigid transform")
    return T.copy()


def joint_difference(a, b, continuous):
    """计算关节角差值 a - b；仅对显式标记为 True 的无界 continuous 旋转关节折叠到 [-π, π)。"""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    d = d.copy()
    mask = np.asarray(continuous, dtype=bool)
    if np.any(mask):
        d[mask] = (d[mask] + np.pi) % (2.0 * np.pi) - np.pi
    return d


def select_closest_bounded_branch(
    q, reference, lower, upper, multi_turn_mask=None
):
    """在满足 lower <= q + 2πk <= upper (k ∈ Z) 的合法同伦分支中，
    为显式标记在 multi_turn_mask 中的旋转关节选择最接近 reference 的等价关节角。

    注意：当 multi_turn_mask 为 None 时，绝不根据限位跨度猜测周期性（直接返回原值拷贝），
    以防止对无界或大跨度 prismatic 平移关节误执行模 2π 运算。
    """
    q_arr = finite_vector(q, name="q")
    ref_arr = finite_vector(reference, len(q_arr), name="reference")
    if multi_turn_mask is None:
        return q_arr.copy()

    mask = np.asarray(multi_turn_mask, dtype=bool)
    if mask.shape != q_arr.shape:
        raise ValueError("multi_turn_mask shape mismatch")
    if not np.any(mask):
        return q_arr.copy()

    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    out = q_arr.copy()
    two_pi = 2.0 * np.pi
    eps = 1e-9

    for i in np.nonzero(mask)[0]:
        qi = float(q_arr[i])
        ri = float(ref_arr[i])
        li = float(lo[i])
        ui = float(hi[i])

        if not np.isfinite(li) and not np.isfinite(ui):
            out[i] = ri + ((qi - ri + np.pi) % two_pi - np.pi)
            continue

        k_min = (
            math.ceil((li - eps - qi) / two_pi)
            if np.isfinite(li)
            else round((ri - qi) / two_pi) - 2
        )
        k_max = (
            math.floor((ui + eps - qi) / two_pi)
            if np.isfinite(ui)
            else round((ri - qi) / two_pi) + 2
        )

        best_val = qi
        best_dist = abs(qi - ri) if (li - eps <= qi <= ui + eps) else math.inf
        for k in range(int(k_min), int(k_max) + 1):
            cand = qi + k * two_pi
            if li - eps <= cand <= ui + eps:
                cand_clipped = float(np.clip(cand, li, ui))
                dist = abs(cand_clipped - ri)
                if dist < best_dist - 1e-12:
                    best_dist = dist
                    best_val = cand_clipped
        out[i] = best_val

    return out


def bounded_joint_difference(a, b, lower, upper, multi_turn_mask=None):
    """在考虑有限限位内合法 2πk 等价分支的前提下，计算 a 相对于 b 的最小等价差（用于候选解去重）。"""
    a_branch = select_closest_bounded_branch(
        a, b, lower, upper, multi_turn_mask=multi_turn_mask
    )
    return a_branch - np.asarray(b, dtype=float)


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
    early_stop_ratio: float = 0.60
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

    def cost_gradient(self, tau, d_tau_d_active, names, *, tau_reference=None):
        tau = finite_vector(tau, len(names), "gravity efforts")
        J = np.asarray(d_tau_d_active, dtype=float)
        if J.ndim != 2 or J.shape[0] != len(names) or not np.all(np.isfinite(J)):
            raise ValueError("invalid gravity effort derivative")
        caps, weights = self.vectors(names)
        r = tau / caps
        excess = np.maximum(np.abs(r) - self.soft_ratio, 0.0)
        n = len(names)

        abs_cost = 0.5 * self.absolute_weight * float(np.sum(weights * r * r)) / n
        excess_cost = (
            0.5 * self.excess_weight * float(np.sum(weights * excess * excess)) / n
        )
        dcost_dtau = (
            self.absolute_weight * weights * r / caps
            + self.excess_weight * weights * excess * np.sign(r) / caps
        ) / n

        rel_cost = 0.0
        if self.relative_weight > 0.0:
            ref = finite_vector(
                tau_reference, len(names), "reference gravity efforts"
            )
            diff = (tau - ref) / caps
            rel_cost = (
                0.5 * self.relative_weight * float(np.sum(weights * diff * diff)) / n
            )
            dcost_dtau += self.relative_weight * weights * diff / caps / n

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
                pin_maps.get("active_multi_turn", pin_maps["active_continuous"]),
            )
        else:
            mask = pin_maps["active_continuous"]
            q[mask] = ref[mask] + joint_difference(q, ref, mask)[mask]
    return q


def estimate_arm_max_reach(pin_model, pin_maps, ee_frame_id):
    """从 Pinocchio 模型解析肩->TCP 的最大几何距离。"""
    arm_names = [
        n for n in pin_maps["active_names"] if n not in WAIST_JOINT_NAMES
    ]
    if not arm_names:
        return None

    side_prefix = (
        "LHand" if any(n.startswith("LHand") for n in arm_names) else "RHand"
    )
    reach = 0.0
    for link_idx in (3, 5):
        name = f"{side_prefix}_Link{link_idx}"
        if name not in pin_maps["name_to_joint_id"]:
            return None
        jid = pin_maps["name_to_joint_id"][name]
        seg = float(np.linalg.norm(pin_model.jointPlacements[jid].translation))
        if seg <= 1e-9:
            return None
        reach += seg

    frame = pin_model.frames[ee_frame_id]
    tcp_off = float(np.linalg.norm(frame.placement.translation))
    return reach + tcp_off


def model_com_with_jacobian(
    model, data, q_full, mass_points=None, active_v=None, derivatives=True
):
    """整机质心（URDF 根坐标系）及其对活跃关节的雅可比，含固定在根上的质量。

    Pinocchio 的 centerOfMass / jacobianCenterOfMass 只累加 joint 1..n 的子树质量，
    不含 model.inertias[0]（URDF 根 link 以及经 fixed joint 挂在根上的 link）。
    这部分质量 m0 不随 q 运动，质心为 c0（根坐标系），因此：
        com   = (M_mov * c_mov + m0 * c0) / (M_mov + m0)
        J_com = M_mov / (M_mov + m0) * J_mov
    与 configure_model_metadata 统计的 model_mass_kg（全部 link 质量）保持一致。
    mass_points 参数仅为兼容旧调用保留，计算不再使用。
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
        pin_model,
        pin_data,
        full,
        pin_maps.get("mass_points"),
        active_v,
        derivatives=False,
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
            pin_maps.get("active_periodic", pin_maps["active_continuous"]),
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


def _build_effective_bounds(
    lower,
    upper,
    q_ref,
    _legacy_waist_mask=None,
    virtual_half_windows=None,
    active_names=None,
):
    """构建有效关节限位：当显式配置 virtual_half_windows 时，作为硬搜索窗口与真实限位求交。

    第 4 个位置参数是旧版的腰部掩码，仅为兼容既有位置参数调用而保留，不参与计算：
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


def _compute_nullspace_gradient_and_cost(
    pin_model,
    pin_data,
    q_full,
    q_active_cur,
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
    pin_maps=None,
    derivatives=True,
    components=None,
    relative_gravity_gain=0.0,
    posture_use_equivalent_branch=False,
):
    """计算零空间负梯度 z = -dH/dq 与标量代价 H(q)。

    关键设计：
      - 默认 posture_use_equivalent_branch=False：对所有有限限位 revolute 关节（包括 ±360° 的腰部与臂自转），
        使用真实编码器坐标差 (q_active_cur - q_ref)，确保零空间始终将关节拉回真实的物理 q_ready（如 0°），
        绝不会因为 200° 等价于 -160° 而把关节反向推向 +360°。
      - 仅当显式设置 posture_use_equivalent_branch=True 时，才使用 2πk 等价角度差。
    """
    if (
        posture_use_equivalent_branch
        and pin_maps is not None
        and "active_multi_turn" in pin_maps
    ):
        diff = bounded_joint_difference(
            q_active_cur,
            q_ref,
            lower,
            upper,
            pin_maps["active_multi_turn"],
        )
    else:
        periodic = (
            (
                pin_maps.get("active_periodic", pin_maps["active_continuous"])
                if pin_maps is not None
                else np.zeros(len(q_ref), dtype=bool)
            )
            if load_context is None
            else load_context["periodic"]
        )
        diff = joint_difference(q_active_cur, q_ref, periodic)

    z = -ns_gain * diff
    posture = 0.5 * float(np.dot(ns_gain * (diff ** 2), movable_mask))
    cost = posture
    parts = {
        "posture": posture,
        "joint_limit": 0.0,
        "absolute_load": 0.0,
        "excess_load": 0.0,
        "relative_load": 0.0,
        "com": 0.0,
    }

    # 1. 单侧安全裕度排斥项
    margin = max(float(SAFE_MARGIN), 0.02)
    if limit_gain > 0.0:
        b_low = np.minimum(q_ref, lower + margin)
        b_high = np.maximum(q_ref, upper - margin)
        under = np.maximum(0.0, b_low - q_active_cur) / margin
        over = np.maximum(0.0, q_active_cur - b_high) / margin
        z += limit_gain * (under - over) / margin
        c_lim = (
            0.5
            * limit_gain
            * float(np.dot(under ** 2 + over ** 2, movable_mask))
        )
        cost += c_lim
        parts["joint_limit"] = c_lim

    # 2. 静态重力矩与质心目标
    if load_context is not None:
        physics = load_physics(
            pin_model,
            pin_data,
            pin_maps,
            q_active_cur,
            load_context,
            derivatives,
        )
        c_load, g_load, load_parts = load_context["policy"].cost_gradient(
            physics["tau"],
            physics["d_tau_d_active"],
            load_context["guard_names"],
            tau_reference=tau_g_ref,
        )
        if derivatives:
            z -= g_load
        cost += c_load
        # peak_ratio / load_ok 随代价一并返回，调用方择优时无需再算一次重力矩
        parts.update(
            absolute_load=load_parts["absolute"],
            excess_load=load_parts["excess"],
            relative_load=load_parts["relative"],
            peak_ratio=physics["peak_ratio"],
            load_ok=physics["load_ok"],
        )
        if include_waist and com_gain > 0.0 and com_xy_ref is not None:
            ref_com = finite_vector(com_xy_ref, 3, "reference COM")
            ghat = np.asarray(pin_model.gravity.linear, dtype=float)
            ghat = ghat / max(float(np.linalg.norm(ghat)), 1e-30)
            P = np.eye(3) - np.outer(ghat, ghat)
            dcom = P @ (physics["com"] - ref_com)
            if derivatives:
                z -= com_gain * (physics["Jcom"].T @ dcom)
            parts["com"] = 0.5 * com_gain * float(dcom @ dcom)
            cost += parts["com"]
    else:
        valid_eff = (
            effort_limits is not None
            and np.all(np.isfinite(effort_limits))
            and np.all(effort_limits > 0.0)
        )
        if valid_eff and (gravity_gain > 0.0 or relative_gravity_gain > 0.0):
            tau_full = np.array(
                pin.computeGeneralizedGravity(pin_model, pin_data, q_full),
                copy=True,
            )
            tau_cur = tau_full[active_v_indices]
            inv_eff_sq = 1.0 / (effort_limits ** 2)
            n_act = max(1, len(active_v_indices))

            dcost_dtau = np.zeros_like(tau_cur)
            if gravity_gain > 0.0:
                abs_c = (
                    0.5
                    * gravity_gain
                    * float(np.sum((tau_cur ** 2) * inv_eff_sq))
                    / n_act
                )
                cost += abs_c
                parts["absolute_load"] = abs_c
                dcost_dtau += gravity_gain * (tau_cur * inv_eff_sq) / n_act

            if relative_gravity_gain > 0.0 and tau_g_ref is not None:
                tau_diff = tau_cur - tau_g_ref
                rel_c = (
                    0.5
                    * relative_gravity_gain
                    * float(np.sum((tau_diff ** 2) * inv_eff_sq))
                    / n_act
                )
                cost += rel_c
                parts["relative_load"] = rel_c
                dcost_dtau += (
                    relative_gravity_gain * (tau_diff * inv_eff_sq) / n_act
                )

            if derivatives:
                dtau_dq_full = np.asarray(
                    pin.computeGeneralizedGravityDerivatives(
                        pin_model, pin_data, q_full
                    )
                ).reshape(pin_model.nv, pin_model.nv)
                dtau_act = dtau_dq_full[
                    np.ix_(active_v_indices, active_v_indices)
                ]
                z -= dtau_act.T @ dcost_dtau

        if include_waist and com_gain > 0.0 and com_xy_ref is not None:
            com_cur, J_com = model_com_with_jacobian(
                pin_model,
                pin_data,
                q_full,
                active_v=active_v_indices,
                derivatives=derivatives,
            )
            ref_com = finite_vector(com_xy_ref, 3, "reference COM")
            ghat = np.asarray(pin_model.gravity.linear, dtype=float)
            ghat = ghat / max(float(np.linalg.norm(ghat)), 1e-30)
            P = np.eye(3) - np.outer(ghat, ghat)
            dcom = P @ (com_cur - ref_com)
            parts["com"] = 0.5 * com_gain * float(dcom @ dcom)
            cost += parts["com"]
            if derivatives:
                z -= com_gain * (J_com.T @ dcom)

    if not np.isfinite(cost) or not np.all(np.isfinite(z)):
        raise ValueError("non-finite secondary objective")
    if components is not None:
        components.update(parts)
    return z * movable_mask, cost


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


def _solve_sns_clamped_step(
    J_active,
    err,
    z_grad,
    null_weight,
    W_inv,
    movable_mask,
    q_active_cur,
    eff_lower,
    eff_upper,
    damping,
    max_step,
    *,
    return_info=False,
):
    """主动集限位钳制（SNS）+ 正交信赖域：求一步既不越界、步长 <= max_step 的关节增量。

    记 D = W_inv（逆度量，关节权重越大 D 越小、该关节越“贵”），A = J_f D_f J_f^T + λI：
      任务步   dq_task = D_f J_f^T A^-1 e_rem
      零空间步 dq_null = w · (D_f z - D_f J_f^T A^-1 J_f D_f z)
    其中 e_rem = e - J dq_clamped 是扣除已钳制关节位移后的剩余误差，z = -∇H 为次任务负梯度。
    零空间投影 D - D J^T A^-1 J D 对称半正定，所以 z^T dq_null >= 0，次任务一定是下降方向
    （与 SVD 回退 _metric_svd_step 的 S Z Z^T S z 一致）。

    循环：每轮把越界最严重的一个自由关节钳制到边界（位移 = 边界 - 当前值，此后不再缩放），
    其余自由关节按剩余信赖域半径 sqrt(max_step^2 - ||dq_clamped||^2) 缩放；
    被钳制关节的越界距离严格小于其本轮位移，故 ||dq_clamped|| < max_step 恒成立。
    至多 n_free + 1 轮；返回的 clamped_converged=False 表示轮数用尽仍有越界（调用方记诊断）。
    """
    if damping < 0.0 or max_step <= 0.0:
        raise ValueError("invalid damping/max_step")

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
        W_inv_free = W_inv[np.ix_(free_mask, free_mask)]

        A = J_free @ W_inv_free @ J_free.T
        A.flat[:: task_dim + 1] += eff_damping

        try:
            chol_fac = cho_factor(A, check_finite=False)
            rhs = cho_solve(chol_fac, err_rem, check_finite=False)
            dq_task_free = W_inv_free @ (J_free.T @ rhs)
            dq_null_free = np.zeros_like(dq_task_free)
            if z_grad is not None and null_weight > 0.0:
                # 度量一致的零空间下降步：w (D z - D J^T A^-1 J D z)
                Dz = W_inv_free @ z_grad[free_mask]
                invA_JDz = cho_solve(chol_fac, J_free @ Dz, check_finite=False)
                dq_null_free = null_weight * (
                    Dz - W_inv_free @ (J_free.T @ invA_JDz)
                )
        except (np.linalg.LinAlgError, ValueError):
            dq_task_free, sec_free = _metric_svd_step(
                J_free,
                err_rem,
                None if z_grad is None else z_grad[free_mask],
                W_inv_free,
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

        if np.max(viol_amt) <= 1e-12:
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

    if return_info:
        return dq_total, dq_null_out, clamped_converged
    return dq_total, dq_null_out


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
    """计算主任务残差与解析雅可比（含一致的 tool_dir 180° 对径点方向导数 -skew(a) @ J[3:]）。"""
    q_full = active_q_to_pinocchio_q(model, pin_maps, q_active)
    pin.forwardKinematics(model, data, q_full)
    pin.updateFramePlacements(model, data)
    current = data.oMf[fid].copy()
    pos_error = target[:3, 3] - np.asarray(current.translation)
    pos = float(np.linalg.norm(pos_error))
    v_idx = pin_maps["active_v_indices"]

    if jacobian:
        J = np.asarray(
            pin.computeFrameJacobian(
                model,
                data,
                q_full,
                fid,
                pin.LOCAL if mode == "full" else pin.LOCAL_WORLD_ALIGNED,
            )
        ).reshape(6, model.nv)[:, v_idx]
    else:
        J = np.zeros((6, len(v_idx)), dtype=float)

    antipodal = False
    if mode == "full":
        delta = current.inverse() * pin.SE3(target[:3, :3], target[:3, 3])
        e = np.array(pin.log6(delta).vector, copy=True)
        angle = float(np.linalg.norm(e[3:]))
        A = (
            np.array(pin.Jlog6(delta.inverse()) @ J, copy=True)
            if jacobian
            else J.copy()
        )
        e[:3] /= characteristic_length
        A[:3] /= characteristic_length
        violation = angle
    else:
        a = np.asarray(current.rotation) @ axis
        b = target[:3, :3] @ axis
        cross = np.cross(a, b)
        sine = float(np.linalg.norm(cross))
        cosine = float(np.clip(a @ b, -1.0, 1.0))
        angle = math.atan2(sine, cosine)

        if mode == "tool_dir":
            edir = b - a
            Jdir = -skew(a) @ J[3:]          # 行：工具轴方向 a 的变化率 da/dq
            if cosine < -1.0 + 1e-8:
                tangent = _antipodal_tangent(a, Jdir if jacobian else None)
                edir = angle * tangent
                antipodal = True
            e = np.r_[pos_error / characteristic_length, edir]
            A = np.vstack([J[:3] / characteristic_length, Jdir])
            violation = angle
        elif mode == "tool_cone":
            violation = max(0.0, angle - cone)
            row = np.zeros(len(v_idx), dtype=float)
            if violation > 0.0:
                if sine > 1e-9:
                    tangent = cross / sine       # 把 a 转向 b 的转轴
                else:
                    # 对径：任意垂直于 a 的转轴都能减小夹角，取雅可比上最容易转动的那一个
                    tangent = _antipodal_tangent(a, J[3:] if jacobian else None)
                    antipodal = True
                row = tangent @ J[3:]
            e = np.r_[pos_error / characteristic_length, violation]
            A = np.vstack([J[:3] / characteristic_length, row])
        else:
            raise ValueError("invalid orientation mode")

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
):
    """单种子 DLS IK（merged-v6）：主任务为末端位姿，零空间次任务为回中/限位裕度/负荷/质心。

    返回 (q, diag)。q 优先取“位姿达标点”中按负荷峰值容差带 + 次任务代价择优的那个；
    没有达标点时返回主任务误差最小的迭代点（diag["converged"] = False）。
    迭代全程满足有效限位（URDF 限位 ∩ virtual_half_windows），由 SNS 钳制 + 边界吸附保证。
    diag["stop_reason"]：SETTLED / SETTLE_LIMIT / SMALL_STEP / STALLED_NO_PROGRESS /
    LINE_SEARCH_STALLED / TIME_BUDGET / ITERATION_LIMIT。
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
        waist,
        virtual_half_windows,
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
            pin.centerOfMass(pin_model, pin_data, ref_full, False)
            ref_com = np.array(pin_data.com[0], dtype=float, copy=True)

    shoulder_id = next(
        (
            pin_maps["name_to_joint_id"][n]
            for n in pin_maps["active_names"]
            if n not in WAIST_JOINT_NAMES and n.endswith("_Link2")
        ),
        None,
    )
    shoulder_ref = (
        None
        if shoulder_id is None
        else np.array(pin_data.oMi[shoulder_id].translation, copy=True)
    )
    if waist_release and waist_max_reach is None:
        waist_max_reach = estimate_arm_max_reach(
            pin_model, pin_maps, ee_frame_id
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
        t_val = np.clip(
            (distance - WAIST_RELEASE_START * waist_max_reach)
            / max(
                (WAIST_RELEASE_END - WAIST_RELEASE_START) * waist_max_reach,
                1e-6,
            ),
            0.0,
            1.0,
        )
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
    periodic_unbounded = pin_maps.get(
        "active_periodic", pin_maps["active_continuous"]
    )
    D_base = np.diag(1.0 / weights)
    waist_idx = np.nonzero(waist)[0]
    has_waist_dof = bool(np.any(waist & movable))

    def objective(values, q_full_val, derivatives, parts=None):
        return _compute_nullspace_gradient_and_cost(
            pin_model,
            pin_data,
            q_full_val,
            values,
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
            pin_maps=pin_maps,
            derivatives=derivatives,
            components=parts,
            relative_gravity_gain=rel_grav_gain,
            posture_use_equivalent_branch=posture_use_equivalent_branch,
        )

    def terms(values, jac=True):
        return dls_task_terms(
            pin_model,
            pin_data,
            pin_maps,
            ee_frame_id,
            values,
            target,
            mode,
            axis,
            cone,
            characteristic_length,
            jac,
        )

    def is_pose(t_dict):
        return t_dict["pos_err"] <= pos_tol and t_dict["ori_violation"] <= ori_tol

    def snap(trial):
        """试探点只允许浮点舍入级越界（<= BOUND_SNAP_TOL），吸附回边界；真实越界返回 None。"""
        viol = max(float(np.max(lo - trial)), float(np.max(trial - hi)))
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

    for it in range(int(max_iter)):
        if deadline is not None and time.monotonic() >= deadline:
            diag["stop_reason"] = "TIME_BUDGET"
            break

        t = terms(q, jac=True)
        shoulder_cur = (
            np.array(pin_data.oMi[shoulder_id].translation, copy=True)
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

        # 逆度量 D：手臂 = 1；腰部在近距离目标时“贵”(1/near_weight)，远目标或回程时逐步释放
        D = D_base.copy()
        if has_waist_dof:
            alpha = 1.0 if waist_load_release else 0.0
            if release_enabled and shoulder_cur is not None:
                current_distance = float(
                    np.linalg.norm(target[:3, 3] - shoulder_cur)
                )
                waist_diff = joint_difference(q, ref, periodic_unbounded)[
                    waist & movable
                ]
                ret = float(
                    np.linalg.norm(waist_diff)
                    / max(WAIST_RETURN_RELEASE_RAD, 1e-6)
                )
                alpha = max(
                    alpha,
                    ref_release,
                    release_fraction(current_distance),
                    float(np.clip(ret, 0.0, 1.0)),
                )
            D[waist_idx, waist_idx] = (
                1.0 - alpha
            ) / near_weight + alpha / far_weight

        # 主任务误差越小，零空间次任务权重越大（达标后再放大 1.5 倍做姿态整理）
        null_weight = (1.5 if pose_ok else 1.0) / (
            1.0
            + (
                float(np.linalg.norm(t["error"]))
                / max(NULLSPACE_ERR_SCALE, 1e-6)
            )
            ** 2
        )

        dq, dq_null, clamped_ok = _solve_sns_clamped_step(
            t["J"],
            t["error"],
            z,
            null_weight,
            D,
            movable,
            q,
            lo,
            hi,
            damping,
            max_step,
            return_info=True,
        )
        diag["sns_clamp_incomplete"] |= not clamped_ok

        # 末端已达标且零空间步长收敛：最后试一次整步，仍达标且择优通过才采纳
        if pose_ok and float(np.linalg.norm(dq_null)) < POSTURE_SETTLE_TOL:
            trial = snap(q + dq)
            if trial is not None:
                tt = terms(trial, jac=False)
                if is_pose(tt):
                    tparts = {}
                    trial_cost = objective(trial, tt["q_full"], False, tparts)[1]
                    if offer_pose(trial, trial_cost, tparts):
                        q = trial
            diag["stop_reason"] = "SETTLED"
            break

        if float(np.linalg.norm(dq)) < 1e-10:
            diag["stop_reason"] = "SMALL_STEP"
            break

        # 回溯线搜索（严格单调）：先试“主任务 + 零空间”合成步，失败再试纯主任务步（按需才求解）
        directions = [dq]
        need_pure = (not pose_ok) and float(np.linalg.norm(dq_null)) > 1e-12
        accepted = False
        d_idx = 0
        while d_idx < len(directions) and not accepted and not timed_out:
            direction = directions[d_idx]
            for backtrack in range(int(backtrack_steps)):
                if deadline is not None and time.monotonic() >= deadline:
                    timed_out = True
                    break
                trial = snap(q + (0.5 ** backtrack) * direction)
                if trial is None:
                    continue
                tt = terms(trial, jac=False)
                if pose_ok:
                    # 姿态整理阶段：必须保持位姿达标，且次任务代价严格下降
                    progress = (
                        is_pose(tt)
                        and objective(trial, tt["q_full"], False)[1]
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
                pure_dq, _ = _solve_sns_clamped_step(
                    t["J"],
                    t["error"],
                    None,
                    0.0,
                    D,
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

    last = terms(q, jac=False)
    if is_pose(last):
        lparts = {}
        offer_pose(q, objective(q, last["q_full"], False, lparts)[1], lparts)
    elif last["merit"] < best_merit:
        best_q = q.copy()

    result = pose_best["q"] if pose_best["q"] is not None else best_q
    # 严格在有效限位 [lo, hi]（包含 virtual_half_windows 约束）内选择最接近起始 seed 的合法 2πk 分支
    multi_turn = pin_maps.get(
        "active_multi_turn",
        pin_maps.get("active_periodic", pin_maps["active_continuous"]),
    )
    result = select_closest_bounded_branch(
        result, seed, lo, hi, multi_turn_mask=multi_turn
    )

    final = terms(result, jac=False)
    parts = {}
    objective(result, final["q_full"], False, parts)
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


def make_scatter_seeds(q_prev, q_ready, lower, upper, rng):
    """生成种子序列：q_prev（就近、连续性最好）-> q_ready（标准姿态）-> 两者附近由小到大的扰动 -> 限位中点。

    q_prev 放在第一位：从当前构型出发的 DLS 天然收敛到最近的解，最不容易“绕大圈”；
    q_ready 作为第二个基准种子提供“姿态合理”的另一族解。对无界 continuous 关节不产生 NaN 中点。
    """
    seeds = [
        clip_to_bounds(q_prev, lower, upper),
        clip_to_bounds(q_ready, lower, upper),
    ]
    # 交错排列 (轮次 -> 尺度从大到小 -> q_prev / q_ready)，保证前 max_seeds 个种子足够分散
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
    return unique_seed_list(seeds)


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
        if not model.existFrame(name):
            errors.append(f"{name}: model frame missing")
            continue
        points.append((model.getFrameId(name), mass, xyz))

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
        early_stop_ratio=float(config.get("load_early_stop_ratio", 0.60)),
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
        "multi_turn": pin_maps.get(
            "active_multi_turn", pin_maps["active_periodic"]
        ).copy(),
        "model_mass_kg": pin_maps["model_mass_kg"],
        "mass_points": pin_maps["mass_points"],
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
        model,
        data,
        q_full,
        context["mass_points"],
        context["active_v"],
        derivatives,
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