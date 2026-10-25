"""Pure numerical policies for a STATIC model-load-aware IK solver.

No ROS, actuator commands, effort fallbacks, or implied hardware safety limits.
Revolute capacities are joint-side N*m; prismatic capacities are N.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Sequence
import math
import time

import numpy as np


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
    R = T[:3, :3]
    if (not np.allclose(T[3], [0, 0, 0, 1], atol=1e-9, rtol=0)
            or not np.allclose(R.T @ R, np.eye(3), atol=1e-6, rtol=0)
            or not np.isclose(np.linalg.det(R), 1.0, atol=1e-6, rtol=0)):
        raise ValueError("target rotation/bottom row is not a rigid transform")
    return T.copy()


def joint_difference(a, b, continuous):
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    d = d.copy()
    mask = np.asarray(continuous, dtype=bool)
    d[mask] = (d[mask] + np.pi) % (2 * np.pi) - np.pi
    return d


def skew(v):
    x, y, z = v
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


@dataclass(frozen=True)
class JointStateSnapshot:
    """Immutable complete state. Do not re-stamp an old cached state as new.

    source_age_at_receipt_s must use the same ROS clock as the message stamp.
    None explicitly means the source timestamp was not checked.
    """
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
        if not names or len(names) != len(positions) or len(set(names)) != len(names):
            raise ValueError("JointState names/positions are empty, duplicated or mismatched")
        stamp = msg.header.stamp
        stamp_ns = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
        source_age = None
        if now_ros_ns is not None and stamp_ns != 0:
            source_age = (int(now_ros_ns) - stamp_ns) / 1e9
        return cls(dict(zip(names, positions)),
                   time.monotonic() if received_monotonic is None else received_monotonic,
                   source_age)

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
    """All numeric defaults below are optimizer preferences, NOT rated limits."""
    allowable_efforts: Mapping[str, float]
    guard_joint_names: Sequence[str] | None = None
    joint_weights: Mapping[str, float] = field(default_factory=dict)
    absolute_weight: float = 0.05
    excess_weight: float = 0.5
    relative_weight: float = 0.0
    soft_ratio: float = 0.7
    early_stop_ratio: float = 0.6
    peak_tie: float = 0.02
    continuity_weight: float = 1.0
    mean_load_weight: float = 0.1
    reference_weight: float = 0.02

    def __post_init__(self):
        capacities = {str(k): float(v) for k, v in self.allowable_efforts.items()}
        weights = {str(k): float(v) for k, v in self.joint_weights.items()}
        if not capacities or any(not math.isfinite(v) or v <= 0 for v in capacities.values()):
            raise ValueError("explicit positive finite allowable_efforts are required; no fallback")
        if any(not math.isfinite(v) or v <= 0 for v in weights.values()):
            raise ValueError("joint load weights must be positive and finite")
        for name in ("absolute_weight", "excess_weight", "relative_weight",
                     "peak_tie", "continuity_weight", "mean_load_weight", "reference_weight"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be nonnegative and finite")
        if not 0 < self.soft_ratio < 1:
            raise ValueError("soft_ratio must lie strictly between 0 and 1")
        if not 0 < self.early_stop_ratio <= self.soft_ratio:
            raise ValueError("early_stop_ratio must be positive and <= soft_ratio")
        guards = None if self.guard_joint_names is None else tuple(self.guard_joint_names)
        if guards is not None and (not guards or len(set(guards)) != len(guards)):
            raise ValueError("guard_joint_names must be nonempty and unique")
        object.__setattr__(self, "allowable_efforts", MappingProxyType(capacities))
        object.__setattr__(self, "joint_weights", MappingProxyType(weights))
        object.__setattr__(self, "guard_joint_names", guards)

    def vectors(self, names):
        missing = [n for n in names if n not in self.allowable_efforts]
        if missing:
            raise ValueError(f"missing confirmed allowable effort for: {missing}")
        return (np.array([self.allowable_efforts[n] for n in names]),
                np.array([self.joint_weights.get(n, 1.) for n in names]))

    def cost_gradient(self, tau, d_tau_d_active, names, *, tau_reference=None):
        tau = finite_vector(tau, len(names), "gravity efforts")
        J = np.asarray(d_tau_d_active, dtype=float)
        if J.ndim != 2 or J.shape[0] != len(names) or not np.all(np.isfinite(J)):
            raise ValueError("invalid gravity effort derivative")
        caps, weights = self.vectors(names)
        r = tau / caps
        excess = np.maximum(np.abs(r) - self.soft_ratio, 0.)
        # Mean normalization prevents gain changing merely with guard count.
        n = len(names)
        abs_cost = .5 * self.absolute_weight * np.sum(weights * r*r) / n
        excess_cost = .5 * self.excess_weight * np.sum(weights * excess*excess) / n
        dcost_dtau = (self.absolute_weight * weights * r / caps
                     + self.excess_weight * weights * excess * np.sign(r) / caps) / n
        rel_cost = 0.
        if self.relative_weight:
            ref = finite_vector(tau_reference, len(names), "reference gravity efforts")
            diff = (tau-ref) / caps
            rel_cost = .5 * self.relative_weight * np.sum(weights * diff*diff) / n
            dcost_dtau += self.relative_weight * weights * diff / caps / n
        gradient = J.T @ dcost_dtau
        cost = float(abs_cost + excess_cost + rel_cost)
        if not np.all(np.isfinite(gradient)) or not math.isfinite(cost):
            raise ValueError("non-finite load objective")
        return cost, gradient, {"absolute": float(abs_cost), "excess": float(excess_cost),
                                "relative": float(rel_cost)}

    def evaluate(self, tau, names):
        tau = finite_vector(tau, len(names), "gravity efforts")
        caps, _ = self.vectors(names)
        with np.errstate(over="ignore", invalid="ignore"):
            ratios = np.abs(tau) / caps
        if not np.all(np.isfinite(ratios)):
            raise ValueError("non-finite normalized load; check capacities/model")
        i = int(np.argmax(ratios))
        return {"ratios": ratios, "peak_ratio": float(ratios[i]),
                "mean_square_ratio": float(np.mean(ratios**2)),
                "peak_joint": names[i],
                # No tolerance that can admit a configuration beyond the configured cap.
                "load_ok": bool(np.all(ratios <= 1.0)),
                "overloaded_joints": [n for n, r in zip(names, ratios) if r > 1.]}


def metric_step(J, error, gradient, inverse_weights, *, damping=1e-3, rank_rtol=1e-9):
    """Damped task + metric-consistent EXACT linearized nullspace descent.

    Uses the nullspace of J sqrt(D), not a damped pseudoprojector.
    Singular values <= rank_rtol*max(1,s_max) are treated as numerically zero.
    This is a local, numerical-rank statement, not nonlinear task invariance.
    """
    J = np.asarray(J, dtype=float)
    e = finite_vector(error, J.shape[0], "task error")
    d = finite_vector(inverse_weights, J.shape[1], "inverse weights")
    g = finite_vector(gradient, J.shape[1], "gradient")
    if np.any(d <= 0) or damping < 0 or not math.isfinite(damping):
        raise ValueError("invalid metric/damping")
    if not np.all(np.isfinite(J)):
        raise ValueError("non-finite task Jacobian")
    if J.shape[1] == 0:
        return np.empty(0), np.empty(0), 0
    sqrt_d = np.sqrt(d)
    B = J * sqrt_d
    U, singular, Vt = np.linalg.svd(B, full_matrices=True)
    cutoff = rank_rtol * max(1., float(singular[0]) if singular.size else 1.)
    rank = int(np.sum(singular > cutoff))
    V = Vt.T
    task = np.zeros(J.shape[1])
    if rank:
        s = singular[:rank]
        task = sqrt_d * (V[:, :rank] @ ((s / (s*s + damping*damping)) * (U[:, :rank].T @ e)))
    Z = V[:, rank:]
    null = -sqrt_d * (Z @ (Z.T @ (sqrt_d * g)))
    return task, null, rank


def select_load_candidate(candidates, policy):
    """Never trade an overload for better smoothness/pose/average load."""
    valid = [c for c in candidates if c.get("pose_ok") and c.get("load_ok")
             and c.get("bounds_ok") and np.isfinite(c.get("peak_ratio", np.nan))]
    if not valid:
        return None
    minimum_peak = min(c["peak_ratio"] for c in valid)
    tied = [c for c in valid if c["peak_ratio"] <= minimum_peak + policy.peak_tie]
    def key(c):
        cost = (policy.continuity_weight * c["continuity_cost"]
                + policy.mean_load_weight * c["mean_square_ratio"]
                + policy.reference_weight * c["reference_cost"])
        return cost, c["peak_ratio"], c["candidate_index"]
    return min(tied, key=key)


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