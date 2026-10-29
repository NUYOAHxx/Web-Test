#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""IK 求解器候选解评分、筛选、排序与结果类型封装。"""

from enum import Enum

import numpy as np

from ha_arm_ik.ik_common import (
    ORI_TOL,
    POS_TOL,
    SAFE_MARGIN,
    TOOL_AXIS_X,
    joint_difference as joint_difference,
    joint_dist as joint_dist,
    joint_dist_periodic,
    joint_limit_penalty,
    joint_margin,
    normalize,
    normalize_cone_half_angle,
    pose_error,
    sew_angle_from_points,
    tool_cone_angle_error,
    tool_direction_angle_error,
    validate_orientation_mode,
    wrap_to_pi,
)
from ha_arm_ik.robot_model import (
    E_JOINT,
    REF_DIR,
    S_JOINT,
    W_JOINT,
    WAIST_JOINT_NAMES,
)

try:
    from ha_arm_ik.robot_model import get_sew_joints
except ImportError:
    def get_sew_joints(chain=None, s_joint=None, e_joint=None, w_joint=None):
        return (
            s_joint or getattr(chain, "s_joint", None) or S_JOINT,
            e_joint or getattr(chain, "e_joint", None) or E_JOINT,
            w_joint or getattr(chain, "w_joint", None) or W_JOINT,
        )

# 兼容导出：保留旧公共路径
_COMPAT_EXPORTS = (joint_difference, joint_dist)

W_NOM = 50.0
W_SMOOTH = 8.0
W_LIMIT = 10.0
W_WAIST_RETURN = 2000.0
W_TORQUE = 500.0
PHI_TIE_TOL = 1e-5


class Status(Enum):
    """solve() 公共状态词表。"""
    SUCCESS = "SUCCESS"
    NEAR_LIMIT = "NEAR_LIMIT"
    APPROXIMATE = "APPROXIMATE"
    NO_VALID_POSE = "NO_VALID_POSE"
    NO_RESULT = "NO_RESULT"
    OUTSIDE_WORKSPACE = "OUTSIDE_WORKSPACE"


class FailureReason(Enum):
    """solve() 失败原因（SUCCESS / NEAR_LIMIT 时为 None）。"""
    QP_INFEASIBLE_ALL_SEEDS = "QP_INFEASIBLE_ALL_SEEDS"
    QP_SOLVER_ERROR = "QP_SOLVER_ERROR"
    SEEDS_EXHAUSTED = "SEEDS_EXHAUSTED"
    TOTAL_ITER_BUDGET_HIT = "TOTAL_ITER_BUDGET_HIT"
    NOT_CONVERGED = "NOT_CONVERGED"
    WORKSPACE_REJECTED = "WORKSPACE_REJECTED"


_VALID_STATUS_STRS = {status.value for status in Status}


def status_from_str(s):
    """把内部状态字符串转换为 Status；未知值返回 None。"""
    return Status(s) if s in _VALID_STATUS_STRS else None


def failure_reason_for_status(status_str):
    """为失败的公共状态字符串返回默认失败原因。"""
    if status_str == "NO_RESULT":
        return FailureReason.SEEDS_EXHAUSTED
    if status_str in ("NO_VALID_POSE", "APPROXIMATE"):
        return FailureReason.NOT_CONVERGED
    return None


class IKResult(dict):
    """solve() 的诊断结果包装器，继承自 dict 保证向后兼容。"""

    def __init__(
        self,
        info_dict=None,
        status=None,
        failure_reason=None,
        failure_detail="",
        per_seed=None,
    ):
        super().__init__(info_dict or {})
        self.status = (
            status_from_str(status) if isinstance(status, str) else status
        )
        self.failure_reason = failure_reason
        self.failure_detail = failure_detail or ""
        self.per_seed = list(per_seed) if per_seed else []

    @property
    def ok(self):
        return self.status in (Status.SUCCESS, Status.NEAR_LIMIT)

    @property
    def failed(self):
        return not self.ok

    def __repr__(self):
        parts = [f"status={self.status.value if self.status else 'NONE'}"]
        if self.failure_reason is not None:
            parts.append(f"failure_reason={self.failure_reason.value}")
        if "pos_err" in self:
            parts.append(f"pos_err={self['pos_err'] * 1000:.3f}mm")
        if "num_candidates" in self:
            parts.append(f"candidates={self['num_candidates']}")
        return f"IKResult({', '.join(parts)})"


def make_result(
    info_dict,
    status_str,
    failure_reason=None,
    failure_detail="",
    per_seed=None,
):
    """打包构造 IKResult 对象。"""
    if failure_reason is None:
        failure_reason = failure_reason_for_status(status_str)
    return IKResult(
        info_dict,
        status=status_str,
        failure_reason=failure_reason,
        failure_detail=failure_detail,
        per_seed=per_seed,
    )


def is_valid_status(status):
    """判断状态字符串是否允许伴随有效关节角 q 返回。"""
    return status in ["SUCCESS", "NEAR_LIMIT", "APPROXIMATE"]


def phi_sort_value(c):
    """提取用于排序的臂角偏离值（过滤 PHI_TIE_TOL 内微弱扰动）。"""
    x = c.get("phi_abs", np.inf)
    if not np.isfinite(x):
        return np.inf
    return 0.0 if x < PHI_TIE_TOL else float(x)


def candidate_sort_key(c):
    """8 级分层候选解排序键：位姿达标 > 代价 > 臂角差 > 连续平滑 > 限位裕度 > 预备姿态距离 > 位置误差 > 姿态误差。"""
    return (
        not c.get("pose_ok", False),
        c.get("cost", c.get("cost_no_torque", np.inf)),
        phi_sort_value(c),
        c.get("dq_prev", np.inf),
        -c.get("margin", -np.inf),
        c.get("dq_ready", np.inf),
        c.get("pos_err", np.inf),
        c.get("ori_err", np.inf),
    )


def candidate_status(c):
    """根据候选解字典的评估字段映射为对应的公共状态字符串。"""
    if c is None:
        return "NO_RESULT"
    if not c.get("pose_ok", False):
        return "NO_VALID_POSE"
    if c.get("margin", -np.inf) < SAFE_MARGIN:
        return "NEAR_LIMIT"
    return "SUCCESS"


def _waist_indices(chain, n):
    """识别躯干腰部关节在活动关节向量中的下标。"""
    joints = getattr(chain, "active_joints", None)
    if joints is not None and len(joints) == n:
        names = [j.get("name") if isinstance(j, dict) else None for j in joints]
        if all(name is not None for name in names):
            return np.array(
                [i for i, name in enumerate(names) if name in WAIST_JOINT_NAMES],
                dtype=np.intp,
            )
    if n == len(WAIST_JOINT_NAMES) + 7:
        return np.arange(len(WAIST_JOINT_NAMES), dtype=np.intp)
    return np.zeros(0, dtype=np.intp)


def score_candidate(
    chain,
    q,
    T_target,
    psi_ready,
    q_prev,
    q_ready,
    lower,
    upper,
    orientation_mode="full",
    tool_axis=None,
    tool_cone_half_angle=None,
    apply_waist_return=False,
    periodic_mask=None,
):
    """评估单个候选解并返回位姿、各项代价分量与诊断指标。"""
    q = np.asarray(q, dtype=float)
    orientation_mode = validate_orientation_mode(orientation_mode)
    tool_axis = TOOL_AXIS_X if tool_axis is None else normalize(tool_axis)
    psi_ready_ok = psi_ready is not None and bool(np.isfinite(psi_ready))
    tool_cone_half_angle_val = normalize_cone_half_angle(tool_cone_half_angle)

    psi = np.nan
    phi = np.nan
    pos_err = np.inf
    ori_err = np.inf
    tool_dir_angle_err = np.nan
    tool_cone_violation = np.nan
    Tq = None

    try:
        S_joint, E_joint, W_joint = get_sew_joints(chain)
        Tq, joint_T = chain.fk_pose_and_joints(q, (S_joint, E_joint, W_joint))

        dp = Tq[:3, 3] - T_target[:3, 3]
        pos_err = float(np.linalg.norm(dp))

        if orientation_mode == "full":
            _, drot = pose_error(Tq, T_target)
            ori_err = float(np.linalg.norm(drot))
        elif orientation_mode == "tool_dir":
            tool_dir_angle_err = tool_direction_angle_error(
                T_current=Tq, T_target=T_target, tool_axis=tool_axis
            )
            ori_err = tool_dir_angle_err
        elif orientation_mode == "tool_cone":
            tool_cone_violation, tool_dir_angle_err = tool_cone_angle_error(
                T_current=Tq,
                T_target=T_target,
                tool_axis=tool_axis,
                cone_half_angle=tool_cone_half_angle_val,
            )
            ori_err = tool_cone_violation

        psi = float(sew_angle_from_points(
            joint_T[S_joint][:3, 3],
            joint_T[E_joint][:3, 3],
            joint_T[W_joint][:3, 3],
            REF_DIR,
        ))
        phi = float(wrap_to_pi(psi - psi_ready)) if psi_ready_ok else np.nan
    except Exception:
        pass

    margin, margin_idx, _ = joint_margin(q, lower, upper)
    dq_prev = joint_dist_periodic(q, q_prev, periodic_mask)
    dq_ready = joint_dist_periodic(q, q_ready, periodic_mask)

    waist_dev = float("nan")
    waist_idx = _waist_indices(chain, len(q))
    if waist_idx.size:
        waist_mask = (
            np.asarray(periodic_mask, dtype=bool)[waist_idx]
            if periodic_mask is not None
            else None
        )
        waist_dev = joint_dist_periodic(
            q[waist_idx], np.asarray(q_ready, dtype=float)[waist_idx], waist_mask
        )

    limit_pen = joint_limit_penalty(q, lower, upper, safe_margin=SAFE_MARGIN)
    pose_ok = (pos_err <= POS_TOL) and (ori_err <= ORI_TOL)

    if np.isfinite(phi) or (not psi_ready_ok and Tq is not None):
        psi_cost = W_NOM * phi * phi if np.isfinite(phi) else 0.0
        smooth_cost = W_SMOOTH * dq_prev * dq_prev
        limit_cost = W_LIMIT * limit_pen
        waist_cost = (
            W_WAIST_RETURN * waist_dev * waist_dev
            if apply_waist_return and np.isfinite(waist_dev)
            else 0.0
        )
        cost_no_torque = psi_cost + smooth_cost + limit_cost + waist_cost
    else:
        psi_cost = np.inf
        smooth_cost = np.inf
        limit_cost = np.inf
        waist_cost = np.inf
        cost_no_torque = np.inf

    res = {
        "q": q.copy(),
        "pos_err": pos_err,
        "ori_err": ori_err,
        "orientation_mode": orientation_mode,
        "tool_axis": None if tool_axis is None else np.asarray(tool_axis, dtype=float).copy(),
        "tool_dir_angle_err": float(tool_dir_angle_err),
        "tool_dir_angle_err_deg": (
            float(np.rad2deg(tool_dir_angle_err))
            if np.isfinite(tool_dir_angle_err)
            else np.nan
        ),
        "tool_cone_half_angle": float(tool_cone_half_angle_val),
        "tool_cone_half_angle_deg": float(np.rad2deg(tool_cone_half_angle_val)),
        "tool_cone_violation": float(tool_cone_violation),
        "tool_cone_violation_deg": (
            float(np.rad2deg(tool_cone_violation))
            if np.isfinite(tool_cone_violation)
            else np.nan
        ),
        "psi": psi,
        "phi": phi,
        "phi_abs": abs(phi) if np.isfinite(phi) else np.inf,
        "phi_deg": float(np.rad2deg(phi)) if np.isfinite(phi) else np.nan,
        "margin": margin,
        "margin_idx": margin_idx,
        "dq_prev": dq_prev,
        "dq_ready": dq_ready,
        "waist_dev": waist_dev,
        "limit_penalty": limit_pen,
        "psi_cost": float(psi_cost),
        "smooth_cost": float(smooth_cost),
        "limit_cost": float(limit_cost),
        "waist_cost": float(waist_cost),
        "cost_no_torque": float(cost_no_torque),
        "torque_weighted_cost": 0.0,
        "cost": float(cost_no_torque),
        "pose_ok": pose_ok,
    }
    res["status"] = candidate_status(res)
    return res


def evaluate_config_quality(pin_model, pin_data, pin_maps, q, ee_frame_id):
    """计算构型质量诊断指标：末端 Jacobian 条件数与最小奇异值。"""
    try:
        import pinocchio as pin

        q = np.asarray(q, dtype=float)
        idx = list(pin_maps["active_q_indices"])
        if q.shape != (len(idx),) or not np.all(np.isfinite(q)):
            raise ValueError("q must be a finite active-joint vector")
        template = pin_maps.get("q_template")
        q_full = (
            pin.neutral(pin_model)
            if template is None
            else np.array(template, dtype=float, copy=True)
        )
        cont = np.asarray(
            pin_maps.get("active_continuous", np.zeros(len(idx), dtype=bool)),
            dtype=bool,
        )
        for k, q_idx in enumerate(idx):
            if cont[k]:
                q_full[q_idx] = np.cos(q[k])
                q_full[q_idx + 1] = np.sin(q[k])
            else:
                q_full[q_idx] = float(q[k])
        J_full = pin.computeFrameJacobian(
            pin_model, pin_data, q_full, ee_frame_id, pin.LOCAL
        )
        active_v_indices = np.asarray(pin_maps["active_v_indices"], dtype=np.intp)
        J = np.asarray(J_full, dtype=float)[:, active_v_indices]
        J[:3, :] /= 0.5
        sv = np.linalg.svd(J, compute_uv=False)
        return {
            "jac_min_sv": float(sv[-1]),
            "jac_cond": float(sv[0] / max(sv[-1], 1e-12)),
        }
    except Exception:
        return {"jac_min_sv": float("nan"), "jac_cond": float("nan")}


def has_valid_torque(c):
    """判断候选解是否具备有限的力矩代价和力矩范数。"""
    return (
        c is not None
        and "torque_cost" in c
        and "torque_l2" in c
        and np.isfinite(c.get("torque_cost", np.inf))
        and np.isfinite(c.get("torque_l2", np.inf))
    )


def update_candidate_final_cost_with_torque(candidate):
    """把 solver 计算出的力矩代价合并进共享候选代价。"""
    if candidate is None:
        return

    cost_no_torque = candidate.get(
        "cost_no_torque", candidate.get("cost", np.inf)
    )
    candidate["cost_no_torque"] = float(cost_no_torque)

    if not has_valid_torque(candidate):
        candidate["torque_weighted_cost"] = np.inf
        candidate["cost"] = np.inf
        return

    torque_cost = float(candidate["torque_cost"])
    torque_weighted_cost = float(W_TORQUE * torque_cost)
    candidate["torque_weighted_cost"] = torque_weighted_cost
    candidate["cost"] = float(cost_no_torque + torque_weighted_cost)


def select_best_candidate(candidates, use_torque=True):
    """选择最优候选解并返回 (best, public_status)。"""
    if len(candidates) == 0:
        return None, "NO_RESULT"

    valid = [c for c in candidates if c.get("pose_ok", False)]

    if len(valid) == 0:
        best = dict(
            sorted(
                candidates,
                key=lambda c: (
                    c.get("pos_err", np.inf),
                    c.get("ori_err", np.inf),
                    phi_sort_value(c),
                    c.get("dq_prev", np.inf),
                    -c.get("margin", -np.inf),
                ),
            )[0]
        )
        best["selected_reason"] = "no_valid_pose_use_best_pose_error"
        return best, "NO_VALID_POSE"

    if use_torque:
        torque_valid = [c for c in valid if has_valid_torque(c)]
        pool = torque_valid if len(torque_valid) > 0 else valid
    else:
        pool = valid

    pool.sort(key=candidate_sort_key)
    best = dict(pool[0])
    best["selected_reason"] = "weighted_cost"
    best["selection_W_TORQUE"] = float(W_TORQUE)
    best["selection_cost_has_torque"] = bool(
        use_torque and has_valid_torque(best)
    )
    return best, candidate_status(best)


def finalize_ik_result(
    solver,
    candidates,
    per_seed,
    select_kwargs=None,
    annotate=None,
    update_prev=True,
    allow_approximate=False,
    failure_diagnosis=None,
):
    """所有 IK solver 的共享收尾与诊断包装逻辑。"""
    best, status = select_best_candidate(
        candidates,
        **(select_kwargs or {}),
    )
    if best is not None:
        if annotate is not None:
            annotate(best)
        best["num_candidates"] = len(candidates)
        best["approximate"] = False
        best["approximate_accepted"] = False

    if best is not None and is_valid_status(status):
        if update_prev:
            solver.q_prev = best["q"].copy()
        return best["q"].copy(), status, make_result(
            best, status, per_seed=per_seed or None
        )

    if (
        allow_approximate
        and best is not None
        and status == "NO_VALID_POSE"
        and "q" in best
    ):
        best["approximate"] = True
        best["approximate_accepted"] = True
        best["original_status"] = status
        best["status"] = "APPROXIMATE"
        if update_prev:
            solver.q_prev = best["q"].copy()

        return best["q"].copy(), "APPROXIMATE", make_result(
            best, "APPROXIMATE", per_seed=per_seed or None
        )

    per_seed = per_seed or []
    if failure_diagnosis is not None:
        failure_reason, failure_detail = failure_diagnosis(candidates, per_seed)
    elif not candidates:
        failure_reason = FailureReason.SEEDS_EXHAUSTED
        failure_detail = (
            "no candidate survived any solve pass "
            "(seed chain empty or all engines raised)"
        )
    else:
        failure_reason = FailureReason.NOT_CONVERGED
        n_stalled = sum(1 for s in per_seed if s.get("stalled"))
        failure_detail = (
            f"{len(candidates)} candidates, none within tolerance "
            f"(POS_TOL={POS_TOL:.1e}); {n_stalled} seeds stalled"
        )

    return None, status, make_result(
        best or {},
        status,
        failure_reason=failure_reason,
        failure_detail=failure_detail,
        per_seed=per_seed or None,
    )
