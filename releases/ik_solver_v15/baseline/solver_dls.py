#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Pinocchio 阻尼最小二乘 IK 核心（工業級重構版）。

本模組包含單 seed 數值算法、無狀態腰部自適應釋放、SNS 限位鉗制重投影、
以及基於 q_ref 錨定的防漂移/抗重力矩/穩重心零空間優化。
候選打分與 portfolio 編排保留在 ``ha_arm_ik.solver``。
"""

import math

import numpy as np
from scipy.linalg import cho_factor, cho_solve

try:
    import pinocchio as pin
except Exception:
    pin = None

from ha_arm_ik.ik_base import (
    SAFE_MARGIN,
    TOOL_AXIS_X,
    WAIST_JOINT_NAMES,
    clip_to_bounds,
    closest_direction_on_cone,
    normalize,
    normalize_cone_half_angle,
    skew,
    unique_seed_list,
    validate_orientation_mode,
)

# =============================================================================
# 1. 基礎數值與收斂容差參數
# =============================================================================
PIN_IK_MAX_ITER = 500
PIN_IK_DAMPING = 1e-6
PIN_IK_POS_TOL = 5e-4
PIN_IK_ORI_TOL = 5e-4
PIN_IK_MAX_STEP = 0.5             # 單次迭代關節步長上限（弧度）
PIN_IK_STALL_ITERS = 25           # 連續微進展迭代次數上限 -> 放棄
PIN_IK_STALL_DELTA_M = 5e-5       # 單次迭代進展閾值（米，= 0.05 mm）
PIN_IK_MAX_VALID_CANDIDATES = 2
WAIST_RETURN_TOL = 0.10           # 外層早停的腰部回中閾值（弧度）
N_NOISE_PER_BASE = 16
NOISE_SCALES = [0.02, 0.05, 0.10, 0.20]

# =============================================================================
# 2. 零空間防漂移與流形穩態收斂參數
# =============================================================================
NULLSPACE_GAIN = 0.30             # 手臂冗餘關節拉向 q_ref 的次任務增益
WAIST_NULL_GAIN = 0.50            # 腰部關節拉向 q_ref（直立位）的次任務增益
NULLSPACE_ERR_SCALE = 0.06        # 主任務誤差小於此值時零空間整理全功率激活
POSTURE_SETTLE_TOL = 1e-4         # 末端收斂後，零空間步長小於此閾值(rad)才算穩態收斂
MAX_SETTLE_EXTRA_ITERS = 50       # 末端達標後允許繼續在零空間整理姿態的最大步數

# 兼容舊版外部導入的別名常量
WAIST_POST_GAIN = 0.30
WAIST_POST_ITERS = MAX_SETTLE_EXTRA_ITERS
WAIST_POST_STEP = 0.05
WAIST_POST_TOL = POSTURE_SETTLE_TOL

# =============================================================================
# 3. 無狀態腰部自適應釋放參數（路徑無關，去返程完全對稱）
# =============================================================================
WAIST_TASK_WEIGHT = 6.0           # 近距離時腰部關節空間代價（手臂 = 1.0，鎖死穩重心）
WAIST_TASK_WEIGHT_FAR = 1.0       # 遠目標釋放後的腰部權重（與手臂平權協同，防過度彎腰）
WAIST_RELEASE_START = 0.75        # 肩->目標距離/最大臂展 達到此比例開始放腰
WAIST_RELEASE_END = 0.94          # 達到此比例腰完全釋放
WAIST_RETURN_RELEASE_RAD = 0.20   # 返程解鎖尺度(rad)：腰偏離 q_ref 時保持釋放通道以便退回

# =============================================================================
# 4. 電機重力矩卸載、重心穩定與限位保護參數
# =============================================================================
GRAVITY_TORQUE_GAIN = 0.04        # 相對重力矩卸載增益（抑制遠伸時腰/肩大力矩）
COM_XY_GAIN = 0.12                # 相對質心(CoM)水平防傾倒增益（拉回待命重心投影）
JOINT_LIMIT_GAIN = 0.03           # 單側安全裕度(SAFE_MARGIN)排斥增益
WAIST_YAW_VIRTUAL_LIM = np.deg2rad(90.0)   # 腰部旋轉(±360°)相對 q_ref 的最大工作偏角
ARM_ROLL_VIRTUAL_LIM = np.deg2rad(150.0)   # 臂自轉關節(±360°)相對 q_ref 的最大工作偏角
MAX_SNS_CLAMP_PASSES = 3          # 單步內非對稱限位碰撞重投影最大次數


# =============================================================================
# 配置與 Pinocchio 映射
# =============================================================================
def build_pinocchio_joint_maps(model, active_joint_names):
    """建立 active_joint_names 與 Pinocchio 關節索引之間的映射。"""
    name_to_joint_id = {}
    name_to_q_idx = {}
    name_to_v_idx = {}

    for jid in range(1, model.njoints):
        name = model.names[jid]
        name_to_joint_id[name] = jid
        name_to_q_idx[name] = model.joints[jid].idx_q
        name_to_v_idx[name] = model.joints[jid].idx_v

    active_q_indices = []
    active_v_indices = []
    active_names = []
    effort_limits = []

    for name in active_joint_names:
        if name not in name_to_joint_id:
            raise KeyError(
                f"Active joint '{name}' not found in Pinocchio model. "
                f"Available: {list(name_to_joint_id.keys())}"
            )
        active_q_indices.append(name_to_q_idx[name])
        active_v_indices.append(name_to_v_idx[name])
        active_names.append(name)

        jid = name_to_joint_id[name]
        eff = float(model.effortLimit[model.joints[jid].idx_v])
        effort_limits.append(eff if eff > 1e-9 else 1.0)

    return {
        "name_to_joint_id": name_to_joint_id,
        "name_to_q_idx": name_to_q_idx,
        "name_to_v_idx": name_to_v_idx,
        "active_q_indices": active_q_indices,
        "active_v_indices": active_v_indices,
        "active_names": active_names,
        "effort_limits": np.array(effort_limits, dtype=float),
    }


def active_q_to_pinocchio_q(model, pin_maps, q_active):
    """由活躍關節值構造完整的 Pinocchio q 向量（非活躍關節置為 neutral）。"""
    q_full = pin.neutral(model)
    for k, q_idx in enumerate(pin_maps["active_q_indices"]):
        q_full[q_idx] = float(q_active[k])
    return q_full


def pinocchio_q_to_active_q(pin_maps, q_full):
    """從完整 Pinocchio q 向量中提取活躍關節值。"""
    return np.array(
        [float(q_full[idx]) for idx in pin_maps["active_q_indices"]],
        dtype=float,
    )


def estimate_arm_max_reach(pin_model, pin_maps, ee_frame_id):
    """從 Pinocchio 模型解析肩->TCP 的最大幾何距離。"""
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


def evaluate_pose_physics(
    pin_model, pin_data, pin_maps, q_active, q_ref=None
):
    """計算給定關節構型下的重心偏移、底座傾覆力矩及各關節重力負載指標。"""
    q_active = np.asarray(q_active, dtype=float)
    q_full = active_q_to_pinocchio_q(pin_model, pin_maps, q_active)
    active_v_indices = pin_maps["active_v_indices"]
    effort_limits = pin_maps["effort_limits"]

    pin.computeGeneralizedGravity(pin_model, pin_data, q_full)
    tau_g = pin_data.g[active_v_indices].copy()
    tau_norm = float(np.linalg.norm(tau_g))
    motor_load_pct = (np.abs(tau_g) / np.maximum(effort_limits, 1e-6)) * 100.0

    pin.centerOfMass(pin_model, pin_data, q_full, False)
    com_xyz = pin_data.com[0].copy()
    total_mass = float(pin_data.mass[0]) if len(pin_data.mass) > 0 else 0.0
    com_xy_dist_m = float(np.linalg.norm(com_xyz[:2]))
    overturning_moment = total_mass * 9.81 * com_xy_dist_m

    drift_deg = 0.0
    if q_ref is not None:
        drift_deg = float(np.rad2deg(np.linalg.norm(q_active - q_ref)))

    return {
        "tau_g_nm": tau_g,
        "tau_norm_nm": tau_norm,
        "motor_load_pct": motor_load_pct,
        "max_motor_load_pct": float(np.max(motor_load_pct)),
        "com_xyz_m": com_xyz,
        "com_xy_offset_mm": com_xy_dist_m * 1000.0,
        "overturning_moment_nm": overturning_moment,
        "posture_drift_deg": drift_deg,
    }


def _build_effective_bounds(lower, upper, q_ref, waist_bool_mask):
    """為 ±360° 大範圍旋轉關節構建僅依賴於 q_ref 的固定虛擬工作盒。

    邊界僅由 (lower, upper, q_ref) 唯一決定，與起始種子 seed (q_prev) 無關，
    從拓撲上阻斷 ±180° 奇異翻轉與 ±360° 多圈纏繞。
    """
    eff_lower = np.asarray(lower, dtype=float).copy()
    eff_upper = np.asarray(upper, dtype=float).copy()
    span = eff_upper - eff_lower

    wide_joints = span > np.deg2rad(300.0)
    for k in np.nonzero(wide_joints)[0]:
        half_win = (
            WAIST_YAW_VIRTUAL_LIM
            if waist_bool_mask[k]
            else ARM_ROLL_VIRTUAL_LIM
        )
        eff_lower[k] = max(eff_lower[k], q_ref[k] - half_win)
        eff_upper[k] = min(eff_upper[k], q_ref[k] + half_win)

    return eff_lower + 1e-9, eff_upper - 1e-9


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
):
    """計算以 q_ref 為全局唯一零點的複合零空間負梯度 (-dH/dq) 與標量代價 H(q)。"""
    diff_q = q_ref - q_active_cur
    z_grad = ns_gain * diff_q
    cost = 0.5 * float(np.dot(ns_gain * (diff_q ** 2), movable_mask))

    # 1. 單側安全裕度排斥項：僅當關節比 q_ref 更靠近物理極限且進入 SAFE_MARGIN 時激活，
    #    保證在 q = q_ref 處梯度嚴格為 0，同時防止運動中貼邊觸發 LOW_MARGIN。
    if limit_gain > 0.0:
        margin_safe = max(float(SAFE_MARGIN), 0.02)
        b_low = np.minimum(q_ref, lower + margin_safe)
        b_high = np.maximum(q_ref, upper - margin_safe)

        under_pen = np.maximum(0.0, b_low - q_active_cur) / margin_safe
        over_pen = np.maximum(0.0, q_active_cur - b_high) / margin_safe

        z_limit = (
            limit_gain * (under_pen - over_pen) / margin_safe
        )
        z_grad += z_limit
        cost += 0.5 * limit_gain * float(
            np.dot((under_pen ** 2) + (over_pen ** 2), movable_mask)
        )

    # 2. 相對重力矩卸載項：最小化 0.5 * w_g * ||(tau_g(q) - tau_g(q_ref)) / eff_lim||^2
    if gravity_gain > 0.0 and tau_g_ref is not None:
        try:
            dtau_dq_full = pin.computeGeneralizedGravityDerivatives(
                pin_model, pin_data, q_full
            )
            tau_g_cur = pin_data.g[active_v_indices]
            dtau_dq_active = dtau_dq_full[
                np.ix_(active_v_indices, active_v_indices)
            ]
            tau_diff = tau_g_cur - tau_g_ref
            inv_eff_sq = 1.0 / (effort_limits ** 2)
            z_gravity = -gravity_gain * (
                dtau_dq_active.T @ (tau_diff * inv_eff_sq)
            )
            if np.all(np.isfinite(z_gravity)):
                z_grad += z_gravity
                cost += 0.5 * gravity_gain * float(
                    np.sum((tau_diff ** 2) * inv_eff_sq)
                )
        except Exception:
            pass

    # 3. 相對整機質心 (CoM) 水平穩定項：最小化 0.5 * w_c * ||CoM_xy(q) - CoM_xy(q_ref)||^2
    if include_waist and com_gain > 0.0 and com_xy_ref is not None:
        try:
            J_com_full = pin.jacobianCenterOfMass(
                pin_model, pin_data, q_full, False
            )
            com_xy_diff = pin_data.com[0][:2] - com_xy_ref
            J_com_xy = J_com_full[:2, active_v_indices]
            z_com = -com_gain * (J_com_xy.T @ com_xy_diff)
            if np.all(np.isfinite(z_com)):
                z_grad += z_com
                cost += 0.5 * com_gain * float(com_xy_diff @ com_xy_diff)
        except Exception:
            pass

    return z_grad * movable_mask, cost


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
):
    """帶 Active-Set 限位鉗制重投影（SNS）與對稱 Riemannian 零空間算子的 DLS 求解。"""
    n_active = len(q_active_cur)
    free_mask = movable_mask.copy()
    dq_clamped = np.zeros(n_active, dtype=float)
    dq_null_out = np.zeros(n_active, dtype=float)
    task_dim = err.size

    # 自適應阻尼：誤差較大時略增阻尼防奇異點超調，收斂期精確回到基礎阻尼
    err_sq = float(err @ err)
    eff_damping = damping + min(1e-3, 1e-4 * err_sq)

    for pass_idx in range(MAX_SNS_CLAMP_PASSES):
        if not np.any(free_mask):
            break

        err_rem = err - J_active @ dq_clamped
        J_free = J_active[:, free_mask]
        W_inv_free = W_inv[np.ix_(free_mask, free_mask)]

        A = J_free @ W_inv_free @ J_free.T
        A.flat[:: task_dim + 1] += eff_damping

        chol_fac = None
        try:
            chol_fac = cho_factor(A, check_finite=False)
            rhs = cho_solve(chol_fac, err_rem, check_finite=False)
        except (np.linalg.LinAlgError, ValueError):
            rhs = np.linalg.solve(A, err_rem)

        dq_task_free = W_inv_free @ (J_free.T @ rhs)

        # 對稱正定零空間投影算子：dq_null = N_W (W_inv * z_grad)
        # 若使用 W_inv 縮放梯度，穩態條件為 dH/dq = J^T * mu（完全與度量矩陣 W 無關）
        dq_null_free = np.zeros_like(dq_task_free)
        if z_grad is not None and null_weight > 0.0:
            z_free = z_grad[free_mask]
            Jz = J_free @ z_free
            if chol_fac is not None:
                J_free_invA_Jz = cho_solve(chol_fac, Jz, check_finite=False)
            else:
                J_free_invA_Jz = np.linalg.solve(A, Jz)
            dq_null_free = null_weight * (
                z_free - W_inv_free @ (J_free.T @ J_free_invA_Jz)
            )

        dq_total = dq_clamped.copy()
        dq_total[free_mask] = dq_task_free + dq_null_free

        dq_null_out.fill(0.0)
        dq_null_out[free_mask] = dq_null_free

        dq_norm = math.sqrt(float(dq_total @ dq_total))
        if dq_norm > max_step:
            scale = max_step / dq_norm
            dq_total *= scale
            dq_null_out *= scale

        q_trial = q_active_cur + dq_total
        under = (q_trial < eff_lower) & free_mask
        over = (q_trial > eff_upper) & free_mask
        violated = under | over

        if not np.any(violated) or pass_idx == MAX_SNS_CLAMP_PASSES - 1:
            return dq_total, dq_null_out

        dq_clamped[under] = eff_lower[under] - q_active_cur[under]
        dq_clamped[over] = eff_upper[over] - q_active_cur[over]
        free_mask[violated] = False

    return dq_clamped, dq_null_out


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
):
    """使用 Pinocchio FK + Jacobian 的阻尼最小二乘 IK（單主循環統一收斂與流形精修）。"""
    orientation_mode = validate_orientation_mode(orientation_mode)
    tool_cone_half_angle = normalize_cone_half_angle(tool_cone_half_angle)
    if tool_axis is None:
        tool_axis = TOOL_AXIS_X

    tool_axis = normalize(tool_axis)
    seed = np.asarray(seed, dtype=float)

    if q_ref is None:
        q_ref = seed.copy()
    else:
        q_ref = np.asarray(q_ref, dtype=float).copy()

    # 支持動態讀取模組級全局變量或外部顯式傳入的覆蓋參數
    eff_ns_gain = NULLSPACE_GAIN if nullspace_gain is None else float(nullspace_gain)
    eff_waist_ns_gain = (
        WAIST_NULL_GAIN if waist_null_gain is None else float(waist_null_gain)
    )
    eff_waist_weight = (
        WAIST_TASK_WEIGHT
        if waist_task_weight is None
        else float(waist_task_weight)
    )
    eff_waist_weight_far = (
        WAIST_TASK_WEIGHT_FAR
        if waist_task_weight_far is None
        else float(waist_task_weight_far)
    )
    eff_grav_gain = (
        GRAVITY_TORQUE_GAIN
        if gravity_torque_gain is None
        else float(gravity_torque_gain)
    )
    eff_com_gain = COM_XY_GAIN if com_xy_gain is None else float(com_xy_gain)
    eff_lim_gain = (
        JOINT_LIMIT_GAIN
        if joint_limit_gain is None
        else float(joint_limit_gain)
    )

    active_q_indices = np.asarray(pin_maps["active_q_indices"], dtype=np.intp)
    active_v_indices = np.asarray(pin_maps["active_v_indices"], dtype=np.intp)
    effort_limits = pin_maps["effort_limits"]
    n_active = len(active_v_indices)

    if joint_weights is None:
        joint_weights = np.ones(n_active, dtype=float)
    else:
        joint_weights = np.asarray(joint_weights, dtype=float).copy()

    R_des = np.asarray(T_target[:3, :3], dtype=float, order="C")
    p_des = np.asarray(T_target[:3, 3], dtype=float)
    oMdes = pin.SE3(R_des, p_des)
    d_des = normalize(R_des @ tool_axis)

    W_inv_base = np.diag(1.0 / np.maximum(joint_weights, 1e-6))

    ns_gain = np.full(n_active, eff_ns_gain, dtype=float)
    for k, name in enumerate(pin_maps["active_names"]):
        if name in WAIST_JOINT_NAMES:
            ns_gain[k] = eff_waist_ns_gain
    use_nullspace = bool(
        np.any(ns_gain > 0.0)
        or eff_grav_gain > 0.0
        or eff_lim_gain > 0.0
        or (include_waist and eff_com_gain > 0.0)
    )

    fixed_joint_mask = (
        np.asarray(fixed_joint_mask, dtype=bool)
        if fixed_joint_mask is not None
        else np.zeros(n_active, dtype=bool)
    )
    movable_mask = ~fixed_joint_mask
    waist_bool_mask = np.array(
        [name in WAIST_JOINT_NAMES for name in pin_maps["active_names"]],
        dtype=bool,
    )

    eff_lower, eff_upper = _build_effective_bounds(
        lower, upper, q_ref, waist_bool_mask
    )
    q_init = np.clip(seed, eff_lower, eff_upper)
    q_full = active_q_to_pinocchio_q(pin_model, pin_maps, q_init)

    # 預計算參考姿態 q_ref 下的基準肩部位置、基準重力矩與基準質心投影
    q_ref_full = active_q_to_pinocchio_q(pin_model, pin_maps, q_ref)
    pin.forwardKinematics(pin_model, pin_data, q_ref_full)
    pin.updateFramePlacements(pin_model, pin_data)

    waist_release_active = bool(
        waist_release
        and include_waist
        and np.any(waist_bool_mask & movable_mask)
    )
    if waist_release_active and waist_max_reach is None:
        waist_max_reach = estimate_arm_max_reach(
            pin_model, pin_maps, ee_frame_id
        )
    waist_release_active = bool(waist_release_active and waist_max_reach)

    shoulder_jid = None
    p_shoulder_ref = None
    if waist_release_active:
        for name in pin_maps["active_names"]:
            if name not in WAIST_JOINT_NAMES and name.endswith("_Link2"):
                shoulder_jid = pin_maps["name_to_joint_id"][name]
                break
        if shoulder_jid is None:
            waist_release_active = False
        else:
            p_shoulder_ref = pin_data.oMi[shoulder_jid].translation.copy()

    tau_g_ref = None
    com_xy_ref = None
    if use_nullspace:
        if eff_grav_gain > 0.0:
            try:
                pin.computeGeneralizedGravity(pin_model, pin_data, q_ref_full)
                tau_g_ref = pin_data.g[active_v_indices].copy()
            except Exception:
                tau_g_ref = None
        if include_waist and eff_com_gain > 0.0:
            try:
                pin.centerOfMass(pin_model, pin_data, q_ref_full, False)
                com_xy_ref = pin_data.com[0][:2].copy()
            except Exception:
                com_xy_ref = None

    waist_indices = np.nonzero(waist_bool_mask & movable_mask)[0]
    waist_w_inv_stiff = (
        1.0 / eff_waist_weight if eff_waist_weight > 0 else 1.0
    )
    waist_w_inv_far = (
        1.0 / eff_waist_weight_far if eff_waist_weight_far > 0 else 1e3
    )

    # 預計算固定中性肩部到目標點的基準距離釋放量 alpha_dist_ref（目標不變則恆定）
    alpha_dist_ref = 0.0
    if waist_release_active and p_shoulder_ref is not None:
        dist_ref = float(np.linalg.norm(p_des - p_shoulder_ref))
        r_start = WAIST_RELEASE_START * waist_max_reach
        r_span = (WAIST_RELEASE_END - WAIST_RELEASE_START) * waist_max_reach
        t_ref = np.clip((dist_ref - r_start) / max(r_span, 1e-4), 0.0, 1.0)
        alpha_dist_ref = float(t_ref * t_ref * (3.0 - 2.0 * t_ref))

    seed_diag = {
        "converged": False,
        "iters": 0,
        "final_pos_err": float("nan"),
        "final_ori_err": float("nan"),
        "stalled": False,
        "error_history": [],
    }

    prev_pos_err = None
    stall_count = 0
    settle_iters = 0
    best_converged_q_full = None
    best_converged_cost = float("inf")
    W_inv_cur = W_inv_base.copy()

    for it in range(max_iter):
        pin.forwardKinematics(pin_model, pin_data, q_full)
        pin.updateFramePlacements(pin_model, pin_data)

        oMcur = pin_data.oMf[ee_frame_id]
        if orientation_mode == "full":
            err_se3 = pin.log6(oMcur.inverse() * oMdes)
            pos_norm = math.sqrt(float(err_se3.linear @ err_se3.linear))
            ori_norm = math.sqrt(float(err_se3.angular @ err_se3.angular))
            ori_violation = ori_norm
            err = err_se3.vector

            J_full = pin.computeFrameJacobian(
                pin_model, pin_data, q_full, ee_frame_id, pin.LOCAL
            )
            J_active = J_full[:, active_v_indices]

        elif orientation_mode == "tool_dir":
            R_cur = np.asarray(oMcur.rotation, dtype=float)
            p_cur = np.asarray(oMcur.translation, dtype=float)
            d_cur = normalize(R_cur @ tool_axis)

            e_pos = p_des - p_cur
            e_dir = d_des - d_cur
            pos_norm = float(np.linalg.norm(e_pos))
            ori_norm = float(
                np.arccos(np.clip(float(np.dot(d_cur, d_des)), -1.0, 1.0))
            )
            ori_violation = ori_norm

            J_full = pin.computeFrameJacobian(
                pin_model, pin_data, q_full, ee_frame_id, pin.WORLD
            )
            J_active_full = J_full[:, active_v_indices]
            Jv = J_active_full[:3, :]
            Jw = J_active_full[3:, :]
            J_dir = -skew(d_cur) @ Jw

            err = np.concatenate([e_pos, e_dir])
            J_active = np.vstack([Jv, J_dir])

        else:  # "tool_cone"
            R_cur = np.asarray(oMcur.rotation, dtype=float)
            p_cur = np.asarray(oMcur.translation, dtype=float)
            d_cur = normalize(R_cur @ tool_axis)

            e_pos = p_des - p_cur
            d_closest, ori_norm, cone_violation, inside = (
                closest_direction_on_cone(
                    d_current=d_cur,
                    d_target=d_des,
                    cone_half_angle=tool_cone_half_angle,
                )
            )
            pos_norm = float(np.linalg.norm(e_pos))
            ori_violation = cone_violation

            J_full = pin.computeFrameJacobian(
                pin_model, pin_data, q_full, ee_frame_id, pin.WORLD
            )
            J_active_full = J_full[:, active_v_indices]
            Jv = J_active_full[:3, :]
            Jw = J_active_full[3:, :]

            if inside or cone_violation <= ori_tol:
                e_dir = np.zeros(3, dtype=float)
                J_dir = np.zeros_like(Jw)
            else:
                e_dir = d_closest - d_cur
                J_dir = -skew(d_cur) @ Jw

            err = np.concatenate([e_pos, e_dir])
            J_active = np.vstack([Jv, J_dir])

        seed_diag["iters"] = it + 1
        seed_diag["final_pos_err"] = pos_norm
        seed_diag["final_ori_err"] = ori_norm
        seed_diag["error_history"].append(
            {"iteration": it + 1, "pos_err": pos_norm, "ori_err": ori_norm}
        )

        q_active_cur = q_full[active_q_indices]
        task_converged = (pos_norm < pos_tol) and (ori_violation < ori_tol)

        z_grad = None
        posture_cost = 0.0
        null_weight = 0.0
        if use_nullspace:
            z_grad, posture_cost = _compute_nullspace_gradient_and_cost(
                pin_model=pin_model,
                pin_data=pin_data,
                q_full=q_full,
                q_active_cur=q_active_cur,
                q_ref=q_ref,
                lower=lower,
                upper=upper,
                ns_gain=ns_gain,
                active_v_indices=active_v_indices,
                effort_limits=effort_limits,
                movable_mask=movable_mask,
                include_waist=include_waist,
                tau_g_ref=tau_g_ref,
                com_xy_ref=com_xy_ref,
                gravity_gain=eff_grav_gain,
                com_gain=eff_com_gain,
                limit_gain=eff_lim_gain,
            )
            err_norm = float(np.linalg.norm(err))
            # 末端收斂後略微提升零空間步進速率（1.5x），加速到達唯一穩態點
            base_nw = 1.5 if task_converged else 1.0
            null_weight = base_nw / (
                1.0 + (err_norm / max(NULLSPACE_ERR_SCALE, 1e-6)) ** 2
            )

        if task_converged:
            seed_diag["converged"] = True
            if posture_cost < best_converged_cost:
                best_converged_cost = posture_cost
                best_converged_q_full = q_full.copy()

            if not use_nullspace or settle_iters >= MAX_SETTLE_EXTRA_ITERS:
                break
            settle_iters += 1

        # 無狀態腰部自適應釋放（基準肩距 + 當前肩距 + 腰偏離角回中通道）
        if waist_release_active:
            p_shoulder_cur = pin_data.oMi[shoulder_jid].translation
            dist_cur = float(np.linalg.norm(p_des - p_shoulder_cur))
            r_start = WAIST_RELEASE_START * waist_max_reach
            r_span = (WAIST_RELEASE_END - WAIST_RELEASE_START) * waist_max_reach
            t_cur = np.clip((dist_cur - r_start) / max(r_span, 1e-4), 0.0, 1.0)
            alpha_dist_cur = float(t_cur * t_cur * (3.0 - 2.0 * t_cur))

            waist_dev = float(
                np.linalg.norm((q_active_cur - q_ref)[waist_indices])
            )
            alpha_return = float(
                np.clip(
                    waist_dev / max(WAIST_RETURN_RELEASE_RAD, 1e-4), 0.0, 1.0
                )
            )
            alpha = max(alpha_dist_ref, alpha_dist_cur, alpha_return)

            w_inv_waist = (
                1.0 - alpha
            ) * waist_w_inv_stiff + alpha * waist_w_inv_far
            W_inv_cur[waist_indices, waist_indices] = w_inv_waist

        dq_active, dq_null = _solve_sns_clamped_step(
            J_active=J_active,
            err=err,
            z_grad=z_grad,
            null_weight=null_weight,
            W_inv=W_inv_cur,
            movable_mask=movable_mask,
            q_active_cur=q_active_cur,
            eff_lower=eff_lower,
            eff_upper=eff_upper,
            damping=damping,
            max_step=max_step,
        )

        if (
            task_converged
            and float(np.linalg.norm(dq_null)) < POSTURE_SETTLE_TOL
        ):
            q_next = np.clip(q_active_cur + dq_active, eff_lower, eff_upper)
            q_full[active_q_indices] = q_next
            best_converged_q_full = q_full.copy()
            break

        q_active_next = np.clip(
            q_active_cur + dq_active, eff_lower, eff_upper
        )
        q_full[active_q_indices] = q_active_next

        if not task_converged:
            if prev_pos_err is not None and (
                abs(prev_pos_err - pos_norm) < PIN_IK_STALL_DELTA_M
            ):
                stall_count += 1
            else:
                stall_count = 0
            prev_pos_err = pos_norm

            if stall_count >= PIN_IK_STALL_ITERS:
                seed_diag["stalled"] = True
                break

    if best_converged_q_full is not None:
        q_full = best_converged_q_full

    return pinocchio_q_to_active_q(pin_maps, q_full), seed_diag


def make_scatter_seeds(q_prev, q_ready, lower, upper, rng):
    """從 q_prev、q_ready 和噪聲生成 IK 兜底用的散佈種子。"""
    seeds = []
    q_prev = clip_to_bounds(q_prev, lower, upper)
    q_ready = clip_to_bounds(q_ready, lower, upper)

    # 將 q_ready 與 q_prev 作為前兩個確定性基準種子
    seeds.append(q_ready.copy())
    seeds.append(q_prev.copy())
    bases = [q_prev, q_ready]

    for base in bases:
        for scale in NOISE_SCALES:
            for _ in range(max(1, N_NOISE_PER_BASE // len(NOISE_SCALES))):
                q_seed = base + rng.normal(scale=scale, size=len(base))
                q_seed = clip_to_bounds(q_seed, lower, upper)
                seeds.append(q_seed)

    q_mid = 0.5 * (lower + upper)
    seeds.append(clip_to_bounds(q_mid, lower, upper))

    return unique_seed_list(seeds)
