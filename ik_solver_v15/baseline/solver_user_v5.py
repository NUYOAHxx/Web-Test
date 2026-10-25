#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""ha_arm_ik.solver -- merged-v5, original HAArmIKSolver entry.

核心特性：
  1. 严格模式下，若调用方未显式传入 q_prev，自动使用通过新鲜度校验的实测活动关节状态作为默认分支与连续性参考。
  2. 区分“候选解 2πk 等价分支对齐/去重”与“零空间物理坐标回中”，所有分支选择函数均显式传入基于 URDF 关节类型的 multi_turn_mask。
  3. 显式配置的虚拟搜索窗口 (dls_virtual_half_windows) 在迭代、分支选择与最终候选检查中保持统一的硬约束语义。
  4. 拆分严格负荷预算 (load_budget_s) 与普通 IK 求解预算 (solve_budget_s)。
"""

from collections import defaultdict, deque
from enum import Enum
import math
from pathlib import Path
import threading
import time
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

try:
    import ssik
except Exception:
    ssik = None

try:
    import pinocchio as pin
except Exception:
    pin = None

from ha_arm_ik.ik_candidate import (
    PHI_TIE_TOL,
    W_LIMIT,
    W_NOM,
    W_SMOOTH,
    W_TORQUE,
    W_WAIST_RETURN,
    FailureReason,
    IKResult,
    Status,
    candidate_sort_key,
    candidate_status,
    failure_reason_for_status,
    finalize_ik_result,
    has_valid_torque,
    is_valid_status,
    make_result,
    phi_sort_value,
    score_candidate,
    select_best_candidate,
    status_from_str,
    update_candidate_final_cost_with_torque,
)
from ha_arm_ik.ik_common import (
    ORIENTATION_MODES,
    ORI_TOL,
    POS_TOL,
    SAFE_MARGIN,
    TOOL_AXIS_X,
    TOOL_CONE_HALF_ANGLE,
    T_origin,
    T_prismatic,
    T_revolute,
    clip_to_bounds,
    closest_direction_on_cone,
    invert_T,
    joint_center_from_bounds,
    joint_dist,
    joint_limit_barrier,
    joint_limit_barrier_jac,
    joint_limit_penalty,
    joint_margin,
    joint_margin_from_bounds,
    joint_range_scale,
    make_T,
    make_T_from_xyz_rpy,
    make_translation_T,
    normalize,
    normalize_cone_half_angle,
    pose_error,
    pose_errors,
    project_to_plane,
    rot_axis,
    rotx,
    roty,
    rotz,
    rpy_to_R,
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
    validate_orientation_mode,
    validate_q_vector,
    validate_transform4x4,
    wrap_to_pi,
)
from ha_arm_ik.robot_model import (
    ARM_CONFIGS,
    E_JOINT,
    REF_DIR,
    S_JOINT,
    W_JOINT,
    WAIST_JOINT_NAMES,
    SerialChain,
    estimate_chain_max_reach_from_joint,
    get_fixed_joint_origin,
    get_urdf_root_link,
    joint_origin_position_at_zero,
    parse_vec,
)
from ha_arm_ik.solver_dls import (
    COM_XY_GAIN,
    GRAVITY_TORQUE_GAIN,
    JOINT_LIMIT_GAIN,
    N_NOISE_PER_BASE,
    NOISE_SCALES,
    NULLSPACE_ERR_SCALE,
    NULLSPACE_GAIN,
    PIN_IK_DAMPING,
    PIN_IK_MAX_ITER,
    PIN_IK_MAX_STEP,
    PIN_IK_MAX_VALID_CANDIDATES,
    PIN_IK_ORI_TOL,
    PIN_IK_POS_TOL,
    PIN_IK_STALL_DELTA_M,
    PIN_IK_STALL_ITERS,
    RELATIVE_GRAVITY_GAIN,
    WAIST_NULL_GAIN,
    WAIST_RETURN_TOL,
    WAIST_TASK_WEIGHT,
    WAIST_TASK_WEIGHT_FAR,
    JointStateSnapshot,
    LoadPolicy,
    _build_effective_bounds,
    active_q_to_pinocchio_q,
    bounded_joint_difference,
    build_pinocchio_joint_maps,
    configure_model_metadata,
    dls_task_terms,
    evaluate_pose_physics,
    finite_vector,
    joint_difference,
    json_safe,
    load_physics,
    make_scatter_seeds,
    pinocchio_ik_one_seed,
    pinocchio_q_to_active_q,
    prepare_load_context,
    select_closest_bounded_branch,
    validate_transform,
)

# =============================================================================
# CONFIG
# =============================================================================

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


def unique_add(solutions, q, tol=1e-4):
    q = np.asarray(q, dtype=float)

    for old in solutions:
        if np.linalg.norm(q - old) < tol:
            return False

    solutions.append(q.copy())
    return True


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
        S_joint=S_joint,
        E_joint=E_joint,
        W_joint=W_joint,
        ref_dir=ref_dir,
    )
    psi_err = float(abs(wrap_to_pi(psi_q - psi_des)))

    margin, margin_idx = joint_margin_from_bounds(q, lower, upper)

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
    """Get Pinocchio frame id by name. Raises KeyError if not found."""
    if not model.existFrame(frame_name):
        available = [model.frames[i].name for i in range(model.nframes)]
        raise KeyError(
            f"Frame '{frame_name}' not found in Pinocchio model. "
            f"Available frames ({len(available)}): {available[:30]} ..."
        )

    return model.getFrameId(frame_name)


# =============================================================================
# Pinocchio DLS candidate orchestration
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
):
    """执行多 seed DLS 搜索：统一虚拟窗口边界、显式 multi_turn_mask 分支对齐与严格负荷早停比较。"""
    params = dict(dls_params or {})
    search = dict(load_search or {})
    context = params.get("load_context")
    strict = context is not None

    max_seeds = int(search.get("max_seeds", 10))
    min_seeds = int(search.get("min_seeds", 3))
    min_unlocked = int(search.get("min_unlocked", 2))
    if min(max_seeds, min_seeds, min_unlocked) < 1 or min_seeds > max_seeds:
        raise ValueError("invalid load search seed counts")
    if include_waist and max_seeds < min_unlocked + 1:
        raise ValueError(
            "max_seeds must include locked and unlocked comparisons"
        )

    waist_mask = np.array(
        [n in WAIST_JOINT_NAMES for n in pin_maps["active_names"]],
        dtype=bool,
    )
    # 统一计算有效搜索边界（含显式配置的 virtual_half_windows）
    eff_lower, eff_upper = _build_effective_bounds(
        lower,
        upper,
        q_ready,
        waist_mask,
        params.get("virtual_half_windows"),
        active_names=pin_maps["active_names"],
    )

    seeds = make_scatter_seeds(q_prev, q_ready, eff_lower, eff_upper, rng)
    masks = [None] * len(seeds)
    if include_waist:
        locked = clip_to_bounds(q_prev.copy(), eff_lower, eff_upper)
        locked[waist_mask] = q_ready[waist_mask]
        seeds.insert(0, locked)
        masks.insert(0, waist_mask)

    near = (
        WAIST_TASK_WEIGHT
        if params.get("waist_task_weight") is None
        else float(params["waist_task_weight"])
    )
    weights = np.array(
        [near if n in WAIST_JOINT_NAMES else 1.0 for n in pin_maps["active_names"]],
        dtype=float,
    )

    candidates, per_seed, unique = [], [], []
    unlocked = 0
    released = False
    stop = "SEED_LIMIT"
    multi_turn = pin_maps.get(
        "active_multi_turn",
        pin_maps.get("active_periodic", pin_maps["active_continuous"]),
    )

    for seed_idx, (seed, fixed) in enumerate(
        zip(seeds[:max_seeds], masks[:max_seeds])
    ):
        deadline = params.get("deadline")
        if deadline is not None and time.monotonic() >= deadline:
            stop = "TIME_BUDGET"
            break
        if fixed is None:
            unlocked += 1

        started = time.monotonic()
        try:
            q, diag = pinocchio_ik_one_seed(
                pin_model,
                pin_data,
                pin_maps,
                ee_frame_id,
                T_target_pin,
                seed,
                eff_lower,
                eff_upper,
                orientation_mode=orientation_mode,
                tool_axis=tool_axis,
                tool_cone_half_angle=tool_cone_half_angle,
                joint_weights=weights,
                q_ref=q_ready,
                include_waist=include_waist,
                fixed_joint_mask=fixed,
                waist_release=include_waist,
                waist_load_release=released,
                **params,
            )
            # 在有效搜索限位 [eff_lower, eff_upper] 内对齐至最接近当前关节角 q_prev 的合法 2πk 分支
            q = select_closest_bounded_branch(
                q, q_prev, eff_lower, eff_upper, multi_turn_mask=multi_turn
            )

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

            if strict:
                physical = load_physics(
                    pin_model, pin_data, pin_maps, q, context, False
                )
                info.update(
                    load_ok=physical["load_ok"],
                    load_peak_ratio=physical["peak_ratio"],
                    load_peak_joint=physical["peak_joint"],
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
                    **diag,
                    "fixed_waist": fixed is not None,
                    "pose_ok": bool(info["pose_ok"]),
                    "margin": float(info["margin"]),
                    "duplicate": duplicate,
                    "elapsed_ms": (time.monotonic() - started) * 1000.0,
                }
            )

            if strict:
                low = [
                    c
                    for c in candidates
                    if c.get("pose_ok")
                    and c.get("load_ok")
                    and c.get("margin", -np.inf) >= SAFE_MARGIN
                ]
                if (
                    seed_idx + 1 >= min_seeds
                    and (not include_waist or unlocked >= min_unlocked)
                    and low
                    and min(c["load_peak_ratio"] for c in low)
                    <= context["policy"].early_stop_ratio
                ):
                    stop = "LOW_LOAD_AFTER_REQUIRED_COMPARISON"
                    break
            elif (
                info.get("pose_ok")
                and info.get("margin", -np.inf) >= SAFE_MARGIN
                and seed_idx >= 1
            ):
                stop = "LEGACY_POSE_EARLY_STOP"
                break
        except Exception as exc:
            per_seed.append(
                {
                    "seed_idx": seed_idx,
                    "converged": False,
                    "fixed_waist": fixed is not None,
                    "exception": f"{type(exc).__name__}: {exc}",
                    "stop_reason": "VISIBLE_SEED_ERROR",
                    "elapsed_ms": (time.monotonic() - started) * 1000.0,
                }
            )

    if strict:
        context["search_diagnostics"] = {
            "stop_reason": stop,
            "unlocked_seeds_attempted": unlocked,
            "waist_released_for_load": released,
            "seeds_attempted": len(per_seed),
        }

    sorted_candidates = sorted(candidates, key=candidate_sort_key)
    return (
        candidates,
        [c["q"] for c in sorted_candidates[:PIN_TOP_K_FOR_DIRECT]],
        per_seed,
    )


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

    tau = gravity_torque_for_active_q(
        pin_model=pin_model,
        pin_data=pin_data,
        pin_maps=pin_maps,
        active_joint_names=active_joint_names,
        q_active=candidate["q"],
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
            "version": "merged-v5",
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
        if self.urdf_has_world_mount or self.urdf_root_link != self.base_link:
            self.T_world_from_base = make_T_from_xyz_rpy(
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
        # 基于 URDF 关节类型显式构造多圈旋转掩码，绝不把 prismatic 关节当作周期关节
        self.multi_turn_mask = np.array(
            [
                j["type"] == "continuous"
                or (
                    j["type"] == "revolute"
                    and np.isfinite(self.lower[i])
                    and np.isfinite(self.upper[i])
                    and (self.upper[i] - self.lower[i]) >= 2.0 * np.pi - 1e-6
                )
                for i, j in enumerate(self.chain.active_joints)
            ],
            dtype=bool,
        )

        try:
            self.psi_ready = float(self.chain.psi(self.q_ready))
        except Exception:
            self.psi_ready = float("nan")

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
            if (
                self.urdf_has_world_mount
                or self.urdf_root_link != self.base_link
            ):
                g = G_WORLD
            else:
                g = self.g_mounted_base
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
        waist_mask = np.array(
            [n in WAIST_JOINT_NAMES for n in self.active_joint_names],
            dtype=bool,
        )
        self.search_lower, self.search_upper = _build_effective_bounds(
            self.lower,
            self.upper,
            self.q_ready,
            waist_mask,
            self.dls_virtual_half_windows,
            active_names=self.active_joint_names,
        )

        if self.pin_model is not None:
            self.pin_ee_frame_id = get_pinocchio_frame_id(
                self.pin_model,
                self.ee_link,
            )

        self._check_model()

        if self.verbose:
            self.print_info()

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
                self.pin_maps["q_template"] = context["q_template"].copy()
                self.pin_maps["effort_limits"] = context["policy"].vectors(
                    self.active_joint_names
                )[0]
                self.pin_maps["effort_basis"] = (
                    "confirmed_allowable_joint_side_efforts"
                )
            return evaluate_pose_physics(
                self.pin_model,
                self.pin_data,
                self.pin_maps,
                q_active,
                q_ref=self.q_ready,
            )
        finally:
            if self.pin_maps is not None:
                self.pin_maps.pop("q_template", None)
                self.pin_maps["effort_limits"] = self.pin_maps[
                    "urdf_effort_limits"
                ].copy()
                self.pin_maps["effort_basis"] = (
                    "URDF reference, not confirmed continuous rating"
                )
            self._solve_lock.release()

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
        print("=" * 80)
        print("URDF root link:", self.urdf_root_link)
        print("solver base link:", self.base_link)
        print(
            "Pinocchio target frame mode:",
            "world/root"
            if (
                self.urdf_has_world_mount
                or self.urdf_root_link != self.base_link
            )
            else "local/solver_base",
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
            if (
                self.urdf_has_world_mount
                or self.urdf_root_link != self.base_link
            ):
                T_target_pin = T_target_world
            else:
                T_target_pin = T_target_local

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
                "characteristic_length": self._load_config.get(
                    "dls_characteristic_length", 0.5
                ),
            }

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
                            "load_min_seeds", 3
                        ),
                        "min_unlocked": self._load_config.get(
                            "load_min_unlocked_seeds", 2
                        ),
                    },
                )
            )
            candidates.extend(pin_candidates)
            per_seed.extend(pin_per_seed)

            if self.use_direct_psi:
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
        """执行严格负荷门控、力矩指标注入、最优候选选择及受负荷门控约束的近似回退。"""
        strict_load_safe_pool = []
        if self.load_mode == "strict":
            error = self._load_snapshot_error()
            if error:
                return self._load_failure(
                    "STALE_OR_CHANGED_CONTEXT", error, per_seed
                )
            candidates, strict_load_safe_pool = self._filter_load_candidates(
                candidates,
                T_target_world,
                orientation_mode,
                tool_axis,
                tool_cone_half_angle,
                return_load_safe_pool=True,
            )
            if not candidates:
                if allow_approximate and strict_load_safe_pool:
                    candidates = strict_load_safe_pool
                else:
                    code = (
                        "NO_LOAD_SAFE_SOLUTION"
                        if self.last_load_diagnostics.get("pose_valid_count", 0)
                        or len(strict_load_safe_pool) == 0
                        else "NO_VALID_POSE_OR_PHYSICS"
                    )
                    return self._load_failure(
                        code,
                        "No candidate passed bounds and confirmed static-load gates",
                        per_seed,
                    )

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
                    T_target_world,
                    orientation_mode,
                    tool_axis,
                    tool_cone_half_angle,
                ):
                    return self._load_failure(
                        "FINAL_RECHECK_FAILED",
                        "Selected candidate failed recomputation",
                        per_seed,
                    )
                error = self._load_snapshot_error()
                if error:
                    return self._load_failure(
                        "STALE_OR_CHANGED_CONTEXT", error, per_seed
                    )
                self.last_load_diagnostics.update(
                    accepted=True,
                    code="ACCEPTED",
                    peak_ratio=best["load_peak_ratio"],
                    peak_joint=best["load_peak_joint"],
                    ratios=best["load_guard_ratios"],
                    gravity_efforts=best["load_guard_efforts"],
                    position_error_m=best["load_final_pos_err"],
                    orientation_violation_rad=best["load_final_ori_violation"],
                )
            else:
                self.last_load_diagnostics.update(
                    accepted=True, code="ACCEPTED_WITH_LOAD_PROTECTION_OFF"
                )
            if update_prev:
                self.q_prev = best["q"].copy()
            return (
                best["q"].copy(),
                status,
                make_result(
                    {
                        k: v
                        for k, v in best.items()
                        if not k.startswith("load_")
                    },
                    status,
                    per_seed=per_seed or None,
                ),
            )

        if (
            allow_approximate
            and best is not None
            and status == "NO_VALID_POSE"
            and "q" in best
        ):
            if self.load_mode == "strict":
                self._check_load_candidate(
                    best,
                    T_target_world,
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
                error = self._load_snapshot_error()
                if error:
                    return self._load_failure(
                        "STALE_OR_CHANGED_CONTEXT", error, per_seed
                    )
                self.last_load_diagnostics.update(
                    accepted=True,
                    code="APPROXIMATE_ACCEPTED_LOAD_SAFE",
                    peak_ratio=best["load_peak_ratio"],
                    peak_joint=best["load_peak_joint"],
                    ratios=best["load_guard_ratios"],
                    gravity_efforts=best["load_guard_efforts"],
                    position_error_m=best["load_final_pos_err"],
                    orientation_violation_rad=best["load_final_ori_violation"],
                )
            else:
                self.last_load_diagnostics.update(
                    accepted=True,
                    code="APPROXIMATE_WITH_LOAD_PROTECTION_OFF",
                )

            best["approximate"] = True
            best["approximate_accepted"] = True
            best["original_status"] = status
            best["status"] = "APPROXIMATE"
            best["approximate_policy"] = (
                "allow_approximate=True: use best candidate by pose error "
                "(strict bounds and static-load gates enforced when load_mode='strict')"
            )

            if update_prev:
                self.q_prev = best["q"].copy()
            return (
                best["q"].copy(),
                "APPROXIMATE",
                make_result(
                    {
                        k: v
                        for k, v in best.items()
                        if not k.startswith("load_")
                    },
                    "APPROXIMATE",
                    per_seed=per_seed or None,
                ),
            )

        failure_reason = None
        if best is None:
            failure_reason = FailureReason.SEEDS_EXHAUSTED
            failure_detail = (
                "no candidate survived any solve pass "
                "(seed chain empty or all engines raised)"
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

        # 3. direct-psi stage
        if self.use_direct_psi:
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
        """Preserve legacy formatter/status vocabulary, even on a new load failure."""
        self.last_load_diagnostics.update(
            accepted=False, code=code, detail=str(detail)
        )
        status = "NO_VALID_POSE"
        return (
            None,
            status,
            make_result(
                {},
                status,
                failure_reason=FailureReason.NOT_CONVERGED,
                failure_detail=f"[{code}] {detail}",
                per_seed=per_seed or None,
            ),
        )

    def update_joint_state(self, snapshot_or_msg, *, now_ros_ns=None):
        """Call in the existing subscriber; no ROS import/subscription is created."""
        if isinstance(snapshot_or_msg, JointStateSnapshot):
            snapshot = snapshot_or_msg
        else:
            snapshot = JointStateSnapshot.from_joint_state(
                snapshot_or_msg, now_ros_ns=now_ros_ns
            )
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
            float(self._load_config.get("dls_characteristic_length", 0.5)),
            False,
        )
        pose_ok = bool(
            candidate.get("pose_ok", False)
            and task["pos_err"] <= POS_TOL
            and task["ori_violation"] <= ORI_TOL
        )
        physical = load_physics(
            self.pin_model,
            self.pin_data,
            self.pin_maps,
            q,
            self._load_context,
            False,
        )
        candidate.update(
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
            load_final_pos_err=task["pos_err"],
            load_final_ori_violation=task["ori_violation"],
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
        shortlist = [
            c
            for c in eligible
            if c["load_peak_ratio"]
            <= minimum + self._load_context["policy"].peak_tie
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
                    failure_reason=FailureReason.NOT_CONVERGED,
                    failure_detail="[BUSY] one solve per instance; use a single worker",
                ),
            )
        old_prev = self.q_prev.copy()
        try:
            self.last_load_diagnostics = {
                "version": "merged-v5",
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
                    self.pin_maps["q_template"] = context["q_template"].copy()
                    self.pin_maps["effort_limits"] = context["policy"].vectors(
                        self.active_joint_names
                    )[0]
                    self.pin_maps["effort_basis"] = (
                        "confirmed_allowable_joint_side_efforts"
                    )
                    self.joint_efforts_active = self.pin_maps[
                        "effort_limits"
                    ].copy()
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
                    self._solve_T_root_base = np.array(
                        self.pin_data.oMf[
                            self.pin_model.getFrameId(self.base_link)
                        ].homogeneous,
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
            if self.load_mode == "strict" and result[0] is not None:
                error = self._load_snapshot_error()
                if error:
                    self.q_prev = old_prev
                    return self._load_failure(
                        "STALE_OR_CHANGED_CONTEXT", error
                    )
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
                self.pin_maps.pop("q_template", None)
                self.pin_maps["effort_limits"] = self.pin_maps[
                    "urdf_effort_limits"
                ].copy()
                self.pin_maps["effort_basis"] = (
                    "URDF reference, not confirmed continuous rating"
                )
                self.joint_efforts_active = self.pin_maps[
                    "effort_limits"
                ].copy()
            self._solve_lock.release()