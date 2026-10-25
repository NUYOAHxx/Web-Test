#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""ha_arm_ik.solver —— HAArmIKSolver：7 轴单臂 / 腰 + 单臂 9 轴逆运动学求解入口（merged-v15）。

公共接口（自用户原版起签名不变）
  HAArmIKSolver(urdf, side, use_ssik, use_pinocchio_ik, use_torque_cost, use_direct_psi,
                include_waist, verbose, config)
  solve(T_target, frame, q_prev, update_prev, verbose, allow_approximate, orientation_mode,
        tool_axis, tool_cone_half_angle, *, joint_state=None) -> (q 或 None, status, IKResult)
  update_joint_state / evaluate_physics / workspace_diagnostic / check_workspace / print_info / close
  last_load_diagnostics：最近一次 solve 的完整诊断（负荷、搜索阶段、跳变、舒适窗、失败码）

模块分层（上层只调用下层）
  solver        本文件：配置解析与校验、搜索阶段编排、候选硬约束过滤与择优、strict 负荷保护流程
  solver_dls    数值核心：单种子 DLS/SNS 迭代、静态负荷模型、腰部舒适窗推导、任务空间种子库
  ik_candidate  候选打分 score_candidate、择优 select_best_candidate、结果类型 IKResult / Status
  ik_common     几何、限位、2πk 分支对齐等纯函数
  robot_model   URDF 解析、SerialChain 正运动学、ARM_CONFIGS / WAIST_JOINT_NAMES

一次 solve 的数据流
  solve()                      加锁；strict：冻结关节快照 -> 负荷上下文（q 模板 + 确认的容许力矩）-> 截止时间
   └ _solve_impl()             目标换到求解基座 / URDF 根坐标系；q_prev 校验（strict 未传时取实测关节角）
      ├ _collect_portfolio()   组装 DLS 参数与搜索参数
      │  └ collect_pinocchio_candidates_and_seeds()   多阶段多种子搜索（阶段与种子顺序见该函数）
      │     └ 每个种子：pinocchio_ik_one_seed -> 对齐到最接近 q_prev 的 2πk 分支 -> score_candidate
      │                 -> 跳变 -> strict 负荷 -> 去重入池 -> 早停判定
      └ _finalize_result()     跳变策略 -> strict 负荷复核 + 峰值容差带 -> 力矩代价 -> select_best_candidate
                               -> strict 最终复核 + 快照新鲜度 -> 近似回退 -> 失败码（_classify_empty_pool）

候选约束分层（与 solver_dls 文件头一致）
  硬约束  H1 位姿达标（ik_common.POS_TOL / ORI_TOL；full / tool_dir / tool_cone）
          H2 有效限位 = URDF ∩ position_bounds ∩ dls_virtual_half_windows（迭代中由 SNS 钳制保证）
          H3 静态负荷：守护关节 |τ_g| / 容许力矩 <= 1（strict）
          H4 不绕大圈：单关节相对 q_prev 的转动量 <= max_joint_jump_rad（joint_jump_policy）
  偏好    P1 腰部舒适窗 dls_waist_comfort：重新定位类求解先只在窗内搜索
  择优    S1 负荷峰值容差带（最低峰值 + peak_tie 以内同档）
          S2 同档内交给 select_best_candidate（连续性 / 标准姿态 / 限位裕度 / 腰部回中 / 可选力矩代价）

搜索阶段（collect_pinocchio_candidates_and_seeds；每阶段最多 load_max_seeds 个名额，共享时间预算）
  continuation  续算（目标在 FK(q_prev) 附近）：q_prev ± dls_continuation_max_step 内随动 + 锁腰，有局部解即采用；
                v15 起续算不做达标后的姿态整理（CONTINUATION_POSTURE_SETTLE），q_prev 已是解时原样返回
  comfort       重新定位：有效限位 ∩ 腰部舒适窗
  full          有效限位：未启用舒适窗时的唯一常规阶段，或舒适窗内没有可接受解时的回退
  jump_box      只找到跳变超限解时，在 q_prev ± max_joint_jump_rad 内重搜（dls_jump_limit_rescue）
  unboxed       上限 < π 的跳变框内没有可接受解时，不加框重搜一次
  常规阶段的种子顺序：q_prev -> 锁腰（腰部 = q_ready）-> q_ready -> 种子库（dls_seed_library_*）-> 随机扰动

可选引擎：SSIK（ssik 包）与 direct-psi（臂角参数化最小二乘）只用于 7 轴（include_waist=False），默认关闭。
版本演进与各项改动的数据见 README_v14.md（v15 的续算漂移修复与提速见 README_v15.md）。
"""

from collections import deque
from collections.abc import Mapping
import copy
import hashlib
import math
from pathlib import Path
import threading
import time
import warnings

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
    W_TORQUE,
    FailureReason,
    IKResult,
    Status,
    candidate_sort_key,
    is_valid_status,
    make_result,
    score_candidate,
    select_best_candidate,
    update_candidate_final_cost_with_torque,
)
from ha_arm_ik.ik_common import (
    ORI_TOL,
    POS_TOL,
    SAFE_MARGIN,
    TOOL_AXIS_X,
    bounded_joint_difference,
    clip_to_bounds,
    closest_direction_on_cone,
    invert_T,
    joint_center_from_bounds,
    joint_difference,
    joint_dist,
    joint_margin_from_bounds,
    joint_range_scale,
    make_T_from_xyz_rpy,
    make_translation_T,
    multi_turn_mask_from_chain,
    normalize,
    normalize_cone_half_angle,
    pose_error,
    rpy_to_R,
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
    validate_orientation_mode,
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
)
from ha_arm_ik.solver_dls import (
    PIN_IK_DAMPING,
    PIN_IK_MAX_ITER,
    PIN_IK_MAX_STEP,
    PIN_IK_ORI_TOL,
    PIN_IK_POS_TOL,
    WAIST_TASK_WEIGHT,
    WAIST_COMFORT_CONTINUATION_STEP,
    WAIST_COMFORT_EDGE_SEEDS,
    SEED_LIBRARY_MIN_SEPARATION,
    SEED_LIBRARY_RNG_SEED,
    SEED_LIBRARY_SEEDS,
    SEED_LIBRARY_SIZE,
    JointStateSnapshot,
    LoadPolicy,
    _build_effective_bounds,
    active_multi_turn_mask,
    active_periodic_mask,
    active_q_to_pinocchio_q,
    build_pinocchio_joint_maps,
    build_seed_library,
    configure_model_metadata,
    derive_waist_comfort_bounds,
    dls_task_terms,
    estimate_arm_max_reach,
    evaluate_pose_physics,
    finite_vector,
    load_physics,
    make_scatter_seeds,
    pinocchio_ik_one_seed,
    prepare_load_context,
    seed_library_target,
    select_library_seeds,
    validate_transform,
)

# 公共接口：求解器类，以及调用方常用的结果 / 状态 / 负荷输入类型（定义在 ik_candidate / solver_dls，
# 在此再导出以保持 `from ha_arm_ik.solver import IKResult` 等既有用法）
__all__ = [
    "HAArmIKSolver",
    "IKResult",
    "Status",
    "FailureReason",
    "make_result",
    "JointStateSnapshot",
    "LoadPolicy",
]

SOLVER_VERSION = "merged-v15"   # 写入 last_load_diagnostics["version"]

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
                    continuation_max_step ∩ 跳变框，DLS 单步 <= CONTINUATION_DLS_STEP_FRACTION × 该值）+ 锁腰续算
                    （q_prev 的腰部偏离 q_ready <= 该值时才跑；锁腰解负荷超 soft_ratio 时释放腰部、重跑随动续算）。
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

    # v8：舒适窗与有效限位求交；不收窄任何关节、或不含 q_ready 时退化为单阶段
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
    # 阶段 2 使用与 v7 完全相同的随机扰动：在阶段 1 消耗随机数之前复制生成器状态
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
    edge_pool = []                  # 阶段 1 未达标种子的 (主任务 merit, seed_idx, q)

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

    # ---- 阶段 1：舒适窗（未启用舒适窗时即 v7 的唯一阶段）--------------------------
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

    # ---- 阶段 2：舒适窗内无可接受解 -> 放开到有效限位 -----------------------------
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
        if self.load_mode == "strict":
            stale = self._stale_context_failure(per_seed)
            if stale:
                return stale
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
                {k: v for k, v in best.items() if not k.startswith("load_")},
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
            float(self.dls_characteristic_length),
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
            if self.load_mode == "strict" and result[0] is not None:
                stale = self._stale_context_failure()
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