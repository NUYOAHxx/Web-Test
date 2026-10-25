"""测试用的 solver_dls 旧入口薄封装（merged-v14 起从生产代码移出）。

生产代码只使用内部实现 `_SecondaryObjective`（次任务代价与梯度）和 `_sns_clamped_core`
（主动集钳制 SNS 步），这两个旧入口只剩测试在用：
  - 有限差分测试按“整函数”验证次任务梯度与 SNS 步的性质；
  - test_v7_speed 用它们逐值复算 tests/reference/solver_dls_v6.py 记录的内部调用。
函数体与 merged-v13 原文一致，只把模块内名字改为经 `ha_arm_ik.solver_dls` 访问
（调用时取属性，测试对 D._chol_factor 等的 patch 仍然生效）。
"""
import numpy as np

import ha_arm_ik.solver_dls as D


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
    """零空间负梯度 z = -dH/dq 与标量代价 H(q)（返回 (z, H)）。

    与单种子迭代共用 _SecondaryObjective；配置负荷上下文时按 v6 语义由 q_active_cur
    重算 Pinocchio q（不使用传入的 q_full），未配置时使用传入的 q_full。
    """
    objective = D._SecondaryObjective(
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
        load_context=load_context,
        relative_gravity_gain=relative_gravity_gain,
        posture_use_equivalent_branch=posture_use_equivalent_branch,
    )
    q_active_cur = np.asarray(q_active_cur, dtype=float)
    if load_context is not None:
        q_full = D.active_q_to_pinocchio_q(pin_model, pin_maps, q_active_cur)
    return objective(q_active_cur, q_full, derivatives, components, need_z=True)


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
    """SNS 钳制步的矩阵入口：W_inv 为对角阵时走向量化实现（与矩阵实现逐位一致）。

    返回 (dq_total, dq_null)；return_info=True 时再附 clamped_converged。数学说明见
    solver_dls._sns_clamped_core。
    """
    W = np.asarray(W_inv, dtype=float)
    d_inv = None
    if W.ndim == 2 and W.shape[0] == W.shape[1]:
        diagonal = np.diagonal(W)
        if np.array_equal(W, np.diag(diagonal)):
            d_inv = diagonal.copy()
    dq_total, dq_null_out, clamped_converged = D._sns_clamped_core(
        J_active,
        err,
        z_grad,
        null_weight,
        d_inv,
        None if d_inv is not None else W,
        movable_mask,
        q_active_cur,
        eff_lower,
        eff_upper,
        damping,
        max_step,
    )
    if return_info:
        return dq_total, dq_null_out, clamped_converged
    return dq_total, dq_null_out
