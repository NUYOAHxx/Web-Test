#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""ha_arm_ik.chained_ik —— 9 轴（腰 2 + 臂 7）腰臂协同与指定臂角（Direct-Psi）两阶段级联逆运动学求解器。

设计理念与求解流水线：
  1. 宏观定位阶段（Stage 1: 9-DOF Macro Positioning）：
     使用主力 Pinocchio DLS 求解器进行全机身（腰 2 + 臂 7）逆运动学计算，
     处理大范围作业空间可达性、静态重力矩卸载、整机水平质心（CoM）防倾覆保护与腰部舒适窗。
  2. 微观定肘阶段（Stage 2: 7-DOF Direct-Psi Elbow Steering）：
     锁定阶段一求得的腰部角度 q_waist，冻结胸腔基座（Body_Link）在空间中的三维位姿，将末端目标投影至该基座；
     以阶段一的手臂角度为高精度种子，调用 Direct-Psi（基于解析雅可比与 SciPy TRF 优化），
     在严格保持末端位姿精度的同时，将手肘臂角强力收敛至用户显式指定的期望值 psi_des。
  3. 平滑回退兜底（Graceful Fallback）：
     若用户指定的期望臂角在当前腰部姿态下受关节物理限位阻挡而无法达成，自动回退到阶段一
     求得的自然合规解，并在诊断字典中标记 psi_fallback=True，确保生产环境绝对不因臂角偏好
     而导致末端到达失败。
"""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from .ik_base import (
    IKResult,
    Status,
    invert_T,
    is_valid_status,
    wrap_to_pi,
)
from .single_arm_ik import (
    HAArmIKSolver,
    SingleArmIKSolver,
    solve_direct_psi,
)

__all__ = [
    "Chained9DofArmAngleSolver",
    "solve_9dof_with_arm_angle",
]


def solve_9dof_with_arm_angle(
    solver_9dof: HAArmIKSolver,
    solver_7dof: HAArmIKSolver,
    T_target: np.ndarray,
    psi_des: float,
    frame: str = "world",
    q_prev: Optional[np.ndarray] = None,
    update_prev: bool = True,
    verbose: Optional[bool] = None,
    allow_approximate: bool = False,
    orientation_mode: str = "full",
    tool_axis: Any = None,
    tool_cone_half_angle: Any = None,
    *,
    joint_state: Any = None,
    w_psi: float = 20.0,
    psi_tol: float = 0.20,
    max_direct_nfev: int = 150,
) -> Tuple[Optional[np.ndarray], str, Any]:
    """通过两阶段级联架构在 9 轴上求解逆运动学并显式指定臂角。

    Args:
        solver_9dof: 9 轴腰臂协同求解器 (include_waist=True)。
        solver_7dof: 7 轴纯手臂求解器 (include_waist=False)。
        T_target: 目标 4x4 齐次变换矩阵。
        psi_des: 用户期望的手臂肘部角度 [rad]（绕肩-腕连线的带符号转角）。
        frame: "world"（默认，URDF 根/世界安装系）或 "local"（9 轴基座系）。
        q_prev: 前一时刻的 9 维关节角，用于平滑与续算。
        update_prev: 是否更新求解器内部的 q_prev。
        verbose: 是否打印详细调试信息。
        allow_approximate: 是否允许近似解。
        orientation_mode: 姿态模式 ("full", "tool_dir", "tool_cone")。
        tool_axis: 工具主轴向量。
        tool_cone_half_angle: 锥角半角 [rad]。
        joint_state: 实时 JointStateSnapshot 快照（strict 模式必需）。
        w_psi: 阶段二 Direct-Psi 优化中臂角残差权重。
        psi_tol: 阶段二容许的手肘角度误差 [rad]（默认 0.2 rad ≈ 11.5°）。
        max_direct_nfev: 阶段二 TRF 优化的最大函数评估次数。

    Returns:
        (q_9dof, status, result):
            - q_9dof: 9 维关节角度向量 [q_waist(2), q_arm(7)]，若失败则为 None。
            - status: 状态字符串（如 "SUCCESS"）。
            - result: IKResult 字典对象，包含诊断、各关节角度及实际达成的 psi 值。
    """
    psi_des = float(wrap_to_pi(psi_des))
    T_target = np.asarray(T_target, dtype=float)

    # =========================================================================
    # 阶段一：9 轴宏观定位求解（腰臂协同、重力矩卸载与防倾覆）
    # =========================================================================
    q_9dof, status, result = solver_9dof.solve(
        T_target=T_target,
        frame=frame,
        q_prev=q_prev,
        update_prev=False,  # 暂不更新，待阶段二完成后统一更新
        verbose=verbose,
        allow_approximate=allow_approximate,
        orientation_mode=orientation_mode,
        tool_axis=tool_axis,
        tool_cone_half_angle=tool_cone_half_angle,
        joint_state=joint_state,
    )

    if not is_valid_status(status) or q_9dof is None:
        return None, status, result

    q_waist = q_9dof[:2]
    q_arm_seed = q_9dof[2:]

    # =========================================================================
    # 坐标转换：将世界/根坐标系下的目标位姿投影到当前腰部构型下的手臂基座
    # =========================================================================
    if frame == "world":
        T_target_world = T_target.copy()
        T_target_root = invert_T(solver_9dof.T_world_from_base) @ T_target_world
    else:
        T_target_root = T_target.copy()
        T_target_world = solver_9dof.T_world_from_base @ T_target_root

    fk_res = solver_9dof.chain.fk_all(q_9dof)
    link_T, _ = fk_res
    arm_base_link = solver_7dof.base_link
    if arm_base_link not in link_T:
        raise KeyError(
            f"arm_base_link '{arm_base_link}' not found in 9-dof forward kinematics links."
        )

    T_root_to_armbase = link_T[arm_base_link]
    T_target_in_armbase = invert_T(T_root_to_armbase) @ T_target_root

    # =========================================================================
    # 阶段二：7 轴 Direct-Psi 微观定肘（以阶段一手肘为初值，强力约束臂角）
    # =========================================================================
    lower, upper = solver_7dof.chain.bounds()
    direct_sols = solve_direct_psi(
        chain=solver_7dof.chain,
        T_target=T_target_in_armbase,
        psi_des=psi_des,
        seeds=[q_arm_seed],
        orientation_mode=orientation_mode,
        tool_axis=tool_axis,
        tool_cone_half_angle=tool_cone_half_angle,
        verbose=bool(verbose),
        w_psi=w_psi,
        psi_tol=psi_tol,
        max_nfev=max_direct_nfev,
        bounds=(lower, upper),
    )

    # =========================================================================
    # 阶段三：结果拼装与优雅回退
    # =========================================================================
    if direct_sols:
        best_arm = direct_sols[0]["q"]
        achieved_psi = direct_sols[0]["psi"]
        psi_err = direct_sols[0]["psi_err"]
        q_final = np.concatenate([q_waist, best_arm])

        # 检查整机安全闭环：
        # 1. 检查单关节跳变限幅 (H4 硬约束保护)
        jump_safe = True
        if q_prev is not None and hasattr(solver_9dof, "max_joint_jump_rad") and solver_9dof.max_joint_jump_rad is not None:
            max_jump = float(np.max(np.abs(q_final - q_prev)))
            if max_jump > solver_9dof.max_joint_jump_rad:
                jump_safe = False

        # 2. strict 模式静态力矩安全复核 (H3 硬约束保护)
        strict_safe = True
        if solver_9dof.load_mode == "strict" and solver_9dof.pin_model is not None:
            try:
                phys = solver_9dof.evaluate_physics(q_final, joint_state=joint_state)
                if phys.get("max_motor_load_pct") is not None and phys["max_motor_load_pct"] > 100.0:
                    strict_safe = False
            except Exception:
                pass

        if jump_safe and strict_safe:
            info = dict(result)
            info["q"] = q_final
            info["solve_method"] = "chained_9dof_direct_psi"
            info["source"] = "direct_psi"
            info["psi"] = float(achieved_psi)
            info["psi_err"] = float(psi_err)
            info["psi_target"] = float(psi_des)
            info["psi_fallback"] = False

            if update_prev:
                solver_9dof.q_prev = q_final.copy()
                solver_7dof.q_prev = best_arm.copy()

            return q_final, status, IKResult(info)

    # 若指定臂角物理不可达、跳变超限或力矩超载，平滑回退到阶段一的自然合格解
    current_psi = float(solver_7dof.chain.psi(q_arm_seed))
    info = dict(result)
    info["q"] = q_9dof
    info["psi"] = current_psi
    info["psi_err"] = float(abs(wrap_to_pi(current_psi - psi_des)))
    info["psi_target"] = float(psi_des)
    info["psi_fallback"] = True

    if update_prev:
        solver_9dof.q_prev = q_9dof.copy()
        solver_7dof.q_prev = q_arm_seed.copy()

    return q_9dof, status, IKResult(info)


class Chained9DofArmAngleSolver:
    """9 轴腰臂协同 + 指定臂角两阶段级联高内聚逆运动学求解器。

    使用两阶段分层设计：
      - 阶段一：9 轴 Pinocchio DLS 宏观规划，保证大作业空间可达性与重力矩安全；
      - 阶段二：7 轴 Direct-Psi 微观定肘，保证手肘精确指向用户期望的臂角。
    """

    def __init__(
        self,
        urdf: Union[str, Path],
        side: str = "right",
        verbose: bool = False,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.urdf = str(Path(urdf).expanduser().resolve())
        self.side = side.lower()
        self.verbose = verbose
        self.config = dict(config) if config else {}

        # 9 轴宏观定位求解器
        self.solver_9dof = HAArmIKSolver(
            urdf=self.urdf,
            side=self.side,
            include_waist=True,
            use_direct_psi=False,
            verbose=self.verbose,
            config=self.config,
        )

        # 7 轴纯手臂求解器
        cfg_7dof = self.config.copy()
        self.solver_7dof = HAArmIKSolver(
            urdf=self.urdf,
            side=self.side,
            include_waist=False,
            use_direct_psi=False,
            verbose=self.verbose,
            config=cfg_7dof,
        )

        self.chain_9dof = self.solver_9dof.chain
        self.chain_7dof = self.solver_7dof.chain
        self.active_joint_names = self.solver_9dof.active_joint_names

    def solve(
        self,
        T_target: np.ndarray,
        psi_des: float,
        frame: str = "world",
        q_prev: Optional[np.ndarray] = None,
        update_prev: bool = True,
        verbose: Optional[bool] = None,
        allow_approximate: bool = False,
        orientation_mode: str = "full",
        tool_axis: Any = None,
        tool_cone_half_angle: Any = None,
        *,
        joint_state: Any = None,
        w_psi: float = 20.0,
        psi_tol: float = 0.20,
        max_direct_nfev: int = 150,
    ) -> Tuple[Optional[np.ndarray], str, Any]:
        """求解 9 轴关节角并强力指定臂角。"""
        if verbose is None:
            verbose = self.verbose

        return solve_9dof_with_arm_angle(
            solver_9dof=self.solver_9dof,
            solver_7dof=self.solver_7dof,
            T_target=T_target,
            psi_des=psi_des,
            frame=frame,
            q_prev=q_prev,
            update_prev=update_prev,
            verbose=verbose,
            allow_approximate=allow_approximate,
            orientation_mode=orientation_mode,
            tool_axis=tool_axis,
            tool_cone_half_angle=tool_cone_half_angle,
            joint_state=joint_state,
            w_psi=w_psi,
            psi_tol=psi_tol,
            max_direct_nfev=max_direct_nfev,
        )

    def compute_psi(self, q: np.ndarray) -> float:
        """根据 9 维关节角度计算当前手臂相对于胸腔的真实臂角 [rad]。"""
        q = np.asarray(q, dtype=float)
        if len(q) == 9:
            q_arm = q[2:]
        elif len(q) == 7:
            q_arm = q
        else:
            raise ValueError(f"Expected 9 or 7 joint angles, got {len(q)}")
        return float(self.chain_7dof.psi(q_arm))

    def close(self) -> None:
        """释放底层 C++ 资源。"""
        self.solver_9dof.close()
        self.solver_7dof.close()

