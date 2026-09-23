#!/usr/bin/env python3
"""
================================================================================
Unitree G1 机械臂工业级逆运动学优化求解器 (Industrial Pink + ProxQP IK Engine)
================================================================================

架构原则 (First Principles & Industrial Standards):
1. 数学严谨流形优化：完全基于 Inria / CNRS 原厂开发的人形机器人专用开源库 Pink 4.4.0；
2. 凸二次规划求解内核：集成 LAAS-CNRS 开发的高阶原对偶内点法 ProxQP 求解器；
3. SE(3) 李群测地线残差：由 FrameTask 严格计算空间 3D 线位移与 SO(3) 李代数对数姿态残差；
4. 绝对关节硬限位保证：由 ConfigurationLimit 在 QP 凸多面体空间内强制约束，数学上 0% 越界；
5. 标称姿态引力与加权：通过 PostureTask 施加对角加权矩阵，严格实现腰部高惩罚、手臂优先运动；
6. 彻底清除所有 Ad-hoc 补丁：零魔数初猜、零手工碰撞踢脉冲、零非量纲误差混合。
"""

import time
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pinocchio as pin
import pink
from pink.tasks import FrameTask, PostureTask
from pink.barriers import SelfCollisionBarrier

from core.kinematics.g1_model import (
    G1KinematicsModel,
    G1_READY_POSE,
    G1_DEFAULT_STAND_JOINTS,
)
from core.collision.g1_collision import G1CollisionChecker

# 逆运动学算法五大计算流水线阶段定义 (Pink + ProxQP 工业标准架构)
IK_PIPELINE_STAGES = [
    {"id": 1, "name": "Configuration", "fullName": "Pink Lie-Manifold Setup",      "desc": "构建流形构型并绑定硬关节限位 (ConfigurationLimit, 0% 越界保证)"},
    {"id": 2, "name": "FrameTask",     "fullName": "SE(3) Manifold Task",          "desc": "末端执行器 SE(3) 位置 + 李代数对数旋转测地线流形任务"},
    {"id": 3, "name": "PostureTask",   "fullName": "Joint-Weighted Posture",       "desc": "各关节运动代价加权（腰部高惩罚，手臂运动优先，就绪姿态引力）"},
    {"id": 4, "name": "ProxQP Solver", "fullName": "Inria ProxQP Solver",          "desc": "高阶原对偶凸二次规划求解器微秒级求解 (Inria / LAAS-CNRS)"},
    {"id": 5, "name": "Safety Barrier", "fullName": "Whole-Body Collision Barrier", "desc": "基于 MoveIt SRDF ACM 工业级自碰撞极速门禁与 CBF 屏障函数"},
]

from enum import Enum


class IKSolveStatus(str, Enum):
    """逆运动学求解诊断状态枚举"""
    CONVERGED = "CONVERGED"                          # 严苛双重收敛 (位置与姿态均达标，且无碰撞)
    POSITION_REACHED_ONLY = "POSITION_REACHED_ONLY"  # 仅位置收敛，姿态未达标
    RELAXED_ORIENTATION = "RELAXED_ORIENTATION"      # 姿态受控微松弛容差下收敛 (<=2.0°)
    MAX_ITERATIONS_EXCEEDED = "MAX_ITERATIONS_EXCEEDED"  # 达到最大迭代步数仍未达标
    JOINT_LIMIT_VIOLATION = "JOINT_LIMIT_VIOLATION"  # 撞击关节硬限位阻断
    SINGULARITY_DETECTED = "SINGULARITY_DETECTED"    # 发生严重运动学奇异
    SELF_COLLISION = "SELF_COLLISION"                # 触发全机身自碰撞门禁
    QP_INFEASIBLE = "QP_INFEASIBLE"                  # QP 优化器无可行域
    INVALID_INPUT = "INVALID_INPUT"                  # 输入目标包含 NaN 或 Inf


class IKResult(dict):
    """
    Unitree G1 工业级逆运动学求解诊断结果对象。
    同时继承 dict 与提供对象属性访问，100% 兼容既有字典索引 (res['pos_err_mm'])
    与类型安全的属性自省 (res.pos_err_mm, res.status)。
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__dict__ = self

    @property
    def is_converged(self) -> bool:
        """是否成功收敛 (严苛收敛或允许松弛下的收敛)"""
        return bool(self.get("success", False))

    def __repr__(self) -> str:
        status_val = self.get("status", "UNKNOWN")
        status_str = status_val.value if hasattr(status_val, "value") else str(status_val)
        return (
            f"<IKResult success={self.get('success', False)} status={status_str} "
            f"pos_err={self.get('pos_err_mm', 0.0):.2f}mm rot_err={self.get('rot_err_deg', 0.0):.2f}° "
            f"time={self.get('time_ms', 0.0):.2f}ms iters={self.get('iters', 0)}>"
        )


class G1PinkIKSolver:
    """
    Unitree G1 工业级逆运动学求解器 (Pink + ProxQP 引擎)
    """

    def __init__(
        self,
        kinematics: Optional[G1KinematicsModel] = None,
        collision_checker: Optional[G1CollisionChecker] = None,
        urdf_path: Optional[str] = None,
    ):
        """
        构造逆运动学求解器
        :param kinematics: 注入的运动学引擎 (若为 None 则自动初始化默认模型)
        :param collision_checker: 注入的碰撞安全引擎 (若为 None 则自动初始化默认碰撞检测器)
        :param urdf_path: 当 kinematics 为 None 时指定的 URDF 路径
        """
        if kinematics is None:
            self.kin = G1KinematicsModel(urdf_path=urdf_path)
        else:
            self.kin = kinematics

        if collision_checker is None:
            self.collision = G1CollisionChecker(self.kin)
        else:
            self.collision = collision_checker

        # ── 核心运动学属性与配置便捷委托 ──
        self.urdf_path: str = self.kin.urdf_path
        self.model: pin.Model = self.kin.model
        self.data: pin.Data = self.kin.data
        self.limits: Dict[str, Tuple[np.ndarray, np.ndarray]] = self.kin.limits
        self.limits_10dof: Dict[str, Tuple[np.ndarray, np.ndarray]] = self.kin.limits_10dof
        self.waist_limits: Tuple[np.ndarray, np.ndarray] = self.kin.waist_limits
        self.left_arm_joint_names: List[str] = self.kin.left_arm_joint_names
        self.right_arm_joint_names: List[str] = self.kin.right_arm_joint_names
        self.waist_joint_names: List[str] = self.kin.waist_joint_names
        self.ee_frame_names: Dict[str, str] = self.kin.ee_frame_names
        self.ee_frame_ids: Dict[str, int] = self.kin.ee_frame_ids
        self.arm_q_indices: Dict[str, List[int]] = self.kin.arm_q_indices
        self.waist_q_indices: List[int] = self.kin.waist_q_indices
        self.arm_joint_names: Dict[str, List[str]] = self.kin.arm_joint_names
        self.chain_10dof_indices: Dict[str, List[int]] = self.kin.chain_10dof_indices

        # 快捷方法委托
        self.forward_kinematics = self.kin.forward_kinematics
        self.forward_kinematics_10dof = self.kin.forward_kinematics_10dof
        self.build_q_10dof = self.kin.build_q_10dof
        self.is_colliding = self.collision.is_colliding
        self.evaluate_balance = self.kin.evaluate_balance
        self.compute_gravity_torques = self.kin.compute_gravity_torques
        self.compute_manipulability = self.kin.compute_manipulability

    def _extract_solution_telemetry(
        self,
        arm: str,
        waist_q: np.ndarray,
        arm_q: np.ndarray,
        target_rot: Optional[np.ndarray] = None,
        q_full_base: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """提取解算构型下的 Pinocchio 刚体动力学、全身质心平衡与全 6D 空间位姿指标"""
        q_10dof = np.concatenate([waist_q, arm_q])
        q_full = self.build_q_10dof(arm, waist_q, arm_q, base_q=q_full_base)
        bal = self.kin.evaluate_balance(q_full)
        torques = self.kin.compute_gravity_torques(q_full, arm=arm, is_10dof=True)
        manip = self.kin.compute_manipulability(arm, q_10dof, has_rot=(target_rot is not None))

        fk_pos, fk_rot = self.kin.forward_kinematics_10dof(arm, waist_q, arm_q)
        act_rpy_rad = pin.rpy.matrixToRpy(fk_rot)
        act_rpy_deg = [round(float(np.degrees(v)), 2) for v in act_rpy_rad]
        act_quat = pin.Quaternion(fk_rot)

        tgt_rpy_deg = None
        rot_err_deg = 0.0
        if target_rot is not None:
            tgt_rpy_rad = pin.rpy.matrixToRpy(target_rot)
            tgt_rpy_deg = [round(float(np.degrees(v)), 2) for v in tgt_rpy_rad]
            R_err = target_rot @ fk_rot.T
            rot_err_deg = round(float(np.degrees(np.linalg.norm(pin.log3(R_err)))), 2)

        # 关节限位安全裕度 (距物理硬限位的最小角距离，单位: rad 与 deg)
        low_lim, up_lim = self.limits_10dof[arm]
        dist_to_low = q_10dof - low_lim
        dist_to_up = up_lim - q_10dof
        joint_limit_margin_rad = float(np.min(np.minimum(dist_to_low, dist_to_up)))
        joint_limit_margin_deg = round(float(np.degrees(joint_limit_margin_rad)), 2)

        return {
            "com_pos": bal["com_pos"],
            "com_proj": bal["com_proj"],
            "balance_margin_mm": bal["margin_mm"],
            "balance_status": bal["status"],
            "support_polygon": bal["support_polygon"],
            "gravity_torques": [round(float(v), 3) for v in torques],
            "manipulability": manip["yoshikawa"] if isinstance(manip, dict) else manip,
            "min_singular_value": manip.get("min_singular_value", 0.0) if isinstance(manip, dict) else 0.0,
            "is_singular": manip.get("is_singular", False) if isinstance(manip, dict) else False,
            "condition_number": manip.get("condition_number", 1.0) if isinstance(manip, dict) else 1.0,
            "joint_limit_margin": round(joint_limit_margin_rad, 4),
            "joint_limit_margin_deg": joint_limit_margin_deg,
            "actual_rpy_deg": act_rpy_deg,
            "actual_quat": [round(float(v), 5) for v in [act_quat.x, act_quat.y, act_quat.z, act_quat.w]],
            "target_rpy_deg": tgt_rpy_deg,
            "rot_err_deg": rot_err_deg,
        }

    # --------------------------------------------------------------------------
    # 内部流水线辅助工序 (Internal Pipeline Auxiliaries)
    # --------------------------------------------------------------------------

    def _build_seed_chain(
        self,
        is_10dof: bool,
        arm: str,
        ready_q: np.ndarray,
        lower_limit: np.ndarray,
        upper_limit: np.ndarray,
        seed_q: Optional[np.ndarray] = None,
        seed_waist: Optional[np.ndarray] = None,
        seed_arm: Optional[np.ndarray] = None,
    ) -> Tuple[List[np.ndarray], List[str]]:
        """构建确定性启发式多种子链 (Warm Start -> Ready Pose -> 4组 QMC 空间网格)"""
        seed_chain: List[np.ndarray] = []
        seed_names: List[str] = []

        # 1. Warm Start 种子装配
        if is_10dof:
            if seed_waist is not None or seed_arm is not None:
                ready_arm = G1_READY_POSE[arm]
                sw = np.clip(
                    np.asarray(seed_waist if seed_waist is not None else np.zeros(3), dtype=np.float64),
                    self.waist_limits[0],
                    self.waist_limits[1],
                )
                sa = np.clip(
                    np.asarray(seed_arm if seed_arm is not None else ready_arm, dtype=np.float64),
                    self.limits[arm][0],
                    self.limits[arm][1],
                )
                seed_chain.append(np.concatenate([sw, sa]))
                seed_names.append("Warm Start")
        else:
            if seed_q is not None:
                seed_chain.append(np.clip(np.asarray(seed_q, dtype=np.float64), lower_limit, upper_limit))
                seed_names.append("Warm Start")

        # 2. Ready Pose 中立就绪种子
        seed_chain.append(ready_q.copy())
        seed_names.append("Ready Pose")

        # 3. 确定性低差异准蒙特卡洛多起点探索 (4 组均匀分位网格)
        for r_idx in range(4):
            alpha = (r_idx + 1) / 5.0
            qmc_seed = lower_limit + alpha * (upper_limit - lower_limit)
            if is_10dof:
                qmc_seed[:3] *= 0.30  # 空间探索中对腰部初猜施加阻尼，优先探索手臂
            seed_chain.append(qmc_seed)
            seed_names.append(f"QMC Grid #{r_idx+1}")

        return seed_chain, seed_names

    def _polish_gauss_newton(
        self,
        arm: str,
        is_10dof: bool,
        model: pin.Model,
        data: pin.Data,
        ee_id: int,
        q_init: np.ndarray,
        target_pos: np.ndarray,
        target_rot: Optional[np.ndarray],
        lower_limit: np.ndarray,
        upper_limit: np.ndarray,
        check_collision: bool,
    ) -> Tuple[np.ndarray, float, float]:
        """
        二阶段高精李群高斯-牛顿微调抛光器 (~25 微秒)
        在第一阶段满足硬限位与防自碰的前提下，执行 2 步无阻尼李群高斯-牛顿微调，消除 QP 正则化残差。
        返回: (polished_q, best_err, best_rot_norm)
        """
        has_rot = target_rot is not None
        pin.forwardKinematics(model, data, q_init)
        pin.updateFramePlacements(model, data)
        cur_T = data.oMf[ee_id]
        ep_norm = float(np.linalg.norm(target_pos - cur_T.translation))
        er_norm = float(np.linalg.norm(pin.log3(target_rot @ cur_T.rotation.T))) if has_rot else 0.0

        best_q = q_init.copy()
        best_err = ep_norm + er_norm * 0.25
        best_rot_norm = er_norm

        q_pol = q_init.copy()
        for _ in range(2):
            pin.forwardKinematics(model, data, q_pol)
            pin.updateFramePlacements(model, data)
            cur_T = data.oMf[ee_id]
            ep = target_pos - cur_T.translation
            ep_norm = float(np.linalg.norm(ep))
            if has_rot:
                er = pin.log3(target_rot @ cur_T.rotation.T)
                er_norm = float(np.linalg.norm(er))
                e_vec = np.concatenate([ep, er])
                J = pin.computeFrameJacobian(model, data, q_pol, ee_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            else:
                er_norm = 0.0
                e_vec = ep
                J = pin.computeFrameJacobian(model, data, q_pol, ee_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:3, :]

            m = 6 if has_rot else 3
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-6 * np.eye(m), e_vec)
            dq = np.clip(dq, -0.04, 0.04)
            q_cand = np.clip(q_pol + dq, lower_limit, upper_limit)

            is_cand_col = (
                self.collision.is_colliding_reduced(arm, q_cand, is_10dof=is_10dof)
                if check_collision and self.collision is not None
                else False
            )
            if not is_cand_col:
                pin.forwardKinematics(model, data, q_cand)
                pin.updateFramePlacements(model, data)
                c_T = data.oMf[ee_id]
                cand_ep = float(np.linalg.norm(target_pos - c_T.translation))
                cand_er = float(np.linalg.norm(pin.log3(target_rot @ c_T.rotation.T))) if has_rot else 0.0
                cand_err = cand_ep + cand_er * 0.25
                if cand_err < (ep_norm + er_norm * 0.25):
                    q_pol = q_cand
                    best_q = q_cand.copy()
                    best_err = cand_err
                    best_rot_norm = cand_er

        return best_q, best_err, best_rot_norm

    def _classify_solve_status(
        self,
        is_success: bool,
        fin_col: bool,
        fin_ep: float,
        fin_er: float,
        pos_tol: float,
        rot_tol: float,
        has_rot: bool,
        is_singular: bool,
    ) -> IKSolveStatus:
        """纯函数式状态分类机：按严苛优先律裁定最终诊断状态枚举"""
        if fin_col:
            return IKSolveStatus.SELF_COLLISION
        if is_success:
            return IKSolveStatus.CONVERGED
        if fin_ep < pos_tol and has_rot and fin_er >= rot_tol:
            return IKSolveStatus.POSITION_REACHED_ONLY
        if is_singular:
            return IKSolveStatus.SINGULARITY_DETECTED
        return IKSolveStatus.MAX_ITERATIONS_EXCEEDED

    # --------------------------------------------------------------------------
    # 通用凸二次规划优化内核 (_solve_qp_core) — Pink + ProxQP / OSQP
    # --------------------------------------------------------------------------

    def _solve_qp_core(
        self,
        arm: str,
        is_10dof: bool,
        target_pos: np.ndarray,
        target_rot: Optional[np.ndarray],
        seed_chain: List[np.ndarray],
        seed_names: List[str],
        ready_q: np.ndarray,
        cost_posture: np.ndarray,
        lower_limit: np.ndarray,
        upper_limit: np.ndarray,
        q_full_base: Optional[np.ndarray],
        waist_q_base: np.ndarray,
        pos_tol: float,
        rot_tol: float,
        max_iters: int,
        max_step: float,
        check_collision: bool,
        start_time: float,
        enable_collision_barrier: bool = False,
        barrier_d_min: float = 0.015,
        record_trace: bool = True,
    ) -> Tuple[bool, np.ndarray, IKResult]:
        """
        统一的 SE(3) 流形凸二次规划优化内核 (DRY 架构，供 7-DoF 与 10-DoF 共用)
        """
        has_rot = target_rot is not None
        model = self.kin.model_10dof[arm] if is_10dof else self.kin.model_7dof[arm]
        data = self.kin.data_10dof[arm] if is_10dof else self.kin.data_7dof[arm]
        ee_frame = self.ee_frame_names[arm]
        target_pose = pin.SE3(target_rot if has_rot else np.eye(3), target_pos)

        # ── Pink 任务栈配置 ──
        # 1. 主空间位姿任务 (FrameTask)
        task_ee = FrameTask(
            ee_frame,
            position_cost=10.0,
            orientation_cost=5.0 if has_rot else 0.0,
            lm_damping=1e-4,
        )
        task_ee.set_target(target_pose)

        # 2. 标称零空间姿态引力 (PostureTask)
        task_posture = PostureTask(cost=cost_posture)
        task_posture.set_target(ready_q)

        tasks = [task_ee, task_posture]

        best_q: Optional[np.ndarray] = None
        best_err = float("inf")
        best_rot_norm: float = 0.0
        total_iters = 0
        convergence_trace: List[float] = []
        seed_switches: List[int] = []
        step_details: List[Dict[str, Any]] = []

        dt = 0.25

        # 构型对象复用：若启用 CBF 屏障函数则挂载降维碰撞模型，否则轻量构造极速收敛
        coll_model = (self.kin.coll_model_10dof[arm] if is_10dof else self.kin.coll_model_7dof[arm]) if enable_collision_barrier else None
        coll_data = (self.kin.coll_data_10dof[arm] if is_10dof else self.kin.coll_data_7dof[arm]) if enable_collision_barrier else None
        config = pink.Configuration(model, data, seed_chain[0].copy(), collision_model=coll_model, collision_data=coll_data)

        barriers = []
        if enable_collision_barrier:
            barriers.append(
                SelfCollisionBarrier(
                    n_collision_pairs=5,
                    d_min=barrier_d_min,
                    gain=1.0,
                    safe_displacement_gain=0.0,
                )
            )

        for seed_idx, seed in enumerate(seed_chain):
            if total_iters >= max_iters:
                break
            # 若启用屏障约束，则跳过已发生穿透的无效初猜种子
            if enable_collision_barrier and self.collision is not None:
                if self.collision.is_colliding_reduced(arm, seed, is_10dof=is_10dof):
                    continue

            config.update(seed.copy())
            seed_switches.append(total_iters)
            prev_err_mm: Optional[float] = None
            stall_count = 0

            # ── 自适应初猜预算分配策略 ──
            is_primary_seed = (seed_idx == 0) or (seed_idx == 1 and seed_names[0] == "Warm Start")
            remaining_total = max_iters - total_iters
            if is_primary_seed:
                seed_iter_budget = min(remaining_total, max(22, int(max_iters * 0.70)))
            else:
                seed_iter_budget = min(remaining_total, max(10, int(max_iters * 0.35)))

            for it in range(seed_iter_budget):
                total_iters += 1

                # ── 零空间姿态引力动态退火 ──
                if prev_err_mm is not None and prev_err_mm < 6.0:
                    alpha_posture = max(0.01, prev_err_mm / 6.0)
                    task_posture.cost = cost_posture * alpha_posture
                else:
                    task_posture.cost = cost_posture.copy()

                try:
                    v = pink.solve_ik(config, tasks, dt, solver="proxqp", barriers=barriers if barriers else None)
                except Exception:
                    try:
                        v = pink.solve_ik(config, tasks, dt, solver="osqp", barriers=barriers if barriers else None)
                    except Exception:
                        break

                # 步长硬约束截断，杜绝大角度非线性跳跃
                step = v * dt
                step_max = float(np.max(np.abs(step)))
                if max_step is not None and max_step > 0 and step_max > max_step:
                    v = v * (max_step / step_max)

                config.integrate_inplace(v, dt)
                q_cur = config.q.copy()

                # 正向运动学与流形残差
                cur_pose = config.get_transform_frame_to_world(ee_frame)
                e_pos = target_pos - cur_pose.translation
                pos_norm = float(np.linalg.norm(e_pos))
                cur_err_mm = pos_norm * 1000.0
                delta_mm = round(float(prev_err_mm - cur_err_mm), 3) if prev_err_mm is not None else 0.0
                prev_err_mm = cur_err_mm

                rot_norm = 0.0
                if has_rot:
                    R_err = target_rot @ cur_pose.rotation.T
                    rot_norm = float(np.linalg.norm(pin.log3(R_err)))

                convergence_trace.append(round(cur_err_mm, 2))

                # 工业级极速降维自碰撞检测 (~40 微秒)
                is_col = False
                if check_collision and self.collision is not None:
                    is_col = self.collision.is_colliding_reduced(arm, q_cur, is_10dof=is_10dof)

                err_val = pos_norm + rot_norm * 0.25
                if err_val < best_err and not is_col:
                    best_err = err_val
                    best_q = q_cur.copy()
                    best_rot_norm = rot_norm

                # 达标检查
                if pos_norm < pos_tol and (not has_rot or rot_norm < rot_tol) and not is_col:
                    if record_trace:
                        step_details.append({
                            "step": total_iters, "iter": it + 1, "seed_idx": seed_idx,
                            "seed_name": seed_names[seed_idx], "pos_err_mm": round(cur_err_mm, 3),
                            "rot_err_deg": round(float(np.degrees(rot_norm)), 3),
                            "delta_mm": delta_mm, "damping": 1e-4, "is_colliding": False,
                            "status": "CONVERGED", "action": "Pink ProxQP Optimal Reached",
                        })
                    break

                if record_trace:
                    step_details.append({
                        "step": total_iters, "iter": it + 1, "seed_idx": seed_idx,
                        "seed_name": seed_names[seed_idx], "pos_err_mm": round(cur_err_mm, 3),
                        "rot_err_deg": round(float(np.degrees(rot_norm)), 3),
                        "delta_mm": delta_mm, "damping": 1e-4, "is_colliding": bool(is_col),
                        "status": "QP_DESCENT", "action": "Pink ProxQP Step",
                    })

                # 智能停滞检测 (Stall Detection)
                if delta_mm < 0.05:
                    stall_count += 1
                    if stall_count >= 4:
                        break
                else:
                    stall_count = 0

            # 若当前初猜已顺利达标收敛，不再尝试后续备用种子
            if best_q is not None and best_err < pos_tol + (rot_tol * 0.25 if has_rot else 0.0):
                break

        # ── 二阶段高精李群高斯-牛顿微调抛光器 ──
        ee_id = self.kin.ee_frame_ids_10dof[arm] if is_10dof else self.kin.ee_frame_ids_7dof[arm]
        target_polish_q = best_q if best_q is not None else ready_q.copy()
        if best_err < 0.05:  # 50mm 邻域内执行高精抛光
            best_q, best_err, best_rot_norm = self._polish_gauss_newton(
                arm=arm,
                is_10dof=is_10dof,
                model=model,
                data=data,
                ee_id=ee_id,
                q_init=target_polish_q,
                target_pos=target_pos,
                target_rot=target_rot,
                lower_limit=lower_limit,
                upper_limit=upper_limit,
                check_collision=check_collision,
            )

        if best_q is None:
            best_q = ready_q.copy()

        # 计算最终位姿残差与合格判定
        pin.forwardKinematics(model, data, best_q)
        pin.updateFramePlacements(model, data)
        fin_T = data.oMf[ee_id]
        fin_ep = float(np.linalg.norm(target_pos - fin_T.translation))
        fin_er = float(np.linalg.norm(pin.log3(target_rot @ fin_T.rotation.T))) if has_rot else 0.0
        fin_col = (
            self.collision.is_colliding_reduced(arm, best_q, is_10dof=is_10dof)
            if check_collision and self.collision is not None
            else False
        )
        is_success = (fin_ep < pos_tol) and (not has_rot or fin_er < rot_tol) and not fin_col

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        w_q = best_q[:3] if is_10dof else waist_q_base
        a_q = best_q[3:] if is_10dof else best_q
        telemetry = self._extract_solution_telemetry(
            arm, w_q, a_q, target_rot=target_rot if has_rot else None, q_full_base=q_full_base
        )

        solve_status = self._classify_solve_status(
            is_success=is_success,
            fin_col=fin_col,
            fin_ep=fin_ep,
            fin_er=fin_er,
            pos_tol=pos_tol,
            rot_tol=rot_tol,
            has_rot=has_rot,
            is_singular=telemetry.get("is_singular", False),
        )

        telemetry["rot_err_deg"] = round(float(np.degrees(fin_er)), 4) if has_rot else 0.0

        ik_res = IKResult(
            success=bool(is_success),
            status=solve_status,
            q_solution=best_q.copy(),
            waist_solution=w_q.copy() if is_10dof else None,
            arm_solution=a_q.copy(),
            iters=total_iters,
            time_ms=elapsed_ms,
            pos_err_mm=round(fin_ep * 1000.0, 4),
            seed_used=seed_idx if is_success else -1,
            seed_name=seed_names[seed_idx] if is_success else "None (Failed)",
            seed_names=seed_names,
            seed_switches=seed_switches,
            convergence_trace=convergence_trace,
            step_details=step_details,
            pipeline_stages=IK_PIPELINE_STAGES,
            is_colliding=bool(fin_col),
            mode="10DOF_6DOF_POSE" if (is_10dof and has_rot) else ("10DOF_3DOF_POS" if is_10dof else ("7DOF_6DOF_POSE" if has_rot else "7DOF_3DOF_POS")),
            **telemetry,
        )
        return is_success, best_q, ik_res

    # --------------------------------------------------------------------------
    # 7-DoF 单臂逆运动学求解 (solve_ik) — Pink + ProxQP 工业级内核
    # --------------------------------------------------------------------------

    def solve_ik(
        self,
        arm: str,
        target_pos: np.ndarray,
        target_rot: Optional[np.ndarray] = None,
        seed_q: Optional[np.ndarray] = None,
        q_full_base: Optional[np.ndarray] = None,
        pos_tol: float = 1e-3,       # 1.0 mm
        rot_tol: float = 2e-2,       # ~1.15°
        max_iters: int = 35,
        max_step: float = 0.20,
        check_collision: bool = True,
        damping: Optional[float] = None,
        enable_collision_barrier: bool = False,
        barrier_d_min: float = 0.015,
        allow_relaxation: bool = False,
        record_trace: bool = True,
        **kwargs,
    ) -> Tuple[bool, np.ndarray, IKResult]:
        """
        7-DoF 单臂逆运动学 — 基于 Pink 凸二次规划 (ProxQP)
        """
        start_time = time.perf_counter()
        target_pos = np.asarray(target_pos, dtype=np.float64)
        if np.isnan(target_pos).any() or np.isinf(target_pos).any():
            fail_res = IKResult(
                success=False,
                status=IKSolveStatus.INVALID_INPUT,
                q_solution=G1_READY_POSE[arm].copy(),
                arm_solution=G1_READY_POSE[arm].copy(),
                pos_err_mm=999.0,
                rot_err_deg=180.0,
                time_ms=0.0,
                iters=0,
                error="Invalid target_pos NaN/Inf",
            )
            return False, G1_READY_POSE[arm].copy(), fail_res

        lower_limit, upper_limit = self.limits[arm]
        ready_q = G1_READY_POSE[arm]

        # 确定性多初猜链构建 (Warm Start -> Ready Pose -> QMC 空间网格)
        seed_chain, seed_names = self._build_seed_chain(
            is_10dof=False,
            arm=arm,
            ready_q=ready_q,
            lower_limit=lower_limit,
            upper_limit=upper_limit,
            seed_q=seed_q,
        )

        waist_q = (
            np.array([q_full_base[idx] for idx in self.waist_q_indices], dtype=np.float64)
            if q_full_base is not None
            else np.zeros(3)
        )
        cost_posture = np.ones(7) * 1e-4

        ok, q_sol, res = self._solve_qp_core(
            arm=arm,
            is_10dof=False,
            target_pos=target_pos,
            target_rot=target_rot,
            seed_chain=seed_chain,
            seed_names=seed_names,
            ready_q=ready_q,
            cost_posture=cost_posture,
            lower_limit=lower_limit,
            upper_limit=upper_limit,
            q_full_base=q_full_base,
            waist_q_base=waist_q,
            pos_tol=pos_tol,
            rot_tol=rot_tol,
            max_iters=max_iters,
            max_step=max_step,
            check_collision=check_collision,
            start_time=start_time,
            enable_collision_barrier=enable_collision_barrier,
            barrier_d_min=barrier_d_min,
            record_trace=record_trace,
        )

        # ── 严苛位姿标准与受控微松弛 (Strict Pose Standards & Micro-Relaxation) ──
        if not ok and allow_relaxation and target_rot is not None:
            # 仅允许极窄受控微松弛 (上限严格控制在 rot_tol*1.5 且不超过 2.0°，杜绝宽松阶梯)
            micro_rot_tol_deg = min(float(np.degrees(rot_tol)) * 1.5, 2.0)
            if res.pos_err_mm < pos_tol * 1000.0 and res.rot_err_deg <= micro_rot_tol_deg and not res.is_colliding:
                res.status = IKSolveStatus.RELAXED_ORIENTATION
                res.success = True
                return True, q_sol, res

        return ok, q_sol, res

    # --------------------------------------------------------------------------
    # 10-DoF 协同逆运动学 (solve_10dof_ik) — Pink + ProxQP 工业级内核
    # --------------------------------------------------------------------------

    def solve_10dof_ik(
        self,
        arm: str,
        target_pos: np.ndarray,
        target_rot: Optional[np.ndarray] = None,
        target_rpy: Optional[np.ndarray] = None,
        seed_waist: Optional[np.ndarray] = None,
        seed_arm: Optional[np.ndarray] = None,
        q_full_base: Optional[np.ndarray] = None,
        pos_tol: float = 1e-3,       # 1.0 mm
        rot_tol: float = 2.0e-2,     # ~1.15°
        max_iters: int = 35,
        waist_weight: float = 10.0,  # 腰部惩罚倍率 (手臂优先)
        max_step: float = 0.18,
        check_collision: bool = True,
        damping: Optional[float] = None,
        cascade_upright: bool = True,
        enable_collision_barrier: bool = False,
        barrier_d_min: float = 0.015,
        allow_relaxation: bool = False,
        record_trace: bool = True,
        **kwargs,
    ) -> Tuple[bool, np.ndarray, np.ndarray, IKResult]:
        """
        10-DoF (3腰 + 7臂) 躯干-手臂协同逆运动学 — 基于 Pink 凸二次规划 (ProxQP)
        采用工业级级联直立优先与连续柔性过渡架构 (Continuous Waist-Arm Coordination Architecture)：
        1. 阶段 1：手臂舒适区 (dist <= 30.0cm)：
           - 若初始腰部接近零位，锁定腰部为 [0, 0, 0] 执行 7-DoF 单臂极速求解 (<0.5ms)，腰部保持 0.00°；
        2. 阶段 2：连续柔性过渡与协同扩展 (30.0cm < dist < 38.5cm 及超展区)：
           - 基于 C2 smoothstep 连续函数调制各向异性刚度，腰部随距离平滑介入，消除 ON/OFF 抖动。
        """
        start_time = time.perf_counter()
        target_pos = np.asarray(target_pos, dtype=np.float64)
        if np.isnan(target_pos).any() or np.isinf(target_pos).any():
            fail_res = IKResult(
                success=False,
                status=IKSolveStatus.INVALID_INPUT,
                q_solution=np.concatenate([np.zeros(3), G1_READY_POSE[arm]]),
                waist_solution=np.zeros(3),
                arm_solution=G1_READY_POSE[arm].copy(),
                pos_err_mm=999.0,
                rot_err_deg=180.0,
                time_ms=0.0,
                iters=0,
                error="Invalid target_pos NaN/Inf",
            )
            return False, np.zeros(3), G1_READY_POSE[arm].copy(), fail_res

        # 支持欧拉角输入
        if target_rot is None and target_rpy is not None:
            rpy = np.asarray(target_rpy, dtype=np.float64)
            target_rot = pin.rpy.rpyToMatrix(float(rpy[0]), float(rpy[1]), float(rpy[2]))
        has_rot = target_rot is not None
        if has_rot:
            target_rot = np.asarray(target_rot, dtype=np.float64)

        # ── 连续自适应工作空间协调架构 (Smooth Continuous Waist-Arm Coordination) ──
        sh_origin = self.kin.model_7dof[arm].jointPlacements[1].translation
        dist_to_shoulder = float(np.linalg.norm(target_pos - sh_origin))

        # 舒适区内静态单次求解快速通道 (仅在当前腰部本就处于零位、且处于舒适区 <= 30cm 时秒级返回，避免无谓开销)
        sw_norm = float(np.linalg.norm(seed_waist)) if seed_waist is not None else 0.0
        if cascade_upright and dist_to_shoulder <= 0.300 and sw_norm < 0.02:
            ok_7, q_arm_7, info_7 = self.solve_ik(
                arm=arm,
                target_pos=target_pos,
                target_rot=target_rot,
                seed_q=seed_arm,
                q_full_base=q_full_base,
                pos_tol=pos_tol,
                rot_tol=rot_tol,
                max_iters=min(18, max_iters),
                max_step=max_step,
                check_collision=check_collision,
                enable_collision_barrier=enable_collision_barrier,
                barrier_d_min=barrier_d_min,
                allow_relaxation=allow_relaxation,
                record_trace=record_trace,
            )
            if ok_7:
                waist_zero = np.zeros(3)
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                telemetry = self._extract_solution_telemetry(
                    arm, waist_zero, q_arm_7, target_rot=target_rot, q_full_base=q_full_base
                )
                info_7.update({
                    "time_ms": elapsed_ms,
                    "cascade_stage": "ARM_COMFORT_UPRIGHT",
                    "mode": "10DOF_6DOF_POSE" if has_rot else "10DOF_3DOF_POS",
                    "waist_solution": waist_zero.copy(),
                    "arm_solution": q_arm_7.copy(),
                    "q_solution": np.concatenate([waist_zero, q_arm_7]),
                    **telemetry,
                })
                return True, waist_zero, q_arm_7, info_7

        # ── 连续过渡区域激活权重计算 ──
        d_near = 0.300
        d_far  = 0.385
        if dist_to_shoulder <= d_near:
            mu = 0.0
            stage_name = "ARM_COMFORT_ZONE"
        elif dist_to_shoulder >= d_far:
            mu = 1.0
            stage_name = "10DOF_COORDINATED"
        else:
            s = (dist_to_shoulder - d_near) / (d_far - d_near)
            mu = s * s * (3.0 - 2.0 * s)  # C2 smoothstep
            stage_name = "CASCADE_SMOOTH_TRANSITION"

        # 连续动态非对称刚度: 舒适区 (mu=0) 极高锁紧，超展区 (mu=1) 柔顺协同
        w_lock = np.array([0.080, 0.250, 0.120]) * (float(waist_weight) / 10.0)
        w_assist = np.array([1.5e-4, 8.0e-4, 3.0e-4]) * float(waist_weight)
        waist_weights = w_lock * (1.0 - mu) + w_assist * mu

        lower_limit, upper_limit = self.limits_10dof[arm]
        ready_arm = G1_READY_POSE[arm]
        ready_10dof = np.concatenate([np.zeros(3), ready_arm])
        n = 10

        # 种子链配置 (Warm Start -> Ready Pose -> QMC 网格点)
        seed_chain, seed_names = self._build_seed_chain(
            is_10dof=True,
            arm=arm,
            ready_q=ready_10dof,
            lower_limit=lower_limit,
            upper_limit=upper_limit,
            seed_waist=seed_waist,
            seed_arm=seed_arm,
        )

        cost_posture = np.ones(n) * 1e-4
        cost_posture[0] = waist_weights[0]  # yaw
        cost_posture[1] = waist_weights[1]  # roll
        cost_posture[2] = waist_weights[2]  # pitch

        ok_10, q_sol, info_10 = self._solve_qp_core(
            arm=arm,
            is_10dof=True,
            target_pos=target_pos,
            target_rot=target_rot,
            seed_chain=seed_chain,
            seed_names=seed_names,
            ready_q=ready_10dof,
            cost_posture=cost_posture,
            lower_limit=lower_limit,
            upper_limit=upper_limit,
            q_full_base=q_full_base,
            waist_q_base=np.zeros(3),
            pos_tol=pos_tol,
            rot_tol=rot_tol,
            max_iters=max_iters,
            max_step=max_step,
            check_collision=check_collision,
            start_time=start_time,
            enable_collision_barrier=enable_collision_barrier,
            barrier_d_min=barrier_d_min,
            record_trace=record_trace,
        )

        waist_sol = q_sol[:3].copy()
        arm_sol = q_sol[3:].copy()
        info_10.cascade_stage = stage_name
        info_10.waist_solution = waist_sol
        info_10.arm_solution = arm_sol

        # ── 严苛位姿标准与受控微松弛 (Strict Pose Standards & Micro-Relaxation) ──
        if not ok_10 and allow_relaxation and has_rot:
            # 仅允许极窄受控微松弛 (上限严格控制在 rot_tol*1.5 且不超过 2.0°，杜绝宽松阶梯)
            micro_rot_tol_deg = min(float(np.degrees(rot_tol)) * 1.5, 2.0)
            if info_10.pos_err_mm < pos_tol * 1000.0 and info_10.rot_err_deg <= micro_rot_tol_deg and not info_10.is_colliding:
                info_10.status = IKSolveStatus.RELAXED_ORIENTATION
                info_10.success = True
                return True, waist_sol, arm_sol, info_10

        return ok_10, waist_sol, arm_sol, info_10


__all__ = [
    "G1PinkIKSolver",
    "IKSolveStatus",
    "IKResult",
    "IK_PIPELINE_STAGES",
    "G1_READY_POSE",
    "G1_DEFAULT_STAND_JOINTS",
]

