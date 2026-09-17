#!/usr/bin/env python3
"""
================================================================================
Unitree G1 机械臂混合逆运动学核心优化求解器 (Hybrid IK Numerical Solver)
================================================================================

职责：
1. 纯数学与优化算法核心：带自适应阻尼的最小二乘 (DLS) 与加权广义逆 (Weighted DLS)；
2. 多级精选种子链智能重启 (Smart Multi-Seed Pipeline)；
3. 零空间多任务投影优化 (Null-Space Projection)；
4. 通过依赖注入 (Dependency Injection) 接入运动学引擎与安全碰撞检测器，实现彻底解耦。
"""

import time
from typing import Dict, List, Optional, Tuple
import numpy as np
import pinocchio as pin

from deploy.kinematics.g1_model import (
    G1KinematicsModel,
    G1_JOINT_LIMITS,
    G1_WAIST_LIMITS,
    G1_READY_POSE,
)
from deploy.collision.g1_collision import G1CollisionChecker


class G1HybridIKSolver:
    """
    Unitree G1 混合逆运动学优化求解器
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
        # 依赖注入：运动学模型
        if kinematics is None:
            self.kin = G1KinematicsModel(urdf_path=urdf_path)
        else:
            self.kin = kinematics

        # 依赖注入：碰撞安全引擎
        if collision_checker is None:
            self.collision = G1CollisionChecker(self.kin)
        else:
            self.collision = collision_checker

        # ── 向后兼容性属性映射 (Backward-compatible Aliases) ──
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
        self.chain_10dof_indices: Dict[str, List[int]] = self.kin.chain_10dof_indices

        # 快捷方法委托
        self.forward_kinematics = self.kin.forward_kinematics
        self.forward_kinematics_10dof = self.kin.forward_kinematics_10dof
        self.is_colliding = self.collision.is_colliding

    # --------------------------------------------------------------------------
    # 7-DoF 单臂混合逆运动学求解 (solve_ik)
    # --------------------------------------------------------------------------

    def solve_ik(
        self,
        arm: str,
        target_pos: np.ndarray,
        target_rot: Optional[np.ndarray] = None,
        seed_q: Optional[np.ndarray] = None,
        q_full_base: Optional[np.ndarray] = None,
        pos_tol: float = 1e-3,        # 位置容差: 1mm
        rot_tol: float = 2e-2,        # 姿态容差: ~1.1 度
        max_iters: int = 80,
        damping: float = 0.01,
        max_step: float = 0.2,
        check_collision: bool = True, # 是否启用机身防碰撞门禁
    ) -> Tuple[bool, np.ndarray, Dict[str, float]]:
        """
        求解 7-DoF 高精度逆运动学 (带多级精选种子、自碰撞安全门禁与零空间姿态优化)
        :param arm: "left_arm" 或 "right_arm"
        :param target_pos: (3,) 空间目标位置 [x, y, z]
        :param target_rot: (3, 3) 目标旋转矩阵 (可选，若为 None 则只约束位置)
        :param seed_q: (7,) 初始猜测种子 (首选热启动)
        :param q_full_base: (29,) 全身基础姿势 (可选)
        :param check_collision: 是否激活机身防穿透安全保护 (默认 True)
        :return: (success: bool, q_solution: np.ndarray(7,), info: dict)
        """
        start_time = time.perf_counter()
        target_pos = np.asarray(target_pos, dtype=np.float64)
        has_rot = target_rot is not None
        if has_rot:
            target_rot = np.asarray(target_rot, dtype=np.float64)

        lower_limit, upper_limit = self.limits[arm]
        ready_q = G1_READY_POSE[arm]

        # ── 1. 构建精选种子链 (Smart Seeding Pipeline) ──
        seed_chain: List[np.ndarray] = []
        if seed_q is not None:
            seed_chain.append(np.clip(np.asarray(seed_q, dtype=np.float64), lower_limit, upper_limit))
        seed_chain.append(ready_q.copy())

        # 肘部伸直反向种子 (跳出局部极小)
        flipped_elbow = ready_q.copy()
        flipped_elbow[3] = 0.0
        seed_chain.append(flipped_elbow)

        # 4 组限位空间伪随机种子 (终极兜底)
        for _ in range(4):
            rnd = lower_limit + np.random.rand(7) * (upper_limit - lower_limit)
            seed_chain.append(rnd)

        best_q: Optional[np.ndarray] = None
        best_err = float("inf")
        total_iters = 0

        if q_full_base is None:
            q_full = pin.neutral(self.model)
        else:
            q_full = q_full_base.copy()

        indices = self.arm_q_indices[arm]
        ee_frame_id = self.ee_frame_ids[arm]
        iters_per_seed = 35

        # ── 2. 多级启发式迭代求解 ──
        for seed_idx, current_seed in enumerate(seed_chain):
            q_arm = current_seed.copy()

            for it in range(iters_per_seed):
                total_iters += 1
                for i, idx in enumerate(indices):
                    q_full[idx] = q_arm[i]

                # 前向运动学
                pin.forwardKinematics(self.model, self.data, q_full)
                pin.updateFramePlacements(self.model, self.data)

                oMf = self.data.oMf[ee_frame_id]
                cur_pos = oMf.translation
                cur_rot = oMf.rotation

                # 计算位姿误差
                pos_err = target_pos - cur_pos
                pos_norm = np.linalg.norm(pos_err)

                if has_rot:
                    R_err = target_rot @ cur_rot.T
                    rot_err = pin.log3(R_err)
                    rot_norm = np.linalg.norm(rot_err)
                    err_vec = np.hstack([pos_err, rot_err * 0.5])
                    err_val = pos_norm + rot_norm * 0.1
                else:
                    rot_norm = 0.0
                    err_vec = pos_err
                    err_val = pos_norm

                # 碰撞检查 (解耦自碰撞引擎)
                is_col = False
                if check_collision and self.collision is not None:
                    is_col = self.collision.is_colliding(arm, q_full, update_fk=False)

                # 仅记录无碰撞的历史最佳姿态
                if err_val < best_err and not is_col:
                    best_err = err_val
                    best_q = q_arm.copy()

                # 收敛判据（必须同时满足位姿容差且无机身自碰撞）
                if pos_norm < pos_tol and (not has_rot or rot_norm < rot_tol):
                    if not is_col:
                        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                        return True, q_arm, {
                            "iters": total_iters,
                            "time_ms": elapsed_ms,
                            "pos_err_mm": pos_norm * 1000.0,
                            "rot_err_deg": np.degrees(rot_norm),
                            "seed_used": seed_idx,
                            "is_colliding": False,
                        }
                    else:
                        # 发生自碰撞：在肩部施加反冲脉冲，继续搜索
                        if arm == "left_arm":
                            q_arm[1] += 0.08
                        else:
                            q_arm[1] -= 0.08

                # 自适应阻尼
                cur_damping = max(5e-4, min(0.02, pos_norm * 0.05))

                # 提取雅可比子空间
                J_full = pin.computeFrameJacobian(
                    self.model, self.data, q_full, ee_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
                )
                J_arm = J_full[:, indices] if has_rot else J_full[:3, indices]

                # 阻尼最小二乘伪逆
                m = J_arm.shape[0]
                damping_matrix = (cur_damping ** 2) * np.eye(m)
                J_pinv = J_arm.T @ np.linalg.solve(J_arm @ J_arm.T + damping_matrix, np.eye(m))

                # 主任务位移量
                dq_main = J_pinv @ err_vec

                # 零空间次级任务：就绪姿态 + 限位排斥 + 防碰撞外展势场
                q_mid = (lower_limit + upper_limit) / 2.0
                q_range = upper_limit - lower_limit
                grad_limits = -0.1 * (q_arm - q_mid) / (q_range ** 2 + 1e-4)
                grad_posture = -0.2 * (q_arm - ready_q)

                grad_repulse = np.zeros(7)
                if check_collision and self.collision is not None:
                    grad_repulse = self.collision.get_repulsion_gradient_7dof(arm, q_arm)

                null_proj = np.eye(7) - J_pinv @ J_arm
                dq_null = null_proj @ (grad_posture + grad_limits + grad_repulse)

                dq = dq_main + dq_null

                # 单步截断
                step_norm = np.linalg.norm(dq)
                if step_norm > max_step:
                    dq *= max_step / step_norm

                q_arm = np.clip(q_arm + dq, lower_limit, upper_limit)

        # 超时未完全收敛：返回无碰撞最佳解（若无则安全回退至就绪位）
        if best_q is None:
            best_q = ready_q.copy()
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return False, best_q, {
            "iters": total_iters,
            "time_ms": elapsed_ms,
            "pos_err_mm": best_err * 1000.0 if best_err != float("inf") else 999.0,
            "rot_err_deg": 0.0,
            "seed_used": -1,
            "is_colliding": False,
        }

    # --------------------------------------------------------------------------
    # 10-DoF 躯干-手臂协同加权逆运动学求解 (solve_10dof_ik)
    # --------------------------------------------------------------------------

    def solve_10dof_ik(
        self,
        arm: str,
        target_pos: np.ndarray,
        target_rot: Optional[np.ndarray] = None,
        seed_waist: Optional[np.ndarray] = None,
        seed_arm: Optional[np.ndarray] = None,
        q_full_base: Optional[np.ndarray] = None,
        pos_tol: float = 2e-3,        # 位置容差: 2mm
        rot_tol: float = 3e-2,        # 姿态容差: ~1.7 度
        max_iters: int = 50,
        waist_weight: float = 8.0,    # 腰部相对手臂的阻尼惩罚倍率 (手臂优先)
        max_step: float = 0.15,
        check_collision: bool = True,
    ) -> Tuple[bool, np.ndarray, np.ndarray, Dict[str, float]]:
        """
        求解 10-DoF (3-DoF 腰部 + 7-DoF 手臂) 躯干-手臂加权协同逆运动学
        采用非对称加权自适应阻尼最小二乘 (Weighted DLS) 与自碰撞安全闭环
        :param arm: "left_arm" 或 "right_arm"
        :param target_pos: (3,) 空间目标位置 [x, y, z]
        :param target_rot: (3, 3) 目标旋转矩阵 (可选)
        :param seed_waist: (3,) 初始腰部角度 (用于热启动)
        :param seed_arm: (7,) 初始手臂角度 (用于热启动)
        :param waist_weight: 腰部惩罚权重 (默认 8.0，实现“近处不动腰，远处才弯腰”)
        :param check_collision: 是否激活机身防穿透安全保护 (默认 True)
        :return: (success: bool, waist_q: np.ndarray(3,), arm_q: np.ndarray(7,), info: dict)
        """
        start_time = time.perf_counter()
        target_pos = np.asarray(target_pos, dtype=np.float64)
        has_rot = target_rot is not None
        if has_rot:
            target_rot = np.asarray(target_rot, dtype=np.float64)

        lower_limit, upper_limit = self.limits_10dof[arm]
        ready_arm = G1_READY_POSE[arm]
        ready_waist = np.zeros(3, dtype=np.float64)
        ready_10dof = np.concatenate([ready_waist, ready_arm])

        # ── 1. 构建 10-DoF 多级精选种子链 ──
        seed_chain: List[np.ndarray] = []
        if seed_waist is not None and seed_arm is not None:
            sw = np.clip(np.asarray(seed_waist, dtype=np.float64), self.waist_limits[0], self.waist_limits[1])
            sa = np.clip(np.asarray(seed_arm, dtype=np.float64), self.limits[arm][0], self.limits[arm][1])
            seed_chain.append(np.concatenate([sw, sa]))
        elif seed_arm is not None:
            sa = np.clip(np.asarray(seed_arm, dtype=np.float64), self.limits[arm][0], self.limits[arm][1])
            seed_chain.append(np.concatenate([ready_waist, sa]))

        seed_chain.append(ready_10dof.copy())

        # 腰部前倾俯仰种子 (前倾 ~14 度)
        bend_waist = ready_10dof.copy()
        bend_waist[2] = 0.25
        seed_chain.append(bend_waist)

        # 目标方位角偏航转向预估种子
        yaw_angle = np.arctan2(target_pos[1], max(1e-3, target_pos[0]))
        yaw_angle = np.clip(yaw_angle, lower_limit[0], upper_limit[0])
        turn_waist = ready_10dof.copy()
        turn_waist[0] = yaw_angle * 0.5
        seed_chain.append(turn_waist)

        # 3 组伪随机种子
        for _ in range(3):
            rnd = lower_limit + np.random.rand(10) * (upper_limit - lower_limit)
            seed_chain.append(rnd)

        # ── 2. 构造非对称加权度量矩阵 W (腰部惩罚大，手臂惩罚小) ──
        weights = np.array([
            waist_weight * 0.8,   # waist_yaw
            waist_weight * 1.5,   # waist_roll
            waist_weight * 1.0,   # waist_pitch
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0  # 7个手臂关节
        ], dtype=np.float64)
        W_inv = np.diag(1.0 / weights)

        best_q: Optional[np.ndarray] = None
        best_err = float("inf")
        total_iters = 0

        if q_full_base is None:
            q_full = pin.neutral(self.model)
        else:
            q_full = q_full_base.copy()

        indices = self.chain_10dof_indices[arm]
        ee_frame_id = self.ee_frame_ids[arm]
        iters_per_seed = 30

        # ── 3. 协同优化主循环 ──
        for seed_idx, current_seed in enumerate(seed_chain):
            q_10dof = current_seed.copy()

            for it in range(iters_per_seed):
                total_iters += 1
                for i, idx in enumerate(indices):
                    q_full[idx] = q_10dof[i]

                pin.forwardKinematics(self.model, self.data, q_full)
                pin.updateFramePlacements(self.model, self.data)

                oMf = self.data.oMf[ee_frame_id]
                cur_pos = oMf.translation
                cur_rot = oMf.rotation

                pos_err = target_pos - cur_pos
                pos_norm = np.linalg.norm(pos_err)

                if has_rot:
                    R_err = target_rot @ cur_rot.T
                    rot_err = pin.log3(R_err)
                    rot_norm = np.linalg.norm(rot_err)
                    err_vec = np.hstack([pos_err, rot_err * 0.5])
                    err_val = pos_norm + rot_norm * 0.1
                else:
                    rot_norm = 0.0
                    err_vec = pos_err
                    err_val = pos_norm

                is_col = False
                if check_collision and self.collision is not None:
                    is_col = self.collision.is_colliding(arm, q_full, update_fk=False)

                if err_val < best_err and not is_col:
                    best_err = err_val
                    best_q = q_10dof.copy()

                # 收敛判据
                if pos_norm < pos_tol and (not has_rot or rot_norm < rot_tol):
                    if not is_col:
                        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                        return True, q_10dof[:3].copy(), q_10dof[3:].copy(), {
                            "iters": total_iters,
                            "time_ms": elapsed_ms,
                            "pos_err_mm": pos_norm * 1000.0,
                            "rot_err_deg": np.degrees(rot_norm),
                            "seed_used": seed_idx,
                            "is_colliding": False,
                        }
                    else:
                        if arm == "left_arm":
                            q_10dof[4] += 0.08
                        else:
                            q_10dof[4] -= 0.08

                cur_damping = max(1e-3, min(0.03, pos_norm * 0.05))

                J_full = pin.computeFrameJacobian(
                    self.model, self.data, q_full, ee_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
                )
                J = J_full[:, indices] if has_rot else J_full[:3, indices]

                m = J.shape[0]
                A = J @ W_inv @ J.T + (cur_damping ** 2) * np.eye(m)
                dq_main = W_inv @ J.T @ np.linalg.solve(A, err_vec)

                # 零空间次级任务：腰部回正 + 就绪姿态 + 限位 + 防穿透排斥
                grad_waist = -0.6 * q_10dof[:3]
                grad_arm = -0.2 * (q_10dof[3:] - ready_arm)
                q_mid = (lower_limit + upper_limit) / 2.0
                q_range = upper_limit - lower_limit
                grad_limits = -0.1 * (q_10dof - q_mid) / (q_range ** 2 + 1e-4)

                grad_repulse = np.zeros(10)
                if check_collision and self.collision is not None:
                    grad_repulse = self.collision.get_repulsion_gradient_10dof(arm, q_10dof)

                N_proj = np.eye(10) - W_inv @ J.T @ np.linalg.solve(A, J)
                dq_null = N_proj @ (np.concatenate([grad_waist, grad_arm]) + grad_limits + grad_repulse)

                dq = dq_main + dq_null

                step_norm = np.linalg.norm(dq)
                if step_norm > max_step:
                    dq *= max_step / step_norm

                q_10dof = np.clip(q_10dof + dq, lower_limit, upper_limit)

        if best_q is None:
            best_q = np.concatenate([np.zeros(3), ready_arm])
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return False, best_q[:3].copy(), best_q[3:].copy(), {
            "iters": total_iters,
            "time_ms": elapsed_ms,
            "pos_err_mm": best_err * 1000.0 if best_err != float("inf") else 999.0,
            "rot_err_deg": 0.0,
            "seed_used": -1,
            "is_colliding": False,
        }
