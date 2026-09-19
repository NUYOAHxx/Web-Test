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
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pinocchio as pin

from deploy.kinematics.g1_model import (
    G1KinematicsModel,
    G1_JOINT_LIMITS,
    G1_WAIST_LIMITS,
    G1_READY_POSE,
    G1_DEFAULT_STAND_JOINTS,
)
from deploy.collision.g1_collision import G1CollisionChecker

# 逆运动学算法五大计算流水线阶段定义 (Stack-of-Tasks 工业标准架构)
IK_PIPELINE_STAGES = [
    {"id": 1, "name": "Seed Init",    "fullName": "Multi-Seed Warm Start",         "desc": "热启动 + 启发式多级种子链采样（目标偏航预估 / 腰部俯仰 / 随机保底）"},
    {"id": 2, "name": "Pinocchio FK", "fullName": "Reduced-Model FK (4.1x)",       "desc": "Pinocchio 10-DoF 裁剪子模型高速前向运动学 (<1μs/次) 与 SE(3) 末端位姿提取"},
    {"id": 3, "name": "SoT P1 Pos",   "fullName": "SVD-DLS Priority-1 Position",  "desc": "P1 主任务：SVD 自适应阻尼最小二乘位置求解 (Nakamura & Hanafusa 1986)"},
    {"id": 4, "name": "SoT P2 Rot",   "fullName": "SoT Priority-2 Orientation",   "desc": "P2 次任务：SO(3) 李代数姿态残差在 P1 零空间内严格分层求解 (Mansard 2009)"},
    {"id": 5, "name": "Null APF",     "fullName": "Null-Space Gradient Projection", "desc": "P3 最低优先级：关节限位势场 + CoM 质心平衡 + 碰撞排斥梯度投影到 P1∩P2 零空间"},
]


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
        self, arm: str, waist_q: np.ndarray, arm_q: np.ndarray, target_rot: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """提取解算构型下的 Pinocchio 刚体动力学、全身质心平衡与全 6D 空间位姿指标"""
        q_10dof = np.concatenate([waist_q, arm_q])
        q_full = self.build_q_10dof(arm, waist_q, arm_q)
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

        return {
            "com_pos": bal["com_pos"],
            "com_proj": bal["com_proj"],
            "balance_margin_mm": bal["margin_mm"],
            "balance_status": bal["status"],
            "support_polygon": bal["support_polygon"],
            "gravity_torques": [round(float(v), 3) for v in torques],
            "manipulability": manip,
            "actual_rpy_deg": act_rpy_deg,
            "actual_quat": [round(float(v), 5) for v in [act_quat.x, act_quat.y, act_quat.z, act_quat.w]],
            "target_rpy_deg": tgt_rpy_deg,
            "rot_err_deg": rot_err_deg,
        }

    # --------------------------------------------------------------------------
    # 工业标准工具函数 (Industrial-Grade Utilities)
    # --------------------------------------------------------------------------

    @staticmethod
    def _svd_dls(
        J: np.ndarray,
        sigma_0: float = 0.05,
        lambda_max: float = 0.12,
    ) -> np.ndarray:
        """
        SVD 自适应阻尼最小二乘伪逆 (Selective Damped Least-Squares)
        ─────────────────────────────────────────────────────────────
        参考文献: Nakamura & Hanafusa (1986) + Deo & Walker (1995) 逐奇异值阻尼
        原理:
          - 对雅可比 J 做 SVD: J = U Σ V^T
          - 对每个奇异值 σ_i 单独计算阻尼因子:
              若 σ_i ≥ σ₀: λᵢ = 0  (远离奇异点，不阻尼，接近精确伪逆)
              若 σ_i < σ₀: λᵢ² = λ_max² · (1-(σ_i/σ₀)²)  (平滑过渡)
          - 阻尼伪逆: J⁺ = V diag(σ_i/(σ_i²+λᵢ²)) U^T
        相比固定阻尼: 大奇异值时精度更高，小奇异值时稳定性更好
        :param J: 任意维度雅可比矩阵 (m×n)
        :param sigma_0: 奇异值阈值（触发阻尼的边界）
        :param lambda_max: 最大阻尼系数
        :return: J 的阻尼伪逆 (n×m)
        """
        U, sigma, Vt = np.linalg.svd(J, full_matrices=False)
        lambda_sq = np.where(
            sigma < sigma_0,
            lambda_max ** 2 * (1.0 - (sigma / sigma_0) ** 2),
            0.0,
        )
        sigma_inv = sigma / (sigma ** 2 + lambda_sq + 1e-16)
        return Vt.T @ (sigma_inv[:, np.newaxis] * U.T)

    @staticmethod
    def _joint_limit_gradient(
        q: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        margin_frac: float = 0.12,
        gain: float = 0.20,
    ) -> np.ndarray:
        """
        关节限位梯度势场 (Joint-Limit Repulsive Potential)
        ─────────────────────────────────────────────────
        在关节行程内侧 margin_frac (12%) 处激活排斥梯度，平滑推离硬限位
        :param q: 当前关节角 (n,)
        :param lower: 下限 (n,)
        :param upper: 上限 (n,)
        :param margin_frac: 激活区域占全程的比例
        :param gain: 梯度强度
        :return: 排斥梯度 (n,)，朝向关节中点
        """
        q_range = upper - lower + 1e-8
        q_mid   = (lower + upper) * 0.5
        q_norm  = (q - q_mid) / (q_range * 0.5)   # 归一化到 [-1, 1]
        thresh  = 1.0 - margin_frac
        active  = np.abs(q_norm) > thresh
        grad    = np.zeros_like(q)
        excess  = (np.abs(q_norm) - thresh) / margin_frac   # [0, 1]
        grad[active] = -gain * np.sign(q_norm[active]) * excess[active]
        return grad

    # --------------------------------------------------------------------------
    # 7-DoF 单臂逆运动学求解 (solve_ik) — SVD-DLS + Stack-of-Tasks
    # --------------------------------------------------------------------------

    def solve_ik(
        self,
        arm: str,
        target_pos: np.ndarray,
        target_rot: Optional[np.ndarray] = None,
        seed_q: Optional[np.ndarray] = None,
        q_full_base: Optional[np.ndarray] = None,
        pos_tol: float = 1e-3,
        rot_tol: float = 2e-2,
        max_iters: int = 60,
        max_step: float = 0.20,
        check_collision: bool = True,
        damping: Optional[float] = None,
        **kwargs,
    ) -> Tuple[bool, np.ndarray, Dict[str, float]]:
        """
        7-DoF 单臂逆运动学 — Stack-of-Tasks + SVD-DLS
        ─────────────────────────────────────────────
        P1 (最高优先级): 位置        —— SVD-DLS 伪逆
        P2 (次级优先级): 姿态在P1零空间 —— 不破坏已收敛的位置
        P3 (最低优先级): 关节限位 + 碰撞排斥 —— 次任务梯度
        """
        start_time = time.perf_counter()
        target_pos = np.asarray(target_pos, dtype=np.float64)
        has_rot = target_rot is not None
        if has_rot:
            target_rot = np.asarray(target_rot, dtype=np.float64)

        lower_limit, upper_limit = self.limits[arm]
        ready_q = G1_READY_POSE[arm]
        n = 7  # 关节数

        # ── 种子链 ──
        seed_chain: List[np.ndarray] = []
        seed_names: List[str] = []
        if seed_q is not None:
            seed_chain.append(np.clip(np.asarray(seed_q, dtype=np.float64), lower_limit, upper_limit))
            seed_names.append("Warm Start")
        seed_chain.append(ready_q.copy())
        seed_names.append("Ready Pose")
        flipped = ready_q.copy(); flipped[3] = 0.0
        seed_chain.append(flipped)
        seed_names.append("Flipped Elbow")
        for i in range(4):
            seed_chain.append(lower_limit + np.random.rand(n) * (upper_limit - lower_limit))
            seed_names.append(f"Random #{i+1}")

        best_q: Optional[np.ndarray] = None
        best_err = float("inf")
        best_rot_norm: float = 0.0
        total_iters = 0
        convergence_trace: List[float] = []
        seed_switches: List[int] = []
        step_details: List[Dict[str, Any]] = []

        if q_full_base is None:
            q_full = pin.neutral(self.model)
        else:
            q_full = q_full_base.copy()

        indices = self.arm_q_indices[arm]
        ee_frame_id = self.ee_frame_ids[arm]
        I_n = np.eye(n)

        for seed_idx, seed in enumerate(seed_chain):
            if total_iters >= max_iters:
                break
            q_arm = seed.copy()
            seed_switches.append(total_iters)
            prev_err_mm: Optional[float] = None
            stall_count = 0
            per_seed_budget = min(40, max_iters - total_iters)

            for it in range(per_seed_budget):
                total_iters += 1
                for i, idx in enumerate(indices):
                    q_full[idx] = q_arm[i]

                # ── FK ──
                pin.forwardKinematics(self.model, self.data, q_full)
                pin.updateFramePlacements(self.model, self.data)
                oMf = self.data.oMf[ee_frame_id]
                cur_pos = oMf.translation
                cur_rot = oMf.rotation

                # ── 误差 ──
                e_pos = target_pos - cur_pos
                pos_norm = float(np.linalg.norm(e_pos))
                cur_err_mm = pos_norm * 1000.0
                delta_mm = round(float(prev_err_mm - cur_err_mm), 3) if prev_err_mm is not None else 0.0
                prev_err_mm = cur_err_mm

                rot_norm = 0.0
                e_rot = np.zeros(3)
                if has_rot:
                    R_err = target_rot @ cur_rot.T
                    e_rot = pin.log3(R_err)
                    rot_norm = float(np.linalg.norm(e_rot))

                convergence_trace.append(round(cur_err_mm, 2))
                err_val = pos_norm + rot_norm * 0.1

                # ── 碰撞检查 ──
                is_col = False
                if check_collision and self.collision is not None:
                    is_col = self.collision.is_colliding(arm, q_full, update_fk=False)

                if err_val < best_err and not is_col:
                    best_err = err_val
                    best_q = q_arm.copy()
                    best_rot_norm = rot_norm

                status = "SEED_INIT" if it == 0 else "SoT_DESCENT"
                action = f"Seed ({seed_names[seed_idx]})" if it == 0 else "SVD-DLS SoT P1+P2+P3"
                if pos_norm < pos_tol and (not has_rot or rot_norm < rot_tol):
                    if not is_col:
                        step_details.append({"step": total_iters, "iter": it+1, "seed_idx": seed_idx,
                            "seed_name": seed_names[seed_idx], "pos_err_mm": round(cur_err_mm, 2),
                            "rot_err_deg": round(float(np.degrees(rot_norm)), 2),
                            "delta_mm": delta_mm, "damping": 0.0, "is_colliding": False,
                            "status": "CONVERGED", "action": "Target Reached"})
                        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                        return True, q_arm, {
                            "iters": total_iters, "time_ms": elapsed_ms,
                            "pos_err_mm": pos_norm * 1000.0, "rot_err_deg": float(np.degrees(rot_norm)),
                            "seed_used": seed_idx, "seed_name": seed_names[seed_idx],
                            "seed_names": seed_names, "seed_switches": seed_switches,
                            "convergence_trace": convergence_trace, "step_details": step_details,
                            "pipeline_stages": IK_PIPELINE_STAGES, "is_colliding": False,
                        }
                    else:
                        q_arm[1] += 0.08 if arm == "left_arm" else -0.08
                        step_details.append({"step": total_iters, "iter": it+1, "seed_idx": seed_idx,
                            "seed_name": seed_names[seed_idx], "pos_err_mm": round(cur_err_mm, 2),
                            "rot_err_deg": round(float(np.degrees(rot_norm)), 2), "delta_mm": delta_mm,
                            "damping": 0.0, "is_colliding": True, "status": "COLLISION_PULSE",
                            "action": "Repulsion Pulse"})
                else:
                    step_details.append({"step": total_iters, "iter": it+1, "seed_idx": seed_idx,
                        "seed_name": seed_names[seed_idx], "pos_err_mm": round(cur_err_mm, 2),
                        "rot_err_deg": round(float(np.degrees(rot_norm)), 2), "delta_mm": delta_mm,
                        "damping": 0.0, "is_colliding": bool(is_col),
                        "status": status, "action": action})

                # ── 雅可比 ──
                J_full = pin.computeFrameJacobian(
                    self.model, self.data, q_full, ee_frame_id,
                    pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
                J_pos = J_full[:3, indices]   # 3×7
                J_rot = J_full[3:, indices]   # 3×7

                # ── Stack-of-Tasks ──
                # P1: 位置主任务 (SVD-DLS)
                J1_pinv = self._svd_dls(J_pos)          # 7×3
                dq = J1_pinv @ e_pos                     # 7
                N1 = I_n - J1_pinv @ J_pos               # 7×7 (P1零空间)

                if has_rot:
                    # P2: 姿态次任务，严格在P1零空间内求解（不破坏位置）
                    J2_aug = J_rot @ N1                   # 3×7, P1零空间中的姿态雅可比
                    J2_aug_pinv = self._svd_dls(J2_aug)  # 7×3
                    dq += N1 @ (J2_aug_pinv @ e_rot)     # 7
                    N12 = N1 @ (I_n - J2_aug_pinv @ J2_aug)  # P1∩P2零空间
                else:
                    N12 = N1

                # P3: 关节限位 + 碰撞排斥（最低优先级，投影到 P1∩P2 零空间）
                grad_limits = self._joint_limit_gradient(q_arm, lower_limit, upper_limit)
                grad_posture = -0.15 * (q_arm - ready_q)
                grad_repulse = np.zeros(n)
                if check_collision and self.collision is not None:
                    grad_repulse = self.collision.get_repulsion_gradient_7dof(arm, q_arm)
                dq += N12 @ (grad_posture + grad_limits + grad_repulse)

                # ── 步长截断 + 关节限位裁剪 ──
                step_norm = np.linalg.norm(dq)
                if step_norm > max_step:
                    dq *= max_step / step_norm
                q_arm = np.clip(q_arm + dq, lower_limit, upper_limit)

                # ── 停滞检测：连续 8 步无改善则换种子 ──
                if delta_mm < 0.05:
                    stall_count += 1
                    if stall_count >= 8:
                        break
                else:
                    stall_count = 0

        if best_q is None:
            best_q = ready_q.copy()
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return False, best_q, {
            "iters": total_iters, "time_ms": elapsed_ms,
            "pos_err_mm": best_err * 1000.0 if best_err != float("inf") else 999.0,
            "rot_err_deg": float(np.degrees(best_rot_norm)),
            "seed_used": -1, "seed_name": "None (Failed)",
            "seed_names": seed_names, "seed_switches": seed_switches,
            "convergence_trace": convergence_trace, "step_details": step_details,
            "pipeline_stages": IK_PIPELINE_STAGES, "is_colliding": False,
        }


    # 10-DoF 协同逆运动学 (solve_10dof_ik) — Stack-of-Tasks 工业标准
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
        pos_tol: float = 1e-3,       # 位置收敛容差: 1.0 mm
        rot_tol: float = 2.0e-2,     # 姿态收敛容差: ~1.15°
        max_iters: int = 200,        # 全局最大迭代步数
        waist_weight: float = 8.0,   # 腰部惩罚倍率（手臂优先）
        max_step: float = 0.18,      # 单步最大关节位移 (rad)
        check_collision: bool = True,
        damping: Optional[float] = None,
        **kwargs,
    ) -> Tuple[bool, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        10-DoF (3腰 + 7臂) 躯干-手臂协同逆运动学 — Stack-of-Tasks 工业标准架构
        ─────────────────────────────────────────────────────────────────────────
        优先级层级:
          P1: 末端位置  (3D, 最高优先级) — SVD-DLS 自适应阻尼伪逆
          P2: 末端姿态  (3D, 在P1零空间) — 严格分层，位置已收敛后才修正姿态
          P3: 关节限位 + CoM平衡 + 碰撞排斥 (最低优先级, 在P1∩P2零空间)
        积分方式: pin.integrate (Riemannian流形积分，10-DoF裁剪子模型 4.1x加速)
        参考文献: Mansard & Chaumette (2009), Nakamura & Hanafusa (1986)
        """
        start_time = time.perf_counter()
        target_pos = np.asarray(target_pos, dtype=np.float64)

        # 支持欧拉角输入
        if target_rot is None and target_rpy is not None:
            rpy = np.asarray(target_rpy, dtype=np.float64)
            target_rot = pin.rpy.rpyToMatrix(float(rpy[0]), float(rpy[1]), float(rpy[2]))
        has_rot = target_rot is not None
        if has_rot:
            target_rot = np.asarray(target_rot, dtype=np.float64)

        # ── 运动学约束 ──
        lower_limit, upper_limit = self.limits_10dof[arm]
        ready_arm  = G1_READY_POSE[arm]
        ready_10dof = np.concatenate([np.zeros(3), ready_arm])
        n = 10  # 关节数
        I_n = np.eye(n)

        # 加权伪逆权重矩阵 W^{-1} (腰部高惩罚 → 手臂优先)
        w = np.ones(n)
        w[:3] = waist_weight          # 腰部 3 关节权重放大
        W_inv = np.diag(1.0 / w)     # W^{-1}: 各关节运动代价倒数
        W_inv_sqrt = np.diag(1.0 / np.sqrt(w))  # W^{-1/2}: 用于加权 SVD

        # ── 裁剪子模型 (4.1x 加速) ──
        model_r  = self.kin.model_10dof[arm]
        data_r   = self.kin.data_10dof[arm]
        ee_r_id  = self.kin.ee_frame_ids_10dof[arm]

        # ── 种子链 ──
        seed_chain: List[np.ndarray] = []
        seed_names: List[str] = []

        # 1. 热启动种子
        if seed_waist is not None and seed_arm is not None:
            sw = np.clip(np.asarray(seed_waist, dtype=np.float64), self.waist_limits[0], self.waist_limits[1])
            sa = np.clip(np.asarray(seed_arm,   dtype=np.float64), self.limits[arm][0], self.limits[arm][1])
            seed_chain.append(np.concatenate([sw, sa]))
            seed_names.append("Warm Start")

        # 2. 就绪姿态
        seed_chain.append(ready_10dof.copy())
        seed_names.append("Ready Pose")

        # 3. 腰部前倾种子（高前伸目标）
        fwd_seed = ready_10dof.copy()
        dx = float(target_pos[0]) - 0.25
        fwd_seed[2] = float(np.clip(dx * 0.6, self.waist_limits[0][2], self.waist_limits[1][2]))
        seed_chain.append(fwd_seed)
        seed_names.append("Waist Pitch")

        # 4. 目标偏航预估种子
        yaw_seed = ready_10dof.copy()
        yaw_est = float(np.arctan2(target_pos[1], target_pos[0] + 1e-6))
        yaw_seed[0] = float(np.clip(yaw_est * 0.5, self.waist_limits[0][0], self.waist_limits[1][0]))
        seed_chain.append(yaw_seed)
        seed_names.append("Yaw Estimate")

        # 5. 随机保底种子 × 3
        for ri in range(3):
            rnd = lower_limit + np.random.rand(n) * (upper_limit - lower_limit)
            rnd[:3] *= 0.35   # 腰部种子缩小范围，避免奇异姿态
            seed_chain.append(rnd)
            seed_names.append(f"Random #{ri+1}")

        # ── 全局最优追踪 ──
        best_q:        Optional[np.ndarray] = None
        best_err:      float = float("inf")
        best_rot_norm: float = 0.0
        total_iters:   int   = 0
        convergence_trace: List[float] = []
        seed_switches:     List[int]   = []
        step_details:      List[Dict[str, Any]] = []

        # ── 全身 q_full（用于碰撞检查 / CoM）──
        if q_full_base is None:
            q_full = pin.neutral(self.model)
        else:
            q_full = q_full_base.copy()
        indices = self.chain_10dof_indices[arm]

        # ── 主循环：多种子 Stack-of-Tasks ──
        for seed_idx, seed in enumerate(seed_chain):
            if total_iters >= max_iters:
                break
            q = np.clip(seed.copy(), lower_limit, upper_limit)
            seed_switches.append(total_iters)
            prev_err_mm: Optional[float] = None
            stall_count: int = 0
            per_seed_budget = min(40, max_iters - total_iters)

            for it in range(per_seed_budget):
                total_iters += 1

                # ── 2. FK (裁剪子模型，4.1x 加速) ──
                pin.forwardKinematics(model_r, data_r, q)
                pin.updateFramePlacements(model_r, data_r)
                oMf    = data_r.oMf[ee_r_id]
                cur_pos = oMf.translation
                cur_rot = oMf.rotation

                # ── 误差计算 ──
                e_pos    = target_pos - cur_pos
                pos_norm = float(np.linalg.norm(e_pos))
                cur_err_mm = pos_norm * 1000.0
                delta_mm   = round(float(prev_err_mm - cur_err_mm), 3) if prev_err_mm is not None else 0.0
                prev_err_mm = cur_err_mm

                rot_norm = 0.0
                e_rot    = np.zeros(3)
                if has_rot:
                    R_err    = target_rot @ cur_rot.T
                    e_rot    = pin.log3(R_err)              # SO(3) 李代数误差
                    rot_norm = float(np.linalg.norm(e_rot))

                convergence_trace.append(round(cur_err_mm, 2))
                err_val = pos_norm + rot_norm * 0.1

                # ── 碰撞检查 ──
                is_col = False
                if check_collision and self.collision is not None:
                    for i, idx in enumerate(indices):
                        q_full[idx] = q[i]
                    is_col = self.collision.is_colliding(arm, q_full, update_fk=True)

                if err_val < best_err and not is_col:
                    best_err      = err_val
                    best_q        = q.copy()
                    best_rot_norm = rot_norm

                # ── 收敛判据 ──
                if pos_norm < pos_tol and (not has_rot or rot_norm < rot_tol) and not is_col:
                    step_details.append({
                        "step": total_iters, "iter": it + 1,
                        "seed_idx": seed_idx, "seed_name": seed_names[seed_idx],
                        "pos_err_mm": round(cur_err_mm, 2),
                        "rot_err_deg": round(float(np.degrees(rot_norm)), 2),
                        "delta_mm": delta_mm, "damping": 0.0,
                        "is_colliding": False, "status": "CONVERGED",
                        "action": "Target Reached",
                    })
                    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                    telemetry  = self._extract_solution_telemetry(arm, q[:3], q[3:], target_rot=target_rot)
                    return True, q[:3].copy(), q[3:].copy(), {
                        "iters": total_iters, "time_ms": elapsed_ms,
                        "pos_err_mm": pos_norm * 1000.0,
                        "rot_err_deg": float(np.degrees(rot_norm)),
                        "seed_used": seed_idx, "seed_name": seed_names[seed_idx],
                        "seed_names": seed_names, "seed_switches": seed_switches,
                        "convergence_trace": convergence_trace,
                        "step_details": step_details,
                        "pipeline_stages": IK_PIPELINE_STAGES,
                        "is_colliding": False,
                        "mode": "10DOF_6DOF_POSE" if has_rot else "10DOF_3DOF_POS",
                        **telemetry,
                    }

                # ── 碰撞排斥脉冲 ──
                if is_col:
                    q[4] += 0.08 if arm == "left_arm" else -0.08
                    step_details.append({
                        "step": total_iters, "iter": it + 1,
                        "seed_idx": seed_idx, "seed_name": seed_names[seed_idx],
                        "pos_err_mm": round(cur_err_mm, 2),
                        "rot_err_deg": round(float(np.degrees(rot_norm)), 2),
                        "delta_mm": delta_mm, "damping": 0.0,
                        "is_colliding": True, "status": "COLLISION_PULSE",
                        "action": "Repulsion Pulse",
                    })
                    continue

                step_details.append({
                    "step": total_iters, "iter": it + 1,
                    "seed_idx": seed_idx, "seed_name": seed_names[seed_idx],
                    "pos_err_mm": round(cur_err_mm, 2),
                    "rot_err_deg": round(float(np.degrees(rot_norm)), 2),
                    "delta_mm": delta_mm, "damping": 0.0,
                    "is_colliding": False,
                    "status": "SEED_INIT" if it == 0 else "SoT_DESCENT",
                    "action": f"Seed ({seed_names[seed_idx]})" if it == 0 else "SoT P1+P2+P3",
                })

                # ── 3. 雅可比 (裁剪子模型) ──
                J_red = pin.computeFrameJacobian(
                    model_r, data_r, q, ee_r_id,
                    pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)  # 6×10
                J_pos_r = J_red[:3, :]   # 3×10
                J_rot_r = J_red[3:, :]   # 3×10

                # ── 4. Stack-of-Tasks (P1: 位置) — 加权 SVD-DLS ──
                # 加权形式: J_w = J @ W^{-1/2}, J^#_W = W^{-1/2} (J_w)^#
                J1_w      = J_pos_r @ W_inv_sqrt           # 3×10 (列缩放)
                J1_w_pinv = self._svd_dls(J1_w)            # 10×3
                J1_pinv   = W_inv_sqrt @ J1_w_pinv         # 10×3 加权伪逆
                dq        = J1_pinv @ e_pos                 # P1 主任务步
                N1        = I_n - J1_pinv @ J_pos_r        # P1 零空间投影子 (10×10)

                if has_rot:
                    # ── 5. SoT P2: 姿态在P1零空间 — 不破坏已收敛的位置 ──
                    J2_aug      = J_rot_r @ N1              # 3×10 (投影到P1零空间)
                    J2_aug_pinv = self._svd_dls(J2_aug)     # 10×3
                    dq         += N1 @ (J2_aug_pinv @ e_rot)
                    N12         = N1 @ (I_n - J2_aug_pinv @ J2_aug)  # P1∩P2 联合零空间
                else:
                    N12 = N1

                # ── 6. P3: 关节限位势场 + CoM 平衡 + 碰撞排斥 (最低优先级) ──
                # 关节限位排斥
                grad_limits  = self._joint_limit_gradient(q, lower_limit, upper_limit)

                # 关节舒适姿态引力（腰部归零，手臂回到 ready 姿态）
                grad_posture = np.zeros(n)
                grad_posture[:3] = -0.20 * q[:3]              # 腰部归中心
                grad_posture[3:] = -0.12 * (q[3:] - ready_arm) # 手臂回 ready

                # CoM 平衡梯度（全身质心保持在支撑多边形内）
                grad_com = np.zeros(n)
                try:
                    for i, idx in enumerate(indices):
                        q_full[idx] = q[i]
                    com_cur  = pin.centerOfMass(self.model, self.data, q_full)[:2]
                    com_ref  = np.array([0.02, 0.0])
                    J_com    = self.kin.compute_com_jacobian(q_full, arm=arm, is_10dof=True)
                    grad_com = -0.30 * (J_com[:2, :].T @ (com_cur - com_ref))
                except Exception:
                    pass

                # 碰撞排斥梯度
                grad_repulse = np.zeros(n)
                if check_collision and self.collision is not None:
                    try:
                        grad_repulse = self.collision.get_repulsion_gradient_10dof(arm, q)
                    except Exception:
                        pass

                dq += N12 @ (grad_posture + grad_limits + grad_com + grad_repulse)

                # ── 7. Riemannian 流形积分 + 限位截断 ──
                step_norm = np.linalg.norm(dq)
                if step_norm > max_step:
                    dq *= max_step / step_norm

                q = np.clip(
                    pin.integrate(model_r, q, dq),
                    lower_limit, upper_limit
                )

                # 停滞检测：连续 8 步无改善则换种子
                if delta_mm < 0.05:
                    stall_count += 1
                    if stall_count >= 8:
                        break
                else:
                    stall_count = 0

        # ── 失败兜底：返回全程最佳解 ──
        if best_q is None:
            best_q = ready_10dof.copy()
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        telemetry  = self._extract_solution_telemetry(
            arm, best_q[:3], best_q[3:], target_rot=target_rot if has_rot else None)
        return False, best_q[:3].copy(), best_q[3:].copy(), {
            "iters": total_iters, "time_ms": elapsed_ms,
            "pos_err_mm": best_err * 1000.0 if best_err != float("inf") else 999.0,
            "rot_err_deg": float(np.degrees(best_rot_norm)) if has_rot else 0.0,
            "seed_used": -1, "seed_name": "None (Failed)",
            "seed_names": seed_names, "seed_switches": seed_switches,
            "convergence_trace": convergence_trace,
            "step_details": step_details,
            "pipeline_stages": IK_PIPELINE_STAGES,
            "is_colliding": False,
            "mode": "10DOF_6DOF_POSE" if has_rot else "10DOF_3DOF_POS",
            **telemetry,
        }

