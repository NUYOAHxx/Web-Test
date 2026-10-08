"""V1 双臂 QP Position IK:Pose → q_goal (16 DoF = 腰2 + 左7 + 右7)。

每次迭代求解:
    min_dq  1/2 (J dq - e)^T W (J dq - e) + 1/2 (q+dq-q_seed)^T R (q+dq-q_seed)
    s.t.    max(q_min-q, -dq_max) <= dq <= min(q_max-q, dq_max)
    q <- q + step_scale * dq

只负责 "目标 Pose → 关节目标",不做碰撞/规划(交给 PlanningScene / OMPL)。
误差与 Jacobian 均在世界系(LOCAL_WORLD_ALIGNED):e = [p_d - p ; log3(R_d R^T)]。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import numpy as np
import pinocchio as pin
import qpsolvers
import scipy.sparse as sp

ARM_NAMES = ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
             "wrist_roll", "wrist_pitch", "wrist_yaw"]
DEFAULT_URDF = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../resources/g1/g1_29dof.urdf"))


class IKFailureReason(str, Enum):
    NONE = "none"
    MAX_ITERATIONS = "max_iterations"
    QP_FAILED = "qp_failed"
    STALLED = "stalled"          # Δq 过小但误差仍未收敛(目标不可达/卡在限位)
    INVALID_INPUT = "invalid_input"


@dataclass
class IKConfig:
    max_iterations: int = 100
    position_tolerance: float = 1e-3      # m
    orientation_tolerance: float = 1e-2   # rad
    step_scale: float = 1.0               # α
    stall_tolerance: float = 1e-7         # ||Δq|| 低于此视为停滞
    # 无进展早停:连续 no_progress_iterations 次迭代最优误差下降不足 progress_rel 即放弃本次
    progress_rel: float = 1e-3
    no_progress_iterations: int = 20      # 放宽:近收敛时更需要耐心
    # 多起点重试
    max_restarts: int = 20                # 增加默认次数(DLS提速后耗时极低)
    restart_noise: float = 0.3
    restart_rng_seed: int = 0
    # 最终精细迭代:所有重启结束后,如果最优误差 < near_tol 倍容差则再尝试
    near_tol_factor: float = 10.0         # 误差 < 这个倍容差时进入精细阶段
    fine_iterations: int = 200            # 精细阶段最大迭代次数
    fine_step_scale: float = 0.5          # 精细阶段步长缩小
    # 每步最大步长 (rad)
    max_step_waist: float = 0.10
    max_step_arm: float = 0.15
    # 任务权重 (W 对角)
    weight_position: float = 1.0
    weight_orientation: float = 0.5
    # Seed 权重 (R 对角)。阶段1 默认为 0:非零的 seed 拉力在 seed 较远时会把解卡在
    # 容差边缘(实测全局 seed 成功率 20%→90%),冗余选解由阶段2 零空间精修完成。
    seed_weight_waist: float = 0.0
    seed_weight_left_arm: float = 0.0
    seed_weight_right_arm: float = 0.0
    damping: float = 1.0e-6               # 保证 H 正定

    # 阶段2: 零空间向 seed 精修
    refine_iterations: int = 30           # 0 = 关闭
    refine_seed_weight: float = 1.0
    refine_position_slack: float = 0.9    # (已废弃,保留兼容;精修改用相对精修前误差的阈值)
    limit_margin: float = 1.0e-4          # 距限位的安全裕度 (rad)
    waist_joints: List[str] = field(
        default_factory=lambda: ["waist_yaw_joint", "waist_pitch_joint"])
    left_tip: str = "left_wrist_yaw_link"
    right_tip: str = "right_wrist_yaw_link"
    qp_solver: str = "osqp"
    # 主循环是否用完整 QP(默认关闭:用 DLS+投影,速度 17x)
    # 开启用于调试或有大量限位活跃的场景
    use_qp_step: bool = False


@dataclass
class IKResult:
    success: bool
    solution: np.ndarray
    position_error_left: float
    orientation_error_left: float
    position_error_right: float
    orientation_error_right: float
    total_error: float
    iterations: int
    failure_reason: IKFailureReason = IKFailureReason.NONE


class DualArmIKSolver:
    """q 顺序: [waist_yaw, waist_pitch, left(7), right(7)]。"""

    def __init__(self, config: Optional[IKConfig] = None,
                 urdf_path: str = DEFAULT_URDF) -> None:
        self.cfg = config or IKConfig()
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()

        self.joint_names: List[str] = (
            list(self.cfg.waist_joints)
            + [f"left_{n}_joint" for n in ARM_NAMES]
            + [f"right_{n}_joint" for n in ARM_NAMES])
        self.nq = len(self.joint_names)
        self.idx_q = np.array([self._joint(n).idx_q for n in self.joint_names])
        self.idx_v = np.array([self._joint(n).idx_v for n in self.joint_names])
        self.q_min = self.model.lowerPositionLimit[self.idx_q] + self.cfg.limit_margin
        self.q_max = self.model.upperPositionLimit[self.idx_q] - self.cfg.limit_margin
        self.tip_ids = [self.model.getFrameId(self.cfg.left_tip),
                        self.model.getFrameId(self.cfg.right_tip)]
        self._q_ref = pin.neutral(self.model)   # 其余关节(如 waist_roll)固定

        nw = len(self.cfg.waist_joints)
        c = self.cfg
        self.dq_max = np.concatenate([np.full(nw, c.max_step_waist),
                                      np.full(14, c.max_step_arm)])
        wt = [c.weight_position] * 3 + [c.weight_orientation] * 3
        self.W = np.diag(wt * 2)
        self.R = np.diag(np.concatenate([np.full(nw, c.seed_weight_waist),
                                         np.full(7, c.seed_weight_left_arm),
                                         np.full(7, c.seed_weight_right_arm)]))

    # ── 基础 ──
    def _joint(self, name: str):
        if not self.model.existJointName(name):
            raise KeyError(f"模型中缺少关节 {name}")
        return self.model.joints[self.model.getJointId(name)]

    def _full_q(self, q16: np.ndarray) -> np.ndarray:
        q = self._q_ref.copy()
        q[self.idx_q] = q16
        return q

    def forward_kinematics(self, q16: np.ndarray) -> List[pin.SE3]:
        pin.forwardKinematics(self.model, self.data, self._full_q(q16))
        pin.updateFramePlacements(self.model, self.data)
        return [self.data.oMf[i].copy() for i in self.tip_ids]

    def jacobian(self, q16: np.ndarray) -> np.ndarray:
        """12×16,世界对齐。"""
        pin.computeJointJacobians(self.model, self.data, self._full_q(q16))
        pin.updateFramePlacements(self.model, self.data)
        return np.vstack([
            pin.getFrameJacobian(self.model, self.data, i,
                                 pin.LOCAL_WORLD_ALIGNED)[:, self.idx_v]
            for i in self.tip_ids])

    @staticmethod
    def pose_error(target: pin.SE3, current: pin.SE3) -> np.ndarray:
        """6D 误差 [p_d-p ; log3(R_d R^T)](世界系)。"""
        return np.concatenate([
            target.translation - current.translation,
            pin.log3(target.rotation @ current.rotation.T)])

    def _errors(self, q, targets):
        poses = self.forward_kinematics(q)
        return np.concatenate([self.pose_error(t, p)
                               for t, p in zip(targets, poses)])

    @staticmethod
    def _norms(e: np.ndarray):
        return (np.linalg.norm(e[0:3]), np.linalg.norm(e[3:6]),
                np.linalg.norm(e[6:9]), np.linalg.norm(e[9:12]))

    def _converged(self, e: np.ndarray) -> bool:
        pl, ol, pr, orr = self._norms(e)
        c = self.cfg
        return (max(pl, pr) < c.position_tolerance
                and max(ol, orr) < c.orientation_tolerance)

    # ── 单步 QP ──
    def solve_step(self, q: np.ndarray, q_seed: np.ndarray,
                   e: np.ndarray) -> Optional[np.ndarray]:
        """QP 单步。默认用 DLS+盒投影(~36μs); use_qp_step=True 时用 OSQP(~600μs)。"""
        J = self.jacobian(q)
        JtW = J.T @ self.W
        H = JtW @ J + self.R + self.cfg.damping * np.eye(self.nq)
        g = -JtW @ e + self.R @ (q - q_seed)
        lb = np.maximum(self.q_min - q, -self.dq_max)
        ub = np.minimum(self.q_max - q, self.dq_max)
        if self.cfg.use_qp_step:
            kw = dict(eps_abs=1e-9, eps_rel=1e-9, max_iter=8000, verbose=False) \
                if self.cfg.qp_solver == "osqp" else {}
            dq = qpsolvers.solve_qp(sp.csc_matrix(H), g, lb=lb, ub=ub,
                                    solver=self.cfg.qp_solver, **kw)
            if dq is None or not np.all(np.isfinite(dq)):
                return None
            return dq
        # 默认: DLS 直接求解 + 盒投影(无等式约束时等价)
        try:
            dq = np.linalg.solve(H, -g)
        except np.linalg.LinAlgError:
            return None
        if not np.all(np.isfinite(dq)):
            return None
        return np.clip(dq, lb, ub)

    def _refine_to_seed(self, q: np.ndarray, seed: np.ndarray,
                        targets) -> np.ndarray:
        """阶段2:min ||q+dq-seed||²  s.t.  J dq = e (保持位姿)、限位、步长。
        失败/精度变差则回退到输入 q。"""
        c = self.cfg
        best = q
        n = self.nq
        # 精修不得明显破坏已有精度:误差上限 = max(2×精修前误差, 容差的 1%)
        err_limit = max(2.0 * float(np.linalg.norm(self._errors(q, targets))),
                        1e-2 * c.position_tolerance)
        Rm = c.refine_seed_weight * np.eye(n) + c.damping * np.eye(n)
        for _ in range(c.refine_iterations):
            e = self._errors(q, targets)
            J = self.jacobian(q)
            lb = np.maximum(self.q_min - q, -self.dq_max)
            ub = np.minimum(self.q_max - q, self.dq_max)
            kw = dict(eps_abs=1e-10, eps_rel=1e-10, max_iter=8000, verbose=False) \
                if c.qp_solver == "osqp" else {}
            dq = qpsolvers.solve_qp(sp.csc_matrix(Rm), Rm @ (q - seed),
                                    A=sp.csc_matrix(J), b=e, lb=lb, ub=ub,
                                    solver=c.qp_solver, **kw)
            if dq is None or not np.all(np.isfinite(dq)):
                break
            q_new = np.clip(q + dq, self.q_min, self.q_max)
            e_new = self._errors(q_new, targets)
            if not self._converged(e_new) or np.linalg.norm(e_new) > err_limit:
                break
            moved = np.linalg.norm(q_new - q)
            q = q_new
            best = q
            if moved < 1e-6:
                break
        return best

    def _polish(self, q: np.ndarray, seed: np.ndarray, targets,
                max_steps: int = 12, rel: float = 1e-2) -> np.ndarray:
        """收敛后将误差压到 ≤ tol*rel;任何一步变差或失败则保留上一个好解。"""
        c = self.cfg
        e = self._errors(q, targets)
        for _ in range(max_steps):
            pl, ol, pr, orr = self._norms(e)
            if (max(pl, pr) <= c.position_tolerance * rel and
                    max(ol, orr) <= c.orientation_tolerance * rel):
                break
            dq = self.solve_step(q, seed, e)
            if dq is None:
                break
            q_new = np.clip(q + dq, self.q_min, self.q_max)
            e_new = self._errors(q_new, targets)
            if np.linalg.norm(e_new) >= np.linalg.norm(e):
                break
            q, e = q_new, e_new
        return q

    # ── 主入口 ──
    def solve(self, seed: np.ndarray, left_target: pin.SE3,
              right_target: pin.SE3) -> IKResult:
        """先从 seed 求解;失败则多起点重启(少量扰动 seed,多数全域随机)。
        seed 偏置(冗余选解)始终指向调用方给的原始 seed。"""
        seed = np.asarray(seed, dtype=float)
        if seed.shape != (self.nq,):
            return self._result(False, seed, np.full(12, np.nan), 0,
                                IKFailureReason.INVALID_INPUT)
        targets = [left_target, right_target]
        seed = np.clip(seed, self.q_min, self.q_max)
        best = self._solve_once(seed, seed, targets)
        total_it = best.iterations
        rng = np.random.default_rng(self.cfg.restart_rng_seed)
        n = self.cfg.max_restarts
        seed_perturb_count = min(4, n // 4)
        for k in range(n):
            if best.success:
                break
            if k < seed_perturb_count:
                start = np.clip(seed + rng.normal(0, self.cfg.restart_noise, self.nq),
                                self.q_min, self.q_max)
            else:
                start = rng.uniform(self.q_min, self.q_max)
            cand = self._solve_once(start, seed, targets)
            total_it += cand.iterations
            if cand.success or cand.total_error < best.total_error:
                best = cand

        # ① 所有重启结束后:如果最优误差在 near_tol 倒容差内则从最优状态精细求解
        c = self.cfg
        if not best.success:
            pos_err = max(best.position_error_left, best.position_error_right)
            ori_err = max(best.orientation_error_left, best.orientation_error_right)
            if (pos_err < c.position_tolerance * c.near_tol_factor and
                    ori_err < c.orientation_tolerance * c.near_tol_factor):
                fine = self._solve_fine(best.solution, seed, targets)
                total_it += fine.iterations
                if fine.success or fine.total_error < best.total_error:
                    best = fine

        best.iterations = total_it
        return best

    def _solve_fine(self, start: np.ndarray, seed: np.ndarray, targets) -> IKResult:
        """精细阶段:步长缩小、迭代次数更多,从接近收敛的状态继续。"""
        orig_scale = self.cfg.step_scale
        self.cfg.step_scale = self.cfg.fine_step_scale
        r = self._solve_once(start, seed, targets,
                             max_it_override=self.cfg.fine_iterations)
        self.cfg.step_scale = orig_scale
        return r

    def _solve_once(self, start: np.ndarray, seed: np.ndarray,
                    targets, max_it_override: int = 0) -> IKResult:
        """单次局部迭代:从 start 出发,偏置指向 seed。
        max_it_override > 0 时覆盖 cfg.max_iterations。"""
        q = np.clip(start, self.q_min, self.q_max)
        e = self._errors(q, targets)
        reason = IKFailureReason.MAX_ITERATIONS
        it = 0
        max_it = max_it_override if max_it_override > 0 else self.cfg.max_iterations
        best_err, best_it = float(np.linalg.norm(e)), 0
        while it < max_it:
            if self._converged(e):
                break
            dq = self.solve_step(q, seed, e)
            if dq is None:
                reason = IKFailureReason.QP_FAILED
                break
            q_next = np.clip(q + self.cfg.step_scale * dq, self.q_min, self.q_max)
            step = np.linalg.norm(q_next - q)
            q = q_next
            e = self._errors(q, targets)
            it += 1
            if step < self.cfg.stall_tolerance:
                reason = IKFailureReason.STALLED
                break
            err = float(np.linalg.norm(e))
            if err < best_err * (1.0 - self.cfg.progress_rel):
                best_err, best_it = err, it
            elif it - best_it >= self.cfg.no_progress_iterations:
                reason = IKFailureReason.STALLED     # 误差不再下降:局部极小/不可达
                break
        # 统一收尾:无论从哪条路径收敛都走抛光+精修
        if self._converged(e):
            q = self._polish(q, seed, targets)
            if self.cfg.refine_iterations > 0:
                q = self._refine_to_seed(q, seed, targets)
                q = self._polish(q, seed, targets)
            e = self._errors(q, targets)
            return self._result(True, q, e, it, IKFailureReason.NONE)
        return self._result(False, q, e, it, reason)


    def _result(self, ok, q, e, it, reason) -> IKResult:
        pl, ol, pr, orr = self._norms(e)
        return IKResult(ok, q.copy(), pl, ol, pr, orr,
                        float(np.linalg.norm(e)), it, reason)
