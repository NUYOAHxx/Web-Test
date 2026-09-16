#!/usr/bin/env python3
"""
Unitree G1 机械臂高精度混合逆运动学 (Hybrid Inverse Kinematics) 解算器
基于 Pinocchio 动力学计算图，结合带零空间投影的阻尼最小二乘 (DLS) 与多级精选种子启发式搜索。
"""

import os
import time
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pinocchio as pin


# 宇树 G1 机械臂物理限位 (单位: 弧度)
G1_JOINT_LIMITS = {
    # 左臂 7-DoF
    "left_shoulder_pitch_joint": (-3.0892, 2.6704),
    "left_shoulder_roll_joint": (-1.5882, 2.2515),
    "left_shoulder_yaw_joint": (-2.6180, 2.6180),
    "left_elbow_joint": (-1.0472, 2.0944),
    "left_wrist_roll_joint": (-1.9722, 1.9722),
    "left_wrist_pitch_joint": (-1.6144, 1.6144),
    "left_wrist_yaw_joint": (-1.6144, 1.6144),
    # 右臂 7-DoF
    "right_shoulder_pitch_joint": (-3.0892, 2.6704),
    "right_shoulder_roll_joint": (-2.2515, 1.5882),
    "right_shoulder_yaw_joint": (-2.6180, 2.6180),
    "right_elbow_joint": (-1.0472, 2.0944),
    "right_wrist_roll_joint": (-1.9722, 1.9722),
    "right_wrist_pitch_joint": (-1.6144, 1.6144),
    "right_wrist_yaw_joint": (-1.6144, 1.6144),
}

# 宇树 G1 腰部物理限位 (单位: 弧度)
G1_WAIST_LIMITS = {
    "waist_yaw_joint": (-2.6180, 2.6180),    # 左右旋转 ±150 度
    "waist_roll_joint": (-0.5200, 0.5200),   # 左右侧倾 ±30 度
    "waist_pitch_joint": (-0.5200, 0.5200),  # 前后俯仰 ±30 度
}

# 人体仿生预备就绪姿态 (肘部自然朝下微弯)
G1_READY_POSE = {
    "left_arm": np.array([0.2, 0.2, 0.0, 0.5, 0.0, 0.0, 0.0], dtype=np.float64),
    "right_arm": np.array([0.2, -0.2, 0.0, 0.5, 0.0, 0.0, 0.0], dtype=np.float64),
}


class G1HybridIKSolver:
    """
    Unitree G1 机械臂混合逆运动学求解器
    """

    def __init__(self, urdf_path: Optional[str] = None):
        if urdf_path is None:
            # 默认使用 ws_moveit 下的标准 URDF，或当前项目下的资源
            default_paths = [
                "/home/parallels/ws_moveit/src/g1_description/urdf/g1_29dof.urdf",
                os.path.join(os.path.dirname(__file__), "../resources/g1/g1_29dof.urdf"),
            ]
            for p in default_paths:
                if os.path.exists(p):
                    urdf_path = os.path.abspath(p)
                    break

        if not urdf_path or not os.path.exists(urdf_path):
            raise FileNotFoundError(f"无法找到 G1 URDF 模型文件: {urdf_path}")

        self.urdf_path = urdf_path
        self.model = pin.buildModelFromUrdf(self.urdf_path)
        self.data = self.model.createData()

        # 关节名称映射
        self.left_arm_joint_names = [
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
        ]

        self.right_arm_joint_names = [
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        ]

        # 末端连杆名称
        self.ee_frame_names = {
            "left_arm": "left_wrist_yaw_link",
            "right_arm": "right_wrist_yaw_link",
        }

        # 缓存各臂在全模型中的 q 索引
        self.arm_q_indices = {
            "left_arm": [self.model.getJointId(name) - 1 for name in self.left_arm_joint_names],
            "right_arm": [self.model.getJointId(name) - 1 for name in self.right_arm_joint_names],
        }

        # 提取限位数组 (7,)
        self.limits = {}
        for arm, jnames in [("left_arm", self.left_arm_joint_names), ("right_arm", self.right_arm_joint_names)]:
            lower = np.array([G1_JOINT_LIMITS[name][0] for name in jnames], dtype=np.float64)
            upper = np.array([G1_JOINT_LIMITS[name][1] for name in jnames], dtype=np.float64)
            self.limits[arm] = (lower, upper)

        # 腰部 3 自由度关节及 10-DoF 协同链定义
        self.waist_joint_names = [
            "waist_yaw_joint",
            "waist_roll_joint",
            "waist_pitch_joint",
        ]
        self.waist_q_indices = [self.model.getJointId(name) - 1 for name in self.waist_joint_names]
        waist_lower = np.array([G1_WAIST_LIMITS[name][0] for name in self.waist_joint_names], dtype=np.float64)
        waist_upper = np.array([G1_WAIST_LIMITS[name][1] for name in self.waist_joint_names], dtype=np.float64)
        self.waist_limits = (waist_lower, waist_upper)

        # 10-DoF 链 (3 腰 + 7 臂)
        self.chain_10dof_joint_names = {
            "left_arm": self.waist_joint_names + self.left_arm_joint_names,
            "right_arm": self.waist_joint_names + self.right_arm_joint_names,
        }
        self.chain_10dof_indices = {
            "left_arm": self.waist_q_indices + self.arm_q_indices["left_arm"],
            "right_arm": self.waist_q_indices + self.arm_q_indices["right_arm"],
        }
        self.limits_10dof = {}
        for arm in ["left_arm", "right_arm"]:
            l_10 = np.concatenate([waist_lower, self.limits[arm][0]])
            u_10 = np.concatenate([waist_upper, self.limits[arm][1]])
            self.limits_10dof[arm] = (l_10, u_10)

    def forward_kinematics(
        self, arm: str, q_arm: np.ndarray, q_full_base: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算正向运动学 (FK)
        :param arm: "left_arm" 或 "right_arm"
        :param q_arm: (7,) 单臂关节角
        :param q_full_base: (29,) 全身基础姿势 (可选)
        :return: (pos, rot_matrix)
        """
        if q_full_base is None:
            q_full = pin.neutral(self.model)
        else:
            q_full = q_full_base.copy()

        indices = self.arm_q_indices[arm]
        for i, idx in enumerate(indices):
            q_full[idx] = q_arm[i]

        pin.forwardKinematics(self.model, self.data, q_full)
        pin.updateFramePlacements(self.model, self.data)

        frame_id = self.model.getFrameId(self.ee_frame_names[arm])
        oMf = self.data.oMf[frame_id]
        return oMf.translation.copy(), oMf.rotation.copy()

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
    ) -> Tuple[bool, np.ndarray, Dict[str, float]]:
        """
        求解高精度逆运动学 (带多级精选种子与零空间姿态优化)
        :param arm: "left_arm" 或 "right_arm"
        :param target_pos: (3,) 目标空间坐标 [x, y, z]
        :param target_rot: (3, 3) 目标旋转矩阵 (可选，若为 None 则只约束位置)
        :param seed_q: (7,) 初始猜测种子 (首选热启动)
        :param q_full_base: (29,) 全身姿态
        :return: (success: bool, q_solution: np.ndarray, info: dict)
        """
        start_time = time.perf_counter()
        target_pos = np.asarray(target_pos, dtype=np.float64)
        has_rot = target_rot is not None
        if has_rot:
            target_rot = np.asarray(target_rot, dtype=np.float64)

        lower_limit, upper_limit = self.limits[arm]
        ready_q = G1_READY_POSE[arm]

        # 构建精选种子链 (Smart Seeding Pipeline)
        seed_chain = []
        if seed_q is not None:
            seed_chain.append(np.clip(np.asarray(seed_q, dtype=np.float64), lower_limit, upper_limit))
        seed_chain.append(ready_q.copy())

        # 肘部反向种子
        flipped_elbow = ready_q.copy()
        flipped_elbow[3] = 0.0  # 伸直
        seed_chain.append(flipped_elbow)

        # 精选多样化种子 (Smart Seeding)
        # 4 组限位内均匀随机种子 (终极兜底)
        for _ in range(4):
            rnd = lower_limit + np.random.rand(7) * (upper_limit - lower_limit)
            seed_chain.append(rnd)

        best_q = None
        best_err = float("inf")
        total_iters = 0

        # 全局基础姿势
        if q_full_base is None:
            q_full = pin.neutral(self.model)
        else:
            q_full = q_full_base.copy()

        indices = self.arm_q_indices[arm]
        ee_frame_id = self.model.getFrameId(self.ee_frame_names[arm])

        # 每个种子分配 35 步快速判定 (通常 5-15 步即可收敛)
        iters_per_seed = 35

        for seed_idx, current_seed in enumerate(seed_chain):
            q_arm = current_seed.copy()
            prev_err = float("inf")

            for it in range(iters_per_seed):
                total_iters += 1
                # 填入当前关节角
                for i, idx in enumerate(indices):
                    q_full[idx] = q_arm[i]

                # 前向运动学与雅可比计算
                pin.forwardKinematics(self.model, self.data, q_full)
                pin.updateFramePlacements(self.model, self.data)

                oMf = self.data.oMf[ee_frame_id]
                cur_pos = oMf.translation
                cur_rot = oMf.rotation

                # 位置误差
                pos_err = target_pos - cur_pos
                pos_norm = np.linalg.norm(pos_err)

                # 姿态误差
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

                if err_val < best_err:
                    best_err = err_val
                    best_q = q_arm.copy()

                # 收敛判据
                if pos_norm < pos_tol and (not has_rot or rot_norm < rot_tol):
                    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                    return True, q_arm, {
                        "iters": total_iters,
                        "time_ms": elapsed_ms,
                        "pos_err_mm": pos_norm * 1000.0,
                        "rot_err_deg": np.degrees(rot_norm),
                        "seed_used": seed_idx,
                    }

                # 自适应阻尼：远离目标时增加阻尼保稳定，临近目标时降低阻尼实现二次收敛
                cur_damping = max(5e-4, min(0.02, pos_norm * 0.05))

                # 计算空间雅可比并提取单臂 7 维子空间
                J_full = pin.computeFrameJacobian(
                    self.model, self.data, q_full, ee_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
                )
                if has_rot:
                    J_arm = J_full[:, indices]
                else:
                    J_arm = J_full[:3, indices]

                # 阻尼最小二乘伪逆
                m = J_arm.shape[0]
                damping_matrix = (cur_damping ** 2) * np.eye(m)
                J_pinv = J_arm.T @ np.linalg.solve(J_arm @ J_arm.T + damping_matrix, np.eye(m))

                # 主任务步长
                dq_main = J_pinv @ err_vec

                # 零空间次级任务：
                # 1. 引导手臂维持人体舒适下垂角 (ready_q)
                # 2. 排斥势场远离关节限位边界
                q_mid = (lower_limit + upper_limit) / 2.0
                q_range = upper_limit - lower_limit
                grad_limits = -0.1 * (q_arm - q_mid) / (q_range ** 2 + 1e-4)
                grad_posture = -0.2 * (q_arm - ready_q)
                null_proj = np.eye(7) - J_pinv @ J_arm
                dq_null = null_proj @ (grad_posture + grad_limits)

                dq = dq_main + dq_null

                # 限制单步角位移，防止过冲
                step_norm = np.linalg.norm(dq)
                if step_norm > max_step:
                    dq *= max_step / step_norm

                # 更新并施加硬件硬限位截断 (Box Clamping)
                q_arm = np.clip(q_arm + dq, lower_limit, upper_limit)

        # 超时未完全达到容差，返回最接近解
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return False, best_q, {
            "iters": total_iters,
            "time_ms": elapsed_ms,
            "pos_err_mm": best_err * 1000.0,
            "rot_err_deg": 0.0,
            "seed_used": -1,
        }

    def forward_kinematics_10dof(
        self, arm: str, q_waist: np.ndarray, q_arm: np.ndarray, q_full_base: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算 10-DoF (3腰 + 7臂) 正向运动学
        :param arm: "left_arm" 或 "right_arm"
        :param q_waist: (3,) 腰部关节角 [yaw, roll, pitch]
        :param q_arm: (7,) 单臂关节角
        :param q_full_base: (29,) 全身基础姿势 (可选)
        :return: (pos, rot_matrix)
        """
        if q_full_base is None:
            q_full = pin.neutral(self.model)
        else:
            q_full = q_full_base.copy()

        indices = self.chain_10dof_indices[arm]
        q_10dof = np.concatenate([np.asarray(q_waist, dtype=np.float64), np.asarray(q_arm, dtype=np.float64)])
        for i, idx in enumerate(indices):
            q_full[idx] = q_10dof[i]

        pin.forwardKinematics(self.model, self.data, q_full)
        pin.updateFramePlacements(self.model, self.data)

        frame_id = self.model.getFrameId(self.ee_frame_names[arm])
        oMf = self.data.oMf[frame_id]
        return oMf.translation.copy(), oMf.rotation.copy()

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
        waist_weight: float = 8.0,    # 腰部相对手臂的阻尼惩罚倍率 (手臂优先: 8.0 vs 1.0)
        max_step: float = 0.15,
    ) -> Tuple[bool, np.ndarray, np.ndarray, Dict[str, float]]:
        """
        求解 10 自由度 (3-DoF 腰部 + 7-DoF 手臂) 躯干-手臂协同加权逆运动学
        采用加权自适应阻尼最小二乘法 (Weighted DLS) 与零空间腰部直立回正势场
        :param arm: "left_arm" 或 "right_arm"
        :param target_pos: (3,) 空间目标位置 [x, y, z]
        :param target_rot: (3, 3) 目标旋转矩阵 (可选)
        :param seed_waist: (3,) 初始腰部角度 (可选，用于热启动)
        :param seed_arm: (7,) 初始手臂角度 (可选，用于热启动)
        :param waist_weight: 腰部惩罚权重 (默认 8.0，实现“近处不动腰，远处才弯腰”)
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

        # 1. 构建精选种子链 (Smart Seeding Pipeline for 10-DoF)
        seed_chain = []
        if seed_waist is not None and seed_arm is not None:
            sw = np.clip(np.asarray(seed_waist, dtype=np.float64), self.waist_limits[0], self.waist_limits[1])
            sa = np.clip(np.asarray(seed_arm, dtype=np.float64), self.limits[arm][0], self.limits[arm][1])
            seed_chain.append(np.concatenate([sw, sa]))
        elif seed_arm is not None:
            sa = np.clip(np.asarray(seed_arm, dtype=np.float64), self.limits[arm][0], self.limits[arm][1])
            seed_chain.append(np.concatenate([ready_waist, sa]))

        # Level 2: 默认直立就绪位
        seed_chain.append(ready_10dof.copy())

        # Level 3: 腰部前倾俯仰种子 (前倾 0.25 rad / 14.3 度，针对远距离或低处目标)
        bend_waist = ready_10dof.copy()
        bend_waist[2] = 0.25  # waist_pitch_joint 前倾
        seed_chain.append(bend_waist)

        # Level 4: 根据目标方位角预估腰部转向种子 (waist_yaw 转向目标)
        yaw_angle = np.arctan2(target_pos[1], max(1e-3, target_pos[0]))
        yaw_angle = np.clip(yaw_angle, lower_limit[0], upper_limit[0])
        turn_waist = ready_10dof.copy()
        turn_waist[0] = yaw_angle * 0.5  # 适度偏航
        seed_chain.append(turn_waist)

        # Level 5: 3 组限位内均匀随机种子 (终极兜底)
        for _ in range(3):
            rnd = lower_limit + np.random.rand(10) * (upper_limit - lower_limit)
            seed_chain.append(rnd)

        # 2. 构造非对称加权度量矩阵 W (腰部惩罚大，手臂惩罚小)
        weights = np.array([
            waist_weight * 0.8,   # waist_yaw
            waist_weight * 1.5,   # waist_roll (侧倾代价最大)
            waist_weight * 1.0,   # waist_pitch (弯腰前倾适度)
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0  # 7个手臂关节
        ], dtype=np.float64)
        W_inv = np.diag(1.0 / weights)

        best_q = None
        best_err = float("inf")
        total_iters = 0

        if q_full_base is None:
            q_full = pin.neutral(self.model)
        else:
            q_full = q_full_base.copy()

        indices = self.chain_10dof_indices[arm]
        ee_frame_id = self.model.getFrameId(self.ee_frame_names[arm])
        iters_per_seed = 30

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

                if err_val < best_err:
                    best_err = err_val
                    best_q = q_10dof.copy()

                if pos_norm < pos_tol and (not has_rot or rot_norm < rot_tol):
                    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                    return True, q_10dof[:3].copy(), q_10dof[3:].copy(), {
                        "iters": total_iters,
                        "time_ms": elapsed_ms,
                        "pos_err_mm": pos_norm * 1000.0,
                        "rot_err_deg": np.degrees(rot_norm),
                        "seed_used": seed_idx,
                    }

                cur_damping = max(1e-3, min(0.03, pos_norm * 0.05))

                J_full = pin.computeFrameJacobian(
                    self.model, self.data, q_full, ee_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
                )
                if has_rot:
                    J = J_full[:, indices]
                else:
                    J = J_full[:3, indices]

                m = J.shape[0]
                A = J @ W_inv @ J.T + (cur_damping ** 2) * np.eye(m)
                dq_main = W_inv @ J.T @ np.linalg.solve(A, err_vec)

                # 零空间次级任务：
                # 1. 腰部直立回正势场 (极强刚度拉向 0)
                # 2. 手臂维持仿生就绪姿态
                # 3. 软限位排斥力
                grad_waist = -0.6 * q_10dof[:3]
                grad_arm = -0.2 * (q_10dof[3:] - ready_arm)
                q_mid = (lower_limit + upper_limit) / 2.0
                q_range = upper_limit - lower_limit
                grad_limits = -0.1 * (q_10dof - q_mid) / (q_range ** 2 + 1e-4)

                N_proj = np.eye(10) - W_inv @ J.T @ np.linalg.solve(A, J)
                dq_null = N_proj @ (np.concatenate([grad_waist, grad_arm]) + grad_limits)

                dq = dq_main + dq_null

                step_norm = np.linalg.norm(dq)
                if step_norm > max_step:
                    dq *= max_step / step_norm

                q_10dof = np.clip(q_10dof + dq, lower_limit, upper_limit)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return False, best_q[:3].copy(), best_q[3:].copy(), {
            "iters": total_iters,
            "time_ms": elapsed_ms,
            "pos_err_mm": best_err * 1000.0,
            "rot_err_deg": 0.0,
            "seed_used": -1,
        }
