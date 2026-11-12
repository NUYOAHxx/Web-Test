#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""碰撞排斥梯度: 见证点雅可比 + C¹ 双层单侧二次排斥势场。

零空间避障次任务的数学核: FCL 最近见证点 + Pinocchio 点雅可比解析求 ∂d/∂q,
构造 C¹ 单侧排斥势垒 H_col, 负梯度 z_col 投影至度量一致零空间。
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pinocchio as pin

from .collision_constraints import CollisionReport


class CollisionGradient:
    """见证点雅可比与排斥势场梯度 (Mixin)。"""

    # 声明 Mixin 依赖的宿主属性类型，提升 IDE 静态检查与可读性
    enabled: bool
    active_joint_names: List[str]
    pin_model: pin.Model
    pin_data: pin.Data
    geom_model: pin.GeometryModel
    geom_data: pin.GeometryData
    pin_maps: Dict[str, Any]
    obstacle_geom_ids: Dict[str, int]
    geom_to_link: Dict[int, str]
    _pair_active: List[bool]

    def _point_jacobian(self, gid: int, p_world: np.ndarray) -> np.ndarray:
        """几何体 gid 上空间见证点 p_world 对主动关节 q 的 3xN 线速度雅可比。

        v(p_W) = v(o_j) + ω_j × (p_W - o_j) = (J_trans - [p_W - o_j]_× J_rot) dq
        """
        n_act = len(self.active_joint_names)
        jid = self.geom_model.geometryObjects[gid].parentJoint
        if jid <= 0:
            return np.zeros((3, n_act), dtype=float)
        J_lwa = pin.getJointJacobian(
            self.pin_model, self.pin_data, jid, pin.LOCAL_WORLD_ALIGNED
        )
        J_act = J_lwa[:, self.pin_maps["active_v_indices"]]

        # 利用向量叉乘性质: -[r]_× @ ω = ω × r，避免每次构造 3x3 反对称矩阵与矩阵乘法
        r = p_world - self.pin_data.oMi[jid].translation
        return J_act[:3, :] + np.cross(J_act[3:, :], r, axisa=0, axisb=0, axisc=0)

    def compute_collision_repulsion_gradient(
        self,
        q: np.ndarray,
        influence_distance: float = 0.08,
        safety_margin: float = 0.0,
        repulsion_gain: float = 0.25,
        compute_gradient: bool = True,
    ) -> Tuple[np.ndarray, float, CollisionReport]:
        """零空间排斥次任务负梯度 z_col、势能代价 H_col 及碰撞报告。

        只覆盖机器人 × 环境障碍物排斥 (自碰撞归 MoveIt)。
        compute_gradient=False 为 cost-only 模式: 跳过雅可比, z_col 恒零。

        C¹ 连续双层单侧二次势垒 (影响半径 d_inf), 对 d < d_inf 的对:
          u = (d_inf - d) / d_inf
          H_col = 0.5 * repulsion_gain * ∑ u²
          z_col = ∑ (repulsion_gain / d_inf) * u * ∂d/∂q
        d < safety_margin 时叠加增益 2x 的内层高刚度项, 阈值处函数值与导数
        均连续 (无跳变振颤)。∂d/∂q 由 FCL 见证点经 _point_jacobian 精确求得。

        距离查询两段式 (mesh 档热路径关键优化): mesh×图元全量 GJK 距离
        要下钻 BVH 到三角形级 (~35ms/52 对), 而 DLS 每迭代只需排斥信号。
        先以 security_margin=d_inf 膨胀判碰 (OBB 层剪枝, ~0.05ms) 粗筛,
        仅膨胀命中 (d < d_inf 精确) 的对跑 pin.computeDistances 拿见证点
        距离/方向; 未命中的对 d >= d_inf, 势场与梯度恒零, 直接跳过。
        """
        n_act = len(q)
        z_col = np.zeros(n_act, dtype=float)

        # 提前拦截未启用或无环境障碍物场景，省去不必要的运动学与碰撞更新
        if not self.enabled or not self.obstacle_geom_ids:
            return (
                z_col,
                0.0,
                CollisionReport(
                    is_collision=False,
                    colliding_pairs=[],
                    min_obstacle_distance=float("inf"),
                ),
            )

        # 同步正运动学与关节雅可比 (修复原代码重复调用两次的冗余 bug)
        self._sync_transforms(q, compute_joint_jacobians=compute_gradient)
        # 仅激活机器人 × 障碍物碰撞对 (对表只含障碍物对, 见 collision_pair)
        d_inf = max(float(influence_distance), float(safety_margin) + 0.01, 1e-4)
        self._update_active_pairs(d_inf)

        obs_gids = set(self.obstacle_geom_ids.values())
        collision_pairs = self.geom_model.collisionPairs

        # 收集本轮真正需要评估的障碍物碰撞对索引
        active_obs_pairs: List[Tuple[int, int, int]] = []
        for k, cp in enumerate(collision_pairs):
            gid1, gid2 = cp.first, cp.second
            if (gid1 in obs_gids or gid2 in obs_gids) and self._pair_active[k]:
                active_obs_pairs.append((k, gid1, gid2))

        if not active_obs_pairs:
            return (
                z_col,
                0.0,
                CollisionReport(
                    is_collision=False,
                    colliding_pairs=[],
                    min_obstacle_distance=float("inf"),
                ),
            )

        # 两段式距离查询: 第一段膨胀判碰粗筛 (OBB 剪枝, 微秒级),
        # 只对"膨胀命中"的对跑第二段精确 GJK 距离。势场只在 d < d_inf
        # 时非零, 膨胀未命中的对 (d >= d_inf) 无需距离值。
        requests = self.geom_data.collisionRequests
        saved_margins = [float(cr.security_margin) for cr in requests]
        for cr in requests:
            cr.security_margin = d_inf
        try:
            pin.computeCollisions(self.geom_model, self.geom_data, False)
        finally:
            for cr, m in zip(requests, saved_margins):
                cr.security_margin = m

        near_pairs = [
            t
            for t in active_obs_pairs
            if self.geom_data.collisionResults[t[0]].isCollision()
        ]
        if not near_pairs:
            return (
                z_col,
                0.0,
                CollisionReport(
                    is_collision=False,
                    colliding_pairs=[],
                    min_obstacle_distance=float("inf"),
                ),
            )

        act = self.geom_data.activeCollisionPairs
        saved_act = [bool(act[k]) for k in range(len(act))]
        try:
            near_indices = {t[0] for t in near_pairs}
            for k in range(len(act)):
                act[k] = k in near_indices
            pin.computeDistances(self.geom_model, self.geom_data)
        finally:
            for k, v in enumerate(saved_act):
                act[k] = v

        active_obs_pairs = near_pairs

        # 预计算循环外常量与局部属性绑定，加速热路径迭代
        threshold = self._env_pair_threshold(safety_margin)
        inv_d_inf = 1.0 / d_inf
        outer_grad_scale = repulsion_gain * inv_d_inf

        inner_gain = 2.0 * repulsion_gain
        scale_inner = max(threshold, 0.01)
        inv_scale_inner = 1.0 / scale_inner
        inner_grad_scale = inner_gain * inv_scale_inner

        dist_results = self.geom_data.distanceResults
        geom_objs = self.geom_model.geometryObjects
        geom_to_link = self.geom_to_link

        colliding_pairs: List[Tuple[str, str]] = []
        min_obs_d = float("inf")
        cost_col = 0.0

        for k, gid1, gid2 in active_obs_pairs:
            dr = dist_results[k]
            d = float(dr.min_distance)
            if d < min_obs_d:
                min_obs_d = d

            if d <= threshold:
                name1 = geom_to_link.get(gid1, geom_objs[gid1].name)
                name2 = geom_to_link.get(gid2, geom_objs[gid2].name)
                colliding_pairs.append((name1, name2))

            # 进入势场影响半径才计算势能代价与排斥梯度
            if d < d_inf:
                # 1. 计算双层 C¹ 势能代价与标量梯度系数 (Cost-only 模式下同样生效)
                u_outer = (d_inf - d) * inv_d_inf
                cost_col += 0.5 * repulsion_gain * (u_outer * u_outer)
                grad_coeff = outer_grad_scale * u_outer

                if d < threshold:
                    u_inner = (threshold - d) * inv_scale_inner
                    cost_col += 0.5 * inner_gain * (u_inner * u_inner)
                    grad_coeff += inner_grad_scale * u_inner

                # 2. 计算见证点排斥梯度 (仅当 compute_gradient=True 时求解雅可比)
                if compute_gradient:
                    p1 = np.asarray(dr.getNearestPoint1(), dtype=float)
                    p2 = np.asarray(dr.getNearestPoint2(), dtype=float)
                    normal = np.asarray(dr.normal, dtype=float)

                    n_norm = float(np.linalg.norm(normal))
                    if n_norm > 1e-9:
                        u_away_1 = -normal / n_norm
                    else:
                        diff = (p1 - p2) if d >= 0.0 else (p2 - p1)
                        dn = float(np.linalg.norm(diff))
                        u_away_1 = (diff / dn) if dn > 1e-9 else np.array([0.0, 0.0, 1.0])

                    if gid1 in obs_gids:
                        robot_gid, p_robot, u_away = gid2, p2, -u_away_1
                    else:
                        robot_gid, p_robot, u_away = gid1, p1, u_away_1

                    J = self._point_jacobian(robot_gid, p_robot)
                    z_col += grad_coeff * (J.T @ u_away)

        has_obs = bool(colliding_pairs)
        report = CollisionReport(
            is_collision=has_obs,
            colliding_pairs=colliding_pairs,
            min_obstacle_distance=min_obs_d,
        )
        return z_col, cost_col, report