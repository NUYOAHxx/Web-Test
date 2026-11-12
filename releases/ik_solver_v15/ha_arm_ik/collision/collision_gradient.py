#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""碰撞排斥梯度: 见证点雅可比 + C¹ 双层单侧二次排斥势场。

数学核修复与增强:
* 见证点优先法向准则: 当 d > 0 时严格采用 (p_robot - p_obs) / ||p_robot - p_obs|| 作为远离单位向量,
  当 d < 0 (EPA 穿透) 时采用 (p_obs - p_robot) 结合法向与球心指向校验, 彻底消除 coal/FCL 不同图元法向符号反转导致的"反向吸入"死锁。
* 并发安全: 全程受 _obstacle_lock 保护, 防止后台 MoveIt 场景同步重建 geom_data 时引发 C++ 内存越界。
"""
from __future__ import annotations

import threading
from typing import Any, Dict, List, Tuple

import numpy as np
import pinocchio as pin

from .collision_constraints import CollisionReport


class CollisionGradient:
    """见证点雅可比与排斥势场梯度 (RobotCollisionManager 的功能组件)。"""

    _obstacle_lock: threading.RLock
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

    def _compute_safe_away_direction(
        self,
        robot_gid: int,
        obs_gid: int,
        p_robot: np.ndarray,
        p_obs: np.ndarray,
        d: float,
        raw_normal: np.ndarray,
    ) -> np.ndarray:
        """计算使机器人几何体远离障碍物的解析单位方向向量 u_away = ∂d/∂p_robot。

        消除对 FCL/coal DistanceResult.normal 符号约定的盲目依赖:
        1. 非穿透区 (d > 1e-7): 最近见证点连线 (p_robot - p_obs) 恒为精确远离方向;
        2. 穿透区 (d < -1e-7): 见证点位于对方表面内部, (p_obs - p_robot) 指向逃离穿透的最短路径;
        3. 接触临界区 (|d| <= 1e-7): 使用 raw_normal 并用几何体中心连线校验正负号。
        """
        diff = (p_robot - p_obs) if d >= 0.0 else (p_obs - p_robot)
        dn = float(np.linalg.norm(diff))
        if dn > 1e-7:
            return diff / dn

        # 见证点重合时 (刚好接触 d ≈ 0), 使用幾何中心连线校准法线方向
        c_robot = self.geom_data.oMg[robot_gid].translation
        c_obs = self.geom_data.oMg[obs_gid].translation
        center_vec = c_robot - c_obs
        cn = float(np.linalg.norm(center_vec))

        n_norm = float(np.linalg.norm(raw_normal))
        if n_norm > 1e-9:
            u_n = raw_normal / n_norm
            if cn > 1e-9 and float(np.dot(u_n, center_vec)) < 0.0:
                u_n = -u_n
            return u_n

        return (center_vec / cn) if cn > 1e-9 else np.array([0.0, 0.0, 1.0])

    def compute_collision_repulsion_gradient(
        self,
        q: np.ndarray,
        influence_distance: float = 0.08,
        safety_margin: float = 0.0,
        repulsion_gain: float = 0.25,
        compute_gradient: bool = True,
    ) -> Tuple[np.ndarray, float, CollisionReport]:
        """零空间排斥次任务负梯度 z_col、势能代价 H_col 及碰撞报告 (线程安全)。"""
        n_act = len(q)
        z_col = np.zeros(n_act, dtype=float)

        with self._obstacle_lock:
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

            self._sync_transforms(q, compute_joint_jacobians=compute_gradient)
            d_inf = max(float(influence_distance), float(safety_margin) + 0.01, 1e-4)
            self._update_active_pairs(d_inf)

            obs_gids = set(self.obstacle_geom_ids.values())
            collision_pairs = self.geom_model.collisionPairs

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

            # 第一段: security_margin = d_inf 膨胀判碰粗筛 (OBB 剪枝, 微秒级)
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

            # 第二段: 仅对进入影响半径 d_inf 的膨胀命中对运行精确 GJK 距离查询
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

            for k, gid1, gid2 in near_pairs:
                dr = dist_results[k]
                d = float(dr.min_distance)
                if d < min_obs_d:
                    min_obs_d = d

                if d <= threshold:
                    name1 = geom_to_link.get(gid1, geom_objs[gid1].name)
                    name2 = geom_to_link.get(gid2, geom_objs[gid2].name)
                    colliding_pairs.append((name1, name2))

                if d < d_inf:
                    u_outer = (d_inf - d) * inv_d_inf
                    cost_col += 0.5 * repulsion_gain * (u_outer * u_outer)
                    grad_coeff = outer_grad_scale * u_outer

                    if d < threshold:
                        u_inner = (threshold - d) * inv_scale_inner
                        cost_col += 0.5 * inner_gain * (u_inner * u_inner)
                        grad_coeff += inner_grad_scale * u_inner

                    if compute_gradient:
                        p1 = np.asarray(dr.getNearestPoint1(), dtype=float)
                        p2 = np.asarray(dr.getNearestPoint2(), dtype=float)
                        normal = np.asarray(dr.normal, dtype=float)

                        if gid1 in obs_gids:
                            robot_gid, obs_gid = gid2, gid1
                            p_robot, p_obs = p2, p1
                        else:
                            robot_gid, obs_gid = gid1, gid2
                            p_robot, p_obs = p1, p2

                        u_away = self._compute_safe_away_direction(
                            robot_gid, obs_gid, p_robot, p_obs, d, normal
                        )
                        J = self._point_jacobian(robot_gid, p_robot)
                        z_col += grad_coeff * (J.T @ u_away)

            report = CollisionReport(
                is_collision=bool(colliding_pairs),
                colliding_pairs=colliding_pairs,
                min_obstacle_distance=min_obs_d,
            )
            return z_col, cost_col, report