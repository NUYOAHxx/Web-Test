#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""判碰约束: check_collision 机器人×障碍物净空判碰 (security_margin 制)。

判据语义: 障碍物净空 < safety_margin 判碰, 逐请求 security_margin 膨胀实现。
自碰撞不在本包检测范围 (归 MoveIt 按 SRDF ACM 口径在规划/执行层负责)。
"""
from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Any, Dict, List, Tuple

import numpy as np
import pinocchio as pin


@dataclass
class CollisionReport:
    """碰撞检测诊断报告 (只含机器人×障碍物通道)。"""

    is_collision: bool                      # 是否与障碍物碰撞 (净空 < safety_margin)
    colliding_pairs: List[Tuple[str, str]]  # 碰撞对列表 [(linkA, obstacleName)]
    min_obstacle_distance: float            # 机器人距障碍物的全局最近欧氏距离 [m]


class CollisionConstraints:
    """check_collision 判碰检查 (RobotCollisionManager 的功能组件)。"""

    _obstacle_lock: threading.RLock
    enabled: bool
    obstacles: Dict[str, Any]
    geom_model: pin.GeometryModel
    geom_data: pin.GeometryData
    geom_to_link: Dict[int, str]
    _pair_active: List[bool]

    def check_collision(
        self,
        q: np.ndarray,
        stop_at_first: bool = True,
        safety_margin: float = 0.0,
        compute_exact_min_distance: bool = False,
    ) -> CollisionReport:
        """机器人×障碍物碰撞检测 (线程安全, 支持可选全局精确最小距离查询)。

        Args:
            q: 机器人关节角度向量。
            stop_at_first: 发现第一个碰撞点即快速熔断退出 (提升检测吞吐)。
            safety_margin: 安全裕度 [m]——净空 < margin 判碰 (margin=0 等价穿透)。
            compute_exact_min_distance: 若为 True, 即使未发生碰撞也计算全局精确最短距离
                (用于监控曲线绘制或安全状态机, 避免未碰撞时返回 inf 跳变)。
        """
        with self._obstacle_lock:
            # 1. 未启用或无障碍物时秒级快速短路
            if not self.enabled or not self.obstacles:
                return CollisionReport(
                    is_collision=False,
                    colliding_pairs=[],
                    min_obstacle_distance=float("inf"),
                )
            self._sync_transforms(q)

            # 2. 设置安全裕度与激活表
            margin_f = float(max(safety_margin, 0.0))
            saved_margins = [
                float(cr.security_margin) for cr in self.geom_data.collisionRequests
            ]
            for cr in self.geom_data.collisionRequests:
                cr.security_margin = margin_f

            self._update_active_pairs(
                float("inf") if compute_exact_min_distance else margin_f
            )

            # 3. 运行碰撞检测并保证异常安全地还原 security_margin
            try:
                pin.computeCollisions(
                    self.geom_model, self.geom_data, stop_at_first
                )
            finally:
                for cr, m in zip(self.geom_data.collisionRequests, saved_margins):
                    cr.security_margin = m

            # 4. 局部属性绑定，加速热路径循环
            collision_pairs = self.geom_model.collisionPairs
            collision_results = self.geom_data.collisionResults
            geom_objs = self.geom_model.geometryObjects
            geom_to_link = self.geom_to_link
            pair_active = self._pair_active

            colliding_pairs: List[Tuple[str, str]] = []
            hit_ks: List[int] = []

            for k, is_active in enumerate(pair_active):
                if not is_active:
                    continue

                if collision_results[k].isCollision():
                    cp = collision_pairs[k]
                    gid1, gid2 = cp.first, cp.second
                    name1 = geom_to_link.get(gid1, geom_objs[gid1].name)
                    name2 = geom_to_link.get(gid2, geom_objs[gid2].name)
                    colliding_pairs.append((name1, name2))
                    hit_ks.append(k)

                    if stop_at_first:
                        break

            min_obs_d = float("inf")

            # 5. 靶向补测或全量精确距离提取
            if compute_exact_min_distance:
                pin.computeDistances(self.geom_model, self.geom_data)
                for dr in self.geom_data.distanceResults:
                    d = float(dr.min_distance)
                    if d < min_obs_d:
                        min_obs_d = d
            elif hit_ks:
                act = self.geom_data.activeCollisionPairs
                n_act = len(act)
                hit_set = set(hit_ks)
                saved_states: Dict[int, bool] = {}

                try:
                    for k in range(n_act):
                        target_state = k in hit_set
                        if act[k] != target_state:
                            saved_states[k] = act[k]
                            act[k] = target_state
                    pin.computeDistances(self.geom_model, self.geom_data)
                finally:
                    for k, prev_state in saved_states.items():
                        act[k] = prev_state

                dist_results = self.geom_data.distanceResults
                for k in hit_ks:
                    d = float(dist_results[k].min_distance)
                    if d < min_obs_d:
                        min_obs_d = d

            return CollisionReport(
                is_collision=bool(colliding_pairs),
                colliding_pairs=colliding_pairs,
                min_obstacle_distance=min_obs_d,
            )