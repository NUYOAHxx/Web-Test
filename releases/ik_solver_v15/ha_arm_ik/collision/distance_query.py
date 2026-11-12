#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""精确距离查询: 指定障碍物的逐连杆最短距离与见证点。

查询接口需要全部机器人-障碍物对的精确距离, 不做 broad-phase 粗筛
(粗筛只在判碰/排斥梯度的热路径上启用)——本模块全激活跑一轮 GJK。
"""
from __future__ import annotations

import threading
from typing import Any, Dict, Optional

import numpy as np
import pinocchio as pin


class DistanceQuery:
    """GJK 距离 / 见证点查询 (RobotCollisionManager 的功能组件)。"""

    _obstacle_lock: threading.RLock
    obstacle_geom_ids: Dict[str, int]
    geom_model: pin.GeometryModel
    geom_data: pin.GeometryData
    geom_to_link: Dict[int, str]

    def compute_distances_to_obstacle(
        self, q: np.ndarray, obstacle_name: str
    ) -> Dict[str, Any]:
        """精确计算各连杆到指定障碍物的最短欧氏距离与空间见证接触点 (线程安全)。"""
        with self._obstacle_lock:
            target_gid = self.obstacle_geom_ids.get(obstacle_name)
            if target_gid is None:
                raise KeyError(f"未找到障碍物 '{obstacle_name}'")

            # 1. 状态同步与全激活距离计算
            self._sync_transforms(q)
            self._update_active_pairs(float("inf"))  # 全激活
            pin.computeDistances(self.geom_model, self.geom_data)

            # 2. 局部属性绑定，加速循环解析
            collision_pairs = self.geom_model.collisionPairs
            dist_results = self.geom_data.distanceResults
            geom_objs = self.geom_model.geometryObjects
            geom_to_link = self.geom_to_link

            link_dists: Dict[str, float] = {}
            min_d = float("inf")
            closest_link: Optional[str] = None
            closest_idx: Optional[int] = None
            closest_is_first: bool = False

            # 3. 遍历碰撞对，收集逐连杆距离与最近几何体索引
            for k, cp in enumerate(collision_pairs):
                gid1, gid2 = cp.first, cp.second
                if gid1 != target_gid and gid2 != target_gid:
                    continue

                dr = dist_results[k]
                d = float(dr.min_distance)

                is_target_first = gid1 == target_gid
                link_gid = gid2 if is_target_first else gid1
                l_name = geom_to_link.get(link_gid, geom_objs[link_gid].name)

                # 更新该连杆的最短距离
                prev_d = link_dists.get(l_name)
                if prev_d is None or d < prev_d:
                    link_dists[l_name] = d

                # 仅记录最小距离的索引与方向，避免在循环内部重复调用 np.asarray 转换点坐标
                if d < min_d:
                    min_d = d
                    closest_link = l_name
                    closest_idx = k
                    closest_is_first = is_target_first

            # 4. 仅对最终的全局最近接触对提取 3D 见证点
            witness_on_robot: Optional[np.ndarray] = None
            witness_on_obstacle: Optional[np.ndarray] = None

            if closest_idx is not None:
                best_dr = dist_results[closest_idx]
                p1 = np.asarray(best_dr.getNearestPoint1(), dtype=float)
                p2 = np.asarray(best_dr.getNearestPoint2(), dtype=float)
                witness_on_robot = p2 if closest_is_first else p1
                witness_on_obstacle = p1 if closest_is_first else p2

            return {
                "obstacle": obstacle_name,
                "min_distance": min_d,
                "closest_link": closest_link,
                "link_distances": link_dists,
                "witness_on_robot": witness_on_robot,
                "witness_on_obstacle": witness_on_obstacle,
            }