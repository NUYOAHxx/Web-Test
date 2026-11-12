#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""碰撞对管理: 机器人-环境对集合组装与环境净空判据。

碰撞对表只含机器人×障碍物对——自碰撞不在本包检测范围 (归 MoveIt 按 SRDF ACM 口径在规划/执行层负责)。
障碍物增删后走 rebuild_pairs 重组。
"""
from __future__ import annotations

from itertools import product
from typing import Dict, List, Optional

import pinocchio as pin


class CollisionPairManager:
    """碰撞对集合组装与环境判据 (RobotCollisionManager 的功能组件)。"""

    link_geom_ids: Dict[str, List[int]]
    geom_to_link: Dict[int, str]
    obstacle_geom_ids: Dict[str, int]
    geom_model: pin.GeometryModel
    geom_data: pin.GeometryData
    _pair_active: List[bool]
    _cached_robot_geoms: Optional[List[int]]

    def robot_geoms(self) -> List[int]:
        """全部机器人连杆几何 id (障碍物几何除外，带简单缓存)。"""
        cached = getattr(self, "_cached_robot_geoms", None)
        if cached is not None:
            return cached

        # 展平机器人连杆的所有几何体 ID
        gids: List[int] = [
            gid for gid_list in self.link_geom_ids.values() for gid in gid_list
        ]

        self._cached_robot_geoms = gids
        return gids

    def invalidate_robot_geoms_cache(self) -> None:
        """机器人模型几何体变更时清空缓存 (一般仅模型重载时调用)。"""
        self._cached_robot_geoms = None

    @staticmethod
    def _env_pair_threshold(safety_margin: float) -> float:
        """环境判据: 净空 < safety_margin 即判碰 (margin=0 时等价穿透)。"""
        return float(safety_margin)

    def rebuild_pairs(self) -> None:
        """重组机器人-障碍物碰撞对 (全部碰撞对均为机器人×障碍物对)。"""
        self.geom_model.removeAllCollisionPairs()
        add_pair = self.geom_model.addCollisionPair

        robot_gids = self.robot_geoms()
        obs_gids = list(self.obstacle_geom_ids.values())

        # 仅在两端几何体均存在时生成碰撞对
        if robot_gids and obs_gids:
            for r_gid, obs_gid in product(robot_gids, obs_gids):
                add_pair(pin.CollisionPair(r_gid, obs_gid))

        # 重建 GeometryData 与 broad-phase 激活表 (默认全激活, 首帧粗筛后收缩)
        self.geom_data = pin.GeometryData(self.geom_model)
        n_pairs = len(self.geom_model.collisionPairs)
        self._pair_active = [True] * n_pairs

        # 清除 Broadphase 拓扑缓存 (碰撞对列表已变更)
        if hasattr(self, "_bp_cache"):
            self._bp_cache = None