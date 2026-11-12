#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Broad-phase 粗筛: 包围球保守下界激活对缓存 (DLS 每迭代热路径)。

激活结果写 self._pair_active 与 geom_data.activeCollisionPairs——判碰 / 排斥梯度 / 距离查询
共享同一份激活状态, 必须在 pin.computeDistances / computeCollisions 之前调用。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pinocchio as pin

from .collision_config import BROADPHASE_ENABLED


def compute_geometry_bounding_radius(
    geo: Any, fallback_radius: float = 0.05
) -> float:
    """计算单个 coal/FCL 几何体绕局部原点的保守外接球半径。

    严格遵循三角不等式上界: R = aabb_radius + ||aabb_center||。
    无论机器人连杆凸包还是外部环境障碍物 (含偏心 Mesh), 均統一调用此函数,
    消除因遗漏 ||aabb_center|| 导致的 Broad-phase 漏检。
    """
    try:
        if getattr(geo, "aabb_radius", -1.0) < 0.0:
            geo.computeLocalAABB()
        r = float(geo.aabb_radius) + float(np.linalg.norm(geo.aabb_center))
        return max(r, 0.0)
    except Exception:
        return float(fallback_radius)


class Broadphase:
    """包围球保守下界粗筛 (RobotCollisionManager 的功能组件)。"""

    geom_model: pin.GeometryModel
    geom_data: pin.GeometryData
    pin_data: pin.Data
    obstacle_geom_ids: Dict[str, int]
    geom_bounding_radii: np.ndarray
    _pair_active: List[bool]
    _bp_cache: Optional[Dict[str, Any]]

    def _compute_bounding_spheres(self) -> np.ndarray:
        """预计算每几何体的局部外接球半径 (以几何原点为球心, 保守上界)。"""
        n_geoms = self.geom_model.ngeoms
        radii = np.zeros(n_geoms, dtype=float)
        geom_objs = self.geom_model.geometryObjects

        for g in range(n_geoms):
            radii[g] = compute_geometry_bounding_radius(geom_objs[g].geometry)
        return radii

    def _update_active_pairs(self, influence_distance: float) -> int:
        """按几何球心距 - 半径和保守下界筛选激活碰撞对 (向量化热路径)。

        保守下界: d >= |c1 - c2| - (R1 + R2)。下界 >= influence_distance 时屏蔽。
        返回保持激活的对数。
        """
        n_pairs = len(self.geom_model.collisionPairs)
        if n_pairs == 0:
            self._pair_active = []
            return 0

        # 开关未开启或无穷大距离查询: 全部激活回退
        if not BROADPHASE_ENABLED or np.isinf(influence_distance):
            self._pair_active = [True] * n_pairs
            act = self.geom_data.activeCollisionPairs
            for k in range(len(act)):
                act[k] = True
            return n_pairs

        d_inf = float(influence_distance)
        n_geoms = self.geom_model.ngeoms

        # 1. 结构与对端索引缓存 (仅随碰撞对增删/模型变动/半径表更新时重建拓扑)
        topo_key = (
            n_pairs,
            n_geoms,
            len(self.obstacle_geom_ids),
            id(self.geom_model),
            len(self.geom_bounding_radii),
        )
        cached = getattr(self, "_bp_cache", None)

        if cached is None or cached["topo_key"] != topo_key:
            if len(self.geom_bounding_radii) != n_geoms:
                self.geom_bounding_radii = self._compute_bounding_spheres()

            first = np.empty(n_pairs, dtype=np.intp)
            second = np.empty(n_pairs, dtype=np.intp)
            for k, cp in enumerate(self.geom_model.collisionPairs):
                first[k] = cp.first
                second[k] = cp.second

            cached = {
                "topo_key": topo_key,
                "first": first,
                "second": second,
                # 预计算对端几何半径之和，避免每轮迭代重复求和
                "radii_sum": self.geom_bounding_radii[first]
                + self.geom_bounding_radii[second],
                "centers": np.empty((n_geoms, 3), dtype=float),
            }
            self._bp_cache = cached

        # 2. 批量提取全部几何体的世界坐标球心
        centers = cached["centers"]
        geom_objs = self.geom_model.geometryObjects
        oMi_all = self.pin_data.oMi

        for g in range(n_geoms):
            go = geom_objs[g]
            if go.parentJoint > 0:
                oMi = oMi_all[go.parentJoint]
                # c = p_joint + R_joint @ p_geom_local
                centers[g] = (
                    oMi.translation + oMi.rotation @ go.placement.translation
                )
            else:
                # 挂在 universe (joint 0) 上的环境障碍物直接取 placement.translation
                centers[g] = go.placement.translation

        # 3. 向量化粗筛 (坐标差 -> 欧氏范数 -> 保守下界)
        diff = centers[cached["first"]] - centers[cached["second"]]
        dist_centers = np.sqrt(np.einsum("ij,ij->i", diff, diff))
        d_lb = dist_centers - cached["radii_sum"]

        active_arr = d_lb < d_inf

        # 4. 同步状态至 Python 列表与 C++ geom_data.activeCollisionPairs
        self._pair_active = active_arr.tolist()
        act_cpp = self.geom_data.activeCollisionPairs
        for k, is_act in enumerate(self._pair_active):
            act_cpp[k] = is_act

        return int(np.count_nonzero(active_arr))