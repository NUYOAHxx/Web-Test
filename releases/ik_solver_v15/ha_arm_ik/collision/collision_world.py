#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CollisionWorld —— 碰撞场景统一门面 + CollisionContext 上下文 + RobotCollisionManager 引擎。

架构演进与修复要点:
* 引入 CollisionContext: 显式聚合运动学与几何状态, 为向组合模式 (Composition) 解耦提供清晰的数据边界。
* 消除外部状态污染: 对传入的 pin_maps 字典做浅拷贝隔离, 防止 set_frozen_template 隐式篡改主求解器字典。
* 实例级 RLock 并发安全: 统一初始化实例级 _obstacle_lock, 保护 CollisionWorld 所有门面查询。
* 移除冗余未用的 import hppfcl, 提升跨环境兼容性。
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
import os
from pathlib import Path
import threading
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pinocchio as pin

from ..ik_base import invert_T
from .broadphase import Broadphase
from .collision_config import BODY_HULL_LINKS
from .collision_constraints import CollisionConstraints, CollisionReport
from .collision_gradient import CollisionGradient
from .collision_pair import CollisionPairManager
from .distance_query import DistanceQuery
from .environment_collision import (
    MoveItSceneSync,
    ObstacleManager,
    PrimitiveObstacle,
)
from .robot_collision import (
    RobotCollisionGeometry,
    _apply_hull_decomposition,
    _resolve_package_dirs,
)
from .trajectory_validator import TrajectoryValidator


@dataclass
class CollisionContext:
    """碰撞引擎状态上下文容器 (支持长期架构向组合模式演进)。"""

    pin_model: Optional[pin.Model]
    pin_data: Optional[pin.Data]
    pin_maps: Optional[Dict[str, Any]]
    geom_model: pin.GeometryModel
    geom_data: pin.GeometryData
    link_geom_ids: Dict[str, List[int]]
    geom_to_link: Dict[int, str]
    obstacle_geom_ids: Dict[str, int]
    T_world_from_base: np.ndarray
    T_base_from_world: np.ndarray


class RobotCollisionManager(
    RobotCollisionGeometry,
    ObstacleManager,
    CollisionPairManager,
    Broadphase,
    DistanceQuery,
    CollisionGradient,
    CollisionConstraints,
    TrajectoryValidator,
):
    """模型无关机器人碰撞管理器 (只做机器人 × 环境障碍物检测)。"""

    def __init__(
        self,
        chain: Any,
        geom_model: Optional[pin.GeometryModel] = None,
        link_geom_ids: Optional[Dict[str, List[int]]] = None,
        geom_to_link: Optional[Dict[int, str]] = None,
        pin_model: Optional[pin.Model] = None,
        pin_maps: Optional[Dict[str, Any]] = None,
        T_world_from_base: Optional[np.ndarray] = None,
        active_joint_names: Optional[List[str]] = None,
        urdf_path: Optional[Union[str, Path]] = None,
        pin_uses_root_frame: bool = True,
    ):
        # 实例级可重入锁: 隔离不同实例的锁竞争, 保护读写并发安全
        self._obstacle_lock = threading.RLock()
        self.chain = chain
        self.obstacles: Dict[str, PrimitiveObstacle] = {}
        self.obstacle_geom_ids: Dict[str, int] = {}
        self.enabled: bool = True

        # pin_data 自建独立副本, pin_maps 浅拷贝隔离, 严禁污染主求解器内部状态
        self.pin_model = pin_model
        self.pin_data = pin_model.createData() if pin_model is not None else None
        self.pin_maps = dict(pin_maps) if pin_maps is not None else None
        self.active_joint_names = active_joint_names or (
            [j["name"] for j in chain.active_joints]
            if hasattr(chain, "active_joints")
            else []
        )

        if T_world_from_base is not None:
            self.T_world_from_base = np.asarray(
                T_world_from_base, dtype=float
            ).reshape(4, 4)
        else:
            self.T_world_from_base = np.eye(4, dtype=float)
        self.T_base_from_world = invert_T(self.T_world_from_base)
        self._pin_uses_root_frame = bool(pin_uses_root_frame)

        self._urdf_path = urdf_path
        if (
            geom_model is not None
            and link_geom_ids is not None
            and geom_to_link is not None
        ):
            self.geom_model = geom_model
            self.link_geom_ids = link_geom_ids
            self.geom_to_link = geom_to_link
        else:
            self.geom_model = pin.GeometryModel()
            self.link_geom_ids = {}
            self.geom_to_link = {}
            self._build_capsule_proxies()
            self._build_body_hull_proxies()

        self.rebuild_pairs()
        self.geom_data = pin.GeometryData(self.geom_model)
        self.geom_bounding_radii = self._compute_bounding_spheres()
        self._pair_active: List[bool] = [True] * len(self.geom_model.collisionPairs)

    @property
    def context(self) -> CollisionContext:
        """导出当前碰撞上下文快照，供组合式下游组件使用。"""
        return CollisionContext(
            pin_model=self.pin_model,
            pin_data=self.pin_data,
            pin_maps=self.pin_maps,
            geom_model=self.geom_model,
            geom_data=self.geom_data,
            link_geom_ids=self.link_geom_ids,
            geom_to_link=self.geom_to_link,
            obstacle_geom_ids=self.obstacle_geom_ids,
            T_world_from_base=self.T_world_from_base,
            T_base_from_world=self.T_base_from_world,
        )

    @classmethod
    def from_solver(
        cls,
        solver: Any,
        package_dirs: Optional[List[str]] = None,
        geometry: str = "mesh",
    ) -> "RobotCollisionManager":
        """从 IK 求解器实例自省并自动构建碰撞管理器 (7 轴纯臂 / 9 轴腰+臂)。"""
        base = getattr(solver, "solver_9dof", solver)
        chain = solver.chain

        pin_model = base.pin_model
        pin_maps = base.pin_maps
        urdf_path = getattr(base, "urdf", None) or getattr(base, "urdf_path", None)
        T_world_from_base = base.T_world_from_base
        active_joint_names = base.active_joint_names
        pin_uses_root_frame = getattr(base, "_pin_uses_root_frame", True)

        geometry_mode = str(geometry or "capsule").strip().lower()
        if geometry_mode not in ("capsule", "mesh"):
            raise ValueError(
                f"geometry 必须是 'capsule' 或 'mesh', 收到 '{geometry}'"
            )

        geom_model = None
        link_geom_ids: Dict[str, List[int]] = {}
        geom_to_link: Dict[int, str] = {}

        if geometry_mode == "mesh":
            if pin_model is None or not urdf_path or not os.path.exists(str(urdf_path)):
                raise RuntimeError(
                    "geometry='mesh' 需要 solver 持有有效 pin_model 与 URDF 路径, "
                    f"实际 pin_model={pin_model is not None}, urdf={urdf_path!r}"
                )

            all_package_dirs = _resolve_package_dirs()
            if package_dirs:
                all_package_dirs = list(package_dirs) + all_package_dirs

            try:
                geom_model = pin.buildGeomFromUrdf(
                    pin_model,
                    str(urdf_path),
                    pin.GeometryType.COLLISION,
                    package_dirs=all_package_dirs,
                )
            except Exception as e:
                raise RuntimeError(
                    f"geometry='mesh': 从 URDF 加载碰撞网格失败 ({e})"
                ) from e

            if geom_model.ngeoms == 0:
                raise RuntimeError("geometry='mesh': URDF 中未找到 collision 几何")

            frames = pin_model.frames
            for gid in range(geom_model.ngeoms):
                go = geom_model.geometryObjects[gid]
                l_name = frames[go.parentFrame].name
                geom_to_link[gid] = l_name
                link_geom_ids.setdefault(l_name, []).append(gid)

            _apply_hull_decomposition(
                geom_model, pin_model, all_package_dirs, urdf_path
            )

        mgr = cls(
            chain=chain,
            geom_model=geom_model,
            link_geom_ids=link_geom_ids if geom_model else None,
            geom_to_link=geom_to_link if geom_model else None,
            pin_model=pin_model,
            pin_maps=pin_maps,
            T_world_from_base=T_world_from_base,
            active_joint_names=active_joint_names,
            urdf_path=urdf_path,
            pin_uses_root_frame=pin_uses_root_frame,
        )
        mgr.geometry = geometry_mode
        return mgr

    def _sync_transforms(
        self, q: np.ndarray, compute_joint_jacobians: bool = False
    ) -> None:
        """将关节角 q 同步到 Pinocchio FK 并刷新 FCL 几何内存位姿。"""
        with self._obstacle_lock:
            if len(q) == self.pin_model.nq:
                q_pin = q
            else:
                from ..solver_dls import active_q_to_pinocchio_q
                q_pin = active_q_to_pinocchio_q(self.pin_model, self.pin_maps, q)
            if compute_joint_jacobians:
                pin.computeJointJacobians(self.pin_model, self.pin_data, q_pin)
            else:
                pin.forwardKinematics(self.pin_model, self.pin_data, q_pin)
            pin.updateGeometryPlacements(
                self.pin_model, self.pin_data, self.geom_model, self.geom_data
            )

    def set_frozen_template(self, q_full: Any) -> None:
        """显式设定冻结关节的全身模板，供镜像臂包络 FK 使用 (仅修改内部隔离副本)。"""
        if self.pin_model is None or self.pin_maps is None:
            return
        with self._obstacle_lock:
            if q_full is None:
                self.pin_maps.pop("q_template", None)
                return
            if isinstance(q_full, Mapping):
                q_pin = pin.neutral(self.pin_model)
                all_specs = self.pin_maps.get("all_specs", {})
                for name, val in q_full.items():
                    s = all_specs.get(name)
                    if s is None:
                        continue
                    idx = s["idx_q"]
                    if s["continuous"]:
                        v = float(val)
                        q_pin[idx : idx + 2] = (math.cos(v), math.sin(v))
                    else:
                        q_pin[idx] = float(val)
            else:
                q_pin = np.asarray(q_full, dtype=float).reshape(-1)
                if q_pin.shape[0] != self.pin_model.nq:
                    raise ValueError(
                        f"q_full 维度不匹配: 期望 {self.pin_model.nq}, 实际 {q_pin.shape[0]}"
                    )
            self.pin_maps["q_template"] = q_pin


class CollisionWorld:
    """碰撞场景统一门面——solver 侧唯一依赖。"""

    def __init__(self, manager: RobotCollisionManager):
        self.manager = manager

    def update_robot_configuration(self, q: np.ndarray) -> None:
        """更新全部机器人碰撞几何体位姿 (FK 同步, 线程安全)。"""
        self.manager._sync_transforms(q)

    def update_environment(self, obstacle_states: List[Dict[str, Any]]) -> None:
        """更新静态/动态环境物体 (整体对账: 注册/更新/移除, 幂等)。"""
        self.manager.sync_scene_objects(obstacle_states)

    def get_candidate_pairs(self) -> List[Tuple[int, int]]:
        """返回当前潜在危险碰撞对 (broad-phase 激活对, (gid1, gid2) 列表)。"""
        mgr = self.manager
        with mgr._obstacle_lock:
            mgr._update_active_pairs(0.0)
            pair_active = mgr._pair_active
            return [
                (cp.first, cp.second)
                for k, cp in enumerate(mgr.geom_model.collisionPairs)
                if pair_active[k]
            ]

    def query_active_distances(self) -> Dict[Tuple[int, int], float]:
        """查询活跃碰撞对的精确 GJK 距离 {(gid1, gid2): d}。"""
        mgr = self.manager
        with mgr._obstacle_lock:
            pin.computeDistances(mgr.geom_model, mgr.geom_data)
            out: Dict[Tuple[int, int], float] = {}
            dist_results = mgr.geom_data.distanceResults
            pair_active = mgr._pair_active
            for k, cp in enumerate(mgr.geom_model.collisionPairs):
                if pair_active[k]:
                    out[(cp.first, cp.second)] = float(dist_results[k].min_distance)
            return out

    def compute_collision_gradients(
        self, q: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, float, CollisionReport]:
        """计算排斥梯度 (透传 manager.compute_collision_repulsion_gradient)。"""
        return self.manager.compute_collision_repulsion_gradient(q, **kwargs)

    def validate_configuration(self, q: np.ndarray, **kwargs) -> CollisionReport:
        """验证指定构型的安全性 (透传 check_collision)。"""
        return self.manager.check_collision(q, **kwargs)

    def validate_trajectory(self, q_sequence: Any) -> Dict[str, Any]:
        """逐点验证一段轨迹的安全性 (逐帧 check_collision, 首碰熔断)。"""
        return self.manager.validate_trajectory(q_sequence)

    def __getattr__(self, name: str) -> Any:
        """未定义属性透传 manager，兼容平滑过渡。"""
        return getattr(self.manager, name)