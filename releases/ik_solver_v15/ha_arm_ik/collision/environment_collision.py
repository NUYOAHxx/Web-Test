#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""环境障碍物管理: 解析图元障碍物生命周期 + MoveIt 规划场景自动感知。

架构演进与修复要点:
* 并发安全: 采用可重入锁 (RLock) 协同保护写路径与主线程读路径, 彻底消除 geom_model/geom_data 重建期间的 C++ 段错误风险。
* 批量对账单次重建: sync_scene_objects 将增删改合并后再执行单次 rebuild_pairs(), 真正将 O(N) 次 GeometryData 重建降为 O(1)。
* 复合障碍物与 SE(3) 位姿合成: 完整支持单个 CollisionObject 含多个 primitives/meshes 的复合场景, 并精确计算 T_world = T_object * T_sub。
* 统一日志与 coal/hppfcl 兼容: 全面替换裸 print 为标准 logger, 统一使用 compute_geometry_bounding_radius 计算粗筛半径。
"""
from __future__ import annotations

from dataclasses import dataclass
import logging
import threading
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pinocchio as pin

try:
    import coal
except ImportError:
    import hppfcl as coal

from .broadphase import compute_geometry_bounding_radius

logger = logging.getLogger(__name__)


@dataclass
class PrimitiveObstacle:
    """机器人作业空间内的解析几何障碍物 (box / sphere / cylinder / cone / mesh)。

    size 语义按 shape 区分 (均为全尺寸, 非半尺寸):
      - box:      [dx, dy, dz]  三轴边长 [m]
      - sphere:   [d]           直径 [m]
      - cylinder: [d, h]        直径 + 全高 [m] (轴向为局部 +Z, 朝向由 rpy/R 决定)
      - cone:     [d, h]        底面直径 + 全高 [m] (轴向局部 +Z, 顶朝 +Z 底朝 -Z)
      - mesh:     size 未用      mesh_path 指向的 STL/DAE 文件原样加载

    支持指定坐标系 ("world" 或 "base"), 自动依据机器人安装矩阵转换至统一计算坐标系。
    """

    name: str
    size: np.ndarray                  # [m], 语义按 shape 见 docstring
    position: np.ndarray              # [x, y, z] 中心点位置 [m]
    rpy: Optional[np.ndarray] = None  # [roll, pitch, yaw] 欧拉角 [rad]
    R: Optional[np.ndarray] = None    # 3x3 旋转矩阵
    frame: str = "world"              # "world" 或 "base"
    shape: str = "box"                # "box" | "sphere" | "cylinder" | "cone" | "mesh"
    mesh_path: Optional[str] = None   # shape="mesh" 时的碰撞网格文件路径
    source: str = "manual"            # "manual" 或 "moveit"

    _SHAPE_NDIM = {"box": 3, "sphere": 1, "cylinder": 2, "cone": 2, "mesh": 0}

    def __post_init__(self):
        self.shape = str(self.shape or "box").lower()
        if self.shape not in self._SHAPE_NDIM:
            raise ValueError(
                f"不支持的障碍物形状 '{self.shape}' "
                "(可选 box/sphere/cylinder/cone/mesh)"
            )
        if self._SHAPE_NDIM[self.shape] > 0:
            self.size = np.asarray(self.size, dtype=float).reshape(
                self._SHAPE_NDIM[self.shape]
            )
        else:
            self.size = np.asarray(self.size, dtype=float).reshape(-1)
        self.position = np.asarray(self.position, dtype=float).reshape(3)

        if self.R is None:
            if self.rpy is not None:
                self.rpy = np.asarray(self.rpy, dtype=float).reshape(3)
                self.R = self._rpy_to_matrix(self.rpy)
            else:
                self.R = np.eye(3, dtype=float)
        else:
            self.R = np.asarray(self.R, dtype=float).reshape(3, 3)

    def to_coal_geometry(self):
        """按 shape 构造对应 coal 碰撞图元。"""
        if self.shape == "sphere":
            return coal.Sphere(self.size[0] * 0.5)
        if self.shape == "cylinder":
            # coal.Cylinder(radius, halfLength), 轴向局部 +Z
            return coal.Cylinder(self.size[0] * 0.5, self.size[1] * 0.5)
        if self.shape == "cone":
            # coal.Cone(radius, halfLength), 轴向局部 +Z, 底面朝 -Z
            return coal.Cone(self.size[0] * 0.5, self.size[1] * 0.5)
        if self.shape == "mesh":
            if not self.mesh_path:
                raise FileNotFoundError(f"mesh 障碍物 '{self.name}' 未提供 mesh_path")
            return coal.MeshLoader().load(self.mesh_path)
        return coal.Box(self.size[0], self.size[1], self.size[2])

    @staticmethod
    def _quat_to_matrix(quat) -> Optional[np.ndarray]:
        """(x,y,z,w) 四元数 -> 3x3 旋转矩阵, 零四元数返回 None (不引 scipy)。"""
        qx, qy, qz, qw = (float(v) for v in quat)
        n2 = qx * qx + qy * qy + qz * qz + qw * qw
        if n2 <= 1e-12:
            return None

        s = 2.0 / n2
        return np.array(
            [
                [1 - s * (qy * qy + qz * qz), s * (qx * qy - qz * qw), s * (qx * qz + qy * qw)],
                [s * (qx * qy + qz * qw), 1 - s * (qx * qx + qz * qz), s * (qy * qz - qx * qw)],
                [s * (qx * qz - qy * qw), s * (qy * qz + qx * qw), 1 - s * (qx * qx + qy * qy)],
            ],
            dtype=float,
        )

    @staticmethod
    def _rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
        r, p, y = rpy
        cr, sr = np.cos(r), np.sin(r)
        cp, sp = np.cos(p), np.sin(p)
        cy, sy = np.cos(y), np.sin(y)
        return np.array(
            [
                [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                [-sp, cp * sr, cp * cr],
            ],
            dtype=float,
        )

    def to_se3(
        self,
        T_world_from_base: Optional[np.ndarray] = None,
        T_base_from_world: Optional[np.ndarray] = None,
        pin_uses_root_frame: bool = True,
    ) -> pin.SE3:
        """获取在 Pinocchio 模型根系（URDF 根）下的 SE3 位姿。

        依据 pin_uses_root_frame 与 frame ("world" / "base") 自动完成双向坐标转换:
        1) pin_uses_root_frame == True (Pinocchio 根为 World):
           - frame="base": T_root = T_world_from_base @ T_raw
           - frame="world": T_root = T_raw
        2) pin_uses_root_frame == False (Pinocchio 根为机器人基座 Base):
           - frame="base": T_root = T_raw
           - frame="world": T_root = T_base_from_world @ T_raw
        """
        T_raw = np.eye(4, dtype=float)
        T_raw[:3, :3] = self.R
        T_raw[:3, 3] = self.position

        is_base = self.frame.lower() in ("base", "local")

        if pin_uses_root_frame:
            if is_base:
                if T_world_from_base is not None:
                    T_root = T_world_from_base @ T_raw
                    return pin.SE3(T_root[:3, :3], T_root[:3, 3])
                return pin.SE3(T_raw[:3, :3], T_raw[:3, 3])
            else:
                return pin.SE3(T_raw[:3, :3], T_raw[:3, 3])
        else:
            if is_base:
                return pin.SE3(T_raw[:3, :3], T_raw[:3, 3])
            else:
                if T_base_from_world is not None:
                    T_root = T_base_from_world @ T_raw
                    return pin.SE3(T_root[:3, :3], T_root[:3, 3])
                elif T_world_from_base is not None:
                    from ..ik_base import invert_T
                    T_root = invert_T(T_world_from_base) @ T_raw
                    return pin.SE3(T_root[:3, :3], T_root[:3, 3])
                return pin.SE3(T_raw[:3, :3], T_raw[:3, 3])


class ObstacleManager:
    """解析障碍物生命周期管理 (add/update/remove/场景对账)。

    _obstacle_lock: 实例级 RLock (由 RobotCollisionManager.__init__ 初始化,
    此处保留类级回退以防独立混入测试)。写操作与检测热路径均通过该锁保护。
    """

    _obstacle_lock = threading.RLock()

    def _obstacle_to_se3(self, obs: PrimitiveObstacle) -> pin.SE3:
        """计算障碍物在 Pinocchio 模型根系下的 SE3 位姿。"""
        return obs.to_se3(
            T_world_from_base=getattr(self, "T_world_from_base", None),
            T_base_from_world=getattr(self, "T_base_from_world", None),
            pin_uses_root_frame=getattr(self, "_pin_uses_root_frame", True),
        )

    def add_obstacle(
        self,
        name: str,
        size: Union[List[float], Tuple[float, ...], np.ndarray],
        position: Union[List[float], Tuple[float, ...], np.ndarray],
        rpy: Optional[Union[List[float], Tuple[float, ...], np.ndarray]] = None,
        R: Optional[np.ndarray] = None,
        frame: str = "world",
        shape: str = "box",
        mesh_path: Optional[str] = None,
        source: str = "manual",
    ) -> PrimitiveObstacle:
        """添加障碍物 (box/sphere/cylinder/cone/mesh, 语义见 PrimitiveObstacle)。"""
        with self._obstacle_lock:
            return self._add_obstacle_impl(
                name,
                size=size,
                position=position,
                rpy=rpy,
                R=R,
                frame=frame,
                shape=shape,
                mesh_path=mesh_path,
                rebuild=True,
                source=source,
            )

    def _add_obstacle_impl(
        self,
        name: str,
        size: Union[List[float], Tuple[float, ...], np.ndarray],
        position: Union[List[float], Tuple[float, ...], np.ndarray],
        rpy: Optional[Union[List[float], Tuple[float, ...], np.ndarray]] = None,
        R: Optional[np.ndarray] = None,
        frame: str = "world",
        shape: str = "box",
        mesh_path: Optional[str] = None,
        rebuild: bool = True,
        source: str = "manual",
    ) -> PrimitiveObstacle:
        if name in self.obstacles:
            self._remove_obstacle_impl(name, rebuild=False)

        obs = PrimitiveObstacle(
            name=name,
            size=size,
            position=position,
            rpy=rpy,
            R=R,
            frame=frame,
            shape=shape,
            mesh_path=mesh_path,
            source=source,
        )
        coal_geo = obs.to_coal_geometry()
        radius = compute_geometry_bounding_radius(
            coal_geo, fallback_radius=float(np.linalg.norm(obs.size) * 0.5)
        )

        self.obstacles[name] = obs
        se3_placement = self._obstacle_to_se3(obs)
        go = pin.GeometryObject(name, 0, 0, se3_placement, coal_geo)
        obs_gid = self.geom_model.addGeometryObject(go)
        self.obstacle_geom_ids[name] = obs_gid

        self.geom_bounding_radii = np.concatenate(
            [self.geom_bounding_radii, [radius]]
        )
        if rebuild:
            self.rebuild_pairs()
        return obs

    def update_obstacle(
        self,
        name: str,
        position: Optional[Union[List[float], Tuple[float, ...], np.ndarray]] = None,
        size: Optional[Union[List[float], Tuple[float, ...], np.ndarray]] = None,
        rpy: Optional[Union[List[float], Tuple[float, ...], np.ndarray]] = None,
        R: Optional[np.ndarray] = None,
        frame: Optional[str] = None,
    ) -> bool:
        """动态更新障碍物的位置、朝向或形状 (size 语义随该障碍物的 shape)。"""
        with self._obstacle_lock:
            return self._update_obstacle_impl(
                name,
                position=position,
                size=size,
                rpy=rpy,
                R=R,
                frame=frame,
                rebuild=True,
            )

    def _update_obstacle_impl(
        self,
        name: str,
        position: Optional[Union[List[float], Tuple[float, ...], np.ndarray]] = None,
        size: Optional[Union[List[float], Tuple[float, ...], np.ndarray]] = None,
        rpy: Optional[Union[List[float], Tuple[float, ...], np.ndarray]] = None,
        R: Optional[np.ndarray] = None,
        frame: Optional[str] = None,
        rebuild: bool = True,
    ) -> bool:
        if name not in self.obstacles:
            return False

        obs = self.obstacles[name]
        if frame is not None:
            obs.frame = frame
        if position is not None:
            obs.position = np.asarray(position, dtype=float).reshape(3)
        if rpy is not None:
            obs.rpy = np.asarray(rpy, dtype=float).reshape(3)
            obs.R = PrimitiveObstacle._rpy_to_matrix(obs.rpy)
        if R is not None:
            obs.R = np.asarray(R, dtype=float).reshape(3, 3)
            obs.rpy = None
        if size is not None:
            obs.size = np.asarray(size, dtype=float).reshape(
                PrimitiveObstacle._SHAPE_NDIM[obs.shape]
            )
            self._add_obstacle_impl(
                name,
                obs.size,
                obs.position,
                rpy=obs.rpy,
                R=obs.R,
                frame=obs.frame,
                shape=obs.shape,
                mesh_path=obs.mesh_path,
                rebuild=rebuild,
                source=getattr(obs, "source", "manual"),
            )
        else:
            obs_gid = self.obstacle_geom_ids[name]
            new_placement = self._obstacle_to_se3(obs)
            self.geom_model.geometryObjects[obs_gid].placement = new_placement
            if (
                hasattr(self, "geom_data")
                and self.geom_data is not None
                and obs_gid < len(self.geom_data.oMg)
            ):
                self.geom_data.oMg[obs_gid] = new_placement

        return True

    def remove_obstacle(self, name: str) -> bool:
        """移除指定的外部障碍物。"""
        with self._obstacle_lock:
            return self._remove_obstacle_impl(name, rebuild=True)

    def _remove_obstacle_impl(self, name: str, rebuild: bool = True) -> bool:
        if name not in self.obstacles:
            return False
        del self.obstacles[name]
        new_geom_model = pin.GeometryModel()
        new_link_geom_ids: Dict[str, List[int]] = {}
        new_geom_to_link: Dict[int, str] = {}

        for l_name, gids in self.link_geom_ids.items():
            for gid in gids:
                go = self.geom_model.geometryObjects[gid]
                new_gid = new_geom_model.addGeometryObject(go)
                new_link_geom_ids.setdefault(l_name, []).append(new_gid)
                new_geom_to_link[new_gid] = l_name

        del self.obstacle_geom_ids[name]
        new_obs_ids = {}
        for obs_name, obs in self.obstacles.items():
            old_go = self.geom_model.geometryObjects[self.obstacle_geom_ids[obs_name]]
            go = pin.GeometryObject(
                old_go.name,
                old_go.parentJoint,
                old_go.parentFrame,
                self._obstacle_to_se3(obs),
                old_go.geometry,
            )
            new_gid = new_geom_model.addGeometryObject(go)
            new_obs_ids[obs_name] = new_gid

        self.geom_model = new_geom_model
        self.link_geom_ids = new_link_geom_ids
        self.geom_to_link = new_geom_to_link
        self.obstacle_geom_ids = new_obs_ids
        self.invalidate_robot_geoms_cache()
        self.geom_bounding_radii = self._compute_bounding_spheres()
        if rebuild:
            self.rebuild_pairs()
        return True

    def clear_obstacles(self, source: Optional[str] = None) -> None:
        """清空障碍物。若指定 source 则仅清空该来源 (如 'moveit' 或 'manual')。"""
        with self._obstacle_lock:
            if not self.obstacles:
                return
            if source is None:
                self.obstacles.clear()
                self.obstacle_geom_ids.clear()
                new_geom_model = pin.GeometryModel()
                new_link_geom_ids: Dict[str, List[int]] = {}
                new_geom_to_link: Dict[int, str] = {}
                for l_name, gids in self.link_geom_ids.items():
                    for gid in gids:
                        go = self.geom_model.geometryObjects[gid]
                        new_gid = new_geom_model.addGeometryObject(go)
                        new_link_geom_ids.setdefault(l_name, []).append(new_gid)
                        new_geom_to_link[new_gid] = l_name
                self.geom_model = new_geom_model
                self.link_geom_ids = new_link_geom_ids
                self.geom_to_link = new_geom_to_link
                self.invalidate_robot_geoms_cache()
                self.geom_bounding_radii = self._compute_bounding_spheres()
                self.rebuild_pairs()
            else:
                to_del = [
                    k for k, v in self.obstacles.items()
                    if getattr(v, "source", "manual") == source
                ]
                for name in to_del:
                    self._remove_obstacle_impl(name, rebuild=False)
                if to_del:
                    self.rebuild_pairs()

    @staticmethod
    def _obstacle_bounding_radius(obs: PrimitiveObstacle) -> float:
        """障碍物外接球半径 (局部系, 含 ||aabb_center|| 偏移), 供 broad-phase 粗筛表使用。"""
        geo = obs.to_coal_geometry()
        return compute_geometry_bounding_radius(
            geo, fallback_radius=float(np.linalg.norm(obs.size) * 0.5)
        )

    def _register_prebuilt(
        self,
        name: str,
        geometry: Any,
        pos: Any,
        R: Any,
        frame: str = "world",
        source: str = "manual",
    ) -> None:
        """注册已构建的 coal 几何为障碍物 (mesh 内联三角形路径)。"""
        with self._obstacle_lock:
            return self._register_prebuilt_impl(
                name, geometry, pos, R, frame=frame, rebuild=True, source=source
            )

    def _register_prebuilt_impl(
        self,
        name: str,
        geometry: Any,
        pos: Any,
        R: Any,
        frame: str = "world",
        rebuild: bool = True,
        source: str = "manual",
    ) -> None:
        if name in self.obstacles:
            self._remove_obstacle_impl(name, rebuild=False)

        obs = PrimitiveObstacle(
            name=name,
            size=[],
            position=pos,
            R=R,
            frame=frame,
            shape="mesh",
            source=source,
        )
        # 严格按 aabb_radius + ||aabb_center|| 计算外接球半径, 修复偏心 Mesh 漏检
        radius = compute_geometry_bounding_radius(geometry, fallback_radius=0.05)

        se3_placement = self._obstacle_to_se3(obs)
        go = pin.GeometryObject(name, 0, 0, se3_placement, geometry)
        obs_gid = self.geom_model.addGeometryObject(go)
        self.obstacle_geom_ids[name] = obs_gid
        self.obstacles[name] = obs
        self.geom_bounding_radii = np.concatenate(
            [self.geom_bounding_radii, [radius]]
        )
        if rebuild:
            self.rebuild_pairs()

    def sync_scene_objects(self, objs: List[Dict[str, Any]]) -> None:
        """场景物体列表整体对账进 manager (注册/更新/移除, 幂等且单次 rebuild)。"""
        with self._obstacle_lock:
            self._sync_scene_objects_impl(objs)

    def _sync_scene_objects_impl(self, objs: List[Dict[str, Any]]) -> None:
        """批量对账实现: 仅剔除 moveit 来源的过期障碍物, 原地更新位姿, 单次批量重构几何模型。"""
        scene_ids = {o["id"] for o in objs}

        # 1. 来源隔离: 仅移除 source == 'moveit' 且不在新场景中的障碍物, 保留用户 manual 障碍物
        to_remove = {
            name
            for name, obs in self.obstacles.items()
            if getattr(obs, "source", "manual") == "moveit" and name not in scene_ids
        }

        # 2. 分类处理: 纯位姿更新 vs 几何结构变更/新增
        in_place_updates: List[Tuple[str, List[float], np.ndarray, str]] = []
        to_add_or_recreate: List[Dict[str, Any]] = []

        for o in objs:
            otype = o.get("type")
            if otype not in ("box", "sphere", "cylinder", "cone", "mesh"):
                continue
            pos = [float(v) for v in o["pos"]]
            quat = o.get("quat")
            R = PrimitiveObstacle._quat_to_matrix(quat) if quat else np.eye(3)
            if R is None:
                R = np.eye(3)
            size = [float(v) for v in o.get("size", [])]
            frame = str(o.get("frame") or "world")
            mesh_path = o.get("mesh_path")
            prebuilt = o.get("coal_geometry")

            if otype == "mesh" and prebuilt is None and not mesh_path:
                logger.warning("[SceneSync] mesh 物体 '%s' 无可用几何, 跳过", o["id"])
                continue

            oid = o["id"]
            if oid in self.obstacles:
                existing = self.obstacles[oid]
                size_arr = np.asarray(size, dtype=float) if size else np.empty(0)
                # 同型同尺寸且非预建 mesh: 原地更新位姿与标记来源, 无需重构 GeometryModel
                if (
                    existing.shape == otype
                    and np.array_equal(existing.size, size_arr)
                    and prebuilt is None
                ):
                    existing.source = "moveit"
                    in_place_updates.append((oid, pos, R, frame))
                    continue

            to_add_or_recreate.append(
                {
                    "id": oid,
                    "type": otype,
                    "pos": pos,
                    "R": R,
                    "size": size,
                    "frame": frame,
                    "mesh_path": mesh_path,
                    "coal_geometry": prebuilt,
                }
            )

        # 3. 执行原地位姿更新 (毫秒级, 不重建模型与碰撞对)
        for oid, pos, R, frame in in_place_updates:
            self._update_obstacle_impl(
                oid, position=pos, R=R, frame=frame, rebuild=False
            )

        # 4. 若无过期障碍物且无结构变更/新增, 直接完成
        if not to_remove and not to_add_or_recreate:
            return

        # 5. 单次批量重建 GeometryModel: 消除循环重建的 O(K*N) 锁竞争与段错误风险
        for name in to_remove:
            self.obstacles.pop(name, None)
            self.obstacle_geom_ids.pop(name, None)

        for po in to_add_or_recreate:
            self.obstacles.pop(po["id"], None)
            self.obstacle_geom_ids.pop(po["id"], None)

        new_geom_model = pin.GeometryModel()
        new_link_geom_ids: Dict[str, List[int]] = {}
        new_geom_to_link: Dict[int, str] = {}

        # 复制机器人已有连杆几何
        for l_name, gids in self.link_geom_ids.items():
            for gid in gids:
                go = self.geom_model.geometryObjects[gid]
                new_gid = new_geom_model.addGeometryObject(go)
                new_link_geom_ids.setdefault(l_name, []).append(new_gid)
                new_geom_to_link[new_gid] = l_name

        # 复制存活障碍物 (包含本地 manual 障碍物与未改变结构的 moveit 障碍物)
        new_obs_ids: Dict[str, int] = {}
        for obs_name, obs in list(self.obstacles.items()):
            old_gid = self.obstacle_geom_ids.get(obs_name)
            if old_gid is not None and old_gid < len(self.geom_model.geometryObjects):
                old_go = self.geom_model.geometryObjects[old_gid]
                go = pin.GeometryObject(
                    old_go.name,
                    old_go.parentJoint,
                    old_go.parentFrame,
                    self._obstacle_to_se3(obs),
                    old_go.geometry,
                )
                new_gid = new_geom_model.addGeometryObject(go)
                new_obs_ids[obs_name] = new_gid

        # 挂载待新增/重建的 moveit 障碍物
        for po in to_add_or_recreate:
            oid = po["id"]
            otype = po["type"]
            prebuilt = po.get("coal_geometry")
            if prebuilt is not None:
                obs = PrimitiveObstacle(
                    name=oid,
                    size=[],
                    position=po["pos"],
                    R=po["R"],
                    frame=po["frame"],
                    shape="mesh",
                    source="moveit",
                )
                coal_geo = prebuilt
            else:
                obs = PrimitiveObstacle(
                    name=oid,
                    size=po["size"],
                    position=po["pos"],
                    R=po["R"],
                    frame=po["frame"],
                    shape=otype,
                    mesh_path=po.get("mesh_path"),
                    source="moveit",
                )
                coal_geo = obs.to_coal_geometry()

            se3_placement = self._obstacle_to_se3(obs)
            go = pin.GeometryObject(oid, 0, 0, se3_placement, coal_geo)
            new_gid = new_geom_model.addGeometryObject(go)
            self.obstacles[oid] = obs
            new_obs_ids[oid] = new_gid

        # 原子切换模型映射并单次更新碰撞对
        self.geom_model = new_geom_model
        self.link_geom_ids = new_link_geom_ids
        self.geom_to_link = new_geom_to_link
        self.obstacle_geom_ids = new_obs_ids
        self.invalidate_robot_geoms_cache()
        self.geom_bounding_radii = self._compute_bounding_spheres()
        self.rebuild_pairs()


class MoveItSceneSync:
    """MoveIt 规划场景 -> RobotCollisionManager 的自动障碍物同步器 (可选挂载)。"""

    _RESYNC_MERGE_S = 0.2         # 门铃合并窗口 (吸收 ADD+回声多条消息)
    _FETCH_TIMEOUT_S = 5.0        # 单次服务重拉超时
    _RESYNC_SERVICE_WAIT_S = 3.0  # 常驻期服务重拉 wait_for_service 超时

    def __init__(
        self,
        manager: Any,
        scene_service: str = "/get_planning_scene",
        monitored_topic: str = "/monitored_planning_scene",
        probe_timeout_s: float = 3.0,
        verbose: bool = True,
    ):
        self.manager = manager
        self.scene_service = scene_service
        self.monitored_topic = monitored_topic
        self.verbose = bool(verbose)

        self._context = None
        self._node = None
        self._executor = None
        self._scene_cli = None
        self._scene_sub = None
        self._spin_thread = None
        self._resync_pending = threading.Event()
        self._pending_lock = threading.Lock()
        self._stop = threading.Event()
        self._probe_timeout_s = float(probe_timeout_s)
        self._scene_types = None
        self._srv_type = None

        try:
            import rclpy  # noqa: F401
            from moveit_msgs.msg import PlanningScene  # noqa: F401
            from moveit_msgs.srv import GetPlanningScene  # noqa: F401
        except ImportError:
            if self.verbose:
                logger.info(
                    "[SceneSync] rclpy / moveit_msgs 未安装: 不启用 MoveIt "
                    "场景感知, 环境避障退化为仅手动 add_obstacle"
                )
            return

        self._spin_thread = threading.Thread(
            target=self._spin_loop, daemon=True, name="ha-arm-ik-scene-sync"
        )
        self._spin_thread.start()

    def _probe_moveit(self, rclpy: Any, GetPlanningScene: Any) -> bool:
        """探测 move_group 是否在线 (后台线程, 一次性 context 节点)。"""
        ctx = None
        node = None
        try:
            from rclpy.context import Context

            ctx = Context()
            rclpy.init(context=ctx)
            node = rclpy.create_node("ha_arm_ik_scene_sync_probe", context=ctx)
            cli = node.create_client(GetPlanningScene, self.scene_service)
            if not cli.wait_for_service(timeout_sec=self._probe_timeout_s):
                return False
            return True
        except Exception as e:
            self._probe_error = str(e)
            return False
        finally:
            try:
                if node is not None:
                    node.destroy_node()
                if ctx is not None and ctx.ok():
                    rclpy.shutdown(context=ctx)
            except Exception:
                pass

    def _on_scene_msg(self, msg: Any) -> None:
        """场景变化门铃: 只置事件, 解析与重拉全在 spin 线程做 (回调零阻塞)。"""
        with self._pending_lock:
            self._resync_pending.set()

    def _spin_loop(self) -> None:
        """后台线程主体: 探测 -> 常驻 (门铃 -> 合并窗口 -> 服务重拉 -> 对账)。"""
        import rclpy
        from moveit_msgs.msg import PlanningScene
        from moveit_msgs.srv import GetPlanningScene
        from rclpy.qos import DurabilityPolicy, QoSProfile

        self._scene_types = (PlanningScene,)
        self._srv_type = GetPlanningScene

        self._probe_error = None
        if not self._probe_moveit(rclpy, GetPlanningScene):
            if self.verbose:
                reason = f" ({self._probe_error})" if self._probe_error else ""
                logger.info(
                    "[SceneSync] 未探测到 move_group (%s 超时 %ss%s): 不启用 MoveIt "
                    "场景感知, 环境避障退化为仅手动 add_obstacle",
                    self.scene_service,
                    self._probe_timeout_s,
                    reason,
                )
            return

        try:
            from rclpy.context import Context

            self._context = Context()
            rclpy.init(context=self._context)
            self._node = rclpy.create_node(
                "ha_arm_ik_scene_sync", context=self._context
            )
            self._executor = rclpy.executors.SingleThreadedExecutor(
                context=self._context
            )
            self._executor.add_node(self._node)
            self._scene_cli = self._node.create_client(
                GetPlanningScene, self.scene_service
            )
            self._scene_sub = self._node.create_subscription(
                PlanningScene,
                self.monitored_topic,
                self._on_scene_msg,
                QoSProfile(
                    depth=1,
                    durability=DurabilityPolicy.VOLATILE,
                ),
            )
        except Exception as e:
            if self.verbose:
                logger.warning("[SceneSync] 常驻初始化失败 (按无 MoveIt 处理): %s", e)
            self.close()
            return

        if self.verbose:
            logger.info(
                "[SceneSync] 已挂载: 监听 %s 门铃 -> %s 重拉 -> manager 自动增删障碍物",
                self.monitored_topic,
                self.scene_service,
            )

        with self._pending_lock:
            self._resync_pending.set()

        while not self._stop.is_set() and rclpy.ok(context=self._context):
            try:
                self._executor.spin_once(timeout_sec=0.1)
            except Exception:
                time.sleep(0.1)
                continue

            if self._resync_pending.is_set():
                with self._pending_lock:
                    self._resync_pending.clear()
                time.sleep(self._RESYNC_MERGE_S)
                try:
                    self._reconcile_scene()
                except Exception as e:
                    if self.verbose:
                        logger.warning(
                            "[SceneSync] 场景重拉失败 (保留现有障碍物): %s", e
                        )

    def _reconcile_scene(self) -> None:
        objs = self._fetch_scene_via_service()
        if objs is None:
            return
        self.manager.sync_scene_objects(objs)
        if self.verbose:
            logger.info(
                "[SceneSync] 场景同步: %d 个障碍物图元已对账进 manager", len(objs)
            )

    @staticmethod
    def _pose_msg_to_se3(pose_msg: Any) -> Tuple[np.ndarray, np.ndarray]:
        """将 geometry_msgs/Pose 转为 (pos_3, R_3x3)。"""
        pos = np.array(
            [pose_msg.position.x, pose_msg.position.y, pose_msg.position.z],
            dtype=float,
        )
        quat = (
            pose_msg.orientation.x,
            pose_msg.orientation.y,
            pose_msg.orientation.z,
            pose_msg.orientation.w,
        )
        R = PrimitiveObstacle._quat_to_matrix(quat)
        if R is None:
            R = np.eye(3, dtype=float)
        return pos, R

    @staticmethod
    def _matrix_to_quat(R: np.ndarray) -> List[float]:
        """3x3 旋转矩阵转 (qx, qy, qz, qw) 四元数。"""
        q = pin.Quaternion(R)
        return [float(q.x), float(q.y), float(q.z), float(q.w)]

    @classmethod
    def _compose_obj_and_sub_pose(
        cls, co_pose: Any, sub_pose: Optional[Any]
    ) -> Tuple[List[float], List[float]]:
        """精确合成物体级位姿 T_co 与子图元相对位姿 T_sub: T_world = T_co * T_sub。"""
        p_co, R_co = cls._pose_msg_to_se3(co_pose)
        if sub_pose is None:
            return p_co.tolist(), cls._matrix_to_quat(R_co)

        p_sub, R_sub = cls._pose_msg_to_se3(sub_pose)
        p_world = p_co + R_co @ p_sub
        R_world = R_co @ R_sub
        return p_world.tolist(), cls._matrix_to_quat(R_world)

    def _fetch_scene_via_service(self) -> Optional[List[Dict[str, Any]]]:
        """经 /get_planning_scene 拉取当前世界物体, 支持复合多图元物体并合成 SE(3) 位姿。"""
        if self._scene_types is None or self._node is None:
            return None
        from moveit_msgs.msg import PlanningSceneComponents

        try:
            if not self._scene_cli.wait_for_service(
                timeout_sec=self._RESYNC_SERVICE_WAIT_S
            ):
                return None
            req = self._srv_type.Request()
            req.components.components = (
                PlanningSceneComponents.WORLD_OBJECT_NAMES
                | PlanningSceneComponents.WORLD_OBJECT_GEOMETRY
            )
            fut = self._scene_cli.call_async(req)

            deadline = time.time() + self._FETCH_TIMEOUT_S
            while not fut.done() and time.time() < deadline:
                self._executor.spin_once(timeout_sec=0.05)
            if not fut.done() or fut.result() is None:
                return None

            world = fut.result().scene.world
            out: List[Dict[str, Any]] = []
            for co in world.collision_objects:
                op = co.operation
                if isinstance(op, (bytes, bytearray)):
                    op = op[0] if op else 0
                if op != 0:
                    continue

                frame_id = co.header.frame_id or "world"
                total_parts = len(co.primitives) + len(co.meshes)

                # 1. 遍历全部解析图元 (支持多图元复合障碍物, 如多部件桌架)
                for idx, prim in enumerate(co.primitives):
                    sub_p = (
                        co.primitive_poses[idx]
                        if idx < len(co.primitive_poses)
                        else None
                    )
                    pos, quat = self._compose_obj_and_sub_pose(co.pose, sub_p)
                    part_id = co.id if total_parts == 1 else f"{co.id}_prim_{idx}"

                    entry: Dict[str, Any] = {
                        "id": part_id,
                        "frame": frame_id,
                        "pos": pos,
                        "quat": quat,
                    }
                    if prim.type == prim.BOX:
                        entry["type"] = "box"
                        entry["size"] = list(prim.dimensions)
                    elif prim.type == prim.SPHERE:
                        entry["type"] = "sphere"
                        entry["size"] = [prim.dimensions[0] * 2.0]
                    elif prim.type == prim.CYLINDER:
                        entry["type"] = "cylinder"
                        entry["size"] = [prim.dimensions[1] * 2.0, prim.dimensions[0]]
                    elif prim.type == prim.CONE:
                        entry["type"] = "cone"
                        entry["size"] = [prim.dimensions[1] * 2.0, prim.dimensions[0]]
                    else:
                        continue
                    out.append(entry)

                # 2. 遍历全部三角网格图元
                for idx, mesh_msg in enumerate(co.meshes):
                    mesh_geo = self._mesh_msg_to_coal(mesh_msg)
                    if mesh_geo is None:
                        logger.warning(
                            "[SceneSync] mesh 物体 '%s' (子网格 %d) 无内联三角形, 跳过",
                            co.id,
                            idx,
                        )
                        continue
                    sub_p = (
                        co.mesh_poses[idx] if idx < len(co.mesh_poses) else None
                    )
                    pos, quat = self._compose_obj_and_sub_pose(co.pose, sub_p)
                    part_id = co.id if total_parts == 1 else f"{co.id}_mesh_{idx}"

                    out.append(
                        {
                            "id": part_id,
                            "frame": frame_id,
                            "type": "mesh",
                            "coal_geometry": mesh_geo,
                            "size": [],
                            "pos": pos,
                            "quat": quat,
                        }
                    )

            return out
        except Exception:
            return None

    @staticmethod
    def _mesh_msg_to_coal(mesh_msg: Any):
        """shape_msgs/Mesh 内联三角形 -> coal BVHModelOBBRSS (无三角形返回 None)。"""
        if not mesh_msg.triangles or not mesh_msg.vertices:
            return None
        try:
            verts = np.array(
                [(float(v.x), float(v.y), float(v.z)) for v in mesh_msg.vertices],
                dtype=float,
            )
            tris = np.array(
                [
                    (
                        int(t.vertex_indices[0]),
                        int(t.vertex_indices[1]),
                        int(t.vertex_indices[2]),
                    )
                    for t in mesh_msg.triangles
                ],
                dtype=np.int64,
            )
            bvh = coal.BVHModelOBBRSS()
            bvh.beginModel(len(tris), len(verts))
            for v in verts:
                bvh.addVertex(v)
            bvh.addTriangles(tris)
            bvh.endModel()
            bvh.computeLocalAABB()
            return bvh
        except Exception:
            return None

    @property
    def active(self) -> bool:
        """是否处于常驻同步状态 (探测到 move_group 且线程在跑)。"""
        return (
            self._node is not None
            and self._spin_thread is not None
            and self._spin_thread.is_alive()
        )

    def close(self) -> None:
        """停后台线程并彻底销毁 ROS 资源 (幂等, 可安全多次调用)。"""
        self._stop.set()
        if (
            self._spin_thread is not None
            and self._spin_thread is not threading.current_thread()
            and self._spin_thread.is_alive()
        ):
            self._spin_thread.join(timeout=2.0)
        self._spin_thread = None

        if self._executor is not None:
            try:
                if self._node is not None:
                    self._executor.remove_node(self._node)
                self._executor.shutdown()
            except Exception:
                pass
            self._executor = None

        if self._node is not None:
            try:
                self._node.destroy_node()
            except Exception:
                pass
            self._node = None

        if self._context is not None:
            try:
                if self._context.ok():
                    import rclpy

                    rclpy.shutdown(context=self._context)
            except Exception:
                pass
            self._context = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass