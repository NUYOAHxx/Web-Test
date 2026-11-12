#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""环境障碍物管理: 解析图元障碍物生命周期 + MoveIt 规划场景自动感知。

* PrimitiveObstacle: box / sphere / cylinder / cone / mesh 解析障碍物数据结构
* ObstacleManager: 障碍物生命周期 (add/update/remove/场景对账)
* MoveItSceneSync: 后台线程感知 /monitored_planning_scene 门铃 -> /get_planning_scene 重拉 -> manager.sync_scene_objects 幂等对账
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import coal
import numpy as np
import pinocchio as pin


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

    def to_se3(self, T_world_from_base: Optional[np.ndarray] = None) -> pin.SE3:
        """获取在 Pinocchio 模型根系（URDF 根）下的 SE3 位姿。

        障碍物几何挂在 joint 0，placement 由 Pinocchio 按模型根（URDF 根）
        解释，与机器人碰撞几何同系。"world" 即 URDF 根系（MoveIt frame_id 同口径，
        solver 的 target_to_local 也是此约定），原样放置；"base" 为求解器基座系
        （7 轴时是 Body_Link），需乘安装位姿 T_world_from_base 换到模型根。
        """
        T_raw = np.eye(4, dtype=float)
        T_raw[:3, :3] = self.R
        T_raw[:3, 3] = self.position

        if self.frame.lower() == "base" and T_world_from_base is not None:
            T_root = T_world_from_base @ T_raw
            return pin.SE3(T_root[:3, :3], T_root[:3, 3])
        return pin.SE3(self.R, self.position)


class ObstacleManager:
    """解析障碍物生命周期管理 (add/update/remove/场景对账)。

    _obstacle_lock: 生命周期操作互斥锁 (类级共享: add/update/remove 重建
    geom_model/半径表/对表非线程安全, 并发写方必须串行)。类级锁跨实例
    共享无害——锁只影响障碍物变更时延, 不影响检测路径。
    """

    _obstacle_lock = threading.Lock()

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
    ) -> PrimitiveObstacle:
        """添加障碍物 (box/sphere/cylinder/cone/mesh, 语义见 PrimitiveObstacle)。

        shape="mesh" 时 size 未用, mesh_path 必须给出且文件存在, 否则
        抛 FileNotFoundError (调用方捕获降级)。
        """
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
    ) -> PrimitiveObstacle:
        if name in self.obstacles:
            self._remove_obstacle_impl(name)

        obs = PrimitiveObstacle(
            name=name,
            size=size,
            position=position,
            rpy=rpy,
            R=R,
            frame=frame,
            shape=shape,
            mesh_path=mesh_path,
        )
        self.obstacles[name] = obs

        se3_placement = obs.to_se3(self.T_world_from_base)
        go = pin.GeometryObject(name, 0, 0, se3_placement, obs.to_coal_geometry())
        obs_gid = self.geom_model.addGeometryObject(go)
        self.obstacle_geom_ids[name] = obs_gid

        # 障碍物外接球半径同步入 broad-phase 表 (统一 FCL computeLocalAABB 口径)
        self.geom_bounding_radii = np.concatenate(
            [
                self.geom_bounding_radii,
                [self._obstacle_bounding_radius(obs)],
            ]
        )
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
        """动态更新障碍物的位置、朝向或形状 (size 语义随该障碍物的 shape)。

        朝向三选一传参: rpy 与 R 同时给出时以 R 为准, rpy 同步为 R 的反推欧拉角。
        """
        with self._obstacle_lock:
            return self._update_obstacle_impl(
                name, position=position, size=size, rpy=rpy, R=R, frame=frame
            )

    def _update_obstacle_impl(
        self,
        name: str,
        position: Optional[Union[List[float], Tuple[float, ...], np.ndarray]] = None,
        size: Optional[Union[List[float], Tuple[float, ...], np.ndarray]] = None,
        rpy: Optional[Union[List[float], Tuple[float, ...], np.ndarray]] = None,
        R: Optional[np.ndarray] = None,
        frame: Optional[str] = None,
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
            # 尺寸变化改变 FCL 几何本身, 走 add (内部先 remove) 重建
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
            )
        else:
            obs_gid = self.obstacle_geom_ids[name]
            self.geom_model.geometryObjects[obs_gid].placement = obs.to_se3(
                self.T_world_from_base
            )

        return True

    def remove_obstacle(self, name: str) -> bool:
        """移除指定的外部障碍物。"""
        with self._obstacle_lock:
            return self._remove_obstacle_impl(name)

    def _remove_obstacle_impl(self, name: str) -> bool:
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
            # 复用旧 geom_model 里的 GeometryObject (几何指针原样保留):
            # 重建时重调 to_coal_geometry 对 prebuilt mesh 型会炸 (无
            # mesh_path 可加载), 且重复构建解析几何纯属浪费。
            old_go = self.geom_model.geometryObjects[self.obstacle_geom_ids[obs_name]]
            go = pin.GeometryObject(
                old_go.name,
                old_go.parentJoint,
                old_go.parentFrame,
                obs.to_se3(self.T_world_from_base),
                old_go.geometry,
            )
            new_gid = new_geom_model.addGeometryObject(go)
            new_obs_ids[obs_name] = new_gid

        self.geom_model = new_geom_model
        self.link_geom_ids = new_link_geom_ids
        self.geom_to_link = new_geom_to_link
        self.obstacle_geom_ids = new_obs_ids
        # gid 前移, 半径表必须同步重建否则后续障碍的半径索引错位
        self.geom_bounding_radii = self._compute_bounding_spheres()
        self.rebuild_pairs()
        return True

    @staticmethod
    def _obstacle_bounding_radius(obs: PrimitiveObstacle) -> float:
        """障碍物外接球半径 (局部系), 供 broad-phase 粗筛表使用。"""
        geo = obs.to_coal_geometry()
        try:
            if geo.aabb_radius < 0.0:
                geo.computeLocalAABB()
            return max(float(geo.aabb_radius), 0.0)
        except Exception:
            # 兜底: 按"最远顶点距中心"保守估计
            return float(np.linalg.norm(obs.size) * 0.5)

    def _register_prebuilt(self, name: str, geometry, pos, R) -> None:
        """注册已构建的 coal 几何为障碍物 (mesh 内联三角形路径)。

        与 add_obstacle 同口径: Placement 挂 joint 0、半径入 broad-phase
        表、rebuild_pairs 重组对表; PrimitiveObstacle 记录 shape="mesh"
        仅作生命周期登记 (to_coal_geometry 不会被再调)。
        """
        with self._obstacle_lock:
            return self._register_prebuilt_impl(name, geometry, pos, R)

    def _register_prebuilt_impl(self, name: str, geometry, pos, R) -> None:
        obs = PrimitiveObstacle(
            name=name,
            size=[],
            position=pos,
            R=R,
            frame="world",
            shape="mesh",
        )
        # mesh 兜底半径: aabb_radius 已在 _mesh_msg_to_coal 算好
        radius = 0.0
        try:
            radius = max(float(geometry.aabb_radius), 0.0)
        except Exception:
            pass

        se3_placement = obs.to_se3(self.T_world_from_base)
        go = pin.GeometryObject(name, 0, 0, se3_placement, geometry)
        obs_gid = self.geom_model.addGeometryObject(go)
        self.obstacle_geom_ids[name] = obs_gid
        self.obstacles[name] = obs
        self.geom_bounding_radii = np.concatenate(
            [
                self.geom_bounding_radii,
                [radius],
            ]
        )
        self.rebuild_pairs()

    def sync_scene_objects(self, objs: List[Dict[str, Any]]) -> None:
        """场景物体列表整体对账进 manager (注册/更新/移除, 幂等)。

        MoveIt frame_id 即 URDF 根, 直接 frame="world"; quat 恒转换显式
        传入 (单位阵也传), 保证对账后障碍物姿态与场景严格一致, 而非
        残留旧值; scene 里消失的 id 一并移除。全程持 _obstacle_lock
        (对账的 remove+add 序列必须原子, 否则并发方看到中间态)。
        """
        with self._obstacle_lock:
            self._sync_scene_objects_impl(objs)

    def _sync_scene_objects_impl(self, objs: List[Dict[str, Any]]) -> None:
        """批量对账实现: 零散增删合并为单次 rebuild, 比逐个 remove/add 省多次 geom_model 重建
        (多障碍场景对账开销从 O(n) 次重建降到 1 次)。
        """
        scene_ids = {o["id"] for o in objs}
        for name in list(self.obstacles.keys()):
            if name not in scene_ids:
                self._remove_obstacle_impl(name)

        for o in objs:
            if o.get("type") not in ("box", "sphere", "cylinder", "cone", "mesh"):
                continue
            pos = [float(v) for v in o["pos"]]
            quat = o.get("quat")
            R = PrimitiveObstacle._quat_to_matrix(quat) if quat else np.eye(3)
            if R is None:
                R = np.eye(3)
            size = [float(v) for v in o["size"]]
            mesh_path = o.get("mesh_path")
            prebuilt = o.get("coal_geometry")

            if o["type"] == "mesh" and prebuilt is None and not mesh_path:
                # mesh 物体既无内联几何也无文件路径: 无法构建碰撞几何, 跳过
                # (fetch 层已滤过一次, 这里防调用方直传未滤列表)
                print(f"[SceneSync] mesh 物体 '{o['id']}' 无可用几何, 跳过")
                continue

            if o["id"] in self.obstacles:
                existing = self.obstacles[o["id"]]
                size_arr = np.asarray(size, dtype=float) if size else np.empty(0)
                # 同型同尺寸 (mesh 恒走原地): 仅位姿变, 原地更新 placement
                # 免 remove+add 重建整表 (高频门铃场景的主要开销来源)
                if (
                    existing.shape == o["type"]
                    and np.array_equal(existing.size, size_arr)
                    and prebuilt is None
                ):
                    self._update_obstacle_impl(o["id"], position=pos, R=R)
                    continue
                self._remove_obstacle_impl(o["id"])

            if prebuilt is not None:
                # mesh 物体: 已还原的 coal 几何直接注册 (绕过 to_coal_geometry)
                self._register_prebuilt_impl(o["id"], prebuilt, pos, R)
                continue

            self._add_obstacle_impl(
                o["id"],
                size=size,
                position=pos,
                R=R,
                frame="world",
                shape=o["type"],
                mesh_path=mesh_path,
            )


class MoveItSceneSync:
    """MoveIt 规划场景 -> RobotCollisionManager 的自动障碍物同步器 (可选挂载)。

    设计语义: "有 MoveIt 才感知, 没有 MoveIt 就不做环境避障"——构造后
    后台线程经 /get_planning_scene 探测 move_group, 失败则全程零 ROS 资源
    (manager 保持手动 add_obstacle)。探测成功后订阅
    /monitored_planning_scene 当"变化门铃", 每次门铃经服务重拉全量世界
    物体整体对账进 manager (单一事实来源, 幂等); move_group 中途消失时
    保留现有障碍物等下次门铃。ROS 依赖全部延迟 import, 纯库环境可加载。
    独立 rclpy Context 避免与调用方已有 rclpy.init 冲突。
    """

    _RESYNC_MERGE_S = 0.2         # 门铃合并窗口 (吸收 ADD+回声多条消息)
    _FETCH_TIMEOUT_S = 5.0        # 单次服务重拉超时
    _RESYNC_SERVICE_WAIT_S = 3.0  # 常驻期服务重拉 wait_for_service 超时

    def __init__(
        self,
        manager,
        scene_service: str = "/get_planning_scene",
        monitored_topic: str = "/monitored_planning_scene",
        probe_timeout_s: float = 3.0,
        verbose: bool = True,
    ):
        self.manager = manager
        self.scene_service = scene_service
        self.monitored_topic = monitored_topic
        self.verbose = bool(verbose)

        self._context = None  # 独立 rclpy Context (探测成功才创建)
        self._node = None
        self._executor = None
        self._scene_cli = None
        self._scene_sub = None
        self._spin_thread = None
        self._resync_pending = threading.Event()
        self._pending_lock = threading.Lock()
        self._stop = threading.Event()
        self._probe_timeout_s = float(probe_timeout_s)
        self._scene_types = None  # (PlanningScene,) 导入成功才有值
        self._srv_type = None

        # 零依赖快检: rclpy / moveit_msgs 未安装 (纯库环境) 直接不启动
        try:
            import rclpy  # noqa: F401
            from moveit_msgs.msg import PlanningScene  # noqa: F401
            from moveit_msgs.srv import GetPlanningScene  # noqa: F401
        except ImportError:
            if self.verbose:
                print(
                    "[SceneSync] rclpy / moveit_msgs 未安装: 不启用 MoveIt "
                    "场景感知, 环境避障退化为仅手动 add_obstacle"
                )
            return

        # 探测在后台线程做: 求解器构造路径绝不等服务探测 (3s 超时量级);
        # 探测失败由线程自行拆干净资源, 本实例退化为 inactive (no-op)
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
            # 服务在 = move_group 在线; 首帧快照由常驻线程的初始 resync 拉取
            return True
        except Exception as e:
            # 探测失败原因由 _spin_loop 统一打印, 这里只标记
            self._probe_error = str(e)
            return False
        finally:
            # 回收一次性探测节点 (常驻期自建全新 context)
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
        """后台线程主体: 探测 -> 常驻 (门铃 -> 合并窗口 -> 服务重拉 -> 对账)。

        探测失败: 线程直接退出, 不创建常驻资源。move_group 中途消失:
        保留现有障碍物等下次门铃, 不做定时轮询空转。
        """
        import rclpy
        from moveit_msgs.msg import PlanningScene
        from moveit_msgs.srv import GetPlanningScene
        from rclpy.qos import DurabilityPolicy, QoSProfile

        self._scene_types = (PlanningScene,)
        self._srv_type = GetPlanningScene

        # 探测期: move_group 不在线就全程零常驻资源
        self._probe_error = None
        if not self._probe_moveit(rclpy, GetPlanningScene):
            if self.verbose:
                reason = f" ({self._probe_error})" if self._probe_error else ""
                print(
                    f"[SceneSync] 未探测到 move_group ({self.scene_service} "
                    f"超时 {self._probe_timeout_s}s{reason}): 不启用 MoveIt "
                    "场景感知, 环境避障退化为仅手动 add_obstacle"
                )
            return

        # 常驻期: 独立 context + 节点 + 门铃订阅
        try:
            from rclpy.context import Context

            self._context = Context()
            rclpy.init(context=self._context)
            self._node = rclpy.create_node(
                "ha_arm_ik_scene_sync", context=self._context
            )
            # 独立 context 的 spin 必须走绑定该 context 的 executor
            # (rclpy.spin_once 只作用于全局默认 context)
            self._executor = rclpy.executors.SingleThreadedExecutor(
                context=self._context
            )
            self._executor.add_node(self._node)
            self._scene_cli = self._node.create_client(
                GetPlanningScene, self.scene_service
            )
            # 门铃 QoS: 发布端 durability=VOLATILE, 订阅侧必须一致 (请求值
            # 高于提供值会静默收不到)。本订阅只当"可能变了"的信号。
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
                print(f"[SceneSync] 常驻初始化失败 (按无 MoveIt 处理): {e}")
            self.close()
            return

        if self.verbose:
            print(
                f"[SceneSync] 已挂载: 监听 {self.monitored_topic} 门铃 -> "
                f"{self.scene_service} 重拉 -> manager 自动增删障碍物"
            )

        # 首帧快照: 门铃是 VOLATILE, 本节点收不到启动前的历史快照,
        # 常驻启动后先主动拉一次 (与后续门铃路径共用 _reconcile_scene)
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
                # 合并窗口: 吸收同一事件的 ADD+回声等多条门铃消息
                time.sleep(self._RESYNC_MERGE_S)
                try:
                    self._reconcile_scene()
                except Exception as e:
                    if self.verbose:
                        print(f"[SceneSync] 场景重拉失败 (保留现有障碍物): {e}")

    def _reconcile_scene(self) -> None:
        objs = self._fetch_scene_via_service()
        if objs is None:
            return  # 服务不可用: 保持现状, 等下次门铃重试
        self.manager.sync_scene_objects(objs)
        if self.verbose:
            print(f"[SceneSync] 场景同步: {len(objs)} 个障碍物已对账进 manager")

    def _fetch_scene_via_service(self) -> Optional[List[Dict[str, Any]]]:
        """经 /get_planning_scene 拉取当前世界物体, 返回归一化条目列表或 None。

        条目: {id, type(box/sphere/cylinder/cone/mesh), size(全尺寸),
        pos, quat[, mesh_path]}。sphere 直径 x2 归一为全尺寸; 物体级
        co.pose 优先, 退化用 primitive_poses[0]/mesh_poses[0]; mesh 物体
        经 package:// 解析出绝对路径 (解析不到的跳过并打印)。
        """
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

            # future 只能靠本线程 spin 完成——边等边 spin executor,
            # 纯 sleep 会永远不 done
            deadline = time.time() + self._FETCH_TIMEOUT_S
            while not fut.done() and time.time() < deadline:
                self._executor.spin_once(timeout_sec=0.05)
            if not fut.done() or fut.result() is None:
                return None

            world = fut.result().scene.world
            out: List[Dict[str, Any]] = []
            for co in world.collision_objects:
                # operation 归一化: rclpy 反序列化怪癖, 可能给 bytes
                op = co.operation
                if isinstance(op, (bytes, bytearray)):
                    op = op[0] if op else 0

                if op != 0:
                    continue

                def _obj_pose(co=co, pose_list=None):
                    if pose_list is not None:
                        return pose_list
                    cp = co.pose
                    if (
                        abs(cp.position.x)
                        + abs(cp.position.y)
                        + abs(cp.position.z)
                        > 1e-9
                    ):
                        return cp
                    if co.primitive_poses:
                        return co.primitive_poses[0]
                    if co.mesh_poses:
                        return co.mesh_poses[0]
                    return cp

                entry: Dict[str, Any] = {
                    "id": co.id,
                    "frame": co.header.frame_id or "base_Base",
                }

                if co.primitives:
                    prim = co.primitives[0]
                    # 位姿优先级与 moveit_bridge 同口径: move_group 规范化后
                    # 物体级位姿在 co.pose, primitive_poses 是相对位姿
                    # (实测物体级 pose=(0.5,0,0) 而 primitive_poses[0]=(0,0,0));
                    # 必须走 _obj_pose 的智能回退, 不可直取 primitive_poses[0]
                    p = _obj_pose(co)
                    if prim.type == prim.BOX:
                        entry["type"] = "box"
                        entry["size"] = list(prim.dimensions)
                    elif prim.type == prim.SPHERE:
                        entry["type"] = "sphere"
                        entry["size"] = [prim.dimensions[0] * 2.0]
                    elif prim.type == prim.CYLINDER:
                        # shape_msgs cylinder/cone dimensions=[height, radius]
                        # (CONE_HEIGHT=0/CONE_RADIUS=1), 翻转为 manager 的
                        # [直径, 全高] 全尺寸语义
                        entry["type"] = "cylinder"
                        entry["size"] = [prim.dimensions[1] * 2.0, prim.dimensions[0]]
                    elif prim.type == prim.CONE:
                        entry["type"] = "cone"
                        entry["size"] = [prim.dimensions[1] * 2.0, prim.dimensions[0]]
                    else:
                        continue
                elif co.meshes:
                    # mesh 物体: shape_msgs/Mesh 内联三角形 (vertices +
                    # vertex_indices), 直接还原为 coal BVH 几何, 无需
                    # 资源文件; 无内联三角形的空壳跳过。
                    mesh_geo = self._mesh_msg_to_coal(co.meshes[0])
                    if mesh_geo is None:
                        print(
                            f"[SceneSync] mesh 物体 '{co.id}' 无内联三角形, 跳过"
                        )
                        continue
                    # mesh 物体姿态归 mesh_poses (无则回退物体级位姿)
                    p = _obj_pose(co, co.mesh_poses[0] if co.mesh_poses else None)
                    entry["type"] = "mesh"
                    entry["coal_geometry"] = mesh_geo
                    entry["size"] = []
                else:
                    continue

                entry["pos"] = [p.position.x, p.position.y, p.position.z]
                entry["quat"] = [
                    p.orientation.x,
                    p.orientation.y,
                    p.orientation.z,
                    p.orientation.w,
                ]
                out.append(entry)
            return out
        except Exception:
            return None

    @staticmethod
    def _mesh_msg_to_coal(mesh_msg: Any):
        """shape_msgs/Mesh 内联三角形 -> coal BVHModelOBB (无三角形返回 None)。

        coal 4.x 绑定: BVH 构建走 camelCase (beginModel/addSubModel/
        endModel), 与 MeshLoader 返回的几何同类, GJK 判碰口径一致。
        """
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
            # coal 4.x 绑定: addTriangles 只收 (N,3) Eigen 索引矩阵, 配合
            # addVertex 逐个入顶点 (addSubModel 的 vector<Vec3s> 重载不
            # 接受 numpy/list, 这里逐顶点最稳)。必须用 OBBRSS 节点——与
            # MeshLoader 同款; OBB 节点与机器人侧 OBBRSS 几何判碰时
            # coal 报 "node type BV_OBB and node type BV_OBBRSS is not
            # yet supported"。
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
        """停后台线程并彻底销毁 ROS 资源 (幂等, 可安全多次调用)。

        每项资源独立 try: 任一销毁失败不阻断其余资源回收。
        """
        self._stop.set()
        # 从自身 spin 线程内调用 (异常清理路径) 时不能 join 自己
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