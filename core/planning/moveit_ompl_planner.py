#!/usr/bin/env python3
"""
================================================================================
Unitree G1 MoveIt 2 OMPL (Open Motion Planning Library) 全局运动规划客户端
(MoveIt 2 OMPL Motion Planning Client for Unitree G1 Humanoid Robot)
================================================================================

定位：高层宏观全局避障路径规划客户端
1. 封装 ROS 2 MoveIt 2 OMPL (C++ 底层高性能采样规划流水线)；
2. 支持规划组：
   - left_arm (7-DoF 单臂)
   - right_arm (7-DoF 单臂)
   - left_arm_torso (10-DoF 躯干-手臂协同)
   - right_arm_torso (10-DoF 躯干-手臂协同)
   - both_arms (14-DoF 双臂协同)
3. 支持经典 OMPL 采样算法切换：
   - RRTConnect (RRTConnectkConfigDefault, 毫秒级双向扩展，推荐宏观避障)
   - RRT* (RRTstarkConfigDefault, 渐进最优路径规划)
   - PRM (PRMkConfigDefault, 概率路图算法，适合多目标查询)
4. 支持目标形式：
   - 6D 空间笛卡尔末端目标位姿 (PoseStamped / XYZ + RPY / 四元数)
   - 高维关节空间配置目标 (Joint-Space Goal)
5. 支持动态障碍物与规划场景 (PlanningScene) 管理：
   - 动态向场景添加立方体 (Box)、圆柱体 (Cylinder) 障碍物
   - OMPL 在规划时基于 28 对 SRDF ACM 与环境障碍物进行实时碰撞避免
6. 输出标准平滑轨迹对象 (JointTrajectory)，包含位置、速度、加速度与时间戳。
"""

import time
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

# ROS 2 与 MoveIt 消息包
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import Pose, Point, Quaternion
from trajectory_msgs.msg import JointTrajectory
from control_msgs.action import FollowJointTrajectory
from shape_msgs.msg import SolidPrimitive

import moveit_msgs.msg
import moveit_msgs.srv


# 规划组末端执行器 Link 映射 (与 resources/g1/g1_29dof.srdf 严格一致)
PLANNING_GROUP_EE_LINKS: Dict[str, str] = {
    "left_arm": "left_wrist_yaw_link",
    "right_arm": "right_wrist_yaw_link",
    "left_arm_torso": "left_wrist_yaw_link",
    "right_arm_torso": "right_wrist_yaw_link",
}

# 规划组基座 Link 映射
PLANNING_GROUP_BASE_LINKS: Dict[str, str] = {
    "left_arm": "torso_link",
    "right_arm": "torso_link",
    "left_arm_torso": "pelvis",
    "right_arm_torso": "pelvis",
    "both_arms": "torso_link",
}

# 默认规划算法 ID
DEFAULT_PLANNER_IDS = {
    "RRTConnect": "RRTConnectkConfigDefault",
    "RRTstar": "RRTstarkConfigDefault",
    "PRM": "PRMkConfigDefault",
}


@dataclass
class PlanResult:
    """OMPL 运动规划输出结果数据类"""
    success: bool
    planning_time: float
    joint_names: List[str] = field(default_factory=list)
    trajectory: Optional[JointTrajectory] = None
    waypoints: List[np.ndarray] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)
    error_code: int = 1
    error_message: str = ""
    ik_solution: Optional[Dict[str, float]] = None
    ik_time_ms: float = 0.0

    def summary(self) -> str:
        """格式化输出规划指标摘要"""
        if not self.success:
            return f"[OMPL 规划失败] 错误码: {self.error_code} | 原因: {self.error_message}"
        num_pts = len(self.waypoints)
        total_time = self.timestamps[-1] if self.timestamps else 0.0
        ik_str = f" | 前置 Pink IK: {self.ik_time_ms:.1f}ms" if self.ik_time_ms > 0 else ""
        return (
            f"[OMPL 规划成功] 耗时: {self.planning_time * 1000.0:.2f}ms{ik_str} | "
            f"路标点: {num_pts} 个 | 轨迹执行时长: {total_time:.2f}s | "
            f"受控关节数: {len(self.joint_names)}"
        )


class MoveItOMPLPlanner:
    """
    Unitree G1 MoveIt 2 OMPL 规划引擎高层客户端接口
    """

    def __init__(self, node: Optional[Node] = None, wait_for_services: bool = True, timeout_sec: float = 5.0):
        """
        初始化 OMPL 规划客户端
        :param node: 外部传入的 rclpy.node.Node 实例；若为 None 则自建独立后台节点
        :param wait_for_services: 是否在初始化时阻塞等待 MoveIt 规划服务上线
        :param timeout_sec: 等待服务上线的超时时间 (秒)
        """
        self._owns_node = False
        if node is None:
            if not rclpy.ok():
                rclpy.init()
            self.node = Node("g1_ompl_planning_client")
            self._owns_node = True
        else:
            self.node = node

        # 1. 运动规划服务客户端 (/plan_kinematic_path)
        self.plan_service_client = self.node.create_client(
            moveit_msgs.srv.GetMotionPlan, "/plan_kinematic_path"
        )

        # 2. 规划场景应用客户端 (/apply_planning_scene)
        self.scene_service_client = self.node.create_client(
            moveit_msgs.srv.ApplyPlanningScene, "/apply_planning_scene"
        )

        # 3. 规划场景快速广播发布者 (/planning_scene)
        self.scene_publisher = self.node.create_publisher(
            moveit_msgs.msg.PlanningScene, "/planning_scene", 10
        )

        # 场景中已添加障碍物缓存
        self._attached_obstacles: List[str] = []
        self._pink_solver = None

        if wait_for_services:
            self._wait_for_services(timeout_sec)

    def _wait_for_services(self, timeout_sec: float = 5.0):
        """检查并等待 MoveIt 后台服务上线"""
        self.node.get_logger().info("正在连接 MoveIt 2 规划服务 (/plan_kinematic_path)...")
        ready = self.plan_service_client.wait_for_service(timeout_sec=timeout_sec)
        if ready:
            self.node.get_logger().info("[OK] 成功接入 MoveIt 2 OMPL 规划服务端！")
        else:
            self.node.get_logger().warn(
                f"[WARN] 未检测到 /plan_kinematic_path 服务（等待超时 {timeout_sec}s）。"
                "请确保已启动 MoveIt move_group 节点 (ros2 launch launch/g1_moveit_ompl.launch.py)。"
            )

    # ==========================================================================
    # 核心规划接口 1: 笛卡尔末端 6D 空间目标规划
    # ==========================================================================
    def plan_to_pose(
        self,
        group_name: str,
        target_pos: Union[List[float], np.ndarray],
        target_rpy: Optional[Union[List[float], np.ndarray]] = None,
        target_quat: Optional[Union[List[float], Tuple[float, ...]]] = None,
        planner_id: str = "RRTConnectkConfigDefault",
        allowed_planning_time: float = 5.0,
        num_planning_attempts: int = 5,
        start_joints: Optional[Dict[str, float]] = None,
        pos_tolerance: float = 0.005,
        rot_tolerance: float = 0.05,
        reference_frame: str = "world",
    ) -> PlanResult:
        """
        向指定 6D 空间笛卡尔位姿规划无碰撞关节运动轨迹
        :param group_name: 规划组名称 ("left_arm", "right_arm", "left_arm_torso", "right_arm_torso")
        :param target_pos: [X, Y, Z] 目标三维坐标 (米)
        :param target_rpy: [Roll, Pitch, Yaw] 目标欧拉角 (弧度，可选)
        :param target_quat: [x, y, z, w] 目标四元数 (可选，优先级高于 target_rpy)
        :param planner_id: OMPL 算法 ID ("RRTConnectkConfigDefault", "RRTstarkConfigDefault", "PRMkConfigDefault")
        :param allowed_planning_time: 最大允许求解耗时 (秒)
        :param num_planning_attempts: 最大并行/重试求解次数
        :param start_joints: 可选指定起始关节状态字典 {jname: rad}；若为 None 则从当前机器人状态规划
        :param pos_tolerance: 容许位置误差 (米，默认 5mm)
        :param rot_tolerance: 容许姿态旋转误差 (弧度，默认 ~2.8度)
        :param reference_frame: 坐标系，默认 "world"
        """
        if group_name not in PLANNING_GROUP_EE_LINKS:
            return PlanResult(
                success=False,
                planning_time=0.0,
                error_message=f"未知的规划组 '{group_name}'，有效组: {list(PLANNING_GROUP_EE_LINKS.keys())}",
            )

        ee_link = PLANNING_GROUP_EE_LINKS[group_name]

        # 构造标准请求
        req = moveit_msgs.srv.GetMotionPlan.Request()
        mpr = req.motion_plan_request
        mpr.group_name = group_name
        mpr.planner_id = planner_id
        mpr.allowed_planning_time = float(allowed_planning_time)
        mpr.num_planning_attempts = int(num_planning_attempts)
        mpr.max_velocity_scaling_factor = 1.0
        mpr.max_acceleration_scaling_factor = 1.0

        # 设置起始状态 (若提供)
        if start_joints:
            mpr.start_state.joint_state.name = list(start_joints.keys())
            mpr.start_state.joint_state.position = [float(v) for v in start_joints.values()]
            mpr.start_state.is_diff = True

        # 构造目标约束 (Goal Constraints)
        constraints = moveit_msgs.msg.Constraints()
        constraints.name = f"{group_name}_cartesian_goal"

        # 1. 位置约束 (PositionConstraint)
        pos_con = moveit_msgs.msg.PositionConstraint()
        pos_con.header.frame_id = reference_frame
        pos_con.link_name = ee_link
        pos_con.weight = 1.0

        target_point = Point(x=float(target_pos[0]), y=float(target_pos[1]), z=float(target_pos[2]))
        # 约束包络盒 (以目标点为中心的容差球/盒)
        bbox = SolidPrimitive()
        bbox.type = SolidPrimitive.SPHERE
        bbox.dimensions = [float(pos_tolerance)]
        pos_con.constraint_region.primitives.append(bbox)

        bbox_pose = Pose()
        bbox_pose.position = target_point
        bbox_pose.orientation.w = 1.0
        pos_con.constraint_region.primitive_poses.append(bbox_pose)
        constraints.position_constraints.append(pos_con)

        # 2. 姿态约束 (OrientationConstraint，若提供了朝向)
        quat_msg = self._resolve_orientation(target_rpy, target_quat)
        if quat_msg is not None:
            ori_con = moveit_msgs.msg.OrientationConstraint()
            ori_con.header.frame_id = reference_frame
            ori_con.link_name = ee_link
            ori_con.orientation = quat_msg
            ori_con.absolute_x_axis_tolerance = float(rot_tolerance)
            ori_con.absolute_y_axis_tolerance = float(rot_tolerance)
            ori_con.absolute_z_axis_tolerance = float(rot_tolerance)
            ori_con.weight = 1.0
            constraints.orientation_constraints.append(ori_con)

        mpr.goal_constraints.append(constraints)

        # 发起调用并返回解析结果
        return self._call_motion_planner(req)

    def _get_pink_solver(self):
        """惰性加载 Pink 高精度凸优化求解器"""
        if self._pink_solver is None:
            from core.solver.g1_pink_ik import G1PinkIKSolver
            self._pink_solver = G1PinkIKSolver()
        return self._pink_solver

    # ==========================================================================
    # 核心规划接口 2: MoveIt 2 链路 + Pink IK 深度协同闭环规划
    # ==========================================================================
    def plan_to_pose_with_pink_ik(
        self,
        group_name: str,
        target_pos: Union[List[float], np.ndarray],
        target_rpy: Optional[Union[List[float], np.ndarray]] = None,
        target_quat: Optional[Union[List[float], Tuple[float, ...]]] = None,
        planner_id: str = "RRTConnectkConfigDefault",
        allowed_planning_time: float = 5.0,
        num_planning_attempts: int = 5,
        start_joints: Optional[Dict[str, float]] = None,
        tolerance: float = 0.01,
        seed_waist: Optional[np.ndarray] = None,
        seed_arm: Optional[np.ndarray] = None,
    ) -> PlanResult:
        """
        【MoveIt 2 链路 + Pink IK 混合规划管道】
        1. 首先通过 Pink 10-DoF/7-DoF 凸优化求解器求解最佳全局关节构型
           （实现手臂优先、腰部辅助、CoM 质心稳定、关节硬限位 100% 遵守）；
        2. 将精准目标构型作为关节空间目标注入 MoveIt 2 OMPL 规划管道；
        3. 由 MoveIt 2 OMPL (RRTConnect / RRT*) 负责环境全局避障与时间最优平滑轨迹规划。
        """
        import pinocchio as pin
        solver = self._get_pink_solver()
        t0 = time.time()

        pos = np.array(target_pos, dtype=np.float64)
        rot = None
        if target_quat is not None and len(target_quat) == 4:
            qx, qy, qz, qw = [float(v) for v in target_quat]
            if qx**2 + qy**2 + qz**2 + qw**2 > 1e-4:
                rot = pin.Quaternion(qw, qx, qy, qz).normalized().toRotationMatrix()
        elif target_rpy is not None and len(target_rpy) == 3:
            rot = pin.rpy.rpyToMatrix(float(target_rpy[0]), float(target_rpy[1]), float(target_rpy[2]))

        arm_id = "left_arm" if "left" in group_name else "right_arm"
        arm_names = solver.left_arm_joint_names if arm_id == "left_arm" else solver.right_arm_joint_names

        # 区分 7-DoF 与 10-DoF 规划组
        if group_name in ("left_arm_torso", "right_arm_torso"):
            ok, w_sol, a_sol, ik_info = solver.solve_10dof_ik(
                arm=arm_id,
                target_pos=pos,
                target_rot=rot,
                seed_waist=seed_waist,
                seed_arm=seed_arm,
            )
            if not ok or ik_info.pos_err_mm > 5.0:
                dt_ik = (time.time() - t0) * 1000.0
                return PlanResult(
                    success=False,
                    planning_time=time.time() - t0,
                    error_code=-31,
                    error_message=f"Pink IK 10-DoF 前置逆解失败 (残差: {ik_info.pos_err_mm:.2f}mm, 耗时: {dt_ik:.1f}ms)",
                )
            target_joints = {}
            for name, val in zip(solver.waist_joint_names, w_sol):
                target_joints[name] = float(val)
            for name, val in zip(arm_names, a_sol):
                target_joints[name] = float(val)

        elif group_name in ("left_arm", "right_arm"):
            ok, a_sol, ik_info = solver.solve_ik(
                arm=arm_id,
                target_pos=pos,
                target_rot=rot,
                seed_chain=seed_arm,
            )
            if not ok or ik_info.pos_err_mm > 5.0:
                dt_ik = (time.time() - t0) * 1000.0
                return PlanResult(
                    success=False,
                    planning_time=time.time() - t0,
                    error_code=-31,
                    error_message=f"Pink IK 7-DoF 前置逆解失败 (残差: {ik_info.pos_err_mm:.2f}mm, 耗时: {dt_ik:.1f}ms)",
                )
            target_joints = {name: float(val) for name, val in zip(arm_names, a_sol)}
        else:
            return PlanResult(
                success=False,
                planning_time=0.0,
                error_message=f"未知的规划组: {group_name}",
            )

        ik_time_ms = (time.time() - t0) * 1000.0

        # 将 Pink IK 求解得到的精确实体构型交由 MoveIt 2 OMPL 生成无碰撞轨迹
        plan_res = self.plan_to_joints(
            group_name=group_name,
            target_joints=target_joints,
            planner_id=planner_id,
            allowed_planning_time=allowed_planning_time,
            num_planning_attempts=num_planning_attempts,
            start_joints=start_joints,
            tolerance=tolerance,
        )

        plan_res.ik_solution = target_joints
        plan_res.ik_time_ms = ik_time_ms
        return plan_res

    # ==========================================================================
    # 核心规划接口 3: 关节空间目标配置规划
    # ==========================================================================
    def plan_to_joints(
        self,
        group_name: str,
        target_joints: Dict[str, float],
        planner_id: str = "RRTConnectkConfigDefault",
        allowed_planning_time: float = 5.0,
        num_planning_attempts: int = 5,
        start_joints: Optional[Dict[str, float]] = None,
        tolerance: float = 0.01,
    ) -> PlanResult:
        """
        向指定关节角目标配置规划全局避障轨迹
        :param group_name: 规划组名称
        :param target_joints: 目标关节角映射字典 {jname: target_rad}
        :param planner_id: 规划算法
        :param allowed_planning_time: 允许耗时
        :param start_joints: 起始关节角配置
        :param tolerance: 关节到达容差 (弧度)
        """
        req = moveit_msgs.srv.GetMotionPlan.Request()
        mpr = req.motion_plan_request
        mpr.group_name = group_name
        mpr.planner_id = planner_id
        mpr.allowed_planning_time = float(allowed_planning_time)
        mpr.num_planning_attempts = int(num_planning_attempts)
        mpr.max_velocity_scaling_factor = 1.0
        mpr.max_acceleration_scaling_factor = 1.0

        if start_joints:
            mpr.start_state.joint_state.name = list(start_joints.keys())
            mpr.start_state.joint_state.position = [float(v) for v in start_joints.values()]
            mpr.start_state.is_diff = True

        constraints = moveit_msgs.msg.Constraints()
        constraints.name = f"{group_name}_joint_goal"

        for jname, target_val in target_joints.items():
            jc = moveit_msgs.msg.JointConstraint()
            jc.joint_name = jname
            jc.position = float(target_val)
            jc.tolerance_above = float(tolerance)
            jc.tolerance_below = float(tolerance)
            jc.weight = 1.0
            constraints.joint_constraints.append(jc)

        mpr.goal_constraints.append(constraints)
        return self._call_motion_planner(req)

    # ==========================================================================
    # 规划场景与环境障碍物管理 (Obstacle Management)
    # ==========================================================================
    def add_box_obstacle(
        self,
        name: str,
        position: Union[List[float], Tuple[float, float, float]],
        size: Union[List[float], Tuple[float, float, float]],
        orientation: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
        frame_id: str = "world",
    ) -> bool:
        """
        向 MoveIt 规划场景中动态添加长方体/立方体障碍物
        :param name: 障碍物唯一标识符 ID
        :param position: [X, Y, Z] 几何中心坐标 (米)
        :param size: [dx, dy, dz] 长宽高三维尺寸 (米)
        :param orientation: [x, y, z, w] 空间四元数朝向
        :param frame_id: 基准坐标系
        """
        co = moveit_msgs.msg.CollisionObject()
        co.header.frame_id = frame_id
        co.id = name
        co.operation = moveit_msgs.msg.CollisionObject.ADD

        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.BOX
        primitive.dimensions = [float(size[0]), float(size[1]), float(size[2])]

        pose = Pose()
        pose.position.x = float(position[0])
        pose.position.y = float(position[1])
        pose.position.z = float(position[2])
        pose.orientation.x = float(orientation[0])
        pose.orientation.y = float(orientation[1])
        pose.orientation.z = float(orientation[2])
        pose.orientation.w = float(orientation[3])

        co.primitives.append(primitive)
        co.primitive_poses.append(pose)

        return self._apply_collision_object(co)

    def add_cylinder_obstacle(
        self,
        name: str,
        position: Union[List[float], Tuple[float, float, float]],
        radius: float,
        height: float,
        orientation: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
        frame_id: str = "world",
    ) -> bool:
        """
        向 MoveIt 规划场景中动态添加立柱/圆柱体障碍物
        :param name: 障碍物 ID
        :param position: 几何中心坐标
        :param radius: 半径 (米)
        :param height: 高度 (米)
        """
        co = moveit_msgs.msg.CollisionObject()
        co.header.frame_id = frame_id
        co.id = name
        co.operation = moveit_msgs.msg.CollisionObject.ADD

        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.CYLINDER
        primitive.dimensions = [float(height), float(radius)]

        pose = Pose()
        pose.position.x = float(position[0])
        pose.position.y = float(position[1])
        pose.position.z = float(position[2])
        pose.orientation.x = float(orientation[0])
        pose.orientation.y = float(orientation[1])
        pose.orientation.z = float(orientation[2])
        pose.orientation.w = float(orientation[3])

        co.primitives.append(primitive)
        co.primitive_poses.append(pose)

        return self._apply_collision_object(co)

    def remove_obstacle(self, name: str) -> bool:
        """从场景中移除指定障碍物"""
        co = moveit_msgs.msg.CollisionObject()
        co.id = name
        co.operation = moveit_msgs.msg.CollisionObject.REMOVE
        res = self._apply_collision_object(co)
        if name in self._attached_obstacles:
            self._attached_obstacles.remove(name)
        return res

    def clear_all_obstacles(self) -> bool:
        """清空所有动态注入的场景障碍物"""
        for name in list(self._attached_obstacles):
            self.remove_obstacle(name)
        return True

    def _apply_collision_object(self, co: moveit_msgs.msg.CollisionObject) -> bool:
        """应用碰撞对象变更至 PlanningScene"""
        ps = moveit_msgs.msg.PlanningScene()
        ps.is_diff = True
        ps.world.collision_objects.append(co)

        if self.scene_service_client.service_is_ready():
            req = moveit_msgs.srv.ApplyPlanningScene.Request()
            req.scene = ps
            future = self.scene_service_client.call_async(req)
            self._spin_until_future_complete(future, timeout_sec=2.0)
            if future.result() and future.result().success:
                if co.operation == moveit_msgs.msg.CollisionObject.ADD and co.id not in self._attached_obstacles:
                    self._attached_obstacles.append(co.id)
                return True

        # 降级备选：通过 Topic 广播
        self.scene_publisher.publish(ps)
        if co.operation == moveit_msgs.msg.CollisionObject.ADD and co.id not in self._attached_obstacles:
            self._attached_obstacles.append(co.id)
        return True

    # ==========================================================================
    # 底层通信调用与结果解析辅助函数
    # ==========================================================================
    def _call_motion_planner(self, req: moveit_msgs.srv.GetMotionPlan.Request) -> PlanResult:
        """调用 /plan_kinematic_path 服务并解析 OMPL 规划轨迹"""
        if not self.plan_service_client.service_is_ready():
            return PlanResult(
                success=False,
                planning_time=0.0,
                error_message="MoveIt /plan_kinematic_path 服务不可用，请确保 move_group 节点已启动",
            )

        t0 = time.perf_counter()
        future = self.plan_service_client.call_async(req)
        self._spin_until_future_complete(future, timeout_sec=req.motion_plan_request.allowed_planning_time + 3.0)
        dt = time.perf_counter() - t0

        if not future.done() or future.result() is None:
            return PlanResult(
                success=False,
                planning_time=dt,
                error_message="规划服务请求超时或未返回响应",
            )

        resp: moveit_msgs.srv.GetMotionPlan.Response = future.result()
        mpr = resp.motion_plan_response
        error_code = mpr.error_code.val

        # MoveIt 错误码 1 代表 SUCCESS
        if error_code != 1:
            return PlanResult(
                success=False,
                planning_time=dt,
                error_code=error_code,
                error_message=f"MoveIt 规划返回错误码: {error_code}",
            )

        # 提取并解析关节轨迹
        jt = mpr.trajectory.joint_trajectory
        joint_names = list(jt.joint_names)
        waypoints: List[np.ndarray] = []
        timestamps: List[float] = []

        for pt in jt.points:
            waypoints.append(np.array(pt.positions, dtype=np.float64))
            t_sec = pt.time_from_start.sec + pt.time_from_start.nanosec * 1e-9
            timestamps.append(float(t_sec))

        return PlanResult(
            success=True,
            planning_time=dt,
            joint_names=joint_names,
            trajectory=jt,
            waypoints=waypoints,
            timestamps=timestamps,
            error_code=1,
            error_message="SUCCESS",
        )

    def _spin_until_future_complete(self, future, timeout_sec: float = 5.0):
        """非阻塞等待 future 完成"""
        start = time.time()
        while not future.done() and (time.time() - start) < timeout_sec:
            rclpy.spin_once(self.node, timeout_sec=0.02)

    @staticmethod
    def _resolve_orientation(
        rpy: Optional[Union[List[float], np.ndarray]] = None,
        quat: Optional[Union[List[float], Tuple[float, ...]]] = None,
    ) -> Optional[Quaternion]:
        """将欧拉角或四元数统一转化为 geometry_msgs.msg.Quaternion"""
        if quat is not None and len(quat) == 4:
            return Quaternion(x=float(quat[0]), y=float(quat[1]), z=float(quat[2]), w=float(quat[3]))

        if rpy is not None and len(rpy) == 3:
            roll, pitch, yaw = float(rpy[0]), float(rpy[1]), float(rpy[2])
            cy = math.cos(yaw * 0.5)
            sy = math.sin(yaw * 0.5)
            cp = math.cos(pitch * 0.5)
            sp = math.sin(pitch * 0.5)
            cr = math.cos(roll * 0.5)
            sr = math.sin(roll * 0.5)

            w = cr * cp * cy + sr * sp * sy
            x = sr * cp * cy - cr * sp * sy
            y = cr * cp * cy + sr * cp * sy
            z = cr * cp * sy - sr * sp * cy
            return Quaternion(x=x, y=y, z=z, w=w)

        return None

    # ==========================================================================
    # 轨迹执行控制器接口 (实时驱动 RViz2 中的 3D 模型平滑运动)
    # ==========================================================================
    def execute_plan(self, plan_res: PlanResult, group_name: str, timeout_sec: float = 10.0) -> bool:
        """
        将规划生成的无碰撞轨迹发送至 FollowJointTrajectory 控制器执行，
        实时更新 /joint_states 并在 RViz2 桌面端驱动 3D 机器人模型动态平滑运动。
        """
        if not plan_res.success or plan_res.trajectory is None:
            self.node.get_logger().error("无法执行轨迹：规划结果为空或规划失败")
            return False

        action_name = f"{group_name}_controller/follow_joint_trajectory"
        action_client = ActionClient(self.node, FollowJointTrajectory, action_name)
        if not action_client.wait_for_server(timeout_sec=3.0):
            self.node.get_logger().error(f"控制器 Action Server 未就绪: {action_name}")
            return False

        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory = plan_res.trajectory

        future = action_client.send_goal_async(goal_msg)
        self._spin_until_future_complete(future, timeout_sec=2.0)
        goal_handle = future.result()
        if not goal_handle or not goal_handle.accepted:
            self.node.get_logger().error("控制器拒绝了轨迹执行请求")
            return False

        res_future = goal_handle.get_result_async()
        total_dur = plan_res.timestamps[-1] if plan_res.timestamps else 5.0
        self._spin_until_future_complete(res_future, timeout_sec=total_dur + 5.0)
        return True
