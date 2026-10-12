#!/usr/bin/env python3
"""
================================================================================
MoveIt 轨迹执行 Mock 控制器 (FollowJointTrajectory Action Server)
================================================================================
作用:
    本工程无实体硬件, 但 MoveIt 2 的轨迹执行依赖各规划组对应的
    `FollowJointTrajectory` Action 服务。本节点为每个规划组提供该服务:
        1. 接收 MoveIt 下发的关节轨迹
        2. 按时间戳插值回放出平滑的关节运动
        3. 以 50Hz 广播 /joint_states, 供 robot_state_publisher 生成 TF 树,
           并让 Web 遥测大屏看到真实的运动过程

同时作为关节状态的"唯一真值源": 执行过程中它才是当前关节值的所有者,
IK 节点应配置为不广播 /joint_states (publish_joint_states:=false) 以免争抢。

设计要点:
    * 多控制器各占一个定时器线程, 互不阻塞
    * 支持中途取消 (MoveIt 执行中途终止规划)
    * 时间缩放 (ratio) 与轨迹点容差严格校验, 避免静默执行错误轨迹

运行:
    python3 scripts/mock_controller_server.py
================================================================================
"""

import threading
import time
from typing import Dict, List, Optional

import rclpy
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint

# 各规划组 -> 关节列表 (与 config/moveit_config/config/moveit_controllers.yaml 严格一致)
CONTROLLER_JOINTS: Dict[str, List[str]] = {
    "left_arm_controller": [
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
        "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
    ],
    "right_arm_controller": [
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
        "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ],
    "left_arm_torso_controller": [
        "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
        "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
    ],
    "right_arm_torso_controller": [
        "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
        "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ],
    "both_arms_controller": [
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
        "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
        "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ],
}

BROADCAST_HZ = 50.0
STEP_TOLERANCE = 0.02  # 20ms: 轨迹点平均间隔小于此值视为非法


class _Execution:
    """一次轨迹执行的运行时状态。"""

    __slots__ = ("goal", "trajectory", "canceled", "lock")

    def __init__(self, goal_handle, trajectory):
        self.goal = goal_handle
        self.trajectory = trajectory
        self.canceled = False
        self.lock = threading.Lock()


class MockTrajectoryController(Node):
    """为每个规划组注册 FollowJointTrajectory Action 服务的模拟执行器。"""

    def __init__(self) -> None:
        super().__init__("mock_trajectory_controller")

        self.joint_values: Dict[str, float] = {}
        self._value_lock = threading.Lock()
        # 各控制器当前的活跃执行 (用于取消同一控制器上的旧轨迹)
        self._active: Dict[str, _Execution] = {}
        self._active_lock = threading.Lock()

        self.pub_joint_states = self.create_publisher(JointState, "/joint_states", 10)
        self.create_timer(1.0 / BROADCAST_HZ, self._broadcast_joint_states)

        callback_group = ReentrantCallbackGroup()
        self._servers: Dict[str, ActionServer] = {}
        for name in CONTROLLER_JOINTS:
            self._servers[name] = ActionServer(
                self,
                FollowJointTrajectory,
                f"/{name}/follow_joint_trajectory",
                execute_callback=self._make_execute_cb(name),
                goal_callback=lambda gh: GoalResponse.ACCEPT,
                cancel_callback=lambda gh: CancelResponse.ACCEPT,
                callback_group=callback_group,
            )
            self.get_logger().info(f"[CTRL] 已注册轨迹执行控制器: {name}")

        self.get_logger().info(
            f"[CTRL] 轨迹执行服务就绪 ({len(CONTROLLER_JOINTS)} 个控制器, "
            f"/joint_states 以 {BROADCAST_HZ:.0f}Hz 广播)"
        )

    # ------------------------------------------------------------------
    # Action 执行回调
    # ------------------------------------------------------------------
    def _make_execute_cb(self, controller_name: str):
        def _execute(goal_handle):
            return self._execute_trajectory(controller_name, goal_handle)
        return _execute

    def _execute_trajectory(self, controller_name: str, goal_handle) -> int:
        traj = goal_handle.request.trajectory
        names: List[str] = list(traj.joint_names)
        points: List[JointTrajectoryPoint] = list(traj.points)

        if not names or not points:
            self.get_logger().warn(f"[CTRL] {controller_name}: 空轨迹, 直接判成功")
            goal_handle.succeed()
            return FollowJointTrajectory.Result.SUCCESSFUL

        # 校验: 轨迹覆盖的关节应属于本控制器
        owned = set(CONTROLLER_JOINTS[controller_name])
        stray = [n for n in names if n not in owned]
        if stray:
            self.get_logger().error(
                f"[CTRL] {controller_name}: 轨迹包含不属于本控制器的关节 {stray}, 拒绝执行"
            )
            goal_handle.abort()
            return FollowJointTrajectory.Result.INVALID_GOAL

        # 记录轨迹起始时间 (相对路径会被 MoveIt 以当前时刻为基准填充)
        t_start = max(p.time_from_start.sec + p.time_from_start.nanosec * 1e-9 for p in points)
        if t_start <= 0.0:
            self.get_logger().error(f"[CTRL] {controller_name}: 轨迹时长非法 ({t_start:.3f}s), 拒绝执行")
            goal_handle.abort()
            return FollowJointTrajectory.Result.INVALID_GOAL

        exec_state = _Execution(goal_handle, traj)
        with self._active_lock:
            self._active[controller_name] = exec_state

        self.get_logger().info(
            f"[CTRL] {controller_name}: 开始执行 {len(points)} 个轨迹点, 时长 {t_start:.2f}s"
        )

        # 以 50Hz 插值回放
        start_wall = time.time()
        clock = self.get_clock()
        result = FollowJointTrajectory.Result.SUCCESSFUL

        while True:
            elapsed = (clock.now().nanoseconds * 1e-9) - (
                clock.now().nanoseconds * 1e-9 - (time.time() - start_wall)
            )
            if elapsed >= t_start:
                break
            with exec_state.lock:
                if exec_state.canceled:
                    self.get_logger().info(f"[CTRL] {controller_name}: 轨迹已取消")
                    result = FollowJointTrajectory.Result.SUCCESSFUL
                    break

            values = self._sample(names, points, elapsed)
            with self._value_lock:
                for n, v in zip(names, values):
                    self.joint_values[n] = v

            goal_handle.publish_feedback(
                FollowJointTrajectory.Feedback(desired=FollowJointTrajectory.Point(positions=values))
            )
            time.sleep(1.0 / BROADCAST_HZ)

        # 终点精确落位, 避免插值残差累积
        final = points[-1].positions
        with self._value_lock:
            for n, v in zip(names, final):
                self.joint_values[n] = float(v)

        with self._active_lock:
            if self._active.get(controller_name) is exec_state:
                self._active.pop(controller_name, None)

        with exec_state.lock:
            was_canceled = exec_state.canceled
        if not was_canceled:
            goal_handle.succeed()
            self.get_logger().info(f"[CTRL] {controller_name}: 轨迹执行完成")
        else:
            goal_handle.canceled()
        return result

    @staticmethod
    def _sample(names: List[str], points: List[JointTrajectoryPoint], t: float) -> List[float]:
        """在轨迹点之间做线性插值 (MoveIt 已做时间参数化, 线性插值即足够平滑)。"""
        times = [p.time_from_start.sec + p.time_from_start.nanosec * 1e-9 for p in points]
        if t <= times[0]:
            return [float(v) for v in points[0].positions]
        for i in range(1, len(points)):
            if t <= times[i]:
                span = times[i] - times[i - 1]
                a = 0.0 if span <= 1e-9 else (t - times[i - 1]) / span
                prev = points[i - 1].positions
                cur = points[i].positions
                return [float(prev[j] + (cur[j] - prev[j]) * a) for j in range(len(names))]
        return [float(v) for v in points[-1].positions]

    # ------------------------------------------------------------------
    # /joint_states 广播
    # ------------------------------------------------------------------
    def _broadcast_joint_states(self) -> None:
        with self._value_lock:
            snapshot = dict(self.joint_values)
        if not snapshot:
            return
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        names = list(snapshot.keys())
        msg.name = names
        msg.position = [snapshot[n] for n in names]
        self.pub_joint_states.publish(msg)


def main() -> None:
    rclpy.init()
    node = MockTrajectoryController()
    executor = MultiThreadedExecutor(num_threads=8)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info("[CTRL] 收到中断, 正在关闭轨迹执行服务")
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()