#!/usr/bin/env python3
"""双臂 IK ROS 2 (Jazzy) 节点:提供 moveit_msgs/srv/GetPositionIK 服务。

调用约定:
  ik_request.ik_link_names       = [left_wrist_yaw_link, right_wrist_yaw_link](顺序任意)
  ik_request.pose_stamped_vector = 与上对应的两个目标 Pose,frame_id 须为 base_frame(默认 pelvis)或空
  ik_request.robot_state.joint_state = seed(当前关节状态,缺失关节按 0)
  返回 solution.joint_state = 16 个关节 (waist_yaw, waist_pitch, 左7, 右7)
上层拿到后用 MoveIt 的 joint-space goal 规划(碰撞检测/OMPL 均由 MoveIt 负责)。

运行: python3 core/ros/dual_arm_ik_node.py --ros-args -p urdf_path:=/abs/g1_29dof.urdf
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import rclpy  # noqa: E402
from moveit_msgs.msg import MoveItErrorCodes  # noqa: E402
from moveit_msgs.srv import GetPositionIK  # noqa: E402
from rclpy.node import Node  # noqa: E402
from sensor_msgs.msg import JointState  # noqa: E402

from core.ros.ik_service_core import PoseInput, solve_request  # noqa: E402
from core.solver.dual_arm_ik import DEFAULT_URDF, DualArmIKSolver, IKConfig  # noqa: E402


class DualArmIKNode(Node):
    def __init__(self) -> None:
        super().__init__("dual_arm_ik")
        self.declare_parameter("urdf_path", DEFAULT_URDF)
        self.declare_parameter("base_frame", "pelvis")
        self.declare_parameter("service_name", "~/compute_ik")
        self.declare_parameter("max_iterations", 100)
        self.declare_parameter("position_tolerance", 1e-3)
        self.declare_parameter("orientation_tolerance", 1e-2)

        cfg = IKConfig(
            max_iterations=self.get_parameter("max_iterations").value,
            position_tolerance=self.get_parameter("position_tolerance").value,
            orientation_tolerance=self.get_parameter("orientation_tolerance").value,
        )
        self.solver = DualArmIKSolver(cfg, self.get_parameter("urdf_path").value)
        self.base_frame = self.get_parameter("base_frame").value
        self.create_service(GetPositionIK,
                            self.get_parameter("service_name").value, self._on_request)
        self.get_logger().info(
            f"DualArmIK ready: joints={self.solver.joint_names}, base={self.base_frame}")

    def _on_request(self, request: GetPositionIK.Request,
                    response: GetPositionIK.Response) -> GetPositionIK.Response:
        r = request.ik_request
        poses = [PoseInput(p.header.frame_id,
                           (p.pose.position.x, p.pose.position.y, p.pose.position.z),
                           (p.pose.orientation.x, p.pose.orientation.y,
                            p.pose.orientation.z, p.pose.orientation.w))
                 for p in r.pose_stamped_vector]
        js = r.robot_state.joint_state
        code, sol, res = solve_request(self.solver, self.base_frame,
                                       list(r.ik_link_names), poses,
                                       list(js.name), list(js.position))
        response.error_code = MoveItErrorCodes(val=code)
        if sol:
            out = JointState()
            out.header.stamp = self.get_clock().now().to_msg()
            out.name = list(sol.keys())
            out.position = list(sol.values())
            response.solution.joint_state = out
        if res is not None:
            self.get_logger().info(
                f"IK code={code} iters={res.iterations} err={res.total_error:.2e} "
                f"reason={res.failure_reason.value}")
        return response


def main() -> None:
    rclpy.init()
    node = DualArmIKNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
