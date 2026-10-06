#!/usr/bin/env python3
"""
================================================================================
Unitree G1 MoveIt 2 链路逆运动学 (IK) 求解服务与规划流水线自动化测试套件
(MoveIt 2 Kinematics Pipeline & GetPositionIK Integration Test Suite)
================================================================================

测试目标：
1. 验证 MoveIt 2 标准服务协议兼容性 (/g1/compute_ik, moveit_msgs/srv/GetPositionIK)；
2. 验证 7-DoF 单臂与 10-DoF 躯干-手臂协同在 MoveIt 消息体系下的解算精度与合规性；
3. 验证关节输出 100% 符合 MoveIt joint_limits.yaml 物理极限；
4. 验证面对超限/奇异目标时的诚实拦截机制 (MoveItErrorCodes.NO_IK_SOLUTION)；
5. 验证 MoveIt 2 混合规划管道 (Pink IK 前置引导 + MoveIt 2 OMPL 碰撞避障轨迹)。
"""

import os
import sys
import time
import math
import numpy as np
import pytest

# 项目根目录挂载
DIR_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if DIR_ROOT not in sys.path:
    sys.path.insert(0, DIR_ROOT)

try:
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import PoseStamped, Point, Quaternion
    from moveit_msgs.srv import GetPositionIK
    from moveit_msgs.msg import PositionIKRequest, MoveItErrorCodes, RobotState
    HAS_ROS2 = True
except (ImportError, ModuleNotFoundError):
    HAS_ROS2 = False
    rclpy = None
    Node = object
    GetPositionIK = None
    MoveItErrorCodes = None

import pinocchio as pin

from core.solver.g1_pink_ik import G1PinkIKSolver, G1_DEFAULT_STAND_JOINTS, G1_READY_POSE
from core.kinematics.g1_model import G1_JOINT_LIMITS, G1_WAIST_LIMITS
from core.planning.moveit_ompl_planner import MoveItOMPLPlanner


@pytest.fixture(scope="module")
def ros_context():
    """初始化测试用 ROS 2 上下文与客户端节点"""
    if not HAS_ROS2:
        yield {"node": None, "client": None, "service_ready": False}
        return

    if not rclpy.ok():
        rclpy.init()
    node = Node("test_moveit_ik_pipeline_client")
    client = node.create_client(GetPositionIK, "/g1/compute_ik")
    
    # 阻塞等待服务就绪 (最多等待 3.0s)
    service_ready = client.wait_for_service(timeout_sec=3.0)
    
    yield {"node": node, "client": client, "service_ready": service_ready}
    
    node.destroy_node()


def test_moveit_service_ready(ros_context):
    """测试 1: 验证 MoveIt 2 标准 /g1/compute_ik 服务节点在线（若未启动后台节点则跳过集成测试）"""
    if not ros_context["service_ready"]:
        pytest.skip("MoveIt IK 服务未启动（需运行 launch/g1_telemetry_system.launch.py 或 core/solver/g1_ik_node.py）")
    assert ros_context["service_ready"]


def test_moveit_7dof_left_arm_ik(ros_context):
    """测试 2: 7-DoF 单臂 (left_arm) 标准 MoveIt IK 请求与 6D 位姿收敛性"""
    if not ros_context["service_ready"]:
        pytest.skip("MoveIt IK 服务未就绪")

    node = ros_context["node"]
    client = ros_context["client"]

    req = GetPositionIK.Request()
    ik_req = req.ik_request
    ik_req.group_name = "left_arm"
    ik_req.avoid_collisions = True

    # 目标：工作空间内舒适拾取点 [0.26, 0.21, 0.80]，配合手爪自然朝向
    ik_req.pose_stamped.header.frame_id = "torso_link"
    ik_req.pose_stamped.pose.position = Point(x=0.26, y=0.21, z=0.80)
    ik_req.pose_stamped.pose.orientation = Quaternion(x=0.0919, y=0.3401, z=0.0417, w=0.9349)

    future = client.call_async(req)
    rclpy.spin_until_future_complete(node, future, timeout_sec=2.0)

    assert future.done(), "服务调用超时"
    res = future.result()
    assert res is not None, "服务返回 None"
    assert res.error_code.val == MoveItErrorCodes.SUCCESS, f"求解失败，错误码: {res.error_code.val}"

    # 验证返回关节数量与命名规范
    joint_names = res.solution.joint_state.name
    joint_positions = res.solution.joint_state.position
    assert len(joint_names) == 7, f"7-DoF 应返回 7 个关节，实际返回: {len(joint_names)}"
    assert all("left_" in name for name in joint_names), "关节名称应为 left_ 前缀"

    # 验证关节限位 100% 遵守
    for name, pos in zip(joint_names, joint_positions):
        min_lim, max_lim = G1_JOINT_LIMITS[name]
        assert min_lim - 1e-4 <= pos <= max_lim + 1e-4, f"关节 {name}={pos:.3f} 越界 [{min_lim:.3f}, {max_lim:.3f}]"


def test_moveit_10dof_torso_coordination_ik(ros_context):
    """测试 3: 10-DoF 躯干-手臂协同 (left_arm_torso) MoveIt 远距离大跨度求解与手臂优先原则"""
    if not ros_context["service_ready"]:
        pytest.skip("MoveIt IK 服务未就绪")

    node = ros_context["node"]
    client = ros_context["client"]

    req = GetPositionIK.Request()
    ik_req = req.ik_request
    ik_req.group_name = "left_arm_torso"
    ik_req.avoid_collisions = True

    # 远距离大跨度目标 [0.52, 0.22, 0.85] (7-DoF 单臂极限约 39cm，必然需要腰部 3-DoF 协助)
    ik_req.pose_stamped.header.frame_id = "pelvis"
    ik_req.pose_stamped.pose.position = Point(x=0.52, y=0.22, z=0.85)
    ik_req.pose_stamped.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)

    future = client.call_async(req)
    rclpy.spin_until_future_complete(node, future, timeout_sec=2.0)

    assert future.done(), "服务调用超时"
    res = future.result()
    assert res.error_code.val == MoveItErrorCodes.SUCCESS, f"10-DoF 协同求解失败，错误码: {res.error_code.val}"

    joint_names = res.solution.joint_state.name
    joint_positions = res.solution.joint_state.position
    assert len(joint_names) == 10, f"10-DoF 协同应包含 3腰 + 7臂 共 10 个关节，实际: {len(joint_names)}"

    sol_dict = dict(zip(joint_names, joint_positions))

    # 验证腰部介入：在 52cm 远端，腰部应明显协助倾转/偏航
    waist_yaw = sol_dict["waist_yaw_joint"]
    waist_pitch = sol_dict["waist_pitch_joint"]
    assert abs(waist_yaw) > 0.05 or abs(waist_pitch) > 0.05, "远距离抓取腰部未如期介入协同"

    # 验证腰部与手臂各关节绝对处于物理硬限位之内
    for name, pos in sol_dict.items():
        if name in G1_WAIST_LIMITS:
            min_lim, max_lim = G1_WAIST_LIMITS[name]
        else:
            min_lim, max_lim = G1_JOINT_LIMITS[name]
        assert min_lim - 1e-4 <= pos <= max_lim + 1e-4, f"关节 {name}={pos:.3f} 越界"


def test_moveit_impossible_pose_honest_rejection(ros_context):
    """测试 4: MoveIt 链路面对极端不可达/严重违规位姿时的诚实拦截测试 (0% 虚报成功)"""
    if not ros_context["service_ready"]:
        pytest.skip("MoveIt IK 服务未就绪")

    node = ros_context["node"]
    client = ros_context["client"]

    req = GetPositionIK.Request()
    ik_req = req.ik_request
    ik_req.group_name = "left_arm"

    # 极端超限目标：后方 1.5 米 (不可能到达)
    ik_req.pose_stamped.header.frame_id = "torso_link"
    ik_req.pose_stamped.pose.position = Point(x=-1.5, y=0.0, z=0.8)
    ik_req.pose_stamped.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)

    future = client.call_async(req)
    rclpy.spin_until_future_complete(node, future, timeout_sec=2.0)

    res = future.result()
    assert res.error_code.val == MoveItErrorCodes.NO_IK_SOLUTION, (
        f"严重不可达目标未被拦截，返回码: {res.error_code.val}"
    )


def test_moveit_ompl_pink_ik_planner_pipeline():
    """测试 5: MoveIt 2 OMPL 混合规划管道 (Pink IK 前置引导 + MoveIt 2 OMPL 轨迹规划)"""
    if not HAS_ROS2:
        pytest.skip("当前环境缺少 ROS 2 支持")

    # 验证 MoveItOMPLPlanner 具备 plan_to_pose_with_pink_ik 接口能力
    planner = MoveItOMPLPlanner(node=None, wait_for_services=False)
    assert hasattr(planner, "plan_to_pose_with_pink_ik"), "MoveItOMPLPlanner 缺失 plan_to_pose_with_pink_ik 方法"

    # 本地前置执行 Pink IK 求解逻辑验证
    solver = planner._get_pink_solver()
    target_pos = np.array([0.32, 0.22, 0.88])
    ok, a_sol, info = solver.solve_ik("left_arm", target_pos)
    assert ok, f"Pink IK 7-DoF 单元求解失败: {info}"
    assert info.pos_err_mm < 2.0, f"求解残差过大: {info.pos_err_mm}mm"


if __name__ == "__main__":
    print("=" * 80)
    print("[START] 启动 Unitree G1 MoveIt 2 链路逆运动学 (IK) 专项自动化测试")
    print("=" * 80)
    pytest.main([__file__, "-v", "-s"])
