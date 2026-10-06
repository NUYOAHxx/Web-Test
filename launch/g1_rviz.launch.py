#!/usr/bin/env python3
"""
================================================================================
Unitree G1 10-DoF 逆运动学与机身状态 RViz 纯可视化监控标准调度启动文件
(G1 RViz Visualization Launch File - Cleanly Decoupled)
================================================================================

定位：纯可视化展示端 (Zero Kinematics / Pure Display)
- 彻底解耦：不包含任何逆运动学求解逻辑；
- 广播全机身 TF 坐标树 (robot_state_publisher)；
- 启动 RViz2 挂载 10-DoF 专属 Marker 与关节状态监控看板；
- 支持可选级联拉起 IK 求解节点 (solver:=true)。
"""

import os
import sys
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    dir_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    urdf_file = os.path.join(dir_root, "resources/g1/g1_29dof.urdf")
    rviz_config_file = os.path.join(dir_root, "visualizer/rviz/view_10dof_ik.rviz")

    # 读取 URDF 内容
    robot_description_content = ""
    if os.path.exists(urdf_file):
        with open(urdf_file, "r", encoding="utf-8") as f:
            robot_description_content = f.read()

    # 启动参数
    solver_arg = DeclareLaunchArgument(
        name="solver",
        default_value="false",
        description="是否同时在后台拉起 Headless IK 求解服务节点 (true/false)",
    )

    # 1. 机器人状态发布者 (TF 树广播)
    rsp_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[{"robot_description": robot_description_content}],
    )

    # 2. RViz2 独立可视化窗口
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", rviz_config_file] if os.path.exists(rviz_config_file) else [],
    )

    # 3. 可选：后台 IK 解算服务节点 (按需解耦拉起)
    ik_solver_process = ExecuteProcess(
        cmd=[sys.executable, os.path.join(dir_root, "core/solver/g1_ik_node.py")],
        output="screen",
        name="g1_ik_solver",
        condition=IfCondition(LaunchConfiguration("solver")),
    )

    return LaunchDescription([
        solver_arg,
        rsp_node,
        rviz_node,
        ik_solver_process,
    ])
