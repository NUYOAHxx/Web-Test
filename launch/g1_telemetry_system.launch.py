#!/usr/bin/env python3
"""
================================================================================
Unitree G1 10-DoF 逆运动学与 Web 遥测监控系统标准 ROS 2 调度启动文件
(G1 IK & Telemetry System Launch File)
================================================================================

调度组件：
1. g1_ik_solver: 核心 10-DoF 躯干-手臂协同加权逆解与 28 对碰撞避免无头服务节点；
2. robot_state_publisher: 广播全机身 TF 坐标树与模型描述；
3. g1_telemetry_hub: Web 纯被动遥测监控与目标指令分发服务端 (默认端口 8080)；
4. [可选] rviz2: 桌面端 3D 交互式可视化与 Marker 监控。
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

    # URDF 模型路径查找
    default_urdf = "/home/parallels/ws_moveit/src/g1_description/urdf/g1_29dof.urdf"
    if not os.path.exists(default_urdf):
        default_urdf = os.path.join(dir_root, "resources/g1/g1_29dof.urdf")

    default_rviz_config = "/home/parallels/ws_moveit/src/g1_description/rviz/view_robot.rviz"

    # 读取 URDF 内容供 robot_state_publisher 使用
    robot_description_content = ""
    if os.path.exists(default_urdf):
        with open(default_urdf, "r", encoding="utf-8") as f:
            robot_description_content = f.read()

    # 启动参数
    launch_rviz_arg = DeclareLaunchArgument(
        name="rviz",
        default_value="false",
        description="是否同时启动桌面端 RViz2 监控 (true/false)"
    )

    port_arg = DeclareLaunchArgument(
        name="port",
        default_value="8080",
        description="Web 遥测监控大屏 HTTP 服务端口"
    )

    # 1. 核心无头 IK 解算服务节点 (Headless ROS 2 Node)
    ik_solver_process = ExecuteProcess(
        cmd=[sys.executable, os.path.join(dir_root, "core/solver/g1_ik_node.py")],
        output="screen",
        name="g1_ik_solver"
    )

    # 2. 机器人状态发布者 (发布 TF 坐标树)
    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[{"robot_description": robot_description_content}],
    )

    # 3. Web 纯被动遥测监控与目标指令分发服务端
    web_server_process = ExecuteProcess(
        cmd=[sys.executable, os.path.join(dir_root, "visualizer/web/server.py"), "--port", LaunchConfiguration("port")],
        output="screen",
        name="g1_telemetry_hub"
    )

    # 4. 可选桌面端 RViz2
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", default_rviz_config] if os.path.exists(default_rviz_config) else [],
        condition=IfCondition(LaunchConfiguration("rviz"))
    )

    return LaunchDescription([
        launch_rviz_arg,
        port_arg,
        ik_solver_process,
        robot_state_publisher_node,
        web_server_process,
        rviz_node,
    ])
