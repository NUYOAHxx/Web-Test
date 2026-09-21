#!/usr/bin/env python3
"""
================================================================================
Unitree G1 MoveIt 2 OMPL (Open Motion Planning Library) 标准调度启动文件
(G1 MoveIt 2 OMPL Motion Planning Pipeline Launch File)
================================================================================

调度组件：
1. robot_state_publisher: 广播全机身 TF 坐标树与模型描述；
2. mock_controller_server: 仿真控制器服务端，发布 /joint_states 并支持 Trajectory 执行；
3. move_group: MoveIt 2 核心节点，载入 OMPL 流水线 (RRTConnect / RRT* / PRM) 与场景监视器；
4. [可选] rviz2: 启动带有 MotionPlanning 插件的 RViz2 桌面规划交互窗口 (默认 false)。
"""

import os
import sys
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    dir_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    # 1. 核心模型与配置文件定位
    urdf_file = os.path.join(dir_root, "resources/g1/g1_29dof.urdf")
    srdf_file = os.path.join(dir_root, "resources/g1/g1_29dof.srdf")
    kinematics_file = os.path.join(dir_root, "config/moveit/kinematics.yaml")
    joint_limits_file = os.path.join(dir_root, "config/moveit/joint_limits.yaml")
    controllers_file = os.path.join(dir_root, "config/moveit/moveit_controllers.yaml")
    rviz_config_file = os.path.join(dir_root, "config/moveit/moveit.rviz")

    # 2. 构建 MoveIt 2 配置对象
    moveit_config = (
        MoveItConfigsBuilder("g1_29dof", package_name="g1_moveit_config")
        .robot_description(file_path=urdf_file)
        .robot_description_semantic(file_path=srdf_file)
        .robot_description_kinematics(file_path=kinematics_file)
        .joint_limits(file_path=joint_limits_file)
        .planning_pipelines(pipelines=["ompl"])
        .trajectory_execution(file_path=controllers_file)
        .planning_scene_monitor(
            publish_robot_description=True, publish_robot_description_semantic=True
        )
        .to_moveit_configs()
    )

    # 3. 启动参数
    launch_rviz_arg = DeclareLaunchArgument(
        name="rviz",
        default_value="false",
        description="是否启动带有 MotionPlanning 插件的 RViz2 桌面端窗口 (true/false)",
    )

    # 4. Robot State Publisher 节点
    rsp_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[moveit_config.robot_description],
    )

    # 5. 仿真轨迹执行与状态同步节点 (提供 FollowJointTrajectory 并广播 /joint_states)
    mock_controller_node = Node(
        package="g1_moveit_config",
        executable="mock_controller_server.py",
        name="mock_trajectory_server",
        output="screen",
    )

    # 6. Move Group 核心服务节点 (挂载 OMPL 规划流水线与 28 对 ACM 碰撞检测)
    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        name="move_group",
        output="screen",
        parameters=[moveit_config.to_dict()],
        arguments=["--ros-args", "--log-level", "info"],
    )

    # 7. 可选 RViz2 交互窗口
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", rviz_config_file] if os.path.exists(rviz_config_file) else [],
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.planning_pipelines,
            moveit_config.robot_description_kinematics,
            moveit_config.joint_limits,
        ],
        condition=IfCondition(LaunchConfiguration("rviz")),
    )

    return LaunchDescription([
        launch_rviz_arg,
        rsp_node,
        mock_controller_node,
        move_group_node,
        rviz_node,
    ])
