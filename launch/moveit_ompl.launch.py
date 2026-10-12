#!/usr/bin/env python3
"""
================================================================================
MoveIt 2 OMPL 运动规划系统启动入口
================================================================================
启动组件:
    1. robot_state_publisher   —— 广播全机身 TF 坐标树
    2. move_group              —— OMPL 全局无碰撞轨迹规划服务
                                   (RRTConnect / RRT* / PRM)
    3. mock_controller_server  —— 轨迹执行服务 (FollowJointTrajectory),
                                 由 scripts/mock_controller_server.py 提供
    4. rviz2 (可选)            —— 加载 MotionPlanning 插件的桌面端界面

与旧版的差异 (修复记录):
    * 此前依赖一个并不存在的 `humanoid_moveit_config` ament 包, 启动必定抛
      PackageNotFoundError; 同时 `ompl_planning.yaml` 从未被接线加载。
    * 现改为直接读取仓库内 config/moveit_config/ 下的 yaml 并以 move_group
      参数注入, 不依赖任何额外 ament 包, 启动自包含。

用法:
    # 无头模式 (仅规划服务)
    python3 ~/ros2_ws/launch_via_venv.py launch/moveit_ompl.launch.py

    # 启动前请先 source ROS 2 环境
================================================================================
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _read(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _yaml_to_dict(text: str) -> dict:
    """极简 YAML -> dict 解析 (仅处理 move_group 所需的扁平/二级结构)。

    刻意不引入 yaml 依赖到 launch 侧: 本工程的规划配置结构简单且稳定,
    且 launch 文件需在 ROS 环境下独立可用。
    """
    import yaml  # 延迟导入, 避免非规划场景的启动路径强依赖
    return yaml.safe_load(text) or {}


def generate_launch_description() -> LaunchDescription:
    dir_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    urdf_file = os.path.join(dir_root, "resources/humanoid/humanoid_29dof.urdf")
    srdf_file = os.path.join(dir_root, "resources/humanoid/humanoid_29dof.srdf")
    cfg_dir = os.path.join(dir_root, "config/moveit_config/config")
    kinematics_file = os.path.join(cfg_dir, "kinematics.yaml")
    joint_limits_file = os.path.join(cfg_dir, "joint_limits.yaml")
    ompl_file = os.path.join(cfg_dir, "ompl_planning.yaml")
    controllers_file = os.path.join(cfg_dir, "moveit_controllers.yaml")
    rviz_config_file = os.path.join(dir_root, "config/moveit_config/moveit.rviz")
    mock_server = os.path.join(dir_root, "scripts/mock_controller_server.py")

    robot_description = _read(urdf_file)
    robot_description_semantic = _read(srdf_file)
    kinematics_cfg = _yaml_to_dict(_read(kinematics_file))
    joint_limits_cfg = _yaml_to_dict(_read(joint_limits_file))
    ompl_cfg = _yaml_to_dict(_read(ompl_file))
    controllers_cfg = _yaml_to_dict(_read(controllers_file))

    launch_rviz_arg = DeclareLaunchArgument(
        name="rviz",
        default_value="false",
        description="是否启动带 MotionPlanning 插件的 RViz2 窗口 (true/false)",
    )

    # ── 1. TF 广播 ──
    rsp_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[{"robot_description": robot_description}],
    )

    # ── 2. OMPL 规划服务 ──
    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        name="move_group",
        output="screen",
        parameters=[
            {"robot_description": robot_description},
            {"robot_description_semantic": robot_description_semantic},
            # 规划组运动学求解器
            {
                "robot_description_planning.joint_groups": kinematics_cfg,
                # TOTG 时间参数化必须读取该命名空间下的加速度限值,
                # 缺失会导致 "No acceleration limit was defined" 并使规划管线失败
                "robot_description_planning.joint_limits": joint_limits_cfg,
            },
            # OMPL 管线 (RRTConnect / RRT* / PRM 与各组规划器映射)
            {"ompl": ompl_cfg},
            # 轨迹执行: 控制器管理器 + 关节限位
            controllers_cfg,
            {
                "trajectory_execution.managed_controllers": [
                    f"{name}/{name}"
                    for name in controllers_cfg.get("moveit_simple_controller_manager", {})
                    .get("controller_names", [])
                ],
                "moveit_manage_controllers": True,
                "publish_robot_description": True,
                "publish_robot_description_semantic": True,
                "publish_planning_scene": True,
                # 本工程无物理机器人, 规划后不自动执行
                "trajectory_execution.execution_duration_monitoring": True,
            },
            {"planning_scene_monitor.robot_description": robot_description},
            {"planning_scene_monitor.robot_description_semantic": robot_description_semantic},
            {"default_planning_pipeline": "ompl"},
            {"planning_pipelines": ["ompl"]},
        ],
    )

    # ── 3. 轨迹执行服务 ──
    # 仓库内普通 python 脚本, 非 ament 可执行文件, 故用 ExecuteProcess 启动
    mock_controller_process = ExecuteProcess(
        cmd=["python3", mock_server],
        output="screen",
        name="mock_trajectory_controller",
    )

    # ── 4. RViz ──
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", rviz_config_file] if os.path.exists(rviz_config_file) else [],
        condition=IfCondition(LaunchConfiguration("rviz")),
    )

    return LaunchDescription([
        launch_rviz_arg,
        rsp_node,
        mock_controller_process,
        move_group_node,
        rviz_node,
    ])