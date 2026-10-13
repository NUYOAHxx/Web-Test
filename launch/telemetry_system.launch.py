#!/usr/bin/env python3
"""
================================================================================
人形机器人 9-DoF 逆运动学与 Web 遥测监控系统标准 ROS 2 调度启动文件
(Humanoid IK & Telemetry System Launch File)
================================================================================

调度组件：
1. ik_solver: 核心 9-DoF 躯干-手臂协同加权逆解与碰撞避免无头服务节点；
2. robot_state_publisher: 广播全机身 TF 坐标树与模型描述；
3. telemetry_hub: Web 纯被动遥测监控与目标指令分发服务端 (默认端口 8080)；
4. [可选] rviz2: 桌面端 3D 交互式可视化与 Marker 监控。

可配置项 (全部通过启动参数, 无需改码即可换模型 / 换求解器)：
    port           Web 服务端口
    urdf           机器人 URDF 路径 (影响 IK 节点 + 大屏运动学 + TF 树)
    srdf           MoveIt SRDF 路径, 用于允许碰撞矩阵过滤
    solver_script  求解器节点脚本路径 (pink / dual_arm 两套实现可互换)
    rviz           是否同时启动桌面端 RViz2

三个组件共用同一个 URDF —— 此前 URDF 仅在启动文件内硬编码, 换模型需要改码;
现在模型路径贯穿 IK 节点、大屏服务端与 TF 树, 保持一致。
"""

import os
import sys
from launch import LaunchContext, LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _build_robot_state_publisher(context, dir_root):
    """
    按启动参数读取 URDF 内容并生成 robot_state_publisher 节点。

    必须在运行期 (而非描述生成期) 读取文件 —— 声明式生成时 LaunchConfiguration
    尚未求值, 此时读到的只能是被硬编码覆盖的默认值。

    :param context: 启动上下文, 用于对 LaunchConfiguration 求值
    :param dir_root: 仓库根目录
    :return: 节点动作列表 —— OpaqueFunction 会**迭代**本函数的返回值,
             返回单个 Node 会抛 "'Node' object is not iterable"
    """
    from launch_ros.parameter_descriptions import ParameterValue

    urdf_path = LaunchConfiguration("urdf").perform(context)
    description = ""
    if os.path.exists(urdf_path):
        with open(urdf_path, "r", encoding="utf-8") as f:
            description = f.read()
    else:
        # 不静默失败: TF 树缺失会让大屏的连杆位姿全部为空, 原因却无从追溯
        print(f"[警告] URDF 不存在, robot_state_publisher 将无模型描述: {urdf_path}")

    return [Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[{"robot_description": ParameterValue(description, value_type=str)}],
    )]


def generate_launch_description():
    dir_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    # ── 默认模型路径 (作为启动参数的默认值, 非硬编码) ──
    default_urdf = os.path.join(dir_root, "resources/humanoid/humanoid_29dof.urdf")
    default_srdf = os.path.join(dir_root, "resources/humanoid/humanoid_29dof.srdf")
    default_solver = os.path.join(dir_root, "core/solver/ik_node.py")
    default_rviz_config = os.path.join(dir_root, "visualizer/rviz/view_9dof_ik.rviz")

    urdf_arg = DeclareLaunchArgument(
        name="urdf", default_value=default_urdf,
        description="机器人 URDF 路径 (供 IK 节点、大屏运动学与 TF 树共用)")
    srdf_arg = DeclareLaunchArgument(
        name="srdf", default_value=default_srdf,
        description="MoveIt SRDF 路径, 用于允许碰撞矩阵 (ACM) 过滤")
    solver_arg = DeclareLaunchArgument(
        name="solver_script", default_value=default_solver,
        description="求解器节点脚本路径 (pink 9-DoF 协同解 / dual_arm 双臂独立解 可互换)")
    port_arg = DeclareLaunchArgument(
        name="port", default_value="8080",
        description="Web 遥测监控大屏 HTTP 服务端口")
    topic_prefix_arg = DeclareLaunchArgument(
        name="topic_prefix", default_value="/humanoid",
        description="ROS 2 话题命名空间前缀 (求解器与大屏共用; 兼容旧命名传 /g1)")
    launch_rviz_arg = DeclareLaunchArgument(
        name="rviz", default_value="false",
        description="是否同时启动桌面端 RViz2 监控 (true/false)")

    # ── 1. 核心无头 IK 解算服务节点 ──
    # solver_script 路径可切换, 但模型路径参数名对两套实现一致 (dual_arm 节点
    # 用 urdf_path), 故统一透传。
    ik_solver_process = ExecuteProcess(
        cmd=[
            sys.executable, LaunchConfiguration("solver_script"),
            "--ros-args",
            # "-p name:=value" 必须是**单个** argv token, rcl 才会正确解析。
            # 用嵌套列表让 launch 把 "name:=" 与路径替换拼成一个参数:
            #   拆成独立列表项 → "-p urdf_path:=" 缺值, 节点直接解析失败退出;
            #   写进 f-string    → 退化成字符串字面量, 替换不生效, 参数被静默忽略。
            "-p", ["urdf_path:=", LaunchConfiguration("urdf")],
            "-p", ["srdf_path:=", LaunchConfiguration("srdf")],
            "-p", ["topic_prefix:=", LaunchConfiguration("topic_prefix")],
        ],
        output="screen",
        name="ik_solver",
    )

    # ── 2. 机器人状态发布者 (TF 坐标树) ──
    rsp = OpaqueFunction(
        function=_build_robot_state_publisher, args=[dir_root])

    # ── 3. Web 纯被动遥测监控与目标指令分发服务端 ──
    web_server_process = ExecuteProcess(
        cmd=[
            sys.executable, os.path.join(dir_root, "visualizer/web/server.py"),
            "--port", LaunchConfiguration("port"),
            "--urdf", LaunchConfiguration("urdf"),
            "--srdf", LaunchConfiguration("srdf"),
        ],
        output="screen",
        name="telemetry_hub",
    )

    # ── 4. 可选桌面端 RViz2 ──
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", default_rviz_config] if os.path.exists(default_rviz_config) else [],
        condition=IfCondition(LaunchConfiguration("rviz")),
    )

    return LaunchDescription([
        urdf_arg, srdf_arg, solver_arg, port_arg, topic_prefix_arg, launch_rviz_arg,
        ik_solver_process, rsp, web_server_process, rviz_node,
    ])


# ── 直接执行入口 ────────────────────────────────────────────────────────────
# 有了本块, 该文件既可由 `ros2 launch` 调度, 也可由指定解释器直接运行:
#     .venv/bin/python3 launch/telemetry_system.launch.py port:=8080
# 后者对本工程是必需的 —— `ros2` 命令的 shebang 指向系统 python, 经它启动时
# 启动文件内的 sys.executable 会解析成系统解释器, 从而把求解器节点也拉起在
# 没有 pinocchio / rclpy 的环境里。直接执行则由 venv 解释器全程接管。
def main():
    from launch import LaunchService
    from launch.launch_description_sources import PythonLaunchDescriptionSource

    launch_file = os.path.abspath(__file__)
    context = LaunchContext(argv=sys.argv[1:])

    description = PythonLaunchDescriptionSource(launch_file).get_launch_description(context)
    service = LaunchService()
    service.include_launch_description(description)
    service.run()


if __name__ == "__main__":
    main()
