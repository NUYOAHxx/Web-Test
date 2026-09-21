#!/usr/bin/env bash
# ==============================================================================
# Unitree G1 MoveIt 2 OMPL (Open Motion Planning Library) 一键启动调度脚本
# ==============================================================================

set -e

DIR_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$DIR_ROOT"

echo "======================================================================"
echo " 🚀 正在启动 Unitree G1 MoveIt 2 OMPL 全局运动规划服务系统..."
echo "======================================================================"
echo " 规划流水线：OMPL (RRTConnect / RRT* / PRM) + Time-Optimal Parameterization"
echo " 规划组支持："
echo "   • left_arm (7-DoF 单臂)"
echo "   • right_arm (7-DoF 单臂)"
echo "   • left_arm_torso (10-DoF 躯干-手臂协同)"
echo "   • right_arm_torso (10-DoF 躯干-手臂协同)"
echo "   • both_arms (14-DoF 双臂协同)"
echo "----------------------------------------------------------------------"

# 检查并加载 ROS 2 与 MoveIt 环境
if [ -f "/opt/ros/jazzy/setup.bash" ]; then
    source /opt/ros/jazzy/setup.bash
elif [ -f "/opt/ros/humble/setup.bash" ]; then
    source /opt/ros/humble/setup.bash
else
    echo "❌ 错误: 未检测到系统 ROS 2 环境 (/opt/ros/jazzy 或 /opt/ros/humble)"
    exit 1
fi

if [ -f "$HOME/ws_moveit/install/setup.bash" ]; then
    source "$HOME/ws_moveit/install/setup.bash"
fi

# 检查是否传入 --rviz 参数
RVIZ_ARG="false"
for arg in "$@"; do
    if [ "$arg" == "--rviz" ] || [ "$arg" == "rviz:=true" ]; then
        RVIZ_ARG="true"
    fi
done

# 清理遗留 move_group 进程
pkill -f "moveit_ros_move_group" 2>/dev/null || true
sleep 0.5

echo "✔️ 正在通过 ros2 launch 调度 MoveIt 2 move_group 规划核心 (rviz:=$RVIZ_ARG)..."
exec ros2 launch launch/g1_moveit_ompl.launch.py rviz:="$RVIZ_ARG"
