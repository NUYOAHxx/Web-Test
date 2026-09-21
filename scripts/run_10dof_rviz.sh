#!/usr/bin/env bash
# ==============================================================================
# Unitree G1 10-DoF (3-DoF 腰部 + 7-DoF 手臂) 协同逆运动学 RViz 可视化一键启动脚本
# ==============================================================================

set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$DIR"

# 检查 ROS 2 环境
if [ -f "/opt/ros/jazzy/setup.bash" ]; then
    source /opt/ros/jazzy/setup.bash
elif [ -f "/opt/ros/humble/setup.bash" ]; then
    source /opt/ros/humble/setup.bash
fi

if [ -f "$HOME/ws_moveit/install/setup.bash" ]; then
    source "$HOME/ws_moveit/install/setup.bash"
fi

URDF_PATH="$DIR/resources/g1/g1_29dof.urdf"
RVIZ_CFG="$DIR/visualizer/rviz/view_10dof_ik.rviz"

echo "=================================================================="
echo "    启动 Unitree G1 10-DoF 躯干-手臂协同逆运动学 RViz 可视化看板"
echo "=================================================================="

PIDS=()
cleanup() {
    echo -e "\n正在关闭可视化后台进程..."
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
        fi
    done
    wait 2>/dev/null || true
    echo "清理完成，已安全退出。"
}
trap cleanup EXIT INT TERM

# 1. 检查是否已经有 robot_state_publisher 运行
if ! pgrep -f "robot_state_publisher" > /dev/null; then
    echo "[1/3] 启动 robot_state_publisher..."
    ros2 run robot_state_publisher robot_state_publisher --ros-args -p robot_description:="$(cat "$URDF_PATH")" > /dev/null 2>&1 &
    PIDS+=($!)
    sleep 1
else
    echo "[1/3] 检测到已存在 robot_state_publisher，自动复用。"
fi

# 2. 检查是否已经有 rviz2 运行
if ! pgrep -f "rviz2" > /dev/null; then
    echo "[2/3] 启动 RViz2 3D 渲染窗口..."
    rviz2 -d "$RVIZ_CFG" > /dev/null 2>&1 &
    PIDS+=($!)
    sleep 2
else
    echo "[2/3] 检测到已存在 RViz2 窗口，自动接入可视化。"
fi

# 3. 启动 10-DoF 交互式解算与终端监控控制台
echo "[3/3] 进入 10-DoF 交互式控制台与数据监控报表..."
python3 "$DIR/visualizer/rviz/visualize_10dof_rviz.py"
