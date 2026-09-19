#!/usr/bin/env bash
# ==============================================================================
# Unitree G1 10-DoF 逆运动学解算与 Web 数字孪生遥测大屏一键启动调度脚本
# ==============================================================================

set -e

DIR_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$DIR_ROOT"

echo "======================================================================"
echo " 🚀 正在一键启动 Unitree G1 10-DoF 逆运动学与 Web 遥测监控系统..."
echo "======================================================================"
echo " 架构：算展彻底解耦 (Headless IK 节点后台解算 + Web 端纯被动遥测与指令分发)"
echo " 标准 ROS 2 话题："
echo "   • 笛卡尔指令: /g1/kinematics/target_pose"
echo "   • 实际末端反馈: /g1/kinematics/actual_pose"
echo "   • 关节状态总线: /joint_states"
echo "   • 算法度量与诊断: /g1/kinematics/solver_metrics"
echo "   • 28对全域安全雷达: /g1/safety/collision_status"
echo "----------------------------------------------------------------------"

# 检查 ROS 2 环境
if [ -f "/opt/ros/jazzy/setup.bash" ]; then
    source /opt/ros/jazzy/setup.bash
elif [ -f "/opt/ros/humble/setup.bash" ]; then
    source /opt/ros/humble/setup.bash
fi

if [ -f "$HOME/ws_moveit/install/setup.bash" ]; then
    source "$HOME/ws_moveit/install/setup.bash"
fi

# 清理历史旧进程
fuser -k 8080/tcp 2>/dev/null || true
pkill -f "g1_ik_node.py" 2>/dev/null || true

# 检查是否使用 ros2 launch 还是直接调度
if command -v ros2 &> /dev/null; then
    echo "✔️ 检测到 ROS 2 环境，使用 ros2 launch 调度系统..."
    exec ros2 launch deploy/launch/g1_telemetry_system.launch.py "$@"
else
    echo "⚠️ 未检测到全局 ros2 命令，使用 Python 多进程直接调度..."
    python3 deploy/solver/g1_ik_node.py &
    IK_PID=$!
    trap "kill $IK_PID 2>/dev/null || true" EXIT INT TERM
    python3 deploy/visualizer/web/server.py --port 8080
fi
