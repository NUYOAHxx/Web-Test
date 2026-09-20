#!/bin/bash
# ==============================================================================
# Unitree G1 实时高精度可视化监控仪表盘 (Web Dashboard) 一键启动脚本
# ==============================================================================

set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$DIR"

PORT="${1:-8080}"

# 若该端口已有遗留实例占用，自动释放
if fuser "$PORT/tcp" > /dev/null 2>&1; then
    echo "[提示] 检测到端口 $PORT 已被占用，正在释放历史实例..."
    fuser -k "$PORT/tcp" > /dev/null 2>&1 || true
    sleep 0.5
fi

# 尝试加载 ROS 2 环境 (若存在) 以便自动同步发布 /joint_states 供 RViz 联动
if [ -f "/opt/ros/jazzy/setup.bash" ]; then
    source /opt/ros/jazzy/setup.bash > /dev/null 2>&1 || true
elif [ -f "/opt/ros/humble/setup.bash" ]; then
    source /opt/ros/humble/setup.bash > /dev/null 2>&1 || true
fi

if [ -f "$HOME/ws_moveit/install/setup.bash" ]; then
    source "$HOME/ws_moveit/install/setup.bash" > /dev/null 2>&1 || true
fi

echo "=================================================================="
echo "    Unitree G1 实时高精度可视化监控仪表盘 (Web Dashboard)"
echo "    访问地址: http://localhost:$PORT"
echo "=================================================================="

# 检查并拉起无头 IK 求解服务节点 (始终加载最新算法代码)
if pgrep -f "g1_ik_node.py" > /dev/null 2>&1; then
    echo "[提示] 检测到旧版本 g1_ik_node.py 实例，正在重启以加载最新算法代码..."
    pkill -9 -f "g1_ik_node.py" > /dev/null 2>&1 || true
    sleep 0.5
fi

echo "[提示] 启动后台 Headless 10-DoF IK 求解服务节点 (core/solver/g1_ik_node.py)..."
python3 -u "$DIR/core/solver/g1_ik_node.py" &
IK_PID=$!
trap "kill -9 $IK_PID 2>/dev/null || true" EXIT INT TERM
sleep 0.5

python3 -u "$DIR/visualizer/web/server.py" --port "$PORT"
