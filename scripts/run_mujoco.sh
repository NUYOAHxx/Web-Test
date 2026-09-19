#!/usr/bin/env bash
# ==============================================================================
# Unitree G1 53-DoF 全身物理仿真与强化学习策略部署运行脚本
# ==============================================================================

set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$DIR"

# 优先激活虚拟环境
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

python3 deploy/simulation/deploy_mujoco53.py "$@"
