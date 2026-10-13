#!/usr/bin/env bash
# ==============================================================================
# 人形机器人 Web 数字孪生遥测大屏 —— 一键启动
# ==============================================================================
#
# 用法:
#   bash scripts/start_web.sh                      # 默认模型 + Pink 求解器
#   bash scripts/start_web.sh --port 9090
#   bash scripts/start_web.sh --urdf resources/humanoid/humanoid_29dof.urdf
#   bash scripts/start_web.sh --solver dual_arm   # 切换解算器
#   bash scripts/start_web.sh --topic-prefix /g1 # 话题命名空间 (默认 /humanoid)
#   bash scripts/start_web.sh --rviz              # 同时启动桌面端 RViz2
#
# 设计要点 (相对早期版本):
#   1. 仓库路径由脚本自身位置推导, 不再硬编码绝对路径, 仓库可整体移动/改名;
#   2. ament 标记收在仓库内 (resources/ament/), 不依赖 ~/ros2_ws 等外部目录;
#   3. 不再需要 launch_via_venv.py 绕行 —— 直接用 venv 的解释器调用
#      launch 文件, 彻底避开 `ros2` 命令 shebang 指向系统 python 的问题。
# ==============================================================================
set -euo pipefail

# ── 仓库根目录 (本脚本位于 <repo>/scripts/) ──
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO"

PORT=8080
URDF=""
SOLVER="pink"
SRDF=""
RVIZ="false"
TOPIC_PREFIX="/humanoid"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --port)   PORT="$2"; shift 2 ;;
        --urdf)   URDF="$2"; shift 2 ;;
        --srdf)   SRDF="$2"; shift 2 ;;
        --solver) SOLVER="$2"; shift 2 ;;
        --topic-prefix) TOPIC_PREFIX="$2"; shift 2 ;;
        --rviz)   RVIZ="true"; shift ;;
        -h|--help) sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; exit 0 ;;
        *) echo "未知参数: $1 (用 --help 查看用法)" >&2; exit 2 ;;
    esac
done

# ── Python 解释器 ──
# venv 必须以 --system-site-packages 创建: rclpy / pinocchio 是针对系统解释器
# 编译的 C 扩展, 隔离的 venv 反而 import 不到。
VENV_PY="${REPO}/.venv/bin/python3"
if [[ ! -x "$VENV_PY" ]]; then
    echo "[错误] 未找到 venv 解释器: $VENV_PY" >&2
    echo "       请先创建:  python3 -m venv --system-site-packages .venv" >&2
    echo "       再安装依赖: .venv/bin/pip install -r requirements.txt" >&2
    exit 1
fi

# ── ROS 2 环境 ──
ROS_SETUP="/opt/ros/jazzy/setup.bash"
[[ -f "$ROS_SETUP" ]] || { echo "[错误] 未找到 ROS 2 安装: $ROS_SETUP" >&2; exit 1; }
# shellcheck disable=SC1091
# source 期间必须关闭 -u: ROS 2 的 setup.bash 会读取未定义的探测变量
# (如 AMENT_TRACE_SETUP_FILES), 在 set -u 下会直接中断脚本。
set +u
source "$ROS_SETUP"
set -u

# ── ament 资源前缀 (解析 URDF 中的 package:// 引用) ──
# 话题前缀经环境变量传给大屏服务端, 与求解器的 ROS 参数同源 ——
# 两端各写一份时不一致不会报错, 只是静默失联 (指令已下发但机器人不动)。
export ROBOT_TOPIC_PREFIX="$TOPIC_PREFIX"

export AMENT_PREFIX_PATH="${REPO}/resources/ament/install/humanoid_description:${AMENT_PREFIX_PATH:-/opt/ros/jazzy}"

# ── 求解器选择 ──
case "$SOLVER" in
    pink)     SOLVER_SCRIPT="core/solver/ik_node.py" ;;
    dual_arm) SOLVER_SCRIPT="core/ros/dual_arm_ik_node.py" ;;
    *) echo "[错误] 未知求解器: $SOLVER (可选: pink / dual_arm)" >&2; exit 2 ;;
esac

[[ -n "$URDF" ]] && URDF="$(readlink -f "$URDF")" || URDF="$(readlink -f "${REPO}/resources/humanoid/humanoid_29dof.urdf")"
[[ -n "$SRDF" ]] && SRDF="$(readlink -f "$SRDF")" || SRDF="$(readlink -f "${REPO}/resources/humanoid/humanoid_29dof.srdf")"

for f in "$URDF" "$SRDF" "$REPO/$SOLVER_SCRIPT"; do
    [[ -f "$f" ]] || { echo "[错误] 文件不存在: $f" >&2; exit 1; }
done

# ── 清理残留进程 ──
# 注意: 不能只靠 fuser 释放端口 —— IK 节点与 robot_state_publisher 并不监听
# 该端口, 残留实例会继续向同一批 ROS 话题发布, 互相回灌导致持续重解算。
# 必须按进程精确清理。
cleanup_stale() {
    fuser -k "${PORT}/tcp" >/dev/null 2>&1 || true
    # 用 [.] 技巧: pkill 的模式若匹配到执行本脚本的 shell 自身会自杀
    pkill -f "core/solver/ik_node[.]py"        >/dev/null 2>&1 || true
    pkill -f "core/ros/dual_arm_ik_node[.]py"  >/dev/null 2>&1 || true
    pkill -f "visualizer/web/server[.]py"      >/dev/null 2>&1 || true
    pkill -f "robot_state_publisher"           >/dev/null 2>&1 || true
    sleep 2
}
cleanup_stale

# 兜底校验: 确认没有遗留实例再继续, 避免叠加多个发布者。
#
# 注意 `|| true` 的必要性: 无匹配进程时 pgrep 退出码为 1, 在 set -o pipefail 下
# 该管道整体失败, 而赋���式会沿用命令替换的退出码 —— 于是 set -e 在这里直接把
# 脚本终止。该语句位于所有输出之前, 表现为脚本静默退出、什么都不打印。
STALE="$(pgrep -fc "core/solver/ik_node[.]py" 2>/dev/null | head -1 || true)"
STALE="${STALE//[^0-9]/}"   # 去掉任何非数字字符 (pgrep 无匹配时输出为空)
STALE="${STALE:-0}"
if [[ "$STALE" -gt 1 ]]; then
    echo "[警告] 仍检测到 ${STALE} 个 IK 节点实例, 强制清理"
    pkill -9 -f "core/solver/ik_node[.]py" >/dev/null 2>&1 || true
    sleep 2
fi

cat <<EOF
────────────────────────────────────────────────────────────
 人形机器人 Web 遥测大屏
   端口      : ${PORT}   →  http://localhost:${PORT}
   求解器    : ${SOLVER}  (${SOLVER_SCRIPT})
   话题前缀  : ${TOPIC_PREFIX}
   URDF      : ${URDF}
   SRDF      : ${SRDF}
   ament 前缀: ${REPO}/resources/ament
 按 Ctrl+C 停止
────────────────────────────────────────────────────────────
EOF

# ── 启动 ──
# 直接用 venv 解释器执行 launch 描述, 不用 `ros2 launch`:
#   `ros2` 是 /usr/bin/python3 的脚本, 其 shebang 决定了解释器, 内部再通过
#   sys.executable 拉起节点时会绕过 venv, 导致 rclpy/pinocchio 找不到。
#   直接执行 launch 文件则由 venv 解释器全程接管。
exec "$VENV_PY" "${REPO}/launch/telemetry_system.launch.py" \
     port:="${PORT}" \
     urdf:="${URDF}" \
     srdf:="${SRDF}" \
     solver_script:="${SOLVER_SCRIPT}" \
     rviz:="${RVIZ}" \
     topic_prefix:="${TOPIC_PREFIX}"
