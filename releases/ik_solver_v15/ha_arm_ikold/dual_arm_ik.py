# -*- coding: utf-8 -*-
"""
dual_arm_ik.py —— 腰 2 + 左臂 7 + 右臂 7 的 16 维联合逆运动学（模式一：左右手独立目标）

独立模块：只依赖 numpy 和 Pinocchio 4.x，不 import solver.py / solver_dls.py / ik_common.py /
ik_candidate.py / robot_model.py。位姿容差、限位裕度、跳变上限、姿态模式、刚体校验等判据的数值
从单臂代码复制（第 1 节，tests/test_dual_arm_ik.py 逐项核对两边一致）。

为什么在 16 维上联合求解
    两条手臂只通过腰耦合：J = [[J_Lw, J_L, 0], [J_Rw, 0, J_R]]。腰角必须同时让左手和右手可达；
    两臂各自带腰求解会得到两个不同的腰角，主从分解（先解一只手、锁腰再解另一只）是贪心的。
    所以状态只有一个 16 维向量，迭代和校验都在它上面做；搜索时利用“腰给定后两臂解耦”的结构
    生成种子（组合种子库、主从分解）。

一次 solve
    1. 解析目标：每只手是 4×4 位姿（URDF 根坐标系）/ "hold"（保持 q_prev 处的世界位姿）/
       "keep"（该臂关节角保持 q_prev）/ "free"（不约束，只回中）/ None（= "hold"：只给一只手目标时
       另一只手默认不动）。两只都是 None 报错；至少一只手有位姿或 hold 目标。
    2. 可达性预检（证明，不搜索）：目标位置对所有腰角都超出臂展 -> X_UNREACHABLE；两目标间距超过
       “两手臂根关节间距 + 两臂臂展”或没有同一个腰角让两手都在臂展内 -> NOT_JOINTLY_REACHABLE。
       状态都是 OUTSIDE_WORKSPACE，耗时 < 1 ms。
    3. 续算（路点跟踪，单臂 v9 规则）：每只手的目标都在 FK(q_prev) 附近（位置差 <= 0.1 × 臂展且姿态差
       <= 0.1 rad）时，先从 q_prev 在 q_prev ± continuation_max_step（0.15 rad）内求解、不做达标后的姿态整理，
       有解即采用（含 NEAR_LIMIT）。关节步长随手的移动量走，q_prev 已是解时原样返回（伺服循环里不漂）。
    4. 联合搜索（最多用预算的 70%）：种子依次为 q_prev -> q_ready -> 组合种子库 -> 随机扰动，
       第一个满足全部判据的解即返回。迭代限位 = 物理限位内缩 SAFE_MARGIN，
       joint_jump_policy="reject" 时再与 q_prev ± max_joint_jump_rad 取交集（跳变框；种子库也只取框内样本）。
       跳变框起作用时框内搜索到 55%，之后到 70% 不加框从 q_ready / 种子库再试几次：解在框内即采用，
       在框外 -> JOINT_JUMP_LIMIT（有解，但要转过上限）。
    5. 近限位补救（到 78%）：内缩限位里没解时，用物理限位从误差最小的终点续算，解出即 NEAR_LIMIT。
    6. 失败诊断（到 95%，不加跳变框）：左手单独、右手单独（各 9 轴）能否解出；都能解出时用主从分解拼成
       16 维种子再联合精化（能救回就返回成功），仍失败 -> NOT_JOINTLY_REACHABLE。诊断结论是搜索结果，
       不是证明；诊断没做完（预算用完）-> TIME_BUDGET_EXCEEDED。

单种子迭代（_run_seed）
    两只手的 6 维误差上下堆叠（full：log6 本体误差；tool_dir / tool_cone：位置 + 工具轴方向），
    带权阻尼最小二乘求步长（腰部逆度量按“肩 -> 目标距离 / 臂展”释放，常数与单臂相同）；
    零空间把关节拉向 q_ready；越过限位的关节钉在边界、去掉该列重解（SNS，最多 4 轮）；
    回溯线搜索；每 12 步检查一次进展，停滞即换种子；达标后在零空间整理姿态。

返回 (q, status, DualIKResult)：q 为 16 维（腰 2 | 左 7 | 右 7，按 layout 顺序），无解时为 None；
status 与单臂同一词表：SUCCESS / NEAR_LIMIT / APPROXIMATE / NO_VALID_POSE / OUTSIDE_WORKSPACE /
NO_RESULT；失败原因见 result.failure_reason（第 3 节）。
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pinocchio as pin

DUAL_SOLVER_VERSION = "dual-p2.1"

# =============================================================================
# 1. 判据与迭代常数：数值复制自单臂代码（不 import）
# =============================================================================
POS_TOL = 5e-4                               # ik_common.POS_TOL（m）
ORI_TOL = 5e-4                               # ik_common.ORI_TOL（rad）
SAFE_MARGIN = 0.02                           # ik_common.SAFE_MARGIN（rad）：限位裕度低于此值 -> NEAR_LIMIT
PERIODIC_MIN_SPAN_RAD = 2.0 * np.pi + 1e-6   # ik_common：跨度 >= 2π 的关节可做 2πk 等价平移
TOOL_AXIS_X = np.array([1.0, 0.0, 0.0])      # ik_common.TOOL_AXIS_X：TCP 局部 +X 为工具轴
TOOL_CONE_HALF_ANGLE = np.deg2rad(30.0)      # ik_common.TOOL_CONE_HALF_ANGLE
ORIENTATION_MODES = ("full", "tool_dir", "tool_cone")
RIGID_BOTTOM_ROW_TOL = 1e-9                  # ik_common.validate_transform4x4
RIGID_ROTATION_TOL = 1e-6
MAX_JOINT_JUMP_RAD = np.pi                   # solver.MAX_JOINT_JUMP_RAD：单关节相对 q_prev 超过半圈 = 绕大圈
JOINT_JUMP_POLICIES = ("reject", "prefer")   # solver.JOINT_JUMP_POLICIES
JUMP_COMPARE_TOL = 1e-9                      # solver.joint_jump_metrics 的比较余量
RNG_SEED = 120                               # solver.RNG_SEED
CONTINUATION_TOL = 0.10                      # solver_dls.WAIST_COMFORT_CONTINUATION_STEP：目标相对 FK(q_prev)
                                             # 位置差 <= 0.1 × 臂展（肩 -> TCP）且姿态差 <= 0.1 rad -> 续算（路点跟踪）
CONTINUATION_MAX_STEP = 0.15                 # solver.CONTINUATION_MAX_STEP：续算框 q_prev ± 0.15 rad（逐关节）
CONTINUATION_DLS_STEP_FRACTION = 0.5         # solver.CONTINUATION_DLS_STEP_FRACTION：续算时 DLS 单步 <= 0.5 × 续算框

MAX_STEP = 0.5                  # solver_dls.PIN_IK_MAX_STEP：单步关节 2-范数上限（rad）
DAMPING_MIN = 1e-6              # solver_dls.PIN_IK_DAMPING：λ² 下限
STALL_ITERS = 12                # solver_dls.PIN_IK_STALL_ITERS：每 12 步检查一次进展
STALL_RTOL = 0.025              # solver_dls.PIN_IK_STALL_RTOL：误差相对下降 < 2.5% -> 停滞
NULLSPACE_GAIN = 0.30           # 手臂关节拉向 q_ref 的零空间增益
WAIST_NULL_GAIN = 0.50          # 腰关节拉向 q_ref 的零空间增益
NULLSPACE_ERR_SCALE = 0.06      # 零空间权重 = (达标 1.5 / 未达标 1.0) / (1 + (|e| / 0.06)^2)
POSTURE_SETTLE_TOL = 1e-4       # 姿态整理：步长小于此值（rad）即收敛
SETTLE_COST_RTOL = 1e-3         # 姿态整理：回中代价相对下降 < 0.1% 记为慢步
SETTLE_PATIENCE = 3             # 连续 3 个慢步即收敛
MAX_SETTLE_ITERS = 50           # solver_dls.MAX_SETTLE_EXTRA_ITERS
WAIST_TASK_WEIGHT = 6.0         # 近目标：腰部步长代价为手臂的 6 倍（逆度量 1/6）
WAIST_TASK_WEIGHT_FAR = 1.0     # 远目标：腰部与手臂同价
WAIST_RELEASE_START = 0.75      # 肩 -> 目标距离 / 臂展 达到此比例开始放腰
WAIST_RELEASE_END = 0.94        # 达到此比例完全放开
WAIST_RETURN_RELEASE_RAD = 0.20  # 腰偏离 q_ref 时保持释放（便于退回）

# =============================================================================
# 2. 双臂求解自己的参数
# =============================================================================
CHAR_LENGTH = 0.5               # m：位置误差除以它，与转角同量纲
DAMPING_LM = 1e-3               # λ² = DAMPING_MIN + DAMPING_LM · |e|²（离目标远时加大阻尼）
MAX_ITER = 150                  # 单种子迭代上限
SNS_ROUNDS = 4                  # 限位饱和最多重解 4 轮
LINE_SEARCH_STEPS = 4           # 回溯线搜索：步长 1, 1/2, 1/4, 1/8
NULL_STEP_MAX = 0.2             # 零空间原始步长上限（rad）
PROGRESS_GATES = ((30, 0.1), (60, 0.02))   # 第 30 / 60 步时缩放误差 |e| 仍大于 0.1 / 0.02 -> 放弃该种子
                                # （0.1 约 5 cm 或 0.1 rad；统计：这样的种子几乎都不会收敛）
NULL_STEP_RATIO = 2.0           # 未达标时零空间步长 <= 2 × |e|（e 为缩放后的 12 维误差）
TIME_BUDGET_S = 0.1             # 一次 solve 的总时间上限（含失败诊断）
JOINT_PHASE_FRACTION = 0.70     # 联合搜索最多用到预算的 70%
JUMP_PROBE_FRACTION = 0.55      # 跳变框起作用时：框内搜索到 55%，之后到 70% 做一次不加框的探测
JUMP_PROBE_SEEDS = 4            # 探测用的种子库种子数（另加 q_ready）：找到的解若跳变超限 -> JOINT_JUMP_LIMIT
NEAR_LIMIT_PHASE_FRACTION = 0.78
DIAGNOSIS_PHASE_FRACTION = 0.95
DIAGNOSIS_LIBRARY_SEEDS = 3     # 失败诊断里每只手单独求解时用的种子库种子数（另加 q_ready）
MAX_SEEDS = 24                  # 联合搜索的种子个数上限（通常先碰到时间预算）
SEED_LIBRARY_SIZE = 16384       # 每只手臂的关节样本数；0 = 不用种子库（建库约 0.3 s，进程内缓存）
SEED_LIBRARY_SEEDS = 8          # 联合搜索里插入的组合种子个数
SEED_LIBRARY_RNG_SEED = 13      # 与单臂 SEED_LIBRARY_RNG_SEED 相同（库内容可复现）
WAIST_GRID = (6, 11)            # 组合种子库的腰网格（第 1 个腰关节 × 第 2 个腰关节）
LIBRARY_ROT_WEIGHT = 0.5        # 种子库距离² = (位置差 / CHAR_LENGTH)² + 0.5² × (3 − tr(ΔR))
LIBRARY_WAIST_WEIGHT = 0.05     # 组合种子打分：腰偏离 q_prev 的惩罚
RANDOM_SEED_SIGMA = 0.6         # 随机种子：q_ready 附近的正态扰动（rad）
RANDOM_SEED_MODE = "local"
REACH_GRID = (26, 51)           # 可达性预检的腰网格
LIBRARY_CHUNK = 256             # 种子库查询的分块大小（66 格 × 256 × 13 < OpenBLAS 多线程门槛 262144）

_SCALE6 = np.array([1.0 / CHAR_LENGTH] * 3 + [1.0] * 3)
_SCALE6_COL = _SCALE6[:, None].copy()
_INV_L = 1.0 / CHAR_LENGTH
_TWO_PI = 2.0 * np.pi
_PROGRESS_GATES = {int(n): float(x) ** 2 for n, x in PROGRESS_GATES}
_EYE = {6: np.eye(6), 12: np.eye(12)}

try:
    # numpy 的 solve gufunc（np.linalg.solve 内部调用的就是它）：12×12 一次约 2.5 µs，公开接口约 6 µs。
    # 矩阵 = J D Jᵀ + λ²I（λ² >= 1e-6）总是对称正定；私有接口不可用时退回 np.linalg.solve。
    from numpy.linalg import _umath_linalg as _np_umath_linalg
    _solve_vec = _np_umath_linalg.solve1
    if not np.allclose(_solve_vec(np.diag([2.0, 4.0]), np.array([2.0, 4.0])), [1.0, 1.0]):
        raise ImportError("unexpected solve1 behaviour")
except Exception:  # pragma: no cover
    _solve_vec = np.linalg.solve

# =============================================================================
# 3. 状态与失败原因
# =============================================================================
SUCCESS = "SUCCESS"
NEAR_LIMIT = "NEAR_LIMIT"
APPROXIMATE = "APPROXIMATE"
NO_VALID_POSE = "NO_VALID_POSE"
OUTSIDE_WORKSPACE = "OUTSIDE_WORKSPACE"
NO_RESULT = "NO_RESULT"
VALID_STATUSES = (SUCCESS, NEAR_LIMIT, APPROXIMATE)   # 这些状态伴随非空 q 返回

LEFT_UNREACHABLE = "LEFT_UNREACHABLE"            # 左手单独（腰 + 左臂）也解不出
RIGHT_UNREACHABLE = "RIGHT_UNREACHABLE"
BOTH_UNREACHABLE = "BOTH_UNREACHABLE"
NOT_JOINTLY_REACHABLE = "NOT_JOINTLY_REACHABLE"  # 两只手各自可解，但同时不可解（腰角冲突）
JOINT_JUMP_LIMIT = "JOINT_JUMP_LIMIT"            # 只找到跳变超限的解
TIME_BUDGET_EXCEEDED = "TIME_BUDGET_EXCEEDED"    # 预算用完，诊断未完成
SEEDS_EXHAUSTED = "SEEDS_EXHAUSTED"              # 种子用完（未做诊断）
SOLVER_BUSY = "SOLVER_BUSY"                      # 同一实例正被另一个线程调用

TARGET_KINDS = ("pose", "hold", "keep", "free")

DEFAULT_CONFIG = {
    "time_budget_s": TIME_BUDGET_S,         # 唯一的时间预算名
    "max_seeds": MAX_SEEDS,
    "max_iter": MAX_ITER,
    "joint_jump_policy": "reject",
    "continuation_local": True,             # 路点跟踪：先在 q_prev ± continuation_max_step 内求局部解（单臂 v9 规则）
    "continuation_max_step": CONTINUATION_MAX_STEP,
    "max_joint_jump_rad": MAX_JOINT_JUMP_RAD,   # None = 不限；标量；{关节名: rad 或 None}
    "seed_library_size": SEED_LIBRARY_SIZE,
    "seed_library_seeds": SEED_LIBRARY_SEEDS,
    "waist_grid": WAIST_GRID,
    "posture_settle": True,
    "diagnose_failures": True,
}

_LOCAL = pin.ReferenceFrame.LOCAL
_LWA = pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
_CJJ = pin.computeJointJacobians
_FK = pin.forwardKinematics
_UFP = pin.updateFramePlacement
_GFJ = pin.getFrameJacobian
_LOG6 = pin.log6
_JLOG6 = pin.Jlog6
_now = time.perf_counter


# =============================================================================
# 4. 关节布局与结果
# =============================================================================
@dataclass(frozen=True)
class DualArmLayout:
    """16 个活动关节的名字和顺序、两个 TCP 帧、臂展起算关节、q_ready。默认值按真机命名
    （与单臂 robot_model.py 相同，复制而非引用）。合成模型或别的机器人按需覆盖。"""

    waist_joints: tuple = ("Hip_Link1", "Body_Link")
    left_joints: tuple = tuple(f"LHand_Link{i}" for i in range(1, 8))
    right_joints: tuple = tuple(f"RHand_Link{i}" for i in range(1, 8))
    left_tcp: str = "left_l_dg_tcp"
    right_tcp: str = "right_l_dg_tcp"
    left_shoulder: str = "LHand_Link2"      # 单臂 s_joint：腰部释放用的臂展从这里起算
    right_shoulder: str = "RHand_Link2"
    q_ready_waist: tuple = (0.0, 0.0)
    q_ready_left: tuple = (-0.523599, -0.785398, 0.0, -0.698132, 0.0, 1.570796, 0.0)
    q_ready_right: tuple = (0.523599, -0.785398, 0.0, -0.698132, 0.0, 1.570796, 0.0)

    @property
    def joint_names(self):
        return list(self.waist_joints) + list(self.left_joints) + list(self.right_joints)

    @property
    def q_ready(self):
        return np.array(
            list(self.q_ready_waist) + list(self.q_ready_left) + list(self.q_ready_right), dtype=float
        )

    def key(self):
        return (
            tuple(self.waist_joints), tuple(self.left_joints), tuple(self.right_joints),
            self.left_tcp, self.right_tcp, self.left_shoulder, self.right_shoulder,
            tuple(np.round(self.q_ready, 12)),
        )


DEFAULT_LAYOUT = DualArmLayout()

WAIST = slice(0, 2)
LEFT = slice(2, 9)
RIGHT = slice(9, 16)
_ARM = {"left": LEFT, "right": RIGHT}


class DualIKResult(dict):
    """solve() 的诊断结果（dict，另有 status / failure_reason / failure_detail / per_seed 属性）。"""

    def __init__(self, info=None, status=NO_RESULT, failure_reason=None, failure_detail="", per_seed=None):
        super().__init__(info or {})
        self.status = status
        self.failure_reason = failure_reason
        self.failure_detail = failure_detail or ""
        self.per_seed = list(per_seed or [])
        self["status"] = status
        self["failure_reason"] = failure_reason

    @property
    def ok(self):
        return self.status in (SUCCESS, NEAR_LIMIT)

    def __str__(self):
        parts = [f"status={self.status}"]
        if self.failure_reason:
            parts.append(f"failure_reason={self.failure_reason}")
        for side in ("left", "right"):
            h = self.get(side) or {}
            if h.get("pos_err") is not None:
                parts.append(f"{side}={1e3 * h['pos_err']:.3f}mm/{1e3 * h['ori_err']:.3f}mrad")
        if "seeds_used" in self:
            parts.append(f"seeds={self['seeds_used']}")
        if "elapsed_s" in self:
            parts.append(f"t={1e3 * self['elapsed_s']:.1f}ms")
        return "DualIKResult(" + ", ".join(parts) + ")"

    __repr__ = __str__


# =============================================================================
# 5. 小工具
# =============================================================================
def _norm3(v):
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def _wrap(x):
    """角度差折叠到 [-π, π)。"""
    return (x + np.pi) % _TWO_PI - np.pi


def _smoothstep(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def validate_transform(T, name="T_target"):
    """4×4 刚体变换（或 pin.SE3）校验，返回 pin.SE3。规则与单臂 validate_transform4x4 相同。"""
    if isinstance(T, pin.SE3):
        T = T.homogeneous
    T = np.asarray(T, dtype=float)
    if T.shape != (4, 4):
        raise ValueError(f"{name} must be a 4x4 matrix, got shape {T.shape}")
    if not np.all(np.isfinite(T)):
        raise ValueError(f"{name} contains non-finite values")
    rot = T[:3, :3]
    if not np.allclose(T[3], [0.0, 0.0, 0.0, 1.0], atol=RIGID_BOTTOM_ROW_TOL, rtol=0.0):
        raise ValueError(f"{name} bottom row must be [0, 0, 0, 1]")
    if not (
        np.allclose(rot.T @ rot, np.eye(3), atol=RIGID_ROTATION_TOL, rtol=0.0)
        and np.isclose(np.linalg.det(rot), 1.0, atol=RIGID_ROTATION_TOL, rtol=0.0)
    ):
        raise ValueError(f"{name} rotation block is not a proper rotation matrix")
    return pin.SE3(rot.copy(), T[:3, 3].copy())


def _unit(v, name):
    v = np.asarray(v, dtype=float).reshape(-1)
    if v.shape != (3,) or not np.all(np.isfinite(v)):
        raise ValueError(f"{name} must be a finite 3-vector")
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        raise ValueError(f"{name} must be non-zero")
    return v / n


def _cone_half_angle(x):
    x = TOOL_CONE_HALF_ANGLE if x is None else float(x)
    if not np.isfinite(x) or x <= 0.0 or x >= np.pi:
        raise ValueError("cone_half_angle must be in radians and within (0, pi)")
    return x


def _per_side(value, is_pair):
    """单个值 -> (值, 值)；(左, 右) -> 原样。"""
    if is_pair(value):
        return tuple(value)
    return (value, value)


# =============================================================================
# 6. 运动学模型（Pinocchio）
# =============================================================================
class _Model:
    """Pinocchio 模型 + 16 个活动关节的索引、限位和角度向量 <-> Pinocchio q 转换。"""

    def __init__(self, urdf_path, layout):
        self.urdf_path = str(Path(urdf_path))
        model = pin.buildModelFromUrdf(self.urdf_path)
        self.model, self.data = model, model.createData()
        names = layout.joint_names
        if len(layout.waist_joints) != 2 or len(layout.left_joints) != 7 or len(layout.right_joints) != 7:
            raise ValueError("layout must list 2 waist joints, 7 left-arm joints and 7 right-arm joints")
        if len(set(names)) != 16:
            raise ValueError("layout joint names must be unique")
        self.names = names
        self.jids = []
        for nm in names:
            if not model.existJointName(nm):
                raise ValueError(f"joint {nm!r} not found in URDF")
            self.jids.append(int(model.getJointId(nm)))
        idx_q, idx_v, nqs, revolute = [], [], [], []
        lo, hi = np.full(16, -np.inf), np.full(16, np.inf)
        for i, jid in enumerate(self.jids):
            j = model.joints[jid]
            if j.nv != 1 or j.nq not in (1, 2):
                raise ValueError(f"joint {names[i]!r} must be a 1-DoF joint (got nq={j.nq}, nv={j.nv})")
            short = j.shortname()
            idx_q.append(j.idx_q)
            idx_v.append(j.idx_v)
            nqs.append(j.nq)
            revolute.append(short.startswith("JointModelR"))
            if j.nq == 1:
                a, b = float(model.lowerPositionLimit[j.idx_q]), float(model.upperPositionLimit[j.idx_q])
                lo[i] = a if abs(a) < 1e10 else -np.inf
                hi[i] = b if abs(b) < 1e10 else np.inf
        self.lower, self.upper = lo, hi
        if np.any(self.lower > self.upper):
            raise ValueError("joint lower limit above upper limit")
        idx_q, nqs = np.array(idx_q), np.array(nqs)
        lin = np.flatnonzero(nqs == 1)
        ub = np.flatnonzero(nqs == 2)
        self.ai_lin, self.qi_lin = lin, idx_q[lin]
        self.ai_ub, self.qi_cos, self.qi_sin = ub, idx_q[ub], idx_q[ub] + 1
        self.has_ub = ub.size > 0
        self.vcols = np.array(idx_v)
        self.vcols_identity = model.nv == 16 and np.array_equal(self.vcols, np.arange(16))
        self.revolute = np.array(revolute)
        # 2πk 等价平移：跨度 >= 2π 的转动关节；跳变与回中按短弧计的只有两侧无界的 continuous 关节
        self.periodic_capable = self.revolute & ((self.upper - self.lower) >= PERIODIC_MIN_SPAN_RAD)
        self.unbounded = ~np.isfinite(self.lower) & ~np.isfinite(self.upper)
        self.template = pin.neutral(model)

        self.fid = {"left": self._frame_id(layout.left_tcp, "left TCP"),
                    "right": self._frame_id(layout.right_tcp, "right TCP")}
        sup = {s: [int(x) for x in model.supports[int(model.frames[self.fid[s]].parentJoint)]]
               for s in ("left", "right")}
        waist_ids, left_ids, right_ids = self.jids[0:2], self.jids[2:9], self.jids[9:16]
        for w, nm in zip(waist_ids, layout.waist_joints):
            if w not in sup["left"] or w not in sup["right"]:
                raise ValueError(f"waist joint {nm!r} must be an ancestor of both TCP frames")
        for side, ids, other in (("left", left_ids, "right"), ("right", right_ids, "left")):
            pos = [sup[side].index(j) if j in sup[side] else -1 for j in ids]
            if min(pos) < 0:
                raise ValueError(f"{side}-arm joints must all lie on the chain to the {side} TCP")
            if any(j in sup[other] for j in ids):
                raise ValueError(f"{side}-arm joints must not lie on the {other}-arm chain")
            if pos != sorted(pos):
                raise ValueError(f"{side}-arm joints must be listed from shoulder to wrist")
            if max(sup[side].index(w) for w in waist_ids) > pos[0]:
                raise ValueError("both waist joints must come before the arm joints")
        # 两条 TCP 链上不在 16 个活动关节里的可动关节（取 joint_state / neutral 的值）：有的话，
        # 手臂根关节位置与臂展上界就依赖这些关节，可达性预检不再是证明 -> 关掉预检（种子库只是启发式，照用）
        active = set(self.jids)
        self.extra_chain_joints = sorted({str(model.names[j]) for side in ("left", "right") for j in sup[side]
                                          if j != 0 and j not in active})
        # 躯干帧：两个腰关节里更靠近末端的那个关节坐标系；手臂相对它的位姿与腰角无关
        self.torso_jid = max(waist_ids, key=lambda j: sup["left"].index(j))
        self.arm_root_jid = {"left": left_ids[0], "right": right_ids[0]}
        self.shoulder_jid = {}
        self.reach_shoulder = {}
        self.reach_root = {}
        for side, nm, ids in (("left", layout.left_shoulder, left_ids), ("right", layout.right_shoulder, right_ids)):
            if not model.existJointName(nm) or int(model.getJointId(nm)) not in ids:
                raise ValueError(f"{side} shoulder joint {nm!r} must be one of the {side}-arm joints")
            sj = int(model.getJointId(nm))
            self.shoulder_jid[side] = sj
            self.reach_shoulder[side] = self._reach_after(sj, self.fid[side])
            self.reach_root[side] = self._reach_after(self.arm_root_jid[side], self.fid[side])

    def _frame_id(self, name, what):
        m = self.model
        if m.existFrame(name, pin.FrameType.BODY):
            return int(m.getFrameId(name, pin.FrameType.BODY))
        if m.existFrame(name):
            return int(m.getFrameId(name))
        raise ValueError(f"{what} frame {name!r} not found in URDF")

    def _reach_after(self, start_jid, fid):
        """start 关节原点 -> TCP 距离的上界：之后各关节原点偏移长度 + TCP 帧偏移长度（单臂同一定义）。"""
        m = self.model
        frame = m.frames[fid]
        reach = float(np.linalg.norm(frame.placement.translation))
        jid = int(frame.parentJoint)
        while jid > 0 and jid != start_jid:
            reach += float(np.linalg.norm(m.jointPlacements[jid].translation))
            jid = int(m.parents[jid])
        return reach

    def full_q(self, q, template):
        qf = template.copy()
        qf[self.qi_lin] = q[self.ai_lin]
        if self.has_ub:
            a = q[self.ai_ub]
            qf[self.qi_cos] = np.cos(a)
            qf[self.qi_sin] = np.sin(a)
        return qf

    def fk(self, q, template):
        _FK(self.model, self.data, self.full_q(q, template))
        return {s: pin.SE3(_UFP(self.model, self.data, self.fid[s])) for s in ("left", "right")}

    def torso(self):
        return pin.SE3(self.data.oMi[self.torso_jid])


# =============================================================================
# 7. 组合种子库
# =============================================================================
class _ArmLibrary:
    """一只手臂的关节样本 + TCP 相对躯干帧的位姿（与腰角无关）。

    距离 d² = |Δp|² / CHAR_LENGTH² + LIBRARY_ROT_WEIGHT² · (3 − tr(R_sᵀR_t))，展开成矩阵乘法：
    d²[g, k] = G[g] · F[k] + |p_t,g|² / CHAR_LENGTH²，F = [p_s, vec(R_s), |p_s|²/L² + 3w²]，
    G = [−2 p_t / L², −w² vec(R_t), 1]。样本按 LIBRARY_CHUNK 分块做批量乘法：每块的规模低于 OpenBLAS
    的多线程门槛，查询不会起 BLAS 线程（多进程争用 CPU 时多线程 GEMM 会慢一个数量级）。"""

    def __init__(self, Q, P, Rf, a, b):
        self.Q = Q
        self.QT = np.ascontiguousarray(Q.T)       # 按关节取列（跳变框过滤用）
        self.a, self.b = a, b                     # 采样范围（无界关节为 ±π）
        w2 = LIBRARY_ROT_WEIGHT ** 2
        p2 = np.einsum("ij,ij->i", P, P)
        F_T = np.hstack([P, Rf, (p2 * _INV_L ** 2 + 3.0 * w2)[:, None]]).T      # (13, N)
        n, ch = F_T.shape[1], LIBRARY_CHUNK
        pad = (-n) % ch
        if pad:                                   # 补齐到整块：补的列距离极大，不会被选中
            fill = np.zeros((13, pad))
            fill[12] = 1e30
            F_T = np.hstack([F_T, fill])
        self.F3 = np.ascontiguousarray(F_T.reshape(13, -1, ch).transpose(1, 0, 2))   # (块数, 13, ch)
        self._offs = (np.arange(self.F3.shape[0]) * ch)[:, None]

    def inside(self, lo, hi, periodic, center):
        """跳变框 [lo, hi]（7 维）内的样本掩码。None = 框不比采样范围紧（全部样本可用）。
        周期关节按 2π 折叠到 center（q_prev）附近比较。"""
        tight = (lo > self.a + 1e-12) | (hi < self.b - 1e-12)
        if not tight.any():
            return None
        QT = self.QT
        ok = np.ones(QT.shape[1], dtype=bool)
        for j in np.flatnonzero(tight):
            v = QT[j]
            if periodic[j]:
                v = center[j] + (v - center[j] + np.pi) % _TWO_PI - np.pi
            ok &= (v >= lo[j] - 1e-12) & (v <= hi[j] + 1e-12)
        return ok

    def best(self, T, Rg, og, mask=None, out=None):
        """目标换到各腰网格点的躯干坐标系，每格取距离最近的样本。返回 (每格距离, 每格样本)。

        mask（跳变框内的样本）：每格在“各块最近样本”里取框内最近的一个（块是 256 个随机样本，块内最近的
        在框外时放弃这一块；近似但几乎不增加开销）；没有框内候选的格点距离为 inf。
        out = 调用方的距离缓冲（免去每次分配 8 MB）。"""
        ng = len(og)
        cols = np.arange(ng)
        pt = np.einsum("gji,gj->gi", Rg, T.translation[None, :] - og)
        Rt = np.einsum("gji,jk->gik", Rg, T.rotation).reshape(ng, 9)
        G = np.empty((ng, 13))
        G[:, :3] = (-2.0 * _INV_L ** 2) * pt
        G[:, 3:12] = (-(LIBRARY_ROT_WEIGHT ** 2)) * Rt
        G[:, 12] = 1.0
        F3 = self.F3
        if out is not None and out.shape == (F3.shape[0], ng, LIBRARY_CHUNK):
            S = np.matmul(G, F3, out=out)                            # (块数, ng, ch)
        else:
            S = np.matmul(G, F3)
        kmin = S.argmin(axis=2)
        vmin = np.take_along_axis(S, kmin[:, :, None], axis=2)[:, :, 0]
        if mask is not None:
            kg = np.minimum(kmin + self._offs, len(mask) - 1)
            vmin = np.where(mask[kg], vmin, np.inf)
        c = vmin.argmin(axis=0)
        k = np.minimum(c * LIBRARY_CHUNK + kmin[c, cols], len(self.Q) - 1)
        d2 = vmin[c, cols] + np.einsum("gi,gi->g", pt, pt) * _INV_L ** 2
        return np.sqrt(np.maximum(d2, 0.0)), self.Q[k]


class _Library:
    """两只手臂各一个种子库 + 腰网格（种子用）+ 细腰网格上的手臂根关节位置（可达性预检用）。"""

    def __init__(self, mdl, q_ready, size, waist_grid, lo_s, hi_s):
        rng = np.random.default_rng(SEED_LIBRARY_RNG_SEED)
        tmpl = mdl.template
        self.arms = {}
        if size > 0:
            for side in ("left", "right"):
                sl = _ARM[side]
                a = np.where(np.isfinite(lo_s[sl]), lo_s[sl], -np.pi)
                b = np.where(np.isfinite(hi_s[sl]), hi_s[sl], np.pi)
                Q = rng.uniform(a, b, (size, 7))
                P = np.empty((size, 3))
                Rf = np.empty((size, 9))
                q = q_ready.copy()
                fid = mdl.fid[side]
                for k in range(size):
                    q[sl] = Q[k]
                    _FK(mdl.model, mdl.data, mdl.full_q(q, tmpl))
                    rel = mdl.data.oMi[mdl.torso_jid].actInv(_UFP(mdl.model, mdl.data, fid))
                    P[k] = rel.translation
                    Rf[k] = rel.rotation.reshape(9)
                self.arms[side] = _ArmLibrary(Q, P, Rf, a, b)
        self.grid_w = self._grid(lo_s, hi_s, waist_grid)
        self.grid_R, self.grid_o = self._torso_poses(mdl, q_ready, self.grid_w)
        # 可达性预检：细网格上手臂根关节原点的位置；网格离散带来的误差用相邻格点的最大位移兜底
        nr = REACH_GRID
        fine = self._grid(mdl.lower, mdl.upper, nr)
        self.root_pos = {}
        self.root_slack = {}
        q = q_ready.copy()
        pos = {"left": np.empty((len(fine), 3)), "right": np.empty((len(fine), 3))}
        for g, w in enumerate(fine):
            q[WAIST] = w
            _FK(mdl.model, mdl.data, mdl.full_q(q, tmpl))
            for side in ("left", "right"):
                pos[side][g] = mdl.data.oMi[mdl.arm_root_jid[side]].translation
        for side in ("left", "right"):
            p = pos[side].reshape(nr[0], nr[1], 3)
            steps = [np.linalg.norm(p[1:, :] - p[:-1, :], axis=2).max() if nr[0] > 1 else 0.0,
                     np.linalg.norm(p[:, 1:] - p[:, :-1], axis=2).max() if nr[1] > 1 else 0.0]
            self.root_pos[side] = pos[side]
            self.root_slack[side] = 0.5 * math.hypot(*steps) + 1e-6
        # 两个手臂根关节都固连在躯干上：间距与腰角无关（两手间距上界 = 根间距 + 两臂臂展，精确、无离散误差）
        self.root_gap = float(np.linalg.norm(pos["left"][0] - pos["right"][0]))

    @staticmethod
    def _grid(lo, hi, counts):
        axes = []
        for i in range(2):
            a = lo[i] if np.isfinite(lo[i]) else -np.pi
            b = hi[i] if np.isfinite(hi[i]) else np.pi
            n = int(counts[i])
            if np.isfinite(lo[i]) and np.isfinite(hi[i]):
                axes.append(np.linspace(a, b, n))
            else:
                axes.append(np.linspace(a, b, n, endpoint=False))
        return np.array([(x, y) for x in axes[0] for y in axes[1]], dtype=float)

    @staticmethod
    def _torso_poses(mdl, q_ready, ws):
        q = q_ready.copy()
        R = np.empty((len(ws), 3, 3))
        o = np.empty((len(ws), 3))
        for g, w in enumerate(ws):
            q[WAIST] = w
            _FK(mdl.model, mdl.data, mdl.full_q(q, mdl.template))
            M = mdl.data.oMi[mdl.torso_jid]
            R[g] = M.rotation
            o[g] = M.translation
        return R, o

    def outside_reach(self, side, p, reach):
        """目标位置对细网格上所有腰角都超出臂展（扣除网格离散误差后）-> True。"""
        d = np.linalg.norm(self.root_pos[side] - p[None, :], axis=1).min()
        return bool(d - self.root_slack[side] > reach + POS_TOL)

    def jointly_outside(self, p_left, reach_left, p_right, reach_right):
        """没有任何腰角能让两只手的目标位置同时落在臂展内 -> True：两目标间距超过 根间距 + 两臂臂展，
        或同一网格点上两手不能同时在臂展内（扣除离散误差）。"""
        if _norm3(p_left - p_right) > self.root_gap + reach_left + reach_right + 2.0 * POS_TOL:
            return True
        dl = np.linalg.norm(self.root_pos["left"] - p_left[None, :], axis=1) - self.root_slack["left"]
        dr = np.linalg.norm(self.root_pos["right"] - p_right[None, :], axis=1) - self.root_slack["right"]
        return not bool(np.any((dl <= reach_left + POS_TOL) & (dr <= reach_right + POS_TOL)))


_LIBRARY_CACHE = {}
_LIBRARY_CACHE_MAX = 8
_LIBRARY_CACHE_LOCK = threading.Lock()


def _library_for(mdl, layout, q_ready, size, waist_grid, lo_s, hi_s):
    path = Path(mdl.urdf_path)
    try:
        stamp = (str(path.resolve()), path.stat().st_mtime_ns, path.stat().st_size)
    except OSError:
        stamp = (str(path), None, None)
    key = (stamp, layout.key(), int(size), tuple(int(x) for x in waist_grid),
           tuple(np.round(lo_s, 12)), tuple(np.round(hi_s, 12)), SEED_LIBRARY_RNG_SEED)
    with _LIBRARY_CACHE_LOCK:
        lib = _LIBRARY_CACHE.get(key)
    if lib is None:
        lib = _Library(mdl, q_ready, size, waist_grid, lo_s, hi_s)
        with _LIBRARY_CACHE_LOCK:
            if len(_LIBRARY_CACHE) >= _LIBRARY_CACHE_MAX:
                _LIBRARY_CACHE.pop(next(iter(_LIBRARY_CACHE)))
            _LIBRARY_CACHE[key] = lib
    return lib


# =============================================================================
# 8. 一次求解的任务与上下文
# =============================================================================
class _Hand:
    """一只手的任务：目标位姿 + 姿态模式；row 为它在 12 行任务里的起始行。"""

    __slots__ = ("side", "kind", "fid", "mode", "T", "p", "axis", "dt", "half", "ch", "sh", "perp",
                 "row", "sjid", "reach", "ref_release")


class _Ctx:
    """一次求解（或诊断中的一个子问题）的迭代设置。"""

    __slots__ = ("lo", "hi", "dmask", "gains", "q_ref", "periodic", "pidx", "settle", "max_iter",
                 "waist_free", "template", "m", "hands", "box_lo", "box_hi", "max_step")


# =============================================================================
# 9. 求解器
# =============================================================================
class DualArmIKSolver:
    """腰 2 + 左臂 7 + 右臂 7 的 16 维联合逆运动学（模式一：左右手独立目标）。"""

    def __init__(self, urdf, layout=None, config=None, verbose=False):
        self.layout = DEFAULT_LAYOUT if layout is None else layout
        if not isinstance(self.layout, DualArmLayout):
            raise TypeError("layout must be a DualArmLayout")
        self.verbose = bool(verbose)
        self.config = self._parse_config(config)
        self._m = _Model(urdf, self.layout)
        self.active_joint_names = list(self._m.names)
        self.lower = self._m.lower.copy()
        self.upper = self._m.upper.copy()
        self.q_ready = self.layout.q_ready
        if np.any(self.q_ready < self.lower - 1e-9) or np.any(self.q_ready > self.upper + 1e-9):
            raise ValueError("layout q_ready is outside the joint limits")
        self.q_prev = self.q_ready.copy()
        self._jump = self._resolve_jump_limits(self.config["max_joint_jump_rad"])
        finite_lo, finite_hi = np.isfinite(self.lower), np.isfinite(self.upper)
        m = SAFE_MARGIN + 1e-9
        self._lo_s = np.where(finite_lo, self.lower + m, self.lower)
        self._hi_s = np.where(finite_hi, self.upper - m, self.upper)
        narrow = self._lo_s > self._hi_s           # 跨度 < 2·SAFE_MARGIN 的关节（两侧必有限）：钉在中点
        if np.any(narrow):
            mid = 0.5 * (self.lower[narrow] + self.upper[narrow])
            self._lo_s[narrow] = mid
            self._hi_s[narrow] = mid
        self._gains = np.r_[np.full(2, WAIST_NULL_GAIN), np.full(14, NULLSPACE_GAIN)]
        self._lib = _library_for(self._m, self.layout, self.q_ready, self.config["seed_library_size"],
                                 self.config["waist_grid"], self._lo_s, self._hi_s)
        self._lock = threading.Lock()
        self._ebuf = np.empty(12)
        self._precheck = not self._m.extra_chain_joints     # 可达性预检只在 TCP 链全是活动关节时才是证明
        # 种子库查询的距离缓冲（每个实例一份：库在实例间共享，缓冲不能共享）
        arm = next(iter(self._lib.arms.values()), None)
        self._lib_buf = (None if arm is None else
                         np.empty((arm.F3.shape[0], len(self._lib.grid_w), LIBRARY_CHUNK)))

    # ------------------------------------------------------------------ 配置
    @staticmethod
    def _parse_config(config):
        cfg = dict(DEFAULT_CONFIG)
        if config:
            unknown = set(config) - set(DEFAULT_CONFIG)
            if unknown:
                raise ValueError(f"unknown config keys: {sorted(unknown)}")
            cfg.update(config)
        b = cfg["time_budget_s"]
        if b is not None and (not np.isfinite(float(b)) or float(b) <= 0.0):
            raise ValueError("time_budget_s must be positive (or None = no limit)")
        for k in ("max_seeds", "max_iter"):
            if int(cfg[k]) < 1:
                raise ValueError(f"{k} must be >= 1")
            cfg[k] = int(cfg[k])
        for k in ("seed_library_size", "seed_library_seeds"):
            if int(cfg[k]) < 0:
                raise ValueError(f"{k} must be >= 0")
            cfg[k] = int(cfg[k])
        if cfg["joint_jump_policy"] not in JOINT_JUMP_POLICIES:
            raise ValueError(f"joint_jump_policy must be one of {JOINT_JUMP_POLICIES}")
        g = tuple(int(x) for x in cfg["waist_grid"])
        if len(g) != 2 or min(g) < 1:
            raise ValueError("waist_grid must be two positive counts")
        cfg["waist_grid"] = g
        cfg["posture_settle"] = bool(cfg["posture_settle"])
        cfg["continuation_local"] = bool(cfg["continuation_local"])
        h = float(cfg["continuation_max_step"])
        if not np.isfinite(h) or h <= 0.0:
            raise ValueError("continuation_max_step must be a positive finite angle")
        cfg["continuation_max_step"] = h
        cfg["diagnose_failures"] = bool(cfg["diagnose_failures"])
        return cfg

    def _resolve_jump_limits(self, value):
        """None = 不限；正标量 = 所有关节同一上限；{关节名: rad 或 None}，未列出的取 π（与单臂相同）。"""
        if value is None:
            return None
        names = self.active_joint_names
        if isinstance(value, Mapping):
            unknown = set(value) - set(names)
            if unknown:
                raise ValueError(f"max_joint_jump_rad has unknown joints: {sorted(unknown)}")
            lim = np.array([np.inf if value.get(n, MAX_JOINT_JUMP_RAD) is None
                            else float(value.get(n, MAX_JOINT_JUMP_RAD)) for n in names])
        else:
            lim = np.full(16, float(value))
        if np.any(np.isnan(lim)) or np.any(lim <= 0.0):
            raise ValueError("max_joint_jump_rad must be positive (or None)")
        return lim

    # ------------------------------------------------------------------ 公共小接口
    def fk(self, q, joint_state=None):
        """16 维关节角 -> (T_left, T_right)，4×4，URDF 根坐标系。"""
        q = self._check_q(q, "q")
        poses = self._m.fk(q, self._template(joint_state))
        return poses["left"].homogeneous, poses["right"].homogeneous

    def reset_prev(self, q=None):
        self.q_prev = self.q_ready.copy() if q is None else self._check_q(q, "q")

    @staticmethod
    def _check_q(q, name):
        q = np.asarray(q, dtype=float).reshape(-1)
        if q.shape != (16,):
            raise ValueError(f"{name} must have 16 entries (waist 2 | left 7 | right 7), got {q.shape}")
        if not np.all(np.isfinite(q)):
            raise ValueError(f"{name} contains non-finite values")
        return q.copy()

    def _template(self, joint_state):
        """非活动关节（头、手指等）取 joint_state 里的值，其余取 Pinocchio neutral。"""
        if not joint_state:
            return self._m.template
        model = self._m.model
        tmpl = self._m.template.copy()
        active = set(self.active_joint_names)
        for nm, val in joint_state.items():
            if not model.existJointName(nm):
                raise ValueError(f"joint_state has unknown joint {nm!r}")
            if nm in active:
                continue
            j = model.joints[int(model.getJointId(nm))]
            if j.nq == 1:
                tmpl[j.idx_q] = float(val)
            elif j.nq == 2 and j.nv == 1:
                tmpl[j.idx_q] = math.cos(float(val))
                tmpl[j.idx_q + 1] = math.sin(float(val))
            else:
                raise ValueError(f"joint_state: joint {nm!r} is not a 1-DoF joint")
        return tmpl

    # ------------------------------------------------------------------ solve
    def solve(self, T_left=None, T_right=None, q_prev=None, update_prev=True, verbose=False,
              allow_approximate=False, orientation_mode="full", tool_axis=None, tool_cone_half_angle=None,
              *, joint_state=None, time_budget_s=None):
        """左右手独立目标的 16 维联合求解。

        T_left / T_right：4×4 位姿（或 pin.SE3，URDF 根坐标系）/ "hold" / "keep" / "free" / None。
        None = "hold"（该手保持 q_prev 处的世界位姿）；"free" = 不约束、只回中。两只都是 None 报错
        （ValueError；要两手都不动请显式传 ("hold", "hold")）。
        orientation_mode / tool_axis / tool_cone_half_angle：一个值（两手相同）或 (左, 右)。
        q_prev：16 维；不给时取 joint_state 里的 16 个活动关节（齐全时），否则取上次成功的解（初始 q_ready）。
        time_budget_s：本次的时间上限（不给时取 config["time_budget_s"]）。
        返回 (q 或 None, status, DualIKResult)。result["regime"]：'continuation'（目标在 FK(q_prev) 附近，
        按路点跟踪先求局部解）/ 'reposition'；result["seed_kind"]：采用的解来自哪类种子。
        """
        t0 = _now()
        if not self._lock.acquire(blocking=False):
            res = DualIKResult({"version": DUAL_SOLVER_VERSION}, NO_VALID_POSE, SOLVER_BUSY,
                               "another thread is using this solver instance")
            return None, NO_VALID_POSE, res
        try:
            return self._solve_impl(t0, T_left, T_right, q_prev, update_prev, verbose or self.verbose,
                                    allow_approximate, orientation_mode, tool_axis, tool_cone_half_angle,
                                    joint_state, time_budget_s)
        finally:
            self._lock.release()

    def _solve_impl(self, t0, T_left, T_right, q_prev, update_prev, verbose, allow_approximate,
                    orientation_mode, tool_axis, tool_cone_half_angle, joint_state, time_budget_s):
        cfg = self.config
        budget = cfg["time_budget_s"] if time_budget_s is None else float(time_budget_s)
        if budget is not None and (not np.isfinite(budget) or budget <= 0.0):
            raise ValueError("time_budget_s must be positive")
        end = math.inf if budget is None else t0 + budget
        span = math.inf if budget is None else budget
        dl_joint = t0 + JOINT_PHASE_FRACTION * span
        dl_near = t0 + NEAR_LIMIT_PHASE_FRACTION * span
        dl_diag = t0 + DIAGNOSIS_PHASE_FRACTION * span

        template = self._template(joint_state)
        q_prev_eff = self._resolve_prev(q_prev, joint_state)
        hands, kinds = self._parse_targets(T_left, T_right, q_prev_eff, template, orientation_mode,
                                           tool_axis, tool_cone_half_angle)
        info = {"version": DUAL_SOLVER_VERSION, "targets": dict(kinds), "time_budget_s": budget}

        # 可达性预检（只看位置，按臂展上界判断：判为不可达的一定不可达）
        reach = self._m.reach_root
        pre = self._precheck
        out = [h.side for h in hands if pre and self._lib.outside_reach(h.side, h.p, reach[h.side])]
        reason = None
        if out:
            reason = BOTH_UNREACHABLE if len(out) == 2 else (LEFT_UNREACHABLE if out[0] == "left"
                                                            else RIGHT_UNREACHABLE)
            detail = f"target position beyond arm reach for every waist angle: {', '.join(out)}"
        elif pre and len(hands) == 2 and self._lib.jointly_outside(hands[0].p, reach["left"], hands[1].p,
                                                                    reach["right"]):
            reason = NOT_JOINTLY_REACHABLE
            detail = "each target is within reach for some waist angle, but no waist angle puts both within reach"
        if reason is not None:
            info.update(elapsed_s=_now() - t0, seeds_used=0, iterations=0)
            res = DualIKResult(info, OUTSIDE_WORKSPACE, reason, detail)
            self._log(verbose, res)
            return None, OUTSIDE_WORKSPACE, res

        ctx = self._make_ctx(hands, kinds, q_prev_eff, template, shrink=True)
        per_seed = []
        stats = {"iterations": 0, "seeds": 0}
        best = {"err2": math.inf, "q": None}       # 误差最小的终点（近限位补救 / APPROXIMATE 用）
        fallback = []                               # 位姿达标但不完全合格的候选（NEAR_LIMIT / 跳变超限）

        def attempt(seed, kind, c, deadline):
            ts = _now()
            q, ok, it, why, errs, err2 = self._run_seed(seed, c, deadline)
            stats["iterations"] += it
            stats["seeds"] += 1
            row = {"kind": kind, "iterations": it, "ok": bool(ok), "reason": why,
                   "elapsed_ms": 1e3 * (_now() - ts)}
            for h, (pe, oe) in zip(c.hands, errs):
                row[f"{h.side}_pos_err"] = pe
                row[f"{h.side}_ori_err"] = oe
            per_seed.append(row)
            if c is ctx and err2 < best["err2"]:
                best.update(err2=err2, q=q.copy())
            return q, ok, why

        # 3. 续算（路点跟踪，单臂 v9 规则）：目标就在 FK(q_prev) 附近时，先从 q_prev 在续算框
        #    q_prev ± continuation_max_step 内求解——零空间回中不会一步漂到远处的姿态；有局部解（含 NEAR_LIMIT）
        #    即采用，否则进入联合搜索
        found = None
        regime = "reposition"
        if cfg["continuation_local"] and self._near_prev(hands, q_prev_eff, template):
            regime = "continuation"
            ctx_c = self._make_ctx(hands, kinds, q_prev_eff, template, shrink=True, cont=cfg["continuation_max_step"])
            ctx_c.settle = False    # 不做达标后的姿态整理：回中只来自收敛过程中的零空间步（<= 2|e|），漂移与手的移动量成正比
            seed = np.minimum(np.maximum(q_prev_eff, ctx_c.lo), ctx_c.hi)
            q, ok, _ = attempt(seed, "continuation", ctx_c, dl_joint)
            if ok:
                cand = self._candidate(q, hands, ctx_c, q_prev_eff, "continuation")
                if cand["near_ok"] and cand["jump_ok"]:
                    found = cand
        info["regime"] = regime

        # 4. 联合搜索（跳变框内）
        boxed = bool(ctx.box_lo.any() or ctx.box_hi.any())
        dl_box = t0 + JUMP_PROBE_FRACTION * span if boxed else dl_joint
        for kind, seed in ([] if found is not None else self._seeds(ctx, hands, kinds, q_prev_eff)):
            if stats["seeds"] >= cfg["max_seeds"] or _now() >= dl_box:
                break
            q, ok, _ = attempt(seed, kind, ctx, dl_box)
            if ok:
                cand = self._candidate(q, hands, ctx, q_prev_eff, kind)
                if cand["acceptable"]:
                    found = cand
                    break
                fallback.append(cand)
                if cfg["joint_jump_policy"] == "prefer" and cand["near_ok"] and not cand["jump_ok"]:
                    continue

        # 4b. 跳变框探测：框内没找到时，不加框从 q_ready / 种子库再试几次——找到的解在框内即采用，
        #     在框外就说明“有解但跳变超限”（JOINT_JUMP_LIMIT，不必再做可达性诊断）
        jump_found = False
        if found is None and boxed and _now() < dl_joint:
            ctx_u = self._make_ctx(hands, kinds, q_prev_eff, template, shrink=True, jump_box=False)
            for kind, seed in self._probe_seeds(ctx_u, hands, kinds, q_prev_eff):
                if _now() >= dl_joint:
                    break
                q, ok, _ = attempt(seed, kind, ctx_u, dl_joint)
                if ok:
                    cand = self._candidate(q, hands, ctx_u, q_prev_eff, kind)
                    if cand["acceptable"]:
                        found = cand
                        break
                    fallback.append(cand)
                    if cand["near_ok"] and not cand["jump_ok"]:
                        jump_found = True
                        break

        # 5. 近限位补救：物理限位（仍带跳变框）里从误差最小的终点续算
        if found is None and best["q"] is not None and _now() < dl_near:
            ctx_n = self._make_ctx(hands, kinds, q_prev_eff, template, shrink=False)
            q, ok, _ = attempt(best["q"], "near_limit", ctx_n, dl_near)
            if ok:
                cand = self._candidate(q, hands, ctx_n, q_prev_eff, "near_limit")
                if cand["acceptable"]:
                    found = cand
                else:
                    fallback.append(cand)

        diagnosis = None
        box_bound = False
        if found is None:
            usable = [c for c in fallback if c["near_ok"] and (c["jump_ok"] or cfg["joint_jump_policy"] == "prefer")]
            if best["q"] is not None:
                # 误差最小的终点贴在跳变框边上：卡住它的是 max_joint_jump_rad，不必再做可达性诊断
                qb = best["q"]
                box_bound = bool(np.any(ctx.box_lo & (qb <= ctx.lo + 1e-9)) or np.any(ctx.box_hi & (qb >= ctx.hi - 1e-9)))
            if usable:
                found = min(usable, key=lambda c: (not c["jump_ok"], -c["margin"]))
            elif cfg["diagnose_failures"] and not box_bound and not jump_found and _now() < dl_diag:
                diagnosis, rescued = self._diagnose(hands, kinds, q_prev_eff, template, attempt, dl_diag)
                if rescued is not None:
                    found = rescued

        info.update(seeds_used=stats["seeds"], iterations=stats["iterations"])
        if found is not None:
            q = found["q"]
            status = found["status"]
            info.update(self._public_candidate(found))
            if diagnosis is not None:
                info["diagnosis"] = diagnosis
            info["elapsed_s"] = _now() - t0
            res = DualIKResult(info, status, None if status in (SUCCESS, NEAR_LIMIT) else JOINT_JUMP_LIMIT,
                               per_seed=per_seed)
            if update_prev:
                self.q_prev = q.copy()
            self._log(verbose, res)
            return q.copy(), status, res

        reason, detail = self._failure_reason(fallback, diagnosis, box_bound, _now() >= dl_joint, budget)
        if diagnosis is not None:
            info["diagnosis"] = diagnosis
        if allow_approximate and best["q"] is not None:
            cand = self._candidate(best["q"], hands, ctx, q_prev_eff, "approximate")
            info.update(self._public_candidate(cand))
            info["elapsed_s"] = _now() - t0
            res = DualIKResult(info, APPROXIMATE, reason, detail, per_seed)
            if update_prev:
                self.q_prev = cand["q"].copy()
            self._log(verbose, res)
            return cand["q"].copy(), APPROXIMATE, res
        info["elapsed_s"] = _now() - t0
        res = DualIKResult(info, NO_VALID_POSE, reason, detail, per_seed)
        self._log(verbose, res)
        return None, NO_VALID_POSE, res

    # ------------------------------------------------------------------ 目标与上下文
    def _resolve_prev(self, q_prev, joint_state):
        if q_prev is not None:
            return self._check_q(q_prev, "q_prev")
        if joint_state and all(n in joint_state for n in self.active_joint_names):
            return self._check_q([joint_state[n] for n in self.active_joint_names], "joint_state")
        return self.q_prev.copy()

    def _parse_targets(self, T_left, T_right, q_prev, template, orientation_mode, tool_axis, cone):
        modes = _per_side(orientation_mode, lambda v: isinstance(v, (tuple, list)))
        axes = _per_side(tool_axis, lambda v: isinstance(v, (tuple, list)) and len(v) == 2
                         and all(x is None or np.size(x) == 3 for x in v))
        cones = _per_side(cone, lambda v: isinstance(v, (tuple, list)) and len(v) == 2)
        if T_left is None and T_right is None:
            raise ValueError("T_left and T_right are both None: give at least one hand a target "
                             "(None = 'hold'; pass ('hold', 'hold') to keep both hands where they are)")
        hold_poses = None
        hands, kinds = [], {}
        for idx, (side, T) in enumerate((("left", T_left), ("right", T_right))):
            mode = str(modes[idx]).lower()
            if mode not in ORIENTATION_MODES:
                raise ValueError(f"Unknown orientation_mode: {mode}. Expected one of {ORIENTATION_MODES}")
            if T is None:
                T = "hold"      # dual-p2.1：只给一只手目标时，另一只手默认保持 q_prev 处的世界位姿
            if isinstance(T, str):
                key = T.lower()
                if key in ("keep", "free"):
                    kinds[side] = key
                    continue
                if key != "hold":
                    raise ValueError(f"T_{side} must be a 4x4 pose, 'hold', 'keep', 'free' or None")
                if hold_poses is None:
                    hold_poses = self._m.fk(q_prev, template)
                M, kind = hold_poses[side], "hold"
            else:
                M, kind = validate_transform(T, f"T_{side}"), "pose"
            kinds[side] = kind
            h = _Hand()
            h.side, h.kind, h.fid, h.T = side, kind, self._m.fid[side], M
            h.p = M.translation.copy()
            h.mode = ORIENTATION_MODES.index(mode)
            h.axis = _unit(TOOL_AXIS_X if axes[idx] is None else axes[idx], "tool_axis")
            h.dt = M.rotation @ h.axis
            h.half = _cone_half_angle(cones[idx])
            h.ch, h.sh = math.cos(h.half), math.sin(h.half)
            tmp = np.array([1.0, 0.0, 0.0]) if abs(h.dt[0]) <= 0.9 else np.array([0.0, 1.0, 0.0])
            h.perp = np.cross(h.dt, tmp)
            h.perp /= np.linalg.norm(h.perp)
            h.row = 6 * len(hands)
            h.sjid = self._m.shoulder_jid[side]
            h.reach = self._m.reach_shoulder[side]
            hands.append(h)
        if not hands:
            raise ValueError("at least one hand needs a pose or 'hold' target")
        return hands, kinds

    def _make_ctx(self, hands, kinds, q_prev, template, shrink, cont=None, jump_box=True):
        """cont = 续算框半宽（rad）：限位（q_prev 已在安全裕度带内的一侧用物理限位，框才含 q_prev）
        ∩ q_prev ± cont ∩ 跳变框，DLS 单步 <= CONTINUATION_DLS_STEP_FRACTION × cont。"""
        c = _Ctx()
        if cont is not None:
            lim_lo = np.where(q_prev >= self._lo_s, self._lo_s, self.lower)
            lim_hi = np.where(q_prev <= self._hi_s, self._hi_s, self.upper)
            lo = np.maximum(lim_lo, q_prev - cont)
            hi = np.minimum(lim_hi, q_prev + cont)
        else:
            lim_lo = self._lo_s if shrink else self.lower
            lim_hi = self._hi_s if shrink else self.upper
            lo, hi = lim_lo.copy(), lim_hi.copy()
        c.max_step = MAX_STEP if cont is None else min(MAX_STEP, CONTINUATION_DLS_STEP_FRACTION * cont)
        jump = self._jump
        if jump is not None and jump_box and self.config["joint_jump_policy"] == "reject":
            # 跳变框：上限 < π 的关节，以及非周期关节的 π（周期关节对齐后自然 <= π）
            boxed = (jump < np.pi - 1e-12) | ((jump <= np.pi + 1e-12) & ~self._m.unbounded)
            boxed &= np.isfinite(jump)
            lo[boxed] = np.maximum(lo[boxed], q_prev[boxed] - jump[boxed])
            hi[boxed] = np.minimum(hi[boxed], q_prev[boxed] + jump[boxed])
        bad = lo > hi                           # q_prev 在限位外超过框宽：钉在最近的限位上
        if np.any(bad):
            v = np.minimum(np.maximum(q_prev[bad], self.lower[bad]), self.upper[bad])
            lo[bad] = v
            hi[bad] = v
        c.box_lo = lo > lim_lo + 1e-12          # 跳变框比限位更紧的一侧（失败时判断是否是跳变上限卡住）
        c.box_hi = hi < lim_hi - 1e-12
        dmask = np.ones(16)
        for side in ("left", "right"):
            if kinds[side] == "keep":
                sl = _ARM[side]
                dmask[sl] = 0.0
                lo[sl] = q_prev[sl]
                hi[sl] = q_prev[sl]
        c.lo, c.hi, c.dmask = lo, hi, dmask
        c.box_lo &= dmask > 0.0                 # "keep" 的手臂不算
        c.box_hi &= dmask > 0.0
        c.gains = self._gains * (dmask > 0)
        c.q_ref = self.q_ready
        c.periodic = self._m.unbounded                      # 回中按短弧计的关节（两侧无界）
        c.pidx = [int(i) for i in np.flatnonzero(self._m.unbounded)]
        c.settle = self.config["posture_settle"]
        c.max_iter = self.config["max_iter"]
        c.waist_free = True
        c.template = template
        c.hands = hands
        c.m = 6 * len(hands)
        if hands:
            self._set_ref_release(hands, template)
        return c

    def _set_ref_release(self, hands, template):
        """q_ref 处的肩 -> 目标距离给出的腰部释放系数（与单臂 ref_release 相同）。"""
        _FK(self._m.model, self._m.data, self._m.full_q(self.q_ready, template))
        for h in hands:
            s = self._m.data.oMi[h.sjid].translation
            h.ref_release = self._release(_norm3(h.p - s), h.reach)

    def _near_prev(self, hands, q_prev, template):
        """续算判据（与单臂相同）：每个有目标的手，目标相对 FK(q_prev) 的位置差 <= CONTINUATION_TOL × 臂展
        且姿态差（按该手的姿态模式计）<= CONTINUATION_TOL rad。"""
        _, _, errs = self._evaluate(q_prev, hands, False, self._ebuf[:6 * len(hands)], None, template)
        return all(pe <= CONTINUATION_TOL * h.reach and oe <= CONTINUATION_TOL for h, (pe, oe) in zip(hands, errs))

    @staticmethod
    def _release(dist, reach):
        if not reach or reach <= 1e-9:
            return 0.0
        t = (dist - WAIST_RELEASE_START * reach) / max((WAIST_RELEASE_END - WAIST_RELEASE_START) * reach, 1e-6)
        return _smoothstep(t)

    # ------------------------------------------------------------------ 种子
    def _seeds(self, ctx, hands, kinds, q_prev):
        """按顺序产生 (类别, 16 维种子)；种子库排到时才计算。"""
        seen = []

        un = self._m.unbounded

        def fresh(q):
            if un.any():                        # 周期关节对齐到 q_prev 所在的那一圈（跳变框也按这一圈给出）
                q[un] = q_prev[un] + _wrap(q[un] - q_prev[un])
            q = np.minimum(np.maximum(q, ctx.lo), ctx.hi)
            for s in seen:
                if np.max(np.abs(s - q)) < 1e-6:
                    return None
            seen.append(q)
            return q

        base = q_prev.copy()
        for kind, q in (("q_prev", base), ("q_ready", self._with_locked(self.q_ready, ctx, base))):
            q = fresh(q)
            if q is not None:
                yield kind, q
        if self._lib.arms and self.config["seed_library_seeds"] > 0:
            for q in self._library_seeds(hands, kinds, base, self.config["seed_library_seeds"], box=ctx):
                q = fresh(q)
                if q is not None:
                    yield "library", q
        rng = np.random.default_rng(RNG_SEED)
        lo_u = np.where(np.isfinite(ctx.lo), ctx.lo, -np.pi)
        hi_u = np.where(np.isfinite(ctx.hi), ctx.hi, np.pi)
        n = 0
        while True:
            n += 1
            if RANDOM_SEED_MODE == "uniform" or (RANDOM_SEED_MODE == "mixed" and n % 2 == 0):
                q = rng.uniform(lo_u, hi_u)
            else:
                q = self.q_ready + rng.normal(0.0, RANDOM_SEED_SIGMA, 16)
            q = fresh(self._with_locked(q, ctx, base))
            if q is not None:
                yield "random", q

    def _probe_seeds(self, ctx, hands, kinds, q_prev):
        """跳变框探测的种子：q_ready，再加 JUMP_PROBE_SEEDS 个不加框的种子库种子。"""
        out = [("probe_q_ready", self._with_locked(self.q_ready, ctx, q_prev))]
        if self._lib.arms and JUMP_PROBE_SEEDS > 0:
            out += [("probe_library", q) for q in self._library_seeds(hands, kinds, q_prev, JUMP_PROBE_SEEDS)]
        for kind, q in out:
            yield kind, np.minimum(np.maximum(q, ctx.lo), ctx.hi)

    @staticmethod
    def _with_locked(q, ctx, base):
        q = q.copy()
        locked = ctx.dmask == 0.0
        q[locked] = base[locked]
        return q

    def _library_seeds(self, hands, kinds, base, k, sides=None, box=None):
        """组合种子：每个腰网格点上两臂各取最近样本，按两臂距离之和 + 腰偏离惩罚排序。

        box = 上下文（跳变框）：只取框内的腰网格点与手臂样本（框外的种子会被钳到框边、几乎无用）。"""
        lib = self._lib
        Rg, og, wg = lib.grid_R, lib.grid_o, lib.grid_w
        score = LIBRARY_WAIST_WEIGHT * np.linalg.norm(wg - base[WAIST][None, :], axis=1)
        if box is not None:
            inside = np.all((wg >= box.lo[WAIST] - 1e-12) & (wg <= box.hi[WAIST] + 1e-12), axis=1)
            if not inside.any():
                return []
            score[~inside] = np.inf
        picks = {}
        un = self._m.unbounded
        for h in hands:
            if sides is not None and h.side not in sides:
                continue
            sl = _ARM[h.side]
            mask = None
            if box is not None and box.dmask[sl].any():
                mask = lib.arms[h.side].inside(box.lo[sl], box.hi[sl], un[sl], base[sl])
                if mask is not None and not mask.any():
                    return []
            d, Q = lib.arms[h.side].best(h.T, Rg, og, mask, self._lib_buf)
            score = score + d
            picks[h.side] = Q
        order = np.argsort(score, kind="stable")[:k]
        order = order[np.isfinite(score[order])]
        out = []
        for g in order:
            q = base.copy()
            q[WAIST] = wg[g]
            for side, Q in picks.items():
                q[_ARM[side]] = Q[g]
            out.append(q)
        return out

    # ------------------------------------------------------------------ 单种子迭代
    def _evaluate(self, q, hands, jac, e, J, template):
        """两手任务误差（缩放后）写入 e、雅可比写入 J；返回 (两手都达标, |e|², [(位置误差, 姿态误差)])。

        e 的方向是“当前 -> 目标”，J 满足 de/dq = -J（迭代步 dq = J⁺e）；tests 用有限差分核对。"""
        mdl = self._m
        model, data = mdl.model, mdl.data
        qf = mdl.full_q(q, template)
        if jac:
            _CJJ(model, data, qf)
        else:
            _FK(model, data, qf)
        ok = True
        errs = []
        for h in hands:
            M = _UFP(model, data, h.fid)
            r = h.row
            if h.mode == 0:
                # full：本体系 log6 误差，雅可比 = Jlog6 · 本体雅可比
                delta = M.actInv(h.T)
                nu = _LOG6(delta).vector
                pe = math.hypot(*delta.translation.tolist())
                oe = math.hypot(*nu[3:].tolist())
                e[r:r + 6] = nu * _SCALE6
                if jac:
                    Jb = _GFJ(model, data, h.fid, _LOCAL)
                    if not mdl.vcols_identity:
                        Jb = Jb[:, mdl.vcols]
                    np.dot(_SCALE6_COL * _JLOG6(delta.inverse()), Jb, out=J[r:r + 6])
            else:
                # tool_dir / tool_cone：世界系位置误差 + 工具轴方向误差（锥外取锥面上最近方向为目标）
                dp = h.p - M.translation
                pe = math.hypot(*dp.tolist())
                d = M.rotation @ h.axis
                c = min(1.0, max(-1.0, float(d.dot(h.dt))))
                ang = math.acos(c)
                inside = False
                if h.mode == 1:
                    oe = ang
                    edir = h.dt - d
                elif ang <= h.half:
                    oe = 0.0
                    inside = True
                    edir = None
                else:
                    oe = ang - h.half
                    v = d - c * h.dt
                    nv = math.hypot(*v.tolist())
                    u = v / nv if nv >= 1e-12 else h.perp
                    b = h.ch * h.dt + h.sh * u
                    edir = b / math.hypot(*b.tolist()) - d
                e[r:r + 3] = dp * _INV_L
                if inside:
                    e[r + 3:r + 6] = 0.0
                else:
                    e[r + 3:r + 6] = edir
                if jac:
                    Jw = _GFJ(model, data, h.fid, _LWA)
                    if not mdl.vcols_identity:
                        Jw = Jw[:, mdl.vcols]
                    J[r:r + 3] = Jw[:3] * _INV_L
                    if inside:
                        J[r + 3:r + 6] = 0.0
                    else:
                        # d(d)/dq = ω × d = [d]×ᵀ ω；锥模式下目标方向随 d 绕锥轴转动的分量与 e 正交，
                        # 忽略它不改变 |e|² 的梯度（tests 核对梯度）
                        S = np.array([[0.0, -d[2], d[1]], [d[2], 0.0, -d[0]], [-d[1], d[0], 0.0]])
                        np.dot(S.T, Jw[3:], out=J[r + 3:r + 6])
            if pe > POS_TOL or oe > ORI_TOL:
                ok = False
            errs.append((pe, oe))
        return ok, float(e.dot(e)), errs

    @staticmethod
    def _sns_step(J, e, q, lo, hi, dinv, z, eye, max_step=MAX_STEP):
        """带权阻尼最小二乘 + 零空间步 + 限位饱和：越界关节钉在边界、去掉该列重解（最多 SNS_ROUNDS 次求解）。

        dq = z + D Jᵀ (J D Jᵀ + λ² I)⁻¹ (e − J z)，D = dinv（腰部逆度量），λ² = DAMPING_MIN + DAMPING_LM·|e|²；
        步长 2-范数限幅 max_step（默认 MAX_STEP，续算时更小）。z=None 表示纯任务步。"""
        lam2 = DAMPING_MIN + DAMPING_LM * float(e.dot(e))
        ms2 = max_step * max_step
        Jd = J * dinv
        A = Jd @ J.T
        A += lam2 * eye
        if z is None:
            dq = Jd.T @ _solve_vec(A, e)
        else:
            dq = Jd.T @ _solve_vec(A, e - J @ z)
            dq += z
        n2 = float(dq.dot(dq))
        if n2 > ms2:
            dq *= max_step / math.sqrt(n2)
        qn = q + dq
        viol = (qn < lo) | (qn > hi)
        if not viol.any():
            return dq
        d = dinv.copy()
        zf = np.zeros(16) if z is None else z.copy()
        fixed = np.zeros(16)
        for _ in range(SNS_ROUNDS - 1):
            viol &= d > 0.0
            if not viol.any():
                break
            fixed[viol] = np.where(qn[viol] < lo[viol], lo[viol], hi[viol]) - q[viol]
            d[viol] = 0.0
            zf[viol] = 0.0
            Jd = J * d
            A = Jd @ J.T
            A += lam2 * eye
            base = zf + fixed
            dq = Jd.T @ _solve_vec(A, e - J @ base)
            dq += base
            n2 = float(dq.dot(dq))
            if n2 > ms2:
                dq *= max_step / math.sqrt(n2)
            qn = q + dq
            viol = (qn < lo) | (qn > hi)
        return np.minimum(np.maximum(qn, lo), hi) - q

    def _run_seed(self, q0, ctx, deadline):
        """单种子迭代。返回 (q, 达标, 迭代数, 结束原因, 两手误差, |e|²)。

        未达标：带权 DLS 步（含限幅的零空间回中；零空间步找不到下降时退回纯任务步）+ 回溯线搜索，
        每 STALL_ITERS 步检查一次进展。达标后（posture_settle）：只接受仍达标且回中代价不增的步，
        步长 < POSTURE_SETTLE_TOL、连续 SETTLE_PATIENCE 个慢步或 MAX_SETTLE_ITERS 步后结束。"""
        hands = ctx.hands
        lo, hi, m = ctx.lo, ctx.hi, ctx.m
        tmpl, gains, q_ref, pidx = ctx.template, ctx.gains, ctx.q_ref, ctx.pidx
        max_step = ctx.max_step
        evaluate, sns, release = self._evaluate, self._sns_step, self._release
        oMi = self._m.data.oMi
        eye = _EYE[m]
        q = np.minimum(np.maximum(np.asarray(q0, dtype=float), lo), hi)
        e, J = np.empty(m), np.empty((m, 16))
        e2, J2 = np.empty(m), np.empty((m, 16))
        ok, err2, errs = evaluate(q, hands, True, e, J, tmpl)
        diff = q - q_ref
        for i in pidx:
            diff[i] = (diff[i] + math.pi) % _TWO_PI - math.pi
        h_cost = 0.5 * float(gains.dot(diff * diff))
        hn = h_cost
        diffn = diff
        it = 0
        reason = "max_iter"
        settle_n = slow = 0
        check_err, check_n = math.sqrt(err2), 0
        dinv = ctx.dmask.copy()
        waist_free = ctx.waist_free and ctx.dmask[0] > 0.0
        if waist_free:
            shoulders = [(h.p, h.sjid, h.reach) for h in hands]
            alpha0 = max(h.ref_release for h in hands)
            wr0, wr1 = float(q_ref[0]), float(q_ref[1])
            wp0, wp1 = bool(ctx.periodic[0]), bool(ctx.periodic[1])
        use_null = bool(gains.any())
        while it < ctx.max_iter:
            if ok and not ctx.settle:
                reason = "converged"
                break
            if _now() > deadline:
                reason = "timeout"
                break
            it += 1
            err = math.sqrt(err2)
            if waist_free:
                # 腰部逆度量：肩 -> 目标距离 / 臂展 给出释放系数（两手取大），腰偏离 q_ref 时保持释放
                a = alpha0
                for p, sj, reach in shoulders:
                    d3 = p - oMi[sj].translation
                    a = max(a, release(math.sqrt(float(d3.dot(d3))), reach))
                w0, w1 = float(q[0]) - wr0, float(q[1]) - wr1
                if wp0:
                    w0 = (w0 + math.pi) % _TWO_PI - math.pi
                if wp1:
                    w1 = (w1 + math.pi) % _TWO_PI - math.pi
                a = max(a, min(math.hypot(w0, w1) / WAIST_RETURN_RELEASE_RAD, 1.0))
                dinv[0] = dinv[1] = (1.0 - a) / WAIST_TASK_WEIGHT + a / WAIST_TASK_WEIGHT_FAR
            z = None
            if use_null:
                # 零空间回中：权重随误差衰减（与单臂相同），再限幅——未达标时 |z| <= NULL_STEP_RATIO·|e|，
                # 否则大幅零空间运动的二阶误差会让线搜索在离解 1 mm 处找不到下降方向
                wnull = (1.5 if ok else 1.0) / (1.0 + (err / NULLSPACE_ERR_SCALE) ** 2)
                z = (-wnull) * gains * diff
                cap = NULL_STEP_MAX if ok else min(NULL_STEP_MAX, NULL_STEP_RATIO * err)
                zn = math.sqrt(float(z.dot(z)))
                if zn > cap:
                    z *= cap / zn
            accepted = False
            for zz in ((z, None) if (z is not None and not ok) else (z,)):
                dq = sns(J, e, q, lo, hi, dinv, zz, eye, max_step)
                alpha = 1.0
                for _ in range(LINE_SEARCH_STEPS):
                    qn = q + dq if alpha == 1.0 else q + alpha * dq
                    np.minimum(np.maximum(qn, lo, out=qn), hi, out=qn)
                    okn, err2n, errsn = evaluate(qn, hands, True, e2, J2, tmpl)
                    if ok:
                        if okn:
                            diffn = qn - q_ref
                            for i in pidx:
                                diffn[i] = (diffn[i] + math.pi) % _TWO_PI - math.pi
                            hn = 0.5 * float(gains.dot(diffn * diffn))
                            if hn <= h_cost:
                                accepted = True
                                break
                    elif err2n < err2:
                        accepted = True
                        break
                    alpha *= 0.5
                if accepted:
                    break
            if not accepted:
                reason = "converged" if ok else "no_descent"
                break
            was_ok = ok
            q = qn
            e, e2 = e2, e
            J, J2 = J2, J
            ok, err2, errs = okn, err2n, errsn
            if was_ok:
                settle_n += 1
                rel = (h_cost - hn) / max(h_cost, 1e-12)
                h_cost, diff = hn, diffn
                slow = slow + 1 if rel < SETTLE_COST_RTOL else 0
                if (alpha * math.sqrt(float(dq.dot(dq))) < POSTURE_SETTLE_TOL or slow >= SETTLE_PATIENCE
                        or settle_n >= MAX_SETTLE_ITERS):
                    reason = "converged"
                    break
                continue
            diff = q - q_ref
            for i in pidx:
                diff[i] = (diff[i] + math.pi) % _TWO_PI - math.pi
            if ok:
                h_cost = 0.5 * float(gains.dot(diff * diff))
                settle_n = slow = 0
                continue
            if it in _PROGRESS_GATES and err2 > _PROGRESS_GATES[it]:
                reason = "slow"                 # 这么多步后误差仍大：这个种子几乎不会再收敛，换种子
                break
            check_n += 1
            if check_n >= STALL_ITERS:
                now_err = math.sqrt(err2)
                if now_err > (1.0 - STALL_RTOL) * check_err:
                    reason = "stall"
                    break
                check_err, check_n = now_err, 0
        if ok and reason == "max_iter":
            reason = "converged"
        return q, ok, it, reason, errs, err2

    # ------------------------------------------------------------------ 候选
    def _align(self, q, ref):
        """跨度 >= 2π 的转动关节：在限位内取最接近 ref 的 2πk 等价角。"""
        q = q.copy()
        for i in np.flatnonzero(self._m.periodic_capable):
            lo, hi = self.lower[i], self.upper[i]
            if not np.isfinite(lo) and not np.isfinite(hi):
                q[i] = ref[i] + _wrap(q[i] - ref[i])
                continue
            kmin = math.ceil((lo - q[i]) / _TWO_PI - 1e-12) if np.isfinite(lo) else -10 ** 6
            kmax = math.floor((hi - q[i]) / _TWO_PI + 1e-12) if np.isfinite(hi) else 10 ** 6
            if kmin > kmax:
                continue
            k = min(max(round((ref[i] - q[i]) / _TWO_PI), kmin), kmax)
            q[i] = min(max(q[i] + _TWO_PI * k, lo), hi)
        return q

    def _candidate(self, q, hands, ctx, q_prev, kind):
        qa = self._align(q, q_prev)
        ok, _, errs = self._evaluate(qa, hands, False, self._ebuf[:ctx.m], None, ctx.template)
        moving = ctx.dmask > 0.0
        finite = np.isfinite(self.lower) | np.isfinite(self.upper)
        mask = moving & finite & (self.upper - self.lower > 1e-12)
        each = np.minimum(qa - self.lower, self.upper - qa)
        if np.any(mask):
            idx = int(np.flatnonzero(mask)[np.argmin(each[mask])])
            margin = float(each[idx])
        else:
            idx, margin = 0, math.inf
        diff = qa - q_prev
        un = self._m.unbounded
        diff[un] = _wrap(diff[un])
        absd = np.abs(diff)
        jidx = int(np.argmax(absd))
        jump_ok = True if self._jump is None else bool(np.all(absd <= self._jump + JUMP_COMPARE_TOL))
        near_ok = bool(ok)
        status = (SUCCESS if margin >= SAFE_MARGIN else NEAR_LIMIT) if near_ok else NO_VALID_POSE
        accept = near_ok and margin >= SAFE_MARGIN and (jump_ok or self._jump is None)
        if self.config["joint_jump_policy"] == "prefer":
            accept = accept and jump_ok
        return {"q": qa, "status": status, "near_ok": near_ok, "acceptable": bool(accept),
                "margin": margin, "margin_joint": self.active_joint_names[idx],
                "max_jump": float(absd[jidx]), "jump_joint": self.active_joint_names[jidx],
                "jump_ok": jump_ok, "errs": {h.side: e for h, e in zip(hands, errs)},
                "modes": {h.side: ORIENTATION_MODES[h.mode] for h in hands}, "seed_kind": kind}

    def _public_candidate(self, c):
        q = c["q"]
        out = {"q": q.copy(), "q_waist": q[WAIST].copy(), "q_left": q[LEFT].copy(), "q_right": q[RIGHT].copy(),
               "margin": c["margin"], "margin_joint": c["margin_joint"], "max_jump": c["max_jump"],
               "jump_joint": c["jump_joint"], "jump_ok": c["jump_ok"], "seed_kind": c["seed_kind"]}
        for side in ("left", "right"):
            if side in c["errs"]:
                pe, oe = c["errs"][side]
                out[side] = {"pos_err": pe, "ori_err": oe, "pose_ok": pe <= POS_TOL and oe <= ORI_TOL,
                             "orientation_mode": c["modes"][side]}
            else:
                out[side] = {"pos_err": None, "ori_err": None, "pose_ok": None, "orientation_mode": None}
        return out

    # ------------------------------------------------------------------ 失败诊断
    def _diagnose(self, hands, kinds, q_prev, template, attempt, deadline):
        """左手单独、右手单独（腰 + 该臂，9 轴）能否解出；都能解出时用主从分解拼 16 维种子再联合精化。

        "单独解不出"只在该手的全部诊断种子都自然结束（没被时间预算打断）时才下结论；
        否则 completed=False，失败原因记为 TIME_BUDGET_EXCEEDED。返回 (诊断, 救回的候选或 None)。
        诊断不加跳变框（结论与 max_joint_jump_rad 无关）：拼出的联合解跳变超限时记 jump_limited。"""
        diag = {"left_alone_ok": None, "right_alone_ok": None, "completed": False}
        alone = {}
        for h in hands:
            k2 = {"left": kinds["left"], "right": kinds["right"]}
            other = "right" if h.side == "left" else "left"
            k2[other] = "keep"
            c = self._make_ctx([self._clone_hand(h, 0)], k2, q_prev, template, shrink=True, jump_box=False)
            seeds = [self._with_locked(self.q_ready, c, q_prev)]
            if self._lib.arms:
                seeds += self._library_seeds(c.hands, k2, q_prev, DIAGNOSIS_LIBRARY_SEEDS, sides=(h.side,))
            got = None
            for s in seeds:
                if _now() >= deadline:
                    return diag, None
                q, ok, why = attempt(s, f"{h.side}_alone", c, deadline)
                if ok:
                    got = q
                    break
                if why == "timeout":
                    return diag, None
            alone[h.side] = got
            diag[f"{h.side}_alone_ok"] = got is not None
            if got is not None:
                diag[f"{h.side}_alone_waist"] = got[WAIST].copy()
        if len(hands) == 2 and all(alone.get(s) is not None for s in ("left", "right")):
            ctx = self._make_ctx(hands, kinds, q_prev, template, shrink=True, jump_box=False)
            for master, slave in (("left", "right"), ("right", "left")):
                if _now() >= deadline:
                    return diag, None
                hs = next(h for h in hands if h.side == slave)
                k2 = dict(kinds)
                k2[master] = "keep"
                seed = alone[master].copy()
                seed[_ARM[slave]] = alone[slave][_ARM[slave]]
                c = self._make_ctx([self._clone_hand(hs, 0)], k2, seed, template, shrink=True, jump_box=False)
                c.dmask[WAIST] = 0.0
                c.gains = self._gains * (c.dmask > 0)
                c.lo[WAIST] = seed[WAIST]
                c.hi[WAIST] = seed[WAIST]
                q, ok, why = attempt(seed, f"split_{master}_{slave}", c, deadline)
                if why == "timeout":
                    return diag, None
                if not ok:
                    continue
                q, ok, why = attempt(q, "split_refine", ctx, deadline)
                if why == "timeout":
                    return diag, None
                if ok:
                    cand = self._candidate(q, hands, ctx, q_prev, f"split_{master}")
                    if cand["acceptable"]:
                        diag["completed"] = True
                        diag["rescued_by"] = f"split_{master}"
                        return diag, cand
                    if cand["near_ok"] and not cand["jump_ok"]:
                        diag["jump_limited"] = True
                        diag["jump_joint"] = cand["jump_joint"]
                        diag["max_jump"] = cand["max_jump"]
        elif len(hands) == 1 and alone.get(hands[0].side) is not None:
            # 只有一只手有目标：单独解（另一臂不动）本身就是原问题的解
            ctx = self._make_ctx(hands, kinds, q_prev, template, shrink=True, jump_box=False)
            cand = self._candidate(alone[hands[0].side], hands, ctx, q_prev, f"{hands[0].side}_alone")
            if cand["acceptable"]:
                diag["completed"] = True
                diag["rescued_by"] = f"{hands[0].side}_alone"
                return diag, cand
            if cand["near_ok"] and not cand["jump_ok"]:
                diag["jump_limited"] = True
                diag["jump_joint"] = cand["jump_joint"]
                diag["max_jump"] = cand["max_jump"]
        diag["completed"] = True
        return diag, None

    @staticmethod
    def _clone_hand(h, row):
        c = _Hand()
        for k in _Hand.__slots__:
            setattr(c, k, getattr(h, k))
        c.row = row
        return c

    def _failure_reason(self, fallback, diagnosis, box_bound, joint_phase_timed_out, budget):
        if any(c["near_ok"] and not c["jump_ok"] for c in fallback):
            return JOINT_JUMP_LIMIT, "only solutions exceeding max_joint_jump_rad were found"
        if box_bound:
            return JOINT_JUMP_LIMIT, ("the closest configuration found is held by the max_joint_jump_rad box "
                                      "around q_prev; use an intermediate waypoint or raise the limit")
        if diagnosis is not None and diagnosis.get("jump_limited"):
            return JOINT_JUMP_LIMIT, (f"a solution exists but needs {diagnosis['max_jump']:.3f} rad on "
                                      f"{diagnosis['jump_joint']} (> max_joint_jump_rad); use an intermediate "
                                      "waypoint, raise the limit or joint_jump_policy='prefer'")
        if diagnosis is not None and diagnosis.get("completed"):
            l_ok, r_ok = diagnosis.get("left_alone_ok"), diagnosis.get("right_alone_ok")
            if l_ok is False and r_ok is False:
                return BOTH_UNREACHABLE, "neither hand target was solved on its own (waist + one arm; search result)"
            if l_ok is False:
                return LEFT_UNREACHABLE, "left target was not solved on its own (waist + left arm; search result)"
            if r_ok is False:
                return RIGHT_UNREACHABLE, "right target was not solved on its own (waist + right arm; search result)"
            return NOT_JOINTLY_REACHABLE, ("each hand target is solvable on its own, but the search found no "
                                           "configuration reaching both (search result, not a proof)")
        if diagnosis is not None or joint_phase_timed_out:
            return TIME_BUDGET_EXCEEDED, f"time budget {budget} s used up before the search finished"
        return SEEDS_EXHAUSTED, "all seeds tried without a valid solution"

    @staticmethod
    def _log(verbose, res):
        if verbose:
            print(f"[dual_arm_ik] {res}")