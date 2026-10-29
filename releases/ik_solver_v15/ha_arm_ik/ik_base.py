#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""IK 求解器核心运动学、几何计算、URDF 模型与候选解评估库。

整合了底层 3D 几何与刚体变换、URDF 串联机器人运动学链模型，以及逆运动学候选解打分、
排序与诊断结果包装。
"""

from collections import defaultdict, deque
from enum import Enum
import math
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation as R

__all__ = [
    # 常量
    "TOOL_AXIS_X",
    "TOOL_CONE_HALF_ANGLE",
    "ORIENTATION_MODES",
    "POS_TOL",
    "ORI_TOL",
    "SAFE_MARGIN",
    "PERIODIC_MIN_SPAN_RAD",
    "RIGID_BOTTOM_ROW_TOL",
    "RIGID_ROTATION_TOL",
    "S_JOINT",
    "E_JOINT",
    "W_JOINT",
    "REF_DIR",
    "WAIST_JOINT_NAMES",
    "W_NOM",
    "W_SMOOTH",
    "W_LIMIT",
    "W_WAIST_RETURN",
    "W_TORQUE",
    "PHI_TIE_TOL",
    "ARM_CONFIGS",
    # 状态与结果类型
    "Status",
    "FailureReason",
    "status_from_str",
    "failure_reason_for_status",
    "IKResult",
    "make_result",
    "is_valid_status",
    # 3D 几何与刚体齐次变换
    "normalize",
    "skew",
    "rotx",
    "roty",
    "rotz",
    "rpy_to_R",
    "rot_axis",
    "make_T",
    "T_origin",
    "T_revolute",
    "T_prismatic",
    "invert_T",
    "transform_point",
    "make_translation_T",
    "pose_error",
    "validate_orientation_mode",
    "validate_transform4x4",
    "validate_q_vector",
    "normalize_cone_half_angle",
    "closest_direction_on_cone",
    "tool_direction_from_T",
    "tool_direction_residual",
    "tool_direction_angle_error",
    "tool_cone_residual",
    "tool_cone_angle_error",
    "project_to_plane",
    "sew_angle_from_points",
    "target_frames",
    # 关节限位、距离与周期分支
    "clip_to_bounds",
    "wrap_to_pi",
    "joint_dist",
    "joint_difference",
    "joint_dist_periodic",
    "select_closest_bounded_branch",
    "bounded_joint_difference",
    "multi_turn_mask_from_chain",
    "shift_to_nearest_branch",
    "joint_margin",
    "joint_range_scale",
    "joint_center_from_bounds",
    "joint_limit_penalty",
    "unique_seed_list",
    # URDF 模型与串联链
    "get_sew_joints",
    "get_fixed_joint_origin",
    "get_urdf_root_link",
    "find_chain",
    "SerialChain",
    "estimate_chain_max_reach_from_joint",
    "joint_origin_position_at_zero",
    # 候选解打分、排序与包装
    "candidate_sort_key",
    "candidate_status",
    "score_candidate",
    "has_valid_torque",
    "update_candidate_final_cost_with_torque",
    "select_best_candidate",
]

# ==============================================================================
# 1. 全局常量：几何、容差、代价权重与机器人配置
# ==============================================================================

TOOL_AXIS_X = np.array([1.0, 0.0, 0.0], dtype=float)
TOOL_CONE_HALF_ANGLE = np.deg2rad(30.0)
ORIENTATION_MODES = ("full", "tool_dir", "tool_cone")
POS_TOL = 5e-4
ORI_TOL = 5e-4
SAFE_MARGIN = 0.02
PERIODIC_MIN_SPAN_RAD = 2.0 * np.pi + 1e-6

RIGID_BOTTOM_ROW_TOL = 1e-9
RIGID_ROTATION_TOL = 1e-6
_RIGID_BOTTOM_ROW = np.array([0.0, 0.0, 0.0, 1.0])
_EYE3 = np.eye(3)

# 关节与臂角常量
S_JOINT = "LHand_Link2"
E_JOINT = "LHand_Link4"
W_JOINT = "LHand_Link6"
REF_DIR = np.array([0.0, 0.0, 1.0], dtype=float)
WAIST_JOINT_NAMES = ["Hip_Link1", "Body_Link"]

# 候选解代价权重体系
W_NOM = 50.0
W_SMOOTH = 8.0
W_LIMIT = 10.0
W_WAIST_RETURN = 2000.0
W_TORQUE = 500.0
PHI_TIE_TOL = 1e-5

# 内置人形臂配置
ARM_CONFIGS = {
    "left": {
        "base_link": "Body_Link",
        "ee_link": "left_l_dg_tcp",
        "s_joint": "LHand_Link2",
        "e_joint": "LHand_Link4",
        "w_joint": "LHand_Link6",
        "mount_xyz": np.array([0.0, 0.0, 0.5087]),
        "mount_rpy": np.array([0.0, 0.0, 0.0]),
        "q_ready": np.array([-0.523599, -0.785398, 0.0, -0.698132, 0.0, 1.570796, 0.0]),
    },
    "right": {
        "base_link": "Body_Link",
        "ee_link": "right_l_dg_tcp",
        "s_joint": "RHand_Link2",
        "e_joint": "RHand_Link4",
        "w_joint": "RHand_Link6",
        "mount_xyz": np.array([0.0, 0.0, 0.5087]),
        "mount_rpy": np.array([0.0, 0.0, 0.0]),
        # 左臂镜像：奇数关节反号，偶数关节同号
        "q_ready": np.array([0.523599, -0.785398, 0.0, -0.698132, 0.0, 1.570796, 0.0]),
    },
}


# ==============================================================================
# 2. 状态枚举、诊断与结果封装 IKResult
# ==============================================================================

class Status(Enum):
    """solve() 公共状态词表。"""
    SUCCESS = "SUCCESS"
    NEAR_LIMIT = "NEAR_LIMIT"
    APPROXIMATE = "APPROXIMATE"
    NO_VALID_POSE = "NO_VALID_POSE"
    NO_RESULT = "NO_RESULT"
    OUTSIDE_WORKSPACE = "OUTSIDE_WORKSPACE"


class FailureReason(Enum):
    """solve() 失败原因（SUCCESS / NEAR_LIMIT 时为 None）。"""
    QP_INFEASIBLE_ALL_SEEDS = "QP_INFEASIBLE_ALL_SEEDS"
    QP_SOLVER_ERROR = "QP_SOLVER_ERROR"
    SEEDS_EXHAUSTED = "SEEDS_EXHAUSTED"
    TOTAL_ITER_BUDGET_HIT = "TOTAL_ITER_BUDGET_HIT"
    NOT_CONVERGED = "NOT_CONVERGED"
    WORKSPACE_REJECTED = "WORKSPACE_REJECTED"


_VALID_STATUS_STRS = {status.value for status in Status}


def status_from_str(s):
    """把内部状态字符串转换为 Status；未知值返回 None。"""
    return Status(s) if s in _VALID_STATUS_STRS else None


def failure_reason_for_status(status_str):
    """为失败的公共状态字符串返回默认失败原因。"""
    if status_str == "NO_RESULT":
        return FailureReason.SEEDS_EXHAUSTED
    if status_str in ("NO_VALID_POSE", "APPROXIMATE"):
        return FailureReason.NOT_CONVERGED
    return None


class IKResult(dict):
    """solve() 诊断结果包装器，继承自 dict 保证字典访问与对象属性访问并存。"""

    def __init__(
        self,
        info_dict=None,
        status=None,
        failure_reason=None,
        failure_detail="",
        per_seed=None,
    ):
        super().__init__(info_dict or {})
        self.status = (
            status_from_str(status) if isinstance(status, str) else status
        )
        self.failure_reason = failure_reason
        self.failure_detail = failure_detail or ""
        self.per_seed = list(per_seed) if per_seed else []
        if self.status is not None and "status" not in self:
            self["status"] = self.status.value

    @property
    def ok(self):
        """求解是否成功（含 SUCCESS 与 NEAR_LIMIT）。"""
        return self.status in (Status.SUCCESS, Status.NEAR_LIMIT)

    @property
    def failed(self):
        """求解是否失败。"""
        return not self.ok

    @property
    def status_val(self):
        """状态枚举字符串值。"""
        return self.status.value if self.status else ""

    @property
    def pose_ok(self):
        """位姿是否达标。"""
        return self.get("pose_ok", False)

    @property
    def num_candidates(self):
        """存活候选解数量。"""
        return self.get("num_candidates", 0)

    @property
    def q(self):
        """解向量 (ndarray 或 None)。"""
        return self.get("q")

    @property
    def cost(self):
        """候选解综合代价值。"""
        return self.get("cost")

    @property
    def pos_err(self):
        """末端位置残差 (m)。"""
        return self.get("pos_err")

    @property
    def ori_err(self):
        """末端姿态残差 (rad)。"""
        return self.get("ori_err")

    @property
    def psi(self):
        """臂角 (rad)。"""
        return self.get("psi")

    @property
    def phi(self):
        """臂角相对预备臂角的偏差 (rad)。"""
        return self.get("phi")

    @property
    def margin(self):
        """最小关节限位裕度 (rad)。"""
        return self.get("margin")

    def __repr__(self):
        parts = [f"status={self.status.value if self.status else 'NONE'}"]
        if self.failure_reason is not None:
            parts.append(f"failure_reason={self.failure_reason.value}")
        if "pos_err" in self:
            parts.append(f"pos_err={self['pos_err'] * 1000:.3f}mm")
        if "num_candidates" in self:
            parts.append(f"candidates={self['num_candidates']}")
        return f"IKResult({', '.join(parts)})"


def make_result(
    info_dict,
    status_str,
    failure_reason=None,
    failure_detail="",
    per_seed=None,
):
    """打包构造 IKResult 对象。"""
    if failure_reason is None:
        failure_reason = failure_reason_for_status(status_str)
    return IKResult(
        info_dict,
        status=status_str,
        failure_reason=failure_reason,
        failure_detail=failure_detail,
        per_seed=per_seed,
    )


def is_valid_status(status):
    """判断状态字符串是否允许伴随有效关节角 q 返回。"""
    return status in ("SUCCESS", "NEAR_LIMIT", "APPROXIMATE")


# ==============================================================================
# 3. 3D 线性代数、旋转、刚体齐次变换与空间误差
# ==============================================================================

def normalize(v, eps=1e-12):
    """返回单位向量；非有限或模长小于 eps 时抛出 ValueError。"""
    v = np.asarray(v, dtype=float)
    if not np.all(np.isfinite(v)):
        raise ValueError("vector contains non-finite values")
    n = np.linalg.norm(v)
    if n < eps:
        raise ValueError("zero norm vector")
    return v / n


def skew(v):
    """返回 3x3 反对称矩阵，满足 skew(a) @ b = a x b。"""
    v = np.asarray(v, dtype=float)
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ], dtype=float)


def rotx(a):
    """绕 X 轴的主动旋转矩阵。"""
    c, s = np.cos(a), np.sin(a)
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, c, -s],
        [0.0, s, c],
    ], dtype=float)


def roty(a):
    """绕 Y 轴的主动旋转矩阵。"""
    c, s = np.cos(a), np.sin(a)
    return np.array([
        [c, 0.0, s],
        [0.0, 1.0, 0.0],
        [-s, 0.0, c],
    ], dtype=float)


def rotz(a):
    """绕 Z 轴的主动旋转矩阵。"""
    c, s = np.cos(a), np.sin(a)
    return np.array([
        [c, -s, 0.0],
        [s, c, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=float)


def rpy_to_R(rpy):
    """URDF RPY 约定：R = Rz(yaw) @ Ry(pitch) @ Rx(roll)。"""
    roll, pitch, yaw = rpy
    return rotz(yaw) @ roty(pitch) @ rotx(roll)


def rot_axis(axis, theta):
    """绕归一化三维轴的旋转矩阵。"""
    axis = normalize(axis)
    return R.from_rotvec(axis * theta).as_matrix()


def make_T(R=None, p=None):
    """用 3x3 旋转矩阵与 3 维平移向量构造 4x4 齐次刚体变换。"""
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    if p is not None:
        T[:3, 3] = np.asarray(p, dtype=float)
    return T


def T_origin(xyz, rpy):
    """用平移 xyz 和固定轴 rpy 构造 URDF 刚体变换。"""
    return make_T(rpy_to_R(rpy), xyz)


def T_revolute(axis, q):
    """转动关节位移变换。"""
    return make_T(rot_axis(axis, q))


def T_prismatic(axis, q):
    """移动关节位移变换。"""
    axis = normalize(axis)
    return make_T(np.eye(3), axis * q)


def invert_T(T):
    """刚体变换快速求逆：T^{-1} = [[R^T, -R^T @ p], [0, 1]]。"""
    T = np.asarray(T, dtype=float)
    Tin = np.eye(4)
    Tin[:3, :3] = T[:3, :3].T
    Tin[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return Tin


def transform_point(T, pxyz):
    """用 4x4 变换三维点。"""
    p4 = np.ones(4, dtype=float)
    p4[:3] = np.asarray(pxyz, dtype=float)
    return (np.asarray(T, dtype=float) @ p4)[:3]


def make_translation_T(xyz):
    """返回 4x4 纯平移变换。"""
    T = np.eye(4)
    T[:3, 3] = np.asarray(xyz, dtype=float)
    return T


def pose_error(T_current, T_target):
    """6 维位姿残差：dp = p_cur - p_tar，drot = log(R_tar.T @ R_cur)。"""
    dp = T_current[:3, 3] - T_target[:3, 3]
    R_err = T_target[:3, :3].T @ T_current[:3, :3]
    rotvec = R.from_matrix(R_err).as_rotvec()
    return dp, rotvec


def validate_orientation_mode(orientation_mode):
    """校验并归一化姿态代价模式。"""
    orientation_mode = str(orientation_mode).lower()
    if orientation_mode not in ORIENTATION_MODES:
        raise ValueError(
            f"Unknown orientation_mode: {orientation_mode}. "
            f"Expected one of {ORIENTATION_MODES}"
        )
    return orientation_mode


def validate_transform4x4(T, name="T_target"):
    """校验 4x4 刚体变换（有限、末行 [0, 0, 0, 1]、旋转块正交且 det=1）。"""
    T = np.asarray(T, dtype=float)
    if T.shape != (4, 4):
        raise ValueError(f"{name} must be a 4x4 matrix, got shape {T.shape}")
    if not np.all(np.isfinite(T)):
        raise ValueError(f"{name} contains non-finite values")
    rot = T[:3, :3]
    if not (np.abs(T[3] - _RIGID_BOTTOM_ROW) <= RIGID_BOTTOM_ROW_TOL).all():
        raise ValueError(f"{name} bottom row must be [0, 0, 0, 1]")
    if not (
        (np.abs(rot.T @ rot - _EYE3) <= RIGID_ROTATION_TOL).all()
        and abs(np.linalg.det(rot) - 1.0) <= RIGID_ROTATION_TOL
    ):
        raise ValueError(f"{name} rotation block is not a proper rotation matrix")
    return T


def validate_q_vector(q, n, name="q_prev"):
    """校验关节向量长度，并返回 float ndarray。"""
    q = np.asarray(q, dtype=float)
    if q.shape != (n,):
        raise ValueError(f"{name} must have {n} entries, got shape {q.shape}")
    if not np.all(np.isfinite(q)):
        raise ValueError(f"{name} contains non-finite values")
    return q


def normalize_cone_half_angle(cone_half_angle=None):
    """解析默认锥角半角，并校验其位于 (0, pi)。"""
    if cone_half_angle is None:
        cone_half_angle = TOOL_CONE_HALF_ANGLE
    cone_half_angle = float(cone_half_angle)
    if not np.isfinite(cone_half_angle):
        raise ValueError("cone_half_angle must be finite")
    if cone_half_angle <= 0.0 or cone_half_angle >= np.pi:
        raise ValueError("cone_half_angle must be in radians and within (0, pi)")
    return cone_half_angle


def closest_direction_on_cone(
    d_current,
    d_target,
    cone_half_angle=TOOL_CONE_HALF_ANGLE,
):
    """返回锥内或锥边界上最接近当前方向的允许方向。"""
    d_current = normalize(d_current)
    d_target = normalize(d_target)
    cone_half_angle = normalize_cone_half_angle(cone_half_angle)

    c = float(np.clip(np.dot(d_current, d_target), -1.0, 1.0))
    angle = float(np.arccos(c))
    violation = max(0.0, angle - cone_half_angle)

    if violation <= 0.0:
        return d_current.copy(), angle, 0.0, True

    v = d_current - c * d_target
    n = np.linalg.norm(v)

    if n < 1e-12:
        tmp = np.array([1.0, 0.0, 0.0], dtype=float)
        if abs(np.dot(tmp, d_target)) > 0.9:
            tmp = np.array([0.0, 1.0, 0.0], dtype=float)
        u = normalize(np.cross(d_target, tmp))
    else:
        u = v / n

    d_boundary = np.cos(cone_half_angle) * d_target + np.sin(cone_half_angle) * u
    return normalize(d_boundary), angle, violation, False


def tool_direction_from_T(T, tool_axis=TOOL_AXIS_X):
    """返回在位姿父系中表示的工具轴方向。"""
    axis = normalize(tool_axis)
    return normalize(T[:3, :3] @ axis)


def tool_direction_residual(T_current, T_target, tool_axis=TOOL_AXIS_X):
    """工具方向残差 (d_cur - d_tar)。"""
    d_current = tool_direction_from_T(T_current, tool_axis)
    d_target = tool_direction_from_T(T_target, tool_axis)
    return d_current - d_target, d_current, d_target


def tool_direction_angle_error(T_current, T_target, tool_axis=TOOL_AXIS_X):
    """返回当前和目标工具方向之间的夹角 [rad]。"""
    d_current = tool_direction_from_T(T_current, tool_axis)
    d_target = tool_direction_from_T(T_target, tool_axis)
    c = float(np.clip(np.dot(d_current, d_target), -1.0, 1.0))
    return float(np.arccos(c))


def tool_cone_residual(
    T_current,
    T_target,
    tool_axis=TOOL_AXIS_X,
    cone_half_angle=TOOL_CONE_HALF_ANGLE,
):
    """用于最小二乘的锥形姿态残差。"""
    d_current = tool_direction_from_T(T_current, tool_axis)
    d_target = tool_direction_from_T(T_target, tool_axis)
    d_closest, angle, violation, inside = closest_direction_on_cone(
        d_current=d_current,
        d_target=d_target,
        cone_half_angle=cone_half_angle,
    )
    return (
        d_current - d_closest,
        d_current,
        d_target,
        d_closest,
        angle,
        violation,
        inside,
    )


def tool_cone_angle_error(
    T_current,
    T_target,
    tool_axis=TOOL_AXIS_X,
    cone_half_angle=TOOL_CONE_HALF_ANGLE,
):
    """返回锥形约束超出量和原始方向角 [rad]。"""
    raw_angle = tool_direction_angle_error(
        T_current=T_current,
        T_target=T_target,
        tool_axis=tool_axis,
    )
    cone_half_angle = normalize_cone_half_angle(cone_half_angle)
    violation = max(0.0, raw_angle - cone_half_angle)
    return float(violation), float(raw_angle)


def project_to_plane(v, normal):
    """把向量正交投影到以 normal 为法向的平面。"""
    normal = normalize(normal)
    return v - np.dot(v, normal) * normal


def sew_angle_from_points(S, E, W, ref_dir):
    """根据肩(S)、肘(E)、腕(W)三点计算绕 S->W 轴的带符号臂角 ψ。"""
    S = np.asarray(S, dtype=float)
    E = np.asarray(E, dtype=float)
    W = np.asarray(W, dtype=float)
    ref_dir = np.asarray(ref_dir, dtype=float)

    sw_axis = normalize(W - S)
    elbow_vec = normalize(project_to_plane(E - S, sw_axis))
    ref_vec = normalize(project_to_plane(ref_dir, sw_axis))

    psi = np.arctan2(
        np.dot(sw_axis, np.cross(ref_vec, elbow_vec)),
        np.dot(ref_vec, elbow_vec),
    )
    return float(wrap_to_pi(psi))


def target_frames(
    T_target,
    frame="local",
    T_world_from_base=None,
    T_base_from_world=None,
):
    """在求解器基座系与世界系之间转换目标位姿。"""
    T_target = validate_transform4x4(T_target)
    frame = str(frame).lower()
    if frame == "local":
        if T_world_from_base is None:
            raise ValueError("frame='local' requires T_world_from_base")
        T_target_local = T_target.copy()
        T_target_world = (
            validate_transform4x4(T_world_from_base, "T_world_from_base")
            @ T_target_local
        )
    elif frame == "world":
        if T_base_from_world is None:
            raise ValueError("frame='world' requires T_base_from_world")
        T_target_world = T_target.copy()
        T_target_local = (
            validate_transform4x4(T_base_from_world, "T_base_from_world")
            @ T_target_world
        )
    else:
        raise ValueError(f"Unknown frame: {frame}")
    return T_target_local, T_target_world


# ==============================================================================
# 4. 关节限位、余量、周期分支与距离
# ==============================================================================

def clip_to_bounds(q, lower, upper, eps=1e-9):
    """把 q 裁剪到 [lower + eps, upper - eps]；跨度 <= 2*eps 取中点。"""
    q = np.asarray(q, dtype=float)
    lower_arr = np.asarray(lower, dtype=float)
    upper_arr = np.asarray(upper, dtype=float)
    lo = lower_arr + eps
    hi = upper_arr - eps
    tight = lo > hi
    if tight.any():
        mid = 0.5 * (lower_arr + upper_arr)
        lo = np.where(tight, mid, lo)
        hi = np.where(tight, mid, hi)
    return np.minimum(np.maximum(q, lo), hi)


def wrap_to_pi(x):
    """把角度归一到 [-pi, pi)。"""
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def joint_dist(q, q_ref):
    """关节空间的标准欧氏距离。"""
    return float(np.linalg.norm(np.asarray(q) - np.asarray(q_ref)))


def joint_difference(q, ref, periodic_mask=None):
    """逐关节差 q - ref；periodic_mask 关节取短弧差，其余普通差。"""
    q = np.asarray(q, dtype=float)
    ref = np.asarray(ref, dtype=float)
    diff = q - ref
    if periodic_mask is not None and np.any(periodic_mask):
        diff = np.where(
            np.asarray(periodic_mask, dtype=bool), wrap_to_pi(diff), diff
        )
    return diff


def joint_dist_periodic(q, ref, periodic_mask=None):
    """在可选短弧归一后返回关节空间距离。"""
    return float(np.linalg.norm(joint_difference(q, ref, periodic_mask)))


def select_closest_bounded_branch(
    q, reference, lower, upper, multi_turn_mask=None
):
    """在合法分支 lower <= q + 2πk <= upper 中选择最接近 reference 的角度。"""
    q_arr = np.asarray(q, dtype=float)
    if q_arr.ndim != 1 or not np.all(np.isfinite(q_arr)):
        raise ValueError("q: expected a finite 1-D vector")
    ref_arr = np.asarray(reference, dtype=float)
    if ref_arr.shape != q_arr.shape or not np.all(np.isfinite(ref_arr)):
        raise ValueError("reference: expected a finite vector matching q")
    if multi_turn_mask is None:
        return q_arr.copy()
    mask = np.asarray(multi_turn_mask, dtype=bool)
    if mask.shape != q_arr.shape:
        raise ValueError("multi_turn_mask shape mismatch")
    if not np.any(mask):
        return q_arr.copy()

    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    out = q_arr.copy()
    two_pi = 2.0 * np.pi
    eps = 1e-9

    for i in np.nonzero(mask)[0]:
        qi = float(q_arr[i])
        ri = float(ref_arr[i])
        li = float(lo[i])
        ui = float(hi[i])

        if not np.isfinite(li) and not np.isfinite(ui):
            out[i] = ri + ((qi - ri + np.pi) % two_pi - np.pi)
            continue

        k_min = (
            math.ceil((li - eps - qi) / two_pi)
            if np.isfinite(li)
            else round((ri - qi) / two_pi) - 2
        )
        k_max = (
            math.floor((ui + eps - qi) / two_pi)
            if np.isfinite(ui)
            else round((ri - qi) / two_pi) + 2
        )

        best_val = qi
        best_dist = abs(qi - ri) if (li - eps <= qi <= ui + eps) else math.inf
        for k in range(int(k_min), int(k_max) + 1):
            cand = qi + k * two_pi
            if li - eps <= cand <= ui + eps:
                cand_clipped = float(np.clip(cand, li, ui))
                dist = abs(cand_clipped - ri)
                if dist < best_dist - 1e-12:
                    best_dist = dist
                    best_val = cand_clipped
        out[i] = best_val

    return out


def bounded_joint_difference(a, b, lower, upper, multi_turn_mask=None):
    """计算 a 相对 b 考虑周期分支后的最小等价差（用于去重）。"""
    a_branch = select_closest_bounded_branch(
        a, b, lower, upper, multi_turn_mask=multi_turn_mask
    )
    return a_branch - np.asarray(b, dtype=float)


def multi_turn_mask_from_chain(chain, lower, upper):
    """识别 continuous 关节或跨度 >= 2π 的 revolute 关节。"""
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    mask = np.zeros(len(chain.active_joints), dtype=bool)
    for i, aj in enumerate(chain.active_joints):
        jtype = aj.get("type")
        if jtype == "continuous":
            mask[i] = True
        elif (
            jtype == "revolute"
            and np.isfinite(lower[i]) and np.isfinite(upper[i])
            and (upper[i] - lower[i]) >= 2.0 * np.pi - 1e-6
        ):
            mask[i] = True
    return mask


def shift_to_nearest_branch(q, anchor, lower, upper, periodic_mask):
    """对周期关节做限位内等价角分支归一，返回距 anchor 最近的表示。"""
    q = np.asarray(q, dtype=float)
    mask = np.asarray(periodic_mask, dtype=bool)
    q_out = select_closest_bounded_branch(
        q, anchor, lower, upper, multi_turn_mask=mask
    )

    k_used = 0
    if np.any(mask):
        idx = int(np.nonzero(mask)[0][0])
        k_used = int(round((q_out[idx] - q[idx]) / (2.0 * np.pi)))
    shifted = not np.array_equal(q_out, q)
    return q_out, k_used, shifted


def joint_margin(q, lower, upper):
    """返回最小双侧限位余量、对应关节下标以及全部余量向量。"""
    q = np.asarray(q, dtype=float)
    margin_each = np.minimum(q - lower, upper - q)
    idx = int(np.argmin(margin_each))
    return float(margin_each[idx]), idx, margin_each


def joint_range_scale(lower, upper):
    """返回有限关节跨度；无界或零跨度关节使用 1.0。"""
    r = upper - lower
    scale = np.ones_like(r, dtype=float)
    good = np.isfinite(r) & (r > 1e-9)
    scale[good] = r[good]
    return scale


def joint_center_from_bounds(lower, upper):
    """返回关节范围中心；无界关节用 0。"""
    center = np.zeros_like(lower, dtype=float)
    good = np.isfinite(lower) & np.isfinite(upper)
    center[good] = 0.5 * (lower[good] + upper[good])
    return center


def joint_limit_penalty(q, lower, upper, safe_margin=SAFE_MARGIN, margin_each=None):
    """返回接近限位关节的标量二次 hinge 惩罚。"""
    if margin_each is None:
        _, _, margin_each = joint_margin(q, lower, upper)
    x = np.maximum(0.0, safe_margin - margin_each) / safe_margin
    return float(np.sum(x * x))


def unique_seed_list(seeds, tol=1e-8):
    """按输入顺序去重有限种子向量。"""
    unique = []
    for s in seeds:
        s = np.asarray(s, dtype=float)
        if not np.all(np.isfinite(s)):
            continue
        if not any(np.linalg.norm(s - u) < tol for u in unique):
            unique.append(s.copy())
    return unique


# ==============================================================================
# 5. URDF 模型解析与 SerialChain 串联运动学链
# ==============================================================================

def get_sew_joints(chain=None, s_joint=None, e_joint=None, w_joint=None):
    """获取肩、肘、腕关节点名称，未指定时按链属性兜底。"""
    s = s_joint or getattr(chain, "s_joint", None) or S_JOINT
    e = e_joint or getattr(chain, "e_joint", None) or E_JOINT
    w = w_joint or getattr(chain, "w_joint", None) or W_JOINT
    return s, e, w


def _parse_vec(s):
    """解析空格分隔的浮点数字符串为一维数组。"""
    return np.array([float(x) for x in s.split()], dtype=float)


def get_fixed_joint_origin(urdf_path, parent_link, child_link):
    """从 URDF 检索连接指定父子连杆的 fixed 关节原点 (xyz, rpy)。"""
    root = ET.parse(urdf_path).getroot()
    for joint in root.findall("joint"):
        if joint.attrib.get("type", "") != "fixed":
            continue
        parent_elem = joint.find("parent")
        child_elem = joint.find("child")
        if parent_elem is None or child_elem is None:
            continue
        parent = parent_elem.attrib["link"]
        child = child_elem.attrib["link"]
        if parent != parent_link or child != child_link:
            continue
        origin_elem = joint.find("origin")
        if origin_elem is None:
            return np.zeros(3), np.zeros(3)
        return (
            _parse_vec(origin_elem.attrib.get("xyz", "0 0 0")),
            _parse_vec(origin_elem.attrib.get("rpy", "0 0 0")),
        )
    return None


def get_urdf_root_link(urdf_path):
    """检索 URDF 拓扑树中唯一的根连杆。"""
    root = ET.parse(urdf_path).getroot()
    links = {
        link.attrib.get("name")
        for link in root.findall("link")
        if link.attrib.get("name")
    }
    child_links = {
        joint.find("child").attrib["link"]
        for joint in root.findall("joint")
        if joint.find("child") is not None
    }
    roots = sorted(list(links - child_links))
    if len(roots) != 1:
        raise RuntimeError(f"Expected exactly one URDF root link, got {roots}")
    return roots[0]


def find_chain(urdf_path, base_link, ee_link):
    """使用 BFS 从 URDF 中提取 base_link 到 ee_link 的串联关节链。"""
    root = ET.parse(urdf_path).getroot()
    children = defaultdict(list)
    joint_by_edge = {}

    for joint in root.findall("joint"):
        name = joint.attrib["name"]
        joint_type = joint.attrib.get("type", "")
        parent = joint.find("parent").attrib["link"]
        child = joint.find("child").attrib["link"]

        origin_elem = joint.find("origin")
        xyz = (
            _parse_vec(origin_elem.attrib.get("xyz", "0 0 0"))
            if origin_elem is not None
            else np.zeros(3)
        )
        rpy = (
            _parse_vec(origin_elem.attrib.get("rpy", "0 0 0"))
            if origin_elem is not None
            else np.zeros(3)
        )

        axis_elem = joint.find("axis")
        axis = (
            _parse_vec(axis_elem.attrib.get("xyz", "0 0 1"))
            if axis_elem is not None
            else np.array([0.0, 0.0, 1.0])
        )

        limit_elem = joint.find("limit")
        if limit_elem is not None:
            lower = float(limit_elem.attrib.get("lower", "-inf"))
            upper = float(limit_elem.attrib.get("upper", "inf"))
        else:
            lower = -np.inf
            upper = np.inf

        joint_data = {
            "name": name,
            "type": joint_type,
            "parent": parent,
            "child": child,
            "xyz": xyz,
            "rpy": rpy,
            "axis": axis,
            "lower": lower,
            "upper": upper,
        }
        children[parent].append(child)
        joint_by_edge[(parent, child)] = joint_data

    queue = deque([(base_link, [])])
    visited = set()

    while queue:
        link, path = queue.popleft()
        if link == ee_link:
            return path
        if link in visited:
            continue
        visited.add(link)
        for child in children[link]:
            queue.append((child, path + [joint_by_edge[(link, child)]]))

    raise RuntimeError(f"Cannot find chain from {base_link} to {ee_link}")


class SerialChain:
    """URDF 串联机器人运动学链。"""

    def __init__(
        self,
        urdf_path,
        base_link,
        ee_link,
        s_joint=None,
        e_joint=None,
        w_joint=None,
    ):
        self.urdf_path = urdf_path
        self.base_link = base_link
        self.ee_link = ee_link
        self.s_joint = s_joint
        self.e_joint = e_joint
        self.w_joint = w_joint
        self.joints = find_chain(urdf_path, base_link, ee_link)
        self.active_joints = [
            joint
            for joint in self.joints
            if joint["type"] in ("revolute", "continuous", "prismatic")
        ]
        self._cached_bounds = None

        if len(self.active_joints) not in (7, 9):
            print(
                "WARNING: active joint count is not 7 (arm) or 9 "
                "(waist+arm):",
                len(self.active_joints),
            )

    @property
    def sew_joints(self):
        """返回 (s_joint, e_joint, w_joint) 名称元组。"""
        return get_sew_joints(self)

    def bounds(self):
        """返回所有活动关节的限位上下界数组 (lower, upper)。"""
        if self._cached_bounds is None:
            lower = []
            upper = []
            for joint in self.active_joints:
                if joint["type"] == "continuous":
                    lower.append(-np.pi)
                    upper.append(np.pi)
                else:
                    lower.append(joint["lower"])
                    upper.append(joint["upper"])
            self._cached_bounds = (
                np.array(lower, dtype=float),
                np.array(upper, dtype=float),
            )
        return self._cached_bounds[0].copy(), self._cached_bounds[1].copy()

    def fk(self, q):
        """返回末端执行器（ee_link）的 4x4 位姿变换。"""
        q = np.asarray(q, dtype=float)
        if len(q) != len(self.active_joints):
            raise ValueError(
                f"q size mismatch: got {len(q)}, "
                f"expected {len(self.active_joints)}"
            )
        T = np.eye(4)
        motions = self._joint_motions(q)
        for joint, Tmotion in zip(self.joints, motions):
            T = T @ self._origin_T(joint)
            if Tmotion is not None:
                T = T @ Tmotion
        return T

    def fk_all(self, q):
        """计算所有连杆及关节原点相对于基座的 4x4 齐次变换矩阵。"""
        q = np.asarray(q, dtype=float)
        if len(q) != len(self.active_joints):
            raise ValueError(
                f"q size mismatch: got {len(q)}, "
                f"expected {len(self.active_joints)}"
            )

        T = np.eye(4)
        motions = self._joint_motions(q)

        link_T = {self.base_link: T.copy()}
        joint_origin_T = {}

        for joint, Tmotion in zip(self.joints, motions):
            Tj_origin = T @ self._origin_T(joint)
            joint_origin_T[joint["name"]] = Tj_origin.copy()

            if Tmotion is None:  # fixed
                Tmotion = np.eye(4)

            T = Tj_origin @ Tmotion
            link_T[joint["child"]] = T.copy()

        return link_T, joint_origin_T

    def _origin_T(self, joint):
        """按 (xyz, rpy) 字节缓存关节原点变换矩阵。"""
        key = (
            np.asarray(joint["xyz"], dtype=float).tobytes(),
            np.asarray(joint["rpy"], dtype=float).tobytes(),
        )
        cache = self.__dict__.setdefault("_origin_cache", {})
        T = cache.get(key)
        if T is None:
            T = T_origin(joint["xyz"], joint["rpy"])
            T.flags.writeable = False
            cache[key] = T
        return T

    def _unit_axis(self, axis):
        """按原始转轴字节缓存归一化轴。"""
        key = np.asarray(axis, dtype=float).tobytes()
        cache = self.__dict__.setdefault("_axis_cache", {})
        unit = cache.get(key)
        if unit is None:
            unit = normalize(axis)
            unit.flags.writeable = False
            cache[key] = unit
        return unit

    def _joint_motions(self, q):
        """批量计算与 self.joints 对齐的局部运动变换。"""
        motions = [None] * len(self.joints)
        rows, rotvecs = [], []
        active_idx = 0
        for k, joint in enumerate(self.joints):
            jtype = joint["type"]
            if jtype in ("revolute", "continuous"):
                rotvecs.append(self._unit_axis(joint["axis"]) * q[active_idx])
                rows.append(k)
                active_idx += 1
            elif jtype == "prismatic":
                motions[k] = T_prismatic(joint["axis"], q[active_idx])
                active_idx += 1
            elif jtype != "fixed":
                raise NotImplementedError(f"Unsupported joint type: {jtype}")
        if rotvecs:
            mats = R.from_rotvec(np.array(rotvecs)).as_matrix()
            for k, Rm in zip(rows, mats):
                T = np.eye(4)
                T[:3, :3] = Rm
                motions[k] = T
        return motions

    def fk_pose_and_joints(self, q, joint_names):
        """单次前向递推同时获取末端位姿与指定关节点原点位姿。"""
        q = np.asarray(q, dtype=float)
        if len(q) != len(self.active_joints):
            raise ValueError(
                f"q size mismatch: got {len(q)}, "
                f"expected {len(self.active_joints)}"
            )

        wanted = set(joint_names)
        joint_T = {}
        T = np.eye(4)
        motions = self._joint_motions(q)

        for joint, Tmotion in zip(self.joints, motions):
            T = T @ self._origin_T(joint)
            if joint["name"] in wanted:
                joint_T[joint["name"]] = T.copy()

            if Tmotion is not None:
                T = T @ Tmotion

        missing = wanted.difference(joint_T)
        if missing:
            raise KeyError(f"Joints not found in chain: {sorted(missing)}")
        return T, joint_T

    def joint_position(self, q, joint_name):
        """获取指定关节原点在基座系下的三维坐标。"""
        _, joint_T = self.fk_pose_and_joints(q, (joint_name,))
        return joint_T[joint_name][:3, 3]

    def sew_positions(self, q, s_joint=None, e_joint=None, w_joint=None):
        """单次前向递推获取肩、肘、腕三点在基座系下的三维坐标 (S, E, W)。"""
        s, e, w = get_sew_joints(self, s_joint, e_joint, w_joint)
        _, joint_T = self.fk_pose_and_joints(q, (s, e, w))
        return joint_T[s][:3, 3], joint_T[e][:3, 3], joint_T[w][:3, 3]

    def psi(self, q, s_joint=None, e_joint=None, w_joint=None, ref_dir=REF_DIR):
        """计算给定关节角下的 Shoulder-Elbow-Wrist (SEW) 臂角 ψ。"""
        s, e, w = get_sew_joints(self, s_joint, e_joint, w_joint)
        _, joint_T = self.fk_pose_and_joints(q, (s, e, w))
        return sew_angle_from_points(
            joint_T[s][:3, 3],
            joint_T[e][:3, 3],
            joint_T[w][:3, 3],
            ref_dir,
        )

    def estimate_max_reach(self, anchor_joint_name):
        """估算从指定锚点关节原点到末端的最大理论伸展距离。"""
        return estimate_chain_max_reach_from_joint(self, anchor_joint_name)

    def origin_position_at_zero(self, joint_name):
        """返回零位姿下指定关节原点在基座系下的三维坐标。"""
        return joint_origin_position_at_zero(self, joint_name)


def estimate_chain_max_reach_from_joint(chain, anchor_joint_name):
    """估算串联链从指定锚点关节原点到末端的最大理论伸展距离。"""
    reach = 0.0
    found_anchor = False

    for joint in chain.joints:
        if joint["name"] == anchor_joint_name:
            found_anchor = True
            continue
        if found_anchor:
            xyz = np.asarray(joint.get("xyz", np.zeros(3)), dtype=float)
            reach += float(np.linalg.norm(xyz))

    if not found_anchor:
        raise KeyError(f"Anchor joint {anchor_joint_name} not found in chain.")

    return reach


def joint_origin_position_at_zero(chain, joint_name):
    """返回零位姿下指定关节原点在基座系下的三维坐标。"""
    q0 = np.zeros(len(chain.active_joints), dtype=float)
    return chain.joint_position(q0, joint_name).copy()


# ==============================================================================
# 6. 候选解评估、打分与排序
# ==============================================================================

def _phi_sort_value(c):
    """提取用于排序的臂角偏离值（过滤微弱扰动）。"""
    x = c.get("phi_abs", np.inf)
    if not np.isfinite(x):
        return np.inf
    return 0.0 if x < PHI_TIE_TOL else float(x)


def candidate_sort_key(c):
    """8 级分层候选解排序键：位姿达标 > 代价 > 臂角差 > 平滑 > 余量 > 预备位距离 > 位置误差 > 姿态误差。"""
    return (
        not c.get("pose_ok", False),
        c.get("cost", c.get("cost_no_torque", np.inf)),
        _phi_sort_value(c),
        c.get("dq_prev", np.inf),
        -c.get("margin", -np.inf),
        c.get("dq_ready", np.inf),
        c.get("pos_err", np.inf),
        c.get("ori_err", np.inf),
    )


def candidate_status(c):
    """根据候选解字典的评估字段映射为对应的公共状态字符串。"""
    if c is None:
        return "NO_RESULT"
    if not c.get("pose_ok", False):
        return "NO_VALID_POSE"
    if c.get("margin", -np.inf) < SAFE_MARGIN:
        return "NEAR_LIMIT"
    return "SUCCESS"


def _waist_indices(chain, n):
    """识别躯干腰部关节在活动关节向量中的下标。"""
    cached = getattr(chain, "_waist_indices_cache", None)
    if cached is not None and len(getattr(chain, "active_joints", ())) == n:
        return cached

    joints = getattr(chain, "active_joints", None)
    if joints is not None and len(joints) == n:
        names = [j.get("name") if isinstance(j, dict) else None for j in joints]
        if all(name is not None for name in names):
            idx = np.array(
                [i for i, name in enumerate(names) if name in WAIST_JOINT_NAMES],
                dtype=np.intp,
            )
            if hasattr(chain, "__dict__"):
                chain._waist_indices_cache = idx
            return idx
    if n == len(WAIST_JOINT_NAMES) + 7:
        return np.arange(len(WAIST_JOINT_NAMES), dtype=np.intp)
    return np.zeros(0, dtype=np.intp)


def score_candidate(
    chain,
    q,
    T_target,
    psi_ready,
    q_prev,
    q_ready,
    lower,
    upper,
    orientation_mode="full",
    tool_axis=None,
    tool_cone_half_angle=None,
    apply_waist_return=False,
    periodic_mask=None,
):
    """评估单个候选解并返回位姿残差、各项代价分量与诊断指标字典。"""
    q = np.asarray(q, dtype=float)
    orientation_mode = validate_orientation_mode(orientation_mode)
    tool_axis = TOOL_AXIS_X if tool_axis is None else normalize(tool_axis)
    psi_ready_ok = psi_ready is not None and bool(np.isfinite(psi_ready))
    tool_cone_half_angle_val = normalize_cone_half_angle(tool_cone_half_angle)

    psi = np.nan
    phi = np.nan
    pos_err = np.inf
    ori_err = np.inf
    tool_dir_angle_err = np.nan
    tool_cone_violation = np.nan
    Tq = None

    try:
        s_joint, e_joint, w_joint = get_sew_joints(chain)
        Tq, joint_T = chain.fk_pose_and_joints(q, (s_joint, e_joint, w_joint))

        dp = Tq[:3, 3] - T_target[:3, 3]
        pos_err = float(np.linalg.norm(dp))

        if orientation_mode == "full":
            _, drot = pose_error(Tq, T_target)
            ori_err = float(np.linalg.norm(drot))
        elif orientation_mode == "tool_dir":
            tool_dir_angle_err = tool_direction_angle_error(
                T_current=Tq, T_target=T_target, tool_axis=tool_axis
            )
            ori_err = tool_dir_angle_err
        elif orientation_mode == "tool_cone":
            tool_cone_violation, tool_dir_angle_err = tool_cone_angle_error(
                T_current=Tq,
                T_target=T_target,
                tool_axis=tool_axis,
                cone_half_angle=tool_cone_half_angle_val,
            )
            ori_err = tool_cone_violation

        psi = float(sew_angle_from_points(
            joint_T[s_joint][:3, 3],
            joint_T[e_joint][:3, 3],
            joint_T[w_joint][:3, 3],
            REF_DIR,
        ))
        phi = float(wrap_to_pi(psi - psi_ready)) if psi_ready_ok else np.nan
    except Exception:
        pass

    margin, margin_idx, margin_each = joint_margin(q, lower, upper)
    dq_prev = joint_dist_periodic(q, q_prev, periodic_mask)
    dq_ready = joint_dist_periodic(q, q_ready, periodic_mask)

    waist_dev = float("nan")
    waist_idx = _waist_indices(chain, len(q))
    if waist_idx.size:
        waist_mask = (
            np.asarray(periodic_mask, dtype=bool)[waist_idx]
            if periodic_mask is not None
            else None
        )
        waist_dev = joint_dist_periodic(
            q[waist_idx], np.asarray(q_ready, dtype=float)[waist_idx], waist_mask
        )

    limit_pen = joint_limit_penalty(
        q, lower, upper, safe_margin=SAFE_MARGIN, margin_each=margin_each
    )
    pose_ok = (pos_err <= POS_TOL) and (ori_err <= ORI_TOL)

    if np.isfinite(phi) or (not psi_ready_ok and Tq is not None):
        psi_cost = W_NOM * phi * phi if np.isfinite(phi) else 0.0
        smooth_cost = W_SMOOTH * dq_prev * dq_prev
        limit_cost = W_LIMIT * limit_pen
        waist_cost = (
            W_WAIST_RETURN * waist_dev * waist_dev
            if apply_waist_return and np.isfinite(waist_dev)
            else 0.0
        )
        cost_no_torque = psi_cost + smooth_cost + limit_cost + waist_cost
    else:
        psi_cost = np.inf
        smooth_cost = np.inf
        limit_cost = np.inf
        waist_cost = np.inf
        cost_no_torque = np.inf

    res = {
        "q": q.copy(),
        "pos_err": pos_err,
        "ori_err": ori_err,
        "orientation_mode": orientation_mode,
        "tool_axis": None if tool_axis is None else np.asarray(tool_axis, dtype=float).copy(),
        "tool_dir_angle_err": float(tool_dir_angle_err),
        "tool_dir_angle_err_deg": (
            float(np.rad2deg(tool_dir_angle_err))
            if np.isfinite(tool_dir_angle_err)
            else np.nan
        ),
        "tool_cone_half_angle": float(tool_cone_half_angle_val),
        "tool_cone_half_angle_deg": float(np.rad2deg(tool_cone_half_angle_val)),
        "tool_cone_violation": float(tool_cone_violation),
        "tool_cone_violation_deg": (
            float(np.rad2deg(tool_cone_violation))
            if np.isfinite(tool_cone_violation)
            else np.nan
        ),
        "psi": psi,
        "phi": phi,
        "phi_abs": abs(phi) if np.isfinite(phi) else np.inf,
        "phi_deg": float(np.rad2deg(phi)) if np.isfinite(phi) else np.nan,
        "margin": margin,
        "margin_idx": margin_idx,
        "dq_prev": dq_prev,
        "dq_ready": dq_ready,
        "waist_dev": waist_dev,
        "limit_penalty": limit_pen,
        "psi_cost": float(psi_cost),
        "smooth_cost": float(smooth_cost),
        "limit_cost": float(limit_cost),
        "waist_cost": float(waist_cost),
        "cost_no_torque": float(cost_no_torque),
        "torque_weighted_cost": 0.0,
        "cost": float(cost_no_torque),
        "pose_ok": pose_ok,
    }
    res["status"] = candidate_status(res)
    return res


def has_valid_torque(c):
    """判断候选解是否具备有限的力矩代价和力矩范数。"""
    return (
        c is not None
        and "torque_cost" in c
        and "torque_l2" in c
        and np.isfinite(c.get("torque_cost", np.inf))
        and np.isfinite(c.get("torque_l2", np.inf))
    )


def update_candidate_final_cost_with_torque(candidate):
    """把 solver 计算出的力矩代价合并进共享候选代价。"""
    if candidate is None:
        return

    cost_no_torque = candidate.get(
        "cost_no_torque", candidate.get("cost", np.inf)
    )
    candidate["cost_no_torque"] = float(cost_no_torque)

    if not has_valid_torque(candidate):
        candidate["torque_weighted_cost"] = np.inf
        candidate["cost"] = np.inf
        return

    torque_cost = float(candidate["torque_cost"])
    torque_weighted_cost = float(W_TORQUE * torque_cost)
    candidate["torque_weighted_cost"] = torque_weighted_cost
    candidate["cost"] = float(cost_no_torque + torque_weighted_cost)


def select_best_candidate(candidates, use_torque=True):
    """选择最优候选解并返回 (best, public_status)。"""
    if len(candidates) == 0:
        return None, "NO_RESULT"

    valid = [c for c in candidates if c.get("pose_ok", False)]

    if len(valid) == 0:
        best = dict(
            sorted(
                candidates,
                key=lambda c: (
                    c.get("pos_err", np.inf),
                    c.get("ori_err", np.inf),
                    _phi_sort_value(c),
                    c.get("dq_prev", np.inf),
                    -c.get("margin", -np.inf),
                ),
            )[0]
        )
        best["selected_reason"] = "no_valid_pose_use_best_pose_error"
        return best, "NO_VALID_POSE"

    if use_torque:
        torque_valid = [c for c in valid if has_valid_torque(c)]
        pool = torque_valid if len(torque_valid) > 0 else valid
    else:
        pool = valid

    pool.sort(key=candidate_sort_key)
    best = dict(pool[0])
    best["selected_reason"] = "weighted_cost"
    best["selection_W_TORQUE"] = float(W_TORQUE)
    best["selection_cost_has_torque"] = bool(
        use_torque and has_valid_torque(best)
    )
    return best, candidate_status(best)
