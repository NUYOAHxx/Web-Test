"""共享几何、校验、限位与周期分支工具。

本模块刻意保持与具体求解器（solver）无关，只依赖 NumPy/SciPy。求解器可以基于它实现，
但这里不得反向导入求解器后端。
"""
import math

import numpy as np
from scipy.spatial.transform import Rotation as R


TOOL_AXIS_X = np.array([1.0, 0.0, 0.0], dtype=float)
TOOL_CONE_HALF_ANGLE = np.deg2rad(30.0)
ORIENTATION_MODES = ("full", "tool_dir", "tool_cone")
POS_TOL = 5e-4
ORI_TOL = 5e-4
SAFE_MARGIN = 0.02

PERIODIC_MIN_SPAN_RAD = 2.0 * np.pi + 1e-6


def normalize(v, eps=1e-12):
    """返回单位向量；拒绝含非有限分量或模长小于 ``eps`` 的向量。"""
    v = np.asarray(v, dtype=float)
    if not np.all(np.isfinite(v)):
        raise ValueError("vector contains non-finite values")
    n = np.linalg.norm(v)
    if n < eps:
        raise ValueError("zero norm vector")
    return v / n


def pose_error(T_current, T_target):
    """
    6 维位姿残差：
      dp = 当前位置 - 目标位置
      drot = log(R_target.T @ R_current)
    """
    dp = T_current[:3, 3] - T_target[:3, 3]
    R_err = T_target[:3, :3].T @ T_current[:3, :3]
    rotvec = R.from_matrix(R_err).as_rotvec()
    return dp, rotvec


def skew(v):
    """返回 3x3 反对称矩阵，满足 ``skew(a) @ b = a x b``。"""
    v = np.asarray(v, dtype=float)
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ], dtype=float)


def validate_orientation_mode(orientation_mode):
    """校验并归一化姿态代价模式。"""
    orientation_mode = str(orientation_mode).lower()

    if orientation_mode not in ORIENTATION_MODES:
        raise ValueError(
            f"Unknown orientation_mode: {orientation_mode}. "
            f"Expected one of {ORIENTATION_MODES}"
        )

    return orientation_mode


RIGID_BOTTOM_ROW_TOL = 1e-9
RIGID_ROTATION_TOL = 1e-6
_RIGID_BOTTOM_ROW = np.array([0.0, 0.0, 0.0, 1.0])
_EYE3 = np.eye(3)


def validate_transform4x4(T, name="T_target"):
    """校验 4x4 刚体齐次变换（有限、末行 [0, 0, 0, 1]、旋转块正交且行列式为 +1），返回 float ndarray。

    容差与 solver_dls.validate_transform 相同：末行 1e-9，R^T R = I 与 det(R) = 1 为 1e-6。
    """
    T = np.asarray(T, dtype=float)
    if T.shape != (4, 4):
        raise ValueError(f"{name} must be a 4x4 matrix, got shape {T.shape}")
    if not np.all(np.isfinite(T)):
        raise ValueError(f"{name} contains non-finite values")
    rot = T[:3, :3]
    # 判定与 np.allclose / np.isclose(rtol=0) 逐位等价（参考值有限时 isclose(x, y) ⇔ |x - y| <= atol）
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
        raise ValueError(
            f"{name} must have {n} entries, got shape {q.shape}"
        )
    if not np.all(np.isfinite(q)):
        raise ValueError(f"{name} contains non-finite values")
    return q


def normalize_cone_half_angle(cone_half_angle=None):
    """解析默认锥角半角，并校验其位于 ``(0, pi)``。"""
    if cone_half_angle is None:
        cone_half_angle = TOOL_CONE_HALF_ANGLE

    cone_half_angle = float(cone_half_angle)

    if not np.isfinite(cone_half_angle):
        raise ValueError("cone_half_angle must be finite")
    if cone_half_angle <= 0.0 or cone_half_angle >= np.pi:
        raise ValueError(
            "cone_half_angle must be in radians and within (0, pi)"
        )

    return cone_half_angle


def closest_direction_on_cone(
    d_current,
    d_target,
    cone_half_angle=TOOL_CONE_HALF_ANGLE,
):
    """
    返回锥内或锥边界上最接近当前方向的允许方向。

    锥体定义：
        中心轴 = d_target
        半锥角 = cone_half_angle

    若 d_current 已在锥内：
        最近方向 = d_current；violation = 0
    若在锥外：最近方向在从 d_target 指向 d_current 的大圆方向上，落在锥边界。
    """
    d_current = normalize(d_current)
    d_target = normalize(d_target)
    cone_half_angle = normalize_cone_half_angle(cone_half_angle)

    c = float(np.clip(np.dot(d_current, d_target), -1.0, 1.0))
    angle = float(np.arccos(c))
    violation = max(0.0, angle - cone_half_angle)

    if violation <= 0.0:
        return d_current.copy(), angle, 0.0, True

    # 计算垂直于目标轴、指向当前方向的分量。
    v = d_current - c * d_target
    n = np.linalg.norm(v)

    if n < 1e-12:
        # 反向接近时法向分量退化；选择一个稳定的正交方向。
        tmp = np.array([1.0, 0.0, 0.0], dtype=float)
        if abs(np.dot(tmp, d_target)) > 0.9:
            tmp = np.array([0.0, 1.0, 0.0], dtype=float)

        u = normalize(np.cross(d_target, tmp))
    else:
        u = v / n

    d_boundary = (
        np.cos(cone_half_angle) * d_target
        + np.sin(cone_half_angle) * u
    )

    return normalize(d_boundary), angle, violation, False


def clip_to_bounds(q, lower, upper, eps=1e-9):
    """把 ``q`` 裁剪到 ``[lower + eps, upper - eps]``；不重复校验限位。

    跨度 <= 2·eps 的关节（含 lower == upper 的锁定关节）取区间中点，保证结果不越出 [lower, upper]。
    """
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
    """把角度归一到 ``[-pi, pi)``。"""
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def joint_dist(q, q_ref):
    """返回关节空间的普通欧氏距离。"""
    return float(np.linalg.norm(np.asarray(q) - np.asarray(q_ref)))


def periodic_capable_mask(lower, upper, min_span_rad=PERIODIC_MIN_SPAN_RAD):
    """span >= min_span_rad 的关节可在限位内做一次 ±2π 等价平移。"""
    return (
        np.asarray(upper, dtype=float) - np.asarray(lower, dtype=float)
    ) >= min_span_rad


def joint_difference(q, ref, periodic_mask=None):
    """逐关节差 q-ref；periodic_mask 关节取 wrap_to_pi 短弧差，其余普通差。"""
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
    """在满足 lower <= q + 2πk <= upper (k ∈ Z) 的合法同伦分支中，
    为 multi_turn_mask 中的关节选择最接近 reference 的等价关节角。
    有限限位关节在合法 k 区间内精确搜索并把候选 clip 回 [lower, upper]；
    双侧无界关节按 reference 对齐 wrap。multi_turn_mask 为 None 时绝不根据
    限位跨度猜测周期性（直接返回原值拷贝），防止对无界或大跨度 prismatic
    平移关节误执行模 2π 运算。
    """
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
    """在考虑有限限位内合法 2πk 等价分支的前提下，计算 a 相对于 b 的
    最小等价差（用于候选解去重）。"""
    a_branch = select_closest_bounded_branch(
        a, b, lower, upper, multi_turn_mask=multi_turn_mask
    )
    return a_branch - np.asarray(b, dtype=float)


def multi_turn_mask_from_chain(chain, lower, upper):
    """active_multi_turn 等价判定：continuous 关节，或限位跨度 >= 2π 的
    revolute 关节（与求解器 pin_maps["active_multi_turn"] 语义一致）。
    prismatic 或小跨度 revolute 关节一律 False。"""
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


def shift_to_nearest_branch(q, anchor, lower, upper, periodic_mask, k_max=None):
    """对 periodic 关节做限位内等价角分支归一（基于 select_closest_bounded_branch
    的逐关节 k∈ℤ 搜索），返回距 anchor 最近的表示。

    k_max 参数保留仅为向后兼容，已不参与搜索（k∈ℤ 全搜索，限位天然限定范围）。

    Returns
    -------
    (q_out, k_used, shifted) : k_used 为第一个 mask 关节的平移圈数（诊断用途），
    未平移时为 0。
    """
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
    """返回最小双侧限位余量、对应关节下标以及全部余量。"""
    q = np.asarray(q, dtype=float)
    margin_each = np.minimum(q - lower, upper - q)
    idx = int(np.argmin(margin_each))
    return float(margin_each[idx]), idx, margin_each


def joint_margin_from_bounds(q, lower, upper):
    """返回最小限位余量和达到该余量的关节下标。"""
    margin, index, _ = joint_margin(q, lower, upper)
    return margin, index


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


def joint_limit_barrier(q, lower, upper, margin=0.10):
    """返回进入限位 ``margin`` 范围后激活的二次惩罚。"""
    q = np.asarray(q, dtype=float)

    dist_lower = q - lower
    dist_upper = upper - q
    dist = np.minimum(dist_lower, dist_upper)

    finite = np.isfinite(dist)

    x = np.zeros_like(q, dtype=float)
    x[finite] = (margin - dist[finite]) / margin

    return np.where(x > 0.0, x * x, 0.0)


def joint_limit_penalty(q, lower, upper, safe_margin=SAFE_MARGIN):
    """返回接近限位关节的标量二次 hinge 惩罚。"""
    _, _, margin_each = joint_margin(q, lower, upper)
    x = np.maximum(0.0, safe_margin - margin_each) / safe_margin
    return float(np.sum(x * x))


def joint_limit_barrier_jac(q, lower, upper, margin=0.10):
    """返回 :func:`joint_limit_barrier` 的对角雅可比。"""
    q = np.asarray(q, dtype=float)

    dist_lo = q - lower
    dist_hi = upper - q
    closer_lo = dist_lo <= dist_hi
    dist = np.minimum(dist_lo, dist_hi)

    d_dist = np.where(closer_lo, 1.0, -1.0)

    finite = np.isfinite(dist)
    x = np.zeros_like(q)
    x[finite] = np.maximum(0.0, (margin - dist[finite]) / margin)

    active = x > 0
    diag = np.zeros_like(q)
    mask = active & finite
    diag[mask] = 2.0 * x[mask] * (-1.0 / margin) * d_dist[mask]

    return np.diag(diag)


def rotx(a):
    """返回绕 X 轴的 3x3 主动旋转矩阵。"""
    c, s = np.cos(a), np.sin(a)
    return np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c],
    ], dtype=float)


def roty(a):
    """返回绕 Y 轴的 3x3 主动旋转矩阵。"""
    c, s = np.cos(a), np.sin(a)
    return np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c],
    ], dtype=float)


def rotz(a):
    """返回绕 Z 轴的 3x3 主动旋转矩阵。"""
    c, s = np.cos(a), np.sin(a)
    return np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1],
    ], dtype=float)


def rpy_to_R(rpy):
    """
    URDF 的 RPY 约定：
      R = Rz(yaw) * Ry(pitch) * Rx(roll)
    """
    roll, pitch, yaw = rpy
    return rotz(yaw) @ roty(pitch) @ rotx(roll)


def make_T(Rm=None, pxyz=None):
    """用可选旋转和平移构造 4x4 变换。"""
    T = np.eye(4)
    if Rm is not None:
        T[:3, :3] = Rm
    if pxyz is not None:
        T[:3, 3] = np.asarray(pxyz, dtype=float)
    return T


def T_origin(xyz, rpy):
    """用 xyz 和固定轴 rpy 构造 URDF joint origin 变换。"""
    return make_T(rpy_to_R(rpy), xyz)


def rot_axis(axis, theta):
    """返回绕归一化三维轴的旋转矩阵。"""
    axis = normalize(axis)
    return R.from_rotvec(axis * theta).as_matrix()


def T_revolute(axis, q):
    """返回转动关节的位移变换。"""
    return make_T(rot_axis(axis, q), np.zeros(3))


def T_prismatic(axis, q):
    """返回移动关节的位移变换。"""
    axis = normalize(axis)
    return make_T(np.eye(3), axis * q)


def make_T_from_xyz_rpy(xyz, rpy):
    """
    用 xyz 平移和 URDF 固定轴 RPY 构造 4x4 变换。
    URDF 约定：
        roll 绕固定 X 轴
        pitch 绕固定 Y 轴
        yaw 绕固定 Z 轴

    矩阵约定：
        R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    """
    T = np.eye(4)
    T[:3, :3] = rpy_to_R(rpy)
    T[:3, 3] = np.asarray(xyz, dtype=float)
    return T


def invert_T(T):
    """逆变换刚体变换；旋转块必须是正交矩阵。"""
    T = np.asarray(T, dtype=float)
    Tin = np.eye(4)
    Tin[:3, :3] = T[:3, :3].T
    Tin[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return Tin


def transform_point(T, pxyz):
    """用 4x4 变换变换一个三维点。"""
    p4 = np.ones(4, dtype=float)
    p4[:3] = np.asarray(pxyz, dtype=float)
    return (np.asarray(T, dtype=float) @ p4)[:3]


def make_translation_T(xyz):
    """返回 4x4 纯平移变换。"""
    T = np.eye(4)
    T[:3, 3] = np.asarray(xyz, dtype=float)
    return T


def tool_direction_from_T(T, tool_axis=TOOL_AXIS_X):
    """
    返回在位姿父系中表示的工具轴方向。

    ``tool_axis=[1,0,0]`` 表示 TCP 局部 +X。
    """
    axis = normalize(tool_axis)

    d = T[:3, :3] @ axis
    return normalize(d)


def tool_direction_residual(T_current, T_target, tool_axis=TOOL_AXIS_X):
    """
    仅约束工具方向的姿态残差。

    若该残差为零，表示选定的 TCP 局部轴指向目标方向；绕该轴的 roll 不受约束。
    """
    d_current = tool_direction_from_T(T_current, tool_axis)
    d_target = tool_direction_from_T(T_target, tool_axis)
    return d_current - d_target, d_current, d_target


def tool_direction_angle_error(T_current, T_target, tool_axis=TOOL_AXIS_X):
    """返回当前和目标工具方向之间的夹角，单位为弧度。"""
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
    """
    用于最小二乘的锥形姿态残差。

    在锥内残差为零，在锥边界保持连续。
    """
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
    """返回锥形约束超出量和原始方向角，均为弧度。"""
    raw_angle = tool_direction_angle_error(
        T_current=T_current,
        T_target=T_target,
        tool_axis=tool_axis,
    )
    cone_half_angle = normalize_cone_half_angle(cone_half_angle)
    violation = max(0.0, raw_angle - cone_half_angle)

    return float(violation), float(raw_angle)


def project_to_plane(v, normal):
    """把向量投影到以 ``normal`` 为法向、过原点的平面。"""
    normal = normalize(normal)
    return v - np.dot(v, normal) * normal


def sew_angle_from_points(S, E, W, ref_dir):
    """返回围绕 S->W 轴的带符号 S-E-W 臂角。"""
    S = np.asarray(S, dtype=float)
    E = np.asarray(E, dtype=float)
    W = np.asarray(W, dtype=float)
    ref_dir = np.asarray(ref_dir, dtype=float)

    sw_axis = normalize(W - S)

    elbow_vec = project_to_plane(E - S, sw_axis)
    elbow_vec = normalize(elbow_vec)

    ref_vec = project_to_plane(ref_dir, sw_axis)
    ref_vec = normalize(ref_vec)

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
    """
    在求解器基座系与世界系之间转换目标位姿。

    返回 ``(T_target_local, T_target_world)``。调用方还可基于其中一项推导 solver
    专用目标，例如 Pinocchio 会根据模型根是否等于机器人/世界根选择世界目标或
    求解器局部目标。frame="local" 需要 ``T_world_from_base``，frame="world" 需要
    ``T_base_from_world``；缺失时抛 ValueError（不再在矩阵乘法处抛 TypeError）。
    """
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


def unique_seed_list(seeds, tol=1e-8):
    """按输入顺序返回有限种子向量，并按 ``tol`` 去重。"""
    unique = []
    for s in seeds:
        s = np.asarray(s, dtype=float)

        if not np.all(np.isfinite(s)):
            continue
        duplicate = False

        for u in unique:
            if np.linalg.norm(s - u) < tol:
                duplicate = True
                break
        if not duplicate:
            unique.append(s.copy())

    return unique


def pose_errors(
    chain,
    q,
    T_target,
    orientation_mode="full",
    tool_axis=None,
    tool_cone_half_angle=None,
    T_current=None,
):
    """按指定模式返回 ``(位置误差, 姿态误差)``。"""
    orientation_mode = validate_orientation_mode(orientation_mode)
    tool_cone_half_angle = normalize_cone_half_angle(tool_cone_half_angle)

    if tool_axis is None:
        tool_axis = TOOL_AXIS_X

    tool_axis = normalize(tool_axis)
    Tq = chain.fk(q) if T_current is None else T_current

    dp = Tq[:3, 3] - T_target[:3, 3]
    pos_err = float(np.linalg.norm(dp))

    if orientation_mode == "full":
        _, drot = pose_error(Tq, T_target)
        ori_err = float(np.linalg.norm(drot))
    elif orientation_mode == "tool_dir":
        ori_err = tool_direction_angle_error(
            T_current=Tq,
            T_target=T_target,
            tool_axis=tool_axis,
        )
    elif orientation_mode == "tool_cone":
        ori_err, _ = tool_cone_angle_error(
            T_current=Tq,
            T_target=T_target,
            tool_axis=tool_axis,
            cone_half_angle=tool_cone_half_angle,
        )

    return pos_err, ori_err