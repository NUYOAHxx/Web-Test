"""Small, dependency-light SE(3) utilities. Twists are [linear, angular]."""
from __future__ import annotations

import numpy as np


def skew(v: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(v, dtype=float)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def rotation_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(axis)
    if norm < 1e-12:
        raise ValueError("rotation axis must be non-zero")
    axis = axis / norm
    k = skew(axis)
    return np.eye(3) + np.sin(angle) * k + (1.0 - np.cos(angle)) * (k @ k)


def rpy_matrix(rpy) -> np.ndarray:
    """Return Rz(yaw) @ Ry(pitch) @ Rx(roll), angles in radians."""
    roll, pitch, yaw = np.asarray(rpy, dtype=float)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=float)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=float)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=float)
    return rz @ ry @ rx


def transform(xyz=(0.0, 0.0, 0.0), rpy=(0.0, 0.0, 0.0)) -> np.ndarray:
    t = np.eye(4)
    t[:3, :3] = rpy_matrix(rpy)
    t[:3, 3] = np.asarray(xyz, dtype=float)
    return t


def inverse(t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    r_t = t[:3, :3].T
    out = np.eye(4)
    out[:3, :3] = r_t
    out[:3, 3] = -r_t @ t[:3, 3]
    return out


def so3_log(rotation: np.ndarray) -> np.ndarray:
    """Principal rotation vector for a 3x3 rotation matrix."""
    r = np.asarray(rotation, dtype=float)
    cos_theta = float(np.clip((np.trace(r) - 1.0) * 0.5, -1.0, 1.0))
    theta = float(np.arccos(cos_theta))
    vee = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
    if theta < 1e-8:
        return 0.5 * vee
    if np.pi - theta < 1e-5:
        # The antisymmetric part is ill-conditioned near pi; recover an axis
        # from the symmetric part and choose a sign consistent with vee.
        a = np.maximum((np.diag(r) + 1.0) * 0.5, 0.0)
        axis = np.sqrt(a)
        idx = int(np.argmax(axis))
        if axis[idx] < 1e-8:
            axis = np.array([1.0, 0.0, 0.0])
        else:
            if idx == 0:
                axis[1] = np.copysign(axis[1], r[0, 1] + r[1, 0])
                axis[2] = np.copysign(axis[2], r[0, 2] + r[2, 0])
            elif idx == 1:
                axis[0] = np.copysign(axis[0], r[0, 1] + r[1, 0])
                axis[2] = np.copysign(axis[2], r[1, 2] + r[2, 1])
            else:
                axis[0] = np.copysign(axis[0], r[0, 2] + r[2, 0])
                axis[1] = np.copysign(axis[1], r[1, 2] + r[2, 1])
            axis /= max(np.linalg.norm(axis), 1e-12)
        if np.linalg.norm(vee) > 1e-8 and np.dot(axis, vee) < 0:
            axis = -axis
        return theta * axis
    return theta / (2.0 * np.sin(theta)) * vee


def pose_error(current: np.ndarray, target: np.ndarray) -> np.ndarray:
    """World-frame pose error [target-current translation, Log(Rt Rc^T)]."""
    current, target = np.asarray(current, dtype=float), np.asarray(target, dtype=float)
    return np.concatenate((target[:3, 3] - current[:3, 3],
                           so3_log(target[:3, :3] @ current[:3, :3].T)))


def validate_transform(t: np.ndarray, name: str = "transform") -> np.ndarray:
    t = np.asarray(t, dtype=float)
    if t.shape != (4, 4) or not np.all(np.isfinite(t)):
        raise ValueError(f"{name} must be a finite 4x4 matrix")
    if not np.allclose(t[3], [0, 0, 0, 1], atol=1e-7):
        raise ValueError(f"{name} must have homogeneous bottom row [0, 0, 0, 1]")
    if not np.allclose(t[:3, :3].T @ t[:3, :3], np.eye(3), atol=1e-5):
        raise ValueError(f"{name} rotation must be orthonormal")
    return t
