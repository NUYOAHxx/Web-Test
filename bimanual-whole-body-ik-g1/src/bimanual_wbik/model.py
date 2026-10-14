"""Configurable torso-plus-two-arm kinematic tree for whole-body IK."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Any

import numpy as np

from .math3d import inverse, rotation_axis_angle, transform


@dataclass(frozen=True)
class Joint:
    name: str
    kind: str = "revolute"  # revolute | prismatic
    axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    origin_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    origin_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)
    lower: float = -np.pi
    upper: float = np.pi
    velocity: float = 1.0  # rad/s or m/s

    def __post_init__(self):
        if self.kind not in ("revolute", "prismatic"):
            raise ValueError(f"unsupported joint kind {self.kind!r}")
        axis = np.asarray(self.axis, dtype=float)
        if axis.shape != (3,) or np.linalg.norm(axis) < 1e-12:
            raise ValueError(f"joint {self.name}: axis must be a non-zero 3-vector")
        if not self.lower < self.upper:
            raise ValueError(f"joint {self.name}: lower limit must be less than upper")
        if self.velocity <= 0:
            raise ValueError(f"joint {self.name}: velocity limit must be positive")


@dataclass(frozen=True)
class SphereObstacle:
    """Spherical obstacle in the world frame; robot links use point-radius samples."""
    center: tuple[float, float, float]
    radius: float
    name: str = "obstacle"

    def __post_init__(self):
        if self.radius <= 0 or len(self.center) != 3:
            raise ValueError("obstacle needs a 3D center and positive radius")


class RobotModel:
    """Fixed-base torso chain carrying independent left/right arm chains."""

    def __init__(self, torso: list[Joint], left_arm: list[Joint], right_arm: list[Joint],
                 shoulder_mounts: dict[str, np.ndarray], tool_transforms: dict[str, np.ndarray] | None = None,
                 base_transform: np.ndarray | None = None, *, robot_name: str = "custom",
                 nominal_q: dict[str, float] | None = None,
                 com_proxy: dict[str, Any] | None = None):
        self.torso = tuple(torso)
        self.arms = {"left": tuple(left_arm), "right": tuple(right_arm)}
        self.mounts = {s: np.asarray(shoulder_mounts[s], dtype=float) for s in ("left", "right")}
        self.tools = {s: np.asarray((tool_transforms or {}).get(s, np.eye(4)), dtype=float)
                      for s in ("left", "right")}
        self.base = np.asarray(base_transform if base_transform is not None else np.eye(4), dtype=float)
        self.robot_name = str(robot_name)
        self.com_proxy = dict(com_proxy or {})
        for label, t in [("base", self.base), *[(f"{s} mount", t) for s, t in self.mounts.items()],
                         *[(f"{s} tool", t) for s, t in self.tools.items()]]:
            if t.shape != (4, 4):
                raise ValueError(f"{label} transform must be 4x4")
        names = [j.name for j in self.all_joints]
        if len(names) != len(set(names)):
            raise ValueError("joint names must be unique across torso and arms")
        self.index = {name: i for i, name in enumerate(names)}
        self.nominal = np.zeros(len(names), dtype=float)
        for name, value in (nominal_q or {}).items():
            if name not in self.index:
                raise ValueError(f"nominal configuration references unknown joint {name!r}")
            self.nominal[self.index[name]] = float(value)
        if np.any(self.nominal < self.lower) or np.any(self.nominal > self.upper):
            raise ValueError("nominal configuration violates joint limits")
        self.com_proxy_weight = (
            float(self.com_proxy.get("pelvis_weight", 8.0))
            + float(self.com_proxy.get("torso_weight", 8.0))
            + sum((len(self.arms[side]) + 1)
                  * float(self.com_proxy.get("arm_segment_weight", 1.5))
                  for side in ("left", "right")))
        if self.com_proxy_weight <= 0:
            raise ValueError("COM proxy total weight must be positive")

    @property
    def all_joints(self) -> tuple[Joint, ...]:
        return self.torso + self.arms["left"] + self.arms["right"]

    @property
    def dof(self) -> int:
        return len(self.all_joints)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(j.name for j in self.all_joints)

    @property
    def lower(self) -> np.ndarray:
        return np.array([j.lower for j in self.all_joints])

    @property
    def upper(self) -> np.ndarray:
        return np.array([j.upper for j in self.all_joints])

    @property
    def velocity_limits(self) -> np.ndarray:
        return np.array([j.velocity for j in self.all_joints])

    def _check_q(self, q) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        if q.shape != (self.dof,) or not np.all(np.isfinite(q)):
            raise ValueError(f"q must be a finite vector of shape ({self.dof},)")
        return q

    @staticmethod
    def _advance(frame: np.ndarray, joint: Joint, value: float):
        joint_frame = frame @ transform(joint.origin_xyz, joint.origin_rpy)
        axis_world = joint_frame[:3, :3] @ np.asarray(joint.axis, dtype=float)
        axis_world /= np.linalg.norm(axis_world)
        if joint.kind == "revolute":
            motion = np.eye(4)
            motion[:3, :3] = rotation_axis_angle(np.asarray(joint.axis), value)
        else:
            motion = np.eye(4)
            motion[:3, 3] = np.asarray(joint.axis, dtype=float) * value
        return joint_frame @ motion, joint_frame[:3, 3], axis_world

    def _torso_fk(self, q: np.ndarray):
        frame = self.base.copy()
        records: list[tuple[int, Joint, np.ndarray, np.ndarray]] = []
        for i, joint in enumerate(self.torso):
            frame, origin, axis = self._advance(frame, joint, q[i])
            records.append((i, joint, origin, axis))
        return frame, records

    def _arm_fk(self, q: np.ndarray, side: str):
        if side not in self.arms:
            raise ValueError("side must be 'left' or 'right'")
        torso_frame, records = self._torso_fk(q)
        return self._arm_fk_from_torso(q, side, torso_frame, records)

    def _arm_fk_from_torso(self, q: np.ndarray, side: str, torso_frame: np.ndarray,
                           torso_records: list[tuple[int, Joint, np.ndarray, np.ndarray]]):
        frame = torso_frame @ self.mounts[side]
        records = list(torso_records)
        start = len(self.torso) if side == "left" else len(self.torso) + len(self.arms["left"])
        for local_i, joint in enumerate(self.arms[side]):
            idx = start + local_i
            frame, origin, axis = self._advance(frame, joint, q[idx])
            records.append((idx, joint, origin, axis))
        frame = frame @ self.tools[side]
        return frame, records

    def fk(self, q, side: str) -> np.ndarray:
        """Return world-to-tool homogeneous transform for one hand."""
        q = self._check_q(q)
        return self._arm_fk(q, side)[0]

    def torso_fk(self, q) -> np.ndarray:
        q = self._check_q(q)
        return self._torso_fk(q)[0]

    def jacobian(self, q, side: str) -> np.ndarray:
        """Geometric world-frame Jacobian, row order [vx, vy, vz, wx, wy, wz]."""
        q = self._check_q(q)
        end, records = self._arm_fk(q, side)
        return self._jacobian_from_records(end, records)

    def _jacobian_from_records(self, end: np.ndarray,
                               records: list[tuple[int, Joint, np.ndarray, np.ndarray]]) -> np.ndarray:
        p = end[:3, 3]
        j = np.zeros((6, self.dof))
        for idx, joint, origin, axis in records:
            if joint.kind == "revolute":
                j[:3, idx] = np.cross(axis, p - origin)
                j[3:, idx] = axis
            else:
                j[:3, idx] = axis
        return j

    def fk_jacobians(self, q) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Compute both TCP poses and geometric Jacobians with one shared torso traversal."""
        q = self._check_q(q)
        torso_frame, torso_records = self._torso_fk(q)
        poses, jacobians = {}, {}
        for side in ("left", "right"):
            end, records = self._arm_fk_from_torso(q, side, torso_frame, torso_records)
            poses[side] = end
            jacobians[side] = self._jacobian_from_records(end, records)
        return poses, jacobians

    def collision_points(self, q) -> list[tuple[str, np.ndarray]]:
        """Approximate link centerline samples used by optional obstacle constraints."""
        q = self._check_q(q)
        points: list[tuple[str, np.ndarray]] = [("base", self.base[:3, 3].copy())]
        torso_frame, torso_records = self._torso_fk(q)
        points.extend((f"torso:{r[1].name}", r[2].copy()) for r in torso_records)
        points.append(("torso:end", torso_frame[:3, 3].copy()))
        for side in ("left", "right"):
            _, records = self._arm_fk(q, side)
            points.append((f"{side}:shoulder", (torso_frame @ self.mounts[side])[:3, 3].copy()))
            points.extend((f"{side}:{r[1].name}", r[2].copy()) for r in records[len(self.torso):])
            points.append((f"{side}:tool", self.fk(q, side)[:3, 3].copy()))
        return points

    def approximate_com(self, q) -> np.ndarray:
        """Crude kinematic COM proxy (not a dynamics model); useful for balance penalties."""
        q = self._check_q(q)
        torso = self.torso_fk(q)
        proxy = self.com_proxy
        pelvis_xyz = np.asarray(proxy.get("pelvis_xyz", (0.0, 0.0, 0.0)), dtype=float)
        torso_xyz = np.asarray(proxy.get("torso_xyz", (0.0, 0.0, 0.0)), dtype=float)
        pelvis_point = self.base[:3, 3] + self.base[:3, :3] @ pelvis_xyz
        torso_point = torso[:3, :3] @ torso_xyz + torso[:3, 3]
        samples = [(pelvis_point, float(proxy.get("pelvis_weight", 8.0))),
                   (torso_point, float(proxy.get("torso_weight", 8.0)))]
        arm_weight = float(proxy.get("arm_segment_weight", 1.5))
        for side in ("left", "right"):
            shoulder = torso @ self.mounts[side]
            last = shoulder[:3, 3]
            _, records = self._arm_fk(q, side)
            arm_origins = [r[2] for r in records[len(self.torso):]] + [self.fk(q, side)[:3, 3]]
            for end in arm_origins:
                samples.append(((last + end) * 0.5, arm_weight))
                last = end
        total_mass = sum(w for _, w in samples)
        return sum(p * w for p, w in samples) / total_mass

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RobotModel":
        def joints(rows):
            return [Joint(name=r["name"], kind=r.get("kind", "revolute"),
                          axis=tuple(r.get("axis", (0, 0, 1))),
                          origin_xyz=tuple(r.get("origin_xyz", (0, 0, 0))),
                          origin_rpy=tuple(r.get("origin_rpy", (0, 0, 0))),
                          lower=float(r.get("lower", -np.pi)), upper=float(r.get("upper", np.pi)),
                          velocity=float(r.get("velocity", 1.0))) for r in rows]
        from .math3d import transform as make_transform
        mounts = {s: make_transform(**data["shoulder_mounts"][s]) for s in ("left", "right")}
        tools = {s: make_transform(**data.get("tools", {}).get(s, {})) for s in ("left", "right")}
        base = make_transform(**data.get("base", {}))
        return cls(joints(data["torso"]), joints(data["arms"]["left"]), joints(data["arms"]["right"]),
                   mounts, tools, base, robot_name=data.get("robot_name", "custom"),
                   nominal_q=data.get("nominal_q"), com_proxy=data.get("com_proxy"))

    @classmethod
    def from_json(cls, path: str | Path) -> "RobotModel":
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
