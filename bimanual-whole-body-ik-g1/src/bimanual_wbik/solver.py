"""Bounded whole-body inverse kinematics for independent and coupled hands."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import minimize

from .math3d import inverse, pose_error, skew, validate_transform
from .model import RobotModel, SphereObstacle


@dataclass(frozen=True)
class GraspPair:
    """Fixed object-to-tool transforms: T_world_tool = T_world_object @ T_object_tool."""
    left: np.ndarray
    right: np.ndarray

    def __post_init__(self):
        validate_transform(self.left, "left grasp transform")
        validate_transform(self.right, "right grasp transform")

    @staticmethod
    def infer_object_pose(robot: RobotModel, q) -> np.ndarray:
        """Infer a convenient world object frame at the midpoint of the current hands."""
        left, right = robot.fk(q, "left"), robot.fk(q, "right")
        object_pose = np.eye(4)
        object_pose[:3, :3] = left[:3, :3]
        object_pose[:3, 3] = 0.5 * (left[:3, 3] + right[:3, 3])
        return object_pose

    @classmethod
    def from_current(cls, robot: RobotModel, q, object_pose: np.ndarray | None = None) -> "GraspPair":
        """Capture a rigid grasp from the current hand poses.

        If object_pose is omitted, use the midpoint between hands with the left-hand
        orientation as the object frame. Supply a calibrated object pose in production.
        """
        left, right = robot.fk(q, "left"), robot.fk(q, "right")
        if object_pose is None:
            object_pose = cls.infer_object_pose(robot, q)
        object_pose = validate_transform(object_pose, "object pose")
        return cls(inverse(object_pose) @ left, inverse(object_pose) @ right)


@dataclass(frozen=True)
class BalanceRegion:
    """Convex XY support polygon, vertices ordered counter-clockwise in world frame."""
    vertices: tuple[tuple[float, float], ...]
    margin: float = 0.0

    def __post_init__(self):
        p = np.asarray(self.vertices, dtype=float)
        if p.ndim != 2 or p.shape[1] != 2 or len(p) < 3:
            raise ValueError("support polygon must contain at least three XY vertices")
        if self.margin < 0:
            raise ValueError("balance margin cannot be negative")
        area2 = sum(p[i, 0] * p[(i + 1) % len(p), 1] - p[(i + 1) % len(p), 0] * p[i, 1]
                    for i in range(len(p)))
        if area2 <= 1e-9:
            raise ValueError("support polygon must be non-degenerate and counter-clockwise")
        for i in range(len(p)):
            a, b, c = p[i], p[(i + 1) % len(p)], p[(i + 2) % len(p)]
            ab, bc = b - a, c - b
            turn = ab[0] * bc[1] - ab[1] * bc[0]
            if turn < -1e-8:
                raise ValueError("support polygon must be convex")

    def clearances(self, xy) -> np.ndarray:
        p = np.asarray(self.vertices, dtype=float)
        xy = np.asarray(xy, dtype=float)[:2]
        result = []
        for i, a in enumerate(p):
            edge = p[(i + 1) % len(p)] - a
            # For CCW vertices, the interior is to the left of each edge.
            offset = xy - a
            signed = (edge[0] * offset[1] - edge[1] * offset[0]) / max(np.linalg.norm(edge), 1e-12)
            result.append(signed - self.margin)
        return np.asarray(result)


@dataclass
class IKResult:
    q: np.ndarray
    success: bool
    message: str
    iterations: int
    left_position_error: float
    left_orientation_error: float
    right_position_error: float
    right_orientation_error: float
    closure_error: float | None = None
    object_position_error: float | None = None
    object_orientation_error: float | None = None
    objective: float | None = None

    def as_dict(self, joint_names: Sequence[str] | None = None) -> dict:
        names = joint_names or [f"q{i}" for i in range(len(self.q))]
        return {
            "success": self.success, "message": self.message, "iterations": self.iterations,
            "q": {name: float(value) for name, value in zip(names, self.q)},
            "left_position_error_m": self.left_position_error,
            "left_orientation_error_rad": self.left_orientation_error,
            "right_position_error_m": self.right_position_error,
            "right_orientation_error_rad": self.right_orientation_error,
            "closure_error": self.closure_error,
            "object_position_error_m": self.object_position_error,
            "object_orientation_error_rad": self.object_orientation_error,
            "objective": self.objective,
        }


class WholeBodyIK:
    """Whole-body IK over torso and both arms using bounded SLSQP optimization."""

    def __init__(self, robot: RobotModel, position_weight: float = 1.0,
                 orientation_weight: float = 0.45, posture_weight: float = 1e-4,
                 collision_robot_radius: float = 0.055, tolerance: float = 2e-3,
                 max_iterations: int = 250):
        self.robot = robot
        self.position_weight = float(position_weight)
        self.orientation_weight = float(orientation_weight)
        self.posture_weight = float(posture_weight)
        self.collision_robot_radius = float(collision_robot_radius)
        self.tolerance = float(tolerance)
        self.max_iterations = int(max_iterations)
        if min(self.position_weight, self.orientation_weight) <= 0 or self.posture_weight < 0:
            raise ValueError("task weights must be positive and posture weight non-negative")

    def _bounds(self, q, dt, max_step):
        lo, hi = self.robot.lower.copy(), self.robot.upper.copy()
        if dt is not None:
            if dt <= 0:
                raise ValueError("dt must be positive")
            step = self.robot.velocity_limits * float(dt)
            lo, hi = np.maximum(lo, q - step), np.minimum(hi, q + step)
        if max_step is not None:
            step = np.broadcast_to(np.asarray(max_step, dtype=float), q.shape)
            if np.any(step <= 0):
                raise ValueError("max_step must be positive")
            lo, hi = np.maximum(lo, q - step), np.minimum(hi, q + step)
        return list(zip(lo, hi))

    def _pose_residual(self, current, target):
        e = pose_error(current, target)
        return np.r_[self.position_weight * e[:3], self.orientation_weight * e[3:]]

    @staticmethod
    def _so3_right_jacobian_inverse(phi: np.ndarray) -> np.ndarray:
        """Inverse right Jacobian of SO(3), mapping spatial angular increments to Log."""
        phi = np.asarray(phi, dtype=float)
        theta = float(np.linalg.norm(phi))
        k = skew(phi)
        if theta < 1e-5:
            coefficient = 1.0 / 12.0 + theta * theta / 720.0
        elif abs(np.sin(theta)) < 1e-7:
            coefficient = 1.0 / (theta * theta)
        else:
            coefficient = 1.0 / (theta * theta) - (1.0 + np.cos(theta)) / (2.0 * theta * np.sin(theta))
        return np.eye(3) + 0.5 * k + coefficient * (k @ k)

    def _pose_residual_jacobian(self, current, target, geometric_jacobian):
        """Jacobian of the weighted world-frame pose residual used by the objective."""
        error = pose_error(current, target)
        jac = np.asarray(geometric_jacobian, dtype=float)
        out = np.empty_like(jac)
        out[:3] = -self.position_weight * jac[:3]
        out[3:] = (-self.orientation_weight
                   * self._so3_right_jacobian_inverse(error[3:]) @ jac[3:])
        return out

    def _objective_evaluator(self, q_ref, left_target, right_target):
        """Return a cached residual/Jacobian evaluator shared by SLSQP fun and jac calls."""
        cache = {"q": None, "residual": None, "jacobian": None, "poses": None, "jacobians": None}

        def evaluate(x):
            x = np.asarray(x, dtype=float)
            if cache["q"] is None or not np.array_equal(x, cache["q"]):
                poses, geometric = self.robot.fk_jacobians(x)
                blocks = []
                jac_blocks = []
                for side, target in (("left", left_target), ("right", right_target)):
                    blocks.append(self._pose_residual(poses[side], target))
                    jac_blocks.append(self._pose_residual_jacobian(
                        poses[side], target, geometric[side]))
                posture_scale = np.sqrt(self.posture_weight)
                blocks.append(posture_scale * (x - q_ref))
                jac_blocks.append(posture_scale * np.eye(self.robot.dof))
                cache.update(q=x.copy(), residual=np.concatenate(blocks),
                             jacobian=np.vstack(jac_blocks), poses=poses, jacobians=geometric)
            return cache["residual"], cache["jacobian"], cache["poses"], cache["jacobians"]

        return evaluate

    def _closure_residual_jacobian(self, poses, jacobians, closure_target):
        left, right = poses["left"], poses["right"]
        jl, jr = jacobians["left"], jacobians["right"]
        delta = right[:3, 3] - left[:3, 3]
        rotation_world_to_left = left[:3, :3].T
        relative = inverse(left) @ right
        error = pose_error(relative, closure_target)
        relative_jacobian = np.empty_like(jl)
        relative_jacobian[:3] = rotation_world_to_left @ (
            jr[:3] - jl[:3] + skew(delta) @ jl[3:])
        relative_jacobian[3:] = rotation_world_to_left @ (jr[3:] - jl[3:])
        error_jacobian = np.empty_like(relative_jacobian)
        error_jacobian[:3] = -relative_jacobian[:3]
        error_jacobian[3:] = (-self._so3_right_jacobian_inverse(error[3:])
                              @ relative_jacobian[3:])
        return error, error_jacobian

    def _constraints(self, obstacles, balance_region, payload_mass, payload_pose):
        cons = []
        if obstacles:
            def clearance(q):
                pts = self.robot.collision_points(q)
                return np.concatenate([
                    np.array([np.linalg.norm(p - np.asarray(obs.center)) - obs.radius - self.collision_robot_radius
                              for _, p in pts]) for obs in obstacles
                ])
            cons.append({"type": "ineq", "fun": clearance})
        if balance_region is not None:
            def balance(q):
                com = self.robot.approximate_com(q)
                if payload_mass > 0:
                    if payload_pose is None:
                        raise ValueError("payload_pose is required when payload_mass > 0")
                    payload = validate_transform(payload_pose, "payload pose")[:3, 3]
                    body_weight = self.robot.com_proxy_weight
                    com = (body_weight * com + payload_mass * payload) / (body_weight + payload_mass)
                return balance_region.clearances(com[:2])
            cons.append({"type": "ineq", "fun": balance})
        return cons

    def _result(self, q, result, left_target, right_target, closure_target=None,
                object_target=None, grasps: GraspPair | None = None):
        tl, tr = self.robot.fk(q, "left"), self.robot.fk(q, "right")
        el, er = pose_error(tl, left_target), pose_error(tr, right_target)
        closure = None
        if closure_target is not None:
            actual = inverse(tl) @ tr
            ce = pose_error(actual, closure_target)
            closure = float(np.linalg.norm(ce))
        object_pe = object_re = None
        if object_target is not None and grasps is not None:
            object_from_left = tl @ inverse(grasps.left)
            object_from_right = tr @ inverse(grasps.right)
            eol = pose_error(object_from_left, object_target)
            eor = pose_error(object_from_right, object_target)
            object_pe = float(max(np.linalg.norm(eol[:3]), np.linalg.norm(eor[:3])))
            object_re = float(max(np.linalg.norm(eol[3:]), np.linalg.norm(eor[3:])))
        task_ok = max(np.linalg.norm(el[:3]), np.linalg.norm(er[:3])) <= self.tolerance and \
                  max(np.linalg.norm(el[3:]), np.linalg.norm(er[3:])) <= max(self.tolerance, 5e-3)
        closure_ok = closure is None or closure <= 1e-5
        return IKResult(q=np.asarray(q), success=bool(result.success and task_ok and closure_ok),
                        message=str(result.message), iterations=int(getattr(result, "nit", 0)),
                        left_position_error=float(np.linalg.norm(el[:3])),
                        left_orientation_error=float(np.linalg.norm(el[3:])),
                        right_position_error=float(np.linalg.norm(er[:3])),
                        right_orientation_error=float(np.linalg.norm(er[3:])),
                        closure_error=closure, object_position_error=object_pe,
                        object_orientation_error=object_re, objective=float(result.fun))

    def solve_independent(self, q, left_target: np.ndarray, right_target: np.ndarray, *,
                          posture_reference=None, dt: float | None = None,
                          max_step=None, obstacles: Sequence[SphereObstacle] = (),
                          balance_region: BalanceRegion | None = None) -> IKResult:
        """Track independent 6D hand targets; torso joints participate in optimization."""
        q = self.robot._check_q(q).copy()
        left_target = validate_transform(left_target, "left target")
        right_target = validate_transform(right_target, "right target")
        if np.any(q < self.robot.lower - 1e-9) or np.any(q > self.robot.upper + 1e-9):
            raise ValueError("initial q violates robot joint limits")
        q_ref = q.copy() if posture_reference is None else self.robot._check_q(posture_reference)
        cons = self._constraints(obstacles, balance_region, 0.0, None)

        evaluate = self._objective_evaluator(q_ref, left_target, right_target)

        def objective(x):
            r = evaluate(x)[0]
            return 0.5 * float(r @ r)

        def objective_jacobian(x):
            residual, jacobian = evaluate(x)[:2]
            return jacobian.T @ residual

        result = minimize(objective, q, method="SLSQP", bounds=self._bounds(q, dt, max_step),
                          jac=objective_jacobian, constraints=cons,
                          options={"maxiter": self.max_iterations,
                                                     "ftol": 1e-10, "disp": False})
        return self._result(result.x, result, left_target, right_target)

    def solve_object_carry(self, q, object_target: np.ndarray, grasps: GraspPair, *,
                           posture_reference=None, dt: float | None = None,
                           max_step=None, obstacles: Sequence[SphereObstacle] = (),
                           balance_region: BalanceRegion | None = None,
                           payload_mass: float = 0.0) -> IKResult:
        """Move a shared rigid object while enforcing the hand-to-hand closure exactly.

        The nonlinear equality is T_left^-1 T_right = G_left^-1 G_right. The two
        hand pose objectives are both generated from one object target and fixed grasps.
        """
        q = self.robot._check_q(q).copy()
        object_target = validate_transform(object_target, "object target")
        if payload_mass < 0:
            raise ValueError("payload_mass cannot be negative")
        left_target = object_target @ grasps.left
        right_target = object_target @ grasps.right
        closure_target = inverse(grasps.left) @ grasps.right
        if np.any(q < self.robot.lower - 1e-9) or np.any(q > self.robot.upper + 1e-9):
            raise ValueError("initial q violates robot joint limits")
        q_ref = q.copy() if posture_reference is None else self.robot._check_q(posture_reference)
        cons = self._constraints(obstacles, balance_region, payload_mass, object_target)

        evaluate = self._objective_evaluator(q_ref, left_target, right_target)

        def closure(qv):
            _, _, poses, _ = evaluate(qv)
            relative = inverse(poses["left"]) @ poses["right"]
            return pose_error(relative, closure_target)

        def closure_jacobian(qv):
            _, _, poses, jacobians = evaluate(qv)
            return self._closure_residual_jacobian(poses, jacobians, closure_target)[1]

        cons.append({"type": "eq", "fun": closure, "jac": closure_jacobian})

        def objective(x):
            r = evaluate(x)[0]
            return 0.5 * float(r @ r)

        def objective_jacobian(x):
            residual, jacobian = evaluate(x)[:2]
            return jacobian.T @ residual

        result = minimize(objective, q, method="SLSQP", bounds=self._bounds(q, dt, max_step),
                          jac=objective_jacobian, constraints=cons,
                          options={"maxiter": self.max_iterations,
                                                     "ftol": 1e-11, "disp": False})
        return self._result(result.x, result, left_target, right_target,
                            closure_target=closure_target, object_target=object_target,
                            grasps=grasps)
