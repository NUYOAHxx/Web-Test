#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Candidate static-load-aware IK, not a robot controller.

Independent DLS implementation based on the supplied ha_arm_ik solver's workflow.
Does not depend on the unavailable ik_candidate.py implementation.
solve() returns (q_or_None, status_string, diagnostic_dict); the third value is
NOT the original IKResult type. Acceptance is report["accepted"], not a legacy
is_valid_status() helper.

Requires NumPy and Pinocchio. No ROS import, /joint_states/TF publisher or action.
Only fixed-base URDF trees with fixed/revolute/continuous/prismatic joints.
Mimic, floating, planar and multi-DoF joints fail explicitly, not approximately.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import threading
import time
import xml.etree.ElementTree as ET

import numpy as np

from .load_policy import (
    JointStateSnapshot, LoadPolicy, finite_vector, joint_difference, json_safe,
    metric_step, select_load_candidate, skew, validate_transform,
)


@dataclass(frozen=True)
class SolverConfig:
    pos_tol_m: float = 5e-4
    ori_tol_rad: float = 5e-4
    characteristic_length_m: float = 0.5
    damping: float = 1e-3
    rank_rtol: float = 1e-9
    max_iterations: int = 160
    settle_iterations: int = 40
    max_seeds: int = 10
    min_seeds: int = 3
    min_unlocked_seeds: int = 2
    budget_s: float = 0.5
    state_max_age_s: float = 1.0
    require_source_age: bool = False
    rotation_step_rad: float = 0.15
    translation_step_m: float = 0.02
    seed_noise_rad: float = 0.3
    posture_weight: float = 0.03
    waist_posture_weight: float = 0.05
    waist_metric_weight: float = 3.0
    com_weight: float = 0.02
    limit_weight: float = 0.01
    limit_buffer_rad: float = 0.05
    limit_buffer_m: float = 0.01
    null_gain: float = 1.0
    backtrack_steps: int = 12
    rng_seed: int = 120

    def __post_init__(self):
        positive = ("pos_tol_m", "ori_tol_rad", "characteristic_length_m",
                    "rank_rtol", "budget_s", "state_max_age_s",
                    "rotation_step_rad", "translation_step_m", "seed_noise_rad",
                    "waist_metric_weight", "limit_buffer_rad", "limit_buffer_m")
        nonnegative = ("damping", "posture_weight", "waist_posture_weight",
                       "com_weight", "limit_weight", "null_gain")
        for name in positive + nonnegative:
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0 or (name in positive and value == 0):
                raise ValueError(f"invalid SolverConfig.{name}")
        for name in ("max_iterations", "settle_iterations", "max_seeds",
                     "min_seeds", "min_unlocked_seeds", "backtrack_steps"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.min_seeds > self.max_seeds:
            raise ValueError("min_seeds must be <= max_seeds")


@dataclass(frozen=True)
class JointInfo:
    name: str
    kind: str
    lower: float
    upper: float
    idx_q: int
    nq: int
    idx_v: int
    urdf_effort: float | None


def parse_urdf(xml):
    root = ET.fromstring(xml)
    if root.tag != "robot":
        raise ValueError("expected URDF <robot>, not xacro/source fragments")
    links = {e.get("name"): e for e in root.findall("link")}
    if not links or None in links or len(links) != len(root.findall("link")):
        raise ValueError("invalid/duplicate URDF links")
    joints = {}
    child_to_joint = {}
    for e in root.findall("joint"):
        name, kind = e.get("name"), e.get("type")
        if not name or name in joints:
            raise ValueError("invalid/duplicate URDF joint")
        if kind not in ("fixed", "revolute", "continuous", "prismatic"):
            raise ValueError(f"{name}: unsupported joint type {kind}")
        if e.find("mimic") is not None:
            raise ValueError(f"{name}: mimic joints are not supported by this candidate")
        parent, child = e.find("parent"), e.find("child")
        if parent is None or child is None:
            raise ValueError(f"{name}: missing parent/child")
        p, c = parent.get("link"), child.get("link")
        if p not in links or c not in links or c in child_to_joint:
            raise ValueError(f"{name}: invalid link tree")
        origin = e.find("origin")
        if origin is not None:
            for attr in ("xyz", "rpy"):
                finite_vector([float(v) for v in origin.get(attr, "0 0 0").split()],
                              3, f"{name} origin {attr}")
        if kind != "fixed":
            axis_elem = e.find("axis")
            axis = finite_vector([1., 0., 0.] if axis_elem is None else
                                 [float(v) for v in axis_elem.get("xyz", "1 0 0").split()],
                                 3, f"{name} axis")
            if np.linalg.norm(axis) < 1e-12:
                raise ValueError(f"{name}: zero joint axis")
        if kind == "fixed":
            lower = upper = 0.
        elif kind == "continuous":
            lower, upper = -np.inf, np.inf
        else:
            lim = e.find("limit")
            if lim is None:
                raise ValueError(f"{name}: missing physical position limits")
            lower = float(lim.attrib["lower"])
            upper = float(lim.attrib["upper"])
            if not np.isfinite(lower + upper) or lower >= upper:
                raise ValueError(f"{name}: invalid physical position limits")
        effort = None
        lim = e.find("limit")
        if lim is not None and "effort" in lim.attrib:
            value = float(lim.get("effort"))
            if math.isfinite(value) and value > 0:
                effort = value
        joints[name] = dict(name=name, kind=kind, parent=p, child=c,
                            lower=lower, upper=upper, urdf_effort=effort)
        child_to_joint[c] = name
    roots = set(links) - set(child_to_joint)
    if len(roots) != 1:
        raise ValueError(f"expected one URDF root, got {sorted(roots)}")
    root_link = next(iter(roots))
    for link in links:
        seen = set()
        cur = link
        while cur in child_to_joint:
            if cur in seen:
                raise ValueError("URDF cycle")
            seen.add(cur)
            cur = joints[child_to_joint[cur]]["parent"]
        if cur != root_link:
            raise ValueError("disconnected URDF")
    return root_link, links, joints, child_to_joint


def chain_joint_names(base_link, ee_link, joints, child_to_joint):
    cur, names = ee_link, []
    while cur != base_link:
        if cur not in child_to_joint:
            raise ValueError(f"{base_link} is not an ancestor of {ee_link}")
        name = child_to_joint[cur]
        if joints[name]["kind"] != "fixed":
            names.append(name)
        cur = joints[name]["parent"]
    return list(reversed(names))


def inspect_urdf(xml, base_link, ee_link):
    """No Pinocchio needed; generate an intentionally UNFILLED limit template."""
    root, links, joints, child_map = parse_urdf(xml)
    if base_link not in links or ee_link not in links:
        raise ValueError("unknown base/EE link")
    active = chain_joint_names(base_link, ee_link, joints, child_map)
    moving = [n for n, j in joints.items() if j["kind"] != "fixed"]
    return json_safe({
        "root_link": root, "base_link": base_link, "ee_link": ee_link,
        "active_joint_names": active, "required_state_joints": moving,
        "allowable_efforts": {n: None for n in moving},
        "effort_units": {n: "N" if joints[n]["kind"] == "prismatic" else "N*m" for n in moving},
        "urdf_effort_reference_only": {n: joints[n]["urdf_effort"] for n in moving},
        "position_limits": {n: [joints[n]["lower"], joints[n]["upper"]] for n in moving},
        "continuous_joint_names": [n for n in moving if joints[n]["kind"] == "continuous"],
        "links_without_positive_mass": [
            n for n, e in links.items()
            if e.find("inertial/mass") is None
            or float(e.find("inertial/mass").get("value", "0")) <= 0
        ],
        "warning": "null capacities are deliberately not runnable; confirm real derated joint-side limits",
    })


class LoadAwareIKSolver:
    def __init__(self, urdf_xml, *, base_link, ee_link, policy: LoadPolicy,
                 q_reference=None, config: SolverConfig | None = None,
                 massless_links=(), waist_joint_names=("Hip_Link1", "Body_Link"),
                 position_bounds=None, gravity_root=(0., 0., -9.81)):
        import pinocchio as pin
        self.pin = pin
        self.config = config or SolverConfig()
        if not isinstance(policy, LoadPolicy):
            raise TypeError("policy must be LoadPolicy with explicit capacities")
        self.policy = policy
        self.xml = str(urdf_xml)
        self.model_sha256 = hashlib.sha256(self.xml.encode()).hexdigest()
        root, self.links_xml, source_joints, child_map = parse_urdf(self.xml)
        if base_link not in self.links_xml or ee_link not in self.links_xml:
            raise ValueError("unknown base/EE link")
        self.root_link, self.base_link, self.ee_link = root, base_link, ee_link
        self.active_joint_names = chain_joint_names(base_link, ee_link, source_joints, child_map)
        if not self.active_joint_names:
            raise ValueError("empty IK chain")
        self.model = pin.buildModelFromXML(self.xml)
        self.data = self.model.createData()
        self.gravity_root = finite_vector(gravity_root, 3, "gravity_root")
        if np.linalg.norm(self.gravity_root) < 1e-6:
            raise ValueError("gravity_root must be nonzero")
        self.model.gravity.linear = self.gravity_root
        self.base_id, self.ee_id = self._frame_id(base_link), self._frame_id(ee_link)
        overrides = dict(position_bounds or {})
        unknown = set(overrides) - set(source_joints)
        if unknown:
            raise ValueError(f"unknown position bound override: {sorted(unknown)}")
        self.joints = {}
        for name, src in source_joints.items():
            if src["kind"] == "fixed":
                if name in overrides:
                    raise ValueError(f"cannot override fixed joint {name}")
                continue
            jid = self.model.getJointId(name)
            if jid == 0 or jid >= self.model.njoints:
                raise ValueError(f"joint missing from Pinocchio: {name}")
            pj = self.model.joints[jid]
            expected_nq = 2 if src["kind"] == "continuous" else 1
            if pj.nv != 1 or pj.nq != expected_nq:
                raise ValueError(f"{name}: unsupported Pinocchio nq/nv={pj.nq}/{pj.nv}")
            lo, hi = src["lower"], src["upper"]
            if name in overrides:
                bound = finite_vector(overrides[name], 2, f"{name} bounds")
                if bound[0] >= bound[1]:
                    raise ValueError(f"{name}: invalid override")
                if bound[0] < lo or bound[1] > hi:
                    raise ValueError(f"{name}: override would expand URDF bounds")
                lo, hi = float(bound[0]), float(bound[1])
            self.joints[name] = JointInfo(name, src["kind"], lo, hi,
                                          pj.idx_q, pj.nq, pj.idx_v, src["urdf_effort"])
        self.guard_joint_names = list(policy.guard_joint_names or self.joints)
        if set(self.guard_joint_names) - set(self.joints):
            raise ValueError("unknown guarded joint")
        if set(self.active_joint_names) - set(self.guard_joint_names):
            raise ValueError("every active joint must be guarded")
        self.capacities, self.load_weights = policy.vectors(self.guard_joint_names)
        self.unchecked_joint_names = sorted(set(self.joints) - set(self.guard_joint_names))
        self.active_v = np.array([self.joints[n].idx_v for n in self.active_joint_names])
        self.guard_v = np.array([self.joints[n].idx_v for n in self.guard_joint_names])
        self.lower = np.array([self.joints[n].lower for n in self.active_joint_names])
        self.upper = np.array([self.joints[n].upper for n in self.active_joint_names])
        # A user-supplied finite planning/cable window is not periodic even
        # when the underlying URDF joint is continuous.
        self.continuous = np.array([
            self.joints[n].kind == "continuous"
            and not np.isfinite(self.joints[n].lower)
            and not np.isfinite(self.joints[n].upper)
            for n in self.active_joint_names])
        self.waist = np.array([n in waist_joint_names for n in self.active_joint_names])
        self.steps = np.array([self.config.translation_step_m if self.joints[n].kind == "prismatic"
                               else self.config.rotation_step_rad for n in self.active_joint_names])
        self.limit_buffers = np.array([self.config.limit_buffer_m if self.joints[n].kind == "prismatic"
                                       else self.config.limit_buffer_rad for n in self.active_joint_names])
        if np.any(self.waist) and self.config.max_seeds < 1 + self.config.min_unlocked_seeds:
            raise ValueError("max_seeds must allow a locked seed and the minimum unlocked comparison")
        self.mass_points, self.model_warnings = self._audit_masses(set(massless_links))
        self.total_mass = float(sum(m for _, m, _ in self.mass_points))
        self.q_reference = None if q_reference is None else self._active_values(q_reference, "q_reference")
        if self.q_reference is not None:
            self._check_active_bounds(self.q_reference, "q_reference")
        self.q_prev = None
        self._lock = threading.Lock()

    @classmethod
    def from_urdf_file(cls, path, **kwargs):
        return cls(Path(path).expanduser().read_text(encoding="utf-8"), **kwargs)

    @classmethod
    def from_existing_solver(cls, old, *, policy, config=None, q_reference=None,
                             massless_links=(), position_bounds=None):
        """Reuses configuration/model description, NOT the original scorer.

        No mutation of old_solver, its model, Data, q_prev or source files.
        Explicit q_reference is recommended if old code clipped continuous angles.
        """
        if old.pin_model is None:
            raise ValueError("old solver has no Pinocchio model; supply explicit XML/gravity instead")
        gravity = old.pin_model.gravity.linear
        result = cls.from_urdf_file(
            old.urdf, base_link=old.base_link, ee_link=old.ee_link,
            policy=policy, config=config,
            q_reference=old.q_ready if q_reference is None else q_reference,
            massless_links=massless_links, position_bounds=position_bounds,
            gravity_root=np.array(gravity, copy=True),
        )
        if hasattr(old, "active_joint_names") and list(old.active_joint_names) != result.active_joint_names:
            raise ValueError("old/new active joint order mismatch; no automatic conversion")
        return result

    def _frame_id(self, name):
        if not self.model.existFrame(name):
            raise ValueError(f"frame missing from model: {name}")
        return self.model.getFrameId(name)

    def _audit_masses(self, massless):
        unknown = massless - set(self.links_xml)
        if unknown:
            raise ValueError(f"unknown massless link names: {sorted(unknown)}")
        points, warnings = [], []
        for name, link in self.links_xml.items():
            inertial = link.find("inertial")
            mass_elem = None if inertial is None else inertial.find("mass")
            mass = 0. if mass_elem is None else float(mass_elem.attrib["value"])
            if not np.isfinite(mass) or mass < 0:
                raise ValueError(f"{name}: invalid mass")
            if mass == 0:
                if name not in massless:
                    raise ValueError(f"{name}: mass missing/zero; confirm and list only genuinely massless frames")
                warnings.append(f"explicit_massless_link:{name}")
                continue
            if name in massless:
                raise ValueError(f"{name}: positive mass cannot be whitelisted away")
            origin = inertial.find("origin")
            offset = np.zeros(3) if origin is None else finite_vector(
                [float(v) for v in origin.get("xyz", "0 0 0").split()], 3, f"{name} local COM")
            if origin is not None:
                finite_vector([float(v) for v in origin.get("rpy", "0 0 0").split()], 3,
                              f"{name} inertial rpy")
            inertia = inertial.find("inertia")
            if inertia is None:
                raise ValueError(f"{name}: missing inertia tensor")
            a = inertia.attrib
            I = np.array([[float(a["ixx"]), float(a["ixy"]), float(a["ixz"])],
                          [float(a["ixy"]), float(a["iyy"]), float(a["iyz"])],
                          [float(a["ixz"]), float(a["iyz"]), float(a["izz"])]])
            eig = np.linalg.eigvalsh(I)
            if not np.all(np.isfinite(I)) or eig[0] < -1e-10 or eig[-1] > eig[:2].sum() + 1e-8:
                raise ValueError(f"{name}: nonphysical inertia tensor")
            points.append((self._frame_id(name), mass, offset))
        if not points:
            raise ValueError("no positive model mass")
        return points, warnings

    def _active_values(self, values, name):
        if isinstance(values, dict):
            missing = set(self.active_joint_names) - set(values)
            if missing:
                raise ValueError(f"{name}: missing active joints {sorted(missing)}")
            values = [values[n] for n in self.active_joint_names]
        return finite_vector(values, len(self.active_joint_names), name)

    def _check_active_bounds(self, q, label):
        if np.any(q < self.lower) or np.any(q > self.upper):
            raise ValueError(f"{label}: outside configured physical/planning bounds (not clipped)")

    def _state_context(self, snapshot):
        if not isinstance(snapshot, JointStateSnapshot):
            raise TypeError("snapshot must be JointStateSnapshot")
        missing = set(self.joints) - set(snapshot.positions)
        if missing:
            raise ValueError(f"missing whole-model joint state: {sorted(missing)}")
        context = dict(snapshot.positions)
        for name, info in self.joints.items():
            value = context[name]
            if value < info.lower or value > info.upper:
                raise ValueError(f"observed joint {name} outside configured limits; not clipped")
        return context

    def _full_q(self, q_active, context):
        q = self.pin.neutral(self.model)
        active = dict(zip(self.active_joint_names, q_active))
        for name, info in self.joints.items():
            value = active.get(name, context[name])
            if info.kind == "continuous":
                q[info.idx_q:info.idx_q+2] = (math.cos(value), math.sin(value))
            else:
                q[info.idx_q] = value
        return q

    def _fk(self, q_full):
        self.pin.forwardKinematics(self.model, self.data, q_full)
        self.pin.updateFramePlacements(self.model, self.data)

    def _com(self, q_full, *, jacobian):
        # Includes root/fixed-link masses, even if fixed-base Pinocchio COM
        # conventions omit inertia attached to the universe joint.
        self._fk(q_full)
        if jacobian:
            self.pin.computeJointJacobians(self.model, self.data, q_full)
        moment = np.zeros(3)
        Jcom = np.zeros((3, len(self.active_joint_names)))
        for fid, mass, offset in self.mass_points:
            T = self.data.oMf[fid]
            rotated = np.asarray(T.rotation) @ offset
            p = np.asarray(T.translation) + rotated
            moment += mass * p
            if jacobian:
                J = self.pin.getFrameJacobian(
                    self.model, self.data, fid, self.pin.LOCAL_WORLD_ALIGNED)
                # Some EigenPy versions squeeze the nv=1 return into (6,).
                J = np.asarray(J).reshape(6, self.model.nv)
                Jpoint = J[:3, :] - skew(rotated) @ J[3:, :]
                Jcom += mass * Jpoint[:, self.active_v]
        return moment / self.total_mass, Jcom / self.total_mass

    def _physics(self, q_active, context, *, derivatives):
        q_full = self._full_q(q_active, context)
        tau_full = np.array(self.pin.computeGeneralizedGravity(self.model, self.data, q_full), copy=True)
        tau = tau_full[self.guard_v]
        derivative = np.zeros((len(self.guard_v), len(self.active_v)))
        if derivatives:
            Jtau = np.array(self.pin.computeGeneralizedGravityDerivatives(
                self.model, self.data, q_full), copy=True).reshape(self.model.nv, self.model.nv)
            derivative = Jtau[np.ix_(self.guard_v, self.active_v)]
        com, Jcom = self._com(q_full, jacobian=derivatives)
        if not np.all(np.isfinite(com)) or not np.all(np.isfinite(Jcom)):
            raise ValueError("non-finite whole-model COM")
        load = self.policy.evaluate(tau, self.guard_joint_names)
        return {"q_full": q_full, "tau": tau, "d_tau_d_active": derivative,
                "com_root_m": com, "Jcom": Jcom, **load}

    def _task(self, q_active, context, target, mode, tool_axis, cone, *, jacobian=True):
        q_full = self._full_q(q_active, context)
        self._fk(q_full)
        current = self.data.oMf[self.ee_id].copy()
        p_error = target[:3, 3] - np.asarray(current.translation)
        pos = float(np.linalg.norm(p_error))
        if jacobian:
            J = np.asarray(self.pin.computeFrameJacobian(
                self.model, self.data, q_full, self.ee_id,
                self.pin.LOCAL if mode == "full" else self.pin.LOCAL_WORLD_ALIGNED
            )).reshape(6, self.model.nv)[:, self.active_v]
        else:
            J = np.zeros((6, len(self.active_v)))
        antipodal = False
        if mode == "full":
            delta = current.inverse() * self.pin.SE3(target[:3, :3], target[:3, 3])
            error = np.array(self.pin.log6(delta).vector, copy=True)
            angle = float(np.linalg.norm(error[3:]))
            # d(log(current^-1 desired))/dq = -Jlog6(delta^-1) J_LOCAL.
            correction = self.pin.Jlog6(delta.inverse()) @ J if jacobian else J
            error[:3] /= self.config.characteristic_length_m
            correction = np.array(correction, copy=True)
            correction[:3] /= self.config.characteristic_length_m
            violation = angle
        else:
            a = np.asarray(current.rotation) @ tool_axis
            b = target[:3, :3] @ tool_axis
            cross = np.cross(a, b)
            sin_angle = float(np.linalg.norm(cross))
            dot = float(np.clip(a @ b, -1., 1.))
            angle = math.atan2(sin_angle, dot)
            Jv, Jw = J[:3], J[3:]
            if mode == "tool_dir":
                e_dir = b-a
                J_dir = -skew(a) @ Jw
                if dot < -1 + 1e-8:
                    # Direction difference has a zero gradient at exact opposition.
                    # Choose a deterministic tangent escape; acceptance still checks
                    # the true angle and the actual endpoint.
                    basis = np.eye(3)[int(np.argmin(np.abs(a)))]
                    axis = np.cross(a, basis)
                    axis /= np.linalg.norm(axis)
                    e_dir = angle * axis
                    J_dir = (np.eye(3) - np.outer(a, a)) @ Jw
                    antipodal = True
                error = np.r_[p_error / self.config.characteristic_length_m, e_dir]
                correction = np.vstack([Jv / self.config.characteristic_length_m, J_dir])
                violation = angle
            elif mode == "tool_cone":
                violation = max(0., angle-cone)
                row = np.zeros(len(self.active_v))
                if violation:
                    if sin_angle > 1e-9:
                        axis = cross / sin_angle
                    else:
                        basis = np.eye(3)[int(np.argmin(np.abs(a)))]
                        axis = np.cross(a, basis)
                        axis /= np.linalg.norm(axis)
                        antipodal = True
                    row = axis @ Jw
                error = np.r_[p_error / self.config.characteristic_length_m, violation]
                correction = np.vstack([Jv / self.config.characteristic_length_m, row])
            else:
                raise ValueError("orientation_mode must be full, tool_dir or tool_cone")
        return {"error": error, "J": correction, "position_error_m": pos,
                "orientation_angle_rad": angle, "orientation_violation_rad": violation,
                "pose_ok": bool(pos <= self.config.pos_tol_m and violation <= self.config.ori_tol_rad),
                "merit": .5 * float(error @ error), "antipodal_escape": antipodal}

    def _secondary(self, q, context, reference_physics, *, derivatives):
        physics = self._physics(q, context, derivatives=derivatives)
        cost, gradient, components = self.policy.cost_gradient(
            physics["tau"], physics["d_tau_d_active"], self.guard_joint_names,
            tau_reference=reference_physics["tau"])
        diff = joint_difference(q, self.q_reference, self.continuous)
        w = np.where(self.waist, self.config.waist_posture_weight, self.config.posture_weight)
        posture = .5 * float(np.sum(w * diff*diff))
        gradient += w * diff
        cost += posture
        components["posture"] = posture
        # Real bound buffers only. No arbitrary +/-90 or +/-150 working box.
        under, over = np.zeros_like(q), np.zeros_like(q)
        for i, (lo, hi, buffer) in enumerate(zip(self.lower, self.upper, self.limit_buffers)):
            if np.isfinite(lo) and np.isfinite(hi):
                buffer = min(buffer, .25 * (hi-lo))
            if np.isfinite(lo):
                under[i] = max(0., lo + buffer - q[i]) / buffer
            if np.isfinite(hi):
                over[i] = max(0., q[i] - hi + buffer) / buffer
            gradient[i] += self.config.limit_weight * (over[i]-under[i]) / buffer
        limit_cost = .5 * self.config.limit_weight * float(under @ under + over @ over)
        cost += limit_cost
        components["joint_limit_buffer"] = limit_cost
        ghat = self.gravity_root / np.linalg.norm(self.gravity_root)
        horizontal = np.eye(3) - np.outer(ghat, ghat)
        dcom = horizontal @ (physics["com_root_m"] - reference_physics["com_root_m"])
        length = self.config.characteristic_length_m
        com_cost = .5 * self.config.com_weight * float(dcom @ dcom) / length**2
        gradient += self.config.com_weight * physics["Jcom"].T @ dcom / length**2
        cost += com_cost
        components["reference_com"] = com_cost
        if not math.isfinite(cost) or not np.all(np.isfinite(gradient)):
            raise ValueError("non-finite secondary objective")
        return {"cost": cost, "gradient": gradient, "components": components, "physics": physics}

    def _bounded_direction(self, task, gradient, q, fixed_mask, weights, null_weight):
        free = ~fixed_mask.copy()
        task_step = np.zeros_like(q)
        null_step = np.zeros_like(q)
        rank = 0
        # Freeze outward motion at an already active bound; do not clip the
        # observed state or invent an out-of-range joint increment.
        for _ in range(q.size+1):
            if not np.any(free):
                return np.zeros_like(q), np.zeros_like(q), rank
            t, n, rank = metric_step(task["J"][:, free], task["error"], gradient[free],
                                     1. / weights[free], damping=self.config.damping,
                                     rank_rtol=self.config.rank_rtol)
            task_step.fill(0.)
            null_step.fill(0.)
            task_step[free], null_step[free] = t, null_weight*n
            dq = task_step + null_step
            blocked = free & (((q <= self.lower+1e-12) & (dq < 0))
                              | ((q >= self.upper-1e-12) & (dq > 0)))
            if not np.any(blocked):
                break
            free[blocked] = False
        scale = min(1., 1./max(1., float(np.max(np.abs(dq)/self.steps))))
        for i, step in enumerate(dq):
            if step > 0 and np.isfinite(self.upper[i]):
                scale = min(scale, max(0., (self.upper[i]-q[i])/step))
            elif step < 0 and np.isfinite(self.lower[i]):
                scale = min(scale, max(0., (self.lower[i]-q[i])/step))
        return dq * scale, null_step * scale, rank

    def _candidate(self, q, context, target, mode, axis, cone, q_prev, index):
        # Normalize only genuinely continuous branches around q_prev.
        q = q.copy()
        q[self.continuous] = q_prev[self.continuous] + joint_difference(
            q, q_prev, self.continuous)[self.continuous]
        task = self._task(q, context, target, mode, axis, cone, jacobian=False)
        physics = self._physics(q, context, derivatives=False)
        bounds_ok = bool(np.all(q >= self.lower) and np.all(q <= self.upper))
        return {
            "q": q, "candidate_index": index, "pose_ok": task["pose_ok"],
            "bounds_ok": bounds_ok, "load_ok": physics["load_ok"],
            "position_error_m": task["position_error_m"],
            "orientation_angle_rad": task["orientation_angle_rad"],
            "orientation_violation_rad": task["orientation_violation_rad"],
            "peak_ratio": physics["peak_ratio"], "peak_joint": physics["peak_joint"],
            "mean_square_ratio": physics["mean_square_ratio"],
            "overloaded_joints": physics["overloaded_joints"],
            "gravity_efforts": dict(zip(self.guard_joint_names, physics["tau"])),
            "load_ratios": dict(zip(self.guard_joint_names, physics["ratios"])),
            "com_root_m": physics["com_root_m"],
            "continuity_cost": float(np.mean(joint_difference(q, q_prev, self.continuous)**2)),
            "reference_cost": float(np.mean(joint_difference(q, self.q_reference, self.continuous)**2)),
            "merit": task["merit"],
        }

    def _one_seed(self, seed, fixed, context, target, mode, axis, cone,
                  q_prev, reference_physics, deadline, index, released):
        start = time.monotonic()
        q = seed.copy()
        weights = np.where(self.waist, 1. if released else self.config.waist_metric_weight, 1.)
        best_pose, best_pose_key = None, None
        best_task, best_merit = q.copy(), math.inf
        settled, no_progress = 0, 0
        diag = {"seed_index": index, "fixed_waist": bool(np.any(fixed)),
                "waist_released_for_load": released, "iterations": 0,
                "stop_reason": "ITERATION_LIMIT", "antipodal_escape_used": False}
        for iteration in range(self.config.max_iterations):
            if time.monotonic() >= deadline:
                diag["stop_reason"] = "TIME_BUDGET"
                break
            task = self._task(q, context, target, mode, axis, cone)
            secondary = self._secondary(q, context, reference_physics, derivatives=True)
            diag["iterations"] = iteration+1
            diag["antipodal_escape_used"] |= task["antipodal_escape"]
            if task["merit"] < best_merit:
                best_task, best_merit = q.copy(), task["merit"]
            if task["pose_ok"]:
                physical = secondary["physics"]
                key = (not physical["load_ok"], physical["peak_ratio"], secondary["cost"])
                if best_pose_key is None or key < best_pose_key:
                    best_pose, best_pose_key = q.copy(), key
                settled += 1
                if settled > self.config.settle_iterations:
                    diag["stop_reason"] = "SETTLE_LIMIT"
                    break
            # Exact local task-null separation; nonlinear effects are checked
            # by line search against actual FK at each proposed configuration.
            null_weight = self.config.null_gain / (1.+(np.linalg.norm(task["error"])/.06)**2)
            dq, dq_null, rank = self._bounded_direction(
                task, secondary["gradient"], q, fixed, weights, null_weight)
            diag["last_task_rank"] = rank
            diag["last_null_step_norm"] = float(np.linalg.norm(dq_null))
            if np.linalg.norm(dq) < 1e-10:
                diag["stop_reason"] = "SMALL_STEP"
                break
            accepted = False
            # First try combined task/load descent; retry task-only if the
            # nullspace term prevents progress before task convergence.
            attempts = [dq]
            if not task["pose_ok"] and np.linalg.norm(dq_null) > 1e-12:
                pure, _, _ = self._bounded_direction(
                    task, np.zeros_like(q), q, fixed, weights, 0.)
                attempts.append(pure)
            for direction in attempts:
                for backtrack in range(self.config.backtrack_steps):
                    if time.monotonic() >= deadline:
                        break
                    alpha = .5**backtrack
                    trial = q + alpha * direction
                    if np.any(trial < self.lower) or np.any(trial > self.upper):
                        continue
                    trial_task = self._task(trial, context, target, mode, axis, cone, jacobian=False)
                    if task["pose_ok"]:
                        if not trial_task["pose_ok"]:
                            continue
                        trial_secondary = self._secondary(
                            trial, context, reference_physics, derivatives=False)
                        progress = trial_secondary["cost"] < secondary["cost"] - 1e-13
                    else:
                        progress = (trial_task["pose_ok"]
                                    or trial_task["merit"] < task["merit"] - max(1e-14, 1e-6*task["merit"]))
                        # The antipodal tangent branch is a local escape, so
                        # compare actual angular error, not different residual branches.
                        if task["antipodal_escape"]:
                            progress = (trial_task["orientation_violation_rad"]
                                        < task["orientation_violation_rad"] - 1e-8
                                        and trial_task["position_error_m"]
                                        <= task["position_error_m"] + self.config.pos_tol_m)
                    if progress:
                        q = trial
                        accepted = True
                        break
                if accepted:
                    break
            if not accepted:
                no_progress += 1
                diag["stop_reason"] = "LINE_SEARCH_STALLED"
                break
        # Include the final update, not stale pre-update convergence/physics.
        final = self._candidate(q, context, target, mode, axis, cone, q_prev, index)
        if final["pose_ok"]:
            key = (not final["load_ok"], final["peak_ratio"],
                   self._secondary(q, context, reference_physics, derivatives=False)["cost"])
            if best_pose_key is None or key < best_pose_key:
                best_pose = q.copy()
        elif final["merit"] < best_merit:
            best_task = q.copy()
        chosen_q = best_pose if best_pose is not None else best_task
        candidate = self._candidate(chosen_q, context, target, mode, axis, cone, q_prev, index)
        final_secondary = self._secondary(chosen_q, context, reference_physics, derivatives=False)
        diag.update(elapsed_ms=(time.monotonic()-start)*1000.,
                    final_position_error_m=candidate["position_error_m"],
                    final_orientation_violation_rad=candidate["orientation_violation_rad"],
                    final_peak_ratio=candidate["peak_ratio"],
                    secondary_components=final_secondary["components"],
                    physics_terms_enabled={"absolute": self.policy.absolute_weight > 0,
                                           "excess": self.policy.excess_weight > 0,
                                           "relative": self.policy.relative_weight > 0,
                                           "com": self.config.com_weight > 0},
                    no_progress_count=no_progress)
        return candidate, diag

    def _seeds(self, q_prev):
        seeds = []
        if np.any(self.waist):
            locked = q_prev.copy()
            locked[self.waist] = self.q_reference[self.waist]
            seeds.append((locked, self.waist.copy()))
        seeds.append((q_prev.copy(), np.zeros(q_prev.size, dtype=bool)))
        if np.linalg.norm(joint_difference(q_prev, self.q_reference, self.continuous)) > 1e-6:
            seeds.append((self.q_reference.copy(), np.zeros(q_prev.size, dtype=bool)))
        rng = np.random.default_rng(self.config.rng_seed)
        while len(seeds) < self.config.max_seeds:
            seed = q_prev.copy() if len(seeds) % 2 else self.q_reference.copy()
            noise_scale = self.config.seed_noise_rad * np.where(
                np.array([self.joints[n].kind == "prismatic" for n in self.active_joint_names]),
                self.config.translation_step_m/self.config.rotation_step_rad, 1.)
            seed += rng.normal(size=q_prev.size) * noise_scale
            # Search seeds may be bounded; observed state and q_prev are NEVER clipped.
            seed = np.minimum(np.maximum(seed, self.lower), self.upper)
            seeds.append((seed, np.zeros(q_prev.size, dtype=bool)))
        return seeds[:self.config.max_seeds]

    def evaluate_state(self, snapshot):
        """Read-only diagnostics; still requires complete, fresh model state."""
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("solver is busy; use one worker per instance")
        try:
            error = snapshot.freshness_error(self.config.state_max_age_s,
                                            require_source_age=self.config.require_source_age)
            if error:
                raise ValueError(error)
            context = self._state_context(snapshot)
            q = np.array([context[n] for n in self.active_joint_names])
            return json_safe(self._physics(q, context, derivatives=False))
        finally:
            self._lock.release()

    def solve(self, T_target, frame="local", q_prev=None, update_prev=True,
              verbose=False, allow_approximate=False, orientation_mode="full",
              tool_axis=None, tool_cone_half_angle=None, *, snapshot):
        """Never sends a command. Approximate requests NEVER bypass acceptance."""
        start = time.monotonic()
        report = {
            "accepted": False, "status": "NOT_STARTED", "state_source": "provided_joint_state",
            "model_sha256": self.model_sha256, "root_link": self.root_link,
            "active_joint_names": self.active_joint_names, "guard_joint_names": self.guard_joint_names,
            "unchecked_joint_names": self.unchecked_joint_names, "model_mass_kg": self.total_mass,
            "model_warnings": self.model_warnings,
            "allowable_efforts": dict(self.policy.allowable_efforts),
            "effort_units": {n: "N" if self.joints[n].kind == "prismatic" else "N*m"
                             for n in self.guard_joint_names},
            "urdf_effort_reference_only": {n: self.joints[n].urdf_effort
                                          for n in self.guard_joint_names},
            "approximate_requested": bool(allow_approximate),
            "approximate_accepted": False,
            "collision_check": "NOT_PERFORMED; keep downstream MoveIt validation",
            "dynamic_load_check": "NOT_PERFORMED; static gravity only",
            "per_seed": [], "candidates": [], "search_stop": None,
        }
        if not self._lock.acquire(blocking=False):
            return None, "BUSY", {**report, "status": "BUSY"}
        q_output = None
        try:
            if not isinstance(snapshot, JointStateSnapshot):
                raise TypeError("snapshot must be JointStateSnapshot")
            freshness = snapshot.freshness_error(
                self.config.state_max_age_s, require_source_age=self.config.require_source_age)
            if freshness:
                report["status"], report["reason"] = "STALE_STATE", freshness
                report["elapsed_ms"] = (time.monotonic()-start)*1000.
                report["budget_exceeded"] = time.monotonic() > start + self.config.budget_s
                return None, report["status"], json_safe(report)
            report["source_timestamp_verified"] = snapshot.source_age_at_receipt_s is not None
            context = self._state_context(snapshot)
            T = validate_transform(T_target)
            if orientation_mode not in ("full", "tool_dir", "tool_cone"):
                raise ValueError("invalid orientation_mode")
            axis = finite_vector([1., 0., 0.] if tool_axis is None else tool_axis, 3, "tool_axis")
            if np.linalg.norm(axis) < 1e-12:
                raise ValueError("zero tool_axis")
            axis /= np.linalg.norm(axis)
            cone = math.radians(20.) if tool_cone_half_angle is None else float(tool_cone_half_angle)
            if not math.isfinite(cone) or not 0 <= cone <= np.pi:
                raise ValueError("invalid tool_cone_half_angle")
            current = np.array([context[n] for n in self.active_joint_names])
            if self.q_reference is None:
                self.q_reference = current.copy()
                report["reference_initialized_from_first_state"] = True
            previous = current.copy() if q_prev is None else self._active_values(q_prev, "q_prev")
            self._check_active_bounds(previous, "q_prev")
            report["q_reference"] = self.q_reference.copy()
            report["q_prev_source"] = "snapshot" if q_prev is None else "explicit"
            report["ignored_state_joints"] = sorted(set(context)-set(self.joints))
            self._fk(self._full_q(current, context))
            if frame == "local":
                Troot_base = np.array(self.data.oMf[self.base_id].homogeneous, copy=True)
                target = Troot_base @ T
            elif frame in ("world", "model"):
                # "world" is a legacy alias for this URDF's root, NOT arbitrary map/odom.
                target = T
            else:
                raise ValueError("frame must be local, model or world (URDF-root alias)")
            report["target_root"] = target.copy()
            report["orientation_mode"] = orientation_mode
            reference_physics = self._physics(self.q_reference, context, derivatives=False)
            deadline = start + self.config.budget_s
            unlocked, released = 0, False
            for index, (seed, fixed) in enumerate(self._seeds(previous)):
                if time.monotonic() >= deadline:
                    report["search_stop"] = "TIME_BUDGET"
                    break
                if not np.any(fixed):
                    unlocked += 1
                try:
                    c, diag = self._one_seed(seed, fixed, context, target,
                                             orientation_mode, axis, cone, previous,
                                             reference_physics, deadline, index, released)
                    report["candidates"].append(c)
                    report["per_seed"].append(diag)
                    if np.any(fixed) and c["peak_ratio"] > self.policy.soft_ratio:
                        released = True
                except Exception as exc:
                    # Visible failure, never a silently disabled physics term.
                    report["per_seed"].append(
                        {"seed_index": index, "error": f"{type(exc).__name__}: {exc}",
                         "stop_reason": "PHYSICS_OR_NUMERIC_ERROR"})
                    continue
                best = select_load_candidate(report["candidates"], self.policy)
                enough_unlocked = (not np.any(self.waist)
                                   or unlocked >= self.config.min_unlocked_seeds)
                if (index+1 >= self.config.min_seeds and enough_unlocked and best is not None
                        and best["peak_ratio"] <= self.policy.early_stop_ratio):
                    report["search_stop"] = "LOW_LOAD_AFTER_REQUIRED_COMPARISON"
                    break
            if report["search_stop"] is None:
                report["search_stop"] = "SEED_LIMIT"
            report["unlocked_seeds_attempted"] = unlocked
            report["waist_released_for_load"] = released
            best = select_load_candidate(report["candidates"], self.policy)
            if best is None:
                pose_valid = [c for c in report["candidates"] if c["pose_ok"] and c["bounds_ok"]]
                if pose_valid:
                    report["status"] = "NO_LOAD_SAFE_SOLUTION"
                    report["best_rejected"] = min(pose_valid, key=lambda c: c["peak_ratio"])
                elif not report["candidates"] and any("error" in d for d in report["per_seed"]):
                    report["status"] = "PHYSICS_ERROR"
                elif report["search_stop"] == "TIME_BUDGET":
                    report["status"] = "TIME_BUDGET_EXCEEDED"
                else:
                    report["status"] = "NO_VALID_POSE"
            else:
                final = self._candidate(np.array(best["q"]), context, target, orientation_mode,
                                        axis, cone, previous, best["candidate_index"])
                if not (final["pose_ok"] and final["load_ok"] and final["bounds_ok"]):
                    report["status"] = "FINAL_RECHECK_FAILED"
                else:
                    report["accepted"], report["status"], report["selected"] = True, "SUCCESS", final
                    q_output = final["q"].copy()
            # Freshness is a FINAL acceptance gate too. A long solve cannot
            # turn an old state into a current state by publishing a new stamp.
            freshness = snapshot.freshness_error(
                self.config.state_max_age_s, require_source_age=self.config.require_source_age)
            if freshness:
                report["accepted"], report["status"], report["reason"] = False, "STALE_STATE", freshness
                q_output = None
            if report["accepted"] and update_prev:
                self.q_prev = q_output.copy()
        except Exception as exc:
            report["accepted"], report["status"] = False, "INVALID_INPUT_OR_MODEL"
            report["reason"] = f"{type(exc).__name__}: {exc}"
            q_output = None
        finally:
            report["elapsed_ms"] = (time.monotonic()-start)*1000.
            report["budget_exceeded"] = time.monotonic() > start + self.config.budget_s
            self._lock.release()
        safe_report = json_safe(report)
        if verbose:
            print(json.dumps(safe_report, ensure_ascii=False, allow_nan=False))
        return q_output, safe_report["status"], safe_report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--urdf", required=True, help="URDF XML file; may be exported from running robot_description")
    p.add_argument("--base-link", required=True)
    p.add_argument("--ee-link", required=True)
    p.add_argument("--inspect", action="store_true", help="print UNFILLED config/state requirements; no solve")
    args = p.parse_args()
    xml = Path(args.urdf).expanduser().read_text(encoding="utf-8")
    if not args.inspect:
        p.error("CLI is intentionally inspection-only; integrate solve() using a fresh JointStateSnapshot")
    print(json.dumps(inspect_urdf(xml, args.base_link, args.ee_link), ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())