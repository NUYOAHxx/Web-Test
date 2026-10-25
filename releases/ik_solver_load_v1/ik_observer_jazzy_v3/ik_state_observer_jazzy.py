#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ROS 2 Jazzy: read-only MoveIt state observer for an EXISTING RViz.

Reads JointState + robot_description (topic or GetParameters service).
Publishes ONLY MarkerArray. No IK, actions, JointState publisher or TF publisher.
Pure helpers require NumPy only. ROS imports are lazy for offline tests.
Static gravity estimates are model quantities, NOT measured motor loads.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def vector(text, default):
    values = np.array(default if text is None else text.split(), dtype=float)
    if values.shape != (3,) or not np.all(np.isfinite(values)):
        raise ValueError(f"expected three finite numbers, got {text!r}")
    return values


def rigid_inverse(T):
    result = np.eye(4)
    result[:3, :3] = T[:3, :3].T
    result[:3, 3] = -result[:3, :3] @ T[:3, 3]
    return result


def point_transform(T, p):
    return T[:3, :3] @ np.asarray(p) + T[:3, 3]


def origin(element):
    xyz = vector(None if element is None else element.get("xyz"), (0., 0., 0.))
    r, p, y = vector(None if element is None else element.get("rpy"), (0., 0., 0.))
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    T = np.eye(4)
    T[:3, :3] = [
        [cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
        [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],
        [-sp, cp*sr, cp*cr],
    ]
    T[:3, 3] = xyz
    return T


def axis_rotation(axis, angle):
    x, y, z = axis
    K = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    return np.eye(3) + math.sin(angle)*K + (1.-math.cos(angle))*(K @ K)


def quaternion_matrix(q):
    q = np.asarray(q, dtype=float)
    norm = np.linalg.norm(q)
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("invalid TF quaternion")
    x, y, z, w = q/norm
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])


def tf_matrix(msg):
    t, q = msg.transform.translation, msg.transform.rotation
    T = np.eye(4)
    T[:3, :3] = quaternion_matrix([q.x, q.y, q.z, q.w])
    T[:3, 3] = [t.x, t.y, t.z]
    if not np.all(np.isfinite(T)):
        raise ValueError("non-finite TF")
    return T


def finite_float(text):
    if text is None:
        return None
    value = float(text)
    return value if math.isfinite(value) else None


@dataclass
class Link:
    name: str
    mass: float
    local_com: np.ndarray
    geometry: bool
    mass_valid: bool


@dataclass
class Joint:
    name: str
    kind: str
    parent: str
    child: str
    T_origin: np.ndarray
    axis: np.ndarray
    lower: float | None
    upper: float | None
    effort: float | None
    mimic: tuple[str, float, float] | None


class UrdfModel:
    """Fixed-root URDF tree; no inertial tensor needed for STATIC gravity."""
    def __init__(self, xml):
        if "<!DOCTYPE" in xml.upper() or "<!ENTITY" in xml.upper():
            raise ValueError("DOCTYPE/ENTITY is not accepted")
        root = ET.fromstring(xml)
        if root.tag != "robot":
            raise ValueError("robot_description is not a URDF robot")
        self.sha256 = hashlib.sha256(xml.encode()).hexdigest()
        self.links, self.joints = {}, {}
        self.children = {}
        for element in root.findall("link"):
            name = element.get("name")
            if not name or name in self.links:
                raise ValueError("missing/duplicate link name")
            inertial = element.find("inertial")
            mass_node = None if inertial is None else inertial.find("mass")
            mass = finite_float(None if mass_node is None else mass_node.get("value"))
            valid = mass is not None and mass >= 0
            if mass is not None and mass < 0:
                raise ValueError(f"negative mass: {name}")
            local_com = origin(None if inertial is None else inertial.find("origin"))[:3, 3]
            geometry = element.find("visual") is not None or element.find("collision") is not None
            self.links[name] = Link(name, mass if valid else 0., local_com, geometry, valid)
        parents = {}
        for element in root.findall("joint"):
            name, kind = element.get("name"), element.get("type")
            if not name or name in self.joints:
                raise ValueError("missing/duplicate joint name")
            if kind not in ("fixed", "revolute", "continuous", "prismatic"):
                raise ValueError(f"{name}: unsupported joint type {kind!r}; no guessed floating base")
            pe, ce = element.find("parent"), element.find("child")
            parent = None if pe is None else pe.get("link")
            child = None if ce is None else ce.get("link")
            if parent not in self.links or child not in self.links or child in parents:
                raise ValueError(f"invalid parent/child tree at {name}")
            parents[child] = parent
            ae = element.find("axis")
            axis = vector(None if ae is None else ae.get("xyz"), (1., 0., 0.))
            if kind != "fixed" and np.linalg.norm(axis) < 1e-12:
                raise ValueError(f"zero joint axis: {name}")
            if np.linalg.norm(axis) > 0:
                axis /= np.linalg.norm(axis)
            le = element.find("limit")
            low = finite_float(None if le is None else le.get("lower"))
            high = finite_float(None if le is None else le.get("upper"))
            effort = finite_float(None if le is None else le.get("effort"))
            if effort is not None and effort <= 0:
                effort = None
            if kind in ("continuous", "fixed"):
                low = high = None  # no invented +/-pi mechanical stop
            if low is not None and high is not None and low > high:
                raise ValueError(f"reversed joint limits: {name}")
            me = element.find("mimic")
            mimic = None
            if me is not None:
                multiplier = finite_float(me.get("multiplier", "1"))
                offset = finite_float(me.get("offset", "0"))
                if multiplier is None or offset is None:
                    raise ValueError(f"invalid mimic coefficients: {name}")
                mimic = (me.get("joint"), multiplier, offset)
            self.joints[name] = Joint(name, kind, parent, child, origin(element.find("origin")),
                                      axis, low, high, effort, mimic)
            self.children.setdefault(parent, []).append(name)
        roots = set(self.links)-set(parents)
        if len(roots) != 1:
            raise ValueError(f"expected one URDF root, got {sorted(roots)}")
        self.root = roots.pop()
        self.order, visited = [], set()
        stack = [self.root]
        while stack:
            link = stack.pop()
            if link in visited:
                raise ValueError("cyclic URDF")
            visited.add(link)
            for name in self.children.get(link, ()):
                self.order.append(name)
                stack.append(self.joints[name].child)
        # DFS append above can place a sibling before another sibling's children,
        # but every parent joint is appended before its descendants.
        if visited != set(self.links):
            raise ValueError("disconnected/cyclic URDF")
        self.independent = [name for name, j in self.joints.items()
                            if j.kind != "fixed" and j.mimic is None]
        self._mimic_map = {}
        for name, joint in self.joints.items():
            if joint.kind != "fixed":
                self._mimic_root(name, set())
        self.mass_issues = [name for name, link in self.links.items()
                            if link.geometry and (not link.mass_valid or link.mass <= 0)]
        self.no_inertial = [name for name, link in self.links.items() if not link.mass_valid]

    def _mimic_root(self, name, visiting):
        if name in self._mimic_map:
            return self._mimic_map[name]
        if name in visiting or name not in self.joints:
            raise ValueError(f"missing/cyclic mimic dependency: {name}")
        joint = self.joints[name]
        if joint.kind == "fixed":
            raise ValueError("mimic source must be a movable joint")
        if joint.mimic is None:
            answer = name, 1., 0.
        else:
            source, scale, offset = joint.mimic
            master, source_scale, source_offset = self._mimic_root(source, visiting | {name})
            answer = master, scale*source_scale, scale*source_offset+offset
        self._mimic_map[name] = answer
        return answer

    def kinematics(self, independent_positions):
        q, links, joint_frames = {}, {self.root: np.eye(4)}, {}
        for name, (master, scale, offset) in self._mimic_map.items():
            if master not in independent_positions:
                raise ValueError(f"missing independent joint: {master}")
            value = scale*float(independent_positions[master])+offset
            if not math.isfinite(value):
                raise ValueError(f"non-finite q: {name}")
            q[name] = value
        for name in self.order:
            j = self.joints[name]
            T = links[j.parent] @ j.T_origin
            joint_frames[name] = T
            motion = np.eye(4)
            if j.kind in ("revolute", "continuous"):
                motion[:3, :3] = axis_rotation(j.axis, q[name])
            elif j.kind == "prismatic":
                motion[:3, 3] = j.axis*q[name]
            links[j.child] = T @ motion
        return q, links, joint_frames

    def evaluate(self, positions, gravity_root=None, allow_partial_mass=False):
        q, links, frames = self.kinematics(positions)
        masses, moments, link_coms = {}, {}, {}
        for name, link in self.links.items():
            p = point_transform(links[name], link.local_com)
            link_coms[name] = p
            masses[name], moments[name] = link.mass, link.mass*p
        for name in reversed(self.order):
            j = self.joints[name]
            masses[j.parent] += masses[j.child]
            moments[j.parent] = moments[j.parent]+moments[j.child]
        total = masses[self.root]
        enabled = total > 0 and (allow_partial_mass or not self.mass_issues)
        com = moments[self.root]/total if enabled else None
        physical, actuated = {}, {name: 0. for name in self.independent}
        if enabled and gravity_root is not None:
            g = np.asarray(gravity_root, dtype=float)
            if g.shape != (3,) or not np.all(np.isfinite(g)):
                raise ValueError("invalid gravity")
            for name, j in self.joints.items():
                if j.kind == "fixed":
                    continue
                T = frames[name]
                axis = T[:3, :3] @ j.axis
                force = masses[j.child]*g
                lever_mass = moments[j.child]-masses[j.child]*T[:3, 3]
                # Compensating actuator effort = NEGATIVE gravity generalized force.
                tau = (-float(axis @ force) if j.kind == "prismatic" else
                       -float(axis @ np.cross(lever_mass, g)))
                physical[name] = tau
                master, scale, _ = self._mimic_map[name]
                actuated[master] += scale*tau  # virtual-work reduction for mimic coupling
        else:
            actuated = {}
        return dict(q=q, link_frames=links, joint_frames=frames, link_coms=link_coms,
                    mass_kg=total, com=com, gravity_physical=physical,
                    gravity_actuated=actuated,
                    mass_state=("NO_POSITIVE_MASS" if total <= 0 else
                                "PARTIAL_MODEL" if self.mass_issues else "URDF_MODEL"))


@dataclass
class Sample:
    value: float
    stamp_ns: int
    received: float


class JointCache:
    def __init__(self):
        self.positions, self.efforts = {}, {}

    def clear(self):
        self.positions.clear()
        self.efforts.clear()

    def update(self, names, positions, efforts, stamp_ns, received, allowed=None):
        names, positions, efforts = list(names), list(positions), list(efforts)
        if (not names or len(set(names)) != len(names) or
                (positions and len(positions) != len(names)) or
                (efforts and len(efforts) != len(names)) or
                not all(isinstance(n, str) and n for n in names)):
            raise ValueError("malformed JointState names/array lengths")
        if positions and not np.all(np.isfinite(positions)):
            raise ValueError("non-finite JointState position")
        if stamp_ns < 0:
            raise ValueError("negative JointState timestamp")
        for i, name in enumerate(names):
            if allowed is not None and name not in allowed:
                continue
            if allowed is None and name not in self.positions and len(self.positions) >= 5000:
                continue
            for target, array in ((self.positions, positions), (self.efforts, efforts)):
                if not array:
                    continue
                if allowed is None and name not in target and len(target) >= 5000:
                    continue
                previous = target.get(name)
                if previous and stamp_ns and previous.stamp_ns and stamp_ns < previous.stamp_ns:
                    continue  # do not regress state using out-of-order packets
                if math.isfinite(float(array[i])):
                    target[name] = Sample(float(array[i]), stamp_ns, received)
                else:
                    target.pop(name, None)  # unknown effort is not zero

    def snapshot(self, required, now, timeout, max_skew, now_ros_ns=0):
        missing, stale, future = [], [], []
        samples = []
        for name in required:
            item = self.positions.get(name)
            if item is None:
                missing.append(name)
                continue
            age = now-item.received
            if age > timeout:
                stale.append(name)
            if now_ros_ns > 0 and item.stamp_ns > 0:
                ros_age = (now_ros_ns-item.stamp_ns)*1e-9
                if ros_age > timeout:
                    stale.append(name)
                if ros_age < -max_skew:
                    future.append(name)
            samples.append(item)
        issues = []
        if missing:
            issues.append("missing: "+",".join(missing))
        if stale:
            issues.append("stale: "+",".join(sorted(set(stale))))
        if future:
            issues.append("future timestamp / clock mismatch: "+",".join(future))
        stamps = [s.stamp_ns for s in samples if s.stamp_ns > 0]
        unstamped = [n for n in required if n in self.positions and self.positions[n].stamp_ns == 0]
        if stamps and (max(stamps)-min(stamps))*1e-9 > max_skew:
            issues.append("joint timestamp skew")
        if samples and max(s.received for s in samples)-min(s.received for s in samples) > max_skew:
            issues.append("joint arrival skew")
        return dict(
            valid=not issues, issues=issues,
            positions={n: self.positions[n].value for n in required if n in self.positions},
            efforts={n: s.value for n, s in self.efforts.items()
                     if now-s.received <= timeout and
                     (not now_ros_ns or not s.stamp_ns or
                      -max_skew <= (now_ros_ns-s.stamp_ns)*1e-9 <= timeout)},
            stamp_ns=0 if unstamped else max(stamps, default=0),
            unstamped=unstamped,
            age_s=max((now-s.received for s in samples), default=math.inf),
        )


def joint_margin(joint, position):
    if joint.kind == "continuous" or joint.lower is None or joint.upper is None:
        return None
    return min(position-joint.lower, joint.upper-position)


def load_ratio(tau, effort_limit):
    if tau is None or effort_limit is None or effort_limit <= 0:
        return None
    return abs(tau)/effort_limit


def position_bounds_text(joint):
    if joint.kind == "continuous":
        return "continuous: no hard position stop"
    if joint.lower is None or joint.upper is None:
        return "URDF position limits unknown"
    if joint.kind == "prismatic":
        return f"range=[{joint.lower:.3f}, {joint.upper:.3f}] m"
    return f"range=[{math.degrees(joint.lower):.1f}, {math.degrees(joint.upper):.1f}] deg"


def support_projection(com_root, gravity_root, T_support_root, half_extents=None):
    if com_root is None or gravity_root is None or T_support_root is None:
        return dict(state="UNAVAILABLE", projection=None, margin_m=None)
    com = point_transform(T_support_root, com_root)
    g = T_support_root[:3, :3] @ gravity_root
    if g[2] >= -1e-9 or com[2] < -1e-9:
        return dict(state="INVALID_PLANE_OR_GRAVITY", projection=None, margin_m=None)
    p = com-(com[2]/g[2])*g
    margin = (None if half_extents is None else
              float(min(half_extents[0]-abs(p[0]), half_extents[1]-abs(p[1]))))
    return dict(state=("PROJECTION_ONLY" if margin is None else
                       "INSIDE_MODEL_RECTANGLE" if margin > 0 else "EDGE_OR_OUTSIDE"),
                projection=point_transform(rigid_inverse(T_support_root), p), margin_m=margin)


def reference_notes(model, reference):
    notes = []
    for name, limits in reference.get("joints", {}).items():
        joint = model.joints.get(name)
        if joint is None:
            notes.append(f"{name}: reference name not in loaded model")
            continue
        if joint.kind == "continuous":
            notes.append(f"{name}: continuous; finite reference range is NOT a hard stop")
            continue
        if joint.kind != "revolute":
            notes.append(f"{name}: degree reference cannot be applied to {joint.kind}")
            continue
        low, high = math.radians(limits["lower_deg"]), math.radians(limits["upper_deg"])
        if (joint.lower is None or joint.upper is None or
                abs(joint.lower-low) > 1e-3 or abs(joint.upper-high) > 1e-3):
            notes.append(f"{name}: runtime URDF position limits differ from reference")
    return notes


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


class CsvLog:
    def __init__(self, filename, configuration):
        self.file = None
        if filename is None:
            return
        path = Path(filename).expanduser()
        metadata = path.with_suffix(path.suffix+".meta.json")
        if path.exists() or metadata.exists():
            raise FileExistsError("choose a new log path; no overwrite")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = path.open("x", newline="", encoding="utf-8")
        try:
            with metadata.open("x", encoding="utf-8") as f:
                json.dump(json_safe(configuration), f, indent=2, ensure_ascii=False, allow_nan=False)
            self.writer = csv.writer(self.file)
            self.writer.writerow(["ros_time_ns", "model_sha256", "status", "data_json"])
        except BaseException:
            self.file.close()
            raise

    def write(self, stamp, sha, status, data):
        if self.file is not None:
            self.writer.writerow([stamp, sha, status,
                                  json.dumps(json_safe(data), ensure_ascii=False, allow_nan=False)])
            self.file.flush()

    def close(self):
        if self.file is not None:
            self.file.close()


def argument_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--joint-topic", default="/joint_states")
    p.add_argument("--description-topic", default="/robot_description")
    p.add_argument("--description-node", default="/robot_state_publisher",
                   help="GetParameters fallback node; try /move_group if needed")
    p.add_argument("--description-parameter", default="robot_description")
    p.add_argument("--marker-topic", default="/ha_ik_monitor/markers")
    p.add_argument("--frame-prefix", default="", help="match existing TF frame_prefix, e.g. robot/")
    p.add_argument("--gravity-frame", default="world", help="frame in which --gravity is expressed")
    p.add_argument("--gravity", type=float, nargs=3, default=(0., 0., -9.80665))
    p.add_argument("--tf-time", choices=("state", "latest"), default="state")
    p.add_argument("--support-frame", default=None, help="optional existing TF; its Z=0 is support plane")
    p.add_argument("--support-half-extents", type=float, nargs=2, default=None,
                   help="optional calibrated rectangle half-widths in support frame, metres")
    p.add_argument("--rate", type=float, default=10.)
    p.add_argument("--console-rate", type=float, default=1.)
    p.add_argument("--state-timeout", type=float, default=2.)
    p.add_argument("--max-skew", type=float, default=.5)
    p.add_argument("--joint-labels", action="store_true")
    p.add_argument("--print-joints", action="store_true")
    p.add_argument("--show-link-com", action="store_true")
    p.add_argument("--joint-names", default=None, help="comma-separated display filter; full model still updated")
    p.add_argument("--allow-partial-mass", action="store_true",
                   help="explicitly allow known-mass-only estimates; always labeled partial")
    p.add_argument("--limits-reference", default=None, help="JSON comparison only; never overrides/clips states")
    p.add_argument("--csv", default=None, help="new CSV path; also creates .meta.json")
    p.add_argument("--save-description", default=None, help="optional NEW .urdf path for runtime model audit")
    return p


def validate_args(args):
    for name in ("rate", "console_rate", "state_timeout", "max_skew"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not np.all(np.isfinite(args.gravity)) or np.linalg.norm(args.gravity) < 1e-9:
        raise ValueError("invalid gravity")
    if args.support_half_extents is not None:
        if not args.support_frame or not all(
                math.isfinite(v) and v > 0 for v in args.support_half_extents):
            raise ValueError("support rectangle requires explicit --support-frame and positive dimensions")
    if not args.description_node.startswith("/"):
        raise ValueError("--description-node must be a fully qualified ROS node name")
    if args.frame_prefix.startswith("/"):
        raise ValueError("TF frame names must not start with '/'")


def split_ros_args(argv):
    if "--ros-args" in argv:
        index = argv.index("--ros-args")
        return argv[:index], argv[index:]
    return argv, []


def run_ros(args, ros_args):
    import rclpy
    from rclpy.node import Node
    from rclpy.time import Time
    from rclpy.duration import Duration
    from rclpy.clock import ClockType
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, qos_profile_sensor_data
    from rclpy.executors import ExternalShutdownException
    from rcl_interfaces.srv import GetParameters
    from sensor_msgs.msg import JointState
    from std_msgs.msg import String
    from visualization_msgs.msg import Marker, MarkerArray
    from geometry_msgs.msg import Point
    from tf2_ros import Buffer, TransformListener

    class Observer(Node):
        def __init__(self):
            super().__init__("ik_state_observer")
            self.model, self.cache, self.reference = None, JointCache(), {}
            self.last_status, self.live_markers = "", set()
            self.last_console = self.last_parameter_request = -math.inf
            self.parameter_future = None
            self.last_ros_time = 0
            self.saved_description = False
            self.reference_warnings = []
            if args.limits_reference:
                self.reference = json.loads(Path(args.limits_reference).read_text(encoding="utf-8"))
            self.log = CsvLog(args.csv, dict(
                args=vars(args), script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                python=sys.version, numpy=np.__version__, reference=self.reference,
                state_source="external JointState; effort source not certified",
                torque_method="URDF static gravity with mimic virtual-work reduction",
            ))
            desc_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                                  durability=DurabilityPolicy.TRANSIENT_LOCAL)
            self.create_subscription(String, args.description_topic, self.description_cb, desc_qos)
            self.create_subscription(JointState, args.joint_topic, self.joint_cb, qos_profile_sensor_data)
            # This is the ONLY publisher created by this node.
            self.publisher = self.create_publisher(MarkerArray, args.marker_topic, 10)
            service = args.description_node.rstrip("/")+"/get_parameters"
            self.parameter_client = self.create_client(GetParameters, service)
            self.tf_buffer = Buffer(cache_time=Duration(seconds=10.), node=self)
            self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=False)
            self.timer = self.create_timer(1./args.rate, self.tick)
            self.get_logger().info("Read-only observer: no IK, trajectory action, JointState or TF output.")

        def frame(self, name):
            return args.frame_prefix+name

        def install_model(self, xml, source):
            if self.model is not None and hashlib.sha256(xml.encode()).hexdigest() == self.model.sha256:
                return
            candidate = UrdfModel(xml)
            if args.joint_names:
                unknown = set(args.joint_names.split(","))-set(candidate.joints)
                if unknown:
                    raise ValueError(f"display filter has unknown joints: {sorted(unknown)}")
            reference_warnings = reference_notes(candidate, self.reference)
            if args.save_description and not self.saved_description:
                path = Path(args.save_description).expanduser()
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("x", encoding="utf-8") as f:
                    f.write(xml)
                self.saved_description = True
            self.model = candidate
            self.cache.clear()  # never combine pre-update positions with a new model
            self.reference_warnings = reference_warnings
            self.get_logger().info(
                f"Model loaded from {source}: root={candidate.root}, "
                f"{len(candidate.independent)} independent joints, sha256={candidate.sha256[:12]}")
            for message in self.reference_warnings:
                self.get_logger().warning(message)
            if candidate.mass_issues:
                self.get_logger().warning(
                    "Missing/zero mass on geometry links: "+", ".join(candidate.mass_issues)+
                    "; estimates disabled unless --allow-partial-mass is explicit")
            self.log.write(self.get_clock().now().nanoseconds, candidate.sha256, "MODEL_LOADED",
                           dict(source=source, root=candidate.root, independent=candidate.independent,
                                mass_issues=candidate.mass_issues, no_inertial=candidate.no_inertial,
                                limits={n: dict(type=j.kind, lower=j.lower, upper=j.upper,
                                               effort=j.effort) for n, j in candidate.joints.items()},
                                reference_notes=self.reference_warnings))

        def description_cb(self, msg):
            try:
                self.install_model(msg.data, args.description_topic)
            except Exception as exc:
                self.model = None
                self.cache.clear()
                self.get_logger().error(f"robot_description rejected: {exc}")

        def parameter_done(self, future):
            try:
                response = future.result()
                if self.model is None and response.values and response.values[0].type == 4:
                    self.install_model(response.values[0].string_value,
                                       args.description_node+":"+args.description_parameter)
            except Exception as exc:
                self.get_logger().warning(f"description parameter unavailable: {exc}")

        def request_description(self, now):
            if self.model is not None or now-self.last_parameter_request < 2.:
                return
            if self.parameter_future is not None and not self.parameter_future.done():
                if now-self.last_parameter_request > 5.:
                    self.parameter_future.cancel()
                return
            self.last_parameter_request = now
            if self.parameter_client.service_is_ready():
                request = GetParameters.Request()
                request.names = [args.description_parameter]
                self.parameter_future = self.parameter_client.call_async(request)
                self.parameter_future.add_done_callback(self.parameter_done)

        def joint_cb(self, msg):
            stamp = msg.header.stamp.sec*1_000_000_000+msg.header.stamp.nanosec
            try:
                self.cache.update(msg.name, msg.position, msg.effort, stamp, time.monotonic(),
                                  None if self.model is None else set(self.model.joints))
            except Exception as exc:
                self.get_logger().warning(f"JointState rejected: {exc}")

        def transform(self, target, source, stamp_ns):
            if target == source:
                return np.eye(4)
            when = (Time(nanoseconds=stamp_ns, clock_type=ClockType.ROS_TIME)
                    if args.tf_time == "state" and stamp_ns > 0 else Time())
            return tf_matrix(self.tf_buffer.lookup_transform(
                target, source, when, timeout=Duration(seconds=0.)))

        def publish_markers(self, result, projection, status, snapshot):
            frame = self.frame(self.model.root) if self.model is not None else args.gravity_frame
            stamp = (Time(nanoseconds=snapshot["stamp_ns"], clock_type=ClockType.ROS_TIME).to_msg()
                     if snapshot and snapshot["valid"] and snapshot["stamp_ns"] > 0
                     else self.get_clock().now().to_msg())
            array, keys = MarkerArray(), set()

            def p(value):
                msg = Point()
                msg.x, msg.y, msg.z = map(float, value)
                return msg

            def new(ns, idx, kind, color):
                msg = Marker()
                msg.header.frame_id, msg.header.stamp = frame, stamp
                msg.ns, msg.id, msg.type, msg.action = "ik_observer/"+ns, idx, kind, Marker.ADD
                msg.pose.orientation.w = 1.
                msg.color.r, msg.color.g, msg.color.b, msg.color.a = map(float, color)
                lifetime = max(.1, min(args.state_timeout, max(.5, 3./args.rate)))
                msg.lifetime.sec = int(lifetime)
                msg.lifetime.nanosec = int((lifetime-int(lifetime))*1e9)
                array.markers.append(msg)
                keys.add((msg.ns, idx))
                return msg

            def sphere(ns, idx, value, diameter, color):
                msg = new(ns, idx, Marker.SPHERE, color)
                msg.pose.position = p(value)
                msg.scale.x = msg.scale.y = msg.scale.z = float(diameter)

            def text(ns, idx, value, label, color, size=.035):
                msg = new(ns, idx, Marker.TEXT_VIEW_FACING, color)
                msg.pose.position, msg.text, msg.scale.z = p(value), label, size

            def line(ns, values, color):
                msg = new(ns, 0, Marker.LINE_STRIP, color)
                msg.scale.x, msg.points = .004, [p(v) for v in values]

            com = None if result is None else result["com"]
            banner_position = np.array([0., 0., .5]) if com is None else com+np.array([0., 0., .12])
            text("status", 0, banner_position, "URDF MODEL / NOT MEASURED\n"+status[:260],
                 (.2, .9, 1., 1.) if status.startswith("OK:") and "|" not in status else
                 (1., .65, .15, 1.))
            if result is not None:
                if com is not None:
                    sphere("com", 0, com, .055, (.15, .75, 1., .95))
                    if projection and projection["projection"] is not None:
                        pp = projection["projection"]
                        sphere("com_projection", 0, pp, .025, (.3, .7, 1., .9))
                        line("gravity_line", [com, pp], (.3, .7, 1., .8))
                if args.show_link_com:
                    for idx, (name, value) in enumerate(result["link_coms"].items()):
                        if self.model.links[name].mass > 0:
                            sphere("link_com", idx, value, .012, (.6, .6, 1., .7))
                selected = (set(args.joint_names.split(",")) if args.joint_names else
                            set(self.model.independent))
                for idx, (name, joint) in enumerate(self.model.joints.items()):
                    if joint.kind == "fixed" or name not in selected:
                        continue
                    tau = result["gravity_actuated"].get(name)
                    if joint.mimic is not None:
                        tau = result["gravity_physical"].get(name)
                    ratio = load_ratio(tau, joint.effort)
                    color = (.5, .5, .5, .8)
                    if ratio is not None:
                        t = float(np.clip(ratio, 0., 1.))
                        color = (min(1., 2*t), min(.9, 2*(1-t)), .12, .9)
                    position = result["joint_frames"][name][:3, 3]
                    sphere("gravity_load", idx, position, .025+.02*min(ratio or 0., 1.), color)
                    if args.joint_labels:
                        unit = "N" if joint.kind == "prismatic" else "Nm"
                        estimate = "?" if tau is None else f"{tau:+.2f} {unit}"
                        margin = joint_margin(joint, result["q"][name])
                        bound = ("continuous" if joint.kind == "continuous" else
                                 "limits unknown" if margin is None else
                                 f"margin={margin:.3f} m" if joint.kind == "prismatic" else
                                 f"margin={math.degrees(margin):.1f} deg")
                        pct = "?" if ratio is None else f"{ratio*100:.1f}%"
                        text("joint_labels", idx, position+np.array([0., 0., .045]),
                             f"{name}\ngravity={estimate} / URDF={pct}\n"
                             f"{position_bounds_text(joint)}\n{bound}", color, .022)
                if args.support_half_extents and projection and projection.get("T_support_root") is not None:
                    hx, hy = args.support_half_extents
                    T = rigid_inverse(projection["T_support_root"])
                    corners = [(hx, hy, 0.), (-hx, hy, 0.), (-hx, -hy, 0.),
                               (hx, -hy, 0.), (hx, hy, 0.)]
                    line("support_rectangle", [point_transform(T, c) for c in corners],
                         (.2, .65, 1., .8))
            for ns, idx in self.live_markers-keys:
                msg = Marker()
                msg.header.frame_id, msg.header.stamp = frame, stamp
                msg.ns, msg.id, msg.action = ns, idx, Marker.DELETE
                array.markers.append(msg)
            self.live_markers = keys
            self.publisher.publish(array)

        def tick(self):
            now, ros_now = time.monotonic(), self.get_clock().now().nanoseconds
            if self.last_ros_time and ros_now < self.last_ros_time:
                self.cache.clear()
            self.last_ros_time = ros_now
            self.request_description(now)
            result, projection, snapshot = None, None, None
            status, extra = "WAIT_MODEL: robot_description topic / parameter", {}
            try:
                if self.model is not None:
                    snapshot = self.cache.snapshot(self.model.independent, now, args.state_timeout,
                                                   args.max_skew, ros_now)
                    if not snapshot["valid"]:
                        status = "WAIT_STATE: "+"; ".join(snapshot["issues"])
                    else:
                        gravity = None
                        warnings = (["ZERO_STAMP: receipt freshness / latest TF; not timestamp-aligned"]
                                    if snapshot["unstamped"] else [])
                        try:
                            T = self.transform(self.frame(self.model.root), args.gravity_frame,
                                               snapshot["stamp_ns"])
                            gravity = T[:3, :3] @ np.asarray(args.gravity)
                        except Exception as exc:
                            warnings.append("GRAVITY_TF_UNKNOWN: "+str(exc)[:100])
                        result = self.model.evaluate(snapshot["positions"], gravity, args.allow_partial_mass)
                        status = "OK: "+result["mass_state"] if result["com"] is not None else \
                            "MASS_UNAVAILABLE: "+result["mass_state"]
                        if result["mass_state"] == "PARTIAL_MODEL":
                            warnings.append("partial mass estimate")
                        violated = [n for n, value in result["q"].items()
                                    if (joint_margin(self.model.joints[n], value) is not None and
                                        joint_margin(self.model.joints[n], value) < -1e-6)]
                        if violated:
                            warnings.append("URDF_LIMIT_VIOLATION: "+",".join(violated))
                        overloaded = [n for n, tau in result["gravity_actuated"].items()
                                      if (load_ratio(tau, self.model.joints[n].effort) is not None and
                                          load_ratio(tau, self.model.joints[n].effort) > 1.)]
                        if overloaded:
                            warnings.append("MODEL_GRAVITY_OVER_URDF_EFFORT: "+",".join(overloaded))
                        if args.support_frame:
                            try:
                                T = self.transform(args.support_frame, self.frame(self.model.root),
                                                   snapshot["stamp_ns"])
                                projection = support_projection(result["com"], gravity, T,
                                                                args.support_half_extents)
                                projection["T_support_root"] = T
                            except Exception as exc:
                                warnings.append("SUPPORT_TF_UNKNOWN: "+str(exc)[:80])
                        if warnings:
                            status += " | "+"; ".join(warnings)
                        extra = dict(
                            mass_state=result["mass_state"], mass_kg=result["mass_kg"], com_root_m=result["com"],
                            q=result["q"], gravity_actuated=result["gravity_actuated"],
                            effort_topic_unverified=snapshot["efforts"],
                            urdf_effort_ratio={n: load_ratio(t, self.model.joints[n].effort)
                                               for n, t in result["gravity_actuated"].items()},
                            support=projection, state_age_s=snapshot["age_s"],
                            state_stamp_ns=snapshot["stamp_ns"], unstamped_joints=snapshot["unstamped"],
                            reference_notes=self.reference_warnings)
                self.publish_markers(result, projection, status, snapshot)
            except Exception as exc:
                status = f"MONITOR_ERROR: {type(exc).__name__}: {exc}"
                self.publish_markers(None, None, status, snapshot)
            if now-self.last_console >= 1./args.console_rate:
                self.last_console = now
                self.get_logger().info(status)
                if args.print_joints and result is not None:
                    for name in self.model.independent:
                        j, q = self.model.joints[name], result["q"][name]
                        tau = result["gravity_actuated"].get(name)
                        effort = snapshot["efforts"].get(name)
                        angle = q if j.kind == "prismatic" else math.degrees(q)
                        print(f"{name}: position={angle:+.3f} {'m' if j.kind == 'prismatic' else 'deg'} "
                              f"gravity_model={tau} effort_topic_unverified={effort} "
                              f"{position_bounds_text(j)}", flush=True)
            self.log.write(ros_now, "" if self.model is None else self.model.sha256, status, extra)
            self.last_status = status

        def cleanup(self):
            array = MarkerArray()
            for ns, idx in self.live_markers:
                msg = Marker()
                msg.ns, msg.id, msg.action = ns, idx, Marker.DELETE
                array.markers.append(msg)
            try:
                if rclpy.ok():
                    self.publisher.publish(array)
            finally:
                self.log.close()

    rclpy.init(args=[sys.argv[0]]+ros_args)
    node = None
    try:
        node = Observer()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            try:
                node.cleanup()
            finally:
                node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    application, ros_args = split_ros_args(argv)
    args = argument_parser().parse_args(application)
    try:
        validate_args(args)
        run_ros(args, ros_args)
    except Exception as exc:
        print(f"observer failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())