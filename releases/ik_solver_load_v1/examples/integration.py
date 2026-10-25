"""Integration skeleton. No ROS publisher, trajectory execution or fake limits.

Copy into the user's existing node and adapt its own worker/locking.
Do not copy this file over an existing executor/controller.
"""
from pathlib import Path
import json
import math
import numpy as np

from ha_arm_ik.load_policy import JointStateSnapshot, LoadPolicy
from ha_arm_ik.solver_load_aware import LoadAwareIKSolver, SolverConfig


def build_solver(old_solver, confirmed_limits_json, *, massless_links, q_reference=None):
    """confirmed_limits_json is a name->positive allowable effort JSON mapping.

    These values must come from the actual actuator/joint-side continuous/derated
    operating limits, not arbitrary examples. Unit N*m for revolute, N for sliders.
    """
    limits = json.loads(Path(confirmed_limits_json).read_text(encoding="utf-8"))
    return LoadAwareIKSolver.from_existing_solver(
        old_solver,
        policy=LoadPolicy(allowable_efforts=limits),
        config=SolverConfig(require_source_age=True),
        massless_links=massless_links,
        q_reference=q_reference,
    )


class IntegrationSkeleton:
    def __init__(self, ros_node, solver):
        self.node = ros_node
        self.solver = solver
        self.latest_snapshot = None
        self.last_error = None

    def joint_state_callback(self, msg):
        # Capture receipt ONCE, in the callback. Never re-stamp cached positions
        # when a new IK target arrives. Message must cover the complete model.
        try:
            self.latest_snapshot = JointStateSnapshot.from_joint_state(
                msg, now_ros_ns=self.node.get_clock().now().nanoseconds)
            self.last_error = None
        except (ValueError, TypeError, AttributeError) as exc:
            self.latest_snapshot = None
            self.last_error = str(exc)

    def solve_current_target(self, T_target, *, frame="local", orientation_mode="full"):
        snapshot = self.latest_snapshot
        if snapshot is None:
            return None, "NO_STATE", {"accepted":False,"reason":self.last_error}
        # Call from ONE dedicated worker, not the JointState subscription callback.
        # If newer targets arrive while solving, the caller must discard outdated
        # results using its own target version before any MoveIt plan/execution.
        q, status, report = self.solver.solve(
            np.asarray(T_target), frame=frame, snapshot=snapshot,
            allow_approximate=False, orientation_mode=orientation_mode)
        if not report["accepted"]:
            return None, status, report
        # There is deliberately NO MoveIt execution here. Existing program must:
        # 1. verify latest target version and current complete state/context;
        # 2. plan with MoveIt, preserving collision/position/velocity/acceleration checks;
        # 3. validate whole planned trajectory's static/dynamic loads as required;
        # 4. follow the original authorized execution/stop workflow.
        return q, status, report