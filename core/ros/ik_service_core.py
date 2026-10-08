"""ROS 无关的 IK 服务核心逻辑(便于脱离 ROS 单测)。

对外语义对齐 moveit_msgs/srv/GetPositionIK:
    ik_link_names + pose_stamped_vector + seed(joint_state) → solution(joint_state) + error_code
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pinocchio as pin

from core.solver.dual_arm_ik import DualArmIKSolver, IKResult

# moveit_msgs/msg/MoveItErrorCodes
SUCCESS = 1
FAILURE = 99999
FRAME_TRANSFORM_FAILURE = -21
INVALID_ROBOT_STATE = -24
INVALID_LINK_NAME = -25
NO_IK_SOLUTION = -31


@dataclass
class PoseInput:
    frame_id: str
    position: Tuple[float, float, float]
    quat_xyzw: Tuple[float, float, float, float]


def to_se3(p: PoseInput) -> pin.SE3:
    x, y, z, w = p.quat_xyzw
    n = float(np.linalg.norm([x, y, z, w]))
    if n < 1e-9:
        raise ValueError("四元数为零")
    q = pin.Quaternion(w / n, x / n, y / n, z / n)
    return pin.SE3(q.toRotationMatrix(), np.asarray(p.position, dtype=float))


def build_seed(solver: DualArmIKSolver, names: Sequence[str],
               positions: Sequence[float]) -> np.ndarray:
    """按 solver.joint_names 重排 seed;请求里缺失的关节取限位内的 0。"""
    given: Dict[str, float] = dict(zip(names, positions))
    seed = np.array([given.get(n, 0.0) for n in solver.joint_names], dtype=float)
    return np.clip(seed, solver.q_min, solver.q_max)


def solve_request(
    solver: DualArmIKSolver,
    base_frame: str,
    ik_link_names: Sequence[str],
    poses: Sequence[PoseInput],
    seed_names: Sequence[str],
    seed_positions: Sequence[float],
) -> Tuple[int, Dict[str, float], IKResult | None]:
    """返回 (moveit error_code, {joint_name: value}, IKResult)。"""
    left, right = solver.cfg.left_tip, solver.cfg.right_tip
    if len(ik_link_names) != 2 or len(poses) != 2 or \
            set(ik_link_names) != {left, right}:
        return INVALID_LINK_NAME, {}, None
    by_link = dict(zip(ik_link_names, poses))
    for p in poses:
        if p.frame_id not in ("", base_frame):
            return FRAME_TRANSFORM_FAILURE, {}, None
    if len(seed_names) != len(seed_positions):
        return INVALID_ROBOT_STATE, {}, None
    try:
        lt, rt = to_se3(by_link[left]), to_se3(by_link[right])
    except ValueError:
        return FAILURE, {}, None
    seed = build_seed(solver, seed_names, seed_positions)
    res = solver.solve(seed, lt, rt)
    if not res.success:
        return NO_IK_SOLUTION, {}, res
    return SUCCESS, dict(zip(solver.joint_names, map(float, res.solution))), res
