"""Run small, deterministic simulation-only IK examples."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import BalanceRegion, GraspPair, WholeBodyIK, default_robot
from .math3d import transform


def run(mode: str = "both") -> dict:
    robot = default_robot()
    ik = WholeBodyIK(robot)
    q0 = robot.nominal.copy()
    out = {"model": robot.robot_name, "dof": robot.dof}

    if mode in ("both", "independent"):
        # Generate two compatible, but independently specified, reachable targets.
        q_goal = q0.copy()
        q_goal[robot.index["waist_yaw_joint"]] = 0.10
        q_goal[robot.index["left_shoulder_yaw_joint"]] = -0.12
        q_goal[robot.index["left_elbow_joint"]] = 1.05
        q_goal[robot.index["right_shoulder_yaw_joint"]] = 0.08
        q_goal[robot.index["right_elbow_joint"]] = 0.75
        independent = ik.solve_independent(q0, robot.fk(q_goal, "left"), robot.fk(q_goal, "right"))
        out["independent_hands"] = independent.as_dict(robot.names)

    if mode in ("both", "carry"):
        # Capture rigid grasps from the home pose, then command one shared object pose.
        object_home = GraspPair.infer_object_pose(robot, q0)
        grasps = GraspPair.from_current(robot, q0)
        object_target = object_home @ transform(xyz=(0.035, 0.0, 0.025), rpy=(0.0, 0.0, 0.035))
        balance = BalanceRegion(((-0.5, -0.5), (0.8, -0.5), (0.8, 0.5), (-0.5, 0.5)), margin=0.02)
        carry = ik.solve_object_carry(q0, object_target, grasps,
                                      balance_region=balance, payload_mass=2.0)
        out["coupled_object_carry"] = carry.as_dict(robot.names)

    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Whole-body bimanual IK simulation demo")
    parser.add_argument("--mode", choices=("both", "independent", "carry"), default="both")
    parser.add_argument("--output", type=Path, help="optional JSON output path")
    args = parser.parse_args(argv)
    data = run(args.mode)
    text = json.dumps(data, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0 if all(data[k]["success"] for k in data if isinstance(data[k], dict)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
