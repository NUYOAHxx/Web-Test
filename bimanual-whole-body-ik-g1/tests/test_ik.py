import numpy as np

from bimanual_wbik import BalanceRegion, GraspPair, SphereObstacle, WholeBodyIK, default_robot
from bimanual_wbik.math3d import pose_error, transform


def test_default_model_is_reduced_unitree_g1():
    robot = default_robot()
    assert "Unitree G1" in robot.robot_name
    assert robot.dof == 16
    assert len(robot.torso) == 2
    assert len(robot.arms["left"]) == len(robot.arms["right"]) == 7
    assert robot.names[0] == "waist_yaw_joint"
    assert robot.names[1] == "waist_pitch_joint"
    assert robot.nominal[robot.index["left_elbow_joint"]] == 0.9
    assert np.isclose(robot.com_proxy_weight, 70.8)


def test_geometric_jacobian_matches_finite_difference():
    robot = default_robot()
    q = np.zeros(robot.dof)
    q[robot.index["waist_yaw_joint"]] = 0.15
    q[robot.index["left_shoulder_pitch_joint"]] = -0.23
    q[robot.index["left_elbow_joint"]] = 0.62
    jac = robot.jacobian(q, "left")
    eps = 1e-7
    for idx in (0, 1, 2, 3, 5, 7, 8):
        q2 = q.copy()
        q2[idx] += eps
        numerical = pose_error(robot.fk(q, "left"), robot.fk(q2, "left")) / eps
        assert np.allclose(numerical, jac[:, idx], atol=3e-5, rtol=3e-4)


def test_analytic_objective_and_closure_jacobians_match_finite_difference():
    robot = default_robot()
    ik = WholeBodyIK(robot)
    q = robot.nominal.copy()
    q[robot.index["waist_yaw_joint"]] += 0.07
    q[robot.index["left_shoulder_pitch_joint"]] -= 0.11
    q[robot.index["right_elbow_joint"]] += 0.14
    q_goal = q.copy()
    q_goal[robot.index["left_wrist_pitch_joint"]] += 0.08
    q_goal[robot.index["right_shoulder_yaw_joint"]] -= 0.09
    left_target, right_target = robot.fk(q_goal, "left"), robot.fk(q_goal, "right")
    evaluator = ik._objective_evaluator(q, left_target, right_target)
    residual, jacobian, poses, geometric = evaluator(q)
    analytic_gradient = jacobian.T @ residual
    eps = 1e-6
    numeric_gradient = np.zeros(robot.dof)
    for i in range(robot.dof):
        delta = np.zeros(robot.dof)
        delta[i] = eps
        rp = evaluator(q + delta)[0]
        rm = evaluator(q - delta)[0]
        numeric_gradient[i] = ((rp @ rp) - (rm @ rm)) / (4.0 * eps)
    assert np.allclose(analytic_gradient, numeric_gradient, atol=2e-6, rtol=2e-5)

    grasps = GraspPair.from_current(robot, q)
    closure_target = np.linalg.inv(grasps.left) @ grasps.right
    _, analytic_closure = ik._closure_residual_jacobian(poses, geometric, closure_target)
    numeric_closure = np.zeros_like(analytic_closure)
    for i in range(robot.dof):
        delta = np.zeros(robot.dof)
        delta[i] = eps
        tp = np.linalg.inv(robot.fk(q + delta, "left")) @ robot.fk(q + delta, "right")
        tm = np.linalg.inv(robot.fk(q - delta, "left")) @ robot.fk(q - delta, "right")
        numeric_closure[:, i] = (
            pose_error(tp, closure_target) - pose_error(tm, closure_target)) / (2.0 * eps)
    assert np.allclose(analytic_closure, numeric_closure, atol=3e-5, rtol=3e-4)


def test_independent_hand_targets_are_solved_jointly():
    robot = default_robot()
    ik = WholeBodyIK(robot)
    q0 = robot.nominal.copy()
    q_goal = q0.copy()
    q_goal[robot.index["waist_yaw_joint"]] = 0.08
    q_goal[robot.index["waist_pitch_joint"]] = 0.025
    q_goal[robot.index["left_shoulder_yaw_joint"]] = -0.10
    q_goal[robot.index["left_elbow_joint"]] = 0.72
    q_goal[robot.index["right_shoulder_yaw_joint"]] = 0.07
    q_goal[robot.index["right_elbow_joint"]] = 1.05
    result = ik.solve_independent(q0, robot.fk(q_goal, "left"), robot.fk(q_goal, "right"))
    assert result.success, result.message
    assert result.left_position_error < 2e-3
    assert result.right_position_error < 2e-3
    assert np.all(result.q >= robot.lower - 1e-8)
    assert np.all(result.q <= robot.upper + 1e-8)


def test_coupled_carry_preserves_exact_relative_hand_pose():
    robot = default_robot()
    ik = WholeBodyIK(robot)
    q0 = robot.nominal.copy()
    object_home = GraspPair.infer_object_pose(robot, q0)
    grasps = GraspPair.from_current(robot, q0)
    target = object_home @ transform(xyz=(0.025, 0.0, 0.02), rpy=(0.0, 0.0, 0.025))
    result = ik.solve_object_carry(q0, target, grasps)
    assert result.success, result.message
    assert result.closure_error is not None and result.closure_error < 1e-5
    assert result.left_position_error < 2e-3
    assert result.right_position_error < 2e-3
    assert result.object_position_error is not None and result.object_position_error < 2e-3


def test_dt_limits_each_joint_increment_by_velocity():
    robot = default_robot()
    ik = WholeBodyIK(robot)
    q0 = robot.nominal.copy()
    goal = q0.copy()
    goal[robot.index["waist_yaw_joint"]] = 0.25
    goal[robot.index["left_shoulder_yaw_joint"]] = -0.20
    goal[robot.index["right_shoulder_yaw_joint"]] = 0.15
    dt = 0.002
    result = ik.solve_independent(q0, robot.fk(goal, "left"), robot.fk(goal, "right"), dt=dt)
    assert np.all(np.abs(result.q - q0) <= robot.velocity_limits * dt + 1e-8)


def test_optional_obstacle_and_balance_constraints():
    robot = default_robot()
    ik = WholeBodyIK(robot)
    q = robot.nominal.copy()
    obstacle = SphereObstacle(center=(4.0, 4.0, 4.0), radius=0.1)
    support = BalanceRegion(((-2.0, -2.0), (2.0, -2.0), (2.0, 2.0), (-2.0, 2.0)))
    result = ik.solve_independent(q, robot.fk(q, "left"), robot.fk(q, "right"),
                                  obstacles=[obstacle], balance_region=support)
    assert result.success, result.message


def test_benchmark_pose_sets_are_multi_case_and_within_joint_limits():
    from bimanual_wbik.benchmark import (
        _q_from_profile,
        carry_pose_profiles,
        independent_pose_profiles,
    )

    robot = default_robot()
    independent = independent_pose_profiles()
    carry = carry_pose_profiles()
    assert len(independent) >= 8
    assert len(carry) >= 6
    for profile in (*independent.values(), *carry.values()):
        q = _q_from_profile(robot, profile)
        assert np.all(q >= robot.lower)
        assert np.all(q <= robot.upper)


def test_benchmark_smoke_records_samples_and_writes_artifacts(tmp_path):
    import csv
    import json

    from bimanual_wbik.benchmark import run_benchmark, write_benchmark

    data = run_benchmark(trajectory_steps=2, independent_cases=["neutral"],
                         carry_cases=["home"], include_trajectories=False)
    assert data["summary"]["all"]["samples"] == 2
    assert data["summary"]["all"]["success_rate"] == 1.0
    assert all(len(row["q"]) == 16 for row in data["samples"])
    paths = write_benchmark(data, tmp_path)
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths)
    assert len(json.loads(paths[0].read_text(encoding="utf-8"))["samples"]) == 2
    with paths[1].open(newline="", encoding="utf-8") as stream:
        assert len(list(csv.DictReader(stream))) == 2
    assert "未通过样本" in paths[2].read_text(encoding="utf-8")
