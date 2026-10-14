"""Reproducible multi-pose and trajectory benchmarks for the reference IK model.

This is an offline kinematics benchmark, not a real-time or hardware safety test.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from . import GraspPair, WholeBodyIK, default_robot
from .math3d import inverse, pose_error, transform


def _q_from_profile(robot, profile: dict[str, float]) -> np.ndarray:
    q = robot.nominal.copy()
    for name, value in profile.items():
        q[robot.index[name]] = value
    if np.any(q < robot.lower) or np.any(q > robot.upper):
        raise ValueError(f"profile violates joint limits: {profile}")
    return q


def independent_pose_profiles() -> dict[str, dict[str, float]]:
    """Moderate, repeatable reachable poses spanning the G1 waist and both arms."""
    return {
        "neutral": {},
        "waist_yaw_left": {"waist_yaw_joint": 0.18, "left_shoulder_yaw_joint": -0.10,
                           "right_shoulder_yaw_joint": 0.08},
        "waist_yaw_right": {"waist_yaw_joint": -0.18, "left_shoulder_yaw_joint": -0.08,
                            "right_shoulder_yaw_joint": 0.10},
        "waist_pitch_forward": {"waist_pitch_joint": 0.10,
                                "left_shoulder_pitch_joint": 0.18,
                                "right_shoulder_pitch_joint": 0.16},
        "shoulder_roll_asymmetric": {"left_shoulder_roll_joint": 0.12,
                                     "right_shoulder_roll_joint": -0.12},
        "wide_elbows": {"left_shoulder_yaw_joint": -0.20, "right_shoulder_yaw_joint": 0.20,
                        "left_elbow_joint": 1.10, "right_elbow_joint": 1.10},
        "wrist_orientation": {"left_wrist_pitch_joint": 0.15, "left_wrist_roll_joint": -0.12,
                              "right_wrist_pitch_joint": -0.12, "right_wrist_roll_joint": 0.12},
        "asymmetric_reach": {"waist_yaw_joint": 0.10, "waist_pitch_joint": -0.06,
                             "left_shoulder_pitch_joint": 0.10, "left_elbow_joint": 0.72,
                             "right_shoulder_pitch_joint": 0.20, "right_elbow_joint": 1.05},
        "combined_moderate": {"waist_yaw_joint": -0.12, "waist_pitch_joint": -0.05,
                              "left_shoulder_yaw_joint": -0.18, "left_shoulder_pitch_joint": 0.16,
                              "left_elbow_joint": 1.12, "right_shoulder_yaw_joint": 0.16,
                              "right_shoulder_pitch_joint": 0.19, "right_elbow_joint": 1.10},
        "larger_compound_pose": {"waist_yaw_joint": 0.30, "waist_pitch_joint": 0.12,
                                 "left_shoulder_yaw_joint": -0.35,
                                 "left_shoulder_pitch_joint": 0.05,
                                 "left_shoulder_roll_joint": 0.38, "left_elbow_joint": 1.35,
                                 "right_shoulder_yaw_joint": 0.30,
                                 "right_shoulder_pitch_joint": 0.12,
                                 "right_shoulder_roll_joint": -0.38, "right_elbow_joint": 1.30,
                                 "left_wrist_pitch_joint": 0.12,
                                 "right_wrist_pitch_joint": -0.10},
    }


def carry_pose_profiles() -> dict[str, dict[str, float]]:
    """Rigidly transform the G1 hands while preserving the captured grasp pair."""
    return {
        "home": {},
        "yaw_left": {"waist_yaw_joint": 0.16},
        "yaw_right": {"waist_yaw_joint": -0.16},
        "pitch_forward": {"waist_pitch_joint": 0.10},
        "pitch_backward": {"waist_pitch_joint": -0.10},
        "symmetric_reach": {"left_shoulder_pitch_joint": 0.16,
                             "right_shoulder_pitch_joint": 0.16},
        "elbow_bend": {"left_elbow_joint": 1.12, "right_elbow_joint": 1.12},
        "compound_waist": {"waist_yaw_joint": 0.14, "waist_pitch_joint": 0.08},
    }


def _pose_list(pose: np.ndarray) -> list[list[float]]:
    return np.asarray(pose, dtype=float).round(12).tolist()


def _pose_errors(actual: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    error = pose_error(actual, target)
    return float(np.linalg.norm(error[:3])), float(np.linalg.norm(error[3:]))


def _solve_record(robot, ik, *, sample_id: str, mode: str, suite: str,
                  case_id: str, step: int, progress: float | None,
                  q_start: np.ndarray, left_target: np.ndarray,
                  right_target: np.ndarray, object_target: np.ndarray | None = None,
                  grasps: GraspPair | None = None) -> tuple[dict[str, Any], np.ndarray]:
    start = time.perf_counter()
    result = None
    error_message = None
    try:
        if mode == "independent":
            result = ik.solve_independent(q_start, left_target, right_target)
        else:
            assert object_target is not None and grasps is not None
            result = ik.solve_object_carry(q_start, object_target, grasps)
        q = np.asarray(result.q, dtype=float).copy()
    except Exception as exc:  # retain failures in the benchmark output
        q = np.asarray(q_start, dtype=float).copy()
        error_message = f"{type(exc).__name__}: {exc}"
    elapsed = time.perf_counter() - start

    actual_left, actual_right = robot.fk(q, "left"), robot.fk(q, "right")
    left_pe, left_re = _pose_errors(actual_left, left_target)
    right_pe, right_re = _pose_errors(actual_right, right_target)
    closure_error = result.closure_error if result is not None else None
    object_pe = result.object_position_error if result is not None else None
    object_re = result.object_orientation_error if result is not None else None
    margin = float(np.min(np.minimum(q - robot.lower, robot.upper - q)))
    row = {
        "sample_id": sample_id,
        "mode": mode,
        "suite": suite,
        "case_id": case_id,
        "step": int(step),
        "progress": None if progress is None else float(progress),
        "success": bool(result.success) if result is not None else False,
        "message": str(result.message) if result is not None else error_message,
        "solve_time_s": float(elapsed),
        "iterations": int(result.iterations) if result is not None else 0,
        "left_position_error_m": left_pe,
        "left_orientation_error_rad": left_re,
        "right_position_error_m": right_pe,
        "right_orientation_error_rad": right_re,
        "closure_error": closure_error,
        "object_position_error_m": object_pe,
        "object_orientation_error_rad": object_re,
        "min_joint_limit_margin_native": margin,
        "q": q.tolist(),
        "target_left_pose": _pose_list(left_target),
        "target_right_pose": _pose_list(right_target),
        "achieved_left_pose": _pose_list(actual_left),
        "achieved_right_pose": _pose_list(actual_right),
        "target_object_pose": None if object_target is None else _pose_list(object_target),
    }
    return row, q


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"samples": 0, "successes": 0, "success_rate": 0.0}
    times = np.asarray([r["solve_time_s"] for r in rows], dtype=float)
    successes = [r for r in rows if r["success"]]
    summary: dict[str, Any] = {
        "samples": len(rows),
        "successes": len(successes),
        "failures": len(rows) - len(successes),
        "success_rate": len(successes) / len(rows),
        "solve_time_mean_s": float(np.mean(times)),
        "solve_time_median_s": float(np.median(times)),
        "solve_time_p95_s": float(np.percentile(times, 95)),
        "solve_time_max_s": float(np.max(times)),
        "iterations_mean": float(np.mean([r["iterations"] for r in rows])),
    }
    for key in ("left_position_error_m", "left_orientation_error_rad",
                "right_position_error_m", "right_orientation_error_rad",
                "closure_error", "object_position_error_m",
                "object_orientation_error_rad"):
        values = [r[key] for r in rows if r[key] is not None]
        if values:
            summary[f"{key}_mean"] = float(np.mean(values))
            summary[f"{key}_max"] = float(np.max(values))
    margins = [r["min_joint_limit_margin_native"] for r in rows]
    summary["min_joint_limit_margin_native"] = float(np.min(margins))
    return summary


def _trajectory_metrics(robot, rows: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: row["step"])
    qs = np.asarray([row["q"] for row in ordered], dtype=float)
    output = _summarize(ordered)
    output["all_frames_success"] = all(row["success"] for row in ordered)
    output["left_target_path_length_m"] = float(sum(
        np.linalg.norm(np.asarray(b["target_left_pose"])[:3, 3] -
                       np.asarray(a["target_left_pose"])[:3, 3])
        for a, b in zip(ordered, ordered[1:])))
    output["right_target_path_length_m"] = float(sum(
        np.linalg.norm(np.asarray(b["target_right_pose"])[:3, 3] -
                       np.asarray(a["target_right_pose"])[:3, 3])
        for a, b in zip(ordered, ordered[1:])))
    achieved_left = [robot.fk(q, "left") for q in qs]
    achieved_right = [robot.fk(q, "right") for q in qs]
    output["left_achieved_path_length_m"] = float(sum(
        np.linalg.norm(b[:3, 3] - a[:3, 3])
        for a, b in zip(achieved_left, achieved_left[1:])))
    output["right_achieved_path_length_m"] = float(sum(
        np.linalg.norm(b[:3, 3] - a[:3, 3])
        for a, b in zip(achieved_right, achieved_right[1:])))
    if len(qs) > 1:
        deltas = np.diff(qs, axis=0)
        output["joint_travel_by_joint_native"] = {
            name: float(np.sum(np.abs(deltas[:, index])))
            for index, name in enumerate(robot.names)
        }
        output["max_joint_step_by_joint_native"] = {
            name: float(np.max(np.abs(deltas[:, index])))
            for index, name in enumerate(robot.names)
        }
        output["discrete_joint_acceleration_rms_by_joint_native"] = {
            name: float(np.sqrt(np.mean(np.diff(qs[:, index], n=2) ** 2)))
            if len(qs) > 2 else 0.0
            for index, name in enumerate(robot.names)
        }
    else:
        output["joint_travel_by_joint_native"] = {name: 0.0 for name in robot.names}
        output["max_joint_step_by_joint_native"] = {name: 0.0 for name in robot.names}
        output["discrete_joint_acceleration_rms_by_joint_native"] = {
            name: 0.0 for name in robot.names
        }
    return output


def run_benchmark(*, trajectory_steps: int = 9,
                  independent_cases: list[str] | None = None,
                  carry_cases: list[str] | None = None,
                  include_trajectories: bool = True) -> dict[str, Any]:
    """Run all static target and continuous trajectory tests; returns JSON-safe data."""
    if trajectory_steps < 2:
        raise ValueError("trajectory_steps must be at least 2")
    robot = default_robot()
    ik = WholeBodyIK(robot)
    q_home = robot.nominal.copy()
    samples: list[dict[str, Any]] = []
    independent_profiles = independent_pose_profiles()
    carry_profiles = carry_pose_profiles()

    selected_independent = independent_cases or list(independent_profiles)
    selected_carry = carry_cases or list(carry_profiles)
    for case_id in selected_independent:
        if case_id not in independent_profiles:
            raise ValueError(f"unknown independent pose case: {case_id}")
        q_goal = _q_from_profile(robot, independent_profiles[case_id])
        left_target, right_target = robot.fk(q_goal, "left"), robot.fk(q_goal, "right")
        row, _ = _solve_record(robot, ik, sample_id=f"independent/static/{case_id}",
                               mode="independent", suite="static_pose", case_id=case_id,
                               step=0, progress=None, q_start=q_home,
                               left_target=left_target, right_target=right_target)
        samples.append(row)

    object_home = GraspPair.infer_object_pose(robot, q_home)
    grasps = GraspPair.from_current(robot, q_home, object_home)
    for case_id in selected_carry:
        if case_id not in carry_profiles:
            raise ValueError(f"unknown carry pose case: {case_id}")
        q_goal = _q_from_profile(robot, carry_profiles[case_id])
        left_goal, _ = robot.fk(q_goal, "left"), robot.fk(q_goal, "right")
        object_target = left_goal @ inverse(grasps.left)
        left_target, right_target = object_target @ grasps.left, object_target @ grasps.right
        row, _ = _solve_record(robot, ik, sample_id=f"carry/static/{case_id}",
                               mode="carry", suite="static_pose", case_id=case_id,
                               step=0, progress=None, q_start=q_home,
                               left_target=left_target, right_target=right_target,
                               object_target=object_target, grasps=grasps)
        samples.append(row)

    if include_trajectories:
        progress_values = np.linspace(0.0, 1.0, trajectory_steps)
        independent_paths = {
            "lateral_sweep": {"waist_yaw_joint": 0.20, "left_shoulder_yaw_joint": -0.18,
                              "right_shoulder_yaw_joint": 0.18, "left_elbow_joint": 1.08,
                              "right_elbow_joint": 1.08},
            "waist_wrist_sweep": {"waist_yaw_joint": -0.12, "waist_pitch_joint": 0.10,
                                  "left_shoulder_pitch_joint": 0.14,
                                  "right_shoulder_pitch_joint": 0.18,
                                  "left_wrist_roll_joint": 0.16,
                                  "right_wrist_roll_joint": -0.16},
        }
        for path_id, profile in independent_paths.items():
            q_goal = _q_from_profile(robot, profile)
            q_current = q_home.copy()
            for step, alpha in enumerate(progress_values):
                q_reference = (1.0 - alpha) * q_home + alpha * q_goal
                left_target = robot.fk(q_reference, "left")
                right_target = robot.fk(q_reference, "right")
                row, q_current = _solve_record(
                    robot, ik, sample_id=f"independent/trajectory/{path_id}/{step:03d}",
                    mode="independent", suite="trajectory", case_id=path_id,
                    step=step, progress=float(alpha), q_start=q_current,
                    left_target=left_target, right_target=right_target)
                samples.append(row)

        carry_paths = {
            "object_waist_yaw": {"waist_yaw_joint": 0.20},
            "object_waist_pitch": {"waist_pitch_joint": -0.12},
        }
        for path_id, profile in carry_paths.items():
            q_goal = _q_from_profile(robot, profile)
            q_current = q_home.copy()
            for step, alpha in enumerate(progress_values):
                q_reference = (1.0 - alpha) * q_home + alpha * q_goal
                left_reference = robot.fk(q_reference, "left")
                object_target = left_reference @ inverse(grasps.left)
                left_target, right_target = object_target @ grasps.left, object_target @ grasps.right
                row, q_current = _solve_record(
                    robot, ik, sample_id=f"carry/trajectory/{path_id}/{step:03d}",
                    mode="carry", suite="trajectory", case_id=path_id,
                    step=step, progress=float(alpha), q_start=q_current,
                    left_target=left_target, right_target=right_target,
                    object_target=object_target, grasps=grasps)
                samples.append(row)

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    trajectory_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in samples:
        groups["all"].append(row)
        groups[f"mode:{row['mode']}"] .append(row)
        groups[f"suite:{row['suite']}"] .append(row)
        if row["suite"] == "trajectory":
            trajectory_groups[f"{row['mode']}/{row['case_id']}"].append(row)
    summary = {key: _summarize(rows) for key, rows in groups.items()}
    trajectories = {key: _trajectory_metrics(robot, rows)
                    for key, rows in trajectory_groups.items()}
    return {
        "metadata": {
            "benchmark": "bimanual whole-body IK multi-pose and trajectory benchmark",
            "model": robot.robot_name,
            "dof": robot.dof,
            "joint_names": list(robot.names),
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "trajectory_steps": trajectory_steps if include_trajectories else 0,
            "notes": [
                "目标位姿由参考模型中的可达关节构型生成。",
                "搬运场景对 G1 关节链生成可达目标，并保持标定的双手刚性抓取关系。",
                "轨迹按前一帧返回的关节解 warm-start。",
                "求解耗时为 IK 调用周围的墙钟时间，不含写报告时间。",
                "离散加速度是按采样点索引计算的平滑度代理，不是物理加速度。",
                "这是离线运动学基准，不证明实时性或实机安全性。",
            ],
        },
        "summary": summary,
        "trajectory_metrics": trajectories,
        "samples": samples,
    }


def write_benchmark(data: dict[str, Any], output_dir: str | Path) -> tuple[Path, Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "ik_benchmark.json"
    csv_path = output_dir / "ik_benchmark_samples.csv"
    report_path = output_dir / "ik_benchmark_report.md"
    json_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    scalar_fields = [
        "sample_id", "mode", "suite", "case_id", "step", "progress", "success", "message",
        "solve_time_s", "iterations", "left_position_error_m", "left_orientation_error_rad",
        "right_position_error_m", "right_orientation_error_rad", "closure_error",
        "object_position_error_m", "object_orientation_error_rad", "min_joint_limit_margin_native",
    ]
    pose_fields = ["target_left_pose", "target_right_pose", "achieved_left_pose",
                   "achieved_right_pose", "target_object_pose", "q"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=scalar_fields + pose_fields)
        writer.writeheader()
        for row in data["samples"]:
            csv_row = {key: row.get(key) for key in scalar_fields}
            csv_row.update({key: json.dumps(row.get(key), separators=(",", ":"), ensure_ascii=False)
                            for key in pose_fields})
            writer.writerow(csv_row)

    summary = data["summary"]
    lines = [
        "# 双臂全身 IK 多目标与轨迹基准报告", "",
        f"生成时间（UTC）：{data['metadata']['created_utc']}",
        f"参考模型：{data['metadata']['model']}（{data['metadata']['dof']} DoF）",
        f"样本数：{summary['all']['samples']}；成功：{summary['all']['successes']}；"
        f"失败：{summary['all']['failures']}；成功率：{summary['all']['success_rate']:.1%}", "",
        "## 按模式/场景汇总", "",
        "| 分组 | 样本数 | 成功率 | 平均耗时 (ms) | P95 耗时 (ms) | 平均迭代 | 最大左手位置误差 (mm) | 最大右手位置误差 (mm) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key in ("mode:independent", "mode:carry", "suite:static_pose", "suite:trajectory"):
        if key not in summary:
            continue
        item = summary[key]
        lines.append(
            f"| {key} | {item['samples']} | {item['success_rate']:.1%} | "
            f"{item['solve_time_mean_s'] * 1000:.3f} | {item['solve_time_p95_s'] * 1000:.3f} | "
            f"{item['iterations_mean']:.1f} | {item.get('left_position_error_m_max', float('nan')) * 1000:.4f} | "
            f"{item.get('right_position_error_m_max', float('nan')) * 1000:.4f} |"
        )
    lines.extend(["", "## 轨迹指标", ""])
    if data["trajectory_metrics"]:
        lines.extend([
            "| 轨迹 | 帧数 | 成功率 | 左/右目标路径长度 (m) | 平均耗时 (ms) | 单步最大关节变化（见 JSON/CSV） |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ])
        for key, item in data["trajectory_metrics"].items():
            lines.append(
                f"| {key} | {item['samples']} | {item['success_rate']:.1%} | "
                f"{item['left_target_path_length_m']:.4f} / {item['right_target_path_length_m']:.4f} | "
                f"{item['solve_time_mean_s'] * 1000:.3f} | 按关节列于 JSON |"
            )
    else:
        lines.append("本次未运行轨迹场景。")
    failures = [row for row in data["samples"] if not row["success"]]
    lines.extend(["", "## 未通过样本", ""])
    if failures:
        lines.extend([
            "未通过项仍保留在汇总成功率中。优化器状态为成功但误差超限的样本也会被视为失败；完整逐帧数据见 CSV/JSON。",
            "",
            "| 样本 | 优化器消息 | 耗时 (s) | 左/右手位置误差 (mm) | 左/右手姿态误差 (rad) | 闭链误差 |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ])
        for row in failures:
            message = str(row["message"]).replace("|", "\\|")
            lines.append(
                f"| {row['sample_id']} | {message} | {row['solve_time_s']:.3f} | "
                f"{row['left_position_error_m'] * 1000:.3f} / {row['right_position_error_m'] * 1000:.3f} | "
                f"{row['left_orientation_error_rad']:.5f} / {row['right_orientation_error_rad']:.5f} | "
                f"{row['closure_error'] if row['closure_error'] is not None else '—'} |"
            )
    else:
        lines.append("所有样本均通过。")
    lines.extend(["", "## 指标口径与边界", ""])
    lines.extend(f"- {note}" for note in data["metadata"]["notes"])
    lines.extend([
        "- 位置误差单位为米，姿态误差单位为弧度；CSV/JSON 保留每帧目标位姿、实际位姿和全身关节向量。",
        "- G1 16 个活动关节均为旋转关节，角度单位为弧度。",
        "- 本基准仅用于离线算法回归。参考模型结果不代表具体实机精度、控制周期性能、碰撞安全或动力学能力。",
    ])
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, csv_path, report_path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Multi-pose and trajectory IK benchmark")
    parser.add_argument("--output-dir", type=Path, default=Path("results/g1-ik-benchmark"))
    parser.add_argument("--trajectory-steps", type=int, default=9,
                        help="number of inclusive samples per trajectory (minimum 2)")
    parser.add_argument("--quick", action="store_true",
                        help="run 2 cases per mode with 3 trajectory samples")
    args = parser.parse_args(argv)
    if args.quick:
        data = run_benchmark(trajectory_steps=3,
                             independent_cases=list(independent_pose_profiles())[:2],
                             carry_cases=list(carry_pose_profiles())[:2])
    else:
        data = run_benchmark(trajectory_steps=args.trajectory_steps)
    paths = write_benchmark(data, args.output_dir)
    print(json.dumps({"summary": data["summary"],
                      "trajectory_metrics": data["trajectory_metrics"],
                      "files": [str(path) for path in paths]}, indent=2, ensure_ascii=False))
    return 0 if data["summary"]["all"]["success_rate"] == 1.0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
