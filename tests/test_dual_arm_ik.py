#!/usr/bin/env python3
"""
================================================================================
双臂 16-DoF 协同逆运动学求解器单元测试
================================================================================
覆盖 core/solver/dual_arm_ik.py 的核心行为:
    * 关节映射与配置结构 (q 顺序 / 限位 / 步长上限)
    * forward_kinematics / jacobian / pose_error 基础几何原语
    * solve 的收敛性: 以自身 FK 构造的自洽目标必然可解 (往返一致性)
    * 输入校验: 错误维度的 seed 必须被拒绝而不是崩溃
    * 多起点重启机制的有效性
    * seed 数组越界时应被夹紧到合法限位内

说明: 该求解器使用 2 自由度腰部 (waist_yaw, waist_pitch) + 双臂各 7 关节,
      q 共 16 维。waist_roll 等其余关节固定于 neutral。
================================================================================
"""

import os
import sys

import numpy as np
import pinocchio as pin

DIR_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if DIR_ROOT not in sys.path:
    sys.path.insert(0, DIR_ROOT)

from core.solver.dual_arm_ik import (
    DualArmIKSolver,
    IKConfig,
    IKFailureReason,
)


def _make_solver():
    return DualArmIKSolver()


def test_joint_layout_and_limits():
    """关节顺序、维度与限位应当自洽。"""
    print("-" * 70)
    print("▶ 1. 关节映射与限位结构")
    s = _make_solver()

    assert s.nq == 16, f"q 应为 16 维 (2腰+7左臂+7右臂), 实际 {s.nq}"
    assert s.joint_names[:2] == list(s.cfg.waist_joints), "前两维应为腰部关节"
    assert "left_shoulder_pitch_joint" in s.joint_names, "应包含左肩关节"
    assert "right_wrist_yaw_joint" in s.joint_names, "应包含右腕关节"

    # 限位必须是"下界 < 上界"且长度匹配
    assert len(s.q_min) == s.nq and len(s.q_max) == s.nq
    assert np.all(s.q_min < s.q_max), "存在下限 >= 上限的关节"
    # 步长上限维度匹配, 且为正值
    assert len(s.dq_max) == s.nq and np.all(s.dq_max > 0)
    # 任务权重矩阵应为方阵
    assert s.W.shape == (12, 12), f"W 应为 12x12, 实际 {s.W.shape}"
    print(f"  [OK] 16 维关节映射正确, 限位/步长/权重维度自洽")


def test_forward_kinematics_shape():
    """FK 应返回左右末端各一个 SE3 位姿。"""
    print("-" * 70)
    print("▶ 2. 正向运动学")
    s = _make_solver()
    q = np.zeros(s.nq)
    poses = s.forward_kinematics(q)

    assert isinstance(poses, list) and len(poses) == 2, "应返回左右两个末端位姿"
    for p in poses:
        assert isinstance(p, pin.SE3)
    # 零位时两个末端应有不同的 Y 向坐标 (左右镜像)
    assert abs(poses[0].translation[1] - poses[1].translation[1]) > 0.05, \
        "零位下左右末端 Y 应左右对称分离"
    print(f"  L.y={poses[0].translation[1]:+.4f}  R.y={poses[1].translation[1]:+.4f} (左右分离正常)")


def test_jacobian_shape():
    """雅可比应为 12x16 (双臂各 6D 位姿约束)。"""
    print("-" * 70)
    print("▶ 3. 雅可比矩阵")
    s = _make_solver()
    J = s.jacobian(np.zeros(s.nq))
    assert J.shape == (12, s.nq), f"J 应为 12x{s.nq}, 实际 {J.shape}"
    print(f"  [OK] 雅可比维度 {J.shape} 正确")


def test_pose_error():
    """pose_error 对相同位姿应给出零误差, 对纯平移应给出正确的线性误差。"""
    print("-" * 70)
    print("▶ 4. 位姿误差度量")
    s = _make_solver()
    T = pin.SE3(np.eye(3), np.array([0.1, 0.2, 0.3]))
    e = s.pose_error(T, pin.SE3(np.eye(3), np.array([0.1, 0.2, 0.3])))
    assert np.allclose(e, 0.0, atol=1e-12), "相同位姿误差应为 0"

    # 沿 X 轴平移 0.05m -> 位置误差模长应为 0.05
    e2 = s.pose_error(T, pin.SE3(np.eye(3), np.array([0.15, 0.2, 0.3])))
    assert abs(np.linalg.norm(e2[:3]) - 0.05) < 1e-9, \
        f"平移 0.05m 的位置误差应为 0.05, 实际 {np.linalg.norm(e2[:3]):.6f}"
    print("  [OK] 零误差与纯平移误差度量正确")


def test_solve_roundtrip():
    """往返一致性: 用自身 FK 生成的目标位姿, 求解器应能还原。"""
    print("-" * 70)
    print("▶ 5. 求解往返一致性 (最核心的正确性)")
    s = _make_solver()
    rng = np.random.default_rng(7)

    q_true = np.clip(
        s.q_min + rng.random(s.nq) * (s.q_max - s.q_min) * 0.6,
        s.q_min, s.q_max,
    )
    poses = s.forward_kinematics(q_true)

    res = s.solve(q_true.copy(), poses[0], poses[1])
    assert res.success, f"自洽目标应可解, 实际失败: {res.failure_reason}"
    assert res.position_error_left < 1e-3, f"左臂位置残差过大: {res.position_error_left}"
    assert res.position_error_right < 1e-3, f"右臂位置残差过大: {res.position_error_right}"
    assert res.orientation_error_left < 1e-2, f"左臂姿态残差过大: {res.orientation_error_left}"
    assert res.orientation_error_right < 1e-2, f"右臂姿态残差过大: {res.orientation_error_right}"
    print(f"  [OK] 往返求解成功: 位置残差 L/R = "
          f"{res.position_error_left:.2e}/{res.position_error_right:.2e} m")


def test_solve_respects_limits():
    """求解得到的关节值必须落在配置限位内。"""
    print("-" * 70)
    print("▶ 6. 求解结果限位合规")
    s = _make_solver()
    rng = np.random.default_rng(11)
    for _ in range(3):
        q_true = np.clip(s.q_min + rng.random(s.nq) * (s.q_max - s.q_min) * 0.6,
                         s.q_min, s.q_max)
        poses = s.forward_kinematics(q_true)
        res = s.solve(q_true.copy(), poses[0], poses[1])
        assert np.all(res.solution >= s.q_min - 1e-6), "解越过下限"
        assert np.all(res.solution <= s.q_max + 1e-6), "解越过上限"
    print("  [OK] 多次求解结果均在限位内")


def test_invalid_seed_rejected():
    """维度错误的 seed 应被明确拒绝, 返回 INVALID_INPUT 而非抛异常。"""
    print("-" * 70)
    print("▶ 7. 非法输入处理")
    s = _make_solver()
    q0 = np.zeros(s.nq)
    poses = s.forward_kinematics(q0)

    for bad in [np.zeros(3), np.zeros(15), np.zeros(17), np.zeros((s.nq, 2))]:
        res = s.solve(bad, poses[0], poses[1])
        assert not res.success, f"非法 seed {bad.shape} 不应报告成功"
        assert res.failure_reason == IKFailureReason.INVALID_INPUT, \
            f"非法 seed 应返回 INVALID_INPUT, 实际 {res.failure_reason}"
    print(f"  [OK] 4 种非法维度均被正确拒绝 (INVALID_INPUT)")


def test_seed_clamped_to_limits():
    """越界的 seed 应被夹紧到合法限位内, 而不是直接崩溃或产出越界解。"""
    print("-" * 70)
    print("▶ 8. seed 越界夹紧")
    s = _make_solver()
    q0 = np.zeros(s.nq)
    poses = s.forward_kinematics(q0)

    huge = np.full(s.nq, 99.0)          # 远超所有限位
    res = s.solve(huge, poses[0], poses[1])
    assert not np.any(np.isnan(res.solution)), "夹紧后不应产生 NaN"
    assert np.all(res.solution <= s.q_max + 1e-6), "夹紧后的解不应越上限"
    print("  [OK] 越界 seed 被安全夹紧, 未产生 NaN 或越界解")


def test_unreachable_target_reported():
    """远超工作空间的目标应被诚实报告为失败, 而非伪造一个假解。"""
    print("-" * 70)
    print("▶ 9. 不可达目标的诚实失败")
    s = _make_solver()
    q0 = np.zeros(s.nq)
    # 构造一个伸手绝对够不到的目标 (3 米外)
    far = pin.SE3(np.eye(3), np.array([3.0, 0.0, 0.0]))
    near = s.forward_kinematics(q0)[0]

    res = s.solve(q0.copy(), far, near)
    assert not res.success, "不可达目标不应报告成功"
    assert res.failure_reason is not None, "失败时应给出失败原因"
    print(f"  [OK] 不可达目标诚实失败 (reason={res.failure_reason})")


def test_restarts_effect():
    """同一目标下, 更多重启次数不应得到更差的最终误差。"""
    print("-" * 70)
    print("▶ 10. 多起点重启机制有效性")
    rng = np.random.default_rng(5)

    s_few = DualArmIKSolver(config=IKConfig(max_restarts=0))
    s_many = DualArmIKSolver(config=IKConfig(max_restarts=20))
    q_true = np.clip(s_few.q_min + rng.random(16) * (s_few.q_max - s_few.q_min) * 0.6,
                     s_few.q_min, s_few.q_max)
    poses = s_few.forward_kinematics(q_true)

    # 用一个较差的初始点起解, 让重启机制有机会发挥作用
    bad_seed = np.clip(q_true + 0.25, s_few.q_min, s_few.q_max)
    r_few = s_few.solve(bad_seed.copy(), poses[0], poses[1])
    r_many = s_many.solve(bad_seed.copy(), poses[0], poses[1])

    assert r_many.total_error <= r_few.total_error + 1e-9, \
        f"更多重启应不劣: few={r_few.total_error:.6e} many={r_many.total_error:.6e}"
    print(f"  [OK] 重启 0 次 err={r_few.total_error:.3e} | 重启 20 次 err={r_many.total_error:.3e}")


if __name__ == "__main__":
    print("=" * 70)
    print("[START] 双臂 16-DoF 协同逆解算器单元测试")
    print("=" * 70)
    test_joint_layout_and_limits()
    test_forward_kinematics_shape()
    test_jacobian_shape()
    test_pose_error()
    test_solve_roundtrip()
    test_solve_respects_limits()
    test_invalid_seed_rejected()
    test_seed_clamped_to_limits()
    test_unreachable_target_reported()
    test_restarts_effect()
    print("=" * 70)
    print("[SUCCESS] 双臂求解器全部测试通过")
    print("=" * 70)