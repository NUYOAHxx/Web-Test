#!/usr/bin/env python3
"""
Unitree G1 10-DoF (3-DoF 腰部 + 7-DoF 手臂) 躯干-手臂协同加权逆运动学自动化测试套件
支持 pytest tests/ 自动发现与直接命令行执行。
"""

import os
import sys
import numpy as np
import pytest
import pinocchio as pin

dir_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if dir_root not in sys.path:
    sys.path.insert(0, dir_root)

from deploy.solver.g1_hybrid_ik import G1HybridIKSolver


@pytest.fixture(scope="module")
def solver():
    return G1HybridIKSolver()


def test_10dof_near_target_arm_priority(solver):
    """测试 1: 近处舒适区目标 (验证手臂优先，腰部微动/保持直立)"""
    arm = "left_arm"
    target_near = np.array([0.22, 0.22, 0.95])

    ok, w_q, a_q, info = solver.solve_10dof_ik(arm, target_near, waist_weight=8.0)
    assert ok, f"10-DoF 求解失败: err={info.get('pos_err_mm')}mm"

    fk_pos, _ = solver.forward_kinematics_10dof(arm, w_q, a_q)
    err_mm = np.linalg.norm(fk_pos - target_near) * 1000.0
    assert err_mm < 2.0, f"近处目标跟踪误差过大: {err_mm:.2f} mm"

    # 手臂优先机制：腰部各关节偏差应当很小 (< 8°)
    max_waist_deg = float(np.max(np.abs(np.degrees(w_q))))
    assert max_waist_deg < 8.0, f"手臂优先机制失效，腰部偏角过大: {max_waist_deg:.1f}°"


def test_10dof_far_target_extended_reach(solver):
    """测试 2: 极限大跨度远距离抓取 (50 cm 远，7-DoF 单臂够不着，10-DoF 辅助解出)"""
    arm = "left_arm"
    target_far = np.array([0.48, 0.25, 0.95])

    # 7-DoF 单臂受限于最大臂展 (37.7cm) 必然不可达
    ok_7dof, _, info_7dof = solver.solve_ik(arm, target_far)
    assert not ok_7dof or info_7dof["pos_err_mm"] > 50.0, "单臂不应能触及远超臂展的目标"

    # 10-DoF 腰臂协同应能成功收敛
    ok_10dof, w_q, a_q, info_10dof = solver.solve_10dof_ik(arm, target_far, waist_weight=8.0)
    assert ok_10dof, f"10-DoF 远距离协同求解失败: {info_10dof}"

    fk_pos, _ = solver.forward_kinematics_10dof(arm, w_q, a_q)
    err_mm = np.linalg.norm(fk_pos - target_far) * 1000.0
    assert err_mm < 2.5, f"10-DoF 远距离目标残差过大: {err_mm:.2f} mm"


def test_10dof_low_target_bending_pickup(solver):
    """测试 3: 极限低矮拾取 (模拟俯身捡拾 Z=0.65m 物体)"""
    arm = "left_arm"
    target_low = np.array([0.25, 0.22, 0.65])

    ok, w_q, a_q, info = solver.solve_10dof_ik(arm, target_low, waist_weight=8.0)
    assert ok, f"10-DoF 俯身拾取求解失败: {info}"

    fk_pos, _ = solver.forward_kinematics_10dof(arm, w_q, a_q)
    err_mm = np.linalg.norm(fk_pos - target_low) * 1000.0
    assert err_mm < 2.5, f"俯身拾取残差过大: {err_mm:.2f} mm"

    # 腰部俯仰角应明显弯曲以协助下探 (pitch > 10°)
    waist_pitch_deg = float(np.degrees(w_q[2]))
    assert waist_pitch_deg > 10.0, f"俯身拾取未检测到腰部俯仰前屈: {waist_pitch_deg:.1f}°"


def test_10dof_limits_compliance(solver):
    """测试 4: 随机采样 10-DoF 求解完全遵守关节限位与安全边界"""
    arm = "left_arm"
    lower_10, upper_10 = solver.limits_10dof[arm]
    np.random.seed(42)

    for _ in range(15):
        # 随机采样可达配置正解作为目标
        q_rand = lower_10 + np.random.rand(10) * (upper_10 - lower_10)
        tgt_p, _ = solver.forward_kinematics_10dof(arm, q_rand[:3], q_rand[3:])

        ok, w_ans, a_ans, _ = solver.solve_10dof_ik(arm, tgt_p, waist_weight=8.0)
        if ok:
            ans_10 = np.concatenate([w_ans, a_ans])
            assert np.all(ans_10 >= lower_10 - 1e-4), "解违背了 10-DoF 关节下限"
            assert np.all(ans_10 <= upper_10 + 1e-4), "解违背了 10-DoF 关节上限"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
