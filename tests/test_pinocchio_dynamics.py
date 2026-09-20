#!/usr/bin/env python3
"""
================================================================================
Pinocchio 刚体动力学与全身体态平衡引擎自动化单元测试与性能验证
================================================================================
"""

import sys
import os
import time
import numpy as np

# 加入项目根目录
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.kinematics.g1_model import G1KinematicsModel, G1_READY_POSE
from core.solver.g1_hybrid_ik import G1HybridIKSolver


def test_pinocchio_model_reduction():
    """测试 1. 10-DoF 轻量裁剪子模型构建与正解/雅可比等价性"""
    print("----------------------------------------------------------------------")
    print("▶ 1. 测试 10-DoF 裁剪子模型 (pin.buildReducedModel)...")
    kin = G1KinematicsModel()

    assert "left_arm" in kin.model_10dof
    assert "right_arm" in kin.model_10dof
    assert kin.model_10dof["left_arm"].nq == 10
    assert kin.model_10dof["left_arm"].nv == 10

    q_10 = np.array([0.1, -0.05, 0.15, 0.2, 0.2, 0.0, 0.5, 0.0, 0.0, 0.0], dtype=np.float64)

    # 裁剪模型 FK
    pos_red, rot_red = kin.forward_kinematics_reduced("left_arm", q_10)

    # 全模型 FK
    pos_full, rot_full = kin.forward_kinematics_10dof("left_arm", q_10[:3], q_10[3:])

    err_pos = np.linalg.norm(pos_red - pos_full)
    err_rot = np.linalg.norm(rot_red - rot_full)

    print(f"  FK 空间位置欧氏差: {err_pos:.2e} m")
    print(f"  FK 姿态旋转阵残差: {err_rot:.2e}")
    assert err_pos < 1e-6, f"裁剪模型与全模型 FK 不一致: {err_pos}"
    assert err_rot < 1e-6, f"裁剪模型与全模型旋转不一致: {err_rot}"

    # 雅可比一致性
    J_red = kin.compute_jacobian_reduced("left_arm", q_10, has_rot=True)
    q_full = kin.build_q_10dof("left_arm", q_10[:3], q_10[3:])
    J_full_sub = kin.compute_subspace_jacobian(q_full, "left_arm", is_10dof=True, has_rot=True)
    err_J = np.linalg.norm(J_red - J_full_sub)
    print(f"  Jacobian 矩阵残差: {err_J:.2e}")
    assert err_J < 1e-6, f"裁剪模型雅可比不一致: {err_J}"
    print("  ✔️ 10-DoF 裁剪子模型验证完全通过！")


def test_pinocchio_com_and_balance():
    """测试 2. 全身质心 CoM 与双足多边形平衡安全评估"""
    print("----------------------------------------------------------------------")
    print("▶ 2. 测试全身质心 CoM 与双足多边形静平衡安全评估...")
    kin = G1KinematicsModel()

    # 标称直立就绪姿态
    q_stand = kin.build_q_10dof("left_arm", np.zeros(3), G1_READY_POSE["left_arm"])
    bal = kin.evaluate_balance(q_stand)

    print(f"  直立姿态质心: {bal['com_pos']} m")
    print(f"  平衡裕度: {bal['margin_mm']} mm, 状态: {bal['status']}")
    assert bal["status"] == "STABLE", f"标称姿态应为 STABLE，实测: {bal['status']}"
    assert bal["margin_mm"] > 50.0, f"标称姿态裕度应大于 50mm，实测: {bal['margin_mm']}"

    # 极端前倾俯仰测试 (弯腰 0.5 rad ≈ 28.6度)
    q_lean = kin.build_q_10dof("left_arm", np.array([0.0, 0.0, 0.5]), G1_READY_POSE["left_arm"])
    bal_lean = kin.evaluate_balance(q_lean)
    print(f"  过度前倾姿态质心: {bal_lean['com_pos']} m")
    print(f"  前倾平衡裕度: {bal_lean['margin_mm']} mm, 状态: {bal_lean['status']}")
    assert bal_lean["com_pos"][0] > bal["com_pos"][0], "弯腰前倾时质心 X 必须前移"
    print("  ✔️ 全身质心与体态平衡评估验证通过！")


def test_pinocchio_gravity_and_dynamics():
    """测试 3. 广义重力补偿力矩与动力学"""
    print("----------------------------------------------------------------------")
    print("▶ 3. 测试广义重力补偿力矩 (pin.computeGeneralizedGravity)...")
    kin = G1KinematicsModel()

    q_stand = kin.build_q_10dof("left_arm", np.zeros(3), G1_READY_POSE["left_arm"])
    tau_g = kin.compute_gravity_torques(q_stand, arm="left_arm", is_10dof=True)

    print(f"  直立姿态 10 轴重力力矩 (N·m): {np.round(tau_g, 2)}")
    assert len(tau_g) == 10
    # 腰部俯仰电机抵抗上半身重量，力矩应显著非零
    assert abs(tau_g[2]) > 0.5, f"腰部俯仰重力力矩过小: {tau_g[2]}"
    print("  ✔️ 广义重力补偿力矩测试通过！")


def test_pinocchio_manipulability():
    """测试 4. Yoshikawa 可操作度与奇异点分析"""
    print("----------------------------------------------------------------------")
    print("▶ 4. 测试 Yoshikawa 可操作度度量与奇异点检测...")
    kin = G1KinematicsModel()

    q_10 = np.concatenate([np.zeros(3), G1_READY_POSE["left_arm"]])
    manip = kin.compute_manipulability("left_arm", q_10, has_rot=True)
    print(f"  可操作度指标 w: {manip['yoshikawa']}, 最小奇异值: {manip['min_singular_value']}")
    assert manip["yoshikawa"] > 0.01, f"就绪姿态可操作度异常: {manip['yoshikawa']}"
    assert not manip["is_singular"], "就绪姿态不应为奇异点"
    print("  ✔️ 可操作度与奇异点分析测试通过！")


def test_hybrid_ik_with_com_balance():
    """测试 5. 混合 IK 求解器集成零空间 CoM 平衡约束"""
    print("----------------------------------------------------------------------")
    print("▶ 5. 测试 10-DoF 混合 IK 求解 (集成零空间 CoM 平衡约束)...")
    solver = G1HybridIKSolver()

    targets = [
        np.array([0.35, 0.22, 0.85]),
        np.array([0.45, 0.15, 0.75]),
        np.array([0.25, 0.35, 0.80]),
    ]

    for idx, tgt in enumerate(targets):
        t0 = time.perf_counter()
        ok, waist_q, arm_q, info = solver.solve_10dof_ik("left_arm", tgt)
        dt_ms = (time.perf_counter() - t0) * 1000.0

        print(f"  目标 #{idx+1} {tgt}: 成功={ok}, 耗时={dt_ms:.2f}ms, 残差={info.get('pos_err_mm', 0):.2f}mm, 平衡={info.get('balance_status')} (裕度: {info.get('balance_margin_mm')}mm)")
        assert ok, f"目标 #{idx+1} 求解失败"
        assert info.get("pos_err_mm", 999.0) < 2.0, f"目标 #{idx+1} 残差过大: {info.get('pos_err_mm')}"
        assert "com_pos" in info, "返回字典缺少 com_pos"
        assert "gravity_torques" in info, "返回字典缺少 gravity_torques"
        assert "manipulability" in info, "返回字典缺少 manipulability"

    print("  ✔️ 混合 IK 求解与 CoM 平衡零空间约束验证完全通过！")


if __name__ == "__main__":
    test_pinocchio_model_reduction()
    test_pinocchio_com_and_balance()
    test_pinocchio_gravity_and_dynamics()
    test_pinocchio_manipulability()
    test_hybrid_ik_with_com_balance()
    print("======================================================================")
    print("🎉 所有 Pinocchio 动力学、平衡安全与求解器集成测试 100% 成功！")
    print("======================================================================")
