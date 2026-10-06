#!/usr/bin/env python3
"""
通用人形机器人 2-DoF 腰部 + 7-DoF 手臂 (9-DoF) Pink IK 优化求解器自动化测试套件
支持 pytest tests/ 自动发现与独立命令行执行。
"""

import numpy as np
import pinocchio as pin
import pytest

from core.solver.humanoid_pink_ik import (
    HumanoidPinkIKSolver,
    HumanoidKinematicsAdapter,
    IKSolveStatus,
)


@pytest.fixture(scope="module")
def solver():
    return HumanoidPinkIKSolver()


def test_adapter_initialization():
    """测试通用运动学适配器加载与 2-DoF 腰部拓扑特性。"""
    adapter = HumanoidKinematicsAdapter()
    assert adapter.waist_dim == 2
    assert "left_arm" in adapter.chain_coord_joint_names
    assert "right_arm" in adapter.chain_coord_joint_names
    assert len(adapter.chain_coord_joint_names["left_arm"]) == 9


def test_invalid_input_handling(solver):
    """测试目标坐标包含非法浮点数 (NaN / Inf) 时的健壮性拦截。"""
    nan_target = [np.nan, 0.2, 0.8]
    ok, q, res = solver.solve_ik("left_arm", nan_target)
    assert not ok
    assert res.status == IKSolveStatus.INVALID_INPUT

    ok_c, w_c, a_c, res_c = solver.solve_coordinated_ik("left_arm", nan_target)
    assert not ok_c
    assert res_c.status == IKSolveStatus.INVALID_INPUT


def test_coordinated_ik_near_reach(solver):
    """测试 9-DoF 协同求解 (水平取物任务)。"""
    target = np.array([0.35, 0.22, 0.85])
    ok, w_q, a_q, res = solver.solve_coordinated_ik("left_arm", target)
    assert ok, f"协同求解失败: {res}"
    assert res.status == IKSolveStatus.CONVERGED
    assert res.pos_err_mm < 2.0
    assert len(w_q) == 2
    assert len(a_q) == 7
    assert res.within_joint_limits


def test_coordinated_ik_6dof_full_pose(solver):
    """测试 9-DoF 全位姿 (位置 + 姿态) 逆运动学求解。"""
    target_pos = np.array([0.35, 0.22, 0.85])
    rpy_deg = np.array([0.0, 15.0, 0.0])
    rpy_rad = np.radians(rpy_deg)
    target_rot = pin.rpy.rpyToMatrix(float(rpy_rad[0]), float(rpy_rad[1]), float(rpy_rad[2]))

    ok, w_q, a_q, res = solver.solve_coordinated_ik(
        "left_arm",
        target_pos,
        target_rot=target_rot,
        pos_tol=1e-3,
        rot_tol=0.03,
    )
    assert ok, f"全位姿协同求解失败: {res}"
    assert res.pos_err_mm < 2.0
    assert res.rot_err_deg < 2.0
    assert res.within_joint_limits


def test_right_arm_mirror_reach(solver):
    """测试右臂对称求解。"""
    target = np.array([0.35, -0.22, 0.85])
    ok, w_q, a_q, res = solver.solve_coordinated_ik("right_arm", target)
    assert ok, f"右臂协同求解失败: {res}"
    assert res.status == IKSolveStatus.CONVERGED
    assert res.pos_err_mm < 2.0
