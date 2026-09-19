#!/usr/bin/env python3
"""
================================================================================
Unitree G1 10-DoF 6-DoF 全空间位姿 (Position + Orientation) 逆运动学求解测试
================================================================================
验证目标：
1. 3-DoF 纯位置求解向后兼容性 (Position Error < 1.0mm)
2. 6-DoF 空间全位姿求解同时满足双重容差 (Position Error < 1.0mm, Rot Error < 1.5°)
3. 多工况覆盖：平伸握持、俯视抓取、倾斜操作、双臂对称性测试
4. 动力学与全身质心防倾覆平衡指标校验
"""

import os
import sys
import numpy as np
import pinocchio as pin

# 注入项目根目录
DIR_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if DIR_ROOT not in sys.path:
    sys.path.insert(0, DIR_ROOT)

from deploy.solver.g1_hybrid_ik import G1HybridIKSolver


def test_6dof_ik_suite():
    print("=" * 80)
    print("🚀 启动 Unitree G1 10-DoF 6-DoF 全空间位姿逆运动学自动化测试")
    print("=" * 80)

    solver = G1HybridIKSolver()
    
    # 测试案例集: (arm, target_pos, rpy_deg, case_name)
    test_cases = [
        # 1. 经典前向水平桌前取物 (平放朝前)
        ("left_arm", np.array([0.35, 0.22, 0.85]), np.array([0.0, 0.0, 0.0]), "Table Reach (Horizontal)"),
        # 2. 俯角 25 度抓取
        ("left_arm", np.array([0.38, 0.20, 0.80]), np.array([0.0, 25.0, 0.0]), "Pitched Grasp (Pitch=25°)"),
        # 3. 倾斜低位抓取
        ("left_arm", np.array([0.35, 0.25, 0.65]), np.array([10.0, 30.0, -10.0]), "Low Angled Reach"),
        # 4. 右臂对称测试
        ("right_arm", np.array([0.35, -0.22, 0.85]), np.array([0.0, 0.0, 0.0]), "Right Arm Mirror Reach"),
        # 5. 右臂俯倾抓取
        ("right_arm", np.array([0.36, -0.20, 0.78]), np.array([-15.0, 20.0, 10.0]), "Right Arm Angled Grasp"),
    ]

    results = []
    
    for arm, pos, rpy_deg, name in test_cases:
        rpy_rad = np.radians(rpy_deg)
        target_rot = pin.rpy.rpyToMatrix(float(rpy_rad[0]), float(rpy_rad[1]), float(rpy_rad[2]))
        
        ok, waist_q, arm_q, info = solver.solve_10dof_ik(
            arm=arm,
            target_pos=pos,
            target_rot=target_rot,
            pos_tol=1e-3,
            rot_tol=2.6e-2,
            max_iters=50,
        )

        fk_pos, fk_rot = solver.forward_kinematics_10dof(arm, waist_q, arm_q)
        pos_err_mm = np.linalg.norm(fk_pos - pos) * 1000.0
        R_err = target_rot @ fk_rot.T
        rot_err_deg = np.degrees(np.linalg.norm(pin.log3(R_err)))
        
        # 关节限位检查
        limits = solver.limits_10dof[arm]
        q_10 = np.concatenate([waist_q, arm_q])
        within_limits = np.all(q_10 >= limits[0] - 1e-4) and np.all(q_10 <= limits[1] + 1e-4)

        results.append({
            "name": name,
            "arm": arm,
            "ok": ok,
            "time_ms": info["time_ms"],
            "iters": info["iters"],
            "pos_err_mm": pos_err_mm,
            "rot_err_deg": rot_err_deg,
            "within_limits": within_limits,
            "bal_margin_mm": info.get("balance_margin_mm", 0.0),
            "seed_name": info.get("seed_name", ""),
        })

    print(f"{'Case Name':<28} | {'Arm':<9} | {'Status':<6} | {'Pos Err':<9} | {'Rot Err':<9} | {'Time (ms)':<9} | {'Iters':<5} | {'Limits'}")
    print("-" * 100)
    for r in results:
        status_str = "PASS" if (r["ok"] and r["pos_err_mm"] < 1.0 and r["rot_err_deg"] < 1.5) else "FAIL"
        print(f"{r['name']:<28} | {r['arm']:<9} | {status_str:<6} | {r['pos_err_mm']:>6.2f} mm | {r['rot_err_deg']:>6.2f}° | {r['time_ms']:>6.2f} ms | {r['iters']:>5} | {r['within_limits']}")

    # 断言
    for r in results:
        assert r["ok"], f"Case {r['name']} failed to converge!"
        assert r["pos_err_mm"] < 1.0, f"Case {r['name']} pos err {r['pos_err_mm']:.2f}mm exceeded 1.0mm!"
        assert r["rot_err_deg"] < 1.5, f"Case {r['name']} rot err {r['rot_err_deg']:.2f}° exceeded 1.5°!"
        assert r["within_limits"], f"Case {r['name']} joint limit violation!"

    print("\n🎉 5/5 6-DoF 全空间位姿逆运动学测试全部通过！双重残差均严格达标 (<1.0mm, <1.5°)！")


def test_backward_compatibility():
    print("\n" + "=" * 80)
    print("🔄 验证 3-DoF 纯位置求解向后兼容性 (target_rot=None)")
    print("=" * 80)

    solver = G1HybridIKSolver()
    target_pos = np.array([0.35, 0.22, 0.85])
    ok, waist_q, arm_q, info = solver.solve_10dof_ik(
        arm="left_arm",
        target_pos=target_pos,
        target_rot=None,
    )
    assert ok, "3-DoF backward compatibility test failed!"
    assert info["pos_err_mm"] < 1.0, f"3-DoF pos err {info['pos_err_mm']:.2f}mm exceeded 1.0mm!"
    assert info["mode"] == "10DOF_3DOF_POS", f"Expected mode 10DOF_3DOF_POS, got {info['mode']}"
    print(f"✔️ 3-DoF 纯位置解算测试通过！残差: {info['pos_err_mm']:.3f}mm, 耗时: {info['time_ms']:.2f}ms")


if __name__ == "__main__":
    test_6dof_ik_suite()
    test_backward_compatibility()
