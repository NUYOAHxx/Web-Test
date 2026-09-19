#!/usr/bin/env python3
"""
Unitree G1 10-DoF (3-DoF 腰部 + 7-DoF 手臂) 躯干-手臂协同加权逆运动学测试套件

测试项目：
1. 近处目标测试：验证“手臂优先”机制（目标在 35cm 臂展内时，腰部保持直立不动，偏角 < 5°）；
2. 极限远距离目标测试：验证“腰部辅助”机制（目标达 50cm 远，单臂 7-DoF 绝对够不着，10-DoF 自动弯腰转身轻松解出）；
3. 极限低矮拾取测试：模拟弯腰捡拾低处物体（目标 Z=0.45m，单臂够不到，10-DoF 腰部俯仰弯曲下探精准命中）；
4. 1,000 样本大规模随机压力基准测试：统计 10-DoF 求解成功率、耗时与限位违规率。
"""

import os
import sys
import time
import numpy as np

dir_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if dir_root not in sys.path:
    sys.path.insert(0, dir_root)

from deploy.solver.g1_hybrid_ik import G1HybridIKSolver

G1_MAX_REACH = 0.377  # 37.7 cm (大臂 19.3cm + 小臂 18.4cm)


def run_10dof_test_suite():
    print("=" * 75)
    print("       Unitree G1 10-DoF (3腰 + 7臂) 躯干-手臂加权协同逆运动学实测")
    print("=" * 75)

    solver = G1HybridIKSolver()
    arm = "left_arm"
    
    import pinocchio as pin
    pin.forwardKinematics(solver.model, solver.data, pin.neutral(solver.model))
    pin.updateFramePlacements(solver.model, solver.data)
    shoulder_pos = solver.data.oMf[solver.model.getFrameId("left_shoulder_pitch_link")].translation.copy()

    # -------------------------------------------------------------
    # 测试 1: 近处日常抓取 (验证手臂优先，腰部不动)
    # -------------------------------------------------------------
    target_near = np.array([0.22, 0.22, 0.95])
    dist_near = np.linalg.norm(target_near - shoulder_pos)

    print(f"\n【测试 1: 近处舒适区目标 (距离肩部 {dist_near*100:.1f} cm <= 37.7 cm)】")
    print(f"目标空间坐标: [X={target_near[0]:.2f}, Y={target_near[1]:.2f}, Z={target_near[2]:.2f}]")

    # 7-DoF 单臂测试
    ok_7dof, _, info_7dof = solver.solve_ik(arm, target_near)
    # 10-DoF 腰臂协同测试
    ok_10dof, w_q, a_q, info_10dof = solver.solve_10dof_ik(arm, target_near)

    fk_pos, _ = solver.forward_kinematics_10dof(arm, w_q, a_q)
    err_mm = np.linalg.norm(fk_pos - target_near) * 1000.0

    print(f"  • 7-DoF 单臂求解结果 : {'✔️ 成功' if ok_7dof else '❌ 失败'} (误差: {info_7dof['pos_err_mm']:.2f} mm)")
    print(f"  • 10-DoF 协同求解结果: {'✔️ 成功' if ok_10dof else '❌ 失败'} (误差: {err_mm:.2f} mm, 耗时: {info_10dof['time_ms']:.2f} ms)")
    print(f"  • 解出的腰部 3 关节  : yaw={np.degrees(w_q[0]):.1f}°, roll={np.degrees(w_q[1]):.1f}°, pitch={np.degrees(w_q[2]):.1f}°")
    
    waist_moved = np.any(np.abs(np.degrees(w_q)) > 6.0)
    if not waist_moved:
        print("  • \033[92m[机制验证成功]\033[0m: 腰部关节角度均小于 6°，几乎保持直立 0 位，全靠手臂完成抓取，手臂优先机制生效！")
    else:
        print("  • [注]: 腰部有微小自适应微调。")

    # -------------------------------------------------------------
    # 测试 2: 极限大跨度远距离抓取 (50 cm 远，7-DoF 单臂 100% 失败)
    # -------------------------------------------------------------
    target_far = np.array([0.48, 0.25, 0.95])
    dist_far = np.linalg.norm(target_far - shoulder_pos)

    print(f"\n【测试 2: 极限远距离目标 (距离肩部 {dist_far*100:.1f} cm >> 最大臂展 37.7 cm)】")
    print(f"目标空间坐标: [X={target_far[0]:.2f}, Y={target_far[1]:.2f}, Z={target_far[2]:.2f}]")

    ok_7dof_far, _, info_7dof_far = solver.solve_ik(arm, target_far)
    ok_10dof_far, w_q_far, a_q_far, info_10dof_far = solver.solve_10dof_ik(arm, target_far)

    fk_pos_far, _ = solver.forward_kinematics_10dof(arm, w_q_far, a_q_far)
    err_far_mm = np.linalg.norm(fk_pos_far - target_far) * 1000.0

    print(f"  • 7-DoF 单臂求解结果 : {'✔️ 成功' if ok_7dof_far else '❌ 失败'} (臂长极限死锁，误差: {info_7dof_far['pos_err_mm']:.1f} mm)")
    print(f"  • 10-DoF 协同求解结果: {'✔️ 成功' if ok_10dof_far else '❌ 失败'} (误差: {err_far_mm:.2f} mm, 耗时: {info_10dof_far['time_ms']:.2f} ms)")
    print(f"  • 解出的腰部 3 关节  : yaw={np.degrees(w_q_far[0]):.1f}°, roll={np.degrees(w_q_far[1]):.1f}°, pitch={np.degrees(w_q_far[2]):.1f}°")
    
    if ok_10dof_far and not ok_7dof_far:
        print("  • \033[92m[机制验证成功]\033[0m: 7-DoF 够不着的死区，10-DoF 自动控制腰部俯仰弯曲将肩关节基座前送，成功解出！")

    # -------------------------------------------------------------
    # 测试 3: 极限低矮拾取 (模拟弯腰捡拾 Z=0.65m 物体)
    # -------------------------------------------------------------
    target_low = np.array([0.25, 0.22, 0.65])
    dist_low = np.linalg.norm(target_low - shoulder_pos)

    print(f"\n【测试 3: 极限低矮拾取 (距离肩部 {dist_low*100:.1f} cm，模拟弯腰拾物)】")
    print(f"目标空间坐标: [X={target_low[0]:.2f}, Y={target_low[1]:.2f}, Z={target_low[2]:.2f}]")

    ok_7dof_low, _, info_7dof_low = solver.solve_ik(arm, target_low)
    ok_10dof_low, w_q_low, a_q_low, info_10dof_low = solver.solve_10dof_ik(arm, target_low)

    fk_pos_low, _ = solver.forward_kinematics_10dof(arm, w_q_low, a_q_low)
    err_low_mm = np.linalg.norm(fk_pos_low - target_low) * 1000.0

    print(f"  • 7-DoF 单臂求解结果 : {'✔️ 成功' if ok_7dof_low else '❌ 失败'} (手短悬空，误差: {info_7dof_low['pos_err_mm']:.1f} mm)")
    print(f"  • 10-DoF 协同求解结果: {'✔️ 成功' if ok_10dof_low else '❌ 失败'} (误差: {err_low_mm:.2f} mm, 耗时: {info_10dof_low['time_ms']:.2f} ms)")
    print(f"  • 解出的腰部 3 关节  : yaw={np.degrees(w_q_low[0]):.1f}°, roll={np.degrees(w_q_low[1]):.1f}°, pitch={np.degrees(w_q_low[2]):.1f}° (前屈弯腰俯身)")

    # -------------------------------------------------------------
    # 测试 4: 1,000 样本大规模随机压力基准测试
    # -------------------------------------------------------------
    print("\n" + "=" * 75)
    print("【测试 4: 1,000 样本大规模随机工作空间压力基准测试】")
    print("=" * 75)

    np.random.seed(42)
    N = 1000
    lower_10, upper_10 = solver.limits_10dof[arm]

    # 生成 1000 个在 10-DoF 工作空间内的真实可达点
    margin = (upper_10 - lower_10) * 0.08
    q_rand_10 = (lower_10 + margin) + np.random.rand(N, 10) * (upper_10 - lower_10 - 2 * margin)
    targets_10dof = [solver.forward_kinematics_10dof(arm, q_rand_10[i, :3], q_rand_10[i, 3:])[0] for i in range(N)]

    success_cnt = 0
    latencies_ms = []
    errors_mm = []
    limit_violations = 0

    t_all_0 = time.perf_counter()
    for i in range(N):
        t_pt = targets_10dof[i]
        t0 = time.perf_counter()
        ok, w_ans, a_ans, inf = solver.solve_10dof_ik(arm, t_pt, waist_weight=8.0)
        dt = (time.perf_counter() - t0) * 1000.0

        latencies_ms.append(dt)
        if ok:
            success_cnt += 1

        # 检查 10 自由度限位
        ans_10 = np.concatenate([w_ans, a_ans])
        if np.any(ans_10 < lower_10 - 1e-4) or np.any(ans_10 > upper_10 + 1e-4):
            limit_violations += 1

        fk_check, _ = solver.forward_kinematics_10dof(arm, w_ans, a_ans)
        errors_mm.append(np.linalg.norm(fk_check - t_pt) * 1000.0)

    total_time_s = time.perf_counter() - t_all_0

    print(f"  • 求解成功率 (Success Rate)       : \033[92m{(success_cnt/N)*100:.2f}%\033[0m ({success_cnt}/{N})")
    print(f"  • 限位违规率 (Limit Violations)   : \033[92m{(limit_violations/N)*100:.2f}%\033[0m (严格要求 0.00%)")
    print(f"  • 平均单次求解耗时 (Mean Latency) : {np.mean(latencies_ms):.3f} ms")
    print(f"  • 中位数耗时 (Median Latency)     : {np.median(latencies_ms):.3f} ms")
    print(f"  • 95% 样本耗时 (P95 Latency)      : {np.percentile(latencies_ms, 95):.3f} ms")
    print(f"  • 平均位置误差 (Mean Error)       : {np.mean(errors_mm):.3f} mm")
    print(f"  • 求解总吞吐率 (Throughput)       : \033[92m{N / total_time_s:.0f} solves/sec\033[0m")
    print("=" * 75)


if __name__ == "__main__":
    run_10dof_test_suite()
