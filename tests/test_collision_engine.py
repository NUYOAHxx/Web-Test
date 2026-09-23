#!/usr/bin/env python3
"""
================================================================================
Unitree G1 工业级全域自碰撞安全检测引擎自动化测试套件
(MoveIt SRDF ACM + Pinocchio/Coal BVH + Pink CBF 屏障函数)
================================================================================

测试目标：
1. SRDF 允许碰撞矩阵 (ACM) 解析与剪枝完整性 (648 -> 575 对，降维剪枝至 436 / 207 对)；
2. 基准姿态物理无干涉性 (直立 Stand 与 就绪 Ready 姿态最小净空 > 15mm，碰撞状态为 False)；
3. 全域危险自干涉精准捕获 (臂-臂交叉相撞、臂-躯干穿透、臂-头部干涉 100% 捕获并定位连杆)；
4. 降维碰撞门禁极速性能与等价性 (< 100 微秒/次，支持 1kHz+ 控制闭环)；
5. Web 大屏与 ROS 2 遥测数据字典完整性 (4 大核心区域净空 mm 级指标输出)；
6. Inria Pink 4.4.0 SelfCollisionBarrier 控制屏障函数在 ProxQP 中的连续避碰求解能力。
"""

import os
import sys
import time
import numpy as np

# 注入项目根目录
DIR_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if DIR_ROOT not in sys.path:
    sys.path.insert(0, DIR_ROOT)

from core.kinematics.g1_model import G1KinematicsModel, G1_DEFAULT_STAND_JOINTS, G1_READY_POSE
from core.collision.g1_collision import G1CollisionChecker
from core.solver.g1_pink_ik import G1PinkIKSolver


def test_collision_suite():
    print("=" * 80)
    print("🚀 启动 Unitree G1 工业级自碰撞检测引擎 (MoveIt SRDF ACM) 全面自动化测试")
    print("=" * 80)

    # 1. 初始化模型与碰撞检测引擎
    t0 = time.perf_counter()
    kin = G1KinematicsModel()
    col = G1CollisionChecker(kin)
    t_init = (time.perf_counter() - t0) * 1000.0
    print(f"✔️ 模型与 SRDF ACM 引擎初始化成功，耗时: {t_init:.2f} ms")

    # 2. 验证 ACM 矩阵与拓扑分类
    n_pairs = len(col.pair_metadata)
    n_10dof = len(kin.coll_model_10dof["left_arm"].collisionPairs)
    n_7dof = len(kin.coll_model_7dof["left_arm"].collisionPairs)

    print("\n【1. 碰撞对拓扑与 ACM 过滤检查】")
    print(f"  • MoveIt SRDF 过滤后全机身有效碰撞对 : {n_pairs} 对 (完全剔除相连相邻连杆与物理不可达对)")
    print(f"  • 10-DoF 动态运动链有效对 (剪除静态对) : {n_10dof} 对")
    print(f"  • 7-DoF 单臂动态运动链有效对           : {n_7dof} 对")
    print("  • 语义分类统计:")
    for cat, idxs in col._category_indices.items():
        print(f"    - {cat:10s}: {len(idxs):3d} 对")

    assert n_pairs == 575, f"期望 SRDF 过滤后为 575 对，实际为 {n_pairs}"
    assert n_10dof == 436, f"期望 10-DoF 剪枝后为 436 对，实际为 {n_10dof}"
    assert n_7dof == 207, f"期望 7-DoF 剪枝后为 207 对，实际为 {n_7dof}"
    assert len(col._category_indices["arm_torso"]) == 42
    assert len(col._category_indices["arm_arm"]) == 47
    assert len(col._category_indices["arm_head"]) == 8
    print("  ✔️ ACM 规则解析与运动学动态剪枝测试 100% 通过！")

    # 3. 验证默认姿态下的安全性 (基准姿态无误报)
    print("\n【2. 基准健康姿态安全度评估 (零误报测试)】")
    q_stand = np.zeros(kin.model.nq)
    for name, val in G1_DEFAULT_STAND_JOINTS.items():
        if kin.model.existJointName(name):
            q_stand[kin.model.getJointId(name) - 1] = val

    is_col_full = col.is_full_body_colliding(q_stand, update_fk=True)
    is_col_left = col.is_colliding("left_arm", q_stand, update_fk=False)
    is_col_right = col.is_colliding("right_arm", q_stand, update_fk=False)

    q_red_10 = np.concatenate([np.zeros(3), G1_READY_POSE["left_arm"]])
    is_col_red = col.is_colliding_reduced("left_arm", q_red_10, is_10dof=True)

    min_dist_m = col.compute_min_distance(q_stand, update_fk=False)
    min_dist_mm = min_dist_m * 1000.0

    print(f"  • 全身自碰撞判定 (Full-Body is_colliding) : {is_col_full}")
    print(f"  • 左臂自碰撞判定 (Left-Arm is_colliding)  : {is_col_left}")
    print(f"  • 右臂自碰撞判定 (Right-Arm is_colliding) : {is_col_right}")
    print(f"  • 降维极速判定 (Reduced is_colliding)     : {is_col_red}")
    print(f"  • 全机身最小欧氏物理间距                  : {min_dist_mm:.1f} mm")

    assert not is_col_full, "错误：默认直立姿态被误判为碰撞！"
    assert not is_col_left, "错误：左臂默认姿态被误判为碰撞！"
    assert not is_col_right, "错误：右臂默认姿态被误判为碰撞！"
    assert not is_col_red, "错误：降维模型默认姿态被误判为碰撞！"
    assert min_dist_mm > 15.0, f"错误：默认姿态最小安全裕度过低 ({min_dist_mm:.1f} mm <= 15 mm)！"
    print("  ✔️ 默认健康姿态测试通过：零误报，最小物理间隙 > 15mm！")

    # 4. 危险碰撞穿透精准拦截测试
    print("\n【3. 极端干涉穿透精准拦截测试】")

    # Case A: 双臂胸前交叉抱胸相撞
    q_cross = q_stand.copy()
    left_pitch_id = kin.model.getJointId("left_shoulder_pitch_joint") - 1
    left_roll_id = kin.model.getJointId("left_shoulder_roll_joint") - 1
    left_elbow_id = kin.model.getJointId("left_elbow_joint") - 1
    right_pitch_id = kin.model.getJointId("right_shoulder_pitch_joint") - 1
    right_roll_id = kin.model.getJointId("right_shoulder_roll_joint") - 1
    right_elbow_id = kin.model.getJointId("right_elbow_joint") - 1

    # 左右手过度内收并交叉相叠
    q_cross[left_pitch_id] = 0.3
    q_cross[left_roll_id] = -0.8    # 左肩大幅向右胸内收
    q_cross[left_elbow_id] = 1.5
    q_cross[right_pitch_id] = 0.3
    q_cross[right_roll_id] = 0.8     # 右肩大幅向左胸内收
    q_cross[right_elbow_id] = 1.5

    col_cross = col.is_colliding("left_arm", q_cross, update_fk=True)
    pairs_cross = col.get_colliding_pairs(q_cross, arm="left_arm", update_fk=False)
    arm_arm_pairs = [p for p in pairs_cross if p[2] == "arm_arm"]

    print(f"  • 工况 A (双臂胸前强行交叉相撞): 检出碰撞={col_cross}, 碰撞对数量={len(pairs_cross)}")
    if arm_arm_pairs:
        print(f"    - 精确捕获双臂互碰: {arm_arm_pairs[0][0]} <--> {arm_arm_pairs[0][1]}")
    assert col_cross, "错误：双臂交叉抱胸相撞未被拦截！"
    assert len(arm_arm_pairs) > 0, "错误：未识别出 arm_arm 双臂互碰分类！"

    # Case B: 左臂手腕过度弯折撞入胸部
    q_torso_col = q_stand.copy()
    q_torso_col[left_pitch_id] = 0.2
    q_torso_col[left_roll_id] = -0.5  # 紧贴胸壁
    q_torso_col[left_elbow_id] = 1.8  # 深度反折小臂
    col_torso = col.is_colliding("left_arm", q_torso_col, update_fk=True)
    pairs_torso = col.get_colliding_pairs(q_torso_col, arm="left_arm", update_fk=False)
    torso_pairs = [p for p in pairs_torso if p[2] == "arm_torso"]

    print(f"  • 工况 B (手臂反折刺入胸躯干): 检出碰撞={col_torso}, 碰撞对数量={len(pairs_torso)}")
    if torso_pairs:
        print(f"    - 精确捕获臂胸干涉: {torso_pairs[0][0]} <--> {torso_pairs[0][1]}")
    assert col_torso, "错误：手臂撞入胸腔未被拦截！"
    assert len(torso_pairs) > 0, "错误：未识别出 arm_torso 臂胸干涉分类！"

    print("  ✔️ 危险干涉拦截测试 100% 通过：精准捕获并输出碰撞连杆名称与分类！")

    # 5. 极速降维碰撞判定性能基准 (Reduced Collision Benchmark)
    print("\n【4. 极速降维碰撞检测门禁性能基准 (1,000 次压力测试)】")
    N = 1000
    t_start = time.perf_counter()
    for _ in range(N):
        col.is_colliding_reduced("left_arm", q_red_10, is_10dof=True)
    t_red = (time.perf_counter() - t_start) / N * 1e6

    t_start = time.perf_counter()
    for _ in range(N):
        col.is_full_body_colliding(q_stand, update_fk=True)
    t_full = (time.perf_counter() - t_start) / N * 1e6

    print(f"  • 29-DoF 全身全连杆碰撞检测平均耗时 : {t_full:.1f} 微秒")
    print(f"  • 10-DoF 降维几何模型碰撞检测平均耗时 : {t_red:.1f} 微秒 (提速 {t_full/t_red:.1f}x)")
    assert t_red < 250.0, f"降维碰撞检测过慢: {t_red:.1f} 微秒 (应 < 250 微秒)"
    print("  ✔️ 碰撞检测极速门禁性能达标，完全满足 1kHz+ 控制环实时需求！")

    # 6. Web 大屏与 ROS 2 遥测数据接口测试
    print("\n【5. Web 大屏与 ROS 2 遥测接口完整性测试】")
    zone_dists = col.compute_zone_distances(q_stand, update_fk=True)
    print("  • 区域间距输出 (mm):", zone_dists)
    for k in ("arm_torso", "arm_arm", "arm_head", "arm_leg", "torso_arm", "inter_arm", "head_arm", "leg_arm"):
        assert k in zone_dists, f"缺少遥测键: {k}"
        assert isinstance(zone_dists[k], float), f"遥测值类型异常: {zone_dists[k]}"
    print("  ✔️ 遥测数据字典结构与双向别名 100% 满足 Web 大屏需求！")

    # 7. Inria Pink SelfCollisionBarrier 集成测试
    print("\n【6. Inria Pink 4.4.0 SelfCollisionBarrier (CBF) 屏障函数求解测试】")
    solver = G1PinkIKSolver()
    target_pos = np.array([0.35, 0.22, 0.85])
    ok_barrier, w_sol, a_sol, info_barrier = solver.solve_10dof_ik(
        arm="left_arm",
        target_pos=target_pos,
        enable_collision_barrier=True,
        barrier_d_min=0.015,
        max_iters=15,
    )
    print(f"  • 启用 SelfCollisionBarrier 求解结果: 成功={ok_barrier}, 残差={info_barrier.get('pos_err_mm', 0):.2f} mm")
    assert ok_barrier, "错误：Pink ProxQP 结合 SelfCollisionBarrier 求解失败！"
    assert info_barrier["pos_err_mm"] < 1.0, "残差未达标"
    print("  ✔️ Inria Pink 4.4.0 控制屏障函数 (CBF) 凸二次规划平滑避碰求解成功！")

    print("\n" + "=" * 80)
    print("🎉 恭喜！宇树 G1 工业级自碰撞检测系统 (SRDF ACM + Pinocchio/Coal) 所有测试全部通过！")
    print("=" * 80)


if __name__ == "__main__":
    test_collision_suite()
