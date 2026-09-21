#!/usr/bin/env python3
"""
================================================================================
Unitree G1 MoveIt 2 OMPL 运动规划演示与基准测试套件
(MoveIt 2 OMPL Planning Demo & Benchmark for Unitree G1)
================================================================================

测试与演示内容：
1. 7-DoF 单臂快速无障碍规划 (left_arm, RRTConnect)；
2. 10-DoF 躯干-手臂协同远距离规划 (left_arm_torso, RRTConnect)；
3. 动态注入环境障碍物，验证 OMPL 全局避障绕行规划能力；
4. 对比 RRTConnect 与 RRT* 的规划耗时与路标点质量。
"""

import os
import sys
import time
import argparse
import numpy as np

# 导入项目根目录
dir_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if dir_root not in sys.path:
    sys.path.insert(0, dir_root)

import rclpy
from core.planning.moveit_ompl_planner import MoveItOMPLPlanner, PlanResult


def run_ompl_demo():
    print("=" * 80)
    print(" 🚀 Unitree G1 MoveIt 2 OMPL 全局运动规划基准演示")
    print("=" * 80)

    # 1. 初始化 ROS 2 客户端
    if not rclpy.ok():
        rclpy.init()

    node = rclpy.create_node("g1_ompl_demo_runner")
    planner = MoveItOMPLPlanner(node=node, wait_for_services=True, timeout_sec=5.0)

    if not planner.plan_service_client.service_is_ready():
        print("\n❌ 错误: MoveIt 规划服务 (/plan_kinematic_path) 未就绪！")
        print("💡 请先在另一个终端窗口中启动 MoveIt 服务端：")
        print("   bash scripts/run_moveit_ompl.sh\n")
        return

    print("✔️ MoveIt 2 OMPL 规划服务端已就绪，开始测试流水线...\n")

    # 定义标准基准就绪姿态 (确保即使外部 RViz 交互改变了机器人姿态，基准压测亦能 100% 确定性复现)
    ready_joints = {
        "left_shoulder_pitch_joint": 0.2,
        "left_shoulder_roll_joint": 0.2,
        "left_shoulder_yaw_joint": 0.0,
        "left_elbow_joint": 0.5,
        "left_wrist_roll_joint": 0.0,
        "left_wrist_pitch_joint": 0.0,
        "left_wrist_yaw_joint": 0.0,
        "waist_yaw_joint": 0.0,
        "waist_roll_joint": 0.0,
        "waist_pitch_joint": 0.0,
    }

    # ──────────────────────────────────────────────────────────────────────────
    # 【测试 1】7-DoF 左臂自由空间点对点规划 (left_arm, RRTConnect)
    # ──────────────────────────────────────────────────────────────────────────
    print("-" * 80)
    print("▶ 【测试 1】7-DoF 左臂点对点快速规划 (left_arm | 算法: RRTConnect)")
    target_pos_1 = [0.25, 0.22, 0.85]
    print(f"  • 目标末端空间坐标: {target_pos_1}")

    t0 = time.perf_counter()
    res1 = planner.plan_to_pose(
        group_name="left_arm",
        target_pos=target_pos_1,
        planner_id="RRTConnectkConfigDefault",
        allowed_planning_time=3.0,
        start_joints=ready_joints,
    )
    dt1 = time.perf_counter() - t0

    if res1.success:
        print(f"  ✔️ {res1.summary()}")
        print(f"  • 第一路标点关节角: {np.round(res1.waypoints[0], 3)}")
        print(f"  • 终点路标点关节角: {np.round(res1.waypoints[-1], 3)}")
    else:
        print(f"  ❌ 规划失败: {res1.error_message}")

    # ──────────────────────────────────────────────────────────────────────────
    # 【测试 2】10-DoF 躯干-手臂协同大跨度规划 (left_arm_torso, RRTConnect)
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "-" * 80)
    print("▶ 【测试 2】10-DoF 躯干-手臂协同大跨度目标规划 (left_arm_torso | 算法: RRTConnect)")
    target_pos_2 = [0.42, 0.32, 0.95]
    print(f"  • 远距离目标坐标: {target_pos_2} (需要腰部 3-DoF 旋转倾斜介入协同)")

    t0 = time.perf_counter()
    res2 = planner.plan_to_pose(
        group_name="left_arm_torso",
        target_pos=target_pos_2,
        planner_id="RRTConnectkConfigDefault",
        allowed_planning_time=5.0,
        start_joints=ready_joints,
    )
    dt2 = time.perf_counter() - t0

    if res2.success:
        print(f"  ✔️ {res2.summary()}")
        print(f"  • 受控关节列表 (含腰部): {res2.joint_names}")
    else:
        print(f"  ❌ 规划失败: {res2.error_message}")

    # ──────────────────────────────────────────────────────────────────────────
    # 【测试 3】动态注入障碍物并执行避障绕行规划 (Obstacle Avoidance)
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "-" * 80)
    print("▶ 【测试 3】动态环境障碍物注入与 C-Space 避障规划")
    obs_name = "test_table_obstacle"
    # 将障碍物放置在直行路径中间 (比如 X=0.20, Y=0.20, Z=0.80)，尺寸 8cm x 8cm x 15cm
    obs_pos = [0.20, 0.20, 0.80]
    obs_size = [0.08, 0.08, 0.15]
    print(f"  • 在直行路径中插入长方体障碍物: {obs_name}")
    print(f"    - 中心坐标: {obs_pos} | 尺寸: {obs_size}")

    planner.add_box_obstacle(obs_name, position=obs_pos, size=obs_size)
    time.sleep(0.5)  # 等待 PlanningScene 更新同步

    # 目标末端点位设在障碍物外侧/上方，避开障碍物碰撞包络
    target_pos_3 = [0.32, 0.22, 0.88]
    print(f"  • 目标末端点位: {target_pos_3} (需要绕行避开障碍物)")

    # 3.1 7-DoF 单臂绕行避障 (RRTConnect)
    print("  [3.1] 7-DoF 单臂绕行避障 (left_arm | 算法: RRTConnect)...")
    res3_arm = planner.plan_to_pose(
        group_name="left_arm",
        target_pos=target_pos_3,
        planner_id="RRTConnectkConfigDefault",
        allowed_planning_time=5.0,
        start_joints=ready_joints,
    )
    print(f"    • {res3_arm.summary()}")

    # 3.2 10-DoF 躯干-手臂协同大冗余极速避障 (left_arm_torso | 算法: RRTConnect)
    print("  [3.2] 10-DoF 躯干-手臂协同大冗余极速避障 (left_arm_torso | 算法: RRTConnect)...")
    res3_torso = planner.plan_to_pose(
        group_name="left_arm_torso",
        target_pos=target_pos_3,
        planner_id="RRTConnectkConfigDefault",
        allowed_planning_time=5.0,
        start_joints=ready_joints,
    )
    print(f"    • {res3_torso.summary()}")

    # 清理障碍物
    print("  • 清理测试障碍物...")
    planner.remove_obstacle(obs_name)
    time.sleep(0.2)

    # ──────────────────────────────────────────────────────────────────────────
    # 【测试 4】RRT* 渐进最优全局规划 (自由空间收敛与平滑度评测)
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "-" * 80)
    print("▶ 【测试 4】RRT* 渐进最优算法全局规划 (left_arm | 算法: RRT* | 5.0s 充分优化)")
    target_pos_4 = [0.25, 0.22, 0.85]
    print(f"  • 目标末端空间坐标: {target_pos_4}")
    res4_star = planner.plan_to_pose(
        group_name="left_arm",
        target_pos=target_pos_4,
        planner_id="RRTstarkConfigDefault",
        allowed_planning_time=5.0,
        start_joints=ready_joints,
    )
    print(f"    • {res4_star.summary()}")

    # ──────────────────────────────────────────────────────────────────────────
    # 汇总输出性能对比表
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print(" 📊 MoveIt 2 OMPL 规划性能总结报表")
    print("=" * 80)
    print(f"{'测试项':<32} | {'规划组':<16} | {'算法':<12} | {'成功':<6} | {'耗时(ms)':<10} | {'路标点数'}")
    print("-" * 90)
    items = [
        ("7-DoF 自由空间快速规划", "left_arm", "RRTConnect", res1.success, res1.planning_time * 1000.0, len(res1.waypoints)),
        ("10-DoF 躯干协同大目标规划", "left_arm_torso", "RRTConnect", res2.success, res2.planning_time * 1000.0, len(res2.waypoints)),
        ("7-DoF 动态障碍物绕行避障", "left_arm", "RRTConnect", res3_arm.success, res3_arm.planning_time * 1000.0, len(res3_arm.waypoints)),
        ("10-DoF 躯干协同极速避障", "left_arm_torso", "RRTConnect", res3_torso.success, res3_torso.planning_time * 1000.0, len(res3_torso.waypoints)),
        ("渐进最优路径优化评测", "left_arm", "RRT*", res4_star.success, res4_star.planning_time * 1000.0, len(res4_star.waypoints)),
    ]
    for name, group, algo, succ, ms, pts in items:
        status_str = "✔️ YES" if succ else "❌ NO"
        print(f"{name:<32} | {group:<16} | {algo:<12} | {status_str:<6} | {ms:<10.2f} | {pts}")
    print("=" * 90)


if __name__ == "__main__":
    run_ompl_demo()
