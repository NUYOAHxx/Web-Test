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

    # 3.1 RRTConnect 求解
    print("  [3.1] 使用 RRTConnect 双向扩展树规划...")
    res3_rrt = planner.plan_to_pose(
        group_name="left_arm",
        target_pos=target_pos_3,
        planner_id="RRTConnectkConfigDefault",
        allowed_planning_time=5.0,
    )
    print(f"    • {res3_rrt.summary()}")

    # 3.2 RRT* 最优规划求解
    print("  [3.2] 使用 RRT* 渐进最优算法规划...")
    res3_star = planner.plan_to_pose(
        group_name="left_arm",
        target_pos=target_pos_3,
        planner_id="RRTstarkConfigDefault",
        allowed_planning_time=5.0,
    )
    print(f"    • {res3_star.summary()}")

    # 清理障碍物
    print("  • 清理测试障碍物...")
    planner.remove_obstacle(obs_name)
    time.sleep(0.2)

    # ──────────────────────────────────────────────────────────────────────────
    # 汇总输出性能对比表
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print(" 📊 MoveIt 2 OMPL 规划性能总结报表")
    print("=" * 80)
    print(f"{'测试项':<30} | {'规划组':<16} | {'算法':<14} | {'成功':<6} | {'耗时(ms)':<10} | {'路标点数'}")
    print("-" * 88)
    items = [
        ("7-DoF 自由空间目标", "left_arm", "RRTConnect", res1.success, res1.planning_time * 1000.0, len(res1.waypoints)),
        ("10-DoF 躯干协同大目标", "left_arm_torso", "RRTConnect", res2.success, res2.planning_time * 1000.0, len(res2.waypoints)),
        ("障碍物绕行规划 (双向树)", "left_arm", "RRTConnect", res3_rrt.success, res3_rrt.planning_time * 1000.0, len(res3_rrt.waypoints)),
        ("障碍物渐进最优 (RRT*)", "left_arm", "RRT*", res3_star.success, res3_star.planning_time * 1000.0, len(res3_star.waypoints)),
    ]
    for name, group, algo, succ, ms, pts in items:
        status_str = "✔️ YES" if succ else "❌ NO"
        print(f"{name:<30} | {group:<16} | {algo:<14} | {status_str:<6} | {ms:<10.2f} | {pts}")
    print("=" * 88)


if __name__ == "__main__":
    run_ompl_demo()
