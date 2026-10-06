#!/usr/bin/env python3
"""
================================================================================
Unitree G1 MoveIt 2 + Pink IK 桌面端可视化实时运动演示程序
(Live 3D Motion Demo: Pink IK + MoveIt 2 OMPL Pipeline in RViz2)
================================================================================

功能：
依次执行 4 个典型工况任务，每次使用 Pink IK 求解最优关节构型并由 MoveIt 2 OMPL
规划轨迹后通过 Action 控制器执行，直接在 RViz2 桌面端驱动 3D 模型平滑运动：
1. 【工况 ①】桌面抓取 (Table Reach)
2. 【工况 ②】远距大跨度探取 (Long Reach, 腰部自适应大角度介入协同)
3. 【工况 ③】俯身低位拾取 (Low Pick, 躯干俯仰前屈协助下探)
4. 【工况 ④】平滑回正就绪 (Return to Stand, 腰部平滑归零)
"""

import os
import sys
import time
import numpy as np

DIR_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if DIR_ROOT not in sys.path:
    sys.path.insert(0, DIR_ROOT)

import rclpy
from rclpy.node import Node
from core.planning.moveit_ompl_planner import MoveItOMPLPlanner


def run_live_motion_demo():
    print("=" * 80)
    print(" [DEMO] Unitree G1 MoveIt 2 + Pink IK 实时 3D 运动演示")
    print("   (请观察屏幕上的 RViz2 桌面端窗口，机器人模型将开始执行轨迹运动)")
    print("=" * 80)

    if not rclpy.ok():
        rclpy.init()

    node = Node("g1_live_motion_demo_runner")
    planner = MoveItOMPLPlanner(node=node, wait_for_services=True, timeout_sec=5.0)

    if not planner.plan_service_client.service_is_ready():
        print("[FAIL] 错误: MoveIt /plan_kinematic_path 服务未就绪！")
        return

    motions = [
        {
            "title": "工况 ①：桌面抓取 (Table Reach)",
            "pos": [0.35, 0.22, 0.85],
            "rpy": [0.0, np.radians(20.0), 0.0],
            "group": "left_arm_torso",
            "desc": "手臂在舒适区伸出抓取，腰部保持直立稳定",
        },
        {
            "title": "工况 ②：远端大跨度探取 (Long Reach)",
            "pos": [0.55, 0.20, 0.82],
            "rpy": [0.0, 0.0, 0.0],
            "group": "left_arm_torso",
            "desc": "突破 7-DoF 单臂极限 (38cm)，腰部自适应偏航倾转大角度协助延伸",
        },
        {
            "title": "工况 ③：俯身低位拾取 (Low Pick)",
            "pos": [0.32, 0.20, 0.65],
            "rpy": [0.0, np.radians(40.0), 0.0],
            "group": "left_arm_torso",
            "desc": "捡拾地面低位物体，躯干俯仰大角度前屈下探",
        },
        {
            "title": "工况 ④：平滑复位就绪 (Return to Stand)",
            "pos": [0.18, 0.22, 0.88],
            "rpy": [0.0, 0.0, 0.0],
            "group": "left_arm_torso",
            "desc": "手爪收回胸前舒适区，腰部零空间优雅归零",
        },
    ]

    for i, m in enumerate(motions, 1):
        print(f"\n▶ [{i}/4] 正在执行【{m['title']}】...")
        print(f"  • 目标末端空间坐标: {m['pos']} (说明: {m['desc']})")

        # 1. Pink IK 前置解算 + MoveIt 2 OMPL 规划
        t_plan_start = time.time()
        plan_res = planner.plan_to_pose_with_pink_ik(
            group_name=m["group"],
            target_pos=m["pos"],
            target_rpy=m["rpy"],
            planner_id="RRTConnectkConfigDefault",
            allowed_planning_time=3.0,
        )

        if not plan_res.success:
            print(f"  [FAIL] 规划失败: {plan_res.error_message}")
            continue

        print(f"  [OK] {plan_res.summary()}")
        print(f"  [START] 正在将轨迹发送至控制器，驱动 RViz2 3D 模型平滑运动...")

        # 2. 调度 FollowJointTrajectory 执行轨迹并在 RViz2 中平滑回放
        ok_exec = planner.execute_plan(plan_res, group_name=m["group"])
        if ok_exec:
            print(f"  [OK] 【{m['title']}】动作执行完成！")
        else:
            print(f"  [WARN] 执行出现异常")

        # 停顿 1 秒以便观察
        time.sleep(1.0)

    print("\n" + "=" * 80)
    print("[SUCCESS] 4 组 MoveIt 2 + Pink IK 运动工况全部演示完毕！")
    print("   您可以在 RViz2 界面中用鼠标交互拖动末端 Interactive Marker 手柄进行更多自主测试。")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_live_motion_demo()
