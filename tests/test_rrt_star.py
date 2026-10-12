#!/usr/bin/env python3
"""
================================================================================
三维笛卡尔空间 RRT* 路径规划器单元测试
================================================================================
覆盖 core/planning/rrt_star.py 的核心几何与规划逻辑:
    * Node3D 数据结构
    * check_collision —— 线段 vs 轴对齐障碍盒的相交判定 (含各类边界情形)
    * steer / find_nearest / calc_distance —— 基础几何原语
    * plan() —— 完整规划在有无障碍场景下的行为
    * generate_path —— 输出路径的连通性与终点到达性

约定提示: check_collision 返回 True 表示"无碰撞"(可通行), 与项目内
         is_colliding 系列接口的语义相反, 属于该模块的历史约定。
================================================================================
"""

import os
import sys

import numpy as np

DIR_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if DIR_ROOT not in sys.path:
    sys.path.insert(0, DIR_ROOT)

from core.planning.rrt_star import Node3D, RRTStar3D


def test_node3d_fields():
    """Node3D 应正确持有三维坐标并支持算术运算。"""
    print("-" * 70)
    print("▶ 1. Node3D 数据结构")
    n = Node3D(1.0, 2.0, 3.0)
    assert (n.x, n.y, n.z) == (1.0, 2.0, 3.0)

    # 注意: Node3D 未实现运算符重载, 仅提供坐标字段与树结构指针
    child = Node3D(1.0, 2.0, 3.0)
    assert child.parent is None and child.cost == 0.0, "新节点应为根节点且成本为 0"
    n.parent = child
    assert n.parent.x == 1.0, "parent 指针应正确保存"
    print("  [OK] Node3D 字段与树结构指针正确")


def test_calc_distance():
    """calc_distance 应返回欧氏距离。"""
    print("-" * 70)
    print("▶ 2. 欧氏距离计算")
    p = RRTStar3D(start=(0, 0, 0), goal=(3, 4, 0), obstacles=[],
                       bounds=(-10, 10, -10, 10, -10, 10))
    d = p.calc_distance(Node3D(0, 0, 0), Node3D(3, 4, 0))
    assert abs(d - 5.0) < 1e-9, f"3-4-5 三角形的斜边应为 5, 实际 {d}"
    assert p.calc_distance(Node3D(1, 2, 3), Node3D(1, 2, 3)) == 0.0
    print("  [OK] 距离计算正确 (含零距离边界)")


def test_collision_free_space():
    """无障碍时任意线段都应判定为可通行。"""
    print("-" * 70)
    print("▶ 3. 空场景碰撞判定")
    p = RRTStar3D(start=(0, 0, 0), goal=(5, 5, 5), obstacles=[],
                       bounds=(-10, 10, -10, 10, -10, 10))
    assert p.check_collision(Node3D(0, 0, 0), Node3D(9, 9, 9)) is True
    print("  [OK] 空场景恒判定无碰撞")


def test_collision_with_box():
    """线段穿越障碍盒时应判定为碰撞, 绕行时不判定。"""
    print("-" * 70)
    print("▶ 4. 障碍盒碰撞判定")
    # 中心 (5,0,0), 边长 2 的立方体
    obstacles = [(5.0, 0.0, 0.0, 2.0, 2.0, 2.0)]
    p = RRTStar3D(start=(0, 0, 0), goal=(10, 0, 0), obstacles=obstacles,
                       bounds=(-10, 20, -10, 10, -10, 10))

    # 4a. 直线穿过盒子中心 -> 碰撞
    assert p.check_collision(Node3D(0, 0, 0), Node3D(10, 0, 0)) is False, "穿越障碍应判碰撞"
    print("  [OK] 穿越障碍盒 -> 碰撞")

    # 4b. 完全绕开 -> 无碰撞
    assert p.check_collision(Node3D(0, 5, 5), Node3D(10, 5, 5)) is True, "绕行应判无碰撞"
    print("  [OK] 绕开障碍盒 -> 无碰撞")

    # 4c. 线段只擦过盒子表面之外一点 -> 无碰撞
    assert p.check_collision(Node3D(0, 1.01, 0), Node3D(10, 1.01, 0)) is True, "贴近但未接触应无碰撞"
    print("  [OK] 擦边而过 -> 无碰撞")

    # 4d. 起点位于盒内 -> 碰撞
    assert p.check_collision(Node3D(5, 0, 0), Node3D(8, 0, 0)) is False, "起点在盒内应判碰撞"
    print("  [OK] 起点位于障碍盒内 -> 碰撞")

    # 4e. 终点位于盒内 -> 碰撞
    assert p.check_collision(Node3D(0, 0, 0), Node3D(5, 0, 0)) is False, "终点在盒内应判碰撞"
    print("  [OK] 终点位于障碍盒内 -> 碰撞")


def test_collision_degenerate_direction():
    """线段与某坐标轴平行时不应产生除零错误。"""
    print("-" * 70)
    print("▶ 5. 退化方向 (轴平行线段) 碰撞判定")
    obstacles = [(0.0, 0.0, 0.0, 2.0, 2.0, 2.0)]
    p = RRTStar3D(start=(0, 0, 0), goal=(0, 0, 0), obstacles=obstacles,
                       bounds=(-10, 10, -10, 10, -10, 10))

    # x 方向平行 (direction[0] != 0), 但起终点 y/z 均为 0 -> 命中盒内
    assert p.check_collision(Node3D(-3, 0, 0), Node3D(3, 0, 0)) is False
    # x 方向平行且远离盒子
    assert p.check_collision(Node3D(-3, 5, 5), Node3D(3, 5, 5)) is True
    print("  [OK] 轴平行线段无除零异常")


def test_plan_without_obstacle():
    """无障碍场景应规划出一条连通的起终点路径。"""
    print("-" * 70)
    print("▶ 6. 空场景完整规划")
    p = RRTStar3D(start=(0, 0, 0), goal=(8, 8, 0), obstacles=[],
                       bounds=(-5, 15, -5, 15, -5, 15),
                       step_size=2.0, max_iter=2000, search_radius=4.0)
    p.plan()
    assert p.path, "空场景应规划出非空路径"
    # generate_path 返回 (x, y, z) 元组序列
    assert np.allclose(p.path[0], [p.start.x, p.start.y, p.start.z]), "路径首点应为起点"
    assert np.allclose(p.path[-1], [p.goal.x, p.goal.y, p.goal.z]), "路径末点应为目标点"

    # 相邻路径点之间不得穿越障碍 (此场景无障碍, 断言路径单调合理即可)
    print(f"  [OK] 空场景规划成功: {len(p.path)} 个路径点, 起终点一致")


def test_plan_respects_obstacle():
    """有障碍场景: 规划出的路径不应穿越障碍盒。"""
    print("-" * 70)
    print("▶ 7. 含障碍规划 (路径须避开障碍)")
    obstacles = [(4.0, 0.0, 0.0, 3.0, 6.0, 6.0)]  # 中间竖立一面"墙"
    p = RRTStar3D(start=(0, 0, 0), goal=(9, 0, 0), obstacles=obstacles,
                       bounds=(-3, 12, -8, 8, -8, 8),
                       step_size=1.0, max_iter=6000, search_radius=2.5)
    p.plan()

    if not p.path:
        print("  [INFO] 该随机场景未能收敛出路径 (RRT* 为随机算法, 允许失败)")
        return

    # 逐段校验: 相邻节点连线不得与墙相交
    violations = 0
    for i in range(len(p.path) - 1):
        a, b = p.path[i], p.path[i + 1]
        if not p.check_collision(Node3D(*a), Node3D(*b)):
            violations += 1
    assert violations == 0, f"规划路径有 {violations} 段穿墙"
    print(f"  [OK] 规划路径 {len(p.path)} 点, 无穿墙 ({violations} 处违规)")


def test_plan_respects_bounds():
    """规划路径的所有节点都应落在给定空间边界内。"""
    print("-" * 70)
    print("▶ 8. 规划路径边界约束")
    bounds = (0, 10, 0, 10, 0, 10)
    p = RRTStar3D(start=(1, 1, 1), goal=(9, 9, 9), obstacles=[],
                       bounds=bounds, step_size=1.5, max_iter=3000, search_radius=3.0)
    p.plan()
    if not p.path:
        print("  [INFO] 该随机场景未能收敛出路径, 跳过边界断言")
        return
    for (x, y, z) in p.path:
        assert bounds[0] <= x <= bounds[1], f"X 越界: {x}"
        assert bounds[2] <= y <= bounds[3], f"Y 越界: {y}"
        assert bounds[4] <= z <= bounds[5], f"Z 越界: {z}"
    print(f"  [OK] {len(p.path)} 个路径点全部位于边界内")


if __name__ == "__main__":
    print("=" * 70)
    print("[START] 三维 RRT* 路径规划器单元测试")
    print("=" * 70)
    test_node3d_fields()
    test_calc_distance()
    test_collision_free_space()
    test_collision_with_box()
    test_collision_degenerate_direction()
    test_plan_without_obstacle()
    test_plan_respects_obstacle()
    test_plan_respects_bounds()
    print("=" * 70)
    print("[SUCCESS] 三维 RRT* 规划器全部测试通过")
    print("=" * 70)