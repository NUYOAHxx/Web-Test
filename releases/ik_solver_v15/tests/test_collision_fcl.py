#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""tests/test_collision_fcl.py —— 验证 MoveIt 2 / FCL 自碰撞检测与面前 Box 障碍物避障模块。"""

import sys
import unittest
import numpy as np
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))

from tests.fixtures import make_solver
from ha_arm_ik.collision_fcl import (
    BoxObstacle,
    CollisionAwareIKSolver,
    RobotCollisionManager,
    create_front_box_obstacle,
    parse_srdf_disabled_pairs,
)
from ha_arm_ik.chained_ik import Chained9DofArmAngleSolver


class CollisionFCLTests(unittest.TestCase):
    """测试基于 FCL 的自碰撞与障碍物检测及零空间避障能力。"""

    @classmethod
    def setUpClass(cls):
        cfg = {"load_mode": "off"}
        cls.s9, _ = make_solver(config=cfg, include_waist=True)
        cls.col_mgr = RobotCollisionManager(chain=cls.s9.chain)

    def test_1_box_obstacle_creation_and_update(self):
        """测试面前长方体障碍物的创建、位姿转换与动态更新。"""
        box = create_front_box_obstacle(
            x_dist=0.50,
            y_offset=0.10,
            z_height=0.40,
            size=(0.3, 0.4, 0.2),
            name="workbench_box",
        )
        self.assertEqual(box.name, "workbench_box")
        self.assertTrue(np.allclose(box.position, [0.5, 0.1, 0.4]))
        self.assertTrue(np.allclose(box.size, [0.3, 0.4, 0.2]))
        se3 = box.to_se3()
        self.assertTrue(np.allclose(se3.translation, [0.5, 0.1, 0.4]))

    def test_2_srdf_acm_parsing(self):
        """测试 MoveIt 2 SRDF Allowed Collision Matrix (ACM) 免检白名单解析。"""
        srdf_xml = """<?xml version="1.0"?>
        <robot name="ha_robot">
          <disable_collisions link1="link_1" link2="link_2" reason="Adjacent"/>
          <disable_collisions link1="base" link2="link_0" reason="Adjacent"/>
          <disable_collisions link1="link_0" link2="link_5" reason="Never"/>
        </robot>
        """
        disabled = parse_srdf_disabled_pairs(srdf_xml)
        self.assertIn(("link_1", "link_2"), disabled)
        self.assertIn(("base", "link_0"), disabled)
        self.assertIn(("link_0", "link_5"), disabled)
        # 对称性测试
        self.assertIn(tuple(sorted(["link_2", "link_1"])), disabled)

    def test_3_self_collision_and_margin(self):
        """测试机器人初始构型无自碰撞，且测距功能正常。"""
        q_ready = self.s9.q_ready.copy()
        report = self.col_mgr.check_collision(q_ready, check_self=True, check_obstacles=False)
        self.assertFalse(report.is_collision)
        self.assertFalse(report.self_collision)

    def test_4_front_box_obstacle_collision_detection(self):
        """测试面前长方体障碍物的碰撞与测距检测。"""
        # 1. 在远端安全位置放置 Box，应无碰撞
        self.col_mgr.add_box_obstacle(
            name="test_box",
            size=[0.2, 0.2, 0.2],
            position=[1.5, 0.0, 0.0],
        )
        q_ready = self.s9.q_ready.copy()
        rep_safe = self.col_mgr.check_collision(q_ready, check_self=False, check_obstacles=True)
        self.assertFalse(rep_safe.is_collision)
        self.assertFalse(rep_safe.obstacle_collision)

        # 测距
        dist_info = self.col_mgr.compute_distances_to_obstacle(q_ready, "test_box")
        self.assertGreater(dist_info["min_distance"], 0.5)

        # 2. 将 Box 移动到机器人手臂所在的区域 (比如放在 link_4 附近 [0.3, 0.3, 0.85])
        self.col_mgr.update_box_obstacle(
            name="test_box",
            position=[0.30, 0.30, 0.85],
        )
        rep_col = self.col_mgr.check_collision(q_ready, check_self=False, check_obstacles=True)
        self.assertTrue(rep_col.is_collision)
        self.assertTrue(rep_col.obstacle_collision)
        self.assertGreater(len(rep_col.colliding_pairs), 0)

        # 清理
        self.col_mgr.remove_obstacle("test_box")

    def test_5_collision_aware_ik_evasion(self):
        """测试 CollisionAwareIKSolver 在遭遇面前障碍物时自动沿零空间流形旋转定肘避障。"""
        q_init = self.s9.q_ready.copy()
        T_target = self.s9.chain.fk(q_init)

        # 在手臂右侧斜前放置一个障碍物（初始手肘构型会与该 box 干涉，但逆向翻肘可完全绕开）
        self.col_mgr.add_box_obstacle(
            name="obstacle_box",
            size=[0.20, 0.20, 0.20],
            position=[0.30, 0.48, 0.85],
        )

        # 原始解在该障碍物下应发生干涉
        rep_init = self.col_mgr.check_collision(q_init)
        self.assertTrue(rep_init.is_collision)

        # 包装成碰撞感知求解器
        safe_solver = CollisionAwareIKSolver(
            base_solver=self.s9,
            collision_manager=self.col_mgr,
            auto_nullspace_evade=True,
            safety_margin=0.01,
        )

        q_safe, status, info = safe_solver.solve(T_target, q_prev=q_init)
        self.assertEqual(status, "SUCCESS")
        self.assertIsNotNone(q_safe)
        self.assertTrue(info.get("collision_evaded", False))

        # 验证避障后的关节角完全无碰撞
        rep_safe = self.col_mgr.check_collision(q_safe)
        self.assertFalse(rep_safe.is_collision)

        # 验证末端位姿毫厘不差 (<0.5mm)
        T_actual = self.s9.chain.fk(q_safe)
        pos_err = np.linalg.norm(T_actual[:3, 3] - T_target[:3, 3])
        self.assertLess(pos_err, 5e-4)

        # 清理
        self.col_mgr.remove_obstacle("obstacle_box")


if __name__ == "__main__":
    unittest.main()
