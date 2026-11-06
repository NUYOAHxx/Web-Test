#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from ha_arm_ik.chained_ik import (
    Chained9DofArmAngleSolver,
    fast_nullspace_steer_elbow,
    solve_9dof_with_arm_angle,
)
from ha_arm_ik.ik_base import Status, invert_T
from ha_arm_ik.single_arm_ik import HAArmIKSolver, JointStateSnapshot
from tests.fixtures import make_solver, robot_xml, root_target


class Chained9DofArmAngleSolverTests(unittest.TestCase):
    """验证 9 轴腰臂协同 + 指定臂角两阶段级联求解器及零空间超高速优化。"""

    def setUp(self):
        self.s9, self.state = make_solver(include_waist=True)
        self.s7, _ = make_solver(include_waist=False)
        self.target_world = root_target(self.s9, self.state)
        self.snapshot = JointStateSnapshot(self.state)

    def test_solve_9dof_with_arm_angle_function(self):
        """测试函数接口 solve_9dof_with_arm_angle 能精确指定臂角。"""
        # 先求得自然解的臂角
        q_stage1, status, _ = self.s9.solve(
            self.target_world, frame="world", joint_state=self.snapshot
        )
        self.assertEqual(status, Status.SUCCESS)
        natural_psi = float(self.s7.chain.psi(q_stage1[2:]))

        # 设定一个明确的目标臂角（在自然臂角基础上偏移 20 度）
        target_psi = natural_psi + np.deg2rad(20.0)

        q_final, status, result = solve_9dof_with_arm_angle(
            solver_9dof=self.s9,
            solver_7dof=self.s7,
            T_target=self.target_world,
            psi_des=target_psi,
            frame="world",
            joint_state=self.snapshot,
            psi_tol=0.15,
            fast_mode=True,
        )

        self.assertEqual(status, "SUCCESS")
        self.assertIsNotNone(q_final)
        self.assertEqual(len(q_final), 9)

        # 验证手肘角度确实收敛到 target_psi
        achieved_psi = float(self.s7.chain.psi(q_final[2:]))
        self.assertAlmostEqual(achieved_psi, target_psi, places=2)
        self.assertAlmostEqual(result["psi"], target_psi, places=2)
        self.assertFalse(result.get("psi_fallback", True))
        self.assertIn(result.get("source"), ["fast_nullspace", "direct_psi_trf"])

        # 验证末端位姿依然精确达标
        fk_final = self.s9.chain.fk(q_final)
        T_target_root = invert_T(self.s9.T_world_from_base) @ self.target_world
        pos_err = np.linalg.norm(fk_final[:3, 3] - T_target_root[:3, 3])
        self.assertLess(pos_err, 5e-4)

    def test_fast_nullspace_steer_standalone(self):
        """测试 fast_nullspace_steer_elbow 独立步进函数的高速高精特性。"""
        q_stage1, status, _ = self.s9.solve(
            self.target_world, frame="world", joint_state=self.snapshot
        )
        self.assertEqual(status, Status.SUCCESS)
        q_arm_seed = q_stage1[2:]

        link_T, _ = self.s9.chain.fk_all(q_stage1)
        T_target_root = invert_T(self.s9.T_world_from_base) @ self.target_world
        T_armbase = invert_T(link_T[self.s7.base_link]) @ T_target_root

        natural_psi = float(self.s7.chain.psi(q_arm_seed))
        target_psi = natural_psi + np.deg2rad(15.0)

        res = fast_nullspace_steer_elbow(
            chain=self.s7.chain,
            T_target=T_armbase,
            psi_des=target_psi,
            q_arm_seed=q_arm_seed,
            psi_tol=0.15,
            max_steps=6,
        )
        self.assertIsNotNone(res)
        self.assertTrue(res["success"])
        self.assertLess(res["pos_err"], 5e-4)
        self.assertLess(res["ori_err"], 5e-4)
        self.assertAlmostEqual(res["psi"], target_psi, delta=0.15)

    def test_chained_class_initialization_and_steer(self):
        """测试 Chained9DofArmAngleSolver 类封装并测试不同臂角下的末端一致性。"""
        xml, names = robot_xml(other=True, continuous=True)
        with TemporaryDirectory() as tmp:
            urdf_path = Path(tmp) / "robot.urdf"
            urdf_path.write_text(xml)

            solver = Chained9DofArmAngleSolver(
                urdf=str(urdf_path),
                side="left",
                verbose=False,
                config={
                    "load_allowable_efforts": {n: 300.0 for n in names + ["OtherJoint"]},
                    "load_mode": "off",
                },
            )

            q_ref = np.array([0.05, 0.1, 0.25, -0.6, 0.4, -0.8, 0.3, 0.6, -0.2])
            T_target = solver.chain_9dof.fk(q_ref)

            # 角度 1: 期望臂角 -1.8 rad (使用快速零空间)
            q1, s1, r1 = solver.solve(T_target, psi_des=-1.8, frame="local", fast_mode=True)
            self.assertEqual(s1, "SUCCESS")
            psi1 = solver.compute_psi(q1)
            self.assertAlmostEqual(psi1, -1.8, delta=0.15)

            # 角度 2: 期望臂角 -2.2 rad (使用快速零空间)
            q2, s2, r2 = solver.solve(T_target, psi_des=-2.2, frame="local", fast_mode=True)
            self.assertEqual(s2, "SUCCESS")
            psi2 = solver.compute_psi(q2)
            self.assertAlmostEqual(psi2, -2.2, delta=0.15)

            # 两个解的手肘角度不同，但末端位姿误差均极小
            self.assertNotAlmostEqual(psi1, psi2, places=1)
            T_res1 = solver.chain_9dof.fk(q1)
            T_res2 = solver.chain_9dof.fk(q2)
            self.assertLess(np.linalg.norm(T_res1[:3, 3] - T_target[:3, 3]), 5e-4)
            self.assertLess(np.linalg.norm(T_res2[:3, 3] - T_target[:3, 3]), 5e-4)

            solver.close()

    def test_solve_9dof_tracking_continuous_trajectory(self):
        """测试 9 轴一体化解析加权 LM 连续轨迹跟踪模式（100 路点无缝连续，中位耗时 < 1ms）。"""
        q_start = self.s9.q_ready.copy()
        T_start = self.s9.chain.fk(q_start)
        psi_base = float(self.s7.chain.psi(q_start[2:]))
        center = T_start[:3, 3] + np.array([-0.03, 0.0, 0.0])

        q_curr = q_start.copy()
        success_count = 0
        lats = []

        for i in range(100):
            angle = 2.0 * np.pi * (i / 100)
            T_tgt = T_start.copy()
            T_tgt[0, 3] = center[0] + 0.08 * 0.3 * np.cos(angle)
            T_tgt[1, 3] = center[1] + 0.08 * 0.7 * np.sin(angle)
            T_tgt[2, 3] = center[2] + 0.08 * 0.4 * np.sin(2.0 * angle)
            p_des = psi_base + np.deg2rad(25.0) * np.sin(angle)

            import time
            t0 = time.perf_counter()
            q_sol, st, res = solve_9dof_with_arm_angle(
                solver_9dof=self.s9,
                solver_7dof=self.s7,
                T_target=T_tgt,
                psi_des=p_des,
                frame="world",
                q_prev=q_curr,
                psi_tol=0.035,
                tracking_fastpath=True,
            )
            lats.append((time.perf_counter() - t0) * 1000.0)

            self.assertEqual(st, "SUCCESS")
            self.assertIsNotNone(q_sol)
            achieved_psi = float(self.s7.chain.psi(q_sol[2:]))
            self.assertAlmostEqual(achieved_psi, p_des, delta=0.05)

            # 验证末端位姿误差 < 0.5mm
            fk_res = self.s9.chain.fk(q_sol)
            T_tgt_root = invert_T(self.s9.T_world_from_base) @ T_tgt
            pos_err = np.linalg.norm(fk_res[:3, 3] - T_tgt_root[:3, 3])
            self.assertLess(pos_err, 5e-4)

            success_count += 1
            q_curr = q_sol

        self.assertEqual(success_count, 100)
        # 验证中位耗时 < 1.0 ms (超高速)
        self.assertLess(np.median(lats), 1.0)


if __name__ == "__main__":
    unittest.main()
