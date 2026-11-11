"""多核 CPU 并发多种子求解单元测试与正确性验证。"""
import unittest
import numpy as np
import os

from tests import fixtures as F
from ha_arm_ik.single_arm_ik import HAArmIKSolver, JointStateSnapshot, POS_TOL


class ParallelIKSolverTests(unittest.TestCase):
    def test_parallel_matches_sequential_on_repositioning(self):
        """验证多核并发求解与串行求解在重定位模式下的收敛性与解质量一致。"""
        s_seq, state = F.make_solver(config={"load_mode": "off", "collision_avoidance": False, "parallel_workers": 1})
        s_par, _ = F.make_solver(config={"load_mode": "off", "collision_avoidance": False, "parallel_workers": 2})

        try:
            names = s_seq.active_joint_names
            q0 = np.array([state[n] for n in names])
            rng = np.random.default_rng(42)
            qt = np.clip(q0 + rng.uniform(-0.6, 0.6, len(q0)), -2.4, 2.4)
            T = F.root_target(s_seq, state, qt)

            q_seq, st_seq, res_seq = s_seq.solve(T, frame="world", q_prev=q0)
            q_par, st_par, res_par = s_par.solve(T, frame="world", q_prev=q0)

            self.assertEqual(st_seq, "SUCCESS")
            self.assertEqual(st_par, "SUCCESS")
            self.assertLess(res_seq.pos_err, POS_TOL)
            self.assertLess(res_par.pos_err, POS_TOL)
        finally:
            s_seq.close()
            s_par.close()

    def test_continuation_mode_fast_path(self):
        """验证续算模式在并发模式下仍走单核直通快速路径，保持 ~1ms 伺服响应。"""
        s, state = F.make_solver(config={"load_mode": "off", "collision_avoidance": False, "parallel_workers": 4})
        try:
            names = s.active_joint_names
            q0 = np.array([state[n] for n in names])
            T0 = F.root_target(s, state, q0)

            q, status, res = s.solve(T0, frame="world", q_prev=q0)
            self.assertEqual(status, "SUCCESS")
            self.assertTrue(s.last_load_diagnostics["continuation"]["hold"])
            np.testing.assert_allclose(q, q0, atol=1e-12)
        finally:
            s.close()

    def test_context_manager_lifecycle(self):
        """验证上下文管理器能够自动创建并在退出时回收多进程池。"""
        s, state = F.make_solver(config={"load_mode": "off", "collision_avoidance": False, "parallel_workers": 2})
        with s:
            pool = s._ensure_parallel_pool()
            self.assertIsNotNone(pool)
        self.assertIsNone(s._parallel_pool)


if __name__ == "__main__":
    unittest.main()
