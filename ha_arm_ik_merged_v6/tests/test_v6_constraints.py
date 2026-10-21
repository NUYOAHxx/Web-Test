"""merged-v6 约束与修正的回归测试（合成 URDF + 真实 Pinocchio；ik_common / ik_candidate 为测试替身）。"""
import re
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pinocchio as pin

from tests.fixtures import context, make_solver, robot_xml, root_target
import tests.fixtures as F
from ha_arm_ik import solver as S
import ha_arm_ik.solver_dls as D
from ha_arm_ik.solver_dls import JointStateSnapshot


def solve(s, state, q=None, **kw):
    return s.solve(root_target(s, state, q), frame="world",
                   joint_state=JointStateSnapshot(state), **kw)


class ComTests(unittest.TestCase):
    def setUp(self):
        self.s, self.state = make_solver()
        self.ctx = context(self.s, self.state)
        self.q = np.array([self.state[n] for n in self.s.active_joint_names]) + 0.1

    def manual_com(self, q):
        """按 URDF 全部 link 的质量点手工求质心（含根 link 与固定 link）。"""
        m, d = self.s.pin_model, self.s.pin_data
        full = D.active_q_to_pinocchio_q(m, self.s.pin_maps, q)
        pin.forwardKinematics(m, d, full)
        pin.updateFramePlacements(m, d)
        total, acc = 0.0, np.zeros(3)
        for fid, mass, xyz in self.s.pin_maps["mass_points"]:
            acc += mass * (d.oMf[fid].act(np.asarray(xyz)))
            total += mass
        return acc / total, full

    def test_com_includes_root_and_fixed_masses(self):
        expected, full = self.manual_com(self.q)
        com, J = D.model_com_with_jacobian(self.s.pin_model, self.s.pin_data, full,
                                           active_v=self.s.pin_maps["active_v_indices"])
        np.testing.assert_allclose(com, expected, atol=1e-12)
        self.assertGreater(self.s.pin_model.inertias[0].mass, 0.0)  # 夹具根 link 有质量

    def test_com_jacobian_matches_finite_difference(self):
        _, full = self.manual_com(self.q)
        _, J = D.model_com_with_jacobian(self.s.pin_model, self.s.pin_data, full,
                                         active_v=self.s.pin_maps["active_v_indices"])
        num = np.zeros_like(J)
        for j in range(len(self.q)):
            dq = np.eye(len(self.q))[j] * 1e-6
            num[:, j] = (self.manual_com(self.q + dq)[0] - self.manual_com(self.q - dq)[0]) / 2e-6
        np.testing.assert_allclose(J, num, atol=1e-8)


class NullspaceAndBoundsTests(unittest.TestCase):
    def test_sns_nullspace_step_is_descent_with_nonuniform_metric(self):
        rng = np.random.default_rng(5)
        for _ in range(500):
            n = 9
            J = rng.normal(size=(6, n))
            W = np.diag(rng.uniform(0.15, 1.0, size=n))
            z = rng.normal(size=n)
            lo, hi = -np.full(n, 10.0), np.full(n, 10.0)
            dq, dqn = D._solve_sns_clamped_step(J, rng.normal(size=6) * 0.1, z, 1.0, W,
                                                np.ones(n, bool), np.zeros(n), lo, hi, 1e-6, 10.0)
            self.assertGreaterEqual(float(z @ dqn), -1e-12)
            # 零空间步不应扰动主任务（阻尼极小）
            self.assertLess(np.linalg.norm(J @ dqn), 1e-4 * max(1.0, np.linalg.norm(z)))

    def test_sns_rounding_stays_within_snap_tolerance(self):
        rng = np.random.default_rng(9)
        for _ in range(3000):
            n = 9
            J = rng.normal(size=(6, n))
            lo, hi = -rng.uniform(0.05, 3, n), rng.uniform(0.05, 3, n)
            q = np.clip(rng.uniform(lo, hi), lo, hi)
            dq, _ = D._solve_sns_clamped_step(J, rng.normal(size=6), rng.normal(size=n), 1.0,
                                              np.eye(n), np.ones(n, bool), q, lo, hi, 1e-6, 0.5)
            t = q + dq
            self.assertLessEqual(max(np.max(lo - t), np.max(t - hi)), D.BOUND_SNAP_TOL)

    def test_scatter_seeds_start_with_q_prev(self):
        lo, hi = -np.full(3, 3.0), np.full(3, 3.0)
        q_prev, q_ready = np.array([1.0, -1.0, 0.5]), np.zeros(3)
        seeds = D.make_scatter_seeds(q_prev, q_ready, lo, hi, np.random.default_rng(0))
        np.testing.assert_array_equal(seeds[0], q_prev)
        np.testing.assert_array_equal(seeds[1], q_ready)


class SeedLoopTests(unittest.TestCase):
    def setUp(self):
        self.s, self.state = make_solver()
        self.ctx = context(self.s, self.state)
        self.qm = np.array([self.state[n] for n in self.s.active_joint_names])

    def one_seed(self, target, **kw):
        return D.pinocchio_ik_one_seed(self.s.pin_model, self.s.pin_data, self.s.pin_maps,
                                       self.s.pin_ee_frame_id, target, self.qm, self.s.lower,
                                       self.s.upper, q_ref=self.s.q_ready, include_waist=True,
                                       load_context=self.ctx, **kw)

    def test_unreachable_locked_waist_seed_stops_on_stall(self):
        qt = self.qm.copy(); qt[:2] += [0.6, -0.6]      # 目标需要腰部大幅运动
        target = root_target(self.s, self.state, qt)
        waist = np.array([n in S.WAIST_JOINT_NAMES for n in self.s.active_joint_names])
        q, diag = self.one_seed(target, fixed_joint_mask=waist, max_iter=500)
        self.assertFalse(diag["converged"])
        self.assertEqual(diag["stop_reason"], "STALLED_NO_PROGRESS")
        self.assertLess(diag["iters"], 200)

    def test_timeout_inside_line_search_is_time_budget(self):
        target = root_target(self.s, self.state, self.qm + 0.3)
        clock = iter([0.0] + [100.0] * 10000)
        with patch.object(D.time, "monotonic", lambda: next(clock)):
            q, diag = self.one_seed(target, deadline=50.0, max_iter=50)
        self.assertEqual(diag["stop_reason"], "TIME_BUDGET")

    def test_converges_and_reports_peak(self):
        target = root_target(self.s, self.state, self.qm + 0.2)
        q, diag = self.one_seed(target, max_iter=200)
        self.assertTrue(diag["converged"])
        self.assertIn(diag["stop_reason"], ("SETTLED", "SETTLE_LIMIT", "SMALL_STEP"))
        self.assertTrue(diag["final_load_ok"])
        self.assertGreater(diag["final_peak_ratio"], 0.0)


def build_wide_link5(limit=2 * np.pi):
    """Link5 改为 ±limit 的宽范围 revolute（多圈关节），用于回中 / 分支测试。"""
    xml, names = robot_xml(other=True, continuous=False)
    pat = re.compile(r'(<joint name="LHand_Link5" type="revolute">.*?<limit lower=")-3(" upper=")3(")', re.S)
    xml, n = pat.subn(r'\g<1>%.6f\g<2>%.6f\g<3>' % (-limit, limit), xml)
    assert n == 1
    tmp = TemporaryDirectory(); path = Path(tmp.name) / "robot.urdf"; path.write_text(xml)
    qref = np.array([.05, .1, .25, -.6, .4, -.8, .3, .6, -.2])
    S.ARM_CONFIGS["left"] = {"base_link": "link_1", "ee_link": "tcp", "s_joint": "LHand_Link2",
                             "e_joint": "LHand_Link4", "w_joint": "LHand_Link6",
                             "mount_xyz": np.zeros(3), "mount_rpy": np.zeros(3), "q_ready": qref[2:]}
    cfg = {"load_allowable_efforts": {n: 300. for n in names + ["OtherJoint"]},
           "load_massless_links": ["tcp"], "load_budget_s": 5., "load_state_max_age_s": 60.,
           "dls_max_iter": 300, "q_ready": qref}
    s = S.HAArmIKSolver(path, "left", include_waist=True, verbose=False, config=cfg)
    s._tmp = tmp
    return s, names, qref


class PostureAndBranchTests(unittest.TestCase):
    def test_multi_turn_joint_recenters_physically_and_result_keeps_improvement(self):
        s, names, qref = build_wide_link5()
        self.addCleanup(s._tmp.cleanup)
        i5 = s.active_joint_names.index("LHand_Link5")
        state = dict(zip(names, qref)); state["OtherJoint"] = .4
        state["LHand_Link5"] = qref[6] + 4.8            # 5.1 rad，等价角 -1.18 rad
        ctx = D.prepare_load_context(s.pin_model, s.pin_maps, s._load_config,
                                     JointStateSnapshot(state))
        s.pin_maps["q_template"] = ctx["q_template"].copy()
        seed = np.array([state[n] for n in s.active_joint_names])
        T = root_target(s, state)
        q, diag = D.pinocchio_ik_one_seed(s.pin_model, s.pin_data, s.pin_maps, s.pin_ee_frame_id,
                                          T, seed, s.lower, s.upper, q_ref=qref, include_waist=True,
                                          max_iter=300, load_context=ctx)
        self.assertTrue(diag["converged"])
        # 物理坐标回中：朝 q_ready(0.3) 方向移动，绝不朝 +2π 上限漂移；且择优保留了这次姿态改善
        self.assertLess(q[i5], seed[i5] - 0.05)
        q2, status, _ = s.solve(T, frame="world", joint_state=JointStateSnapshot(state))
        self.assertEqual(status, "VALID")
        self.assertLess(q2[i5], seed[i5])
        self.assertLessEqual(abs(q2[i5] - seed[i5]), np.pi)   # 不绕大圈


class JointJumpAndFailureCodeTests(unittest.TestCase):
    def setUp(self):
        self.s, self.state = make_solver()
        self.qm = np.array([self.state[n] for n in self.s.active_joint_names])

    def test_default_jump_limit_reported_on_success(self):
        q, status, _ = solve(self.s, self.state, self.qm + 0.3)
        self.assertEqual(status, "VALID")
        d = self.s.last_load_diagnostics
        self.assertTrue(d["joint_jump_ok"])
        self.assertLessEqual(d["joint_jump_max_rad"], np.pi)
        self.assertAlmostEqual(d["joint_jump_max_rad"], float(np.max(np.abs(q - self.qm))), places=9)

    def test_reject_policy_fails_with_joint_jump_code(self):
        s, state = make_solver(config={"max_joint_jump_rad": 0.05})
        qm = np.array([state[n] for n in s.active_joint_names])
        q, status, result = solve(s, state, qm + 0.4)
        self.assertIsNone(q)
        self.assertEqual(s.last_load_diagnostics["code"], "JOINT_JUMP_LIMIT")
        self.assertIn("JOINT_JUMP_LIMIT", result.failure_detail)

    def test_prefer_policy_returns_flagged_solution(self):
        s, state = make_solver(config={"max_joint_jump_rad": 0.05, "joint_jump_policy": "prefer"})
        qm = np.array([state[n] for n in s.active_joint_names])
        q, status, _ = solve(s, state, qm + 0.4)
        self.assertEqual(status, "VALID")
        self.assertFalse(s.last_load_diagnostics["joint_jump_ok"])

    def test_per_joint_limits_and_validation(self):
        s, state = make_solver(config={"max_joint_jump_rad": {"Hip_Link1": 0.5, "LHand_Link7": None}})
        names = s.active_joint_names
        self.assertEqual(s.max_joint_jump[names.index("Hip_Link1")], 0.5)
        self.assertTrue(np.isinf(s.max_joint_jump[names.index("LHand_Link7")]))
        self.assertEqual(s.max_joint_jump[names.index("LHand_Link2")], np.pi)
        with self.assertRaises(ValueError):
            make_solver(config={"max_joint_jump_rad": {"NoSuchJoint": 1.0}})
        with self.assertRaises(ValueError):
            make_solver(config={"joint_jump_policy": "maybe"})

    def test_seed_exceptions_are_reported_not_misread_as_load(self):
        def boom(*a, **k):
            raise RuntimeError("synthetic seed failure")
        with patch.object(S, "pinocchio_ik_one_seed", boom):
            q, status, result = solve(self.s, self.state, self.qm + 0.2)
        self.assertIsNone(q)
        d = self.s.last_load_diagnostics
        self.assertEqual(d["code"], "SEED_ERRORS")
        self.assertIn("synthetic seed failure", result.failure_detail)
        self.assertGreater(d["search"]["seed_errors"], 0)

    def test_unreachable_target_is_pose_failure_not_load_failure(self):
        T = root_target(self.s, self.state)
        T[:3, 3] += [5.0, 0.0, 0.0]                      # 远超臂展
        q, status, result = self.s.solve(T, frame="world", joint_state=JointStateSnapshot(self.state))
        self.assertIsNone(q)
        self.assertEqual(self.s.last_load_diagnostics["code"], "NO_VALID_POSE_OR_PHYSICS")


class LoadPolicyCacheTests(unittest.TestCase):
    def test_vectors_cached_and_read_only(self):
        p = D.LoadPolicy({"a": 10.0, "b": 20.0}, joint_weights={"a": 2.0})
        c1, w1 = p.vectors(["a", "b"])
        c2, w2 = p.vectors(("a", "b"))
        self.assertIs(c1, c2)
        np.testing.assert_array_equal(w1, [2.0, 1.0])
        with self.assertRaises(ValueError):
            c1[0] = 1.0
        with self.assertRaises(ValueError):
            p.vectors(["a", "missing"])


if __name__ == "__main__":
    unittest.main()
