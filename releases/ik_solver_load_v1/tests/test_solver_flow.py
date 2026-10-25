from dataclasses import replace
import json
import time
import unittest
from unittest.mock import patch
import numpy as np

from ha_arm_ik.load_policy import JointStateSnapshot, LoadPolicy
from ha_arm_ik.solver_load_aware import LoadAwareIKSolver, SolverConfig, inspect_urdf
from tests.fixtures import make_solver, target_for, robot_xml, planar_xml


class TestSolverFlow(unittest.TestCase):
    def test_reaches_known_full_target_and_checks_final(self):
        s, state = make_solver()
        q_goal = s.q_reference + np.linspace(-.08,.08,9)
        T = target_for(s, state, q_goal)
        q, status, r = s.solve(T, frame="model", snapshot=JointStateSnapshot(state))
        self.assertEqual(status, "SUCCESS", r)
        self.assertTrue(r["accepted"])
        self.assertLessEqual(r["selected"]["position_error_m"], s.config.pos_tol_m)
        self.assertLessEqual(r["selected"]["orientation_violation_rad"], s.config.ori_tol_rad)
        self.assertLessEqual(r["selected"]["peak_ratio"], 1.)
        self.assertIsNotNone(q)

    def test_each_accepted_state_has_load_and_pose_gates(self):
        s, state = make_solver()
        for scale in (0., .05, -.07):
            with self.subTest(scale=scale):
                T = target_for(s, state, s.q_reference + scale)
                q, status, r = s.solve(T, frame="model", snapshot=JointStateSnapshot(state))
                self.assertEqual(status, "SUCCESS", r)
                self.assertTrue(r["selected"]["load_ok"] and r["selected"]["pose_ok"])
                self.assertTrue(r["selected"]["bounds_ok"])

    def test_no_approximate_overload_bypass(self):
        xml, names = robot_xml()
        policy = LoadPolicy({n:.001 for n in names+["OtherJoint"]})
        config = SolverConfig(max_iterations=1, settle_iterations=1, max_seeds=3,
                              min_seeds=3, budget_s=3., state_max_age_s=20.)
        s, state = make_solver(policy=policy, config=config)
        q, status, r = s.solve(target_for(s,state), frame="model", allow_approximate=True,
                               snapshot=JointStateSnapshot(state))
        self.assertIsNone(q)
        self.assertEqual(status, "NO_LOAD_SAFE_SOLUTION")
        self.assertFalse(r["accepted"])
        self.assertFalse(r["approximate_accepted"])

    def test_locked_solution_does_not_skip_unlocked_comparison(self):
        s, state = make_solver()
        q, status, r = s.solve(target_for(s,state), frame="model", snapshot=JointStateSnapshot(state))
        self.assertEqual(status, "SUCCESS", r)
        self.assertTrue(r["per_seed"][0]["fixed_waist"])
        self.assertGreaterEqual(r["unlocked_seeds_attempted"], 2)

    def test_high_locked_load_releases_waist_metric(self):
        _, names = robot_xml()
        policy = LoadPolicy({n:.001 for n in names+["OtherJoint"]})
        config = SolverConfig(max_iterations=1, settle_iterations=1, max_seeds=3,
                              min_seeds=3, budget_s=3., state_max_age_s=20.)
        s, state = make_solver(policy=policy, config=config)
        _, _, r = s.solve(target_for(s,state), frame="model", snapshot=JointStateSnapshot(state))
        self.assertTrue(r["waist_released_for_load"])
        self.assertTrue(r["per_seed"][1]["waist_released_for_load"])

    def test_missing_nonactive_state_rejected(self):
        s, state = make_solver()
        T = target_for(s,state); del state["OtherJoint"]
        q, status, r = s.solve(T, snapshot=JointStateSnapshot(state))
        self.assertIsNone(q)
        self.assertIn("missing whole-model", r["reason"])

    def test_out_of_bounds_observation_not_clipped(self):
        s, state = make_solver()
        T = target_for(s,state); state["Hip_Link1"] = 4.
        q, _, r = s.solve(T, snapshot=JointStateSnapshot(state))
        self.assertIsNone(q)
        self.assertIn("not clipped", r["reason"])

    def test_old_state_rejected(self):
        s, state = make_solver()
        q, status, r = s.solve(target_for(s,state),
                               snapshot=JointStateSnapshot(state, time.monotonic()-100))
        self.assertIsNone(q)
        self.assertEqual(status, "STALE_STATE")

    def test_invalid_target_rejected(self):
        s, state = make_solver()
        T = np.eye(4); T[0,0] = 2.
        q, _, r = s.solve(T, snapshot=JointStateSnapshot(state))
        self.assertIsNone(q)
        self.assertIn("rigid transform", r["reason"])

    def test_missing_limits_fail_at_construction(self):
        with self.assertRaisesRegex(ValueError, "missing confirmed"):
            make_solver(policy=LoadPolicy({"Hip_Link1": 100.}))

    def test_active_joint_cannot_be_unguarded(self):
        with self.assertRaisesRegex(ValueError, "every active joint"):
            make_solver(policy=LoadPolicy({"Hip_Link1":100.}, guard_joint_names=["Hip_Link1"]))

    def test_explicit_subset_reports_unchecked_inactive_joint(self):
        _, names = robot_xml()
        p = LoadPolicy({n:300. for n in names}, guard_joint_names=names)
        s, state = make_solver(policy=p)
        _, _, r = s.solve(target_for(s,state), frame="model", snapshot=JointStateSnapshot(state))
        self.assertEqual(r["unchecked_joint_names"], ["OtherJoint"])

    def test_position_override_must_not_expand_urdf(self):
        with self.assertRaisesRegex(ValueError, "expand"):
            make_solver(position_bounds={"Hip_Link1": [-4.,4.]})

    def test_continuous_explicit_window_not_periodic(self):
        s, _ = make_solver(position_bounds={"LHand_Link3": [-2.,2.]})
        self.assertFalse(s.continuous[4])
        self.assertEqual(s.lower[4], -2.)

    def test_missing_mass_is_not_silently_zero(self):
        xml, names = robot_xml()
        with self.assertRaisesRegex(ValueError, "mass missing/zero"):
            LoadAwareIKSolver(xml, base_link="base", ee_link="tcp",
                               policy=LoadPolicy({n:300. for n in names+["OtherJoint"]}))

    def test_mimic_is_explicitly_unsupported(self):
        xml, names = robot_xml()
        xml = xml.replace('<axis xyz="0 1 0"/>', '<mimic joint="Body_Link"/><axis xyz="0 1 0"/>', 1)
        with self.assertRaisesRegex(ValueError, "mimic"):
            inspect_urdf(xml, "base", "tcp")

    def test_inspection_template_has_no_fabricated_limits(self):
        xml, _ = robot_xml()
        result = inspect_urdf(xml, "base", "tcp")
        self.assertTrue(all(v is None for v in result["allowable_efforts"].values()))
        self.assertIn("OtherJoint", result["required_state_joints"])
        self.assertIn("LHand_Link3", result["continuous_joint_names"])

    def test_diagnostics_are_standard_json(self):
        s, state = make_solver()
        _, _, r = s.solve(target_for(s,state), frame="model", snapshot=JointStateSnapshot(state))
        json.dumps(r, allow_nan=False)

    def test_zero_tool_axis_rejected(self):
        s, state = make_solver()
        q, _, r = s.solve(target_for(s,state), tool_axis=[0,0,0], snapshot=JointStateSnapshot(state))
        self.assertIsNone(q)
        self.assertIn("zero tool_axis", r["reason"])

    def test_invalid_mode_rejected(self):
        s, state = make_solver()
        q, _, r = s.solve(target_for(s,state), orientation_mode="bad", snapshot=JointStateSnapshot(state))
        self.assertIsNone(q)
        self.assertIn("orientation_mode", r["reason"])

    def test_busy_instance_refuses_concurrent_solve(self):
        s, state = make_solver()
        s._lock.acquire()
        try:
            q, status, _ = s.solve(np.eye(4), snapshot=JointStateSnapshot(state))
            self.assertIsNone(q)
            self.assertEqual(status, "BUSY")
        finally:
            s._lock.release()

    def test_physics_exception_visible(self):
        s, state = make_solver()
        with patch.object(s, "_physics", side_effect=RuntimeError("synthetic_physics_failure")):
            q, _, r = s.solve(np.eye(4), snapshot=JointStateSnapshot(state))
        self.assertIsNone(q)
        self.assertFalse(r["accepted"])
        self.assertIn("synthetic_physics_failure", r["reason"])

    def test_final_recheck_catches_fake_candidate(self):
        s, state = make_solver(policy=LoadPolicy(
            {n:.001 for n in robot_xml()[1]+["OtherJoint"]}))
        T = target_for(s,state)
        qref = s.q_reference.copy()
        candidate = s._candidate(qref, state, T, "full", np.array([1.,0,0]), .3, qref, 0)
        candidate.update(load_ok=True, peak_ratio=.1)
        with patch.object(s, "_one_seed", return_value=(candidate, {"fixed_waist":False})):
            q, status, r = s.solve(T, frame="model", snapshot=JointStateSnapshot(state))
        self.assertIsNone(q)
        self.assertEqual(status, "FINAL_RECHECK_FAILED")
        self.assertFalse(r["accepted"])

    def test_state_can_expire_during_solve(self):
        cfg = SolverConfig(state_max_age_s=.02, budget_s=2., max_seeds=3)
        s, state = make_solver(config=cfg)
        T = target_for(s,state); qref = s.q_reference.copy()
        candidate = s._candidate(qref, state, T, "full", np.array([1.,0,0]), .3, qref, 0)
        def slow(*args, **kwargs):
            time.sleep(.025)
            return candidate, {"fixed_waist":False}
        with patch.object(s, "_one_seed", side_effect=slow):
            q, status, r = s.solve(T, frame="model", snapshot=JointStateSnapshot(state))
        self.assertIsNone(q)
        self.assertEqual(status, "STALE_STATE")

    def test_time_budget_not_silent_success(self):
        cfg = SolverConfig(budget_s=1e-9, state_max_age_s=20.)
        s, state = make_solver(config=cfg)
        q, status, r = s.solve(target_for(s,state), frame="model", snapshot=JointStateSnapshot(state))
        self.assertIsNone(q)
        self.assertEqual(status, "TIME_BUDGET_EXCEEDED")

    def test_failed_solve_does_not_update_previous(self):
        s, state = make_solver()
        s.q_prev = np.ones(9)*.123
        old = s.q_prev.copy()
        T = np.eye(4); T[0,0] = 2.
        s.solve(T, snapshot=JointStateSnapshot(state))
        np.testing.assert_array_equal(s.q_prev, old)

    def test_local_base_uses_current_inactive_waist(self):
        xml, names = robot_xml()
        policy = LoadPolicy({n:300. for n in names+["OtherJoint"]})
        s = LoadAwareIKSolver(xml, base_link="link_1", ee_link="tcp", policy=policy,
                              q_reference=[.25,-.6,.4,-.8,.3,.6,-.2],
                              massless_links=["tcp"],
                              config=SolverConfig(budget_s=3., state_max_age_s=20.,
                                                  max_seeds=3, min_seeds=3, settle_iterations=3))
        _, state = make_solver()
        state["Body_Link"] = .8
        Troot = target_for(s,state)
        base = np.array(s.data.oMf[s.base_id].homogeneous, copy=True)
        Tlocal = np.linalg.inv(base) @ Troot
        q, status, r = s.solve(Tlocal, frame="local", snapshot=JointStateSnapshot(state))
        self.assertEqual(status, "SUCCESS", r)
        np.testing.assert_allclose(r["target_root"], Troot, atol=1e-12)

    def test_tool_dir_solve(self):
        s, state = make_solver()
        q, status, r = s.solve(target_for(s,state), frame="model", orientation_mode="tool_dir",
                               snapshot=JointStateSnapshot(state))
        self.assertEqual(status, "SUCCESS", r)

    def test_tool_cone_solve(self):
        s, state = make_solver()
        q, status, r = s.solve(target_for(s,state), frame="model", orientation_mode="tool_cone",
                               snapshot=JointStateSnapshot(state))
        self.assertEqual(status, "SUCCESS", r)


if __name__ == "__main__":
    unittest.main()