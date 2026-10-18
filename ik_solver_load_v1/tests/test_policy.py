import math
import time
import unittest
from types import SimpleNamespace
import numpy as np

from ha_arm_ik.load_policy import (
    JointStateSnapshot, LoadPolicy, joint_difference, json_safe, metric_step,
    select_load_candidate, validate_transform,
)


class TestPolicy(unittest.TestCase):
    def setUp(self):
        self.names = ["a", "b"]
        self.policy = LoadPolicy({"a": 10., "b": 20.}, absolute_weight=.3,
                                 excess_weight=2., joint_weights={"a": 2.})

    def test_reject_empty_capacities(self):
        with self.assertRaises(ValueError):
            LoadPolicy({})

    def test_reject_zero_capacity(self):
        with self.assertRaises(ValueError):
            LoadPolicy({"a": 0.})

    def test_reject_nonfinite_capacity(self):
        with self.assertRaises(ValueError):
            LoadPolicy({"a": np.inf})

    def test_reject_negative_capacity(self):
        with self.assertRaises(ValueError):
            LoadPolicy({"a": -2.})

    def test_missing_capacity(self):
        with self.assertRaises(ValueError):
            self.policy.vectors(["missing"])

    def test_capacities_immutable(self):
        with self.assertRaises(TypeError):
            self.policy.allowable_efforts["a"] = 1.

    def test_overload_is_per_joint_not_average(self):
        p = LoadPolicy({"a": 10., "b": 1000.})
        result = p.evaluate([10.01, 0.], self.names)
        self.assertFalse(result["load_ok"])
        self.assertEqual(result["overloaded_joints"], ["a"])

    def test_exact_limit_accepted(self):
        self.assertTrue(self.policy.evaluate([10., -20.], self.names)["load_ok"])

    def test_invalid_tau_never_zero_fallback(self):
        with self.assertRaises(ValueError):
            self.policy.evaluate([np.nan, 0.], self.names)

    def test_peak_joint(self):
        r = self.policy.evaluate([1., -18.], self.names)
        self.assertEqual(r["peak_joint"], "b")
        self.assertAlmostEqual(r["peak_ratio"], .9)

    def test_absolute_gradient_finite_difference(self):
        J = np.array([[2., -1., 3.], [.2, .3, -.6]])
        tau = np.array([8., -18.])
        _, analytic, _ = self.policy.cost_gradient(tau, J, self.names)
        numerical = []
        for j in range(3):
            eps = 1e-6
            plus = self.policy.cost_gradient(tau+eps*J[:, j], J, self.names)[0]
            minus = self.policy.cost_gradient(tau-eps*J[:, j], J, self.names)[0]
            numerical.append((plus-minus)/(2*eps))
        np.testing.assert_allclose(analytic, numerical, atol=1e-9)

    def test_relative_gradient_finite_difference(self):
        p = LoadPolicy({"a": 10., "b": 20.}, absolute_weight=0., excess_weight=0.,
                       relative_weight=.2)
        J = np.eye(2)
        tau, ref = np.array([6., -8.]), np.array([4., 3.])
        _, grad, _ = p.cost_gradient(tau, J, self.names, tau_reference=ref)
        np.testing.assert_allclose(grad, .2*(tau-ref)/np.array([10.,20.])**2/2)

    def test_soft_penalty_off_below_threshold(self):
        _, _, c = self.policy.cost_gradient([1.,2.], np.eye(2), self.names)
        self.assertEqual(c["excess"], 0.)

    def test_soft_penalty_on_above_threshold(self):
        _, _, c = self.policy.cost_gradient([9.,18.], np.eye(2), self.names)
        self.assertGreater(c["excess"], 0.)

    def test_configured_peak_preference(self):
        def c(i, peak, smooth, safe=True):
            return dict(candidate_index=i, peak_ratio=peak, continuity_cost=smooth,
                        reference_cost=0., mean_square_ratio=.1,
                        pose_ok=True, bounds_ok=True, load_ok=safe)
        candidates = [c(0, 1.01, 0., False), c(1, .8, .01), c(2, .5, 10.)]
        self.assertEqual(select_load_candidate(candidates, self.policy)["candidate_index"], 2)

    def test_near_equal_load_prefers_continuity(self):
        p = LoadPolicy({"a": 10.}, peak_tie=.02)
        a = dict(candidate_index=0, peak_ratio=.5, continuity_cost=1.,
                 reference_cost=0., mean_square_ratio=.1, pose_ok=True, bounds_ok=True, load_ok=True)
        b = dict(a, candidate_index=1, peak_ratio=.51, continuity_cost=0.)
        self.assertIs(select_load_candidate([a,b], p), b)

    def test_no_safe_candidate(self):
        self.assertIsNone(select_load_candidate([], self.policy))


class TestProjection(unittest.TestCase):
    def test_metric_counterexample_fixed(self):
        J = np.array([[1.,1.]])
        task, null, rank = metric_step(J, [0.], [-1.,-1.], [1.,.25], damping=0)
        np.testing.assert_allclose(task, [0.,0.], atol=1e-12)
        np.testing.assert_allclose(null, [0.,0.], atol=1e-12)
        self.assertEqual(rank, 1)

    def test_nullspace_no_task_leakage(self):
        rng = np.random.default_rng(3)
        J = rng.normal(size=(6,9))
        gradient = rng.normal(size=9)
        _, null, _ = metric_step(J, np.zeros(6), gradient, np.linspace(.1,2.,9))
        np.testing.assert_allclose(J@null, 0., atol=1e-12)
        self.assertLessEqual(gradient@null, 1e-12)

    def test_full_rank_no_nullspace(self):
        _, null, _ = metric_step(np.eye(3), [0.,0.,0.], [1.,2.,3.], [1.,1.,1.])
        np.testing.assert_allclose(null, 0., atol=1e-12)

    def test_task_step_sign(self):
        task, _, _ = metric_step(np.eye(2), [1.,-2.], [0.,0.], [1.,1.], damping=0.)
        np.testing.assert_allclose(task, [1.,-2.])

    def test_bad_metric(self):
        with self.assertRaises(ValueError):
            metric_step(np.eye(2), [0.,0.], [0.,0.], [0.,1.])

    def test_continuous_difference_only(self):
        d = joint_difference([3.5,3.5], [-3.,-3.], [True,False])
        self.assertLess(abs(d[0]), .3)
        self.assertEqual(d[1], 6.5)


class TestSnapshot(unittest.TestCase):
    def test_snapshot_copies_input(self):
        values = {"a": 1.}
        s = JointStateSnapshot(values)
        values["a"] = 3.
        self.assertEqual(s.positions["a"], 1.)

    def test_snapshot_immutable(self):
        s = JointStateSnapshot({"a": 1.})
        with self.assertRaises(TypeError):
            s.positions["a"] = 0.

    def test_old_receipt(self):
        s = JointStateSnapshot({"a": 1.}, received_monotonic=10.)
        self.assertEqual(s.freshness_error(.5, now=11.), "receipt_state_stale")

    def test_old_source(self):
        s = JointStateSnapshot({"a": 1.}, received_monotonic=10., source_age_at_receipt_s=2.)
        self.assertEqual(s.freshness_error(.5, now=10.), "source_state_stale")

    def test_future_source(self):
        s = JointStateSnapshot({"a": 1.}, received_monotonic=10., source_age_at_receipt_s=-1.)
        self.assertEqual(s.freshness_error(.5, now=10.), "source_timestamp_in_future")

    def test_source_verification_required(self):
        s = JointStateSnapshot({"a": 1.}, received_monotonic=10.)
        self.assertEqual(s.freshness_error(.5, require_source_age=True, now=10.),
                         "source_timestamp_unverified")

    def test_ros_message_capture(self):
        msg = SimpleNamespace(name=["a"], position=[.2],
                              header=SimpleNamespace(stamp=SimpleNamespace(sec=2,nanosec=0)))
        s = JointStateSnapshot.from_joint_state(msg, now_ros_ns=2_100_000_000)
        self.assertAlmostEqual(s.source_age_at_receipt_s, .1)

    def test_duplicate_ros_joint_names_rejected(self):
        msg = SimpleNamespace(name=["a","a"], position=[.2,.3])
        with self.assertRaises(ValueError):
            JointStateSnapshot.from_joint_state(msg)

    def test_transform_validation(self):
        np.testing.assert_array_equal(validate_transform(np.eye(4)), np.eye(4))

    def test_reflection_rejected(self):
        T = np.eye(4); T[0,0] = -1.
        with self.assertRaises(ValueError):
            validate_transform(T)

    def test_json_nonfinite_is_explicit_null(self):
        self.assertEqual(json_safe({"a": np.inf, "b": np.array([np.nan])}), {"a": None, "b": [None]})


if __name__ == "__main__":
    unittest.main()