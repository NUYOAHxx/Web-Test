import math
import time
import unittest
from types import SimpleNamespace
import numpy as np
from tests.stubs import install
install()
from ha_arm_ik.solver_dls import JointStateSnapshot, LoadPolicy, joint_difference, json_safe, validate_transform

class TestPolicy(unittest.TestCase):

    def setUp(self):
        self.names = ['a', 'b']
        self.policy = LoadPolicy({'a': 10.0, 'b': 20.0}, absolute_weight=0.3, excess_weight=2.0, joint_weights={'a': 2.0})

    def test_reject_empty_capacities(self):
        with self.assertRaises(ValueError):
            LoadPolicy({})

    def test_reject_zero_capacity(self):
        with self.assertRaises(ValueError):
            LoadPolicy({'a': 0.0})

    def test_reject_nonfinite_capacity(self):
        with self.assertRaises(ValueError):
            LoadPolicy({'a': np.inf})

    def test_reject_negative_capacity(self):
        with self.assertRaises(ValueError):
            LoadPolicy({'a': -2.0})

    def test_missing_capacity(self):
        with self.assertRaises(ValueError):
            self.policy.vectors(['missing'])

    def test_capacities_immutable(self):
        with self.assertRaises(TypeError):
            self.policy.allowable_efforts['a'] = 1.0

    def test_overload_is_per_joint_not_average(self):
        p = LoadPolicy({'a': 10.0, 'b': 1000.0})
        result = p.evaluate([10.01, 0.0], self.names)
        self.assertFalse(result['load_ok'])
        self.assertEqual(result['overloaded_joints'], ['a'])

    def test_exact_limit_accepted(self):
        self.assertTrue(self.policy.evaluate([10.0, -20.0], self.names)['load_ok'])

    def test_invalid_tau_never_zero_fallback(self):
        with self.assertRaises(ValueError):
            self.policy.evaluate([np.nan, 0.0], self.names)

    def test_peak_joint(self):
        r = self.policy.evaluate([1.0, -18.0], self.names)
        self.assertEqual(r['peak_joint'], 'b')
        self.assertAlmostEqual(r['peak_ratio'], 0.9)

    def test_absolute_gradient_finite_difference(self):
        J = np.array([[2.0, -1.0, 3.0], [0.2, 0.3, -0.6]])
        tau = np.array([8.0, -18.0])
        _, analytic, _ = self.policy.cost_gradient(tau, J, self.names)
        numerical = []
        for j in range(3):
            eps = 1e-06
            plus = self.policy.cost_gradient(tau + eps * J[:, j], J, self.names)[0]
            minus = self.policy.cost_gradient(tau - eps * J[:, j], J, self.names)[0]
            numerical.append((plus - minus) / (2 * eps))
        np.testing.assert_allclose(analytic, numerical, atol=1e-09)

    def test_relative_gradient_finite_difference(self):
        p = LoadPolicy({'a': 10.0, 'b': 20.0}, absolute_weight=0.0, excess_weight=0.0, relative_weight=0.2)
        J = np.eye(2)
        tau, ref = (np.array([6.0, -8.0]), np.array([4.0, 3.0]))
        _, grad, _ = p.cost_gradient(tau, J, self.names, tau_reference=ref)
        np.testing.assert_allclose(grad, 0.2 * (tau - ref) / np.array([10.0, 20.0]) ** 2 / 2)

    def test_soft_penalty_off_below_threshold(self):
        _, _, c = self.policy.cost_gradient([1.0, 2.0], np.eye(2), self.names)
        self.assertEqual(c['excess'], 0.0)

    def test_soft_penalty_on_above_threshold(self):
        _, _, c = self.policy.cost_gradient([9.0, 18.0], np.eye(2), self.names)
        self.assertGreater(c['excess'], 0.0)

class TestSnapshot(unittest.TestCase):

    def test_snapshot_copies_input(self):
        values = {'a': 1.0}
        s = JointStateSnapshot(values)
        values['a'] = 3.0
        self.assertEqual(s.positions['a'], 1.0)

    def test_snapshot_immutable(self):
        s = JointStateSnapshot({'a': 1.0})
        with self.assertRaises(TypeError):
            s.positions['a'] = 0.0

    def test_old_receipt(self):
        s = JointStateSnapshot({'a': 1.0}, received_monotonic=10.0)
        self.assertEqual(s.freshness_error(0.5, now=11.0), 'receipt_state_stale')

    def test_old_source(self):
        s = JointStateSnapshot({'a': 1.0}, received_monotonic=10.0, source_age_at_receipt_s=2.0)
        self.assertEqual(s.freshness_error(0.5, now=10.0), 'source_state_stale')

    def test_future_source(self):
        s = JointStateSnapshot({'a': 1.0}, received_monotonic=10.0, source_age_at_receipt_s=-1.0)
        self.assertEqual(s.freshness_error(0.5, now=10.0), 'source_timestamp_in_future')

    def test_source_verification_required(self):
        s = JointStateSnapshot({'a': 1.0}, received_monotonic=10.0)
        self.assertEqual(s.freshness_error(0.5, require_source_age=True, now=10.0), 'source_timestamp_unverified')

    def test_ros_message_capture(self):
        msg = SimpleNamespace(name=['a'], position=[0.2], header=SimpleNamespace(stamp=SimpleNamespace(sec=2, nanosec=0)))
        s = JointStateSnapshot.from_joint_state(msg, now_ros_ns=2100000000)
        self.assertAlmostEqual(s.source_age_at_receipt_s, 0.1)

    def test_duplicate_ros_joint_names_rejected(self):
        msg = SimpleNamespace(name=['a', 'a'], position=[0.2, 0.3])
        with self.assertRaises(ValueError):
            JointStateSnapshot.from_joint_state(msg)

    def test_transform_validation(self):
        np.testing.assert_array_equal(validate_transform(np.eye(4)), np.eye(4))

    def test_reflection_rejected(self):
        T = np.eye(4)
        T[0, 0] = -1.0
        with self.assertRaises(ValueError):
            validate_transform(T)

    def test_json_nonfinite_is_explicit_null(self):
        self.assertEqual(json_safe({'a': np.inf, 'b': np.array([np.nan])}), {'a': None, 'b': [None]})
