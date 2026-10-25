import unittest
import numpy as np
from ha_arm_ik.load_policy import LoadPolicy, JointStateSnapshot
from ha_arm_ik.solver_load_aware import LoadAwareIKSolver
from tests.fixtures import make_solver, target_for, robot_xml, inertial


class TestPinocchioMath(unittest.TestCase):
    def setUp(self):
        self.s, self.state = make_solver()
        self.q = np.array([self.state[n] for n in self.s.active_joint_names])

    def test_continuous_joint_nq2_mapping(self):
        q = self.q.copy(); q[4] = 3.8
        full = self.s._full_q(q, self.state)
        info = self.s.joints["LHand_Link3"]
        self.assertEqual(info.nq, 2)
        np.testing.assert_allclose(full[info.idx_q:info.idx_q+2], [np.cos(3.8), np.sin(3.8)])
        self.assertTrue(np.isneginf(self.s.lower[4]))
        self.assertTrue(np.isposinf(self.s.upper[4]))

    def test_inactive_joint_not_neutral(self):
        full = self.s._full_q(self.q, self.state)
        self.assertEqual(full[self.s.joints["OtherJoint"].idx_q], .4)

    def test_inactive_joint_changes_physics(self):
        a = self.s._physics(self.q, self.state, derivatives=False)
        state = dict(self.state, OtherJoint=1.4)
        b = self.s._physics(self.q, state, derivatives=False)
        self.assertGreater(np.linalg.norm(a["com_root_m"]-b["com_root_m"]), .001)
        self.assertGreater(np.linalg.norm(a["tau"]-b["tau"]), .01)

    def test_gravity_derivative_matches_finite_difference(self):
        actual = self.s._physics(self.q, self.state, derivatives=True)["d_tau_d_active"]
        numeric = np.zeros_like(actual)
        for j in range(self.q.size):
            dq = np.eye(self.q.size)[j]*1e-6
            plus = self.s._physics(self.q+dq, self.state, derivatives=False)["tau"]
            minus = self.s._physics(self.q-dq, self.state, derivatives=False)["tau"]
            numeric[:,j] = (plus-minus)/2e-6
        np.testing.assert_allclose(actual, numeric, atol=2e-7, rtol=2e-6)

    def test_com_jacobian_matches_finite_difference(self):
        actual = self.s._physics(self.q, self.state, derivatives=True)["Jcom"]
        numeric = np.zeros_like(actual)
        for j in range(self.q.size):
            dq = np.eye(self.q.size)[j]*1e-6
            plus = self.s._physics(self.q+dq, self.state, derivatives=False)["com_root_m"]
            minus = self.s._physics(self.q-dq, self.state, derivatives=False)["com_root_m"]
            numeric[:,j] = (plus-minus)/2e-6
        np.testing.assert_allclose(actual, numeric, atol=2e-8, rtol=1e-5)

    def test_gravity_is_potential_gradient_including_root_mass(self):
        physics = self.s._physics(self.q, self.state, derivatives=False)
        tau = physics["tau"][:self.q.size]
        numeric = []
        for j in range(self.q.size):
            dq = np.eye(self.q.size)[j]*1e-6
            def potential(q):
                p = self.s._physics(q, self.state, derivatives=False)
                return -self.s.total_mass * self.s.gravity_root @ p["com_root_m"]
            numeric.append((potential(self.q+dq)-potential(self.q-dq))/2e-6)
        np.testing.assert_allclose(tau, numeric, atol=5e-7, rtol=1e-5)

    def test_full_log_residual_jacobian(self):
        target = target_for(self.s, self.state, self.q + np.linspace(-.2,.3,self.q.size))
        actual = self.s._task(self.q, self.state, target, "full", np.array([1.,0,0]), .2)
        numeric = np.zeros_like(actual["J"])
        for j in range(self.q.size):
            dq = np.eye(self.q.size)[j]*1e-7
            plus = self.s._task(self.q+dq, self.state, target, "full",
                                np.array([1.,0,0]), .2, jacobian=False)["error"]
            minus = self.s._task(self.q-dq, self.state, target, "full",
                                 np.array([1.,0,0]), .2, jacobian=False)["error"]
            numeric[:,j] = -(plus-minus)/2e-7
        np.testing.assert_allclose(actual["J"], numeric, atol=2e-7, rtol=1e-5)

    def test_tool_direction_jacobian(self):
        target = target_for(self.s, self.state, self.q + .2)
        actual = self.s._task(self.q, self.state, target, "tool_dir", np.array([1.,0,0]), .2)
        numeric = np.zeros_like(actual["J"])
        for j in range(self.q.size):
            dq = np.eye(self.q.size)[j]*1e-7
            plus = self.s._task(self.q+dq, self.state, target, "tool_dir",
                                np.array([1.,0,0]), .2, jacobian=False)["error"]
            minus = self.s._task(self.q-dq, self.state, target, "tool_dir",
                                 np.array([1.,0,0]), .2, jacobian=False)["error"]
            numeric[:,j] = -(plus-minus)/2e-7
        np.testing.assert_allclose(actual["J"], numeric, atol=2e-7, rtol=1e-5)

    def test_outside_cone_jacobian(self):
        target = target_for(self.s, self.state, self.q + .3)
        axis, cone = np.array([1.,0,0]), .01
        actual = self.s._task(self.q, self.state, target, "tool_cone", axis, cone)
        self.assertGreater(actual["orientation_violation_rad"], .05)
        numeric = np.zeros_like(actual["J"])
        for j in range(self.q.size):
            dq = np.eye(self.q.size)[j]*1e-7
            plus = self.s._task(self.q+dq, self.state, target, "tool_cone",
                                axis, cone, jacobian=False)["error"]
            minus = self.s._task(self.q-dq, self.state, target, "tool_cone",
                                 axis, cone, jacobian=False)["error"]
            numeric[:,j] = -(plus-minus)/2e-7
        np.testing.assert_allclose(actual["J"], numeric, atol=3e-7, rtol=1e-5)

    def test_inside_cone_has_no_orientation_row(self):
        target = target_for(self.s, self.state)
        task = self.s._task(self.q, self.state, target, "tool_cone", np.array([1.,0,0]), .5)
        self.assertEqual(task["orientation_violation_rad"], 0.)
        np.testing.assert_array_equal(task["J"][-1], 0.)

    def test_entire_secondary_gradient_finite_difference(self):
        ref = self.s._physics(self.s.q_reference, self.state, derivatives=False)
        q = self.q + np.linspace(-.05,.05,self.q.size)
        secondary = self.s._secondary(q, self.state, ref, derivatives=True)
        numeric = []
        for j in range(q.size):
            dq = np.eye(q.size)[j]*1e-6
            p = self.s._secondary(q+dq, self.state, ref, derivatives=False)["cost"]
            m = self.s._secondary(q-dq, self.state, ref, derivatives=False)["cost"]
            numeric.append((p-m)/2e-6)
        np.testing.assert_allclose(secondary["gradient"], numeric, atol=1e-8, rtol=1e-5)

    def test_root_mass_in_com(self):
        xml = ('<robot name="root_mass"><link name="base">'+inertial(10,(1,0,0))+'</link>'
               '<link name="tip">'+inertial(2,(.5,0,0))+'</link>'
               '<joint name="j" type="revolute"><parent link="base"/><child link="tip"/>'
               '<axis xyz="0 1 0"/><limit lower="-2" upper="2" effort="10" velocity="1"/></joint></robot>')
        s = LoadAwareIKSolver(xml, base_link="base", ee_link="tip",
                              policy=LoadPolicy({"j": 100.}), q_reference=[0.])
        p = s._physics(np.array([0.]), {"j":0.}, derivatives=True)
        self.assertAlmostEqual(s.total_mass, 12.)
        self.assertAlmostEqual(p["com_root_m"][0], 11/12)

    def test_model_effort_not_capacity(self):
        self.assertEqual(self.s.joints["OtherJoint"].urdf_effort, 100.)
        self.assertEqual(self.s.policy.allowable_efforts["OtherJoint"], 300.)

    def test_continuous_reference_outside_pi_is_not_clipped(self):
        q = self.q.copy(); q[4] = -3.865597
        xml, names = robot_xml()
        policy = LoadPolicy({n:300. for n in names+["OtherJoint"]})
        s = LoadAwareIKSolver(xml, base_link="base", ee_link="tcp", q_reference=q,
                              policy=policy, massless_links=["tcp"])
        self.assertEqual(s.q_reference[4], -3.865597)


if __name__ == "__main__":
    unittest.main()