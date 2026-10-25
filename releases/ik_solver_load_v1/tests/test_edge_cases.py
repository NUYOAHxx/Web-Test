from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import math
import unittest
import numpy as np

from ha_arm_ik.load_policy import JointStateSnapshot, LoadPolicy
from ha_arm_ik.solver_load_aware import LoadAwareIKSolver, SolverConfig, inspect_urdf
from tests.fixtures import inertial, make_solver, target_for, robot_xml, planar_xml


class TestEdgeCases(unittest.TestCase):
    def test_antipodal_direction_escapes_and_converges(self):
        xml = ('<robot name="rotation"><link name="base">'+inertial(2)+'</link>'
               '<link name="tip">'+inertial(1,(.1,0,0))+'</link>'
               '<joint name="j" type="revolute"><parent link="base"/><child link="tip"/>'
               '<axis xyz="0 0 1"/><limit lower="-4" upper="4" effort="10" velocity="1"/></joint></robot>')
        s = LoadAwareIKSolver(xml, base_link="base", ee_link="tip", q_reference=[0.],
                              policy=LoadPolicy({"j":10.}),
                              config=SolverConfig(max_seeds=1,min_seeds=1,budget_s=2.,state_max_age_s=5.))
        T = np.eye(4); T[:3,:3] = np.diag([-1.,-1.,1.])
        q, status, r = s.solve(T,frame="model",orientation_mode="tool_dir",
                               snapshot=JointStateSnapshot({"j":0.}))
        self.assertEqual(status,"SUCCESS",r)
        self.assertTrue(r["per_seed"][0]["antipodal_escape_used"])
        self.assertLess(r["selected"]["orientation_violation_rad"],s.config.ori_tol_rad)

    def test_prismatic_effort_units_and_task(self):
        xml = ('<robot name="slider"><link name="base">'+inertial(2)+'</link>'
               '<link name="tip">'+inertial(1)+'</link>'
               '<joint name="j" type="prismatic"><parent link="base"/><child link="tip"/>'
               '<axis xyz="1 0 0"/><limit lower="-1" upper="1" effort="10" velocity="1"/></joint></robot>')
        s = LoadAwareIKSolver(xml, base_link="base", ee_link="tip",q_reference=[0.],
                              gravity_root=[-9.81,0,0], policy=LoadPolicy({"j":20.}),
                              config=SolverConfig(max_seeds=1,min_seeds=1,budget_s=2.,state_max_age_s=5.))
        T = np.eye(4); T[0,3]=.2
        q,status,r = s.solve(T, frame="model",snapshot=JointStateSnapshot({"j":0.}))
        self.assertEqual(status,"SUCCESS",r)
        self.assertAlmostEqual(r["selected"]["gravity_efforts"]["j"],9.81,places=8)
        self.assertEqual(r["effort_units"]["j"],"N")
        self.assertLess(abs(q[0]-.2),s.config.pos_tol_m)

    def test_static_absolute_objective_improves_synthetic_case(self):
        # A deterministic regression fixture, NOT a user-robot performance claim.
        cfg=SolverConfig(max_seeds=1,min_seeds=1,max_iterations=250,
                         settle_iterations=120,budget_s=3.,state_max_age_s=10.,
                         posture_weight=.001,com_weight=0.,limit_weight=0.,rotation_step_rad=.05)
        ref=[.1,-1.,1.4]
        rows=[]
        for enabled in (False,True):
            policy=LoadPolicy({f"j{i}":100. for i in range(3)},
                              absolute_weight=5. if enabled else 0.,
                              excess_weight=5. if enabled else 0.)
            s=LoadAwareIKSolver(planar_xml(),base_link="base",ee_link="tcp",
                                q_reference=ref,policy=policy,config=cfg,
                                massless_links=["tcp"],waist_joint_names=())
            state=dict(zip(s.active_joint_names,ref))
            T=target_for(s,state)
            q,status,r=s.solve(T,frame="model",orientation_mode="tool_cone",
                               tool_cone_half_angle=math.pi,snapshot=JointStateSnapshot(state))
            self.assertEqual(status,"SUCCESS",r)
            rows.append(r["selected"])
        self.assertLess(rows[1]["peak_ratio"], rows[0]["peak_ratio"]-.01)
        self.assertLess(rows[1]["mean_square_ratio"], rows[0]["mean_square_ratio"])
        self.assertLessEqual(rows[1]["position_error_m"],cfg.pos_tol_m)

    def test_existing_solver_adapter_uses_fresh_model_and_preserves_old(self):
        original,state=make_solver()
        with TemporaryDirectory() as tmp:
            path=Path(tmp)/"fixture.urdf";path.write_text(original.xml)
            old=SimpleNamespace(urdf=path,base_link=original.base_link,ee_link=original.ee_link,
                                pin_model=original.model,q_ready=original.q_reference.copy(),
                                active_joint_names=original.active_joint_names)
            new=LoadAwareIKSolver.from_existing_solver(old,policy=original.policy,
                                                       massless_links=["tcp"])
            self.assertIsNot(new.model,old.pin_model)
            np.testing.assert_array_equal(new.q_reference,old.q_ready)
            new.model.gravity.linear=np.array([0.,-9.81,0.])
            np.testing.assert_array_equal(old.pin_model.gravity.linear,[0.,0.,-9.81])

    def test_adapter_checks_joint_order(self):
        original,state=make_solver()
        with TemporaryDirectory() as tmp:
            path=Path(tmp)/"fixture.urdf";path.write_text(original.xml)
            old=SimpleNamespace(urdf=path,base_link=original.base_link,ee_link=original.ee_link,
                                pin_model=original.model,q_ready=original.q_reference,
                                active_joint_names=list(reversed(original.active_joint_names)))
            with self.assertRaisesRegex(ValueError,"order mismatch"):
                LoadAwareIKSolver.from_existing_solver(old,policy=original.policy,massless_links=["tcp"])

    def test_zero_axis_fails_before_model_build(self):
        xml,_=robot_xml()
        xml=xml.replace('<axis xyz="0 1 0"/>','<axis xyz="0 0 0"/>',1)
        with self.assertRaisesRegex(ValueError,"zero joint axis"):
            inspect_urdf(xml,"base","tcp")

    def test_floating_joint_rejected(self):
        xml,_=robot_xml()
        xml=xml.replace('type="revolute"','type="floating"',1)
        with self.assertRaisesRegex(ValueError,"unsupported joint type"):
            inspect_urdf(xml,"base","tcp")

    def test_source_stamp_required_when_configured(self):
        s,state=make_solver(config=SolverConfig(require_source_age=True,state_max_age_s=5.))
        q,status,r=s.solve(target_for(s,state),frame="model",snapshot=JointStateSnapshot(state))
        self.assertIsNone(q)
        self.assertEqual(status,"STALE_STATE")
        self.assertEqual(r["reason"],"source_timestamp_unverified")

    def test_budget_preserves_model_valid_best_candidate(self):
        # A small search budget need not discard an already validated candidate.
        s,state=make_solver()
        s.config=replace(s.config,budget_s=.02)
        q,status,r=s.solve(target_for(s,state),frame="model",snapshot=JointStateSnapshot(state))
        if q is not None:
            self.assertTrue(r["accepted"])
            self.assertLessEqual(r["selected"]["peak_ratio"],1.)
        else:
            self.assertFalse(r["accepted"])
            self.assertIn(status,["TIME_BUDGET_EXCEEDED","STALE_STATE"])

    def test_bounded_large_range_joint_has_no_virtual_150_degree_window(self):
        xml,names=robot_xml(continuous=False)
        xml=xml.replace('lower="-3" upper="3"','lower="-6" upper="6"')
        ref=np.array([.05,.1,.25,-.6,3.8,-.8,.3,.6,-.2])
        s=LoadAwareIKSolver(xml,base_link="base",ee_link="tcp",q_reference=ref,
                            policy=LoadPolicy({n:300. for n in names+["OtherJoint"]}),
                            massless_links=["tcp"])
        self.assertFalse(s.continuous[4])
        self.assertEqual(s.lower[4],-6.)
        self.assertEqual(s.upper[4],6.)
        self.assertEqual(s.q_reference[4],3.8)

    def test_continuous_solve_keeps_nearby_unwrapped_branch(self):
        s,state=make_solver()
        state["LHand_Link3"]=3.8
        q,status,r=s.solve(target_for(s,state),frame="model",snapshot=JointStateSnapshot(state))
        self.assertEqual(status,"SUCCESS",r)
        self.assertLess(abs(q[4]-3.8),np.pi)

    def test_reference_pose_outside_bounded_joint_limit_is_not_clipped(self):
        with self.assertRaisesRegex(ValueError,"q_reference.*not clipped"):
            make_solver(position_bounds={"Hip_Link1":[.1,.2]})


if __name__ == "__main__":
    unittest.main()