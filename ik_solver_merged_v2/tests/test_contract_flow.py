"""Legacy orchestration tests with explicitly substituted unavailable modules."""
import ast
import inspect
from pathlib import Path
import time
import unittest
from unittest.mock import patch
import numpy as np
from tests.stubs import install, MAKE_RESULT_CALLS
install()
from tests.fixtures import make_solver,root_target,context
from ha_arm_ik.solver_dls import JointStateSnapshot
from ha_arm_ik import solver as module


class LegacyContractTests(unittest.TestCase):
    def test_original_constructor_signature(self):
        self.assertEqual(list(inspect.signature(module.HAArmIKSolver.__init__).parameters),
                         ["self","urdf","side","use_ssik","use_pinocchio_ik","use_torque_cost",
                          "use_direct_psi","include_waist","verbose","config"])

    def test_original_solve_positionals_and_one_keyword_extension(self):
        sig=inspect.signature(module.HAArmIKSolver.solve)
        names=[p.name for p in sig.parameters.values() if p.kind!=p.KEYWORD_ONLY]
        self.assertEqual(names,["self","T_target","frame","q_prev","update_prev","verbose",
                                "allow_approximate","orientation_mode","tool_axis","tool_cone_half_angle"])
        self.assertEqual(sig.parameters["joint_state"].kind,inspect.Parameter.KEYWORD_ONLY)

    def test_explicit_root_gravity_configuration(self):
        s,_=make_solver(config={"gravity_root":[0.,-9.81,0.]})
        np.testing.assert_array_equal(s.pin_model.gravity.linear,[0.,-9.81,0.])
        with self.assertRaisesRegex(ValueError,"gravity_root"):
            make_solver(config={"gravity_root":[0.,0.,0.]})

    def test_only_original_solver_class_is_used(self):
        tree=ast.parse(Path(module.__file__).read_text())
        self.assertEqual([n.name for n in tree.body if isinstance(n,ast.ClassDef)],
                         ["DirectPsiProblem","HAArmIKSolver"])
        self.assertNotIn("solver_load_aware",Path(module.__file__).read_text())

    def test_original_functions_and_methods_retained(self):
        root=Path(__file__).parents[1]
        old=ast.parse((root/"baseline/solver.py").read_text())
        new=ast.parse(Path(module.__file__).read_text())
        def names(tree):
            return {n.name for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        self.assertTrue(names(old)<=names(new))

    def test_optional_direct_psi_source_functions_not_replaced(self):
        root=Path(__file__).parents[1]
        old=ast.parse((root/"baseline/solver.py").read_text())
        new=ast.parse(Path(module.__file__).read_text())
        for name in ("solve_direct_psi","collect_ssik_candidates_and_seeds","DirectPsiProblem"):
            a=next(n for n in old.body if getattr(n,"name",None)==name)
            b=next(n for n in new.body if getattr(n,"name",None)==name)
            self.assertEqual(ast.dump(a,include_attributes=False),ast.dump(b,include_attributes=False))

    def test_legacy_result_factory_and_type_retained(self):
        s,state=make_solver()
        q,status,result=s.solve(root_target(s,state),frame="world",joint_state=JointStateSnapshot(state))
        self.assertIsNotNone(q,s.last_load_diagnostics)
        self.assertEqual(status,"VALID")
        self.assertIsInstance(result,module.IKResult)
        self.assertFalse(isinstance(result,dict))
        self.assertTrue(s.last_load_diagnostics["accepted"])
        self.assertFalse(any(k.startswith("load_") for k in MAKE_RESULT_CALLS[-1][0]))

    def test_update_joint_state_keeps_original_solve_call(self):
        s,state=make_solver()
        T=root_target(s,state)
        s.update_joint_state(JointStateSnapshot(state))
        q,status,result=s.solve(T,frame="world")
        self.assertIsNotNone(q,s.last_load_diagnostics)

    def test_missing_allowable_limits_fail_closed(self):
        s,state=make_solver(config={"load_allowable_efforts":{}})
        q,status,result=s.solve(root_target(s,state),frame="world",joint_state=JointStateSnapshot(state))
        self.assertIsNone(q)
        self.assertEqual(status,"NO_VALID_POSE")
        self.assertIsInstance(result,module.IKResult)
        self.assertEqual(s.last_load_diagnostics["code"],"LOAD_CONFIG_OR_STATE_INVALID")
        self.assertIn("allowable",result.failure_detail)

    def test_missing_snapshot_fail_closed(self):
        s,state=make_solver()
        q,status,result=s.solve(root_target(s,state),frame="world")
        self.assertIsNone(q)
        self.assertIn("Snapshot",result.failure_detail)

    def test_partial_full_state_not_filled_neutral(self):
        s,state=make_solver()
        T=root_target(s,state);del state["OtherJoint"]
        q,status,result=s.solve(T,frame="world",joint_state=JointStateSnapshot(state))
        self.assertIsNone(q)
        self.assertIn("OtherJoint",result.failure_detail)

    def test_old_snapshot_fail_closed(self):
        s,state=make_solver()
        q,_,result=s.solve(root_target(s,state),frame="world",
                           joint_state=JointStateSnapshot(state,time.monotonic()-100))
        self.assertIsNone(q)
        self.assertIn("stale",result.failure_detail)

    def test_source_age_can_be_required(self):
        s,state=make_solver(config={"load_require_source_age":True})
        q,_,result=s.solve(root_target(s,state),frame="world",joint_state=JointStateSnapshot(state))
        self.assertIsNone(q)
        self.assertIn("source_timestamp_unverified",result.failure_detail)

    def test_overload_no_approximate_bypass(self):
        names=["Hip_Link1","Body_Link"]+[f"LHand_Link{i}" for i in range(1,8)]+["OtherJoint"]
        s,state=make_solver(config={"load_allowable_efforts":{n:.001 for n in names},"dls_max_iter":1})
        old=s.q_prev.copy()
        q,status,result=s.solve(root_target(s,state),frame="world",allow_approximate=True,
                                joint_state=JointStateSnapshot(state))
        self.assertIsNone(q)
        self.assertEqual(status,"NO_VALID_POSE")
        self.assertEqual(s.last_load_diagnostics["code"],"NO_LOAD_SAFE_SOLUTION")
        self.assertIn("NO_LOAD_SAFE_SOLUTION",result.failure_detail)
        np.testing.assert_array_equal(s.q_prev,old)

    def test_unlock_comparison_before_low_load_early_stop(self):
        s,state=make_solver()
        q,_,_=s.solve(root_target(s,state),frame="world",joint_state=JointStateSnapshot(state))
        self.assertIsNotNone(q,s.last_load_diagnostics)
        self.assertGreaterEqual(s.last_load_diagnostics["search"]["unlocked_seeds_attempted"],2)

    def test_original_status_helper_controls_valid_status(self):
        s,state=make_solver()
        with patch.object(module,"is_valid_status",return_value=False):
            q,_,_=s.solve(root_target(s,state),frame="world",joint_state=JointStateSnapshot(state))
        self.assertIsNone(q)
        self.assertFalse(s.last_load_diagnostics["accepted"])

    def test_legacy_selector_receives_only_safe_shortlist(self):
        s,state=make_solver()
        original=module.select_best_candidate
        seen=[]
        def spy(candidates,**kw):
            seen.extend(candidates)
            self.assertTrue(all(c["load_ok"] and c["load_pose_ok"] and c["load_bounds_ok"] for c in candidates))
            return original(candidates,**kw)
        with patch.object(module,"select_best_candidate",side_effect=spy):
            q,_,_=s.solve(root_target(s,state),frame="world",joint_state=JointStateSnapshot(state))
        self.assertIsNotNone(q)
        self.assertTrue(seen)

    def test_new_gate_covers_other_engine_candidates(self):
        s,state=make_solver()
        T=root_target(s,state)
        candidate=module.score_candidate(s.chain,s.q_ready,T,s.psi_ready,s.q_prev,s.q_ready,
                                          s.lower,s.upper)
        candidate["q"]=s.q_ready+1.  # lies about pose_ok; independent FK must catch it.
        candidate["pose_ok"]=True
        with patch.object(s,"_collect_portfolio",return_value=([candidate],[],[])):
            q,_,_=s.solve(T,frame="world",joint_state=JointStateSnapshot(state))
        self.assertIsNone(q)
        self.assertEqual(s.last_load_diagnostics["eligible_count"],0)

    def test_final_recheck_catches_changed_physics(self):
        s,state=make_solver()
        original=s._check_load_candidate
        selected=False
        original_selector=module.select_best_candidate
        def selector(*args,**kwargs):
            nonlocal selected
            result=original_selector(*args,**kwargs)
            selected=True
            return result
        def check(*args,**kwargs):
            if selected:return False
            return original(*args,**kwargs)
        with patch.object(s,"_check_load_candidate",side_effect=check), \
                patch.object(module,"select_best_candidate",side_effect=selector):
            q,_,_=s.solve(root_target(s,state),frame="world",joint_state=JointStateSnapshot(state))
        self.assertIsNone(q)
        self.assertEqual(s.last_load_diagnostics["code"],"FINAL_RECHECK_FAILED")

    def test_observation_not_clipped(self):
        s,state=make_solver()
        T=root_target(s,state);state["Hip_Link1"]=4.
        q,_,result=s.solve(T,frame="world",joint_state=JointStateSnapshot(state))
        self.assertIsNone(q)
        self.assertIn("not clipped",result.failure_detail)

    def test_qprev_not_clipped_in_strict_mode(self):
        s,state=make_solver()
        qprev=s.q_prev.copy();qprev[0]=4.
        q,_,result=s.solve(root_target(s,state),frame="world",q_prev=qprev,
                           joint_state=JointStateSnapshot(state))
        self.assertIsNone(q)
        self.assertIn("q_prev",result.failure_detail)

    def test_complete_state_rechecked_during_evaluate_physics(self):
        s,state=make_solver()
        s.update_joint_state(JointStateSnapshot(state))
        physical=s.evaluate_physics(s.q_ready)
        self.assertTrue(physical["full_state_available"])
        self.assertEqual(physical["effort_basis"],"confirmed_allowable_joint_side_efforts")
        self.assertIn("tau_g_nm",physical)
        self.assertNotIn("q_template",s.pin_maps)

    def test_context_restored_after_success(self):
        s,state=make_solver()
        s.solve(root_target(s,state),frame="world",joint_state=JointStateSnapshot(state))
        self.assertNotIn("q_template",s.pin_maps)
        self.assertIsNone(s._load_context)
        self.assertIsNone(s._solve_T_root_base)
        np.testing.assert_array_equal(s.pin_maps["effort_limits"],s.pin_maps["urdf_effort_limits"])

    def test_explicit_off_mode_is_labelled_unprotected(self):
        s,state=make_solver(config={"load_mode":"off"})
        q,status,result=s.solve(s.chain.fk(s.q_ready),frame="local")
        self.assertIsNotNone(q,s.last_load_diagnostics)
        self.assertFalse(s.last_load_diagnostics["protection_enabled"])
        self.assertIn("OFF",s.last_load_diagnostics["code"])

    def test_no_silent_model_mass_fallback(self):
        s,state=make_solver(config={"load_massless_links":[]})
        q,_,result=s.solve(root_target(s,state),frame="world",joint_state=JointStateSnapshot(state))
        self.assertIsNone(q)
        self.assertIn("missing/zero mass",result.failure_detail)

    def test_unguarded_active_joint_rejected(self):
        s,state=make_solver(config={"load_guard_joint_names":["Hip_Link1"]})
        q,_,result=s.solve(root_target(s,state),frame="world",joint_state=JointStateSnapshot(state))
        self.assertIsNone(q)
        self.assertIn("active joints",result.failure_detail)

    def test_single_worker_busy_keeps_factory_type(self):
        s,state=make_solver()
        s._solve_lock.acquire()
        try:
            q,status,result=s.solve(np.eye(4))
            self.assertIsNone(q)
            self.assertIsInstance(result,module.IKResult)
            self.assertIn("BUSY",result.failure_detail)
        finally:s._solve_lock.release()

    def test_close_refuses_inflight_solve(self):
        s,state=make_solver()
        s._solve_lock.acquire()
        try:
            with self.assertRaises(RuntimeError):s.close()
        finally:s._solve_lock.release()

    def test_current_inactive_waist_used_for_arm_only_local_frame(self):
        s,state=make_solver(include_waist=False)
        state["Body_Link"]=.8
        Tlocal=s.chain.fk(s.q_ready)
        q,status,result=s.solve(Tlocal,frame="local",joint_state=JointStateSnapshot(state))
        self.assertIsNotNone(q,s.last_load_diagnostics)
        self.assertLess(s.last_load_diagnostics["position_error_m"],5e-4)

    def test_context_change_rejects_result(self):
        s,state=make_solver()
        snapshot=JointStateSnapshot(state)
        original=s._finalize_result
        def moved(*args,**kwargs):
            changed=dict(state,OtherJoint=state["OtherJoint"]+.2)
            s.update_joint_state(JointStateSnapshot(changed))
            return original(*args,**kwargs)
        with patch.object(s,"_finalize_result",side_effect=moved):
            q,_,result=s.solve(root_target(s,state),frame="world",joint_state=snapshot)
        self.assertIsNone(q)
        self.assertIn("CONTEXT_CHANGED",result.failure_detail)

    def test_nearby_full_goal_three_orientation_modes(self):
        for mode in ("full","tool_dir","tool_cone"):
            with self.subTest(mode=mode):
                s,state=make_solver()
                T=root_target(s,state,s.q_ready+np.linspace(-.03,.03,len(s.q_ready)))
                q,_,_=s.solve(T,frame="world",orientation_mode=mode,
                               joint_state=JointStateSnapshot(state))
                self.assertIsNotNone(q,s.last_load_diagnostics)
                self.assertLessEqual(s.last_load_diagnostics["orientation_violation_rad"],5e-4)


if __name__=="__main__":
    unittest.main()