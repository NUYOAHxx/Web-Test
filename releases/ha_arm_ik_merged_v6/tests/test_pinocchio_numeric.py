"""Real Pinocchio/NumPy numerical tests; ancillary project imports use test doubles."""
import unittest
import numpy as np
from tests.stubs import install
install()
from tests.fixtures import make_solver,context,root_target
from ha_arm_ik import solver_dls as dls


class RealPinocchioNumericTests(unittest.TestCase):
    def setUp(self):
        self.s,self.state=make_solver()
        self.ctx=context(self.s,self.state)
        self.q=self.s.q_ready.copy()

    def physics(self,q,derivative=False):
        return dls.load_physics(self.s.pin_model,self.s.pin_data,self.s.pin_maps,q,self.ctx,derivative)

    def test_gravity_derivative(self):
        analytic=self.physics(self.q,True)["d_tau_d_active"]
        numeric=np.zeros_like(analytic)
        for j in range(len(self.q)):
            dq=np.eye(len(self.q))[j]*1e-6
            numeric[:,j]=(self.physics(self.q+dq)["tau"]-self.physics(self.q-dq)["tau"])/2e-6
        np.testing.assert_allclose(analytic,numeric,atol=3e-7,rtol=1e-5)

    def test_com_jacobian(self):
        analytic=self.physics(self.q,True)["Jcom"]
        numeric=np.zeros_like(analytic)
        for j in range(len(self.q)):
            dq=np.eye(len(self.q))[j]*1e-6
            numeric[:,j]=(self.physics(self.q+dq)["com"]-self.physics(self.q-dq)["com"])/2e-6
        np.testing.assert_allclose(analytic,numeric,atol=2e-8,rtol=1e-5)

    def test_potential_gradient_includes_root_and_fixed_link_mass(self):
        tau=self.physics(self.q)["tau"][:len(self.q)]
        numeric=[]
        for j in range(len(self.q)):
            dq=np.eye(len(self.q))[j]*1e-6
            def U(q):
                return -self.ctx["model_mass_kg"]*self.s.pin_model.gravity.linear@self.physics(q)["com"]
            numeric.append((U(self.q+dq)-U(self.q-dq))/2e-6)
        np.testing.assert_allclose(tau,numeric,atol=5e-7,rtol=1e-5)

    def test_task_jacobians_three_modes(self):
        target=root_target(self.s,self.state,self.q+np.linspace(-.2,.25,len(self.q)))
        for mode in ("full","tool_dir","tool_cone"):
            with self.subTest(mode=mode):
                def terms(q,jac):
                    return dls.dls_task_terms(self.s.pin_model,self.s.pin_data,self.s.pin_maps,
                        self.s.pin_ee_frame_id,q,target,mode,np.array([1.,0.,0.]),.01,.5,jac)
                analytic=terms(self.q,True)["J"]
                numeric=np.zeros_like(analytic)
                for j in range(len(self.q)):
                    dq=np.eye(len(self.q))[j]*1e-7
                    numeric[:,j]=-(terms(self.q+dq,False)["error"]-terms(self.q-dq,False)["error"])/2e-7
                np.testing.assert_allclose(analytic,numeric,atol=3e-7,rtol=1e-5)

    def test_entire_secondary_gradient(self):
        ref=self.physics(self.q)
        q=self.q+np.linspace(-.03,.03,len(self.q))
        def obj(q,deriv):
            return dls._compute_nullspace_gradient_and_cost(
                self.s.pin_model,self.s.pin_data,
                dls.active_q_to_pinocchio_q(self.s.pin_model,self.s.pin_maps,q),
                q,self.q,self.s.lower,self.s.upper,np.full(len(q),.3),
                np.array(self.s.pin_maps["active_v_indices"]),self.s.pin_maps["effort_limits"],
                np.ones(len(q),dtype=bool),True,ref["tau"],ref["com"],.04,.12,.03,
                load_context=self.ctx,pin_maps=self.s.pin_maps,derivatives=deriv)
        z,_=obj(q,True)
        numeric=[]
        for j in range(len(q)):
            dq=np.eye(len(q))[j]*1e-6
            numeric.append((obj(q+dq,False)[1]-obj(q-dq,False)[1])/2e-6)
        np.testing.assert_allclose(-z,numeric,atol=1e-8,rtol=1e-5)

    def test_metric_counterexample(self):
        primary,secondary=dls._metric_svd_step(np.array([[1.,1.]]),np.array([0.]),
                                                np.array([1.,1.]),np.diag([1.,.25]),0.)
        np.testing.assert_allclose(primary,0.,atol=1e-12)
        np.testing.assert_allclose(secondary,0.,atol=1e-12)

    def test_metric_full_spd_nullspace_descent(self):
        rng=np.random.default_rng(2)
        J=rng.normal(size=(6,9));A=rng.normal(size=(9,9))
        D=A@A.T+np.eye(9)
        gradient=rng.normal(size=9)
        _,dq=dls._metric_svd_step(J,np.zeros(6),-gradient,D,1e-6)
        np.testing.assert_allclose(J@dq,0.,atol=1e-11)
        self.assertLessEqual(gradient@dq,1e-10)

    def test_continuous_mapping_and_roundtrip_branch(self):
        q=self.q.copy();q[4]=-3.865597
        full=dls.active_q_to_pinocchio_q(self.s.pin_model,self.s.pin_maps,q)
        spec=self.s.pin_maps["all_specs"]["LHand_Link3"]
        np.testing.assert_allclose(full[spec["idx_q"]:spec["idx_q"]+2],[np.cos(q[4]),np.sin(q[4])])
        back=dls.pinocchio_q_to_active_q(self.s.pin_maps,full,reference=q)
        np.testing.assert_allclose(back,q,atol=1e-12)

    def test_nonactive_state_not_zero(self):
        full=dls.active_q_to_pinocchio_q(self.s.pin_model,self.s.pin_maps,self.q)
        self.assertEqual(full[self.s.pin_maps["all_specs"]["OtherJoint"]["idx_q"]],.4)

    def test_raw_effort_no_one_fallback(self):
        raw=self.s.pin_maps["urdf_effort_limits"]
        self.assertTrue(np.all(raw==300.))
        model=self.s.pin_model.copy()
        model.effortLimit[:]=0.
        mapping=dls.build_pinocchio_joint_maps(model,self.s.active_joint_names)
        self.assertTrue(np.all(np.isnan(mapping["effort_limits"])))

    def test_continuous_scatter_midpoint_finite(self):
        seeds=dls.make_scatter_seeds(self.q,self.q,self.s.lower,self.s.upper,np.random.default_rng(3))
        self.assertTrue(all(np.all(np.isfinite(q)) for q in seeds))

    def test_no_implicit_reference_angle_windows(self):
        lo,hi=dls._build_effective_bounds(np.array([-7.,-7.]),np.array([7.,7.]),
                                           np.array([0.,0.]),np.array([True,False]))
        np.testing.assert_array_equal(lo,[-7.,-7.])
        np.testing.assert_array_equal(hi,[7.,7.])

    def test_explicit_search_window_is_separate(self):
        lo,hi=dls._build_effective_bounds(np.array([-7.]),np.array([7.]),
                                           np.array([1.]),np.array([True]),np.array([.5]))
        np.testing.assert_array_equal(lo,[.5])
        np.testing.assert_array_equal(hi,[1.5])

    def test_secondary_and_task_step_respect_bounds(self):
        q=np.array([.99,0.])
        dq,_=dls._solve_sns_clamped_step(np.array([[1.,1.]]),np.array([1.]),
            np.array([1.,-1.]),1.,np.eye(2),np.ones(2,dtype=bool),q,
            np.array([-1.,-1.]),np.array([1.,1.]),1e-6,.5)
        self.assertTrue(np.all(q+dq<=1.) and np.all(q+dq>=-1.))

    def test_final_seed_diagnostics_match_returned_configuration(self):
        target=root_target(self.s,self.state,self.q+.03)
        q,diag=dls.pinocchio_ik_one_seed(self.s.pin_model,self.s.pin_data,self.s.pin_maps,
            self.s.pin_ee_frame_id,target,self.q,self.s.lower,self.s.upper,max_iter=100,
            q_ref=self.q,include_waist=True,load_context=self.ctx)
        task=dls.dls_task_terms(self.s.pin_model,self.s.pin_data,self.s.pin_maps,
            self.s.pin_ee_frame_id,q,target,"full",np.array([1.,0,0]),.3,.5,False)
        self.assertAlmostEqual(diag["final_pos_err"],task["pos_err"],places=12)
        self.assertAlmostEqual(diag["final_ori_violation"],task["ori_violation"],places=12)

    def test_absolute_load_objective_improves_synthetic_redundant_arm(self):
        """This is a synthetic regression, not a user-robot performance claim."""
        import pinocchio as pin
        from tests.fixtures import planar_xml
        xml=planar_xml()
        model=pin.buildModelFromXML(xml)
        data=model.createData()
        names=["j0","j1","j2"]
        qref=np.array([.1,-1.,1.4])
        results=[]
        for gain in (0.,5.):
            mapping=dls.build_pinocchio_joint_maps(model,names)
            dls.configure_model_metadata(model,mapping,xml,massless_links=["tcp"])
            cfg={"load_allowable_efforts":{n:100. for n in names},
                 "load_absolute_gain":gain,"load_excess_gain":gain,
                 "load_state_max_age_s":100.}
            ctx=dls.prepare_load_context(model,mapping,cfg,
                dls.JointStateSnapshot(dict(zip(names,qref))))
            mapping["q_template"]=ctx["q_template"]
            pin.forwardKinematics(model,data,dls.active_q_to_pinocchio_q(model,mapping,qref))
            pin.updateFramePlacements(model,data)
            target=np.array(data.oMf[model.getFrameId("tcp")].homogeneous,copy=True)
            q,diag=dls.pinocchio_ik_one_seed(
                model,data,mapping,model.getFrameId("tcp"),target,qref,
                mapping["active_lower"],mapping["active_upper"],max_iter=250,
                q_ref=qref,nullspace_gain=.001,com_xy_gain=0.,joint_limit_gain=0.,
                orientation_mode="tool_cone",tool_cone_half_angle=np.pi,load_context=ctx)
            self.assertTrue(diag["converged"])
            self.assertLessEqual(diag["final_pos_err"],dls.PIN_IK_POS_TOL)
            results.append(dls.load_physics(model,data,mapping,q,ctx))
        self.assertLess(results[1]["peak_ratio"],results[0]["peak_ratio"]-.01)
        self.assertLess(results[1]["mean_square_ratio"],results[0]["mean_square_ratio"])

    def test_one_dof_prismatic_mapping_gravity_and_solve(self):
        import pinocchio as pin
        from tests.fixtures import inertial
        xml=(f'<robot name="slider"><link name="base">{inertial(5)}</link>'
             f'<link name="slider">{inertial(2)}</link>'
             '<joint name="slide" type="prismatic"><parent link="base"/>'
             '<child link="slider"/><axis xyz="0 0 1"/>'
             '<limit lower="0" upper="1" effort="100" velocity="1"/></joint></robot>')
        model=pin.buildModelFromXML(xml)
        data=model.createData()
        mapping=dls.build_pinocchio_joint_maps(model,["slide"])
        dls.configure_model_metadata(model,mapping,xml)
        ctx=dls.prepare_load_context(model,mapping,{"load_allowable_efforts":{"slide":100.}},
                                    dls.JointStateSnapshot({"slide":.2}))
        mapping["q_template"]=ctx["q_template"]
        p=dls.load_physics(model,data,mapping,np.array([.2]),ctx,True)
        np.testing.assert_allclose(p["tau"],[19.62],atol=1e-10)
        np.testing.assert_allclose(p["Jcom"][:,0],[0.,0.,2./7.],atol=1e-12)
        target=np.eye(4)
        target[2,3]=.25
        q,diag=dls.pinocchio_ik_one_seed(
            model,data,mapping,model.getFrameId("slider"),target,np.array([.2]),
            mapping["active_lower"],mapping["active_upper"],q_ref=np.array([.2]),
            load_context=ctx)
        self.assertTrue(diag["converged"])
        np.testing.assert_allclose(q,[.25],atol=dls.PIN_IK_POS_TOL)

    def test_antipodal_tool_direction_escape_one_dof(self):
        import pinocchio as pin
        from tests.fixtures import inertial
        xml=(f'<robot name="hinge"><link name="base">{inertial(5)}</link>'
             f'<link name="tool">{inertial(2)}</link>'
             '<joint name="spin" type="continuous"><parent link="base"/>'
             '<child link="tool"/><axis xyz="0 0 1"/>'
             '<limit effort="100" velocity="1"/></joint></robot>')
        model=pin.buildModelFromXML(xml)
        mapping=dls.build_pinocchio_joint_maps(model,["spin"])
        dls.configure_model_metadata(model,mapping,xml)
        ctx=dls.prepare_load_context(model,mapping,{"load_allowable_efforts":{"spin":100.}},
                                    dls.JointStateSnapshot({"spin":0.}))
        mapping["q_template"]=ctx["q_template"]
        target=np.diag([-1.,-1.,1.,1.])
        for mode in ("tool_dir","tool_cone"):
            with self.subTest(mode=mode):
                q,diag=dls.pinocchio_ik_one_seed(
                    model,model.createData(),mapping,model.getFrameId("tool"),target,
                    np.array([0.]),mapping["active_lower"],mapping["active_upper"],
                    q_ref=np.array([0.]),orientation_mode=mode,tool_cone_half_angle=.1,
                    load_context=ctx)
                self.assertTrue(diag["antipodal_escape_used"])
                self.assertTrue(diag["converged"],diag)
                self.assertTrue(np.all(np.isfinite(q)))


if __name__=="__main__":
    unittest.main()