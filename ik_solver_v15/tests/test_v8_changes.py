"""merged-v8 改动的回归测试（合成 URDF + 真实 Pinocchio；v9 起 ik_common / ik_candidate / robot_model 为用户提供的真实模块）。

覆盖：
  - 腰部舒适窗推导（derive_waist_comfort_bounds）：重力矩正弦律、负荷 / 质心两条判据在窗口边缘恰好取等、
    直立平衡时窗口 = 平衡角 ± asin(s)、偏航类 ±90°、容许力矩收窄、无重力、显式 dict / 关闭 / 非法配置；
  - 两阶段搜索：窗内有解不回退；窗内无解回退到有效限位（q_prev -> comfort_edge -> 扰动），仍然求解成功；
  - 求解类别：小位移续算不加窗（与关闭舒适窗逐值一致）；腰部大幅跟踪 + 跳变限制全程连续；阈值配置校验；
  - 臂展估计：沿运动链累加（合成模型 1.15 m），是肩 -> TCP 距离的紧上界；
  - load_mode="off" 的质心参考与代价同口径（含根 link 质量）：q_ready 处零空间不再漂移；
  - DLS 内部容差取 min(PIN_IK_*_TOL, ik_common.*_TOL)；
  - 关节与连杆同名的 URDF：帧按 BODY 查询，初始化成功且解与唯一命名模型逐值一致。
"""
import contextlib
import io
import math
import unittest
import weakref
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pinocchio as pin

from tests.fixtures import make_solver, robot_xml, root_target
from ha_arm_ik import solver as S
import ha_arm_ik.solver_dls as D
from ha_arm_ik.solver_dls import JointStateSnapshot

QREF = np.array([.05, .1, .25, -.6, .4, -.8, .3, .6, -.2])
FAR_WAIST_TARGET = np.array([-0.9, 1.9, 0.2, -0.3, 0.1, -0.3, 0.1, 0.2, 0.0])   # 肩->目标 1.52 m > 臂展 1.15 m


def solve(s, state, q=None, T=None, **kw):
    target = root_target(s, state, q) if T is None else T
    return s.solve(target, frame="world", joint_state=JointStateSnapshot(state), **kw)


def gravity_tau(s, q_active, joint_index):
    """与 derive_waist_comfort_bounds 同一构型口径：非活跃关节取 URDF 零位。"""
    template = s.pin_maps.pop("q_template", None)
    try:
        full = D.active_q_to_pinocchio_q(s.pin_model, s.pin_maps, np.asarray(q_active, dtype=float))
    finally:
        if template is not None:
            s.pin_maps["q_template"] = template
    iv = s.pin_maps["active_v_indices"][joint_index]
    return float(pin.computeGeneralizedGravity(s.pin_model, s.pin_data, full)[iv])


def solver_from_xml(xml, base_link, *, include_waist=True, config=None):
    tmp = TemporaryDirectory()
    path = Path(tmp.name) / "robot.urdf"
    path.write_text(xml)
    S.ARM_CONFIGS["left"] = {
        "base_link": base_link, "ee_link": "tcp", "s_joint": "LHand_Link2",
        "e_joint": "LHand_Link4", "w_joint": "LHand_Link6",
        "mount_xyz": np.zeros(3), "mount_rpy": np.zeros(3), "q_ready": QREF[2:],
    }
    names = robot_xml()[1] + ["OtherJoint"]
    cfg = {"load_allowable_efforts": {n: 300. for n in names},
           "load_massless_links": ["tcp"], "load_budget_s": 3., "load_state_max_age_s": 20.,
           "load_max_seeds": 4, "dls_max_iter": 100,
           "q_ready": QREF if include_waist else QREF[2:]}
    cfg.update(config or {})
    try:
        s = S.HAArmIKSolver(path, "left", include_waist=include_waist, verbose=False, config=cfg)
    except BaseException:
        tmp.cleanup()
        raise
    s._test_tmp = tmp
    s._test_finalize = weakref.finalize(s, tmp.cleanup)
    return s


class WaistComfortDeriveTests(unittest.TestCase):
    """auto 舒适窗的物理推导（合成模型：Hip_Link1 绕 y 俯仰，Body_Link 绕 z 偏航）。"""

    @classmethod
    def setUpClass(cls):
        cls.s, cls.state = make_solver()
        cls.info = cls.s.waist_comfort_info

    def test_pitch_torque_is_exact_sinusoid(self):
        e = self.info["Hip_Link1"]
        self.assertEqual((e["kind"], e["source"], e["cap_basis"]), ("gravity", "auto", "max_gravity_torque"))
        self.assertAlmostEqual(e["tau_ready_nm"], gravity_tau(self.s, QREF, 0), places=9)
        c, tb = e["tau_max_nm"], e["balance_angle"]
        self.assertAlmostEqual(gravity_tau(self.s, np.r_[tb, QREF[1:]], 0), 0.0, places=9)
        for theta in np.linspace(-2.5, 2.5, 21):
            tau = gravity_tau(self.s, np.r_[theta, QREF[1:]], 0)
            self.assertAlmostEqual(abs(tau), c * abs(math.sin(theta - tb)), places=8)
        # 前重的上半身：q_ready 处已有前倾力矩，平衡角在 q_ready 后方
        self.assertLess(tb, QREF[0])

    def test_pitch_window_edges_are_where_a_criterion_binds(self):
        e = self.info["Hip_Link1"]
        lo, hi = self.s.waist_comfort_lower[0], self.s.waist_comfort_upper[0]
        self.assertEqual((lo, hi), (e["lower"], e["upper"]))
        self.assertTrue(lo < QREF[0] < hi)
        self.assertEqual(e["limited_by_gravity"], {"lower": True, "upper": True})
        tau0, t_load, t_com = e["tau_ready_nm"], e["load_threshold_nm"], e["com_threshold_nm"]
        self.assertAlmostEqual(t_load, max(0.7 * e["cap_nm"], abs(tau0)), places=12)
        self.assertAlmostEqual(t_com, 0.7 * e["tau_max_nm"], places=12)

        def margins(theta):
            tau = gravity_tau(self.s, np.r_[theta, QREF[1:]], 0)
            return t_load - abs(tau), t_com - abs(tau - tau0)

        for edge, outward in ((lo, -1.0), (hi, 1.0)):
            m = margins(edge)
            self.assertGreaterEqual(min(m), -1e-7)
            self.assertLess(min(m), 1e-6)                       # 至少一条判据在边缘取等
            self.assertGreater(min(margins(edge - 1e-3 * outward)), 0.0)
            self.assertLess(min(margins(edge + 1e-3 * outward)), 0.0)
        for theta in np.linspace(lo, hi, 25):
            self.assertGreaterEqual(min(margins(theta)), -1e-7)

    def test_balanced_upright_window_is_balance_angle_pm_asin_s(self):
        tb = self.info["Hip_Link1"]["balance_angle"]
        q0 = np.r_[tb, QREF[1:]]
        lo, hi, info = D.derive_waist_comfort_bounds(
            self.s.pin_model, self.s.pin_data, self.s.pin_maps, q0,
            self.s.search_lower, self.s.search_upper)
        half = math.asin(0.7)
        self.assertAlmostEqual(lo[0], tb - half, places=6)
        self.assertAlmostEqual(hi[0], tb + half, places=6)
        self.assertAlmostEqual(abs(info["Hip_Link1"]["tau_ready_nm"]), 0.0, places=9)

    def test_yaw_axis_is_gravity_free_pm_90deg(self):
        e = self.info["Body_Link"]
        self.assertEqual(e["kind"], "gravity_free")
        self.assertAlmostEqual(e["axis_gravity_cos"], math.cos(QREF[0]), places=12)   # 偏航轴随俯仰倾斜
        self.assertAlmostEqual(self.s.waist_comfort_lower[1], QREF[1] - math.pi / 2, places=12)
        self.assertAlmostEqual(self.s.waist_comfort_upper[1], QREF[1] + math.pi / 2, places=12)
        np.testing.assert_array_equal(self.s.waist_comfort_lower[2:], self.s.search_lower[2:])
        np.testing.assert_array_equal(self.s.waist_comfort_upper[2:], self.s.search_upper[2:])

    def test_low_allowable_effort_narrows_load_side(self):
        efforts = {n: 300. for n in robot_xml()[1] + ["OtherJoint"]}
        efforts["Hip_Link1"] = 70.
        s, _ = make_solver(config={"load_allowable_efforts": efforts})
        e, base = s.waist_comfort_info["Hip_Link1"], self.info["Hip_Link1"]
        self.assertEqual(e["cap_basis"], "allowable_effort")
        self.assertAlmostEqual(e["load_threshold_nm"], 49.0, places=12)
        self.assertGreaterEqual(s.waist_comfort_lower[0], base["lower"] - 1e-12)
        self.assertLessEqual(s.waist_comfort_upper[0], base["upper"] + 1e-12)
        self.assertLess(s.waist_comfort_upper[0] - s.waist_comfort_lower[0], base["upper"] - base["lower"] - 1e-3)
        loads = [abs(gravity_tau(s, np.r_[t, QREF[1:]], 0)) for t in (s.waist_comfort_lower[0], s.waist_comfort_upper[0])]
        self.assertAlmostEqual(max(loads), 49.0, places=5)  # 负荷侧边缘：负载率恰为 70%

    def test_zero_gravity_makes_every_waist_joint_gravity_free(self):
        model = self.s.pin_model.copy()
        model.gravity.linear = np.zeros(3)
        lo, hi, info = D.derive_waist_comfort_bounds(
            model, model.createData(), self.s.pin_maps, QREF, self.s.search_lower, self.s.search_upper)
        self.assertEqual({info[n]["kind"] for n in ("Hip_Link1", "Body_Link")}, {"gravity_free"})
        np.testing.assert_allclose(hi[:2] - lo[:2], [math.pi, math.pi], atol=1e-12)

    def test_non_sinusoidal_torque_disables_window_for_that_joint(self):
        real = D.pin.computeGeneralizedGravity
        iv = self.s.pin_maps["active_v_indices"][0]
        iq = self.s.pin_maps["all_specs"]["Hip_Link1"]["idx_q"]

        def distorted(model, data, q):                      # 人为加入非正弦项
            tau = np.array(real(model, data, q), copy=True)
            tau[iv] += 5.0 * float(q[iq]) ** 2
            return tau

        with patch.object(D.pin, "computeGeneralizedGravity", distorted):
            lo, hi, info = D.derive_waist_comfort_bounds(
                self.s.pin_model, self.s.pin_data, self.s.pin_maps, QREF, self.s.search_lower, self.s.search_upper)
        self.assertEqual(info["Hip_Link1"]["kind"], "gravity_check_failed")
        self.assertEqual((lo[0], hi[0]), (-np.inf, np.inf))
        self.assertEqual(info["Body_Link"]["kind"], "gravity_free")

    def test_explicit_off_and_invalid_specs(self):
        s, _ = make_solver(config={"dls_waist_comfort": {"Hip_Link1": 0.2, "Body_Link": None}})
        self.assertAlmostEqual(s.waist_comfort_lower[0], QREF[0] - 0.2, places=12)
        self.assertAlmostEqual(s.waist_comfort_upper[0], QREF[0] + 0.2, places=12)
        self.assertEqual((s.waist_comfort_lower[1], s.waist_comfort_upper[1]), (s.search_lower[1], s.search_upper[1]))
        self.assertEqual(set(s.waist_comfort_info), {"Hip_Link1"})
        self.assertEqual(s.waist_comfort_info["Hip_Link1"]["source"], "explicit")
        for spec in ("off", None, False):
            s, _ = make_solver(config={"dls_waist_comfort": spec})
            self.assertIsNone(s.waist_comfort_lower)
        for spec in ("bogus", {"LHand_Link3": 0.1}, 0.3):
            with self.assertRaises(ValueError):
                make_solver(config={"dls_waist_comfort": spec})

    def test_print_info_reports_windows(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.s.print_info()
        text = out.getvalue()
        self.assertIn("waist comfort window Hip_Link1", text)
        self.assertIn("waist comfort window Body_Link", text)


class ComfortSearchTests(unittest.TestCase):
    """两阶段搜索：舒适窗是偏好，窗内无解时放开到有效限位，不减少可解目标。"""

    def test_near_target_stays_in_comfort_stage(self):
        s, state = make_solver()
        q, _, _ = solve(s, state, QREF + np.r_[0.1, -0.1, 0.05, 0.05, -0.05, 0.05, 0.0, 0.0, 0.0])
        self.assertIsNotNone(q, s.last_load_diagnostics)
        wc = s.last_load_diagnostics["waist_comfort"]
        self.assertEqual((wc["enabled"], wc["regime"], wc["applied"]), (True, "reposition", True))
        self.assertFalse(wc["fallback_to_full_bounds"])
        self.assertEqual(wc["full_bounds_seeds"], 0)
        self.assertEqual((wc["selected_stage"], wc["selected_in_comfort"]), ("comfort", True))
        self.assertFalse(s.last_load_diagnostics["search"]["waist_comfort"]["fallback_to_full_bounds"])

    def test_fallback_when_window_has_no_solution(self):
        s, state = make_solver(config={"dls_waist_comfort": {"Hip_Link1": 0.05, "Body_Link": 0.05}})
        q, _, result = solve(s, state, FAR_WAIST_TARGET)
        self.assertIsNotNone(q, s.last_load_diagnostics)
        wc = s.last_load_diagnostics["waist_comfort"]
        self.assertTrue(wc["fallback_to_full_bounds"])
        self.assertEqual((wc["selected_stage"], wc["selected_in_comfort"]), ("full", False))
        search = s.last_load_diagnostics["search"]["waist_comfort"]
        self.assertEqual((search["enabled"], search["comfort_solution_found"], search["fallback_to_full_bounds"]),
                         (True, False, True))
        per_seed = list(result.per_seed)
        stages = [p["search_stage"] for p in per_seed]
        self.assertEqual(stages, sorted(stages, key=("comfort", "full").index))   # 先窗内、后放开
        full_kinds = [p["seed_kind"] for p in per_seed if p["search_stage"] == "full"]
        self.assertEqual(full_kinds[:2], ["q_prev", "comfort_edge"])
        self.assertNotIn("locked", full_kinds)
        self.assertGreater(max(abs(q[0] - QREF[0]), abs(q[1] - QREF[1])), 0.05)
        # 同一目标在 auto 窗内可解，无需回退
        s2, state2 = make_solver()
        q2, _, _ = solve(s2, state2, FAR_WAIST_TARGET)
        self.assertIsNotNone(q2)
        self.assertEqual(s2.last_load_diagnostics["waist_comfort"]["selected_in_comfort"], True)

    def test_regime_follows_target_displacement(self):
        """目标相对 FK(q_prev) 的位移决定是否加窗：小位移 = 续算（不加窗，与关闭舒适窗逐值一致），大位移 = 重新定位。"""
        s, state = make_solver()
        s_off, state_off = make_solver(config={"dls_waist_comfort": None})
        for d, regime in [(0.003, "continuation"), (0.05, "reposition")]:
            T = root_target(s, state, QREF + d)
            q, _, _ = solve(s, state, T=T, q_prev=QREF.copy())
            q_off, _, _ = solve(s_off, state_off, T=T, q_prev=QREF.copy())
            self.assertIsNotNone(q, s.last_load_diagnostics)
            wc = s.last_load_diagnostics["waist_comfort"]
            search = s.last_load_diagnostics["search"]["waist_comfort"]
            self.assertEqual((wc["enabled"], wc["regime"], wc["applied"]), (True, regime, regime == "reposition"))
            self.assertEqual((search["regime"], search["enabled"]), (regime, regime == "reposition"))
            if regime == "continuation":
                np.testing.assert_array_equal(q, q_off)
                self.assertEqual(wc["full_bounds_seeds"], 0)
        self.assertIsNone(s_off.last_load_diagnostics["waist_comfort"]["regime"])

    def test_large_waist_tracking_with_jump_limit_is_continuous(self):
        """腰部大幅跟踪（超出舒适窗）+ max_joint_jump_rad：每个路点都按续算求解，无卡腰 / 放开瞬变。

        舒适窗不影响续算：每个路点都与关闭舒适窗逐值相同。v14 行为（续算整理姿态）下俯仰跟踪到窗外；
        v15 续算不整理姿态，腰部在 q_ready 停留更久（锁腰分支，手臂补偿），终点仍在窗内。
        """
        cfg = {"max_joint_jump_rad": 0.25}

        def run():
            s, state = make_solver(config=cfg)
            s_off, state_off = make_solver(config=dict(cfg, dls_waist_comfort=None))
            q_prev = q_prev_off = QREF.copy()
            steps = []
            for t in np.linspace(0.0, 1.0, 41)[1:]:
                qt = QREF + t * np.r_[-1.2, 1.1, 0.2, -0.2, 0.2, 0.2, -0.2, 0.2, 0.1]
                T = root_target(s, state, qt)
                q, _, _ = solve(s, state, T=T, q_prev=q_prev)
                q_off, _, _ = solve(s_off, state_off, T=T, q_prev=q_prev_off)
                self.assertIsNotNone(q, (t, s.last_load_diagnostics))
                np.testing.assert_array_equal(q, q_off)
                steps.append(float(np.max(np.abs(q - q_prev))))
                self.assertEqual(s.last_load_diagnostics["waist_comfort"]["regime"], "continuation")
                q_prev, q_prev_off = q, q_off
            return s, q_prev, steps

        # v9（真实 ik_candidate）：锁腰 -> 随动只过渡一次，该步 0.137 rad（v8 同为 0.137），其余 < 0.1
        for settle in (False, True):
            with patch.object(S, "CONTINUATION_POSTURE_SETTLE", settle):
                s, q_end, steps = run()
            self.assertLess(max(steps), 0.15)
            self.assertLessEqual(sum(st >= 0.1 for st in steps), 1)
            if settle:
                self.assertLess(q_end[0], s.waist_comfort_lower[0] - 0.1)  # 俯仰跟踪到窗外，没有被窗口挡住

    def test_continuation_tol_config(self):
        s, state = make_solver(config={"dls_waist_comfort_continuation_tol": (0.0, 0.0)})
        q, _, _ = solve(s, state, QREF + 0.01, q_prev=QREF.copy())
        self.assertIsNotNone(q)
        wc = s.last_load_diagnostics["waist_comfort"]
        self.assertEqual((wc["regime"], wc["applied"]), ("reposition", True))   # 阈值 0：任何位移都加窗
        for bad in [(-0.1, 0.1), (0.1,), (float("nan"), 0.1), (0.1, float("inf")), 0.1, "ab"]:
            with self.assertRaises(ValueError, msg=repr(bad)):
                make_solver(config={"dls_waist_comfort_continuation_tol": bad})

    def test_without_waist_comfort_is_disabled(self):
        s, state = make_solver(include_waist=False)
        self.assertIsNone(s.waist_comfort_lower)
        self.assertIsNone(s.dls_waist_max_reach)
        q, _, _ = solve(s, state, QREF[2:] + 0.05)
        self.assertIsNotNone(q)
        self.assertFalse(s.last_load_diagnostics["waist_comfort"]["enabled"])


class ArmReachTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s, cls.state = make_solver()

    def test_reach_sums_offsets_after_shoulder(self):
        s = self.s
        self.assertAlmostEqual(s.dls_waist_max_reach, 0.25 + 0.3 + 0.25 + 0.15 + 0.08 + 0.12, places=12)
        legacy = D.estimate_arm_max_reach(s.pin_model, s.pin_maps, s.pin_ee_frame_id)
        self.assertEqual(legacy, s.dls_waist_max_reach)          # 合成模型的 "_Link2" 约定指向同一关节
        self.assertIsNone(D.estimate_arm_max_reach(s.pin_model, s.pin_maps, s.pin_ee_frame_id, "OtherJoint"))
        self.assertIsNone(D.estimate_arm_max_reach(s.pin_model, s.pin_maps, s.pin_ee_frame_id, "missing"))

    def test_reach_is_tight_upper_bound(self):
        s = self.s
        sid = s.pin_maps["name_to_joint_id"]["LHand_Link2"]
        rng = np.random.default_rng(3)

        def distance(q):
            full = D.active_q_to_pinocchio_q(s.pin_model, s.pin_maps, q)
            pin.forwardKinematics(s.pin_model, s.pin_data, full)
            pin.updateFramePlacements(s.pin_model, s.pin_data)
            return float(np.linalg.norm(s.pin_data.oMf[s.pin_ee_frame_id].translation
                                        - s.pin_data.oMi[sid].translation))

        for _ in range(200):
            self.assertLessEqual(distance(rng.uniform(-3, 3, 9)), s.dls_waist_max_reach + 1e-12)
        self.assertAlmostEqual(distance(np.r_[QREF[:2], np.zeros(7)]), s.dls_waist_max_reach, places=12)


class OffModeReferenceComTests(unittest.TestCase):
    """load_mode="off"：质心参考与代价同为整机质心（含根 link 质量），q_ready 处质心项梯度为零。"""

    def run_seed(self, s, T):
        q, _ = D.pinocchio_ik_one_seed(
            s.pin_model, s.pin_data, s.pin_maps, s.pin_ee_frame_id, T, QREF.copy(), s.lower, s.upper,
            q_ref=QREF, include_waist=True, gravity_torque_gain=0.0, relative_gravity_gain=0.0,
            joint_limit_gain=0.0, max_iter=100)
        return q

    def test_reference_com_matches_cost_and_no_drift(self):
        s, state = make_solver(config={"load_mode": "off"})
        T = root_target(s, state, QREF)
        full = D.active_q_to_pinocchio_q(s.pin_model, s.pin_maps, QREF)
        com_total, _ = D.model_com_with_jacobian(s.pin_model, s.pin_data, full, derivatives=False)
        np.testing.assert_array_equal(D._reference_com(s.pin_model, s.pin_data, full), com_total)
        self.assertLessEqual(np.max(np.abs(self.run_seed(s, T) - QREF)), 1e-12)

        def moving_com(model, data, q_full):              # v7 口径：不含根 link 质量
            pin.centerOfMass(model, data, q_full, False)
            return np.array(data.com[0], copy=True)

        with patch.object(D, "_reference_com", moving_com):
            self.assertGreater(np.max(np.abs(self.run_seed(s, T) - QREF)), 1e-5)


class ToleranceTests(unittest.TestCase):
    def capture_dls_params(self, s, state):
        captured = {}
        original = S.collect_pinocchio_candidates_and_seeds

        def spy(*args, **kwargs):
            captured.update(kwargs["dls_params"])
            return original(*args, **kwargs)

        with patch.object(S, "collect_pinocchio_candidates_and_seeds", spy):
            solve(s, state, QREF + 0.05)
        return captured

    def test_dls_tolerance_not_looser_than_final_check(self):
        s, state = make_solver()
        params = self.capture_dls_params(s, state)
        self.assertEqual(params["pos_tol"], min(D.PIN_IK_POS_TOL, S.POS_TOL))
        self.assertEqual(params["ori_tol"], min(D.PIN_IK_ORI_TOL, S.ORI_TOL))
        self.assertEqual(params["shoulder_joint"], "LHand_Link2")
        self.assertEqual(params["waist_max_reach"], s.dls_waist_max_reach)
        with patch.object(S, "POS_TOL", 2e-4), patch.object(S, "ORI_TOL", 3e-4):
            params = self.capture_dls_params(s, state)
        self.assertEqual((params["pos_tol"], params["ori_tol"]), (2e-4, 3e-4))


class SameNameUrdfTests(unittest.TestCase):
    """关节按子连杆命名（joint 名 = link 名）：帧按 BODY 查询，初始化成功且解与唯一命名模型逐值一致。"""

    @staticmethod
    def same_name_xml():
        xml, names = robot_xml()
        for i, n in enumerate(names):
            xml = xml.replace(f'"link_{i}"', f'"{n}"')
        return xml.replace('<joint name="tcp_fixed"', '<joint name="tcp"')

    def test_frames_resolve_to_bodies(self):
        # include_waist=False 时 base_link 取配置值（与关节同名的 Body_Link）；带腰时取 URDF 根
        s = solver_from_xml(self.same_name_xml(), "Body_Link", include_waist=False)
        for fid, name in ((s.pin_ee_frame_id, "tcp"), (s.pin_base_frame_id, "Body_Link")):
            frame = s.pin_model.frames[fid]
            self.assertEqual((frame.name, frame.type), (name, pin.FrameType.BODY))
        with self.assertRaises(KeyError):
            S.get_pinocchio_frame_id(s.pin_model, "no_such_frame")

    def test_solutions_identical_to_unique_names(self):
        for include_waist in (True, False):
            with self.subTest(include_waist=include_waist):
                xml, names = robot_xml()
                unique = solver_from_xml(xml, "link_1", include_waist=include_waist)
                same = solver_from_xml(self.same_name_xml(), "Body_Link", include_waist=include_waist)
                state = dict(zip(names, QREF))
                state["OtherJoint"] = .4
                self.assertEqual(unique.dls_waist_max_reach, same.dls_waist_max_reach)
                rng = np.random.default_rng(5)
                for _ in range(3):
                    qt = QREF + rng.uniform(-0.4, 0.4, 9)
                    if not include_waist:
                        qt = qt[2:]
                    qa = solve(unique, state, qt)[0]
                    qb = solve(same, state, qt)[0]
                    self.assertIsNotNone(qa)
                    np.testing.assert_array_equal(qa, qb)


if __name__ == "__main__":
    unittest.main()
