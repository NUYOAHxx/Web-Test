"""merged-v9 改动的回归测试（合成 URDF + 真实 Pinocchio + 用户提供的真实 ik_common / ik_candidate / robot_model）。

覆盖：
  - 续算局部优先：腰部大幅跟踪不再在“锁腰 / 随动”分支间切换（关闭时复现 v8 的大跳变）；普通跟踪腰部不漂、
    逐帧步长接近目标步长；max_joint_jump_rad=0.25 时腰部大幅跟踪全部成功；重新定位与关闭时逐值一致；
    续算框内无解时回到常规搜索；诊断字段；配置校验；
  - 跳变框：只有超限解时照旧报 JOINT_JUMP_LIMIT（框内无解 -> 不加框重搜）；
  - psi(q_ready) 无定义时告警，且候选打分不再全部为 inf；
  - ik_common：normalize 拒绝非有限、validate_transform4x4 刚体校验、clip_to_bounds 零跨度、target_frames 缺变换；
  - ik_candidate：psi_ready=NaN 打分、无腰链 waist_cost、状态词表、finalize_ik_result(per_seed=None)、
    evaluate_config_quality（continuous 关节 / q_template / 输入校验）；
  - robot_model：S/E/W 兜底名与 ARM_CONFIGS 一致、fk_pose_and_joints 缺关节报错。
"""
import math
import unittest
import warnings
from unittest.mock import patch

import numpy as np
import pinocchio as pin

from tests.fixtures import make_solver, root_target
from ha_arm_ik import solver as S
import ha_arm_ik.ik_candidate as C
import ha_arm_ik.ik_common as K
import ha_arm_ik.robot_model as RM
import ha_arm_ik.solver_dls as D
from ha_arm_ik.solver_dls import JointStateSnapshot

QREF = np.array([.05, .1, .25, -.6, .4, -.8, .3, .6, -.2])
BENCH = {"load_max_seeds": 10, "dls_max_iter": 500, "load_budget_s": 2.0}
WAIST_RAMP = np.r_[-1.2, 1.1, 0.2, -0.2, 0.2, 0.2, -0.2, 0.2, 0.1]


def solve(s, state, q=None, T=None, **kw):
    target = root_target(s, state, q) if T is None else T
    return s.solve(target, frame="world", joint_state=JointStateSnapshot(state), **kw)


def sine_targets(s, state, n, t_end, waist_amp):
    """与 bench/v9/run_targets.py 相同的正弦路点（rng 种子 3）；waist_amp=None 时腰部不动。"""
    qm = np.array([state[k] for k in s.active_joint_names])
    rng = np.random.default_rng(3)
    amp = rng.uniform(0.2, 0.5, size=len(qm))
    amp[:2] = 0.0 if waist_amp is None else waist_amp
    ph = rng.uniform(0, 2 * np.pi, size=len(qm))
    w = rng.uniform(0.5, 1.0, size=len(qm))
    t_all = np.linspace(0, t_end, n + 1)[1:]
    return qm, [root_target(s, state, np.clip(qm + amp * (np.sin(w * t + ph) - np.sin(ph)), -2.8, 2.8))
                for t in t_all]


def track(s, state, targets, q0):
    q_prev, steps, local = np.array(q0, dtype=float), [], []
    for T in targets:
        q, status, _ = solve(s, state, T=T, q_prev=q_prev)
        if q is None:
            return None, steps, local, s.last_load_diagnostics
        steps.append(float(np.max(np.abs(np.asarray(q) - q_prev))))
        local.append(bool(s.last_load_diagnostics["continuation"]["local_used"]))
        q_prev = np.asarray(q, dtype=float)
    return q_prev, steps, local, None


class ContinuationLocalTests(unittest.TestCase):
    def test_waist_ramp_has_no_branch_switch(self):
        """腰部大幅跟踪（无跳变限制）：v9 全程局部续算、单步 < 0.15；关闭续算局部即 v8，出现 > 0.5 rad 的分支切换。

        v15 续算不整理姿态：锁腰分支保持更久（手臂补偿，约前 3/4 路点腰部留在 q_ready），之后逐帧过渡；
        v14 行为（CONTINUATION_POSTURE_SETTLE = True）在第 5 个路点就转到随动。两者都只有一步 >= 0.1。
        """
        s, state = make_solver(config=BENCH)
        s_off, state_off = make_solver(config=dict(BENCH, dls_continuation_local=False))
        targets = [root_target(s, state, QREF + t * WAIST_RAMP) for t in np.linspace(0.0, 1.0, 41)[1:]]
        q_end, steps, local, fail = track(s, state, targets, QREF)
        self.assertIsNone(fail)
        self.assertTrue(all(local))
        self.assertLess(max(steps), 0.15)
        self.assertLessEqual(sum(st >= 0.1 for st in steps), 1)
        with patch.object(S, "CONTINUATION_POSTURE_SETTLE", True):
            _, steps14, local14, fail14 = track(s, state, targets, QREF)
        self.assertIsNone(fail14)
        self.assertTrue(all(local14))
        self.assertLess(max(steps14), 0.15)
        self.assertLessEqual(sum(st >= 0.1 for st in steps14), 1)   # v14：锁腰 -> 随动只过渡一次
        _, steps_off, local_off, fail_off = track(s_off, state_off, targets, QREF)
        self.assertIsNone(fail_off)
        self.assertFalse(any(local_off))
        self.assertGreater(max(steps_off), 0.5)

    def test_sine_waist_tracking_with_jump_limit_all_succeed(self):
        """bench 的腰部大幅正弦跟踪（前 120 个路点）+ max_joint_jump_rad=0.25：全部成功，单步不超过续算框。"""
        s, state = make_solver(config=dict(BENCH, max_joint_jump_rad=0.25))
        qm, targets = sine_targets(s, state, 239, 12.0, [0.9, 1.2])
        q_end, steps, local, fail = track(s, state, targets[:120], qm)
        self.assertIsNone(fail, fail)
        self.assertTrue(all(local))
        self.assertLessEqual(max(steps), S.CONTINUATION_MAX_STEP + 1e-9)

    def test_ordinary_tracking_keeps_waist_at_ready(self):
        """普通跟踪（腰部目标不动）：腰部留在 q_ready（v8 在负荷容差带边界来回切换，单步 0.198 rad）。"""
        s, state = make_solver(config=BENCH)
        qm, targets = sine_targets(s, state, 119, 6.0, None)
        q_prev, steps = qm.copy(), []
        for T in targets:
            q, status, _ = solve(s, state, T=T, q_prev=q_prev)
            self.assertIsNotNone(q)
            q = np.asarray(q, dtype=float)
            steps.append(float(np.max(np.abs(q - q_prev))))
            self.assertLess(float(np.max(np.abs(q[:2] - s.q_ready[:2]))), 0.02)
            q_prev = q
        self.assertLess(max(steps), 0.03)

    def test_reposition_identical_to_local_off(self):
        s, state = make_solver()
        s_off, state_off = make_solver(config={"dls_continuation_local": False})
        rng = np.random.default_rng(11)
        for _ in range(4):
            qt = rng.uniform(-2.5, 2.5, size=9)
            qt[:2] = rng.uniform(-0.8, 0.8, size=2)
            q, st, _ = solve(s, state, qt)
            q_off, st_off, _ = solve(s_off, state_off, qt)
            self.assertEqual(st, st_off)
            if q is not None:
                np.testing.assert_array_equal(q, q_off)
            cont = s.last_load_diagnostics["continuation"]
            self.assertEqual((cont["regime"], cont["local_used"]), ("reposition", False))

    def test_diagnostics_for_local_solution(self):
        s, state = make_solver()
        q, status, result = solve(s, state, QREF + 0.01, q_prev=QREF)
        self.assertEqual(status, "SUCCESS")
        d = s.last_load_diagnostics
        self.assertEqual(d["continuation"], {"local_enabled": True, "regime": "continuation", "local_used": True,
                                             "hold": False})
        self.assertTrue(d["search"]["continuation_local_used"])
        self.assertEqual(d["search"]["stop_reason"], "CONTINUATION_LOCAL")
        stages = {p.get("search_stage") for p in result.per_seed}
        self.assertEqual(stages, {"continuation"})
        self.assertLessEqual(float(np.max(np.abs(np.asarray(q) - QREF))), S.CONTINUATION_MAX_STEP + 1e-9)

    def test_no_local_solution_falls_back_to_regular_search(self):
        """续算框太小、框内无可接受解：回到常规搜索并照常求解（续算候选留在池中）。"""
        s, state = make_solver(config={"dls_continuation_max_step": 1e-3})
        q, status, result = solve(s, state, QREF + 0.01, q_prev=QREF)
        self.assertEqual(status, "SUCCESS")
        d = s.last_load_diagnostics
        self.assertEqual((d["continuation"]["regime"], d["continuation"]["local_used"]), ("continuation", False))
        stages = [p.get("search_stage") for p in result.per_seed]
        self.assertIn("continuation", stages)
        self.assertTrue(any(st in ("comfort", "full") for st in stages))

    def test_config_validation(self):
        for bad in ("yes", 1, None):
            with self.assertRaises(ValueError):
                make_solver(config={"dls_continuation_local": bad})
        for bad in (0.0, -0.1, float("nan"), float("inf"), True, "a", None):
            with self.assertRaises(ValueError):
                make_solver(config={"dls_continuation_max_step": bad})
        s, _ = make_solver(config={"dls_continuation_local": False, "dls_continuation_max_step": 0.2})
        self.assertEqual((s.dls_continuation_local, s.dls_continuation_max_step), (False, 0.2))


class JumpBoxTests(unittest.TestCase):
    def test_only_over_limit_solutions_still_report_joint_jump_limit(self):
        s, state = make_solver(config={"max_joint_jump_rad": 0.05})
        qm = np.array([state[n] for n in s.active_joint_names])
        q, status, result = solve(s, state, qm + 0.4)
        self.assertIsNone(q)
        d = s.last_load_diagnostics
        self.assertEqual(d["code"], "JOINT_JUMP_LIMIT")
        self.assertTrue(d["search"]["jump_limit_search_box"])
        self.assertTrue(d["search"]["jump_limit_unboxed_retry"])

    def test_box_not_used_for_prefer_policy(self):
        s, state = make_solver(config={"max_joint_jump_rad": 0.05, "joint_jump_policy": "prefer"})
        qm = np.array([state[n] for n in s.active_joint_names])
        q, status, _ = solve(s, state, qm + 0.4)
        self.assertEqual(status, "SUCCESS")
        self.assertFalse(s.last_load_diagnostics["search"]["jump_limit_search_box"])


class PsiReadyTests(unittest.TestCase):
    def test_nan_psi_ready_scores_stay_finite(self):
        """psi_ready = NaN：臂角项记 0，其余代价照算（旧实现所有候选 cost = inf，择优失效）。"""
        s, state = make_solver()
        q = s.q_ready + 0.05
        info = C.score_candidate(s.chain, q, root_target(s, state, q), float("nan"), s.q_ready,
                                 s.q_ready, s.lower, s.upper, apply_waist_return=True)
        self.assertEqual(info["psi_cost"], 0.0)
        self.assertTrue(math.isfinite(info["cost_no_torque"]))
        self.assertGreater(info["waist_cost"], 0.0)
        self.assertTrue(info["pose_ok"])

    def test_solver_warns_when_psi_ready_undefined(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with patch.object(RM.SerialChain, "psi", lambda self, q, *a, **k: float("nan")):
                s, state = make_solver()
        self.assertTrue(any(issubclass(w.category, RuntimeWarning) and "psi(q_ready)" in str(w.message)
                            for w in caught))
        self.assertTrue(math.isnan(s.psi_ready))
        self.assertIsNotNone(s.psi_ready_error)
        q, status, _ = solve(s, state, QREF + 0.01, q_prev=QREF)       # 续算
        self.assertEqual(status, "SUCCESS")
        q, status, _ = solve(s, state, QREF + 0.4)                     # 重新定位（含 direct-psi 阶段）
        self.assertIn(status, ("SUCCESS", "NEAR_LIMIT"))


class IkCommonTests(unittest.TestCase):
    def test_normalize_rejects_non_finite(self):
        for bad in ([np.nan, 0, 0], [np.inf, 1, 0]):
            with self.assertRaises(ValueError):
                K.normalize(bad)
        with self.assertRaises(ValueError):
            K.normalize([0, 0, 0])
        np.testing.assert_allclose(K.normalize([3, 0, 4]), [0.6, 0, 0.8])

    def test_validate_transform_requires_rigid(self):
        good = np.eye(4)
        good[:3, :3] = pin.rpy.rpyToMatrix(0.3, -0.2, 1.0)
        good[:3, 3] = [0.1, 0.2, 0.3]
        np.testing.assert_array_equal(K.validate_transform4x4(good), good)
        scaled, bottom, mirror = good.copy(), good.copy(), good.copy()
        scaled[:3, :3] *= 1.01
        bottom[3] = [0, 0, 1e-6, 1]
        mirror[:3, 0] *= -1
        for bad in (scaled, bottom, mirror):
            with self.assertRaises(ValueError):
                K.validate_transform4x4(bad)

    def test_clip_to_bounds_zero_span(self):
        lo, hi = np.array([0.3, -1.0, 0.0]), np.array([0.3, 1.0, 1e-9])
        q = K.clip_to_bounds(np.array([5.0, 5.0, 5.0]), lo, hi)
        self.assertEqual(q[0], 0.3)
        self.assertAlmostEqual(q[1], 1.0 - 1e-9, places=15)
        self.assertTrue(lo[2] <= q[2] <= hi[2])

    def test_target_frames_requires_transform(self):
        T = np.eye(4)
        with self.assertRaises(ValueError):
            K.target_frames(T, "local")
        with self.assertRaises(ValueError):
            K.target_frames(T, "world")
        shift = np.eye(4)
        shift[:3, 3] = [0, 0, 1]
        local, world = K.target_frames(T, "local", T_world_from_base=shift)
        np.testing.assert_array_equal(world, shift)
        bad = shift.copy()
        bad[:3, :3] *= 2
        with self.assertRaises(ValueError):
            K.target_frames(T, "world", T_base_from_world=bad)


class IkCandidateTests(unittest.TestCase):
    def setUp(self):
        self.s, self.state = make_solver()

    def score(self, q, **kw):
        s = self.s
        T = root_target(s, self.state, q)
        return C.score_candidate(s.chain, np.asarray(q, dtype=float), T, s.psi_ready, s.q_ready,
                                 s.q_ready, s.lower, s.upper, **kw)

    def test_status_vocabulary_matches_public_status(self):
        info = self.score(QREF + 0.05)
        self.assertEqual(info["status"], "SUCCESS")
        self.assertEqual(info["status"], C.candidate_status(info))
        near = QREF.copy()
        near[5] = self.s.upper[5] - 0.01          # LHand_Link4（revolute ±3）
        info = self.score(near)
        self.assertEqual((info["status"], C.candidate_status(info)), ("NEAR_LIMIT", "NEAR_LIMIT"))
        s = self.s
        bad = C.score_candidate(s.chain, QREF + 0.3, root_target(s, self.state, QREF), s.psi_ready,
                                QREF, s.q_ready, s.lower, s.upper)
        self.assertEqual((bad["status"], C.candidate_status(bad)), ("NO_VALID_POSE", "NO_VALID_POSE"))

    def test_chain_without_waist_has_zero_waist_cost(self):
        s7, state7 = make_solver(include_waist=False)
        q = s7.q_ready + 0.05
        info = C.score_candidate(s7.chain, q, root_target(s7, state7, q), s7.psi_ready, s7.q_ready,
                                 s7.q_ready, s7.lower, s7.upper, apply_waist_return=True)
        self.assertEqual(info["waist_cost"], 0.0)
        self.assertTrue(math.isfinite(info["cost_no_torque"]))

    def test_waist_indices_follow_joint_names(self):
        idx = C._waist_indices(self.s.chain, len(QREF))
        np.testing.assert_array_equal(idx, [0, 1])

    def test_finalize_without_per_seed(self):
        class Holder:
            q_prev = None
        q, status, result = C.finalize_ik_result(Holder(), [], None)
        self.assertIsNone(q)
        self.assertEqual(status, "NO_RESULT")
        self.assertEqual(result.failure_reason, C.FailureReason.SEEDS_EXHAUSTED)
        bad = self.score(QREF)
        bad = dict(bad, pose_ok=False)
        q, status, result = C.finalize_ik_result(Holder(), [bad], None)
        self.assertEqual((q, status), (None, "NO_VALID_POSE"))
        self.assertEqual(result.failure_reason, C.FailureReason.NOT_CONVERGED)

    def test_evaluate_config_quality_handles_continuous_and_template(self):
        s, state = self.s, self.state
        root_target(s, state)  # 写入 q_template（含 OtherJoint 实测值）
        q = QREF + 0.1
        out = C.evaluate_config_quality(s.pin_model, s.pin_data, s.pin_maps, q, s.pin_ee_frame_id)
        full = D.active_q_to_pinocchio_q(s.pin_model, s.pin_maps, q)
        J = np.asarray(pin.computeFrameJacobian(s.pin_model, s.pin_data, full, s.pin_ee_frame_id, pin.LOCAL))
        J = J[:, np.asarray(s.pin_maps["active_v_indices"])]
        J[:3] /= 0.5
        sv = np.linalg.svd(J, compute_uv=False)
        self.assertTrue(any(s.pin_maps["active_continuous"]))
        self.assertAlmostEqual(out["jac_min_sv"], float(sv[-1]), places=12)
        self.assertAlmostEqual(out["jac_cond"], float(sv[0] / sv[-1]), places=6)
        for bad in (q[:5], np.r_[q[:-1], np.nan]):
            out = C.evaluate_config_quality(s.pin_model, s.pin_data, s.pin_maps, bad, s.pin_ee_frame_id)
            self.assertTrue(math.isnan(out["jac_min_sv"]))


class RobotModelTests(unittest.TestCase):
    def test_default_sew_joint_names_match_arm_config(self):
        """S/E/W 兜底名与 ARM_CONFIGS 的 LHand_Link* 命名一致：链未给 s/e/w 时臂角仍有定义。"""
        self.assertEqual((RM.S_JOINT, RM.E_JOINT, RM.W_JOINT), ("LHand_Link2", "LHand_Link4", "LHand_Link6"))
        s, _ = make_solver()
        chain = RM.SerialChain(s.chain.urdf_path, s.chain.base_link, s.chain.ee_link)
        self.assertTrue(math.isfinite(chain.psi(QREF)))
        self.assertAlmostEqual(chain.psi(QREF), s.chain.psi(QREF), places=12)

    def test_fk_pose_and_joints_missing_joint_raises(self):
        s, _ = make_solver()
        with self.assertRaises(KeyError):
            s.chain.fk_pose_and_joints(QREF, ["LHand_Link2", "NoSuchJoint"])
        T, joints = s.chain.fk_pose_and_joints(QREF, ["LHand_Link2", "LHand_Link4"])
        self.assertEqual(set(joints), {"LHand_Link2", "LHand_Link4"})
        np.testing.assert_allclose(T, s.chain.fk(QREF), atol=1e-12)


if __name__ == "__main__":
    unittest.main()
