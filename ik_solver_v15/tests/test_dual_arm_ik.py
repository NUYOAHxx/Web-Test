# -*- coding: utf-8 -*-
"""ha_arm_ik/dual_arm_ik.py（腰 2 + 左右各 7 轴的 16 维联合 IK）的测试。

只用 numpy + Pinocchio 和合成双臂 URDF（tests/dual_fixtures.py）。与单臂判据数值一致性的检查需要单臂模块，
缺失时跳过。正确性用例关闭时间预算（config time_budget_s=None），结果与机器快慢无关。
"""

import ast
import json
import math
import subprocess
import sys
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pinocchio as pin

from ha_arm_ik import dual_arm_ik as D
from tests.dual_fixtures import ALL_JOINTS, two_arm_urdf_path, synthetic_layout

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "ha_arm_ik" / "dual_arm_ik.py"
NO_BUDGET = {"time_budget_s": None}


def _solver(config=None, layout=None):
    return D.DualArmIKSolver(two_arm_urdf_path(), layout=layout or synthetic_layout(), config=config)


class _IndependentFK:
    """与求解器无关的 FK（直接用 Pinocchio，按关节名填 q），用于核对结果。"""

    def __init__(self):
        self.model = pin.buildModelFromUrdf(two_arm_urdf_path())
        self.data = self.model.createData()
        self.fid = {s: self.model.getFrameId(f"{s}_tcp") for s in ("left", "right")}

    def __call__(self, q):
        qf = pin.neutral(self.model)
        for i, name in enumerate(ALL_JOINTS):
            j = self.model.joints[self.model.getJointId(name)]
            if j.nq == 1:
                qf[j.idx_q] = q[i]
            else:
                qf[j.idx_q], qf[j.idx_q + 1] = math.cos(q[i]), math.sin(q[i])
        pin.framesForwardKinematics(self.model, self.data, qf)
        return {s: self.data.oMf[self.fid[s]].copy() for s in ("left", "right")}


def _pose_err(M, T):
    T = pin.SE3(np.asarray(T)[:3, :3], np.asarray(T)[:3, 3])
    return (float(np.linalg.norm(M.translation - T.translation)),
            float(np.linalg.norm(pin.log3(T.rotation.T @ M.rotation))))


def _random_q(s, rng, inset=0.05):
    lo = np.where(np.isfinite(s.lower), s.lower, -np.pi)
    hi = np.where(np.isfinite(s.upper), s.upper, np.pi)
    return rng.uniform(lo + inset, hi - inset)


class DualShared(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s = _solver(NO_BUDGET)
        cls.fk = _IndependentFK()

    def setUp(self):
        self.s.reset_prev()

    def assertPoseOk(self, q, T, side):
        pe, oe = _pose_err(self.fk(q)[side], T)
        self.assertLessEqual(pe, D.POS_TOL)
        self.assertLessEqual(oe, D.ORI_TOL)

    def assertWithinLimits(self, q, margin=0.0):
        lo, hi = self.s.lower, self.s.upper
        fin_lo, fin_hi = np.isfinite(lo), np.isfinite(hi)
        self.assertTrue(np.all(q[fin_lo] >= lo[fin_lo] + margin - 1e-9))
        self.assertTrue(np.all(q[fin_hi] <= hi[fin_hi] - margin + 1e-9))


# =============================================================================
# 1. 独立性与判据一致性
# =============================================================================
class TestIndependence(unittest.TestCase):
    FORBIDDEN = ("ha_arm_ik.solver", "ha_arm_ik.solver_dls", "ha_arm_ik.ik_common", "ha_arm_ik.ik_candidate",
                 "ha_arm_ik.robot_model", "solver", "solver_dls", "ik_common", "ik_candidate", "robot_model", "scipy")

    def test_source_imports_only_numpy_and_pinocchio(self):
        tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                names.add((node.module or "").split(".")[0] if node.level == 0 else "<relative>")
        allowed = {"__future__", "math", "threading", "time", "collections", "dataclasses", "pathlib", "numpy",
                   "pinocchio"}
        self.assertLessEqual(names, allowed, f"unexpected imports: {sorted(names - allowed)}")

    def test_loading_module_pulls_in_no_single_arm_code(self):
        code = (
            "import importlib.util, json, sys\n"
            f"spec = importlib.util.spec_from_file_location('dual_arm_ik_standalone', {str(MODULE_PATH)!r})\n"
            "mod = importlib.util.module_from_spec(spec); sys.modules[spec.name] = mod\n"
            "spec.loader.exec_module(mod)\n"
            "print(json.dumps(sorted(sys.modules)))\n"
        )
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120, check=True)
        mods = json.loads(out.stdout.strip().splitlines()[-1])
        bad = [m for m in mods if any(m == f or m.startswith(f + ".") for f in self.FORBIDDEN)]
        self.assertEqual(bad, [])


class TestCriteriaMatchSingleArm(unittest.TestCase):
    """容差、判据、DLS 常数与单臂代码数值一致（双臂文件里是复制的数值，不是引用）。"""

    @classmethod
    def setUpClass(cls):
        try:
            from ha_arm_ik import ik_common, robot_model, solver, solver_dls
        except Exception as exc:  # pragma: no cover - 单臂模块不在时跳过
            raise unittest.SkipTest(f"single-arm modules not importable: {exc}")
        cls.C, cls.S, cls.L, cls.R = ik_common, solver, solver_dls, robot_model

    def test_tolerances_and_criteria(self):
        C, S = self.C, self.S
        for name in ("POS_TOL", "ORI_TOL", "SAFE_MARGIN", "PERIODIC_MIN_SPAN_RAD", "TOOL_CONE_HALF_ANGLE",
                     "RIGID_BOTTOM_ROW_TOL", "RIGID_ROTATION_TOL"):
            self.assertEqual(getattr(D, name), getattr(C, name), name)
        np.testing.assert_array_equal(D.TOOL_AXIS_X, C.TOOL_AXIS_X)
        self.assertEqual(tuple(D.ORIENTATION_MODES), tuple(C.ORIENTATION_MODES))
        self.assertEqual(D.MAX_JOINT_JUMP_RAD, S.MAX_JOINT_JUMP_RAD)
        self.assertEqual(tuple(D.JOINT_JUMP_POLICIES), tuple(S.JOINT_JUMP_POLICIES))
        self.assertEqual(D.RNG_SEED, S.RNG_SEED)
        self.assertEqual(D.CONTINUATION_MAX_STEP, S.CONTINUATION_MAX_STEP)
        self.assertEqual(D.CONTINUATION_DLS_STEP_FRACTION, S.CONTINUATION_DLS_STEP_FRACTION)
        self.assertEqual(D.CONTINUATION_TOL, self.L.WAIST_COMFORT_CONTINUATION_STEP)
        for status in ("SUCCESS", "NEAR_LIMIT", "APPROXIMATE", "NO_VALID_POSE", "OUTSIDE_WORKSPACE", "NO_RESULT"):
            self.assertEqual(getattr(D, status), status)

    def test_dls_constants(self):
        L = self.L
        pairs = {"MAX_STEP": "PIN_IK_MAX_STEP", "DAMPING_MIN": "PIN_IK_DAMPING", "STALL_ITERS": "PIN_IK_STALL_ITERS",
                 "STALL_RTOL": "PIN_IK_STALL_RTOL", "NULLSPACE_GAIN": "NULLSPACE_GAIN",
                 "WAIST_NULL_GAIN": "WAIST_NULL_GAIN", "NULLSPACE_ERR_SCALE": "NULLSPACE_ERR_SCALE",
                 "POSTURE_SETTLE_TOL": "POSTURE_SETTLE_TOL", "SETTLE_COST_RTOL": "SETTLE_COST_RTOL",
                 "SETTLE_PATIENCE": "SETTLE_PATIENCE", "MAX_SETTLE_ITERS": "MAX_SETTLE_EXTRA_ITERS",
                 "WAIST_TASK_WEIGHT": "WAIST_TASK_WEIGHT", "WAIST_TASK_WEIGHT_FAR": "WAIST_TASK_WEIGHT_FAR",
                 "WAIST_RELEASE_START": "WAIST_RELEASE_START", "WAIST_RELEASE_END": "WAIST_RELEASE_END",
                 "WAIST_RETURN_RELEASE_RAD": "WAIST_RETURN_RELEASE_RAD", "SEED_LIBRARY_RNG_SEED": "SEED_LIBRARY_RNG_SEED"}
        for mine, theirs in pairs.items():
            self.assertEqual(getattr(D, mine), getattr(L, theirs), mine)

    def test_default_layout_matches_robot_model(self):
        # 重新执行 robot_model.py 取文件里的数值：单臂测试（tests/fixtures.make_solver）会改写
        # robot_model.ARM_CONFIGS["left"]，全量运行时模块里的值不一定是文件里的值
        import importlib.util
        spec = importlib.util.spec_from_file_location("_robot_model_dual_file", self.R.__file__)
        R = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(R)
        lay = D.DEFAULT_LAYOUT
        self.assertEqual(list(lay.waist_joints), list(R.WAIST_JOINT_NAMES))
        for side, tcp, sh, qr in (("left", lay.left_tcp, lay.left_shoulder, lay.q_ready_left),
                                  ("right", lay.right_tcp, lay.right_shoulder, lay.q_ready_right)):
            cfg = R.ARM_CONFIGS[side]
            self.assertEqual(tcp, cfg["ee_link"])
            self.assertEqual(sh, cfg["s_joint"])
            np.testing.assert_array_equal(np.array(qr), cfg["q_ready"])


# =============================================================================
# 2. 模型、FK、雅可比
# =============================================================================
class TestModel(DualShared):
    def test_layout_validation(self):
        bad = [synthetic_layout().__class__(left_tcp="nope", right_tcp="right_tcp"),
               synthetic_layout().__class__(left_tcp="left_tcp", right_tcp="right_tcp",
                                            left_joints=tuple(f"LHand_Link{i}" for i in (2, 1, 3, 4, 5, 6, 7))),
               synthetic_layout().__class__(left_tcp="left_tcp", right_tcp="right_tcp",
                                            right_joints=tuple(f"LHand_Link{i}" for i in range(1, 8))),
               synthetic_layout().__class__(left_tcp="left_tcp", right_tcp="right_tcp", waist_joints=("Hip_Link1", "X")),
               synthetic_layout().__class__(left_tcp="left_tcp", right_tcp="right_tcp", left_shoulder="RHand_Link2")]
        for lay in bad:
            with self.assertRaises(ValueError):
                _solver(layout=lay)
        with self.assertRaises(TypeError):
            D.DualArmIKSolver(two_arm_urdf_path(), layout={"left_tcp": "left_tcp"})

    def test_layout_order_and_limits(self):
        self.assertEqual(self.s.active_joint_names, ALL_JOINTS)
        self.assertEqual(self.s.lower[0], -1.0)
        self.assertEqual(self.s.upper[1], 2.5)
        self.assertTrue(np.isinf(self.s.lower[4]) and np.isinf(self.s.upper[11]))   # 两臂第 3 关节 continuous

    def test_fk_matches_independent_pinocchio(self):
        rng = np.random.default_rng(0)
        for _ in range(10):
            q = _random_q(self.s, rng)
            TL, TR = self.s.fk(q)
            ref = self.fk(q)
            for T, side in ((TL, "left"), (TR, "right")):
                pe, oe = _pose_err(ref[side], T)
                self.assertLess(pe, 1e-12)
                self.assertLess(oe, 1e-9)

    def test_continuous_joints_are_2pi_periodic(self):
        q = self.s.q_ready.copy()
        q[4] += 0.7
        q2 = q.copy()
        q2[4] += 2 * np.pi
        q2[11] -= 4 * np.pi
        for a, b in zip(self.s.fk(q), self.s.fk(q2)):
            np.testing.assert_allclose(a, b, atol=1e-12)

    def _fd(self, mode, cone=None):
        rng = np.random.default_rng(1)
        worst_J, worst_g = 0.0, 0.0
        for _ in range(8):
            qa = _random_q(self.s, rng, 0.1)
            TL, TR = self.s.fk(qa + rng.normal(0.0, 0.4, 16))
            hands, _ = self.s._parse_targets(TL, TR, qa, self.s._m.template, mode, None, cone)
            e, J = np.empty(12), np.empty((12, 16))
            self.s._evaluate(qa, hands, True, e, J, self.s._m.template)
            Jfd, gfd, h = np.empty((12, 16)), np.empty(16), 1e-7
            for i in range(16):
                dq = np.zeros(16)
                dq[i] = h
                ep, em = np.empty(12), np.empty(12)
                self.s._evaluate(qa + dq, hands, False, ep, None, self.s._m.template)
                self.s._evaluate(qa - dq, hands, False, em, None, self.s._m.template)
                Jfd[:, i] = (ep - em) / (2 * h)
                gfd[i] = (ep @ ep - em @ em) / (4 * h)
            worst_J = max(worst_J, np.abs(Jfd + J).max() / max(1.0, np.abs(J).max()))
            g = -J.T @ e
            worst_g = max(worst_g, np.abs(g - gfd).max() / max(1e-9, np.abs(g).max()))
        return worst_J, worst_g

    def test_jacobian_finite_difference_full_and_tool_dir(self):
        for mode in ("full", "tool_dir"):
            wj, wg = self._fd(mode)
            self.assertLess(wj, 1e-6, mode)
            self.assertLess(wg, 1e-6, mode)

    def test_tool_cone_gradient_finite_difference(self):
        # 锥外的目标方向随 d 转动的分量与误差正交：雅可比不是 de/dq，但 |e|² 的梯度准确
        _, wg = self._fd("tool_cone", 0.05)
        self.assertLess(wg, 1e-6)


# =============================================================================
# 3. 求解：可达目标、跟踪、目标类型、姿态模式
# =============================================================================
class TestSolve(DualShared):
    def test_random_reachable_targets(self):
        rng = np.random.default_rng(2024)
        for k in range(30):
            q_true = _random_q(self.s, rng)
            TL, TR = self.s.fk(q_true)
            self.s.reset_prev()
            q, status, res = self.s.solve(TL, TR)
            self.assertEqual(status, D.SUCCESS, f"target {k}: {res}")
            self.assertIsNone(res.failure_reason)
            self.assertTrue(res.ok)
            self.assertPoseOk(q, TL, "left")
            self.assertPoseOk(q, TR, "right")
            self.assertWithinLimits(q, D.SAFE_MARGIN)
            self.assertGreaterEqual(res["margin"], D.SAFE_MARGIN)
            np.testing.assert_array_equal(res["q"], q)
            np.testing.assert_array_equal(np.r_[res["q_waist"], res["q_left"], res["q_right"]], q)
            for side in ("left", "right"):
                self.assertTrue(res[side]["pose_ok"])
                self.assertLessEqual(res[side]["pos_err"], D.POS_TOL)
            np.testing.assert_array_equal(self.s.q_prev, q)        # update_prev 默认 True

    def test_tracking_small_steps(self):
        q = self.s.q_ready.copy()
        TL0, TR0 = self.s.fk(q)
        worst_jump, seeds = 0.0, []
        for k in range(1, 31):
            TL, TR = TL0.copy(), TR0.copy()
            TL[:3, 3] += [0.004 * k, 0.002 * k, 0.003 * k]
            TR[:3, 3] += [0.004 * k, -0.002 * k, -0.002 * k]
            q_new, status, res = self.s.solve(TL, TR)
            self.assertEqual(status, D.SUCCESS, f"step {k}: {res}")
            worst_jump = max(worst_jump, float(np.abs(q_new - q).max()))
            seeds.append(res["seeds_used"])
            q = q_new
        self.assertLess(worst_jump, 0.1)          # 5 mm 级步长 -> 关节小步，不换构型
        self.assertEqual(set(seeds), {1})

    def test_continuation_from_unsettled_posture(self):
        """路点跟踪（单臂 v9 续算规则）：q_prev 远离 q_ready 时，小步目标也不让姿态漂走（不做达标后的整理）。"""
        rng = np.random.default_rng(21)
        qa, qb = _random_q(self.s, rng, 0.3), _random_q(self.s, rng, 0.3)
        qb = qa + 0.6 * (qb - qa) / np.abs(qb - qa).max()          # 每步每关节 0.04 rad
        q = qa.copy()
        for k in range(1, 16):
            TL, TR = self.s.fk(qa + (qb - qa) * k / 15)
            q_new, status, res = self.s.solve(TL, TR, q_prev=q)
            self.assertIn(status, (D.SUCCESS, D.NEAR_LIMIT), f"step {k}: {res}")
            self.assertEqual(res["regime"], "continuation")
            self.assertEqual(res["seed_kind"], "continuation")
            self.assertLess(float(np.abs(q_new - q).max()), 0.1, f"step {k}")
            self.assertPoseOk(q_new, TL, "left")
            self.assertPoseOk(q_new, TR, "right")
            q = q_new

    def test_continuation_returns_solved_q_prev_unchanged(self):
        rng = np.random.default_rng(22)
        q0 = _random_q(self.s, rng, 0.1)
        TL, TR = self.s.fk(q0)
        q, status, res = self.s.solve(TL, TR, q_prev=q0)
        self.assertEqual(status, D.SUCCESS, str(res))
        np.testing.assert_array_equal(q, q0)            # 已是解：不向 q_ready 整理（伺服循环里不会自己漂）
        self.assertEqual(res["iterations"], 0)

    def test_continuation_switch_and_regime(self):
        rng = np.random.default_rng(23)
        q0 = _random_q(self.s, rng, 0.1)
        TL, TR = self.s.fk(q0 + 0.01)
        s2 = _solver({**NO_BUDGET, "continuation_local": False})
        _, status, res = s2.solve(TL, TR, q_prev=q0)
        self.assertEqual(res["regime"], "reposition")
        self.assertEqual(status, D.SUCCESS, str(res))
        TLf, TRf = self.s.fk(self.s.q_ready)                         # 远目标：常规搜索
        _, status, res = self.s.solve(TLf, TRf, q_prev=q0)
        self.assertEqual(res["regime"], "reposition")
        with self.assertRaises(ValueError):
            _solver({"continuation_max_step": 0.0})

    def test_q_prev_slightly_outside_limits(self):
        q0 = self.s.q_ready.copy()
        q0[3] = self.s.upper[3] + 0.05                                # 实测关节角略超限位
        TL, TR = self.s.fk(np.clip(q0, self.s.lower, self.s.upper))
        q, status, res = self.s.solve(TL, TR, q_prev=q0)
        self.assertIn(status, (D.SUCCESS, D.NEAR_LIMIT), str(res))
        self.assertWithinLimits(q)
        self.assertPoseOk(q, TL, "left")

    def test_hold_keep_free(self):
        rng = np.random.default_rng(3)
        q0 = self.s.q_ready + rng.normal(0.0, 0.2, 16)
        q0 = np.clip(q0, self.s.lower + 0.05, self.s.upper - 0.05)
        _, TR = self.s.fk(self.s.q_ready + rng.normal(0.0, 0.25, 16))
        TL_prev, _ = self.s.fk(q0)
        # hold：左手保持 q_prev 处的世界位姿
        q, status, res = self.s.solve("hold", TR, q_prev=q0)
        self.assertEqual(status, D.SUCCESS, str(res))
        self.assertPoseOk(q, TL_prev, "left")
        self.assertPoseOk(q, TR, "right")
        self.assertEqual(res["targets"], {"left": "hold", "right": "pose"})
        # keep：左臂关节角原样保持（腰可以动）
        q, status, res = self.s.solve("keep", TR, q_prev=q0)
        self.assertEqual(status, D.SUCCESS, str(res))
        np.testing.assert_array_equal(q[D.LEFT], q0[D.LEFT])
        self.assertPoseOk(q, TR, "right")
        # None = "hold"（dual-p2.1 默认）：结果与显式 "hold" 逐位相同
        q_hold = self.s.solve("hold", TR, q_prev=q0)[0]
        q, status, res = self.s.solve(None, TR, q_prev=q0)
        self.assertEqual(status, D.SUCCESS, str(res))
        np.testing.assert_array_equal(q, q_hold)
        self.assertEqual(res["targets"], {"left": "hold", "right": "pose"})
        # "free"：不约束左手，只回中
        q, status, res = self.s.solve("free", TR, q_prev=q0)
        self.assertEqual(status, D.SUCCESS, str(res))
        self.assertPoseOk(q, TR, "right")
        self.assertIsNone(res["left"]["pos_err"])
        self.assertEqual(res["targets"]["left"], "free")
        # ("keep", None)：左臂关节不动、右手保持位姿（None = "hold"）
        _, TR_prev = self.s.fk(q0)
        q, status, res = self.s.solve("keep", None, q_prev=q0)
        self.assertEqual(status, D.SUCCESS, str(res))
        np.testing.assert_array_equal(q[D.LEFT], q0[D.LEFT])
        self.assertPoseOk(q, TR_prev, "right")
        self.assertEqual(res["targets"], {"left": "keep", "right": "hold"})
        # ("hold", "hold") 允许：两手都不动，q_prev 已是解 -> 原样返回
        q, status, res = self.s.solve("hold", "hold", q_prev=q0)
        self.assertEqual(status, D.SUCCESS, str(res))
        np.testing.assert_array_equal(q, q0)
        # 两只都是 None 报错；至少一只手要有位姿 / hold 目标
        with self.assertRaisesRegex(ValueError, "both None"):
            self.s.solve(None, None, q_prev=q0)
        for args in (("keep", "keep"), ("keep", "free"), ("free", "free"), ("FREE", "keep")):
            with self.assertRaisesRegex(ValueError, "at least one hand"):
                self.s.solve(*args, q_prev=q0)
        with self.assertRaisesRegex(ValueError, "'free' or None"):
            self.s.solve("grab", TR)

    def test_orientation_modes(self):
        rng = np.random.default_rng(4)
        q_true = _random_q(self.s, rng, 0.2)
        TL, TR = self.s.fk(q_true)
        for mode in ("tool_dir", "tool_cone"):
            self.s.reset_prev()
            q, status, res = self.s.solve(TL, TR, orientation_mode=mode)
            self.assertEqual(status, D.SUCCESS, f"{mode}: {res}")
            got = self.fk(q)
            for side, T in (("left", TL), ("right", TR)):
                self.assertLessEqual(np.linalg.norm(got[side].translation - T[:3, 3]), D.POS_TOL)
                ang = math.acos(np.clip((got[side].rotation @ D.TOOL_AXIS_X) @ (T[:3, :3] @ D.TOOL_AXIS_X), -1, 1))
                limit = D.ORI_TOL if mode == "tool_dir" else D.TOOL_CONE_HALF_ANGLE + D.ORI_TOL
                self.assertLessEqual(ang, limit + 1e-9)
                self.assertEqual(res[side]["orientation_mode"], mode)
        # 每只手各自的模式与参数：左 full，右 tool_cone（半角 0.2 rad，工具轴 +Z）
        self.s.reset_prev()
        axis = np.array([0.0, 0.0, 1.0])
        q, status, res = self.s.solve(TL, TR, orientation_mode=("full", "tool_cone"), tool_axis=(None, axis),
                                      tool_cone_half_angle=(None, 0.2))
        self.assertEqual(status, D.SUCCESS, str(res))
        self.assertPoseOk(q, TL, "left")
        got = self.fk(q)["right"]
        ang = math.acos(np.clip((got.rotation @ axis) @ (TR[:3, :3] @ axis), -1, 1))
        self.assertLessEqual(ang, 0.2 + D.ORI_TOL)
        self.assertEqual((res["left"]["orientation_mode"], res["right"]["orientation_mode"]), ("full", "tool_cone"))

    def test_joint_state_supplies_q_prev(self):
        rng = np.random.default_rng(5)
        q0 = np.clip(self.s.q_ready + rng.normal(0.0, 0.2, 16), self.s.lower + 0.05, self.s.upper - 0.05)
        TL, TR = self.s.fk(q0)
        state = dict(zip(ALL_JOINTS, q0.tolist()))
        q, status, res = self.s.solve(TL, TR, joint_state=state, update_prev=False)
        self.assertEqual(status, D.SUCCESS)
        self.assertEqual(res["seed_kind"], "continuation")  # q_prev 本身就是解：续算第一步即达标
        self.assertEqual(res["seeds_used"], 1)
        self.assertPoseOk(q, TL, "left")
        with self.assertRaises(ValueError):
            self.s.solve(TL, TR, joint_state={"no_such_joint": 0.0})

    def test_update_prev_flag(self):
        rng = np.random.default_rng(6)
        TL, TR = self.s.fk(_random_q(self.s, rng, 0.2))
        before = self.s.q_prev.copy()
        q, status, _ = self.s.solve(TL, TR, update_prev=False)
        self.assertEqual(status, D.SUCCESS)
        np.testing.assert_array_equal(self.s.q_prev, before)
        self.s.solve(TL, TR)
        np.testing.assert_array_equal(self.s.q_prev, q)

    def test_deterministic_across_instances(self):
        other = _solver(NO_BUDGET)
        rng = np.random.default_rng(7)
        for _ in range(8):
            TL, TR = self.s.fk(_random_q(self.s, rng))
            self.s.reset_prev()
            other.reset_prev()
            q1, s1, r1 = self.s.solve(TL, TR)
            q2, s2, r2 = other.solve(TL, TR)
            self.assertEqual(s1, s2)
            np.testing.assert_array_equal(q1, q2)
            self.assertEqual(r1["seeds_used"], r2["seeds_used"])

    def test_library_is_shared_between_instances(self):
        self.assertIs(_solver(NO_BUDGET)._lib, self.s._lib)
        self.assertIsNot(_solver({"seed_library_size": 1024})._lib, self.s._lib)


# =============================================================================
# 4. 跳变策略、近限位、失败分类、时间预算
# =============================================================================
class TestPoliciesAndFailures(DualShared):
    def _far_target(self):
        q = self.s.q_ready.copy()
        q[1] = 1.5
        q[2] += 1.2
        return self.s.fk(q)

    def test_jump_policy_reject(self):
        s = _solver({"max_joint_jump_rad": 0.3, "time_budget_s": 0.5})
        TL, TR = self._far_target()
        q, status, res = s.solve(TL, TR)
        self.assertIsNone(q)
        self.assertEqual(status, D.NO_VALID_POSE)
        self.assertEqual(res.failure_reason, D.JOINT_JUMP_LIMIT)
        # 近目标：解出且每个关节都在 ±0.3 rad 内
        TL2, TR2 = s.fk(s.q_ready + 0.05)
        q, status, res = s.solve(TL2, TR2)
        self.assertEqual(status, D.SUCCESS, str(res))
        self.assertLessEqual(res["max_jump"], 0.3 + 1e-9)
        self.assertTrue(res["jump_ok"])

    def test_jump_policy_prefer(self):
        s = _solver({"max_joint_jump_rad": 0.3, "joint_jump_policy": "prefer", "time_budget_s": 0.5})
        TL, TR = self._far_target()
        q, status, res = s.solve(TL, TR)
        self.assertEqual(status, D.SUCCESS, str(res))
        self.assertIsNone(res.failure_reason)
        self.assertFalse(res["jump_ok"])
        self.assertGreater(res["max_jump"], 0.3)
        self.assertPoseOk(q, TL, "left")

    def test_jump_limited_target_is_classified_within_budget(self):
        """解需要超过 max_joint_jump_rad 的转动：框内搜索 + 不加框探测 -> JOINT_JUMP_LIMIT（不报超时或不可达）。"""
        s = _solver({"max_joint_jump_rad": 0.3})                     # 默认预算 0.1 s
        TL, TR = self._far_target()
        q, status, res = s.solve(TL, TR)
        self.assertIsNone(q)
        self.assertEqual(res.failure_reason, D.JOINT_JUMP_LIMIT, str(res))
        self.assertNotIn("diagnosis", res)
        self.assertLess(res["elapsed_s"], 0.1 + 0.02)

    def test_library_seeds_respect_jump_box(self):
        rng = np.random.default_rng(24)
        s = self.s
        for _ in range(5):
            q_prev = _random_q(s, rng)
            TL, TR = s.fk(_random_q(s, rng))
            hands, kinds = s._parse_targets(TL, TR, q_prev, s._m.template, "full", None, None)
            ctx = s._make_ctx(hands, kinds, q_prev, s._m.template, shrink=True)
            self.assertTrue(ctx.box_lo.any() or ctx.box_hi.any())
            seeds = s._library_seeds(hands, kinds, q_prev, 8, box=ctx)
            self.assertGreater(len(seeds), 0)
            b = ~s._m.unbounded
            for q in seeds:
                self.assertTrue(np.all(q[b] >= ctx.lo[b] - 1e-9) and np.all(q[b] <= ctx.hi[b] + 1e-9))

    def test_jump_limit_per_joint_dict(self):
        s = _solver({"max_joint_jump_rad": {"Body_Link": 0.1, "LHand_Link1": None}})
        self.assertAlmostEqual(s._jump[1], 0.1)
        self.assertTrue(np.isinf(s._jump[2]))
        self.assertAlmostEqual(s._jump[3], D.MAX_JOINT_JUMP_RAD)
        for bad in ({"nope": 0.1}, -1.0, 0.0):
            with self.assertRaises(ValueError):
                _solver({"max_joint_jump_rad": bad})

    def test_near_limit_classification(self):
        q = self.s.q_ready.copy()
        q[3] = self.s.upper[3] - 0.005                      # 左臂第 2 关节离上限 5 mrad
        TL, TR = self.s.fk(q)
        hands, kinds = self.s._parse_targets(TL, TR, q, self.s._m.template, "full", None, None)
        ctx = self.s._make_ctx(hands, kinds, q, self.s._m.template, shrink=False)
        c = self.s._candidate(q, hands, ctx, q, "test")
        self.assertEqual(c["status"], D.NEAR_LIMIT)
        self.assertAlmostEqual(c["margin"], 0.005, places=9)
        self.assertEqual(c["margin_joint"], "LHand_Link2")
        self.assertFalse(c["acceptable"])
        self.assertTrue(c["near_ok"])

    def test_outside_workspace_is_immediate(self):
        far = np.eye(4)
        far[:3, 3] = [3.0, 0.5, 0.5]
        far_r = np.eye(4)
        far_r[:3, 3] = [-3.0, -0.5, 0.5]
        _, TR = self.s.fk(self.s.q_ready)
        for args, reason in (((far, TR), D.LEFT_UNREACHABLE), ((self.s.fk(self.s.q_ready)[0], far_r), D.RIGHT_UNREACHABLE),
                             ((far, far_r), D.BOTH_UNREACHABLE)):
            t = time.perf_counter()
            q, status, res = self.s.solve(*args)
            self.assertLess(time.perf_counter() - t, 0.05)
            self.assertIsNone(q)
            self.assertEqual(status, D.OUTSIDE_WORKSPACE)
            self.assertEqual(res.failure_reason, reason)
            self.assertEqual(res["seeds_used"], 0)

    def _not_jointly_pair(self):
        # 两组构型各自 FK 出一只手的目标：两目标间距 3.10 m > 两臂根距离 + 两臂展（2.905 m），不可能同时到达
        qa = np.array([-0.268, 1.49, 1.419, -0.735, 1.912, 0.0, 1.386, 0.337, 1.157,
                       -0.412, -2.091, -2.48, -0.359, 2.655, 0.441, 1.461])
        qb = np.array([0.862, 0.848, -1.465, -1.201, 1.743, 0.043, 2.502, 0.152, 0.703,
                       -1.078, 0.113, -2.169, 0.0, -1.512, 0.51, 0.012])
        return self.s.fk(qa)[0], self.s.fk(qb)[1]

    def test_not_jointly_reachable_precheck(self):
        TL, TR = self._not_jointly_pair()
        q, status, res = self.s.solve(TL, TR)
        self.assertIsNone(q)
        self.assertEqual(status, D.OUTSIDE_WORKSPACE)
        self.assertEqual(res.failure_reason, D.NOT_JOINTLY_REACHABLE)
        # 每只手单独都能解（另一只手 "free"；None 现在是 "hold"）
        for args in ((TL, "free"), ("free", TR)):
            self.s.reset_prev()
            self.assertEqual(self.s.solve(*args)[1], D.SUCCESS)

    def test_pair_distance_bound_is_exact_and_waist_independent(self):
        s = self.s
        lib, m = s._lib, s._m
        self.assertEqual(m.extra_chain_joints, [])
        self.assertTrue(s._precheck)
        model, data = m.model, m.data
        gaps = []
        for w in ((0.0, 0.0), (1.2, -2.0), (-0.8, 2.3)):
            q = s.q_ready.copy()
            q[:2] = w
            pin.forwardKinematics(model, data, m.full_q(q, m.template))
            gaps.append(np.linalg.norm(data.oMi[m.arm_root_jid["left"]].translation
                                       - data.oMi[m.arm_root_jid["right"]].translation))
        np.testing.assert_allclose(gaps, lib.root_gap, atol=1e-12)
        rl, rr = m.reach_root["left"], m.reach_root["right"]
        half = 0.5 * (lib.root_gap + rl + rr) + 0.002
        u, c = np.array([0.0, 1.0, 0.0]), np.array([0.3, 0.0, 1.0])
        self.assertTrue(lib.jointly_outside(c + half * u, rl, c - half * u, rr))

    def test_not_jointly_reachable_by_diagnosis(self):
        # 预检判不出来时（模拟），由搜索 + 失败诊断给出 NOT_JOINTLY_REACHABLE，且在预算内结束。
        # 预算 0.3 s（诊断窗口 = 预算的 17%）：0.1 s 时诊断窗口只有 17 ms，机器慢或有 CPU 争用时会来不及做完
        # （结果是 TIME_BUDGET_EXCEEDED，也是正确行为，但不是本测试要核对的结论）
        budget = 0.3
        s = _solver({"time_budget_s": budget})
        TL, TR = self._not_jointly_pair()
        with mock.patch.object(s._lib, "jointly_outside", return_value=False):
            t = time.perf_counter()
            q, status, res = s.solve(TL, TR)
            elapsed = time.perf_counter() - t
        self.assertIsNone(q)
        self.assertEqual(status, D.NO_VALID_POSE)
        self.assertEqual(res.failure_reason, D.NOT_JOINTLY_REACHABLE)
        self.assertTrue(res["diagnosis"]["left_alone_ok"])
        self.assertTrue(res["diagnosis"]["right_alone_ok"])
        self.assertLess(elapsed, budget + 0.03)

    def test_allow_approximate(self):
        TL, TR = self._not_jointly_pair()
        s = _solver({"time_budget_s": 0.05})
        with mock.patch.object(s._lib, "jointly_outside", return_value=False):
            q, status, res = s.solve(TL, TR, allow_approximate=True)
        self.assertEqual(status, D.APPROXIMATE)
        self.assertIsNotNone(q)
        self.assertFalse(res.ok)
        self.assertIsNotNone(res.failure_reason)
        self.assertGreater(res["left"]["pos_err"] + res["right"]["pos_err"], D.POS_TOL)
        self.assertWithinLimits(q)

    def test_time_budget_is_respected(self):
        TL, TR = self._not_jointly_pair()
        s = _solver()
        for budget in (0.01, 0.03):
            with mock.patch.object(s._lib, "jointly_outside", return_value=False):
                t = time.perf_counter()
                q, status, res = s.solve(TL, TR, time_budget_s=budget)
                elapsed = time.perf_counter() - t
            self.assertIsNone(q)
            self.assertLess(elapsed, budget + 0.02)
            self.assertIn(res.failure_reason, (D.TIME_BUDGET_EXCEEDED, D.NOT_JOINTLY_REACHABLE))
        self.assertEqual(s.config["time_budget_s"], D.TIME_BUDGET_S)

    def test_busy_instance_reports_solver_busy(self):
        TL, TR = self.s.fk(self.s.q_ready)
        self.s._lock.acquire()
        try:
            q, status, res = self.s.solve(TL, TR)
        finally:
            self.s._lock.release()
        self.assertIsNone(q)
        self.assertEqual(status, D.NO_VALID_POSE)
        self.assertEqual(res.failure_reason, D.SOLVER_BUSY)

    def test_threads_with_separate_instances(self):
        rng = np.random.default_rng(8)
        targets = [self.s.fk(_random_q(self.s, rng, 0.2)) for _ in range(4)]
        out = {}

        def run(i):
            s = _solver(NO_BUDGET)
            out[i] = [s.solve(TL, TR, update_prev=False)[1] for TL, TR in targets]

        ts = [threading.Thread(target=run, args=(i,)) for i in range(2)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        self.assertEqual(out[0], out[1])
        self.assertEqual(set(out[0]), {D.SUCCESS})


# =============================================================================
# 5. 输入校验
# =============================================================================
class TestValidation(DualShared):
    def test_bad_inputs(self):
        TL, TR = self.s.fk(self.s.q_ready)
        bad_rot = TL.copy()
        bad_rot[:3, :3] *= 1.01
        bad_row = TL.copy()
        bad_row[3, 0] = 1e-6
        cases = [dict(T_left=bad_rot, T_right=TR), dict(T_left=bad_row, T_right=TR),
                 dict(T_left=TL[:3], T_right=TR), dict(T_left=np.full((4, 4), np.nan), T_right=TR),
                 dict(T_left=TL, T_right=TR, orientation_mode="xyz"),
                 dict(T_left=TL, T_right=TR, orientation_mode="tool_cone", tool_cone_half_angle=0.0),
                 dict(T_left=TL, T_right=TR, orientation_mode="tool_dir", tool_axis=[0.0, 0.0, 0.0]),
                 dict(T_left=TL, T_right=TR, q_prev=np.zeros(15)),
                 dict(T_left=TL, T_right=TR, q_prev=np.full(16, np.inf)),
                 dict(T_left=TL, T_right=TR, time_budget_s=-1.0)]
        for kw in cases:
            with self.assertRaises(ValueError, msg=str(list(kw))):
                self.s.solve(**kw)
        q, status, _ = self.s.solve(pin.SE3(TL[:3, :3], TL[:3, 3]), TR)     # pin.SE3 也接受
        self.assertEqual(status, D.SUCCESS)

    def test_bad_config(self):
        for cfg in ({"unknown": 1}, {"time_budget_s": 0.0}, {"max_seeds": 0}, {"max_iter": 0},
                    {"joint_jump_policy": "maybe"}, {"waist_grid": (0, 3)}, {"seed_library_size": -1}):
            with self.assertRaises(ValueError, msg=str(cfg)):
                _solver(cfg)

    def test_fk_and_reset_prev_validate(self):
        with self.assertRaises(ValueError):
            self.s.fk(np.zeros(14))
        with self.assertRaises(ValueError):
            self.s.reset_prev(np.zeros(3))
        self.s.reset_prev(np.zeros(16))
        np.testing.assert_array_equal(self.s.q_prev, np.zeros(16))


if __name__ == "__main__":
    unittest.main()
