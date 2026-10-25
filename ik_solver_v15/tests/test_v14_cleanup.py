"""merged-v14（基于 v13，只做清理）：公共接口保留、冗余项已删除、合并后的帮手与原实现等价。

求解行为与 v13 逐值一致由离线回放验证（486 条记录：一次性 / 跟踪 / 跳变 / 失败路径 / 配置报错 / 7 轴 /
direct-psi；除耗时、版本号与快照时间戳外 0 差异，见 README_v14.md）；本文件覆盖接口层面的约定。
"""
import inspect
import unittest

import numpy as np

from tests.fixtures import make_solver, robot_xml, root_target
from tests import dls_compat
from ha_arm_ik import solver as S
import ha_arm_ik.ik_candidate as K
import ha_arm_ik.ik_common as C
import ha_arm_ik.solver_dls as D
from ha_arm_ik.solver_dls import JointStateSnapshot


class PublicApiTests(unittest.TestCase):
    def test_constructor_and_solve_signatures(self):
        self.assertEqual(
            list(inspect.signature(S.HAArmIKSolver.__init__).parameters),
            ["self", "urdf", "side", "use_ssik", "use_pinocchio_ik", "use_torque_cost",
             "use_direct_psi", "include_waist", "verbose", "config"],
        )
        sig = inspect.signature(S.HAArmIKSolver.solve)
        self.assertEqual(
            list(sig.parameters),
            ["self", "T_target", "frame", "q_prev", "update_prev", "verbose", "allow_approximate",
             "orientation_mode", "tool_axis", "tool_cone_half_angle", "joint_state"],
        )
        self.assertIs(sig.parameters["joint_state"].kind, inspect.Parameter.KEYWORD_ONLY)

    def test_reexports_are_the_defining_objects(self):
        for name in S.__all__:
            self.assertTrue(hasattr(S, name), name)
        for name in ("IKResult", "Status", "FailureReason", "make_result"):
            self.assertIs(getattr(S, name), getattr(K, name), name)
        for name in ("JointStateSnapshot", "LoadPolicy"):
            self.assertIs(getattr(S, name), getattr(D, name), name)
        # 测试与调用方经 solver 模块访问 / patch 的其它名字仍在
        for name in ("ARM_CONFIGS", "WAIST_JOINT_NAMES", "POS_TOL", "ORI_TOL", "score_candidate",
                     "select_best_candidate", "pinocchio_ik_one_seed",
                     "collect_pinocchio_candidates_and_seeds", "get_pinocchio_frame_id",
                     "CONTINUATION_MAX_STEP", "LOCKED_SEED_SKIP_TOL"):
            self.assertTrue(hasattr(S, name), name)

    def test_version(self):
        # v15 起版本号为 merged-v15（tests/test_v15_changes.VersionTests）；这里只核对写入诊断的就是 SOLVER_VERSION
        self.assertTrue(S.SOLVER_VERSION.startswith("merged-v1"))
        s, _ = make_solver(config={"dls_seed_library_size": 0})
        self.assertEqual(s.last_load_diagnostics["version"], S.SOLVER_VERSION)


class RemovedNamesTests(unittest.TestCase):
    def test_solver_dls_aliases_and_test_only_entries_removed(self):
        for name in ("PIN_IK_STALL_DELTA_M", "PIN_IK_MAX_VALID_CANDIDATES", "WAIST_RETURN_TOL",
                     "WAIST_POST_GAIN", "WAIST_POST_ITERS", "WAIST_POST_STEP", "WAIST_POST_TOL",
                     "ARM_ROLL_VIRTUAL_LIM", "MAX_SNS_CLAMP_PASSES",
                     "_compute_nullspace_gradient_and_cost", "_solve_sns_clamped_step"):
            self.assertFalse(hasattr(D, name), name)

    def test_branch_tools_are_ik_common_functions(self):
        for name in ("joint_difference", "bounded_joint_difference", "select_closest_bounded_branch"):
            self.assertIs(getattr(D, name), getattr(C, name), name)
            self.assertIs(getattr(S, name), getattr(C, name), name)

    def test_unused_solver_imports_removed(self):
        for name in ("defaultdict", "Enum", "ET", "rotx", "T_origin", "joint_limit_barrier",
                     "finalize_ik_result", "W_SMOOTH", "NULLSPACE_GAIN", "json_safe",
                     "pinocchio_q_to_active_q"):
            self.assertFalse(hasattr(S, name), name)


class SignatureCleanupTests(unittest.TestCase):
    def test_effective_bounds_windows_are_keyword_only(self):
        lo, hi = D._build_effective_bounds(
            np.array([-7.0]), np.array([7.0]), np.array([1.0]),
            virtual_half_windows={"j": 0.5}, active_names=["j"],
        )
        np.testing.assert_array_equal(lo, [0.5])
        np.testing.assert_array_equal(hi, [1.5])
        with self.assertRaises(TypeError):  # 旧版第 4 个位置参数（不参与计算的腰部掩码）已删除
            D._build_effective_bounds(np.array([-7.0]), np.array([7.0]), np.array([1.0]),
                                      np.array([True]))

    def test_removed_parameters(self):
        self.assertNotIn("mass_points", inspect.signature(D.model_com_with_jacobian).parameters)
        self.assertNotIn("orientation_weight", inspect.signature(D.select_library_seeds).parameters)

    def test_mask_helpers_fallback_chain(self):
        maps = {"active_continuous": np.array([True, False, False])}
        np.testing.assert_array_equal(D.active_periodic_mask(maps), [True, False, False])
        np.testing.assert_array_equal(D.active_multi_turn_mask(maps), [True, False, False])
        maps["active_periodic"] = np.array([False, True, False])
        np.testing.assert_array_equal(D.active_multi_turn_mask(maps), [False, True, False])
        maps["active_multi_turn"] = np.array([False, True, True])
        np.testing.assert_array_equal(D.active_multi_turn_mask(maps), [False, True, True])
        np.testing.assert_array_equal(D.active_periodic_mask(maps), [False, True, False])

    def test_solver_multi_turn_mask_matches_ik_common(self):
        s, _ = make_solver(config={"dls_seed_library_size": 0})
        expected = C.multi_turn_mask_from_chain(s.chain, s.lower, s.upper)
        np.testing.assert_array_equal(s.multi_turn_mask, expected)
        np.testing.assert_array_equal(s.multi_turn_mask, s.pin_maps["active_multi_turn"])
        self.assertTrue(expected.any())  # 合成模型的 LHand_Link3 是 continuous


class ConfigValidationTests(unittest.TestCase):
    """合并后的配置校验帮手：报错消息与 v13 逐字相同。"""

    CASES = {
        "dls_continuation_local": ("yes", "dls_continuation_local must be True or False"),
        "dls_continuation_max_step": (0.0, "dls_continuation_max_step must be a positive finite angle"),
        "dls_seed_library_size": (-1, "dls_seed_library_size must be a nonnegative integer (0 = off)"),
        "dls_seed_library_seeds": (2.0, "dls_seed_library_seeds must be a nonnegative integer"),
        "dls_seed_library_min_separation": (
            float("inf"), "dls_seed_library_min_separation must be a finite angle >= 0"),
        "dls_waist_comfort_early_fallback": (1, "dls_waist_comfort_early_fallback must be True or False"),
        "dls_waist_comfort_patience": (0, "dls_waist_comfort_patience must be an integer >= 1"),
        "dls_jump_limit_rescue": (None, "dls_jump_limit_rescue must be True or False"),
        "dls_locked_reach_check": ("no", "dls_locked_reach_check must be True or False"),
        "dls_seed_library_continuation": (0, "dls_seed_library_continuation must be True or False"),
    }

    def test_messages(self):
        for key, (value, message) in self.CASES.items():
            with self.subTest(key=key), self.assertRaises(ValueError) as caught:
                make_solver(config={key: value})
            self.assertEqual(str(caught.exception), message)

    def test_helpers(self):
        self.assertEqual(S._finite_angle("0.25", "m", positive=True), 0.25)
        self.assertEqual(S._finite_angle(0, "m", positive=False), 0.0)
        for bad, positive in ((0, True), (True, False), ("x", False), (None, False),
                              (float("nan"), False), (-1e-9, False)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                S._finite_angle(bad, "m", positive=positive)
        self.assertEqual(S._int_at_least(np.int64(3), 1, "m"), 3)
        for bad in (True, np.bool_(True), 3.0, "3", 0):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                S._int_at_least(bad, 1, "m")
        self.assertIs(S._config_flag({"k": np.bool_(True)}, "k", False), True)
        self.assertIs(S._config_flag({}, "k", False), False)


class LoadContextRestoreTests(unittest.TestCase):
    def test_confirmed_efforts_only_during_strict_calls(self):
        names = robot_xml()[1] + ["OtherJoint"]
        s, state = make_solver(config={"dls_seed_library_size": 0,
                                       "load_allowable_efforts": {n: 250.0 for n in names}})
        T = root_target(s, state, s.q_ready + 0.05)
        q, _, _ = s.solve(T, frame="world", joint_state=JointStateSnapshot(state))
        self.assertIsNotNone(q)
        s.evaluate_physics(s.q_ready, joint_state=JointStateSnapshot(state))
        self.assertNotIn("q_template", s.pin_maps)
        np.testing.assert_array_equal(s.pin_maps["effort_limits"], s.pin_maps["urdf_effort_limits"])
        np.testing.assert_array_equal(s.joint_efforts_active, s.pin_maps["urdf_effort_limits"])
        self.assertTrue(np.all(s.pin_maps["urdf_effort_limits"] == 300.0))
        self.assertEqual(s.pin_maps["effort_basis"], "URDF reference, not confirmed continuous rating")
        self.assertEqual(s.last_load_diagnostics["code"], "ACCEPTED")


class DlsCompatTests(unittest.TestCase):
    def test_sns_diagonal_metric_matches_matrix_path_bitwise(self):
        rng = np.random.default_rng(4)
        for _ in range(100):
            n = 9
            J = rng.normal(size=(6, n))
            w = rng.uniform(0.2, 1.0, n)
            lo, hi = -rng.uniform(0.05, 2, n), rng.uniform(0.05, 2, n)
            q = rng.uniform(lo, hi)
            e, z = rng.normal(size=6), rng.normal(size=n)
            movable = np.ones(n, dtype=bool)
            movable[rng.integers(0, n)] = False
            args = (movable, q, lo, hi, 1e-6, 0.5)
            vec = D._sns_clamped_core(J, e, z, 1.0, w.copy(), None, *args)
            mat = D._sns_clamped_core(J, e, z, 1.0, None, np.diag(w), *args)
            compat = dls_compat._solve_sns_clamped_step(J, e, z, 1.0, np.diag(w), *args, return_info=True)
            for a, b, c in zip(vec, mat, compat):
                np.testing.assert_array_equal(a, b)
                np.testing.assert_array_equal(a, c)


if __name__ == "__main__":
    unittest.main()
