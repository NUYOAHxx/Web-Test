"""merged-v13（基于 v12）的回归测试：提速 + 成功率（单种子 DLS 迭代、约束与择优规则不变）。

覆盖：
  - 种子库 build_seed_library：形状、限位内、可复现、只读、位姿 = FK、一半样本在舒适窗内；
    select_library_seeds：目标与某样本位姿相同时该样本排第一、只用限位内样本、入选种子彼此 / 与已排队种子
    关节差 >= min_separation、tool_dir / tool_cone 口径；同一模型的求解器实例共享同一份库；
  - 求解器默认行为：种子库种子排在 q_prev / 锁腰 / q_ready 之后；续算类求解不用种子库（跟踪基准里换到
    远处分支的路点）；舒适阶段提前回退（COMFORT_PATIENCE）；锁腰种子可达性检查（locked_arm_reach_bound 是
    严格上界）；跳变重搜（舒适阶段之后提前 / 常规阶段之后）；配置校验；
  - 四项全部关闭 = v12 逐值一致（tests/reference/solver_v12.py 是 v12 ha_arm_ik/solver.py 的副本；v14 只把
    它对 solver_dls 的调用改成新签名：_build_effective_bounds 用关键字参数、不再导入已删除的兼容常量，逻辑未改）。
"""
import importlib.util
from pathlib import Path
import unittest
from unittest import mock

import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation

from tests.fixtures import make_solver, root_target
from tests import test_v12_changes as V12T
import ha_arm_ik
from ha_arm_ik import solver as S
import ha_arm_ik.solver_dls as D
from ha_arm_ik.solver_dls import JointStateSnapshot

REFERENCE = Path(__file__).resolve().parent / "reference"
BENCH = {"load_max_seeds": 10, "dls_max_iter": 500, "load_budget_s": 60.0}
OFF = {"dls_seed_library_size": 0, "dls_waist_comfort_early_fallback": False,
       "dls_jump_limit_rescue": False, "dls_locked_reach_check": False}
QREF = np.array([.05, .1, .25, -.6, .4, -.8, .3, .6, -.2])
FAR_WAIST_TARGET = np.array([-0.9, 1.9, 0.2, -0.3, 0.1, -0.3, 0.1, 0.2, 0.0])   # 与 test_v8_changes 相同
NARROW_COMFORT = {"dls_waist_comfort": {"Hip_Link1": 0.05, "Body_Link": 0.05}}

# 回放基准（bench/v13）里 v12 只找到跳变超限解（JOINT_JUMP_LIMIT）的两个目标：q_prev 是上一个目标的解
JUMP_EARLY = (  # 舒适阶段已出现跳变超限解 -> 放开阶段直接在跳变框内搜索
    "full",
    [[0.8913715405235342, 0.4135295548298828, -0.18560733828946527, 0.7782624804074241],
     [0.3741916559978593, -0.4402410236840596, 0.8161914270850869, -0.43896656456967886],
     [0.25580731288663505, -0.7969825270028257, -0.5471576284087696, 1.0574634837631571],
     [0.0, 0.0, 0.0, 1.0]],
    [0.78115922846521, 0.16442818054878164, 2.891622737563128, 2.9827822243135973, 0.7868271649687764,
     -0.9324415056970944, -0.19862133641498395, -2.977690668236203, -1.4557305321061382],
)
JUMP_LATE = (  # 舒适阶段没有位姿解、放开阶段只有跳变超限解 -> 常规阶段之后在跳变框内重搜
    "tool_dir",
    [[0.5600909952412875, 0.7073823961635692, -0.4311707581080978, -0.3047389496456027],
     [0.505223246569709, -0.7041407736249465, -0.4989341059142227, -1.0401311413519707],
     [-0.6565421145480216, 0.06161100970402306, -0.751768937445585, 0.8678161221263966],
     [0.0, 0.0, 0.0, 1.0]],
    [0.05, 0.1, -0.10509896056359766, 2.812128789325915, -5.2875645844727055, -0.6153870713726608,
     -0.5457150493153804, 0.40877360789324113, -2.8104250762590435],
)


# 跟踪基准（bench/v13/track，与 bench/v12/track_detail.py 相同的双臂模型与 120 个路点）左臂第 38 个路点：
# q_prev = 第 37 个路点的解。q_prev 附近的解限位裕度都略低于 SAFE_MARGIN（NEAR_LIMIT），续算局部解不成立；
# 续算类求解也用种子库时会换成 1.27 rad 外的合格解
TRACK_38 = (
    [[0.4938865583256401, 0.40468345561211905, -0.7696150779832011, -0.8338474239056171],
     [-0.8695254792291462, 0.2311081165026421, -0.4364796438070446, -0.8228430355345866],
     [0.0012282005445498857, 0.884771348564382, 0.46602376793784583, 0.8149144351374551],
     [0.0, 0.0, 0.0, 1.0]],
    [-0.9815478882281271, -0.6738346147292483, -1.0720320917750388, -0.9106250584582019, -0.04420038862689911,
     -0.3052704409526118, -0.055020985835696895, 1.1177972059798724, 0.3025264234860093],
)


def _load_v12():
    spec = importlib.util.spec_from_file_location("_ha_arm_ik_solver_v12", REFERENCE / "solver_v12.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


S12 = _load_v12()


def make_v12(config=None):
    """与 tests.fixtures.make_solver 相同的模型与配置，求解器类换成 v12 的 HAArmIKSolver。"""
    with mock.patch.object(ha_arm_ik, "solver", S12):
        return make_solver(config=config)


def solve(s, state, T, mode="full", q_prev=None):
    q, status, result = s.solve(np.asarray(T, dtype=float), frame="world", joint_state=JointStateSnapshot(state),
                                orientation_mode=mode, q_prev=None if q_prev is None else np.array(q_prev, dtype=float))
    return (None if q is None else np.asarray(q, dtype=float)), str(status), result


def seed_rows(result):
    return [(p.get("seed_kind"), p.get("search_stage"), p.get("iters"), p.get("stop_reason"))
            for p in (result.per_seed or [])]


def random_targets(s, state, n, seed):
    rng = np.random.default_rng(seed)
    lo, hi = s.lower.copy(), s.upper.copy()
    lo[~np.isfinite(lo)], hi[~np.isfinite(hi)] = -np.pi, np.pi
    out = []
    for i in range(n):
        T = root_target(s, state, rng.uniform(lo + 0.02, hi - 0.02))
        mode = ("full", "tool_dir", "tool_cone")[i % 3]
        if mode != "full":
            T[:3, :3] = T[:3, :3] @ Rotation.from_rotvec(rng.uniform(-1, 1, 3)).as_matrix()
        out.append((mode, T))
    return out


class SeedLibraryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s, cls.state = make_solver(config=BENCH)
        cls.lib = cls.s.seed_library

    def test_shape_bounds_and_read_only(self):
        s, lib = self.s, self.lib
        n = len(s.active_joint_names)
        self.assertEqual(lib["size"], D.SEED_LIBRARY_SIZE)
        self.assertEqual((lib["q"].shape, lib["p"].shape, lib["R"].shape),
                         ((D.SEED_LIBRARY_SIZE, n), (D.SEED_LIBRARY_SIZE, 3), (D.SEED_LIBRARY_SIZE, 3, 3)))
        lo = np.where(np.isfinite(s.search_lower), s.search_lower, s.q_ready - np.pi)
        hi = np.where(np.isfinite(s.search_upper), s.search_upper, s.q_ready + np.pi)
        self.assertTrue(np.all((lib["q"] >= lo) & (lib["q"] <= hi)))
        unbounded = ~np.isfinite(s.search_lower)
        self.assertTrue(np.any(unbounded))   # 夹具的 LHand_Link3 是 continuous：在 q_ready ± π 内采样
        for key in ("q", "p", "R"):
            self.assertFalse(lib[key].flags.writeable)
            with self.assertRaises(ValueError):
                lib[key][0] = 0.0
        # 一半样本落在腰部舒适窗内（舒适阶段只用窗内样本）
        self.assertIsNotNone(s.waist_comfort_lower)
        half = lib["q"][: D.SEED_LIBRARY_SIZE // 2]
        inside = np.all((half >= s.waist_comfort_lower) & (half <= s.waist_comfort_upper), axis=1)
        self.assertTrue(np.all(inside))

    def test_poses_match_forward_kinematics(self):
        s, lib = self.s, self.lib
        build = D._make_q_builder(s.pin_model, s.pin_maps)
        data = s.pin_model.createData()
        for i in (0, 1, 777, D.SEED_LIBRARY_SIZE // 2, D.SEED_LIBRARY_SIZE - 1):
            pin.forwardKinematics(s.pin_model, data, build(lib["q"][i]))
            M = data.oMi[lib["base_joint"]].actInv(pin.updateFramePlacement(s.pin_model, data, s.pin_ee_frame_id))
            np.testing.assert_allclose(lib["p"][i], M.translation, atol=1e-12)
            np.testing.assert_allclose(lib["R"][i], M.rotation, atol=1e-12)

    def test_reproducible_and_off(self):
        s = self.s
        comfort = (s.waist_comfort_lower, s.waist_comfort_upper)
        args = (s.pin_model, s.pin_model.createData(), s.pin_maps, s.pin_ee_frame_id,
                s.search_lower, s.search_upper, s.q_ready)
        a = D.build_seed_library(*args, 256, comfort_bounds=comfort)
        b = D.build_seed_library(*args, 256, comfort_bounds=comfort)
        for key in ("q", "p", "R"):
            np.testing.assert_array_equal(a[key], b[key])
        self.assertIsNone(D.build_seed_library(*args, 0))
        self.assertIsNone(D.build_seed_library(*args, None))

    def test_select_exact_match_first_and_separation(self):
        lib, s = self.lib, self.s
        lo = np.where(np.isfinite(s.search_lower), s.search_lower, -np.inf)
        hi = np.where(np.isfinite(s.search_upper), s.search_upper, np.inf)
        i = 4321
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = lib["R"][i], lib["p"][i]
        for mode in ("full", "tool_dir", "tool_cone"):
            picked = D.select_library_seeds(lib, T, mode, None, 0.3, 0.5, lo, hi, 4)
            self.assertEqual(len(picked), 4)
            np.testing.assert_array_equal(picked[0], lib["q"][i])
            for a in range(4):
                for b in range(a):
                    self.assertGreaterEqual(float(np.max(np.abs(picked[a] - picked[b]))),
                                            D.SEED_LIBRARY_MIN_SEPARATION)
        # 已排队的种子参与去重：与样本 i 太近 -> 不再选它
        near = lib["q"][i] + 0.1
        picked = D.select_library_seeds(lib, T, "full", None, 0.3, 0.5, lo, hi, 3, exclude=[near])
        self.assertTrue(all(float(np.max(np.abs(q - near))) >= D.SEED_LIBRARY_MIN_SEPARATION for q in picked))
        self.assertEqual(D.select_library_seeds(lib, T, "full", None, 0.3, 0.5, lo, hi, 0), [])
        self.assertEqual(D.select_library_seeds(None, T, "full", None, 0.3, 0.5, lo, hi, 3), [])

    def test_select_only_inside_bounds(self):
        lib = self.lib
        n = lib["q"].shape[1]
        lo, hi = np.full(n, -np.inf), np.full(n, np.inf)
        hi[2], lo[3] = 0.0, 0.0
        i = int(np.flatnonzero((lib["q"][:, 2] > 0.0) & (lib["q"][:, 3] < 0.0))[0])   # 框外的样本作目标
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = lib["R"][i], lib["p"][i]
        for mode in ("full", "tool_dir", "tool_cone"):
            picked = D.select_library_seeds(lib, T, mode, None, 0.3, 0.5, lo, hi, 3, min_separation=0.0)
            self.assertEqual(len(picked), 3)
            for q in picked:
                self.assertTrue(np.all((q >= lo) & (q <= hi)), mode)   # 只用框内样本，不截断
        empty = self.s.q_ready - 0.4
        self.assertEqual(D.select_library_seeds(lib, T, "full", None, 0.3, 0.5, empty, empty, 3), [])

    def test_tool_cone_ignores_angle_inside_cone(self):
        """tool_cone：工具轴夹角在锥角以内的样本只按位置排序（姿态差 = 夹角超出锥角的部分）。"""
        lib = self.lib
        i = 99
        T = np.eye(4)
        T[:3, 3] = lib["p"][i]
        T[:3, :3] = lib["R"][i] @ Rotation.from_rotvec([0.0, 0.0, 0.25]).as_matrix()   # 绕 z 转 0.25：x 轴偏 0.25 rad
        lo, hi = np.full(lib["q"].shape[1], -np.inf), np.full(lib["q"].shape[1], np.inf)
        picked = D.select_library_seeds(lib, T, "tool_cone", None, 0.3, 0.5, lo, hi, 1)
        np.testing.assert_array_equal(picked[0], lib["q"][i])
        far = D.select_library_seeds(lib, T, "tool_dir", None, 0.3, 0.5, lo, hi, 1, pool=1)
        self.assertEqual(len(far), 1)

    def test_instances_share_one_library(self):
        s2, _ = make_solver(config=BENCH)
        self.assertIs(s2.seed_library, self.lib)
        s3, _ = make_solver(config=dict(BENCH, dls_seed_library_size=512))
        self.assertIsNot(s3.seed_library, self.lib)
        self.assertEqual(s3.seed_library["size"], 512)
        s4, _ = make_solver(config=dict(BENCH, dls_seed_library_size=0))
        self.assertIsNone(s4.seed_library)


class SearchOrderTests(unittest.TestCase):
    def test_library_seeds_follow_lead_seeds(self):
        s, state = make_solver(config=dict(BENCH, **NARROW_COMFORT))
        q, status, result = solve(s, state, root_target(s, state, FAR_WAIST_TARGET), q_prev=QREF + 0.05)
        self.assertIsNotNone(q, s.last_load_diagnostics)
        search = s.last_load_diagnostics["search"]
        self.assertGreater(search["library_seeds_queued"], 0)
        self.assertEqual(search["locked_seed_unreachable"], 1)   # 远目标：锁腰种子够不到，跳过（不出现在 per_seed）
        # q_prev != q_ready：舒适阶段 q_prev -> (锁腰，已跳过) -> q_ready（"scatter"）-> 种子库；
        # 放开阶段 q_prev -> comfort_edge -> q_ready -> 种子库；种子库之后才是随机扰动种子
        for stage, lead in (("comfort", ["q_prev", "scatter"]), ("full", ["q_prev", "comfort_edge", "scatter"])):
            kinds = [p["seed_kind"] for p in result.per_seed if p["search_stage"] == stage]
            if stage == "comfort":
                self.assertIn("library", kinds)
            if "library" not in kinds:   # 放开阶段在前几个种子就早停时排不到种子库
                self.assertEqual(kinds, lead[:len(kinds)], stage)
                continue
            first = kinds.index("library")
            self.assertEqual(kinds[:first], lead, stage)
            n_lib = kinds.count("library")
            self.assertEqual(kinds[first:first + n_lib], ["library"] * n_lib, stage)
            self.assertLessEqual(n_lib, D.SEED_LIBRARY_SEEDS)

    def test_comfort_stage_falls_back_early(self):
        """舒适窗内无解：种子库种子跑完即放开（v12 要在窗内跑满 max_seeds 个种子）。"""
        s, state = make_solver(config=dict(BENCH, **NARROW_COMFORT))
        T = root_target(s, state, FAR_WAIST_TARGET)
        q, _, result = solve(s, state, T)
        self.assertIsNotNone(q, s.last_load_diagnostics)
        search = s.last_load_diagnostics["search"]
        self.assertTrue(search["comfort_patience_stop"])
        self.assertEqual(search["waist_comfort"]["comfort_stop_reason"], "COMFORT_PATIENCE")
        comfort = [p for p in result.per_seed if p["search_stage"] == "comfort"]
        self.assertLessEqual(len(comfort), 2 + 1 + D.SEED_LIBRARY_SEEDS)
        self.assertEqual([p["seed_kind"] for p in result.per_seed if p["search_stage"] == "full"][:2],
                         ["q_prev", "comfort_edge"])
        s2, state2 = make_solver(config=dict(BENCH, **NARROW_COMFORT, dls_waist_comfort_early_fallback=False))
        q2, _, result2 = solve(s2, state2, T)
        self.assertIsNotNone(q2)
        self.assertFalse(s2.last_load_diagnostics["search"]["comfort_patience_stop"])
        self.assertGreater(len([p for p in result2.per_seed if p["search_stage"] == "comfort"]), len(comfort))

    def test_patience_without_library(self):
        s, state = make_solver(config=dict(BENCH, **NARROW_COMFORT, dls_seed_library_size=0,
                                           dls_waist_comfort_patience=2))
        q, _, result = solve(s, state, root_target(s, state, FAR_WAIST_TARGET))
        self.assertIsNotNone(q, s.last_load_diagnostics)
        self.assertTrue(s.last_load_diagnostics["search"]["comfort_patience_stop"])
        self.assertEqual(s.last_load_diagnostics["search"]["library_seeds_queued"], 0)
        # 第 2 个名额之后停：锁腰种子被可达性检查跳过时也占名额
        self.assertLessEqual(len([p for p in result.per_seed if p["search_stage"] == "comfort"]), 2)


class ContinuationLibraryTests(V12T.TwoArmCase):
    """续算类求解默认不用种子库：搜索与关闭种子库相同，解留在 q_prev 附近。"""

    def run_case(self, config):
        s = self.make("left", config=dict(load_max_seeds=10, dls_max_iter=500, **config))
        state = {n: 0.0 for n in V12T.ALL_JOINTS}
        state.update(dict(zip(s.active_joint_names, s.q_ready)))
        T, q_prev = TRACK_38
        q, status, result = s.solve(np.array(T), frame="world", q_prev=np.array(q_prev),
                                    joint_state=JointStateSnapshot(state))
        self.assertIsNotNone(q, s.last_load_diagnostics)
        step = float(np.max(np.abs(D.joint_difference(np.asarray(q, dtype=float), np.array(q_prev),
                                                      s.multi_turn_mask))))
        return np.asarray(q, dtype=float), str(status), result, dict(s.last_load_diagnostics), step

    # TRACK_38 的续算解离限位 0.016 < SAFE_MARGIN，续算无局部解 -> 常规搜索（v14 整理姿态后如此；v15 续算不整理，
    # 但第一个达标点同样贴限位，照常整理后结果相同）
    def test_continuation_skips_library(self):
        q, status, result, diag, step = self.run_case({})
        self.assertEqual(diag["continuation"]["regime"], "continuation")
        self.assertFalse(diag["continuation"]["local_used"])
        self.assertTrue(diag["search"]["library_skipped_continuation"])
        self.assertEqual(diag["search"]["library_seeds_queued"], 0)
        self.assertNotIn("library", [p["seed_kind"] for p in result.per_seed])
        self.assertEqual(status, "NEAR_LIMIT")
        self.assertLessEqual(step, S.CONTINUATION_MAX_STEP)
        q0, status0, result0, diag0, _ = self.run_case({"dls_seed_library_size": 0})
        np.testing.assert_array_equal(q, q0)
        self.assertEqual((status, diag["code"]), (status0, diag0["code"]))
        self.assertEqual(seed_rows(result), seed_rows(result0))
        self.assertFalse(diag0["search"]["library_skipped_continuation"])   # 没有种子库时无所谓跳过

    def test_library_in_continuation_switches_branch(self):
        q, status, result, diag, step = self.run_case({"dls_seed_library_continuation": True})
        self.assertFalse(diag["search"]["library_skipped_continuation"])
        self.assertGreater(diag["search"]["library_seeds_queued"], 0)
        chosen = [p for p in result.per_seed if p["seed_idx"] == result.get("seed_used")]
        self.assertEqual([p["seed_kind"] for p in chosen], ["library"])
        self.assertEqual(status, "SUCCESS")
        self.assertGreater(step, 1.0)

    def test_reposition_still_uses_library(self):
        s, state = make_solver(config=dict(BENCH, **NARROW_COMFORT))
        q, _, result = solve(s, state, root_target(s, state, FAR_WAIST_TARGET), q_prev=QREF + 0.05)
        self.assertIsNotNone(q)
        search = s.last_load_diagnostics["search"]
        self.assertEqual(search["waist_comfort"]["regime"], "reposition")
        self.assertFalse(search["library_skipped_continuation"])
        self.assertGreater(search["library_seeds_queued"], 0)


class LockedReachTests(unittest.TestCase):
    def test_bound_is_a_strict_upper_bound(self):
        s, _ = make_solver(config=BENCH)
        joint, bound = s.dls_locked_reach
        data = s.pin_model.createData()
        build = D._make_q_builder(s.pin_model, s.pin_maps)
        rng = np.random.default_rng(5)
        lo = np.where(np.isfinite(s.lower), s.lower, -np.pi)
        hi = np.where(np.isfinite(s.upper), s.upper, np.pi)
        reach = []
        for _ in range(2000):
            pin.forwardKinematics(s.pin_model, data, build(rng.uniform(lo, hi)))
            tcp = pin.updateFramePlacement(s.pin_model, data, s.pin_ee_frame_id).translation
            reach.append(float(np.linalg.norm(tcp - data.oMi[joint].translation)))
        self.assertLess(max(reach), bound)
        self.assertGreater(max(reach), 0.9 * bound)   # 不是过分宽松的上界
        # 手臂首关节（LHand_Link1）原点 -> TCP：.1 + .25 + .3 + .25 + .15 + .08 + .12 = 1.25
        self.assertEqual(s.pin_model.names[joint], "LHand_Link1")
        self.assertAlmostEqual(bound, 1.25, places=12)
        no_waist, _ = make_solver(config=BENCH, include_waist=False)
        self.assertIsNone(no_waist.dls_locked_reach)

    def test_unreachable_locked_seed_is_skipped(self):
        s, state = make_solver(config=BENCH)
        T = root_target(s, state, FAR_WAIST_TARGET)
        q, _, result = solve(s, state, T)
        self.assertIsNotNone(q)
        search = s.last_load_diagnostics["search"]
        self.assertGreaterEqual(search["locked_seed_unreachable"], 1)
        self.assertTrue(search["locked_seed_skipped"])
        self.assertNotIn("locked", [p["seed_kind"] for p in result.per_seed])
        s2, state2 = make_solver(config=dict(BENCH, dls_locked_reach_check=False))
        q2, _, result2 = solve(s2, state2, T)
        self.assertIsNotNone(q2)
        self.assertEqual(s2.last_load_diagnostics["search"]["locked_seed_unreachable"], 0)
        locked = [p for p in result2.per_seed if p["seed_kind"] == "locked"]
        self.assertTrue(locked)
        self.assertFalse(any(p["pose_ok"] for p in locked))   # 关掉检查时锁腰种子照跑、也确实解不出

    def test_near_target_keeps_locked_seed(self):
        s, state = make_solver(config=dict(BENCH, dls_continuation_local=False))
        qt = QREF + np.r_[0.0, 0.0, 0.3, -0.2, 0.2, 0.2, -0.2, 0.2, 0.1]
        q, _, result = solve(s, state, root_target(s, state, qt))
        self.assertIsNotNone(q)
        self.assertEqual(s.last_load_diagnostics["search"]["locked_seed_unreachable"], 0)


class JumpRescueTests(unittest.TestCase):
    def check_case(self, case, early):
        mode, T, q_prev = case
        s, state = make_solver(config=BENCH)
        q, status, result = solve(s, state, T, mode, q_prev)
        d = s.last_load_diagnostics
        self.assertIsNotNone(q, d)
        self.assertTrue(d["search"]["jump_limit_rescue"])
        self.assertEqual(d["search"]["jump_limit_rescue_early"], early)
        self.assertEqual(d["waist_comfort"]["selected_stage"], "jump_box")
        stages = [p["search_stage"] for p in result.per_seed]
        self.assertEqual("full" in stages, not early)   # 提前重搜代替放开阶段
        self.assertLessEqual(float(np.max(np.abs(q - np.asarray(q_prev)))), np.pi + 1e-9)
        # 关闭：与 v12 一样只找到跳变超限解
        s2, state2 = make_solver(config=dict(BENCH, **OFF))
        q2, _, _ = solve(s2, state2, T, mode, q_prev)
        self.assertIsNone(q2)
        self.assertEqual(s2.last_load_diagnostics["code"], "JOINT_JUMP_LIMIT")
        self.assertFalse(s2.last_load_diagnostics["search"]["jump_limit_rescue"])

    def test_early_rescue(self):
        self.check_case(JUMP_EARLY, True)

    def test_rescue_after_regular_stages(self):
        self.check_case(JUMP_LATE, False)

    def test_no_rescue_with_jump_box_or_prefer_policy(self):
        """上限 < π 时 v9 的跳变框与不加框重搜照旧（不跑 jump_box）；prefer 策略不重搜。"""
        for extra in ({"max_joint_jump_rad": 0.05}, {"joint_jump_policy": "prefer"}):
            s, state = make_solver(config=dict(BENCH, **extra))
            qm = np.array([state[n] for n in s.active_joint_names])
            q, _, result = solve(s, state, root_target(s, state, qm + 0.4))
            self.assertFalse(s.last_load_diagnostics["search"]["jump_limit_rescue"], extra)
            self.assertNotIn("jump_box", [p["search_stage"] for p in (result.per_seed or [])])


class ConfigTests(unittest.TestCase):
    def test_defaults(self):
        s, _ = make_solver()
        self.assertEqual((s.dls_seed_library_size, s.dls_seed_library_seeds, s.dls_seed_library_min_separation),
                         (16384, 3, 0.5))
        self.assertEqual((s.dls_waist_comfort_early_fallback, s.dls_waist_comfort_patience),
                         (True, 4))
        self.assertEqual((s.dls_jump_limit_rescue, s.dls_locked_reach_check), (True, True))
        self.assertIs(s.dls_seed_library_continuation, False)
        self.assertEqual(s.last_load_diagnostics["version"], S.SOLVER_VERSION)

    def test_validation(self):
        bad = {
            "dls_seed_library_size": (-1, 1.5, "8", True),
            "dls_seed_library_seeds": (-1, 2.0, "3", True, None),
            "dls_seed_library_min_separation": (-0.1, float("nan"), float("inf"), "a", True, None),
            "dls_waist_comfort_early_fallback": ("yes", 1, None),
            "dls_waist_comfort_patience": (0, -1, 2.0, True, None),
            "dls_jump_limit_rescue": ("yes", 1, None),
            "dls_locked_reach_check": ("yes", 1, None),
            "dls_seed_library_continuation": ("yes", 1, None),
        }
        for key, values in bad.items():
            for value in values:
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    make_solver(config={key: value})
        for off in (0, None, False):
            s, _ = make_solver(config={"dls_seed_library_size": off})
            self.assertIsNone(s.seed_library)

    def test_library_seed_count_zero_queues_nothing(self):
        s, state = make_solver(config=dict(BENCH, dls_seed_library_seeds=0, **NARROW_COMFORT))
        q, _, result = solve(s, state, root_target(s, state, FAR_WAIST_TARGET))
        self.assertIsNotNone(q)
        self.assertEqual(s.last_load_diagnostics["search"]["library_seeds_queued"], 0)
        self.assertNotIn("library", [p["seed_kind"] for p in result.per_seed])


class V12EquivalenceTests(unittest.TestCase):
    """四项全部关闭时与 v12 逐值一致：解、状态、失败码、负荷峰值与每个种子的类别 / 阶段 / 迭代数 / 停止原因。"""

    def assert_same(self, a, b, label):
        (qa, sa, ra, da), (qb, sb, rb, db) = a, b
        self.assertEqual((sa, da.get("code")), (sb, db.get("code")), label)
        if qa is None or qb is None:
            self.assertIs(qa, qb, label)
        else:
            np.testing.assert_array_equal(qa, qb, err_msg=label)
        self.assertEqual(seed_rows(ra), seed_rows(rb), label)
        self.assertEqual(repr(da.get("peak_ratio")), repr(db.get("peak_ratio")), label)

    def run_pair(self, s13, st13, s12, st12, T, mode, q_prev):
        out = []
        for s, st in ((s13, st13), (s12, st12)):
            q, status, r = solve(s, st, T, mode, q_prev)
            out.append((q, status, r, dict(s.last_load_diagnostics)))
        return out

    @mock.patch.object(S, "CONTINUATION_POSTURE_SETTLE", True)   # v15 续算不整理姿态是另一项改动
    def test_off_matches_v12(self):
        s13, st13 = make_solver(config=dict(BENCH, **OFF))
        s12, st12 = make_v12(config=dict(BENCH, **OFF))
        self.assertIsInstance(s12, S12.HAArmIKSolver)
        self.assertNotIsInstance(s12, S.HAArmIKSolver)
        targets = random_targets(s13, st13, 9, seed=31)
        # 一次性求解（q_prev = q_ready）
        for k, (mode, T) in enumerate(targets):
            self.assert_same(*self.run_pair(s13, st13, s12, st12, T, mode, s13.q_ready), f"one-shot {k} {mode}")
        # 链式求解（q_prev = 上一个解；覆盖续算与跳变检查）
        q_prev = s13.q_ready.copy()
        for k, (mode, T) in enumerate(targets[:6]):
            a, b = self.run_pair(s13, st13, s12, st12, T, mode, q_prev)
            self.assert_same(a, b, f"chain {k} {mode}")
            if a[0] is not None:
                q_prev = a[0]
        # 小步跟踪（续算阶段）
        for k in range(3):
            T = root_target(s13, st13, QREF + 0.01 * (k + 1))
            self.assert_same(*self.run_pair(s13, st13, s12, st12, T, "full", QREF + 0.01 * k), f"track {k}")
        for mode, T, q_prev in (JUMP_EARLY, JUMP_LATE):
            a, b = self.run_pair(s13, st13, s12, st12, T, mode, q_prev)
            self.assert_same(a, b, f"jump {mode}")
            self.assertEqual(a[3]["code"], "JOINT_JUMP_LIMIT")


if __name__ == "__main__":
    unittest.main()
