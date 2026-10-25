"""merged-v7 提速改动的回归测试（合成 URDF + 真实 Pinocchio；v9 起 ik_common / ik_candidate / robot_model 为用户提供的真实模块）。

参考实现 tests/reference/solver_dls_v6.py 是 merged-v6 ha_arm_ik/solver_dls.py 的原样副本。
等价性口径：
  - “逐值一致”= 不设容差的精确相等（数组用 array_equal，±0 视为相等，NaN 位置一致）；
  - v7 直接调用 LAPACK dpotrf/dpotrs，与 scipy 1.18 的 cho_factor/cho_solve 封装相差 1e-12 量级，
    逐值对比时把 v7 的 _chol_factor/_chol_solve 换回 scipy 版本；不换时按 1e-9 容差对比；
  - 单种子对比时方案1 的四个迭代常量恢复为 v6 取值（停滞 25 次 / 5%、姿态整理 50 步、耐心 3）。
merged-v8 的有意改动在逐值对比中按 v6 口径处理：
  - load_mode="off" 的质心参考（v8 含根 link 质量）在对比时换回 v6 的 pin.centerOfMass 口径；
  - 新关键字 shoulder_joint 不传给 v6（合成模型下与 v6 的 "_Link2" 约定指向同一关节）；
  - 臂展 waist_max_reach 由求解器显式传入，v6 / v8 收到同一个值；
  - 腰部舒适窗只改变传给单种子求解的限位与种子，两边收到相同实参，因此对比保持逐值一致。
"""
import contextlib
import dataclasses
import importlib.util
import math
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial.transform import Rotation

from tests.fixtures import context, make_solver, root_target
from ha_arm_ik import solver as S
import ha_arm_ik.solver_dls as D
from tests import dls_compat
from ha_arm_ik.solver_dls import JointStateSnapshot


def _load_v6_reference():
    name = "solver_dls_v6_reference"
    if name not in sys.modules:
        path = Path(__file__).with_name("reference") / "solver_dls_v6.py"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module          # dataclass 处理需要模块已注册
        spec.loader.exec_module(module)
    return sys.modules[name]


R6 = _load_v6_reference()

V6_ITER_CONSTANTS = {"PIN_IK_STALL_ITERS": 25, "PIN_IK_STALL_RTOL": 0.05,
                     "MAX_SETTLE_EXTRA_ITERS": 50, "SETTLE_PATIENCE": 3}
LAPACK_CHOL = (D._chol_factor, D._chol_solve)     # v7 的 LAPACK 直调实现（导入时保存，免受补丁影响）


def _v6_reference_com(model, data, q_full):
    """v6 / v7 在 load_mode="off" 下的质心参考口径（不含根 link 质量；v8 已修正，仅用于逐值对比）。"""
    import pinocchio as pin
    pin.centerOfMass(model, data, q_full, False)
    return np.array(data.com[0], dtype=float, copy=True)


@contextlib.contextmanager
def v6_numerics(*, cholesky=True, constants=True):
    """临时恢复 v6 的 scipy Cholesky 封装与迭代常量（只影响 v7 模块 D）。"""
    with contextlib.ExitStack() as stack:
        if constants:
            for key, value in V6_ITER_CONSTANTS.items():
                stack.enter_context(patch.object(D, key, value))
        if cholesky:
            stack.enter_context(patch.object(
                D, "_chol_factor", lambda A: cho_factor(A, check_finite=False)))
            stack.enter_context(patch.object(
                D, "_chol_solve", lambda c, b: cho_solve(c, b, check_finite=False)))
        yield


def _copy(value):
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {k: _copy(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_copy(v) for v in value)
    return value


def assert_same(test, a, b, path="value"):
    """递归逐值比较（无容差）：dict / list / tuple / ndarray / 标量。"""
    if isinstance(a, dict) or isinstance(b, dict):
        test.assertIsInstance(a, dict, path)
        test.assertIsInstance(b, dict, path)
        test.assertEqual(sorted(a), sorted(b), path)
        for key in a:
            assert_same(test, a[key], b[key], f"{path}[{key!r}]")
    elif isinstance(a, (list, tuple)) and not isinstance(a, str):
        test.assertEqual(len(a), len(b), path)
        for i, (x, y) in enumerate(zip(a, b)):
            assert_same(test, x, y, f"{path}[{i}]")
    elif isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        x, y = np.asarray(a), np.asarray(b)
        test.assertEqual(x.shape, y.shape, path)
        if x.dtype.kind == "f" or y.dtype.kind == "f":
            same = np.array_equal(x, y, equal_nan=True)
            detail = float(np.nanmax(np.abs(x - y))) if x.size and not same else 0.0
            test.assertTrue(same, f"{path}: max |diff| = {detail:.3e}")
        else:
            test.assertTrue(np.array_equal(x, y), path)
    elif isinstance(a, float) or isinstance(b, float):
        test.assertTrue(a == b or (math.isnan(a) and math.isnan(b)), f"{path}: {a!r} != {b!r}")
    else:
        test.assertEqual(a, b, path)


def solve(s, state, q=None, T=None, **kw):
    target = root_target(s, state, q) if T is None else T
    return s.solve(target, frame="world", joint_state=JointStateSnapshot(state), **kw)


def rotated_target(s, state, q, rotvec):
    T = root_target(s, state, q)
    T[:3, :3] = T[:3, :3] @ Rotation.from_rotvec(rotvec).as_matrix()
    return T


class _SeedReplay:
    """截获求解流程中的每次单种子调用，现场用 v6 参考实现与 v7 各算一遍（无时间预算）并逐值比较；
    同时记录 v6 单种子内部对主任务 / 次任务 / SNS 的全部调用，供 v7 公共函数离线复算。"""

    def __init__(self, test, solver):
        self.test, self.s = test, solver
        self.original = D.pinocchio_ik_one_seed
        self.calls = 0
        self.failures = []          # 求解器会吞掉种子内异常（含断言失败），在此单独记录
        self.kinds = set()
        self.inner = {"task": [], "objective": [], "sns": []}
        self.stats = {"task": 0, "objective": 0, "objective_load": 0, "sns": 0,
                      "modes": set(), "antipodal": 0, "lapack_worst": 0.0}
        self.contexts = {}          # id(v7 上下文) -> (v7 上下文, v6 上下文)
        self.v7_of_v6 = {}          # id(v6 上下文) -> v7 上下文

    def v6_context(self, ctx7):
        if ctx7 is None:
            return None
        key = id(ctx7)
        if key not in self.contexts:
            snap = ctx7["snapshot"]
            snap6 = R6.JointStateSnapshot(
                dict(snap.positions), snap.received_monotonic, snap.source_age_at_receipt_s
            )
            ctx6 = R6.prepare_load_context(
                self.s.pin_model, self.s.pin_maps, self.s._load_config, snap6
            )
            assert_same(self.test, ctx6["q_template"], ctx7["q_template"], "q_template")
            self.contexts[key] = (ctx7, ctx6)
            self.v7_of_v6[id(ctx6)] = ctx7
        return self.contexts[key][1]

    @contextlib.contextmanager
    def _record_inner(self):
        def wrap(name, kind):
            original = getattr(R6, name)

            def recorder(*args, **kwargs):
                # 只复制数组实参；字典实参（pin_maps / load_context）保留引用以便映射到 v7 上下文
                arr = lambda v: v.copy() if isinstance(v, np.ndarray) else v
                record = (tuple(arr(v) for v in args), {k: arr(v) for k, v in kwargs.items()})
                out = original(*args, **kwargs)
                self.inner[kind].append(record + (_copy(out), _copy(kwargs.get("components"))))
                return out

            return patch.object(R6, name, recorder)

        with wrap("dls_task_terms", "task"), wrap(
            "_compute_nullspace_gradient_and_cost", "objective"
        ), wrap("_solve_sns_clamped_step", "sns"):
            yield

    def _v7_context_like(self, ctx6):
        """v6 单种子内部会浅复制上下文并按 gravity_torque_gain 替换 policy；在 v7 侧做同样的派生。"""
        base6, base7 = self.current
        if ctx6 is base6:
            return base7
        derived = dict(base7)
        for key, value in ctx6.items():
            if key != "policy" and value is not base6.get(key):
                derived[key] = value
        pol6, pol7 = ctx6["policy"], base7["policy"]
        changes = {}
        for f in dataclasses.fields(pol6):
            if f.init and f.name != "early_stop_ratio":
                a, b = getattr(pol6, f.name), getattr(pol7, f.name)
                same = np.array_equal(a, b) if isinstance(a, np.ndarray) else a == b
                if not same:
                    changes[f.name] = a
        if changes:
            derived["policy"] = dataclasses.replace(pol7, **changes)
        return derived

    def _check_inner(self):
        """用 v7 公共函数复算刚记录的 v6 内部调用：主任务项 / 次任务（含负荷）/ SNS 逐值一致。"""
        t, st = self.test, self.stats
        for args, kwargs, out, _ in self.inner["task"]:
            assert_same(t, D.dls_task_terms(*args, **kwargs), out, f"task[{st['task']}]")
            st["task"] += 1
            st["modes"].add(args[6] if len(args) > 6 else kwargs.get("mode"))
            st["antipodal"] += bool(out["antipodal"])
        for args, kwargs, out, comps in self.inner["objective"]:
            kwargs = dict(kwargs)
            ctx = kwargs.get("load_context")
            if ctx is not None:
                kwargs["load_context"] = self._v7_context_like(ctx)
                st["objective_load"] += 1
            if comps is not None:
                kwargs["components"] = {}
            got = dls_compat._compute_nullspace_gradient_and_cost(*args, **kwargs)
            assert_same(t, got, out, f"objective[{st['objective']}]")
            if comps is not None:
                assert_same(t, kwargs["components"], comps, f"objective[{st['objective']}] components")
            st["objective"] += 1
        for args, kwargs, out, _ in self.inner["sns"]:
            with v6_numerics(constants=False):
                got = dls_compat._solve_sns_clamped_step(*args, **kwargs)
            assert_same(t, got, out, f"sns[{st['sns']}]")
            # LAPACK 直调（显式恢复 v7 实现；外层求解处于 scipy 补丁下）：与 scipy 封装差 1e-12 量级
            with patch.object(D, "_chol_factor", LAPACK_CHOL[0]), patch.object(
                D, "_chol_solve", LAPACK_CHOL[1]
            ):
                got = dls_compat._solve_sns_clamped_step(*args, **kwargs)
            for x, y in zip(got[:2], out[:2]):
                st["lapack_worst"] = max(st["lapack_worst"], float(np.max(np.abs(x - y))))
            st["sns"] += 1
        for records in self.inner.values():
            records.clear()

    def __call__(self, *args, **kwargs):
        try:
            kwargs = dict(kwargs, deadline=None)
            kw6 = dict(kwargs, load_context=self.v6_context(kwargs.get("load_context")))
            shoulder = kw6.pop("shoulder_joint", None)
            self.test.assertIn(shoulder, (None, D._legacy_shoulder_joint(args[2])))
            self.current = (kw6["load_context"], kwargs.get("load_context"))
            with self._record_inner():
                q6, diag6 = R6.pinocchio_ik_one_seed(*args, **kw6)
            with patch.object(D, "_reference_com", _v6_reference_com):
                q7, diag7 = self.original(*args, **kwargs)
            assert_same(self.test, q7, q6, f"seed {self.calls} q")
            assert_same(self.test, diag7, diag6, f"seed {self.calls} diag")
            self._check_inner()
        except BaseException as exc:
            self.failures.append(f"{type(exc).__name__}: {exc}")
            raise
        self.calls += 1
        self.kinds.add((
            kwargs.get("orientation_mode"),
            kwargs.get("fixed_joint_mask") is not None,
            bool(kwargs.get("waist_load_release")),
            kwargs.get("load_context") is not None,
        ))
        return q7, diag7


class V6EquivalenceTests(unittest.TestCase):
    """方案2（实现优化）与 v6 数值逐值一致：单种子求解、主任务项、次任务、SNS、种子生成。"""

    @classmethod
    def setUpClass(cls):
        cls.replays = []
        rng = np.random.default_rng(11)
        far = []
        for _ in range(3):
            qt = rng.uniform(-2.9, 2.9, size=9)
            qt[:2] = rng.uniform(-0.8, 0.8, size=2)
            far.append(qt)
        cases = [
            # (配置, include_waist, 目标构造, orientation_mode)
            ({}, True, lambda s, st, q0: (q0 + [0.3, -0.2, 0.4, -0.3, 0.2, 0.3, -0.4, 0.2, 0.1], None), "full"),
            ({}, True, lambda s, st, q0: (far[0], None), "full"),
            ({}, True, lambda s, st, q0: (None, rotated_target(s, st, far[1], [0.4, -0.7, 0.2])), "tool_dir"),
            ({}, True, lambda s, st, q0: (None, rotated_target(s, st, far[2], [-0.5, 0.3, 0.6])), "tool_cone"),
            # 工具轴反向 180°：触发对径点逃逸
            ({}, True, lambda s, st, q0: (None, rotated_target(s, st, q0, [0.0, math.pi, 0.0])), "tool_dir"),
            # 锁腰种子超出软阈值 -> 腰部释放（v7 顺序下会重跑 q_prev）
            # （v6 的早停阈值固定默认 0.60，须显式给出 <= soft_ratio 的值；该目标超出锁腰时的手臂可达范围，
            # v13 默认跳过锁腰种子，这里关掉可达性检查以覆盖腰部负荷释放）
            ({"load_soft_ratio": 0.05, "load_early_stop_ratio": 0.05, "dls_locked_reach_check": False}, True,
             lambda s, st, q0: (q0 + [0.2, -0.3, 0.5, 0.4, -0.3, 0.2, 0.3, -0.2, 0.4], None), "full"),
            # 负荷保护关闭：load_context=None 路径
            ({"load_mode": "off"}, True, lambda s, st, q0: (far[0], None), "full"),
            # 无腰部的 7 轴配置
            ({}, False, lambda s, st, q0: (None, rotated_target(
                s, st, q0 + [0.3, -0.2, 0.25, 0.3, -0.4, 0.2, 0.1], [0.3, 0.2, -0.4])), "tool_cone"),
        ]
        for config, include_waist, build, mode in cases:
            cfg = {"load_budget_s": 60.0, **config}
            s, state = make_solver(config=cfg, include_waist=include_waist)
            q0 = np.array([state[n] for n in s.active_joint_names])
            replay = _SeedReplay(None, s)
            cls.replays.append((s, state, q0, build, mode, replay))

    def _run_case(self, index):
        s, state, q0, build, mode, replay = self.replays[index]
        replay.test = self
        q_arg, T = build(s, state, q0)
        with v6_numerics(), patch.object(S, "pinocchio_ik_one_seed", replay):
            solve(s, state, q_arg, T, orientation_mode=mode)
        self.assertEqual(replay.failures, [])
        self.assertGreater(replay.calls, 0)
        search = s.last_load_diagnostics.get("search")
        if search is not None:
            self.assertEqual(search["seed_errors"], 0, search)
        return replay

    def test_seed_solver_matches_v6_exactly(self):
        kinds = set()
        stats = {"task": 0, "objective": 0, "objective_load": 0, "sns": 0, "antipodal": 0,
                 "modes": set(), "lapack_worst": 0.0}
        calls = 0
        for i in range(len(self.replays)):
            with self.subTest(case=i):
                replay = self._run_case(i)
                kinds |= replay.kinds
                calls += replay.calls
                for key, value in replay.stats.items():
                    if key == "modes":
                        stats[key] |= value
                    elif key == "lapack_worst":
                        stats[key] = max(stats[key], value)
                    else:
                        stats[key] += value
        # 单种子：三种姿态模式、锁腰种子、腰部负荷释放、非负荷路径均已覆盖
        self.assertGreater(calls, 10)
        self.assertEqual({k[0] for k in kinds}, {"full", "tool_dir", "tool_cone"})
        self.assertTrue(any(k[1] for k in kinds), "锁腰种子未覆盖")
        self.assertTrue(any(k[2] for k in kinds), "腰部负荷释放未覆盖")
        self.assertTrue(any(not k[3] for k in kinds), "非负荷路径未覆盖")
        # 内部调用：主任务（含对径点）/ 次任务（含与不含负荷）/ SNS
        self.assertGreater(stats["task"], 100)
        self.assertEqual(stats["modes"], {"full", "tool_dir", "tool_cone"})
        self.assertGreater(stats["antipodal"], 0)
        self.assertGreater(stats["objective_load"], 0)
        self.assertLess(stats["objective_load"], stats["objective"])
        self.assertGreater(stats["sns"], 50)
        self.assertLessEqual(stats["lapack_worst"], 1e-9)

    def test_task_evaluator_single_frame_matches_public(self):
        s, state = make_solver()
        rng = np.random.default_rng(5)
        fid = s.pin_ee_frame_id
        axis = np.array([1.0, 0.0, 0.0])
        for mode in ("full", "tool_dir", "tool_cone"):
            for _ in range(5):
                qt = np.clip(rng.uniform(-2.5, 2.5, size=9), s.lower, s.upper)
                T = rotated_target(s, state, qt, rng.uniform(-1, 1, 3))
                terms = D._make_task_terms(s.pin_model, s.pin_data, s.pin_maps, fid, T, mode, axis, 0.2)
                q = np.clip(qt + rng.normal(scale=0.3, size=9), s.lower, s.upper)
                ref = R6.dls_task_terms(s.pin_model, s.pin_data, s.pin_maps, fid, q, T, mode, axis, 0.2)
                assert_same(self, terms(q, jacobian=True, all_frames=False, need_A=True), ref, mode)
                assert_same(self, D.dls_task_terms(s.pin_model, s.pin_data, s.pin_maps, fid, q, T,
                                                   mode, axis, 0.2), ref, mode)
                ref0 = R6.dls_task_terms(s.pin_model, s.pin_data, s.pin_maps, fid, q, T, mode, axis,
                                         0.2, jacobian=False)
                assert_same(self, D.dls_task_terms(s.pin_model, s.pin_data, s.pin_maps, fid, q, T,
                                                   mode, axis, 0.2, jacobian=False), ref0, mode)
                fast = terms(q, jacobian=False)
                self.assertIsNone(fast["J"])
                for key in ("error", "pos_err", "ori_err", "ori_violation", "merit", "antipodal", "q_full"):
                    assert_same(self, fast[key], ref0[key], f"{mode} {key}")

    def test_q_builder_and_vector_norm(self):
        s, state = make_solver()
        context(s, state)
        build = D._make_q_builder(s.pin_model, s.pin_maps)
        rng = np.random.default_rng(9)
        for _ in range(20):
            q = rng.uniform(-3, 3, size=9)
            assert_same(self, build(q), R6.active_q_to_pinocchio_q(s.pin_model, s.pin_maps, q), "q_full")
            x = rng.normal(size=int(rng.integers(1, 12)))
            self.assertEqual(D._vec_norm(x), float(np.linalg.norm(x)))
        with self.assertRaises(ValueError):
            build(np.full(9, np.nan))
        with self.assertRaises(ValueError):
            build(np.zeros(8))

    def test_lapack_cholesky_matches_scipy(self):
        rng = np.random.default_rng(1)
        for n in (1, 3, 6, 9):
            M = rng.normal(size=(n, n))
            A = M @ M.T + 1e-3 * np.eye(n)
            b = rng.normal(size=n)
            B = rng.normal(size=(n, 2))
            c = D._chol_factor(A)
            ref = cho_factor(A, check_finite=False)
            U = np.triu(c)
            np.testing.assert_allclose(U, np.triu(ref[0]), rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(U.T @ U, A, rtol=1e-10, atol=1e-10)
            np.testing.assert_allclose(D._chol_solve(c, b), cho_solve(ref, b), rtol=1e-9, atol=1e-10)
            np.testing.assert_allclose(D._chol_solve(c, B), cho_solve(ref, B), rtol=1e-9, atol=1e-10)
            self.assertFalse(np.shares_memory(c, A))
        with self.assertRaises(np.linalg.LinAlgError):
            D._chol_factor(np.array([[1.0, 2.0], [2.0, 1.0]]))
        with self.assertRaises(np.linalg.LinAlgError):
            D._chol_factor(-np.eye(3))


class ScatterSeedPrefixTests(unittest.TestCase):
    """make_scatter_seeds(max_count) == v6 全量去重结果的前 max_count 个，随机数状态一致。"""

    def check(self, q_prev, q_ready, lower, upper, seed=3):
        rng6 = np.random.default_rng(seed)
        full = R6.make_scatter_seeds(q_prev, q_ready, lower, upper, rng6)
        rng7 = np.random.default_rng(seed)
        assert_same(self, D.make_scatter_seeds(q_prev, q_ready, lower, upper, rng7), full, "full")
        self.assertEqual(rng7.bit_generator.state, rng6.bit_generator.state)
        for count in range(1, len(full) + 3):
            rng7 = np.random.default_rng(seed)
            got = D.make_scatter_seeds(q_prev, q_ready, lower, upper, rng7, max_count=count)
            assert_same(self, got, full[:count], f"max_count={count}")
            self.assertEqual(rng7.bit_generator.state, rng6.bit_generator.state, count)
        return full

    def test_prefix_matches_v6(self):
        lower, upper = np.full(9, -2.9), np.full(9, 2.9)
        q_ready = np.array([.05, .1, .25, -.6, .4, -.8, .3, .6, -.2])
        self.check(q_ready + 0.3, q_ready, lower, upper)
        # q_prev == q_ready：第 2 个种子重复，前缀需扩大
        full = self.check(q_ready.copy(), q_ready, lower, upper)
        self.assertEqual(len(full), 34)
        # 窄限位：大量扰动种子被截到同一边界，前缀多轮扩大
        full = self.check(np.array([0.0005]), np.array([0.0]), np.array([0.0]), np.array([0.001]))
        self.assertLess(len(full), 10)
        # 无界 continuous 关节
        self.check(np.array([0.2, 3.5]), np.array([0.0, 0.0]), np.array([-1.0, -np.inf]),
                   np.array([1.0, np.inf]))

    def test_invalid_max_count(self):
        z = np.zeros(2)
        with self.assertRaises(ValueError):
            D.make_scatter_seeds(z, z, z - 1, z + 1, np.random.default_rng(0), max_count=0)


class EarlyStopDefaultTests(unittest.TestCase):
    """方案1：低负荷早停阈值默认 min(0.70, soft_ratio)；显式配置优先。"""

    def policy(self, **config):
        s, state = make_solver(config=config)
        return context(s, state)["policy"]

    def test_defaults_and_explicit(self):
        self.assertEqual(D.LOAD_EARLY_STOP_RATIO, 0.70)
        self.assertEqual(self.policy().early_stop_ratio, 0.70)
        self.assertEqual(self.policy(load_soft_ratio=0.65).early_stop_ratio, 0.65)
        self.assertEqual(self.policy(load_early_stop_ratio=0.5).early_stop_ratio, 0.5)
        self.assertEqual(self.policy(load_early_stop_ratio=None).early_stop_ratio, 0.70)
        self.assertEqual(self.policy(load_early_stop_ratio=0.60).early_stop_ratio, 0.60)

    def test_explicit_above_soft_ratio_rejected(self):
        with self.assertRaises(ValueError):
            self.policy(load_soft_ratio=0.5, load_early_stop_ratio=0.6)

    def test_iteration_constants(self):
        # v8：姿态整理恢复 v6 的 50 步 / 耐心 3（停滞检测仍为 v7 的 12 次 / 2.5%）
        self.assertEqual((D.PIN_IK_STALL_ITERS, D.MAX_SETTLE_EXTRA_ITERS, D.SETTLE_PATIENCE), (12, 50, 3))
        self.assertEqual(D.PIN_IK_STALL_RTOL, 0.025)


class SeedOrderTests(unittest.TestCase):
    """方案3：q_prev 先行；其解合格且腰部 ≈ q_ready 时跳过等价的锁腰种子。

    这里的目标多为 FK(q_prev)（续算），v9 默认的续算局部优先会在 q_prev 种子后直接停止，
    因此本组测试统一关闭 dls_continuation_local，专门检查 v7 的种子顺序规则（重新定位时仍按此执行）。
    """

    def run_solve(self, config=None, q=None, **kw):
        s, state = make_solver(config={"dls_continuation_local": False, **(config or {})})
        q_sol, _, result = solve(s, state, q, **kw)
        search = s.last_load_diagnostics.get("search") or {}
        per_seed = list(getattr(result, "per_seed", None) or [])
        return s, q_sol, search, per_seed

    def test_equivalent_locked_seed_skipped(self):
        s, q, search, per_seed = self.run_solve()
        self.assertIsNotNone(q, s.last_load_diagnostics)
        self.assertEqual(search["seed_order"], "q_prev_first")
        self.assertTrue(search["locked_seed_skipped"])
        self.assertEqual(search["seeds_attempted"], 1)
        self.assertEqual(search["seed_slots_used"], 2)
        self.assertEqual(search["stop_reason"], "LOW_LOAD_AFTER_REQUIRED_COMPARISON")
        self.assertEqual([p["seed_kind"] for p in per_seed], ["q_prev"])
        waist = np.abs(np.array(q)[:2] - s.q_ready[:2])
        self.assertLessEqual(float(waist.max()), S.LOCKED_SEED_SKIP_TOL)

    def test_v6_order_when_tolerance_none(self):
        s, q, search, per_seed = self.run_solve({"locked_seed_skip_tol": None})
        self.assertIsNotNone(q, s.last_load_diagnostics)
        self.assertEqual(search["seed_order"], "locked_first")
        self.assertFalse(search["locked_seed_skipped"])
        self.assertGreaterEqual(search["seeds_attempted"], 2)
        self.assertEqual(per_seed[0]["seed_kind"], "locked")
        self.assertTrue(per_seed[0]["fixed_waist"])

    def test_locked_seed_runs_when_waist_moves(self):
        # 目标超出锁腰时的手臂可达范围（v13 默认跳过锁腰种子，见 test_v13_changes）；这里检查 v7 的种子顺序
        s, state = make_solver(config={"dls_continuation_local": False, "dls_locked_reach_check": False})
        qt = np.array([state[n] for n in s.active_joint_names])
        qt[:2] += [0.6, -0.6]
        q, _, result = solve(s, state, qt)
        self.assertIsNotNone(q, s.last_load_diagnostics)
        search = s.last_load_diagnostics["search"]
        kinds = [p["seed_kind"] for p in result.per_seed]
        self.assertEqual(kinds[:2], ["q_prev", "locked"])
        self.assertFalse(search["locked_seed_skipped"])
        self.assertGreaterEqual(search["unlocked_seeds_attempted"], 1)

    def test_release_reruns_q_prev_within_seed_limit(self):
        s, state = make_solver(config={"load_soft_ratio": 0.05, "dls_continuation_local": False,
                                       "dls_locked_reach_check": False})
        qt = np.array([state[n] for n in s.active_joint_names])
        qt[:2] += [0.4, -0.3]
        q, _, result = solve(s, state, qt)
        search = s.last_load_diagnostics["search"]
        kinds = [p["seed_kind"] for p in result.per_seed] if result.per_seed else []
        self.assertEqual(kinds[:3], ["q_prev", "locked", "q_prev_release"], search)
        self.assertTrue(search["q_prev_release_rerun"])
        self.assertTrue(search["waist_released_for_load"])
        self.assertLessEqual(search["seed_slots_used"], s.load_search["max_seeds"] if hasattr(s, "load_search") else 4)
        self.assertLessEqual(len(kinds), 4)
        released = [p for p in result.per_seed if p["seed_kind"] == "q_prev_release"]
        self.assertTrue(all(p["waist_released_for_load"] for p in released))

    def test_invalid_tolerance_rejected(self):
        for value in (-0.1, float("nan"), float("inf"), "abc"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                make_solver(config={"locked_seed_skip_tol": value})

    def test_skip_without_load_protection(self):
        s, state = make_solver(config={"load_mode": "off", "dls_continuation_local": False})
        q, _, result = solve(s, state)
        self.assertIsNotNone(q)
        self.assertEqual([p["seed_kind"] for p in result.per_seed], ["q_prev"])
        s, state = make_solver(config={"load_mode": "off", "locked_seed_skip_tol": None,
                                       "dls_continuation_local": False})
        q, _, result = solve(s, state)
        self.assertIsNotNone(q)
        self.assertEqual([p["seed_kind"] for p in result.per_seed][:2], ["locked", "q_prev"])

    def test_no_waist_configuration_unchanged(self):
        s, state = make_solver(include_waist=False, config={"dls_continuation_local": False})
        q, _, result = solve(s, state)
        self.assertIsNotNone(q, s.last_load_diagnostics)
        search = s.last_load_diagnostics["search"]
        self.assertEqual(search["seed_order"], "no_waist")
        self.assertFalse(search["locked_seed_skipped"])
        self.assertTrue(all(p["seed_kind"] != "locked" for p in result.per_seed))


if __name__ == "__main__":
    unittest.main()
