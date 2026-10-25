"""merged-v15：单臂续算漂移修复（行为改变）+ 逐位不变提速。

行为改变只在续算（q_prev 随动）阶段：CONTINUATION_POSTURE_SETTLE = False 时续算 DLS 不做达标后的姿态整理，
q_prev 已是可接受解时原样返回（continuation hold）。一次性 / 重定位 / 失败路径 / 配置报错 / 7 轴 / direct-psi
与 v14 逐值一致：486 条离线回放在 v14 模式（CONTINUATION_POSTURE_SETTLE = True）下 0 差异，默认模式只有
跟踪类记录变化（README_v15.md）。提速改动对照 v14 原写法逐位验证（本文件后半部分）。
"""
import unittest
import warnings
from unittest import mock

import numpy as np
from scipy.spatial.transform import Rotation as R

from tests import fixtures as F
from ha_arm_ik import solver as S
import ha_arm_ik.ik_common as C
import ha_arm_ik.solver_dls as D
from ha_arm_ik.ik_common import T_origin, T_prismatic, T_revolute
from ha_arm_ik.solver_dls import JointStateSnapshot


# --------------------------------------------------------------------------- 续算漂移
class ContinuationHoldTests(unittest.TestCase):
    """目标不动、q_prev 已是解：v15 原样返回；目标小步移动：关节变化与手的移动量成正比。"""

    def setUp(self):
        self.s, self.state = F.make_solver()
        self.names = self.s.active_joint_names
        rng = np.random.default_rng(5)
        lo, hi = self.s.search_lower + 0.1, self.s.search_upper - 0.1
        self.q0 = np.clip(self.s.q_ready + rng.normal(0, 0.6, len(self.names)), lo, hi)
        st = dict(self.state)
        st.update(zip(self.names, self.q0))
        self.T0 = F.root_target(self.s, st, self.q0)
        self.s.pin_maps.pop("q_template", None)

    def solve(self, T, q_prev):
        st = dict(self.state)
        st.update(zip(self.names, q_prev))
        q, status, result = self.s.solve(T, frame="world", q_prev=q_prev, joint_state=JointStateSnapshot(st))
        self.assertIsNotNone(q, self.s.last_load_diagnostics)
        return np.asarray(q, dtype=float), str(status), result, dict(self.s.last_load_diagnostics)

    def test_default_is_no_settle(self):
        self.assertFalse(S.CONTINUATION_POSTURE_SETTLE)
        self.assertEqual(D.NO_SETTLE_NULL_STEP_RATIO, 2.0)

    def test_static_target_returns_q_prev_unchanged(self):
        q = self.q0.copy()
        for _ in range(3):
            q_new, status, result, diag = self.solve(self.T0, q)
            np.testing.assert_array_equal(q_new, q)          # 逐位相同：伺服循环里机器人不动
            self.assertEqual(status, "SUCCESS")
            self.assertEqual(diag["continuation"]["regime"], "continuation")
            self.assertTrue(diag["continuation"]["local_used"])
            self.assertTrue(diag["continuation"]["hold"])
            self.assertTrue(diag["search"]["continuation_hold"])
            rows = [(p["seed_kind"], p["search_stage"], p["stop_reason"], p["iters"]) for p in result.per_seed]
            self.assertEqual(rows, [("q_prev", "continuation", "POSE_REACHED", 1)])   # 不跑锁腰续算
            q = q_new

    @mock.patch.object(S, "CONTINUATION_POSTURE_SETTLE", True)
    def test_v14_mode_drifts_on_static_target(self):
        q_new, status, result, diag = self.solve(self.T0, self.q0.copy())
        self.assertGreater(float(np.abs(q_new - self.q0).max()), 0.05)   # v14：每次调用 0.10~0.15 rad
        self.assertFalse(diag["continuation"]["hold"])

    def move(self, steps):
        """目标每次平移 1.2 mm（与 README_v15 的数据相同），返回每次调用的关节最大变化与迭代数。"""
        q, out = self.q0.copy(), []
        for k in range(steps):
            T = self.T0.copy()
            T[:3, 3] += np.array([0.001, 0.0005, -0.0005]) * (k + 1)
            q_new, status, result, diag = self.solve(T, q)
            out.append((float(np.abs(q_new - q).max()), sum(p["iters"] for p in result.per_seed), diag))
            q = q_new
        return out

    def test_small_motion_gives_proportional_steps(self):
        for step, iters, diag in self.move(5):
            self.assertLess(step, 0.01)          # v15：0.002~0.003 rad / 1.2 mm
            self.assertLessEqual(iters, 3)
            self.assertTrue(diag["continuation"]["local_used"])
            self.assertFalse(diag["continuation"]["hold"])   # 目标变了：不是 hold，是收敛到新目标的第一个达标点

    @mock.patch.object(S, "CONTINUATION_POSTURE_SETTLE", True)
    def test_v14_mode_small_motion_drifts(self):
        self.assertGreater(self.move(1)[0][0], 0.05)


class DlsPostureSettleTests(unittest.TestCase):
    """pinocchio_ik_one_seed(posture_settle=False) 的单种子语义。"""

    def setUp(self):
        self.s, self.state = F.make_solver()
        self.ctx = F.context(self.s, self.state)
        self.qm = np.array([self.state[n] for n in self.s.active_joint_names])
        # 种子离开 q_ref（= q_ready = 夹具状态），默认语义下零空间整理有事可做
        self.qs = self.qm + 0.3 * np.array([1, -1, 1, -1, 1, 1, -1, 1, -1])

    def one_seed(self, target, seed, **kw):
        return D.pinocchio_ik_one_seed(self.s.pin_model, self.s.pin_data, self.s.pin_maps,
                                       self.s.pin_ee_frame_id, target, seed, self.s.lower, self.s.upper,
                                       q_ref=self.s.q_ready, include_waist=True, load_context=self.ctx,
                                       max_iter=200, **kw)

    def target(self, q):
        T = F.root_target(self.s, self.state, q)
        self.s.pin_maps["q_template"] = self.ctx["q_template"].copy()
        return T

    def test_seed_solution_returned_as_is(self):
        T = self.target(self.qs)
        q, diag = self.one_seed(T, self.qs, posture_settle=False)
        np.testing.assert_array_equal(q, self.qs)
        self.assertEqual((diag["stop_reason"], diag["iters"]), ("POSE_REACHED", 1))
        self.assertTrue(diag["converged"])
        q14, diag14 = self.one_seed(T, self.qs)                  # 默认 posture_settle=True：照常整理
        self.assertGreater(diag14["iters"], 1)
        self.assertGreater(float(np.abs(q14 - self.qs).max()), 1e-3)

    def near_limit_seed(self):
        qt = self.qs.copy()
        qt[5] = self.s.upper[5] - 0.25 * C.SAFE_MARGIN          # 肘关节（LHand_Link4，有界）贴上限
        return qt, self.target(qt)

    def test_near_limit_seed_keeps_settling(self):
        qt, T = self.near_limit_seed()
        q, diag = self.one_seed(T, qt, posture_settle=False)
        # 达标点离限位 0.005 < SAFE_MARGIN：继续整理到裕度 >= SAFE_MARGIN 的第一个点（3 次迭代，动 0.04 rad；
        # posture_settle=True 整理 51 次、动 0.43 rad）
        self.assertEqual(diag["stop_reason"], "POSE_REACHED")
        self.assertGreater(diag["iters"], 1)
        self.assertTrue(diag["converged"])
        gap = np.minimum(q - self.s.lower, self.s.upper - q)
        self.assertGreaterEqual(float(gap.min()), C.SAFE_MARGIN)
        self.assertLess(float(np.abs(q - qt).max()), 0.1)

    def test_margin_bounds_define_the_margin(self):
        qt, T = self.near_limit_seed()
        inf = np.full(len(qt), np.inf)
        q, diag = self.one_seed(T, qt, posture_settle=False, margin_bounds=(-inf, inf))   # 无界关节不计裕度
        np.testing.assert_array_equal(q, qt)
        self.assertEqual((diag["stop_reason"], diag["iters"]), ("POSE_REACHED", 1))

    def test_margin_bounds_validation(self):
        T = self.target(self.qs)
        n = len(self.qs)
        bad = [(np.zeros(n - 1), np.ones(n - 1)), (np.full(n, np.nan), np.ones(n)), (np.zeros(n), np.ones((n, 1)))]
        for bounds in bad:
            with self.assertRaises(ValueError):
                self.one_seed(T, self.qs, posture_settle=False, margin_bounds=bounds)
        self.one_seed(T, self.qs, margin_bounds=bad[0])   # posture_settle=True 时不使用 margin_bounds

    def test_null_step_capped_before_pose_reached(self):
        T = self.target(self.qm + 0.2)
        calls = []
        core = D._sns_clamped_core

        def spy(J, err, z, null_weight, d_inv, *rest):
            movable = rest[1]
            raw = 0.0 if z is None else null_weight * float(np.linalg.norm((d_inv * z)[movable]))
            calls.append((raw, float(np.linalg.norm(err))))
            return core(J, err, z, null_weight, d_inv, *rest)

        with mock.patch.object(D, "_sns_clamped_core", spy):
            q, diag = self.one_seed(T, self.qm, posture_settle=False)
        self.assertEqual(diag["stop_reason"], "POSE_REACHED")
        self.assertEqual(len(calls), diag["iters"] - 1)          # 达标点直接返回，不再求步
        for raw, err in calls:
            self.assertLessEqual(raw, D.NO_SETTLE_NULL_STEP_RATIO * err * (1 + 1e-12))
        calls.clear()
        with mock.patch.object(D, "_sns_clamped_core", spy):
            self.one_seed(T, self.qm)                            # v14 语义：零空间步不限幅
        self.assertTrue(any(raw > D.NO_SETTLE_NULL_STEP_RATIO * err for raw, err in calls))


# --------------------------------------------------------------------------- 逐位不变提速
def _vt_v14(value):
    T = np.asarray(value, dtype=float)
    if T.shape != (4, 4) or not np.all(np.isfinite(T)):
        raise ValueError("target must be a finite 4x4 transform")
    rot = T[:3, :3]
    if (not np.allclose(T[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9, rtol=0.0)
            or not np.allclose(rot.T @ rot, np.eye(3), atol=1e-6, rtol=0.0)
            or not np.isclose(np.linalg.det(rot), 1.0, atol=1e-6, rtol=0.0)):
        raise ValueError("target rotation/bottom row is not a rigid transform")
    return T.copy()


def _vt4_v14(T, name="T_target"):
    T = np.asarray(T, dtype=float)
    if T.shape != (4, 4):
        raise ValueError(f"{name} must be a 4x4 matrix, got shape {T.shape}")
    if not np.all(np.isfinite(T)):
        raise ValueError(f"{name} contains non-finite values")
    rot = T[:3, :3]
    if not np.allclose(T[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9, rtol=0.0):
        raise ValueError(f"{name} bottom row must be [0, 0, 0, 1]")
    if not (np.allclose(rot.T @ rot, np.eye(3), atol=1e-6, rtol=0.0)
            and np.isclose(np.linalg.det(rot), 1.0, atol=1e-6, rtol=0.0)):
        raise ValueError(f"{name} rotation block is not a proper rotation matrix")
    return T


def _contiguous_range_v14(indices):
    if indices is None:
        return None
    values = np.asarray(indices).ravel()
    if values.size == 0 or values.dtype.kind not in "iu":
        return None
    start = int(values[0])
    if start < 0 or not np.array_equal(values, np.arange(start, start + values.size)):
        return None
    return start, start + int(values.size)


def _fk_all_v14(chain, q):
    q = np.asarray(q, dtype=float)
    T, active_idx = np.eye(4), 0
    link_T, joint_origin_T = {chain.base_link: T.copy()}, {}
    for joint in chain.joints:
        Tj_origin = T @ T_origin(joint["xyz"], joint["rpy"])
        joint_origin_T[joint["name"]] = Tj_origin.copy()
        if joint["type"] in ("revolute", "continuous"):
            Tmotion = T_revolute(joint["axis"], q[active_idx])
            active_idx += 1
        elif joint["type"] == "prismatic":
            Tmotion = T_prismatic(joint["axis"], q[active_idx])
            active_idx += 1
        else:
            Tmotion = np.eye(4)
        T = Tj_origin @ Tmotion
        link_T[joint["child"]] = T.copy()
    return link_T, joint_origin_T


def _fk_pose_v14(chain, q, joint_names):
    q = np.asarray(q, dtype=float)
    wanted, joint_T, T, active_idx = set(joint_names), {}, np.eye(4), 0
    for joint in chain.joints:
        T = T @ T_origin(joint["xyz"], joint["rpy"])
        if joint["name"] in wanted:
            joint_T[joint["name"]] = T.copy()
        if joint["type"] in ("revolute", "continuous"):
            T = T @ T_revolute(joint["axis"], q[active_idx])
            active_idx += 1
        elif joint["type"] == "prismatic":
            T = T @ T_prismatic(joint["axis"], q[active_idx])
            active_idx += 1
    return T, joint_T


def _outcome(f, *args):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            out = np.asarray(f(*args))
            res = ("ok", out.shape, out.tobytes())
        except Exception as exc:  # noqa: BLE001 - 对照异常类型与文本
            res = ("err", type(exc).__name__, str(exc))
    return res, [str(w.message) for w in caught]


def _same(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


class BitwiseSpeedupTests(unittest.TestCase):
    def test_validate_transform_matches_v14(self):
        rng = np.random.default_rng(3)
        cases = [np.eye(4), np.eye(4).tolist(), np.zeros((4, 4)), np.eye(3), np.eye(4)[None]]
        for i in range(800):
            M = np.eye(4)
            M[:3, :3] = R.random(random_state=int(rng.integers(1 << 30))).as_matrix()
            M[:3, 3] = rng.normal(size=3)
            k = i % 7
            if k == 1:      # 正交性容差边界（1e-6 两侧）
                M[:3, :3] *= 1.0 + rng.choice([-1, 1]) * rng.uniform(4e-7, 6e-7)
            elif k == 2:    # 末行容差边界（1e-9 两侧）
                M[3, rng.integers(4)] += rng.choice([-1, 1]) * rng.uniform(0.5e-9, 1.5e-9)
            elif k == 3:
                M[rng.integers(3), rng.integers(3)] += rng.normal() * 1e-6
            elif k == 4:    # 反射（det = -1）
                M[:3, 0] *= -1.0
            elif k == 5:
                M[rng.integers(4), rng.integers(4)] = rng.choice([np.nan, np.inf, -np.inf])
            elif k == 6:    # 溢出
                M[:3, :3] *= rng.choice([1e160, 1e200, -1e155])
            cases.append(M)
        for M in cases:
            self.assertEqual(_outcome(D.validate_transform, M), _outcome(_vt_v14, M))
            self.assertEqual(_outcome(C.validate_transform4x4, M), _outcome(_vt4_v14, M))

    def test_contiguous_range_matches_v14(self):
        rng = np.random.default_rng(1)
        cases = [None, [], [3], [0, 1, 2], [2, 3, 5], [-1, 0, 1], np.arange(4, 9).reshape(5, 1),
                 np.array([1, 2, 3], dtype=np.uint8), np.array([True, False]), [1.0, 2.0], (7, 8),
                 np.array([2 ** 62, 2 ** 62 + 1], dtype=np.uint64), [0, 0], [3, 2, 1]]
        for _ in range(2000):
            k, s = int(rng.integers(0, 8)), int(rng.integers(-2, 20))
            a = np.arange(s, s + k)
            if rng.random() < 0.3 and k > 1:
                a[int(rng.integers(k))] += int(rng.choice([-1, 1]))
            cases.append(a.astype(rng.choice([np.int64, np.int32, np.int16, np.intp])))
        for c in cases:
            new, old = D._contiguous_range(c), _contiguous_range_v14(c)
            self.assertEqual(new, old)
            if new is not None:
                self.assertEqual([type(x) for x in new], [int, int])

    def test_chain_fk_matches_per_joint_v14(self):
        specials = [0.0, -0.0, 1e-300, 1e-12, 1e-8, 3e-7, np.pi, -np.pi, 2 * np.pi, 1e3, -1e3]
        rng = np.random.default_rng(7)
        for include_waist in (True, False):
            s, _ = F.make_solver(include_waist=include_waist)
            chain = s.chain
            names = (chain.s_joint, chain.e_joint, chain.w_joint)
            m = len(chain.active_joints)
            for it in range(300):
                q = (rng.choice(specials, size=m) if it % 3 == 0
                     else rng.uniform(-50, 50, size=m) if it % 3 == 1 else rng.uniform(-np.pi, np.pi, size=m))
                (la, ja), (lb, jb) = _fk_all_v14(chain, q), chain.fk_all(q)
                self.assertEqual((list(la), list(ja)), (list(lb), list(jb)))
                self.assertTrue(all(_same(la[k], lb[k]) for k in la) and all(_same(ja[k], jb[k]) for k in ja))
                (Ta, pa), (Tb, pb) = _fk_pose_v14(chain, q, names), chain.fk_pose_and_joints(q, names)
                self.assertTrue(_same(Ta, Tb) and pa.keys() == pb.keys() and all(_same(pa[k], pb[k]) for k in pa))
                lb[chain.ee_link][:] = 0.0                     # 返回值是副本：改它不影响缓存
            self.assertTrue(_same(chain.fk(q), _fk_all_v14(chain, q)[0][chain.ee_link]))
            self.assertFalse(chain._origin_T(chain.joints[0]).flags.writeable)

    def test_sns_core_matches_reference_on_captured_calls(self):
        """真实求解中捕获的 SNS 调用逐个与 v14 原实现对照（含钳制、固定关节、续算框）。"""
        s, state = F.make_solver(config={"load_max_seeds": 6})
        qm = np.array([state[n] for n in s.active_joint_names])
        calls = []
        core = D._sns_clamped_core

        def spy(*args):
            calls.append([a.copy() if isinstance(a, np.ndarray) else a for a in args])
            return core(*args)

        rng = np.random.default_rng(11)
        with mock.patch.object(D, "_sns_clamped_core", spy):
            for k in range(4):
                qt = np.clip(qm + rng.uniform(-1.2, 1.2, len(qm)), -2.95, 2.95)
                qt[k] = 2.99 if k % 2 else -2.99                 # 目标贴限位：迭代中会钳制
                s.solve(F.root_target(s, state, qt), frame="world", joint_state=JointStateSnapshot(state),
                        q_prev=qm if k < 2 else None)
        self.assertGreater(len(calls), 50)
        clamped = 0
        for args in calls:
            a, b = core(*args), D._sns_clamped_core_reference(*args)
            self.assertTrue(_same(a[0], b[0]) and _same(a[1], b[1]) and a[2] == b[2])
            clamped += bool(np.any((args[7] + a[0] <= args[8]) | (args[7] + a[0] >= args[9])))
        self.assertGreater(clamped, 0)

    def test_sns_core_matches_reference_on_random_problems(self):
        rng = np.random.default_rng(5)
        for it in range(600):
            n, m = int(rng.integers(1, 10)), int(rng.choice([3, 6]))
            J = rng.normal(size=(m, n))
            if it % 4 == 0:
                J = np.asfortranarray(J)
            err = rng.normal(size=m) * rng.choice([1e-6, 1e-2, 1.0])
            z = None if it % 5 == 0 else rng.normal(size=n)
            d_inv = rng.uniform(0.1, 2.0, size=n)
            movable = rng.random(n) > 0.2
            lo = rng.uniform(-2, -0.1, size=n)
            hi = rng.uniform(0.1, 2, size=n)
            q = rng.uniform(lo, hi)
            if it % 3 == 0:                                      # 贴限位
                j = int(rng.integers(n))
                q[j] = hi[j] if rng.random() < 0.5 else lo[j]
            if it % 11 == 0:
                lo[int(rng.integers(n))] = -np.inf
            args = (J, err, z, float(rng.choice([0.0, 0.3, 1.0])), d_inv, None, movable, q, lo, hi,
                    float(rng.choice([0.0, 1e-6, 1e-3])), float(rng.choice([0.05, 0.3, 1.0])))
            a, b = D._sns_clamped_core(*args), D._sns_clamped_core_reference(*args)
            self.assertTrue(_same(a[0], b[0]) and _same(a[1], b[1]) and a[2] == b[2], it)


class VersionTests(unittest.TestCase):
    def test_version(self):
        self.assertEqual(S.SOLVER_VERSION, "merged-v15")
        s, _ = F.make_solver(config={"dls_seed_library_size": 0})
        self.assertEqual(s.last_load_diagnostics["version"], "merged-v15")


if __name__ == "__main__":
    unittest.main()
