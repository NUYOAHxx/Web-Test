"""一致性回放：equiv_run.py <pkg_root> <out.pkl>
在给定包目录下跑固定场景（构造属性、一次性求解、路点跟踪、配置变体、7 轴 direct-psi、失败路径、
物理评估 / 工作空间诊断、配置报错），把全部输出规范化（浮点按位）后存盘，供 equiv_cmp.py 逐值比较。
"""
import contextlib, enum, hashlib, io, os, pickle, sys, time, traceback
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np

root, out = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
only = set(sys.argv[3].split(",")) if len(sys.argv) > 3 and sys.argv[3] else None
sys.path.insert(0, root)
os.chdir(root)
from scipy.spatial.transform import Rotation as Rot  # noqa: E402
import ha_arm_ik.robot_model as RM  # noqa: E402
from ha_arm_ik import solver as S  # noqa: E402
from ha_arm_ik.solver_dls import JointStateSnapshot  # noqa: E402
from tests import fixtures as F  # noqa: E402
from tests import test_v12_changes as T  # noqa: E402
from ha_arm_ik.ik_candidate import IKResult  # noqa: E402
import re  # noqa: E402

IGNORE = {"elapsed_ms", "version", "frozen_snapshot_received_monotonic", "received_monotonic"}


def canon(x, depth=0):
    if depth > 40:
        return ("deep",)
    if isinstance(x, IKResult):
        return ("IKResult", canon(dict(x), depth + 1), str(x.status), str(x.failure_reason),
                x.failure_detail, canon(x.per_seed, depth + 1))
    if isinstance(x, dict) or hasattr(x, "items") and callable(x.items) and not isinstance(x, np.ndarray):
        try:
            items = list(x.items())
        except Exception:
            return ("obj", type(x).__name__)
        return ("d", tuple(sorted(((str(k), canon(v, depth + 1)) for k, v in items if str(k) not in IGNORE),
                                  key=lambda kv: kv[0])))
    if isinstance(x, np.ndarray):
        if x.dtype == object:
            return ("ndo", x.shape, tuple(canon(v, depth + 1) for v in x.ravel()))
        return ("nd", x.dtype.str, x.shape, np.ascontiguousarray(x).tobytes())
    if isinstance(x, (list, tuple)):
        return ("l", tuple(canon(v, depth + 1) for v in x))
    if isinstance(x, (set, frozenset)):
        return ("s", tuple(sorted(repr(canon(v, depth + 1)) for v in x)))
    if isinstance(x, (bool, np.bool_)):
        return ("b", bool(x))
    if isinstance(x, (int, np.integer)):
        return ("i", int(x))
    if isinstance(x, (float, np.floating)):
        v = float(x)
        return ("f", "nan" if v != v else v.hex())
    if isinstance(x, enum.Enum):
        return ("e", type(x).__name__, str(x.value))
    if x is None or isinstance(x, str):
        return x
    return ("obj", type(x).__name__)


RESULTS = {}


def record(key, fn):
    try:
        with contextlib.redirect_stdout(io.StringIO()) as buf:
            val = fn()
        RESULTS[key] = ("ok", canon(val), re.sub(r"/tmp/[^/\s]+", "<TMP>", buf.getvalue()))
    except Exception as exc:  # noqa: BLE001
        RESULTS[key] = ("err", type(exc).__name__, str(exc))


def solver_attrs(s):
    names = ["q_ready", "q_prev", "lower", "upper", "search_lower", "search_upper", "waist_comfort_lower",
             "waist_comfort_upper", "waist_comfort_info", "multi_turn_mask", "psi_ready", "psi_ready_error",
             "dls_arm_max_reach", "dls_waist_max_reach", "dls_locked_reach", "max_joint_jump", "joint_jump_policy",
             "locked_seed_skip_tol", "active_joint_names", "base_link", "ee_link", "urdf_has_world_mount",
             "T_world_from_base", "T_base_from_world", "g_mounted_base", "mount_quat", "workspace_max_radius",
             "workspace_anchor_local_xyz", "workspace_max_j7_radius_from_anchor", "joint_efforts_active",
             "use_ssik", "use_pinocchio_ik", "use_torque_cost", "use_direct_psi", "use_local_seeds",
             "dls_continuation_max_step", "dls_seed_library_size", "dls_seed_library_seeds",
             "dls_seed_library_min_separation", "dls_waist_comfort_patience", "pin_base_frame_id", "pin_ee_frame_id"]
    out = {n: getattr(s, n, "<missing>") for n in names}
    lib = getattr(s, "seed_library", None)
    out["seed_library"] = None if lib is None else hashlib.sha256(
        lib["q"].tobytes() + lib["p"].tobytes() + lib["R"].tobytes()).hexdigest() + f":{lib['base_joint']}:{lib['size']}"
    out["pin_maps_keys"] = sorted(k for k in (s.pin_maps or {}) if not k.startswith("_"))
    return out


def solve_seq(prefix, s, state, targets, mode="full", chain=False, frame="world", **kw):
    q_last = s.q_ready.copy()
    for k, Tt in enumerate(targets):
        key = f"{prefix}/{mode}/{k}"
        snap = None if state is None else JointStateSnapshot(state)
        holder = {}

        def run(Tt=Tt, q_last=q_last.copy(), snap=snap):
            q, st, r = s.solve(Tt, frame=frame, q_prev=q_last, orientation_mode=mode,
                               **({} if snap is None else {"joint_state": snap}), **kw)
            holder["q"] = q
            return {"q": q, "status": st, "result": r, "diag": s.last_load_diagnostics,
                    "q_prev_after": s.q_prev}
        record(key, run)
        if chain:
            q = holder.get("q")
            q_last = s.q_ready.copy() if q is None else np.array(q, dtype=float)


def two_arm_solver(side, extra=None, **ctor):
    tmp = TemporaryDirectory()
    path = Path(tmp.name) / "two_arm.urdf"
    path.write_text(T.two_arm_xml())
    RM.ARM_CONFIGS.clear()
    RM.ARM_CONFIGS.update(T.arm_configs())
    cfg = dict(T.SOLVER_CFG, load_max_seeds=10, dls_max_iter=500, load_budget_s=60.0)
    cfg.update(extra or {})
    s = S.HAArmIKSolver(path, side, include_waist=ctor.pop("include_waist", True), verbose=False, config=cfg, **ctor)
    s._tmp = tmp
    state = {k: 0.0 for k in T.ALL_JOINTS}
    if side == "right":
        state = T.mirror_state(state) if hasattr(T, "mirror_state") else state
    state = {**{k: 0.0 for k in T.ALL_JOINTS}, **dict(zip(s.active_joint_names, s.q_ready))}
    for k in list(state):
        state.setdefault(k, 0.0)
    return s, state


def all_names_state(s, state):
    full = dict(state)
    for n in s.pin_maps["all_names"]:
        full.setdefault(n, 0.0)
    return full


def random_targets(s, state, n, mode, rng, margin=0.02):
    lo, hi = s.search_lower.copy(), s.search_upper.copy()
    lo[~np.isfinite(lo)], hi[~np.isfinite(hi)] = -np.pi, np.pi
    Ts = []
    for _ in range(n):
        qt = rng.uniform(lo + margin, hi - margin)
        Tt = F.root_target(s, state, qt)
        if mode != "full":
            Tt[:3, :3] = Tt[:3, :3] @ Rot.from_rotvec(rng.uniform(-1, 1, 3)).as_matrix()
        Ts.append(Tt)
    s.pin_maps.pop("q_template", None)
    return Ts


def track_targets(s, state, n, amp=0.5, waist_amp=0.0):
    q0 = s.q_ready.copy()
    Ts = []
    for k in range(n):
        ph = 2 * np.pi * k / n
        dq = amp * np.sin(ph + np.arange(len(q0)) * 0.7) * 0.4
        wmask = np.array([nm in RM.WAIST_JOINT_NAMES for nm in s.active_joint_names])
        dq[wmask] = waist_amp * np.sin(ph)
        Ts.append(F.root_target(s, state, np.clip(q0 + dq, s.search_lower + 0.01, s.search_upper - 0.01)))
    s.pin_maps.pop("q_template", None)
    return Ts


def scen_two(name, side, extra=None, n_full=16, n_tool=8, n_track=0, waist_amp=0.0, ctor=None, kw=None):
    if only and name not in only:
        return
    rng = np.random.default_rng(sum(map(ord, name)))
    s, state = two_arm_solver(side, extra, **(ctor or {}))
    state = all_names_state(s, state)
    record(f"{name}/attrs", lambda: solver_attrs(s))
    for mode, n in (("full", n_full), ("tool_dir", n_tool), ("tool_cone", n_tool)):
        if n:
            solve_seq(f"{name}/oneshot", s, state, random_targets(s, state, n, mode, rng), mode=mode, **(kw or {}))
    if n_track:
        solve_seq(f"{name}/track", s, state, track_targets(s, state, n_track, waist_amp=waist_amp), chain=True,
                  **(kw or {}))
    s._tmp.cleanup()


t0 = time.time()
scen_two("two_left", "left", n_full=24, n_tool=12, n_track=40, waist_amp=0.4)
scen_two("two_right", "right", n_full=24, n_tool=12, n_track=40, waist_amp=0.4)
scen_two("two_left_jump1", "left", {"max_joint_jump_rad": 1.0}, n_full=16, n_tool=0, n_track=20, waist_amp=0.4)
scen_two("two_left_nolib", "left", {"dls_seed_library_size": 0}, n_full=16, n_tool=6)
scen_two("two_left_v12flags", "left", {"dls_seed_library_size": 0, "dls_waist_comfort_early_fallback": False,
                                       "dls_jump_limit_rescue": False, "dls_locked_reach_check": False},
         n_full=12, n_tool=4)
scen_two("two_left_nocomfort", "left", {"dls_waist_comfort": None, "locked_seed_skip_tol": None,
                                        "dls_continuation_local": False}, n_full=12, n_tool=4, n_track=16,
         waist_amp=0.3)
scen_two("two_left_explicit", "left", {"dls_waist_comfort": {"Hip_Link1": 0.3, "Body_Link": 0.5},
                                       "dls_virtual_half_windows": {"Body_Link": 2.0},
                                       "joint_jump_policy": "prefer", "max_joint_jump_rad": {"Body_Link": 0.8},
                                       "dls_seed_library_continuation": True},
         n_full=12, n_tool=4, n_track=12, waist_amp=0.3)
scen_two("two_left_off", "left", {"load_mode": "off"}, n_full=12, n_tool=4, n_track=12,
         kw={"allow_approximate": True})
scen_two("two_left_strict_approx", "left", {}, n_full=8, n_tool=0, kw={"allow_approximate": True})

# 不可达目标（近似回退 / 失败码）
if not only or "two_unreach" in only:
    s, state = two_arm_solver("left")
    state = all_names_state(s, state)
    Tf = F.root_target(s, state, s.q_ready)
    s.pin_maps.pop("q_template", None)
    for k, d in enumerate((1.5, 3.0)):
        Tu = Tf.copy(); Tu[0, 3] += d
        for approx in (False, True):
            solve_seq(f"two_unreach/{k}/{approx}", s, state, [Tu], allow_approximate=approx)
    s2, state2 = two_arm_solver("left", {"load_mode": "off"})
    for k, d in enumerate((1.5,)):
        Tu = Tf.copy(); Tu[0, 3] += d
        for approx in (False, True):
            solve_seq(f"two_unreach_off/{k}/{approx}", s2, None, [Tu], allow_approximate=approx)
    # 失败路径：缺状态 / 预算非法 / 局部坐标
    record("two_fail/nostate", lambda: s.solve(Tf, frame="world")[1:])
    record("two_fail/nostate_diag", lambda: s.last_load_diagnostics)
    sb, stb = two_arm_solver("left", {"load_budget_s": -1.0})
    record("two_fail/budget", lambda: (sb.solve(Tf, frame="world", joint_state=JointStateSnapshot(all_names_state(sb, stb)))[1:],
                                       dict(sb.last_load_diagnostics)))
    record("two_local", lambda: s.solve(np.linalg.inv(s._solve_T_root_base) @ Tf if s._solve_T_root_base is not None
                                        else Tf, frame="local", joint_state=JointStateSnapshot(state))[:2])
    record("two_eval_physics", lambda: s.evaluate_physics(s.q_ready, joint_state=JointStateSnapshot(state)))
    record("two_eval_physics_off", lambda: s2.evaluate_physics(s2.q_ready))
    record("two_workspace", lambda: s.workspace_diagnostic(Tf, frame="world"))
    record("two_workspace_tcp", lambda: s.workspace_diagnostic(Tf, frame="world", target_is_tcp=True,
                                                               tcp_offset_xyz=[0.0, 0.0, 0.1]))
    record("two_check_ws", lambda: s.check_workspace(Tf, frame="world"))
    record("two_print_info", lambda: S.HAArmIKSolver(s.urdf, "left", include_waist=True, verbose=True,
                                                     config=dict(T.SOLVER_CFG)).psi_ready)
    record("two_update_state", lambda: s.update_joint_state(JointStateSnapshot(state)).positions)
    record("two_close", lambda: (s.close(), s.pin_model))

# 构造报错（类型与消息）
if not only or "cfg_errors" in only:
    bad = [{"load_mode": "x"}, {"dls_continuation_local": 1}, {"dls_continuation_max_step": 0},
           {"dls_continuation_max_step": True}, {"dls_seed_library_size": -1}, {"dls_seed_library_size": 1.5},
           {"dls_seed_library_seeds": True}, {"dls_seed_library_min_separation": -0.1},
           {"dls_waist_comfort_early_fallback": "yes"}, {"dls_waist_comfort_patience": 0},
           {"dls_jump_limit_rescue": None}, {"dls_locked_reach_check": 0}, {"dls_seed_library_continuation": 1},
           {"dls_waist_comfort_continuation_tol": (0.1,)}, {"joint_jump_policy": "x"},
           {"max_joint_jump_rad": -1}, {"max_joint_jump_rad": {"nope": 1.0}}, {"locked_seed_skip_tol": -1},
           {"dls_waist_comfort": "bad"}, {"dls_waist_comfort": {"LHand_Link1": 0.2}},
           {"dls_virtual_half_windows": {"nope": 1.0}}, {"position_bounds": {"Hip_Link1": [0.5, 0.4]}},
           {"q_ready": [9.0] * 9}, {"gravity_root": [0, 0, 0]}]
    for k, b in enumerate(bad):
        record(f"cfg_errors/{k}", lambda b=b: solver_attrs(two_arm_solver("left", b)[0]))
    record("cfg_errors/side", lambda: two_arm_solver("middle"))

# 宽限位夹具（含 continuous 关节）与 7 轴模式
if not only or "wide" in only:
    rng = np.random.default_rng(7)
    s, state = F.make_solver(config={"load_max_seeds": 6, "dls_max_iter": 200})
    record("wide/attrs", lambda: solver_attrs(s))
    for mode, n in (("full", 12), ("tool_dir", 6), ("tool_cone", 6)):
        solve_seq("wide/oneshot", s, state, random_targets(s, state, n, mode, rng), mode=mode)
    solve_seq("wide/track", s, state, track_targets(s, state, 12, waist_amp=0.3), chain=True)
    s7, state7 = F.make_solver(include_waist=False, config={"load_max_seeds": 4, "dls_max_iter": 150})
    record("wide7/attrs", lambda: solver_attrs(s7))
    solve_seq("wide7/oneshot", s7, state7, random_targets(s7, state7, 8, "full", rng))

if not only or "direct" in only:
    rng = np.random.default_rng(11)
    xml, names = F.robot_xml(other=True, continuous=True)
    tmp = TemporaryDirectory()
    path = Path(tmp.name) / "robot.urdf"
    path.write_text(xml)
    qref = np.array([.05, .1, .25, -.6, .4, -.8, .3, .6, -.2])
    S.ARM_CONFIGS["left"] = {"base_link": "link_1", "ee_link": "tcp", "s_joint": "LHand_Link2",
                             "e_joint": "LHand_Link4", "w_joint": "LHand_Link6", "mount_xyz": np.zeros(3),
                             "mount_rpy": np.zeros(3), "q_ready": qref[2:]}
    all_names = names + ["OtherJoint"]
    for tag, extra in (("off", {"load_mode": "off"}), ("off_local", {"load_mode": "off", "use_local_seeds": True})):
        cfg = {"load_allowable_efforts": {n: 300. for n in all_names}, "load_massless_links": ["tcp"],
               "load_budget_s": 3., "load_state_max_age_s": 20., "load_max_seeds": 4, "dls_max_iter": 100,
               "q_ready": qref[2:], **extra}
        sd = S.HAArmIKSolver(path, "left", use_direct_psi=True, include_waist=False, verbose=False, config=cfg)
        state = dict(zip(names, qref)); state["OtherJoint"] = .4
        record(f"direct_{tag}/attrs", lambda: solver_attrs(sd))
        targets = random_targets(sd, state, 6, "full", rng)
        solve_seq(f"direct_{tag}/oneshot", sd, None, targets, frame="local")
        solve_seq(f"direct_{tag}/tool", sd, None, random_targets(sd, state, 3, "tool_dir", rng), mode="tool_dir",
                  frame="local")
    tmp.cleanup()

pickle.dump(RESULTS, open(out, "wb"))
errs = sum(1 for v in RESULTS.values() if v[0] == "err")
print(f"{len(RESULTS)} records ({errs} errors) in {time.time() - t0:.1f}s -> {out}")
