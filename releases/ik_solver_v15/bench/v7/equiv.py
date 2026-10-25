"""逐位等价性采样：equiv.py <pkg_root> <v6|v7compat|v7> <out.pkl>
v7compat = v7 代码 + 方案1/3 关闭（4 个迭代常量回 v6、early_stop 0.60、locked_seed_skip_tol=None）+ Cholesky 换回 scipy。"""
import sys, os, pickle, time
import numpy as np
root, mode, out = os.path.abspath(sys.argv[1]), sys.argv[2], os.path.abspath(sys.argv[3])
sys.path.insert(0, root); os.chdir(root)
from tests.fixtures import make_solver, root_target
import ha_arm_ik.solver_dls as D
from ha_arm_ik.solver_dls import JointStateSnapshot
from scipy.spatial.transform import Rotation as R
from scipy.linalg import cho_factor, cho_solve

extra = {}
if mode == "v7compat":
    D.PIN_IK_STALL_ITERS = 25; D.PIN_IK_STALL_RTOL = 0.05; D.MAX_SETTLE_EXTRA_ITERS = 50; D.SETTLE_PATIENCE = 3
    D._chol_factor = lambda A: cho_factor(A, check_finite=False)
    D._chol_solve = lambda c, b: cho_solve(c, b, check_finite=False)
    extra = {"load_early_stop_ratio": 0.60, "locked_seed_skip_tol": None}

def strip(ps):
    rows = []
    for p in ps:
        p = {k: v for k, v in p.items() if k != "elapsed_ms"}
        p.pop("seed_kind", None)
        rows.append(p)
    return rows

def run_case(s, state, targets, modes, q_prevs=None, chain_prev=False):
    rows = []
    q_prev = None
    for i, (T, m) in enumerate(zip(targets, modes)):
        kw = {}
        if chain_prev:
            kw["q_prev"] = q_prev
        q, st, r = s.solve(T, frame="world", joint_state=JointStateSnapshot(state), orientation_mode=m, **kw)
        d = s.last_load_diagnostics
        search = dict(d.get("search") or {})
        for k in ("seed_slots_used", "seed_order", "locked_seed_skipped", "q_prev_release_rerun"):
            search.pop(k, None)
        rows.append(dict(q=None if q is None else np.array(q, copy=True), status=str(st), code=d.get("code"),
                         peak=d.get("peak_ratio"), search=search, per_seed=strip(getattr(r, "per_seed", None) or []),
                         detail=getattr(r, "failure_detail", None)))
        if chain_prev and q is not None:
            q_prev = q
    return rows

results = {}
base_cfg = {"load_max_seeds": 10, "dls_max_iter": 500, "load_budget_s": 30.0, **extra}
s, state = make_solver(config=base_cfg)
qm = np.array([state[n] for n in s.active_joint_names])
for name, n, seed in (("near_full", 40, 7), ("far_full", 60, 11), ("far_tool_dir", 50, 13), ("far_tool_cone", 50, 17)):
    rng = np.random.default_rng(seed); Ts = []; ms = []
    for _ in range(n):
        if name == "near_full": qt = np.clip(qm + rng.uniform(-0.6, 0.6, size=len(qm)), -2.8, 2.8)
        else: qt = rng.uniform(-2.9, 2.9, size=len(qm)); qt[:2] = rng.uniform(-0.8, 0.8, size=2)
        T = root_target(s, state, qt); m = name.split("_", 1)[1]
        if m != "full": T[:3, :3] = T[:3, :3] @ R.from_rotvec(rng.uniform(-1, 1, 3)).as_matrix()
        Ts.append(T); ms.append(m)
    results[name] = run_case(s, state, Ts, ms)
# 180° 对径：目标工具轴与 q_prev 处工具轴正好相反
T0 = root_target(s, state, qm); rng = np.random.default_rng(23); Ts = []; ms = []
for k in range(16):
    qt = np.clip(qm + rng.uniform(-0.4, 0.4, size=len(qm)), -2.8, 2.8)
    T = root_target(s, state, qt); T[:3, :3] = T0[:3, :3] @ R.from_euler("y", np.pi).as_matrix()
    Ts.append(T); ms.append("tool_dir" if k % 2 == 0 else "tool_cone")
results["antipodal"] = run_case(s, state, Ts, ms)
# 路点跟踪（腰不动），q_prev 逐点传递
rng = np.random.default_rng(3); amp = rng.uniform(0.2, 0.5, size=len(qm)); amp[:2] = 0
ph = rng.uniform(0, 2*np.pi, size=len(qm)); w = rng.uniform(0.5, 1.0, size=len(qm))
Ts = [root_target(s, state, np.clip(qm + amp*(np.sin(w*t+ph)-np.sin(ph)), -2.8, 2.8)) for t in np.linspace(0, 6.0, 120)[1:]]
results["tracking"] = run_case(s, state, Ts, ["full"]*len(Ts), chain_prev=True)
# 非 strict（load_mode=off）与不含腰
for name, cfg_extra, kw in (("off_full", {"load_mode": "off"}, {}), ("nowaist_full", {}, {"include_waist": False})):
    s2, state2 = make_solver(config={**base_cfg, **cfg_extra}, **kw)
    qm2 = np.array([state2[n] for n in s2.active_joint_names]); rng = np.random.default_rng(31); Ts = []
    for _ in range(30):
        qt = rng.uniform(-2.9, 2.9, size=len(qm2))
        if kw.get("include_waist", True): qt[:2] = rng.uniform(-0.8, 0.8, size=2)
        Ts.append(root_target(s2, state2, qt))
    results[name] = run_case(s2, state2, Ts, ["full"]*len(Ts))
pickle.dump(results, open(out, "wb"))
print(mode, {k: len(v) for k, v in results.items()})
