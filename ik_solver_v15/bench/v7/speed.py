"""计时与质量：speed.py <pkg_root> <v6|plan2[+stall][+settle][+early][+skip]|v7> <out.pkl> [repeat]
plan2 = v7 代码 + 方案1/3 关闭（只保留实现优化，Cholesky 用 LAPACK）。"""
import sys, os, pickle, time, json
import numpy as np
root, mode, out = os.path.abspath(sys.argv[1]), sys.argv[2], os.path.abspath(sys.argv[3])
repeat = int(sys.argv[4]) if len(sys.argv) > 4 else 1
sys.path.insert(0, root); os.chdir(root)
from tests.fixtures import make_solver, root_target
import ha_arm_ik.solver_dls as D
from ha_arm_ik.solver_dls import JointStateSnapshot
from scipy.spatial.transform import Rotation as R
extra = {}
# 可选常量覆盖：<mode>@NAME=value[,NAME=value]，例如 v7@PIN_IK_STALL_RTOL=0.025
mode, _, overrides = mode.partition("@")
for item in filter(None, overrides.split(",")):
    key, value = item.split("=")
    setattr(D, key, type(getattr(D, key))(value))
if mode.startswith("plan2"):
    # plan2[+stall][+settle][+early][+skip]：在“仅实现优化”基础上逐项打开方案1/3
    toks = set(mode.split("+")[1:])
    if "stall" not in toks: D.PIN_IK_STALL_ITERS = 25; D.PIN_IK_STALL_RTOL = 0.05
    if "settle" not in toks: D.MAX_SETTLE_EXTRA_ITERS = 50; D.SETTLE_PATIENCE = 3
    if "early" not in toks: extra["load_early_stop_ratio"] = 0.60
    if "skip" not in toks: extra["locked_seed_skip_tol"] = None
cfg = {"load_max_seeds": 10, "dls_max_iter": 500, "load_budget_s": 2.0, **extra}
s, state = make_solver(config=cfg)
qm = np.array([state[n] for n in s.active_joint_names]); qr = s.q_ready
cases = {}
for name, n, seed in (("near_full", 40, 7), ("far_full", 60, 11), ("far_tool_dir", 50, 13), ("far_tool_cone", 50, 17)):
    rng = np.random.default_rng(seed); Ts = []; ms = []
    for _ in range(n):
        if name == "near_full": qt = np.clip(qm + rng.uniform(-0.6, 0.6, size=len(qm)), -2.8, 2.8)
        else: qt = rng.uniform(-2.9, 2.9, size=len(qm)); qt[:2] = rng.uniform(-0.8, 0.8, size=2)
        T = root_target(s, state, qt); m = name.split("_", 1)[1]
        if m != "full": T[:3, :3] = T[:3, :3] @ R.from_rotvec(rng.uniform(-1, 1, 3)).as_matrix()
        Ts.append(T); ms.append(m)
    cases[name] = (Ts, ms, False)
rng = np.random.default_rng(3); amp = rng.uniform(0.2, 0.5, size=len(qm)); amp[:2] = 0
ph = rng.uniform(0, 2*np.pi, size=len(qm)); w = rng.uniform(0.5, 1.0, size=len(qm))
Ts = [root_target(s, state, np.clip(qm + amp*(np.sin(w*t+ph)-np.sin(ph)), -2.8, 2.8)) for t in np.linspace(0, 6.0, 120)[1:]]
cases["tracking"] = (Ts, ["full"]*len(Ts), True)
res = {}
for name, (Ts, ms, chain) in cases.items():
    best_ms = None; rows = None
    for rep in range(repeat):
        q_prev = qm.copy(); times = []; rr = []
        for T, m in zip(Ts, ms):
            kw = {"q_prev": q_prev} if chain else {}
            t0 = time.perf_counter()
            q, st, r = s.solve(T, frame="world", joint_state=JointStateSnapshot(state), orientation_mode=m, **kw)
            times.append(1e3 * (time.perf_counter() - t0))
            d = s.last_load_diagnostics; sr = d.get("search") or {}
            ps = getattr(r, "per_seed", None) or []
            rr.append(dict(q=None if q is None else np.array(q), peak=d.get("peak_ratio", np.nan), code=d.get("code"),
                           seeds=len(ps), iters=sum(p.get("iters", 0) for p in ps), skipped=bool(sr.get("locked_seed_skipped")),
                           ref_prev=q_prev.copy()))
            if chain and q is not None: q_prev = q
        best_ms = times if best_ms is None else [min(a, b) for a, b in zip(best_ms, times)]
        rows = rr
    for r_, t_ in zip(rows, best_ms): r_["ms"] = t_
    res[name] = rows
pickle.dump(res, open(out, "wb"))
summary = {}
for name, rows in res.items():
    ok = [r for r in rows if r["q"] is not None]; ms_ = [r["ms"] for r in rows]
    summary[name] = dict(succ=f"{len(ok)}/{len(rows)}", med=round(float(np.median(ms_)), 2), p90=round(float(np.percentile(ms_, 90)), 2),
                         mean=round(float(np.mean(ms_)), 2), seeds=round(float(np.mean([r["seeds"] for r in rows])), 2),
                         iters=round(float(np.mean([r["iters"] for r in rows])), 1))
print(mode, json.dumps(summary))
