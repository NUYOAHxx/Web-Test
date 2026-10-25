"""v8 基准采样：run_targets.py <pkg_root> <out.pkl> [repeat] [json_config]

目标集与 bench/v7/speed.py 完全相同（near_full×40 / far_full×60 / far_tool_dir×50 / far_tool_cone×50 /
路点跟踪×119），对 v6 / v7 / v8 包根目录都能运行。每个目标记录解、耗时（repeat 次取最短）、负荷峰值、
v8 的 waist_comfort 诊断与每个种子的 (seed_kind, stop_reason, iters, pose_ok, search_stage)。
json_config 会合并进求解器 config，例如 '{"dls_waist_comfort": "off"}'。
"""
import json
import os
import pickle
import sys
import time

import numpy as np

root, out = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
repeat = int(sys.argv[3]) if len(sys.argv) > 3 else 1
extra = json.loads(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] else {}
sys.path.insert(0, root)
os.chdir(root)
from scipy.spatial.transform import Rotation as R  # noqa: E402
from tests.fixtures import make_solver, root_target  # noqa: E402
from ha_arm_ik.solver_dls import JointStateSnapshot  # noqa: E402

cfg = {"load_max_seeds": 10, "dls_max_iter": 500, "load_budget_s": 2.0, **extra}
s, state = make_solver(config=cfg)
qm = np.array([state[n] for n in s.active_joint_names])
cases = {}
for name, n, seed in (("near_full", 40, 7), ("far_full", 60, 11), ("far_tool_dir", 50, 13), ("far_tool_cone", 50, 17)):
    rng = np.random.default_rng(seed)
    Ts, modes = [], []
    for _ in range(n):
        if name == "near_full":
            qt = np.clip(qm + rng.uniform(-0.6, 0.6, size=len(qm)), -2.8, 2.8)
        else:
            qt = rng.uniform(-2.9, 2.9, size=len(qm))
            qt[:2] = rng.uniform(-0.8, 0.8, size=2)
        T = root_target(s, state, qt)
        mode = name.split("_", 1)[1]
        if mode != "full":
            T[:3, :3] = T[:3, :3] @ R.from_rotvec(rng.uniform(-1, 1, 3)).as_matrix()
        Ts.append(T)
        modes.append(mode)
    cases[name] = (Ts, modes, False)
rng = np.random.default_rng(3)
amp = rng.uniform(0.2, 0.5, size=len(qm))
amp[:2] = 0
ph = rng.uniform(0, 2 * np.pi, size=len(qm))
w = rng.uniform(0.5, 1.0, size=len(qm))
Ts = [root_target(s, state, np.clip(qm + amp * (np.sin(w * t + ph) - np.sin(ph)), -2.8, 2.8))
      for t in np.linspace(0, 6.0, 120)[1:]]
cases["tracking"] = (Ts, ["full"] * len(Ts), True)

res = {"_qm": qm, "_names": list(s.active_joint_names), "_root": root,
       "_comfort": None if getattr(s, "waist_comfort_lower", None) is None
       else (np.array(s.waist_comfort_lower), np.array(s.waist_comfort_upper))}
for name, (Ts, modes, chain) in cases.items():
    best = None
    for _ in range(repeat):
        q_prev, rows = qm.copy(), []
        for T, mode in zip(Ts, modes):
            kw = {"q_prev": q_prev} if chain else {}
            t0 = time.perf_counter()
            q, _, r = s.solve(T, frame="world", joint_state=JointStateSnapshot(state), orientation_mode=mode, **kw)
            dt = 1e3 * (time.perf_counter() - t0)
            d = s.last_load_diagnostics
            ps = getattr(r, "per_seed", None) or []
            rows.append(dict(q=None if q is None else np.array(q), ms=dt, peak=d.get("peak_ratio", np.nan),
                             code=d.get("code"), wc=dict(d.get("waist_comfort") or {}),
                             seeds=[(p.get("seed_kind"), p.get("stop_reason"), p.get("iters", 0),
                                     bool(p.get("pose_ok")), p.get("search_stage")) for p in ps]))
            if chain and q is not None:
                q_prev = q
        if best is not None:
            for a, b in zip(rows, best):
                a["ms"] = min(a["ms"], b["ms"])
        best = rows
    res[name] = best
pickle.dump(res, open(out, "wb"))
print("done", out, {k: sum(r["q"] is not None for r in v) for k, v in res.items() if not k.startswith("_")})
