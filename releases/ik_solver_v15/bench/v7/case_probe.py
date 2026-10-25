"""case_probe.py <root> <mode> <case> <idx...>：重解指定目标，打印每个种子的诊断"""
import sys, os
import numpy as np
root, mode, case = os.path.abspath(sys.argv[1]), sys.argv[2], sys.argv[3]; idxs = [int(x) for x in sys.argv[4:]]
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
    toks = set(mode.split("+")[1:])
    if "stall" not in toks: D.PIN_IK_STALL_ITERS = 25; D.PIN_IK_STALL_RTOL = 0.05
    if "settle" not in toks: D.MAX_SETTLE_EXTRA_ITERS = 50; D.SETTLE_PATIENCE = 3
    if "early" not in toks: extra["load_early_stop_ratio"] = 0.60
    if "skip" not in toks: extra["locked_seed_skip_tol"] = None
cfg = {"load_max_seeds": 10, "dls_max_iter": 500, "load_budget_s": 2.0, **extra}
s, state = make_solver(config=cfg)
qm = np.array([state[n] for n in s.active_joint_names]); qr = s.q_ready
seeds = {"near_full": 7, "far_full": 11, "far_tool_dir": 13, "far_tool_cone": 17}
n = {"near_full": 40, "far_full": 60, "far_tool_dir": 50, "far_tool_cone": 50}[case]
rng = np.random.default_rng(seeds[case]); Ts = []; ms = []
for _ in range(n):
    if case == "near_full": qt = np.clip(qm + rng.uniform(-0.6, 0.6, size=len(qm)), -2.8, 2.8)
    else: qt = rng.uniform(-2.9, 2.9, size=len(qm)); qt[:2] = rng.uniform(-0.8, 0.8, size=2)
    T = root_target(s, state, qt); m = case.split("_", 1)[1]
    if m != "full": T[:3, :3] = T[:3, :3] @ R.from_rotvec(rng.uniform(-1, 1, 3)).as_matrix()
    Ts.append(T); ms.append(m)
for i in idxs:
    q, st, r = s.solve(Ts[i], frame="world", joint_state=JointStateSnapshot(state), orientation_mode=ms[i])
    d = s.last_load_diagnostics; sr = d.get("search") or {}
    print(f"[{mode}] {case}#{i} peak={d.get('peak_ratio'):.4f} waist={np.abs(np.array(q)[:2]-qr[:2]).max():.3f} stop={sr.get('stop_reason')} cands={sr.get('candidates')}")
    for p in r.per_seed:
        print(f"   {p.get('seed_kind','?'):15s} fixed={p['fixed_waist']!s:5s} it={p.get('iters'):4d} conv={p.get('converged')!s:5s} stop={p.get('stop_reason'):22s} pose_ok={p.get('pose_ok')} dup={p.get('duplicate')} rel={p.get('waist_released_for_load')}")
