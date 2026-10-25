"""probe_steps.py <pkg_root> [json_config] [tracking|tracking_waist] [thr]：逐帧打印单步 > thr 的路点。

与 run_targets.py 相同的正弦路点；每个大步长打印选中种子类别 / 搜索阶段、是否采用续算局部解、腰部变化、
目标步长与各种子结果；续算类求解没有局部解（回到常规搜索）时打印 NONLOCAL。
"""
import json, os, sys, time
import numpy as np
root = os.path.abspath(sys.argv[1]); extra = json.loads(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] else {}
case = sys.argv[3] if len(sys.argv) > 3 else "tracking_waist"
thr = float(sys.argv[4]) if len(sys.argv) > 4 else 0.15
sys.path.insert(0, root); os.chdir(root)
from tests.fixtures import make_solver, root_target
from ha_arm_ik.solver_dls import JointStateSnapshot
cfg = {"load_max_seeds": 10, "dls_max_iter": 500, "load_budget_s": 2.0, **extra}
s, state = make_solver(config=cfg)
qm = np.array([state[n] for n in s.active_joint_names]); names = s.active_joint_names
rng = np.random.default_rng(3)
amp = rng.uniform(0.2, 0.5, size=len(qm))
if case == "tracking_waist": amp[:2] = [0.9, 1.2]; tt = np.linspace(0, 12.0, 240)[1:]
else: amp[:2] = 0; tt = np.linspace(0, 6.0, 120)[1:]
ph = rng.uniform(0, 2*np.pi, size=len(qm)); w = rng.uniform(0.5, 1.0, size=len(qm))
qts = [np.clip(qm + amp*(np.sin(w*t+ph)-np.sin(ph)), -2.8, 2.8) for t in tt]
q_prev = qm.copy(); steps = []; kinds = {}
for k, qt in enumerate(qts):
    T = root_target(s, state, qt)
    q, st, r = s.solve(T, frame="world", joint_state=JointStateSnapshot(state), orientation_mode="full", q_prev=q_prev)
    d = s.last_load_diagnostics
    if q is None:
        print(k, "FAIL", d.get("code")); continue
    q = np.array(q); dq = np.abs(q - q_prev); j = int(np.argmax(dq))
    ps = getattr(r, "per_seed", None) or []; sel = r.get("seed_used")
    p = next((p for p in ps if p.get("seed_idx") == sel), {})
    kind = (p.get("seed_kind"), p.get("search_stage")); kinds[kind] = kinds.get(kind, 0) + 1
    cont = d.get("continuation") or {}
    tstep = np.max(np.abs(qt - (qts[k-1] if k else qm)))
    if not cont.get('local_used') and cont.get('regime') == 'continuation':
        print("  NONLOCAL", k, [(x.get('seed_kind'), x.get('search_stage'), repr(x.get('joint_jump_max_rad')), round(float(x.get('margin', np.nan)),4), x.get('joint_jump_ok')) for x in ps if x.get('search_stage')=='continuation'])
    if dq[j] > thr:
        print(f"{k:3d} step {dq[j]:.3f} {names[j]:12s} kind={kind} local={cont.get('local_used')} regime={cont.get('regime')} "
              f"waist {np.round(q_prev[:2],3)}->{np.round(q[:2],3)} qready_w {np.round(s.q_ready[:2],3)} target_dstep {tstep:.3f} "
              f"seeds={[(x.get('seed_kind'), x.get('search_stage'), bool(x.get('pose_ok')), round(float(x.get('joint_jump_max_rad', np.nan)),3)) for x in ps]}")
    steps.append(dq[j]); q_prev = q
steps = np.array(steps)
print("n", len(steps), "max", steps.max().round(3), "p99", np.percentile(steps, 99).round(3), ">0.3", int((steps > 0.3).sum()), ">0.2", int((steps > 0.2).sum()), kinds)
