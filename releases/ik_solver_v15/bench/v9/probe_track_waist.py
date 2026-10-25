"""probe_track_waist.py <pkg_root> [json_config] [n_steps]：逐步打印 track_waist 路径的择优来源与首个失败原因。"""
import json, os, sys
import numpy as np
root = os.path.abspath(sys.argv[1]); sys.path.insert(0, root); os.chdir(root)
from tests.fixtures import make_solver, root_target
from ha_arm_ik.solver_dls import JointStateSnapshot
cfg = {"load_max_seeds": 10, "dls_max_iter": 500, "load_budget_s": 2.0}
if len(sys.argv) > 2 and sys.argv[2]: cfg.update(json.loads(sys.argv[2]))
n_steps = int(sys.argv[3]) if len(sys.argv) > 3 else 40
s, state = make_solver(config=cfg)
qm = np.array([state[n] for n in s.active_joint_names])
rng = np.random.default_rng(3)
amp = rng.uniform(0.2, 0.5, size=len(qm)); amp[:2] = [0.9, 1.2]
ph = rng.uniform(0, 2 * np.pi, size=len(qm)); w = rng.uniform(0.5, 1.0, size=len(qm))
path = [np.clip(qm + amp * (np.sin(w * t + ph) - np.sin(ph)), -2.8, 2.8) for t in np.linspace(0, 12.0, 240)[1:]]
q_prev = qm.copy()
for k, qt in enumerate(path[:n_steps]):
    T = root_target(s, state, qt)
    q, status, res = s.solve(T, frame="world", joint_state=JointStateSnapshot(state), q_prev=q_prev)
    d = s.last_load_diagnostics
    if q is None:
        print(f"{k:3d} FAIL {d.get('code')} {str(res.failure_detail)[:160]}")
        ps = res.per_seed or []
        print("     seeds:", [(p.get('seed_kind'), p.get('stop_reason'), bool(p.get('pose_ok')), round(p.get('joint_jump_max_rad', np.nan), 3) if 'joint_jump_max_rad' in p else None) for p in ps][:8])
        break
    kinds = [p.get("seed_kind") for p in (res.per_seed or [])]
    sel = res.get("seed_kind") or res.get("seed_used")
    print(f"{k:3d} {status:10s} step {np.max(np.abs(q-q_prev)):.3f} waist {q[0]-qm[0]:+.3f},{q[1]-qm[1]:+.3f} tgt_waist {qt[0]-qm[0]:+.3f},{qt[1]-qm[1]:+.3f} sel {sel} kinds {kinds} cost {res.get('cost', float('nan')):.1f} wst {res.get('waist_cost', float('nan')):.1f} tq {res.get('torque_weighted_cost', float('nan')):.1f}")
    q_prev = q
