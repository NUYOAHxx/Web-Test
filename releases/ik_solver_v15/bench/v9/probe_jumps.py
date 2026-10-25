"""probe_jumps.py <pkg_root> [json_config]：track_waist 路径上 >0.3 rad 的单步跳变来源（选中种子类别、腰部偏离、代价分项）。"""
import json, os, sys
import numpy as np
root = os.path.abspath(sys.argv[1]); sys.path.insert(0, root); os.chdir(root)
from tests.fixtures import make_solver, root_target
from ha_arm_ik.solver_dls import JointStateSnapshot
cfg = {"load_max_seeds": 10, "dls_max_iter": 500, "load_budget_s": 2.0}
if len(sys.argv) > 2 and sys.argv[2]: cfg.update(json.loads(sys.argv[2]))
s, state = make_solver(config=cfg)
qm = np.array([state[n] for n in s.active_joint_names])
rng = np.random.default_rng(3)
amp = rng.uniform(0.2, 0.5, size=len(qm)); amp[:2] = [0.9, 1.2]
ph = rng.uniform(0, 2 * np.pi, size=len(qm)); w = rng.uniform(0.5, 1.0, size=len(qm))
path = [np.clip(qm + amp * (np.sin(w * t + ph) - np.sin(ph)), -2.8, 2.8) for t in np.linspace(0, 12.0, 240)[1:]]
q_prev = qm.copy(); kinds_sel = {}; locked_steps = 0
for k, qt in enumerate(path):
    T = root_target(s, state, qt)
    q, status, res = s.solve(T, frame="world", joint_state=JointStateSnapshot(state), q_prev=q_prev)
    if q is None:
        print(k, "FAIL", s.last_load_diagnostics.get("code")); continue
    ps = res.per_seed or []
    si = res.get("seed_used")
    kind = ps[si]["seed_kind"] if si is not None and si < len(ps) else res.get("source")
    kinds_sel[kind] = kinds_sel.get(kind, 0) + 1
    step = np.abs(q - q_prev)
    if step.max() > 0.3:
        print(f"{k:3d} step {step.max():.3f}@{s.active_joint_names[int(step.argmax())]} sel {kind} waist_dev {np.abs(q[:2]-qm[:2]).max():.3f} prev_waist_dev {np.abs(q_prev[:2]-qm[:2]).max():.3f} "
              f"cost {res['cost']:.1f} psi {res['psi_cost']:.1f} sm {res['smooth_cost']:.2f} wst {res['waist_cost']:.1f} tq {res['torque_weighted_cost']:.1f}")
    q_prev = q
print("selected kinds:", kinds_sel)
