"""A/B 交替计时 worker：worker.py <pkg_root> <model> [json_extra]
stdin 每行 "mode k"，stdout 每行 JSON {ms, ok, h, seeds, iters}；与 bench/v13/stress.py 相同的模型 / 配置 / 目标。"""
import hashlib, json, os, sys, time
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
root, model = os.path.abspath(sys.argv[1]), sys.argv[2]
extra = json.loads(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3] else {}
chain = bool(extra.pop("_chain", False))
sys.path.insert(0, root); os.chdir(root)
from scipy.spatial.transform import Rotation as R  # noqa
from tests.fixtures import make_solver, root_target  # noqa
from ha_arm_ik.solver_dls import JointStateSnapshot  # noqa
tmp = TemporaryDirectory()
if model == "two_arm":
    from tests import test_v12_changes as T
    import ha_arm_ik.ik_base as RM
    path = Path(tmp.name) / "two_arm.urdf"; path.write_text(T.two_arm_xml())
    RM.ARM_CONFIGS.clear(); RM.ARM_CONFIGS.update(T.arm_configs())
    from ha_arm_ik import solver as S
    cfg = dict(T.SOLVER_CFG, load_max_seeds=10, dls_max_iter=500, load_budget_s=0.5); cfg.update(extra)
    side = cfg.pop("_side", "left")
    s = S.HAArmIKSolver(path, side, include_waist=True, verbose=False, config=cfg)
    state = {k: 0.0 for k in T.ALL_JOINTS}; state.update(dict(zip(s.active_joint_names, s.q_ready)))
else:
    cfg = {"load_max_seeds": 10, "dls_max_iter": 500, "load_budget_s": 0.5}; cfg.update(extra)
    s, state = make_solver(config=cfg)
lo, hi = s.lower.copy(), s.upper.copy(); lo[~np.isfinite(lo)], hi[~np.isfinite(hi)] = -np.pi, np.pi
rng = np.random.default_rng(2024)
n = int(os.environ.get("AB_N", "200"))
cases = {}
for mode in ("full", "tool_dir", "tool_cone"):
    Ts = []
    for _ in range(n if mode == "full" else n // 2):
        qt = rng.uniform(lo + 0.02, hi - 0.02)
        Tt = root_target(s, state, qt)
        if mode != "full":
            Tt[:3, :3] = Tt[:3, :3] @ R.from_rotvec(rng.uniform(-1, 1, 3)).as_matrix()
        Ts.append(Tt)
    cases[mode] = Ts
s.pin_maps.pop("q_template", None)
last = {m: s.q_ready.copy() for m in cases}
def solve(mode, k):
    Tt = cases[mode][k]
    qp = last[mode] if chain else s.q_ready
    t0 = time.perf_counter()
    q, st, r = s.solve(Tt, frame="world", q_prev=qp.copy(), joint_state=JointStateSnapshot(state), orientation_mode=mode)
    ms = 1e3 * (time.perf_counter() - t0)
    if chain:
        last[mode] = s.q_ready.copy() if q is None else np.array(q, dtype=float)
    ps = getattr(r, "per_seed", None) or []
    h = "none" if q is None else hashlib.sha1(np.asarray(q, dtype=float).tobytes()).hexdigest()[:16]
    return dict(ms=ms, ok=q is not None, h=h, st=str(st), seeds=len(ps), iters=int(sum(p.get("iters", 0) for p in ps)))
for k in range(3):   # 预热
    solve("full", k)
if chain:
    last = {m: s.q_ready.copy() for m in cases}
sys.stdout.write(json.dumps({"ready": True, "sizes": {m: len(v) for m, v in cases.items()}}) + "\n"); sys.stdout.flush()
for line in sys.stdin:
    line = line.strip()
    if not line or line == "quit":
        break
    mode, k = line.split()
    sys.stdout.write(json.dumps(solve(mode, int(k))) + "\n"); sys.stdout.flush()
