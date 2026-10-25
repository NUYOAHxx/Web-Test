"""压力基准：stress.py <pkg_root> <out.pkl> [model] [n] [json_config]

model: two_arm（tests/test_v12_changes 的双臂合成 URDF，左臂 + 腰，限位接近真实人形）| wide（tests/fixtures，±3 rad）
目标 = 关节空间均匀采样（限位内收 0.02 rad）的 FK，必然可达；一次性求解（q_prev = q_ready），三种姿态模式。
记录：是否解出、耗时、失败码、种子数、DLS 总迭代、每个种子 (kind, stop, iters, pose_ok)。
"""
import json
import os
import pickle
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

root, out = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
model = sys.argv[3] if len(sys.argv) > 3 else "two_arm"
n = int(sys.argv[4]) if len(sys.argv) > 4 else 100
extra = json.loads(sys.argv[5]) if len(sys.argv) > 5 and sys.argv[5] else {}
chain = bool(extra.pop("_chain", False))   # True：q_prev = 上一个目标的解（未解出时回到 q_ready）
replay = extra.pop("_replay", None)         # pkl 路径：q_prev = 该结果中上一个目标的解（成对比较用，序列固定）
replay = None if replay is None else pickle.load(open(replay, "rb"))
sys.path.insert(0, root)
os.chdir(root)
from scipy.spatial.transform import Rotation as R  # noqa: E402
from tests.fixtures import make_solver, root_target  # noqa: E402
from ha_arm_ik.solver_dls import JointStateSnapshot  # noqa: E402

tmp = TemporaryDirectory()
if model == "two_arm":
    from tests import test_v12_changes as T
    import ha_arm_ik.robot_model as RM
    path = Path(tmp.name) / "two_arm.urdf"
    path.write_text(T.two_arm_xml())
    configs = T.arm_configs()
    RM.ARM_CONFIGS.clear()
    RM.ARM_CONFIGS.update(configs)
    from ha_arm_ik import solver as S
    cfg = dict(T.SOLVER_CFG, load_max_seeds=10, dls_max_iter=500, load_budget_s=0.5)
    cfg.update(extra)
    s = S.HAArmIKSolver(path, "left", include_waist=True, verbose=False, config=cfg)
    state = {k: 0.0 for k in T.ALL_JOINTS}
    state.update(dict(zip(s.active_joint_names, s.q_ready)))
else:
    cfg = {"load_max_seeds": 10, "dls_max_iter": 500, "load_budget_s": 0.5}
    cfg.update(extra)
    s, state = make_solver(config=cfg)
lo, hi = s.lower.copy(), s.upper.copy()
lo[~np.isfinite(lo)], hi[~np.isfinite(hi)] = -np.pi, np.pi
rng = np.random.default_rng(2024)
cases = {}
for mode in ("full", "tool_dir", "tool_cone"):
    Ts, qs = [], []
    for _ in range(n if mode == "full" else n // 2):
        qt = rng.uniform(lo + 0.02, hi - 0.02)
        Tt = root_target(s, state, qt)
        if mode != "full":
            Tt[:3, :3] = Tt[:3, :3] @ R.from_rotvec(rng.uniform(-1, 1, 3)).as_matrix()
        Ts.append(Tt)
        qs.append(qt)
    cases[mode] = (Ts, qs)
res = {"_root": root, "_model": model, "_cfg": cfg, "_names": list(s.active_joint_names), "_chain": chain, "_replay": replay is not None}
for mode, (Ts, qs) in cases.items():
    rows = []
    q_last = s.q_ready.copy()
    for k, (Tt, qt) in enumerate(zip(Ts, qs)):
        if replay is not None:
            prev = replay[mode][k - 1]["q"] if k > 0 else None
            q_last = s.q_ready.copy() if prev is None else np.array(prev, dtype=float)
        t0 = time.perf_counter()
        q, st, r = s.solve(Tt, frame="world", q_prev=(q_last if (chain or replay is not None) else s.q_ready).copy(),
                           joint_state=JointStateSnapshot(state), orientation_mode=mode)
        if chain and replay is None:
            q_last = s.q_ready.copy() if q is None else np.array(q, dtype=float)
        ms = 1e3 * (time.perf_counter() - t0)
        d = s.last_load_diagnostics
        ps = getattr(r, "per_seed", None) or []
        rows.append(dict(q=None if q is None else np.array(q), qt=qt, ms=ms, status=str(st), code=d.get("code"),
                         peak=d.get("peak_ratio", np.nan), cost=float(r.get("cost", np.nan)) if q is not None else np.nan,
                         seeds=[(p.get("seed_kind"), p.get("stop_reason"), p.get("iters", 0), bool(p.get("pose_ok")),
                                 p.get("search_stage"), float(p.get("final_pos_err", np.nan)),
                                 float(p.get("elapsed_ms", np.nan)), p.get("joint_jump_ok"), p.get("margin")) for p in ps],
                         fail=getattr(r, "failure_detail", "")))
    res[mode] = rows
    ok = [x for x in rows if x["q"] is not None]
    ms = np.array([x["ms"] for x in rows])
    print(f"{model} {mode:9s} solved {len(ok)}/{len(rows)}  ms mean {ms.mean():.1f} med {np.median(ms):.1f} "
          f"p90 {np.percentile(ms, 90):.1f} max {ms.max():.1f}  seeds mean {np.mean([len(x['seeds']) for x in rows]):.1f} "
          f"iters mean {np.mean([sum(p[2] for p in x['seeds']) for x in rows]):.0f}", flush=True)
pickle.dump(res, open(out, "wb"))
tmp.cleanup()
