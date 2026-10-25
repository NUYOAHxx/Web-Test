"""v8 质量 / 耗时对比：compare.py <eval_pkg_root> a.pkl b.pkl ... [--json out.json]

H = <eval_pkg_root> 的 strict 零空间次任务代价（同一把尺子评估所有版本；交付时用 v7 包根目录），
另报腰部逐关节偏离 q_ready（均值 / 最大 / >0.8 rad 个数）、最小限位裕度 <0.1 个数、负荷峰值、
耗时（ms，run_targets.py 的 repeat 次取最短）、舒适窗回退次数，以及相对第一个 pkl 的配对 dH。
"""
import json
import os
import pickle
import sys

import numpy as np

args = sys.argv[1:]
json_out = None
if "--json" in args:
    i = args.index("--json")
    json_out = os.path.abspath(args[i + 1])
    del args[i:i + 2]
eval_root, paths = os.path.abspath(args[0]), [os.path.abspath(p) for p in args[1:]]
sys.path.insert(0, eval_root)
os.chdir(eval_root)
import pinocchio as pin  # noqa: E402
from tests.fixtures import make_solver  # noqa: E402
import ha_arm_ik.solver_dls as D  # noqa: E402
from ha_arm_ik.solver_dls import JointStateSnapshot  # noqa: E402
# merged-v14 起整函数形式的次任务入口移到 tests/dls_compat.py（生产代码不再使用）
_nullspace_gradient_and_cost = getattr(D, "_compute_nullspace_gradient_and_cost", None)
if _nullspace_gradient_and_cost is None:
    from tests.dls_compat import _compute_nullspace_gradient_and_cost as _nullspace_gradient_and_cost  # noqa: E402

s, state = make_solver(config={"load_max_seeds": 10, "dls_max_iter": 500, "load_budget_s": 2.0})
M, Dd, maps = s.pin_model, s.pin_data, s.pin_maps
ctx = D.prepare_load_context(M, maps, s._load_config, JointStateSnapshot(state))
maps["q_template"] = ctx["q_template"].copy()
v = np.array(maps["active_v_indices"])
qr, lo, hi = s.q_ready, s.search_lower, s.search_upper
bounded = np.isfinite(lo) & np.isfinite(hi)
waist = np.array([n in D.WAIST_JOINT_NAMES for n in s.active_joint_names])
ns_gain = np.where(waist, D.WAIST_NULL_GAIN, D.NULLSPACE_GAIN)
com_ref = D.load_physics(M, Dd, maps, qr, ctx)["com"]


def metrics(q):
    full = D.active_q_to_pinocchio_q(M, maps, q)
    _, H = _nullspace_gradient_and_cost(
        M, Dd, full, q, qr, lo, hi, ns_gain, v, maps["effort_limits"], np.ones(len(q), bool), True, None,
        com_ref, D.GRAVITY_TORQUE_GAIN, D.COM_XY_GAIN, D.JOINT_LIMIT_GAIN, load_context=ctx, pin_maps=maps,
        derivatives=False)
    return dict(H=float(H), waist=np.abs(q - qr)[waist], margin=float(np.min(np.minimum(q - lo, hi - q)[bounded])),
                peak=float(D.load_physics(M, Dd, maps, q, ctx)["peak_ratio"]))


runs = [(os.path.basename(p).rsplit(".", 1)[0], pickle.load(open(p, "rb"))) for p in paths]
wnames = [n for n, w in zip(s.active_joint_names, waist) if w]
report = {}
for case in [k for k in runs[0][1] if not k.startswith("_")]:
    print(f"== {case}")
    report[case], per = {}, {}
    for tag, res in runs:
        rows = res[case]
        ms_ = [metrics(r["q"]) if r["q"] is not None else None for r in rows]
        per[tag] = ms_
        ok = [m for m in ms_ if m is not None]
        W = np.array([m["waist"] for m in ok])
        entry = dict(
            success=f"{len(ok)}/{len(rows)}", H_mean=round(float(np.mean([m["H"] for m in ok])), 5),
            waist={n: dict(mean=round(float(W[:, j].mean()), 3), max=round(float(W[:, j].max()), 3),
                           gt_0p8=int((W[:, j] > 0.8).sum())) for j, n in enumerate(wnames)},
            waist_any_gt_0p8=int((W.max(axis=1) > 0.8).sum()),
            margin_lt_0p1=int(sum(m["margin"] < 0.1 for m in ok)),
            peak_mean=round(float(np.mean([m["peak"] for m in ok])), 4),
            ms_mean=round(float(np.mean([r["ms"] for r in rows])), 1),
            ms_median=round(float(np.median([r["ms"] for r in rows])), 1),
            ms_p90=round(float(np.percentile([r["ms"] for r in rows], 90)), 1),
            comfort_fallbacks=sum(bool((r.get("wc") or {}).get("fallback_to_full_bounds")) for r in rows),
            selected_outside_comfort=sum((r.get("wc") or {}).get("selected_in_comfort") is False for r in rows))
        report[case][tag] = entry
        wtxt = "  ".join(f"{n} mean {e['mean']:.3f} max {e['max']:.3f} >0.8 {e['gt_0p8']}" for n, e in entry["waist"].items())
        print(f"  {tag:12s} succ {entry['success']}  H {entry['H_mean']:.4f}  {wtxt}  margin<0.1 {entry['margin_lt_0p1']}"
              f"  peak {entry['peak_mean']:.3f}  ms mean {entry['ms_mean']:.1f} p90 {entry['ms_p90']:.1f}"
              f"  fallback {entry['comfort_fallbacks']}")
    base = runs[0][0]
    for tag, _ in runs[1:]:
        A, B = per[base], per[tag]
        idx = [i for i in range(len(A)) if A[i] and B[i]]
        d = np.array([B[i]["H"] - A[i]["H"] for i in idx])
        rel = 100 * d.mean() / max(np.mean([A[i]["H"] for i in idx]), 1e-12)
        report[case][tag]["vs_" + base] = dict(dH_mean=round(float(d.mean()), 6), dH_rel_pct=round(float(rel), 2),
                                               worse_gt_1e3=int((d > 1e-3).sum()), better_gt_1e3=int((d < -1e-3).sum()))
        print(f"    {tag} vs {base}: dH {d.mean():+.5f} ({rel:+.2f}%)  worse {int((d > 1e-3).sum())}  better {int((d < -1e-3).sum())}")
if json_out:
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump({"eval_root_objective": "strict nullspace cost of eval package", "runs": [t for t, _ in runs],
                   "cases": report}, f, ensure_ascii=False, indent=1)
