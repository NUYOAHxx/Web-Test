"""v9 质量 / 耗时对比（在 v8 基础上增加真实 ik_candidate 代价与跟踪连续性）：compare.py <eval_pkg_root> a.pkl b.pkl ... [--json out.json]

H = <eval_pkg_root> 的 strict 零空间次任务代价（同一把尺子评估所有版本；交付时用 v7 包根目录），
另报腰部逐关节偏离 q_ready（均值 / 最大 / >0.8 rad 个数）、最小限位裕度 <0.1 个数、负荷峰值、
耗时（ms，run_targets.py 的 repeat 次取最短）、舒适窗回退次数，以及相对第一个 pkl 的配对 dH。
v9 新增：C = 真实 ik_candidate 择优代价（score_candidate + W_TORQUE·torque_cost，不含与 q_prev 有关的平滑项，
即 W_NOM·φ² + W_LIMIT·限位罚 + W_WAIST_RETURN·腰偏离² + W_TORQUE·mean(τ/τ_allow)²），分项 Cw（腰回中）/ Ct（力矩）；
路点跟踪另报单步 max|q - q_prev| 的最大值 / p99 / >0.3 rad 次数。
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
from ha_arm_ik.ik_base import score_candidate  # noqa: E402
from ha_arm_ik.solver import add_torque_metrics_to_candidate  # noqa: E402
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
    c = score_candidate(s.chain, q, s.chain.fk(q), s.psi_ready, q, qr, s.lower, s.upper, apply_waist_return=True)
    add_torque_metrics_to_candidate(c, M, Dd, maps, s.active_joint_names, s.joint_efforts_active)
    return dict(H=float(H), waist=np.abs(q - qr)[waist], margin=float(np.min(np.minimum(q - lo, hi - q)[bounded])),
                peak=float(D.load_physics(M, Dd, maps, q, ctx)["peak_ratio"]),
                C=float(c["cost"]), Cw=float(c["waist_cost"]), Ct=float(c["torque_weighted_cost"]))


runs = [(os.path.basename(p).rsplit(".", 1)[0], pickle.load(open(p, "rb"))) for p in paths]
wnames = [n for n, w in zip(s.active_joint_names, waist) if w]
report = {}
for case in [k for k in runs[0][1] if not k.startswith("_") and all(k in r for _, r in runs)]:
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
            C_mean=round(float(np.mean([m["C"] for m in ok])), 3),
            Cw_mean=round(float(np.mean([m["Cw"] for m in ok])), 3),
            Ct_mean=round(float(np.mean([m["Ct"] for m in ok])), 3),
            ms_mean=round(float(np.mean([r["ms"] for r in rows])), 1),
            ms_median=round(float(np.median([r["ms"] for r in rows])), 1),
            ms_p90=round(float(np.percentile([r["ms"] for r in rows], 90)), 1),
            comfort_fallbacks=sum(bool((r.get("wc") or {}).get("fallback_to_full_bounds")) for r in rows),
            selected_outside_comfort=sum((r.get("wc") or {}).get("selected_in_comfort") is False for r in rows))
        if res.get("_chain_" + case):
            qs, prev, steps = [r["q"] for r in rows], res["_qm"], []
            for qq in qs:
                if qq is not None:
                    steps.append(float(np.max(np.abs(qq - prev))))
                    prev = qq
            steps = np.array(steps)
            entry.update(step_max=round(float(steps.max()), 3), step_p99=round(float(np.percentile(steps, 99)), 3),
                         step_gt_0p3=int((steps > 0.3 + 1e-6).sum()),
                         # v9：续算类求解中没有局部解、回到常规搜索的次数（v9 之前的 pkl 无此诊断，记 None）
                         nonlocal_=(None if not any("cont" in r for r in rows) else
                                    sum((r.get("cont") or {}).get("regime") == "continuation"
                                        and not (r.get("cont") or {}).get("local_used") for r in rows)))
        report[case][tag] = entry
        wtxt = "  ".join(f"{n} mean {e['mean']:.3f} max {e['max']:.3f} >0.8 {e['gt_0p8']}" for n, e in entry["waist"].items())
        print(f"  {tag:12s} succ {entry['success']}  H {entry['H_mean']:.4f}  {wtxt}  margin<0.1 {entry['margin_lt_0p1']}"
              f"  peak {entry['peak_mean']:.3f}  C {entry['C_mean']:.2f} (Cw {entry['Cw_mean']:.2f} Ct {entry['Ct_mean']:.2f})"
              f"  ms mean {entry['ms_mean']:.1f} p90 {entry['ms_p90']:.1f}  fallback {entry['comfort_fallbacks']}"
              + (f"  step max {entry['step_max']:.3f} p99 {entry['step_p99']:.3f} >0.3 {entry['step_gt_0p3']}"
                 f"  nonlocal {entry['nonlocal_']}"
                 if "step_max" in entry else ""))
    base = runs[0][0]
    for tag, _ in runs[1:]:
        A, B = per[base], per[tag]
        idx = [i for i in range(len(A)) if A[i] and B[i]]
        d = np.array([B[i]["H"] - A[i]["H"] for i in idx])
        rel = 100 * d.mean() / max(np.mean([A[i]["H"] for i in idx]), 1e-12)
        dc = np.array([B[i]["C"] - A[i]["C"] for i in idx])
        relc = 100 * dc.mean() / max(np.mean([A[i]["C"] for i in idx]), 1e-12)
        report[case][tag]["vs_" + base] = dict(dH_mean=round(float(d.mean()), 6), dH_rel_pct=round(float(rel), 2),
                                               worse_gt_1e3=int((d > 1e-3).sum()), better_gt_1e3=int((d < -1e-3).sum()),
                                               dC_mean=round(float(dc.mean()), 4), dC_rel_pct=round(float(relc), 2))
        print(f"    {tag} vs {base}: dH {d.mean():+.5f} ({rel:+.2f}%)  worse {int((d > 1e-3).sum())}  better {int((d < -1e-3).sum())}"
              f"  dC {dc.mean():+.3f} ({relc:+.2f}%)")
if json_out:
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump({"eval_root_objective": "strict nullspace cost of eval package", "runs": [t for t, _ in runs],
                   "cases": report}, f, ensure_ascii=False, indent=1)
