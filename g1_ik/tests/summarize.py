"""Summarise results/benchmark.json into a markdown table (results/summary.md)."""
import json
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
B = json.load(open(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results", "benchmark.json")))
rows = B["rows"]
names = []
for r in rows:
    if r["scenario"] not in names:
        names.append(r["scenario"])


def f(v, fmt):
    return "-" if v is None or (isinstance(v, float) and np.isnan(v)) else fmt.format(v)


lines = ["| 场景 | n | 成功率 | 最终模式分布 | 位置误差 中位/最大 [mm] | 姿态误差 最大 [deg] | 腰部位移 中位/最大 [rad] | 最大力矩利用率 中位/最大 | 最小 mesh 距离 最小 [mm] | 胶囊约束被违反但 mesh 通过 | 求解时间 中位/P90/最大 [s] |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
for n in names:
    R = [r for r in rows if r["scenario"] == n]
    ok = [r for r in R if r["success"]]
    t = np.array([r["time"] for r in R])
    modes = {}
    for r in ok:
        modes[r["final_mode"]] = modes.get(r["final_mode"], 0) + 1
    md = ", ".join(f"{k}:{v}" for k, v in sorted(modes.items())) or "-"
    if ok:
        pe = np.array([r["pos_err"] for r in ok]) * 1e3
        re = np.degrees([r["rot_err"] for r in ok])
        wd = np.array([r["waist_disp"] for r in ok])
        ut = np.array([r["max_util"] for r in ok])
        dm = np.array([r["min_mesh_dist"] for r in ok]) * 1e3
        cv = sum(bool(r.get("capsule_violation")) for r in ok)
        cells = [f"{np.median(pe):.2f}/{pe.max():.2f}", f"{re.max():.2f}", f"{np.median(wd):.3f}/{wd.max():.3f}",
                 f"{np.median(ut):.2f}/{ut.max():.2f}", f"{dm.min():.1f}", f"{cv}/{len(ok)}"]
    else:
        cells = ["-"] * 6
    lines.append(f"| {n} | {len(R)} | {len(ok)}/{len(R)} ({len(ok) / len(R):.0%}) | {md} | " + " | ".join(cells)
                 + f" | {np.median(t):.2f}/{np.percentile(t, 90):.2f}/{t.max():.2f} |")
txt = "\n".join(lines)
# escalation reasons
esc = {}
for r in rows:
    for s in r["stages"]:
        if s.get("reason"):
            key = (r["scenario"], s["mode"], s["reason"].split(" (")[0].split(" 0")[0])
            esc[key] = esc.get(key, 0) + 1
txt += "\n\n升级(escalation)原因统计:\n" + "\n".join(f"- {k[0]}: {k[1]} -> 下一级, 原因 '{k[2]}' x{v}" for k, v in sorted(esc.items()))
# per-layer time (mean per request)
lay = {}
for r in rows:
    for s in r["stages"]:
        for k, v in s["times"].items():
            lay.setdefault((r["scenario"], k), []).append(v)
txt += "\n\n各层耗时 (每个 stage 平均, s):\n"
for n in names:
    txt += f"- {n}: " + ", ".join(f"{k}={np.mean(lay[(n, k)]):.3f}" for k in ["seed", "qp", "nlp", "validate"] if (n, k) in lay) + "\n"
parts = B.get("parts", {})
if parts:
    p = next(iter(parts.values()))
    txt += f"\n初始化 IKSolver: {p['init_time']:.2f}s；预热(构建 CasADi/IPOPT NLP，按 mode/pair 集缓存): 每个进程 {np.mean([v['warmup_time'] for v in parts.values()]):.1f}s，单个 NLP 构建平均 {np.mean([v['nlp_build_time'] / max(v['nlp_builds'], 1) for v in parts.values()]):.2f}s\n"
print(txt)
open(os.path.join(ROOT, "results", "summary.md"), "w").write(txt + "\n")
