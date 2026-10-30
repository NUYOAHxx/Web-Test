"""v12 右臂基准（v9 代码 + 右臂 q_ready 数值）：mirror_bench.py <pkg_root> [out.json]

合成双臂 URDF（tests/test_v12_changes.two_arm_xml：右臂轴向量与左臂相同、限位镜像），对同一组目标的镜像分别求解：
  left      左臂（v9 = v12）
  right     右臂，v12（ARM_CONFIGS["right"]["q_ready"] = 左臂镜像数值；扰动噪声与左臂相同，不镜像）
  right_v9  右臂，v9 原写法（q_ready 取左臂数值）
目标集、求解参数与 bench/v10/mirror_bench.py 相同（近处 40 / 远处 60 / 路点跟踪 120），结果可以直接对照。
统计成功数、与“左臂解的镜像”的偏差、臂角差、腰部偏离、负荷峰值、耗时；右臂解不是左臂解镜像的目标，
另外逐个比较两侧解的择优代价 C（IKResult["cost"]）、负荷峰值、腰部偏离、限位裕度和臂角偏差 |phi|。
"""
import json
import os
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

root = os.path.abspath(sys.argv[1])
out = sys.argv[2] if len(sys.argv) > 2 else None
sys.path.insert(0, root)
os.chdir(root)
from tests import test_v12_changes as T  # noqa: E402
from tests.fixtures import root_target  # noqa: E402
import ha_arm_ik.ik_base as RM  # noqa: E402
import ha_arm_ik.solver_dls as D  # noqa: E402
from ha_arm_ik import solver as S  # noqa: E402
from ha_arm_ik.solver_dls import JointStateSnapshot  # noqa: E402

tmp = TemporaryDirectory()
path = Path(tmp.name) / "two_arm.urdf"
path.write_text(T.two_arm_xml())
q_ready_right = T.ARM_CONFIGS_FILE["right"]["q_ready"].copy()
RM.ARM_CONFIGS.clear()
RM.ARM_CONFIGS.update(T.arm_configs(T.Q_READY_LEFT_V9, q_ready_right))
cfg = dict(T.SOLVER_CFG, load_max_seeds=10, dls_max_iter=500, load_budget_s=60.0)
left = S.HAArmIKSolver(path, "left", include_waist=True, verbose=False, config=cfg)
right = S.HAArmIKSolver(path, "right", include_waist=True, verbose=False, config=cfg)
right_v9 = S.HAArmIKSolver(path, "right", include_waist=True, verbose=False,
                           config=dict(cfg, q_ready=np.r_[0.0, 0.0, T.Q_READY_LEFT_V9]))
s9 = T.S9

state_l = {n: 0.0 for n in T.ALL_JOINTS}
state_l.update(dict(zip(left.active_joint_names, left.q_ready)))
state_r = T.mirror_state(state_l)
state_r9 = dict(state_r)
state_r9.update(dict(zip(right_v9.active_joint_names, right_v9.q_ready)))

Mm = T.M
p_l = root_target(left, state_l, left.q_ready)[:3, 3]
p_r = root_target(right, state_r, right.q_ready)[:3, 3]
p_r9 = root_target(right_v9, state_r9, right_v9.q_ready)[:3, 3]
ready = {
    "left_q_ready": left.q_ready.tolist(), "right_q_ready": right.q_ready.tolist(),
    "right_q_ready_v9": right_v9.q_ready.tolist(),
    "psi_ready": {"left": left.psi_ready, "right": right.psi_ready, "right_v9": right_v9.psi_ready},
    "tcp_mirror_err_mm": float(1e3 * np.linalg.norm(p_r - Mm @ p_l)),
    "tcp_mirror_err_v9_mm": float(1e3 * np.linalg.norm(p_r9 - Mm @ p_l)),
    "waist_comfort": {"left": [left.waist_comfort_lower[:2].tolist(), left.waist_comfort_upper[:2].tolist()],
                      "right": [right.waist_comfort_lower[:2].tolist(), right.waist_comfort_upper[:2].tolist()]},
}


def run(s, state, Ts, q0, chain):
    rows, q_prev = [], np.array(q0, dtype=float)
    for Tt in Ts:
        t0 = time.perf_counter()
        q, _, r = s.solve(Tt, frame="world", q_prev=q_prev, joint_state=JointStateSnapshot(state))
        ms = 1e3 * (time.perf_counter() - t0)
        d = s.last_load_diagnostics
        q = None if q is None else np.asarray(q, dtype=float)
        rows.append({"q": q, "ms": ms, "peak": d.get("peak_ratio", np.nan), "code": d.get("code"),
                     "psi": None if q is None else float(s.chain.psi(q)),
                     "cost": float(r.get("cost", np.nan)), "waist_dev": float(r.get("waist_dev", np.nan)),
                     "margin": float(r.get("margin", np.nan)), "phi_abs": float(r.get("phi_abs", np.nan))})
        if chain and q is not None:
            q_prev = q
    return rows


rng = np.random.default_rng(31)
sets = {}
qs = []
for _ in range(40):
    qt = left.q_ready + rng.uniform(-0.6, 0.6, size=9)
    qs.append(np.clip(qt, left.lower + 0.05, left.upper - 0.05))
sets["near"] = (qs, False)
qs = []
for _ in range(60):
    qt = rng.uniform(-2.0, 2.0, size=9)
    qt[:2] = rng.uniform(-0.6, 0.6, size=2)
    qs.append(np.clip(qt, left.lower + 0.05, left.upper - 0.05))
sets["far"] = (qs, False)
rng3 = np.random.default_rng(3)
amp = rng3.uniform(0.2, 0.5, size=9)
amp[:2] = [0.5, 0.8]
ph = rng3.uniform(0, 2 * np.pi, size=9)
w = rng3.uniform(0.5, 1.0, size=9)
qs = [np.clip(left.q_ready + amp * (np.sin(w * t + ph) - np.sin(ph)), left.lower + 0.05, left.upper - 0.05)
      for t in np.linspace(0, 12.0, 121)[1:]]
sets["tracking"] = (qs, True)

report = {"ready": ready, "sets": {}}
for name, (qs, chain) in sets.items():
    Tl = [root_target(left, state_l, q) for q in qs]
    Tr = [root_target(right, state_r, s9 * q) for q in qs]
    Tr9 = [root_target(right_v9, state_r9, s9 * q) for q in qs]
    rl = run(left, state_l, Tl, left.q_ready, chain)
    rr = run(right, state_r, Tr, right.q_ready, chain)
    rr9 = run(right_v9, state_r9, Tr9, right_v9.q_ready, chain)
    entry = {"n": len(qs)}
    for tag, rows, s in (("left", rl, left), ("right", rr, right), ("right_v9", rr9, right_v9)):
        ok = [r for r in rows if r["q"] is not None]
        e = {"solved": len(ok), "ms_median": float(np.median([r["ms"] for r in rows])),
             "peak_mean": float(np.nanmean([r["peak"] for r in ok])) if ok else None,
             "cost_mean": float(np.mean([r["cost"] for r in ok])) if ok else None,
             "waist_dev_mean": float(np.mean([np.max(np.abs(r["q"][:2] - s.q_ready[:2])) for r in ok])) if ok else None}
        if tag != "left":
            diffs, psi_d, same, cmp = [], [], 0, []
            for k, (a, b) in enumerate(zip(rl, rows)):
                if a["q"] is None or b["q"] is None:
                    if (a["q"] is None) != (b["q"] is None):
                        cmp.append({"k": k, "left_solved": a["q"] is not None, "right_solved": b["q"] is not None})
                    continue
                d = float(np.max(np.abs(D.joint_difference(b["q"], s9 * a["q"], s.multi_turn_mask))))
                diffs.append(d)
                psi_d.append(abs(np.angle(np.exp(1j * (b["psi"] + a["psi"])))))
                same += d <= 1e-6
                if d > 1e-6 and tag == "right":
                    cmp.append({"k": k, "dev_rad": d, "code": [a["code"], b["code"]],
                                "cost": [a["cost"], b["cost"]], "peak": [a["peak"], b["peak"]],
                                "waist_dev": [a["waist_dev"], b["waist_dev"]], "margin": [a["margin"], b["margin"]],
                                "phi_abs": [a["phi_abs"], b["phi_abs"]]})
            e.update({"mirror_of_left": same, "max_dev_rad": float(max(diffs)) if diffs else None,
                      "dev_gt_0p1": int(sum(d > 0.1 for d in diffs)),
                      "psi_dev_mean_rad": float(np.mean(psi_d)) if psi_d else None,
                      "psi_dev_max_rad": float(max(psi_d)) if psi_d else None})
            if tag == "right":
                e["not_mirrored"] = cmp
        if chain:
            steps = []
            prev = s.q_ready
            for r in rows:
                if r["q"] is not None:
                    steps.append(float(np.max(np.abs(D.joint_difference(r["q"], prev, s.multi_turn_mask)))))
                    prev = r["q"]
            e["max_step"] = float(max(steps)) if steps else None
            e["max_step_index"] = int(np.argmax(steps)) if steps else None
            e["max_step_after_first"] = float(max(steps[1:])) if len(steps) > 1 else None
            e["steps_gt_0p3"] = int(sum(x > 0.3 for x in steps))
        entry[tag] = e
    report["sets"][name] = entry
    print(name, json.dumps(entry, default=float))
print("ready", json.dumps(ready, default=float))
if out:
    Path(out).write_text(json.dumps(report, indent=2, default=float))
tmp.cleanup()
