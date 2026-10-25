"""v12 路点跟踪细节：track_detail.py <pkg_root> [max_joint_jump_rad] [out.json]

与 mirror_bench.py 相同的合成双臂 URDF 与 120 个跟踪路点，左臂与右臂（v12 数值 q_ready）逐路点记录：单步
max|q - q_prev|、续算类别 / 是否用了续算局部解、选中解来自哪个种子（seed_kind / search_stage）、与左臂解镜像的偏差。
max_joint_jump_rad 缺省 = 求解器默认（π）；传 0.25 = v9 文档建议的跟踪设置。
"""
import json
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

root = os.path.abspath(sys.argv[1])
jump = float(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] not in ("", "default") else None
out = sys.argv[3] if len(sys.argv) > 3 else None
sys.path.insert(0, root)
os.chdir(root)
from tests import test_v12_changes as T  # noqa: E402
from tests.fixtures import root_target  # noqa: E402
import ha_arm_ik.robot_model as RM  # noqa: E402
import ha_arm_ik.solver_dls as D  # noqa: E402
from ha_arm_ik import solver as S  # noqa: E402
from ha_arm_ik.solver_dls import JointStateSnapshot  # noqa: E402

tmp = TemporaryDirectory()
path = Path(tmp.name) / "two_arm.urdf"
path.write_text(T.two_arm_xml())
configs = T.arm_configs()
RM.ARM_CONFIGS.clear()
RM.ARM_CONFIGS.update(configs)
cfg = dict(T.SOLVER_CFG, load_max_seeds=10, dls_max_iter=500, load_budget_s=60.0)
if jump is not None:
    cfg["max_joint_jump_rad"] = jump
left = S.HAArmIKSolver(path, "left", include_waist=True, verbose=False, config=cfg)
right = S.HAArmIKSolver(path, "right", include_waist=True, verbose=False, config=cfg)
state_l = {n: 0.0 for n in T.ALL_JOINTS}
state_l.update(dict(zip(left.active_joint_names, left.q_ready)))
state_r = T.mirror_state(state_l)

rng3 = np.random.default_rng(3)
amp = rng3.uniform(0.2, 0.5, size=9)
amp[:2] = [0.5, 0.8]
ph = rng3.uniform(0, 2 * np.pi, size=9)
w = rng3.uniform(0.5, 1.0, size=9)
qs = [np.clip(left.q_ready + amp * (np.sin(w * t + ph) - np.sin(ph)), left.lower + 0.05, left.upper - 0.05)
      for t in np.linspace(0, 12.0, 121)[1:]]


def track(s, state, signs):
    rows, q_prev = [], s.q_ready.copy()
    for qt in qs:
        q, _, r = s.solve(root_target(s, state, signs * qt), frame="world", q_prev=q_prev,
                          joint_state=JointStateSnapshot(state))
        d = s.last_load_diagnostics
        cont = dict(d.get("continuation") or {})
        kind = stage = None
        for p in getattr(r, "per_seed", None) or []:
            if p.get("seed_idx") == r.get("seed_used"):
                kind, stage = p.get("seed_kind"), p.get("search_stage")
        step = None if q is None else float(np.max(np.abs(D.joint_difference(q, q_prev, s.multi_turn_mask))))
        rows.append({"q": None if q is None else np.asarray(q, float), "step": step, "code": d.get("code"),
                     "regime": cont.get("regime"), "local_used": cont.get("local_used"),
                     "seed_kind": kind, "stage": stage, "cost": float(r.get("cost", np.nan))})
        if q is not None:
            q_prev = np.asarray(q, float)
    return rows


rl = track(left, state_l, np.ones(9))
rr = track(right, state_r, T.S9)
dev = [None if a["q"] is None or b["q"] is None
       else float(np.max(np.abs(D.joint_difference(b["q"], T.S9 * a["q"], right.multi_turn_mask))))
       for a, b in zip(rl, rr)]
first = next((k for k, x in enumerate(dev) if x is not None and x > 1e-6), None)
summary = {"max_joint_jump_rad": jump, "first_divergence": first,
           "mirror_of_left": int(sum(x is not None and x <= 1e-6 for x in dev)),
           "dev_gt_0p1": int(sum(x is not None and x > 0.1 for x in dev))}
for tag, rows in (("left", rl), ("right", rr)):
    steps = [r["step"] for r in rows if r["step"] is not None]
    summary[tag] = {"solved": sum(r["q"] is not None for r in rows), "max_step": max(steps),
                    "steps_gt_0p3": int(sum(x > 0.3 for x in steps)),
                    "local_used": int(sum(bool(r["local_used"]) for r in rows)),
                    "regimes": {k: sum(r["regime"] == k for r in rows) for k in {r["regime"] for r in rows}},
                    "cost_mean": float(np.mean([r["cost"] for r in rows if r["q"] is not None]))}
print(json.dumps(summary, default=str))
lo = max(0, (first or 0) - 2)
for k in range(lo, min(len(qs), lo + 8)):
    a, b = rl[k], rr[k]
    print(k, "dev", None if dev[k] is None else round(dev[k], 4),
          "| L step", None if a["step"] is None else round(a["step"], 4), a["regime"], a["local_used"], a["seed_kind"], a["stage"],
          "| R step", None if b["step"] is None else round(b["step"], 4), b["regime"], b["local_used"], b["seed_kind"], b["stage"])
if out:
    Path(out).write_text(json.dumps({"summary": summary, "dev": dev,
                                     "left": [{k: v for k, v in r.items() if k != "q"} for r in rl],
                                     "right": [{k: v for k, v in r.items() if k != "q"} for r in rr]},
                                    indent=1, default=str))
tmp.cleanup()
