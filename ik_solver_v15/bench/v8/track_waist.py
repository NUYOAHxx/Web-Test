"""腰部大幅路点跟踪（俯仰超出舒适窗）：track_waist.py <pkg_root> [json_config]

239 个路点：各关节绕 q_ready 做正弦运动（腰部幅值 0.9 / 1.2 rad，其余 0.2~0.5 rad），q_prev = 上一个解。
统计成功数、单步最大关节变化（max|q - q_prev|）、舒适窗回退次数与求解类别（v8）。
对 v6 / v7 / v8 包根目录都能运行；json_config 合并进求解器 config，例如 '{"max_joint_jump_rad": 0.25}'。
"""
import json
import os
import sys
import time
from collections import Counter

import numpy as np

root = os.path.abspath(sys.argv[1])
sys.path.insert(0, root)
os.chdir(root)
from tests.fixtures import make_solver, root_target  # noqa: E402
from ha_arm_ik.solver_dls import JointStateSnapshot  # noqa: E402

cfg = {"load_max_seeds": 10, "dls_max_iter": 500, "load_budget_s": 2.0}
if len(sys.argv) > 2 and sys.argv[2]:
    cfg.update(json.loads(sys.argv[2]))
s, state = make_solver(config=cfg)
qm = np.array([state[n] for n in s.active_joint_names])
rng = np.random.default_rng(3)
amp = rng.uniform(0.2, 0.5, size=len(qm))
amp[:2] = [0.9, 1.2]
ph = rng.uniform(0, 2 * np.pi, size=len(qm))
w = rng.uniform(0.5, 1.0, size=len(qm))
path = [np.clip(qm + amp * (np.sin(w * t + ph) - np.sin(ph)), -2.8, 2.8)
        for t in np.linspace(0, 12.0, 240)[1:]]
q_prev = qm.copy()
steps, ms, pitch, regimes = [], [], [], Counter()
ok = fb = 0
for qt in path:
    T = root_target(s, state, qt)
    t0 = time.perf_counter()
    q, _, _ = s.solve(T, frame="world", joint_state=JointStateSnapshot(state), q_prev=q_prev)
    ms.append(1e3 * (time.perf_counter() - t0))
    wc = s.last_load_diagnostics.get("waist_comfort") or {}
    regimes[wc.get("regime")] += 1
    if q is None:
        continue
    ok += 1
    steps.append(np.max(np.abs(q - q_prev)))
    pitch.append(q[0])
    fb += bool(wc.get("fallback_to_full_bounds"))
    q_prev = q
steps = np.array(steps)
print(f"{os.path.basename(root)} {json.dumps(cfg, sort_keys=True)}\n"
      f"  ok {ok}/{len(path)}  step max {steps.max():.3f} p99 {np.percentile(steps, 99):.3f} "
      f">0.3: {int((steps > 0.3).sum())}  fallback {fb}  regime {dict(regimes)}\n"
      f"  pitch [{min(pitch):+.3f}, {max(pitch):+.3f}]  "
      f"target pitch [{min(p[0] for p in path):+.3f}, {max(p[0] for p in path):+.3f}]  ms mean {np.mean(ms):.1f}")
