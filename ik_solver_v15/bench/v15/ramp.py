"""v15 腰部斜坡跟踪：python bench/v15/ramp.py

tests/test_v9_changes 的 WAIST_RAMP：目标 = FK(QREF + t·WAIST_RAMP)，t = 1/40 … 1（需要腰部转动），q_prev = 上一帧的解。
打印每帧关节最大变化与腰角；v15 默认与 v14 模式（CONTINUATION_POSTURE_SETTLE = True）各一遍。
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np  # noqa: E402

from tests.fixtures import make_solver, root_target  # noqa: E402
from tests import test_v9_changes as V  # noqa: E402
from ha_arm_ik import solver as S  # noqa: E402

for settle in (False, True):
    S.CONTINUATION_POSTURE_SETTLE = settle
    s, state = make_solver(config=V.BENCH)
    targets = [root_target(s, state, V.QREF + t * V.WAIST_RAMP) for t in np.linspace(0.0, 1.0, 41)[1:]]
    q_prev, steps, waist = V.QREF.copy(), [], []
    for T in targets:
        q, status, res = V.solve(s, state, T=T, q_prev=q_prev)
        q = np.asarray(q, dtype=float)
        steps.append(float(np.abs(q - q_prev).max()))
        waist.append(q[:2].copy())
        q_prev = q
    steps = np.array(steps)
    moved = [k for k, w in enumerate(waist) if np.abs(w - V.QREF[:2]).max() > 1e-6]
    first = moved[0] if moved else len(steps)
    locked, after = steps[:first], steps[first:]
    print(f"[{'v14 模式' if settle else 'v15 默认'}] 40 帧：单步最大 {steps.max():.3f} rad，>= 0.1 的 {int((steps >= 0.1).sum())} 帧；"
          f"腰保持 q_ready 的前 {first} 帧每帧 {locked.min() if first else 0:.3f}~{locked.max() if first else 0:.3f}，"
          f"腰开始动之后每帧 {after.min() if after.size else 0:.3f}~{after.max() if after.size else 0:.3f}；"
          f"关节路程合计 {steps.sum():.2f} rad（∞-范数逐帧相加）")
    print("    ", [round(float(x), 3) for x in steps])
S.CONTINUATION_POSTURE_SETTLE = False
