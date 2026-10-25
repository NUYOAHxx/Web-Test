"""v15 续算漂移测量：python bench/v15/drift.py

合成 9 轴夹具（tests/fixtures.make_solver），4 组随机构型（q_ready + N(0, 0.6)，限位内缩 0.1），q_prev = 上一次的解：
  (1) 静止：目标 = FK(q0) 不动，连续调用 5 次，每次的关节最大变化 |q - q_prev|∞ 与耗时；
  (2) 小步：目标每次平移 (1, 0.5, -0.5) mm（1.22 mm），连续 10 次，每次的关节最大变化与 DLS 迭代数。
各跑两遍：v15 默认（solver.CONTINUATION_POSTURE_SETTLE = False）与 v14 模式（True；与 v14 逐位相同）。
"""
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np  # noqa: E402

from tests import fixtures as F  # noqa: E402
from ha_arm_ik import solver as S  # noqa: E402
from ha_arm_ik.solver_dls import JointStateSnapshot  # noqa: E402

s, state = F.make_solver()
names = s.active_joint_names
lo, hi = s.search_lower + 0.1, s.search_upper - 0.1


def call(T, q):
    st = dict(state)
    st.update(zip(names, q))
    t0 = time.perf_counter()
    qn, status, res = s.solve(T, frame="world", q_prev=q, joint_state=JointStateSnapshot(st))
    ms = 1e3 * (time.perf_counter() - t0)
    qn = np.asarray(qn, dtype=float)
    return qn, float(np.abs(qn - q).max()), ms, sum(p.get("iters", 0) for p in res.per_seed), \
        bool(s.last_load_diagnostics["continuation"].get("hold", False))


for settle in (False, True):
    S.CONTINUATION_POSTURE_SETTLE = settle
    tag = "v14 模式" if settle else "v15 默认"
    rng = np.random.default_rng(5)
    static, times, holds, moving, iters_all = [], [], 0, [], []
    for trial in range(4):
        q0 = np.clip(s.q_ready + rng.normal(0, 0.6, len(names)), lo, hi)
        st = dict(state)
        st.update(zip(names, q0))
        T0 = F.root_target(s, st, q0)
        s.pin_maps.pop("q_template", None)
        q, row = q0.copy(), []
        for _ in range(5):
            q, d, ms, _, hold = call(T0, q)
            row.append(round(d, 4))
            times.append(ms)
            holds += hold
        static.append(row)
        q, row, its = q0.copy(), [], []
        for k in range(10):
            T = T0.copy()
            T[:3, 3] += np.array([0.001, 0.0005, -0.0005]) * (k + 1)
            q, d, _, it, _ = call(T, q)
            row.append(round(d, 4))
            its.append(it)
        moving.append(row)
        iters_all.append(its)
    a = np.array(static)
    print(f"[{tag}] 静止 20 次：最大变化 {a.max():.4f} rad，均值 {a.mean():.4f}，为 0 的 {int((a == 0).sum())} 次，"
          f"hold {holds} 次；耗时中位 {np.median(times):.1f} ms")
    for row in static:
        print("    ", row)
    m = np.array(moving)
    print(f"[{tag}] 小步 1.22 mm × 10：每次关节最大变化 {m.min():.4f}~{m.max():.4f} rad，"
          f"> 0.05 的 {int((m > 0.05).sum())}/{m.size} 次")
    for row, its in zip(moving, iters_all):
        print("    ", row, "iters", its)
S.CONTINUATION_POSTURE_SETTLE = False
