"""成对计时汇总：paired_timing.py <dir>

<dir> 下是按 v7, v8, v6, v8, v7, v8 交替顺序运行 run_targets.py（repeat 1）得到的 pair_<序号>_<版本>.pkl。
v7 vs v8：各取 2 次运行逐目标最短耗时的均值；v6 vs v8：相邻两次运行（第 3、4 次）。同时检查重复运行的解逐值一致。
"""
import collections
import glob
import os
import pickle
import sys

import numpy as np

d = os.path.abspath(sys.argv[1])
runs = collections.defaultdict(list)
for p in sorted(glob.glob(os.path.join(d, "pair_*_v*.pkl")), key=lambda p: int(os.path.basename(p).split("_")[1])):
    runs[os.path.basename(p).rsplit("_", 1)[1][:-4]].append(pickle.load(open(p, "rb")))
cases = ["near_full", "far_full", "far_tool_dir", "far_tool_cone", "tracking"]
print("runs:", {k: len(v) for k, v in runs.items()})
for tag, rs in runs.items():
    for c in cases:
        same = all((a["q"] is None and b["q"] is None)
                   or (a["q"] is not None and b["q"] is not None and np.array_equal(a["q"], b["q"]))
                   for r in rs[1:] for a, b in zip(rs[0][c], r[c]))
        if not same:
            print("NONDETERMINISTIC", tag, c)


def ms_min(rs, c):
    return np.min(np.array([[row["ms"] for row in r[c]] for r in rs]), axis=0)


print("case           |   v7 ms |   v8 ms | v8/v7  |   v6 ms | v8 ms(相邻) | v6/v8")
for c in cases:
    m7, m8 = ms_min(runs["v7"][:2], c).mean(), ms_min(runs["v8"][:2], c).mean()
    m6 = np.array([row["ms"] for row in runs["v6"][0][c]]).mean()
    m8s = np.array([row["ms"] for row in runs["v8"][1][c]]).mean()
    print(f"{c:14s} | {m7:7.1f} | {m8:7.1f} | {100 * (m8 / m7 - 1):+5.1f}% | {m6:7.1f} | {m8s:11.1f} | {m6 / m8s:.2f}x")
