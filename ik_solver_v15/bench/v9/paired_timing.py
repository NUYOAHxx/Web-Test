"""成对计时汇总：paired_timing.py <dir>

<dir> 下是按 ABBA 顺序（v8, v9, v9, v8）单进程依次运行 run_targets.py（repeat 1）得到的 pair_<序号>_<版本>.pkl。
每个版本取 2 次运行逐目标最短耗时，再按场景求均值；同时检查同一版本两次运行的解逐值一致。
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
cases = [k for k in next(iter(runs.values()))[0] if not k.startswith("_")]
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


print(f"{'case':14s} | {'v8 ms':>7s} | {'v9 ms':>7s} | v9/v8")
for c in cases:
    a, b = ms_min(runs["v8"], c).mean(), ms_min(runs["v9"], c).mean()
    print(f"{c:14s} | {a:7.1f} | {b:7.1f} | {100 * (b / a - 1):+6.1f}%")
