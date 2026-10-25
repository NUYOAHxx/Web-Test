"""v12 / v13 压力基准成对比较：compare.py <pkg_root> <results_dir>

results_dir 下 v12_<组>.pkl 与 v13_<组>.pkl（stress.py 的输出），组 = two / wide / two_rep / wide_rep：
  two / wide       一次性求解（q_prev = q_ready），400 个目标（full 200 + tool_dir 100 + tool_cone 100）；
  two_rep / wide_rep  回放：q_prev = v12 链式求解（v12_<模型>_chain.pkl）中上一个目标的解，两版输入完全相同。
输出：解出数、失败码、耗时（均值 / 中位数 / p90 / p99 / 最大，ms）、两版都解出的目标上的代价差均值 dC（v13 - v12，
负 = v13 更优）与腰部在舒适窗内的比例差 inC（v13 - v12）。
"""
import collections
import os
import pickle
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

root, rdir = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
sys.path.insert(0, root)
os.chdir(root)
from tests.fixtures import make_solver  # noqa: E402


def comfort_info(model):
    if model == "two_arm":
        from tests import test_v12_changes as T
        import ha_arm_ik.robot_model as RM
        from ha_arm_ik import solver as S
        tmp = TemporaryDirectory()
        path = Path(tmp.name) / "two_arm.urdf"
        path.write_text(T.two_arm_xml())
        RM.ARM_CONFIGS.clear()
        RM.ARM_CONFIGS.update(T.arm_configs())
        s = S.HAArmIKSolver(path, "left", include_waist=True, verbose=False,
                            config=dict(T.SOLVER_CFG, dls_seed_library_size=0))
    else:
        s, _ = make_solver(config={"dls_seed_library_size": 0})
    waist = np.array([n in ("Hip_Link1", "Body_Link") for n in s.active_joint_names])
    return waist, s.waist_comfort_lower, s.waist_comfort_upper


INFO = {"two": comfort_info("two_arm"), "wide": comfort_info("wide")}


def summary(rows):
    ms = np.array([x["ms"] for x in rows])
    fails = collections.Counter(x["code"] for x in rows if x["q"] is None)
    return (f"{sum(x['q'] is not None for x in rows):3d}/{len(rows)} 均值 {ms.mean():5.1f} 中位 {np.median(ms):5.1f} "
            f"p90 {np.percentile(ms, 90):5.0f} p99 {np.percentile(ms, 99):5.0f} 最大 {ms.max():5.0f} 失败 {dict(fails)}")


for group in ("two", "wide", "two_rep", "wide_rep"):
    a = pickle.load(open(os.path.join(rdir, f"v12_{group}.pkl"), "rb"))
    b = pickle.load(open(os.path.join(rdir, f"v13_{group}.pkl"), "rb"))
    waist, lo, hi = INFO[group.split("_")[0]]
    ra = [x for m in ("full", "tool_dir", "tool_cone") for x in a[m]]
    rb = [x for m in ("full", "tool_dir", "tool_cone") for x in b[m]]
    both = [(x, y) for x, y in zip(ra, rb) if x["q"] is not None and y["q"] is not None]
    inside = lambda q: bool(np.all(q[waist] >= lo[waist] - 1e-9) and np.all(q[waist] <= hi[waist] + 1e-9))
    dC = np.mean([y["cost"] - x["cost"] for x, y in both])
    inC = np.mean([inside(y["q"]) for _, y in both]) - np.mean([inside(x["q"]) for x, _ in both])
    only = sum(x["q"] is not None and y["q"] is None for x, y in zip(ra, rb))
    print(f"[{group}]")
    print("  v12 " + summary(ra))
    print("  v13 " + summary(rb))
    print(f"  dC {dC:+.1f}  inC {inC:+.3f}  v12 解出而 v13 未解出 {only}")
