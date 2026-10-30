"""路点跟踪耗时：track_time.py <pkg_root> <cfg> <jump|default> <out.json>

与 bench/v12/track_detail.py 相同的双臂合成模型与 120 个路点（左臂、右臂各跑一遍），记录每步耗时、单步
max|q - q_prev|、续算局部解是否成立、代价与状态。cfg：off = v13 四项改动全部关闭（= v12，见
tests/test_v13_changes.V12EquivalenceTests）| v13 = 默认 | v13_libcont = 续算类求解也用种子库（对照）。
jump：default = 求解器默认跳变上限（π）| 数值 = max_joint_jump_rad（0.25 为 v9 文档建议的跟踪设置）。
"""
import json, os, sys, time
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
root = os.path.abspath(sys.argv[1])
name, jump, out = sys.argv[2], sys.argv[3], os.path.abspath(sys.argv[4])
sys.path.insert(0, root); os.chdir(root)
from tests import test_v12_changes as T
from tests.fixtures import root_target
import ha_arm_ik.ik_base as RM
import ha_arm_ik.solver_dls as D
from ha_arm_ik import solver as S
from ha_arm_ik.solver_dls import JointStateSnapshot
CFGS = {
    "off": dict(dls_seed_library_size=0, dls_waist_comfort_early_fallback=False, dls_jump_limit_rescue=False, dls_locked_reach_check=False),
    "v13_libcont": dict(dls_seed_library_continuation=True),
    "v13": {},
}
tmp = TemporaryDirectory(); path = Path(tmp.name) / "two_arm.urdf"; path.write_text(T.two_arm_xml())
RM.ARM_CONFIGS.clear(); RM.ARM_CONFIGS.update(T.arm_configs())
cfg = dict(T.SOLVER_CFG, load_max_seeds=10, dls_max_iter=500, load_budget_s=60.0, **CFGS[name])
if jump != "default":
    cfg["max_joint_jump_rad"] = float(jump)
left = S.HAArmIKSolver(path, "left", include_waist=True, verbose=False, config=cfg)
right = S.HAArmIKSolver(path, "right", include_waist=True, verbose=False, config=cfg)
state_l = {n: 0.0 for n in T.ALL_JOINTS}; state_l.update(dict(zip(left.active_joint_names, left.q_ready)))
state_r = T.mirror_state(state_l)
rng3 = np.random.default_rng(3)
amp = rng3.uniform(0.2, 0.5, size=9); amp[:2] = [0.5, 0.8]
ph = rng3.uniform(0, 2*np.pi, size=9); w = rng3.uniform(0.5, 1.0, size=9)
qs = [np.clip(left.q_ready + amp*(np.sin(w*t+ph)-np.sin(ph)), left.lower+0.05, left.upper-0.05) for t in np.linspace(0, 12.0, 121)[1:]]
def track(s, state, signs):
    rows, q_prev = [], s.q_ready.copy()
    for qt in qs:
        Tt = root_target(s, state, signs*qt)
        t0 = time.perf_counter()
        q, _, r = s.solve(Tt, frame="world", q_prev=q_prev, joint_state=JointStateSnapshot(state))
        ms = (time.perf_counter() - t0) * 1000
        d = s.last_load_diagnostics; cont = dict(d.get("continuation") or {})
        step = None if q is None else float(np.max(np.abs(D.joint_difference(q, q_prev, s.multi_turn_mask))))
        rows.append(dict(ms=ms, step=step, code=d.get("code"), local=cont.get("local_used"), regime=cont.get("regime"),
                         cost=float(r.get("cost", np.nan)), status=r.get("status"), margin=float(r.get("margin", np.nan)),
                         q=None if q is None else [float(x) for x in q]))
        if q is not None:
            q_prev = np.asarray(q, float)
    return rows
res = {"left": track(left, state_l, np.ones(9)), "right": track(right, state_r, T.S9)}
json.dump(res, open(out, "w"))
for side in ("left", "right"):
    R = res[side]; st = [x["step"] for x in R if x["step"] is not None]; ms = np.array([x["ms"] for x in R])
    print(name, jump, side, "solved", sum(x["step"] is not None for x in R), "max_step %.3f" % max(st), "gt0.3", sum(x > 0.3 for x in st),
          "local", sum(bool(x["local"]) for x in R), "ms mean %.1f p90 %.1f max %.1f" % (ms.mean(), np.percentile(ms, 90), ms.max()),
          "cost %.1f" % np.mean([x["cost"] for x in R]), "near_limit", sum(x["status"] == "NEAR_LIMIT" for x in R))
