"""跨版本逐位对比用：dump_right.py <pkg_root> <out.npz>（v12 代码；v11 代码用 bench/v11/dump_right.py ... declared）

合成双臂 URDF、目标集（近处 40、远处 60、跟踪 120）与求解参数都和 bench/v11/dump_right.py 相同；右臂 q_ready
取 ARM_CONFIGS 里写的数值。保存解 / 失败码 / 负荷峰值，用 bench/v11/compare_dumps.py 比较。
"""
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

root, out = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
sys.path.insert(0, root)
os.chdir(root)
from tests import test_v12_changes as T  # noqa: E402
from tests.fixtures import root_target  # noqa: E402
import ha_arm_ik.ik_base as RM  # noqa: E402
from ha_arm_ik import solver as S  # noqa: E402
from ha_arm_ik.solver_dls import JointStateSnapshot  # noqa: E402

tmp = TemporaryDirectory()
path = Path(tmp.name) / "two_arm.urdf"
path.write_text(T.two_arm_xml())
cfg = dict(T.SOLVER_CFG, load_max_seeds=10, dls_max_iter=500, load_budget_s=60.0)
configs = T.arm_configs()
RM.ARM_CONFIGS.clear()
RM.ARM_CONFIGS.update(configs)
left = S.HAArmIKSolver(path, "left", include_waist=True, verbose=False, config=cfg)
right = S.HAArmIKSolver(path, "right", include_waist=True, verbose=False, config=cfg)
s9 = T.S9
state_l = {n: 0.0 for n in T.ALL_JOINTS}
state_l.update(dict(zip(left.active_joint_names, left.q_ready)))
state_r = T.mirror_state(state_l)

rng = np.random.default_rng(31)
sets = [([np.clip(left.q_ready + rng.uniform(-0.6, 0.6, size=9), left.lower + 0.05, left.upper - 0.05)
          for _ in range(40)], False)]
qs = []
for _ in range(60):
    qt = rng.uniform(-2.0, 2.0, size=9)
    qt[:2] = rng.uniform(-0.6, 0.6, size=2)
    qs.append(np.clip(qt, left.lower + 0.05, left.upper - 0.05))
sets.append((qs, False))
rng3 = np.random.default_rng(3)
amp = rng3.uniform(0.2, 0.5, size=9)
amp[:2] = [0.5, 0.8]
ph = rng3.uniform(0, 2 * np.pi, size=9)
w = rng3.uniform(0.5, 1.0, size=9)
sets.append(([np.clip(left.q_ready + amp * (np.sin(w * t + ph) - np.sin(ph)), left.lower + 0.05, left.upper - 0.05)
              for t in np.linspace(0, 12.0, 121)[1:]], True))

Q, codes, peaks = [], [], []
for qs, chain in sets:
    q_prev = right.q_ready.copy()
    for qt in qs:
        q, _, _ = right.solve(root_target(right, state_r, s9 * qt), frame="world", q_prev=q_prev,
                              joint_state=JointStateSnapshot(state_r))
        d = right.last_load_diagnostics
        Q.append(np.full(9, np.nan) if q is None else np.asarray(q, dtype=float))
        codes.append(str(d.get("code")))
        peaks.append(float(d.get("peak_ratio", np.nan)))
        if chain and q is not None:
            q_prev = np.asarray(q, dtype=float)
# v9 代码没有扰动噪声镜像：noise 记为全 +1（与 v11 的 [+1, -1, s] 对照）
np.savez(out, q=np.array(Q), code=np.array(codes), peak=np.array(peaks), q_ready=right.q_ready,
         noise=np.ones(9))
print("v12", root, len(Q), "solved", int(np.sum(~np.isnan(np.array(Q)[:, 0]))))
