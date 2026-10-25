#!/usr/bin/env python3
"""合成基准（离线）：固定随机种子生成同一批 6D 目标，统计成功率 / 耗时 / 负荷峰值 / 最大关节跳变。

用法（在包根目录，即本文件上一级目录运行）：
    python bench/bench_targets.py            # 默认配置
    python bench/bench_targets.py '{"load_min_seeds": 3, "load_min_unlocked_seeds": 2}'
说明：使用 tests/fixtures.py 的合成 9 轴 URDF 和 tests/stubs.py 测试替身，不代表实机 URDF 结果。
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from tests.fixtures import make_solver, root_target  # noqa: E402
from ha_arm_ik.solver_dls import JointStateSnapshot  # noqa: E402

CASES = (("near_full", 40, 7), ("far_full", 60, 11), ("far_tool_dir", 50, 13), ("far_tool_cone", 50, 17))


def main():
    extra = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    cfg = {"load_max_seeds": 10, "dls_max_iter": 500, "load_budget_s": 2.0, **extra}
    s, state = make_solver(config=cfg)
    q_meas = np.array([state[n] for n in s.active_joint_names])
    out = {}
    for name, n, seed in CASES:
        rng = np.random.default_rng(seed)
        rows = []
        for _ in range(n):
            if name == "near_full":
                qt = np.clip(q_meas + rng.uniform(-0.6, 0.6, size=len(q_meas)), -2.8, 2.8)
            else:
                qt = rng.uniform(-2.9, 2.9, size=len(q_meas))
                qt[:2] = rng.uniform(-0.8, 0.8, size=2)
            T = root_target(s, state, qt)
            mode = name.split("_", 1)[1]
            if mode != "full":  # 方向/锥约束：姿态加随机扰动，只约束工具轴
                T[:3, :3] = T[:3, :3] @ R.from_rotvec(rng.uniform(-1, 1, 3)).as_matrix()
            t0 = time.monotonic()
            q, _status, _res = s.solve(
                T, frame="world", joint_state=JointStateSnapshot(state), orientation_mode=mode
            )
            ms = (time.monotonic() - t0) * 1000.0
            d = s.last_load_diagnostics
            rows.append(dict(
                ok=q is not None, ms=ms, peak=d.get("peak_ratio", np.nan), code=d.get("code"),
                jump=float(np.max(np.abs(q - q_meas))) if q is not None else np.nan,
            ))
        ok = [r for r in rows if r["ok"]]
        out[name] = {
            "success": f"{len(ok)}/{n}",
            "median_ms": round(float(np.median([r["ms"] for r in rows])), 1),
            "p90_ms": round(float(np.percentile([r["ms"] for r in rows], 90)), 1),
            "mean_peak": round(float(np.mean([r["peak"] for r in ok])), 4) if ok else None,
            "jump_gt_pi": int(sum(r["jump"] > np.pi for r in ok)),
            "max_jump": round(float(np.max([r["jump"] for r in ok])), 2) if ok else None,
            "fail_codes": sorted({str(r["code"]) for r in rows if not r["ok"]}),
        }
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
