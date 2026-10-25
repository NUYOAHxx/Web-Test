#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""双臂 16 维联合 IK（ha_arm_ik/dual_arm_ik.py）性能测试。

在仓库根目录运行：python bench/dual/bench_dual.py [--quick]
合成双臂 URDF（tests/dual_fixtures.py），默认配置（time_budget_s = 0.1 s）。单进程顺序执行，计时含 solve() 全部开销。

B1 一次性可达      目标 = 随机关节角（限位内缩 0.05 rad）的 FK，q_prev = q_ready
B2 大范围换姿      q_prev = 一个随机构型，目标 = 另一个随机构型的 FK
B3 连续跟踪        20 条关节空间直线轨迹 × 30 步（每关节每步 <= 0.02 rad），q_prev = 上一步的解
B4 单手保持        左手 "hold"（保持 q_prev 处位姿；dual-p2.1 起 None 即 "hold"）/ 左臂 "keep"，
                   右手换到新目标（腰不变时可达）
B5 失败与混合用例  (a) 左手目标在任何腰角下都超出臂展（可证明不可达） (b) 两手间距超过可能的最大间距
                   (c) 左右手目标来自两个独立随机构型 (d) 左手位置可达、姿态随机（部分不可达）
"""

import argparse
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pinocchio as pin  # noqa: E402

from ha_arm_ik import dual_arm_ik as D  # noqa: E402
from tests.dual_fixtures import two_arm_urdf_path, synthetic_layout  # noqa: E402


def stats_ms(ts):
    t = np.asarray(ts, dtype=float)
    if t.size == 0:
        return {}
    return {"n": int(t.size), "median": float(np.median(t)), "mean": float(t.mean()),
            "p90": float(np.percentile(t, 90)), "p99": float(np.percentile(t, 99)), "max": float(t.max())}


def fmt(st):
    if not st:
        return "-"
    return "中位 {median:.1f} / 均值 {mean:.1f} / p90 {p90:.1f} / p99 {p99:.1f} / 最长 {max:.1f} ms".format(**st)


def count(values):
    out = {}
    for v in values:
        out[str(v)] = out.get(str(v), 0) + 1
    return dict(sorted(out.items(), key=lambda kv: (len(kv[0]), kv[0])))


class Runner:
    def __init__(self):
        t = time.perf_counter()
        self.s = D.DualArmIKSolver(two_arm_urdf_path(), layout=synthetic_layout())
        self.init_s = time.perf_counter() - t
        s = self.s
        self.lo = np.where(np.isfinite(s.lower), s.lower, -np.pi)
        self.hi = np.where(np.isfinite(s.upper), s.upper, np.pi)
        model = pin.buildModelFromUrdf(two_arm_urdf_path())
        self.model, self.data = model, model.createData()
        self.fid = {k: model.getFrameId(f"{k}_tcp") for k in ("left", "right")}

    def rand_q(self, rng, inset=0.05):
        return rng.uniform(self.lo + inset, self.hi - inset)

    def check(self, q, targets):
        """独立 FK 核对：返回 (位姿全部达标, 限位全部满足)。"""
        qf = pin.neutral(self.model)
        for i, name in enumerate(self.s.active_joint_names):
            j = self.model.joints[self.model.getJointId(name)]
            if j.nq == 1:
                qf[j.idx_q] = q[i]
            else:
                qf[j.idx_q], qf[j.idx_q + 1] = np.cos(q[i]), np.sin(q[i])
        pin.framesForwardKinematics(self.model, self.data, qf)
        ok = True
        for side, T in targets.items():
            M = self.data.oMf[self.fid[side]]
            pe = np.linalg.norm(M.translation - T[:3, 3])
            oe = np.linalg.norm(pin.log3(T[:3, :3].T @ M.rotation))
            ok &= bool(pe <= D.POS_TOL and oe <= D.ORI_TOL)
        lo, hi = self.s.lower, self.s.upper
        lim = bool(np.all(q[np.isfinite(lo)] >= lo[np.isfinite(lo)] - 1e-9)
                   and np.all(q[np.isfinite(hi)] <= hi[np.isfinite(hi)] + 1e-9))
        return ok, lim

    def solve(self, TL, TR, q_prev, **kw):
        t = time.perf_counter()
        q, status, res = self.s.solve(TL, TR, q_prev=q_prev, **kw)
        return q, status, res, 1e3 * (time.perf_counter() - t)


def run_set(r, cases, name):
    """cases: [(TL, TR, q_prev, check_targets, fixed)]；fixed = None 或 (关节切片, 应保持不变的值)。"""
    ts, statuses, seeds, reasons, bad, t_fail = [], [], [], [], 0, []
    for TL, TR, q_prev, tg, fixed in cases:
        q, status, res, dt = r.solve(TL, TR, q_prev, update_prev=False)
        ts.append(dt)
        statuses.append(status)
        seeds.append(res.get("seeds_used"))
        if res.failure_reason:
            reasons.append(res.failure_reason)
        if q is None:
            t_fail.append(dt)
        else:
            ok, lim = r.check(q, tg)
            if fixed is not None:
                ok &= bool(np.array_equal(q[fixed[0]], fixed[1]))
            bad += (not ok) or (not lim)
    ok_n = sum(s in (D.SUCCESS, D.NEAR_LIMIT) for s in statuses)
    first = sum(1 for s, n in zip(statuses, seeds) if s in (D.SUCCESS, D.NEAR_LIMIT) and n == 1)
    out = {"name": name, "n": len(cases), "ok": ok_n, "ok_rate": ok_n / max(1, len(cases)),
           "first_seed_ok": first, "time_ms": stats_ms(ts), "status": count(statuses),
           "seeds": count(seeds), "failure_reasons": count(reasons), "verify_bad": bad}
    if t_fail:
        out["fail_time_ms"] = stats_ms(t_fail)
    return out


def b1(r, n):
    rng = np.random.default_rng(7)
    cases = []
    for _ in range(n):
        TL, TR = r.s.fk(r.rand_q(rng))
        cases.append((TL, TR, r.s.q_ready.copy(), {"left": TL, "right": TR}, None))
    return run_set(r, cases, "B1 一次性可达（q_prev = q_ready）")


def b2(r, n):
    rng = np.random.default_rng(8)
    cases = []
    for _ in range(n):
        q_prev = r.rand_q(rng)
        TL, TR = r.s.fk(r.rand_q(rng))
        cases.append((TL, TR, q_prev, {"left": TL, "right": TR}, None))
    return run_set(r, cases, "B2 大范围换姿（q_prev = 另一随机构型）")


def b3(r, n_traj, n_step):
    rng = np.random.default_rng(9)
    ts, statuses, seeds, jumps, moves, bad = [], [], [], [], [], 0
    for _ in range(n_traj):
        qa, qb = r.rand_q(rng, 0.3), r.rand_q(rng, 0.3)
        qb = qa + 0.6 * (qb - qa) / max(1e-9, np.abs(qb - qa).max())       # 每关节最多走 0.6 rad
        q_prev = qa.copy()
        TL0, TR0 = r.s.fk(qa)
        for k in range(1, n_step + 1):
            TL, TR = r.s.fk(qa + (qb - qa) * k / n_step)
            q, status, res, dt = r.solve(TL, TR, q_prev, update_prev=False)
            ts.append(dt)
            statuses.append(status)
            seeds.append(res.get("seeds_used"))
            moves.append(1e3 * max(np.linalg.norm(TL[:3, 3] - TL0[:3, 3]), np.linalg.norm(TR[:3, 3] - TR0[:3, 3])))
            TL0, TR0 = TL, TR
            if q is None:
                continue
            ok, lim = r.check(q, {"left": TL, "right": TR})
            bad += (not ok) or (not lim)
            jumps.append(float(np.abs(q - q_prev).max()))
            q_prev = q
    jumps = np.array(jumps)
    ok_n = sum(s in (D.SUCCESS, D.NEAR_LIMIT) for s in statuses)
    return {"name": "B3 连续跟踪（q_prev = 上一步解）", "n": len(ts), "ok": ok_n, "ok_rate": ok_n / len(ts),
            "first_seed_ok": sum(1 for s, n in zip(statuses, seeds) if s in (D.SUCCESS, D.NEAR_LIMIT) and n == 1),
            "time_ms": stats_ms(ts), "status": count(statuses), "seeds": count(seeds), "verify_bad": bad,
            "hand_step_mm_median": float(np.median(moves)), "hand_step_mm_max": float(np.max(moves)),
            "joint_step_rad_median": float(np.median(jumps)), "joint_step_rad_p99": float(np.percentile(jumps, 99)),
            "joint_step_rad_max": float(np.max(jumps)), "joint_steps_over_0p1": int(np.sum(jumps > 0.1))}


def b4(r, n):
    rng = np.random.default_rng(10)
    out = []
    for mode in ("hold", "keep"):
        cases = []
        for _ in range(n):
            qa = r.rand_q(rng)
            qc = qa.copy()
            qc[D.RIGHT] = r.rand_q(rng)[D.RIGHT]
            TL_hold, _ = r.s.fk(qa)
            _, TR = r.s.fk(qc)
            if mode == "hold":      # 左手位姿保持（腰、左臂可动）
                cases.append((mode, TR, qa, {"left": TL_hold, "right": TR}, None))
            else:                   # 左臂关节不动（腰动时左手跟着动）
                cases.append((mode, TR, qa, {"right": TR}, (D.LEFT, qa[D.LEFT])))
        out.append(run_set(r, cases, f"B4 左手 \"{mode}\"、右手换新目标"))
    return out


def b5(r, n):
    rng = np.random.default_rng(11)
    out = []
    # (a) 独立于求解器算出：任何腰角下左臂根关节离原点最远 rho、根关节之后的臂展上界 reach，
    #     目标到原点的距离 > rho + reach 即一定不可达
    model, data = r.model, r.data
    jid = model.getJointId(r.s.layout.left_joints[0])
    rho = 0.0
    for w0 in np.linspace(r.s.lower[0], r.s.upper[0], 31):
        for w1 in np.linspace(r.s.lower[1], r.s.upper[1], 31):
            qf = pin.neutral(model)
            for i, nm in enumerate(r.s.layout.waist_joints):
                qf[model.joints[model.getJointId(nm)].idx_q] = (w0, w1)[i]
            pin.forwardKinematics(model, data, qf)
            rho = max(rho, float(np.linalg.norm(data.oMi[jid].translation)))
    fr = model.frames[r.fid["left"]]
    reach = float(np.linalg.norm(fr.placement.translation))
    j = fr.parentJoint
    while j != jid:
        reach += float(np.linalg.norm(model.jointPlacements[j].translation))
        j = model.parents[j]
    cases = []
    for _ in range(n):
        _, TR = r.s.fk(r.rand_q(rng))
        d = rng.normal(size=3)
        far = np.eye(4)
        far[:3, 3] = (rho + reach + rng.uniform(0.02, 0.5)) * d / np.linalg.norm(d)
        cases.append((far, TR, r.s.q_ready.copy(), {}, None))
    out.append(run_set(r, cases, "B5a 左手目标在任何腰角下都超出臂展"))
    m = r.s._m
    pin.forwardKinematics(m.model, m.data, m.full_q(r.s.q_ready, m.template))
    bound = (np.linalg.norm(m.data.oMi[m.arm_root_jid["left"]].translation - m.data.oMi[m.arm_root_jid["right"]].translation)
             + m.reach_root["left"] + m.reach_root["right"])
    cases = []
    while len(cases) < n:
        qa, qb = r.rand_q(rng), r.rand_q(rng)
        qa[5] = qb[12] = 0.0
        TL, _ = r.s.fk(qa)
        _, TR = r.s.fk(qb)
        if np.linalg.norm(TL[:3, 3] - TR[:3, 3]) > bound + 0.01:
            cases.append((TL, TR, r.s.q_ready.copy(), {}, None))
    out.append(run_set(r, cases, "B5b 两手间距超过最大可能间距（各自可达）"))
    cases = []
    for _ in range(n):
        TL, _ = r.s.fk(r.rand_q(rng))
        _, TR = r.s.fk(r.rand_q(rng))
        cases.append((TL, TR, r.s.q_ready.copy(), {"left": TL, "right": TR}, None))
    out.append(run_set(r, cases, "B5c 左右手目标来自两个独立随机构型（多数可达、部分不可同时到达）"))
    cases = []
    for _ in range(n):
        TL, TR = r.s.fk(r.rand_q(rng))
        TL = TL.copy()
        TL[:3, :3] = pin.exp3(rng.normal(size=3) * 3.0)     # 随机姿态
        cases.append((TL, TR, r.s.q_ready.copy(), {"left": TL, "right": TR}, None))
    out.append(run_set(r, cases, "B5d 左手位置可达、姿态随机（部分不可达）"))
    return out


def iteration_cost(r):
    rng = np.random.default_rng(12)
    it, t_total = 0, 0.0
    for _ in range(100):
        TL, TR = r.s.fk(r.rand_q(rng))
        q, status, res, dt = r.solve(TL, TR, r.s.q_ready.copy(), update_prev=False)
        it += res.get("iterations", 0)
        t_total += sum(row["elapsed_ms"] for row in res.per_seed)
    return {"iterations": it, "us_per_iteration": 1e3 * t_total / max(1, it)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="小样本（冒烟用）")
    args = ap.parse_args()
    q = args.quick
    r = Runner()
    for _ in range(5):                       # 预热
        TL, TR = r.s.fk(r.rand_q(np.random.default_rng(0)))
        r.solve(TL, TR, r.s.q_ready.copy(), update_prev=False)
    results = [b1(r, 100 if q else 1000), b2(r, 50 if q else 300), b3(r, 4 if q else 20, 30)]
    results += b4(r, 50 if q else 200)
    results += b5(r, 20 if q else 100)
    data = {"version": D.DUAL_SOLVER_VERSION, "python": platform.python_version(), "numpy": np.__version__,
            "pinocchio": pin.__version__, "machine": platform.machine(), "cpu_count": __import__("os").cpu_count(),
            "init_s": r.init_s, "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in r.s.config.items()},
            "iteration_cost": iteration_cost(r), "results": results}
    lines = [f"dual_arm_ik {D.DUAL_SOLVER_VERSION} | Python {data['python']} | numpy {data['numpy']} | "
             f"Pinocchio {data['pinocchio']} | {data['cpu_count']} CPU | 初始化 {1e3 * r.init_s:.0f} ms（含种子库）",
             f"每次迭代 {data['iteration_cost']['us_per_iteration']:.0f} µs（100 个 B1 目标平均）", ""]
    for res in results:
        lines.append(f"{res['name']}  N={res['n']}")
        lines.append(f"  成功 {res['ok']}/{res['n']} = {100 * res['ok_rate']:.1f}%   首种子成功 {res['first_seed_ok']}   "
                     f"核验不合格 {res['verify_bad']}")
        lines.append(f"  耗时 {fmt(res['time_ms'])}")
        lines.append(f"  状态 {res['status']}   种子数 {res['seeds']}")
        if res.get("failure_reasons"):
            lines.append(f"  失败原因 {res['failure_reasons']}")
        if res.get("fail_time_ms"):
            lines.append(f"  失败用例耗时 {fmt(res['fail_time_ms'])}")
        if "hand_step_mm_median" in res:
            lines.append(f"  每步手移动 中位 {res['hand_step_mm_median']:.1f} / 最大 {res['hand_step_mm_max']:.1f} mm；"
                         f"每步关节最大变化 中位 {res['joint_step_rad_median']:.3f} / p99 {res['joint_step_rad_p99']:.3f} / "
                         f"最大 {res['joint_step_rad_max']:.3f} rad（参考轨迹每步 <= 0.020），> 0.1 rad 的步数 "
                         f"{res['joint_steps_over_0p1']}")
        lines.append("")
    text = "\n".join(lines)
    print(text)
    if not q:
        suffix = "pin" + pin.__version__.replace(".", "_")
        out = Path(__file__).resolve().parent
        (out / f"result_{suffix}.txt").write_text(text, encoding="utf-8")
        (out / f"result_{suffix}.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
