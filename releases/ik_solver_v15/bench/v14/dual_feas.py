"""16 自由度（腰 2 + 双臂 2×7）可行性微基准：同一个简化 DLS（阻尼最小二乘 + 回溯 + 限位裁剪，无零空间次任务 / SNS / 负荷），
比较 (A) 腰 + 左臂 9 轴单目标、(B) 16 轴联合双目标、(C) 主从分解（先 9 轴解左臂，再锁腰 7 轴解右臂；失败再换主臂）。
目标 = 限位内随机 16 维关节角的双臂 TCP 位姿（两臂同一腰角下必然同时可达）。种子 = q_ready（单种子）。"""
import sys, time
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import pinocchio as pin
from tests import test_v12_changes as T
import ha_arm_ik.solver_dls as D

tmp = TemporaryDirectory()
path = Path(tmp.name) / "two_arm.urdf"
path.write_text(T.two_arm_xml())
m = pin.buildModelFromUrdf(str(path))
d = m.createData()
names = list(T.ALL_JOINTS)
maps = D.build_pinocchio_joint_maps(m, names)
vidx = np.array(maps["active_v_indices"])
n = len(names)
lo, hi = np.full(n, -np.inf), np.full(n, np.inf)
for i, nm in enumerate(names):
    j = m.joints[maps["name_to_joint_id"][nm]]
    if not j.shortname().startswith("JointModelRUB"):
        lo[i], hi[i] = m.lowerPositionLimit[j.idx_q], m.upperPositionLimit[j.idx_q]
cfgs = T.arm_configs()
q_ready = np.concatenate([[0.0, 0.0], cfgs["left"]["q_ready"], cfgs["right"]["q_ready"]])
fl, fr = m.getFrameId("left_tcp", pin.BODY), m.getFrameId("right_tcp", pin.BODY)
WAIST, LEFT, RIGHT = np.arange(0, 2), np.arange(2, 9), np.arange(9, 16)
L, POS_TOL, ORI_TOL = 0.5, 5e-4, 5e-4


def fk(q):
    pin.framesForwardKinematics(m, d, D.active_q_to_pinocchio_q(m, maps, q))
    return d.oMf[fl].homogeneous.copy(), d.oMf[fr].homogeneous.copy()


def task(q, targets, jac=True):
    full = D.active_q_to_pinocchio_q(m, maps, q)
    if jac:
        pin.computeJointJacobians(m, d, full)
    else:
        pin.forwardKinematics(m, d, full)
    pin.updateFramePlacements(m, d)
    es, Js, ok = [], [], True
    for fid, Tt in targets:
        M = d.oMf[fid]
        dp = Tt[:3, 3] - M.translation
        eo = M.rotation @ pin.log3(M.rotation.T @ Tt[:3, :3])
        ok &= bool(np.linalg.norm(dp) <= POS_TOL and np.linalg.norm(eo) <= ORI_TOL)
        es += [dp / L, eo]
        if jac:
            J = pin.getFrameJacobian(m, d, fid, pin.LOCAL_WORLD_ALIGNED)[:, vidx]
            Js += [J[:3] / L, J[3:]]
    return np.concatenate(es), (np.vstack(Js) if jac else None), ok


def dls(q0, targets, cols, max_iter=300, lam2=1e-4, max_step=0.5):
    q = q0.copy()
    e, J, ok = task(q, targets)
    for it in range(max_iter):
        if ok:
            return q, True, it
        Ja = J[:, cols]
        dq = Ja.T @ np.linalg.solve(Ja @ Ja.T + lam2 * np.eye(len(e)), e)
        nrm = np.linalg.norm(dq)
        if nrm > max_step:
            dq *= max_step / nrm
        f0, alpha = e @ e, 1.0
        for _ in range(6):
            qn = q.copy()
            qn[cols] = np.clip(q[cols] + alpha * dq, lo[cols], hi[cols])
            en, Jn, okn = task(qn, targets)
            if en @ en < f0:
                q, e, J, ok = qn, en, Jn, okn
                break
            alpha *= 0.5
        else:
            return q, False, it            # 线搜索停滞
    return q, ok, max_iter


rng = np.random.default_rng(16)
N = 200
lo_s = np.where(np.isfinite(lo), lo + 0.1, -np.pi)
hi_s = np.where(np.isfinite(hi), hi - 0.1, np.pi)
rows = []
for k in range(N):
    qt = rng.uniform(lo_s, hi_s)
    Tl, Tr = fk(qt)
    r = {}
    t0 = time.perf_counter(); _, ok, it = dls(q_ready, [(fl, Tl)], np.r_[WAIST, LEFT]); r["A"] = (ok, it, time.perf_counter() - t0)
    t0 = time.perf_counter(); _, ok, it = dls(q_ready, [(fl, Tl), (fr, Tr)], np.arange(n)); r["B"] = (ok, it, time.perf_counter() - t0)
    # C：主从分解（左主 -> 右从；失败换右主 -> 左从）
    t0 = time.perf_counter(); its = 0; okc = False; order = None
    for first, second, f1, T1, f2, T2 in ((LEFT, RIGHT, fl, Tl, fr, Tr), (RIGHT, LEFT, fr, Tr, fl, Tl)):
        q1, ok1, i1 = dls(q_ready, [(f1, T1)], np.r_[WAIST, first]); its += i1
        if not ok1:
            continue
        q2, ok2, i2 = dls(q1, [(f2, T2)], second); its += i2
        if ok2:
            okc, order = True, ("L" if first is LEFT else "R"); break
    r["C"] = (okc, its, time.perf_counter() - t0, order)
    rows.append(r)

def summary(key):
    ok = np.array([r[key][0] for r in rows]); it = np.array([r[key][1] for r in rows]); ms = 1e3 * np.array([r[key][2] for r in rows])
    per_it = ms[it > 0].sum() / max(1, it[it > 0].sum())
    return ok.mean(), np.median(it[ok]) if ok.any() else np.nan, per_it, np.median(ms)

for key, label in (("A", "A 腰+左臂 9 轴，单目标"), ("B", "B 16 轴联合，双目标"), ("C", "C 主从分解（失败换主臂）")):
    s, it, per_it, ms = summary(key)
    print(f"{label:24s} 成功 {100 * s:5.1f}%  成功者迭代中位 {it:5.0f}  每迭代 {per_it * 1e3:6.0f} µs  耗时中位 {ms:6.1f} ms")
print("C 中左主成功", sum(r["C"][3] == "L" for r in rows), "右主补救", sum(r["C"][3] == "R" for r in rows))
print("B 成功而 C 失败", sum(r["B"][0] and not r["C"][0] for r in rows), " C 成功而 B 失败", sum(r["C"][0] and not r["B"][0] for r in rows))
# 双臂都能单独解出（9 轴，各自带腰）但联合 / 分解失败的比例
tmp.cleanup()
