"""Benchmark of the G1 IK prototype: reachable / hard targets per mode, waist-needed, box with payload,
near arm-arm collision. All numbers are measured on this machine (time.perf_counter).

usage: python tests/run_benchmark.py [--n N] [--out results/benchmark.json]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pinocchio as pin

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from g1ik.config import load_config  # noqa: E402
from g1ik.solver import IKSolver, IKRequest  # noqa: E402


class Gen:
    """Target generator. 'reachable' targets = FK of random configurations that are collision free for
    BOTH the exact meshes (>= 15 mm) and the conservative capsule model (>= d_safe), and torque feasible."""

    def __init__(self, S, seed=0):
        self.S, self.r, self.rng = S, S.robot, np.random.default_rng(seed)
        self.dsafe = S.cfg["collision"]["nlp_d_safe"]

    def ok(self, q, payload=None, free=None):
        S, r = self.S, self.r
        coll, dmin, _, _ = S.coll.mesh_check(q)
        if coll or dmin < 0.015:
            return False
        pairs = S.coll.relevant_pairs(free) if free is not None else S.coll.cap_pairs
        if pairs and S.coll.capsule_distances(q, pairs).min() < self.dsafe + 0.002:
            return False
        return r.torque_util(r.static_torque(q, payload or {})).max() < 0.85

    def config(self, sides, waist=None, q_base=None, payload=None, frac=0.75):
        r = self.r
        for _ in range(5000):
            q = (r.q_nominal if q_base is None else q_base).copy()
            for s in sides:
                idx = r.arm_idx[s]
                lo, hi = r.lower[idx], r.upper[idx]
                mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
                q[idx] = self.rng.uniform(mid - frac * half, mid + frac * half)
            if waist is not None:
                q[r.waist_idx] = waist() if callable(waist) else waist
            free = r.mode_free_idx("M3", ["left", "right"])
            if self.ok(q, payload, free):
                return q
        raise RuntimeError("could not sample a feasible configuration")

    def waist_far(self):
        yaw = self.rng.choice([-1, 1]) * self.rng.uniform(0.45, 0.9)
        pitch = self.rng.uniform(0.15, 0.45)
        return np.array([yaw, pitch])

    def waist_any(self):
        return np.array([self.rng.uniform(-0.6, 0.6), self.rng.uniform(-0.1, 0.4)])


def run_case(S, name, req, expect=None, extra=None):
    res = S.solve(req)
    sols = res["solutions"]
    best = sols[0] if sols else None
    row = dict(scenario=name, mode=req.mode, success=bool(res["success"]), time=res["time"],
               final_mode=best["mode"] if best else None,
               stages=[dict(mode=e["mode"], waist_bound=e["waist_bound"], n_valid=e["n_valid"],
                            reason=e.get("escalate_reason"), times=e["times"]) for e in res["log"]],
               n_solutions=len([s for s in sols if s.get("valid")]))
    if best is not None:
        row.update(pos_err=best["pos_err"], rot_err=best["rot_err"], rel_pos_err=best["rel_pos_err"],
                   rel_rot_err=best["rel_rot_err"], waist_disp=best["waist_disp"], max_util=best["max_util"],
                   util_joint=best["util_joint"], min_mesh_dist=best["min_mesh_dist"],
                   min_mesh_pair=list(best["min_mesh_pair"]), min_cap_dist=best["min_cap_dist"],
                   fail=best["fail"], q=best["q"].tolist(), capsule_violation=best.get("capsule_violation"),
                   nlp_status=best.get("nlp_status"))
    if expect is not None:
        row["expect"] = expect
    if extra:
        row.update(extra)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "results", "benchmark.json"))
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    cfg = load_config()
    t0 = time.perf_counter()
    S = IKSolver(cfg)
    init_time = time.perf_counter() - t0
    r = S.robot
    G = Gen(S, seed=0)
    N = a.n
    rows = []
    qn = r.q_nominal.copy()

    def want(tag):
        return not a.only or tag in a.only.split(",")

    def log(row):
        rows.append(row)
        print(f"[{row['scenario']:<22}] ok={row['success']!s:<5} t={row['time']:.2f}s mode={row.get('final_mode')} "
              f"pos={row.get('pos_err', np.nan):.1e} wd={row.get('waist_disp', np.nan):.3f} "
              f"util={row.get('max_util', np.nan):.2f} dmesh={row.get('min_mesh_dist', np.nan):.3f} "
              f"fail={row.get('fail')} stages={[(s['mode'], s['waist_bound'], s['n_valid']) for s in row['stages']]}",
              flush=True)

    # --- warm-up (builds the cached CasADi/IPOPT problems; build time reported separately)
    t0 = time.perf_counter()
    for side in ["left", "right"]:
        q = G.config([side])
        S.solve(IKRequest(targets={side: r.fk(q)[side]}, mode="M1"))
        S.solve(IKRequest(targets={side: r.fk(q)[side]}, mode="M2"))
    q = G.config(["left", "right"])
    T = r.fk(q)
    S.solve(IKRequest(targets={"left": T["left"], "right": T["right"]}, mode="M3"))
    S.solve(IKRequest(targets={"left": T["left"], "right": T["right"]}, mode="M3",
                      relative=T["left"].inverse() * T["right"], object_mass=1.0))
    warm = time.perf_counter() - t0
    G = Gen(S, seed=1)

    # 1) M1 reachable (single arm, waist locked at current = 0)
    if want("m1"):
        for k in range(N):
            side = ["left", "right"][k % 2]
            q = G.config([side])
            log(run_case(S, f"M1_reachable_{side[0]}", IKRequest(targets={side: r.fk(q)[side]}, mode="M1")))
    # 2) M1 hard: targets generated with a large waist motion; M1 is expected to fail (or find an
    #    alternative arm config); this measures how often M1 alone suffices.
    # hard targets: generated with a large waist motion AND the wrist centre outside the reach sphere
    # (L1+L2) of the shoulder at the current waist -> provably unreachable for M1.
    hard = []
    geo = S.seeds.geo
    while len(hard) < N:
        side = ["left", "right"][len(hard) % 2]
        q = G.config([side], waist=G.waist_far, frac=0.6)
        d0 = r.model.createData()
        pin.forwardKinematics(r.model, d0, qn)
        Wc = r.fk(q)[side].act(geo[side]["w_in_ee"])
        if np.linalg.norm(Wc - d0.oMi[geo[side]["js"]].translation) > geo[side]["L1"] + geo[side]["L2"] + 0.01:
            hard.append((side, q))
    if want("m1hard"):
        for side, q in hard:
            log(run_case(S, "M1_hard", IKRequest(targets={side: r.fk(q)[side]}, mode="M1")))
    # 3) M2 reachable: waist + one arm, generated with moderate waist
    if want("m2"):
        for k in range(N):
            side = ["left", "right"][k % 2]
            q = G.config([side], waist=G.waist_any)
            log(run_case(S, "M2_reachable", IKRequest(targets={side: r.fk(q)[side]}, mode="M2")))
    # 4) auto on the hard set (waist-needed): expected M1 -> M2 escalation
    if want("auto"):
        for side, q in hard:
            log(run_case(S, "auto_waist_needed", IKRequest(targets={side: r.fk(q)[side]}, mode="auto"),
                         extra=dict(gen_waist=q[r.waist_idx].tolist())))
    # 5) specific waist-needed target: left hand low and to the right side of the body (cross-body pick)
    if want("specific"):
        Rdown = pin.rpy.rpyToMatrix(0.0, np.pi / 2, 0.0)  # tcp x-axis pointing down
        for p in [(0.35, -0.15, 0.0), (0.40, -0.05, -0.05), (0.30, -0.25, 0.05)]:
            T = pin.SE3(Rdown, np.array(p))
            log(run_case(S, "specific_cross_body", IKRequest(targets={"left": T}, mode="auto"),
                         extra=dict(target=list(p))))
    # 6) torque-driven escalation: heavy single-hand payload on far-reaching targets
    if want("torque"):
        for k in range(max(N // 2, 5)):
            side = ["left", "right"][k % 2]
            for _ in range(200):
                q = G.config([side])
                p = r.fk(q)[side].translation
                if p[0] > 0.30:  # far forward reach -> large shoulder torque
                    break
            for m in [3.0, 4.5]:
                log(run_case(S, f"torque_payload_{m}kg", IKRequest(targets={side: r.fk(q)[side]}, mode="auto",
                                                                  payload={side: m}), extra=dict(payload=m)))
    # 7) M3 separate targets (waist + both arms)
    if want("m3"):
        for k in range(N):
            q = G.config(["left", "right"], waist=G.waist_any)
            T = r.fk(q)
            log(run_case(S, "M3_separate", IKRequest(targets={"left": T["left"], "right": T["right"]}, mode="M3")))
    # 8) both hands hold a box (relative pose constraint + payload), auto (M1 = both arms, waist locked -> M3).
    #    (a) box_reachable: the box pose is defined in the TORSO frame of a sampled waist configuration and
    #        a certificate solve (both arms, waist fixed at that sample) must succeed -> known feasible.
    #        The test then starts from the nominal waist (0, 0).
    #    (b) box_random: box pose sampled in the pelvis frame, feasibility unknown (hard set).
    width = 0.30
    torso_fid = r.model.getFrameId("torso_link")

    def box_targets(Tb):
        TL = pin.SE3(Tb.rotation, Tb.act(np.array([0, width / 2, 0])))   # TCP x (forearm) along box forward
        TR = pin.SE3(Tb.rotation, Tb.act(np.array([0, -width / 2, 0])))
        return TL, TR

    if want("box"):
        n_ok, tries = 0, 0
        while n_ok < N and tries < 20 * N:
            tries += 1
            w = np.array([G.rng.uniform(-0.6, 0.6), G.rng.uniform(0.0, 0.4)])
            qw = qn.copy()
            qw[r.waist_idx] = w
            dd = r.model.createData()
            pin.framesForwardKinematics(r.model, dd, qw)
            Tloc = pin.SE3(pin.rpy.rpyToMatrix(0, 0, G.rng.uniform(-0.3, 0.3)),
                           np.array([G.rng.uniform(0.30, 0.42), G.rng.uniform(-0.06, 0.06), G.rng.uniform(0.0, 0.15)]))
            Tb = dd.oMf[torso_fid] * Tloc
            TL, TR = box_targets(Tb)
            mass = [1.0, 3.0, 5.0][n_ok % 3]
            cert = S.solve(IKRequest(targets={"left": TL, "right": TR}, mode="M1", q_current=qw,
                                     relative=TL.inverse() * TR, object_mass=mass))
            if not cert["success"]:
                continue
            n_ok += 1
            log(run_case(S, "box_reachable", IKRequest(targets={"left": TL, "right": TR}, mode="auto",
                                                        relative=TL.inverse() * TR, object_mass=mass),
                         extra=dict(gen_waist=w.tolist(), object_mass=mass, box=Tb.homogeneous.tolist())))
        print(f"box_reachable: {n_ok} certified boxes out of {tries} sampled", flush=True)
        for k in range(N):
            c = np.array([G.rng.uniform(0.28, 0.45), G.rng.uniform(-0.20, 0.20), G.rng.uniform(-0.10, 0.20)])
            Tb = pin.SE3(pin.rpy.rpyToMatrix(0, 0, G.rng.uniform(-0.6, 0.6)), c)
            TL, TR = box_targets(Tb)
            mass = [1.0, 3.0, 5.0][k % 3]
            log(run_case(S, "box_random_hard", IKRequest(targets={"left": TL, "right": TR}, mode="auto",
                                                          relative=TL.inverse() * TR, object_mass=mass),
                         extra=dict(object_mass=mass, box=Tb.homogeneous.tolist())))
    # 9) arms near each other: hands crossing in front of the chest, 4-8 cm apart (M3), and a target that
    #    puts the left hand INSIDE the locked right forearm (M1) -> must be rejected.
    if want("near"):
        # (a) hands side by side in front of the chest, TCP origins 11-14 cm apart (hand meshes ~1-4 cm apart)
        for k in range(N):
            gap = G.rng.uniform(0.11, 0.14)
            c = np.array([G.rng.uniform(0.28, 0.40), G.rng.uniform(-0.08, 0.08), G.rng.uniform(-0.05, 0.15)])
            TL = pin.SE3(np.eye(3), c + np.array([0, gap / 2, 0]))
            TR = pin.SE3(np.eye(3), c - np.array([0, gap / 2, 0]))
            log(run_case(S, "near_hands_side_by_side", IKRequest(targets={"left": TL, "right": TR}, mode="M3"),
                         extra=dict(gap=gap)))
        # (b) hard: crossed hands (left hand right of the right hand)
        for k in range(5):
            gap = G.rng.uniform(0.12, 0.16)
            c = np.array([G.rng.uniform(0.30, 0.38), G.rng.uniform(-0.03, 0.03), G.rng.uniform(0.05, 0.15)])
            TL = pin.SE3(pin.rpy.rpyToMatrix(0, 0, -0.5), c - np.array([0, gap / 2, 0]))
            TR = pin.SE3(pin.rpy.rpyToMatrix(0, 0, 0.5), c + np.array([0, gap / 2, 0]))
            log(run_case(S, "near_crossed_hard", IKRequest(targets={"left": TL, "right": TR}, mode="M3"),
                         extra=dict(gap=gap)))
        for k in range(3):
            q = qn.copy()
            pr = r.fk(q)["right"]
            T = pin.SE3(pr.rotation, pr.translation + np.array([0.0, 0.0, 0.01 * k]))
            log(run_case(S, "infeasible_inside_right_hand", IKRequest(targets={"left": T}, mode="M1"),
                         expect="fail"))
    out = dict(init_time=init_time, warmup_time=warm, nlp_builds=S.nlp.n_build, nlp_build_time=S.nlp.build_time,
               rows=rows, machine=os.uname().nodename, cpu=os.cpu_count())
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1, default=float)
    print("saved", a.out)


if __name__ == "__main__":
    main()
