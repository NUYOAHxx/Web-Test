"""Orchestrator: modes (M1/M2/M3/auto), layers L1-L4, validation and ranking."""
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pinocchio as pin

from .collision import CollisionModel, JointDistanceTables
from .model import RobotModel
from .nlp import NLPLayer
from .qp import QPLayer
from .seeds import SeedGenerator


def as_se3(T):
    if isinstance(T, pin.SE3):
        return T
    T = np.asarray(T)
    return pin.SE3(T[:3, :3], T[:3, 3])


@dataclass
class IKRequest:
    targets: Dict[str, object]                   # {"left": SE3/4x4, "right": ...} TCP targets in pelvis frame
    mode: str = "auto"                           # "M1" | "M2" | "M3" | "auto"
    q_current: Optional[np.ndarray] = None       # 16-vector (solver order); None -> nominal
    payload: Dict[str, float] = field(default_factory=dict)  # kg held by each hand
    relative: Optional[object] = None            # desired T_left^-1 T_right (both hands hold one object)
    object_mass: float = 0.0                     # if >0 with two hands: split equally between hands
    top_k: Optional[int] = None


class IKSolver:
    def __init__(self, cfg):
        t0 = time.perf_counter()
        self.cfg = cfg
        self.robot = RobotModel(cfg)
        self.coll = CollisionModel(self.robot, cfg)
        self.qp = QPLayer(self.robot, self.coll, cfg)
        self.nlp = NLPLayer(self.robot, self.coll, cfg)
        self.jt = JointDistanceTables(self.robot, self.coll, cfg)
        self.nlp.jt = self.jt
        self.seeds = SeedGenerator(self.robot, cfg)
        self.init_time = time.perf_counter() - t0

    # ------------------------------------------------------------------ public
    def solve(self, req: IKRequest):
        t_start = time.perf_counter()
        r, sc = self.robot, self.cfg["solver"]
        q_cur = r.q_nominal.copy() if req.q_current is None else np.asarray(req.q_current, float).copy()
        targets = {s: as_se3(T) for s, T in req.targets.items()}
        rel = as_se3(req.relative) if req.relative is not None else None
        payload = dict(req.payload)
        if req.object_mass > 0:
            for s in r.sides:
                payload[s] = payload.get(s, 0.0) + req.object_mass / len(r.sides)
        sides = sorted(targets)
        log = []
        if req.mode == "auto":
            stages = [("M1", None)] + [("M2" if len(sides) == 1 else "M3", b) for b in sc["waist_bounds_schedule"]]
        else:
            stages = [(req.mode, None)]
        pool = []  # valid solutions of all attempted stages
        result = None
        thr = sc["auto_torque_util_threshold"]
        for mode, wb in stages:
            sols, info = self._solve_mode(mode, sides, targets, q_cur, payload, rel, wb)
            valid = [s for s in sols if s["valid"]]
            last_sols = sols
            entry = dict(mode=mode, waist_bound=wb, n_valid=len(valid), **info)
            if valid:
                entry["best_max_util"] = valid[0]["max_util"]
            if req.mode != "auto":
                result = sols
                log.append(entry)
                break
            if not valid:
                entry["escalate_reason"] = "no valid solution (pose/collision/torque/limits)"
                log.append(entry)
                continue
            pool += valid
            if min(v["max_util"] for v in valid) > thr and (mode, wb) != stages[-1]:
                entry["escalate_reason"] = (f"torque utilisation {valid[0]['max_util']:.2f} > "
                                            f"{sc['auto_torque_util_threshold']}")
                log.append(entry)
                continue
            log.append(entry)
            result = valid
            break
        k = req.top_k or sc["top_k"]
        if req.mode == "auto":
            # pool over all stages: prefer solutions under the torque threshold (ranked normally);
            # if none is under it, return the lowest-utilisation ones.
            good = [v for v in pool if v["max_util"] <= thr]
            if not pool:  # nothing valid: return the (invalid) best attempts of the last stage for diagnosis
                result = self._rank(last_sols, q_cur)[:k]
            elif good:
                result = self._rank(good, q_cur)[:k]
            else:
                result = \
                self._rank(sorted(pool, key=lambda v: v["max_util"]), q_cur, keep_order=True)[:k]
        else:
            result = self._rank(result, q_cur)[:k]
        return dict(solutions=result, log=log, time=time.perf_counter() - t_start,
                    success=bool(result) and result[0]["valid"])

    # ------------------------------------------------------------------ one mode
    def _solve_mode(self, mode, sides, targets, q_cur, payload, rel, waist_bound):
        r, sc = self.robot, self.cfg["solver"]
        if mode == "M3" and len(sides) != 2:
            raise ValueError("M3 needs targets for both hands")
        free = r.mode_free_idx(mode, sides)
        pairs = self.coll.relevant_pairs(free)
        offs = self.coll.relevant_offset_pairs(free) if self.cfg["collision"].get("nominal_overlap_slack") is not None \
            else []
        nlp_pairs = list(pairs) + [(i, k) for i, k, _ in offs]
        nlp_lb = [self.cfg["collision"]["nlp_d_safe"]] * len(pairs) + [lb for _, _, lb in offs]
        t = {"seed": 0.0, "qp": 0.0, "nlp": 0.0, "validate": 0.0}
        # ---------------- L1 seeds
        t0 = time.perf_counter()
        seeds = []  # (q0, elbow_targets)
        waist_list = [q_cur[r.waist_idx]] if mode == "M1" else self.seeds.waist_seeds(q_cur, targets)
        for w in waist_list:
            base = q_cur.copy()
            base[r.waist_idx] = w
            seeds.append((base.copy(), None))
            nom = base.copy()
            for s in sides:
                nom[r.arm_idx[s]] = r.q_nominal[r.arm_idx[s]]
            seeds.append((nom, None))
            # swivel-guided seeds (one elbow point per seed, both arms use the same swivel angle)
            pts = {s: self.seeds.swivel_elbow_points(base, s, targets[s], sc["swivel_angles_deg"]) for s in sides}
            n_sw = min(len(v) for v in pts.values()) if pts else 0
            for k in range(n_sw):
                seeds.append((nom.copy(), {s: pts[s][k] for s in sides}))
        if mode != "M1":
            for w in self.seeds.waist_grid(q_cur):
                nom = q_cur.copy()
                nom[r.waist_idx] = w
                for s in sides:
                    nom[r.arm_idx[s]] = r.q_nominal[r.arm_idx[s]]
                seeds.append((nom, None))
        for _ in range(sc["n_random_seeds"]):
            qs = q_cur.copy()
            for s in sides:
                qs[r.arm_idx[s]] = self.seeds.arm_random(s, 1)[0]
            seeds.append((qs, None))
        t["seed"] += time.perf_counter() - t0
        # ---------------- L2 QP
        cands = []
        for q0, el in seeds:
            t0 = time.perf_counter()
            q, info = self.qp.solve(q0, q_cur, free, targets, rel, pairs, elbow_targets=el,
                                    iters=sc["qp_iters"] + (30 if el else 0))
            t["qp"] += time.perf_counter() - t0
            m = self._metrics(q, q_cur, targets, rel, payload, pairs, mesh=False)
            m["seed_kind"] = "swivel" if el else "plain"
            cands.append((q, m))
        # choose diverse, good candidates for refinement
        cands.sort(key=lambda c: c[1]["pos_err"] * 100 + c[1]["rot_err"] + 0.5 * c[1]["waist_disp"]
                   + (0 if c[1]["min_cap_dist"] > 0 else 1.0))
        chosen = []
        for q, m in cands:
            if all(np.max(np.abs(q - c[0])) > 0.15 for c in chosen):
                chosen.append((q, m))
            if len(chosen) >= sc["n_refine"]:
                break
        # ---------------- L3 NLP
        sols = []
        for q, mqp in chosen:
            t0 = time.perf_counter()
            qn, info = self.nlp.solve(q, q_cur, free, targets, payload, rel, nlp_pairs, waist_bound=waist_bound,
                                      pair_lb=nlp_lb)
            t["nlp"] += time.perf_counter() - t0
            t0 = time.perf_counter()
            met = self._metrics(qn, q_cur, targets, rel, payload, pairs, mesh=True)
            t["validate"] += time.perf_counter() - t0
            met.update(q=qn, mode=mode, nlp_status=info["status"], nlp_iters=info["iters"],
                       seed_kind=mqp["seed_kind"], waist_bound=waist_bound,
                       # capsules are a conservative proxy; the mesh check is the ground truth. Flag
                       # solutions accepted by the mesh check although the capsule constraint is violated.
                       capsule_violation=bool(met["min_cap_dist"] < self.cfg["collision"]["nlp_d_safe"] - 1e-3))
            if waist_bound is not None and met["waist_disp_max"] > waist_bound + 1e-6:
                met["valid"] = False
                met["fail"].append("waist_bound")
            sols.append(met)
        sols = self._rank(sols, q_cur)
        return sols, dict(n_seeds=len(seeds), n_pairs=len(pairs), n_free=len(free), times=t)

    # ------------------------------------------------------------------ validation (L4) + metrics
    def _metrics(self, q, q_cur, targets, rel, payload, pairs, mesh=True):
        r, sc = self.robot, self.cfg["solver"]
        T = r.fk(q)
        pos_err = max((np.linalg.norm(T[s].translation - targets[s].translation) for s in targets), default=0.0)
        rot_err = max((np.linalg.norm(pin.log3(targets[s].rotation.T @ T[s].rotation)) for s in targets), default=0.0)
        rel_pos = rel_rot = 0.0
        if rel is not None:
            Trel = T["left"].inverse() * T["right"]
            rel_pos = np.linalg.norm(Trel.translation - rel.translation)
            rel_rot = np.linalg.norm(pin.log3(rel.rotation.T @ Trel.rotation))
        tau = r.static_torque(q, payload)
        util = r.torque_util(tau)
        dcap = self.coll.capsule_distances(q, pairs) if pairs else np.array([np.inf])
        wd = q[r.waist_idx] - q_cur[r.waist_idx]
        out = dict(pos_err=pos_err, rot_err=rot_err, rel_pos_err=rel_pos, rel_rot_err=rel_rot,
                   max_util=float(util.max()), util_joint=r.active_names[int(np.argmax(util))],
                   min_cap_dist=float(dcap.min()), waist_disp=float(np.linalg.norm(wd)),
                   waist_disp_max=float(np.abs(wd).max()), dist_cur=float(np.linalg.norm(q - q_cur)),
                   manip={s: r.manipulability(q, s) for s in targets})
        if mesh:
            coll, dmin, dpair, hits = self.coll.mesh_check(q)
            fail = []
            if pos_err > sc["pos_tol"] or rot_err > sc["rot_tol"]:
                fail.append("pose")
            if rel is not None and (rel_pos > sc["pos_tol"] or rel_rot > sc["rot_tol"]):
                fail.append("relative_pose")
            if coll or dmin < self.cfg["collision"]["validate_min_mesh_distance"]:
                fail.append(f"mesh_collision{hits[:2]}")
            if util.max() > 1.0 + 1e-6:
                fail.append(f"torque({out['util_joint']}={util.max():.2f})")
            if np.any(q < r.lower - 1e-6) or np.any(q > r.upper + 1e-6):
                fail.append("joint_limits")
            out.update(mesh_collision=coll, min_mesh_dist=dmin, min_mesh_pair=dpair, fail=fail, valid=not fail)
        return out

    def _rank(self, sols, q_cur, keep_order=False):
        sc = self.cfg["solver"]
        for s in sols:
            if "min_mesh_dist" in s:
                s["score"] = (sc["rank_w_waist"] * s["waist_disp"] + sc["rank_w_torque"] * s["max_util"]
                              + sc["rank_w_dist_cur"] * s["dist_cur"]
                              + sc["rank_w_clearance"] / max(s["min_mesh_dist"], 1e-3)
                              + (0 if s.get("valid") else 1e3))
        if not keep_order:
            sols = sorted(sols, key=lambda s: s.get("score", 1e9))
        out = []
        for s in sols:  # remove near-duplicates
            if all(np.max(np.abs(s["q"] - o["q"])) > 0.05 for o in out):
                out.append(s)
        return out
