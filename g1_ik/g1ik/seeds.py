"""L1: multi-start seeds.

G1's arm is NOT an exact SRS arm (checked in tests/check_srs.py: shoulder axes offset 13.8 mm / 6.2 mm,
wrist pitch-yaw axes offset 46 mm), so a closed-form SRS solution is not exact. We therefore use
"swivel(arm-angle)-guided numerical seeds": the elbow is pulled towards points on the SRS swivel circle
(computed from approximate shoulder/elbow/wrist centres) during the first half of a QP run, which yields
seeds that cover the 1-D self-motion manifold, plus current / nominal / random seeds.
For a true SRS arm a closed-form IK (Shimizu 2008) could replace `swivel_elbow_points`.
"""
import numpy as np
import pinocchio as pin


class SeedGenerator:
    def __init__(self, robot, cfg, rng=None):
        self.robot, self.cfg = robot, cfg
        self.rng = rng or np.random.default_rng(0)
        m = robot.model
        q0 = np.zeros(m.nq)
        d = m.createData()
        pin.framesForwardKinematics(m, d, q0)
        self.geo = {}
        for s, a in cfg["robot"]["arms"].items():
            js, je, jw = (m.getJointId(a[k]) for k in ("shoulder_joint", "elbow_joint", "wrist_joint"))
            S, E, W = d.oMi[js].translation, d.oMi[je].translation, d.oMi[jw].translation
            T_ee = d.oMf[robot.ee_frame[s]]
            # wrist centre expressed in TCP frame (wrist angles at zero)
            w_in_ee = T_ee.inverse().act(W)
            self.geo[s] = dict(js=js, L1=np.linalg.norm(E - S), L2=np.linalg.norm(W - E), w_in_ee=w_in_ee)

    def swivel_elbow_points(self, q_waist_seed, side, T_target, angles_deg):
        m, r = self.robot.model, self.robot
        g = self.geo[side]
        d = m.createData()
        pin.forwardKinematics(m, d, q_waist_seed)
        S = d.oMi[g["js"]].translation
        W = T_target.act(g["w_in_ee"])
        D = np.linalg.norm(W - S)
        L1, L2 = g["L1"], g["L2"]
        if D >= L1 + L2 - 1e-3 or D < 1e-3:
            return []
        n = (W - S) / D
        a = (L1 ** 2 - L2 ** 2 + D ** 2) / (2 * D)
        C = S + a * n
        rc = np.sqrt(max(L1 ** 2 - a ** 2, 0.0))
        down = np.array([0, 0, -1.0])
        u = down - (down @ n) * n
        if np.linalg.norm(u) < 1e-6:
            u = np.array([1.0, 0, 0]) - n[0] * n
        u /= np.linalg.norm(u)
        v = np.cross(n, u)
        if side == "right":
            v = -v  # positive angle = elbow outward for both arms
        return [C + rc * (np.cos(np.radians(t)) * u + np.sin(np.radians(t)) * v) for t in angles_deg]

    def waist_seeds(self, q_cur, targets):
        """current waist + heuristic 'turn/lean towards the target(s)'."""
        r = self.robot
        out = [q_cur[r.waist_idx].copy()]
        p = np.mean([T.translation for T in targets.values()], axis=0)
        yaw = np.arctan2(p[1], max(p[0], 0.05))
        h = r.q_nominal.copy()
        w = q_cur[r.waist_idx].copy()
        names = r.waist_names
        for n, jn in enumerate(names):
            if "yaw" in jn:
                w[n] = 0.6 * yaw
            elif "pitch" in jn:
                reach = np.linalg.norm(p[:2])
                w[n] = np.clip(0.8 * (reach - 0.30) + 0.5 * max(0.0, 0.10 - p[2]), 0.0, 0.4)
        w = np.clip(w, r.lower[r.waist_idx], r.upper[r.waist_idx])
        if np.linalg.norm(w - out[0]) > 0.05:
            out.append(w)
        # two hands: also turn the torso towards the mean TCP x-axis heading (e.g. box orientation)
        if len(targets) > 1:
            hx = np.mean([T.rotation[:, 0] for T in targets.values()], axis=0)
            w2 = w.copy()
            for n, jn in enumerate(names):
                if "yaw" in jn:
                    w2[n] = 0.5 * (yaw + np.arctan2(hx[1], max(hx[0], 1e-3)))
            w2 = np.clip(w2, r.lower[r.waist_idx], r.upper[r.waist_idx])
            if all(np.linalg.norm(w2 - o) > 0.05 for o in out):
                out.append(w2)
        return out

    def waist_grid(self, q_cur):
        """coarse grid of waist seeds (config solver.waist_seed_grid), used with nominal arms only."""
        r, g = self.robot, self.cfg["solver"].get("waist_seed_grid") or {}
        out = []
        for a in g.get("yaw", [0.0]):
            for b in g.get("pitch", [0.0]):
                w = q_cur[r.waist_idx].copy()
                for n, jn in enumerate(r.waist_names):
                    w[n] = a if "yaw" in jn else (b if "pitch" in jn else w[n])
                out.append(np.clip(w, r.lower[r.waist_idx], r.upper[r.waist_idx]))
        return out

    def arm_random(self, side, k):
        r = self.robot
        idx = r.arm_idx[side]
        lo, hi = r.lower[idx], r.upper[idx]
        mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
        return [self.rng.uniform(mid - 0.8 * half, mid + 0.8 * half) for _ in range(k)]
