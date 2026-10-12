"""L3: CasADi + IPOPT refinement with exact pose / static-torque / capsule-distance constraints."""
import time

import casadi as ca
import numpy as np

from .casadi_kin import SymKinematics, seg_dist_sx


def _flat(R, p):
    return ca.vertcat(ca.reshape(R, 9, 1), p)


class NLPLayer:
    def __init__(self, robot, coll, cfg):
        self.robot, self.coll, self.cfg = robot, coll, cfg
        self.kin = SymKinematics(robot.model)
        self.cache = {}
        k = self.kin
        self.ee = {s: k.frame(fid) for s, fid in robot.ee_frame.items()}
        self.cap_w = [(k.point(j, p0), k.point(j, p1)) for j, (p0, p1) in zip(coll.cap_joint, coll.cap_pts)]
        self.n_build = 0
        self.build_time = 0.0

    def _build(self, free_idx, sides, relative, pairs):
        t0 = time.perf_counter()
        r, k, sc = self.robot, self.kin, self.cfg["solver"]
        nq = r.model.nq
        q = k.q
        # parameters
        P_q = ca.SX.sym("qref", nq)               # current/fixed configuration
        P_tgt = {s: ca.SX.sym(f"T_{s}", 12) for s in sides}
        P_rel = ca.SX.sym("T_rel", 12)
        P_mass = ca.SX.sym("mass", 2)             # payload per side (left, right)
        P_w = ca.SX.sym("w", 3)                   # w_waist, w_reg, w_torque
        x = ca.SX.sym("x", len(free_idx))
        fixed = [i for i in range(nq) if i not in set(free_idx)]
        q_sub = ca.SX.zeros(nq)
        for n, i in enumerate(free_idx):
            q_sub[i] = x[n]
        for i in fixed:
            q_sub[i] = P_q[i]

        g, lbg_tags = [], []
        for s in sides:
            R, p = self.ee[s]
            Rt = ca.reshape(P_tgt[s][:9], 3, 3)
            pt = P_tgt[s][9:]
            g += [p - pt, _rot_err(Rt, R), 3 - ca.trace(Rt.T @ R)]
            lbg_tags += ["pos"] * 3 + ["rot"] * 3 + ["rot_hemi"]
        if relative:
            (RL, pL), (RR, pR) = self.ee["left"], self.ee["right"]
            Rrel = RL.T @ RR
            prel = RL.T @ (pR - pL)
            Rt = ca.reshape(P_rel[:9], 3, 3)
            g += [prel - P_rel[9:], _rot_err(Rt, Rrel), 3 - ca.trace(Rt.T @ Rrel)]
            lbg_tags += ["pos"] * 3 + ["rot"] * 3 + ["rot_hemi"]
        tau = k.tau_g
        for n, s in enumerate(r.sides):
            tau = tau + k.payload_torque(self.ee[s][1], P_mass[n])
        util = tau / ca.DM(r.tau_max)
        g.append(util)
        lbg_tags += ["tau"] * nq
        for (i, j) in pairs:
            d = seg_dist_sx(*self.cap_w[i], *self.cap_w[j]) - r_(self.coll, i, j)
            g.append(d)
            lbg_tags.append("coll")
        # exact 2-joint mesh-distance tables (shoulder socket pairs), see collision.JointDistanceTables
        jt = getattr(self, "jt", None)
        if jt is not None and jt.enabled:
            for expr, _ in jt.casadi_constraints(q, free_idx):
                g.append(expr)
                lbg_tags.append("jt")
        widx = list(r.waist_idx)
        J = P_w[0] * ca.sumsqr(q[widx] - P_q[widx])
        arm = [i for i in free_idx if i not in widx]
        if arm:
            J += P_w[1] * ca.sumsqr(q[arm] - P_q[arm])
        J += P_w[2] * ca.sumsqr(util)
        G = ca.vertcat(*g)
        # substitute q -> q_sub (free vars + fixed params)
        J, G = ca.substitute([J, G], [q], [q_sub])
        params = ca.vertcat(P_q, *[P_tgt[s] for s in sides], P_rel, P_mass, P_w)
        opts = {
            "ipopt.print_level": 0, "print_time": 0, "ipopt.sb": "yes",
            "ipopt.max_iter": int(sc["ipopt_max_iter"]), "ipopt.tol": 1e-7,
            "ipopt.acceptable_tol": 1e-5, "ipopt.acceptable_iter": 5,
            "ipopt.mu_strategy": "adaptive",
        }
        solver = ca.nlpsol("ik", "ipopt", {"x": x, "f": J, "g": G, "p": params}, opts)
        self.n_build += 1
        self.build_time += time.perf_counter() - t0
        return dict(solver=solver, tags=lbg_tags, free=list(free_idx), sides=list(sides), relative=relative)

    def solve(self, q0, q_ref, free_idx, targets, payload, relative=None, pairs=(),
              waist_bound=None, pos_tol=None, rot_tol=None, pair_lb=None):
        """pair_lb: optional per-pair lower bound on the capsule distance (default d_safe)."""
        r, sc, cc = self.robot, self.cfg["solver"], self.cfg["collision"]
        sides = sorted(targets.keys())
        key = (tuple(free_idx), tuple(sides), relative is not None, tuple(pairs))
        if key not in self.cache:
            self.cache[key] = self._build(list(free_idx), sides, relative is not None, list(pairs))
        B = self.cache[key]
        pos_tol = sc["nlp_pos_tol"] if pos_tol is None else pos_tol
        rot_tol = sc["nlp_rot_tol"] if rot_tol is None else rot_tol
        rho = r.torque_margin
        lbg, ubg = [], []
        n_coll = 0
        for tag in B["tags"]:
            if tag == "pos":   # per-axis box => ||e|| <= pos_tol
                b = pos_tol / np.sqrt(3); lbg.append(-b); ubg.append(b)
            elif tag == "rot":  # per-axis box on sin(theta)*axis => theta <= rot_tol (in the cos>0 hemisphere)
                b = np.sin(rot_tol) / np.sqrt(3); lbg.append(-b); ubg.append(b)
            elif tag == "jt":
                lbg.append(float(cc["joint_tables"].get("margin", 0.005))); ubg.append(np.inf)
            elif tag == "rot_hemi":  # 3 - tr <= 1  <=> theta <= 90 deg (rules out the theta~180deg branch)
                lbg.append(-np.inf); ubg.append(1.0)
            elif tag == "tau":
                lbg.append(-rho); ubg.append(rho)
            else:
                lbg.append(cc["nlp_d_safe"] if pair_lb is None else pair_lb[n_coll]); ubg.append(np.inf)
                n_coll += 1
        # torque rows of joints we do not check -> unbounded (none by default)
        lbx = r.lower[free_idx].copy()
        ubx = r.upper[free_idx].copy()
        if waist_bound is not None:
            for n, i in enumerate(free_idx):
                if i in r.waist_idx:
                    lbx[n] = max(lbx[n], q_ref[i] - waist_bound)
                    ubx[n] = min(ubx[n], q_ref[i] + waist_bound)
        p = [q_ref]
        for s in sides:
            T = targets[s]
            p.append(np.concatenate([T.rotation.reshape(-1, order="F"), T.translation]))
        if relative is not None:
            p.append(np.concatenate([relative.rotation.reshape(-1, order="F"), relative.translation]))
        else:
            p.append(np.zeros(12))
        p.append([payload.get(s, 0.0) for s in r.sides])
        p.append([sc["w_waist"], sc["w_reg"], sc["w_torque"]])
        x0 = np.clip(q0[free_idx], lbx, ubx)
        t0 = time.perf_counter()
        sol = B["solver"](x0=x0, p=np.concatenate([np.ravel(v) for v in p]), lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
        dt = time.perf_counter() - t0
        st = B["solver"].stats()
        q = q0.copy()
        q[free_idx] = np.array(sol["x"]).ravel()
        q[[i for i in range(len(q)) if i not in set(free_idx)]] = q_ref[[i for i in range(len(q)) if i not in set(free_idx)]]
        return q, dict(success=bool(st["success"]), status=st["return_status"], time=dt, iters=st.get("iter_count"))


def _rot_err(Rt, R):
    """vee(skew(Rt^T R)) = sin(theta) * axis. Unlike 3 - tr(Rt^T R) its Jacobian is full rank at the
    solution, which keeps the IPOPT constraint qualification (LICQ) healthy."""
    E = Rt.T @ R
    return 0.5 * ca.vertcat(E[2, 1] - E[1, 2], E[0, 2] - E[2, 0], E[1, 0] - E[0, 1])


def r_(coll, i, j):
    return coll.cap_r[i] + coll.cap_r[j]
