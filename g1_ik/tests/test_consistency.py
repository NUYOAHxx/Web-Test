"""Cross-check the hand-written pieces against pinocchio / Coal on random configurations."""
import os, sys
import numpy as np
import casadi as ca
import pinocchio as pin
import coal
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from g1ik.config import load_config
from g1ik.solver import IKSolver
from g1ik.casadi_kin import seg_dist_sx

cfg = load_config()
S = IKSolver(cfg)
r, k, coll = S.robot, S.nlp.kin, S.coll
rng = np.random.default_rng(1)
fk = ca.Function("fk", [k.q], [k.frame(r.ee_frame["left"])[1], k.tau_g + k.payload_torque(k.frame(r.ee_frame["left"])[1], 2.0)])
e_fk = e_tau = e_cap = e_capsx = 0
for _ in range(50):
    q = rng.uniform(r.lower, r.upper)
    p, tau = fk(q)
    T = r.fk(q)["left"]
    e_fk = max(e_fk, np.linalg.norm(np.array(p).ravel() - T.translation))
    # pinocchio static torque with external force on the left TCP
    tau_ref = r.static_torque(q, {"left": 2.0})
    e_tau = max(e_tau, np.abs(np.array(tau).ravel() - tau_ref).max())
    # analytic capsule distance vs Coal capsule distance
    W = coll.capsule_world(q)
    for (i, j) in coll.cap_pairs[:40]:
        d_np = coll.capsule_distances(q, [(i, j)])[0]
        geo = []
        for n in (i, j):
            a, b = W[n]
            L = np.linalg.norm(b - a); z = (b - a) / max(L, 1e-9)
            x = np.cross(z, [1, 0, 0]); x = x if np.linalg.norm(x) > 1e-6 else np.cross(z, [0, 1, 0]); x /= np.linalg.norm(x)
            R = np.column_stack([x, np.cross(z, x), z])
            geo.append((coal.Capsule(coll.cap_r[n], max(L, 1e-6)), coal.Transform3s(R, 0.5 * (a + b))))
        req, res = coal.DistanceRequest(), coal.DistanceResult()
        d_coal = coal.distance(geo[0][0], geo[0][1], geo[1][0], geo[1][1], req, res)
        if d_np > 1e-3:
            e_cap = max(e_cap, abs(d_np - d_coal))
        P = [ca.DM(v) for v in (*W[i], *W[j])]
        d_sx = float(seg_dist_sx(*P)) - coll.cap_r[i] - coll.cap_r[j]
        e_capsx = max(e_capsx, abs(d_sx - d_np))
print(f"max |FK casadi - pinocchio|          = {e_fk:.2e} m")
print(f"max |tau casadi - pinocchio(static)| = {e_tau:.2e} Nm")
print(f"max |capsule numpy - Coal| (d>1mm)   = {e_cap:.2e} m")
print(f"max |capsule casadi - numpy|         = {e_capsx:.2e} m")
