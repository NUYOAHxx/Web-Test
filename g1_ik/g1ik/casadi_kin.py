"""Symbolic (CasADi) forward kinematics / static torque built directly from a pinocchio Model.

The pip wheel of pinocchio ('pin') ships without the pinocchio.casadi bindings, so we re-implement the
few quantities we need for revolute (RX/RY/RZ/RevoluteUnaligned) joints. Static gravity torque is obtained
exactly as the gradient of the potential energy:  g(q) = dU/dq,  U = -sum_k m_k * g . c_k(q).
Verified against pinocchio numerically in tests/test_consistency.py.
"""
import casadi as ca
import numpy as np
import pinocchio as pin


def _rot(axis, q):
    c, s = ca.cos(q), ca.sin(q)
    x, y, z = axis
    C = 1 - c
    return ca.vertcat(
        ca.horzcat(c + x * x * C, x * y * C - z * s, x * z * C + y * s),
        ca.horzcat(y * x * C + z * s, c + y * y * C, y * z * C - x * s),
        ca.horzcat(z * x * C - y * s, z * y * C + x * s, c + z * z * C),
    )


def _axis(jm):
    sn = jm.shortname()
    if sn == "JointModelRX":
        return (1.0, 0.0, 0.0)
    if sn == "JointModelRY":
        return (0.0, 1.0, 0.0)
    if sn == "JointModelRZ":
        return (0.0, 0.0, 1.0)
    if "RevoluteUnaligned" in sn:
        return tuple(np.asarray(jm.extract().axis, dtype=float))
    raise NotImplementedError(f"joint type {sn} not supported")


class SymKinematics:
    def __init__(self, model: pin.Model):
        self.model = model
        self.q = ca.SX.sym("q", model.nq)
        R = [ca.SX.eye(3)]
        p = [ca.SX.zeros(3)]
        for j in range(1, model.njoints):
            jm = model.joints[j]
            Mj = model.jointPlacements[j]
            par = model.parents[j]
            Rj = R[par] @ ca.DM(Mj.rotation) @ _rot(_axis(jm), self.q[jm.idx_q])
            pj = p[par] + R[par] @ ca.DM(Mj.translation)
            R.append(Rj)
            p.append(pj)
        self.R, self.p = R, p
        g = np.asarray(model.gravity.linear, dtype=float)
        U = 0
        for j in range(1, model.njoints):
            I = model.inertias[j]
            c = p[j] + R[j] @ ca.DM(I.lever)
            U = U - I.mass * ca.dot(ca.DM(g), c)
        self.tau_g = ca.gradient(U, self.q)
        self.g = g

    def point(self, joint, local):
        return self.p[joint] + self.R[joint] @ ca.DM(np.asarray(local, dtype=float))

    def frame(self, fid):
        f = self.model.frames[fid]
        j = f.parentJoint
        return self.R[j] @ ca.DM(f.placement.rotation), self.point(j, f.placement.translation)

    def payload_torque(self, frame_pos, mass):
        """Static torque needed to hold a point mass at frame_pos: grad of -m g.p (symbolic mass ok)."""
        U = -mass * ca.dot(ca.DM(self.g), frame_pos)
        return ca.gradient(U, self.q)


def seg_dist_sx(p1, q1, p2, q2, eps=1e-9):
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e, f = ca.dot(d1, d1), ca.dot(d2, d2), ca.dot(d2, r)
    c, b = ca.dot(d1, r), ca.dot(d1, d2)
    den = a * e - b * b
    clip = lambda v: ca.fmin(ca.fmax(v, 0), 1)  # noqa
    s = ca.if_else(den > eps, clip((b * f - c * e) / (den + eps)), 0)
    t = (b * s + f) / (e + eps)
    s = ca.if_else(t < 0, clip(-c / (a + eps)), ca.if_else(t > 1, clip((b - c) / (a + eps)), s))
    t = clip(t)
    diff = (p1 + d1 * s) - (p2 + d2 * t)
    return ca.sqrt(ca.dot(diff, diff) + 1e-12)
