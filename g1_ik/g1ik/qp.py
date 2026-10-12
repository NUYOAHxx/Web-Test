"""L2: placo QP differential IK (joint limits + capsule self-collision avoidance + mode masking)."""
import contextlib
import os
import sys
import time

import numpy as np
import pinocchio as pin
import placo

from .config import rpath


@contextlib.contextmanager
def _quiet():
    """placo prints a warning about neutral-pose self collisions of the URDF meshes; silence it."""
    fd = sys.stdout.fileno()
    saved = os.dup(fd)
    with open(os.devnull, "w") as dn:
        os.dup2(dn.fileno(), fd)
        try:
            yield
        finally:
            os.dup2(saved, fd)
            os.close(saved)


class QPLayer:
    def __init__(self, robot, coll, cfg):
        self.robot, self.coll, self.cfg = robot, coll, cfg
        with _quiet():
            self.pr = placo.RobotWrapper(rpath(cfg, cfg["robot"]["urdf"]), placo.Flags.ignore_collisions)
        pr = self.pr
        # collision model of placo: replace mesh geometry by our capsules (pairs added per call)
        cm = pr.collision_model
        cm.removeAllCollisionPairs()
        self.cap_gid = coll.add_capsules_to(pr.model, cm)
        self.all_joints = list(pr.joint_names())
        # locked joints at reference values
        self.locked = {}
        full = robot.full
        for jn in self.all_joints:
            if jn not in robot.active_names:
                jid = full.getJointId(jn)
                self.locked[jn] = float(robot.q_full_ref[full.joints[jid].idx_q])
        self.hand_link = {s: a["hand_link"] for s, a in cfg["robot"]["arms"].items()}
        self.ee_off = {s: pin.SE3(np.eye(3), np.array(a.get("ee_offset_xyz", [0, 0, 0]), float))
                       for s, a in cfg["robot"]["arms"].items()}
        self.elbow_link = {}
        for s, a in cfg["robot"]["arms"].items():
            self.elbow_link[s] = self._child_link_of(a["elbow_joint"])

    def _child_link_of(self, joint_name):
        m = self.robot.full
        jid = m.getJointId(joint_name)
        for f in m.frames:
            if f.type == pin.FrameType.BODY and f.parentJoint == jid:
                return f.name
        raise KeyError(joint_name)

    def set_q(self, q):
        pr = self.pr
        pr.set_T_world_fbase(np.eye(4))
        for jn, v in self.locked.items():
            pr.set_joint(jn, v)
        for jn in self.robot.active_names:
            pr.set_joint(jn, float(q[self.robot.q_index[jn]]))
        pr.update_kinematics()

    def get_q(self):
        q = np.zeros(self.robot.model.nq)
        for jn in self.robot.active_names:
            q[self.robot.q_index[jn]] = self.pr.get_joint(jn)
        return q

    def set_pairs(self, pairs):
        cm = self.pr.collision_model
        cm.removeAllCollisionPairs()
        for i, k in pairs:
            cm.addCollisionPair(pin.CollisionPair(self.cap_gid[i], self.cap_gid[k]))

    def solve(self, q0, q_ref, free_idx, targets, relative=None, pairs=(), elbow_targets=None,
              iters=None, waist_bound=None):
        """Iterate the QP from q0. targets: {side: SE3 of TCP}. Returns (q, info)."""
        r, sc, cc = self.robot, self.cfg["solver"], self.cfg["collision"]
        iters = iters or sc["qp_iters"]
        t0 = time.perf_counter()
        self.set_pairs(pairs)
        self.set_q(q0)
        S = placo.KinematicsSolver(self.pr)
        S.mask_fbase(True)
        free_names = {n for n in r.active_names if r.q_index[n] in set(free_idx)}
        for jn in self.all_joints:
            if jn not in free_names:
                S.mask_dof(jn)
        S.enable_joint_limits(True)
        S.enable_velocity_limits(False)
        tasks = {}
        for s, T in targets.items():
            T_link = (T * self.ee_off[s].inverse()).homogeneous
            t = S.add_frame_task(self.hand_link[s], T_link)
            t.configure(f"hand_{s}", "soft", 1.0, 1.0)
            tasks[s] = t
        if relative is not None:
            T_ab = (self.ee_off["left"] * relative * self.ee_off["right"].inverse()).homogeneous
            rt = S.add_relative_frame_task(self.hand_link["left"], self.hand_link["right"], T_ab)
            rt.configure("relative", "soft", 10.0, 10.0)
        wfree = [n for n in r.waist_names if n in free_names]
        if wfree:
            jt = S.add_joints_task()
            # anchor the waist at the SEED's value (not q_current): keeps the waist seeds diverse;
            # the minimal-waist-motion preference is enforced by the NLP cost and the ranking.
            jt.set_joints({n: float(q0[r.q_index[n]]) for n in wfree})
            jt.configure("waist_reg", "soft", 1e-2 * sc["w_waist"])
        el_tasks = []
        if elbow_targets:
            for s, p in elbow_targets.items():
                et = S.add_position_task(self.elbow_link[s], p)
                et.configure(f"elbow_{s}", "soft", 0.3)
                el_tasks.append(et)
        if pairs:
            c = S.add_avoid_self_collisions_constraint()
            c.self_collisions_margin = cc["qp_margin"]
            c.self_collisions_trigger = cc["qp_trigger"]
            c.configure("self_collisions", "hard", 1.0)
        reg = S.add_regularization_task(1e-4)
        lam = sc.get("qp_lm_damping", 0.2)
        n_done = 0
        for it in range(iters):
            # Levenberg-Marquardt style damping: large steps far from the target are what drives
            # the undamped iteration into joint-limit local minima.
            e2 = sum(sum(v ** 2 for v in self._err(s, targets[s])) for s in targets)
            reg.configure("regularization", "soft", 1e-4 + lam * e2)
            if el_tasks and it == iters // 2:
                for et in el_tasks:
                    S.remove_task(et)
                el_tasks = []
            try:
                S.solve(True)
            except Exception:  # QP infeasible (e.g. starting in collision): relax collision to soft
                if pairs:
                    c.configure("self_collisions", "soft", 100.0)
                    S.solve(True)
                else:
                    raise
            self.pr.update_kinematics()
            n_done += 1
            if not el_tasks and it > 5:
                err = max(self._err(s, targets[s])[0] for s in targets) if targets else 0
                if err < 1e-4:
                    break
        q = self.get_q()
        return q, dict(time=time.perf_counter() - t0, iters=n_done)

    def _err(self, s, T):
        Tl = self.pr.get_T_world_frame(self.hand_link[s])
        Tc = pin.SE3(Tl[:3, :3], Tl[:3, 3]) * self.ee_off[s]
        return np.linalg.norm(Tc.translation - T.translation), np.linalg.norm(pin.log3(T.rotation.T @ Tc.rotation))
