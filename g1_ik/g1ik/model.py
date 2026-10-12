"""Unified robot model: full URDF -> reduced model with only waist + arm joints active.

All three IK modes use THIS single model; modes only differ by which of its joints are free.
"""
import numpy as np
import pinocchio as pin
import coal

from .config import rpath


class RobotModel:
    def __init__(self, cfg):
        self.cfg = cfg
        rc = cfg["robot"]
        urdf = rpath(cfg, rc["urdf"])
        mesh_dir = rpath(cfg, rc["mesh_dir"])
        full, cgeom, vgeom = pin.buildModelsFromUrdf(urdf, mesh_dir)
        full.gravity.linear = np.array(rc["gravity"], dtype=float)
        self.full = full

        self.waist_names = list(rc["waist_joints"])
        self.arm_names = {s: list(a["joints"]) for s, a in rc["arms"].items()}
        self.sides = list(rc["arms"].keys())
        active = self.waist_names + [j for s in self.sides for j in self.arm_names[s]]
        self.active_names = active

        # reference configuration for locked joints
        q_ref = pin.neutral(full)
        for jn, v in rc.get("locked_joint_values", {}).items():
            jid = full.getJointId(jn)
            q_ref[full.joints[jid].idx_q] = v
        self.q_full_ref = q_ref
        lock_ids = [jid for jid in range(1, full.njoints) if full.names[jid] not in active]

        # extra collision geometry from visual meshes (links without <collision>)
        for ln in cfg["collision"].get("collision_from_visual", []):
            for go in vgeom.geometryObjects:
                if full.frames[go.parentFrame].name == ln:
                    g2 = pin.GeometryObject(go)
                    g2.name = go.name + "_fromvis"
                    cgeom.addGeometryObject(g2)

        self.model, (self.cmesh, self.vis) = pin.buildReducedModel(full, [cgeom, vgeom], lock_ids, q_ref)
        self.model.gravity.linear = np.array(rc["gravity"], dtype=float)
        m = self.model
        assert m.nq == m.nv == len(active), "only 1-DOF revolute active joints supported"
        # order of q in reduced model
        self.q_index = {m.names[j]: m.joints[j].idx_q for j in range(1, m.njoints)}
        self.waist_idx = np.array([self.q_index[n] for n in self.waist_names])
        self.arm_idx = {s: np.array([self.q_index[n] for n in self.arm_names[s]]) for s in self.sides}
        self.lower = m.lowerPositionLimit.copy()
        self.upper = m.upperPositionLimit.copy()
        tau = m.effortLimit.copy()
        for jn, v in rc.get("torque_limit_override", {}).items():
            tau[self.q_index[jn]] = v
        self.tau_max = tau
        self.torque_margin = float(rc.get("torque_margin", 1.0))

        # end-effector (TCP) frames
        self.ee_frame = {}
        for s, a in rc["arms"].items():
            fid = m.getFrameId(a["hand_link"])
            f = m.frames[fid]
            plc = f.placement * pin.SE3(np.eye(3), np.array(a.get("ee_offset_xyz", [0, 0, 0]), dtype=float))
            name = f"{s}_tcp"
            self.ee_frame[s] = m.addFrame(pin.Frame(name, f.parentJoint, fid, plc, pin.FrameType.OP_FRAME))
        self.data = m.createData()

        # nominal configuration
        self.q_nominal = np.zeros(m.nq)
        for jn, v in rc.get("nominal_q", {}).items():
            self.q_nominal[self.q_index[jn]] = v
        self.q_nominal = np.clip(self.q_nominal, self.lower, self.upper)

        # mesh collision pairs are set later from SRDF (collision.py)
        self.cmesh_data = None

    # ------------------------------------------------------------------ helpers
    def joint_of_q(self, i):
        for j in range(1, self.model.njoints):
            if self.model.joints[j].idx_q == i:
                return j
        raise KeyError(i)

    def mode_free_idx(self, mode, sides):
        """Indices of free joints. mode: 'M1' (arms only), 'M2'/'M3' (waist + arms)."""
        idx = [] if mode == "M1" else list(self.waist_idx)
        for s in sides:
            idx += list(self.arm_idx[s])
        return np.array(sorted(idx))

    def fk(self, q):
        pin.framesForwardKinematics(self.model, self.data, q)
        return {s: self.data.oMf[fid].copy() for s, fid in self.ee_frame.items()}

    def static_torque(self, q, payload):
        """tau = g(q) - sum_s J_s^T f_s, f_s = payload_s * gravity (force applied ON the robot)."""
        m, d = self.model, self.data
        tau = pin.computeGeneralizedGravity(m, d, q).copy()
        if payload:
            pin.computeJointJacobians(m, d, q)
            pin.updateFramePlacements(m, d)
            for s, mass in payload.items():
                if mass <= 0:
                    continue
                J = pin.getFrameJacobian(m, d, self.ee_frame[s], pin.LOCAL_WORLD_ALIGNED)[:3]
                f = mass * m.gravity.linear
                tau -= J.T @ f
        return tau

    def torque_util(self, tau):
        return np.abs(tau) / self.tau_max

    def manipulability(self, q, side):
        m, d = self.model, self.data
        pin.computeJointJacobians(m, d, q)
        pin.updateFramePlacements(m, d)
        J = pin.getFrameJacobian(m, d, self.ee_frame[side], pin.LOCAL_WORLD_ALIGNED)[:, self.arm_idx[side]]
        return float(np.sqrt(max(np.linalg.det(J @ J.T), 0.0)))
