"""Collision modelling for the three layers.

* capsules: fitted from meshes (PCA axis + bounding radius), stored in link-local frames,
  used by the QP layer (placo, through Coal capsules) and the NLP layer (analytic, CasADi).
* SRDF: MoveIt-style <disable_collisions> list -> pruned link-pair whitelist.
* mode relevance: a pair is only checked if some FREE joint changes the relative pose of the two links
  (symmetric difference of the joints supporting each link).
* mesh model: exact Coal (BVH mesh) check for the final validation.
"""
import json
import os
import xml.etree.ElementTree as ET

import numpy as np
import pinocchio as pin
import coal
import trimesh

from .config import rpath


# ----------------------------------------------------------------------------- capsule fitting
def _bounding_capsule(P, axis=None):
    """Capsule containing all points P (N,3), axis = principal component (or the given unit axis)."""
    c = P.mean(0)
    if axis is None:
        _, _, Vt = np.linalg.svd(P - c, full_matrices=False)
        a = Vt[0]
    else:
        a = np.asarray(axis, float) / np.linalg.norm(axis)
    t = (P - c) @ a
    perp = np.linalg.norm((P - c) - np.outer(t, a), axis=1)
    r = perp.max()
    h = np.sqrt(np.maximum(r * r - perp * perp, 0.0))
    t_hi = np.max(t - h)  # segment must reach at least here
    t_lo = np.min(t + h)
    if t_lo > t_hi:  # sphere-like
        t_lo = t_hi = 0.5 * (t_lo + t_hi)
        r = np.max(np.linalg.norm(P - (c + a * t_lo), axis=1))
    return c + a * t_lo, c + a * t_hi, float(r)


def fit_capsules(robot, cfg, force=False):
    ccfg = cfg["collision"]
    cache = rpath(cfg, ccfg["capsule_cache"])
    if os.path.exists(cache) and not force:
        with open(cache) as f:
            caps = json.load(f)
        want = set(ccfg["capsule_links"].keys()) | set((ccfg.get("manual_capsules") or {}).keys())
        man = [c for c in caps if "#m" in c["name"]]
        sig = json.dumps(ccfg["capsule_links"], sort_keys=True)
        if os.path.exists(cache + ".sig") and open(cache + ".sig").read() != sig:
            caps = []
        if caps and set(c["link"] for c in caps) == want and len(man) == sum(len(v) for v in (ccfg.get("manual_capsules") or {}).values()):
            return caps
    m = robot.model
    caps = []
    pad = float(ccfg.get("capsule_padding", 0.0))
    for link, lst in (ccfg.get("manual_capsules") or {}).items():
        for k, c in enumerate(lst):
            caps.append(dict(link=link, name=f"{link}#m{k}", p0=list(c["p0"]), p1=list(c["p1"]), r=float(c["r"])))
    for link, spec in ccfg["capsule_links"].items():
        if link in (ccfg.get("manual_capsules") or {}):
            continue
        fid = m.getFrameId(link)
        fr = m.frames[fid]
        pts = []
        for gm in (robot.vis, robot.cmesh):
            for go in gm.geometryObjects:
                if go.parentFrame != fid or not go.meshPath:
                    continue
                mesh = trimesh.load(go.meshPath, force="mesh")
                V = np.asarray(mesh.vertices) * go.meshScale
                # vertices: geometry frame -> joint frame -> link frame
                T = fr.placement.inverse() * go.placement
                pts.append(V @ T.rotation.T + T.translation)
            if pts:
                break  # prefer visual mesh
        if not pts:
            raise RuntimeError(f"no mesh for capsule link {link}")
        P = np.vstack(pts)
        n = int(spec.get("n", 1))
        # axis_pca: capsule axis = k-th principal axis of the whole link (e.g. 1 for flat parts like a
        # hand plate: slice along the long axis, each capsule spans the width -> radius ~ half thickness)
        Vt = np.linalg.svd(P - P.mean(0), full_matrices=False)[2]
        cap_axis = Vt[int(spec["axis_pca"])] if "axis_pca" in spec else None
        if n == 1:
            chunks = [P]
        else:
            c = P.mean(0)
            a = Vt[0]
            t = (P - c) @ a
            edges = np.linspace(t.min(), t.max(), n + 1)
            chunks = [P[(t >= edges[i]) & (t <= edges[i + 1])] for i in range(n)]
        for k, Q in enumerate(chunks):
            p0, p1, r = _bounding_capsule(Q, cap_axis)
            caps.append(dict(link=link, name=f"{link}#{k}", p0=p0.tolist(), p1=p1.tolist(), r=r + pad))
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    with open(cache, "w") as f:
        json.dump(caps, f, indent=1)
    with open(cache + ".sig", "w") as f:
        f.write(json.dumps(ccfg["capsule_links"], sort_keys=True))
    return caps


# ----------------------------------------------------------------------------- SRDF
def load_srdf_disabled(path):
    if not os.path.exists(path):
        return set()
    root = ET.parse(path).getroot()
    return {frozenset((d.get("link1"), d.get("link2"))) for d in root.findall("disable_collisions")}


def link_of_geom(model, go):
    name = model.frames[go.parentFrame].name
    return name


# ----------------------------------------------------------------------------- distances
def seg_seg_closest(p1, q1, p2, q2, eps=1e-12):
    """Closest points between segments [p1,q1] and [p2,q2] (Ericson, RTCD 5.1.9). numpy."""
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e, f = d1 @ d1, d2 @ d2, d2 @ r
    if a <= eps and e <= eps:
        return p1, p2
    if a <= eps:
        s, t = 0.0, np.clip(f / e, 0, 1)
    else:
        c = d1 @ r
        if e <= eps:
            t, s = 0.0, np.clip(-c / a, 0, 1)
        else:
            b = d1 @ d2
            den = a * e - b * b
            s = np.clip((b * f - c * e) / den, 0, 1) if den > eps else 0.0
            t = (b * s + f) / e
            if t < 0:
                t, s = 0.0, np.clip(-c / a, 0, 1)
            elif t > 1:
                t, s = 1.0, np.clip((b - c) / a, 0, 1)
    return p1 + d1 * s, p2 + d2 * t


class CollisionModel:
    def __init__(self, robot, cfg):
        self.robot = robot
        self.cfg = cfg
        self.cfg_coll = cfg["collision"]
        m = robot.model
        extra_off = {frozenset(p) for p in cfg["collision"].get("capsule_disabled_link_pairs", [])}
        self.caps = fit_capsules(robot, cfg)
        self.disabled = load_srdf_disabled(rpath(cfg, cfg["collision"]["srdf"]))
        # capsule endpoints in parent-joint frame of the reduced model
        self.cap_joint = []
        self.cap_pts = []
        for c in self.caps:
            fr = m.frames[m.getFrameId(c["link"])]
            p0 = fr.placement.act(np.array(c["p0"]))
            p1 = fr.placement.act(np.array(c["p1"]))
            self.cap_joint.append(fr.parentJoint)
            self.cap_pts.append((p0, p1))
        self.cap_r = np.array([c["r"] for c in self.caps])
        # supports (set of joints between root and link) for mode relevance
        self.link_support = {}
        for c in self.caps:
            j = m.frames[m.getFrameId(c["link"])].parentJoint
            self.link_support[c["link"]] = set(m.supports[j]) - {0}
        # candidate capsule pairs (not disabled, different links)
        self.cap_pairs = []
        for i in range(len(self.caps)):
            for k in range(i + 1, len(self.caps)):
                la, lb = self.caps[i]["link"], self.caps[k]["link"]
                if la == lb or frozenset((la, lb)) in self.disabled:
                    continue
                if frozenset((la, lb)) in extra_off:
                    continue
                if self.cap_joint[i] == self.cap_joint[k]:
                    continue  # rigidly attached to the same body
                self.cap_pairs.append((i, k))
        self.cap_pairs_disabled_nominal = []
        self.cap_pairs_offset = []  # (i, k, lower_bound) for pairs overlapping at nominal
        self.capsule_skipped_link_pairs = set(extra_off)
        if cfg["collision"].get("disable_capsule_pairs_overlapping_at_nominal", True):
            d = self.capsule_distances(robot.q_nominal)
            keep = []
            for (i, k), dd in zip(self.cap_pairs, d):
                if dd < 0:
                    self.cap_pairs_disabled_nominal.append((self.caps[i]["name"], self.caps[k]["name"], float(dd)))
                    self.capsule_skipped_link_pairs.add(frozenset((self.caps[i]["link"], self.caps[k]["link"])))
                    # keep them in the NLP with a relative bound: "not closer than at nominal - slack"
                    self.cap_pairs_offset.append((i, k, float(dd) - float(cfg["collision"].get("nominal_overlap_slack") or 0.0)))
                else:
                    keep.append((i, k))
            self.cap_pairs = keep
        self._setup_mesh_pairs()

    # ---------------------------------------------------------------- mode relevance
    def relevant_pairs(self, free_idx):
        """Capsule pairs whose relative pose is changed by at least one free joint."""
        m = self.robot.model
        free_j = {self.robot.joint_of_q(i) for i in free_idx}
        out = []
        for i, k in self.cap_pairs:
            sa = self.link_support[self.caps[i]["link"]]
            sb = self.link_support[self.caps[k]["link"]]
            if (sa ^ sb) & free_j:
                out.append((i, k))
        return out

    def relevant_offset_pairs(self, free_idx):
        """Nominal-overlapping pairs relevant to the free joints, with their own lower bound."""
        keep = set(self.relevant_pairs_of([(i, k) for i, k, _ in self.cap_pairs_offset], free_idx))
        return [(i, k, lb) for i, k, lb in self.cap_pairs_offset if (i, k) in keep]

    def relevant_pairs_of(self, pairs, free_idx):
        free_j = {self.robot.joint_of_q(i) for i in free_idx}
        return [(i, k) for i, k in pairs
                if (self.link_support[self.caps[i]["link"]] ^ self.link_support[self.caps[k]["link"]]) & free_j]

    # ---------------------------------------------------------------- numeric capsule distances
    def capsule_world(self, q):
        m, d = self.robot.model, self.robot.data
        pin.forwardKinematics(m, d, q)
        W = []
        for j, (p0, p1) in zip(self.cap_joint, self.cap_pts):
            M = d.oMi[j]
            W.append((M.act(p0), M.act(p1)))
        return W

    def capsule_distances(self, q, pairs=None):
        pairs = self.cap_pairs if pairs is None else pairs
        W = self.capsule_world(q)
        out = np.empty(len(pairs))
        for n, (i, k) in enumerate(pairs):
            a, b = seg_seg_closest(W[i][0], W[i][1], W[k][0], W[k][1])
            out[n] = np.linalg.norm(a - b) - self.cap_r[i] - self.cap_r[k]
        return out

    # ---------------------------------------------------------------- Coal capsules for placo
    def add_capsules_to(self, pmodel, gmodel):
        """Append Coal capsules to a pinocchio GeometryModel of ANOTHER model (e.g. placo's full model).
        Returns geometry indices aligned with self.caps."""
        ids = []
        for c in self.caps:
            fid = pmodel.getFrameId(c["link"])
            fr = pmodel.frames[fid]
            p0, p1 = np.array(c["p0"]), np.array(c["p1"])
            L = np.linalg.norm(p1 - p0)
            z = (p1 - p0) / L if L > 1e-9 else np.array([0, 0, 1.0])
            x = np.cross(z, [1, 0, 0])
            if np.linalg.norm(x) < 1e-6:
                x = np.cross(z, [0, 1, 0])
            x /= np.linalg.norm(x)
            R = np.column_stack([x, np.cross(z, x), z])
            local = pin.SE3(R, 0.5 * (p0 + p1))
            geom = coal.Capsule(c["r"], max(L, 1e-6))  # Coal capsule: radius, full length along z
            go = pin.GeometryObject("cap_" + c["name"], fr.parentJoint, fid, fr.placement * local, geom)
            ids.append(gmodel.addGeometryObject(go))
        return ids

    # ---------------------------------------------------------------- exact mesh check (validation)
    def _setup_mesh_pairs(self):
        m, gm = self.robot.model, self.robot.cmesh
        gm.removeAllCollisionPairs()
        links = [link_of_geom(m, go) for go in gm.geometryObjects]
        self.mesh_pair_links = []
        for i in range(len(links)):
            for k in range(i + 1, len(links)):
                la, lb = links[i], links[k]
                if la == lb or frozenset((la, lb)) in self.disabled:
                    continue
                if gm.geometryObjects[i].parentJoint == gm.geometryObjects[k].parentJoint:
                    continue
                gm.addCollisionPair(pin.CollisionPair(i, k))
                self.mesh_pair_links.append((la, lb))
        self.mesh_data = pin.GeometryData(gm)
        # Convex-hull twin of the mesh model, used for the clearance (distance) query:
        # BVH mesh-mesh distance costs ~0.2 s for all pairs, hull GJK distance ~0.3 ms.
        # Hull contains the mesh, so hull distance <= mesh distance (conservative clearance).
        gh = pin.GeometryModel()
        for go in gm.geometryObjects:
            geo = go.geometry
            if isinstance(geo, coal.BVHModelBase):
                tmp = geo.clone()
                tmp.buildConvexHull(True, "Qt")
                geo = tmp.convex
            gh.addGeometryObject(pin.GeometryObject(go.name, go.parentJoint, go.parentFrame, go.placement, geo))
        for cp in gm.collisionPairs:
            gh.addCollisionPair(pin.CollisionPair(cp.first, cp.second))
        self.hull_model, self.hull_data = gh, pin.GeometryData(gh)
        self.exact_refine_below = float(self.cfg_coll.get("hull_refine_below", 0.02))

    def mesh_check(self, q, exact_distance=False):
        """Exact mesh collision (boolean, Coal BVH) + clearance.
        Returns (in_collision, min_distance, closest_link_pair, colliding_link_pairs).
        min_distance is the convex-hull distance (conservative) unless exact_distance=True."""
        m, d, gm, gd = self.robot.model, self.robot.data, self.robot.cmesh, self.mesh_data
        coll = pin.computeCollisions(m, d, gm, gd, q, False)
        if exact_distance:
            pin.computeDistances(m, d, gm, gd, q)
        else:
            pin.computeDistances(m, d, self.hull_model, self.hull_data, q)
        dists = np.array([r.min_distance for r in gd.distanceResults]) if exact_distance else \
            np.array([r.min_distance for r in self.hull_data.distanceResults])
        if not exact_distance:
            # hull is conservative; refine the near pairs with the exact BVH mesh distance
            # (placements of gd were already updated by computeCollisions)
            for i in np.nonzero(dists < self.exact_refine_below)[0]:
                dists[i] = pin.computeDistance(gm, gd, int(i)).min_distance
        k = int(np.argmin(dists))
        hits = [self.mesh_pair_links[i] for i, r in enumerate(self.mesh_data.collisionResults) if r.isCollision()]
        if coll:
            dists[k] = min(dists[k], 0.0)
        return bool(coll), float(dists[k]), self.mesh_pair_links[k], hits


# ----------------------------------------------------------------------------- 2-joint distance tables
class JointDistanceTables:
    """Exact mesh distance tabulated in joint space for link pairs whose relative pose depends on at most
    TWO active joints (e.g. torso <-> shoulder_roll_link depends only on shoulder pitch + roll; head <->
    shoulder_roll_link likewise). These are exactly the pairs where capsules are hopeless (the shoulder
    socket). A smooth B-spline of the tabulated (signed) distance is used as an NLP constraint.

    Inside the collision region the BVH distance is not informative, so the table stores
    -(joint-space distance to the free region) * inside_scale there (Euclidean distance transform)."""

    def __init__(self, robot, coll, cfg):
        self.robot, self.coll, self.cfg = robot, coll, cfg
        jc = cfg["collision"].get("joint_tables") or {}
        self.enabled = bool(jc.get("enabled", False))
        self.tables = []
        if not self.enabled:
            return
        res = float(jc.get("resolution", 0.04))
        scale = float(jc.get("inside_scale", 0.05))
        cache = rpath(cfg, jc.get("cache", "generated/joint_tables.npz"))
        m = robot.model
        links = set(jc.get("links_pairs_from", []))  # optional filter
        want = []
        for n, (la, lb) in enumerate(coll.mesh_pair_links):
            if frozenset((la, lb)) not in coll.capsule_skipped_link_pairs:
                continue
            ja = m.frames[m.getFrameId(la)].parentJoint
            jb = m.frames[m.getFrameId(lb)].parentJoint
            js = sorted((set(m.supports[ja]) ^ set(m.supports[jb])) - {0})
            if 1 <= len(js) <= 2 and (not links or la in links or lb in links):
                want.append((n, la, lb, [m.joints[j].idx_q for j in js]))
        sig = json.dumps([(w[1], w[2], w[3]) for w in want] + [res, scale])
        data = None
        if os.path.exists(cache):
            z = np.load(cache, allow_pickle=True)
            if str(z["sig"]) == sig:
                data = list(z["tables"])
        if data is None:
            data = [self._tabulate(n, qi, res, scale) for n, _, _, qi in want]
            os.makedirs(os.path.dirname(cache), exist_ok=True)
            np.savez(cache, sig=sig, tables=np.array(data, dtype=object))
        for (n, la, lb, qi), tab in zip(want, data):
            self.tables.append(dict(pair=(la, lb), q_idx=qi, **tab))

    def _tabulate(self, n, qi, res, scale):
        from scipy.ndimage import distance_transform_edt
        r, gm, gd = self.robot, self.robot.cmesh, self.coll.mesh_data
        axes = [np.arange(r.lower[i], r.upper[i] + res, res) for i in qi]
        q = r.q_nominal.copy()
        grid = np.meshgrid(*axes, indexing="ij")
        D = np.zeros(grid[0].shape)
        C = np.zeros(grid[0].shape, bool)
        for idx in np.ndindex(D.shape):
            for k, i in enumerate(qi):
                q[i] = grid[k][idx]
            pin.updateGeometryPlacements(r.model, r.data, gm, gd, q)
            cr = pin.computeCollision(gm, gd, n)
            C[idx] = bool(cr)
            D[idx] = pin.computeDistance(gm, gd, n).min_distance
        if C.any():
            inside = distance_transform_edt(C) * res  # rad to the free region
            D = np.where(C, -inside * scale, np.maximum(D, 0.0))
        return dict(axes=axes, D=D)

    def casadi_constraints(self, q_sym, free_idx):
        """list of (expr, table) for tables touched by the free joints"""
        import casadi as ca
        out = []
        free = set(free_idx)
        for t in self.tables:
            if not (set(t["q_idx"]) & free):
                continue
            f = ca.interpolant(f"jt_{t['pair'][0]}_{t['pair'][1]}", "bspline", [list(a) for a in t["axes"]],
                               t["D"].ravel(order="F").tolist())
            out.append((f(ca.vertcat(*[q_sym[i] for i in t["q_idx"]])), t))
        return out

    def values(self, q):
        from scipy.interpolate import RegularGridInterpolator
        out = []
        for t in self.tables:
            f = RegularGridInterpolator(t["axes"], t["D"], bounds_error=False, fill_value=None)
            out.append(float(f([q[i] for i in t["q_idx"]])[0]))
        return np.array(out)
