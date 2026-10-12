"""Render PNG figures (matplotlib, offscreen) of a few solutions + benchmark summary.
usage: python tests/make_figures.py [--bench results/benchmark.json]"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pinocchio as pin  # noqa: E402
import trimesh  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
from g1ik.config import load_config  # noqa: E402
from g1ik.solver import IKSolver, IKRequest  # noqa: E402

FIG = os.path.join(ROOT, "figures")
_mesh_cache = {}


def _mesh(path, max_faces=6000):
    if path not in _mesh_cache:
        m = trimesh.load(path, force="mesh")
        F = m.faces
        if len(F) > max_faces:
            F = F[np.linspace(0, len(F) - 1, max_faces).astype(int)]
        _mesh_cache[path] = (np.asarray(m.vertices), F)
    return _mesh_cache[path]


def draw_robot(ax, S, q, color="0.75", alpha=0.35, skip_legs=False):
    r = S.robot
    gm = r.vis
    gd = pin.GeometryData(gm)
    pin.updateGeometryPlacements(r.model, r.data, gm, gd, q)
    for k, go in enumerate(gm.geometryObjects):
        if not go.meshPath:
            continue
        name = r.model.frames[go.parentFrame].name
        if skip_legs and any(t in name for t in ["hip", "knee", "ankle"]):
            continue
        V, F = _mesh(go.meshPath)
        M = gd.oMg[k]
        W = (V * go.meshScale) @ M.rotation.T + M.translation
        c = color
        if "left" in name and any(t in name for t in ["shoulder", "elbow", "wrist", "hand"]):
            c = "tab:blue"
        elif "right" in name and any(t in name for t in ["shoulder", "elbow", "wrist", "hand"]):
            c = "tab:orange"
        elif "torso" in name or "waist" in name:
            c = "tab:green"
        pc = Poly3DCollection(W[F], facecolor=c, edgecolor="none", alpha=alpha)
        ax.add_collection3d(pc)


def draw_capsules(ax, S, q, pairs_only=None, color="tab:red"):
    W = S.coll.capsule_world(q)
    u, v = np.mgrid[0:2 * np.pi:12j, 0:np.pi:7j]
    for i, (a, b) in enumerate(W):
        rr = S.coll.cap_r[i]
        for c in (a, b):
            ax.plot_wireframe(c[0] + rr * np.cos(u) * np.sin(v), c[1] + rr * np.sin(u) * np.sin(v),
                              c[2] + rr * np.cos(v), color=color, linewidth=0.3, alpha=0.4)
        ax.plot(*np.c_[a, b], color=color, linewidth=1.0)


def draw_frame(ax, T, L=0.08, lw=2):
    for k, c in enumerate("rgb"):
        e = T.translation + L * T.rotation[:, k]
        ax.plot(*np.c_[T.translation, e], color=c, linewidth=lw)


def draw_box(ax, Tb, size=(0.20, 0.30, 0.15)):
    sx, sy, sz = np.array(size) / 2
    C = np.array([[x, y, z] for x in (-sx, sx) for y in (-sy, sy) for z in (-sz, sz)])
    W = C @ Tb.rotation.T + Tb.translation
    faces = [[0, 1, 3, 2], [4, 5, 7, 6], [0, 1, 5, 4], [2, 3, 7, 6], [0, 2, 6, 4], [1, 3, 7, 5]]
    ax.add_collection3d(Poly3DCollection([W[f] for f in faces], facecolor="gold", edgecolor="k", alpha=0.25))


def setup(ax, title, center=(0.15, 0, 0.1), half=0.45, elev=18, azim=-60):
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("x"), ax.set_ylabel("y"), ax.set_zlabel("z")
    ax.set_title(title, fontsize=9)


def sol_text(s):
    return (f"{s['mode']} valid={s['valid']} pos={s['pos_err'] * 1e3:.2f}mm rot={np.degrees(s['rot_err']):.2f}deg\n"
            f"waist=({s['q'][0]:+.2f},{s['q'][1]:+.2f})rad util={s['max_util']:.2f} dmesh={s['min_mesh_dist'] * 1e3:.1f}mm")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default=os.path.join(ROOT, "results", "benchmark.json"))
    a = ap.parse_args()
    os.makedirs(FIG, exist_ok=True)
    cfg = load_config()
    S = IKSolver(cfg)
    r = S.robot
    qn = r.q_nominal

    # --- 1 capsule model at nominal
    fig = plt.figure(figsize=(10, 5))
    for k, (el, az) in enumerate([(15, -60), (15, 20)]):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        draw_robot(ax, S, qn, alpha=0.25)
        draw_capsules(ax, S, qn)
        setup(ax, f"capsule model ({len(S.coll.caps)} capsules) at nominal", elev=el, azim=az, center=(0.05, 0, 0.0),
              half=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "capsules_nominal.png"), dpi=130)
    plt.close(fig)

    # --- 2 waist-needed target: M1 fails, auto escalates to M2
    # target taken from a benchmark case that M1 could not solve (auto escalated to M2)
    side = "left"
    q_gen = qn.copy()
    q_gen[r.waist_idx] = [0.7, 0.35]
    q_gen[r.arm_idx["left"]] = [-1.0, 0.2, 0.0, 0.3, 0.0, 0.0, 0.0]
    if os.path.exists(a.bench):
        cand = [x for x in json.load(open(a.bench))["rows"] if x["scenario"] == "auto_waist_needed"
                and x["success"] and x["final_mode"] == "M2"]
        if cand:
            x = sorted(cand, key=lambda x: x["waist_disp"])[len(cand) // 2]
            q_gen = np.array(x["q"])
            side = "left" if np.abs(q_gen[r.arm_idx["left"]] - qn[r.arm_idx["left"]]).max() > 1e-6 else "right"
    T = r.fk(q_gen)[side]
    r1 = S.solve(IKRequest(targets={side: T}, mode="M1"))
    r2 = S.solve(IKRequest(targets={side: T}, mode="auto"))
    fig = plt.figure(figsize=(11, 5.2))
    for k, (res, lab) in enumerate([(r1, "M1 (waist locked): best attempt"), (r2, "auto -> " + (r2["solutions"][0]["mode"] if r2["solutions"] else "-"))]):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        s = res["solutions"][0]
        draw_robot(ax, S, s["q"], skip_legs=True)
        draw_frame(ax, T, L=0.1, lw=3)
        setup(ax, lab + "\n" + sol_text(s) + f"\nsolve {res['time']:.2f}s", center=(0.2, 0.0, 0.1), half=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "waist_needed_m1_vs_auto.png"), dpi=130)
    plt.close(fig)
    print("waist-needed:", [(e["mode"], e["n_valid"], e.get("escalate_reason")) for e in r2["log"]])

    # --- 3 box with both hands + payload (Top-K)
    w = np.array([0.4, 0.2])
    qw = qn.copy()
    qw[r.waist_idx] = w
    dd = r.model.createData()
    pin.framesForwardKinematics(r.model, dd, qw)
    Tb = dd.oMf[r.model.getFrameId("torso_link")] * pin.SE3(np.eye(3), np.array([0.36, 0.0, 0.08]))
    TL = pin.SE3(Tb.rotation, Tb.act(np.array([0, 0.15, 0])))
    TR = pin.SE3(Tb.rotation, Tb.act(np.array([0, -0.15, 0])))
    rb = S.solve(IKRequest(targets={"left": TL, "right": TR}, mode="auto", relative=TL.inverse() * TR, object_mass=3.0))
    sols = rb["solutions"]
    fig = plt.figure(figsize=(5.5 * max(len(sols), 1), 5.4))
    for k, s in enumerate(sols):
        ax = fig.add_subplot(1, len(sols), k + 1, projection="3d")
        draw_robot(ax, S, s["q"], skip_legs=True)
        draw_box(ax, Tb)
        draw_frame(ax, TL), draw_frame(ax, TR)
        setup(ax, f"box 3 kg, Top-{k + 1}\n" + sol_text(s), center=(0.2, 0.05, 0.1), half=0.42, azim=-45)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "box_both_hands_topk.png"), dpi=120)
    plt.close(fig)
    print("box:", [(e["mode"], e["n_valid"], e.get("escalate_reason")) for e in rb["log"]], f"{rb['time']:.2f}s")

    # --- 4 near arm-arm: hands side by side, with capsules of the hands
    c = np.array([0.33, 0.0, 0.08])
    TL = pin.SE3(np.eye(3), c + [0, 0.06, 0])
    TR = pin.SE3(np.eye(3), c - [0, 0.06, 0])
    rn = S.solve(IKRequest(targets={"left": TL, "right": TR}, mode="M3"))
    s = rn["solutions"][0]
    fig = plt.figure(figsize=(11, 5.2))
    for k, (el, az) in enumerate([(25, -60), (70, -90)]):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        draw_robot(ax, S, s["q"], skip_legs=True, alpha=0.3)
        draw_capsules(ax, S, s["q"])
        draw_frame(ax, TL), draw_frame(ax, TR)
        setup(ax, "near arm-arm (TCPs 12 cm apart), M3\n" + sol_text(s) + f"\nclosest mesh pair: {s['min_mesh_pair']}",
              center=(0.25, 0.0, 0.05), half=0.3, elev=el, azim=az)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "near_arm_arm.png"), dpi=130)
    plt.close(fig)

    # --- 5 benchmark summary
    if os.path.exists(a.bench):
        B = json.load(open(a.bench))
        rows = B["rows"]
        names = []
        for row in rows:
            if row["scenario"] not in names:
                names.append(row["scenario"])
        succ = [np.mean([x["success"] for x in rows if x["scenario"] == n]) for n in names]
        times = [[x["time"] for x in rows if x["scenario"] == n] for n in names]
        fig, axs = plt.subplots(1, 2, figsize=(14, 5.5))
        axs[0].barh(names, succ, color="tab:blue")
        axs[0].set_xlim(0, 1)
        axs[0].set_xlabel("success rate")
        for i, v in enumerate(succ):
            axs[0].text(v + 0.01, i, f"{v:.0%} (n={len(times[i])})", va="center", fontsize=8)
        axs[0].invert_yaxis()
        axs[1].boxplot(times, orientation="horizontal", tick_labels=names)
        axs[1].set_xlabel("wall-clock solve time per request [s] (this machine)")
        axs[1].invert_yaxis()
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "benchmark_summary.png"), dpi=120)
        plt.close(fig)
        # torque utilisation / waist displacement scatter of successful solutions
        fig, ax = plt.subplots(figsize=(7, 5))
        for n in names:
            xs = [x for x in rows if x["scenario"] == n and x["success"]]
            if xs:
                ax.scatter([x["waist_disp"] for x in xs], [x["max_util"] for x in xs], label=n, s=18)
        ax.axhline(cfg["solver"]["auto_torque_util_threshold"], color="k", ls="--", lw=0.8)
        ax.set_xlabel("waist displacement ||dq_waist|| [rad]")
        ax.set_ylabel("max torque utilisation |tau|/tau_max")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "util_vs_waist.png"), dpi=120)
        plt.close(fig)
    print("figures written to", os.path.abspath(FIG))


if __name__ == "__main__":
    main()
