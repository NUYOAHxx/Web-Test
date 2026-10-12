"""Minimal usage example of the G1 IK prototype.
run from the repo root:  PYTHONPATH=. python examples/solve_example.py"""
import numpy as np
import pinocchio as pin

from g1ik.config import load_config
from g1ik.solver import IKSolver, IKRequest

cfg = load_config("config/g1.yaml")
S = IKSolver(cfg)                      # builds models, capsules, placo QP; NLPs are built lazily + cached
r = S.robot
print("active joints:", r.active_names)

# 1) single hand, auto mode (M1 -> M2 escalation), 1.0 kg payload in the left hand
T = pin.SE3(np.eye(3), np.array([0.40, 0.20, 0.10]))   # TCP pose in the pelvis frame
res = S.solve(IKRequest(targets={"left": T}, mode="auto", payload={"left": 1.0}))
for e in res["log"]:
    print("stage", e["mode"], "valid:", e["n_valid"], "reason:", e.get("escalate_reason"),
          {k: round(v, 3) for k, v in e["times"].items()})
for k, s in enumerate(res["solutions"]):
    print(f"top{k + 1}: valid={s['valid']} mode={s['mode']} pos_err={s['pos_err']:.1e} m "
          f"waist={np.round(s['q'][r.waist_idx], 3)} max_util={s['max_util']:.2f} ({s['util_joint']}) "
          f"min_mesh_dist={s['min_mesh_dist']:.3f} m")

# 2) both hands hold a 30 cm box with a 3 kg object: separate targets + relative pose constraint
Tb = pin.SE3(np.eye(3), np.array([0.36, 0.0, 0.10]))
TL = pin.SE3(Tb.rotation, Tb.act(np.array([0, 0.15, 0])))
TR = pin.SE3(Tb.rotation, Tb.act(np.array([0, -0.15, 0])))
res = S.solve(IKRequest(targets={"left": TL, "right": TR}, mode="auto",
                        relative=TL.inverse() * TR, object_mass=3.0))
s = res["solutions"][0]
print("box:", res["success"], s["mode"], f"{res['time']:.2f}s", "q =", np.round(s["q"], 3))
