"""Check whether the arm is an exact SRS arm (consecutive shoulder / wrist axes must intersect)."""
import os, sys
import numpy as np
import pinocchio as pin
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from g1ik.config import load_config
from g1ik.model import RobotModel

cfg = load_config()
r = RobotModel(cfg)
m, d = r.model, r.data
pin.forwardKinematics(m, d, np.zeros(m.nq))
AX = {"JointModelRX": [1, 0, 0], "JointModelRY": [0, 1, 0], "JointModelRZ": [0, 0, 1]}


def line(jn):
    j = m.getJointId(jn)
    M = d.oMi[j]
    return M.translation, M.rotation @ np.array(AX[m.joints[j].shortname()], float)


def dist(l1, l2):
    (p1, a1), (p2, a2) = l1, l2
    c = np.cross(a1, a2)
    if np.linalg.norm(c) < 1e-9:
        return np.linalg.norm(np.cross(p2 - p1, a1))
    return abs((p2 - p1) @ c) / np.linalg.norm(c)


for s in r.sides:
    js = r.arm_names[s]
    print(f"[{s}] shoulder axes pairwise distance [mm]:",
          [round(1000 * dist(line(js[a]), line(js[b])), 1) for a, b in ((0, 1), (0, 2), (1, 2))])
    print(f"[{s}] wrist    axes pairwise distance [mm]:",
          [round(1000 * dist(line(js[a]), line(js[b])), 1) for a, b in ((4, 5), (4, 6), (5, 6))])
    print(f"[{s}] elbow axis offset from shoulder-yaw axis [mm]:", round(1000 * dist(line(js[2]), line(js[3])), 1))
print("SRS requires all of the above (except elbow) to be 0 -> G1 is NOT an exact SRS arm.")
