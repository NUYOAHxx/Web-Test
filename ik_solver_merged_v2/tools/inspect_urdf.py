#!/usr/bin/env python3
"""Generate UNFILLED load configuration from exported runtime URDF; no ROS/IK."""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("urdf",help="URDF XML exported from the running robot_description")
    args=p.parse_args()
    root=ET.fromstring(Path(args.urdf).expanduser().read_text(encoding="utf-8"))
    names=[]
    reference={}
    unsupported=[]
    for joint in root.findall("joint"):
        name,kind=joint.get("name"),joint.get("type")
        if kind=="fixed":continue
        names.append(name)
        e=joint.find("limit")
        reference[name]={"type":kind,"effort_units":"N" if kind=="prismatic" else "N*m",
                          "URDF_effort_REFERENCE_ONLY":None if e is None else e.get("effort"),
                          "lower":None if e is None or kind=="continuous" else e.get("lower"),
                          "upper":None if e is None or kind=="continuous" else e.get("upper")}
        if kind not in ("continuous","revolute","prismatic") or joint.find("mimic") is not None:
            unsupported.append(name)
    missing_mass=[]
    for link in root.findall("link"):
        e=link.find("inertial/mass")
        if e is None or float(e.get("value","0"))<=0:missing_mass.append(link.get("name"))
    output={
        "config":{
            "load_mode":"strict",
            "load_allowable_efforts":{n:None for n in names},
            "load_massless_links":[],
            "load_require_source_age":True,
            "load_state_max_age_s":1.0,
            "load_budget_s":0.5,
            "load_max_seeds":10,
        },
        "required_full_state_joints":names,
        "massless_CANDIDATES_MUST_CONFIRM":missing_mass,
        "unsupported_joint_names":unsupported,
        "urdf_reference_only":reference,
        "warning":"null limits are intentionally NOT runnable; do not copy test-fixture limits or auto-whitelist real structures",
    }
    print(json.dumps(output,ensure_ascii=False,indent=2,allow_nan=False))
    return 0


if __name__=="__main__":
    raise SystemExit(main())