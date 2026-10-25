#!/usr/bin/env python3
"""Offline validation only. v15 = v14 + continuation drift fix (no posture settling in the q_prev continuation stage; static targets return q_prev unchanged) + bit-identical speed-ups; the user-provided ik_common / ik_candidate / robot_model are tested (no test doubles).
Dual-arm P1+P2 (dual-p2.1: a hand given None holds its pose; "free" = unconstrained): ha_arm_ik/dual_arm_ik.py (independent 16-DoF joint IK, numpy + Pinocchio only) is tested by tests/test_dual_arm_ik.py."""
import ast
import json
from pathlib import Path
import platform
import unittest
import numpy as np
import scipy
import pinocchio as pin


def main():
    suffix="pin"+pin.__version__.replace(".","_")
    counts={}
    for name in ("test_contract_flow","test_pinocchio_numeric","test_policy_state","test_v6_constraints","test_v7_speed","test_v8_changes","test_v9_changes","test_v12_changes","test_v13_changes","test_v14_cleanup","test_v15_changes","test_dual_arm_ik"):
        counts[name]=unittest.defaultTestLoader.loadTestsFromName("tests."+name).countTestCases()
    suite=unittest.defaultTestLoader.discover("tests",top_level_dir=".")
    with open("validation_"+suffix+".txt","w",encoding="utf-8") as out:
        result=unittest.TextTestRunner(stream=out,verbosity=2).run(suite)
    for path in sorted(Path("ha_arm_ik").glob("*.py")):
        ast.parse(path.read_text(),feature_version=(3,12))
    data={
        "python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__,
        "pinocchio":pin.__version__,"tests_run":result.testsRun,
        "failures":len(result.failures),"errors":len(result.errors),"skipped":len(result.skipped),
        "successful":result.wasSuccessful(),"test_counts_by_module":counts,
        "real_pinocchio_used":True,"synthetic_urdf_only":True,
        "actual_robot_model_source_used":True,
        "test_doubles":[],
        "user_ik_candidate_implementation_tested":True,
        "user_ik_common_implementation_tested":True,
        "user_robot_model_implementation_tested":True,
        "user_urdf_tested":False,"user_full_project_tested":False,
        "ros2_jazzy_tested":False,"moveit_tested":False,"hardware_tested":False,
        "python_3_12_check":"syntax only; runtime used Python "+platform.python_version(),
    }
    Path("validation_"+suffix+".json").write_text(json.dumps(data,ensure_ascii=False,indent=2))
    print(json.dumps({k:data[k] for k in ("python","numpy","scipy","pinocchio","tests_run","failures","errors","successful")}))
    return 0 if result.wasSuccessful() else 1


if __name__=="__main__":
    raise SystemExit(main())