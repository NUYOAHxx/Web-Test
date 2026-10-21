#!/usr/bin/env python3
"""Offline validation only. Explicitly records use of missing-module test doubles."""
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
    for name in ("test_contract_flow","test_pinocchio_numeric","test_policy_state","test_v6_constraints"):
        counts[name]=unittest.defaultTestLoader.loadTestsFromName("tests."+name).countTestCases()
    suite=unittest.defaultTestLoader.discover("tests",top_level_dir=".")
    with open("validation_"+suffix+".txt","w",encoding="utf-8") as out:
        result=unittest.TextTestRunner(stream=out,verbosity=2).run(suite)
    for path in (Path("ha_arm_ik/solver.py"),Path("ha_arm_ik/solver_dls.py")):
        ast.parse(path.read_text(),feature_version=(3,12))
    data={
        "python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__,
        "pinocchio":pin.__version__,"tests_run":result.testsRun,
        "failures":len(result.failures),"errors":len(result.errors),"skipped":len(result.skipped),
        "successful":result.wasSuccessful(),"test_counts_by_module":counts,
        "real_pinocchio_used":True,"synthetic_urdf_only":True,
        "actual_robot_model_source_used":True,
        "test_doubles":["ha_arm_ik.ik_common","ha_arm_ik.ik_candidate"],
        "user_ik_candidate_implementation_tested":False,
        "user_ik_common_implementation_tested":False,
        "user_urdf_tested":False,"user_full_project_tested":False,
        "ros2_jazzy_tested":False,"moveit_tested":False,"hardware_tested":False,
        "python_3_12_check":"syntax only; runtime used Python "+platform.python_version(),
    }
    Path("validation_"+suffix+".json").write_text(json.dumps(data,ensure_ascii=False,indent=2))
    print(json.dumps({k:data[k] for k in ("python","numpy","scipy","pinocchio","tests_run","failures","errors","successful")}))
    return 0 if result.wasSuccessful() else 1


if __name__=="__main__":
    raise SystemExit(main())