#!/usr/bin/env python3
"""Run all tests; write versioned QA evidence, not a robot validation certificate."""
import ast
import json
from pathlib import Path
import platform
import sys
import unittest
import numpy as np
import pinocchio as pin


def main():
    version = pin.__version__
    suffix = "pin" + version.replace(".", "_")
    log = Path(f"validation_{suffix}.txt")
    suite = unittest.defaultTestLoader.discover("tests", top_level_dir=".")
    syntax = {}
    for path in (Path("ha_arm_ik/load_policy.py"), Path("ha_arm_ik/solver_load_aware.py")):
        ast.parse(path.read_text(encoding="utf-8"), feature_version=(3,12))
        syntax[str(path)] = "Python 3.12 syntax accepted; not a Python 3.12 runtime test"
    with log.open("w",encoding="utf-8") as stream:
        result = unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    summary = {
        "python": platform.python_version(), "numpy": np.__version__,
        "pinocchio": version, "platform": platform.platform(),
        "tests_run": result.testsRun, "failures": len(result.failures),
        "errors": len(result.errors), "skipped": len(result.skipped),
        "successful": result.wasSuccessful(), "syntax": syntax,
        "actual_pinocchio_library_used": True,
        "synthetic_urdf_only": True,
        "user_urdf_tested": False, "user_full_project_tested": False,
        "ros2_jazzy_tested": False, "moveit_integration_tested": False,
        "hardware_tested": False, "dynamic_load_validated": False,
    }
    Path(f"validation_{suffix}.json").write_text(
        json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({k:summary[k] for k in
                      ("python","numpy","pinocchio","tests_run","failures","errors","successful")}))
    return 0 if result.wasSuccessful() else 1


if __name__=="__main__":
    raise SystemExit(main())