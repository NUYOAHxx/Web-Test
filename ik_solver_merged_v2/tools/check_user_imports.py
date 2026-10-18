#!/usr/bin/env python3
"""Run in the actual user project environment. No test stubs, robot or ROS calls."""
import inspect
import json
from ha_arm_ik import ik_candidate,ik_common,solver,solver_dls


def main():
    result=ik_candidate.make_result(
        {},"NO_VALID_POSE",failure_reason=ik_candidate.FailureReason.NOT_CONVERGED,
        failure_detail="[IMPORT_CHECK] pure formatter contract check")
    if ik_candidate.is_valid_status("NO_VALID_POSE"):
        raise RuntimeError("Unexpected legacy semantics: failure status considered valid")
    data={
        "solver_path":solver.__file__,"solver_dls_path":solver_dls.__file__,
        "ik_candidate_path":ik_candidate.__file__,"ik_common_path":ik_common.__file__,
        "constructor":str(inspect.signature(solver.HAArmIKSolver)),
        "solve":str(inspect.signature(solver.HAArmIKSolver.solve)),
        "result_type":type(result).__module__+"."+type(result).__name__,
        "new_state_api_available":hasattr(solver_dls,"JointStateSnapshot"),
        "contract_scope":"imports, signatures and legacy failure formatter only; not success scoring or robot-model validation",
        "no_ik_or_execution_performed":True,
    }
    print(json.dumps(data,ensure_ascii=False,indent=2))
    return 0


if __name__=="__main__":
    raise SystemExit(main())