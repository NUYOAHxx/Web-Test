"""Reproducible development merge against the saved, hash-recorded source.

Not an installer. Production deliverables are the two generated modules.
"""
import ast
from pathlib import Path
import textwrap

ROOT=Path(__file__).resolve().parents[1]


def entries(text):
    lines=text.splitlines(keepends=True)
    result={}
    for node in ast.parse(text).body:
        if isinstance(node,(ast.FunctionDef,ast.ClassDef)):
            first=min([node.lineno]+[d.lineno for d in getattr(node,"decorator_list",[])])
            result[node.name]="".join(lines[first-1:node.end_lineno]).rstrip()+"\n"
    return result


def replace_node(text,name,replacement,class_name=None):
    tree=ast.parse(text)
    body=tree.body if class_name is None else next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==class_name).body
    node=next(n for n in body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name)
    lines=text.splitlines(keepends=True)
    replacement=replacement.rstrip()+"\n"
    if class_name is not None:
        replacement=textwrap.indent(replacement,"    ")
    return "".join(lines[:node.lineno-1])+replacement+"".join(lines[node.end_lineno:])


def once(text,old,new):
    if text.count(old)!=1:
        raise RuntimeError(f"expected unique merge anchor: {old[:90]!r}; got {text.count(old)}")
    return text.replace(old,new,1)


def main():
    dls=(ROOT/"baseline/solver_dls.py").read_text()
    replacements=entries((ROOT/"tools/dls_updates.pyfrag").read_text())
    old_names=set(entries(dls))
    for name in old_names&set(replacements):
        dls=replace_node(dls,name,replacements[name])
    extra="\n\n".join(replacements[n] for n in replacements if n not in old_names)
    policy=(ROOT/"tools/policy_embed.pyfrag").read_text()
    dls=once(dls,"import math\n","import math\nimport time\nimport xml.etree.ElementTree as ET\nfrom dataclasses import dataclass, field, replace\nfrom types import MappingProxyType\nfrom typing import Mapping, Sequence\n")
    anchor=dls.index("def build_pinocchio_joint_maps")
    dls=dls[:anchor]+policy+"\n\n"+dls[anchor:]+"\n\n"+extra
    dls=replace_node(dls,"make_scatter_seeds",'''def make_scatter_seeds(q_prev,q_ready,lower,upper,rng):
    """Deterministic + noisy seeds; do not create NaN midpoints for continuous joints."""
    seeds=[clip_to_bounds(q_ready,lower,upper),clip_to_bounds(q_prev,lower,upper)]
    for base in (q_prev,q_ready):
        for scale in NOISE_SCALES:
            for _ in range(max(1,N_NOISE_PER_BASE//len(NOISE_SCALES))):
                seeds.append(clip_to_bounds(base+rng.normal(scale=scale,size=len(base)),lower,upper))
    midpoint=np.asarray(q_ready,dtype=float).copy()
    finite=np.isfinite(lower)&np.isfinite(upper)
    midpoint[finite]=.5*(lower[finite]+upper[finite])
    seeds.append(midpoint)
    return unique_seed_list(seeds)
''')
    start=dls.index('"""')
    end=dls.index('"""',start+3)+3
    dls=dls[:start]+'''"""Merged v2 Pinocchio DLS, in the original ha_arm_ik.solver_dls module.

Keeps original public functions/constants. Adds confirmed-limit static load
objectives, continuous-joint/full-state mapping and consistent task/nullspace
math. Portfolio and legacy result formatting remain in solver.py.
No controller, dynamic-load guarantee or global-optimum guarantee.
"""'''+dls[end:]
    dls=dls.replace("# 相對重力矩卸載增益（抑制遠伸時腰/肩大力矩）","# default absolute normalized gravity-load gain in strict mode")
    dls=dls.replace("# 相對質心(CoM)水平防傾倒增益（拉回待命重心投影）","# reference COM-plane soft gain; NOT support/contact safety")
    (ROOT/"ha_arm_ik/solver_dls.py").write_text(dls)

    solver=(ROOT/"baseline/solver.py").read_text()
    start=solver.index('"""')
    end=solver.index('"""',start+3)+3
    solver=solver[:start]+'''"""ha_arm_ik.solver -- merged-v2, original HAArmIKSolver entry.

Preserves the original constructor, positional solve arguments and make_result
factory. Strict static-load protection is now the default and requires confirmed
joint-side capacities, an audited model and a fresh complete joint-state snapshot.
Only the keyword-only joint_state argument and explicit diagnostic/state helpers
are added. No ROS publishing, trajectory execution or collision checking occurs.

local is the configured IK base frame; world is the URDF-root frame, not an
arbitrary map/odom TF. The current inactive-joint state determines that transform
in strict mode. Optional SSIK/direct-psi source remains, but all candidate engines
share the final strict pose/bounds/static-load gate.
"""'''+solver[end:]
    new=entries((ROOT/"tools/solver_updates.pyfrag").read_text())
    solver=replace_node(solver,"collect_pinocchio_candidates_and_seeds",new["collect_pinocchio_candidates_and_seeds"])
    solver=once(solver,"import math\n","import math\nimport time\nimport threading\n")
    solver=once(solver,
        "            self.pin_model.gravity.linear = np.array(g, dtype=float)\n",
        '''            gravity_root = finite_vector(cfg_tune.get("gravity_root", g), 3, "gravity_root")
            if np.linalg.norm(gravity_root) < 1e-9:
                raise ValueError("gravity_root must be nonzero; units m/s^2 in URDF-root frame")
            self.pin_model.gravity.linear = gravity_root
''')
    solver=once(solver,"from ha_arm_ik.solver_dls import (\n",'''from ha_arm_ik.solver_dls import (
    JointStateSnapshot,
    LoadPolicy,
    finite_vector,
    validate_transform,
    joint_difference,
    json_safe,
    configure_model_metadata,
    prepare_load_context,
    load_physics,
    dls_task_terms,
''')
    solver=once(solver,"        cfg_tune = dict(config) if config else {}\n",'''        cfg_tune = dict(config) if config else {}
        self._load_config = cfg_tune.copy()
        self.load_mode = str(cfg_tune.get("load_mode", "strict"))
        if self.load_mode not in ("strict", "off"):
            raise ValueError("load_mode must be strict or explicitly off")
        self._solve_lock = threading.Lock()
        self._latest_joint_state = None
        self._load_context = None
        self._solve_T_root_base = None
        self._load_deadline = None
        self.last_load_diagnostics = {"version": "merged-v2", "accepted": False,
                                      "mode": self.load_mode, "code": "NOT_RUN"}
''')
    first=solver.index("        self.lower, self.upper = self.chain.bounds()")
    last=solver.index("        self.q_prev = self.q_ready.copy()",first)
    solver=solver[:first]+'''        self.lower, self.upper = self.chain.bounds()
        # SerialChain's continuous +/-pi values are NOT physical stops.
        bounds_override = dict(cfg_tune.get("position_bounds", {}))
        active_names = [j["name"] for j in self.chain.active_joints]
        for i, joint in enumerate(self.chain.active_joints):
            if joint["type"] == "continuous":
                self.lower[i], self.upper[i] = -np.inf, np.inf
            if joint["name"] in bounds_override:
                b = finite_vector(bounds_override[joint["name"]], 2, "position override")
                if b[0] >= b[1] or b[0] < self.lower[i] or b[1] > self.upper[i]:
                    raise ValueError("position override must narrow actual URDF bounds")
                self.lower[i], self.upper[i] = b
        raw_ready = (np.concatenate([np.zeros(2), cfg["q_ready"]])
                     if self.include_waist else np.asarray(cfg["q_ready"]))
        self.q_ready = finite_vector(cfg_tune.get("q_ready", raw_ready), len(active_names), "q_ready")
        if self.load_mode == "strict":
            if np.any(self.q_ready < self.lower) or np.any(self.q_ready > self.upper):
                raise ValueError("q_ready outside actual/planning bounds; not clipped in strict mode")
        else:
            self.q_ready = clip_to_bounds(self.q_ready, self.lower, self.upper)
'''+solver[last:]
    solver=once(solver,"            self.joint_efforts_active = get_active_joint_efforts(self.pin_maps)\n",'''            configure_model_metadata(
                self.pin_model, self.pin_maps, Path(self.urdf).read_text(encoding="utf-8"),
                massless_links=cfg_tune.get("load_massless_links", ()),
                position_bounds=cfg_tune.get("position_bounds"),
            )
            self.lower = self.pin_maps["active_lower"].copy()
            self.upper = self.pin_maps["active_upper"].copy()
            self.joint_efforts_active = get_active_joint_efforts(self.pin_maps)
''')
    solver=once(solver,'''        if self.use_pinocchio_ik:
            self.pin_ee_frame_id = get_pinocchio_frame_id(''','''        if self.pin_model is not None:
            self.pin_ee_frame_id = get_pinocchio_frame_id(''')
    solver=replace_node(solver,"evaluate_physics",new["evaluate_physics"],"HAArmIKSolver")
    solver=replace_node(solver,"target_to_local",'''def target_to_local(self,T_target,frame="local"):
    if self._solve_T_root_base is not None:
        T=validate_transform(T_target)
        if frame=="local":
            return T,self._solve_T_root_base@T
        if frame=="world":
            return np.linalg.inv(self._solve_T_root_base)@T,T
        raise ValueError("frame must be local or world (URDF-root legacy convention)")
    return target_frames(T_target,frame=frame,T_world_from_base=self.T_world_from_base,
                         T_base_from_world=self.T_base_from_world)
''',"HAArmIKSolver")
    # The original main body is retained, under a private wrapper target.
    node=next(n for n in ast.parse(solver).body if isinstance(n,ast.ClassDef) and n.name=="HAArmIKSolver")
    old=next(n for n in node.body if isinstance(n,ast.FunctionDef) and n.name=="solve")
    lines=solver.splitlines(keepends=True)
    lines[old.lineno-1]=lines[old.lineno-1].replace("def solve(","def _solve_impl(",1)
    solver="".join(lines)
    solver=once(solver,'''        else:
            q_prev = clip_to_bounds(q_prev, lower, upper)

        rng = np.random.default_rng(RNG_SEED)''','''        else:
            q_prev = finite_vector(q_prev, len(self.active_joint_names), "q_prev")
            if self.load_mode == "strict":
                if np.any(q_prev < lower) or np.any(q_prev > upper):
                    raise ValueError("q_prev outside actual/planning bounds; not clipped")
            else:
                q_prev = clip_to_bounds(q_prev, lower, upper)

        rng = np.random.default_rng(RNG_SEED)''')
    solver=once(solver,'''                "joint_limit_gain": self.dls_joint_limit_gain,
            }
''','''                "joint_limit_gain": self.dls_joint_limit_gain,
                "load_context": self._load_context,
                "deadline": self._load_deadline,
                "characteristic_length": self._load_config.get("dls_characteristic_length", .5),
            }
''')
    solver=once(solver,'''                    dls_params=dls_params,
                )''','''                    dls_params=dls_params,
                    load_search={
                        "max_seeds": self._load_config.get("load_max_seeds", 10),
                        "min_seeds": self._load_config.get("load_min_seeds", 3),
                        "min_unlocked": self._load_config.get("load_min_unlocked_seeds", 2),
                    },
                )''')
    solver=once(solver,'''        """Torque metrics -> select best -> approximate fallback -> result."""
        if self.use_torque_cost''','''        """Legacy result factory retained; strict load gate precedes ranking."""
        if self.load_mode == "strict":
            error = self._load_snapshot_error()
            if error:
                return self._load_failure("STALE_OR_CHANGED_CONTEXT", error, per_seed)
            candidates = self._filter_load_candidates(
                candidates, T_target_world, orientation_mode, tool_axis, tool_cone_half_angle)
            if not candidates:
                code = ("NO_LOAD_SAFE_SOLUTION" if self.last_load_diagnostics.get("pose_valid_count", 0)
                        else "NO_VALID_POSE_OR_PHYSICS")
                return self._load_failure(code, "No candidate passed pose/bounds/confirmed static-load gates", per_seed)
        if self.use_torque_cost''')
    solver=once(solver,'''        if best is not None and is_valid_status(status):
            if update_prev:''','''        if best is not None and is_valid_status(status):
            if self.load_mode == "strict":
                if not self._check_load_candidate(best, T_target_world, orientation_mode,
                                                   tool_axis, tool_cone_half_angle):
                    return self._load_failure("FINAL_RECHECK_FAILED", "Selected candidate failed recomputation", per_seed)
                error = self._load_snapshot_error()
                if error:
                    return self._load_failure("STALE_OR_CHANGED_CONTEXT", error, per_seed)
                self.last_load_diagnostics.update(
                    accepted=True, code="ACCEPTED", peak_ratio=best["load_peak_ratio"],
                    peak_joint=best["load_peak_joint"], ratios=best["load_guard_ratios"],
                    gravity_efforts=best["load_guard_efforts"],
                    position_error_m=best["load_final_pos_err"],
                    orientation_violation_rad=best["load_final_ori_violation"])
            else:
                self.last_load_diagnostics.update(accepted=True, code="ACCEPTED_WITH_LOAD_PROTECTION_OFF")
            if update_prev:''')
    solver=once(solver,'''            allow_approximate
            and best is not None''','''            allow_approximate
            and self.load_mode == "off"
            and best is not None''')
    solver=once(solver,'''    ratio = np.abs(tau) / joint_efforts_active''','''    if not np.all(np.isfinite(joint_efforts_active)) or np.any(joint_efforts_active <= 0):
        raise ValueError("Unknown effort limits; no 1.0 fallback or fake utilization")
    ratio = np.abs(tau) / joint_efforts_active''')
    solver=once(solver,'''                make_result(best, status, per_seed=per_seed or None),''','''                make_result({k: v for k, v in best.items() if not k.startswith("load_")},
                            status, per_seed=per_seed or None),''')
    solver=once(solver,'''            best["approximate_policy"] = (
''','''            self.last_load_diagnostics.update(
                accepted=True, code="APPROXIMATE_WITH_LOAD_PROTECTION_OFF")
            best["approximate_policy"] = (
''')
    solver=replace_node(solver,"close",'''def close(self):
    """Original close interface, protected against an in-flight solve."""
    if not self._solve_lock.acquire(blocking=False):
        raise RuntimeError("cannot close while solve/evaluate is in flight")
    try:
        self.pin_model = None
        self.pin_data = None
        self.pin_maps = None
        self.pin_ee_frame_id = None
    finally:
        self._solve_lock.release()
''',"HAArmIKSolver")
    # New methods live IN the original class; no second Solver class exists.
    for name in ("_load_failure","update_joint_state","_load_snapshot_error",
                 "_check_load_candidate","_filter_load_candidates","solve"):
        solver += "\n"+textwrap.indent(new[name],"    ")+"\n"
    ast.parse(solver)
    (ROOT/"ha_arm_ik/solver.py").write_text(solver)
    print("Merged the original two modules; legacy class and result factory retained.")


if __name__=="__main__":
    main()