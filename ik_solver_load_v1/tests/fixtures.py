from dataclasses import replace
import numpy as np
from ha_arm_ik.load_policy import JointStateSnapshot, LoadPolicy
from ha_arm_ik.solver_load_aware import LoadAwareIKSolver, SolverConfig


def inertial(mass, xyz=(0, 0, 0)):
    return (f'<inertial><origin xyz="{" ".join(map(str, xyz))}" rpy="0 0 0"/>'
            f'<mass value="{mass}"/><inertia ixx=".02" iyy=".02" izz=".02" '
            'ixy="0" ixz="0" iyz="0"/></inertial>')


def robot_xml(*, other=True, continuous=True):
    names = ["Hip_Link1", "Body_Link"] + [f"LHand_Link{i}" for i in range(1, 8)]
    axes = [(0,1,0), (0,0,1), (0,0,1), (0,1,0), (1,0,0),
            (0,1,0), (1,0,0), (0,1,0), (0,0,1)]
    offsets = [(0,0,.3), (0,0,.25), (0,.2,.18), (.1,0,0), (.25,0,0),
               (.3,0,0), (.25,0,0), (.15,0,0), (.08,0,0)]
    masses = [3, 8, 1.5, 2, 1.5, 1.2, .8, .6, .3]
    parts = ['<robot name="synthetic_redundant">',
             f'<link name="base">{inertial(12)}</link>']
    parent = "base"
    for i, name in enumerate(names):
        child = f"link_{i}"
        kind = "continuous" if continuous and name == "LHand_Link3" else "revolute"
        parts.append(f'<link name="{child}">{inertial(masses[i], (.06,0,0))}</link>')
        parts.append(f'<joint name="{name}" type="{kind}"><parent link="{parent}"/>'
                     f'<child link="{child}"/><origin xyz="{" ".join(map(str, offsets[i]))}"/>'
                     f'<axis xyz="{" ".join(map(str, axes[i]))}"/>'
                     '<limit lower="-3" upper="3" effort="300" velocity="2"/></joint>')
        parent = child
    parts += ['<link name="tcp"/>',
              f'<joint name="tcp_fixed" type="fixed"><parent link="{parent}"/>'
              '<child link="tcp"/><origin xyz=".12 0 0"/></joint>']
    if other:
        parts += [f'<link name="other_link">{inertial(3, (.2,0,0))}</link>',
                  '<joint name="OtherJoint" type="revolute"><parent link="link_1"/>'
                  '<child link="other_link"/><origin xyz="0 -.25 0"/>'
                  '<axis xyz="0 1 0"/><limit lower="-2" upper="2" effort="100" velocity="1"/></joint>']
    parts.append("</robot>")
    return "".join(parts), names


def make_solver(*, other=True, continuous=True, policy=None, config=None, **kwargs):
    xml, names = robot_xml(other=other, continuous=continuous)
    all_names = names + (["OtherJoint"] if other else [])
    policy = policy or LoadPolicy({n: 300. for n in all_names})
    config = config or SolverConfig(budget_s=3., state_max_age_s=20.,
                                    max_iterations=90, settle_iterations=8, max_seeds=4)
    qref = np.array([.05, .1, .25, -.6, .4, -.8, .3, .6, -.2])
    solver = LoadAwareIKSolver(xml, base_link="base", ee_link="tcp", policy=policy,
                              q_reference=qref, config=config, massless_links=["tcp"], **kwargs)
    state = dict(zip(names, qref))
    if other:
        state["OtherJoint"] = .4
    return solver, state


def target_for(solver, state, q=None):
    q = np.array([state[n] for n in solver.active_joint_names]) if q is None else np.array(q)
    solver._fk(solver._full_q(q, state))
    return np.array(solver.data.oMf[solver.ee_id].homogeneous, copy=True)


def planar_xml():
    parts = ['<robot name="synthetic_planar">', f'<link name="base">{inertial(10)}</link>']
    parent = "base"
    for i in range(3):
        child = f"arm{i}"
        parts += [f'<link name="{child}">{inertial(2, (.3,0,0))}</link>',
                  f'<joint name="j{i}" type="revolute"><parent link="{parent}"/>'
                  f'<child link="{child}"/><origin xyz="{0 if i == 0 else .6} 0 0"/>'
                  '<axis xyz="0 1 0"/><limit lower="-3" upper="3" effort="80" velocity="2"/></joint>']
        parent = child
    parts += ['<link name="tcp"/>',
              '<joint name="fixed" type="fixed"><parent link="arm2"/><child link="tcp"/>'
              '<origin xyz=".6 0 0"/></joint></robot>']
    return "".join(parts)