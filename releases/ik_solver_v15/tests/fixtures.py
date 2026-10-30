import os
import numpy as np
from pathlib import Path
from tempfile import TemporaryDirectory
import weakref
from tests.stubs import install
install()

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


def make_solver(*,config=None,continuous=True,include_waist=True,other=True):
    if os.environ.get("HA_ARM_IK_USE_UNIFIED", "0") == "1":
        from ha_arm_ik import single_arm_ik as module
    else:
        from ha_arm_ik import solver as module
    xml,names=robot_xml(other=other,continuous=continuous)
    tmp=TemporaryDirectory()
    path=Path(tmp.name)/"robot.urdf"
    path.write_text(xml)
    qref=np.array([.05,.1,.25,-.6,.4,-.8,.3,.6,-.2])
    module.ARM_CONFIGS["left"]={
        "base_link":"link_1","ee_link":"tcp","s_joint":"LHand_Link2",
        "e_joint":"LHand_Link4","w_joint":"LHand_Link6",
        "mount_xyz":np.zeros(3),"mount_rpy":np.zeros(3),"q_ready":qref[2:],
    }
    all_names=names+(["OtherJoint"] if other else [])
    cfg={"load_allowable_efforts":{n:300. for n in all_names},
         "load_massless_links":["tcp"],"load_budget_s":3.,"load_state_max_age_s":20.,
         "load_max_seeds":4,"dls_max_iter":100,
         "q_ready":qref if include_waist else qref[2:]}
    cfg.update(config or {})
    try:
        s=module.HAArmIKSolver(path,"left",include_waist=include_waist,verbose=False,config=cfg)
    except BaseException:
        tmp.cleanup()
        raise
    s._test_tmp=tmp
    s._test_finalize=weakref.finalize(s,tmp.cleanup)
    state=dict(zip(names,qref))
    if other:state["OtherJoint"]=.4
    return s,state


def context(s,state):
    from ha_arm_ik.solver_dls import JointStateSnapshot,prepare_load_context
    ctx=prepare_load_context(s.pin_model,s.pin_maps,s._load_config,JointStateSnapshot(state))
    s.pin_maps["q_template"]=ctx["q_template"].copy()
    return ctx


def root_target(s,state,q=None):
    from ha_arm_ik.solver_dls import active_q_to_pinocchio_q
    import pinocchio as pin
    template=pin.neutral(s.pin_model)
    for name,spec in s.pin_maps["all_specs"].items():
        value=state.get(name,0.)
        if spec["continuous"]:
            template[spec["idx_q"]:spec["idx_q"]+2]=np.cos(value),np.sin(value)
        else:template[spec["idx_q"]]=value
    s.pin_maps["q_template"]=template
    q=np.array([state[n] for n in s.active_joint_names]) if q is None else np.array(q)
    pin.forwardKinematics(s.pin_model,s.pin_data,active_q_to_pinocchio_q(s.pin_model,s.pin_maps,q))
    pin.updateFramePlacements(s.pin_model,s.pin_data)
    return np.array(s.pin_data.oMf[s.pin_ee_frame_id].homogeneous,copy=True)
