"""EXPLICIT TEST DOUBLES for unavailable ik_common/ik_candidate modules.

These files are not production replacements and must NOT be installed into the
user's package. Pinocchio, NumPy, SciPy, the merged modules and robot_model source
are real. Original candidate weights/enums/IKResult internals remain unverified.
"""
from dataclasses import dataclass
from enum import Enum
import importlib.util
from pathlib import Path
import sys
import types
import numpy as np
from scipy.spatial.transform import Rotation

INSTALLED=False
MAKE_RESULT_CALLS=[]


def install():
    global INSTALLED
    if INSTALLED:
        return
    INSTALLED=True
    common=types.ModuleType("ha_arm_ik.ik_common")
    def missing(*a,**k):
        raise NotImplementedError("unimplemented TEST DOUBLE helper; not user source")
    names="""ORIENTATION_MODES ORI_TOL POS_TOL SAFE_MARGIN TOOL_AXIS_X TOOL_CONE_HALF_ANGLE
    T_origin T_prismatic T_revolute clip_to_bounds closest_direction_on_cone invert_T
    joint_center_from_bounds joint_dist joint_limit_barrier joint_limit_barrier_jac
    joint_limit_penalty joint_margin joint_margin_from_bounds joint_range_scale
    make_T make_T_from_xyz_rpy make_translation_T normalize normalize_cone_half_angle
    pose_error pose_errors project_to_plane rot_axis rotx roty rotz rpy_to_R
    sew_angle_from_points skew target_frames tool_cone_angle_error tool_cone_residual
    tool_direction_angle_error tool_direction_from_T tool_direction_residual
    transform_point unique_seed_list validate_orientation_mode validate_q_vector
    validate_transform4x4 wrap_to_pi""".split()
    for name in names:
        setattr(common,name,missing)
    common.ORIENTATION_MODES=("full","tool_dir","tool_cone")
    common.POS_TOL=common.ORI_TOL=5e-4
    common.SAFE_MARGIN=.01
    common.TOOL_AXIS_X=np.array([1.,0.,0.])
    common.TOOL_CONE_HALF_ANGLE=np.deg2rad(20.)
    def normalize(v):
        v=np.asarray(v,dtype=float);n=np.linalg.norm(v)
        if not np.isfinite(n) or n<1e-12:
            raise ValueError("invalid direction")
        return v/n
    common.normalize=normalize
    def skew(v):
        x,y,z=v
        return np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])
    common.skew=skew
    def make_T(R=None,p=None):
        T=np.eye(4)
        if R is not None:T[:3,:3]=R
        if p is not None:T[:3,3]=p
        return T
    common.make_T=make_T
    common.rpy_to_R=lambda rpy:Rotation.from_euler("xyz",rpy).as_matrix()
    common.T_origin=lambda xyz,rpy:make_T(common.rpy_to_R(rpy),xyz)
    common.make_T_from_xyz_rpy=common.T_origin
    common.T_revolute=lambda axis,angle:make_T(Rotation.from_rotvec(normalize(axis)*angle).as_matrix())
    common.T_prismatic=lambda axis,value:make_T(p=np.asarray(axis)*value)
    common.make_translation_T=lambda xyz:make_T(p=xyz)
    common.invert_T=lambda T:np.linalg.inv(T)
    common.transform_point=lambda T,p:T[:3,:3]@p+T[:3,3]
    common.clip_to_bounds=lambda q,lo,hi:np.minimum(np.maximum(np.asarray(q,dtype=float),lo),hi)
    common.wrap_to_pi=lambda x:(np.asarray(x)+np.pi)%(2*np.pi)-np.pi
    def unique(values):
        result=[]
        for q in values:
            q=np.asarray(q,dtype=float)
            if not np.all(np.isfinite(q)):
                raise ValueError("nonfinite test seed")
            if not any(np.linalg.norm(q-r)<1e-8 for r in result):
                result.append(q.copy())
        return result
    common.unique_seed_list=unique
    def cone(value):
        value=common.TOOL_CONE_HALF_ANGLE if value is None else float(value)
        if not 0<=value<=np.pi or not np.isfinite(value):raise ValueError("cone")
        return value
    common.normalize_cone_half_angle=cone
    def orientation(value):
        if value not in common.ORIENTATION_MODES:raise ValueError("mode")
        return value
    common.validate_orientation_mode=orientation
    common.joint_margin=lambda q,lo,hi:float(np.min(np.minimum(np.asarray(q)-lo,hi-np.asarray(q))))
    common.joint_margin_from_bounds=common.joint_margin
    common.sew_angle_from_points=lambda S,E,W,ref:float(np.arctan2((np.cross(E-S,W-S)@ref),
                                                                np.linalg.norm(np.cross(E-S,W-S))+1e-12))
    def frames(T,*,frame,T_world_from_base,T_base_from_world):
        if frame=="local":return np.asarray(T),T_world_from_base@T
        if frame=="world":return T_base_from_world@T,np.asarray(T)
        raise ValueError("frame")
    common.target_frames=frames
    sys.modules[common.__name__]=common

    candidate=types.ModuleType("ha_arm_ik.ik_candidate")
    class FailureReason(Enum):
        SEEDS_EXHAUSTED="seeds_exhausted"
        NOT_CONVERGED="not_converged"
    class Status(Enum):
        VALID="VALID"
        NO_VALID_POSE="NO_VALID_POSE"
    @dataclass
    class FixtureIKResult:
        data:dict
        status:str
        failure_reason:object=None
        failure_detail:str|None=None
        per_seed:object=None
    def make_result(data,status,**kwargs):
        MAKE_RESULT_CALLS.append((data,status,kwargs))
        return FixtureIKResult(dict(data),status,**kwargs)
    def score(chain,q,target,psi,q_prev,q_ready,lower,upper,**kwargs):
        actual=chain.fk(q)
        pe=float(np.linalg.norm(actual[:3,3]-target[:3,3]))
        mode=kwargs.get("orientation_mode","full")
        if mode=="full":
            oe=float(np.linalg.norm(Rotation.from_matrix(actual[:3,:3].T@target[:3,:3]).as_rotvec()))
        else:
            axis=normalize(kwargs.get("tool_axis",common.TOOL_AXIS_X))
            a,b=actual[:3,:3]@axis,target[:3,:3]@axis
            oe=float(np.arccos(np.clip(a@b,-1.,1.)))
            if mode=="tool_cone":oe=max(0.,oe-cone(kwargs.get("tool_cone_half_angle")))
        return {"q":np.asarray(q).copy(),"pose_ok":bool(pe<=common.POS_TOL and oe<=common.ORI_TOL),
                "pos_err":pe,"ori_err":oe,"margin":common.joint_margin(q,lower,upper),
                "waist_dev":float(np.linalg.norm(np.asarray(q)[:2]-np.asarray(q_ready)[:2])),
                "base_cost":float(np.mean((np.asarray(q)-q_prev)**2)),"phi":0.}
    def sortkey(c):
        return (not c.get("pose_ok",False),c.get("base_cost",0.)+.1*c.get("torque_cost",0.),
                c.get("pos_err",np.inf))
    def select(candidates,use_torque=True):
        if not candidates:return None,"NO_VALID_POSE"
        best=min(candidates,key=sortkey)
        return best,"VALID" if best.get("pose_ok") else "NO_VALID_POSE"
    for name in """PHI_TIE_TOL W_LIMIT W_NOM W_SMOOTH W_TORQUE W_WAIST_RETURN""".split():
        setattr(candidate,name,.1)
    for name in """candidate_status failure_reason_for_status finalize_ik_result
    has_valid_torque phi_sort_value status_from_str""".split():
        setattr(candidate,name,missing)
    candidate.FailureReason=FailureReason
    candidate.Status=Status
    candidate.IKResult=FixtureIKResult
    candidate.make_result=make_result
    candidate.score_candidate=score
    candidate.candidate_sort_key=sortkey
    candidate.select_best_candidate=select
    candidate.update_candidate_final_cost_with_torque=lambda c:None
    candidate.is_valid_status=lambda s:s=="VALID"
    sys.modules[candidate.__name__]=candidate

    name="ha_arm_ik.robot_model"
    path=Path(__file__).parent/"reference/robot_model.py"
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)