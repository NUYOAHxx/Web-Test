"""用户工程接入骨架：只计算，不创建 ROS 节点、不规划、不执行。

URDF 从正在运行的 robot_description 导出。
允许值配置必须人工确认，不能直接使用 inspect 输出的 null。
本文件不创建伪造 JointState 或控制目标。
"""
from copy import deepcopy
import json
import numpy as np
from ha_arm_ik.solver import HAArmIKSolver
from ha_arm_ik.solver_dls import JointStateSnapshot
from ha_arm_ik.ik_candidate import is_valid_status


def build_solver(urdf_path, confirmed_config_path, *, side="left", include_waist=True):
    with open(confirmed_config_path, encoding="utf-8") as f:
        record=json.load(f)
    cfg=dict(record["config"])
    if cfg.get("load_mode", "strict") != "strict":
        raise ValueError("此接入骨架要求严格模式，不绕过负荷门槛")
    cfg["load_require_source_age"]=True
    return HAArmIKSolver(urdf_path, side, include_waist=include_waist, config=cfg)


def capture_in_existing_callback(solver, node, msg):
    """完整消息在原回调接收时捕获一次；若输入无效，由调用方显式处理异常。"""
    snapshot=JointStateSnapshot.from_joint_state(
        msg, now_ros_ns=node.get_clock().now().nanoseconds)
    solver.update_joint_state(snapshot)
    return snapshot


def solve_only_in_single_worker(solver, target, original_snapshot, *, frame="local"):
    """original_snapshot 是原样缓存的消息快照，不在目标到来时重新盖章。

    调用方必须另行维护目标版本；MoveIt 前检查它仍是最新目标、状态未失效，
    并检验完整规划轨迹。本函数不调用任何运动执行接口。
    """
    observed=np.array([original_snapshot.positions[n] for n in solver.active_joint_names])
    q,status,result=solver.solve(
        target, frame=frame, q_prev=observed, update_prev=False,
        joint_state=original_snapshot, allow_approximate=False)
    diagnostics=deepcopy(solver.last_load_diagnostics)
    accepted=(q is not None and is_valid_status(status) and diagnostics.get("accepted", False))
    if not accepted:
        return None,status,result,diagnostics
    return q,status,result,diagnostics
