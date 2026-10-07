"""
Unitree G1 机械臂/全身运动轨迹 S-Curve (五次多项式 C^2 连续) 稠密插值器
=============================================================================
功能定位:
  1. 为 MoveIt 2 OMPL 规划出的离散稀疏航路点 (通常仅 2~8 个 Waypoints) 提供高频连续插值；
  2. 采用五次埃尔米特多项式 (Piecewise Quintic Hermite Spline) S-Curve 插值算法；
  3. 实现位置 C^0 连续、速度 C^1 连续、加速度 C^2 连续，完全消除跃度 (Jerk) 冲击与阶跃瞬移；
  4. 支持自适应时间缩放 (针对短距离微小位移规划的视觉平滑兜底)；
  5. 兼容 ROS 2 JointTrajectory 消息协议与纯 NumPy 离线仿真数组。
"""

import time
from typing import List, Tuple, Dict, Optional, Union
import numpy as np


class SCurveInterpolator:
    """
    五次多项式 S-Curve (Quintic Hermite Spline) 轨迹插值器。
    
    数学原理:
      对于区间 [t_k, t_{k+1}]，令 h = t_{k+1} - t_k，局域时间 tau = t - t_k in [0, h]。
      插值多项式:
        q(tau) = c0 + c1*tau + c2*tau^2 + c3*tau^3 + c4*tau^4 + c5*tau^5
      边界匹配条件:
        q(0)  = q_k,      q(h)  = q_{k+1}
        q'(0) = v_k,      q'(h) = v_{k+1}
        q''(0)= a_k,      q''(h)= a_{k+1}
      解析解系数:
        c0 = q_k
        c1 = v_k
        c2 = 0.5 * a_k
        c3 = [20*dq - (8*v_{k+1} + 12*v_k)*h - (3*a_k - a_{k+1})*h^2] / (2*h^3)
        c4 = [-30*dq + (14*v_{k+1} + 16*v_k)*h + (3*a_k - 2*a_{k+1})*h^2] / (2*h^4)
        c5 = [12*dq - 6*(v_{k+1} + v_k)*h + (a_{k+1} - a_k)*h^2] / (2*h^5)
      
      当端点速度与加速度均为 0 时 (rest-to-rest)，该式严格等价于标准五次 Smoothstep S-Curve:
        s(u) = 10*u^3 - 15*u^4 + 6*u^5,  u = tau / h
    """

    def __init__(self, sample_rate_hz: float = 50.0, min_duration: float = 0.80):
        """
        :param sample_rate_hz: 插值输出采样频率 (Hz)，默认 50Hz (20ms 周期)
        :param min_duration: 最小物理回放时长 (秒)，用于视觉展示平滑兜底 (小于此时长自动时间缩放)
        """
        self.sample_rate_hz = max(1.0, float(sample_rate_hz))
        self.dt = 1.0 / self.sample_rate_hz
        self.min_duration = float(min_duration)

    @staticmethod
    def eval_quintic_segment(
        q0: np.ndarray,
        q1: np.ndarray,
        v0: np.ndarray,
        v1: np.ndarray,
        a0: np.ndarray,
        a1: np.ndarray,
        h: float,
        tau: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        对单段区间局域时间 tau 进行五次多项式估值，返回 (position, velocity, acceleration)。
        """
        if h <= 1e-6:
            return q1.copy(), np.zeros_like(q1), np.zeros_like(q1)

        tau = np.clip(tau, 0.0, h)
        dq = q1 - q0
        h2 = h * h
        h3 = h2 * h
        h4 = h3 * h
        h5 = h4 * h

        c0 = q0
        c1 = v0
        c2 = 0.5 * a0
        c3 = (20.0 * dq - (8.0 * v1 + 12.0 * v0) * h - (3.0 * a0 - a1) * h2) / (2.0 * h3)
        c4 = (-30.0 * dq + (14.0 * v1 + 16.0 * v0) * h + (3.0 * a0 - 2.0 * a1) * h2) / (2.0 * h4)
        c5 = (12.0 * dq - 6.0 * (v1 + v0) * h + (a1 - a0) * h2) / (2.0 * h5)

        tau2 = tau * tau
        tau3 = tau2 * tau
        tau4 = tau3 * tau
        tau5 = tau4 * tau

        pos = c0 + c1 * tau + c2 * tau2 + c3 * tau3 + c4 * tau4 + c5 * tau5
        vel = c1 + 2.0 * c2 * tau + 3.0 * c3 * tau2 + 4.0 * c4 * tau3 + 5.0 * c5 * tau4
        acc = 2.0 * c2 + 6.0 * c3 * tau + 12.0 * c4 * tau2 + 20.0 * c5 * tau3
        return pos, vel, acc

    def interpolate_waypoints(
        self,
        positions: List[np.ndarray],
        timestamps: List[float],
        velocities: Optional[List[np.ndarray]] = None,
        accelerations: Optional[List[np.ndarray]] = None,
        apply_min_duration_scale: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """
        对给定的路标点序列进行全局 50Hz 稠密 S-Curve 插值。
        
        :param positions: 路标点位置列表 [np.ndarray(dim), ...]
        :param timestamps: 各路标点时间戳列表 [t0, t1, ..., tN] (秒)
        :param velocities: 可选路标点速度列表
        :param accelerations: 可选路标点加速度列表
        :param apply_min_duration_scale: 是否对超短轨迹启用视觉自适应放缓
        :return: (dense_times, dense_positions, dense_velocities, dense_accelerations, scale_factor)
        """
        n_points = len(positions)
        if n_points == 0:
            raise ValueError("路标点列表不能为空")
        if n_points == 1:
            q0 = np.asarray(positions[0], dtype=np.float64)
            v0 = np.zeros_like(q0)
            a0 = np.zeros_like(q0)
            return (
                np.array([0.0]),
                q0.reshape(1, -1),
                v0.reshape(1, -1),
                a0.reshape(1, -1),
                1.0,
            )

        q_arrs = [np.asarray(p, dtype=np.float64) for p in positions]
        dim = len(q_arrs[0])

        # 处理时间戳
        raw_dur = float(timestamps[-1] - timestamps[0])
        if raw_dur <= 1e-4:
            raw_dur = 0.5

        if apply_min_duration_scale and raw_dur < self.min_duration:
            scale_factor = self.min_duration / raw_dur
            total_dur = self.min_duration
        else:
            scale_factor = 1.0
            total_dur = raw_dur

        scaled_times = [(float(t) - float(timestamps[0])) * scale_factor for t in timestamps]
        scaled_times[0] = 0.0
        scaled_times[-1] = total_dur

        # 处理速度与加速度 (已按时间缩放比例归一化)
        if velocities is not None and len(velocities) == n_points:
            v_arrs = [np.asarray(v, dtype=np.float64) / scale_factor for v in velocities]
        else:
            v_arrs = [np.zeros(dim, dtype=np.float64) for _ in range(n_points)]

        if accelerations is not None and len(accelerations) == n_points:
            a_arrs = [np.asarray(a, dtype=np.float64) / (scale_factor ** 2) for a in accelerations]
        else:
            a_arrs = [np.zeros(dim, dtype=np.float64) for _ in range(n_points)]

        # 稠密时间网格生成
        dense_times = np.arange(0.0, total_dur + self.dt * 0.5, self.dt)
        if dense_times[-1] < total_dur:
            dense_times = np.append(dense_times, total_dur)

        n_dense = len(dense_times)
        dense_q = np.zeros((n_dense, dim), dtype=np.float64)
        dense_v = np.zeros((n_dense, dim), dtype=np.float64)
        dense_a = np.zeros((n_dense, dim), dtype=np.float64)

        seg_idx = 0
        for i, t_now in enumerate(dense_times):
            if t_now >= total_dur:
                dense_q[i] = q_arrs[-1]
                dense_v[i] = v_arrs[-1]
                dense_a[i] = a_arrs[-1]
                continue

            while seg_idx < n_points - 2 and scaled_times[seg_idx + 1] <= t_now:
                seg_idx += 1

            t0 = scaled_times[seg_idx]
            t1 = scaled_times[seg_idx + 1]
            h = max(1e-5, t1 - t0)
            tau = t_now - t0

            q, v, a = self.eval_quintic_segment(
                q_arrs[seg_idx],
                q_arrs[seg_idx + 1],
                v_arrs[seg_idx],
                v_arrs[seg_idx + 1],
                a_arrs[seg_idx],
                a_arrs[seg_idx + 1],
                h,
                tau,
            )
            dense_q[i] = q
            dense_v[i] = v
            dense_a[i] = a

        # 终点严格精准对齐
        dense_q[-1] = q_arrs[-1]
        dense_v[-1] = v_arrs[-1]
        dense_a[-1] = a_arrs[-1]

        return dense_times, dense_q, dense_v, dense_a, scale_factor


def interpolate_ros_trajectory(
    raw_trajectory,
    sample_rate_hz: float = 50.0,
    min_duration: float = 0.80,
):
    """
    为 ROS 2 trajectory_msgs.msg.JointTrajectory 消息生成高频平滑 S-Curve 稠密轨迹消息。
    """
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
    from builtin_interfaces.msg import Duration

    if raw_trajectory is None or not raw_trajectory.points:
        return raw_trajectory

    joint_names = list(raw_trajectory.joint_names)
    pts = raw_trajectory.points

    positions = [np.array(pt.positions, dtype=np.float64) for pt in pts]
    timestamps = [
        pt.time_from_start.sec + pt.time_from_start.nanosec * 1e-9 for pt in pts
    ]
    velocities = (
        [np.array(pt.velocities, dtype=np.float64) for pt in pts]
        if all(len(pt.velocities) == len(pt.positions) for pt in pts)
        else None
    )
    accelerations = (
        [np.array(pt.accelerations, dtype=np.float64) for pt in pts]
        if all(len(pt.accelerations) == len(pt.positions) for pt in pts)
        else None
    )

    interpolator = SCurveInterpolator(sample_rate_hz=sample_rate_hz, min_duration=min_duration)
    d_times, d_q, d_v, d_a, _ = interpolator.interpolate_waypoints(
        positions=positions,
        timestamps=timestamps,
        velocities=velocities,
        accelerations=accelerations,
        apply_min_duration_scale=True,
    )

    dense_msg = JointTrajectory()
    dense_msg.header = raw_trajectory.header
    dense_msg.joint_names = joint_names

    for t, q, v, a in zip(d_times, d_q, d_v, d_a):
        pt_msg = JointTrajectoryPoint()
        pt_msg.positions = [float(val) for val in q]
        pt_msg.velocities = [float(val) for val in v]
        pt_msg.accelerations = [float(val) for val in a]

        sec = int(t)
        nanosec = int((t - sec) * 1e9)
        pt_msg.time_from_start = Duration(sec=sec, nanosec=nanosec)
        dense_msg.points.append(pt_msg)

    return dense_msg
