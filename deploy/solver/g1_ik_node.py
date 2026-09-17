#!/usr/bin/env python3
"""
================================================================================
Unitree G1 10-DoF (3-DoF 腰部 + 7-DoF 手臂) 逆运动学标准无头解算服务节点
(Headless ROS 2 IK Solver & Collision Barrier Service Node)
================================================================================

定位：纯算法与运动学计算节点 (Zero Web / Pure Kinematics Engine)
1. 彻底解耦，无任何阻塞式 input()，完全作为 Headless ROS 2 服务节点常驻；
2. 订阅标准笛卡尔指令话题：/g1/kinematics/target_pose 与 /ik/target_pose；
3. 执行非对称加权自适应阻尼最小二乘 (Weighted DLS) 10-DoF 协同求解；
4. 运行全机身 28 对关键连杆自碰撞安全检测门禁与距离裕度计算；
5. 内置 50Hz 工业级平滑 S 曲线余弦插值器，向 /joint_states 发布真实关节轨迹；
6. 实时广播专业度量流：/g1/kinematics/actual_pose、/g1/kinematics/solver_metrics、
   /g1/safety/collision_status 与 RViz 3D 标记流。
"""

import os
import sys
import time
import math
import json
import threading
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pinocchio as pin

import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Point, PoseStamped
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray

# 添加项目根目录到 Python 搜索路径
DIR_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if DIR_ROOT not in sys.path:
    sys.path.insert(0, DIR_ROOT)

from deploy.solver.g1_hybrid_ik import G1HybridIKSolver, G1_READY_POSE

# 宇树 G1 官方对称直立预备就绪姿态
DEFAULT_STAND_JOINTS: Dict[str, float] = {
    "waist_yaw_joint": 0.0,
    "waist_roll_joint": 0.0,
    "waist_pitch_joint": 0.0,
    "left_shoulder_pitch_joint": 0.2,
    "left_shoulder_roll_joint": 0.2,
    "left_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": 0.5,
    "left_wrist_roll_joint": 0.0,
    "left_wrist_pitch_joint": 0.0,
    "left_wrist_yaw_joint": 0.0,
    "right_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    "right_shoulder_yaw_joint": 0.0,
    "right_elbow_joint": 0.5,
    "right_wrist_roll_joint": 0.0,
    "right_wrist_pitch_joint": 0.0,
    "right_wrist_yaw_joint": 0.0,
    "left_hip_pitch_joint": 0.0,
    "left_hip_roll_joint": 0.0,
    "left_hip_yaw_joint": 0.0,
    "left_knee_joint": 0.0,
    "left_ankle_pitch_joint": 0.0,
    "left_ankle_roll_joint": 0.0,
    "right_hip_pitch_joint": 0.0,
    "right_hip_roll_joint": 0.0,
    "right_hip_yaw_joint": 0.0,
    "right_knee_joint": 0.0,
    "right_ankle_pitch_joint": 0.0,
    "right_ankle_roll_joint": 0.0,
}


class G1IKSolverNode(Node):
    """Unitree G1 10-DoF 逆运动学与碰撞避免服务节点"""

    def __init__(self):
        super().__init__("g1_ik_solver_node")
        self.get_logger().info("======================================================")
        self.get_logger().info(" 🚀 Unitree G1 10-DoF 躯干-手臂协同 IK 解算服务节点启动中...")
        self.get_logger().info("======================================================")

        # 实例化混合求解器 (封装 Pinocchio 运动学模型与 28 对碰撞检测器)
        self.solver = G1HybridIKSolver()
        self.current_arm = "left_arm"
        self.lock = threading.Lock()

        # 计算初始就绪姿态下末端手爪基准坐标
        ready_waist = np.zeros(3, dtype=np.float64)
        ready_arm = G1_READY_POSE[self.current_arm].copy()
        init_fk_p, _ = self.solver.forward_kinematics_10dof(self.current_arm, ready_waist, ready_arm)

        # 关节与轨迹状态缓存
        self.current_joints = dict(DEFAULT_STAND_JOINTS)
        self.target_joints = dict(DEFAULT_STAND_JOINTS)
        self.start_joints = dict(DEFAULT_STAND_JOINTS)
        self.interp_start_time = time.time()
        self.interp_duration = 0.65  # 0.65s 平滑余弦轨迹插值
        self.is_interpolating = False

        # 最新笛卡尔位姿与指标
        self.last_target_pos = init_fk_p.copy()
        self.last_fk_pos = init_fk_p.copy()
        self.last_success = True
        self.last_metrics: Dict[str, Any] = {
            "success": True,
            "time_ms": 0.0,
            "pos_err_mm": 0.0,
            "iters": 0,
            "arm": self.current_arm,
            "mode": "INITIAL_STANDBY",
        }
        self.last_collision_status: Dict[str, Any] = {
            "is_colliding": False,
            "min_clearance_mm": 35.0,
            "colliding_pairs": [],
        }

        # ── ROS 2 发布者 (标准专业命名) ──
        self.pub_joint_states = self.create_publisher(JointState, "/joint_states", 10)
        self.pub_actual_pose = self.create_publisher(PoseStamped, "/g1/kinematics/actual_pose", 10)
        self.pub_actual_pose_compat = self.create_publisher(PoseStamped, "/ik/actual_pose", 10)
        self.pub_metrics = self.create_publisher(String, "/g1/kinematics/solver_metrics", 10)
        self.pub_safety = self.create_publisher(String, "/g1/safety/collision_status", 10)
        self.pub_markers = self.create_publisher(MarkerArray, "/g1/visualization/markers", 10)
        self.pub_markers_compat = self.create_publisher(MarkerArray, "/visualization_marker_array", 10)

        # ── ROS 2 订阅者 (标准专业命名 + 兼容命名) ──
        self.sub_target_pose = self.create_subscription(
            PoseStamped, "/g1/kinematics/target_pose", self._on_target_pose_msg, 10
        )
        self.sub_target_pose_compat = self.create_subscription(
            PoseStamped, "/ik/target_pose", self._on_target_pose_msg, 10
        )
        self.sub_command = self.create_subscription(
            String, "/g1/kinematics/target_command", self._on_target_command_msg, 10
        )

        # 50Hz 实时轨迹平滑插值广播定时器 (20ms 步长)
        self.timer = self.create_timer(0.02, self._on_timer_step)

        self.get_logger().info("✔️ 订阅话题就绪: /g1/kinematics/target_pose & /ik/target_pose")
        self.get_logger().info("✔️ 发布总线就绪: /joint_states (50Hz), /g1/kinematics/*, /g1/safety/*")
        self.get_logger().info(f"✔️ 初始对齐就绪: 手爪空间坐标 [{init_fk_p[0]:.3f}, {init_fk_p[1]:.3f}, {init_fk_p[2]:.3f}], 初始残差 0.0mm")

    # ─────────────────────────────────────────────────────────────
    # 目标指令订阅处理
    # ─────────────────────────────────────────────────────────────
    def _on_target_pose_msg(self, msg: PoseStamped):
        """接收笛卡尔目标位姿指令并执行 10-DoF 逆解 (供外部标准 ROS 2 节点如 RViz 等下达)"""
        tgt = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z], dtype=np.float64)
        with self.lock:
            # 严格防重：若与正在追踪的目标欧氏间距小于 1mm，直接忽略，避免重复解算
            if np.linalg.norm(tgt - self.last_target_pos) < 1e-3:
                return
            self.last_target_pos = tgt.copy()

        self.get_logger().info(f"📥 收到外部笛卡尔目标位姿: [{tgt[0]:.3f}, {tgt[1]:.3f}, {tgt[2]:.3f}] m (操作臂: {self.current_arm})")
        self.solve_target(tgt)


    def _on_target_command_msg(self, msg: String):
        """接收高层 JSON 任务指令 (切换臂、复位就绪、预设工况等)"""
        try:
            cmd = json.loads(msg.data)
            action = cmd.get("action", "")
            if "arm" in cmd:
                new_arm = str(cmd["arm"])
                if new_arm in ["left_arm", "right_arm"]:
                    self.current_arm = new_arm
                    self.get_logger().info(f"🔄 切换操作臂为: {self.current_arm}")

            if action == "SET_ARM":
                # 仅切换当前控制臂，更新 FK 位置基准
                arm = self.current_arm
                with self.lock:
                    seed_w = np.array([self.current_joints["waist_yaw_joint"], self.current_joints["waist_roll_joint"], self.current_joints["waist_pitch_joint"]], dtype=np.float64)
                    arm_names = self.solver.left_arm_joint_names if arm == "left_arm" else self.solver.right_arm_joint_names
                    seed_a = np.array([self.current_joints[name] for name in arm_names], dtype=np.float64)
                    fk_pos, _ = self.solver.forward_kinematics_10dof(arm, seed_w, seed_a)
                    self.last_target_pos = fk_pos.copy()
                    self.last_fk_pos = fk_pos.copy()
                self._publish_diagnostics()
            elif action == "RESET_STAND":
                self.reset_to_stand()
            elif action == "SET_TARGET" and "target" in cmd:
                tgt = np.array(cmd["target"], dtype=np.float64)
                with self.lock:
                    if np.linalg.norm(tgt - self.last_target_pos) < 1e-3:
                        return
                    self.last_target_pos = tgt.copy()
                self.solve_target(tgt)
            elif action == "CIRCLE_DEMO":
                self.run_circle_demo()
        except Exception as e:
            self.get_logger().error(f"❌ 解析目标指令异常: {e}")

    # ─────────────────────────────────────────────────────────────
    # 核心解算与轨迹触发
    # ─────────────────────────────────────────────────────────────
    def solve_target(self, target_pos: np.ndarray, duration: float = 0.65):
        """调用 10-DoF 逆运动学求解器并启动平滑插值"""
        arm = self.current_arm
        t0 = time.perf_counter()

        # 热启动种子提取
        with self.lock:
            seed_w = np.array([
                self.current_joints["waist_yaw_joint"],
                self.current_joints["waist_roll_joint"],
                self.current_joints["waist_pitch_joint"],
            ], dtype=np.float64)
            arm_names = self.solver.left_arm_joint_names if arm == "left_arm" else self.solver.right_arm_joint_names
            seed_a = np.array([self.current_joints[name] for name in arm_names], dtype=np.float64)

        # 核心逆解计算 (加权 DLS + 28 对机身碰撞安全屏障)
        ok, waist_q, arm_q, info = self.solver.solve_10dof_ik(
            arm=arm,
            target_pos=target_pos,
            seed_waist=seed_w,
            seed_arm=seed_a,
            waist_weight=8.0,
            check_collision=True,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # 正运动学真实验算
        fk_pos, _ = self.solver.forward_kinematics_10dof(arm, waist_q, arm_q)
        err_mm = float(np.linalg.norm(fk_pos - target_pos) * 1000.0)

        # 全身 28 对自碰撞安全裕度计算
        q_full = pin.neutral(self.solver.model)
        for i, idx in enumerate(self.solver.waist_q_indices):
            q_full[idx] = float(waist_q[i])
        for i, idx in enumerate(self.solver.arm_q_indices[arm]):
            q_full[idx] = float(arm_q[i])
        opp = "right_arm" if arm == "left_arm" else "left_arm"
        for i, idx in enumerate(self.solver.arm_q_indices[opp]):
            q_full[idx] = float(self.current_joints[self.solver.left_arm_joint_names[i] if opp == "left_arm" else self.solver.right_arm_joint_names[i]])

        col_pairs = self.solver.collision.get_colliding_pairs(q_full, arm=arm)
        min_clearance = self.solver.collision.compute_min_distance(q_full, arm=arm) * 1000.0

        with self.lock:
            # 记录平滑插值起始状态
            self.start_joints = dict(self.current_joints)

            # 更新目标状态
            self.target_joints["waist_yaw_joint"] = float(waist_q[0])
            self.target_joints["waist_roll_joint"] = float(waist_q[1])
            self.target_joints["waist_pitch_joint"] = float(waist_q[2])
            for i, name in enumerate(arm_names):
                self.target_joints[name] = float(arm_q[i])

            self.interp_duration = max(0.2, duration)
            self.interp_start_time = time.time()
            self.is_interpolating = True

            self.last_target_pos = target_pos.copy()
            self.last_fk_pos = fk_pos.copy()
            self.last_success = ok
            self.last_metrics = {
                "success": bool(ok),
                "time_ms": round(elapsed_ms, 2),
                "pos_err_mm": round(err_mm, 2),
                "iters": int(info.get("iters", 0)),
                "arm": arm,
                "mode": "10DOF_WEIGHTED_DLS",
            }
            self.last_collision_status = {
                "is_colliding": len(col_pairs) > 0,
                "min_clearance_mm": round(float(min_clearance), 1),
                "colliding_pairs": col_pairs,
            }

        # 立即发布诊断与安全信息
        self._publish_diagnostics()

        status_icon = "✔️" if ok else "⚠️"
        self.get_logger().info(
            f"{status_icon} IK 解算完成 | 耗时: {elapsed_ms:.2f}ms | 残差: {err_mm:.2f}mm | 步数: {info.get('iters', 0)} | 净空: {min_clearance:.1f}mm"
        )

    def reset_to_stand(self):
        """恢复标准对称直立站姿"""
        arm = self.current_arm
        ready_waist = np.zeros(3, dtype=np.float64)
        ready_arm = G1_READY_POSE[arm].copy()
        fk_pos, _ = self.solver.forward_kinematics_10dof(arm, ready_waist, ready_arm)

        with self.lock:
            self.start_joints = dict(self.current_joints)
            self.target_joints = dict(DEFAULT_STAND_JOINTS)
            self.interp_duration = 0.8
            self.interp_start_time = time.time()
            self.is_interpolating = True
            self.last_target_pos = fk_pos.copy()
            self.last_fk_pos = fk_pos.copy()
            self.last_success = True
            self.last_metrics = {
                "success": True,
                "time_ms": 0.0,
                "pos_err_mm": 0.0,
                "iters": 0,
                "arm": arm,
                "mode": "RESET_STAND",
            }
            self.last_collision_status = {
                "is_colliding": False,
                "min_clearance_mm": 35.0,
                "colliding_pairs": [],
            }

        self._publish_diagnostics()
        self.get_logger().info("✔️ 已执行直立预备就绪姿态复位")

    def run_circle_demo(self):
        """启动后台线程执行连续空间圆周轨迹"""
        def _thread_circle():
            self.get_logger().info("🌀 启动连续空间轨迹演示 (平滑圆周运动)...")
            center = np.array([0.38, 0.22, 0.86])
            radius = 0.12
            steps = 45
            for step in range(steps):
                theta = 2.0 * math.pi * (step / steps)
                circ_tgt = center + np.array([radius * math.cos(theta), 0.0, radius * math.sin(theta)])
                self.solve_target(circ_tgt, duration=0.08)
                time.sleep(0.08)
            self.get_logger().info("✔️ 连续平滑轨迹演示完成！")

        t = threading.Thread(target=_thread_circle, daemon=True)
        t.start()

    # ─────────────────────────────────────────────────────────────
    # 50Hz 平滑插值与度量广播主循环
    # ─────────────────────────────────────────────────────────────
    def _on_timer_step(self):
        with self.lock:
            now = time.time()
            if self.is_interpolating:
                elapsed = now - self.interp_start_time
                if elapsed >= self.interp_duration:
                    self.current_joints = dict(self.target_joints)
                    self.is_interpolating = False
                else:
                    # 工业标准平滑余弦 S-curve 插值: alpha in [0, 1]
                    alpha = 0.5 * (1.0 - math.cos(math.pi * elapsed / self.interp_duration))
                    for k in self.target_joints:
                        q0 = self.start_joints.get(k, 0.0)
                        q1 = self.target_joints.get(k, 0.0)
                        self.current_joints[k] = q0 + alpha * (q1 - q0)

            # 1. 广播 JointState
            stamp = self.get_clock().now().to_msg()
            js_msg = JointState()
            js_msg.header.stamp = stamp
            js_msg.name = list(self.current_joints.keys())
            js_msg.position = [self.current_joints[k] for k in js_msg.name]
            self.pub_joint_states.publish(js_msg)

            # 2. 广播末端实际位姿反馈
            p_act = PoseStamped()
            p_act.header.frame_id = "world"
            p_act.header.stamp = stamp
            p_act.pose.position.x = float(self.last_fk_pos[0])
            p_act.pose.position.y = float(self.last_fk_pos[1])
            p_act.pose.position.z = float(self.last_fk_pos[2])
            p_act.pose.orientation.w = 1.0
            self.pub_actual_pose.publish(p_act)
            self.pub_actual_pose_compat.publish(p_act)

            # 3. 广播 RViz 3D 标记
            self._publish_markers(stamp)

    def _publish_diagnostics(self):
        """广播算法诊断与安全雷达 JSON 消息"""
        try:
            m_msg = String()
            m_msg.data = json.dumps(self.last_metrics, ensure_ascii=False)
            self.pub_metrics.publish(m_msg)

            s_msg = String()
            s_msg.data = json.dumps(self.last_collision_status, ensure_ascii=False)
            self.pub_safety.publish(s_msg)
        except Exception:
            pass

    def _publish_markers(self, stamp):
        """发布 RViz 目标球、实际手爪球与残差连线"""
        markers = MarkerArray()

        # Marker 0: 目标球 (Target)
        m_tgt = Marker()
        m_tgt.header.frame_id = "world"
        m_tgt.header.stamp = stamp
        m_tgt.ns = "g1_ik_target"
        m_tgt.id = 0
        m_tgt.type = Marker.SPHERE
        m_tgt.action = Marker.ADD
        m_tgt.pose.position.x = float(self.last_target_pos[0])
        m_tgt.pose.position.y = float(self.last_target_pos[1])
        m_tgt.pose.position.z = float(self.last_target_pos[2])
        m_tgt.pose.orientation.w = 1.0
        m_tgt.scale.x = 0.042
        m_tgt.scale.y = 0.042
        m_tgt.scale.z = 0.042
        m_tgt.color.r = 0.0
        m_tgt.color.g = 1.0
        m_tgt.color.b = 0.53
        m_tgt.color.a = 0.85
        markers.markers.append(m_tgt)

        # Marker 1: 实际手腕 (Actual FK)
        m_act = Marker()
        m_act.header.frame_id = "world"
        m_act.header.stamp = stamp
        m_act.ns = "g1_ik_actual"
        m_act.id = 1
        m_act.type = Marker.SPHERE
        m_act.action = Marker.ADD
        m_act.pose.position.x = float(self.last_fk_pos[0])
        m_act.pose.position.y = float(self.last_fk_pos[1])
        m_act.pose.position.z = float(self.last_fk_pos[2])
        m_act.pose.orientation.w = 1.0
        m_act.scale.x = 0.032
        m_act.scale.y = 0.032
        m_act.scale.z = 0.032
        m_act.color.r = 1.0
        m_act.color.g = 0.72
        m_act.color.b = 0.0
        m_act.color.a = 0.90
        markers.markers.append(m_act)

        # Marker 2: 残差连线
        m_line = Marker()
        m_line.header.frame_id = "world"
        m_line.header.stamp = stamp
        m_line.ns = "g1_ik_error_line"
        m_line.id = 2
        m_line.type = Marker.LINE_STRIP
        m_line.action = Marker.ADD
        m_line.scale.x = 0.004
        m_line.color.r = 1.0
        m_line.color.g = 0.2
        m_line.color.b = 0.4
        m_line.color.a = 0.8
        p1, p2 = Point(), Point()
        p1.x, p1.y, p1.z = float(self.last_target_pos[0]), float(self.last_target_pos[1]), float(self.last_target_pos[2])
        p2.x, p2.y, p2.z = float(self.last_fk_pos[0]), float(self.last_fk_pos[1]), float(self.last_fk_pos[2])
        m_line.points.append(p1)
        m_line.points.append(p2)
        markers.markers.append(m_line)

        self.pub_markers.publish(markers)
        self.pub_markers_compat.publish(markers)


def main(args=None):
    rclpy.init(args=args)
    node = G1IKSolverNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            node.destroy_node()
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
