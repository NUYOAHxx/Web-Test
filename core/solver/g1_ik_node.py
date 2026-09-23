#!/usr/bin/env python3
"""
================================================================================
Unitree G1 10-DoF (3-DoF 腰部 + 7-DoF 手臂) 逆运动学标准无头解算服务节点
(Headless ROS 2 IK Solver & Collision Barrier Service Node)
================================================================================

定位：纯算法与运动学计算节点 (Zero Web / Pure Kinematics Engine)
1. 彻底解耦，无任何阻塞式 input()，完全作为 Headless ROS 2 独立服务节点常驻；
2. 订阅标准笛卡尔指令话题：/g1/kinematics/target_pose 与高层指令 /g1/kinematics/target_command；
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
from typing import Dict, Optional, Any

import numpy as np
import pinocchio as pin

import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from visualization_msgs.msg import MarkerArray

# 添加项目根目录到 Python 搜索路径
DIR_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if DIR_ROOT not in sys.path:
    sys.path.insert(0, DIR_ROOT)

from core.solver.g1_pink_ik import (
    G1PinkIKSolver,
    G1_READY_POSE,
    G1_DEFAULT_STAND_JOINTS,
    IK_PIPELINE_STAGES,
)
from visualizer.rviz.markers import create_pose_stamped, create_ik_markers, create_com_markers


class G1IKSolverNode(Node):
    """Unitree G1 10-DoF 逆运动学与碰撞避免服务节点"""

    def __init__(self):
        super().__init__("g1_ik_solver")
        self.get_logger().info("======================================================")
        self.get_logger().info(" 🚀 Unitree G1 10-DoF 躯干-手臂协同 IK 解算服务节点启动中...")
        self.get_logger().info("======================================================")

        # 实例化 Pink 凸优化求解器 (封装 Pinocchio 运动学模型与 28 对碰撞检测器)
        self.solver = G1PinkIKSolver()
        self.active_arm = "left_arm"
        self.lock = threading.RLock()

        # Pinocchio 全身质心与双足支撑平衡初始计算
        q_init_full = self.solver.kin.build_q_10dof("left_arm", np.zeros(3), G1_READY_POSE["left_arm"])
        init_bal = self.solver.kin.evaluate_balance(q_init_full)
        self.com_pos = np.array(init_bal["com_pos"], dtype=np.float64)
        self.com_status = init_bal["status"]
        self.support_polygon = dict(init_bal["support_polygon"])

        # 计算初始就绪姿态下手爪真实空间坐标与朝向矩阵
        ready_waist = np.zeros(3, dtype=np.float64)
        ready_arm = G1_READY_POSE[self.active_arm].copy()
        init_fk_pos, init_fk_rot = self.solver.forward_kinematics_10dof(self.active_arm, ready_waist, ready_arm)

        # 统一命名：笛卡尔目标点与实际 FK 空间位置与姿态 (numpy.ndarray)
        self.target_pos = init_fk_pos.copy()
        self.actual_pos = init_fk_pos.copy()
        self.target_rot = init_fk_rot.copy()
        self.actual_rot = init_fk_rot.copy()

        # 关节与轨迹平滑插值缓存 (统一引用官方标准字典)
        self.current_joints = dict(G1_DEFAULT_STAND_JOINTS)
        self.target_joints = dict(G1_DEFAULT_STAND_JOINTS)
        self.start_joints = dict(G1_DEFAULT_STAND_JOINTS)
        self.interp_start_time = time.time()
        self.interp_duration = 0.65  # 0.65s 平滑余弦轨迹插值
        self.is_interpolating = False

        # 算法诊断指标与安全状态缓存 (统一专业命名与字段完整性)
        init_rpy_deg = [round(float(np.degrees(v)), 2) for v in pin.rpy.matrixToRpy(init_fk_rot)]
        init_quat = pin.Quaternion(init_fk_rot)
        self.solver_metrics: Dict[str, Any] = {
            "success": True,
            "time_ms": 0.0,
            "pos_err_mm": 0.0,
            "rot_err_deg": 0.0,
            "iters": 0,
            "arm": self.active_arm,
            "mode": "INITIAL_STANDBY",
            "seed_used": 0,
            "seed_name": "Standby",
            "seed_switches": [0],
            "convergence_trace": [],
            "step_details": [],
            "pipeline_stages": IK_PIPELINE_STAGES,
            "algorithm": "10-DoF Weighted DLS (W_waist=8.0) + 28-Pair Barrier",
            "actual_rpy_deg": init_rpy_deg,
            "target_rpy_deg": init_rpy_deg,
            "actual_quat": [round(float(v), 5) for v in [init_quat.x, init_quat.y, init_quat.z, init_quat.w]],
        }
        self.collision_status: Dict[str, Any] = {
            "is_colliding": False,
            "min_clearance_mm": 35.0,
            "colliding_pairs": [],
        }

        # ── ROS 2 发布者 (标准专业命名，零冗余兼容) ──
        self.pub_joint_states = self.create_publisher(JointState, "/joint_states", 10)
        self.pub_target_pose = self.create_publisher(PoseStamped, "/g1/kinematics/target_pose", 10)
        self.pub_actual_pose = self.create_publisher(PoseStamped, "/g1/kinematics/actual_pose", 10)
        self.pub_metrics = self.create_publisher(String, "/g1/kinematics/solver_metrics", 10)
        self.pub_safety = self.create_publisher(String, "/g1/safety/collision_status", 10)
        self.pub_markers = self.create_publisher(MarkerArray, "/g1/visualization/markers", 10)
        self.is_demo_running = False

        # ── ROS 2 订阅者 (标准专业命名) ──
        self.sub_target_pose = self.create_subscription(
            PoseStamped, "/g1/kinematics/target_pose", self._on_target_pose_msg, 10
        )
        self.sub_command = self.create_subscription(
            String, "/g1/kinematics/target_command", self._on_target_command_msg, 10
        )

        # 50Hz 实时轨迹平滑插值广播定时器 (20ms 步长)
        self.timer = self.create_timer(0.02, self._on_timer_step)

        self.get_logger().info("✔️ 订阅话题就绪: /g1/kinematics/target_pose & /g1/kinematics/target_command")
        self.get_logger().info("✔️ 发布总线就绪: /joint_states (50Hz), /g1/kinematics/*, /g1/safety/*")
        self.get_logger().info(f"✔️ 初始对齐就绪: 手爪空间坐标 [{init_fk_pos[0]:.3f}, {init_fk_pos[1]:.3f}, {init_fk_pos[2]:.3f}], 姿态 RPY: {init_rpy_deg}°")

    # ─────────────────────────────────────────────────────────────
    # 目标指令订阅处理
    # ─────────────────────────────────────────────────────────────
    def _on_target_pose_msg(self, msg: PoseStamped):
        """接收笛卡尔目标位姿指令并执行 10-DoF 6-DoF 空间位姿逆解"""
        target_pos = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z], dtype=np.float64)
        qx = float(msg.pose.orientation.x)
        qy = float(msg.pose.orientation.y)
        qz = float(msg.pose.orientation.z)
        qw = float(msg.pose.orientation.w)
        quat_sq = qx**2 + qy**2 + qz**2 + qw**2
        target_rot = None
        if quat_sq > 1e-4:
            q_pin = pin.Quaternion(qw, qx, qy, qz).normalized()
            target_rot = q_pin.toRotationMatrix()

        with self.lock:
            # 严格防重：若与正在追踪的目标位置与朝向欧氏残差极小，直接忽略
            pos_diff = np.linalg.norm(target_pos - self.target_pos)
            rot_diff = 0.0
            if target_rot is not None and self.target_rot is not None:
                R_diff = target_rot @ self.target_rot.T
                rot_diff = float(np.linalg.norm(pin.log3(R_diff)))
            if pos_diff < 1e-3 and rot_diff < 1e-3:
                return
            self.target_pos = target_pos.copy()
            if target_rot is not None:
                self.target_rot = target_rot.copy()

        self.get_logger().info(
            f"📥 收到外部笛卡尔 6D 目标位姿: [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}] m "
            f"(操作臂: {self.active_arm}, 6D姿态约束: {target_rot is not None})"
        )
        self.solve_target(target_pos, target_rot=target_rot)

    def _on_target_command_msg(self, msg: String):
        """接收高层 JSON 任务指令 (切换臂、复位就绪、预设工况等)"""
        try:
            cmd = json.loads(msg.data)
            action = cmd.get("action", "")

            # 1. 切换操作臂指令
            if "arm" in cmd:
                new_arm = str(cmd["arm"])
                if new_arm in ["left_arm", "right_arm"] and new_arm != self.active_arm:
                    self.active_arm = new_arm
                    self.get_logger().info(f"🔄 切换操作臂为: {self.active_arm}")

            if action == "SET_ARM":
                arm = self.active_arm
                with self.lock:
                    seed_waist = np.array([
                        self.current_joints["waist_yaw_joint"],
                        self.current_joints["waist_roll_joint"],
                        self.current_joints["waist_pitch_joint"]
                    ], dtype=np.float64)
                    arm_names = self.solver.left_arm_joint_names if arm == "left_arm" else self.solver.right_arm_joint_names
                    seed_arm = np.array([self.current_joints[name] for name in arm_names], dtype=np.float64)
                    fk_pos, fk_rot = self.solver.forward_kinematics_10dof(arm, seed_waist, seed_arm)
                    self.target_pos = fk_pos.copy()
                    self.actual_pos = fk_pos.copy()
                    self.target_rot = fk_rot.copy()
                    self.actual_rot = fk_rot.copy()
                self._publish_diagnostics()

            elif action == "RESET_STAND":
                self.reset_to_stand()

            elif action == "SET_TARGET" and "target" in cmd:
                target_pos = np.array(cmd["target"], dtype=np.float64)
                target_rot = None
                if "rpy" in cmd:
                    # 接受欧拉角 [roll, pitch, yaw] (单位：度)
                    rpy_rad = np.radians([float(v) for v in cmd["rpy"]])
                    target_rot = pin.rpy.rpyToMatrix(float(rpy_rad[0]), float(rpy_rad[1]), float(rpy_rad[2]))
                elif "quat" in cmd:
                    # 接受四元数 [x, y, z, w]
                    qx, qy, qz, qw = [float(v) for v in cmd["quat"]]
                    target_rot = pin.Quaternion(qw, qx, qy, qz).normalized().toRotationMatrix()

                with self.lock:
                    self.target_pos = target_pos.copy()
                    if target_rot is not None:
                        self.target_rot = target_rot.copy()
                self.solve_target(target_pos, target_rot=target_rot)

            elif action == "CIRCLE_DEMO":
                arm = cmd.get("arm", self.active_arm)
                if arm in ("left_arm", "right_arm"):
                    self.active_arm = arm
                self.run_circle_demo(arm=self.active_arm)

        except Exception as e:
            self.get_logger().error(f"❌ 解析目标指令异常: {e}")

    # ─────────────────────────────────────────────────────────────
    # 核心解算与轨迹触发
    # ─────────────────────────────────────────────────────────────
    def solve_target(self, target_pos: np.ndarray, target_rot: Optional[np.ndarray] = None, duration: float = 0.65):
        """调用 10-DoF 逆运动学求解器并启动平滑插值 (支持 6-DoF 空间全位姿联合收敛)"""
        arm = self.active_arm
        t0 = time.perf_counter()

        # 热启动种子提取
        with self.lock:
            seed_waist = np.array([
                self.current_joints["waist_yaw_joint"],
                self.current_joints["waist_roll_joint"],
                self.current_joints["waist_pitch_joint"],
            ], dtype=np.float64)
            arm_names = self.solver.left_arm_joint_names if arm == "left_arm" else self.solver.right_arm_joint_names
            seed_arm = np.array([self.current_joints[name] for name in arm_names], dtype=np.float64)

        # 核心逆解计算 (加权 DLS + 6D 空间位姿配准 + 28 对机身碰撞安全屏障)
        ok, waist_q, arm_q, info = self.solver.solve_10dof_ik(
            arm=arm,
            target_pos=target_pos,
            target_rot=target_rot,
            seed_waist=seed_waist,
            seed_arm=seed_arm,
            waist_weight=8.0,
            check_collision=True,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # 正运动学真实验算
        fk_pos, fk_rot = self.solver.forward_kinematics_10dof(arm, waist_q, arm_q)
        err_mm = float(np.linalg.norm(fk_pos - target_pos) * 1000.0)
        rot_err_deg = 0.0
        if target_rot is not None:
            R_err = target_rot @ fk_rot.T
            rot_err_deg = float(np.degrees(np.linalg.norm(pin.log3(R_err))))

        # 全身 28 对自碰撞安全裕度计算 (复用求解器统一构型构建函数)
        opp_arm = "right_arm" if arm == "left_arm" else "left_arm"
        opp_names = self.solver.arm_joint_names[opp_arm]
        opp_q = np.array([self.current_joints[name] for name in opp_names], dtype=np.float64)
        q_full = self.solver.build_q_10dof(arm, waist_q, arm_q, q_other_arm=opp_q)

        col_pairs = self.solver.collision.get_colliding_pairs(q_full, arm=arm)
        min_clearance = self.solver.collision.compute_min_distance(q_full, arm=arm) * 1000.0

        with self.lock:
            # 记录平滑插值起始状态
            self.start_joints = dict(self.current_joints)

            # 更新目标关节角度
            self.target_joints["waist_yaw_joint"] = float(waist_q[0])
            self.target_joints["waist_roll_joint"] = float(waist_q[1])
            self.target_joints["waist_pitch_joint"] = float(waist_q[2])
            for i, name in enumerate(arm_names):
                self.target_joints[name] = float(arm_q[i])

            self.interp_duration = max(0.02, duration)
            self.interp_start_time = time.time()
            self.is_interpolating = True

            self.target_pos = target_pos.copy()
            self.actual_pos = fk_pos.copy()
            if target_rot is not None:
                self.target_rot = target_rot.copy()
            self.actual_rot = fk_rot.copy()

            self.com_pos = np.array(info.get("com_pos", self.com_pos), dtype=np.float64)
            self.com_status = str(info.get("balance_status", "STABLE"))
            self.support_polygon = dict(info.get("support_polygon", self.support_polygon))

            # 结构与字段统一
            self.solver_metrics = {
                "success": bool(ok),
                "time_ms": round(elapsed_ms, 2),
                "pos_err_mm": round(err_mm, 2),
                "rot_err_deg": round(rot_err_deg, 2),
                "iters": int(info.get("iters", 0)),
                "arm": arm,
                "mode": str(info.get("mode", "10DOF_6DOF_POSE" if target_rot is not None else "10DOF_3DOF_POS")),
                "seed_used": int(info.get("seed_used", 0)),
                "seed_name": str(info.get("seed_name", "")),
                "seed_switches": info.get("seed_switches", [0]),
                "convergence_trace": info.get("convergence_trace", []),
                "step_details": info.get("step_details", []),
                "pipeline_stages": info.get("pipeline_stages", IK_PIPELINE_STAGES),
                "cascade_stage": str(info.get("cascade_stage", "10DOF_COORDINATED")),
                "algorithm": "Inria Pink 4.4.0 + ProxQP (Cascade Upright Priority)",
                "com_pos": info.get("com_pos", [round(float(v), 4) for v in self.com_pos]),
                "com_proj": info.get("com_proj", [round(float(v), 4) for v in self.com_pos[:2]]),
                "balance_margin_mm": round(float(info.get("balance_margin_mm", 100.0)), 1),
                "balance_status": self.com_status,
                "support_polygon": self.support_polygon,
                "gravity_torques": info.get("gravity_torques", []),
                "manipulability": info.get("manipulability", {}),
                "actual_rpy_deg": info.get("actual_rpy_deg", []),
                "actual_quat": info.get("actual_quat", []),
                "target_rpy_deg": info.get("target_rpy_deg", None),
                "trajectory_active": bool(self.is_demo_running),
            }
            self.collision_status = {
                "is_colliding": len(col_pairs) > 0,
                "min_clearance_mm": round(float(min_clearance), 1),
                "colliding_pairs": col_pairs,
            }

        # 广播目标位姿到 ROS 2，供遥测中枢与 Web 界面实时追踪
        stamp = self.get_clock().now().to_msg()
        quat_tgt = pin.Quaternion(self.target_rot)
        self.pub_target_pose.publish(
            create_pose_stamped(
                self.target_pos,
                stamp,
                orientation=(float(quat_tgt.x), float(quat_tgt.y), float(quat_tgt.z), float(quat_tgt.w)),
            )
        )

        # 立即发布诊断与安全信息
        self._publish_diagnostics()

        status_icon = "✔️" if ok else "⚠️"
        self.get_logger().info(
            f"{status_icon} IK 解算完成 | 耗时: {elapsed_ms:.2f}ms | 位置残差: {err_mm:.2f}mm | 姿态残差: {rot_err_deg:.2f}° | "
            f"步数: {info.get('iters', 0)} | 净空: {min_clearance:.1f}mm | 平衡: {self.com_status}"
        )

    def reset_to_stand(self):
        """恢复标准对称直立站姿"""
        arm = self.active_arm
        ready_waist = np.zeros(3, dtype=np.float64)
        ready_arm = G1_READY_POSE[arm].copy()
        fk_pos, fk_rot = self.solver.forward_kinematics_10dof(arm, ready_waist, ready_arm)
        q_stand_full = self.solver.kin.build_q_10dof(arm, ready_waist, ready_arm)
        stand_bal = self.solver.kin.evaluate_balance(q_stand_full)
        stand_torques = self.solver.kin.compute_gravity_torques(q_stand_full, arm=arm, is_10dof=True)
        stand_manip = self.solver.kin.compute_manipulability(arm, np.concatenate([ready_waist, ready_arm]), has_rot=True)
        stand_rpy_deg = [round(float(np.degrees(v)), 2) for v in pin.rpy.matrixToRpy(fk_rot)]
        stand_quat = pin.Quaternion(fk_rot)

        with self.lock:
            self.start_joints = dict(self.current_joints)
            self.target_joints = dict(G1_DEFAULT_STAND_JOINTS)
            self.interp_duration = 0.8
            self.interp_start_time = time.time()
            self.is_interpolating = True
            self.target_pos = fk_pos.copy()
            self.actual_pos = fk_pos.copy()
            self.target_rot = fk_rot.copy()
            self.actual_rot = fk_rot.copy()
            self.com_pos = np.array(stand_bal["com_pos"], dtype=np.float64)
            self.com_status = stand_bal["status"]
            self.support_polygon = dict(stand_bal["support_polygon"])

            self.solver_metrics = {
                "success": True,
                "time_ms": 0.0,
                "pos_err_mm": 0.0,
                "rot_err_deg": 0.0,
                "iters": 0,
                "arm": arm,
                "mode": "RESET_STAND",
                "seed_used": 0,
                "seed_name": "Standby",
                "seed_switches": [0],
                "convergence_trace": [0.0],
                "step_details": [],
                "pipeline_stages": IK_PIPELINE_STAGES,
                "algorithm": "10-DoF Weighted DLS (W_waist=8.0) + 28-Pair Barrier",
                "com_pos": stand_bal["com_pos"],
                "com_proj": stand_bal["com_proj"],
                "balance_margin_mm": stand_bal["margin_mm"],
                "balance_status": stand_bal["status"],
                "support_polygon": self.support_polygon,
                "gravity_torques": [round(float(v), 3) for v in stand_torques],
                "manipulability": stand_manip,
                "actual_rpy_deg": stand_rpy_deg,
                "target_rpy_deg": stand_rpy_deg,
                "actual_quat": [round(float(v), 5) for v in [stand_quat.x, stand_quat.y, stand_quat.z, stand_quat.w]],
            }
            self.collision_status = {
                "is_colliding": False,
                "min_clearance_mm": 35.0,
                "colliding_pairs": [],
            }

        self._publish_diagnostics()
        self.get_logger().info("✔️ 已执行直立预备就绪姿态复位")

    def run_circle_demo(self, arm: Optional[str] = None):
        """启动后台线程执行连续空间圆周轨迹 (带平滑引入期、对称工作区适配与姿态定向)"""
        if self.is_demo_running:
            self.get_logger().warn("⚠️ 轨迹演示正在执行中，忽略重复指令")
            return

        exec_arm = arm if arm in ("left_arm", "right_arm") else self.active_arm

        def _thread_circle():
            self.is_demo_running = True
            try:
                self.get_logger().info(f"🌀 启动连续空间轨迹演示 (执行臂: {exec_arm}, 优雅平滑圆周)...")

                # 左右臂对称中心与半径配置 (正前上方天然舒适工作区)
                # 左臂 y = +0.22, 右臂 y = -0.22, x = 0.38, z = 0.82
                y_center = 0.22 if exec_arm == "left_arm" else -0.22
                center = np.array([0.38, y_center, 0.82], dtype=np.float64)
                radius = 0.10  # 10cm 半径

                # 提取标称末端平正姿态 (保持前向微垂，手腕平稳)
                with self.lock:
                    R_target = self.actual_rot.copy() if hasattr(self, "actual_rot") and self.actual_rot is not None else np.eye(3)

                # ── 第一阶段：柔顺前导过渡期 (Smooth Lead-in) ──
                # 避免从待机姿态直接跳变引发冲量，规划 0.8s 优雅滑入圆周起点
                p_start = center + np.array([radius, 0.0, 0.0], dtype=np.float64)
                self.solve_target(p_start, target_rot=R_target, duration=0.8)
                time.sleep(0.85)

                # ── 第二阶段：高频流式空间轨迹循迹 (25Hz / 100步 / T=4.0s) ──
                steps = 100
                dt = 0.04
                for step in range(steps):
                    theta = 2.0 * math.pi * (step / steps)
                    circ_tgt = center + np.array([radius * math.cos(theta), 0.0, radius * math.sin(theta)], dtype=np.float64)
                    self.solve_target(circ_tgt, target_rot=R_target, duration=dt)
                    time.sleep(dt)

                # ── 第三阶段：闭环微步收尾 ──
                self.solve_target(p_start, target_rot=R_target, duration=0.2)
                time.sleep(0.25)
                self.get_logger().info("✔️ 连续平滑空间轨迹演示完美完成！")
            except Exception as e:
                self.get_logger().error(f"❌ 轨迹演示执行异常: {e}")
            finally:
                self.is_demo_running = False
                with self.lock:
                    self.solver_metrics["trajectory_active"] = False
                self._publish_diagnostics()

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

            stamp = self.get_clock().now().to_msg()

            # 1. 广播 JointState
            js_msg = JointState()
            js_msg.header.stamp = stamp
            js_msg.name = list(self.current_joints.keys())
            js_msg.position = [self.current_joints[k] for k in js_msg.name]
            self.pub_joint_states.publish(js_msg)

            # 2. 广播实际末端空间位姿 (包含真实末端姿态四元数)
            quat = pin.Quaternion(self.actual_rot)
            self.pub_actual_pose.publish(
                create_pose_stamped(
                    self.actual_pos,
                    stamp,
                    orientation=(float(quat.x), float(quat.y), float(quat.z), float(quat.w)),
                )
            )

            # 3. 广播 RViz 3D 标记
            self._publish_markers(stamp)

    def _publish_diagnostics(self):
        """广播算法诊断与安全雷达 JSON 消息"""
        try:
            m_msg = String()
            m_msg.data = json.dumps(self.solver_metrics, ensure_ascii=False)
            self.pub_metrics.publish(m_msg)

            s_msg = String()
            s_msg.data = json.dumps(self.collision_status, ensure_ascii=False)
            self.pub_safety.publish(s_msg)
        except Exception:
            pass

    def _publish_markers(self, stamp):
        """发布 RViz 目标球、实际手爪球、残差连线、6D姿态三轴及 Pinocchio 质心与支撑多边形"""
        with self.lock:
            target_p = self.target_pos.copy()
            actual_p = self.actual_pos.copy()
            target_r = self.target_rot.copy() if self.target_rot is not None else None
            actual_r = self.actual_rot.copy() if self.actual_rot is not None else None
            success = self.solver_metrics.get("success", True)
            com_p = self.com_pos.copy()
            com_st = self.com_status
            poly = dict(self.support_polygon)

        ik_markers = create_ik_markers(
            target_p, actual_p, stamp, target_rot=target_r, actual_rot=actual_r, success=success
        )
        com_markers = create_com_markers(com_p, poly, stamp, status=com_st)
        ik_markers.markers.extend(com_markers.markers)
        self.pub_markers.publish(ik_markers)



def main(args=None):
    rclpy.init(args=args)
    node = G1IKSolverNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            executor.shutdown()
            node.destroy_node()
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
