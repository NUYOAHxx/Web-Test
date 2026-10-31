#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2-2-3 构型运动学与 IK 测试套件。

验证 9-DOF 人形机器人（腰部 2-DOF + 单臂 2-2-3 7-DOF 构型）：
  - 躯干腰部 2-DOF: Hip_Link1 (Pitch, Y 轴), Body_Link (Yaw, Z 轴)
  - 肩关节 2-DOF: LHand_Link1 (Yaw, Z 轴), LHand_Link2 (Pitch, Y 轴) -> 肩特征点 S
  - 上臂段: LHand_Link2 到 LHand_Link3 (长 0.35m)，LHand_Link3 坐标系位于肘部
  - 肘关节 2-DOF: LHand_Link3 (Roll, X 轴), LHand_Link4 (Pitch, Y 轴) -> 原点重合于肘特征点 E
  - 前臂段: LHand_Link4 到 LHand_Link5 (长 0.30m)
  - 腕关节 3-DOF: LHand_Link5 (Roll, X 轴), LHand_Link6 (Pitch, Y 轴), LHand_Link7 (Yaw, Z 轴) -> 原点重合于腕特征点 W
  - 末端工具段: LHand_Link7 到 tcp (长 0.15m)
  - 生产限位严格符合 Notion / limits_reference.json
  - 生产真实就绪姿态: [0°, 0°, -30°, -45°, 0°, -40°, 0°, 90°, 0°]
"""

import math
import unittest
import numpy as np
import pinocchio as pin

from ha_arm_ik.ik_base import sew_angle_from_points, REF_DIR, S_JOINT, E_JOINT, W_JOINT
from ha_arm_ik.single_arm_ik import JointStateSnapshot
from tests.fixtures import make_223_solver, robot_223_xml


class Test223KinematicTopology(unittest.TestCase):
    """测试 2-2-3 构型的关节轴、连杆拓扑与坐标系原点分布"""

    @classmethod
    def setUpClass(cls):
        cls.solver, cls.state, cls.q_ready = make_223_solver()

    def test_223_active_joint_names_and_count(self):
        """验证 9-DOF 链关节全称与顺规"""
        expected_names = [
            "Hip_Link1", "Body_Link",
            "LHand_Link1", "LHand_Link2",
            "LHand_Link3", "LHand_Link4",
            "LHand_Link5", "LHand_Link6", "LHand_Link7"
        ]
        self.assertEqual(self.solver.active_joint_names, expected_names)
        self.assertEqual(len(self.solver.active_joint_names), 9)

    def test_223_coordinate_frame_origins_and_coincidence(self):
        """验证 LHand_Link3 坐标系位于肘部，且肘部两轴、腕部三轴几何原点重合"""
        s = self.solver
        q = self.q_ready
        pin.forwardKinematics(s.pin_model, s.pin_data, q)
        pin.updateFramePlacements(s.pin_model, s.pin_data)

        # 获取各关节点在世界坐标系下的三维平移
        pos = {}
        for name in s.active_joint_names:
            jid = s.pin_model.getJointId(name)
            pos[name] = s.pin_data.oMi[jid].translation.copy()
        pos["tcp"] = s.pin_data.oMf[s.pin_ee_frame_id].translation.copy()

        # 1. 肘部重合性：LHand_Link3 (Roll) 与 LHand_Link4 (Pitch) 原点重合于肘部
        dist_elbow_cluster = np.linalg.norm(pos["LHand_Link3"] - pos["LHand_Link4"])
        self.assertAlmostEqual(dist_elbow_cluster, 0.0, places=6,
                               msg="LHand_Link3 与 LHand_Link4 必须同在肘部轴心")

        # 2. 上臂段长度：从肩 S (Link2) 到 肘 E (Link3) 距离为 0.35m
        upper_arm_len = np.linalg.norm(pos["LHand_Link3"] - pos["LHand_Link2"])
        self.assertAlmostEqual(upper_arm_len, 0.35, places=5,
                               msg="上臂长度 (Link2->Link3) 必须为 0.35m")

        # 3. 前臂段长度：从肘 E (Link4) 到 腕 W (Link5) 距离为 0.30m
        forearm_len = np.linalg.norm(pos["LHand_Link5"] - pos["LHand_Link4"])
        self.assertAlmostEqual(forearm_len, 0.30, places=5,
                               msg="前臂长度 (Link4->Link5) 必须为 0.30m")

        # 4. 腕部球面重合性：Link5 (Roll), Link6 (Pitch), Link7 (Yaw) 原点重合
        dist_w56 = np.linalg.norm(pos["LHand_Link5"] - pos["LHand_Link6"])
        dist_w67 = np.linalg.norm(pos["LHand_Link6"] - pos["LHand_Link7"])
        self.assertAlmostEqual(dist_w56, 0.0, places=6, msg="Link5 与 Link6 原点必须重合")
        self.assertAlmostEqual(dist_w67, 0.0, places=6, msg="Link6 与 Link7 原点必须重合")

        # 5. 手端段长度：从腕 (Link7) 到 tcp 距离为 0.15m
        hand_len = np.linalg.norm(pos["tcp"] - pos["LHand_Link7"])
        self.assertAlmostEqual(hand_len, 0.15, places=5, msg="手端长度必须为 0.15m")


class Test223ArmAngleAndSelfMotion(unittest.TestCase):
    """测试 2-2-3 构型下的 SEW 臂角定义与自运动特性"""

    @classmethod
    def setUpClass(cls):
        cls.solver, cls.state, cls.q_ready = make_223_solver()

    def test_sew_joint_constants(self):
        """验证 SEW 特征关节点与系统配置的一致性"""
        s = self.solver
        self.assertEqual(s.s_joint, "LHand_Link2")
        self.assertEqual(s.e_joint, "LHand_Link4")
        self.assertEqual(s.w_joint, "LHand_Link6")
        self.assertEqual((S_JOINT, E_JOINT, W_JOINT), ("LHand_Link2", "LHand_Link4", "LHand_Link6"))

    def test_ready_pose_arm_angle(self):
        """验证生产就绪姿态下的臂角计算为 -180°（或 π）"""
        s = self.solver
        q = self.q_ready
        psi = s.chain.psi(q)
        # 允许 ±180° 等价
        psi_deg = math.degrees(psi)
        self.assertTrue(abs(psi_deg - 180.0) < 0.1 or abs(psi_deg + 180.0) < 0.1,
                        f"就绪态臂角应为 ±180°，实际为 {psi_deg:.2f}°")

    def test_elbow_roll_link3_modulates_arm_angle(self):
        """验证旋转 LHand_Link3（肘部 Roll）改变臂角 ψ，而肩 S 和腕 W 间距保持不变"""
        s = self.solver
        q_base = self.q_ready.copy()

        pin.forwardKinematics(s.pin_model, s.pin_data, q_base)
        pin.updateFramePlacements(s.pin_model, s.pin_data)
        S0 = s.pin_data.oMi[s.pin_model.getJointId("LHand_Link2")].translation.copy()
        W0 = s.pin_data.oMi[s.pin_model.getJointId("LHand_Link6")].translation.copy()
        base_span = np.linalg.norm(W0 - S0)

        prev_psi = s.chain.psi(q_base)
        for delta_deg in [15.0, 30.0, 45.0]:
            q_test = q_base.copy()
            q_test[4] += math.radians(delta_deg)  # 关节 4 为 LHand_Link3

            pin.forwardKinematics(s.pin_model, s.pin_data, q_test)
            pin.updateFramePlacements(s.pin_model, s.pin_data)
            S = s.pin_data.oMi[s.pin_model.getJointId("LHand_Link2")].translation.copy()
            W = s.pin_data.oMi[s.pin_model.getJointId("LHand_Link6")].translation.copy()

            # 肩关节位置不应改变
            np.testing.assert_allclose(S, S0, atol=1e-6)
            # 臂长跨度保持恒定
            span = np.linalg.norm(W - S)
            self.assertAlmostEqual(span, base_span, places=5)

            # 臂角 ψ 发生了显著改变
            psi = s.chain.psi(q_test)
            diff = abs((psi - prev_psi + math.pi) % (2 * math.pi) - math.pi)
            self.assertGreater(diff, math.radians(5.0))

    def test_elbow_pitch_link4_modulates_reach(self):
        """验证弯曲 LHand_Link4（肘部 Pitch）直接改变手臂伸展距离 SW"""
        s = self.solver
        q = self.q_ready.copy()

        # 计算 -40° 时的跨度
        pin.forwardKinematics(s.pin_model, s.pin_data, q)
        S = s.pin_data.oMi[s.pin_model.getJointId("LHand_Link2")].translation
        W = s.pin_data.oMi[s.pin_model.getJointId("LHand_Link6")].translation
        span_ready = np.linalg.norm(W - S)

        # 展开到 -10°（更直）
        q_straight = q.copy()
        q_straight[5] = math.radians(-10.0)
        pin.forwardKinematics(s.pin_model, s.pin_data, q_straight)
        S = s.pin_data.oMi[s.pin_model.getJointId("LHand_Link2")].translation
        W = s.pin_data.oMi[s.pin_model.getJointId("LHand_Link6")].translation
        span_straight = np.linalg.norm(W - S)

        # 屈曲到 -70°（更弯）
        q_bent = q.copy()
        q_bent[5] = math.radians(-70.0)
        pin.forwardKinematics(s.pin_model, s.pin_data, q_bent)
        S = s.pin_data.oMi[s.pin_model.getJointId("LHand_Link2")].translation
        W = s.pin_data.oMi[s.pin_model.getJointId("LHand_Link6")].translation
        span_bent = np.linalg.norm(W - S)

        self.assertGreater(span_straight, span_ready)
        self.assertLess(span_bent, span_ready)


class Test223NotionLimitsAndWorkspaceSolve(unittest.TestCase):
    """测试 2-2-3 构型在生产 Notion 限位下的 7 大典型工况逆解求解"""

    @classmethod
    def setUpClass(cls):
        cls.solver, cls.state, cls.q_ready = make_223_solver()
        s = cls.solver
        pin.forwardKinematics(s.pin_model, s.pin_data, cls.q_ready)
        pin.updateFramePlacements(s.pin_model, s.pin_data)
        cls.T_ready = s.pin_data.oMf[s.pin_ee_frame_id].homogeneous.copy()

        # Notion 真实限位 (度)
        cls.notion_limits = [
            (-120.0, 120.0),  # Hip_Link1
            (-360.0, 360.0),  # Body_Link
            (-360.0, 360.0),  # LHand_Link1
            (-90.0, 45.0),    # LHand_Link2 (肩俯仰)
            (-360.0, 360.0),  # LHand_Link3 (肘自转)
            (-120.0, 40.0),   # LHand_Link4 (肘屈伸)
            (-360.0, 360.0),  # LHand_Link5 (腕自转)
            (60.0, 120.0),    # LHand_Link6 (腕俯仰)
            (-30.0, 30.0),    # LHand_Link7 (腕偏航)
        ]

    def test_notion_limits_configuration(self):
        """验证求解器的限位上下界严格符合 Notion 规格"""
        s = self.solver
        for i, (lo_deg, hi_deg) in enumerate(self.notion_limits):
            lo_rad, hi_rad = math.radians(lo_deg), math.radians(hi_deg)
            self.assertAlmostEqual(s.lower[i], lo_rad, places=4)
            self.assertAlmostEqual(s.upper[i], hi_rad, places=4)

    def test_7_production_cases_convergence_and_limits(self):
        """测试 7 大工业场景：全收敛、高精度（<1mm）、全关节在 Notion 限位内"""
        test_cases = {
            "ready": np.array([0.0, 0.0, 0.0]),
            "fwd5": np.array([0.05, 0.0, 0.0]),
            "fwd10": np.array([0.10, 0.0, 0.0]),
            "reach": np.array([0.15, 0.0, -0.05]),
            "side": np.array([0.0, 0.08, 0.0]),
            "up": np.array([0.0, 0.0, 0.08]),
            "down": np.array([0.0, 0.0, -0.08]),
        }

        s = self.solver
        snapshot = JointStateSnapshot(self.state)

        for case_name, delta in test_cases.items():
            with self.subTest(case=case_name):
                T_target = self.T_ready.copy()
                T_target[:3, 3] += delta

                q_sol, success, info = s.solve(
                    T_target,
                    frame="world",
                    q_prev=self.q_ready,
                    joint_state=snapshot,
                )

                self.assertTrue(success, f"工况 {case_name} 必须成功解出，原因: {info.get('failure_reason')}")
                self.assertIsNotNone(q_sol, f"工况 {case_name} 解向量不可为 None")

                # 正向运动学核验位置误差
                pin.forwardKinematics(s.pin_model, s.pin_data, q_sol)
                pin.updateFramePlacements(s.pin_model, s.pin_data)
                T_fk = s.pin_data.oMf[s.pin_ee_frame_id].homogeneous
                pos_err = np.linalg.norm(T_fk[:3, 3] - T_target[:3, 3])
                self.assertLess(pos_err, 0.001, f"工况 {case_name} 位置误差必须 < 1mm，实际 {pos_err*1000:.3f}mm")

                # 验证 9 关节全部在 Notion 限位内
                for j_idx, (lo_deg, hi_deg) in enumerate(self.notion_limits):
                    val_deg = math.degrees(q_sol[j_idx])
                    j_name = s.active_joint_names[j_idx]
                    self.assertGreaterEqual(
                        val_deg, lo_deg - 0.01,
                        f"工况 {case_name} 关节 {j_name} 越下界: {val_deg:.2f}° < {lo_deg}°"
                    )
                    self.assertLessEqual(
                        val_deg, hi_deg + 0.01,
                        f"工况 {case_name} 关节 {j_name} 越上界: {val_deg:.2f}° > {hi_deg}°"
                    )

                # 验证臂角自运动偏差约束有效
                psi = s.chain.psi(q_sol)
                phi = (psi - s.psi_ready + math.pi) % (2 * math.pi) - math.pi
                self.assertLess(
                    abs(math.degrees(phi)), 30.0,
                    f"工况 {case_name} 臂角偏离就绪态角度过大: {math.degrees(phi):.2f}°"
                )

    def test_223_continuation_smoothness(self):
        """测试 2-2-3 构型在连续路点跟踪中的平滑性（无关节突变）"""
        s = self.solver
        q_prev = self.q_ready.copy()
        snapshot = JointStateSnapshot(self.state)

        # 连续微小位移步进
        for step in range(5):
            delta = np.array([0.01 * (step + 1), 0.0, 0.0])
            T_target = self.T_ready.copy()
            T_target[:3, 3] += delta

            q_sol, success, info = s.solve(
                T_target,
                frame="world",
                q_prev=q_prev,
                joint_state=snapshot,
            )
            self.assertTrue(success)
            step_jump = np.max(np.abs(q_sol - q_prev))
            self.assertLess(step_jump, math.radians(20.0), f"步进 {step} 产生异常关节突变: {math.degrees(step_jump):.2f}°")
            q_prev = q_sol.copy()


if __name__ == "__main__":
    unittest.main()

