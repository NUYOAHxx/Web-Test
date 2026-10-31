import unittest
import numpy as np
from pathlib import Path
from tempfile import TemporaryDirectory

from ha_arm_ik.ik_base import (
    IKResult,
    Status,
    FailureReason,
    failure_reason_for_status,
    make_result,
    SerialChain,
    score_candidate,
    candidate_sort_key,
    select_best_candidate,
    update_candidate_final_cost_with_torque,
)
from ha_arm_ik.single_arm_ik import HAArmIKSolver, SingleArmIKSolver
from tests.fixtures import make_solver, robot_xml


class TestAuditFixes(unittest.TestCase):
    """验证针对 9 项架构与逻辑缺陷的修复情况。"""

    def test_1_ik_result_custom_string_status_and_reason(self):
        """修复 1：非枚举字符串 status/failure_reason 不会导致 AttributeError。"""
        res = IKResult({}, status="CUSTOM_TIMEOUT", failure_reason="CUSTOM_DETAIL")
        self.assertEqual(res.status, "CUSTOM_TIMEOUT")
        self.assertEqual(res["status"], "CUSTOM_TIMEOUT")
        self.assertEqual(res.status_val, "CUSTOM_TIMEOUT")
        self.assertFalse(res.ok)
        self.assertTrue(res.failed)
        rep = repr(res)
        self.assertIn("status=CUSTOM_TIMEOUT", rep)
        self.assertIn("failure_reason=CUSTOM_DETAIL", rep)

        # 同时也验证枚举值
        res2 = IKResult({}, status=Status.SUCCESS)
        self.assertTrue(res2.ok)
        self.assertIn("status=SUCCESS", repr(res2))

    def test_2_score_candidate_decoupled_from_sew_joints(self):
        """修复 2：chain 缺失 SEW 关节时不影响末端位姿正解与残差计算。"""
        s, state = make_solver()
        chain = s.chain
        # 构造一个没有 sew_joints 的链对象
        dummy_chain = SerialChain(
            chain.urdf_path,
            chain.base_link,
            chain.ee_link,
            s_joint="NON_EXISTENT_S",
            e_joint="NON_EXISTENT_E",
            w_joint="NON_EXISTENT_W",
        )
        q = np.zeros(len(dummy_chain.active_joints))
        T_target = dummy_chain.fk(q)
        lower, upper = dummy_chain.bounds()

        info = score_candidate(
            dummy_chain,
            q,
            T_target,
            psi_ready=0.0,
            q_prev=q,
            q_ready=q,
            lower=lower,
            upper=upper,
        )
        # 即使 SEW 关节不存在，末端正解依然计算成功，残差为 0，位姿达标
        self.assertTrue(info["pose_ok"])
        self.assertLess(info["pos_err"], 1e-4)
        self.assertLess(info["ori_err"], 1e-4)
        self.assertTrue(np.isnan(info["psi"]))
        self.assertTrue(np.isnan(info["phi"]))
        self.assertTrue(np.isfinite(info["cost"]))

    def test_3_finalize_result_per_seed_none_safe(self):
        """修复 3：_finalize_result 当 per_seed=None 且未收敛时不抛出 TypeError。"""
        s, state = make_solver()
        s.load_mode = "off"
        dummy_candidate = {
            "q": np.zeros(len(s.active_joint_names)),
            "cost": 10.0,
            "cost_no_torque": 10.0,
            "pos_err": 0.05,
            "ori_err": 0.05,
            "pose_ok": False,
        }
        # 当存在候选解但未收敛 (best is not None)，传入 per_seed=None
        q, status, res = s._finalize_result(
            candidates=[dummy_candidate],
            per_seed=None,
            T_target_local=np.eye(4),
            T_target_world=np.eye(4),
            frame="local",
            orientation_mode="full",
            tool_axis=np.array([1.0, 0.0, 0.0]),
            tool_cone_half_angle=0.0,
            q_prev=s.q_ready,
            update_prev=False,
            allow_approximate=False,
        )
        self.assertIsNone(q)
        self.assertEqual(status, "NO_VALID_POSE")
        self.assertEqual(res.failure_reason, FailureReason.NOT_CONVERGED)

    def test_4_q_prev_none_no_nan_cost_and_sort_stability(self):
        """修复 4：q_prev=None 时 smooth_cost 不污染 NaN，且 candidate_sort_key 防护 NaN。"""
        s, state = make_solver()
        q = np.zeros(len(s.active_joint_names))
        T_target = s.chain.fk(q)
        lower, upper = s.chain.bounds()

        info = score_candidate(
            s.chain,
            q,
            T_target,
            psi_ready=0.0,
            q_prev=None,  # 传入 None
            q_ready=q,
            lower=lower,
            upper=upper,
        )
        self.assertTrue(np.isfinite(info["cost"]))
        self.assertEqual(info["smooth_cost"], 0.0)

        # 测试包含各种 NaN 异常字段的候选解，Timsort 不发生弱序破坏崩溃
        candidates = [
            {"pose_ok": True, "cost": float("nan"), "pos_err": float("nan")},
            {"pose_ok": True, "cost": 1.5, "pos_err": 0.001},
            {"pose_ok": True, "cost": 0.5, "pos_err": 0.002},
        ]
        candidates.sort(key=candidate_sort_key)
        self.assertEqual(candidates[0]["cost"], 0.5)
        self.assertEqual(candidates[1]["cost"], 1.5)
        self.assertTrue(np.isnan(candidates[2]["cost"]))

    def test_5_select_best_candidate_use_torque_false_ignores_torque_cost(self):
        """修复 5：use_torque=False 时按无力矩代价排序。"""
        cand_a = {
            "pose_ok": True,
            "cost_no_torque": 1.0,
            "torque_cost": 100.0,
            "torque_l2": 50.0,
            "pos_err": 0.001,
            "ori_err": 0.001,
            "margin": 0.2,
        }
        cand_b = {
            "pose_ok": True,
            "cost_no_torque": 2.0,
            "torque_cost": 0.0,
            "torque_l2": 0.0,
            "pos_err": 0.001,
            "ori_err": 0.001,
            "margin": 0.2,
        }
        update_candidate_final_cost_with_torque(cand_a)
        update_candidate_final_cost_with_torque(cand_b)
        self.assertGreater(cand_a["cost"], cand_b["cost"])

        # 当 use_torque=False 时，必须按 cost_no_torque 择优，cand_a (1.0 < 2.0) 应当获胜
        best_no_torque, _ = select_best_candidate([cand_a, cand_b], use_torque=False)
        self.assertEqual(best_no_torque["cost_no_torque"], 1.0)
        self.assertFalse(best_no_torque["selection_cost_has_torque"])

        # 当 use_torque=True 时，按总 cost 择优，cand_b 获胜
        best_with_torque, _ = select_best_candidate([cand_a, cand_b], use_torque=True)
        self.assertEqual(best_with_torque["cost_no_torque"], 2.0)
        self.assertTrue(best_with_torque["selection_cost_has_torque"])

    def test_6_serial_chain_bounds_continuous_joints(self):
        """修复 6：SerialChain.bounds() 对 continuous 关节返回 [-inf, inf]。"""
        xml, names = robot_xml(continuous=True)
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "cont_robot.urdf"
            p.write_text(xml)
            chain = SerialChain(str(p), "base", "tcp")
            lo, hi = chain.bounds()
            cont_indices = [
                i for i, j in enumerate(chain.active_joints) if j["type"] == "continuous"
            ]
            self.assertGreater(len(cont_indices), 0)
            for idx in cont_indices:
                self.assertEqual(lo[idx], -np.inf)
                self.assertEqual(hi[idx], np.inf)

    def test_7_failure_reason_for_outside_workspace(self):
        """修复 7：failure_reason_for_status 映射 OUTSIDE_WORKSPACE 为 WORKSPACE_REJECTED。"""
        reason1 = failure_reason_for_status("OUTSIDE_WORKSPACE")
        reason2 = failure_reason_for_status(Status.OUTSIDE_WORKSPACE)
        self.assertEqual(reason1, FailureReason.WORKSPACE_REJECTED)
        self.assertEqual(reason2, FailureReason.WORKSPACE_REJECTED)

        res = make_result({}, "OUTSIDE_WORKSPACE")
        self.assertEqual(res.failure_reason, FailureReason.WORKSPACE_REJECTED)
        self.assertEqual(res["failure_reason"], FailureReason.WORKSPACE_REJECTED.value)

    def test_8_solve_engine_preserves_candidate_source(self):
        """修复 8：solve_engine 保留实际解来源（如 ssik / direct_psi）。"""
        s, state = make_solver()
        s.load_mode = "off"
        cand_ssik = {
            "q": np.zeros(len(s.active_joint_names)),
            "cost": 1.0,
            "cost_no_torque": 1.0,
            "pos_err": 0.0001,
            "ori_err": 0.0001,
            "pose_ok": True,
            "source": "ssik",
        }
        cand_direct = {
            "q": np.zeros(len(s.active_joint_names)),
            "cost": 1.0,
            "cost_no_torque": 1.0,
            "pos_err": 0.0001,
            "ori_err": 0.0001,
            "pose_ok": True,
            "source": "direct_psi",
        }

        # 模拟 finalize
        _, _, res_ssik = s._finalize_result(
            candidates=[cand_ssik],
            per_seed=[],
            T_target_local=np.eye(4),
            T_target_world=np.eye(4),
            frame="local",
            orientation_mode="full",
            tool_axis=np.array([1.0, 0.0, 0.0]),
            tool_cone_half_angle=0.0,
            q_prev=s.q_ready,
            update_prev=False,
            allow_approximate=False,
        )
        self.assertEqual(res_ssik.get("solve_engine"), "ssik")

        _, _, res_direct = s._finalize_result(
            candidates=[cand_direct],
            per_seed=[],
            T_target_local=np.eye(4),
            T_target_world=np.eye(4),
            frame="local",
            orientation_mode="full",
            tool_axis=np.array([1.0, 0.0, 0.0]),
            tool_cone_half_angle=0.0,
            q_prev=s.q_ready,
            update_prev=False,
            allow_approximate=False,
        )
        self.assertEqual(res_direct.get("solve_engine"), "direct_psi")

    def test_9_urdf_non_world_root_link_mount_detected(self):
        """修复 9：URDF 根连杆非 world 时（如 pelvis/base），正确解析固定安装关节。"""
        # 构造带有 pelvis -> link_1 fixed joint 的 URDF
        custom_xml = """<robot name="pelvis_robot">
  <link name="pelvis"/>
  <link name="link_1"/>
  <link name="tcp"/>
  <joint name="pelvis_to_arm" type="fixed">
    <parent link="pelvis"/>
    <child link="link_1"/>
    <origin xyz="0.1 0.2 0.3" rpy="0.0 0.1 0.2"/>
  </joint>
  <joint name="LHand_Link1" type="revolute">
    <parent link="link_1"/>
    <child link="tcp"/>
    <origin xyz="0 0 0.5"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.0" upper="1.0" effort="50.0" velocity="1.0"/>
  </joint>
</robot>"""
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "pelvis_robot.urdf"
            p.write_text(custom_xml)
            from ha_arm_ik.ik_base import get_urdf_root_link, get_fixed_joint_origin
            root = get_urdf_root_link(str(p))
            self.assertEqual(root, "pelvis")
            origin = get_fixed_joint_origin(str(p), parent_link=root, child_link="link_1")
            self.assertIsNotNone(origin)
            xyz, rpy = origin
            np.testing.assert_allclose(xyz, [0.1, 0.2, 0.3])
            np.testing.assert_allclose(rpy, [0.0, 0.1, 0.2])


if __name__ == "__main__":
    unittest.main()
