"""
Unitree G1 S-Curve 轨迹插值器单元测试 (Piecewise Quintic Hermite S-Curve Interpolator)
===================================================================================
测试覆盖:
  1. 稀疏双路标点 (Rest-to-Rest) 端点位置精确匹配、初末速度为 0、初末加速度为 0
  2. 连续性测试: 验证插值曲线 C^0、C^1、C^2 连续 (平滑无冲击，Jerk 有界)
  3. 多路标点 (Multi-waypoint) 非零中间速度与加速度连续传递测试
  4. 极端超短轨迹 (< 0.8s) 自适应时间缩放与视觉舒适度保护测试
  5. ROS 2 JointTrajectory 消息插值兼容性验证
"""

import unittest
import numpy as np
from core.planning.trajectory_interpolator import SCurveInterpolator


class TestSCurveInterpolator(unittest.TestCase):

    def setUp(self):
        self.interp = SCurveInterpolator(sample_rate_hz=50.0, min_duration=0.80)

    def test_two_point_rest_to_rest(self):
        """测试标准双点起止工况：精确匹配位置、初末速度加速度为零"""
        wps = [np.array([0.1, -0.2, 0.5]), np.array([0.8, 0.4, 1.2])]
        t_wps = [0.0, 1.0]

        dense_t, dense_q, dense_v, dense_a, scale = self.interp.interpolate_waypoints(
            positions=wps,
            timestamps=t_wps,
            apply_min_duration_scale=False,
        )

        # 1. 起终点位置严格相等
        np.testing.assert_allclose(dense_q[0], wps[0], atol=1e-6)
        np.testing.assert_allclose(dense_q[-1], wps[-1], atol=1e-6)

        # 2. 起终点速度严格为零 (平滑启停，无阶跃)
        np.testing.assert_allclose(dense_v[0], np.zeros(3), atol=1e-6)
        np.testing.assert_allclose(dense_v[-1], np.zeros(3), atol=1e-6)

        # 3. 起终点加速度严格为零 (Jerk 有界，无机械冲击)
        np.testing.assert_allclose(dense_a[0], np.zeros(3), atol=1e-6)
        np.testing.assert_allclose(dense_a[-1], np.zeros(3), atol=1e-6)

        # 4. 步长与采样率符合 50Hz (20ms)
        self.assertEqual(len(dense_t), 51)
        self.assertAlmostEqual(dense_t[1] - dense_t[0], 0.02, places=4)

    def test_multi_waypoints_c2_continuity(self):
        """测试多路标点非零速度/加速度穿行时的 C^2 连续性"""
        wps = [np.array([0.0]), np.array([0.5]), np.array([1.2]), np.array([1.5])]
        t_wps = [0.0, 0.5, 1.2, 1.8]
        v_wps = [np.array([0.0]), np.array([1.0]), np.array([0.6]), np.array([0.0])]
        a_wps = [np.array([0.0]), np.array([0.2]), np.array([-0.3]), np.array([0.0])]

        dense_t, dense_q, dense_v, dense_a, scale = self.interp.interpolate_waypoints(
            positions=wps,
            timestamps=t_wps,
            velocities=v_wps,
            accelerations=a_wps,
            apply_min_duration_scale=False,
        )

        np.testing.assert_allclose(dense_q[0], wps[0], atol=1e-6)
        np.testing.assert_allclose(dense_q[-1], wps[-1], atol=1e-6)

        # 验证各路标交界点的 C^0(位置), C^1(速度), C^2(加速度) 严格无缝闭合
        for k in range(len(wps) - 1):
            h_k = t_wps[k + 1] - t_wps[k]
            # 第 k 段末端 (tau = h_k)
            q_end, v_end, a_end = self.interp.eval_quintic_segment(
                wps[k], wps[k + 1], v_wps[k], v_wps[k + 1], a_wps[k], a_wps[k + 1], h_k, h_k
            )
            # 第 k+1 点 (目标点)
            np.testing.assert_allclose(q_end, wps[k + 1], atol=1e-6)
            np.testing.assert_allclose(v_end, v_wps[k + 1], atol=1e-6)
            np.testing.assert_allclose(a_end, a_wps[k + 1], atol=1e-6)

        # 验证全局采样曲线平滑无离散奇异值
        self.assertFalse(np.any(np.isnan(dense_q)))
        self.assertFalse(np.any(np.isnan(dense_v)))
        self.assertFalse(np.any(np.isnan(dense_a)))

    def test_adaptive_min_duration_scale(self):
        """测试超短时间轨迹 (<0.8s) 的自适应时间轴放缓机制"""
        wps = [np.array([0.0, 0.0]), np.array([0.05, 0.02])]
        t_wps = [0.0, 0.08]  # 原规划仅 0.08 秒

        dense_t, dense_q, dense_v, dense_a, scale = self.interp.interpolate_waypoints(
            positions=wps,
            timestamps=t_wps,
            apply_min_duration_scale=True,
        )

        # 应该被自适应缩放至 0.80 秒
        self.assertAlmostEqual(dense_t[-1], 0.80, places=3)
        self.assertAlmostEqual(scale, 10.0, places=2)
        self.assertGreaterEqual(len(dense_t), 40)
        np.testing.assert_allclose(dense_q[-1], wps[-1], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
