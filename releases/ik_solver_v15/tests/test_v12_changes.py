"""merged-v12（基于 v9）的回归测试：右臂 q_ready 在 ARM_CONFIGS 里直接写数值（左臂的镜像）。

覆盖：
  - ARM_CONFIGS：左臂 q_ready 不变；右臂 = (-1, +1, -1, +1, -1, +1, -1) ⊙ 左臂（奇数关节反号、偶数同号），
    没有 -0.0；左右条目的键相同（robot_model 不做镜像推导）；
  - 合成双臂 URDF（右臂 = 左臂关于躯干 x-z 面的镜像，关节轴向量与左臂相同、限位镜像；轴序 z,y,x,y,x,y,z，
    正好对应上面的符号）：任意 q 下 FK_R(s ⊙ q) = 镜像(FK_L(q))；用 ARM_CONFIGS 里的两组数值时，q_ready 处
    右臂 TCP 位姿 = 左臂的镜像（7 轴 / 9 轴），psi_ready 互为相反数，腰部舒适窗相同；
  - 求解器：右臂 q_ready 取 ARM_CONFIGS 的数值（9 轴链前面补两个腰关节 0）；
  - 镜像目标下，近处一次性求解与路点跟踪的右臂解 = 左臂解的镜像（1e-6，失败码与负荷峰值相同）。
    扰动种子的噪声左右相同、没有镜像，远处目标不要求逐位镜像（见 bench/v12/mirror_bench.py）。
"""
import copy
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from tests import stubs
stubs.install()
from ha_arm_ik import solver as S  # noqa: E402
import ha_arm_ik.ik_base as RM  # noqa: E402
import ha_arm_ik.solver_dls as D  # noqa: E402
from ha_arm_ik.solver_dls import JointStateSnapshot  # noqa: E402



def _arm_configs_from_file():
    """重新执行 robot_model.py 取 ARM_CONFIGS：其他测试会改写 RM.ARM_CONFIGS["left"]（tests/fixtures.make_solver），
    这里要核对的是文件里写的数值。"""
    spec = importlib.util.spec_from_file_location("_robot_model_v12_file", RM.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return copy.deepcopy(module.ARM_CONFIGS)


ARM_CONFIGS_FILE = _arm_configs_from_file()
SIGNS = np.array([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0])  # 奇数关节反号、偶数关节同号
S9 = np.r_[1.0, -1.0, SIGNS]  # 9 轴链：腰俯仰（轴垂直于镜像面）同号，腰偏航（轴在镜像面内）反号
Q_READY_LEFT_V9 = np.array([-0.523599, -0.785398, 0.0, -0.698132, 0.0, 1.570796, 0.0])

# 合成左臂（躯干 +y 侧）：关节轴、原点、rpy、质量、限位（None = continuous）
ARM_AXIS = [(0, 0, 1), (0, 1, 0), (1, 0, 0), (0, 1, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
ARM_XYZ = [(0, .2, .18), (.1, .02, 0), (.25, 0, -.01), (.3, .01, 0), (.25, 0, 0), (.15, 0, .01), (.08, 0, 0)]
ARM_RPY = [(.3, 0, 0), (0, 0, 0), (0, 0, .1), (0, 0, 0), (0, .05, 0), (0, 0, 0), (0, 0, 0)]
ARM_MASS = [1.5, 2, 1.5, 1.2, .8, .6, .3]
ARM_LIMITS = [(-2.6, 2.9), (-2.2, 1.4), None, (-2.5, .3), (-2.8, 2.7), (-1.0, 2.6), (-2.9, 2.8)]
M = np.diag([1.0, -1.0, 1.0])  # 矢状面（躯干 x-z 面）镜像
ALL_JOINTS = (["Hip_Link1", "Body_Link"] + [f"LHand_Link{i}" for i in range(1, 8)]
              + [f"RHand_Link{i}" for i in range(1, 8)])
SOLVER_CFG = {"load_allowable_efforts": {n: 300.0 for n in ALL_JOINTS},
              "load_massless_links": ["left_tcp", "right_tcp"], "load_budget_s": 60.0,
              "load_state_max_age_s": 600.0, "load_max_seeds": 6, "dls_max_iter": 300}


def _inertial(mass, xyz):
    return (f'<inertial><origin xyz="{" ".join(map(str, xyz))}" rpy="0 0 0"/><mass value="{mass}"/>'
            '<inertia ixx=".02" iyy=".02" izz=".02" ixy="0" ixz="0" iyz="0"/></inertial>')


def two_arm_xml():
    """躯干两侧各一条 7 轴臂：右臂 = 左臂关于躯干 x-z 面的镜像，关节轴向量与左臂相同，限位按 SIGNS 镜像。"""
    parts = ['<robot name="synthetic_two_arm">', f'<link name="base">{_inertial(12, (0, 0, 0))}</link>',
             f'<link name="hip">{_inertial(3, (.06, 0, 0))}</link>',
             f'<link name="torso">{_inertial(8, (.06, 0, 0))}</link>',
             '<joint name="Hip_Link1" type="revolute"><parent link="base"/><child link="hip"/>'
             '<origin xyz="0 0 .3"/><axis xyz="0 1 0"/><limit lower="-1.0" upper="1.5" effort="300" velocity="2"/></joint>',
             '<joint name="Body_Link" type="revolute"><parent link="hip"/><child link="torso"/>'
             '<origin xyz="0 0 .25"/><axis xyz="0 0 1"/><limit lower="-2.5" upper="2.5" effort="300" velocity="2"/></joint>']
    for side in ("L", "R"):
        parent = "torso"
        for i in range(7):
            xyz, rpy, lim, com = list(ARM_XYZ[i]), list(ARM_RPY[i]), ARM_LIMITS[i], [.06, .01, 0]
            if side == "R":
                xyz[1], rpy[0], rpy[2], com[1] = -xyz[1], -rpy[0], -rpy[2], -com[1]
                if lim is not None and SIGNS[i] < 0:
                    lim = (-lim[1], -lim[0])
            child = f"{side.lower()}arm{i + 1}"
            kind = "continuous" if lim is None else "revolute"
            limit = ('<limit effort="300" velocity="2"/>' if lim is None
                     else f'<limit lower="{lim[0]}" upper="{lim[1]}" effort="300" velocity="2"/>')
            parts.append(f'<link name="{child}">{_inertial(ARM_MASS[i], com)}</link>')
            parts.append(f'<joint name="{side}Hand_Link{i + 1}" type="{kind}"><parent link="{parent}"/>'
                         f'<child link="{child}"/><origin xyz="{" ".join(map(str, xyz))}" '
                         f'rpy="{" ".join(map(str, rpy))}"/><axis xyz="{" ".join(map(str, ARM_AXIS[i]))}"/>'
                         f'{limit}</joint>')
            parent = child
        tcp_rpy = (0, .2, .1) if side == "L" else (0, .2, -.1)
        name = "left_tcp" if side == "L" else "right_tcp"
        parts += [f'<link name="{name}"/>',
                  f'<joint name="{name}_fixed" type="fixed"><parent link="{parent}"/><child link="{name}"/>'
                  f'<origin xyz=".12 0 0" rpy="{" ".join(map(str, tcp_rpy))}"/></joint>']
    parts.append("</robot>")
    return "".join(parts)


def arm_configs(q_ready_left=None, q_ready_right=None):
    """合成 URDF 的连杆名 + q_ready 数值（默认取 robot_model.py 文件里写的值）。"""
    q_l = ARM_CONFIGS_FILE["left"]["q_ready"] if q_ready_left is None else q_ready_left
    q_r = ARM_CONFIGS_FILE["right"]["q_ready"] if q_ready_right is None else q_ready_right
    common = {"base_link": "torso", "mount_xyz": np.zeros(3), "mount_rpy": np.zeros(3)}
    return {"left": dict(common, ee_link="left_tcp", s_joint="LHand_Link2", e_joint="LHand_Link4",
                         w_joint="LHand_Link6", q_ready=np.array(q_l, dtype=float)),
            "right": dict(common, ee_link="right_tcp", s_joint="RHand_Link2", e_joint="RHand_Link4",
                          w_joint="RHand_Link6", q_ready=np.array(q_r, dtype=float))}


def mirror_state(state):
    """左臂测量状态 -> 镜像后的全身状态（腰偏航反号，左右臂互换并乘 SIGNS）。"""
    out = {"Hip_Link1": state["Hip_Link1"], "Body_Link": -state["Body_Link"]}
    for i in range(1, 8):
        out[f"RHand_Link{i}"] = SIGNS[i - 1] * state[f"LHand_Link{i}"]
        out[f"LHand_Link{i}"] = SIGNS[i - 1] * state[f"RHand_Link{i}"]
    return out


class TwoArmCase(unittest.TestCase):
    """每个测试临时替换 ARM_CONFIGS（结束后恢复），URDF 写在临时目录。"""

    def setUp(self):
        self._saved = {k: copy.deepcopy(v) for k, v in RM.ARM_CONFIGS.items()}
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "two_arm.urdf"
        self.path.write_text(two_arm_xml())
        configs = arm_configs()
        RM.ARM_CONFIGS.clear()
        RM.ARM_CONFIGS.update(configs)

    def tearDown(self):
        RM.ARM_CONFIGS.clear()
        RM.ARM_CONFIGS.update(self._saved)

    def make(self, side, include_waist=True, config=None):
        cfg = dict(SOLVER_CFG)
        cfg.update(config or {})
        return S.HAArmIKSolver(self.path, side, include_waist=include_waist, verbose=False, config=cfg)

    def pair(self):
        left, right = self.make("left"), self.make("right")
        state_l = {n: 0.0 for n in ALL_JOINTS}
        state_l.update(dict(zip(left.active_joint_names, left.q_ready)))
        return left, right, state_l, mirror_state(state_l)

    @staticmethod
    def tcp(s, state, q):
        from tests.fixtures import root_target
        return root_target(s, state, q)

    @staticmethod
    def solve(s, state, T, q_prev):
        q, _, _ = s.solve(T, frame="world", q_prev=q_prev, joint_state=JointStateSnapshot(state))
        return None if q is None else np.asarray(q, dtype=float), dict(s.last_load_diagnostics)

    def check_mirrored(self, right, ql, qr, dl, dr):
        self.assertEqual(ql is None, qr is None)
        self.assertEqual(dl.get("code"), dr.get("code"))
        if ql is None:
            return
        diff = D.joint_difference(qr, S9 * ql, right.multi_turn_mask)
        self.assertLess(float(np.max(np.abs(diff))), 1e-6)
        self.assertAlmostEqual(dl.get("peak_ratio", 0.0), dr.get("peak_ratio", 0.0), places=9)


class ArmConfigValuesTest(unittest.TestCase):
    def test_q_ready_values(self):
        q_l = ARM_CONFIGS_FILE["left"]["q_ready"]
        q_r = ARM_CONFIGS_FILE["right"]["q_ready"]
        np.testing.assert_array_equal(q_l, Q_READY_LEFT_V9)  # 左臂不变（-30° 起）
        np.testing.assert_array_equal(q_r, [0.523599, -0.785398, 0.0, -0.698132, 0.0, 1.570796, 0.0])
        np.testing.assert_array_equal(q_r, SIGNS * q_l + 0.0)
        for q in (q_l, q_r):
            self.assertEqual(q.dtype, np.float64)
            self.assertEqual(q.shape, (7,))
        self.assertFalse(np.any(np.signbit(q_r) & (q_r == 0)), "right q_ready has -0.0")

    def test_right_entry_has_same_keys_as_left(self):
        self.assertEqual(set(ARM_CONFIGS_FILE["right"]), set(ARM_CONFIGS_FILE["left"]))


class MirroredReadyPoseTest(TwoArmCase):
    def test_fixture_fk_is_mirrored_for_any_q(self):
        left, right = self.make("left", include_waist=False), self.make("right", include_waist=False)
        rng = np.random.default_rng(5)
        for _ in range(20):
            q = rng.uniform(-2.0, 2.0, size=7)
            Tl, Tr = left.chain.fk(q), right.chain.fk(SIGNS * q)
            np.testing.assert_allclose(Tr[:3, 3], M @ Tl[:3, 3], atol=1e-12)
            np.testing.assert_allclose(Tr[:3, :3], M @ Tl[:3, :3] @ M, atol=1e-12)

    def test_ready_pose_is_mirrored(self):
        for include_waist in (False, True):
            with self.subTest(include_waist=include_waist):
                left = self.make("left", include_waist=include_waist)
                right = self.make("right", include_waist=include_waist)
                q_r = ARM_CONFIGS_FILE["right"]["q_ready"]
                np.testing.assert_array_equal(right.q_ready, np.r_[0.0, 0.0, q_r] if include_waist else q_r)
                Tl, Tr = left.chain.fk(left.q_ready), right.chain.fk(right.q_ready)
                np.testing.assert_allclose(Tr[:3, 3], M @ Tl[:3, 3], atol=1e-12)
                np.testing.assert_allclose(Tr[:3, :3], M @ Tl[:3, :3] @ M, atol=1e-12)
                self.assertTrue(np.isfinite(left.psi_ready))
                self.assertAlmostEqual(right.psi_ready, -left.psi_ready, places=12)

    def test_left_values_for_right_arm_are_not_mirrored(self):
        """v9 原写法（右臂取左臂数值）：q_ready 处右臂 TCP 不在左臂 TCP 的镜像位置。"""
        RM.ARM_CONFIGS.update(arm_configs(q_ready_right=Q_READY_LEFT_V9))
        left, right = self.make("left"), self.make("right")
        Tl, Tr = left.chain.fk(left.q_ready), right.chain.fk(right.q_ready)
        self.assertGreater(float(np.linalg.norm(Tr[:3, 3] - M @ Tl[:3, 3])), 0.3)

    def test_waist_comfort_window_is_the_same(self):
        left, right = self.make("left"), self.make("right")
        self.assertIsNotNone(left.waist_comfort_lower)
        # 舒适窗只收窄腰关节（前两个），臂关节保持各自限位
        np.testing.assert_allclose(right.waist_comfort_lower[:2], left.waist_comfort_lower[:2], atol=1e-9)
        np.testing.assert_allclose(right.waist_comfort_upper[:2], left.waist_comfort_upper[:2], atol=1e-9)


class MirrorEquivarianceTest(TwoArmCase):
    def test_near_one_shot_targets(self):
        left, right, state_l, state_r = self.pair()
        rng = np.random.default_rng(23)
        solved = 0
        for k in range(6):
            qt = np.clip(left.q_ready + rng.uniform(-0.6, 0.6, size=9), left.lower + 0.05, left.upper - 0.05)
            Tl, Tr = self.tcp(left, state_l, qt), self.tcp(right, state_r, S9 * qt)
            np.testing.assert_allclose(Tr[:3, 3], M @ Tl[:3, 3], atol=1e-12)
            ql, dl = self.solve(left, state_l, Tl, left.q_ready)
            qr, dr = self.solve(right, state_r, Tr, right.q_ready)
            with self.subTest(target=k):
                self.check_mirrored(right, ql, qr, dl, dr)
            solved += ql is not None
        self.assertGreaterEqual(solved, 5)

    def test_tracking(self):
        left, right, state_l, state_r = self.pair()
        rng = np.random.default_rng(3)
        amp = rng.uniform(0.2, 0.5, size=9)
        amp[:2] = [0.5, 0.8]
        ph = rng.uniform(0, 2 * np.pi, size=9)
        w = rng.uniform(0.5, 1.0, size=9)
        ql_prev, qr_prev = left.q_ready.copy(), right.q_ready.copy()
        for t in np.linspace(0, 6.0, 31)[1:]:
            qt = np.clip(left.q_ready + amp * (np.sin(w * t + ph) - np.sin(ph)), left.lower + 0.05, left.upper - 0.05)
            ql, dl = self.solve(left, state_l, self.tcp(left, state_l, qt), ql_prev)
            qr, dr = self.solve(right, state_r, self.tcp(right, state_r, S9 * qt), qr_prev)
            with self.subTest(t=float(t)):
                self.assertIsNotNone(ql)
                self.check_mirrored(right, ql, qr, dl, dr)
            ql_prev, qr_prev = ql, qr


if __name__ == "__main__":
    unittest.main()
