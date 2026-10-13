"""
机器人描述档案与模型通用性回归测试
================================================================================

本文件守护的是一类**不会报错**的缺陷: 档案与 URDF 不匹配时, 代码照常运行、
照常输出数值, 只是算的是另一个关节 / 另一条链。

Pinocchio 的 ``getJointId()`` 对不存在的名称返回 ``model.njoints`` 而非抛异常;
减一之后恰好落在 ``names`` 的合法区间内, 于是错误索引被静默接受。换模型时若
缺少这层校验, 求解器会对着一条不存在的链求解, 产出一组"看起来合理"实则完全
错误的关节角 —— 面板上一切正常。

覆盖:
    1. 档案校验: 缺失的关节 / Frame / 连杆必须报错, 且报错信息列出实际名称
    2. 关节限位: 唯一真源为 URDF 的 <limit>, 覆盖表仅在 URDF 缺省时生效
    3. 支撑多边形: 由足部几何推导, 非写死
    4. 碰撞对分类: 由档案驱动, 非英文子串猜测
"""

import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from core.kinematics import robot_profile as rp          # noqa: E402
from core.kinematics.humanoid_model import HumanoidKinematicsModel  # noqa: E402
from core.collision.humanoid_collision import HumanoidCollisionChecker  # noqa: E402

URDF = os.path.join(REPO_ROOT, "resources/humanoid/humanoid_29dof.urdf")
SRDF = os.path.join(REPO_ROOT, "resources/humanoid/humanoid_29dof.srdf")


@pytest.fixture(scope="module")
def kin():
    return HumanoidKinematicsModel(urdf_path=URDF, srdf_path=SRDF)


# ==============================================================================
# 1. 档案校验 —— 静默错位必须变成显式失败
# ==============================================================================

def test_missing_joint_raises_with_candidates(kin):
    """档案声明了 URDF 中不存在的关节时必须报错, 且提示模型实际有哪些关节。"""
    bad = rp.RobotProfile.from_dict({
        "name": "broken",
        "waist_joint_names": ["waist_yaw_joint"],
        "left_arm_joint_names": ["根本不存在的手臂关节_xyz"],
        "left_ee_frame": "left_wrist_yaw_link",
        "support_links": ["left_ankle_roll_link"],
    })
    with pytest.raises(rp.RobotProfileError) as exc:
        rp.resolve(kin.model, bad)

    msg = str(exc.value)
    assert "根本不存在的手臂关节_xyz" in msg, "报错信息须点名缺失的关节"
    assert "实际名称示例" in msg, "报错信息须列出模型中实际存在的名称, 便于排查"


def test_missing_ee_frame_raises(kin):
    """末端 Frame 不存在同样必须报错 —— 它一旦错位, FK 目标坐标系就指错了连杆。"""
    bad = rp.RobotProfile.from_dict({
        "waist_joint_names": ["waist_yaw_joint"],
        "left_arm_joint_names": ["left_elbow_joint"],
        "left_ee_frame": "不存在的末端_link",
        "support_links": ["left_ankle_roll_link"],
    })
    with pytest.raises(rp.RobotProfileError, match="不存在的末端_link"):
        rp.resolve(kin.model, bad)


def test_missing_support_link_raises(kin):
    """支撑连杆缺失会让平衡评估无从谈起, 必须报错而非静默退化。"""
    bad = rp.RobotProfile.from_dict({
        "waist_joint_names": ["waist_yaw_joint"],
        "left_arm_joint_names": ["left_elbow_joint"],
        "left_ee_frame": "left_wrist_yaw_link",
        "support_links": ["不存在的脚_link"],
    })
    with pytest.raises(rp.RobotProfileError, match="不存在的脚_link"):
        rp.resolve(kin.model, bad)


def test_missing_support_declaration_raises(kin):
    """未声明支撑连杆且无法推导时, 必须给出可操作的修复建议。"""
    bad = rp.RobotProfile.from_dict({
        "waist_joint_names": ["waist_yaw_joint"],
        "left_arm_joint_names": ["left_elbow_joint"],
        "left_ee_frame": "left_wrist_yaw_link",
        "support_links": [],
    })
    with pytest.raises(rp.RobotProfileError, match="support_links"):
        rp.resolve(kin.model, bad)


def test_unknown_profile_field_rejected():
    """档案里的未知字段必须报错, 否则拼错的字段名会被静默忽略。"""
    with pytest.raises(rp.RobotProfileError, match="未知字段"):
        rp.RobotProfile.from_dict({"waist_joint_names": [], "waist_jont": []})


def test_getjointid_silent_wrong_index_is_guarded(kin):
    """本组测试的存在理由: 记录 Pinocchio 的危险行为, 并确认我们绕开了它。

    getJointId 对不存在的名称返回 njoints, 减一后是合法下标 —— 若代码直接使用
    该返回值而不校验, 就会静默指向最后一个关节。本测试锁定期望的失败行为,
    防止将来有人"顺手"去掉校验。
    """
    bogus = "绝对不存在的关节_name"
    assert not kin.model.existJointName(bogus)

    raw_index = kin.model.getJointId(bogus) - 1
    # 危险: 该下标合法, 指向一个完全不相干的关节, 不会抛任何异常
    assert 0 <= raw_index < kin.model.njoints
    assert kin.model.names[raw_index] != bogus

    # 我们自己的路径必须拦住它
    with pytest.raises(rp.RobotProfileError):
        rp.resolve(kin.model, rp.RobotProfile.from_dict({
            "waist_joint_names": [bogus],
            "left_arm_joint_names": ["left_elbow_joint"],
            "left_ee_frame": "left_wrist_yaw_link",
            "support_links": ["left_ankle_roll_link"],
        }))


# ==============================================================================
# 2. 关节限位 —— 唯一真源为 URDF
# ==============================================================================

def test_limits_come_from_urdf(kin):
    """引擎取到的限位必须逐项等于 URDF 的 <limit>, 而非代码里的副本。"""
    for arm, names in (("left_arm", kin.left_arm_joint_names),
                       ("right_arm", kin.right_arm_joint_names)):
        lo, hi = kin.limits[arm]
        for i, jn in enumerate(names):
            idx = kin.model.getJointId(jn) - 1
            assert lo[i] == pytest.approx(kin.model.lowerPositionLimit[idx], abs=1e-9)
            assert hi[i] == pytest.approx(kin.model.upperPositionLimit[idx], abs=1e-9)

    wlo, whi = kin.waist_limits
    for i, jn in enumerate(kin.waist_joint_names):
        idx = kin.model.getJointId(jn) - 1
        assert wlo[i] == pytest.approx(kin.model.lowerPositionLimit[idx], abs=1e-9)
        assert whi[i] == pytest.approx(kin.model.upperPositionLimit[idx], abs=1e-9)


def test_limit_override_applies_when_urdf_unbounded(kin):
    """URDF 未给限位时 (Pinocchio 填 ±inf), 覆盖值才生效。"""
    name = "left_elbow_joint"
    idx = kin.model.getJointId(name) - 1

    # 构造一个"URDF 未声明限位"的等效模型视图
    saved_lo = kin.model.lowerPositionLimit[idx]
    saved_hi = kin.model.upperPositionLimit[idx]
    try:
        kin.model.lowerPositionLimit[idx] = -1e20
        kin.model.upperPositionLimit[idx] = 1e20

        got = rp.read_joint_limits(kin.model, [name], {name: [-0.3, 0.4]})
        assert got[0].tolist() == [-0.3, 0.4]

        # 无覆盖时退化为对称 ±pi
        fallback = rp.read_joint_limits(kin.model, [name], {})
        assert fallback[0].tolist() == pytest.approx([-np.pi, np.pi])
    finally:
        kin.model.lowerPositionLimit[idx] = saved_lo
        kin.model.upperPositionLimit[idx] = saved_hi


def test_limit_override_ignored_when_urdf_has_limit(kin):
    """URDF 已声明限位时, 覆盖值不得生效 —— 否则又会退回双源分叉的老问题。"""
    name = "left_elbow_joint"
    got = rp.read_joint_limits(kin.model, [name], {name: [-0.1, 0.1]})
    idx = kin.model.getJointId(name) - 1
    assert got[0][0] == pytest.approx(kin.model.lowerPositionLimit[idx])
    assert got[0][0] != pytest.approx(-0.1)


def test_invalid_limit_override_rejected(kin):
    """覆盖值下界大于上界时必须报错, 而不是静默产生空可行域。"""
    with pytest.raises(rp.RobotProfileError, match="大于上界"):
        rp.read_joint_limits(kin.model, ["left_elbow_joint"],
                             {"left_elbow_joint": [1.0, -1.0]})


# ==============================================================================
# 3. 支撑多边形 —— 由几何推导
# ==============================================================================

def test_support_polygon_derived_from_geometry(kin):
    """支撑多边形必须来自足部碰撞几何, 且与写死的标称值不要求相同。"""
    poly = kin.support_polygon
    assert set(poly) == {"x_min", "x_max", "y_min", "y_max"}
    assert poly["x_min"] < poly["x_max"]
    assert poly["y_min"] < poly["y_max"]
    # 双足结构: 横向跨度必然显著大于单脚宽度
    assert (poly["y_max"] - poly["y_min"]) > 0.1, "双足支撑横向跨度应大于单脚宽度"
    assert "AABB" in kin.support_polygon_source


def test_support_polygon_override_wins(kin):
    """档案显式指定时, 推导结果必须让位于指定值。"""
    custom = {"x_min": -0.2, "x_max": 0.2, "y_min": -0.3, "y_max": 0.3}
    m = HumanoidKinematicsModel(
        urdf_path=URDF, srdf_path=SRDF,
        profile=rp.RobotProfile.from_dict({
            "waist_joint_names": list(kin.waist_joint_names),
            "left_arm_joint_names": list(kin.left_arm_joint_names),
            "right_arm_joint_names": list(kin.right_arm_joint_names),
            "left_ee_frame": kin.ee_frame_names["left_arm"],
            "right_ee_frame": kin.ee_frame_names["right_arm"],
            "support_links": list(kin.profile.support_links),
            "support_polygon": custom,
        }),
    )
    assert m.support_polygon == custom
    assert m.support_polygon_source == "档案显式指定"


def test_leg_joints_auto_derived(kin):
    """腿部关节应由支撑连杆的祖先链自动推出, 且覆盖完整的髋-膝-踝。"""
    legs = set(kin.profile.leg_joint_names)
    for expected in ("left_hip_pitch_joint", "left_knee_joint", "left_ankle_roll_joint",
                     "right_hip_pitch_joint", "right_knee_joint", "right_ankle_roll_joint"):
        assert expected in legs, f"腿部关节推导遗漏了 {expected}"


# ==============================================================================
# 4. 碰撞对分类 —— 由档案驱动
# ==============================================================================

def test_collision_categories_not_degenerate(kin):
    """分类必须落到真实身体部位上, 而非全体落入 other。

    旧实现按英文子串猜部位, 换成命名风格不同的模型会整体失效。此处断言
    四个安全分区都有内容, 且总量守恒。
    """
    col = HumanoidCollisionChecker(kin)
    cats = {k: len(v) for k, v in col._category_indices.items()}

    assert cats["arm_arm"] > 0, "双臂互碰分区为空"
    assert cats["arm_torso"] > 0, "臂-躯干分区为空 —— 手臂扫到髋部将无法预警"
    assert cats["arm_leg"] > 0, "臂-腿分区为空"

    total = sum(cats.values())
    assert total == len(col.pair_metadata) == len(kin.coll_model.collisionPairs), \
        "每个碰撞对必须有且只有一个分类"


def test_base_links_classified_as_torso(kin):
    """根关节直挂的骨盆连杆必须归入躯干。

    它们沿关节树上溯到根就断了。若落进 other, "臂-躯干"安全区会凭空少掉
    十几个碰撞对, 而碰撞计算照常进行 —— 大屏一切正常, 实际在漏报。
    """
    col = HumanoidCollisionChecker(kin)
    torso_pairs = [p for p in col.pair_metadata if p[2] == "arm_torso"]
    linked = {p[1] for p in torso_pairs} | {p[0] for p in torso_pairs}

    assert any("pelvis" in n for n in linked), \
        f"骨盆连杆未归入躯干, 臂-躯干分区会漏报; 当前躯干相关连杆: {sorted(linked)}"


def test_head_links_explicitly_classified(kin):
    """头部外壳无独立关节, 只能靠档案显式声明, 否则"臂-头"安全区整块消失。"""
    col = HumanoidCollisionChecker(kin)
    assert len(col._category_indices["arm_head"]) > 0, \
        "臂-头分区为空: 头部连杆未被显式声明, 已随躯干被合并"


def test_collision_zone_distances_available(kin):
    """大屏安全雷达依赖的各分区距离键必须齐全。"""
    import numpy as np
    col = HumanoidCollisionChecker(kin)
    q = pin_neutral_with(kin)
    zones = col.compute_zone_distances(q, arm="left_arm")
    for k in ("arm_torso", "arm_arm", "arm_head", "arm_leg"):
        assert k in zones, f"缺少安全分区距离键 {k}"
        assert isinstance(zones[k], float)


def pin_neutral_with(kin):
    """中立姿态的全身关节向量。"""
    import pinocchio as pin
    q = pin.neutral(kin.model)
    for arm, (lo, hi) in kin.limits.items():
        idx = kin.arm_q_indices[arm]
        for k, j in enumerate(idx):
            v = float(np.clip(0.0, lo[k], hi[k]))
            q[kin.model.joints[j + 1].idx_q] = v
    return q