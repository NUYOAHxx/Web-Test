"""
可达性判定容差的标定测试
================================================================================

注意: 本模块守护的容差**不再用于拦截**, 仅决定界面是否给出"可能不收敛"的
提示。大屏是观测工具而非计算器 —— 用户要的是能把任意目标下发出去并观察结果,
拦截会把"不收敛"这一最值得看的现象挡在入口。

容差 = "距最近的已采样可达位姿多远就不再提示"。若它小于采样间距的尾部,
采样稀疏处会对完全够得着的点误报"可能不收敛" (提示噪声)。

实测 (6 万采样, 左臂):

    自身采样间距   中位 9.9mm / 95分位 19.3mm / 最大 40.1mm
    容差 30mm      误拒 18/3000 = 0.6%      坏角点放过 0/8
    容差 40mm      误拒  1/3000 = 0.0%      坏角点放过 0/8
    容差 50mm      误拒  0/3000 = 0.0%      坏角点放过 1/8   ← 不可接受

容差过大同样有代价: 50mm 会放过距 49mm 的明显不可达目标, 提示随之失效。
因此区间很窄, 需要测试守着。
"""

import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from core.kinematics.humanoid_model import HumanoidKinematicsModel  # noqa: E402
from core.kinematics import reachability  # noqa: E402

SAMPLES = 60000


@pytest.fixture(scope="module")
def index():
    kin = HumanoidKinematicsModel()
    return kin, reachability.build_index(kin, "left_arm", samples=SAMPLES, tolerance=0.040)


def test_tolerance_covers_sampling_gap(index):
    """容差必须不小于采样间距的最大值, 否则稀疏处会误报提示。"""
    _, idx = index
    from scipy.spatial import cKDTree
    tree = cKDTree(idx.points)

    rng = np.random.default_rng(5)
    pick = rng.choice(len(idx.points), 1500, replace=False)
    dist, _ = tree.query(idx.points[pick], k=2)
    own_gap = dist[:, 1]                      # 每个样本点到最近邻的距离
    max_gap = float(own_gap.max())

    tol = idx.tolerance
    assert tol >= max_gap * 0.95, (
        f"容差 {tol * 1000:.0f}mm 小于采样最大间距 {max_gap * 1000:.1f}mm 的 95%, "
        f"稀疏处会误报'可能不收敛'"
    )


def test_tolerance_not_so_loose(index):
    """容差也不可过大, 否则不可达目标会蒙混过关。"""
    _, idx = index
    assert idx.tolerance <= 0.045, \
        f"容差 {idx.tolerance * 1000:.0f}mm 过宽: 实测 50mm 会放过残差 52mm 的坏目标"


def test_false_warning_rate_low(index):
    """对"样本自身"的误提示率必须接近零 —— 样本点必然可达, 不该提示不收敛。"""
    _, idx = index
    from scipy.spatial import cKDTree
    tree = cKDTree(idx.points)

    rng = np.random.default_rng(5)
    pick = rng.choice(len(idx.points), 1500, replace=False)
    dist, _ = tree.query(idx.points[pick], k=2)
    own_gap = dist[:, 1]

    rejected = int((own_gap > idx.tolerance).sum())
    assert rejected / len(pick) < 0.005, \
        f"误提示率 {rejected / len(pick):.1%} 过高, 界面对可达目标也会报警"


def test_unreachable_regions_flagged(index):
    """包围盒角点 (实测残差 52~262mm) 必须全部被标记为"很可能不可达"。

    只要求标记, 不要求拦截 —— 指令照常下发, 由残差与收敛曲线呈现结果。
    """
    _, idx = index
    corners = [
        (0.041, -0.446, 0.722), (0.041, -0.446, 1.374),
        (0.041, 0.449, 0.722), (0.041, 0.449, 1.374),
        (0.505, -0.446, 0.722), (0.505, -0.446, 1.374),
        (0.505, 0.449, 0.722), (0.505, 0.449, 1.374),
    ]
    wrongly = []
    for c in corners:
        ok, gap = idx.query(c)
        if ok:
            wrongly.append((c, gap))
    assert not wrongly, f"以下明显不可达的目标未被标记 (提示将失效): {wrongly}"


def test_high_targets_reachable(index):
    """高处不是禁区 —— 防止过度收紧把上方工作空间也标成不可达。

    实测 [0.30, 0.20, 1.35] 残差 0.45mm, 属于完全可达。
    """
    _, idx = index
    ok, gap = idx.query((0.30, 0.20, 1.35))
    assert ok, f"已知可达的高处目标被标为不可达 (距最近可达位姿 {gap * 1000:.1f}mm)"


def test_query_reports_meaningful_distance(index):
    """不可达时返回的距离必须可操作 —— 界面直接展示给用户。"""
    _, idx = index
    ok, gap = idx.query((0.505, 0.449, 1.374))
    assert not ok
    assert 0.10 < gap < 0.40, f"距离 {gap:.3f}m 不在合理量级, 无法指导用户调整"