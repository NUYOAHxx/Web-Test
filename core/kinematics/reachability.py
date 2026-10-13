"""
可达工作空间索引 (Reachability Index)
================================================================================

为什么不能用包围盒
--------------------------------------------------------------------------------
末端可达集对 9-DoF 人形臂是**非凸**的。一个轴对齐包围盒只是它的超集, 盒子
的角点几乎必然落在可达集之外 —— 而用户正是会去拖角点。

实测 (左臂, 含自碰撞剔除):

    包围盒 8 个角点 → 收敛 0/8, 残差 52 ~ 262 mm
    旧版写死范围的 8 个角点 → 收敛 2/8, 残差最高 574 mm

也就是说"用盒子做可达性校验"这一思路本身是错的: 它必然放过一批无解目标,
表现正是"校验通过但 IK 无法收敛"—— 收紧盒子只能推迟这个问题, 不能消除它。

本模块的做法
--------------------------------------------------------------------------------
在关节限位盒内**稠密采样**, 记录每个样本的末端位置与对应位形, 并建立 KD 树。
校验一个目标时:

1. 先用包围盒做廉价粗筛 (快速排除明显离谱的目标);
2. 再查 KD 树取最近的若干个候选位姿;
3. 逐个做自碰撞判定 —— 只要有一个无碰撞, 即判定为**可达**;
4. 否则明确拒绝, 并报出"距最近可达位姿 XX mm"。

这样校验结果回答的是真正的问题 —— *这个点够不够得着*, 而非 *它在不在某个
矩形里*。容差必须覆盖采样间距的尾部, 否则稀疏处会把可达点误判为不可达。
实测标定见 visualizer/web/server.py 的 ``REACH_INDEX_TOLERANCE_M``。

代价: 构建约 2.3 s (启动时一次); 单次查询最多 12 次碰撞判定, 约 4 ms,
只在用户下发指令时触发。
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pinocchio as pin


class ReachabilityIndex:
    """末端位置的稠密可达性索引。"""

    def __init__(
        self,
        kin,
        arm: str = "left_arm",
        samples: int = 60000,
        seed: int = 20240501,
        tolerance: float = 0.030,
        max_candidates: int = 12,
    ) -> None:
        self.kin = kin
        self.arm = arm
        self.tolerance = float(tolerance)
        self.max_candidates = int(max_candidates)

        lo = np.concatenate([kin.waist_limits[0], kin.limits[arm][0]])
        hi = np.concatenate([kin.waist_limits[1], kin.limits[arm][1]])
        names = kin.chain_9dof_joint_names[arm]
        q_idx = [kin.model.getJointId(n) - 1 for n in names]

        rng = np.random.default_rng(seed)
        pos = np.empty((samples, 3), dtype=np.float64)
        qs = np.empty((samples, len(names)), dtype=np.float64)
        q = pin.neutral(kin.model)
        fid = kin.ee_frame_ids[arm]

        for k in range(samples):
            for i, j in enumerate(q_idx):
                v = float(rng.uniform(lo[i], hi[i]))
                q[j] = v
                qs[k, i] = v
            pin.forwardKinematics(kin.model, kin.data, q)
            pin.updateFramePlacements(kin.model, kin.data)
            pos[k] = kin.data.oMf[fid].translation

        self.points: np.ndarray = pos
        self.configs: np.ndarray = qs
        self.n_samples: int = samples

        try:
            from scipy.spatial import cKDTree
            self._tree = cKDTree(pos)
        except Exception:
            # 无 scipy 时退化为暴力最近邻 —— 慢但功能等价
            self._tree = None

        # 碰撞判定器 (可缺失: 缺失时只按几何位置判定)
        self._collider = self._build_collider()

        self._p_min = pos.min(axis=0)
        self._p_max = pos.max(axis=0)

    def _build_collider(self):
        try:
            from core.collision.humanoid_collision import HumanoidCollisionChecker
            checker = HumanoidCollisionChecker(self.kin)
        except Exception:
            return None
        return lambda q_red: bool(checker.is_colliding_reduced(self.arm, q_red, is_9dof=True))

    # ── 查询 ──────────────────────────────────────────────────────

    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """可达位置包围盒, 仅用于**粗筛**, 不代表可达性。"""
        return self._p_min.copy(), self._p_max.copy()

    def query(self, pos) -> Tuple[bool, float]:
        """判断目标位置是否可达。

        :param pos: (3,) 世界坐标
        :return: ``(是否可达, 距最近可达位姿的距离(米))``
        """
        p = np.asarray(pos, dtype=np.float64).reshape(3)

        # 粗筛: 连包围盒都出不去, 无需查树
        if np.any(p < self._p_min) or np.any(p > self._p_max):
            return False, float(np.linalg.norm(np.clip(p, self._p_min, self._p_max) - p))

        k = min(self.max_candidates, self.n_samples)
        if self._tree is not None:
            dist, idx = self._tree.query(p, k=k)
        else:
            d = np.linalg.norm(self.points - p, axis=1)
            idx = np.argsort(d)[:k]
            dist = d[idx]

        dist = np.atleast_1d(dist)
        idx = np.atleast_1d(idx)
        order = np.argsort(dist)

        nearest = float(dist[order[0]])
        if nearest > self.tolerance:
            return False, nearest

        if self._collider is None:
            return True, nearest

        # 最近的若干个候选里只要有一个无碰撞, 即视为可达 ——
        # 稠密采样下最近点可能恰好落在自碰撞姿态上, 不应因此误杀
        for j in order[:k]:
            if float(dist[j]) > self.tolerance:
                break
            if not self._collider(self.configs[idx[j]]):
                return True, float(dist[j])
        return False, nearest

    def describe(self) -> Dict[str, object]:
        return {
            "samples": self.n_samples,
            "tolerance_mm": round(self.tolerance * 1000.0, 1),
            "bbox": {
                "x": [round(float(self._p_min[0]), 4), round(float(self._p_max[0]), 4)],
                "y": [round(float(self._p_min[1]), 4), round(float(self._p_max[1]), 4)],
                "z": [round(float(self._p_min[2]), 4), round(float(self._p_max[2]), 4)],
            },
            "collision_filter": self._collider is not None,
        }


def build_index(kin, arm: str, samples: int = 60000, tolerance: float = 0.030,
                seed: int = 20240501) -> ReachabilityIndex:
    """便捷构造 (异常向上抛出, 由调用方决定回退策略)。"""
    return ReachabilityIndex(kin, arm=arm, samples=samples, seed=seed, tolerance=tolerance)


def optional_index(kin, arm: str, **kw) -> Optional[ReachabilityIndex]:
    """构造失败时返回 None 而不抛异常 —— 供降级路径使用。"""
    try:
        return build_index(kin, arm, **kw)
    except Exception:
        return None