#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""碰撞检测配置: mesh 资产路径表、凸包缓存目录、机器人几何配置与阈值常量。

支持环境变量与外部 YAML/字典动态覆盖, 兼顾开箱即用与模型无关 (Model-Agnostic) 演进。
"""
from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Dict, List, Tuple

# ---------------------------------------------------------
# 1. 资产路径与包目录配置 (支持环境变量 HA_COLLISION_PACKAGE_DIRS 扩展)
# ---------------------------------------------------------
_DEFAULT_PACKAGE_DIRS = [
    "/workspaces/isaac_ros-dev/src/lam_pm_cad/lam_description",
    "/workspaces/isaac_ros-dev/src/lam_pm_cad/lam_description/cpc_tesollo",
]

_ENV_PACKAGE_DIRS = [
    p.strip()
    for p in os.environ.get("HA_COLLISION_PACKAGE_DIRS", "").split(os.pathsep)
    if p.strip()
]

# 仅保留本地实际存在的有效路径，避免下游加载时重复判断
PACKAGE_DIRS: List[str] = [
    p for p in (_ENV_PACKAGE_DIRS + _DEFAULT_PACKAGE_DIRS) if os.path.isdir(p)
]

# 凸包缓存: 全部凸包共用一个目录, 按 "连杆stem_源mesh哈希" 命名,
# 源 mesh 变更自动失效重建, 首次构建 ~7ms/连杆, 之后毫秒级加载。
HULL_CACHE_DIR: Path = Path(
    os.environ.get(
        "HA_HULL_CACHE_DIR",
        str(Path(__file__).resolve().parent.parent / "assets" / "hull"),
    )
)
HULL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# 2. 凸包分解、连杆白名单与胶囊包络配置
# ---------------------------------------------------------
# 臂-身体碰撞对靠这三个凸包补齐。
BODY_HULL_LINKS: Tuple[str, ...] = ("Hip_Link1", "Body_Link", "Head_Link1")

# mesh 档: collision mesh 逐连杆替换为凸包 (前缀匹配)。
# 凸包 ⊇ 源 mesh, 判碰保守不漏检。灵巧手 ( *l_dg* ) 保留原始 BVH (手指间隙精度优先)。
HULL_DECOMPOSE_LINK_PREFIXES: Tuple[str, ...] = (
    "LHand_Link",
    "RHand_Link",
    "Body_Link",
    "Hip_Link1",
    "Head_Link1",
    "base_Base",
)

# 双臂胶囊包络段表: (登记连杆名, 物理挂载帧名, 胶囊半径[m], 挂载帧系下的骨向量[m])
# 注: 登记名与挂载帧解耦, 挂载帧决定 FK 与雅可比作用的 parentJoint。
DEFAULT_CAPSULE_SEGMENTS: Dict[str, List[Tuple[str, str, float, Tuple[float, float, float]]]] = {
    "left": [
        ("LHand_Link2", "LHand_Link2", 0.0640, (0.0, 0.2378, 0.0)),
        ("LHand_Link4", "LHand_Link4", 0.0640, (0.0, 0.1719, 0.0)),
        ("LHand_Link6", "LHand_Link6", 0.0559, (0.084, 0.0, 0.0)),
        ("LHand_Link7", "LHand_Link7", 0.0422, (0.16, 0.0, 0.0)),
    ],
    "right": [
        ("RHand_Link2", "RHand_Link2", 0.0640, (0.0, 0.2378, 0.0)),
        ("RHand_Link4", "RHand_Link4", 0.0640, (0.0, 0.1719, 0.0)),
        ("RHand_Link6", "RHand_Link6", 0.0559, (0.084, 0.0, 0.0)),
        ("RHand_Link7", "RHand_Link7", 0.0422, (0.16, 0.0, 0.0)),
    ],
}

# ---------------------------------------------------------
# 3. 算法开关与通用阈值
# ---------------------------------------------------------
# 是否启用包围球粗筛 (Broad-phase): 半径口径已统一修复为 aabb_radius + ||aabb_center||, 可安全开启
BROADPHASE_ENABLED: bool = True


@dataclass(frozen=True)
class CollisionConfig:
    """结构化碰撞配置对象 (支持向组合架构与多机器人配置驱动演进)。"""

    package_dirs: Tuple[str, ...] = field(default_factory=lambda: tuple(PACKAGE_DIRS))
    hull_cache_dir: Path = HULL_CACHE_DIR
    body_hull_links: Tuple[str, ...] = BODY_HULL_LINKS
    hull_decompose_prefixes: Tuple[str, ...] = HULL_DECOMPOSE_LINK_PREFIXES
    broadphase_enabled: bool = BROADPHASE_ENABLED
    default_influence_distance: float = 0.08
    default_safety_margin: float = 0.0
    default_repulsion_gain: float = 0.25


@dataclass
class CollisionAvoidanceParams:
    """IK 解算请求级与环境上下文防碰撞参数规范。

    分层职责:
    - 物理硬约束 (Safety Constraint): safety_margin [m]
    - 算法软势场 (Potential Field): influence_distance [m], repulsion_gain
    - 外层评价打分 (Scoring Weight): w_collision
    - 距离测量模式: compute_exact_min_distance
    """

    safety_margin: float = 0.002               # 2mm 硬安全门槛 (验收判定)
    influence_distance: float = 0.09           # 90mm 软排斥势场激活距离
    repulsion_gain: float = 0.35               # 零空间排斥力矩增益
    w_collision: float = 500.0                 # 候选解打分惩罚权重
    compute_exact_min_distance: bool = False   # 是否强制提取全局精确最小距离

    def clone(self) -> "CollisionAvoidanceParams":
        """深拷贝当前参数对象。"""
        return CollisionAvoidanceParams(
            safety_margin=self.safety_margin,
            influence_distance=self.influence_distance,
            repulsion_gain=self.repulsion_gain,
            w_collision=self.w_collision,
            compute_exact_min_distance=self.compute_exact_min_distance,
        )

    def with_overrides(
        self,
        safety_margin: Optional[float] = None,
        influence_distance: Optional[float] = None,
        repulsion_gain: Optional[float] = None,
        w_collision: Optional[float] = None,
        compute_exact_min_distance: Optional[bool] = None,
    ) -> "CollisionAvoidanceParams":
        """返回注入覆盖值后的新参数对象 (不可变链式更新)。"""
        return CollisionAvoidanceParams(
            safety_margin=float(safety_margin) if safety_margin is not None else self.safety_margin,
            influence_distance=float(influence_distance) if influence_distance is not None else self.influence_distance,
            repulsion_gain=float(repulsion_gain) if repulsion_gain is not None else self.repulsion_gain,
            w_collision=float(w_collision) if w_collision is not None else self.w_collision,
            compute_exact_min_distance=bool(compute_exact_min_distance) if compute_exact_min_distance is not None else self.compute_exact_min_distance,
        )

    @classmethod
    def from_any(
        cls,
        obj: Any = None,
        **kwargs,
    ) -> "CollisionAvoidanceParams":
        """从任意来源 (dict / dataclass / None) 解析构建并允许 kwargs 覆盖。"""
        if obj is None:
            base = cls()
        elif isinstance(obj, CollisionAvoidanceParams):
            base = obj.clone()
        elif isinstance(obj, dict):
            base = cls(
                safety_margin=float(obj.get("collision_safety_margin", obj.get("safety_margin", 0.002))),
                influence_distance=float(obj.get("collision_influence_distance", obj.get("influence_distance", 0.09))),
                repulsion_gain=float(obj.get("collision_repulsion_gain", obj.get("repulsion_gain", 0.35))),
                w_collision=float(obj.get("w_collision", 500.0)),
                compute_exact_min_distance=bool(obj.get("compute_exact_min_distance", False)),
            )
        else:
            base = cls()

        s_margin = kwargs.get("collision_safety_margin", kwargs.get("safety_margin"))
        s_inf = kwargs.get("collision_influence_distance", kwargs.get("influence_distance"))
        s_gain = kwargs.get("collision_repulsion_gain", kwargs.get("repulsion_gain"))
        s_w = kwargs.get("w_collision")
        s_exact = kwargs.get("compute_exact_min_distance")

        return base.with_overrides(
            safety_margin=s_margin,
            influence_distance=s_inf,
            repulsion_gain=s_gain,
            w_collision=s_w,
            compute_exact_min_distance=s_exact,
        )
