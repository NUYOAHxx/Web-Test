#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""碰撞检测配置: mesh 资产路径表、凸包缓存目录与阈值常量。

纯配置模块, 无行为逻辑——所有跨模块共享的路径与标量参数集中此处,
供 robot_collision / environment_collision 等模块引用。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

# ---------------------------------------------------------
# 1. 资产路径与包目录配置
# ---------------------------------------------------------
# 加载时按 os.path.isdir 过滤, 目录缺失时该条目自动失效。
# 注意: 条目必须是包目录的父目录 (URDF 的 package://<pkg>/meshes/... 按
# "package_dir / pkg / 余下路径" 拼接解析), 不是 meshes 目录本身。
_RAW_PACKAGE_DIRS = [
    "/workspaces/isaac_ros-dev/src/lam_pm_cad/lam_description",
    "/workspaces/isaac_ros-dev/src/lam_pm_cad/lam_description/cpc_tesollo",
]

# 仅保留本地实际存在的有效路径，避免下游加载时重复判断
PACKAGE_DIRS: List[str] = [p for p in _RAW_PACKAGE_DIRS if os.path.isdir(p)]

# 凸包缓存: 全部凸包共用一个目录, 按 "连杆stem_源mesh哈希" 命名,
# 源 mesh 变更自动失效重建, 首次构建 ~7ms/连杆, 之后毫秒级加载。
HULL_CACHE_DIR: Path = Path(__file__).resolve().parent.parent / "assets" / "hull"
HULL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# 2. 凸包分解与连杆白名单配置
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

# ---------------------------------------------------------
# 3. 算法开关与通用阈值
# ---------------------------------------------------------
# 是否启用包围球粗筛 (Broad-phase)
BROADPHASE_ENABLED: bool = False