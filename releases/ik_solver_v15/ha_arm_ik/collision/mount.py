#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""防碰撞自动挂载与生命周期管理。"""
from __future__ import annotations

import logging
from typing import Any, Dict

from .collision_world import RobotCollisionManager

logger = logging.getLogger(__name__)


def auto_mount(solver: Any, cfg: Dict[str, Any]) -> RobotCollisionManager:
    """按 solver 配置构建 manager 并挂载 MoveIt 场景感知 (失败静默降级)。"""
    if getattr(solver, "pin_model", None) is None:
        raise TypeError("collision_avoidance=True 需要 solver 提供有效 pin_model")

    # 1. 构建碰撞管理器 (mesh / capsule)
    geometry = str(cfg.get("collision_geometry", "mesh")).strip().lower()
    manager = RobotCollisionManager.from_solver(solver, geometry=geometry)

    # 2. 挂载 MoveIt 场景同步 (后台探测，失败不阻断核心求解)
    solver.scene_sync = None
    if cfg.get("collision_scene_sync", True):
        try:
            from .environment_collision import MoveItSceneSync

            solver.scene_sync = MoveItSceneSync(
                manager,
                probe_timeout_s=float(cfg.get("collision_scene_probe_timeout_s", 3.0)),
                verbose=bool(getattr(solver, "verbose", True)),
            )
        except Exception as err:
            if getattr(solver, "verbose", False):
                logger.warning("[Collision] MoveIt 场景同步未启动 (已降级): %s", err)

    return manager


def shutdown(solver: Any) -> None:
    """停止后台同步线程并释放资源 (幂等)。"""
    sync = getattr(solver, "scene_sync", None)
    if sync is not None:
        try:
            getattr(sync, "close", getattr(sync, "shutdown", lambda: None))()
        except Exception:
            pass
        solver.scene_sync = None