#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ha_arm_ik 碰撞检测与防碰撞扩展模块。"""
from __future__ import annotations

try:
    from .auto_mount import auto_mount, build_collision_manager, shutdown
    from .collision_world import CollisionWorld, RobotCollisionManager
except ImportError:
    auto_mount = None
    build_collision_manager = None
    shutdown = None
    CollisionWorld = None
    RobotCollisionManager = None

from .collision_config import (
    BROADPHASE_ENABLED,
    PACKAGE_DIRS,
    CollisionAvoidanceParams,
    CollisionConfig,
)

__all__ = [
    "auto_mount",
    "build_collision_manager",
    "shutdown",
    "CollisionWorld",
    "RobotCollisionManager",
    "CollisionAvoidanceParams",
    "CollisionConfig",
    "BROADPHASE_ENABLED",
    "PACKAGE_DIRS",
]
