#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ha_arm_ik 碰撞检测与防碰撞扩展模块。"""
from __future__ import annotations

try:
    from .mount import auto_mount, shutdown
    from .collision_world import CollisionWorld, RobotCollisionManager
except ImportError:
    auto_mount = None
    shutdown = None
    CollisionWorld = None
    RobotCollisionManager = None

from .collision_config import BROADPHASE_ENABLED, PACKAGE_DIRS

__all__ = [
    "auto_mount",
    "shutdown",
    "CollisionWorld",
    "RobotCollisionManager",
    "BROADPHASE_ENABLED",
    "PACKAGE_DIRS",
]
