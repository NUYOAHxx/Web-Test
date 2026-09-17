#!/usr/bin/env python3
"""
================================================================================
Unitree G1 机械臂逆运动学与碰撞安全系统 (Unified Facade)
================================================================================

本文件为统一对外门面接口 (Facade Pattern)，向后完全兼容已有所有测试与部署脚本。
底层模块已完全解耦为独立专业的工程架构：
- 运动学与动力学引擎：deploy.kinematics.g1_model.G1KinematicsModel
- 几何碰撞安全引擎  ：deploy.collision.g1_collision.G1CollisionChecker
- 数值优化求解引擎  ：deploy.solver.g1_hybrid_ik.G1HybridIKSolver
"""

from deploy.kinematics.g1_model import (
    G1KinematicsModel,
    G1_JOINT_LIMITS,
    G1_WAIST_LIMITS,
    G1_READY_POSE,
)
from deploy.collision.g1_collision import G1CollisionChecker
from deploy.solver.g1_hybrid_ik import G1HybridIKSolver

__all__ = [
    "G1HybridIKSolver",
    "G1KinematicsModel",
    "G1CollisionChecker",
    "G1_JOINT_LIMITS",
    "G1_WAIST_LIMITS",
    "G1_READY_POSE",
]
