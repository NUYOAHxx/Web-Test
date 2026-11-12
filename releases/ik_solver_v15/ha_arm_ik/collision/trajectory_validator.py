#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""轨迹级碰撞安全验证: 逐帧 check_collision 验证与首碰熔断。"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .collision_constraints import CollisionReport


class TrajectoryValidator:
    """轨迹级碰撞验证组件 (RobotCollisionManager 的功能块)。"""

    def validate_trajectory(
        self,
        q_sequence: Any,
        safety_margin: float = 0.0,
        stop_at_first: bool = True,
    ) -> Dict[str, Any]:
        """逐帧验证一段关节轨迹的碰撞安全性。

        Args:
            q_sequence: 关节角度序列 (shape: (N, n_joints) 或列表)。
            safety_margin: 安全裕度 [m] (默认 0.0)。
            stop_at_first: 发现第一个碰撞点即快速熔断退出。

        Returns:
            {
                "is_safe": bool,
                "first_collision_index": Optional[int],
                "colliding_pairs": List[Tuple[str, str]],
                "min_trajectory_distance": float,
                "total_frames_checked": int,
            }
        """
        frames = np.asarray(q_sequence, dtype=float)
        if frames.ndim == 1:
            frames = frames.reshape(1, -1)

        total_frames = frames.shape[0]
        min_traj_d = float("inf")
        first_col_idx: Optional[int] = None
        first_colliding_pairs: List[Tuple[str, str]] = []

        for idx in range(total_frames):
            q = frames[idx]
            rep: CollisionReport = self.check_collision(
                q,
                stop_at_first=stop_at_first,
                safety_margin=safety_margin,
            )

            if np.isfinite(rep.min_obstacle_distance):
                min_traj_d = min(min_traj_d, float(rep.min_obstacle_distance))

            if rep.is_collision:
                first_col_idx = idx
                first_colliding_pairs = list(rep.colliding_pairs)
                if stop_at_first:
                    break

        is_safe = first_col_idx is None
        return {
            "is_safe": is_safe,
            "first_collision_index": first_col_idx,
            "colliding_pairs": first_colliding_pairs,
            "min_trajectory_distance": min_traj_d,
            "total_frames_checked": (first_col_idx + 1) if (first_col_idx is not None and stop_at_first) else total_frames,
        }
