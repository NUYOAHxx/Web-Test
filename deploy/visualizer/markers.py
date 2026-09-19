#!/usr/bin/env python3
"""
================================================================================
Unitree G1 运动学与遥测系统标准 RViz Marker 与几何图元构造工具
================================================================================
"""

from typing import Tuple
import numpy as np
from geometry_msgs.msg import Point, PoseStamped
from visualization_msgs.msg import Marker, MarkerArray


def create_pose_stamped(
    pos: np.ndarray,
    stamp,
    frame_id: str = "world",
    orientation: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
) -> PoseStamped:
    """快速构造包含时间戳与空间坐标的 PoseStamped 消息"""
    msg = PoseStamped()
    msg.header.frame_id = frame_id
    msg.header.stamp = stamp
    msg.pose.position.x = float(pos[0])
    msg.pose.position.y = float(pos[1])
    msg.pose.position.z = float(pos[2])
    msg.pose.orientation.x = float(orientation[0])
    msg.pose.orientation.y = float(orientation[1])
    msg.pose.orientation.z = float(orientation[2])
    msg.pose.orientation.w = float(orientation[3])
    return msg


def create_sphere_marker(
    marker_id: int,
    ns: str,
    pos: np.ndarray,
    radius: float,
    rgba: Tuple[float, float, float, float],
    stamp,
    frame_id: str = "world",
) -> Marker:
    """构造 RViz 球体 Marker"""
    m = Marker()
    m.header.frame_id = frame_id
    m.header.stamp = stamp
    m.ns = ns
    m.id = marker_id
    m.type = Marker.SPHERE
    m.action = Marker.ADD
    m.pose.position.x = float(pos[0])
    m.pose.position.y = float(pos[1])
    m.pose.position.z = float(pos[2])
    m.pose.orientation.w = 1.0
    m.scale.x = float(radius * 2.0)
    m.scale.y = float(radius * 2.0)
    m.scale.z = float(radius * 2.0)
    m.color.r = float(rgba[0])
    m.color.g = float(rgba[1])
    m.color.b = float(rgba[2])
    m.color.a = float(rgba[3])
    return m


def create_line_marker(
    marker_id: int,
    ns: str,
    p1: np.ndarray,
    p2: np.ndarray,
    thickness: float,
    rgba: Tuple[float, float, float, float],
    stamp,
    frame_id: str = "world",
) -> Marker:
    """构造两点之间的直线/残差段 Marker"""
    m = Marker()
    m.header.frame_id = frame_id
    m.header.stamp = stamp
    m.ns = ns
    m.id = marker_id
    m.type = Marker.LINE_STRIP
    m.action = Marker.ADD
    m.scale.x = float(thickness)
    m.color.r = float(rgba[0])
    m.color.g = float(rgba[1])
    m.color.b = float(rgba[2])
    m.color.a = float(rgba[3])

    pt1 = Point()
    pt1.x, pt1.y, pt1.z = float(p1[0]), float(p1[1]), float(p1[2])
    pt2 = Point()
    pt2.x, pt2.y, pt2.z = float(p2[0]), float(p2[1]), float(p2[2])
    m.points.extend([pt1, pt2])
    return m


def create_arrow_marker(
    marker_id: int,
    ns: str,
    p_start: np.ndarray,
    p_end: np.ndarray,
    shaft_diameter: float,
    head_diameter: float,
    head_length: float,
    rgba: Tuple[float, float, float, float],
    stamp,
    frame_id: str = "world",
) -> Marker:
    """构造 RViz 空间矢量箭头 Marker"""
    m = Marker()
    m.header.frame_id = frame_id
    m.header.stamp = stamp
    m.ns = ns
    m.id = marker_id
    m.type = Marker.ARROW
    m.action = Marker.ADD
    m.scale.x = float(shaft_diameter)
    m.scale.y = float(head_diameter)
    m.scale.z = float(head_length)
    m.color.r = float(rgba[0])
    m.color.g = float(rgba[1])
    m.color.b = float(rgba[2])
    m.color.a = float(rgba[3])
    pt1 = Point()
    pt1.x, pt1.y, pt1.z = float(p_start[0]), float(p_start[1]), float(p_start[2])
    pt2 = Point()
    pt2.x, pt2.y, pt2.z = float(p_end[0]), float(p_end[1]), float(p_end[2])
    m.points.extend([pt1, pt2])
    return m


def create_ik_markers(
    target_pos: np.ndarray,
    actual_pos: np.ndarray,
    stamp,
    target_rot: np.ndarray = None,
    actual_rot: np.ndarray = None,
    success: bool = True,
    frame_id: str = "world",
    ns_prefix: str = "g1_ik",
) -> MarkerArray:
    """
    统一构造 IK 视觉反馈 MarkerArray：
    - Marker 0: 目标球 (Target Sphere: 成功荧光绿，失败激光红)
    - Marker 1: 实际手爪球 (Actual FK: 金黄色)
    - Marker 2: 空间残差连线 (Error Line)
    - Marker 3~5: 目标姿态 RGB 空间坐标三轴 (Red=X, Green=Y, Blue=Z)
    - Marker 6~8: 实际手爪 RGB 空间坐标三轴
    """
    markers = MarkerArray()

    # 1. 目标球
    tgt_color = (0.0, 1.0, 0.53, 0.85) if success else (0.95, 0.15, 0.15, 0.85)
    markers.markers.append(
        create_sphere_marker(
            marker_id=0,
            ns=f"{ns_prefix}_target",
            pos=target_pos,
            radius=0.022,
            rgba=tgt_color,
            stamp=stamp,
            frame_id=frame_id,
        )
    )

    # 2. 实际球
    markers.markers.append(
        create_sphere_marker(
            marker_id=1,
            ns=f"{ns_prefix}_actual",
            pos=actual_pos,
            radius=0.018,
            rgba=(1.0, 0.75, 0.0, 0.90),
            stamp=stamp,
            frame_id=frame_id,
        )
    )

    # 3. 残差线
    markers.markers.append(
        create_line_marker(
            marker_id=2,
            ns=f"{ns_prefix}_error_line",
            p1=target_pos,
            p2=actual_pos,
            thickness=0.005,
            rgba=(1.0, 0.2, 0.2, 0.85),
            stamp=stamp,
            frame_id=frame_id,
        )
    )

    # 4. 目标位姿 RGB 空间坐标三轴 (50mm 箭头)
    if target_rot is not None:
        axis_colors = [
            (1.0, 0.2, 0.2, 0.9),  # X: 红色
            (0.2, 1.0, 0.2, 0.9),  # Y: 绿色
            (0.2, 0.5, 1.0, 0.9),  # Z: 蓝色
        ]
        for ax_idx in range(3):
            ax_vec = target_rot[:, ax_idx] * 0.05
            markers.markers.append(
                create_arrow_marker(
                    marker_id=3 + ax_idx,
                    ns=f"{ns_prefix}_target_triad",
                    p_start=target_pos,
                    p_end=target_pos + ax_vec,
                    shaft_diameter=0.004,
                    head_diameter=0.008,
                    head_length=0.012,
                    rgba=axis_colors[ax_idx],
                    stamp=stamp,
                    frame_id=frame_id,
                )
            )

    # 5. 实际手爪 RGB 空间坐标三轴 (45mm 箭头)
    if actual_rot is not None:
        axis_colors = [
            (1.0, 0.3, 0.3, 0.8),
            (0.3, 1.0, 0.3, 0.8),
            (0.3, 0.6, 1.0, 0.8),
        ]
        for ax_idx in range(3):
            ax_vec = actual_rot[:, ax_idx] * 0.045
            markers.markers.append(
                create_arrow_marker(
                    marker_id=6 + ax_idx,
                    ns=f"{ns_prefix}_actual_triad",
                    p_start=actual_pos,
                    p_end=actual_pos + ax_vec,
                    shaft_diameter=0.0035,
                    head_diameter=0.007,
                    head_length=0.010,
                    rgba=axis_colors[ax_idx],
                    stamp=stamp,
                    frame_id=frame_id,
                )
            )

    return markers



def create_com_markers(
    com_pos: np.ndarray,
    support_poly: dict,
    stamp,
    status: str = "STABLE",
    frame_id: str = "world",
    ns_prefix: str = "g1_com",
) -> MarkerArray:
    """
    构造 Pinocchio 全身质心与双足支撑多边形 RViz 标记：
    - Marker 10: 质心 3D 光球 (CoM Sphere: 青色/黄色/红色)
    - Marker 11: 垂直地面垂准线 (Drop Line: 细线连至地面 Z=0)
    - Marker 12: 地面投影圆斑 (Floor Projection Disc)
    - Marker 13: 双足支撑多边形框 (Dual-Foot Support Polygon: LINE_STRIP)
    """
    markers = MarkerArray()

    if status == "STABLE":
        color_sphere = (0.0, 0.94, 1.0, 0.85)     # 荧光青
        color_poly = (0.0, 1.0, 0.53, 0.70)       # 安全绿
    elif status == "LEANING":
        color_sphere = (1.0, 0.72, 0.0, 0.85)     # 警戒金黄
        color_poly = (1.0, 0.72, 0.0, 0.70)
    else:
        color_sphere = (1.0, 0.20, 0.40, 0.90)     # 倾覆红
        color_poly = (1.0, 0.20, 0.40, 0.90)

    # 1. 3D 质心球
    markers.markers.append(
        create_sphere_marker(
            marker_id=10,
            ns=f"{ns_prefix}_sphere",
            pos=com_pos,
            radius=0.035,
            rgba=color_sphere,
            stamp=stamp,
            frame_id=frame_id,
        )
    )

    # 2. 垂直地面垂准线
    ground_proj = np.array([com_pos[0], com_pos[1], 0.002])
    markers.markers.append(
        create_line_marker(
            marker_id=11,
            ns=f"{ns_prefix}_dropline",
            p1=com_pos,
            p2=ground_proj,
            thickness=0.004,
            rgba=color_sphere,
            stamp=stamp,
            frame_id=frame_id,
        )
    )

    # 3. 地面投影球斑
    markers.markers.append(
        create_sphere_marker(
            marker_id=12,
            ns=f"{ns_prefix}_proj_disc",
            pos=ground_proj,
            radius=0.025,
            rgba=color_sphere,
            stamp=stamp,
            frame_id=frame_id,
        )
    )

    # 4. 双足支撑多边形线框 (闭合矩形)
    x_min, x_max = support_poly.get("x_min", -0.09), support_poly.get("x_max", 0.13)
    y_min, y_max = support_poly.get("y_min", -0.17), support_poly.get("y_max", 0.17)
    pts = [
        np.array([x_min, y_min, 0.001]),
        np.array([x_max, y_min, 0.001]),
        np.array([x_max, y_max, 0.001]),
        np.array([x_min, y_max, 0.001]),
        np.array([x_min, y_min, 0.001]),
    ]
    poly_marker = Marker()
    poly_marker.header.frame_id = frame_id
    poly_marker.header.stamp = stamp
    poly_marker.ns = f"{ns_prefix}_support_polygon"
    poly_marker.id = 13
    poly_marker.type = Marker.LINE_STRIP
    poly_marker.action = Marker.ADD
    poly_marker.scale.x = 0.006
    poly_marker.color.r = float(color_poly[0])
    poly_marker.color.g = float(color_poly[1])
    poly_marker.color.b = float(color_poly[2])
    poly_marker.color.a = float(color_poly[3])
    for pt in pts:
        p = Point()
        p.x, p.y, p.z = float(pt[0]), float(pt[1]), float(pt[2])
        poly_marker.points.append(p)
    markers.markers.append(poly_marker)

    return markers
