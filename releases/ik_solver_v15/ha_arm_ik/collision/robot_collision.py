#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""机器人碰撞几何模型构建: 胶囊包络、机身凸包、mesh 凸包化与资产解析。

架构演进与修复要点:
* 配置外部化: 胶囊包络段表抽离至 collision_config.DEFAULT_CAPSULE_SEGMENTS, 支持外部传入自定义段表覆盖。
* 挂载连杆与登记名对齐: 默认段表统一将登记名与挂载帧对齐至实际运动连杆 (Link2/4/6/7), 修复原代码名义连杆与真实 FK 挂载帧错位问题。
"""
from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
import struct
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import xml.etree.ElementTree as ET

import numpy as np
import pinocchio as pin
from scipy.spatial import ConvexHull

# FCL 的 Python 绑定包已由 hppfcl 演进为 coal，优先支持 coal，平滑回退 hppfcl
try:
    import coal
except ImportError:
    import hppfcl as coal

from .collision_config import (
    BODY_HULL_LINKS,
    DEFAULT_CAPSULE_SEGMENTS,
    HULL_CACHE_DIR,
    HULL_DECOMPOSE_LINK_PREFIXES,
    PACKAGE_DIRS,
)

logger = logging.getLogger(__name__)


def align_z_to_vector(vec: np.ndarray) -> np.ndarray:
    """计算将局部 +Z 轴 [0, 0, 1] 旋转对齐至目标向量 vec 的正交旋转矩阵 R ∈ SO(3)。

    用于将 FCL 默认沿局部 Z 轴延伸的 Capsule 精确对齐至连杆骨骼位移方向。
    """
    v = np.asarray(vec, dtype=float).reshape(3)
    norm = float(np.linalg.norm(v))
    if norm < 1e-9:
        return np.eye(3, dtype=float)

    u = v / norm
    z = np.array([0.0, 0.0, 1.0], dtype=float)
    c = float(np.dot(z, u))

    if c > 1.0 - 1e-9:
        return np.eye(3, dtype=float)
    if c < -1.0 + 1e-9:
        return np.diag([1.0, -1.0, -1.0])

    axis = np.cross(z, u)
    s = float(np.linalg.norm(axis))
    axis /= s

    K = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ],
        dtype=float,
    )

    return np.eye(3, dtype=float) + s * K + (1.0 - c) * (K @ K)


def _resolve_package_dirs() -> List[str]:
    """解析 mesh 资产目录: 配置表 + ament_index (若可用) + 当前工作目录按存在性去重。"""
    dirs = [d for d in PACKAGE_DIRS if os.path.isdir(d)]
    dirs.append(os.getcwd())

    unique_dirs: List[str] = []
    seen: Set[str] = set()

    for d in dirs:
        try:
            rp = os.path.realpath(d)
            if os.path.isdir(rp) and rp not in seen:
                seen.add(rp)
                unique_dirs.append(rp)
        except Exception:
            pass

    return unique_dirs


def _sha1_short(path: Path, n: int = 8) -> str:
    """文件内容 sha1 前 n 位 (凸包缓存失效键: 源 mesh 变更即换名重建)。"""
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def _resolve_urdf_collision_mesh(
    urdf_root: ET.Element,
    link: str,
    package_dirs: List[str],
) -> Optional[Path]:
    """从 URDF 解析指定连杆的 collision mesh 绝对路径 (缩放 mesh 直接放弃)。"""
    for l in urdf_root.findall("link"):
        if l.get("name") != link:
            continue

        col = l.find("collision")
        mesh = col.find("geometry/mesh") if col is not None else None
        if mesh is None:
            return None

        scale = mesh.get("scale")
        if scale is not None:
            try:
                s = [float(v) for v in scale.split()]
            except ValueError:
                return None
            if s != [1.0, 1.0, 1.0]:
                return None

        fname = mesh.get("filename") or ""
        if fname.startswith("package://"):
            rel_full = fname[len("package://") :]
            rel_stem = rel_full.split("/", 1)[1] if "/" in rel_full else rel_full
            for d in package_dirs:
                for rel in (rel_full, rel_stem):
                    cand = Path(d) / rel
                    if cand.is_file():
                        return cand
            return None

        cand = Path(fname)
        return cand if cand.is_absolute() and cand.is_file() else None

    return None


def _save_hull_stl(path: Path, verts: np.ndarray, faces: np.ndarray) -> None:
    """凸包网格写二进制 STL (coal MeshLoader 可直接加载的通用交换格式)。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"\0" * 80)
        f.write(struct.pack("<I", len(faces)))
        for a, b, c in faces:
            v1, v2, v3 = verts[a], verts[b], verts[c]
            normal = np.cross(v2 - v1, v3 - v1)
            ln = float(np.linalg.norm(normal))
            normal = normal / ln if ln > 1e-12 else np.zeros(3)
            f.write(
                struct.pack(
                    "<12fH",
                    *normal.tolist(),
                    *v1.tolist(),
                    *v2.tolist(),
                    *v3.tolist(),
                    0,
                )
            )


def _load_or_build_hull(src: Path, cache_dir: Path) -> Any:
    """加载缓存凸包, 未命中则从源 mesh 现算并写盘; 返回 coal 碰撞几何。"""
    cache = cache_dir / f"{src.stem}_{_sha1_short(src)}.STL"
    loader = coal.MeshLoader()

    if cache.is_file():
        bvh = loader.load(str(cache))
        bvh.computeLocalAABB()
        return bvh

    bvh_src = loader.load(str(src))
    verts = np.array(
        [bvh_src.vertex(i) for i in range(bvh_src.num_vertices)],
        dtype=float,
    )
    hull = ConvexHull(verts)
    hv = verts[hull.vertices]
    remap = {v: i for i, v in enumerate(hull.vertices)}
    faces = np.array(
        [[remap[a], remap[b], remap[c]] for a, b, c in hull.simplices],
        dtype=int,
    )

    try:
        _save_hull_stl(cache, hv, faces)
        bvh = loader.load(str(cache))
        bvh.computeLocalAABB()
        return bvh
    except Exception:
        vs = coal.StdVec_Vec3s()
        for v in hv:
            vs.append(v)
        ts = coal.StdVec_Triangle()
        for a, b, c in faces:
            ts.append(coal.Triangle(int(a), int(b), int(c)))
        cv = coal.Convex(vs, ts)
        cv.computeLocalAABB()
        return cv


def _apply_hull_decomposition(
    geom_model: pin.GeometryModel,
    pin_model: pin.Model,
    package_dirs: List[str],
    urdf_path: Union[str, Path],
) -> int:
    """mesh 档 GJK 加速: HULL_DECOMPOSE_LINK_PREFIXES 匹配的连杆 BVH 逐个替换为凸包。"""
    try:
        urdf_root = ET.parse(str(urdf_path)).getroot()
    except Exception as e:
        logger.warning("[Collision] 凸包替换跳过: URDF 解析失败 (%s)", e)
        return 0

    replaced = 0
    frames = pin_model.frames

    for gid in range(geom_model.ngeoms):
        go = geom_model.geometryObjects[gid]
        l_name = frames[go.parentFrame].name
        if not l_name.startswith(HULL_DECOMPOSE_LINK_PREFIXES):
            continue

        src = _resolve_urdf_collision_mesh(urdf_root, l_name, package_dirs)
        if src is None:
            logger.debug("[Collision] %s: collision mesh 未解析到, 保留 BVH", l_name)
            continue

        try:
            hull_geom = _load_or_build_hull(src, HULL_CACHE_DIR)
        except Exception as e:
            logger.warning("[Collision] %s: 凸包构建失败, 保留 BVH (%s)", l_name, e)
            continue

        go.geometry = hull_geom
        replaced += 1

    if replaced:
        logger.info(
            "[Collision] 凸包替换: %d 连杆 BVH -> Convex (缓存 %s)",
            replaced,
            HULL_CACHE_DIR,
        )
    return replaced


class RobotCollisionGeometry:
    """胶囊/机身凸包包络注册 (构建期功能组件, 状态归 manager)。"""

    chain: Any
    pin_model: Optional[pin.Model]
    geom_model: pin.GeometryModel
    link_geom_ids: Dict[str, List[int]]
    geom_to_link: Dict[int, str]
    _urdf_path: Optional[Union[str, Path]]

    def _build_capsule_proxies(
        self,
        segment_table: Optional[
            Dict[str, List[Tuple[str, str, float, Tuple[float, float, float]]]]
        ] = None,
    ) -> None:
        """双臂胶囊包络: 每侧 4 骨段, 支持自定义段表或默认配置表注册。"""
        chain = getattr(self, "chain", None)
        if chain is None or not hasattr(chain, "joints"):
            return

        seg_map = segment_table or DEFAULT_CAPSULE_SEGMENTS

        side = (
            "left"
            if any(j.get("child", "").startswith("LHand") for j in chain.joints)
            else "right"
        )
        other = "right" if side == "left" else "left"

        sides = [side]
        other_check_link = seg_map[other][0][1]
        if self.pin_model is not None and self.pin_model.existFrame(
            other_check_link, pin.FrameType.BODY
        ):
            sides.append(other)

        for s in sides:
            for reg_link, mount_link, r_seg, bone in seg_map[s]:
                bone_vec = np.asarray(bone, dtype=float)
                capsule = coal.Capsule(r_seg, float(np.linalg.norm(bone_vec)))
                capsule.computeLocalAABB()
                place = pin.SE3(align_z_to_vector(bone_vec), bone_vec * 0.5)

                jid, fid = 0, 0
                if self.pin_model is not None and self.pin_model.existFrame(
                    mount_link, pin.FrameType.BODY
                ):
                    fid = self.pin_model.getFrameId(mount_link, pin.FrameType.BODY)
                    jid = self.pin_model.frames[fid].parentJoint

                go = pin.GeometryObject(f"{reg_link}_col", jid, fid, place, capsule)
                gid = self.geom_model.addGeometryObject(go)
                self.link_geom_ids.setdefault(reg_link, []).append(gid)
                self.geom_to_link[gid] = reg_link

    def _build_body_hull_proxies(self) -> None:
        """机身凸包包络 (躯干/骨盆/头): 补胶囊表缺失的身体覆盖。"""
        if self.pin_model is None:
            return
        urdf_path = getattr(self, "_urdf_path", None)
        if not urdf_path or not os.path.exists(str(urdf_path)):
            return

        try:
            urdf_root = ET.parse(str(urdf_path)).getroot()
        except Exception as e:
            logger.warning("[Collision] 机身凸包跳过: URDF 解析失败 (%s)", e)
            return

        package_dirs = _resolve_package_dirs()
        registered = 0

        for link in BODY_HULL_LINKS:
            if not self.pin_model.existFrame(link, pin.FrameType.BODY):
                continue
            try:
                src = _resolve_urdf_collision_mesh(urdf_root, link, package_dirs)
                if src is None:
                    continue
                hull_geom = _load_or_build_hull(src, HULL_CACHE_DIR)
                fid = self.pin_model.getFrameId(link, pin.FrameType.BODY)
                frame = self.pin_model.frames[fid]
                jid = frame.parentJoint
                place = frame.placement
                go = pin.GeometryObject(
                    f"{link}_hull_col",
                    jid,
                    fid,
                    place,
                    hull_geom,
                )
                gid = self.geom_model.addGeometryObject(go)
                self.link_geom_ids.setdefault(link, []).append(gid)
                self.geom_to_link[gid] = link
                registered += 1
            except Exception as e:
                logger.warning("[Collision] %s 凸包注册失败 (跳过): %s", link, e)

        if registered:
            logger.info(
                "[Collision] 机身凸包已挂载: %d/%d (%s)",
                registered,
                len(BODY_HULL_LINKS),
                ", ".join(BODY_HULL_LINKS),
            )