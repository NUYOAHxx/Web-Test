# -*- coding: utf-8 -*-
"""双臂测试 / 基准共用的合成模型（腰 2 + 左右各 7 轴）。

与 tests/test_v12_changes.two_arm_xml() 生成同一个 URDF（几何常数逐项复制），但不 import 单臂代码，
这样双臂测试在没有单臂模块的环境里也能跑。
"""

from __future__ import annotations

import os
import tempfile

import numpy as np

SIGNS = np.array([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0])  # 右臂限位镜像：奇数关节反号
ARM_AXIS = [(0, 0, 1), (0, 1, 0), (1, 0, 0), (0, 1, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
ARM_XYZ = [(0, .2, .18), (.1, .02, 0), (.25, 0, -.01), (.3, .01, 0), (.25, 0, 0), (.15, 0, .01), (.08, 0, 0)]
ARM_RPY = [(.3, 0, 0), (0, 0, 0), (0, 0, .1), (0, 0, 0), (0, .05, 0), (0, 0, 0), (0, 0, 0)]
ARM_MASS = [1.5, 2, 1.5, 1.2, .8, .6, .3]
ARM_LIMITS = [(-2.6, 2.9), (-2.2, 1.4), None, (-2.5, .3), (-2.8, 2.7), (-1.0, 2.6), (-2.9, 2.8)]

ALL_JOINTS = (["Hip_Link1", "Body_Link"] + [f"LHand_Link{i}" for i in range(1, 8)]
              + [f"RHand_Link{i}" for i in range(1, 8)])


def _inertial(mass, xyz):
    return (f'<inertial><origin xyz="{" ".join(map(str, xyz))}" rpy="0 0 0"/><mass value="{mass}"/>'
            '<inertia ixx=".02" iyy=".02" izz=".02" ixy="0" ixz="0" iyz="0"/></inertial>')


def two_arm_xml():
    """躯干两侧各一条 7 轴臂：右臂 = 左臂关于躯干 x-z 面的镜像；第 3 关节 continuous。"""
    parts = ['<robot name="synthetic_two_arm">', f'<link name="base">{_inertial(12, (0, 0, 0))}</link>',
             f'<link name="hip">{_inertial(3, (.06, 0, 0))}</link>',
             f'<link name="torso">{_inertial(8, (.06, 0, 0))}</link>',
             '<joint name="Hip_Link1" type="revolute"><parent link="base"/><child link="hip"/>'
             '<origin xyz="0 0 .3"/><axis xyz="0 1 0"/><limit lower="-1.0" upper="1.5" effort="300" velocity="2"/></joint>',
             '<joint name="Body_Link" type="revolute"><parent link="hip"/><child link="torso"/>'
             '<origin xyz="0 0 .25"/><axis xyz="0 0 1"/><limit lower="-2.5" upper="2.5" effort="300" velocity="2"/></joint>']
    for side in ("L", "R"):
        parent = "torso"
        for i in range(7):
            xyz, rpy, lim, com = list(ARM_XYZ[i]), list(ARM_RPY[i]), ARM_LIMITS[i], [.06, .01, 0]
            if side == "R":
                xyz[1], rpy[0], rpy[2], com[1] = -xyz[1], -rpy[0], -rpy[2], -com[1]
                if lim is not None and SIGNS[i] < 0:
                    lim = (-lim[1], -lim[0])
            child = f"{side.lower()}arm{i + 1}"
            kind = "continuous" if lim is None else "revolute"
            limit = ('<limit effort="300" velocity="2"/>' if lim is None
                     else f'<limit lower="{lim[0]}" upper="{lim[1]}" effort="300" velocity="2"/>')
            parts.append(f'<link name="{child}">{_inertial(ARM_MASS[i], com)}</link>')
            parts.append(f'<joint name="{side}Hand_Link{i + 1}" type="{kind}"><parent link="{parent}"/>'
                         f'<child link="{child}"/><origin xyz="{" ".join(map(str, xyz))}" '
                         f'rpy="{" ".join(map(str, rpy))}"/><axis xyz="{" ".join(map(str, ARM_AXIS[i]))}"/>'
                         f'{limit}</joint>')
            parent = child
        tcp_rpy = (0, .2, .1) if side == "L" else (0, .2, -.1)
        name = "left_tcp" if side == "L" else "right_tcp"
        parts += [f'<link name="{name}"/>',
                  f'<joint name="{name}_fixed" type="fixed"><parent link="{parent}"/><child link="{name}"/>'
                  f'<origin xyz=".12 0 0" rpy="{" ".join(map(str, tcp_rpy))}"/></joint>']
    parts.append("</robot>")
    return "".join(parts)


_URDF_PATH = None


def two_arm_urdf_path():
    """把合成 URDF 写到临时目录（每个进程一次），返回路径。"""
    global _URDF_PATH
    if _URDF_PATH is None or not os.path.exists(_URDF_PATH):
        d = tempfile.mkdtemp(prefix="dual_ik_")
        _URDF_PATH = os.path.join(d, "synthetic_two_arm.urdf")
        with open(_URDF_PATH, "w", encoding="utf-8") as f:
            f.write(two_arm_xml())
    return _URDF_PATH


def synthetic_layout():
    """合成模型的 DualArmLayout：关节名与真机相同，TCP 帧名不同。"""
    from ha_arm_ik.dual_arm_ik import DualArmLayout
    return DualArmLayout(left_tcp="left_tcp", right_tcp="right_tcp")
