"""
pytest 全局环境准备。

机器人模型资源依赖 ament 索引
------------------------------
URDF 中的网格以 ``package://humanoid_description/meshes/...`` 形式引用, 需要
ament_index 才能解析。此前这依赖一项外部的 ``~/ros2_ws/install/humanoid_description``
目录与配套的环境变量, 导致跑测试必须先手工 source 一长串环境, 漏一项就出现
``Mesh ... could not be found`` —— 看起来像代码回归, 实则是环境未就绪。

本文件在测试收集前自动把仓库内的 ament 前缀注入环境, 使 ``pytest`` 可直接运行:

    pytest tests/

标记文件位于 ``resources/ament/install/humanoid_description``, 随仓库一同提交,
其中 meshes/urdf/srdf 均以相对符号链接指向 ``resources/humanoid``, 因此整仓
可任意移动或改名而不会失效。
"""

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
AMENT_PREFIX = os.path.join(REPO_ROOT, "resources", "ament", "install",
                             "humanoid_description")

# 仓库根目录需可导入 (core / launch 等顶层包)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# ament 索引前缀: 前置以便覆盖可能存在的同名外部包
if os.path.isdir(AMENT_PREFIX):
    existing = os.environ.get("AMENT_PREFIX_PATH", "")
    parts = [AMENT_PREFIX] + [p for p in existing.split(os.pathsep) if p and p != AMENT_PREFIX]
    os.environ["AMENT_PREFIX_PATH"] = os.pathsep.join(parts)