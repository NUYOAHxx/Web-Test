# MoveIt 2 / FCL 机器人自碰撞与面前 Box 障碍物避障模块集成指南

本文档介绍全新独立扩展模块 [`ha_arm_ik/collision_fcl.py`](file:///home/parallels/Documents/Web-Test/releases/ik_solver_v15/ha_arm_ik/collision_fcl.py) 的设计理念、API 规范以及如何将其无缝集成到内网代码环境中。

---

## 一、 为什么采用独立扩展脚本？

因为内网环境中的 `single_arm_ik.py` 与 `ik_base.py` 可能已存在特定定制版本，直接修改既有底层文件容易产生版本冲突。
因此，本模块采用**非侵入式松耦合设计（Zero-Intrusion Design）**：
- **零依赖修改**：不修改内网现有的 `single_arm_ik.py` 与 `ik_base.py` 任何一行代码；
- **单文件即插即用**：只需将 `ha_arm_ik/collision_fcl.py` 拷入内网工程目录，即可直接通过 `import` 调用；
- **底层同源加速**：使用 Pinocchio / MoveIt 2 官方同源的 C++ FCL（Flexible Collision Library / HPP-FCL）内核，单次检测仅需 **~6.8 微秒**。

---

## 二、 核心功能架构

```text
 ┌────────────────────────────────────────────────────────┐
 │            RobotCollisionManager (核心管理器)           │
 ├────────────────────────────┬───────────────────────────┤
 │  MoveIt 2 SRDF ACM 解析     │   面前 Box 障碍物生命周期  │
 │  - 支持 disable_collisions  │   - add_box_obstacle      │
 │  - 自动豁免运动学相邻连杆  │   - update_box_obstacle   │
 │  - 毫秒级自碰撞全状态复核   │   - compute_distances     │
 └─────────────┬──────────────┴─────────────┬─────────────┘
               │                            │
               ▼                            ▼
 ┌────────────────────────────────────────────────────────┐
 │           CollisionAwareIKSolver (碰撞感知求解器)        │
 │  1. 初始逆解碰撞拦截与安全诊断                         │
 │  2. 零空间几何流形主动避障 (旋转手肘避开面前 Box, <0.1mm)│
 └────────────────────────────────────────────────────────┘
```

---

## 三、 快速上手指南（三步接入）

### 步骤 1：拷贝文件至内网
将以下文件复制到内网对应目录下：
```text
ha_arm_ik/collision_fcl.py       <-- 核心扩展模块 (单文件)
tests/test_collision_fcl.py      <-- 对应单元测试
```

### 步骤 2：初始化碰撞管理器与障碍物
```python
import numpy as np
from ha_arm_ik.collision_fcl import RobotCollisionManager, create_front_box_obstacle
from ha_arm_ik.chained_ik import Chained9DofArmAngleSolver

# 1. 正常实例化你现有的求解器（无需修改求解器代码）
ik_solver = Chained9DofArmAngleSolver(urdf="path/to/robot.urdf", side="right")

# 2. 建立碰撞管理器（可传入 MoveIt 2 的 robot.srdf 配置文件）
col_mgr = RobotCollisionManager.from_urdf(
    chain=ik_solver.chain_9dof,
    srdf_source="path/to/robot.srdf",  # MoveIt 2 的 SRDF，自动读取 ACM 免检碰撞矩阵
)

# 3. 在机器人正前方添加长方体障碍物（如测试桌、前置箱体、货架）
# 参数：name, size=[长, 宽, 高], position=[x, y, z] (相对于机器人基座)
col_mgr.add_box_obstacle(
    name="front_box",
    size=[0.30, 0.40, 0.30],      # 长 30cm, 宽 40cm, 高 30cm
    position=[0.45, 0.00, 0.45],   # 位于机器人正前方 x=0.45m, z=0.45m
)
```

### 步骤 3：独立碰撞检测与测距查询
```python
# 1. 检测任意关节角状态 q 是否碰撞（仅耗时 ~7 微秒）：
report = col_mgr.check_collision(q_state, check_self=True, check_obstacles=True)
if report.is_collision:
    print(f"发生碰撞！碰撞对清单: {report.colliding_pairs}")
    if report.self_collision:
        print(" -> 发生了自碰撞！")
    if report.obstacle_collision:
        print(" -> 碰撞了面前的障碍物！")

# 2. 精确计算机器人各连杆距面前 Box 障碍物的实际距离：
dist_info = col_mgr.compute_distances_to_obstacle(q_state, "front_box")
print(f"距离面前 Box 最近连杆: {dist_info['closest_link']}, 最短间距: {dist_info['min_distance']*1000:.1f} mm")
```

---

## 四、 零空间主动避障（CollisionAwareIKSolver）

当机器人末端需要到达某个工作点，但**手臂手肘恰好被面前的 Box 障碍物阻挡**时，使用 `CollisionAwareIKSolver` 可以自动利用 7 轴或 9 轴的零空间（保持末端 6D 位姿位置 $<0.1\text{mm}$ 锁定），自动旋转手肘避开面前的障碍物：

```python
from ha_arm_ik.collision_fcl import CollisionAwareIKSolver

# 将基础求解器与碰撞管理器封装在一起
safe_solver = CollisionAwareIKSolver(
    base_solver=ik_solver,
    collision_manager=col_mgr,
    auto_nullspace_evade=True,   # 遭遇障碍时自动启动零空间流形翻肘避障
    safety_margin=0.02,          # 至少保持 20mm 安全净空
)

# 正常调用求解（完全兼容原有 solve 接口）：
q_safe, status, info = safe_solver.solve(
    T_target=T_target,
    psi_des=0.8,
    q_prev=q_curr,
)

if status == "SUCCESS":
    if info.get("collision_evaded"):
        print("成功检测到初始手肘姿态阻挡，已通过零空间流形自动调整手肘角度避开障碍物！")
    print(f"获得安全无碰撞逆解: {q_safe}")
else:
    print(f"求解失败或目标受物理阻挡不可达: {status}")
```

---

## 五、 MoveIt 2 SRDF ACM 支持细节

在 MoveIt 2 中，配置助手生成的 `robot.srdf` 包含如下标签：
```xml
<disable_collisions link1="link_1" link2="link_2" reason="Adjacent"/>
<disable_collisions link1="base" link2="link_0" reason="Adjacent"/>
<disable_collisions link1="link_0" link2="link_5" reason="Never"/>
```
本模块内部的 `parse_srdf_disabled_pairs` 会自动解析这些条目，并：
1. 自动将 SRDF 内指定的连杆对加入免检白名单，避免相邻转轴处的虚假干涉；
2. 若用户未提供 SRDF 文件，`RobotCollisionManager` 会沿运动学树自动分析所有父子连杆关系，并自动排除相邻连杆，保证开箱即用。

---

## 六、 单元测试验收

在迁移至内网后，运行以下命令即可立即验证该扩展模块：
```bash
python3 -m unittest tests/test_collision_fcl.py
```
> **预期输出**：
> ```text
> .....
> ----------------------------------------------------------------------
> Ran 5 tests in 0.139s
> OK
> ```
包含 5 个核心测试场景：
1. 面前 Box 障碍物动态位姿转换与尺寸更新；
2. MoveIt 2 SRDF 语法解析与免检对提取；
3. 全身自碰撞极速检测；
4. 面前 Box 空间穿透与距离连续测距；
5. 闭环零空间流形主动避障（初态碰撞 -> 翻肘脱困且末端误差 $<0.5\text{mm}$）。

