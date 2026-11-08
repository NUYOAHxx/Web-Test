# `ha_arm_ik` — 高性能冗余机械臂逆运动学求解引擎

`ha_arm_ik` 是面向人形机器人冗余运动链（典型如“腰 2 + 臂 7”9 自由度单臂、16 自由度双臂协同及 7 自由度纯手臂）的高性能工业级逆运动学 Python 核心包。支持 6D 位姿硬约束、电机额定静态重力矩保护、整机水平质心（CoM）防倾覆、空间臂角 $\psi$（Direct-Psi）显式控制，以及基于 MoveIt 2 SRDF / FCL 的自碰撞检测与前方障碍物零空间逃逸。

---

## 1. 目录结构

```text
ha_arm_ik/
├── __init__.py                   # 公共导出接口 (SingleArmIKSolver, DualArmIKSolver, Chained9DofArmAngleSolver 等)
├── ik_base.py                    # 底层运动链建模 (SerialChain)、URDF 解析、Pinocchio 桥接与公共数据结构
├── single_arm_ik.py              # 单臂/腰臂协同核心求解器 (merged-v15 统一整合版)
├── dual_arm_ik.py                # 16 自由度双臂协同全身逆运动学求解器 (共腰 2 + 左臂 7 + 右臂 7)
├── chained_ik.py                 # 9 轴腰臂协同 + 指定空间臂角 (Direct-Psi) 级联/混合求解扩展模块
├── collision_fcl.py              # 零侵入 MoveIt 2 SRDF / FCL (hppfcl/coal) 自碰撞与障碍物避障扩展模块
├── solver.py                     # 经典模块化单臂求解主干 (与 solver_dls.py 配套)
├── solver_dls.py                 # 阻尼最小二乘 (DLS)、SNS 限位钳制、四合一次任务势能场与离线种子库
├── IKengine.py                   # 7 轴球腕解析/半解析逆运动学 (SSIK) 子系统
├── README.md                     # 当前包结构与文档总览导航
└── docs/                         # 内部技术规范与评测报告目录
    ├── architecture/             # 架构设计规范文档
    │   ├── HA_ARM_IK_CORE_ENGINE_ARCHITECTURE_SPEC.md
    │   └── CHAINED_9DOF_DIRECT_PSI_ARCHITECTURE_SPEC.md
    └── reports/                  # 性能评测与优化验证报告
        ├── BENCHMARK_REPORT_DIRECT_PSI_VS_BASELINE.md
        └── V15_OPTIMIZATION_REPORT.md
```

---

## 2. 核心源码模块职责速查

| 模块文件 | 核心类 / 函数 | 模块职责说明 |
| :--- | :--- | :--- |
| [`ik_base.py`](ik_base.py) | `SerialChain`, `IKResult`, `Status`, `FailureReason`, `JointStateSnapshot` | 基础运动学树解析、正运动学（FK）、空间臂角 $\psi$ 几何计算、URDF 质量属性与极限臂展估算。 |
| [`single_arm_ik.py`](single_arm_ik.py) | `SingleArmIKSolver`, `LoadPolicy`, `solve_ik` | 单臂（含腰部 9 轴）主求解器。集成五阶段多初值搜索、SNS 主动集限位钳制、四合一零空间势场与 8 级 Pareto 容差带选解。 |
| [`dual_arm_ik.py`](dual_arm_ik.py) | `DualArmIKSolver` | 双臂共腰 16 自由度协同求解器。支持双臂同时或单手独立控制、共享腰部自适应释放与全身力矩/质心平衡。 |
| [`chained_ik.py`](chained_ik.py) | `Chained9DofArmAngleSolver`, `FusedJacobianEngine`, `solve_9dof_with_arm_angle` | 9 轴显式臂角控制扩展。提供两阶段级联精修、解析臂角偏导雅可比与高速连续轨迹跟踪（~0.5 ms）。 |
| [`collision_fcl.py`](collision_fcl.py) | `RobotCollisionManager`, `CollisionAwareIKSolver`, `BoxObstacle` | 独立零侵入防碰撞模块。支持求解器内省自适应建模（Mesh + Capsule 双模包络）、MoveIt 2 SRDF ACM 过滤、多参考系 Box 障碍物与零空间手肘逃逸。 |
| [`solver.py`](solver.py) & [`solver_dls.py`](solver_dls.py) | `pinocchio_ik_one_seed`, `build_seed_library`, `solve_direct_psi` | 模块化拆分版的单臂求解流水线与底层 DLS/SNS 数值迭代算子。 |
| [`IKengine.py`](IKengine.py) | `Arm` | 针对标准 7 自由度球腕构型的解析/半解析逆解器（SSIK），用于快速生成多手肘分支种子。 |

---

## 3. 内部文档导航 (`docs/`)

### 3.1 架构与数学模型规范 (`docs/architecture/`)
1. **[`HA_ARM_IK_CORE_ENGINE_ARCHITECTURE_SPEC.md`](docs/architecture/HA_ARM_IK_CORE_ENGINE_ARCHITECTURE_SPEC.md)**
   - **内容**：深度剖析 `single_arm_ik.py` 与 `ik_base.py` 内部端到端控制流水线、李代数 $\text{log}_6$ 任务残差、腰部逆度量释放、四合一次任务势场（回中/限位/重力矩/CoM）、SNS 钳制、8 级 Pareto 选解及全局超参数标定表。
2. **[`CHAINED_9DOF_DIRECT_PSI_ARCHITECTURE_SPEC.md`](docs/architecture/CHAINED_9DOF_DIRECT_PSI_ARCHITECTURE_SPEC.md)**
   - **内容**：详解“阶段一 9 轴宏观定位 + 胸腔坐标系降维桥接 + 阶段二 7 轴 Direct-Psi 微观定肘 + 阶段三整机安全闭环回退”的两阶段级联架构与状态机设计。

### 3.2 性能评测与优化报告 (`docs/reports/`)
1. **[`BENCHMARK_REPORT_DIRECT_PSI_VS_BASELINE.md`](docs/reports/BENCHMARK_REPORT_DIRECT_PSI_VS_BASELINE.md)**
   - **内容**：涵盖 5 大测试模块、412 个工况点的对比评测（全圆周离散覆盖、50 路点连续轨迹跟踪、奇异极限拉直压测、多姿态模式与可行域测绘）。
2. **[`V15_OPTIMIZATION_REPORT.md`](docs/reports/V15_OPTIMIZATION_REPORT.md)**
   - **内容**：汇总 `merged-v15` 版本在 5 个核心模块上的热路径逐位等价（bit-exact）提速实测数据（最高 13.52× 加速，486 条回归场景 0 差异）与文档注释同步修复清单。

---

## 4. 外层关联文档索引 (`../`)

在 `releases/ik_solver_v15/` 外层目录中，还包含以下配套专题指南与历史版本演进说明：
- **防碰撞集成指南**：[`../COLLISION_FCL_INTEGRATION_GUIDE.md`](../COLLISION_FCL_INTEGRATION_GUIDE.md) — `collision_fcl.py` 内网三步接入、双模几何包络与避障 API 说明。
- **臂角优化与解析雅可比**：[`../ARM_ANGLE_OPTIMIZATION_AND_PIPELINE.md`](../ARM_ANGLE_OPTIMIZATION_AND_PIPELINE.md)、[`../ARM_ANGLE_CONTROL_AND_DIRECT_PSI_ANALYSIS.md`](../ARM_ANGLE_CONTROL_AND_DIRECT_PSI_ANALYSIS.md)、[`../WHY_NOT_SSIK_AND_DIRECT_PSI.md`](../WHY_NOT_SSIK_AND_DIRECT_PSI.md)。
- **版本演进日志**：[`../README_v15.md`](../README_v15.md)、[`../README_v14.md`](../README_v14.md)、[`../README_dual.md`](../README_dual.md)（及 `v6`–`v13` 系列说明）。

