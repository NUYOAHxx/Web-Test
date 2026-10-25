# HA Arm IK 交付快照

本仓库存放 `ha_arm_ik` 逆解迭代包，不是单一可安装工程。根目录只放地图；各版本源码、测试与验证记录在 `releases/` 下互不覆盖。

**不要把某个 `releases/*` 整目录拷进用户工程。** 生产接入通常只替换该包 README 标明的 `ha_arm_ik` 文件，且不要覆盖用户已有的 `__init__.py`、`ik_common.py`、`ik_candidate.py`、`robot_model.py`（除非该版 README 明确要求）。

## 目录

```text
docs/worklog.md              分析笔记与进度记录（原根目录 note）
releases/ha_arm_ik_merged_v6  原结构合并 v6：负荷硬门槛 + 多初值编排
releases/ik_solver_v15        v6→v15 演进包：续算漂移修复、提速、双臂 16 维
releases/ik_solver_load_v1    独立负荷感知分支（新 Solver 类，不覆盖原入口）
releases/ik_solver_merged_v2  把 load_v1 策略并回原 HAArmIKSolver 入口
```

## 两条线，不要混成一个“最新版”

```text
用户原 ha_arm_ik
        │
        ├─ 原结构迭代 ─► v6 ─► … ─► v15（含 DualArmIKSolver）
        │
        └─ 负荷策略另开 ─► load_v1（独立类）─► merged_v2（并回原入口）
```

| 需求 | 使用 | 入口 |
| --- | --- | --- |
| 单臂原入口 + 双臂 + 续算不漂移 | [`releases/ik_solver_v15`](releases/ik_solver_v15/README_v15.md) | `HAArmIKSolver` / `DualArmIKSolver` |
| 原入口 + 负荷策略并回，接口尽量兼容 | [`releases/ik_solver_merged_v2`](releases/ik_solver_merged_v2/README_中文.md) | `HAArmIKSolver` |
| 对照独立负荷分支、不改原 solver | [`releases/ik_solver_load_v1`](releases/ik_solver_load_v1/README.md) | `LoadAwareIKSolver` |
| 对照 v6 约束分层与基线 diff | [`releases/ha_arm_ik_merged_v6`](releases/ha_arm_ik_merged_v6/README_v6.md) | `HAArmIKSolver` |

v15 与 merged_v2 是并行快照，不是互相替换关系。v15 带双臂与续算修复；merged_v2 强调把独立负荷分支并回原 `solver.py` / `solver_dls.py`。

## 各包要点

### `ha_arm_ik_merged_v6`

替换原包中的 `solver.py`、`solver_dls.py`。负荷 `strict` 为硬门槛，种子停滞检测与早停偏速度。说明见包内 `README_v6.md`。

### `ik_solver_v15`

在 v6 之后累计到 v15：续算阶段不再做达标后姿态整理（默认），目标静止时关节变化为 0；另含 `ha_arm_ik/dual_arm_ik.py`（dual-p2.1）。包内 `README_v6.md` … `README_v15.md`、`README_dual.md` 是分版说明，日常看 `README_v15.md`。离线测试 250 项（Pinocchio 4.1.0）。

### `ik_solver_load_v1`

新增 `load_policy.py` + `solver_load_aware.py`，第三返回值是 `report` 字典，不是原 `IKResult`。附带 `ik_observer_jazzy_v3` 只读状态观察器。允许力矩必须显式配置。

### `ik_solver_merged_v2`

负荷目标、完整状态映射、任务 Jacobian 与加权零空间并入原两个文件；保留 `HAArmIKSolver.solve()`。同一套 74 项测试在 Pinocchio 3.9.0 / 4.1.0 下分别通过。合成 URDF；`ik_common.py` / `ik_candidate.py` 仅测试替身。尚未联调真实机器人、完整工程、Jazzy / MoveIt。

## 复跑验证

每个交付包自带 `run_validation.py`。在该包目录下执行：

```bash
cd releases/ik_solver_merged_v2   # 或 v15 / load_v1 / merged_v6
python3 run_validation.py
```

`requirements_tested_pin*.txt` 是当时测试环境快照，不是要求升级 ROS 系统依赖。这些结果覆盖离线合成模型，不代表实机卸载比例或实时性能。

## 文档

- 进度与代码审阅摘录：[`docs/worklog.md`](docs/worklog.md)
- 接入步骤、失败码、回滚：以所选 `releases/*/README*` 为准
