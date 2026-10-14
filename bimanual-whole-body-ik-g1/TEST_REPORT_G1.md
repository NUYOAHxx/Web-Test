# Unitree G1 16-DoF IK 测试报告

日期：2026-10-14  
测试对象：`config/g1.json`（固定 pelvis；腰 yaw/pitch + 双臂各 7 关节）

## 验证摘要

- **自动化测试：9/9 通过**，覆盖 G1 关节配置、几何雅可比、解析目标梯度、解析闭链雅可比、独立双手求解、共持闭链、速度限制、障碍/支撑区域及基准输出。
- **URDF FK 交叉验证：50 组关节姿态、100 个手部 TCP**。与参考项目中 Pinocchio 加载的 G1 URDF 对照，最大位置差为 `2.44e-16 m`，最大旋转矩阵差为 `7.60e-16`。
- **完整离线基准：54/54 成功（100%）**；其中独立双手 28/28、共持 26/26、连续轨迹 36/36。共持闭链误差最大 `3.67e-12`。
- **求解耗时**：全样本中位数 `56.1 ms`、均值 `50.1 ms`、P95 `84.8 ms`、最大 `141.6 ms`；这是当前计算机上的 Python 离线求解时间，不是控制周期或实时性能保证。
- G1 双模式 CLI 演示成功；共持示例闭链误差约 `3.33e-12`。

## 解析导数性能对照

对同一 G1 模型和固定可达目标，分别用解析导数和 SLSQP 默认数值差分重复 5 次。独立双手场景中位耗时 `132.8 ms` 对 `1,380.0 ms`（`10.39×`）；共持场景 `118.3 ms` 对 `1,546.8 ms`（`13.08×`）。两种方法迭代数分别相同（62 次、32 次）。这是单组目标上的本机对照，受任务/机器/缓存影响，不代表所有输入都获得相同加速；完整基准 54 个目标的耗时见下方文件。

## 求解器改动

主任务误差现在通过几何雅可比与 SO(3) 对数映射解析求导；共持相对位姿等式也提供解析雅可比。一次目标评估会共享腰部 FK，并缓存目标函数和导数之间的姿态计算。可选的球障碍/质心约束仍由 SciPy 数值差分，不是碰撞网格或真实动力学模型。

## G1 模型边界

- 关节轴、固定变换、手部/TCP 偏移、位置限位及速度上限与参考 Unitree G1 rev_1_0 URDF 对齐；活动关节顺序可由 `robot.names` 查询。
- pelvis 固定；腿不参与，`waist_roll_joint` 固定为 0。该降阶模型不表示完整 29-DoF 机器人。
- 没有精确 G1 mesh 自碰撞、静力矩、接触、平衡动力学或安全验证。`approximate_com` 和 `SphereObstacle` 仅用于运动学原型。
- 旧的通用 18-DoF 模型结果保留在 `TEST_REPORT.md` 与 `IK_DETAILED_RESULTS.md`，仅作为迁移前历史记录。

## 可复现命令

```bash
cd /workspace/bimanual-whole-body-ik
python3 -m pytest -q
PYTHONPATH=src python -m bimanual_wbik.demo --mode both
PYTHONPATH=src python -m bimanual_wbik.benchmark --output-dir results/g1-ik-benchmark
```

完整逐帧结果：`results/g1-ik-benchmark/ik_benchmark.json`、`ik_benchmark_samples.csv` 与 `ik_benchmark_report.md`。
