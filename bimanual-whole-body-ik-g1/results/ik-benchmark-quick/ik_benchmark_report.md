# 双臂全身 IK 多目标与轨迹基准报告

生成时间（UTC）：2026-10-13T18:07:07Z
参考模型：included generic 18-DoF reference robot（18 DoF）
样本数：16；成功：14；失败：2；成功率：87.5%

## 按模式/场景汇总

| 分组 | 样本数 | 成功率 | 平均耗时 (ms) | P95 耗时 (ms) | 平均迭代 | 最大左手位置误差 (mm) | 最大右手位置误差 (mm) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mode:independent | 8 | 100.0% | 522.111 | 1179.190 | 16.9 | 0.1827 | 0.2031 |
| mode:carry | 8 | 75.0% | 4551.318 | 14691.448 | 69.8 | 112.5443 | 111.9818 |
| suite:static_pose | 4 | 75.0% | 4304.159 | 14221.656 | 66.5 | 112.5443 | 111.9818 |
| suite:trajectory | 12 | 91.7% | 1947.566 | 8228.429 | 35.6 | 3.4123 | 3.3826 |

## 轨迹指标

| 轨迹 | 帧数 | 成功率 | 左/右目标路径长度 (m) | 平均耗时 (ms) | 单步最大关节变化（见 JSON/CSV） |
| --- | ---: | ---: | ---: | ---: | --- |
| independent/lateral_sweep | 3 | 100.0% | 0.0911 / 0.2656 | 478.173 | 按关节列于 JSON |
| independent/lift_rotate | 3 | 100.0% | 0.1577 / 0.1145 | 758.580 | 按关节列于 JSON |
| carry/object_yaw_lift | 3 | 66.7% | 0.1622 / 0.1622 | 4106.019 | 按关节列于 JSON |
| carry/object_pitch_roll | 3 | 100.0% | 0.1358 / 0.0980 | 2447.493 | 按关节列于 JSON |

## 指标口径与边界

- Target poses are generated from reachable reference configurations.
- Carry cases preserve the fixed grasp relation using torso-only motions.
- Trajectory solves warm-start from the previous returned joint state.
- Solve time is wall-clock time around the IK solve call, excluding report writing.
- Discrete acceleration is a per-sample-index smoothness proxy, not physical acceleration.
- This is an offline kinematics benchmark; it does not establish real-time or hardware safety.
- 位置误差单位为米，姿态误差单位为弧度；CSV/JSON 保留每帧目标位姿、实际位姿和全身关节向量。
- 关节行程/单步变化的角关节单位为弧度，升降关节单位为米；这些是原生单位，不合成为物理量。
- 本基准仅用于离线算法回归。参考模型结果不代表具体实机精度、控制周期性能、碰撞安全或动力学能力。
