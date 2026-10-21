# ha_arm_ik merged-v6（solver.py + solver_dls.py）

- 只需用 `ha_arm_ik/solver.py`、`ha_arm_ik/solver_dls.py` 替换原包里的同名文件。入口 `HAArmIKSolver(...).solve(...)` 和其他公开函数签名都没变。
- `ha_arm_ik/__init__.py`、`tests/stubs.py`（ik_common / ik_candidate 测试替身）只是离线测试用的脚手架，**不要**装进正式工程。
- `diff/` 里是相对用户 v5 的 unified diff，`baseline/` 里是原始版和用户 v5 版，供对比测试用。

## 1. 约束分层

| 层 | 编号 | 内容 | 实现 | 默认 |
|---|---|---|---|---|
| 硬约束 | H1 | 末端位姿误差 ≤ pos_tol / ori_tol（full / tool_dir / tool_cone） | DLS 主任务 + 候选复核 | 5e-4 m / 5e-4 rad |
| 硬约束 | H2 | 关节限位 = URDF ∩ 虚拟搜索窗（dls_virtual_half_windows） | SNS 主动集钳制，迭代中不越界；≤1e-9 的舍入超界会吸附回边界 | — |
| 硬约束 | H3 | 每个守护关节的静态重力矩 \|τg\| / 容许值 ≤ 1 | load_mode="strict" 门控 | strict |
| 硬约束 | H4 | 不绕大圈：相对 q_prev 的单关节转动量 ≤ max_joint_jump_rad | solver.py 候选过滤（含 SSIK / direct-psi 候选） | π，reject |
| 软目标 | S1 | 负荷峰值最小；峰值相差 ≤ load_peak_tie 的视为同档 | 单种子内择优 + 候选择优 | 0.02 |
| 软目标 | S2 | 构型合理：q_ready 物理回中 + 限位裕度 + 重力矩 + 整机质心水平偏移 | 零空间次任务；同档候选再交给原 select_best_candidate（连续性 / 标准姿态 / 限位裕度 / 腰部回中） | — |

择优顺序：先用 H1–H4 过滤，再看 S1 峰值容差带，最后按 S2 / 原择优函数排序。

IK 内**不做**的事：自碰撞（交给 MoveIt，诊断里 `collision_check="NOT_PERFORMED"`）、动态力矩（只算静态重力矩）、
显式的可操作度 / 奇异度门槛（只靠 DLS 阻尼和 180° 对径切向处理）。

## 2. 速度与成功率手段

| 手段 | 作用 |
|---|---|
| 种子顺序 q_prev → q_ready → 大到小交错扰动（NOISE_SCALES = 0.60/0.30/0.15/0.05）→ 中点 | 前几个种子就覆盖不同解族；远目标成功率 97% → 100% |
| 停滞检测：每 25 次迭代误差下降 < 5% 就换种子 | v5 的锁腰不可达种子会空转到 500 次迭代（约 350 ms），这是主要耗时 |
| 姿态整理早停：连续 3 步代价相对下降 < 0.1% | 末端达标后不在边际收益上耗时 |
| 早停：≥2 个种子（其中 ≥1 个不锁腰），且已有峰值 ≤ load_early_stop_ratio、跳变合格、限位裕度足够的候选 | 默认偏速度；要更强的负荷寻优见第 3 节 |
| SNS Cholesky 零空间步改为 w(Dz − DJᵀA⁻¹JDz)，与 SVD 回退一致 | 次任务一定是下降方向，不会反向推姿态 |
| 种子内择优用“峰值容差带 + 代价” | 不会因为峰值只差 0.001 就丢掉姿态改善 |
| 对径（180°）死点按雅可比可达方向选切向 | 单轴 / 少轴时也能逃出死点 |
| 负荷向量缓存、关节映射向量化、重力矩复用 | 减少每次迭代的重复计算 |

## 3. 配置项（config 字典）

新增：

| 键 | 取值 | 说明 |
|---|---|---|
| `max_joint_jump_rad` | `π`（默认）/ `None`（不限）/ 正标量 / `{关节名: rad 或 None}` | 字典里没列出的关节取 π。只有无界 continuous 关节的角度差会折叠到 [-π, π)，有限限位关节（包括 ±2π 的多圈 revolute）按实际转动量算 |
| `joint_jump_policy` | `"reject"`（默认）/ `"prefer"` | reject：剔除超限候选，全部超限时报 JOINT_JUMP_LIMIT；prefer：优先选不超限的，全部超限时仍返回结果，并在诊断里记 `joint_jump_ok=False` |

默认值有变化的：`load_min_seeds` 3 → 2，`load_min_unlocked_seeds` 2 → 1（search 字典里的 `min_seeds` / `min_unlocked` 同步改了）。

原有的相关项：`load_peak_tie`(0.02)、`load_early_stop_ratio`(0.60)、`load_soft_ratio`(0.70)、`load_max_seeds`、
`load_budget_s`(0.5)、`solve_budget_s`、`dls_max_iter`、`dls_virtual_half_windows`、`load_allowable_efforts`（必须显式给出）。

solver_dls.py 顶部的模块常量：`PIN_IK_STALL_ITERS=25`、`PIN_IK_STALL_RTOL=0.05`、`SETTLE_COST_RTOL=1e-3`、
`SETTLE_PATIENCE=3`、`POSE_PEAK_BAND_FRACTION=0.25`（种子内容差带 = 0.25 × peak_tie = 0.005）、`BOUND_SNAP_TOL=1e-9`、`NOISE_SCALES`。

两套推荐配置：

```python
# 速度优先（默认）
config = {...}
# 负荷优先：多试种子，平均峰值比 v5 还低，p90 耗时仍是 v5 的约一半（见第 6 节）
config = {..., "load_min_seeds": 3, "load_min_unlocked_seeds": 2}
# 腕部 / 腰部要求更平滑时，可以按关节收紧跳变上限，例如
config = {..., "max_joint_jump_rad": {"Hip_Link1": 1.5, "Body_Link": 1.5}}
```

## 4. 行为变化与兼容性

1. H4 默认开启。v5 可能返回某个关节转动 > π 的解，v6 会剔除这类解。如果要恢复 v5 的行为，设 `max_joint_jump_rad=None`。
2. strict 模式下 q_prev 默认取实测角。做路点序列时，请显式传 q_prev = 上一个路点的解。
3. 种子顺序和扰动尺度变了，所以同一目标返回的解可能和 v5 不同（都满足硬约束）。
4. 早停默认放宽后速度更快，但平均峰值会略高（仍在容差带内）。
5. `LoadPolicy` 删除了 `continuity_weight` / `mean_load_weight` / `reference_weight`。这三个字段在 v5 里只有声明，没有被使用，也没有从 config 读取。外部代码如果直接用这些关键字构造 `LoadPolicy`，会报 TypeError。
6. 整机质心现在计入 universe 惯量（根 link 和固定在根上的 link），所以质心项的数值和 v5 不同（更接近实际）。
7. `_build_effective_bounds` 的第 4 个参数保留了位置兼容，但不参与计算。
8. direct-psi / SSIK 相关函数与用户 v5 一致（AST 比对）。

## 5. 诊断

- 失败码 `last_load_diagnostics["code"]`：
  - `SEED_ERRORS`：所有种子都抛异常（详情里附首个异常）
  - `SEARCH_TIME_BUDGET`：预算用完时一个候选都还没有
  - `NO_CANDIDATE`：没有任何种子产出候选
  - `JOINT_JUMP_LIMIT`：位姿达标的候选全部因跳变超限被剔除（建议插中间路点或放宽上限）
  - `NO_VALID_POSE_OR_PHYSICS`：没有候选达到位姿精度
  - `NO_LOAD_SAFE_SOLUTION`：位姿达标，但限位或负荷门控没通过
  - `SEEDS_EXHAUSTED`：详情附首个种子异常
- 成功时会写 `joint_jump_max_rad` / `joint_jump_joint` / `joint_jump_ok`、`joint_jump_policy` / `joint_jump_limit_rad` / `joint_jump_rejected`。
- `search_diagnostics` 新增 `seed_errors` / `first_seed_error` / `candidates`。
- 单种子 `diag["stop_reason"]`：SETTLED / SETTLE_LIMIT / SMALL_STEP / STALLED_NO_PROGRESS / LINE_SEARCH_STALLED / TIME_BUDGET / ITERATION_LIMIT；
  `diag["bound_snaps"]` 是舍入吸附的次数。

## 6. 验证

- `run_validation.py`：90 项测试（contract_flow 31、pinocchio_numeric 18、policy_state 25、v6_constraints 16）在 Pinocchio 4.1.0 和 3.9.0 下全部通过，结果见 `validation_pin*.json/txt`。
- SNS 随机 2 万例：越界超过容差 0 例、步长超限 0 例、零空间代价上升 0 例（14 例浮点舍入超界都已吸附）。
- 合成基准 `bench/bench_targets.py`（同一批目标，Pin 4.1.0，耗时单位 ms，峰值 = max |τg|/容许值）：

| 场景 | 版本 | 成功 | 中位 / p90 耗时 | 平均峰值 | 跳变 > π |
|---|---|---|---|---|---|
| near_full ×40 | v5 | 40/40 | 45.6 / 385.5 | 0.1235 | 0 |
| | v6 默认 | 40/40 | 18.3 / 43.2 | 0.1239 | 0 |
| | v6 负荷优先 | 40/40 | 54.0 / 96.2 | 0.1239 | 0 |
| far_full ×60 | v5 | 60/60 | 90.3 / 384.4 | 0.0800 | **2**（最大 3.23） |
| | v6 默认 | 60/60 | 36.7 / 83.9 | 0.0822 | 0（最大 3.04） |
| | v6 负荷优先 | 60/60 | 95.7 / 189.0 | 0.0784 | 0 |
| far_tool_dir ×50 | v5 | 50/50 | 156 / 607 | 0.0723 | 0 |
| | v6 默认 | 50/50 | 52.5 / 162.0 | 0.0742 | 0 |
| | v6 负荷优先 | 50/50 | 104.6 / 271.7 | 0.0685 | 0 |
| far_tool_cone ×50 | v5 | 50/50 | 160 / 493 | 0.0776 | 0 |
| | v6 默认 | 50/50 | 67.8 / 156.3 | 0.0791 | 0 |
| | v6 负荷优先 | 50/50 | 148.5 / 204.6 | 0.0728 | 0 |

**验证范围的限制**：ik_common / ik_candidate 用的是测试替身（原择优权重、IKResult 内部逻辑没验证）；URDF 是合成的 9 轴模型；
ROS 2 Jazzy、MoveIt、实机和真实 URDF 都没测。Python 3.12 只做了语法检查（运行时是 3.13）。

## 7. 上线前需要确认

1. 真实 URDF 的限位和各关节容许力矩（`load_allowable_efforts`、`load_guard_joint_names`）。
2. 跳变上限：π 只是“不绕大圈”的底线，腕部 / 腰部是否要更小的值？策略用 reject 还是 prefer？
3. 默认选“速度优先”还是“负荷优先”。
4. 在真实模型上跑一遍 `bench/bench_targets.py`（需要把 tests/fixtures.py 里的模型换成真实 URDF 和力矩表）。
