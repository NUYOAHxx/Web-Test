# ha_arm_ik merged-v7（solver.py + solver_dls.py）：提速版

- 用 `ha_arm_ik/solver.py`、`ha_arm_ik/solver_dls.py` 替换原包里的同名文件即可。入口 `HAArmIKSolver(...).solve(...)` 和所有公开函数签名不变（`make_scatter_seeds` 只多了一个可选关键字参数）。
- `ha_arm_ik/__init__.py`、`tests/stubs.py`（ik_common / ik_candidate 测试替身）只是离线测试脚手架，**不要**装进正式工程。
- `diff/*_v6_to_v7.diff` 是相对 v6 的 unified diff；`baseline/solver_v6.py`、`baseline/solver_dls_v6.py` 是 v6 原文件。
- 约束分层、择优规则、失败码都和 v6 一样（见 README_v6.md）。

## 1. 改了什么

| 方案 | 内容 | 对结果的影响 |
|---|---|---|
| 2 实现优化 | 每个种子预先算好目标位姿、索引、质心投影和容许力矩向量；次任务直接复用主任务已算好的 Pinocchio q；只需要代价时不算梯度；对角逆度量的 SNS 改成向量运算；迭代内只更新末端那一帧的位姿；Cholesky 直接调 LAPACK（dpotrf/dpotrs）；`make_scatter_seeds(max_count=)` 只对用得到的前缀去重 | 算法和运算顺序不变。Cholesky 换回 scipy 时与 v6 **逐值一致**；用 LAPACK 时与 v6 差 ≤ 2e-12 rad |
| 1 迭代参数 | 停滞检测：窗口 25 → 12 次迭代，下降阈值 5% → 2.5%（判定速率和 v6 相当，只是检查更频繁）；姿态整理上限 50 → 15 步；慢步耐心 3 → 2；低负荷早停阈值默认 0.60 → min(0.70, soft_ratio) | 不可达种子更早放弃，姿态整理更早收手；部分目标会选到另一组同样满足约束的解 |
| 3 种子顺序 | q_prev 先跑，锁腰种子第二。如果 q_prev 的解已经合格（位姿 / 跳变 / 裕度，strict 下还要求负荷合格且峰值 ≤ 早停阈值），而且腰部和 q_ready 的差 ≤ `locked_seed_skip_tol`（默认 0.02 rad），就认为锁腰种子和它等价，直接跳过。锁腰种子触发腰部释放时，紧接着用释放状态重跑一次 q_prev（就是 v6 的第二个种子） | 跳过时少跑一个种子；不跳过时候选集和 v6 一样 |

停滞阈值最初按方案 1 只把窗口减到 12、阈值保持 5%。验证时发现，这样会误杀收敛较慢、但本来能收敛的种子：100 个 tool 目标里有 3 个因此丢掉了更好的解，峰值高了 0.02～0.056。其中 2 个丢的是锁腰种子的腰部回中解，腰部偏离最大到 1.33 rad。阈值改成 2.5% 后，这 3 个目标恢复到 v6 的结果，速度只回落 2～13%。

## 2. 实测（合成 9 轴模型，Pin 4.1.0，同一批目标，每个目标取 3 次运行的最短耗时，单位 ms）

| 场景 | v6 中位 / p90 / 均值 | v7 中位 / p90 / 均值 | v7 提速（均值 / p90） | 只用方案 2 |
|---|---|---|---|---|
| near_full ×40 | 27.4 / 67.6 / 40.7 | 14.4 / 27.2 / 17.9 | 2.3× / 2.5× | 1.8× / 1.8× |
| far_full ×60 | 49.8 / 123.2 / 65.3 | 24.9 / 48.4 / 30.9 | 2.1× / 2.5× | 1.9× / 2.0× |
| far_tool_dir ×50 | 79.2 / 244.5 / 113.1 | 30.2 / 81.9 / 42.1 | 2.7× / 3.0× | 2.5× / 2.6× |
| far_tool_cone ×50 | 102.7 / 232.6 / 125.0 | 34.9 / 76.8 / 42.6 | 2.9× / 3.0× | 2.4× / 2.5× |
| 路点跟踪 ×119（腰不动） | 19.7 / 30.9 / 22.6 | 11.0 / 16.3 / 12.3 | 1.8× / 1.9× | 1.8× / 1.9× |

质量（v7 对比 v6，同一批目标逐个配对）：

- 成功率不变（319/319），跳变 > π 仍为 0，最大跳变不变。
- 平均负荷峰值：full 模式和跟踪不变（差 ≤ 0.0001）；tool_dir 0.0742 → 0.0739，tool_cone 0.0791 → 0.0789。
- 100 个 tool 目标里只有 1 个峰值高出 0.01 以上（+0.021，0.058 → 0.080），原因是姿态整理到了 15 步上限。
- q_ready 回中 RMS：tool_dir 0.870 → 0.878，tool_cone 0.868 → 0.871，其余不变；平均腰部偏离变化 ≤ 0.01 rad。
- 只用方案 2 时结果和 v6 相同（差 ≤ 2e-12 rad）。

分项（`bench/v7/speed_breakdown_pin4_1_0.json`）：
- 大部分提速来自方案 2。
- 停滞检测主要压低 p90。
- 姿态整理参数提速 2～14%，tool_cone 最多。
- 早停阈值和锁腰跳过在这套基准上几乎看不出效果：峰值都 < 0.25，跳过只发生在 1/40（near）和 4/119（跟踪）。实机负荷更高、腰部常在 q_ready 附近时，这两项的作用会更明显。

## 3. 配置项与兼容性

| 键 / 常量 | v7 默认 | v6 | 说明 |
|---|---|---|---|
| `locked_seed_skip_tol`（config） | 0.02 rad | — | 腰部逐关节最大偏差 ≤ 该值时跳过锁腰种子；`None` = v6 种子顺序（锁腰在前、不跳过、不重跑）；负数 / NaN / inf 报 ValueError |
| `load_early_stop_ratio`（config） | 不配置时 = min(0.70, `load_soft_ratio`) | 0.60 | 显式配置时以配置为准，必须 ≤ soft_ratio |
| `PIN_IK_STALL_ITERS` / `PIN_IK_STALL_RTOL` | 12 / 0.025 | 25 / 0.05 | solver_dls.py 模块常量 |
| `MAX_SETTLE_EXTRA_ITERS` / `SETTLE_PATIENCE` | 15 / 2 | 50 / 3 | 同上；`WAIST_POST_ITERS` 仍是 `MAX_SETTLE_EXTRA_ITERS` 的别名 |

- 想要**和 v6 同解但保留方案 2 的提速**（1.8～2.5×）：把上面 4 个常量改回 v6 值，并在 config 里设 `load_early_stop_ratio=0.60`、`locked_seed_skip_tol=None`。
- `load_max_seeds` 现在是**种子名额**上限：跳过的锁腰种子和释放后重跑的 q_prev 都占名额，所以实际运行的种子数 ≤ max_seeds，跟 v6 一样。
- `search_diagnostics` 新增 `seed_slots_used`、`seed_order`（`q_prev_first` / `locked_first` / `no_waist`）、`locked_seed_skipped`、`q_prev_release_rerun`；`per_seed` 每项新增 `seed_kind`（`q_prev` / `locked` / `q_prev_release` / `scatter`）。`seeds_attempted` 只统计实际运行的种子，跳过时可能是 1。
- 早停条件里的“至少 min_seeds 个种子”改为按名额计数，并且要求锁腰比较已完成（锁腰种子已运行，或已判定等价而跳过）。
- v6 里只设 `load_soft_ratio` < 0.60 而不设早停阈值会报错；v7 会自动取 soft_ratio。

## 4. 验证

- `run_validation.py`：106 项测试（原有 90 项 + 新增 `tests/test_v7_speed.py` 16 项），在 Pinocchio 4.1.0（numpy 2.5.4）和 3.9.0（numpy 2.3.5）下全部通过，结果见 `validation_pin*.json/txt`。原有测试只改了一项：`test_unlock_comparison_before_low_load_early_stop` 允许“锁腰种子被判等价而跳过”，并补了 `locked_seed_skip_tol=None` 时锁腰种子必跑的检查。
- 整条求解逐值等价（`bench/v7/equiv.py` + `eq_compare.py`，v7 关闭方案 1/3 且 Cholesky 换回 scipy，对比 v6）：8 组场景（near / far / tool_dir / tool_cone / 180° 对径 / 路点跟踪 / load_mode=off / 无腰）共 395 次求解、29,233 次 DLS 迭代。返回的解、状态、失败码和每个种子的诊断（含逐次误差历史）在两个 Pinocchio 版本下都是 **0 处差异**。
- `test_v7_speed.py` 主要覆盖：
  - 在真实求解流程中截获 20 次单种子调用，和 v6 参考实现（`tests/reference/solver_dls_v6.py`，v6 原样副本）逐值一致。覆盖三种姿态模式、锁腰、腰部负荷释放、非负荷路径和无腰配置。
  - 这些调用内部的主任务项 2,574 次、次任务 843 次、SNS 694 次，也和 v6 逐值一致。LAPACK 直调时 SNS 和 v6 最大差 2.3e-12。
  - `make_scatter_seeds` 前缀去重：结果和随机数状态都与 v6 一致。
  - 早停阈值默认值、种子顺序、跳过、重跑和非法配置。

**验证范围的限制**（和 v6 相同）：ik_common / ik_candidate 用的是测试替身；URDF 是合成的 9 轴模型；ROS 2 Jazzy、MoveIt、实机和真实 URDF 都没测；Python 3.12 只做了语法检查。另外两点：
- 前缀去重依赖 `unique_seed_list` “保序、保留首次出现”的逐个去重语义。测试替身是这样实现的，正式工程里的 `ik_common.unique_seed_list` 需要确认。
- LAPACK 直调和 scipy 1.18 封装相差 1e-12 量级，所以 v7 的解与 v6 不是逐位相同的。

## 5. 上线前需要确认

1. 早停阈值默认变成 min(0.70, soft_ratio)，默认配置下就等于 soft_ratio：只要已有不超过软阈值的候选，完成必需的比较后就会停。实机负荷较高（峰值 0.6～0.7）时会比 v6 少试种子，平均峰值可能略高。在意的话请显式设 `load_early_stop_ratio=0.60`。
2. 姿态整理上限 15 步会让 tool 模式的回中 RMS 略增（≤ 0.009 rad）。更看重姿态的话，可以把 `MAX_SETTLE_EXTRA_ITERS` 改回 50，均值慢 2～14%。
3. `locked_seed_skip_tol=0.02` rad 是否合适：越大跳过越多，但被跳过的锁腰解与 q_prev 解的差别也越大。
4. 请在真实 URDF 上复测：把 tests/fixtures.py 里的模型换成真实 URDF 和力矩表，再运行下面的命令。

```bash
python bench/v7/speed.py <v6 包根目录> v6 v6.pkl 3      # v6 包根目录 = 上次交付的 v6 解压目录
python bench/v7/speed.py <v7 包根目录> v7 v7.pkl 3      # 也可用 plan2[+stall][+settle][+early][+skip] 逐项打开
python bench/v7/quality.py v6.pkl v7.pkl                 # 配对比较耗时 / 峰值 / 回中 / 跳变
python bench/v7/case_probe.py <包根目录> v7 far_tool_cone 2   # 单个目标逐种子诊断
python bench/v7/equiv.py <v6 包根目录> v6 a.pkl && python bench/v7/equiv.py <v7 包根目录> v7compat b.pkl && python bench/v7/eq_compare.py a.pkl b.pkl
```
