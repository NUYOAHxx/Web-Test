# ha_arm_ik merged-v14：工程化梳理 + 冗余清理（结果与 v13 逐位一致）+ 16 自由度设计讨论

入口不变：`HAArmIKSolver(urdf, side, use_ssik, use_pinocchio_ik, use_torque_cost, use_direct_psi, include_waist, verbose, config)`，
`solve(...)` 的签名和 `(q, status, IKResult)` 返回不变。只改了 `solver.py`、`solver_dls.py` 和测试；`ik_candidate.py`、
`ik_common.py`、`robot_model.py` 与 v13 逐字节相同（v13 的这三个文件 = v12 = v9 + 右臂 q_ready 数值）。

## 1. 结论

1. **行为不变。** 486 条固定场景回放与 v13 完全一致（浮点逐位比较，0 差异）。场景包括构造属性、一次性求解、路点跟踪、
   配置变体、7 轴 direct-psi、失败路径、物理评估和配置报错。计时用的 800 次求解，两版的解也逐位相同。
2. **性能不变。** ABAB 交替计时，逐目标耗时比（v14 / v13）的中位数为 1.000（two）和 0.997（wide）。均值差 −4% / +1%，
   小于同一版本两轮之间的波动（v13 two 两轮均值 43.9 / 33.6 ms）。
3. **代码更少，静态检查无告警。** solver.py 4098 → 3960 行（代码行 3502 → 3349），solver_dls.py 2844 → 2617 行
   （2303 → 2123）。pyflakes 告警 56 → 0。删掉的是：未用导入 56 个、本地重复的算法副本 3 个、无人读取的兼容常量 9 个、
   旧参数和钩子 5 个、无人读取的上下文键 2 个；另有 2 个只给测试用的旧函数移到了 `tests/dls_compat.py`。
   重复代码合并了 8 处（第 5 节）。
4. **文档改成架构说明。** 两个模块的文件头不再是版本流水，改为“接口 / 分层 / 数据流 / 约束 / 搜索阶段 / 单种子迭代”。
   版本流水移到本文第 9 节。第 2–4 节是完整的设计、功能和配置参考。
5. **16 自由度（腰 2 + 双臂 2×7）建议：以 16 维联合求解为主，主从分解作种子和补救。** 主从分解即先解难的那只手臂
   （9 轴，含腰），再锁住腰解另一只（7 轴）。在简化 DLS 单种子微基准上，联合求解成功 61%，主从分解 48%，两者互补：
   37 例只有联合能解，11 例只有分解能解。接口、失败语义等 7 个问题需要你决定（第 8.6 节），定了再改代码。

## 2. 工程化设计

### 2.1 模块分层（上层只调用下层，无循环依赖）

| 模块 | 职责 | 主要内容 | 行数 |
|---|---|---|---|
| `solver.py` | 入口与编排 | `HAArmIKSolver`：配置解析与校验、坐标系与限位、搜索阶段编排（`collect_pinocchio_candidates_and_seeds`）、候选硬约束过滤与择优、strict 负荷保护、诊断；可选 7 轴引擎 SSIK / direct-psi | 3960 |
| `solver_dls.py` | 数值核心 | 单种子 DLS/SNS 迭代 `pinocchio_ik_one_seed`、次任务 `_SecondaryObjective`、静态负荷模型（`LoadPolicy`、`prepare_load_context`、`load_physics`）、腰部舒适窗推导、任务空间种子库、关节映射 | 2617 |
| `ik_candidate.py`（你的文件） | 打分与择优 | `score_candidate`、`select_best_candidate`、`IKResult` / `Status` / `FailureReason` | 598 |
| `ik_common.py`（你的文件） | 纯函数 | 几何、位姿误差、限位、2πk 分支对齐 | 739 |
| `robot_model.py`（你的文件） | 模型 | URDF 解析、`SerialChain` 正运动学、`ARM_CONFIGS`、`WAIST_JOINT_NAMES` | 357 |

依赖：solver → solver_dls / ik_candidate / ik_common / robot_model；solver_dls → ik_common / robot_model；
ik_candidate → ik_common / robot_model；robot_model → ik_common。`solver.__all__` 只导出 7 个公共名称：
`HAArmIKSolver`、`IKResult`、`Status`、`FailureReason`、`make_result`、`JointStateSnapshot`、`LoadPolicy`。

### 2.2 一次 solve 的数据流

```
solve()                       加锁（一个实例同时只跑一个 solve）；strict：冻结关节快照 -> 负荷上下文 -> 截止时间
 └ _solve_impl()              目标换到求解基座 / URDF 根坐标系；校验 q_prev（strict 下未传时取实测关节角）
    ├ _collect_portfolio()    组装 DLS 参数与搜索参数
    │  └ collect_pinocchio_candidates_and_seeds()        多阶段多种子搜索（2.4 节）
    │     └ 每个种子：pinocchio_ik_one_seed（2.5 节）-> 对齐到离 q_prev 最近的 2πk 分支 -> score_candidate
    │                 -> 跳变 -> strict 负荷 -> 去重入池 -> 早停判定
    └ _finalize_result()      跳变策略 -> strict 负荷复核 + 峰值容差带 -> 力矩代价 -> select_best_candidate
                              -> strict 最终复核 + 快照新鲜度 -> 近似回退 -> 失败码（_classify_empty_pool）
```

### 2.3 约束分层

| 层 | 编号 | 内容 | 实现位置 |
|---|---|---|---|
| 硬约束 | H1 | 末端位姿误差 ≤ POS_TOL / ORI_TOL（full / tool_dir / tool_cone） | DLS 主任务 + 最终复核（同一口径） |
| 硬约束 | H2 | 有效限位 = URDF ∩ `position_bounds` ∩ `dls_virtual_half_windows` | SNS 钳制，迭代中不越界 |
| 硬约束 | H3 | 静态负荷：每个守护关节 \|τ_g\| / 容许力矩 ≤ 1（strict） | 搜索中逐候选计算，择优前后各复核一次 |
| 硬约束 | H4 | 不绕大圈：单关节相对 q_prev 的转动量 ≤ `max_joint_jump_rad` | 候选过滤；上限 < π 时前移为搜索框 |
| 偏好 | P1 | 腰部舒适窗 `dls_waist_comfort`：重新定位类求解先只在窗内搜 | 搜索阶段 comfort → full |
| 择优 | S1 | 负荷峰值容差带：最低峰值 + `load_peak_tie` 以内同档 | `_finalize_result` |
| 择优 | S2 | 同档内交给 `select_best_candidate`：连续性 / 标准姿态（臂角）/ 限位裕度 / 腰部回中 / 可选力矩代价 | ik_candidate |

IK 内不做：自碰撞（交给 MoveIt，诊断里 `collision_check="NOT_PERFORMED"`）、动态力矩（只算静态重力矩）。

### 2.4 搜索阶段与种子顺序

每个阶段最多 `load_max_seeds` 个名额，所有阶段共享时间预算。

| 阶段 | 何时运行 | 搜索范围 |
|---|---|---|
| `continuation` | 续算类求解（目标在 FK(q_prev) 附近：位置 ≤ 0.1 × 臂展且姿态 ≤ 0.1 rad） | q_prev ± `dls_continuation_max_step`；随动 + 锁腰两个种子，有局部解即采用 |
| `comfort` | 重新定位类求解，舒适窗开启 | 有效限位 ∩ 舒适窗 |
| `full` | 未开舒适窗时的唯一常规阶段；或舒适窗内没有可接受解 | 有效限位 |
| `jump_box` | 跳变上限 ≥ π，但只找到跳变超限的解 | q_prev ± `max_joint_jump_rad` |
| `unboxed` | 跳变上限 < π，框内没有可接受解 | 有效限位（不加框重搜一次） |

常规阶段的种子顺序：q_prev → 锁腰（腰 = q_ready）→ q_ready → 种子库（最近邻 3 个）→ 随机扰动（标准差 0.60 / 0.30 /
0.15 / 0.05 交错）。早停条件：至少 `load_min_seeds` 个名额、至少 `load_min_unlocked_seeds` 个不锁腰种子、锁腰比较已完成，
并且已有“位姿达标 + 跳变合格 + 限位裕度足够（strict 下还要求负荷合格且峰值 ≤ 早停阈值）”的候选。

### 2.5 单种子迭代（`pinocchio_ik_one_seed`，每次迭代）

1. 主任务 `_make_task_terms`：位姿残差 e（位置按 `dls_characteristic_length` 归一）和解析雅可比 J。
2. 逆度量 D（对角）：手臂为 1；腰部按“肩 → 目标距离 / 臂展”和“腰偏离 q_ref”平滑释放。无状态，去程和返程对称。
3. 次任务 `_SecondaryObjective`：H(q) = q_ready 回中 + 限位裕度 + 重力矩 + 整机质心水平偏移，z = −∇H。
   零空间权重随 |e| 衰减，达标后放大做姿态整理。
4. `_sns_clamped_core`：主动集限位钳制 + 正交信赖域，一步内不越界且 ‖dq‖ ≤ max_step（失败时退回 `_metric_svd_step`）。
5. 回溯线搜索（先合成步，再纯主任务步）、停滞检测、180° 对径死点逃逸、姿态整理和收敛判定。

返回的 q 是位姿达标点中按“峰值容差带 + 次任务代价”选出的那个；没有达标点时返回主任务误差最小的点。

### 2.6 strict 负荷保护流程

1. `solve()` 取快照：`joint_state` 参数，或 `update_joint_state` 存下的最新快照。
2. `prepare_load_context` 校验快照新鲜度（`load_state_max_age_s`、`load_require_source_age`），要求全身关节都在快照里
   且在限位内（不钳位），守护关节包含全部活动关节，每个守护关节都有确认的容许力矩（不回退到 URDF effort）。
   通过后生成冻结的整机 q 模板，非活动关节取实测值。任何一项不满足返回 `LOAD_CONFIG_OR_STATE_INVALID`，不求解。
3. q_prev 未传时取快照里活动关节的实测值，求解基座位姿也按实测姿态算。截止时间 = 开始时刻 + `load_budget_s`。
4. 搜索中每个候选都算整机静态负荷，早停要求负荷合格。
5. `_finalize_result`：跳变过滤 → 快照新鲜度 → 负荷 / 位姿 / 限位复核 → 峰值容差带 → 择优 → 被选中的候选重新计算一次
   （`FINAL_RECHECK_FAILED`）→ 快照新鲜度和上下文变化检查：非活动关节相对冻结值的变化超过 `load_context_tolerance_*`
   时返回 `STALE_OR_CHANGED_CONTEXT`。
6. `solve()` 返回成功结果前再查一次快照；失败时 q_prev 恢复原值；最后恢复 URDF 力矩上限并清掉上下文。

### 2.7 诊断

`last_load_diagnostics`：`code`（ACCEPTED / 失败码）、`accepted`、负荷峰值与峰值关节、各守护关节负载率、
`joint_jump_*`、`continuation`、`waist_comfort`（含 `selected_stage`、`selected_in_comfort`）、`search`
（`stop_reason`、`seeds_attempted`、`seed_slots_used`、`seed_order`、`locked_seed_skipped`、`continuation_local_used`、
`jump_limit_search_box`、`jump_limit_unboxed_retry`、`library_seeds_queued`、`locked_seed_unreachable`、
`jump_limit_rescue(_early)`、`comfort_patience_stop`、`seed_errors`、`candidates`）、`elapsed_ms`、`budget_exceeded`。
`IKResult.per_seed` 记录每个种子的 `seed_kind`（q_prev / locked / q_prev_release / library / scatter / comfort_edge）、
`search_stage`、迭代数、收敛与停滞情况。

失败码：`NO_CANDIDATE`、`SEED_ERRORS`、`NO_VALID_POSE_OR_PHYSICS`、`JOINT_JUMP_LIMIT`、`NO_LOAD_SAFE_SOLUTION`、
`TIME_BUDGET` / `SEARCH_TIME_BUDGET`、`FINAL_RECHECK_FAILED`、`STALE_OR_CHANGED_CONTEXT`、`LOAD_CONFIG_OR_STATE_INVALID`、
`NO_LOAD_SAFE_APPROXIMATE_SOLUTION`、`INVALID_BUDGET`、`MODEL_UNAVAILABLE`、`SOLVE_ERROR`。

## 3. 功能与效果（v5 → v14 累积）

数据来自各版 README，均为合成模型、Pinocchio 4.1.0。

| 方面 | 功能 | 版本 | 效果 |
|---|---|---|---|
| 约束 | 约束分层：先用 H1–H4 过滤，再在可行解里比 S1 / S2 | v6 | 不会再选到“负荷略低但要绕大圈”的解 |
| 约束 | 不绕大圈 H4（`max_joint_jump_rad` / `joint_jump_policy`） | v6 | 跳变 > π 的解为 0 |
| 约束 | 失败码细分（无候选 / 种子异常 / 位姿不可达 / 跳变超限 / 负荷超限 / 超时） | v6 | 种子异常不再被误报为负荷问题 |
| 正确性 | DLS 容差与最终复核同口径；帧查询按 BODY；臂展沿运动链累加 | v8 | DLS 判达标的解不会被复核拒绝；关节与连杆同名的 URDF 可用；臂展估计 0.62 → 1.15 m |
| 正确性 | 接入真实 ik_common / ik_candidate / robot_model，修正 11 处 | v9 | psi_ready 为 NaN 时择优不再退化等 |
| 负荷安全 | strict：冻结快照、整机负荷门控、最终复核、上下文变化检测（框架是你 v5 原有的） | v5，v6 / v8 修正 | 质心计入根 link；load_mode=off 的质心参考含根质量（q_ready 偏差 73 mm 消除） |
| 成功率 | 种子顺序 q_prev → q_ready → 由大到小的扰动 | v6 | 远目标 97% → 100% |
| 成功率 | 种子库、舒适阶段提前回退、跳变重搜、锁腰可达性检查 | v13 | 解出 395→400、400→400、387→397、384→398（两模型 × 一次性 / 回放） |
| 速度 | 停滞检测、姿态整理早停 | v6 | 去掉 v5 锁腰不可达种子的空转（约 350 ms） |
| 速度 | 热路径优化、停滞窗口 12 / 2.5%、q_prev 先行 + 等价锁腰跳过 | v7 | 均值提速 2.1–2.9×，跟踪 1.8× |
| 速度 | v13 搜索改动（同上） | v13 | 均值 53→33、56→35、60→46、70→49 ms；p90 123→59、170→68、152→78、195→96 ms |
| 姿态 | 腰部舒适窗（由 q_ready 处重力矩推导，只用于重新定位，窗内无解自动放开） | v8 | 不减少可解目标；腰部大幅跟踪 239/239（每次都加窗的方案只有 59/239） |
| 连续性 | 续算局部优先 + 跳变上限前移到搜索 | v9 | 腰部大幅跟踪单步最大 2.47 → 0.15 rad；跳变上限 0.25 时 9/239 → 239/239；每帧 13.4 → 5.6 ms |
| 连续性 | 续算类求解不用种子库 | v13 | 路点跟踪与 v12 逐值相同 |
| 对称性 | 右臂 q_ready 直接写数值 | v12 | q_ready 处右臂 TCP 与左臂镜像距离 730 → 0 mm |
| 工程 | 配置校验、诊断、单实例单 solve 锁、种子库多实例共享（按模型内容缓存） | v6–v13 | — |
| 工程 | 冗余清理、架构文档 | v14 | 与 v13 逐位一致 |

## 4. 配置参考（`config` 字典，共 58 个键）

下表“默认”为不配置时的取值。`dls_*` 增益配 None 时用 solver_dls 的模块常量。

**负荷保护（19 个）**

| 键 | 默认 | 说明 |
|---|---|---|
| `load_mode` | "strict" | "strict" = 负荷门控；"off" = 显式关闭 |
| `load_allowable_efforts` | strict 必填 | {关节名: 容许力矩}；守护关节都要有正有限值，不回退到 URDF |
| `load_guard_joint_names` | None = 全部关节 | 守护关节，必须包含全部活动关节 |
| `load_joint_weights` | {} | 每关节负荷代价权重（缺省 1） |
| `load_soft_ratio` | 0.70 | 负载率软阈值；也是舒适窗推导的 s |
| `load_early_stop_ratio` | min(0.70, soft_ratio) | 低负荷早停阈值，须 ≤ soft_ratio |
| `load_peak_tie` | 0.02 | 峰值容差带 |
| `load_absolute_gain` | `dls_gravity_torque_gain` 或 0.04 | 负载率平方代价权重 |
| `load_excess_gain` / `load_relative_gain` | 0.50 / 0.0 | 超软阈值部分 / 相对参考姿态的代价权重 |
| `load_state_max_age_s` / `load_require_source_age` | 1.0 / False | 快照最大年龄；是否要求源时间戳 |
| `load_context_tolerance_rad` / `load_context_tolerance_m` | 0.01 / 0.001 | 非活动关节相对冻结快照允许的变化 |
| `load_massless_links` | () | URDF 审计时视为无质量的 link |
| `load_budget_s` | 0.5 | strict 时间预算（s） |
| `load_max_seeds` / `load_min_seeds` / `load_min_unlocked_seeds` | 10 / 2 / 1 | 每阶段名额上限；早停前至少比较的名额 / 不锁腰种子数 |

**模型、限位与跳变（7 个）**

| 键 | 默认 | 说明 |
|---|---|---|
| `q_ready` | ARM_CONFIGS（9 轴时前面补两个 0） | 标准姿态；strict 下越界报错，不钳位 |
| `position_bounds` | {} | {关节名: (lo, hi)}，只能收窄 URDF 限位 |
| `gravity_root` | 按安装位姿换算的重力 | URDF 根坐标系下的重力向量（m/s²） |
| `max_joint_jump_rad` | π | 标量 / None / {关节名: rad} |
| `joint_jump_policy` | "reject" | "reject" 剔除超限；"prefer" 全部超限时仍返回 |
| `locked_seed_skip_tol` | 0.02 | q_prev 解的腰部与 q_ready 相差在此以内时跳过锁腰种子；None = v6 顺序 |
| `solve_budget_s` | None | load_mode="off" 时的时间预算 |

**DLS 迭代与次任务（14 个）**

| 键 | 默认 | 说明 |
|---|---|---|
| `dls_max_iter` / `dls_damping` / `dls_max_step` | 500 / 1e-6 / 0.5 | 迭代上限、阻尼、单步 2-范数上限 |
| `dls_characteristic_length` | 0.5 | 位置误差归一化长度（m），DLS 与 strict 复核同一口径 |
| `dls_nullspace_gain` / `dls_waist_null_gain` | 0.30 / 0.50 | 手臂 / 腰部回中增益 |
| `dls_waist_task_weight` / `dls_waist_task_weight_far` | 6.0 / 1.0 | 腰部步长代价（近目标 / 完全释放） |
| `dls_gravity_torque_gain` / `dls_relative_gravity_gain` | 0.04 / 0.0 | 重力矩卸载增益 |
| `dls_com_xy_gain` / `dls_joint_limit_gain` | 0.12 / 0.03 | 质心水平偏移 / 限位排斥增益 |
| `dls_virtual_half_windows` | None | 硬虚拟搜索半窗（以 q_ready 为中心，属于 H2） |
| `dls_posture_use_equivalent_branch` | False | 回中时多圈关节按最近的 2πk 等价分支计算（默认只折叠 continuous 关节） |

**舒适窗、续算与 v13 搜索（12 个）**

| 键 | 默认 | 说明 |
|---|---|---|
| `dls_waist_comfort` | "auto" | "auto" / None / {关节名: 半窗} |
| `dls_waist_comfort_continuation_tol` | (0.1 × 臂展, 0.1 rad) | 续算 / 重新定位判据 |
| `dls_continuation_local` / `dls_continuation_max_step` | True / 0.15 | 续算局部优先及其步长框 |
| `dls_seed_library_size` / `_seeds` / `_min_separation` | 16384 / 3 / 0.5 | 种子库大小（0 关闭）、每阶段种子数、最小关节差 |
| `dls_seed_library_continuation` | False | 续算类求解是否也用种子库 |
| `dls_waist_comfort_early_fallback` / `dls_waist_comfort_patience` | True / 4 | 舒适阶段提前回退 |
| `dls_jump_limit_rescue` / `dls_locked_reach_check` | True / True | 跳变重搜 / 锁腰种子可达性检查 |

**7 轴可选引擎（6 个，只在 `use_direct_psi=True` / `use_ssik=True` 时有效，默认都关）**

`use_local_seeds`（False）、`portfolio_q_ref_mode`（"ready"）、`max_direct_nfev`（150）、`direct_pos_tol` / `direct_ori_tol`
（5e-4）、`direct_psi_tol`（1.5）。

## 5. v14 清理清单

原则：删掉或合并的代码，执行路径、运算顺序、随机数消耗和报错消息都与 v13 相同，由 486 条回放逐位验证。

### 5.1 删除

| 位置 | 内容 |
|---|---|
| solver.py | 未用导入 52 个；`ha_arm_ik.solver` 因此少了 49 个转手名称（名单见第 7 节） |
| solver_dls.py | 未用导入 4 个（`cho_factor`、`cho_solve`、`closest_direction_on_cone`、`normalize`） |
| solver_dls.py | `joint_difference`、`bounded_joint_difference`、`select_closest_bounded_branch` 的本地副本，改用 ik_common 的同一算法（测试验证结果逐位相同） |
| solver_dls.py | 无人读取的兼容常量 9 个：`PIN_IK_STALL_DELTA_M`、`PIN_IK_MAX_VALID_CANDIDATES`、`WAIST_RETURN_TOL`、`WAIST_POST_GAIN/ITERS/STEP/TOL`、`ARM_ROLL_VIRTUAL_LIM`、`MAX_SNS_CLAMP_PASSES` |
| solver_dls.py | `_build_effective_bounds` 的旧腰掩码位置参数（其余参数改为关键字）；`model_com_with_jacobian` 的 `mass_points` 参数；`select_library_seeds` 的 `orientation_weight` 参数；负荷上下文里无人读取的 `multi_turn`、`mass_points` 键 |
| solver.py | 搜索函数里不再使用的 `library_jump_filter` / `library_orientation_weight` 钩子；`__init__` 里只为算有效限位而建的腰掩码 |
| → tests/dls_compat.py | `_compute_nullspace_gradient_and_cost`、`_solve_sns_clamped_step`：只有测试调用的整函数旧入口，测试用它们核对快速路径 |

### 5.2 合并

| 位置 | 合并前 | 合并后 |
|---|---|---|
| solver.py `__init__` | 10 个配置项各自一段校验代码 | `_config_flag` / `_finite_angle` / `_int_at_least`（报错消息逐字不变，校验顺序不变） |
| solver.py 搜索函数 | 三处常规阶段各自组装种子队列 | `regular_queue(lo, hi, rng, *, jump_box=False, checkpoint=False)` |
| solver.py 搜索函数 | 候选判定条件在多处各写一遍 | 统一谓词：`acceptable`、`qualified`、`feasible_but_jump`、`jump_only` |
| solver.py / solver_dls.py | 周期 / 多圈掩码的多层回退写法 | `active_periodic_mask`、`active_multi_turn_mask`、`multi_turn_mask_from_chain` |
| solver.py | 4 处重复的“目标用根坐标系还是本地坐标系”判断 | `self._pin_uses_root_frame` |
| solver.py | solve 与 evaluate_physics 各自切换力矩上限来源 | `_use_confirmed_efforts` / `_use_urdf_efforts` |
| solver.py | 精确解与近似解两处成功收尾 | `_accept(best, status, code, per_seed, update_prev)` |
| solver.py | 四处快照过期检查（_finalize_result 3 处、solve 1 处） | `_stale_context_failure(per_seed=None)` |

### 5.3 保留（有意不删）

- 公共工具 `json_safe`、`pinocchio_q_to_active_q`、`LoadPolicy.cost_gradient`：只有测试在用，但属于公开函数。
- 你原有的接口：`mount_quat`、`workspace_max_radius`、`check_workspace`、`print_workspace_diagnostic`、
  `get_active_joint_efforts`、`unique_add`；`_legacy_shoulder_joint`（未配置肩关节时的回退）。
- SSIK / direct-psi 7 轴可选引擎（约 900 行，默认关，含腰时强制关）：原版就有，删不删要你定（第 7 节）。
- `locked_seed_skip_tol` 在 `__init__` 和搜索函数里各校验一次：两个都是对外入口。

### 5.4 你的三个文件里没被调用的项（只列出，未改）

- ik_candidate.py：`_COMPAT_EXPORTS`、`evaluate_config_quality`、`finalize_ik_result`（后两个有 v9 的测试）。
- ik_common.py：`validate_q_vector`、`periodic_capable_mask`、`shift_to_nearest_branch`、`joint_limit_barrier`、
  `joint_limit_barrier_jac`。

## 6. 验证

- **测试**：194 项，Pinocchio 4.1.0（numpy 2.5.4）和 3.9.0（numpy 2.3.5）下全部通过（`validation_pin*.txt/json`）。
  新增 `tests/test_v14_cleanup.py`（14 项）：公共签名、再导出是同一对象、已删名称确实不存在、分支工具同源且结果一致、
  关键字参数、多圈掩码一致、配置报错消息逐字不变、strict 下力矩上限恢复、SNS 对角 / 矩阵路径逐位一致。
  其余测试只改了调用方式（旧入口改从 `tests/dls_compat.py` 调用、`_build_effective_bounds` 改关键字参数）和
  test_v13 的版本号断言（改用 `S.SOLVER_VERSION`），删掉了 test_v7 里 1 个检查已删常量的断言。
  `tests/reference/solver_v12.py`（v12 solver.py 副本，用于“v13 五项全关 = v12”的测试）只改了它对 solver_dls 的调用：
  3 个已删常量的导入和 3 处关键字参数，逻辑不变。
- **历史基准脚本**：`bench/v8/compare.py`、`bench/v9/compare.py` 在 solver_dls 里找不到整函数次任务入口时改从
  `tests/dls_compat.py` 导入，在 v13 / v14 上算出的数值相同（`diff/bench_v13_to_v14.diff`）。
- **一致性回放**（`bench/v14/equiv_run.py`、`equiv_cmp.py`）：两版各跑同一组场景，486 条记录（含 25 条预期报错），
  浮点逐位比较，忽略耗时、版本号和快照时间戳，差异 0（`equivalence_v13_v14_pin4_1_0.txt`）。
- **计时**（`bench/v14/timing_ab.sh`、`timing_summary.py`，v13 / v14 交替两轮，每轮每模型 400 次一次性求解）：

  | 模型 | 版本 | 解出 | 均值 ms | 中位 ms | p90 ms | 两轮均值 ms |
  |---|---|---|---|---|---|---|
  | two | v13 | 400/400 | 38.8 | 29.8 | 70.7 | 43.9 / 33.6 |
  | two | v14 | 400/400 | 37.2 | 27.9 | 66.8 | 39.7 / 34.7 |
  | wide | v13 | 400/400 | 36.3 | 26.4 | 69.2 | 37.6 / 35.1 |
  | wide | v14 | 400/400 | 36.8 | 27.0 | 71.6 | 38.5 / 35.0 |

  逐目标耗时比中位 1.000 / 0.997，两版的解逐位相同。

## 7. 需要你确认

1. **外部代码如果从 `ha_arm_ik.solver` 导入了转手名称**（例如 `from ha_arm_ik.solver import make_T, W_SMOOTH`），要改成
   从源模块导入。只用 `HAArmIKSolver` 等 7 个公共名称的代码不受影响。49 个名称如下：
   - ik_common（18）：`ORIENTATION_MODES`、`TOOL_CONE_HALF_ANGLE`、`T_origin`、`T_prismatic`、`T_revolute`、
     `joint_limit_barrier`、`joint_limit_barrier_jac`、`joint_limit_penalty`、`joint_margin`、`make_T`、`pose_errors`、
     `project_to_plane`、`rot_axis`、`rotx`、`roty`、`rotz`、`validate_q_vector`、`validate_transform4x4`
   - ik_candidate（11）：`PHI_TIE_TOL`、`W_LIMIT`、`W_NOM`、`W_SMOOTH`、`W_WAIST_RETURN`、`candidate_status`、
     `failure_reason_for_status`、`finalize_ik_result`、`has_valid_torque`、`phi_sort_value`、`status_from_str`
   - solver_dls（13）：`COM_XY_GAIN`、`GRAVITY_TORQUE_GAIN`、`JOINT_LIMIT_GAIN`、`NOISE_SCALES`、`NULLSPACE_ERR_SCALE`、
     `NULLSPACE_GAIN`、`N_NOISE_PER_BASE`、`PIN_IK_STALL_ITERS`、`RELATIVE_GRAVITY_GAIN`、`WAIST_NULL_GAIN`、
     `WAIST_TASK_WEIGHT_FAR`、`json_safe`、`pinocchio_q_to_active_q`
   - robot_model（1）：`parse_vec`
   - 已删除的兼容常量（3）：`PIN_IK_MAX_VALID_CANDIDATES`、`PIN_IK_STALL_DELTA_M`、`WAIST_RETURN_TOL`
   - 标准库（3）：`ET`、`Enum`、`defaultdict`
2. **SSIK / direct-psi 是否删除？** 约 900 行，只支持 7 轴（include_waist=False），默认关；16 自由度方案也用不到它们。
   删除可让 solver.py 降到约 3050 行；如果 7 轴模式还要用，就保留。
3. **数据来自合成模型。** 换成真实 URDF 后建议用 `bench/v14/equiv_run.py` 跑一遍 v13 / v14 对比（应为 0 差异）。

## 8. 16 自由度（腰 2 + 双臂 2×7）设计讨论

### 8.1 问题结构

- q = [腰 2 | 左臂 7 | 右臂 7]；任务是两个末端位姿（full 模式 12 维），冗余 4 维：每只手臂 1 维臂角，加上腰 2 维。
- 两臂只通过腰耦合：J = [[J_Lw, J_L, 0], [J_Rw, 0, J_R]]。腰角给定后，两臂就是两个独立的 7 轴问题。
- 负荷和质心本来就按整机计算：现在 9 轴求解时，另一只手臂按实测角度冻结在负荷上下文里；16 维时它变成活动关节。
- 现有 9 轴单臂求解有一个盲点：腰动时，另一只手的末端在世界坐标系里跟着动。另一只手拿着东西或在做双臂协同时，
  这正是 16 维求解要解决的问题。

### 8.2 三种做法与数据

微基准 `bench/v14/dual_feas.py`：合成双臂 URDF（nq 18 / nv 16），简化 DLS（阻尼最小二乘 + 回溯 + 限位裁剪，没有次任务、
SNS 和负荷），单种子 q_ready，200 组目标。每组目标由限位内随机 16 维关节角做 FK 得到，同一腰角下两臂必然同时可达。

| 做法 | 单种子成功 | 成功者迭代中位 | 每次迭代 | 耗时中位 |
|---|---|---|---|---|
| A 腰 + 左臂 9 轴，单目标（对照：现在的单臂问题） | 84.5% | 10 | 85 µs | 1.0 ms |
| B 16 维联合，双目标 | 61.0% | 19 | 141 µs（1.65×） | 2.7 ms |
| C 主从分解：左臂 9 轴 → 锁腰右臂 7 轴，失败换右臂为主 | 48.0%（左主 68 例 + 换主补救 28 例） | 38 | 146 µs | 6.2 ms |

- B 成功而 C 失败 37 例：主臂选腰角时只顾自己，从臂锁腰后够不到。C 成功而 B 失败 11 例：分解相当于换了一个起点。
- 第三种做法是“腰外层搜索（2 维）+ 两臂各自 7 轴内层求解”。每个腰候选要解两次手臂，没有解析内层时代价是 B 的数倍，
  只适合用来生成种子。
- 单种子成功率不是最终成功率：实际求解是多阶段多种子，9 轴一次性求解目前是 400/400。

### 8.3 推荐设计

1. **主路径：16 维联合 DLS/SNS。** 两个末端各算一次 `_make_task_terms`，残差和雅可比上下堆叠；只有一个末端时走现在的
   代码路径，保证单臂结果逐位不变。SNS、线搜索和停滞检测是按维数写的，可以直接用；180° 对径逃逸要改成按末端分别处理。
2. **腰部逆度量**：两臂各算一个释放系数（肩 → 目标距离 / 臂展、腰偏离 q_ref），取较大者：哪只手需要腰就放开腰。
3. **次任务**：回中和限位裕度扩到 16 维；负荷和质心用整机模型（16 个活动关节都必须在守护列表里，现有校验就是这条规则）。
4. **种子**（每个常规阶段依次）：q_prev → 锁腰（腰锁在 q_ready，两臂 14 维一起解）→ q_ready →
   **主从分解种子**（难的那只手臂 9 轴解，另一只锁腰 7 轴解，拼成 16 维后联合精化；离锁腰可达界更远的一侧算难臂）→
   **组合种子库** → 随机扰动。
   组合种子库：每只手臂一个 7 维库（末端位姿相对躯干），腰候选取 q_prev / q_ready / 舒适窗网格点，两臂各自查最近样本后
   拼接，按两臂任务距离之和排序、去重。不做 16 维联合采样：给定腰角后两臂独立，联合采样要求两只手同时落在目标附近，
   命中率大约是单臂的平方。
5. **约束与阶段**：H1–H4、P1、S1–S2 全部沿用，按 16 维逐关节判断。两臂都是小位移才算续算，任一臂重新定位就按重新定位
   处理。跳变框和续算框变成 16 维盒约束。
6. **择优**：每只手臂各调一次 `score_candidate`，腰回中和腰平滑只算一次。建议按 7 维（躯干坐标系）给每只手臂打分，
   这样臂角是“肘相对躯干”的量，不随腰转动变化。现在 9 轴单臂的臂角在根坐标系里算，单臂模式保持不变。
7. **结果**：q 为 16 维（腰 2、左 7、右 7，按 URDF 活动关节顺序）。IKResult 增加分臂字段（每臂 pos_err / ori_err /
   psi / phi / margin / status），总 status 取两臂中较差的；`last_load_diagnostics` 增加 `per_arm`。
8. **代码组织**（不新增模块和 Solver 类）：HAArmIKSolver 内部把单臂相关的状态（末端帧、SerialChain、q_ready 切片、
   psi_ready、锁腰可达界、种子库）收进按臂的记录，单臂模式只有一项。`collect_pinocchio_candidates_and_seeds` 现在 782 行，
   先按阶段拆成内部函数（续算 / 常规 / 跳变重搜 / 不加框重搜 / 诊断），拆完单独做逐位一致验证，再加双臂。

### 8.4 接口提议

- **方案 A（推荐）**：`HAArmIKSolver(urdf, side="both", include_waist=True, ...)`，新增
  `solve_dual(T_left, T_right, *, frame, q_prev, update_prev, allow_approximate, orientation_mode, tool_axis,
  tool_cone_half_angle, joint_state, mode)`，返回 `(q16 或 None, status, IKResult)`。`orientation_mode` / `tool_axis` /
  `tool_cone_half_angle` 可以给一个值（两臂相同），也可以给 (左, 右)；`mode` 是失败语义（8.6 节）。
  原 `solve()` 和 side="left" / "right" 完全不变。
- **方案 B**：`solve()` 的 `T_target` 接受 `{"left": T, "right": T}`。签名不变，但同一个参数有两种含义。
- 每只手臂的目标可以是 4×4 位姿；`"hold"`：保持该臂末端当前的世界位姿，腰动时这只手要补偿；或 `None`：不给目标，
  该臂只做回中和连续性，末端随腰动。

### 8.5 性能估计

每次迭代约 1.4–1.7×（微基准 1.65×），迭代数约 2×，每个种子约为 9 轴的 3 倍（约 11 ms → 30–35 ms）。
预计典型耗时 60–100 ms，要实测确认。把单种子循环编译化（预计 3–5×）后可以回到 20–30 ms 量级。

### 8.6 需要你决定

1. **接口**：方案 A（新增 `solve_dual`）还是方案 B（`solve` 接受双目标）？
2. **失败语义**：
   - `both_required`（建议默认）：两臂都满足 H1–H4 才算成功，失败码按臂给出；
   - `primary_priority`：主臂必须达标，另一臂尽量接近（在主臂任务的零空间里优化），要给 SNS 加一级任务优先级，建议第二步做；
   - 联合失败时是否自动退回单臂（只解主臂、另一臂 hold）：不建议默认开启，容易掩盖问题。
3. **只有一只手臂有目标时，另一只怎么处理**：hold 末端世界位姿、保持关节角，还是不管？
4. **双臂互碰**：继续全部交给 MoveIt，还是在候选过滤里加两臂胶囊体最小距离检查（每个候选约 0.1 ms 量级）？两只手的
   目标靠得近时，16 维解出两臂交叉的概率明显变大。
5. **臂角参考系**：双臂模式按躯干坐标系（建议），还是与现在 9 轴一致用根坐标系？
6. **时间预算**：`load_budget_s` / `solve_budget_s` 的默认 0.5 s 是否提到 1.0 s？
7. **SSIK / direct-psi 去留**（同第 7 节第 2 条）。

### 8.7 实施步骤（每步单独验证）

1. 重构，不改行为：拆分搜索函数、引入按臂记录；486 条回放逐位一致。
2. 数值核心：多末端任务项、腰释放取较大者；测试雅可比有限差分、16 维 SNS 不越界、单末端逐位不变。
3. 入口与择优：side="both"、`solve_dual`、分臂打分与诊断、strict 整机负荷。
4. 种子：主从分解种子、组合种子库。基准：成功率和耗时对比主从分解基线；路点跟踪（两臂同时动、一臂 hold）。
5. 失败语义扩展（primary_priority、hold）和可选的互碰检查。
6. 真实 URDF 复测。

## 9. 版本记录

| 版本 | 改动 |
|---|---|
| 原版 | SerialChain FK；Pinocchio DLS（SNS 限位钳制、零空间回中 / 重力矩 / 质心、腰部释放，默认开）；direct-psi 与 SSIK（默认关）；重力矩与质心评估；左 / 右臂；含腰 9 轴（include_waist） |
| v5（你的版本） | strict 负荷保护框架（冻结快照、整机负荷门控、最终复核、上下文变化检测、单实例锁）；strict 下未传 q_prev 时用实测关节角；2πk 分支对齐与零空间回中分开处理（multi_turn_mask）；显式虚拟搜索窗 `dls_virtual_half_windows`；`load_budget_s` 与 `solve_budget_s` 分开 |
| v6 | 约束分层 H1–H4 / S1–S2；不绕大圈；失败码细分；整机质心计入根 link；SNS 零空间步与 SVD 回退一致；种子内按峰值容差带 + 代价择优；180° 对径按雅可比可达方向逃逸；停滞检测；种子顺序 q_prev → q_ready → 由大到小的扰动；`load_min_seeds` 3 → 2、`load_min_unlocked_seeds` 2 → 1 |
| v7 | 提速：每种子预计算、次任务复用主任务的 Pinocchio q、对角 SNS 向量化、只更新末端帧、LAPACK Cholesky；停滞窗口 25 → 12、阈值 5% → 2.5%；q_prev 先行 + 等价锁腰跳过（`locked_seed_skip_tol`）；早停阈值 min(0.70, soft_ratio) |
| v8 | 腰部舒适窗 auto（重力矩推导，只用于重新定位）；帧查询按 BODY；DLS 容差取 min(PIN_IK_*, ik_common.*)；臂展与肩关节沿运动链累加；load_mode=off 的质心参考含根质量；姿态整理恢复 50 步 / 耐心 3 |
| v9 | 接入真实 ik_common / ik_candidate / robot_model（修 11 处）；续算局部优先（`dls_continuation_local` / `_max_step`）；跳变上限前移为搜索框 + `unboxed` 重搜；psi_ready 无定义时告警并排除臂角项 |
| v10 / v11 | 实验分支：权重实验（结论：权重不调）、负轴向 continuous 关节修正、扰动噪声镜像。未合入 |
| v12 | robot_model.py 右臂 q_ready 直接写数值；solver 与 v9 相同 |
| v13 | 任务空间种子库；续算类求解不用种子库；舒适阶段提前回退；跳变重搜（`jump_box`）；锁腰种子可达性检查。五项可单独关闭，全关 = v12 |
| v14 | 本文：冗余清理、重复代码合并、文档改为架构说明；结果与 v13 逐位一致 |

## 10. 文件

- `ha_arm_ik/solver.py`、`ha_arm_ik/solver_dls.py`：v14；其余三个源文件与 v13 相同。
- `diff/solver_v13_to_v14.diff`、`diff/solver_dls_v13_to_v14.diff`、`diff/tests_v13_to_v14.diff`（含 run_validation.py）、
  `diff/bench_v13_to_v14.diff`。
- `baseline/solver_v13.py`、`baseline/solver_dls_v13.py`：v13 原文件。
- `tests/test_v14_cleanup.py`、`tests/dls_compat.py`；`run_validation.py` 已加入新测试模块。
- `bench/v14/`：`equiv_run.py` / `equiv_cmp.py`（一致性回放）、`timing_ab.sh` / `timing_summary.py`（交替计时）、
  `dual_feas.py`（16 自由度微基准）及对应结果 `*_pin4_1_0.txt`。