# `ha_arm_ik` (merged-v15) 注释修复与逐位等价性能优化报告

本报告汇总了对 `releases/ik_solver_v15/ha_arm_ik/` 下 5 个核心源码模块（`solver.py`、`solver_dls.py`、`dual_arm_ik.py`、`robot_model.py`、`ik_common.py`）的注释同步修复与热路径提速改动。所有性能优化均严格遵循**浮点逐位等价（bit-exact）**原则，不改变任何接口签名、配置项校验或求解行为。

---

## 1. 验证结论

- **全量测试通过**：`run_validation.py` 共 250 项单元测试全部通过（0 失败，0 错误），包括公共接口签名检查、AST 结构检查（`DirectPsiProblem` / `solve_direct_psi` / `collect_ssik_candidates_and_seeds`）、v6–v15 回归测试及双臂 `test_dual_arm_ik.py`（40 项）。
- **486 条固定场景回放 0 差异**：通过 `bench/v14/equiv_run.py` 与 `bench/v14/equiv_cmp.py` 对照优化前后的 486 条全量场景记录（涵盖一次性求解、路点跟踪、跳变限制、7 轴 direct-psi、失败路径、物理评估、工作空间诊断及配置报错），规范化后逐位比较差异为 0（`records 486 vs 486; differing 0`）。

---

## 2. 性能优化清单与实测数据

| 优化模块与函数 | 问题原因（优化前） | 改进方案（优化后，结果逐位不变） | 单次调用耗时对比 | 加速比 |
| :--- | :--- | :--- | :--- | :---: |
| **`solver.py`**<br>`_check_load_candidate` &<br>`add_torque_metrics_to_candidates` | `strict` 模式下，候选在 `run_stage` 入池时已算过一次 `load_physics` 与 `dls_task_terms`，但在 `_filter_load_candidates` 中对每个候选重复计算第 2 次，`add_torque_metrics_to_candidates` 中又重复调 `computeGeneralizedGravity` 计算第 3 次 | 在单次 `solve` 的 `_load_context` 内按 `q.tobytes()` 缓存 `run_stage` 已算出的 `load_physics` 与未平移分支的位姿误差供 `_filter_load_candidates` 过滤复用，并在 `add_torque_metrics_to_candidate` 中直接复用 `load_guard_efforts`；最终胜出的 `best` 候选在 `_finalize_result` 中仍执行完整的独立重算复核 | `1995.2 µs` → `147.5 µs`<br>（5 候选池） | **13.52×** |
| **`solver_dls.py`**<br>`build_seed_library` &<br>`select_library_seeds` | 每次查库都对 `16384 × 9` 矩阵全量执行 `(Q >= lower) & (Q <= upper)`，且在 `full` 阶段（全部样本均在限位内）仍通过 `idx` 花式索引拷贝 `16384 × 3 × 3`（约 1.57 MB）的 `P` 与 `R` 数组 | 建库时预存只读的列极值 `_q_min` / `_q_max` 与转置视图 `_qt`；查询时只检查比采样范围更紧的关节列，当所有样本均在限位内时直接使用只读原数组，免去 1.57 MB 拷贝与全表布尔扫描 | `full` 阶段：`6715.2 µs` → `1685.6 µs`<br>`comfort` 阶段：`2028.7 µs` → `1317.2 µs` | **3.98×**<br>**1.54×** |
| **`dual_arm_ik.py`**<br>`validate_transform` &<br>`_ArmLibrary.inside` | 单臂 v15 已将 `validate_transform` 的 `np.allclose` / `isclose` 替换为 `abs(x - ref) <= atol`，但双臂仍保留 `np.allclose` 慢路径；`_ArmLibrary.inside` 逐关节过滤时分配多余临时布尔数组 | 将 `dual_arm_ik.validate_transform` 同步为与单臂逐位等价的 `(np.abs(...) <= tol).all()` 写法；`_ArmLibrary.inside` 改用原位 `np.logical_and(..., out=ok)` | `112.2 µs` → `26.8 µs` | **4.18×** |
| **`robot_model.py`**<br>`SerialChain.psi` | `psi()` 连续调用 3 次 `self.joint_position(q, ...)`，每次内部都完整重跑一遍 `self.fk_all(q)`（共算 3 遍整链 FK） | 改为单次调用 `self.fk_all(q)` 同时提取 `S` / `E` / `W` 三个关节原点位置，保持缺关节时的 `KeyError` 顺序与异常文本完全一致 | `512.0 µs` → `246.9 µs` | **2.07×** |
| **`ik_common.py`**<br>`clip_to_bounds` | 每次调用重复执行 `np.asarray(lower, dtype=float)` 与 `np.asarray(upper, dtype=float)` | 合并数组转换并在存在紧约束关节时复用已转换数组 | 消除冗余转换 | — |

---

## 3. 注释与文档问题修复清单

| 文件 | 位置 | 原注释问题 | 修复内容 |
| :--- | :--- | :--- | :--- |
| **`solver.py`** | `collect_pinocchio_candidates_and_seeds` docstring | `continuation` 阶段说明仍停留在 v14，未反映 v15 默认不做达标后姿态整理（`CONTINUATION_POSTURE_SETTLE=False`）及随动续算解等于 `q_prev` 时直接触发 `continuation_hold` 短路（跳过锁腰续算）的行为 | 补充 `CONTINUATION_POSTURE_SETTLE=False` 与 `continuation_hold` 短路说明，并统一函数内 `comfort` / `full` 阶段注释命名 |
| **`solver.py`** | `DirectPsiProblem.__init__` | `w_limit` 与 `limit_margin` 存入实例属性后未在残差或雅可比中使用，缺少说明 | 添加 `#` 行内注释说明二者为兼容旧构造签名保留，实际关节限位由 `least_squares` 的 `bounds` 约束（保持 AST 完全不变） |
| **`solver_dls.py`** | 模块顶层 docstring | 单种子迭代第 3 步与返回值说明仍只描述 `posture_settle=True`，且版本演进只指向 `README_v14.md` | 增补 `posture_settle=False`（未达标零空间步限幅 `NO_SETTLE_NULL_STEP_RATIO × |e|`、首达标合格点 `POSE_REACHED` 直接返回）说明，并补充 `README_v15.md` 引用 |
| **`dual_arm_ik.py`** | 模块头与 `solve` 续算注释、`_Hand`、`_ArmLibrary.best`、`_evaluate` | 将“续算不做达标后姿态整理、静止不漂”误标为“单臂 v9 规则”（实为双臂首创、单臂 v15 才引入）；`_Hand` 和 `_evaluate` 将任务维度硬编码描述为“两手 12 行”；距离缓冲注释写为“8 MB” | 修正为“单臂 v9 续算框 + v15 防漂移规则”；将任务描述改为“受控手（6 × 受控手数，最多 12 行）”；缓冲注释精确为“约 8.6 MB” |
| **`ik_common.py`** | 模块头、`multi_turn_mask_from_chain`、`target_frames` | 存在括号缺失（`revolute 关节与 production solver的 ... 语义一致）。`）及中英文紧贴无空格 | 补齐缺失的左括号并规范中英文空格排版 |
