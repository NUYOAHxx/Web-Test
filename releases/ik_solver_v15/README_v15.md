# ha_arm_ik merged-v15：单臂续算漂移修复 + 逐位不变提速；双臂 dual-p2.1（单手目标时另一只默认 hold）

入口和签名都没变：`HAArmIKSolver(...)` / `solve(...)`，`DualArmIKSolver.solve(...)`。位姿容差也没变，仍是 0.5 mm / 0.5 mrad（`POS_TOL` / `ORI_TOL`）。

## 1. 结论

1. **漂移修好了。** 目标不动、`q_prev` 已经是解时，v15 原样返回：20 次调用里关节变化都是 0；v14 每次仍会变 0.10–0.15 rad。
   目标每次平移 1.2 mm 时，v15 每次变 0.002–0.003 rad，2 次迭代；v14 的前 3–7 次每次变 0.10–0.15 rad。
2. **行为上只改了续算。** 用 486 条固定场景回放核对：
   - v14 模式（`CONTINUATION_POSTURE_SETTLE = True`）与 v14 逐位相同，0 差异；
   - 默认模式只有 121 条跟踪记录变了；一次性、重定位、失败路径、配置报错、7 轴、direct-psi 全部逐位相同。
3. **提速 1.31–1.33×，结果逐位不变。** 用 400 个重定位目标做 ABBA 交替计时，两版 400/400 的解逐位相同：
   - 腰 + 左臂（strict）：均值 51.2 → 39.0 ms，中位 40.2 → 30.3 ms；
   - 合成 9 轴：均值 46.2 → 34.6 ms；
   - 这台机器今天比上次慢。上次 v14 中位 24.9 ms，按同一比例折算，v15 约 19 ms；
   - 续算本身也更快了：跟踪 10.6 → 7.5 ms，静止目标 11.9 → 4.3 ms。
4. **腰部偏好保留。** 舒适窗、锁腰优先、腰回中代价、早停规则都没改，重定位四类的质量与 v14 逐值相同（dH = dC = 0）。
   会改变择优的早停 A（腰离 `q_ready` 更远）没有采用。
5. **双臂 dual-p2.1：** `None` 现在等于 `"hold"`；新增 `"free"` 表示不约束；两只手都是 `None` 时报错。
6. **测试 250 个全部通过，其中新增 16 个。** 环境：Pinocchio 4.1.0，numpy 2.5.4，scipy 1.18.1。

## 2. 续算漂移修复（行为改变）

### 2.1 原因

目标在 FK(q_prev) 附近时，单臂先在 `q_prev ± 0.15 rad` 的续算框里求解（v9 起）。v14 的续算和重定位一样，位姿达标后还会在零空间里整理姿态（回中、限位裕度、负荷），一直整理到 SETTLED。

所以即使 `q_prev` 已经是解，这一步也会把关节往“更好”的姿态推，每次 0.10–0.15 rad（被续算框截住）；下一次调用再从新位置接着推。伺服循环里反复调用，就等于机器人自己在动。

### 2.2 改法（与双臂续算同一规则）

1. **续算阶段的 DLS 不再做达标后的姿态整理**（新参数 `posture_settle=False`）。
   - 迭代中第一个同时满足“位姿达标、负荷合格（strict）、离限位 ≥ SAFE_MARGIN（0.02 rad）”的点直接返回，`stop_reason = "POSE_REACHED"`。
   - 种子本身就满足时，1 次迭代原样返回。
   - 达标点贴限位或超负荷时照常整理，一满足条件就停。
2. **未达标时给零空间步限幅：** `|w·D·z| ≤ NO_SETTLE_NULL_STEP_RATIO × |e|`，比例取 2.0。
   只做第 1 条不够：目标移 1.2 mm 时，收敛的那两三次迭代里零空间步仍能到 0.11–0.15 rad，漂移只是换了个时机。限幅后，姿态调整量与手的移动量成正比。
3. **continuation hold：** 随动续算的解就是 `q_prev` 本身且可接受时，直接返回，不再跑锁腰续算。
   诊断字段：`last_load_diagnostics["continuation"]["hold"]`、`["search"]["continuation_hold"]`。
4. **重定位不受影响。** 目标不在 FK(q_prev) 附近时照常整理姿态。
   开关 `solver.CONTINUATION_POSTURE_SETTLE`：False（默认）是 v15 行为；True 是 v14 行为，结果逐位相同。

限幅比例的取值依据（`bench/v15`）：

| 比例 | 目标每次移 1.2 mm：单次最大关节变化 | 腰部斜坡 40 帧：关节路程 |
|---|---|---|
| **2（默认）** | **0.0033 rad** | **2.77 rad** |
| 5 | 0.0065 | 2.85 |
| 10 | 0.0127 | 2.92 |
| 30 | 0.0375 | 3.00 |
| 不限（只做第 1 条） | 0.150（40 次里 18 次 > 0.05） | 3.42 |

### 2.3 数据

| 场景 | v14 | v15 |
|---|---|---|
| 静止目标，4 组 × 5 次（`bench/v15/drift.py`） | 每次 0.10–0.15 rad（均值 0.102），中位 11.9 ms | 20/20 次变化为 0（hold），中位 4.3 ms |
| 目标每次移 1.2 mm，4 组 × 10 次 | 前 3–7 次每次 0.10–0.15 rad，40 次里 19 次 > 0.05 | 每次 0.0018–0.0033 rad，每次 2 次迭代 |
| 路点跟踪 119 帧（`bench/v9` tracking） | H 0.1061，C 3.47，单步最大 0.020，均值 10.6 ms | H 0.1062，C 3.46，单步最大 0.020，均值 7.5 ms |
| 腰部大幅跟踪 239 帧（tracking_waist） | H 0.4335，C 2600，单步最大 0.150，p99 0.150，均值 7.2 ms | H 0.4331（−0.09%），C 2641（+1.57%），单步最大 0.105，p99 0.081，均值 5.4 ms |
| 腰部斜坡 40 帧（`bench/v15/ramp.py`） | 第 5 帧腰开始动（单步 0.137），之后每帧 0.024–0.045；关节路程 1.35 rad | 前 30 帧腰保持 `q_ready`，手臂补偿，每帧 0.05–0.13；之后每帧 0.04–0.07；关节路程 2.77 rad |
| TRACK_38（贴限位帧） | NEAR_LIMIT，单步 0.051 | 相同 |

- **tracking_waist 的 C +1.57%** 全部来自腰回中项（Cw +1.6%）。v15 只在手移动时回中，所以回得慢一点。
- **腰部斜坡：** v15 让腰一直留在 `q_ready`，直到手臂够不着才放开，这与重定位的锁腰优先一致；v14 是靠姿态整理在第 5 帧就把腰带走了。
  - 两版单步最大分别是 0.131 / 0.137 rad，≥ 0.1 的都只有 1 帧。
  - 代价是这一段手臂动得更多：40 帧关节路程 2.77 vs 1.35 rad（见第 7 节第 2 条）。
  - 限幅比例不影响腰开始动的时刻（各比例都是第 31 帧）。

### 2.4 影响范围（486 条回放，默认模式）

| 组 | 变化记录 |
|---|---|
| two_left / two_right 路点跟踪 | 39/40，39/40 |
| two_left_jump1 跟踪（跳变上限 1 rad） | 16/20 |
| two_left_off / two_left_explicit / wide 跟踪 | 10/12，9/12，7/12 |
| two_local | 1/1 |
| 其余：一次性、重定位、失败路径、配置报错、7 轴、direct-psi、物理评估、工作空间 | 0 |

v14 模式下 486 条都是 0 差异。脚本：`bench/v15/equiv_run_v14mode.py` + `bench/v14/equiv_cmp.py`。

## 3. 提速（结果逐位不变）

| 改动 | 位置 | 单次耗时（同机背靠背，腰 + 左臂 60 个目标） | 等价性验证 |
|---|---|---|---|
| SNS 钳制步改写对角逆度量热路径：首轮不做减法，全部关节自由时不做布尔索引，越界检查用标量 | `solver_dls._sns_clamped_core` | 113 → 74 µs（每次求解约 91 次） | 保留 v14 原实现 `_sns_clamped_core_reference`；捕获的真实调用和 600 个随机问题逐位对照 |
| 次任务求值：常量预计算，限位项按关节做标量运算，有限性检查合并到末尾（失败时按原顺序重查，报原消息） | `_SecondaryObjective` | 93 → 63 µs（每次求解约 113 次） | 486 条回放 |
| 任务项不再拷贝 SE3；q 构造器改用连续切片赋值；主循环里的边界吸附、达标判定做了小改写 | `_make_task_terms` / `_make_q_builder` / `pinocchio_ik_one_seed` | — | 486 条回放 |
| 链 FK：关节原点变换按 (xyz, rpy) 缓存（只读），整条链一次批量调用 `from_rotvec` | `robot_model.SerialChain.fk_all / fk_pose_and_joints` | 612 → 203 µs（每次求解约 3.3 次） | 20012 例逐位相同 |
| 刚体校验：`np.allclose` / `isclose(rtol=0)` 换成等价的 `abs(x − ref) <= atol` | `solver_dls.validate_transform`、`ik_common.validate_transform4x4` | 111 → 25 µs，119 → 22 µs | 40003 例的判定、异常文本、告警都相同 |
| 连续索引判定 | `solver_dls._contiguous_range` | — | 20017 例 |

整体效果（`bench/v15/ab_coord.py`，ABBA 交替，同一时刻只有一个进程在算，单位 ms）：

| 模型 | 模式 | v14 中位 / 均值 / p90 | v15 中位 / 均值 / p90 | 加速 | 解逐位相同 |
|---|---|---|---|---|---|
| 腰 + 左臂（strict） | full ×200 | 43.0 / 57.6 / 94.0 | 32.8 / 43.6 / 71.9 | 1.32× | 200/200 |
| | tool_dir ×100 | 29.2 / 36.1 / 63.7 | 20.7 / 27.6 / 49.0 | 1.31× | 100/100 |
| | tool_cone ×100 | 43.8 / 53.7 / 102.0 | 32.9 / 41.3 / 79.5 | 1.30× | 100/100 |
| 合成 9 轴 | 全部 ×400 | 37.0 / 46.2 / 90.2 | 27.7 / 34.6 / 66.5 | 1.33× | 400/400 |

## 4. 双臂 dual-p2.1

| 每只手的值 | 含义 |
|---|---|
| 4×4 / SE3 | 到这个位姿 |
| `"hold"` | 保持 `q_prev` 处的世界位姿 |
| `"keep"` | 该臂关节角保持 `q_prev` |
| `"free"`（新增） | 不约束，只回中（dual-p2 的 `None`） |
| `None` | 同 `"hold"`（默认） |

- 两只手都是 `None` 时报 `ValueError`；要两手都不动，请显式传 `("hold", "hold")`，此时原样返回 `q_prev`。
- 至少一只手要有位姿或 `"hold"`，否则报错，例如 `("keep", "free")`、`("free", "free")`。
- **升级注意：** 以前传 `None` 表示不约束的调用，请改成 `"free"`。
- 求解逻辑没改，`README_dual.md` 第 9 节的性能数据仍然有效；`DUAL_SOLVER_VERSION = "dual-p2.1"`。

## 5. 改动清单

| 文件 | 改动 |
|---|---|
| `ha_arm_ik/solver.py` | 新增 `CONTINUATION_POSTURE_SETTLE`；`run_stage(posture_settle=...)`；continuation hold 与诊断；`SOLVER_VERSION = "merged-v15"`；文件头 |
| `ha_arm_ik/solver_dls.py` | 新增 `NO_SETTLE_NULL_STEP_RATIO`；`pinocchio_ik_one_seed(..., posture_settle=True, margin_bounds=None)`（默认值即 v14 行为）；第 3 节的提速 |
| `ha_arm_ik/robot_model.py`（你的文件） | `SerialChain` 新增 `_origin_T` / `_unit_axis` / `_joint_motions`，`fk_all` / `fk_pose_and_joints` 改用它们；新增导入 `Rotation`、`make_T`、`normalize` |
| `ha_arm_ik/ik_common.py`（你的文件） | `validate_transform4x4` 的比较写法；新增模块常量 `_RIGID_BOTTOM_ROW`、`_EYE3` |
| `ha_arm_ik/ik_candidate.py` | 未改 |
| `ha_arm_ik/dual_arm_ik.py` | `_parse_targets`（`None` → `"hold"`，新增 `"free"`，两只都 `None` 报错）；文档；版本号 |
| `tests/test_v15_changes.py`（新） | 16 个：静止 hold 逐位相等、v14 模式确有漂移、小步成比例、DLS `posture_settle` 语义（种子即解原样返回、贴限位继续整理、`margin_bounds` 定义与校验、零空间限幅）、刚体校验 / 链 FK / `_contiguous_range` / SNS 与 v14 写法逐位对照、版本号 |
| `tests/test_v6 / v8 / v9 / v13 / v14` | 涉及续算的断言改为 v15 行为，另用 `mock.patch(CONTINUATION_POSTURE_SETTLE=True)` 保留 v14 断言；版本号断言 |
| `tests/test_dual_arm_ik.py` | hold / free / None 的新语义；`test_not_jointly_reachable_by_diagnosis` 预算 0.1 → 0.3 s |
| `bench/v15/` | `drift.py`、`ramp.py`、`ab_coord.py` + `ab_worker.py`、`equiv_run_v14mode.py`，以及结果 `*_pin4_1_0.txt` |
| `bench/dual/bench_dual.py`、`run_validation.py`、`README_dual.md` | 说明与模块列表 |
| `diff/`（新增 6 个） | 相对 dual-p2 的逐文件差异：`solver` / `solver_dls` / `robot_model` / `ik_common` / `dual_arm_ik`，测试与脚本合并为 `tests_v14_to_v15.diff` |

## 6. 验证

- `python run_validation.py`：250 个测试全部通过，结果见 `validation_pin4_1_0.txt/.json`。
- 回放：v14 模式 486 条 0 差异；默认模式 121 条变化，全部是跟踪记录（第 2.4 节）。
- 计时：两种模型 400/400 的解与 v14 逐位相同。
- `test_not_jointly_reachable_by_diagnosis` 原来用 0.1 s 预算，诊断窗口只有 17 ms，机器慢或有 CPU 争用时偶尔会报 TIME_BUDGET_EXCEEDED（这本身也是正确行为）。现在改成 0.3 s，判据和断言都不变。

## 7. 需要你确认

1. **早停 B**（锁腰种子先跑，首个合格种子即停）：保留腰部回中偏好，上次实测能再快 13%，但 200 个解里有 28 个会变。要不要在 v15 基础上加？如果加，建议做成配置项，默认关。
2. **腰部斜坡这类由腰部运动生成的轨迹：** v15 的腰要等手臂够不着才动，手臂路程约为 v14 的 2 倍，单步仍 ≤ 0.131 rad。
   要不要让腰更早分担？那需要在续算里恢复一部分姿态整理，会带回少量漂移（与恢复的整理量成正比）。
3. **配置同义名合并：** 还没做。要不要放进下一版？
4. **你的两个文件** `robot_model.py` / `ik_common.py` 有提速改动，结果逐位不变，约省 1.6 ms/次（约 4%）。如果需要它们保持原样，可以回退。
5. **P3（双手持同一物体）：** 需要你提供物体质量、重心和抓持刚度。
