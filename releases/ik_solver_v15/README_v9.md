# ha_arm_ik merged-v9：接入真实 ik_common / ik_candidate / robot_model + 路点跟踪修正

- 交付 5 个源文件：`ha_arm_ik/solver.py`（改）、`ha_arm_ik/solver_dls.py`（与 v8 相同，未改）、`ha_arm_ik/ik_common.py`、`ha_arm_ik/ik_candidate.py`、`ha_arm_ik/robot_model.py`（即你上传的 `robot.py`；solver 与 ik_candidate 都按 `ha_arm_ik.robot_model` 导入，所以按这个文件名放）。入口 `HAArmIKSolver(...).solve(...)` 和公开函数签名不变，只新增两个可选配置项。
- `ha_arm_ik/__init__.py`、`tests/`、`bench/` 是离线脚手架。v9 起测试不再用替身，全部跑你提供的三个真实文件（v8 的替身留档在 `baseline/tests_stubs_v8.py`）。
- `diff/solver_v8_to_v9.diff` 是 solver 相对 v8 的改动；`diff/*_user_to_v9.diff` 是三个文件相对你上传原件的改动，原件在 `baseline/user_v9_input/`。

## 0. 结论

1. **换成真实 ik_candidate 后，路点跟踪会在两个分支间来回切换。** `W_WAIST_RETURN = 2000` 远大于 `W_SMOOTH = 8`，v8 每帧在全部种子里按代价择优：腰部能回 q_ready 就回（锁腰解），回不了就跟着目标走（随动解）。实测：
   - 腰部大幅跟踪（239 个路点）：单步跳变最大 **2.47 rad**，超过 0.3 rad 的有 7 次；
   - 同一轨迹设 `max_joint_jump_rad=0.25`：切换点被跳变上限拒绝，q_prev 不再更新，后面的路点全部失败，**只成功 9/239**（每帧 157 ms）；
   - 普通跟踪（腰部目标不动）：在负荷容差带边界来回切换，目标每帧只动 0.01 rad，解却跳 **0.198 rad**。
2. **v9 的修正：续算局部优先 + 跳变上限前移到搜索阶段**（第 1、2 节）。一次性目标与 v8 逐值相同；跟踪：

   | 指标 | v8 | v9 |
   |---|---|---|
   | 腰部大幅跟踪：单步最大 / 超过 0.3 rad 的次数 | 2.470 / 7 | **0.150 / 0** |
   | 同上 + `max_joint_jump_rad=0.25`：成功数 | 9/239 | **239/239** |
   | 腰部大幅跟踪：耗时（成对计时） | 13.4 ms | **5.6 ms（−58%）** |
   | 普通跟踪：单步最大 / 择优代价 C | 0.198 / 4.14 | **0.020 / 3.47（−16%）** |

   代价：腰部大幅跟踪的 C 高 1.5%（腰不再一步弹回 q_ready），次任务代价 H 低 7.6%。
3. **三个文件里修了 11 处问题**（第 3 节），影响最大的三处：psi_ready 为 NaN 时所有候选代价都是 inf，择优退化；`clip_to_bounds` 在跨度 < 2·eps 的关节上返回越界值；`evaluate_config_quality` 把 continuous 关节直接写进 Pinocchio q（实际是 cos/sin 两个分量），算出的是另一个构型。
4. **需要你确认**的事项见第 6 节，最重要的是右臂 q_ready 的符号。

## 1. 续算局部优先（`dls_continuation_local=True`，默认）

**判据**：沿用 v8 舒适窗的“续算 / 重新定位”判据，即目标相对 FK(q_prev) 的主任务误差。位置 ≤ 0.1 × 臂展且姿态 ≤ 0.1 rad 算续算（路点跟踪），否则算重新定位。重新定位的求解与 v8 完全相同。

**续算阶段**（先于常规搜索）：

- **随动续算**：种子 q_prev，搜索范围 = 有效限位 ∩ q_prev ± `dls_continuation_max_step`（默认 0.15 rad），DLS 单步上限 0.5 × 该值（从框中心出发，首步不会越框；越框会被 SNS 钳制，线搜索停住）。
- **锁腰续算**：腰锁在 q_ready，手臂从 q_prev 出发。只在 q_prev 腰部离 q_ready 不超过 0.15 rad 时运行，因为更远时它不可能是局部解。锁腰解负荷超过 soft_ratio 时释放腰部，并在框内重跑随动续算（与常规的 q_prev_release 对应）。
- **局部解** = 满足全部硬约束且单关节步长 ≤ 0.15 rad 的候选。有局部解时，只在局部解里按原择优规则（负荷容差带 + ik_candidate 代价）选；没有时进入 v8 常规搜索，续算候选留在池中。bench 中这种回退 0 次。

**为什么要框，而不只是调整种子顺序**：DLS 的零空间整理（回中 / 负荷 / 质心）在一次求解里会把姿态推动 0.2–0.8 rad，而目标只动了 0.01–0.04 rad。框把每帧的姿态变化限制在 0.15 rad 以内；必要的分支过渡（例如锁腰解够不着、开始用腰）会分摊到相邻几帧。bench 中 239 帧里有 5 帧贴框。

**为什么锁腰续算不按 v7 规则跳过**：v7 的规则是“随动解腰部 ≈ q_ready 就跳过锁腰种子”，但腰部等价不代表手臂等价。实测保留这条规则时，普通跟踪会在某一帧被零空间整理带离锁腰分支，之后再也回不来，C 达到 18.9（+363%），而且每帧都贴框。

**0.15 rad 的取值依据**（真实 ik_candidate，bench/v9）：

| `dls_continuation_max_step` | 普通跟踪：单步最大 / C | 腰部大幅跟踪：单步最大 / H / C |
|---|---|---|
| 0.1 | 0.020 / 3.47 | 0.100 / −9.0% / +2.2% |
| **0.15（默认）** | **0.020 / 3.47** | **0.150 / −7.6% / +1.5%** |
| 0.2 | 0.020 / 3.47 | 0.200 / −7.4% / +1.5% |
| 0.3 | 0.198 / 4.12（随动解又在容差带边界切换） | 0.300 / −7.9% / +1.6% |

0.1 时，v8 测试场景（腰部斜坡 41 帧）里锁腰 / 随动在框边上反复切换：单步均值 0.081 rad，p90 贴框 0.100（0.15 时为 0.034 / 0.052）。0.15 是测试档位里不出现这种反复切换的最小值；再大，每帧允许的姿态变化就更大，0.3 时普通跟踪又会出现容差带切换。

**否决的方案**（数据见 `bench/v9`）：

- 只在腰部 ≈ q_ready 时提供锁腰候选：腰部大幅跟踪单步最大 0.646，但普通跟踪腰部漂移，C +1906%。
- 局部种子 + 步长门控、不加框：没有局部解时回退搜索，跳变 2.8 rad。

## 2. 跳变上限前移

`joint_jump_policy="reject"` 且 `max_joint_jump_rad` < π 的关节，所有种子都在 q_prev ± 上限内搜索（再与有效限位、舒适窗求交）。v8 先让 DLS 在零空间里漂出上限，再被跳变过滤整批剔除，这正是上面 9/239 的直接原因。

框内没有可接受解时，按 v8 的方式不加框重搜一次（阶段名 `unboxed`）。这样 v8 能解的目标 v9 仍能解，只有超限解的目标照旧报 `JOINT_JUMP_LIMIT`。`joint_jump_policy="prefer"` 不加框。

## 3. 三个文件：问题与修改

| 文件 | 问题 | 影响 | 修改 |
|---|---|---|---|
| ik_candidate | `psi_ready` 为 NaN 时 phi = NaN，所有候选的 cost = inf | 力矩 / 腰回中 / 限位择优全部失效，只剩位姿排序 | 臂角项记 0，其余代价照算；solver 初始化时给出 RuntimeWarning，原因写入 `self.psi_ready_error` |
| ik_candidate | 腰关节按“9 轴时取前两个”定位；7 轴链设 `apply_waist_return=True` 时代价为 NaN | 换链或关节顺序后算错，NaN 代价导致排序不确定 | 按关节名匹配 `WAIST_JOINT_NAMES`；没有腰关节时腰回中项为 0 |
| ik_candidate | `score_candidate` 的 status 用 `"VALID"` / `"BAD_POSE"` | 与公共词表 `IKResult.status` 不一致 | 改为 `SUCCESS` / `NEAR_LIMIT` / `NO_VALID_POSE`（与 `candidate_status` 同一规则） |
| ik_candidate | `finalize_ik_result(per_seed=None)` 在失败分支遍历 None | 失败时抛 TypeError，失败码丢失 | 失败分支按空列表处理 |
| ik_candidate | `evaluate_config_quality` 把 q 直接写进 neutral q | continuous 关节在 Pinocchio q 中占 (cos, sin) 两个分量，非活动关节没用实测值，Jacobian 条件数算的是另一个构型 | 处理 continuous 关节与 `q_template`，并校验输入（只影响诊断字段） |
| ik_common | `normalize` 不拒绝 NaN（NaN < eps 为 False） | 静默返回 NaN 向量 | 拒绝非有限分量 |
| ik_common | `validate_transform4x4` 只查形状和有限性 | 带缩放 / 剪切 / 镜像、末行不对的矩阵可以进入打分 | 与 `solver_dls.validate_transform` 相同的刚体检查（末行 1e-9，RᵀR = I 与 det = 1 为 1e-6） |
| ik_common | `clip_to_bounds` 在跨度 ≤ 2·eps 的关节上 lower+eps > upper−eps | 返回值越出 [lower, upper] | 这类关节取区间中点 |
| ik_common | `target_frames` 缺 `T_world_from_base` / `T_base_from_world` 时在矩阵乘法处抛 TypeError | 报错信息不可读，且不校验这两个变换 | 明确抛 ValueError，并校验变换 |
| robot_model | S/E/W 兜底名是 `Left_J2/4/6`，与 ARM_CONFIGS 的 `LHand_Link*` 不符 | 兜底生效时找不到关节，臂角恒为 NaN | 改为 `LHand_Link2/4/6` |
| robot_model | `fk_pose_and_joints` 请求的关节不在链上时静默略过 | 调用方随后 KeyError 或用错数据 | 明确抛 KeyError；另修正 q_ready 两行缩进，右臂 q_ready 加核对注释 |

## 4. 实测（合成 9 轴模型，Pin 4.1.0，与 v8 同一批目标 + 239 个路点的腰部大幅跟踪）

质量（`bench/v9/quality_pin4_1_0.txt`）：

| 场景 | 结论 |
|---|---|
| near_full ×40 / far_full ×60 / far_tool_dir ×50 / far_tool_cone ×50 | 与 v8 逐值相同（dH 0、dC 0） |
| 普通跟踪 ×119 | C 4.14 → 3.47（−16.2%），腰部偏离 q_ready 均值 0.002 → 0.000，单步最大 0.198 → 0.020，H +0.15% |
| 腰部大幅跟踪 ×239 | 单步最大 2.470 → 0.150，p99 1.857 → 0.150，超过 0.3 rad：7 → 0；H −7.6%，C +1.5% |
| `dls_continuation_local=False` | 6 类场景全部与 v8 逐值相同 |

`max_joint_jump_rad=0.25`（`bench/v9/jump_limit_0p25_pin4_1_0.txt`）：普通跟踪两版都是 119/119，v9 单步最大 0.020；腰部大幅跟踪 9/239 → 239/239，每帧 156.9 → 5.9 ms。

成对计时（ABBA 各 2 次取最短，`bench/v9/paired_timing_pin4_1_0.txt`）：

| 场景 | v8 ms | v9 ms | 变化 |
|---|---|---|---|
| near_full | 12.4 | 12.7 | +2.6% |
| far_full | 22.6 | 22.6 | +0.3% |
| far_tool_dir | 27.9 | 28.2 | +0.9% |
| far_tool_cone | 30.1 | 30.1 | +0.1% |
| 普通跟踪 | 8.4 | 8.3 | −1.5% |
| 腰部大幅跟踪 | 13.4 | 5.6 | −57.8% |

## 5. 配置项与诊断

| 配置 | 默认 | 说明 |
|---|---|---|
| `dls_continuation_local` | True | 必须是 bool；False 即 v8 行为 |
| `dls_continuation_max_step` | 0.15 | 正的有限角度（rad）。路点更稀、相邻路点单关节位移常超过 0.15 rad 时调到 0.2；超过 0.3 后普通跟踪会重新出现容差带切换 |

诊断：

- `last_load_diagnostics["continuation"] = {"local_enabled", "regime", "local_used"}`；
- `last_load_diagnostics["search"]` 新增 `continuation_local_used`、`jump_limit_search_box`、`jump_limit_unboxed_retry`，采用局部解时 `stop_reason = "CONTINUATION_LOCAL"`；
- `per_seed[*]["search_stage"]` 新增 `"continuation"`、`"unboxed"`；
- 求解器属性 `psi_ready_error`（psi_ready 有定义时为 None）。

## 6. 验证与上线前确认

- 149 项测试（v8 的 127 项 + v9 新增 22 项）在 Pin 3.9.0 / 4.1.0 下全部通过（`validation_pin*.txt/json`）。把三个文件换回你上传的原件后，新增的 13 项文件级测试中 12 项失败，说明这些测试确实覆盖了上述修复。
- 未测：真实 URDF、ROS 2 / MoveIt、真机。
- 需要你确认：
  1. **右臂 q_ready 与左臂完全相同。** 如果右臂关节轴按镜像定义，部分关节应反号，需要按真实 URDF 核对（`robot_model.py` 已加注释）。
  2. **`score_candidate` 的 status 词表已改。** 工程里如果有代码判断 `"VALID"` / `"BAD_POSE"`，需要同步修改；本包内没有这种用法。
  3. **`ik_common.validate_transform4x4` 现在拒绝非刚体矩阵。** 求解入口原本已由 `solver_dls` 做同样检查，行为不变；但直接调用 ik_common、传入未正交化矩阵的其它代码会报错。
  4. **0.15 rad 是按合成模型与 bench 路点间距定的。** 真机上请看 `continuation.local_used` 的比例：经常为 False，说明路点比 0.15 rad 稀，应调大。
  5. 初始化时出现 `psi(q_ready) undefined` 告警，说明 q_ready 处臂角无定义（肘部伸直，或 S→W 与参考方向平行），建议换一个 q_ready。
