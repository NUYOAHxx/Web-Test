抱歉，刚才表格列太宽且公式挤在单元格里，导致排版显得杂乱。下面我把内外层循环和全部超参数重新整理成更清爽的分组排版。
一、内外层划分与 5 层嵌套循环
 * 外层（调度与裁判）：HAArmIKSolver（single_arm_ik.py）+ score_candidate / select_best_candidate（ik_base.py）。负责切换搜索框、派发多种子、过滤硬约束并打分选最优。
 * 内层（单种子迭代）：pinocchio_ik_one_seed（single_arm_ik.py）。只拿单个初始种子，在给定边界内做雅可比与零空间梯度下降。
[外层 1] 五阶段状态机 (collect_pinocchio_candidates_and_seeds)
  │     continuation(续算) -> comfort(舒适窗) -> full(全局) -> unboxed -> jump_box
  ▼
[外层 2] 单阶段多种子队列 (run_stage，最多 max_seeds=10 个名额)
  │     q_prev -> 锁腰(locked) -> q_ready -> 种子库(library) -> 随机扰动(scatter)
  ▼
[内层 3] 单种子 DLS 主迭代 (pinocchio_ik_one_seed，最多 max_iter=250 步)
  │     算残差 e & 雅可比 A -> 算次任务 H & 负梯度 z -> 算逆度量 D 与零空间权重
  ├──▶ [内层 4] SNS 主动集限位钳制 (_sns_clamped_core，最多 n_free+1 轮)
  │             解 Cholesky -> 正交信赖域截断 -> 越界关节钉死在边界重算
  └──▶ [内层 5] 回溯线搜索 (最多 backtrack_steps=8 次)
                按 0.5^k 折半试探 -> 误差/代价严格下降即接受
```[span_6](start_span)[span_6](end_span)

---

## 二、全部超参数速查清单

### 1. 内层 DLS 数值与步长控制 (`single_arm_ik.py`)

*   **`PIN_IK_MAX_ITER = 250`**（主循环）[span_7](start_span)[span_7](end_span)：单种子最大迭代步数[span_8](start_span)[span_8](end_span)。调小提速但复杂目标易未收敛截断；调大拉长死胡同种子耗时[span_9](start_span)[span_9](end_span)。
*   **`PIN_IK_DAMPING = 1e-6`**（`_sns_clamped_core`）[span_10](start_span)[span_10](end_span)：基础阻尼 $\lambda_0$，代入 $M_f = A_f D_f A_f^\top + \lambda_{\text{eff}} I$[span_11](start_span)[span_11](end_span)。调大增强奇异点稳定性但变慢；调小易触发 SVD 慢速回退[span_12](start_span)[span_12](end_span)。
*   **`PIN_IK_POS_TOL = 5e-4` m / `PIN_IK_ORI_TOL = 5e-4` rad**（`is_pose`）[span_13](start_span)[span_13](end_span)：内层达标判据，实际取 $\min(\text{本值}, \text{ik\_base 容差})$[span_14](start_span)[span_14](end_span)。决定何时转入姿态整理[span_15](start_span)[span_15](end_span)。
*   **`PIN_IK_MAX_STEP = 0.5` rad**（信赖域）[span_16](start_span)[span_16](end_span)：单步 2-范数上限 $\Vert{}\Delta q\Vert{}_2 \le 0.5$[span_17](start_span)[span_17](end_span)。调大初期快但易过冲；调小步子稳但收敛慢[span_18](start_span)[span_18](end_span)。
*   **`PIN_IK_BACKTRACK_STEPS = 8`**（线搜索）[span_19](start_span)[span_19](end_span)：步长最多按 $0.5^k$ 折半 8 次[span_20](start_span)[span_20](end_span)。调小省 FK 算力，但陡峭曲面易触发 `LINE_SEARCH_STALLED`[span_21](start_span)[span_21](end_span)。
*   **`PIN_IK_STALL_ITERS = 12` / `PIN_IK_STALL_RTOL = 0.025`**（停滞检测）[span_22](start_span)[span_22](end_span)：未达标时每 12 步误差下降不足 2.5% 即放弃该种子[span_23](start_span)[span_23](end_span)。调严可快筛死胡同，但可能误杀缓慢转弯的好种子[span_24](start_span)[span_24](end_span)。
*   **`BOUND_SNAP_TOL = 1e-9`**（`snap`）[span_25](start_span)[span_25](end_span)：浮点舍入级微小越界直接吸附回边界[span_26](start_span)[span_26](end_span)。
*   **`dls_characteristic_length = 0.5` m**（`_make_task_terms` & 种子库）[span_27](start_span)[span_27](end_span)：位置误差除以 $L_c$ 无量纲化[span_28](start_span)[span_28](end_span)。调小等价于放大位置权重，调大等价于放大姿态权重[span_29](start_span)[span_29](end_span)。

### 2. 内层零空间次任务与物理卸载 (`single_arm_ik.py`)

*   **`NULLSPACE_GAIN = 0.30` / `WAIST_NULL_GAIN = 0.50`**（`_SecondaryObjective`）[span_30](start_span)[span_30](end_span)：手臂/腰部拉向 `q_ref` 的弹簧刚度 $k_i$，代入 $z = -k_i(q_i - q_{\text{ref},i})$[span_31](start_span)[span_31](end_span)。调大姿态更贴近 `q_ready`，过大易与线搜索冲突[span_32](start_span)[span_32](end_span)。
*   **`NULLSPACE_ERR_SCALE = 0.06`**（零空间权重）[span_33](start_span)[span_33](end_span)：代入 $w_{\text{null}} = \frac{\beta}{1 + (\Vert{}e\Vert{}_2 / s_e)^2}$[span_34](start_span)[span_34](end_span)。调大则远距离就过早修姿态（干扰主任务）；调小则到位后才修姿态[span_35](start_span)[span_35](end_span)。
*   **`POSTURE_SETTLE_TOL = 1e-4` rad**（姿态整理）[span_36](start_span)[span_36](end_span)：达标后零空间步长小于此值判稳态收敛（`SETTLED`）[span_37](start_span)[span_37](end_span)。
*   **`MAX_SETTLE_EXTRA_ITERS = 25`**（姿态整理）[span_38](start_span)[span_38](end_span)：达标后最多再花 25 步修姿态，设为 0 则关闭姿态整理[span_39](start_span)[span_39](end_span)。
*   **`SETTLE_COST_RTOL = 1e-3` / `SETTLE_PATIENCE = 3`**（姿态整理）[span_40](start_span)[span_40](end_span)：连续 3 步次任务代价下降 $< 0.1\%$ 即提前停止[span_41](start_span)[span_41](end_span)。
*   **`POSE_PEAK_BAND_FRACTION = 0.25`**（`offer_pose`）[span_42](start_span)[span_42](end_span)：单种子内择优的负荷峰值容差带 $= 0.25 \times \text{peak\_tie}$[span_43](start_span)[span_43](end_span)。
*   **`NO_SETTLE_NULL_STEP_RATIO = 2.0`**（续算模式）[span_44](start_span)[span_44](end_span)：不整理姿态时，未达标前限制零空间原始步长 $\le 2.0 \Vert{}e\Vert{}_2$，防自运动漂移[span_45](start_span)[span_45](end_span)。
*   **`GRAVITY_TORQUE_GAIN = 0.04` / `RELATIVE_GRAVITY_GAIN = 0.0`**（负荷项）[span_46](start_span)[span_46](end_span)：绝对/相对重力矩卸载增益[span_47](start_span)[span_47](end_span)。调大强力压低力矩（肘部下垂），过大破坏自然臂角[span_48](start_span)[span_48](end_span)。
*   **`COM_XY_GAIN = 0.12`**（质心项，仅含腰）[span_49](start_span)[span_49](end_span)：水平质心偏移惩罚增益[span_50](start_span)[span_50](end_span)。调大强力抑制弯腰前倾[span_51](start_span)[span_51](end_span)。
*   **`JOINT_LIMIT_GAIN = 0.03`**（限位项）[span_52](start_span)[span_52](end_span)：距边界 `0.02 rad` 缓冲区内的向内排斥刚度[span_53](start_span)[span_53](end_span)。

### 3. 无状态腰部释放与舒适窗 (`single_arm_ik.py`)

*   **`WAIST_TASK_WEIGHT = 6.0` / `WAIST_TASK_WEIGHT_FAR = 1.0`**（逆度量 $D$）[span_54](start_span)[span_54](end_span)：$d_{\text{waist}} = \frac{1-\alpha}{w_{\text{near}}} + \frac{\alpha}{w_{\text{far}}}$[span_55](start_span)[span_55](end_span)。近处腰比臂贵 6 倍（优先动臂），远处同价[span_56](start_span)[span_56](end_span)。
*   **`WAIST_RELEASE_START = 0.75` / `WAIST_RELEASE_END = 0.94`**（距离释放）[span_57](start_span)[span_57](end_span)：目标距肩达臂展 75% 开始放腰，94% 完全释放[span_58](start_span)[span_58](end_span)。
*   **`WAIST_RETURN_RELEASE_RAD = 0.20` rad**（返程解锁）[span_59](start_span)[span_59](end_span)：腰偏离 `q_ref` 超过 0.20 rad 时保持释放，防止从远点缩回时腰部卡死回不来[span_60](start_span)[span_60](end_span)。
*   **`WAIST_YAW_VIRTUAL_LIM = 90°` / `WAIST_COMFORT_MAX_HALF = 90°`**（舒适窗）[span_61](start_span)[span_61](end_span)：偏航轴及所有腰关节舒适半窗上限，防躯干背对工作区[span_62](start_span)[span_62](end_span)。
*   **`WAIST_GRAVITY_FREE_COS = cos(30°)`**（舒适窗）[span_63](start_span)[span_63](end_span)：转轴与重力夹角 $< 30^\circ$ 视为偏航类（不走正弦重力矩推导）[span_64](start_span)[span_64](end_span)。
*   **`WAIST_COMFORT_SOFT_RATIO = 0.70`**（舒适窗）[span_65](start_span)[span_65](end_span)：重力敏感腰关节的舒适窗按最大重力矩与质心偏移的 70% 推导[span_66](start_span)[span_66](end_span)。
*   **`WAIST_COMFORT_EDGE_SEEDS = 1`**（`comfort` $\to$ `full` 回退）[span_67](start_span)[span_67](end_span)：舒适窗无解时，取窗内误差最小的 1 个终点作为放开后的接力种子[span_68](start_span)[span_68](end_span)。
*   **`WAIST_COMFORT_CONTINUATION_STEP = 0.10`**（工况判定）[span_69](start_span)[span_69](end_span)：相对 `FK(q_prev)` 位置差 $\le 0.1\times\text{臂展}$ 且姿态差 $\le 0.1\text{ rad}$ 判为续算[span_70](start_span)[span_70](end_span)。
*   **`WAIST_COMFORT_EARLY_FALLBACK = True` / `PATIENCE = 4`**（舒适窗检查点）[span_71](start_span)[span_71](end_span)：跑完固定种子与种子库（无库时 4 个名额）仍无解即提前回退 `full`[span_72](start_span)[span_72](end_span)。

### 4. 外层多种子调度、续算与跳变策略 (`single_arm_ik.py`)

*   **`N_NOISE_PER_BASE = 16` / `NOISE_SCALES = [0.60, 0.30, 0.15, 0.05]` / `RNG_SEED = 120`**（`make_scatter_seeds`）[span_73](start_span)[span_73](end_span)：围绕 `q_prev` 和 `q_ready` 生成确定性多尺度高斯扰动种子[span_74](start_span)[span_74](end_span)。
*   **`SEED_LIBRARY_SIZE = 16384` / `SEEDS = 3` / `POOL = 64` / `MIN_SEPARATION = 0.5` rad / `RNG_SEED = 13`**（种子库）[span_75](start_span)[span_75](end_span)：预采样 16384 个构型；按任务空间距离取前 64 个，再按关节 $\infty$-范数间距 $\ge 0.5\text{ rad}$ 选出 3 个不同分支的种子[span_76](start_span)[span_76](end_span)。
*   **`SEED_LIBRARY_CONTINUATION = False`**（续算）[span_77](start_span)[span_77](end_span)：续算时禁用全局种子库，防分支跳变[span_78](start_span)[span_78](end_span)。
*   **`CONTINUATION_LOCAL = True` / `MAX_STEP = 0.15` rad / `POSTURE_SETTLE = False` / `DLS_STEP_FRACTION = 0.5`**（`continuation` 阶段）[span_79](start_span)[span_79](end_span)：在 `q_prev ± 0.15 rad` 框内以 `0.075 rad` 单步上限快搜，关闭姿态整理实现静止零漂移[span_80](start_span)[span_80](end_span)。
*   **`MAX_JOINT_JUMP_RAD = pi` / `joint_jump_policy = "reject"` / `JUMP_LIMIT_RESCUE = True`**（H4 跳变）[span_81](start_span)[span_81](end_span)：单关节相对 `q_prev` 最大跳变限幅；仅因跳变失败时触发 `jump_box` 框内救援重搜[span_82](start_span)[span_82](end_span)。
*   **`LOCKED_SEED_SKIP_TOL = 0.02` rad / `LOCKED_REACH_CHECK = True`**（锁腰剪枝）[span_83](start_span)[span_83](end_span)：`q_prev` 已合格且腰偏离 $\le 0.02\text{ rad}$，或目标超出锁腰臂长上界时跳过锁腰种子[span_84](start_span)[span_84](end_span)。
*   **`load_max_seeds = 10` / `min_seeds = 2` / `min_unlocked_seeds = 1`**（`run_stage`）[span_85](start_span)[span_85](end_span)：单阶段名额上限与早停前必须满足的最小比较次数[span_86](start_span)[span_86](end_span)。

### 5. 静态载荷门控 (`LoadPolicy`) 与安全配置 (`single_arm_ik.py`)

*   **`absolute_weight = 0.05` / `excess_weight = 0.50` / `relative_weight = 0.0`**（`LoadPolicy`）[span_87](start_span)[span_87](end_span)：严格模式重力矩代价权重；超过软阈值后 `0.50` 强力压低峰值[span_88](start_span)[span_88](end_span)。
*   **`soft_ratio = 0.70` / `LOAD_EARLY_STOP_RATIO = 0.70`**（`LoadPolicy`）[span_89](start_span)[span_89](end_span)：$\le 70\%$ 触发低负荷早停，$> 70\%$ 激活超载惩罚并触发锁腰释放[span_90](start_span)[span_90](end_span)。
*   **`peak_tie = 0.02`**（`_filter_load_candidates`）[span_91](start_span)[span_91](end_span)：峰值负载率相差 $\le 2\%$ 视为同档，交由外层综合打分[span_92](start_span)[span_92](end_span)。
*   **`load_state_max_age_s = 1.0` s / `load_budget_s = 0.5` s**（`solve`）[span_93](start_span)[span_93](end_span)：全身快照新鲜度上限与单次求解总时间预算[span_94](start_span)[span_94](end_span)。
*   **`load_context_tolerance_rad = 0.01` / `_m = 0.001`**（`_load_snapshot_error`）[span_95](start_span)[span_95](end_span)：求解期间外部非活跃关节移动超限则作废结果[span_96](start_span)[span_96](end_span)。
*   **`sync_secondary_cost = False` / `w_com = 100.0` / `w_collision = 0.0`**（`process_seed`）[span_97](start_span)[span_97](end_span)：开启时将内层质心/避障代价同步进外层 `cost`，并自动将内层 `waist_null_gain` 提至 `2.0`[span_98](start_span)[span_98](end_span)。

### 6. 外层几何容差与裁判打分权重 (`ik_base.py`)

*   **`POS_TOL = 5e-4` m / `ORI_TOL = 5e-4` rad**[span_99](start_span)[span_99](end_span)：外层 `pose_ok` 最终硬门槛[span_100](start_span)[span_100](end_span)。
*   **`SAFE_MARGIN = 0.02` rad**[span_101](start_span)[span_101](end_span)：最小关节余量低于此值降级为 `NEAR_LIMIT` 并触发二次 hinge 惩罚[span_102](start_span)[span_102](end_span)。
*   **`W_WAIST_RETURN = 2000.0`**[span_103](start_span)[span_103](end_span)：外层最高权重，强力偏好不动腰或少动腰的解[span_104](start_span)[span_104](end_span)。
*   **`W_TORQUE = 500.0`**[span_105](start_span)[span_105](end_span)：次高权重，强力偏好低均方负载率的省力构型[span_106](start_span)[span_106](end_span)。
*   **`W_NOM = 50.0`**[span_107](start_span)[span_107](end_span)：约束 7-DoF 肘部臂角 $\psi$ 贴近 `q_ready`[span_108](start_span)[span_108](end_span)。
*   **`W_LIMIT = 10.0`**[span_109](start_span)[span_109](end_span)：乘以归一化限位惩罚，压制贴近物理限位的解[span_110](start_span)[span_110](end_span)。
*   **`W_SMOOTH = 8.0`**[span_111](start_span)[span_111](end_span)：偏好离上一帧 `q_prev` 欧氏距离更近的解[span_112](start_span)[span_112](end_span)。
*   **`PHI_TIE_TOL = 1e-5` rad**[span_113](start_span)[span_113](end_span)：臂角差小于此值视为 0，防浮点噪声干扰排序[span_114](start_span)[span_114](end_span)。
*   **`TOOL_AXIS_X = [1, 0, 0]` / `TOOL_CONE_HALF_ANGLE = 30°`**[span_115](start_span)[span_115](end_span)：默认工具轴与锥角约束默认半角[span_116](start_span)[span_116](end_span)。
*   **`PERIODIC_MIN_SPAN_RAD = 2*pi + 1e-6`**[span_117](start_span)[span_117](end_span)：判定关节跨度是否允许 $\pm 2\pi$ 平移的阈值[span_118](start_span)[span_118](end_span)。
*   **`RIGID_BOTTOM_ROW_TOL = 1e-9` / `RIGID_ROTATION_TOL = 1e-6`**[span_119](start_span)[span_119](end_span)：校验 $4 \times 4$ 齐次矩阵合法性的容差[span_120](start_span)[span_120](end_span)。

### 7. 备用历史引擎参数（默认全部关闭，`single_arm_ik.py`）

*   **总开关**：`USE_DIRECT_PSI = False`、`USE_LOCAL_SEEDS = False`、`USE_SSIK = False`[span_121](start_span)[span_121](end_span)。
*   **Direct-Psi**：`PORTFOLIO_DIRECT_MAX_NFEV = 150`、`PORTFOLIO_DIRECT_POS_TOL = 5e-4`、`PORTFOLIO_DIRECT_ORI_TOL = 5e-4`、`PORTFOLIO_DIRECT_PSI_TOL = 1.5`、`PORTFOLIO_Q_REF_MODE = "ready"`、`DIRECT_LOCAL_NOISE_SCALES = (0.002, 0.005, 0.01, 0.02, 0.04)`、`PIN_TOP_K_FOR_DIRECT = 15`；内置权重 `w_pos=100, w_ori=20, w_psi=20, w_seed=0.03, w_bias=0.0, w_center=0.005, w_limit=0.05, limit_margin=0.10`[span_122](start_span)[span_122](end_span)。
*   **SSIK**：`SSIK_RESPECT_LIMITS = True`、`SSIK_ALLOW_REFINEMENT = True`、`SSIK_REFINEMENT_MAX_ITERS = 30`、`SSIK_MAX_SOLUTIONS = True`、`SSIK_USE_READY_AS_SEED = True`、`SSIK_TOP_K_FOR_DIRECT = 10`[span_123](start_span)[span_123](end_span)。

