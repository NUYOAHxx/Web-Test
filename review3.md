整套逆运动学求解器采用“外层全局调度与裁判 + 内层单种子数值微调”的双层架构，共包含 5 层嵌套循环。
1. 程序的内外层划分与 5 层嵌套循环全景
1.1 什么是“外层”与“内层”？
 * 外层（宏观调度与裁判层）：位于 HAArmIKSolver（single_arm_ik.py）与 score_candidate / select_best_candidate（ik_base.py）。它不求导、不做梯度下降，只负责三件事：决定在哪个框里搜（阶段切换）、派哪些初始种子去搜（种子队列与并发）、以及所有种子跑完后按硬约束过滤并打分选第一名。
 * 内层（微观数值迭代层）：位于 pinocchio_ik_one_seed（single_arm_ik.py）。它只拿单个初始种子 q_0，在给定的关节限位框 [l, u] 内，利用雅可比矩阵和零空间梯度一步步迭代，试图走到目标位姿并优化姿态/力矩。
1.2 完整的 5 层嵌套循环结构
【外层 - 循环 1：五阶段搜索状态机】 (collect_pinocchio_candidates_and_seeds)
  依次按条件触发：continuation(局部续算) -> comfort(腰部舒适窗) -> full(全局限位) -> unboxed(无框探底) -> jump_box(跳变框救援)
  │
  └──▶ 【外层 - 循环 2：单阶段多种子队列】 (run_stage，最多消耗 max_seeds=10 个名额)
        按序弹出种子：q_prev -> 锁腰(locked) -> q_ready -> 种子库(library) -> 随机扰动(scatter)
        （注：开启多核时，q_prev 先在本地跑以求极速早停，未命中则将剩余种子打包扔进进程池并发）
        │
        └──▶ 【内层 - 循环 3：单种子 DLS 主迭代】 (pinocchio_ik_one_seed，最多 max_iter=250 步)
              计算主任务残差 e & 雅可比 A -> 计算次任务代价 H & 负梯度 z -> 判定收敛/停滞 -> 算逆度量 D 与零空间权重 w_null
              │
              ├──▶ 【内层 - 循环 4：SNS 主动集限位钳制】 (_sns_clamped_core，最多 n_free+1 轮)
              │     解 Cholesky 得自由关节步长 dq_free -> 正交信赖域截断 -> 若有自由关节越界，将其钉死在边界并重算剩余关节
              │
              └──▶ 【内层 - 循环 5：回溯线搜索】 (最多 backtrack_steps=8 次)
                    试探 q + 0.5^k * dq（先试“主+零空间”合成步，失败再试纯主任务步）-> 误差/代价严格下降即接受步长
```[span_8](start_span)[span_8](end_span)

### 1.3 内外层如何交接数据？
1. **外层传给内层**：种子初始角 `seed`、本阶段有效边界 `[lo_s, hi_s]`、参考姿态 `q_ready`、腰部是否锁定掩码 `fixed_joint_mask`、是否已触发负荷释放 `waist_load_release`[span_9](start_span)[span_9](end_span)。
2. **内层还给外层**：迭代收敛（或误差最小）的关节角 `q`，以及包含次任务分量和收敛状态的诊断字典 `diag`[span_10](start_span)[span_10](end_span)。
3. **外层接手处理**：在 `process_seed` 中，先做 $2\pi k$ 周期分支对齐，再调用 `ik_base.score_candidate` 打外层基础分、算单关节跳变、算 `strict` 静态重力矩，去重后放入 `candidates` 候选池[span_11](start_span)[span_11](end_span)[span_12](start_span)[span_12](end_span)。当满足早停条件（如 `strict` 下已比较锁腰且找到负荷 $\le 70\%$ 的解）时，直接跳出循环 2[span_13](start_span)[span_13](end_span)。
4. **最终裁决**：所有阶段结束后，`_finalize_result` 对 `candidates` 执行 **H4 跳变过滤 $\to$ H1-H3/H5 终检 $\to$ S1 峰值容差带筛选 $\to$ 叠加 `W_TORQUE` 力矩分 $\to$ 8 级排序选出 `best`**[span_14](start_span)[span_14](end_span)[span_15](start_span)[span_15](end_span)。

---

## 2. 两个文件的全部超参数完全手册

### 2.1 内层 DLS 基础数值、步长与停滞控制 (`single_arm_ik.py`)

| 参数名 | 默认值 | 使用位置 | 代入公式 / 逻辑与对求解结果的影响 |
| :--- | :--- | :--- | :--- |
| `PIN_IK_MAX_ITER` | `250`[span_16](start_span)[span_16](end_span) | `pinocchio_ik_one_seed` 主循环[span_17](start_span)[span_17](end_span) | 单个种子的最大迭代次数[span_18](start_span)[span_18](end_span)。调小可提速，但远距离/复杂避障可能未收敛就截断；调大对已收敛种子无影响，但会拉长死胡同种子的耗时[span_19](start_span)[span_19](end_span)。 |
| `PIN_IK_DAMPING` ($\lambda_0$) | `1e-6`[span_20](start_span)[span_20](end_span) | `_sns_clamped_core` 海森矩阵[span_21](start_span)[span_21](end_span) | 代入 $\lambda_{\text{eff}} = \lambda_0 + \min(10^{-3}, 10^{-4}\Vert{}e\Vert{}_2^2)$ 与 $M_f = A_f D_f A_f^\top + \lambda_{\text{eff}} I$[span_22](start_span)[span_22](end_span)。调大增强奇异点数值稳定性但拖慢收敛；调小收敛极快但奇异点附近易触发 SVD 慢速回退[span_23](start_span)[span_23](end_span)。 |
| `PIN_IK_POS_TOL` / `PIN_IK_ORI_TOL` | `5e-4` m / `5e-4` rad[span_24](start_span)[span_24](end_span) | `is_pose` 达标判定[span_25](start_span)[span_25](end_span) | 实际取 $\min(\text{本值}, \text{ik\_base.POS\_TOL/ORI\_TOL})$[span_26](start_span)[span_26](end_span)。决定何时进入“姿态整理阶段[span_27](start_span)”[span_27](end_span)。调小精度更高但迭代步数增加；调大易被外层终检拒绝[span_28](start_span)[span_28](end_span)。 |
| `PIN_IK_MAX_STEP` | `0.5` rad[span_29](start_span)[span_29](end_span) | `_sns_clamped_core` 信赖域[span_30](start_span)[span_30](end_span) | 约束单步 2-范数 $\Vert{}\Delta q\Vert{}_2 \le \text{max\_step}$[span_31](start_span)[span_31](end_span)。调大在大位移初期冲得快但容易迈过头导致线搜索频繁回退；调小步子稳但收敛慢[span_32](start_span)[span_32](end_span)。 |
| `PIN_IK_BACKTRACK_STEPS` | `8`[span_33](start_span)[span_33](end_span) | 回溯线搜索循环[span_34](start_span)[span_34](end_span) | 步长按 $0.5^k$ 最多折半 8 次（最小缩至 $1/128$）[span_35](start_span)[span_35](end_span)。调小省去陡峭曲面上的无效 FK 试探，但更容易触发 `LINE_SEARCH_STALLED` 提前终止[span_36](start_span)[span_36](end_span)。 |
| `PIN_IK_STALL_ITERS` / `PIN_IK_STALL_RTOL` | `12` 步 / `0.025`[span_37](start_span)[span_37](end_span) | 停滞检测[span_38](start_span)[span_38](end_span) | 位姿未达标时，每 12 步检查主任务误差范数是否下降 $\ge 2.5\%$，否则直接放弃该种子[span_39](start_span)[span_39](end_span)。调严（步数小/比例高）能极快淘汰不可达的锁腰种子，但可能误杀在狭长山谷中缓慢拐弯的种子[span_40](start_span)[span_40](end_span)。 |
| `BOUND_SNAP_TOL` | `1e-9`[span_41](start_span)[span_41](end_span) | `snap()` 边界吸附[span_42](start_span)[span_42](end_span) | 试探点越界不超过 $10^{-9}$ rad 时视为浮点舍入误差并吸附回边界，否则拒绝该步[span_43](start_span)[span_43](end_span)。 |
| `dls_characteristic_length` ($L_c$) | `0.5` m[span_44](start_span)[span_44](end_span) | `_make_task_terms` & 种子库检索[span_45](start_span)[span_45](end_span) | 位置误差与平移雅可比均除以 $L_c$：$e_{\text{pos}} = \Delta p / L_c$[span_46](start_span)[span_46](end_span)。调小相当于**放大位置权重**（优先追位置再对齐姿态）；调大相当于**放大姿态权重**[span_47](start_span)[span_47](end_span)。 |

---

### 2.2 内层零空间次任务、姿态整理与物理卸载 (`single_arm_ik.py`)

| 参数名 | 默认值 | 使用位置 | 代入公式 / 逻辑与对求解结果的影响 |
| :--- | :--- | :--- | :--- |
| `NULLSPACE_GAIN` / `WAIST_NULL_GAIN` | `0.30` / `0.50`[span_48](start_span)[span_48](end_span) | `_SecondaryObjective` 回中项[span_49](start_span)[span_49](end_span) | 代入 $H_{\text{posture}} = \frac{1}{2}\sum k_i (q_i - q_{\text{ref},i})^2$，$z = -k_i(q_i - q_{\text{ref},i})$[span_50](start_span)[span_50](end_span)。调大则零空间把关节拉回 `q_ready` 的弹簧更硬，但过大易在姿态整理期与线搜索冲突[span_51](start_span)[span_51](end_span)。 |
| `NULLSPACE_ERR_SCALE` ($s_e$) | `0.06`[span_52](start_span)[span_52](end_span) | 零空间权重 $w_{\text{null}}$[span_53](start_span)[span_53](end_span) | 代入 $w_{\text{null}} = \frac{\beta}{1 + (\Vert{}e\Vert{}_2 / s_e)^2}$[span_54](start_span)[span_54](end_span)。调大则远距离时零空间力就过早介入，可能干扰主任务收敛；调小则几乎只在末端到位后才修姿态[span_55](start_span)[span_55](end_span)。 |
| `POSTURE_SETTLE_TOL` | `1e-4` rad[span_56](start_span)[span_56](end_span) | 姿态整理收敛判据[span_57](start_span)[span_57](end_span) | 末端达标后，若零空间步长 $\Vert{}\Delta q_{\text{null}}\Vert{}_2 < 10^{-4}$ 判定稳态收敛（`SETTLED`）[span_58](start_span)[span_58](end_span)。调小会追求更极致的零空间最优，但增加尾迭代耗时[span_59](start_span)[span_59](end_span)。 |
| `MAX_SETTLE_EXTRA_ITERS` | `25`[span_60](start_span)[span_60](end_span) | 姿态整理步数上限[span_61](start_span)[span_61](end_span) | 末端达标后最多再花 25 步在零空间整理姿态（防死循环）[span_62](start_span)[span_62](end_span)。调为 0 则完全不做达标后姿态优化[span_63](start_span)[span_63](end_span)。 |
| `SETTLE_COST_RTOL` / `SETTLE_PATIENCE` | `1e-3` / `3`[span_64](start_span)[span_64](end_span) | 姿态整理边际收益判据[span_65](start_span)[span_65](end_span) | 连续 3 步次任务代价 $H(q)$ 相对下降 $< 0.1\%$ 即提前停止整理[span_66](start_span)[span_66](end_span)。 |
| `POSE_PEAK_BAND_FRACTION` | `0.25`[span_67](start_span)[span_67](end_span) | 单种子内 `offer_pose`[span_68](start_span)[span_68](end_span) | 单种子内择优的负荷峰值容差带 $= 0.25 \times \text{peak\_tie}$[span_69](start_span)[span_69](end_span)。在该容差带内优先选次任务代价 $H(q)$ 最小的迭代点[span_70](start_span)[span_70](end_span)。 |
| `NO_SETTLE_NULL_STEP_RATIO` | `2.0`[span_71](start_span)[span_71](end_span) | 续算模式（`posture_settle=False`）[span_72](start_span)[span_72](end_span) | 未达标时强制限制零空间原始步长 $\le 2.0 \Vert{}e\Vert{}_2$，防止微小目标移动引发剧烈的零空间自运动漂移[span_73](start_span)[span_73](end_span)。 |
| `GRAVITY_TORQUE_GAIN` / `RELATIVE_GRAVITY_GAIN` | `0.04` / `0.0`[span_74](start_span)[span_74](end_span) | `_SecondaryObjective` 负荷项[span_75](start_span)[span_75](end_span) | 分别对应绝对重力矩权重 $k_{\text{abs}}$ 与相对参考力矩保持权重 $k_{\text{rel}}$[span_76](start_span)[span_76](end_span)。调大 `GRAVITY_TORQUE_GAIN` 会强力压低电机力矩（如让肘部下垂），但过大易破坏自然臂角[span_77](start_span)[span_77](end_span)。 |
| `COM_XY_GAIN` | `0.12`[span_78](start_span)[span_78](end_span) | `_SecondaryObjective` 质心项[span_79](start_span)[span_79](end_span) | 代入 $H_{\text{com}} = \frac{1}{2} k_{\text{com}} \Vert{}P_\perp(c - c_{\text{ref}})\Vert{}_2^2$（仅含腰模式生效）[span_80](start_span)[span_80](end_span)。调大则更严格地把整机水平重心锁在支撑面中心，抑制弯腰前倾[span_81](start_span)[span_81](end_span)。 |
| `JOINT_LIMIT_GAIN` | `0.03`[span_82](start_span)[span_82](end_span) | `_SecondaryObjective` 限位项[span_83](start_span)[span_83](end_span) | 代入 $H_{\text{limit}} = \frac{1}{2} k_{\text{lim}} \sum (u_i^2 + o_i^2)$[span_84](start_span)[span_84](end_span)。调大则关节一旦进入距边界 `0.02 rad` 缓冲区，受到的向内排斥力更强[span_85](start_span)[span_85](end_span)。 |

---

### 2.3 无状态腰部释放与腰部舒适窗 (`single_arm_ik.py`)

| 参数名 | 默认值 | 使用位置 | 代入公式 / 逻辑与对求解结果的影响 |
| :--- | :--- | :--- | :--- |
| `WAIST_TASK_WEIGHT` / `WAIST_TASK_WEIGHT_FAR` | `6.0` / `1.0`[span_86](start_span)[span_86](end_span) | 逆度量 $D$ 腰部对角元[span_87](start_span)[span_87](end_span) | $d_{\text{waist}} = \frac{1-\alpha}{w_{\text{near}}} + \frac{\alpha}{w_{\text{far}}}$[span_88](start_span)[span_88](end_span)。`near` 越大，近距离越倾向只动臂不动腰；`far` 越小，远距离腰部越积极参与[span_89](start_span)[span_89](end_span)。 |
| `WAIST_RELEASE_START` / `WAIST_RELEASE_END` | `0.75` / `0.94`[span_90](start_span)[span_90](end_span) | 距离释放函数 $f_{\text{rel}}(d)$[span_91](start_span)[span_91](end_span) | 肩到目标距离占最大臂展的 75% 开始放腰，94% 完全释放[span_92](start_span)[span_92](end_span)。区间前移会让腰部更早介入分担手臂伸展[span_93](start_span)[span_93](end_span)。 |
| `WAIST_RETURN_RELEASE_RAD` | `0.20` rad[span_94](start_span)[span_94](end_span) | 返程解锁通道 $r_{\text{ret}}$[span_95](start_span)[span_95](end_span) | $r_{\text{ret}} = \text{clip}(\|q_{\text{waist}} - q_{\text{ref,waist}}\|_2 / 0.20, 0, 1)$[span_96](start_span)[span_96](end_span)。防止从远点退回近点时腰部被提前锁死回不到正位[span_97](start_span)[span_97](end_span)。 |
| `WAIST_YAW_VIRTUAL_LIM` / `WAIST_COMFORT_MAX_HALF` | `90°` / `90°`[span_98](start_span)[span_98](end_span) | `derive_waist_comfort_bounds`[span_99](start_span)[span_99](end_span) | 偏航轴（重力无关）舒适半窗与所有腰关节舒适半窗的绝对上限，防止躯干背对工作区[span_100](start_span)[span_100](end_span)。 |
| `WAIST_GRAVITY_FREE_COS` | `cos(30°)`[span_101](start_span)[span_101](end_span) | 区分重力敏感/无关腰关节[span_102](start_span)[span_102](end_span) | 转轴与重力夹角 $< 30^\circ$ 视为偏航类（不走正弦重力矩推导）[span_103](start_span)[span_103](end_span)。 |
| `WAIST_COMFORT_SOFT_RATIO` | `0.70`[span_104](start_span)[span_104](end_span) | 舒适窗正弦律推导[span_105](start_span)[span_105](end_span) | 舒适窗内限制重力矩与质心偏移均不超过最大能力的 70%[span_106](start_span)[span_106](end_span)。调小则舒适窗变窄，更容易触发 `full` 回退[span_107](start_span)[span_107](end_span)。 |
| `WAIST_COMFORT_EDGE_SEEDS` | `1`[span_108](start_span)[span_108](end_span) | `comfort` $\to$ `full` 回退[span_109](start_span)[span_109](end_span) | 舒适窗内无解时，挑出窗内误差最小的 1 个未达标终点作为放开限位后的接力种子[span_110](start_span)[span_110](end_span)。 |
| `WAIST_COMFORT_CONTINUATION_STEP` | `0.10`[span_111](start_span)[span_111](end_span) | 续算 vs 重定位判据[span_112](start_span)[span_112](end_span) | 目标相对 `FK(q_prev)` 位置差 $\le 0.1 \times \text{臂展}$ 且姿态差 $\le 0.1\text{ rad}$ 判为“续算”，否则判为“重定位[span_113](start_span)”[span_113](end_span)。 |
| `WAIST_COMFORT_EARLY_FALLBACK` / `PATIENCE` | `True` / `4`[span_114](start_span)[span_114](end_span) | 舒适阶段提前回退检查点[span_115](start_span)[span_115](end_span) | 舒适阶段跑完固定种子与种子库（无种子库时跑完 4 个名额）仍无可接受解，不等耗光名额直接回退到 `full` 阶段[span_116](start_span)[span_116](end_span)。 |

---

### 2.4 外层多种子调度、续算、种子库与跳变策略 (`single_arm_ik.py`)

| 参数名 | 默认值 | 使用位置 | 代入公式 / 逻辑与对求解结果的影响 |
| :--- | :--- | :--- | :--- |
| `N_NOISE_PER_BASE` / `NOISE_SCALES` | `16` / `[0.6, 0.3, 0.15, 0.05]`[span_117](start_span)[span_117](end_span) | `make_scatter_seeds`[span_118](start_span)[span_118](end_span) | 围绕 `q_prev` 和 `q_ready` 生成多尺度高斯扰动种子，兼顾大范围跳出局部极小值与局部细化[span_119](start_span)[span_119](end_span)。 |
| `RNG_SEED` | `120`[span_120](start_span)[span_120](end_span) | `_solve_impl` 随机数生成器[span_121](start_span)[span_121](end_span) | 固定随机种子，保证相同输入下求解结果 100% 确定性可复现[span_122](start_span)[span_122](end_span)。 |
| `SEED_LIBRARY_SIZE` / `SEEDS` / `POOL` / `MIN_SEPARATION` / `RNG_SEED` | `16384` / `3` / `64` / `0.5` rad / `13`[span_123](start_span)[span_123](end_span) | `build_seed_library` & `select_library_seeds`[span_124](start_span)[span_124](end_span) | 初始化预采样 16384 个构型；检索时按任务空间距离 $\sqrt{(\|\Delta p\|/L_c)^2 + \Delta \theta^2}$ 取前 64 个，再按关节 $\infty$-范数间距 $\ge 0.5\text{ rad}$ 去重选出 3 个不同分支的种子[span_125](start_span)[span_125](end_span)。 |
| `SEED_LIBRARY_CONTINUATION` | `False`[span_126](start_span)[span_126](end_span) | 续算类求解[span_127](start_span)[span_127](end_span) | 续算时默认禁用全局种子库，防止跳到远处的离散分支[span_128](start_span)[span_128](end_span)。 |
| `CONTINUATION_LOCAL` / `MAX_STEP` / `POSTURE_SETTLE` / `DLS_STEP_FRACTION` | `True` / `0.15` rad / `False` / `0.5`[span_129](start_span)[span_129](end_span) | `continuation` 续算阶段[span_130](start_span)[span_130](end_span) | 连续路点跟踪时，先在 `q_prev ± 0.15 rad` 框内以 `0.075 rad` 单步上限快搜，且关闭姿态整理（静止目标原样返回，零漂移）[span_131](start_span)[span_131](end_span)。 |
| `MAX_JOINT_JUMP_RAD` / `joint_jump_policy` / `JUMP_LIMIT_RESCUE` | $\pi$ / `"reject"` / `True`[span_132](start_span)[span_132](end_span) | H4 跳变过滤 & `jump_box` 救援[span_133](start_span)[span_133](end_span) | 单关节相对 `q_prev` 跳变上限[span_134](start_span)[span_134](end_span)。`"reject"` 下若上限 $<\pi$ 会前移为搜索框；若仅因跳变超限失败且 `JUMP_LIMIT_RESCUE=True`，触发 `jump_box` 框内救援重搜[span_135](start_span)[span_135](end_span)。 |
| `LOCKED_SEED_SKIP_TOL` / `LOCKED_REACH_CHECK` | `0.02` rad / `True`[span_136](start_span)[span_136](end_span) | 锁腰种子剪枝[span_137](start_span)[span_137](end_span) | `q_prev` 解已合格且腰部偏离 `q_ready` $\le 0.02\text{ rad}$，或目标超出锁腰严格臂长上界时，直接跳过锁腰种子省时间[span_138](start_span)[span_138](end_span)。 |
| `load_max_seeds` / `min_seeds` / `min_unlocked_seeds` | `10` / `2` / `1`[span_139](start_span)[span_139](end_span) | `run_stage` 种子配额与早停[span_140](start_span)[span_140](end_span) | 每阶段最多跑 10 个名额；至少跑满 2 个名额（含 1 个不锁腰且完成锁腰比较）才允许低负荷早停[span_141](start_span)[span_141](end_span)。 |

---

### 2.5 静态载荷门控 (`LoadPolicy`) 与运行时安全配置 (`single_arm_ik.py`)

| 参数名 | 默认值 | 使用位置 | 代入公式 / 逻辑与对求解结果的影响 |
| :--- | :--- | :--- | :--- |
| `absolute_weight` / `excess_weight` / `relative_weight` | `0.05` / `0.50` / `0.0`[span_142](start_span)[span_142](end_span) | `LoadPolicy._cost_terms`[span_143](start_span)[span_143](end_span) | 严格模式下的重力矩二次代价权重：$0.5 w_{\text{abs}} r^2 + 0.5 w_{\text{exc}} \max(|r|-s, 0)^2$[span_144](start_span)[span_144](end_span)。超过软阈值后 `0.50` 的高权重强力压低峰值[span_145](start_span)[span_145](end_span)。 |
| `soft_ratio` / `LOAD_EARLY_STOP_RATIO` | `0.70` / `0.70`[span_146](start_span)[span_146](end_span) | 超载软阈值 & 早停阈值[span_147](start_span)[span_147](end_span) | 负载率 $\le 70\%$ 视为低负荷健康解，触发早停；超过 $70\%$ 激活 `excess_weight` 惩罚并触发锁腰释放[span_148](start_span)[span_148](end_span)。 |
| `peak_tie` | `0.02`[span_149](start_span)[span_149](end_span) | `_filter_load_candidates` S1 分档[span_150](start_span)[span_150](end_span) | 峰值负载率与全局最低峰值相差 $\le 2\%$ 的候选视为同档，进入外层综合打分，避免为了省 1% 力矩选出极扭曲姿态[span_151](start_span)[span_151](end_span)。 |
| `load_state_max_age_s` / `load_budget_s` | `1.0` s / `0.5` s[span_152](start_span)[span_152](end_span) | `prepare_load_context` / `solve`[span_153](start_span)[span_153](end_span) | 全身快照最大允许滞后时间（1s）与单次严格求解的总时间预算（0.5s）[span_154](start_span)[span_154](end_span)。 |
| `load_context_tolerance_rad` / `_m` | `0.01` rad / `0.001` m[span_155](start_span)[span_155](end_span) | `_load_snapshot_error`[span_156](start_span)[span_156](end_span) | 求解期间若外部非活跃关节移动超过此容差，判 `CONTEXT_CHANGED` 作废结果[span_157](start_span)[span_157](end_span)。 |
| `sync_secondary_cost` / `w_com` / `w_collision` | `False` / `100.0` / `0.0`[span_158](start_span)[span_158](end_span) | `process_seed` 外层代价同步[span_159](start_span)[span_159](end_span) | 开启时将内层质心与避障代价加权并入外层 `cost`，并自动将内层 `waist_null_gain` 提至 `2.0` 对齐外层腰部偏好[span_160](start_span)[span_160](end_span)。 |

---

### 2.6 外层几何容差、裁判打分权重 (`ik_base.py`)

| 参数名 | 默认值 | 使用位置 | 代入公式 / 逻辑与对求解结果的影响 |
| :--- | :--- | :--- | :--- |
| `POS_TOL` / `ORI_TOL` | `5e-4` m / `5e-4` rad[span_161](start_span)[span_161](end_span) | `score_candidate` & 终检[span_162](start_span)[span_162](end_span) | 外层判定 `pose_ok` 的最终硬门槛[span_163](start_span)[span_163](end_span)。 |
| `SAFE_MARGIN` | `0.02` rad[span_164](start_span)[span_164](end_span) | `candidate_status` & `joint_limit_penalty`[span_165](start_span)[span_165](end_span) | 最小关节余量 $< 0.02\text{ rad}$ 时状态降级为 `NEAR_LIMIT`，并触发二次 hinge 惩罚 $\sum (\frac{\max(0, 0.02 - m_i)}{0.02})^2$[span_166](start_span)[span_166](end_span)。 |
| `W_WAIST_RETURN` | `2000.0`[span_167](start_span)[span_167](end_span) | `score_candidate`[span_168](start_span)[span_168](end_span) | $2000 \cdot \|q_{\text{waist}} - q_{\text{ready,waist}}\|_2^2$[span_169](start_span)[span_169](end_span)。外层绝对最高权重，强力偏好不动腰或少动腰的解[span_170](start_span)[span_170](end_span)。 |
| `W_TORQUE` | `500.0`[span_171](start_span)[span_171](end_span) | `update_candidate_final_cost_with_torque`[span_172](start_span)[span_172](end_span) | $500 \cdot \frac{1}{n}\sum (\tau_{g,i} / \tau_{\text{eff},i})^2$[span_173](start_span)[span_173](end_span)。强力偏好低均方负载率的省力构型[span_174](start_span)[span_174](end_span)。 |
| `W_NOM` | `50.0`[span_175](start_span)[span_175](end_span) | `score_candidate`[span_176](start_span)[span_176](end_span) | $50 \cdot \phi^2$（$\phi = \psi - \psi_{\text{ready}}$）[span_177](start_span)[span_177](end_span)。约束 7-DoF 肘部臂角贴近标准姿态[span_178](start_span)[span_178](end_span)。 |
| `W_LIMIT` | `10.0`[span_179](start_span)[span_179](end_span) | `score_candidate`[span_180](start_span)[span_180](end_span) | 乘以 `joint_limit_penalty`，压制贴近物理限位的候选解[span_181](start_span)[span_181](end_span)。 |
| `W_SMOOTH` | `8.0`[span_182](start_span)[span_182](end_span) | `score_candidate`[span_183](start_span)[span_183](end_span) | $8 \cdot \|q - q_{\text{prev}}\|_2^2$[span_184](start_span)[span_184](end_span)。偏好离上一帧更近的平滑解[span_185](start_span)[span_185](end_span)。 |
| `PHI_TIE_TOL` | `1e-5` rad[span_186](start_span)[span_186](end_span) | `candidate_sort_key` 第 3 级[span_187](start_span)[span_187](end_span) | 臂角偏离 $|\phi| < 10^{-5}$ 视为 0（无差异），防止微小浮点噪声干扰排序[span_188](start_span)[span_188](end_span)。 |
| `TOOL_AXIS_X` / `TOOL_CONE_HALF_ANGLE` | `[1, 0, 0]` / `30°`[span_189](start_span)[span_189](end_span) | `tool_dir` / `tool_cone` 模式[span_190](start_span)[span_190](end_span) | 默认工具轴方向与锥角约束默认半角[span_191](start_span)[span_191](end_span)。 |
| `PERIODIC_MIN_SPAN_RAD` | $2\pi + 10^{-6}$[span_192](start_span)[span_192](end_span) | `periodic_capable_mask`[span_193](start_span)[span_193](end_span) | 判定关节限位跨度是否允许做 $\pm 2\pi$ 等价平移的阈值[span_194](start_span)[span_194](end_span)。 |
| `RIGID_BOTTOM_ROW_TOL` / `RIGID_ROTATION_TOL` | `1e-9` / `1e-6`[span_195](start_span)[span_195](end_span) | `validate_transform4x4`[span_196](start_span)[span_196](end_span) | 校验输入 $4 \times 4$ 齐次变换矩阵正交性与末行合法性的数值容差[span_197](start_span)[span_197](end_span)。 |

---

### 2.7 备用历史引擎参数（默认全部关闭，`single_arm_ik.py`）
*   **开关**：`USE_DIRECT_PSI = False`、`USE_LOCAL_SEEDS = False`、`USE_SSIK = False`[span_198](start_span)[span_198](end_span)。
*   **Direct-Psi 参数**：`PORTFOLIO_DIRECT_MAX_NFEV = 150`、`PORTFOLIO_DIRECT_POS_TOL = 5e-4`、`PORTFOLIO_DIRECT_ORI_TOL = 5e-4`、`PORTFOLIO_DIRECT_PSI_TOL = 1.5`、`PORTFOLIO_Q_REF_MODE = "ready"`、`DIRECT_LOCAL_NOISE_SCALES = (0.002, 0.005, 0.01, 0.02, 0.04)`、`PIN_TOP_K_FOR_DIRECT = 15`；以及 `solve_direct_psi` 内置最小二乘权重 `w_pos=100.0, w_ori=20.0, w_psi=20.0, w_seed=0.03, w_bias=0.0, w_center=0.005, w_limit=0.05, limit_margin=0.10`[span_199](start_span)[span_199](end_span)。
*   **SSIK 参数**：`SSIK_RESPECT_LIMITS = True`、`SSIK_ALLOW_REFINEMENT = True`、`SSIK_REFINEMENT_MAX_ITERS = 30`、`SSIK_MAX_SOLUTIONS = True`、`SSIK_USE_READY_AS_SEED = True`、`SSIK_TOP_K_FOR_DIRECT = 10`[span_200](start_span)[span_200](end_span)。


