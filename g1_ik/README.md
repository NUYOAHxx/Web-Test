# G1 双臂 + 腰部 IK 原型（Pinocchio + placo QP + CasADi/IPOPT NLP + Coal 校验）

在 Unitree G1（29-DOF rev_1_0 官方 URDF）上实现的多模式 IK 求解器原型。用一个 16-DOF 统一模型（waist_yaw、waist_pitch 加左右各 7 个臂关节）配合 **mode mask** 实现三种模式：

| 模式 | 自由关节 | 说明 |
|---|---|---|
| M1 | 单臂 7 DOF（给两只手的目标时是双臂 14 DOF） | 腰锁定在当前值 |
| M2 | 腰 2 DOF + 单臂 7 DOF | |
| M3 | 腰 2 DOF + 双臂 14 DOF | 两手各自目标；可选相对位姿约束（双手搬箱子）和物体质量（平均分到两只手） |
| auto | 先 M1，失败或 力矩利用率 > 阈值 时升级到 M2/M3 | 每一级的结果和升级原因都写入 `log` |

waist_roll 锁 0（模拟只有 yaw+pitch 的腰）。腿锁在站立名义姿态，pelvis 当作固定基座（世界系 = pelvis 系，z 朝上）。**所有机器人相关信息都在 `config/g1.yaml` 中。**

## 1. 求解流程（每个 mode / stage）

```
L1 种子 ──> L2 placo QP（关节限位 + 胶囊自碰撞，硬约束）──> 多样性筛选 n_refine 个
   ──> L3 CasADi + IPOPT NLP（精确位姿 / 静力矩 / 胶囊距离 / 肩窝距离表）──> L4 全状态校验 ──> Top-K 排序
```

- **L1 种子**：当前构型、名义手臂；臂角（swivel）圆上的肘部点（−60/−20/20/60/100°）作为 QP 前半段的肘部位置任务；随机臂构型；M2/M3 另外加腰部启发式（转向/前倾朝目标；双手时考虑目标朝向）和腰部网格种子（yaw ∈ {−0.5, 0, 0.5} × pitch ∈ {0, 0.3}）。
  - **G1 手臂不是严格的 SRS**（`tests/check_srs.py` 实测：肩部三轴相互距离 13.8 / 0 / 6.2 mm，腕部 0 / 0 / 46 mm，肘偏置 15.8 mm），所以没有用 SRS 解析解，只用臂角做种子引导。用户自己的臂是真 SRS，可以在 `seeds.py` 里接入解析解。
- **L2 QP（placo 0.10）**：KinematicsSolver 用 `mask_dof` 屏蔽非自由关节；手部 frame task（soft）；双手相对 frame task；腰部 joints task（锚定在种子的腰部值，保留种子多样性）。`add_avoid_self_collisions_constraint` 做硬约束（margin 8 mm，trigger 6 cm），作用于本 mode 相关的胶囊对。还加了 Levenberg–Marquardt 式阻尼：正则权重 = 1e‑4 + 0.2·误差²。QP 层只管运动学，力矩不在这里处理。
- **L3 NLP（CasADi 3.8 + IPOPT）**：
  - 变量：自由关节，带关节限位。
  - 位置误差按分量约束 |e_i| ≤ tol/√3（tol = 0.2 mm）。
  - 姿态误差 vee(R*ᵀR) 按分量约束 ≤ sin(tol)/√3（tol = 0.002 rad），并约束 θ ≤ 90°，排除 θ≈180° 的分支。
  - 双手相对位姿的约束形式与上面相同。
  - 静力矩：τ = g(q) − Σ Jᵀ m g（手上有负载），要求 |τ_i/τmax_i| ≤ 0.9。
  - 胶囊距离 ≥ 8 mm，作用于本 mode 相关的所有胶囊对。
  - 肩窝距离表 ≥ 5 mm（见第 4 节）。
  - 代价：w_waist‖Δq_waist‖² + w_reg‖Δq_arm‖² + w_τ‖τ/τmax‖²。
  - 按 (自由关节, 手, 是否有相对约束, pair 集) 构建一次后缓存，目标、负载、当前构型都作为 NLP 参数传入。
- **L4 校验**：
  - Coal 精确 mesh 碰撞（BVH，用 SRDF 剪枝后的 141 个 link 对；rubber hand 没有 collision mesh，所以用它的 visual mesh）。
  - 间隙用凸包距离计算。凸包距离 < 2 cm 的对再用精确 BVH mesh 距离复核。
  - 力矩利用率 ≤ 1、关节限位、位姿误差（2 mm / 1°）、相对位姿误差也都要检查。
  - **只有通过 L4 的解才算成功。**
- **排序**：5·腰部位移 + 1·最大力矩利用率 + 0.2·‖q − q_cur‖ + 0.002/mesh 间隙，然后去重。

## 2. 目录结构

```
g1_ik/
├── config/g1.yaml               # 机器人相关的一切：关节名、TCP、锁定关节、胶囊、碰撞参数、求解参数
├── g1ik/
│   ├── config.py                # 读 yaml，路径相对仓库根目录解析
│   ├── model.py                 # Pinocchio 完整模型 -> 16-DOF 降阶模型，TCP frame，静力矩，可操作度
│   ├── collision.py             # 胶囊拟合、SRDF 读取、mode 相关 pair、numpy 胶囊距离、Coal mesh/凸包校验、肩窝关节空间距离表
│   ├── casadi_kin.py            # 自写 CasADi 符号 FK / 重力矩 / 负载力矩 / 线段-线段距离
│   ├── seeds.py                 # 臂角种子、腰部启发式/网格种子、随机种子
│   ├── qp.py                    # L2 placo QP 层
│   ├── nlp.py                   # L3 CasADi/IPOPT 层
│   └── solver.py                # IKRequest / IKSolver：模式、auto 升级、校验、Top-K
├── tools/gen_collision_srdf.py  # 类 MoveIt Setup Assistant 的碰撞矩阵生成（Adjacent/Default/Always/Never，10000 次采样）
├── generated/                   # 自动生成：g1_collision.srdf、g1_capsules.json、joint_tables.npz
├── tests/
│   ├── test_consistency.py      # 自写 FK/力矩/胶囊距离与 pinocchio/Coal 的一致性检查
│   ├── check_srs.py             # 检查手臂是否为 SRS
│   ├── smoke_m1.py              # M1 快速冒烟测试
│   ├── run_benchmark.py         # 基准测试（场景见第 5 节）
│   ├── run_all.sh               # 分进程跑全部场景并合并成 results/benchmark.json
│   ├── summarize.py             # 生成 results/summary.md
│   └── make_figures.py          # 生成 figures/*.png
├── examples/solve_example.py    # 最小调用示例
├── results/                     # benchmark.json、summary.md、各分组日志
├── figures/                     # PNG 图
└── assets/g1_description/       # 官方 URDF + meshes（来自 unitree_ros，BSD-3，见 LICENSE_unitree_ros）
```

## 3. 怎么运行

```bash
cd g1_ik
uv venv -p 3.12 .venv && . .venv/bin/activate      # 或 python -m venv
uv pip install -r requirements.txt                  # pin(pinocchio+coal) casadi placo numpy scipy trimesh matplotlib pyyaml
export PYTHONPATH=.

python tests/test_consistency.py          # 一致性检查
python examples/solve_example.py          # 调用示例
N=20 bash tests/run_all.sh                # 全部基准（本机约 8 分钟）-> results/benchmark.json
python tests/summarize.py                 # -> results/summary.md
python tests/make_figures.py              # -> figures/*.png
python tools/gen_collision_srdf.py        # 重新生成 SRDF（约 4–5 分钟，换机器人时才需要）
```

调用接口：

```python
from g1ik.config import load_config
from g1ik.solver import IKSolver, IKRequest
S = IKSolver(load_config("config/g1.yaml"))
res = S.solve(IKRequest(targets={"left": T_left, "right": T_right},   # pin.SE3 或 4x4，pelvis 系下的 TCP 位姿
                        mode="auto",                                    # "M1" | "M2" | "M3" | "auto"
                        q_current=None,                                 # 16 维（求解器顺序），None = 名义构型
                        payload={"left": 0.5},                          # 每只手的负载 kg
                        relative=T_left.inverse() * T_right,            # 可选：双手相对位姿
                        object_mass=3.0))                               # 可选：双手共持物体，平均分到两手
res["success"], res["solutions"][0]["q"], res["log"]   # solutions 已按 Top-K 排序；每个解都带完整指标和 fail 原因
```

## 4. 移植到你自己的机器人（改 config 即可）

1. `robot.urdf / mesh_dir`：换成你的 URDF。
2. `locked_joint_values`：腿和其他不参与的关节锁定值。你的机器人没有 waist_roll，这一行删掉即可；URDF 中没有的关节不需要写。
3. `waist_joints`：`[waist_yaw_joint, waist_pitch_joint]` 改成你的关节名。
4. `arms.left/right`：
   - `joints`：7 个关节名，按顺序；
   - `hand_link`：末端 link；
   - `ee_offset_xyz`：TCP 相对 hand_link 的偏移；
   - `shoulder_joint / elbow_joint / wrist_joint`：臂角种子用。你的手臂是 SRS，建议在 `seeds.py` 里加解析解种子。
5. `nominal_q`：名义构型；`torque_limit_override`：URDF effort 不准时在这里覆盖；`torque_margin`：力矩安全系数 ρ。
6. `collision`：
   - 先运行 `tools/gen_collision_srdf.py` 生成新的 SRDF。
   - 在 `capsule_links` 里列出要做胶囊的 link 及段数。扁平零件（手掌）用 `axis_pca: 1`。
   - 胸/躯干这类形状不规则的 link 建议用 `manual_capsules` 手工给。
   - 删除 `generated/g1_capsules.json` 和 `generated/joint_tables.npz` 让它们重新生成。
   - `capsule_disabled_link_pairs`：胶囊近似明显不合理的 link 对，只在胶囊层关闭，mesh 校验仍然检查。
   - `joint_tables`：自动处理「被胶囊层跳过、且相对位姿只依赖 ≤2 个关节」的 link 对（典型就是肩窝）。
7. `solver`：容差、种子数、权重、`auto_torque_util_threshold`、`waist_seed_grid`、`top_k`。

注意：你的机器人胸部和手臂都装在腰关节以上的躯干上，和 G1 一样，所以「腰运动不改变腰以上 link 之间的相对距离」这条 pair 剪枝规则可以直接用（`relevant_pairs`：两个 link 的 support 关节集合的对称差与自由关节有交集，才算相关）。

## 5. 测试结果（本机实测）

测试环境：8 vCPU Intel Xeon、Python 3.12.15、pin 3.8.0 / Coal 3.0.2、CasADi 3.8.1（IPOPT）、placo 0.10.1。单线程顺序求解。时间是 `time.perf_counter` 测得的**每次请求的总墙钟时间**（包含 auto 模式所有 stage），不含一次性的 NLP 构建（预热时完成：每个 NLP 构建约 1.1 s，每个测试进程预热约 11 s；`IKSolver` 初始化约 5.3 s）。

场景说明（`tests/run_benchmark.py`，固定随机种子）：

- **M1_reachable_l/r**：目标来自随机单臂构型（腰 = 0）的 FK。生成时要求 mesh 间隙 ≥ 15 mm、胶囊距离 ≥ d_safe、力矩利用率 < 0.85。用 M1 求解。
- **M1_hard**：用大幅腰部运动（|yaw| 0.45–0.9，pitch 0.15–0.45）生成目标，并要求腕心超出「当前腰位下肩点 + L1 + L2 + 1 cm」的球。用 M1 求解，预期失败。
  - 因为 G1 肩部有 1–2 cm 的轴偏置，这个判据不是严格证明，所以有 2/20 实际上 M1 可达。
- **M2_reachable**：用腰 + 单臂随机构型生成目标，用 M2 求解。
- **auto_waist_needed**：与 M1_hard 是同一组目标，用 auto 求解。
- **specific_cross_body**：3 个固定目标，左手伸到身体右前下方、TCP x 轴朝下，用 auto 求解。
- **torque_payload_3.0kg / 4.5kg**：单手持重。目标是前伸（x > 0.30）的可达目标，用 auto 求解。
- **M3_separate**：用腰 + 双臂随机构型生成两手目标，用 M3 求解。
- **box_reachable**：双手夹持 30 cm 宽的箱子，加相对位姿约束，物体 1/3/5 kg。
  - 箱子位姿定义在某个随机腰位的 torso 系中，并且先用该腰位下的 M1 双臂求解作为「可行性证书」，证书求解成功的箱子才进入测试。
  - 测试从腰 = 0 开始，用 auto 求解（M1 = 双臂锁腰，然后 M3）。
- **box_random_hard**：箱子位姿在 pelvis 系中随机采样，可行性未知。
- **near_hands_side_by_side**：两手并排放在胸前，TCP 相距 11–14 cm（手部 mesh 间隙约 1–2 cm），用 M3 求解。
- **near_crossed_hard**：双手交叉，可行性未知。
- **infeasible_inside_right_hand**：左手目标放在锁定的右手内部（M1），应当被拒绝。

| 场景 | n | 成功率 | 最终模式分布 | 位置误差 中位/最大 [mm] | 姿态误差 最大 [deg] | 腰部位移 中位/最大 [rad] | 最大力矩利用率 中位/最大 | 最小 mesh 距离 最小 [mm] | 胶囊约束被违反但 mesh 通过 | 求解时间 中位/P90/最大 [s] |
|---|---|---|---|---|---|---|---|---|---|---|
| M1_reachable_l | 10 | 10/10 (100%) | M1:10 | 0.20/0.20 | 0.11 | 0.000/0.000 | 0.15/0.23 | 15.2 | 0/10 | 0.30/0.44/0.67 |
| M1_reachable_r | 10 | 10/10 (100%) | M1:10 | 0.20/0.20 | 0.11 | 0.000/0.000 | 0.12/0.18 | 15.4 | 0/10 | 0.37/0.59/0.94 |
| M1_hard | 20 | 2/20 (10%) | M1:2 | 0.20/0.20 | 0.11 | 0.000/0.000 | 0.12/0.16 | 19.4 | 0/2 | 0.73/1.02/1.13 |
| M2_reachable | 20 | 20/20 (100%) | M2:20 | 0.20/0.20 | 0.11 | 0.011/0.171 | 0.16/0.21 | 13.7 | 0/20 | 0.46/0.69/0.84 |
| auto_waist_needed | 20 | 20/20 (100%) | M1:2, M2:18 | 0.20/0.20 | 0.11 | 0.297/0.947 | 0.30/0.51 | 7.1 | 0/20 | 1.39/1.95/2.29 |
| specific_cross_body | 3 | 3/3 (100%) | M2:3 | 0.20/0.20 | 0.11 | 0.532/0.904 | 0.39/0.45 | 5.0 | 0/3 | 1.08/1.18/1.20 |
| torque_payload_3.0kg | 10 | 10/10 (100%) | M1:9, M2:1 | 0.20/0.20 | 0.11 | 0.000/0.021 | 0.62/0.90 | 20.2 | 0/10 | 0.28/0.76/1.28 |
| torque_payload_4.5kg | 10 | 7/10 (70%) | M1:5, M2:2 | 0.20/0.20 | 0.11 | 0.000/0.247 | 0.81/0.90 | 20.2 | 0/7 | 0.75/1.18/1.28 |
| M3_separate | 20 | 20/20 (100%) | M3:20 | 0.20/0.20 | 0.11 | 0.091/0.488 | 0.23/0.46 | 5.0 | 0/20 | 0.98/1.80/3.02 |
| box_reachable | 20 | 18/20 (90%) | M1:7, M3:11 | 0.20/0.20 | 0.11 | 0.038/0.401 | 0.56/0.90 | 6.0 | 0/18 | 2.92/4.92/6.27 |
| box_random_hard | 20 | 5/20 (25%) | M1:2, M3:3 | 0.20/0.20 | 0.11 | 0.002/0.236 | 0.62/0.63 | 5.0 | 0/5 | 3.22/4.05/4.76 |
| near_hands_side_by_side | 20 | 19/20 (95%) | M3:19 | 0.20/0.20 | 0.11 | 0.060/0.416 | 0.27/0.46 | 12.6 | 14/19 | 2.36/2.85/3.18 |
| near_crossed_hard | 5 | 0/5 (0%) | - | - | - | - | - | - | - | 2.59/3.05/3.28 |
| infeasible_inside_right_hand | 3 | 0/3 (0%) | - | - | - | - | - | - | - | 1.25/1.48/1.54 |

说明：
- 位置误差中位数正好是 0.20 mm，因为 NLP 位置容差设为 0.2 mm，IPOPT 停在约束边界上（代价项把解往当前构型拉）。
- 「胶囊约束被违反但 mesh 通过」：手部胶囊对扁平手掌仍然偏保守，两手并排时 IPOPT 会报 `Infeasible_Problem_Detected`，但得到的构型位姿满足要求、精确 mesh 间隙 ≥ 12.6 mm，所以被 L4 接受。每个解都带 `capsule_violation` 标志。
- **没有任何一个被接受的解存在 mesh 碰撞**（L4 是最终判据）。191 个请求中有 2 个失败请求的最优尝试含 mesh 碰撞，被正确拒绝。

auto 升级统计（摘自 `results/summary.md`）：
- auto_waist_needed：18/20 是 M1 无有效解后升级到 M2；另外 2 个直接在 M1 解出。
- torque_payload_4.5kg：
  - 3 次是 M1 有解但力矩利用率 > 0.8，因此升级；
  - 4 次是 M1 无有效解后升级；
  - 3 个请求最终失败：瓶颈都是 **wrist_yaw（τmax = 5 N·m）**，验证时 τ/τmax = 1.01–1.07。腰部无法改变给定 TCP 姿态下腕关节承受的重力矩，所以升级也无济于事。这个失败符合物理。
- box：M1（双臂锁腰）无解时升级到 M3。

各层平均耗时（每个 stage）：
- 单臂：QP 0.07–0.18 s，NLP 0.2–0.6 s，校验约 0.02 s。
- 双臂：QP 0.3–0.5 s，NLP 0.8–1.6 s，校验 0.03–0.05 s。

时间大部分花在 IPOPT 上：每个 stage 要跑 n_refine = 4 个候选，M3 有约 230 个胶囊约束。

图（`figures/`）：
- `capsules_nominal.png`：胶囊模型（30 个胶囊，含 4 个手工躯干胶囊）。
- `waist_needed_m1_vs_auto.png`：同一目标下 M1 的最佳失败尝试与 auto→M2 的解。
- `box_both_hands_topk.png`：双手搬 3 kg 箱子。
- `near_arm_arm.png`：两手相邻，并显示胶囊。
- `benchmark_summary.png`：各场景成功率与时间分布。
- `util_vs_waist.png`：力矩利用率与腰部位移的散点图。

一致性检查（`tests/test_consistency.py`，随机构型）：

| 检查项 | 最大偏差 |
|---|---|
| CasADi FK vs pinocchio | 1.2e‑16 m |
| CasADi 静力矩 vs pinocchio | 7.1e‑15 N·m |
| numpy 胶囊距离 vs Coal | 2.2e‑16 m |
| CasADi 胶囊距离 vs numpy | 1.1e‑8 m（换成 8 段手部胶囊后重测） |

## 6. 与原设计的偏差及原因

1. **没有使用 `pinocchio.casadi`**：pip 版 `pin` wheel 不带 casadi 绑定。所以在 `casadi_kin.py` 里自己写了符号 FK、重力矩（势能梯度）和负载力矩，并与 pinocchio 交叉验证（见上表）。
2. **没有 SRS 解析种子**：G1 手臂不是严格 SRS（见第 1 节），改用臂角引导的数值种子。
3. **QP 层不含力矩约束**（placo 的 `add_joint_space_half_spaces_constraint` 没有使用）。力矩约束只在 NLP 中精确施加，并在 L4 校验。
4. **自碰撞的分层处理**：
   - 胶囊（QP + NLP）+ 精确 mesh（L4）。
   - nominal 时就重叠的 4 对「躯干胶囊 – shoulder_roll 胶囊」，以及 head – shoulder_roll 胶囊对，在胶囊层关闭。这几对在全关节空间几乎恒定贴近，胶囊近似失效。
   - 作为补充，新增了**肩窝关节空间距离表**：这些 link 对的相对位姿只依赖 shoulder_pitch 和 shoulder_roll 两个关节，所以离线在 0.04 rad 网格上算出精确 mesh 距离（碰撞区内部用距离变换给负值），再用 CasADi B-spline 插值作为 NLP 约束（≥ 5 mm）。
   - 加入距离表之前，有 2 个请求的解在 L4 被判为 torso–shoulder_roll 碰撞；加入之后，这类失败在可行场景中消失。
   - 曾尝试「距离不小于 nominal 值」的保守约束，结果几乎锁死了肩部内收，所以默认关闭（`nominal_overlap_slack: null`）。
5. **手部胶囊**：rubber hand 尺寸约 141×109×49 mm，近似一块平板。最初用 2 段沿长轴的胶囊（r ≈ 5 cm），手–手距离被高估约 5 cm。现在改为沿长轴切 4 段、胶囊轴沿手掌宽度方向（r 2.4–3.4 cm），但仍偏保守（见 capsule_violation 统计）。
6. **NLP 不做激活距离预筛选**：使用本 mode 全部相关胶囊对，保证 NLP 结构固定、可以缓存（`nlp_activation` 参数当前未使用）。
7. **位姿约束的写法**：最初用 ‖e‖² ≤ tol² 和 3 − tr ≤ …，这两种形式在最优点梯度为零，破坏 LICQ，IPOPT 大量 `Maximum_Iterations`。改成按分量的盒约束，同时放宽 head–shoulder 胶囊并把 d_safe 从 15 mm 降到 8 mm 后，M1 可达目标的冒烟测试从 4/8 提升到 10/10（两项改动同时生效，没有单独拆分各自的贡献）。
8. **QP 的腰部正则锚定在种子值而非当前值**：否则大腰角种子会被拉回 0，丢失多样性，双手箱子任务会失败。
9. **auto 中原来的「腰部位移上界分级」**（先 ±0.25 rad，再放开）默认关闭（`waist_bounds_schedule: [null]`），因为它使耗时翻倍。最小腰部运动由 NLP 代价 w_waist = 20 和排序共同保证。
10. **校验间隙用凸包距离 + 近距离精确复核**：BVH mesh–mesh 距离全部对算一次约 0.2 s，凸包 GJK 只要约 0.3 ms。碰撞判定本身仍用精确 mesh。
11. 两手持物时，`object_mass` 平均分到两只手（不考虑物体质心位置导致的不均分配）。waist_roll 被锁定且不做力矩检查（用户机器人没有这个关节）。
12. 腿锁定、pelvis 固定且 z 朝上：没有考虑全身平衡（ZMP/CoM）。

## 7. 局限 / 已知问题

- **速度**：单臂约 0.3–1.4 s，双臂 1–3 s，箱子任务约 3 s（中位）。离实时（<10 ms）差很远，适合 TAMP/规划层的离线或准离线调用。提速方向：
  - IPOPT 改用 warm start 或 SQP（例如 `sqpmethod` + qpOASES）；
  - 降低 n_refine，或拿到 K 个有效解就提前停止；
  - 只把接近激活的胶囊对放进 NLP（需要按 pair 集重建或使用松弛变量）；
  - 用 C 代码生成（CasADi codegen）。
- **局部方法**：不保证全局。可行但未解出的例子：box_reachable 有 2/20 失败（证书存在、但从腰 = 0 出发没找到）；near_hands 有 1/20 失败。可行性未知的难例（box_random_hard 25%、near_crossed_hard 0%）成功率低，无法区分「真的不可行」和「没找到」。
- **「hard」集的不可达判据不是严格证明**（G1 不是 SRS），有 2/20 实际可达。
- **力矩模型只有静态**（重力 + 末端负载力），没有动力学、摩擦和电流/温度限制。G1 腕部 5 N·m 的限制使单手负载在约 4 kg 以上时常常不可行。
- **胶囊模型偏保守**，尤其是手部；躯干胶囊是手工给定的，换机器人时需要重新调。
- QP 层在某些种子下会陷入关节限位造成的局部极小，主要靠多种子和 NLP 补救。
- 所有结果都只针对 G1 仿真模型。用户机器人的 SRS 手臂、关节限位和力矩不同，数值不能直接外推。
