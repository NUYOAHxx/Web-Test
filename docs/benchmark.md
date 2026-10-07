# Unitree G1 双臂逆运动学 (Dual-Arm IK) Benchmark 评测体系规范

> **评测目标**：建立针对 Unitree G1 人形机器人（16 DoF：2 DoF 腰部 + 双 7 DoF 手臂）双臂逆运动学求解器的多维度、可量化 Benchmark 评估基准。  
> **核心原则**：破除单一“关节解误差 (Joint Error)”的机械比对思维，从数学正确性、算法收敛性能、种子连续性与 OMPL 端到端规划效能进行全链路评测。  
> **测试基准**：对比 DLS / Levenberg-Marquardt、MoveIt 原生求解器 (KDL/LMA)、双臂 QP-IK 及各组件消融 (Ablation) 变体。

---

## 目录

- [一、评测核心原则与基准认知 (Benchmark Philosophy)](#一评测核心原则与基准认知-benchmark-philosophy)
  - [1.1 破除单一“关节角误差”作为 Ground Truth 的误区](#11-破除单一关节角误差作为-ground-truth-的误区)
  - [1.2 四级评测分层架构 (Benchmark Level 1~4)](#12-四级评测分层架构-benchmark-level-14)
- [二、Level 1：数学与底层实现正确性验证 (Mathematical Correctness)](#二level-1数学与底层实现正确性验证-mathematical-correctness)
  - [2.1 正向运动学 (FK) 一致性校验](#21-正向运动学-fk-一致性校验)
  - [2.2 雅可比矩阵数值差分检验 (Jacobian Numerical Check)](#22-雅可比矩阵数值差分检验-jacobian-numerical-check)
  - [2.3 空间位姿误差 (Pose Error) 连续性与退化验证](#23-空间位姿误差-pose-error-连续性与退化验证)
  - [2.4 QP 单步下降检验 (Single-step Descent Check)](#24-qp-单步下降检验-single-step-descent-check)
- [三、Level 2：算法级性能与鲁棒性评测 (Algorithmic Performance)](#三level-2算法级性能与鲁棒性评测-algorithmic-performance)
  - [3.1 对比基准算法 (Baselines)](#31-对比基准算法-baselines)
  - [3.2 核心量化指标体系 (KPI Matrix)](#32-核心量化指标体系-kpi-matrix)
  - [3.3 成功率 (Success Rate) 的形式化严谨定义](#33-成功率-success-rate-的形式化严谨定义)
  - [3.4 求解耗时与长尾延迟分布 (Tail Latency)](#34-求解耗时与长尾延迟分布-tail-latency)
  - [3.5 迭代收敛次数与雅可比条件数分析](#35-迭代收敛次数与雅可比条件数分析)
- [四、多场景评测数据集设计规范 (Benchmark Datasets)](#四多场景评测数据集设计规范-benchmark-datasets)
  - [4.1 数据集分级架构 (B1 ~ B8)](#41-数据集分级架构-b1--b8)
  - [4.2 严格可达 (Reachable) 目标采样策略](#42-严格可达-reachable-目标采样策略)
  - [4.3 腰部共享耦合评测 (Shared-Waist Benchmark)](#43-腰部共享耦合评测-shared-waist-benchmark)
  - [4.4 种子敏感度实验 (Waist Seed Perturbation)](#44-种子敏感度实验-waist-seed-perturbation)
  - [4.5 连续轨迹跟踪跳变评测 (Seed Continuity Benchmark)](#45-连续轨迹跟踪跳变评测-seed-continuity-benchmark)
- [五、Level 3：系统级有效性与 OMPL 规划协同评测 (System-Level & OMPL)](#五level-3系统级有效性与-ompl-规划协同评测-system-level--ompl)
  - [5.1 规划端评测原则：严禁混淆变量](#51-规划端评测原则严禁混淆变量)
  - [5.2 实验设计 1：固定位姿单纯对比 IK](#52-实验设计-1固定位姿单纯对比-ik)
  - [5.3 实验设计 2：真实系统端到端联合测试](#53-实验设计-2真实系统端到端联合测试)
  - [5.4 关键指标：端到端成功率 (End-to-End Success Rate)](#54-关键指标端到端成功率-end-to-end-success-rate)
  - [5.5 路径质量度量 (Path Quality & Smoothness)](#55-路径质量度量-path-quality--smoothness)
- [六、消融实验与工程交付规范 (Ablation Study & Delivery)](#六消融实验与工程交付规范-ablation-study--delivery)
  - [6.1 消融实验方案 (Ablation Study)](#61-消融实验方案-ablation-study)
  - [6.2 综合评测总表模板](#62-综合评测总表模板)
  - [6.3 最终衡量准则：何为优秀的机器人 IK 求解器](#63-最终衡量准则何为优秀的机器人-ik-求解器)

---

## 一、评测核心原则与基准认知 (Benchmark Philosophy)

### 1.1 破除单一“关节角误差”作为 Ground Truth 的误区

在评价双臂腰部协同逆运动学时，切忌陷入“计算出的关节解 $q$ 与预设的某个参考解 $q_{ref}$ 不一致即视为算法错误”的误区。

Unitree G1 上肢运动学系统为高冗余度系统：

$$
q \in \mathbb{R}^{16} \longrightarrow (T_L, T_R) \in \mathcal{SE}(3) \times \mathcal{SE}(3)
$$

由于广义关节坐标自由度大于任务空间流形维度：

$$
\dim(q) = 16 > \dim(x) = 12 \quad (\text{冗余自由度 } n - m = 4)
$$

对于同一个合法的双臂末端目标位姿对 $(T_L^*, T_R^*)$，机械构型在自运动流形（Self-motion Manifold）上存在无穷多个连续的有效解 $\{q_1, q_2, q_3, \dots\}$。

> [!IMPORTANT]
> **评测第一原则**：  
> 只要关节解 $q$ 满足物理关节限位 $q_{min} \le q \le q_{max}$，且正运动学校验误差满足：
> $$FK_L(q) \approx T_L^* \quad \text{且} \quad FK_R(q) \approx T_R^*$$
> 则该解在数学上即为正确合法的 IK 解。

这与 MoveIt 官方 `KinematicsBase` 接口所强调的 **Seed State 语义**完全吻合：对于常规位置逆运动学 `getPositionIK()`，求解器的核心职责是返回距离当前种子状态（Seed）最近的有效局部最优解，而非脱离上下文随机跳变寻找构型。

因此，Benchmark 方案必须建立一套包含数学正确性、算法性能与规划系统端到端效能的科学评估体系，而非单一比较关节绝对坐标。

---

### 1.2 四级评测分层架构 (Benchmark Level 1~4)

针对 16 DoF 双臂协同逆运动学，评测体系划分为四个递进层级：

```text
                        IK Benchmark 体系架构
                                 │
       ┌─────────────────────────┼─────────────────────────┐
       ▼                         ▼                         ▼
   Level 1                    Level 2                   Level 3
  数学正确性                  算法级性能               OMPL 系统级性能
       │                         │                         │
       ├─ FK 一致性              ├─ 求解成功率 (Success)   ├─ 规划成功率 (Planning Rate)
       ├─ Jacobian 差分验证      ├─ 求解延迟 (P95/P99)     ├─ 规划耗时 (Planning Time)
       ├─ Pose Error 连续性      ├─ 迭代次数 (Iterations)  ├─ 轨迹长度 (Path Length)
       └─ QP 单步下降检验        ├─ 关节限位违规率         └─ 端到端成功率 (E2E Success)
                                 └─ 种子偏离度 (Seed Dist)         │
                                                                   ▼
                                                                Level 4
                                                            鲁棒性与极端工况
                                                                   │
                                                                   ├─ 工作空间临界边界
                                                                   ├─ 运动学奇异区
                                                                   └─ 连续轨迹突变抑制
```

- **Level 1（必须通过）**：用于在代码调试期切断隐蔽 Bug，证明雅可比、误差计算与 QP 建模在数学上无差错；
- **Level 2（核心技术指标）**：技术报告与论文对比的标准量化评测；
- **Level 3（系统级价值）**：证明 IK 产出的目标位姿 $q_{goal}$ 是否真正有助于后续运动规划器求解；
- **Level 4（极限工况测试）**：评测算法在奇异区、工作空间边缘与高频连续控制下的鲁棒性。

---

## 二、Level 1：数学与底层实现正确性验证 (Mathematical Correctness)

在与任何其他算法对比之前，必须首先对求解器各底层数学组件进行独立单元验证。

### 2.1 正向运动学 (FK) 一致性校验

生成 $N$ 组（如 $N = 1000$）在物理限位 $[q_{min}, q_{max}]$ 内均匀分布的随机关节角 $q$：

$$
q \longrightarrow \text{RobotState} \longrightarrow \text{FK} \longrightarrow (T_L, T_R)
$$

若项目中维护了独立的运动学解析实现，可与 MoveIt 原生 `RobotState::getGlobalLinkTransform()` 进行交叉校验：

$$
E_{pos} = \|p_{moveit} - p_{impl}\| < 10^{-6}\text{ m}
$$

$$
E_{rot} = \|R_{moveit}^T R_{impl} - I_{3\times3}\|_F < 10^{-6}
$$

该测试作为底层几何拓扑无误的第一道 Sanity Check。

---

### 2.2 雅可比矩阵数值差分检验 (Jacobian Numerical Check)

由于加权 QP 迭代求解严重依赖几何雅可比矩阵 $J(q) \in \mathbb{R}^{12 \times 16}$，一旦某个关节出现符号错误或排列次序错位，求解器将产生异常发散或抖动。

对每一个关节分量 $i \in \{1, \dots, 16\}$ 施加微小摄动 $\epsilon$（如 $\epsilon = 10^{-6}\text{ rad}$）：

$$
q^{(+i)} = q + \epsilon e_i, \qquad q^{(-i)} = q - \epsilon e_i
$$

通过中心有限差分获得数值雅可比矩阵 $J_{num}$，与解析几何雅可比矩阵 $J_{analytic}$ 对比：

$$
E_J = \frac{\|J_{analytic} - J_{num}\|_F}{\|J_{num}\|_F}
$$

**量化判定准则**：在 100 组随机关节状态下统计，要求相对误差：

$$
E_J < 1 \times 10^{-5}
$$

同时记录最大绝对误差（Max Absolute Error）与平均绝对误差（Mean Absolute Error）。

---

### 2.3 空间位姿误差 (Pose Error) 连续性与退化验证

任务空间误差向量包含左右臂的位置误差与旋转误差：

$$
e = \begin{bmatrix} e_p^L \\ e_R^L \\ e_p^R \\ e_R^R \end{bmatrix} \in \mathbb{R}^{12}
$$

构造已知偏差的合成位姿对 $(T_{current}, T_{target})$，例如施加纯平移 $\Delta p = [10, 0, 0]\text{ mm}$ 与微小纯旋转 $\Delta \theta = 5^\circ$：
- 检验 $\|e_p\| \approx 0.01\text{ m}$；
- 检验 $\|e_R\| \approx 5 \times \frac{\pi}{180}\text{ rad}$。

> [!WARNING]
> **姿态对偶性与奇异性测试**：  
> 必须专项测试 $0^\circ$、接近 $180^\circ$、$\pm\pi$ 邻域，以及四元数对偶性（$q$ 与 $-q$ 代表相同旋转）。验证算法是否具备符号翻转归一化处理（$\text{if } q_1 \cdot q_2 < 0 \implies q_2 \leftarrow -q_2$），避免姿态误差突变引入剧烈速度指令。

---

### 2.4 QP 单步下降检验 (Single-step Descent Check)

给定当前构型 $q_k$ 与目标位姿，由 QP 求解获得单步关节增量 $\Delta q$。

检验一次迭代后的残差收敛趋势：

$$
e_{k} = x^* - x(q_k), \qquad e_{k+1} = x^* - x(q_k + \Delta q)
$$

在未遭遇硬限位截断的常规区间内，单步更新后必须严格保证：

$$
\|e_{k+1}\|_W < \|e_k\|_W
$$

若连续多组测试中频繁出现 $\|e_{k+1}\| > \|e_k\|$，表明雅可比符号、误差方向定义或 QP 线性项梯度矩阵推导存在逻辑缺陷。

---

## 三、Level 2：算法级性能与鲁棒性评测 (Algorithmic Performance)

### 3.1 对比基准算法 (Baselines)

为了全面衡量求解器性能，建立以下三类标准对比方法：

1. **Baseline A：阻尼最小二乘法 (DLS / Levenberg-Marquardt)**  
   标准迭代形式：
   $$\Delta q = J^T (J J^T + \lambda^2 I)^{-1} e$$
   作为经典数值迭代逆运动学基准（关节限位采用后验 Clamping 处理）。
2. **Baseline B：MoveIt 现有求解器 (KDL / LMA / IKFast 插件)**  
   使用 MoveIt 原生 kinematics 插件接口进行求解。  
   *注：若现有插件仅支持单臂 7 DoF 分别求解，则仅用于局部参照，不作为同等 16 DoF 联合优化任务的严格对比项。*
3. **Proposed Method：双臂加权 QP 逆运动学 (Dual-Arm QP-IK)**  
   将 16 DoF 联合空间、种子姿态保持与硬限位约束统一建模求解。

---

### 3.2 核心量化指标体系 (KPI Matrix)

| 评价维度 | 指标名称 | 物理意义 | 期望特性 |
|---|---|---|---|
| **精度与成功率** | **Success Rate (%)** | 满足位姿精度与限位的测试用例比例 | 越高越好 ($\ge 98\%$) |
| **末端残差** | **Position Error (mm)** | 收敛解的末端三维欧氏距离残差 | 均值 $< 1.0\text{ mm}$ |
| | **Rotation Error (deg)** | 收敛解的末端旋转测地距离残差 | 均值 $< 0.5^\circ$ |
| **计算效率** | **Mean Runtime (ms)** | 单次 IK 调用的平均耗时 | 实时级 ($< 3.0\text{ ms}$) |
| | **P95 / P99 Runtime (ms)** | 单次 IK 调用的长尾分位数耗时 | 无突发卡顿 ($< 8.0\text{ ms}$) |
| **数值稳定性** | **Iteration Count** | 达到收敛所需的迭代步数 | 均值 $< 15$ 步 |
| **物理约束** | **Joint-Limit Violation Rate** | 违背关节限位的比例（硬约束） | 严格为 $0\%$ |
| **构型平滑度** | **Seed Distance ($D_{seed}$)** | 求解结果相对于输入 Seed 的加权距离 | 在满足任务下越小越好 |

---

### 3.3 成功率 (Success Rate) 的形式化严谨定义

严禁直接将 `solver.solve() == true` 判定为成功。测试框架必须在外部通过真实正向运动学（FK）重新计算实际位姿并复核约束：

$$
\text{Success}(q) = \begin{cases}
1, & \left( \|e_p^L\| \le \epsilon_p \land \|e_R^L\| \le \epsilon_R \land \|e_p^R\| \le \epsilon_p \land \|e_R^R\| \le \epsilon_R \land q \in [q_{min}, q_{max}] \right) \\
0, & \text{otherwise}
\end{cases}
$$

推荐工程测试阈值设定：
- **位置容差**：$\epsilon_p = 1.0\text{ mm}$（精密操作）或 $5.0\text{ mm}$（常规移动搬运）；
- **姿态容差**：$\epsilon_R = 0.5^\circ$ 或 $1.0^\circ$。

---

### 3.4 求解耗时与长尾延迟分布 (Tail Latency)

机械臂运动规划与控制系统对长尾延迟（Tail Latency）高度敏感，单次 IK 偶发的 50 ms 阻塞将直接破坏控制周期的确定性。

统计指标必须包含分位数与极值分布：

$$
\{\text{Mean}, \quad \text{Median (P50)}, \quad \text{P90}, \quad \text{P95}, \quad \text{P99}, \quad \text{Max}\}
$$

典型评测数据示例：

```text
QP-IK Performance Profile:
  Mean:   1.82 ms
  Median: 1.45 ms
  P95:    3.20 ms
  P99:    4.65 ms
  Max:    8.12 ms (未出现百毫秒级长尾)
```

---

### 3.5 迭代收敛次数与雅可比条件数分析

1. **迭代步数统计**：监控常规工况与奇异临界工况下的迭代耗费，评估步长更新策略与线搜索的加速效能；
2. **雅可比条件数 (Condition Number)**：
   $$\kappa(J) = \frac{\sigma_{max}(J)}{\sigma_{min}(J)}$$
   记录各测试用例中 $\kappa(J)$ 与求解耗时、迭代步数的相关性，分析求解器在近奇异区（$\kappa(J) \gg 10^3$）下的数值阻尼衰减能力。

---

## 四、多场景评测数据集设计规范 (Benchmark Datasets)

评测切忌仅在空间前方平坦区域随机生成 100 个简单位姿，必须建立系统化的测试工况库。

### 4.1 数据集分级架构 (B1 ~ B8)

```text
                          IK Benchmark 数据集矩阵
                                     │
      ┌──────────────┬───────────────┼───────────────┬──────────────┐
      ▼              ▼               ▼               ▼              ▼
   B1 常规可达     B2 工作空间边界   B3 关节限位临界   B4 奇异位形     B5 腰臂耦合
  (1000 cases)     (200 cases)      (200 cases)     (200 cases)    (200 cases)
                                     │
                     ┌───────────────┴───────────────┐
                     ▼                               ▼
               B6 种子扰动                     B7 连续时序轨迹
             (100 × N cases)                    (50 paths)
```

- **B1 Random Reachable (1000 组)**：全工作空间可达采样，评估基准吞吐率；
- **B2 Workspace Boundary (200 组)**：极远伸展、极高或极低作业区，测试极限伸展能力；
- **B3 Near Joint Limits (200 组)**：关节靠近极限值区间，对比 QP 硬约束与 DLS Clamping 差异；
- **B4 Near Singularities (200 组)**：肘部完全伸直、手腕共面等奇异位形区，测试数值稳定性；
- **B5 Dual-arm Coupling (200 组)**：双手大范围交叉或异向展开，验证共享腰部的协调分配；
- **B6 Seed Perturbation (100 个目标 × 10 种种子)**：测试相同目标在不同种子下的构型选择；
- **B7 Continuous Trajectory (50 条空间轨迹)**：连续空间路径插值点，测试相邻解的连续性；
- **B8 Collision / Planning (200 组)**：带环境障碍工况，对接 OMPL 规划。

---

### 4.2 严格可达 (Reachable) 目标采样策略

> [!CAUTION]
> 严禁直接在空间三维包围盒内均匀随机采样位置 $[x, y, z] \in [-1.5, 1.5]^3$。非可达点占据大多数会导致测试结果失真。

**标准数据生成协议（Ground Truth Sampling Protocol）**：

```text
 合法关节空间均匀采样: q_gt ∈ [q_min, q_max]
              │
              ▼
    真实正运动学校验: (T_L^*, T_R^*) = FK(q_gt)  ─── 严格保证空间 100% 物理可达
              │
              ▼
 注入扰动生成初猜种子: q_seed = q_gt + N(0, σ^2) 截断于限位内
              │
              ▼
  输入求解器进行逆解: q_sol = Solver(q_seed, T_L^*, T_R^*)
              │
              ▼
 评估指标计算: FK 残差 ||FK(q_sol) - T^*||、种子距离 ||q_sol - q_seed||
```

数据格式规范存储示例：

```json
{
  "case_id": "B1_0042",
  "q_ground_truth": [0.0, 0.15, -0.2, 0.5, ...],
  "target_pose_left": {"position": [0.35, 0.25, 0.10], "orientation": [0, 0, 0, 1]},
  "target_pose_right": {"position": [0.35, -0.25, 0.10], "orientation": [0, 0, 0, 1]},
  "q_seed": [0.05, 0.10, -0.15, 0.45, ...]
}
```

---

### 4.3 腰部共享耦合评测 (Shared-Waist Benchmark)

Unitree G1 拥有 2 DoF 共享腰部关节（Yaw, Pitch）。这是双臂系统与独立双单臂系统最显著的区别：

$$
x_L = f_L(q_W, q_L), \qquad x_R = f_R(q_W, q_R)
$$

设计对比实验：
- **方案 A（独立解耦 IK）**：先固定腰部或采用独立单臂求解；
- **方案 B（双臂全身 QP-IK）**：由联合雅可比 $J = [J_L; J_R] \in \mathbb{R}^{12 \times 16}$ 统一优化。

**预期结论**：在双手同向大范围搬运或异向避障场景下，方案 B 的求解成功率应显著高于方案 A，且腰部能自适应倾斜补偿双臂伸展不足。

---

### 4.4 种子敏感度实验 (Waist Seed Perturbation)

固定相同的双手末端目标 $(T_L^*, T_R^*)$，人工指定离散的腰部初始种子：

$$
q_{waist}^{seed} \in \{-20^\circ, \quad -10^\circ, \quad 0^\circ, \quad +10^\circ, \quad +20^\circ\}
$$

对比求解器输出：
1. 是否全部收敛至满足容差的有效解；
2. 求解结果的腰部角度是否呈现单调跟随种子倾向；
3. 总关节角改变量 $\|q_{sol} - q_{seed}\|_W$ 是否受到显式最小化抑制。

---

### 4.5 连续轨迹跟踪跳变评测 (Seed Continuity Benchmark)

在连续笛卡尔末端轨迹上，上一帧解作为下一帧的种子：

$$
q_{seed}^{(k)} = q_{sol}^{(k-1)}
$$

统计离散轨迹点间的最大关节阶跃：

$$
\Delta q_k = q_{sol}^{(k)} - q_{sol}^{(k-1)}, \qquad \text{JumpMetric} = \max_k \|\Delta q_k\|_\infty
$$

若轨迹中出现突兀翻转（如肘部反关节跳变、腰部突变 $180^\circ$），则判定位形连续性失效。优秀求解器应呈现平滑连续的关节运动响应曲线。

---

## 五、Level 3：系统级有效性与 OMPL 规划协同评测 (System-Level & OMPL)

### 5.1 规划端评测原则：严禁混淆变量

逆运动学并非孤立存在，其在机器人操作软件栈中的核心价值在于为运动规划器（如 OMPL RRT-Connect）提供高质量、易于规划的目标构型 $q_{goal}$。

> [!WARNING]
> **评测控制变量原则**：  
> 当对比由不同 IK 求解器生成的 $q_{goal}^A$ 与 $q_{goal}^B$ 在 OMPL 中的规划表现时，**必须确保所有规划参数严格一致**：相同的起始状态 $q_{start}$、相同的障碍物环境模型、相同的 RRT 参数、随机数种子序列及超时阈值（Timeout）。

---

### 5.2 实验设计 1：固定位姿单纯对比 IK

输入相同的笛卡尔位姿对与相同的种子，仅评估各 IK 算法的收敛速度、残差精度与种子距离。

### 5.3 实验设计 2：真实系统端到端联合测试

完整的流水线联调评估：

```text
 [笛卡尔空间目标] ───> [IK Solver 求解] ───> q_goal ───> [OMPL 路径规划] ───> [执行轨迹]
```

统计从给出目标到位姿完成规划的全链路耗时与最终有效性。

---

### 5.4 关键指标：端到端成功率 (End-to-End Success Rate)

定义系统级端到端综合成功率：

$$
\text{Success}_{E2E} = \text{Success}_{IK} \times \text{Success}_{Planning}
$$

即使两款求解器的纯运动学成功率相似，构型优选能力的差异在规划层也将显著放大：

| 求解器类型 | IK 成功率 | 目标无碰撞率 | OMPL 规划成功率 | 端到端成功率 (E2E) |
|---|---:|---:|---:|---:|
| **DLS + Clamp** | $96.2\%$ | $81.5\%$ | $74.2\%$ | **$71.4\%$** |
| **MoveIt LMA** | $97.5\%$ | $85.0\%$ | $80.1\%$ | **$78.1\%$** |
| **Dual-Arm QP-IK** | **$98.8\%$** | **$95.4\%$** | **$92.6\%$** | **$91.5\%$** |

> [!NOTE]
> QP-IK 依托**种子姿态保持**与**关节限位软/硬约束优化**，生成的 $q_{goal}$ 天然远离自碰撞与奇异姿态，更接近起始状态流形，因此大幅提升了 OMPL 搜索无碰撞路径的成功率。

---

### 5.5 路径质量度量 (Path Quality & Smoothness)

统计 OMPL 最终生成的有效轨迹几何指标：
1. **关节空间总弧长**：
   $$L_q = \sum_{k=1}^{M-1} \|q_{k+1} - q_k\|_2$$
2. **末端笛卡尔路径迂回度**：真实末端积分轨迹长度与直线欧氏距离之比；
3. **峰值加速度与抖动**：轨迹多项式时间参数化（TOPP-RA）后的导数峰值。

---

## 六、消融实验与工程交付规范 (Ablation Study & Delivery)

### 6.1 消融实验方案 (Ablation Study)

为清晰阐明双臂 QP-IK 内部各项机制的独立贡献，设计递进消融实验：

- **Variant A (Base DLS)**：经典阻尼最小二乘，后验截断关节限位；
- **Variant B (Pure QP)**：仅最小化末端位姿残差 $\frac{1}{2}\|J\Delta q - e\|_W^2 + \frac{1}{2}\|\Delta q\|_R^2$，无限位无种子跟踪；
- **Variant C (QP + JL)**：在 Variant B 基础上加入关节物理限位硬约束 $q_{min} \le q + \Delta q \le q_{max}$；
- **Variant D (QP + Seed)**：在 Variant B 基础上引入种子偏好项 $\frac{1}{2}\|q + \Delta q - q_{seed}\|_R^2$；
- **Variant E (Full QP-IK)**：完整模型（硬限位 + 种子跟踪 + 步长限幅）。

消融量化对比矩阵：

| 实验组 | 算法变体 | 成功率 (%) | 限位违规率 (%) | 种子偏离度 $D_{seed}$ | 平均耗时 (ms) | OMPL 规划成功率 (%) |
|:---:|---|---:|---:|---:|---:|---:|
| A | Base DLS | 94.2 | 12.8 (截断前) | 1.84 | 1.4 | 72.5 |
| B | Pure QP | 95.1 | 14.5 | 1.95 | 1.7 | 73.0 |
| C | QP + JL | 97.4 | **0.0** | 1.42 | 1.9 | 84.6 |
| D | QP + Seed | 96.8 | 8.2 | **0.45** | 1.8 | 87.2 |
| E | **Full QP-IK** | **98.8** | **0.0** | **0.52** | 2.1 | **92.6** |

---

### 6.2 综合评测总表模板

测试报告交付最终汇总模板：

| 方法 | Success Rate | Pos Error (mm) | Rot Error (deg) | Mean Time (ms) | P95 Time (ms) | Mean Iter | Seed Dist | OMPL E2E |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **DLS** | 94.2% | 0.85 | 0.62 | 1.42 | 3.10 | 18.2 | 1.84 | 71.4% |
| **MoveIt KDL**| 91.5% | 1.20 | 0.88 | 2.65 | 6.80 | 24.5 | 2.15 | 68.2% |
| **MoveIt LMA**| 96.5% | 0.65 | 0.45 | 3.10 | 7.50 | 19.8 | 1.62 | 78.1% |
| **Dual-Arm QP**| **98.8%** | **0.32** | **0.21** | **2.12** | **4.20** | **11.4** | **0.52** | **91.5%** |

---

### 6.3 最终衡量准则：何为优秀的机器人 IK 求解器

> [!TIP]
> **评测终局判断准则**：  
> 评判 Unitree G1 双臂逆运动学求解器的优劣，绝非单一指标“比 DLS 快 0.5 毫秒”，而是要求其在**相同物理模型、相同双臂目标位姿与相同种子先验**的约束下：
> 1. **稳定高成功率**：在工作空间边缘与高冗余流形上稳定收敛（Success Rate $\ge 98\%$）；
> 2. **物理硬界完备**：在底层内核天然杜绝关节超限，绝无暴力截断导致的轨迹突变；
> 3. **空间拓扑连续**：保持与先验构型的高度连续性，彻底消除反肘、甩腰等跳变构型；
> 4. **赋能规划系统**：输出高质量的 $q_{goal}$，直接提升 MoveIt 2 / OMPL 的无碰撞规划成功率与运动平滑度。