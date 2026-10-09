# 2-DOF 腰 + 双 7-DOF 臂（16 DOF）IK 求解器设计调研：方法、物理可行性与推荐架构

> 日期：2026-10-10 ｜ 用途：任务与运动规划（TAMP）中的目标构型求解与采样，**不是遥操作**
> 机器人：腰部 2 DOF（yaw + pitch，无 roll）+ 左右各 7-DOF 臂，共 16 DOF；末端抓取并搬运负载
> 需求：一个求解器同时支持三种模式：
> - **M1**：单臂 IK（腰锁定，7 DOF）
> - **M2**：腰 + 单臂协同（9 DOF）
> - **M3**：腰 + 双臂协同（16 DOF，可含双手持同一物体的闭链约束）
>
> 硬性要求：关节极限、负载与重力下的关节力矩极限、自碰撞、远离奇异；**腰部尽量少动**，但在能降低手臂力矩或扩展可达范围时允许腰部参与。
>
> 说明：所有论文、库和数字都来自本次检索或抓取并核实过的页面。§2 的数学推导是标准机器人学内容，不对应某一篇具体论文，原理处标注了出处。本文只讲设计，不含完整实现。

---

## 0. 结论先行

**推荐方案：「单一 Pinocchio 模型 + 关节掩码切换模式 + 多起点种子 + QP 微分 IK 快速收敛 + NLP（CasADi/IPOPT）带力矩与碰撞约束精修 + 全状态校验与多解排序」的分层结构。**

```
             ┌───────────────────────────────────────────────────────────┐
 目标(位姿/TSR,│ L0 模式与问题构建：active joint mask(M1/M2/M3)、任务、负载 wrench │
 负载,模式) ──►│ L1 种子生成：当前构型/名义姿态/解析臂角扫描(若为SRS)/缓存/随机/  │
             │     (可选)cuRobo 批量或学习型种子                                │
             │ L2 快速局部求解：QP 微分 IK（placo / pink / 自写+ProxQP/OSQP）    │
             │     加权任务 + 关节极限 + 自碰撞 + 线性化力矩约束；腰部高代价      │
             │ L3 精修：NLP（Pinocchio-CasADi + IPOPT，或 Drake IK + SNOPT/IPOPT）│
             │     精确位姿等式/容差、力矩不等式、最小距离、腰部位移上界          │
             │ L4 校验与排序：全状态碰撞(Coal)、静力矩利用率、可操作度、腰位移、   │
             │     与当前构型距离 → 返回 Top-K 多样解给 TAMP                     │
             └───────────────────────────────────────────────────────────┘
 腰部策略：先 M1（腰锁定）→ 失败或力矩利用率超阈值时才进入 M2/M3，并加腰部位移上界与高权重
```

**为什么这样选：**
1. **局部求解器会卡住，单一方法不够。** pink 文档明确说微分 IK 是局部算法。IKDiffuser 论文测了一个 17 DOF 的「双臂 + 腰」平台：只用 cuRobo 成功率 83.39%，用好种子后 99.85%；只用 Pink 成功率 69.38%，加种子后 99.04%。所以要「种子 + 局部求解 + 精修」组合。
2. **力矩约束是非线性的**：$\tau=g(q)-J^\top f$。QP 层只能用一阶线性化处理，最终要在 NLP 层精确检查。Pinocchio 提供 `computeStaticTorque` 和 `computeStaticTorqueDerivatives`，CasADi 后端可自动微分，这一层可以直接实现。
3. **三种模式用同一个模型、同一套代价**，只改「哪些关节可以动」。不重建模型，所以锁定关节的重力和负载仍然算进力矩检查。这一点很重要：腰锁定时，腰电机照样承受双臂和负载的重量。

---

## 1. 候选求解方法：原理与在本机器人上的优缺点

### 1.1 解析 7-DOF 臂 IK（臂角 / 肘部参数化）+ 腰部采样

**原理.** 对 S-R-S 构型（球腕-转肘-球肩，类人臂）的 7-DOF 臂，末端位姿确定后，肘部还能绕「肩-腕连线」转一个角度，这个冗余度用**臂角** $\psi$ 参数化。给定位姿 $T$ 和 $\psi$，7 个关节有闭式解：

$$
q_{arm} = \mathrm{IK}_{arm}(T, \psi, \text{branch}),\qquad \text{branch}\in\{\text{肩/肘/腕的离散分支}\}
$$

Shimizu et al.（IEEE T-RO 2008）推导了基于参数化的闭式解，并**解析地**分析了关节极限如何限制 $\psi$ 的可行区间：肩、腕关节的可行性条件可写成 $f(\psi)=a\sin\psi+b\cos\psi+c$ 的形式，由此得到 $\psi$ 的可行区间，用于位置层的冗余分解和避关节极限。Cohn et al.（ICRA 2024）提到 iiwa、Panda、WAM 等常见 7-DOF 臂都有专门的几何解（如 Faria et al. 2018 针对 iiwa），并指出 iiwa 有 8 个离散的「全局构型参数」分支。

**在本机器人上的用法.** 腰只有 2 DOF（yaw、pitch），可以直接网格或随机采样 $(q_{yaw}, q_{pitch})$。对每个采样点，把目标位姿转到肩基座坐标系，再对 $\psi$ 在可行区间内扫描，得到全部解析解：

$$
\{q\} = \bigcup_{(q_w)\in\mathcal{S}_w}\ \bigcup_{\text{branch}}\ \bigcup_{\psi\in\Psi_{feas}(q_w)} \big(q_w,\ \mathrm{IK}_{arm}(T_{shoulder}^{-1}(q_w)\,T^*,\psi)\big)
$$

| 优点 | 缺点 |
|---|---|
| 极快、确定、能枚举所有分支，**是最好的全局种子来源** | **只适用于真正 S-R-S（或有现成几何解）的臂**；有肩/肘/腕偏置的臂没有简洁闭式解（IKFast 对 7-DOF 需要指定 free joint 离散化） |
| 关节极限可以解析地反映到 $\psi$ 区间上 | 不直接处理力矩、碰撞、闭链约束，只能事后筛选 |
| 平滑、可微：Cohn et al. 用它参数化双臂闭链流形 | 腰采样 × 分支 × $\psi$ 扫描，组合数会增长，需要剪枝 |

> 结论：**如果你的臂是 S-R-S 型，强烈建议把它作为 L1 种子生成器**，不作为最终求解器。如果不是，用数值单臂 IK（TRAC-IK 或自写 LM）多起点替代。

### 1.2 数值雅可比方法：DLS、加权伪逆、零空间投影、任务优先级

**阻尼最小二乘（DLS）**，综述见 Chiaverini, Siciliano, Egeland（IEEE TCST 1994）：

$$
\Delta q = J^\top\big(JJ^\top+\lambda^2 I\big)^{-1} e
$$

**Levenberg–Marquardt，阻尼取误差平方范数加小偏置**（Sugihara，IEEE T-RO 2011，"Solvability-Unconcerned IK"）。对不可解、奇异、冗余情况都保持数值稳定，而且不需要计算最小奇异值：

$$
\Delta q = \big(J^\top W_e J + W_n\big)^{-1}J^\top W_e e,\qquad W_n = \tfrac12 e^\top W_e e\, I + \bar W_n
$$

**加权伪逆 + 零空间次任务**，关节空间度量为 $W$：

$$
J_W^{+}=W^{-1}J^\top\big(JW^{-1}J^\top+\lambda^2 I\big)^{-1},\qquad
\Delta q = J_W^{+}e + \big(I-J_W^{+}J\big)\Delta q_0
$$

**两级任务优先级**（Siciliano & Slotine 1991 的经典框架）：

$$
\Delta q = J_1^{+}e_1 + \big(J_2N_1\big)^{+}\big(e_2-J_2J_1^{+}e_1\big),\qquad N_1=I-J_1^{+}J_1
$$

| 优点 | 缺点 |
|---|---|
| 实现简单、每步很便宜；$W$ 天然表达「腰部动得少」（§2.1） | **不等式约束（关节极限、碰撞、力矩）不能直接表达**，只能截断或加惩罚。SNS 类方法（如 arXiv 2204.03974 的 eSNS）可扩展到不等式层级，但实现复杂 |
| LM 对目标不可达也稳定，返回最接近的解 | 局部收敛，需要多起点 |

> 结论：适合作为 L2 的轻量备选，或作为 NLP 前的快速预收敛。约束多时，QP 形式更干净。

### 1.3 QP 微分 IK（加权 / 不等式 / 分层）

**原理.** 每次迭代在当前 $q$ 处线性化，求解 $\Delta q$：

$$
\begin{aligned}
\min_{\Delta q}\ & \sum_k w_k\big\|J_k\Delta q - \alpha e_k\big\|^2 + \Delta q^\top W\Delta q \\
\text{s.t.}\ & q_{min}\ominus q \le \Delta q \le q_{max}\ominus q \quad(\text{关节极限})\\
& -\,n_{ij}^\top J_{ij}\Delta q \le \xi\,(d_{ij}-d_{min}) \quad(\text{碰撞对线性化})\\
& \underline\tau \le \tau(q) + \tfrac{\partial\tau}{\partial q}\Delta q \le \overline\tau \quad(\text{力矩线性化})\\
& J_{rel}\Delta q = -\alpha\,e_{rel}\quad(\text{双手闭链，硬约束})\\
& \Delta q_i = 0,\ i\notin\mathcal{A}_{mode}\quad(\text{模式掩码})
\end{aligned}
$$

迭代 $q\leftarrow q\oplus\Delta q$ 直到收敛。这正是 pink 和 mink 的 `build_ik` 所构造的标准形式：$\min \tfrac12\Delta q^\top H\Delta q + c^\top\Delta q,\ G\Delta q\le h,\ A\Delta q=b$。mink 的碰撞限幅就是上面那行，$\xi$ 为 gain，$d_{min}$ 为最小距离。

**分层 QP（HQP）.** Escande, Mansard, Wieber（IJRR 2014）给出严格优先级的最小二乘 QP 求解器，任意层级都可以有不等式；在只有等式的层级问题上，比迭代投影类分层求解器快约 10 倍。常见工程做法是用加权单层 QP 近似：placo 的 `hard` 加 `soft` 再加权重，就是「硬约束 + 加权软任务」的两级结构。

| 优点 | 缺点 |
|---|---|
| 关节极限、碰撞、闭链等式、模式掩码、**线性化力矩**都能统一在一个凸 QP 中 | 局部、一阶。力矩是 $q$ 的非线性函数，线性化只在小步长内可靠 |
| 每步一个小规模稠密 QP（16 维），可用 ProxQP、OSQP、DAQP 等求解器 | 用于「求一个目标构型」时要迭代很多步，会卡在极限和局部极小（pink 文档明确说明微分 IK 是局部算法） |
| 现成库：placo（C++，有 hard/soft、`RelativeFrameTask`、`ManipulabilityTask`、自碰撞约束、`mask_dof`）、pink（barriers：`SelfCollisionBarrier`、`BodySphericalBarrier`）、mink（`CollisionAvoidanceLimit`、`DofFreezingTask`、闭链等式） | 现成库的运动学 IK **都没有内置「静力矩极限」约束**（placo 的力矩极限在其 dynamics/ID 求解器里），需要自己加线性不等式 |

### 1.4 非线性优化 IK（SQP / IPOPT / SNOPT）

**原理.** 直接在位置层求解：

$$
\begin{aligned}
\min_{q}\ & \|q_w-q_w^{0}\|^2_{W_w} + c_\tau\sum_i\Big(\tfrac{\tau_i(q)}{\tau_i^{max}}\Big)^2 + c_m\,\phi_{manip}(q) + c_s\|q-q_{seed}\|^2_{W}\\
\text{s.t.}\ & \log\!\big(T_{ee}^{*-1}T_{ee}(q)\big)^\vee \in [-\epsilon,\epsilon]\ \ (\text{或 TSR 区间})\\
& q_{min}\le q\le q_{max},\quad |q_w-q_w^{0}|\le \Delta_w\\
& -\tau^{max}\le g(q)-J(q)^\top f \le \tau^{max}\\
& d_{ij}(q)\ge d_{min}\ \ \forall (i,j)\in\mathcal{P}_{coll}
\end{aligned}
$$

现成实现：
- **Drake `InverseKinematics`**：在整个 `MultibodyPlant` 上求解；支持位置/姿态约束（可以相对另一 frame 表达）、`AddMinimumDistanceLowerBoundConstraint`、`Joint::Lock`（锁定子集），可添加自定义约束。`MultibodyPlant::CalcGravityGeneralizedForces` 提供 $\tau_g(q)$。
- **Pinocchio + CasADi（`pinocchio.casadi`）+ IPOPT**：Gepetto 的 jnrh2023 教程 `invgeom_cpin.py` 演示了用 `cpin.framesForwardKinematics` 和 `cpin.log6` 构造符号误差，再用 IPOPT 求解。Unitree xr_teleoperate 的双臂 IK 也是 Pinocchio + CasADi Opti + IPOPT，代价是 `50·平移 + 旋转 + 0.02·正则 + 0.1·平滑`，开启 `warm_start_init_point`、`max_iter=30`，并用 `buildReducedModel` 锁定非手臂关节。
- **TRAC-IK**：KDL 牛顿法（带随机跳出）与 SQP 并行运行，较好地处理关节极限；但**只支持单链**、不检查自碰撞、不支持力矩。只适合当 M1 的种子或单臂 IK。

| 优点 | 缺点 |
|---|---|
| **力矩、碰撞距离、位姿容差、腰部位移上界都可以精确写成约束** | 慢于 QP，对初值敏感（Cohn et al. 指出非凸等式约束只有在好初值下才收敛） |
| 精度高，适合 TAMP 的最终目标构型 | 碰撞距离的梯度在接触切换时不光滑，需要 margin 和 activation distance |

### 1.5 GPU 批量 / 采样式全局 IK

- **cuRobo**：多种子并行优化，论文报告在 RTX 4090 上 37000 IK/s、7600 无碰撞 IK/s。用球体近似做自碰撞和环境碰撞，`link_names` 可指定多个末端。**没有力矩约束。**
- **PyRoki**（JAX，Levenberg–Marquardt）：代价可组合（位姿、关节极限、可操作度、自碰撞、世界碰撞），可在 CPU/GPU/TPU 上运行。论文中 IK-Beam 在 Panda 基准上比 cuRobo 快 1.4–1.7 倍。约束只能作为可微惩罚，不处理硬约束。

| 优点 | 缺点 |
|---|---|
| 大批量多样解，非常适合 **TAMP 的抓取候选筛选和可达性分析** | 需要 GPU（PyRoki 也可在 CPU 运行）；力矩和闭链约束都需要自行扩展或事后筛选 |

### 1.6 学习型种子

- **IKFlow**（RA-L 2022）：条件归一化流，毫秒级生成大量多样近似解。
- **IKDiffuser**（arXiv 2506.13087）：扩散模型，支持运动学树，支持引导采样（如可操作度）。论文中「Dual RM76 and Waist（17 DOF）」平台单独使用时位置误差约 4–5 mm；作为种子交给 cuRobo 和 Pink 时成功率提升见 §0。
- **结论**：只做种子，必须经过数值精修。需要为本机器人采集数据并训练。适合在后期 L1 种子不够用时引入。

---

## 2. 让腰部「尽量少动」的方法与取舍

| 方法 | 数学形式 | 取舍 |
|---|---|---|
| **加权关节度量** | 加权伪逆中的 $W=\mathrm{diag}(w_w I_2,\ w_a I_{14})$，$w_w\gg w_a$；或 QP 正则 $\Delta q^\top W\Delta q$ | 简单、连续。但手臂到达极限时腰仍会被「推」着动，权重只能**降低**腰部参与，不能禁止 |
| **向名义姿态正则** | 代价 $\|q_w-q_w^{nom}\|^2_{W_w}$（NLP），或 QP 中的 posture task | 解与初值无关、可重复，适合 TAMP。$q_w^{nom}$ 可取当前腰角（「不动」）或中立位 |
| **严格优先级** | 末端任务 > 腰部正则，HQP 或 $N_1$ 投影 | 末端误差优先保证；腰部只在零空间内回中。实现比加权复杂 |
| **两阶段（先锁腰，再放开）** | 先 M1；失败、或 $\max_i|\tau_i|/\tau_i^{max}>\rho$、或可操作度过低时，才进入 M2/M3 | **最符合「能不动就不动」的语义**，可解释。代价是多一次求解，阈值需要调 |
| **腰部位移上界** | $|q_w-q_w^{cur}|\le\Delta_w$（盒约束），逐级放宽 $\Delta_w\in\{0, 5°, 15°, \text{全范围}\}$ | 硬保证腰部位移，便于安全论证。太紧会导致不可行，需要逐级放宽 |
| **L1 稀疏惩罚** | $c\,\|q_w-q_w^{cur}\|_1$（QP/NLP 中引入辅助变量） | 倾向「要么不动，要么只动必要的那个轴」（yaw 或 pitch），比二次惩罚更像「尽量不动」。会增加变量 |

**推荐组合**：两阶段，加上逐级放宽的腰部位移上界，再加二次和（可选）L1 腰部代价。放开腰的触发条件有三个：(a) 腰锁定时不可行；(b) 力矩利用率超过阈值；(c) 可操作度低于阈值，或离关节极限太近。

**注意：腰部帮手臂时可能伤到腰部自己。** 腰 pitch 承受整个上身、双臂和负载的重力矩。前倾虽然缩短了手臂力臂、降低了臂力矩，但会**增大腰 pitch 力矩**。所以力矩约束和代价必须**包括腰部关节**（§3）。

---

## 3. 负载与物理可行性：静力矩

### 3.1 静力矩模型

准静态（TAMP 目标构型、慢速搬运）时，关节力矩为：

$$
\tau(q) = g(q) - \sum_{h\in\{L,R\}} J_h(q)^\top f_h^{ext}
$$

其中 $f_h^{ext}$ 是手 $h$ 受到的外部 wrench（负载重力向下，按约定取符号）。Pinocchio 的 `computeStaticTorque(model, data, q, fext)` 直接返回 $g(q)-J^\top f_{ext}$（`fext` 为每个关节局部坐标系中的力），`computeStaticTorqueDerivatives` 给出 $\partial\tau/\partial q$，可直接用于 QP 线性化和 NLP 梯度。`computeGeneralizedGravity` 等价于 `rnea(q,0,0)`。Drake 对应的是 `CalcGravityGeneralizedForces`。

**负载建模的两种方式：**
1. **把负载并入末端连杆惯性**（质量 $m$、质心偏移 $r$），这样 $g(q)$ 里自动包含负载。适合单手持物。
2. **作为外部 wrench**：$f = [\,m\mathbf{g};\ r\times m\mathbf{g}\,]$，作用在手坐标系。适合双手分担，因为分配比例本身可能是变量。

**力矩约束**（加安全裕度 $\rho<1$）：

$$
-\rho\,\tau^{max}\le g(q)-J(q)^\top f\le \rho\,\tau^{max}
$$

文献支持：Load-Capacity-Constrained Arm-Angle Planning（PMC13511702）从重力矩、已知端点负载的 $J^\top$ 映射和单个执行器力矩极限推导**静态负载能力约束**，并把它**投影到臂角 $\psi$ 的一维域**，与关节范围、奇异约束求交集来选择 $\psi$。这与 §1.1 的「解析臂角扫描 + 筛选」完全契合，作者也明确说明该方法只针对慢速静态条件。UTS 的 *A Control Method for Joint Torque Minimization of Redundant Manipulators Handling Large External Forces* 指出：只考虑内部动力学的传统力矩最小化方法，在末端有大外力时可能选出放大力矩的构型；应把外力纳入零空间力矩最小化。

### 3.2 力矩相关的代价（选构型）

| 代价 | 形式 | 说明 |
|---|---|---|
| 力矩利用率 | $\sum_i(\tau_i/\tau_i^{max})^2$ 或 $\max_i|\tau_i|/\tau_i^{max}$（epigraph 变量） | 最直接。按 $\tau^{max}$ 归一化，避免大关节主导 |
| 力可操作度（force manipulability） | 由 $\tau=J^\top f$ 得 $f$ 的椭球 $f^\top JJ^\top f\le1$，其体积与 $1/\sqrt{\det(JJ^\top)}$ 相关；看负载方向 $u$ 的力传递比 $\big(u^\top (JJ^\top)u\big)^{-1/2}$ | 方向性：只关心重力方向的承载能力。基础是 Yoshikawa 1985 的可操作度理论 |
| 动态可操作度 | Yoshikawa 1985（J. Robotic Systems 2:113–118）：基于关节驱动力与末端加速度的动态可操作度椭球 | 搬运需要加减速时更合适；需要惯量矩阵 $M(q)$（Pinocchio CRBA） |
| 远离奇异 | $-\log\sqrt{\det(J_hJ_h^\top)}$ 或最小奇异值下界 | 运动学可操作度（Yoshikawa 1985）；TRAC-IK 的 Manip1–3、placo 的 `ManipulabilityTask`、PyRoki 的 manipulability cost 都基于此 |

### 3.3 双手共持一个物体：负载分担

物体受力平衡（准静态）：

$$
G\begin{bmatrix}f_L\\ f_R\end{bmatrix} = -w_{obj}^{g},\qquad G=\big[\,\mathrm{Ad}^\top_{T_{oL}}\ \ \mathrm{Ad}^\top_{T_{oR}}\,\big]
$$

解为 $f = f_{part} + N_G\,\eta$，其中 $N_G\eta$ 是**内力**（挤压/拉扯，不改变物体平衡）。设计建议：
- **给定 $q$ 时，$\tau$ 关于 $f$ 是线性的。** 可以把 $f_L, f_R$ 作为 NLP 的附加变量，与 $q$ 一起优化（12 个变量），约束包括平衡方程、力矩极限、（可选）接触或摩擦锥。也可以先取最小范数分配 $f=G^{+}(-w)$，再检查力矩。
- 内力通常加小代价 $\|\eta\|^2$，避免夹坏物体。
- 闭链运动学约束 $T_L^{-1}T_R = T_{rel}^*$（6 维等式）在 QP 层作为硬等式（mink 支持闭链等式，placo 有 `RelativeFrameTask`，可配置为 hard），在 NLP 层作为等式约束。

> 本节的平衡和内力分解是标准的双臂协作力学推导，没有对应某一篇检索到的论文。

### 3.4 腰部对力矩的影响（定性）

- 腰 yaw 通常绕竖直轴，重力矩很小（若轴线竖直），主要用来**转向扩展可达范围**，代价低。
- 腰 pitch 承受全上身的重力矩：前倾能缩短手臂力臂，但会增大 pitch 力矩。所以「腰帮手臂省力」必须由**含腰部关节的力矩约束和代价**自动权衡，不能靠手调规则。
- 结论：力矩约束覆盖**全部 16 个关节**，与模式无关。锁定关节也要检查，腰锁定时它同样受力，除非有抱闸并另行设计。

---

## 4. 统一三模式的设计

### 4.1 单一模型 + 关节掩码

| 做法 | 工具 | 说明 |
|---|---|---|
| **主方案：完整 16-DOF 模型 + 活动关节集 $\mathcal{A}$** | 自写 QP/NLP：锁定关节的变量上下界设为当前值，或从决策变量中删除 | 一个模型、一套代价，**力矩和碰撞始终在完整状态上计算** |
| placo | `solver.mask_dof("waist_yaw")` / `unmask_dof(...)` | 运行时切换，不需要重建 |
| mink | `DofFreezingTask(model, dof_indices=[...])` 放进 `constraints` 是硬冻结，放进 tasks 则只是软偏好 | 运行时切换 |
| Drake | 在 `plant_context` 上 `Joint::Lock`，再用带 context 的 `InverseKinematics` 构造函数，被锁关节保持不变 | 官方文档推荐的「对子集求解」方式 |
| Pinocchio | `buildReducedModel(model, joints_to_lock, q_ref)` | 生成一个更小的新模型（可连带几何模型）。适合**预先**为三种模式各建一个模型（Unitree 就是这样锁非手臂关节）。注意：**降阶模型里被锁关节不再是变量，但它们的质量和几何仍在**（固定在 $q_{ref}$），被锁关节自己的力矩要用完整模型校验 |

模式定义：
- **M1**：$\mathcal{A}=$ 某一臂的 7 个关节；另一臂和腰锁定在当前值，但仍参与碰撞和力矩计算。
- **M2**：$\mathcal{A}=$ 腰 2 + 一臂 7；另一臂锁定。注意腰一动，另一只臂会跟着移动，**必须检查它的碰撞和力矩**；如果另一只手正在持物，还需要约束它的末端不动，这时实际上是 M3。
- **M3**：$\mathcal{A}=$ 全部 16 个关节；两个末端任务，持同一物体时加闭链等式。

### 4.2 多任务表述（同一套代价，按模式启用）

$$
\min_{q}\ \underbrace{\sum_{h\in H_{act}}\|e_h(q)\|^2_{W_h}}_{\text{末端（或 NLP 中的约束）}}
+\underbrace{\|q_w-q_w^{ref}\|^2_{W_w}+c_1\|q_w-q_w^{ref}\|_1}_{\text{腰尽量不动}}
+\underbrace{c_\tau\!\sum_i(\tau_i/\tau_i^{max})^2}_{\text{省力}}
+\underbrace{c_m\sum_h\!-\log\mu_h(q)}_{\text{远离奇异}}
+\underbrace{c_s\|q-q_{cur}\|^2_{W}}_{\text{靠近当前}}
$$

- M1/M2：$H_{act}$ 只有一只手。M3：两只手，持物时加 $e_{rel}=0$ 硬约束。
- 「另一只手正在持物但本次不是规划目标」时，把它的当前位姿作为**保持任务**（硬约束），防止腰运动带偏它。

### 4.3 权重设计建议（起点值，需要在实机上调）

| 项 | 建议 | 理由 |
|---|---|---|
| 末端位姿 | NLP 中作为**约束**（容差如 1 mm / 0.5°）；QP 中权重最高（位置约 1，姿态约 0.5–1，单位归一化后） | TAMP 目标必须精确；QP 只负责把解拉近 |
| 腰部 $W_w$ | 手臂关节权重的 10–100 倍，另加逐级放宽的盒约束 | 「尽量少动」，但不至于不可行时也锁死 |
| 力矩利用率 $c_\tau$ | 中等；硬约束放在 $\rho\approx0.8$ 之类的裕度上 | 约束保安全，代价在可行解中挑省力的 |
| 可操作度 $c_m$ | 小 | 只用来打破冗余平局、远离奇异 |
| 靠近当前/种子 $c_s$ | 小 | 让结果可重复，减少无谓的大幅运动 |

> 这些数值是工程起点，不出自论文。可参考的实际配置：Unitree xr_teleoperate 使用 `50·平移 + 1·旋转（另一处为 0.5）+ 0.02·正则 + 0.1·平滑`，说明在 IPOPT 中「位姿项权重远大于正则项」是常见做法。

---

## 5. 全局 vs 局部：多起点与多解选择（TAMP 关键）

**为什么需要**：TAMP 中 IK 是目标采样器。下游运动规划需要**多个不同的**可行构型（不同肘位、腰位、分支）来规避障碍和闭链不可达。只返回一个局部解会导致整个任务规划回溯失败。

**种子来源**（L1，按成本从低到高）：
1. 当前构型、上一次的解（warm start）
2. 名义姿态（腰中立、双臂标准姿态）
3. **解析臂角扫描**（S-R-S 臂）：腰角网格（比如 yaw 和 pitch 各取若干值，从「当前值」开始逐级向外）× 分支 × $\psi$ 可行区间采样；或用 TRAC-IK 的 `Distance` / `Manip1-3` 模式做单臂种子
4. 解缓存（同类抓取任务）
5. 关节极限内的随机重启
6. （可选）cuRobo 或 PyRoki 的 GPU 批量结果；后期可加 IKDiffuser 学习种子

**选择与排序**（L4），对所有通过校验的解计算：

$$
\text{score}(q)= a_1\|q_w-q_w^{cur}\|_{W_w} + a_2\max_i\frac{|\tau_i|}{\tau_i^{max}} + a_3\frac{1}{\min_h\mu_h(q)} + a_4\|q-q_{cur}\|_W + a_5\,\frac{1}{\min d_{ij}(q)}
$$

先按「是否需要动腰」分层（腰位移为 0 的解优先），层内再按分数排序；对解做**多样性去重**（关节空间距离阈值，类似 MTC `ComputeIK` 的最小解间距），返回 Top-K。

---

## 6. 推荐架构（具体到库）

### 6.1 组件选择

| 层 | 推荐 | 备选 | 理由 |
|---|---|---|---|
| 模型、FK、雅可比、重力与静力矩及其导数、碰撞距离 | **Pinocchio**（BSD-2-Clause）+ **Coal**（原 hpp-fcl；`computeDistances`） | Drake MultibodyPlant | Pinocchio 有 `computeStaticTorque(+Derivatives)`、`buildReducedModel`、CasADi 后端，正好覆盖力矩约束需求 |
| L2 QP 微分 IK | **placo**（C++/Python，MIT；`mask_dof`、hard/soft、`RelativeFrameTask`、`ManipulabilityTask`、自碰撞约束）**加自定义线性力矩约束** | pink（Apache-2.0，Python，barriers）；mink（Apache-2.0，MuJoCo，`DofFreezingTask`、碰撞、闭链等式）；或自写 QP + **ProxQP**（BSD-2，RSS 2022）/ OSQP（Apache-2.0） | placo 底层就是 Pinocchio（同一模型）、C++ 性能好、模式切换自然。如果需要更灵活地加力矩约束，自写一个 16 维 QP 并不复杂 |
| L3 NLP 精修 | **Pinocchio-CasADi + IPOPT**（CasADi LGPL-3.0，IPOPT EPL-2.0） | **Drake `InverseKinematics`**（BSD-3；SNOPT 或 IPOPT；自带最小距离约束、`Joint::Lock`） | CasADi 自动微分可以直接写 $g(q)-J^\top f$ 和闭链约束；Unitree 已把同样的 Pinocchio + CasADi + IPOPT 组合用在双臂 IK 上。Drake 更成熟、碰撞约束是现成的，但力矩约束需要自写 |
| L1 种子 | 解析臂角 IK（若为 S-R-S）/ TRAC-IK（BSD-3，单链） | cuRobo（Apache-2.0）/ PyRoki（MIT）批量；IKDiffuser | 种子质量决定全局成功率 |
| L4 校验 | Pinocchio + Coal 全状态碰撞；完整模型静力矩；可操作度 | 与运动规划器共用的碰撞模型（如 MoveIt 或 cuRobo 的球体模型） | **必须在完整 16-DOF 状态上**做碰撞和力矩检查，与求解时的模式无关 |

### 6.2 求解流程（伪流程，不含实现）

1. **输入**：目标（每只手的位姿或 TSR）、模式（M1/M2/M3 或 auto）、负载（质量、质心、持握方式：单手 / 双手）、当前构型、超时、K。
2. **auto 模式的腰部策略**：
   - 第一轮用 M1（或「腰锁定的 M3」，即双臂各自求解、腰不动）。
   - 若失败，或最优解的力矩利用率 > $\rho_1$，或可操作度 < $\mu_{min}$，则放开腰部：腰位移上界从 $\Delta_w^{(1)}$ 开始，失败就放宽到 $\Delta_w^{(2)}$，最后到全范围。
3. **对每个种子**：
   - (a) L2：QP 微分 IK 迭代若干步（加关节极限、自碰撞、线性化力矩、闭链等式、掩码）。
   - (b) L3：IPOPT 精修（精确位姿容差、力矩、最小距离、腰位移上界），warm start 取 (a) 的结果。
   - (c) L4：校验。
4. 汇总、去重、排序，返回 Top-K 和诊断信息（哪个约束起作用、力矩利用率、腰位移）。

### 6.3 模式切换

- 同一 `Model`、同一 `GeometryModel`、同一套代价函数；模式只决定：(i) 活动关节集 $\mathcal{A}$；(ii) 启用哪些末端任务和保持任务；(iii) 是否启用闭链约束与负载分担变量。
- placo 用 `mask_dof` 和 `unmask_dof`；自写 NLP 用变量上下界锁定；也可以为三种模式预建 `buildReducedModel` 以减小 NLP 维度，但力矩和碰撞校验仍用完整模型。

### 6.4 性能预期（只给有出处的参照）

- **力矩约束的 NLP 每次求解要比纯运动学 IK 慢**，因为要计算 $g(q)$、$J^\top f$ 及其导数。具体耗时**需要在你的机器人上实测**，本文不编造数字。
- 可参考：cuRobo 的批量 IK 吞吐（37000/s，无碰撞版 7600/s，RTX 4090）；PyRoki 的 IK-Beam 在 Panda 上比 cuRobo 快 1.4–1.7×；Unitree 的 IPOPT 设置（`max_iter=30` 加 warm start）用于实时遥操作，说明 16 维以内的 Pinocchio + CasADi + IPOPT 在 warm start 下足够快。
- 对 TAMP 来说，瓶颈通常在**多起点数量 × 每次 NLP**。建议并行化（多进程或线程池），或先用 GPU 批量粗筛再精修。

### 6.5 验证与测试建议

1. **单元测试**：随机采样可行 $q$ → FK 得到目标 → 求解 → 比较可达率和误差（与 IKDiffuser 等论文的评测方法一致）。
2. **力矩**：在完整模型上用 `computeStaticTorque` 逐关节检查利用率；负载取最大设计值加裕度。
3. **碰撞**：完整 16-DOF 状态，用 Coal 的 `computeDistances`。凹网格要做凸分解或用凸包（Pinocchio issue 中说明有符号距离需要至少一个形状为凸）。
4. **腰部行为统计**：腰位移分布、动腰的比例、动腰后的力矩改善量。
5. **闭链**：沿解（以及下游轨迹）验证 $T_L^{-1}T_R$ 的残差。

---

## 7. 库对比表

| 库 | 语言 / 许可证 | 求解类型 | 多末端 | 碰撞 | 力矩约束 | 关节锁定 / 模式 | 闭链 / 相对位姿 | 速度特点 |
|---|---|---|---|---|---|---|---|---|
| **Pinocchio** | C++/Python，BSD-2-Clause | 基础库（FK、雅可比、RNEA、`computeStaticTorque` 及导数）；CLIK 示例 | ✔（任意 frame） | Coal 距离 | 提供 $\tau$ 与导数（需自己建约束） | `buildReducedModel` | 自己构造 | 很快（C++）；有 CasADi 自动微分后端 |
| **placo** | C++/Python，MIT | QP 微分 IK + ID | ✔ | ✔ 自碰撞约束 | 运动学 IK 无；**dynamics 求解器有力矩极限** | ✔ `mask_dof` | ✔ `RelativeFrameTask`（hard/soft） | C++ QP（eiquadprog） |
| **pink** | Python，Apache-2.0 | QP 微分 IK | ✔ | ✔ barriers（`SelfCollisionBarrier` 等） | 无 | 通过任务和限制实现 | 自定义任务 | Python，单步较快，需迭代 |
| **mink** | Python（MuJoCo），Apache-2.0 | QP 微分 IK | ✔ | ✔ `CollisionAvoidanceLimit`（任意 geom 对） | 无 | ✔ `DofFreezingTask` | ✔ 闭链等式约束 | Python，单步较快，需迭代 |
| **Drake** | C++/Python，BSD-3-Clause | NLP IK（SNOPT/IPOPT 等）、GlobalIK（MIP）、轨迹优化 | ✔ | ✔ `MinimumDistance` 约束 | 可自定义约束；有 `CalcGravityGeneralizedForces` | ✔ `Joint::Lock` | ✔ 帧间位置/姿态约束 | NLP，速度中等，精度高 |
| **CasADi + IPOPT** | C++/Python，LGPL-3.0 / EPL-2.0 | 通用 NLP | 自定义 | 自定义（距离需外部计算或用解析近似） | ✔（与 `pinocchio.casadi` 配合可精确写出） | 变量上下界 | 自定义 | 依赖问题规模和 warm start |
| **TRAC-IK** | C++，BSD-3-Clause | KDL-RR + SQP 并行 | ✘ 单链 | ✘ | ✘ | 选链即选关节 | ✘ | 快；只适合单臂种子 |
| **cuRobo** | Python/CUDA，Apache-2.0 | GPU 多种子优化 IK、运动生成 | ✔ `link_names` | ✔ 球体近似 | ✘ | 锁关节配置 | 通过多末端位姿 | 37000 IK/s（论文，RTX 4090） |
| **PyRoki** | Python（JAX），MIT | LM 非线性最小二乘（可批量） | ✔ | ✔ 可微惩罚 | 自定义代价（软） | 变量选择 | 自定义代价 | CPU/GPU/TPU；IK-Beam 比 cuRobo 快 1.4–1.7×（Panda） |
| **ProxQP**（proxsuite） | C++/Python，BSD-2-Clause | QP 求解器 | – | – | – | – | – | 面向机器人的稠密/稀疏 QP（RSS 2022） |
| **OSQP** | C，Apache-2.0 | QP 求解器（ADMM） | – | – | – | – | – | 稀疏问题稳健 |
| **TSID** | C++，BSD-2-Clause | 基于 Pinocchio 的任务空间逆动力学 | ✔ | – | ✔（ID 层面） | – | – | 若后续需要力控 / 动力学层可用 |
| bio_ik / pick_ik | C++（MoveIt 插件），BSD-3-Clause | 进化 + 梯度 | ✔ 多 tip | ✘（MoveIt 外部检查） | ✘ | 规划组 | ✘ | 适合 MoveIt 集成，不适合力矩约束 |
| IKFlow / IKDiffuser | Python | 学习型种子 | IKDiffuser ✔（运动学树） | IKDiffuser 隐式降低碰撞率 | ✘ | 需重新训练 | ✘ | 毫秒级批量，精度为毫米级 |

---

## 8. 参考链接（均已核实）

**解析与数值 IK 理论**
- Shimizu et al. 2008，7-DOF 解析 IK 与关节极限（T-RO 24(5):1131–1142）：https://ieeexplore.ieee.org/abstract/document/4631505 （ACM 页：https://dl.acm.org/doi/10.1109/TRO.2008.2003266 ）
- Chiaverini, Siciliano, Egeland 1994，DLS 综述：https://stephanniec.github.io/stepholio/files/leastsqrinvkin.pdf
- Siciliano & Slotine 1991，多任务优先级：https://ui.adsabs.harvard.edu/abs/1991icar.conf...44S/abstract
- Sugihara 2011，LM 可解性无关 IK：https://ieeexplore.ieee.org/abstract/document/5784347/
- Escande, Mansard, Wieber 2014，HQP（IJRR 33(7)）：https://journals.sagepub.com/doi/10.1177/0278364914521306 ；PDF：https://gepettoweb.laas.fr/uploads/Publications/2014_escande_ijrr.pdf
- 分层冗余框架 eSNS：https://export.arxiv.org/pdf/2204.03974v4.pdf
- Yoshikawa 1985，动态可操作度：https://ui.adsabs.harvard.edu/abs/1985JRoS....2..113Y/abstract ；可操作度：https://journals.sagepub.com/doi/10.1177/027836498500400201

**负载与力矩**
- 负载能力约束的臂角规划：https://pmc.ncbi.nlm.nih.gov/articles/PMC13511702/
- 大外力下冗余臂力矩最小化：https://opus.lib.uts.edu.au/rest/bitstreams/13a29c88-9cab-4627-ac56-2fe0fa32c78c/retrieve
- Pinocchio `computeGeneralizedGravity`：https://docs.ros.org/en/jazzy/p/pinocchio/generated/function_namespacepinocchio_1a6cd0f621e83bfa0c74f89aa63682c866.html ；`rnea-derivatives.hpp`（含 `computeStaticTorqueDerivatives`）：https://github.com/stack-of-tasks/pinocchio/blob/eecaea8e/include/pinocchio/algorithm/rnea-derivatives.hpp
- Drake `MultibodyPlant`（`CalcGravityGeneralizedForces`）：https://drake.mit.edu/doxygen_cxx/classdrake_1_1multibody_1_1_multibody_plant.html

**库**
- Pinocchio：https://github.com/stack-of-tasks/pinocchio ；`buildReducedModel` 示例：https://docs.ros.org/en/latest-lts/api/pinocchio/html/build-reduced-model_8py_source.html ；Pinocchio + CasADi IK 示例：https://github.com/Gepetto/jnrh2023/blob/main/invgeom_cpin.py
- Coal（hpp-fcl）：https://github.com/coal-library/coal
- placo：https://github.com/rhoban/placo ；关节掩码：https://placo.readthedocs.io/en/latest/kinematics/joints_mask.html
- pink：https://github.com/stephane-caron/pink ；barriers：https://stephane-caron.github.io/pink/barriers.html
- mink：https://github.com/kevinzakka/mink ；limits：https://kevinzakka.github.io/mink/api/limits.html ；tasks（`DofFreezingTask`）：https://kevinzakka.github.io/mink/api/tasks.html
- Drake IK：https://drake.mit.edu/doxygen_cxx/classdrake_1_1multibody_1_1_inverse_kinematics.html
- ProxQP：https://github.com/Simple-Robotics/proxsuite ；论文：https://roboticsproceedings.org/rss18/p040.pdf
- TRAC-IK：https://github.com/traclabs/trac_ik
- cuRobo：https://arxiv.org/abs/2310.17274 ；PyRoki：https://arxiv.org/abs/2505.03728
- IKDiffuser：https://arxiv.org/abs/2506.13087 ；IKFlow：https://arxiv.org/abs/2111.08933
- Unitree xr_teleoperate 双臂 IK（Pinocchio + CasADi + IPOPT 的实际参考）：https://github.com/unitreerobotics/xr_teleoperate/blob/main/teleop/robot_control/robot_arm_ik.py
- Cohn et al.（7-DOF 解析 IK 与双臂闭链）：https://arxiv.org/abs/2309.08770
