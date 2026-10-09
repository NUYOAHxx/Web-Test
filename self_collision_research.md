# 自碰撞约束在 IK 中的实现：原理、计算开销、MoveIt 能否用、替代方案与推荐

> 日期：2026-10-10 ｜ 对象：2-DOF 腰（yaw + pitch）+ 双 7-DOF SRS 臂，用于 TAMP 的 IK
> 已定架构：Pinocchio 模型 → placo QP 微分 IK → CasADi + IPOPT NLP 精修 → 全状态校验
> 说明：库的行为都来自本次读过的官方文档或源码（placo、pink、mink、Drake、MoveIt 2、cuRobo），数字只引用原文。没有出处的估算会明确标成「估算」。

---

## 0. 先回答四个问题

| 问题 | 简短回答 |
|---|---|
| 自碰撞约束在 IK 里怎么实现？ | 对每一对**可能相撞的连杆**，计算最近距离 $d$、两个最近点（witness points）和法向 $n$。由此得到距离对关节的导数（距离雅可比）$J_d=n^\top(J_{p_B}-J_{p_A})$。QP 层把它写成线性不等式 $d+J_d\Delta q\ge d_{safe}$（或其阻尼版本），NLP 层写成 $d(q)\ge d_{safe}$ 并提供梯度。placo、pink、mink、Drake 都是这个思路。 |
| 计算量大吗？ | 取决于**碰撞对的数量 × 每对的几何复杂度**。单对凸体的距离查询是微秒级：Coal/HPP-FCL 论文中，网格凸包在近距离情形下约 0.8–4.1 µs/对。真正贵的是：(1) 对太多、(2) 用原始网格（非凸）、(3) 每次迭代都重算。把碰撞对剪到几十对、用胶囊/球/凸包代替网格之后，开销通常比 QP 求解本身小，但**必须在你的模型上实测**。 |
| 能用 MoveIt 的碰撞检测吗？ | **用于最终校验和规划：能，而且推荐**（与规划器同一套模型和 ACM）。**放进 placo/CasADi 的迭代循环：不推荐**。MoveIt 的 FCL 后端能给距离、最近点和单位法向，但不给关节空间梯度，要自己用 `RobotState::getJacobian` 拼接；而且要经过 PlanningScene/RobotState 这一层，Bullet 后端甚至没实现距离查询。MoveIt 的 IK 插件（如 KDL）本身不处理碰撞，只是**解出来后用回调拒绝、再随机重启**。 |
| 其他选择？ | Pinocchio + Coal（你已在用 Pinocchio，最自然，placo 内部就是这个）；球/胶囊近似（cuRobo 的做法，解析距离、便宜、可微）；Drake `MinimumDistanceLowerBoundConstraint`；SDF/距离场；学习型自碰撞边界（Koptev et al. RA-L 2021）。 |

**一句话推荐**：用 MoveIt Setup Assistant 生成的 SRDF（禁用碰撞对表）作为唯一的碰撞对来源。三层分别用三种精度的几何：
- QP 层（placo）：胶囊/凸包 + 按模式筛选的少量碰撞对。
- NLP 层：解析胶囊/球距离（CasADi 可符号求导）或 Coal 回调，留安全裕度。
- 校验层：原始网格 + MoveIt/FCL 或 Coal 做全状态检查。

---

## 1. 自碰撞约束是怎么工作的

### 1.1 基本量：距离、最近点、法向

对一对几何体 $A$（挂在连杆 $a$ 上）和 $B$（挂在连杆 $b$ 上），在构型 $q$ 下：

$$
d(q)=\min_{x\in A(q),\,y\in B(q)}\|x-y\|,\qquad p_A,\,p_B=\text{最近点（witness points）},\qquad n=\frac{p_B-p_A}{\|p_B-p_A\|}
$$

穿透时（$d<0$，signed distance），由 EPA 等算法给出穿透深度和方向。距离计算库：Coal（原 HPP-FCL）、FCL、Bullet 都用 GJK 计算凸体距离。

### 1.2 距离雅可比：距离对关节的导数

把两个最近点看成**分别固定在各自连杆上的点**，它们的速度是 $\dot p_A=J_{p_A}(q)\dot q$ 和 $\dot p_B=J_{p_B}(q)\dot q$。这里 $J_{p}$ 是该点的 3×n 线速度雅可比，Pinocchio 中用 `LOCAL_WORLD_ALIGNED` 坐标系、以该点为参考点求 frame/joint 雅可比。于是：

$$
\dot d = n^\top(\dot p_B-\dot p_A)= \underbrace{n^\top\big(J_{p_B}(q)-J_{p_A}(q)\big)}_{J_d(q)\ (1\times n)}\ \dot q
$$

pink 的 `SelfCollisionBarrier` 写的是等价形式：

$$
J_i = n_1^\top J^1_p + (r_1\times n_1)^\top J^1_\omega + n_2^\top J^2_p + (r_2\times n_2)^\top J^2_\omega,\quad n_1=-n_2
$$

其中 $r$ 是关节原点到最近点的向量，也就是把关节处的雅可比平移到最近点上。

**直观理解**：$J_d$ 告诉你「每个关节动一点，这两个连杆之间的距离变化多少」。若某关节同时位于 $a$ 和 $b$ 的上游（比如对左右臂来说，腰就是共同上游），它对 $p_A$、$p_B$ 的贡献相同，在 $J_d$ 中**相互抵消**。所以**腰的运动不改变两臂之间的距离**，这是 §5 剪枝的依据。

### 1.3 QP 层：线性化不等式（velocity damper / barrier）

**(a) 直接线性化**（placo 的做法），要求一步之后距离不小于裕度：

$$
d + J_d\,\Delta q \ \ge\ d_{safe}\quad\Longleftrightarrow\quad -J_d\,\Delta q\le d-d_{safe}
$$

**(b) Velocity damper**（Faverjon & Tournassoud, ICRA 1987；Kanoun, Lamiraux, Wieber, T-RO 2011 将其纳入带不等式的分层 QP）。只在影响距离 $d_i$ 以内激活：

$$
\dot d \ \ge\ -\xi\,\frac{d-d_{s}}{d_i-d_{s}}\qquad (d<d_i)
$$

越靠近就允许越小的接近速度，到 $d_s$ 时接近速度为 0。Kanoun et al. 写成 $-\dot d\le-\lambda(d_{min}-d)$。mink 的 `CollisionAvoidanceLimit` 是同类形式：

$$
-\Delta d\le\begin{cases}\xi(d-d_{min})+\epsilon & d>d_{min}\\ \epsilon & \text{otherwise}\end{cases},\qquad \Delta d=n^\top J\Delta q
$$

只对检测距离 `collision_detection_distance` 以内的对生效，有可选的 `broadphase` 预筛选。

**(c) Control Barrier Function**（pink）：$\frac{\partial h_j}{\partial q}\dot q+\alpha_j(h_j(q))\ge 0$，其中 $h(q)=d(q)-d_{min}$。另外在代价中加入「safe displacement」项（引用 arXiv 2404.12329）。

三者本质相同：**一阶近似 + 让约束随距离逐步收紧**，区别只在增益怎么设。

### 1.4 NLP 层：$d(q)\ge d_{safe}$ 与不光滑问题

$$
\min_q f(q)\quad \text{s.t.}\quad d_k(q)\ge d_{safe}\ \ \forall k\in\mathcal{P},\qquad \nabla_q d_k = J_{d,k}(q)^\top
$$

IPOPT/SNOPT 需要约束值和梯度，上面的 $J_d$ 就是梯度。**难点是不光滑**：
1. **最近点会跳**：两个平面平行（盒对盒、网格面对面）时最近点不唯一，$d$ 在这些构型处不可微，梯度会突变。pink 的源码注释明确写着「对非光滑碰撞几何，行为未定义」，建议用光滑凸体。
2. **对很多对取 min 也不光滑**：Drake 的 `MinimumDistanceLowerBoundConstraint` 因此不逐对加约束，而是把所有对合成一个约束：

$$
\mathrm{SmoothOverMax}\Big(\varphi\big(\tfrac{d_i(q)-d_{inf}}{d_{inf}-lb}\big)/\varphi(-1)\Big)\le 1
$$

其中 $\varphi$ 默认为 QuadraticallySmoothedHinge。只有 $d_i<d_{inf}=lb+\text{offset}$ 的对参与计算。文档说 offset 越小越快（候选对越少），TRI 内部用过 $10^{-6}$ 这样的小值，效果良好；默认 0.01 m。
3. **研究方向**：Montaut et al.（ICRA 2023, arXiv 2209.09012）用随机平滑估计凸体（含网格）碰撞距离的导数，已实现于 HPP-FCL/Pinocchio。

**实践建议**：NLP 层用**球、胶囊这类形状**（距离处处有清晰的解析式），并给 $d_{safe}$ 多留几毫米裕度，抵消近似误差。

### 1.5 placo 的实现（读源码）

`src/placo/kinematics/avoid_self_collisions_constraint.cpp` 的逻辑：
1. 调用 `robot.distances()`，内部是 `pinocchio::computeDistances(model, data, collision_model, geom_data, q)`（Coal 后端），对**所有已启用的碰撞对**计算 `min_distance` 和两个 `nearest_points`。
2. 只对 `min_distance < self_collisions_trigger` 的对生成约束行。
3. $n=$ normalize$(p_B-p_A)$；若距离为负，把 $n$ 取反。
4. 用两个最近点处的 `LOCAL_WORLD_ALIGNED` 雅可比，得到行 $n^\top(J_b-J_a)$，右端为 `min_distance - self_collisions_margin`，即约束 $d+J_d\Delta q\ge \text{margin}$。
5. 可配置为 hard 或 soft（带权重）。

默认值：源码头文件中 `self_collisions_margin = 0.005`、`self_collisions_trigger = 0.01`（m）；文档示例用 0.01 / 0.05。

**placo 文档的三条警告**：
- 碰撞的连杆多时会显著增加计算时间，应使用干净的碰撞模型，尽量用纯几何体（pure shapes）。
- 碰撞形状不能互相重叠，否则要手工指定碰撞对。
- 求解器只看一个小步长，可能卡在局部最优，摆脱它是规划问题。

**碰撞对配置**：默认启用全部碰撞对（`addAllCollisionPairs`）。在 URDF 旁放一个 `collisions.json`（形如 `[["link_a","link_b"], ...]`）即可只启用列出的对，也可调用 `load_collision_pairs`。

**对开销的含义**：触发距离只减少 QP 的约束行数，**不减少距离计算量**，因为每次求解都会计算所有已启用对的距离。所以**碰撞对表要剪得够干净**。

### 1.6 其他库对照

| 库 | 自碰撞形式 | 关键点 |
|---|---|---|
| **pink** `SelfCollisionBarrier` | CBF：$h=d-d_{min}$，梯度见 §1.2 | 基于 Pinocchio + hpp-fcl/Coal；只取最近的 `n_collision_pairs` 对；要求光滑凸几何；示例 `yumi_self_collision.py`、`iiwa14_spheres_collision.srdf` |
| **mink** `CollisionAvoidanceLimit` | velocity damper 不等式 | MuJoCo geom 对；`minimum_distance_from_collisions`、`collision_detection_distance`、`gain`、`broadphase` |
| **Drake** `AddMinimumDistanceLowerBoundConstraint(bound, offset)` | 所有候选对合成一个光滑约束 | 用于 NLP IK；需要把 plant 连接到 SceneGraph；候选对排除父子连杆 |
| **cuRobo** | 机器人用球近似，对球与球计算距离 | `collision_spheres`、`self_collision_ignore`、`self_collision_buffer`；GPU 批量；代价有激活距离 $\eta$ |

---

## 2. 计算开销：由什么决定，怎么降

### 2.1 开销公式（概念上）

$$
T_{coll}\approx T_{FK}+\underbrace{N_{pairs}^{active}}_{\text{剪枝后对数}}\times \underbrace{\bar t_{pair}}_{\text{单对距离（取决于几何）}} + N_{rows}\times T_{J}
$$

单次 IK 的总开销还要乘以迭代次数，再乘以多起点数量。

### 2.2 有出处的数字

- **Coal/HPP-FCL（Montaut et al., RSS 2022，Table I）**：ShapeNet 网格（用其凸包），近距离或浅穿透（−0.1 m ≤ dist ≤ 0.1 m）情形下，单对**距离计算** GJK 约 **0.8–4.1 µs**，Nesterov 加速后约 0.8–2.7 µs；布尔碰撞检测约 0.6–3.3 µs。时间随顶点数增加而增加。论文提醒绝对时间与实现和硬件有关。
- **Coal README**：加速 GJK 在其基准中比 Bullet、原版 FCL（Drake 使用）、libccd（MuJoCo 使用）快 **5–15 倍**；Nesterov 加速最多快 2 倍；对几十到几百个顶点的几何，GJK 远快于把距离问题写成 QP 再用 ProxQP 求解。
- **cuRobo**：论文报告 7600 次无碰撞 IK/s（RTX 4090，含自碰撞和环境碰撞，球近似）。文档说明其 cuboid 碰撞检查比 mesh 快 4 倍；不支持网格对网格，必须把一侧近似为球。
- **MoveIt Setup Assistant**：默认采样 10,000 个随机构型生成自碰撞矩阵（并行计算），推荐把采样密度调到最大。

**估算（非出处数据，需实测）**：本机器人剪枝后约 40–80 对凸体，单对按上面的 1–4 µs 计，一次全对距离约 0.05–0.3 ms 量级。用球/胶囊解析距离会再低一个量级以上。用原始非凸网格则可能慢得多（BVH 加大量三角形对）。**请用 Coal 的 `QueryRequest::enable_timings` 或自己计时实测。**

### 2.3 降开销的标准手段（按收益排序）

1. **碰撞对剪枝（收益最大）**：
   - **相邻连杆**：父子连杆通常永远接触或不可能相撞，直接禁用。Drake 默认排除父子连杆。
   - **永不碰撞（never in collision）**：MoveIt Setup Assistant 在随机构型下采样，标记「总是碰撞 / 从不碰撞 / 默认位姿碰撞 / 相邻」的对，写进 SRDF 的 `<disable_collisions>`。Pinocchio 可直接读：`addAllCollisionPairs()` 之后调 `removeCollisionPairs(model, geom_model, srdf)`。placo 用 `collisions.json`（可脚本生成）。
   - **同一刚体内部的对**：例如被腰带着一起动的「胸 + 头 + 两肩」之间。
2. **几何简化**：球 < 胶囊 < 凸包 ≪ 原始网格。手腕、手爪附近用多个小球或胶囊；大臂、小臂用一个胶囊；躯干用 1–2 个胶囊或盒子。凸包可以保留更多形状细节，代价仍是微秒级。
3. **只计算近处的对**：broadphase（AABB）先剔除远处的对（mink 的 `broadphase`，FCL 的 broadphase 管理器）；激活或触发距离只为近处的对生成约束（placo 的 trigger、Drake 的 influence distance、cuRobo 的激活距离）。
4. **按模式选对**：只计算在当前模式下**相对位姿会改变**的对（§5.2）。
5. **复用缓存**：GJK warm start（Coal 支持缓存的 GJK 初始猜测）；相邻迭代之间的最近点变化很小。
6. **批量或 GPU 计算**：TAMP 中大量 IK 候选可以先用 cuRobo 式的球模型批量筛选。

---

## 3. 能不能用 MoveIt 的碰撞检测？

### 3.1 MoveIt 提供什么

- **后端**：FCL（默认）和 Bullet，通过 `PlanningScene::setActiveCollisionDetector(...)` 切换。Bullet 支持连续碰撞检测（CCD），只返回最深穿透点；文档注明 Bullet 后端**不是线程安全的**。
- **碰撞检查**：`checkSelfCollision`、`checkCollision`，遵循 ACM（Allowed Collision Matrix，来自 SRDF）。
- **距离查询**：`distanceSelf`、`distanceRobot`。`DistanceRequest` 支持 `enable_nearest_points`、`enable_signed_distance`、`distance_threshold`（只计算阈值内的对，注释说可显著减少查询）和 `compute_gradient`。注意源码注释说的 gradient 是「**连接两个最近点的单位向量**」，也就是 §1.1 的 $n$，**不是**关节空间梯度。
- **Bullet 后端的距离查询未实现**：`CollisionEnvBullet::distanceSelf` 和 `distanceRobot` 在源码中只打印「not implemented for Bullet」。要做距离查询，请用 FCL 后端。

### 3.2 放进 placo/CasADi 的迭代循环：可行但不划算

| 方面 | 说明 |
|---|---|
| 能拿到的数据 | $d$、$p_A$、$p_B$、$n$（FCL 后端）。要得到 $J_d$，还需对每个最近点调用 `RobotState::getJacobian(group, link, reference_point)` 并自己拼接 |
| 额外开销 | 每次迭代都要更新 `RobotState`、再经过 `CollisionEnv` 的封装；关节顺序与 Pinocchio 不同，需要映射 |
| 依赖 | `moveit_core`（C++、ROS 2 生态）；Python 绑定（moveit_py）主要面向规划接口 |
| 一致性 | 好处是与规划器用同一套模型、同一个 ACM |
| 结论 | 你的 IK 已经基于 Pinocchio，**用 Pinocchio + Coal 就能拿到同样的距离和最近点，而且关节雅可比是原生的**，不需要跨库拼接 |

### 3.3 放在最终校验和规划里：自然且推荐

- **TAMP 的最终校验**：把 IK 解写进 `RobotState`，调用 `PlanningScene::checkCollision`，用的是**原始网格**、包括 attached objects（手里的负载）和环境。这样 IK 与下游 OMPL 规划用的是同一个碰撞判定，避免「IK 认为无碰撞、规划器却认为碰撞」。
- **ACM / SRDF 复用**：Setup Assistant 生成的 `<disable_collisions>` 可同时给 MoveIt、Pinocchio（`removeCollisionPairs`）和 placo（转成 `collisions.json`）使用。

### 3.4 MoveIt IK 插件的「验证回调」模式及其缺点

MoveIt 的 `RobotState::setFromIK(..., GroupStateValidityCallbackFn constraint)` 把回调包装成 IK 插件的 `solution_callback`。以 KDL 插件 `searchPositionIK` 为例：
1. 从种子求一个 IK 解（求解本身**完全不知道碰撞**）。
2. 调用回调（通常是 `PlanningScene::isStateValid`，即碰撞检查）。
3. 被拒绝就**随机重新播种**（第一次之后每次 `getRandomConfiguration`），直到超时。

| 缺点 | 解释 |
|---|---|
| 生成后测试（generate-and-test），没有梯度引导 | 目标在狭窄可行区域（两臂靠近、手贴躯干）时，大部分随机解都会被拒绝，成功率低、耗时直到超时 |
| 解的质量不可控 | 随机重启得到的构型可能与当前构型差很远（腰或肘大幅翻转） |
| 不知道离碰撞有多近 | 只有布尔结果，没有安全裕度和距离代价 |
| 优点 | 实现简单、与任何 IK 插件兼容、判定精确（原始网格） |

**对比**：把自碰撞写进 QP/NLP 后，求解器会被**推离**碰撞方向，而不是撞上了再重来。所以本项目把自碰撞作为约束放进求解器，MoveIt 只做最后一道精确校验。

---

## 4. 替代方案对比

| 方案 | 原理 | 可微性 | 速度 | 精度 | 适合哪一层 |
|---|---|---|---|---|---|
| **Pinocchio + Coal** | `GeometryModel`、`computeDistances`、SRDF 剪枝；GJK/EPA | 给出最近点和法向，配合 Pinocchio 雅可比得 $J_d$；网格面对面时不光滑 | 单对微秒级（见 §2.2） | 取决于几何（网格或凸包或基本体） | QP 层（placo 内部就是它）、NLP 层（作为 CasADi 外部回调）、校验层 |
| **球 / 胶囊解析模型** | 球对：$d=\|c_i-c_j\|-r_i-r_j$；胶囊对：两线段最短距离减去两半径 | 球对处处光滑（球心不重合时）；胶囊分段光滑；**都可以在 CasADi 中符号表达** | 最快，每对只有几十次浮点运算 | 保守近似，需要调半径 | **NLP 层首选**；QP 层也可以 |
| **cuRobo 球模型（GPU）** | 机器人用球集合表示，GPU 并行 | 可微代价 | 批量最快 | 球近似 | TAMP 候选批量筛选 |
| **Drake** `MinimumDistanceLowerBoundConstraint` | SceneGraph（FCL 等）+ SmoothOverMax | 光滑化 | 中等 | 取决于几何 | 若 NLP 层改用 Drake，可直接使用 |
| **MoveIt（FCL/Bullet）** | PlanningScene + ACM | 只有法向和最近点（FCL） | 有封装开销 | 原始网格，与规划一致 | **最终校验** |
| **Bullet** | GJK/EPA，支持 CCD | 最深穿透点 | 快 | — | 校验、连续碰撞（轨迹段） |
| **SDF / 距离场** | 对连杆或环境预计算有符号距离场，查询为插值 | 光滑（插值） | 查询很快，预计算和内存开销大 | 受分辨率限制 | 环境碰撞更常用；自碰撞用得较少 |
| **学习型自碰撞边界**（Koptev, Figueroa, Billard, RA-L 2021） | 在关节空间学习光滑的自碰撞边界函数（比较了 SVM 和神经网络），把 29-DOF iCub 拆成若干低维子模型，作为 QP IK 的约束 | 处处可微 | 实时 | 是统计近似，有漏报风险 | 研究选项；需要采样训练，代码见 epfl-lasa/Joint-Space-SCA。对 TAMP 的离线 IK 不是必需 |

---

## 5. 针对本机器人的推荐

### 5.1 哪些碰撞对真正重要（2-DOF 腰 + 双 SRS 臂）

假设运动链为：底座/下身 → 腰 yaw → 腰 pitch → 胸部（头、两肩安装在胸部）→ 左右臂（肩 3、肘 1、腕 3）→ 手 / 手爪 →（持握的负载）。

| 类别 | 典型碰撞对 | 何时发生 | 重要性 |
|---|---|---|---|
| **臂对臂** | 左小臂/腕/手 ↔ 右小臂/腕/手；手 ↔ 另一侧大臂 | 双手靠近、交叉、双手持物、递交 | ★★★ 最常见 |
| **手臂对胸/躯干** | 小臂、腕、手 ↔ 胸、腹（腰 pitch 以上的躯干） | 手贴近身体、抱物、肘内收 | ★★★ |
| **手对头/颈** | 手、手爪、负载 ↔ 头、相机 | 抬手过肩、举物到面前 | ★★ |
| **大臂对躯干侧面** | 大臂 ↔ 胸侧 | 肩内收（SRS 肩关节极限附近） | ★★（关节极限常能挡住大部分） |
| **上身对下身/底座** | 胸、手臂、手 ↔ 底座、腿、底盘 | **腰 pitch 前倾较大**、手伸向低处 | ★★（只有腰动时才变化） |
| **持握负载** | 负载 ↔ 另一臂、躯干、头 | 搬运时 | ★★★（必须把负载作为 attached geometry 挂到手上；Pinocchio 中挂到腕关节下的 GeometryObject） |
| 可安全禁用 | 相邻连杆；同一臂上相隔一个关节的连杆（通常由 Setup Assistant 判为 never）；胸、头、双肩之间（同一刚体）；手指之间 | — | 剪掉 |

**SRS 臂特有的注意点**：SRS 臂有 1 维自运动，用臂角 $\psi$ 参数化（肘绕肩-腕连线转动）。同一个手部位姿下，**肘可能向内摆、撞到躯干**，也可能向外摆。所以「肘/大臂 ↔ 躯干」这一对在 IK 中很有用，它会把 $\psi$ 推离身体。如果 L1 层用解析臂角扫描生成种子，可以**在扫描时就用便宜的胶囊距离筛掉 $\psi$ 区间**，这一点与上一份报告中「把约束投影到臂角域」的思路一致。

### 5.2 按模式选碰撞对（利用 §1.2 的结论）

关键事实：**腰的运动不会改变腰下游连杆之间的相对位姿**，因为胸、头和两臂都在腰的下游，腰对它们的作用在 $J_d$ 中抵消。所以：

| 模式 | 活动关节 | 需要计算的自碰撞对 | 可以跳过 |
|---|---|---|---|
| **M1** 单臂（例如左臂） | 左臂 7 | 左臂（含手、负载）↔ {右臂、胸、头、底座} | 右臂 ↔ 胸/头（相对位姿不变，只在初始化时检查一次）；所有上身 ↔ 下身对（腰锁定，相对不变，左臂对底座除外） |
| **M2** 腰 + 左臂 | 腰 2 + 左臂 7 | M1 的所有对，**加上**所有上身连杆（胸、两臂、头、负载）↔ 下身/底座 | 右臂 ↔ 胸/头 |
| **M3** 腰 + 双臂 | 16 | 全部启用的对（经 SRDF 剪枝后） | — |

（另一只臂若在持物，它的负载也要算作上身的一部分。）这样 M1 中只需计算大约一半的对。**全状态校验层仍要检查全部对**，作为最后保险。

### 5.3 分层实现方案

```
                      碰撞对来源：MoveIt Setup Assistant 生成的 SRDF <disable_collisions>
                                   │（一次生成，三处复用）
         ┌─────────────────────────┼──────────────────────────┐
         ▼                         ▼                          ▼
 L2 placo QP 层             L3 CasADi/IPOPT NLP 层        L4 全状态校验
 几何：胶囊/凸包            几何：球或胶囊（符号表达）      几何：原始网格 + 负载 + 环境
 对表：SRDF 剪枝 + 按模式    或 Coal 回调（外部函数）       工具：MoveIt PlanningScene(FCL)
       生成 collisions.json  约束：d_k(q) ≥ d_safe + δ        或 Pinocchio + Coal(网格)
 约束：AvoidSelfCollisions   只对 d<d_act 的对加约束         结果：布尔 + 最小距离
   margin≈1–2 cm，            (或光滑 max 合成，类似 Drake)  → 不通过就丢弃该解或作为
   trigger≈5 cm（起点值）                                    「需要重新求解」的反馈
```

**L2（placo）**：
- 用于 placo 的碰撞 URDF（或单独的 collision 模型）用基本体：大臂、小臂各 1 个胶囊，手 1–3 个胶囊或盒，胸 1–2 个胶囊或盒，头 1 个球。
- `collisions.json` = SRDF 剪枝后的对 ∩ 当前模式需要的对。按模式准备三份，或在加载后用 `load_collision_pairs` 切换。
- 起点参数：`self_collisions_margin` 1–2 cm，`self_collisions_trigger` 约 5 cm（文档示例为 1 cm / 5 cm）。设为 hard 约束；若出现不可行，改为 soft 并加大权重，交给 NLP 层收紧。

**L3（CasADi + IPOPT）**，两种实现：
1. **推荐：符号胶囊/球距离。** 用 `pinocchio.casadi` 算出各胶囊端点的世界坐标（FK 是符号的），在 CasADi 中写线段-线段距离（有 clamp，分段光滑）或球-球距离，IPOPT 自动得到精确梯度。约束形式为 $d_k(q)\ge d_{safe}+\delta$，$\delta$ 是近似误差裕度（几毫米）。对数少时逐对加约束；对数多时只加 $d<d_{act}$ 的对，或用光滑 max 合成一个约束（Drake 的做法）。
2. **备选：Coal 回调。** 用 `casadi.Callback` 封装「Coal 距离 + $J_d$」作为外部函数。优点是可用凸包，缺点是梯度不光滑、回调有开销。

**L4（校验）**：
- 若有 MoveIt 环境：用 `PlanningScene::checkCollision`（原始网格 + attached objects + 世界）。若只用 Pinocchio：用 Coal 对原始网格碰撞模型做检查（注意 Pinocchio issue #1993：有符号距离要求至少一个形状是凸的，非凸网格可用 `buildConvexRepresentation`）。
- 记录最小距离，作为 TAMP 解排序的一项。

### 5.4 开销控制与实测建议

1. 先用 Setup Assistant 生成 SRDF（采样密度调到最大），统计剪枝后的对数。
2. 用 Coal 的 `enable_timings` 或 Python 计时，测三种几何（球/胶囊、凸包、网格）下「全对距离一次」的耗时。
3. 测 placo 单步耗时，在开启和关闭自碰撞约束两种情况下对比。
4. 若碰撞计算占比超过约一半：先继续剪枝和按模式选对，再简化几何，最后才考虑 GPU 或学习型方法。

---

## 6. 参考链接（均已核实）

**理论**
- Faverjon & Tournassoud 1987（velocity damper，ICRA）：见 Kanoun et al. 引用；Kanoun, Lamiraux, Wieber, *Kinematic Control of Redundant Manipulators: Generalizing the Task-Priority Framework to Inequality Task*（T-RO 2011）：https://ieeexplore.ieee.org/document/5766760 ；预印本 https://hal.science/hal-00486755v2/document
- Kanoun, *Real-time prioritized kinematic control under inequality constraints*（RSS 2011）：https://www.roboticsproceedings.org/rss07/p21.html
- Velocity damper 在 MPC 中的应用（Agimus，ICRA 2025）：https://www.agimus-project.eu/media/attachments/2025/01/23/icra_2025__1_-12.pdf
- Montaut et al., *Collision Detection Accelerated: An Optimization Perspective*（RSS 2022，含 µs 级时间表）：https://www.roboticsproceedings.org/rss18/p039.pdf
- Montaut et al., *Differentiable Collision Detection: a Randomized Smoothing Approach*（ICRA 2023）：https://arxiv.org/abs/2209.09012
- Koptev, Figueroa, Billard, *Real-Time Self-Collision Avoidance in Joint Space for Humanoid Robots*（RA-L 2021）：https://ieeexplore.ieee.org/document/9345975 ；代码 https://github.com/epfl-lasa/Joint-Space-SCA

**库与源码**
- placo 自碰撞文档：https://github.com/Rhoban/placo/blob/master/docs/kinematics/avoid_self_collisions_constraint.rst ；源码 https://github.com/Rhoban/placo/blob/master/src/placo/kinematics/avoid_self_collisions_constraint.cpp ；碰撞对配置 https://github.com/Rhoban/placo/blob/master/docs/basics/collisions.rst
- pink `SelfCollisionBarrier`：https://github.com/pink-kinematics/pink/blob/main/pink/barriers/self_collision_barrier.py ；示例 https://github.com/pink-kinematics/pink/tree/main/examples/barriers
- mink limits：https://kevinzakka.github.io/mink/api/limits.html
- Drake `MinimumDistanceLowerBoundConstraint`：https://drake.mit.edu/doxygen_cxx/classdrake_1_1multibody_1_1_minimum_distance_lower_bound_constraint.html
- Coal（原 HPP-FCL）：https://github.com/coal-library/coal
- Pinocchio 碰撞与距离：https://gepettoweb.laas.fr/doc/stack-of-tasks/pinocchio/master/doxygen-html/md_doc_b_examples_c_collisions.html ；SRDF 剪枝 `removeCollisionPairs`：https://github.com/stack-of-tasks/pinocchio/blob/0a64d89b/examples/collisions.py ；有符号距离与凹形状 issue：https://github.com/stack-of-tasks/pinocchio/issues/1993
- cuRobo 碰撞文档：https://curobo.org/get_started/2c_world_collision.html ；机器人配置（`collision_spheres`、`self_collision_ignore`）：https://curobo.org/tutorials/1_robot_configuration.html
- MoveIt 2 Setup Assistant（自碰撞矩阵）：https://moveit.picknik.ai/main/doc/examples/setup_assistant/setup_assistant_tutorial.html
- MoveIt 2 碰撞检测概念（FCL/Bullet）：https://moveit.ai/documentation/concepts/developer_concepts/ ；Bullet 教程：https://moveit.picknik.ai/main/doc/examples/bullet_collision_checker/bullet_collision_checker.html
- MoveIt `DistanceRequest` 源码（`compute_gradient` 只是法向）：https://github.com/moveit/moveit2/blob/main/moveit_core/collision_detection/include/moveit/collision_detection/collision_common.hpp ；Bullet 距离未实现：https://moveit.picknik.ai/main/api/html/collision__env__bullet_8cpp_source.html
- MoveIt KDL 插件（回调拒绝 + 随机重启）：https://github.com/moveit/moveit2/blob/main/moveit_kinematics/kdl_kinematics_plugin/src/kdl_kinematics_plugin.cpp
