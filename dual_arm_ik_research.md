# 双臂机器人（共享可动腰部）逆运动学（IK）调研简报：面向任务与运动规划（TAMP）

> 调研日期：2026-10-10 ｜ 适用场景：**双 7-DOF 臂 + 可动腰/躯干（共享运动链 / 运动学树）** 的**任务与运动规划**（不是遥操作）
> 说明：文中所有论文、项目、数字均来自本次实际检索/抓取的页面，链接附后。未能核实的内容一律未写入。数字均为原文作者在其实验设置下报告的结果，不同论文之间不可直接横向比较。

---

## 0. TL;DR

1. **不要把它当作「两条独立的 7-DOF 链」**。腰部关节同时出现在左右两条末端链上，在 IK 和规划里都是**树结构**问题：任何一侧末端的目标都会通过腰部关节「拉动」另一侧。正确做法是在**一个整体关节向量** `q = [q_waist, q_L, q_R]`（例如 2+7+7=16 维）上联合求解，同时施加两个末端位姿、关节极限、自碰撞以及（双手持物时）相对位姿约束。
2. **IK 在 TAMP 中的角色**主要有三个：(a) 目标/抓取位姿 → 关节构型的**采样器**（PDDLStream 的 stream、MoveIt Task Constructor 的 `ComputeIK`）；(b) 约束流形上的**投影/参数化**算子（双手持同一物体的闭链约束）；(c) 轨迹优化中的**约束/初值**（TrajOpt、Drake、cuRobo）。
3. **工具上的关键事实**：TRAC-IK / KDL 位置 IK 都是**单链**（`base_link → tip_link`），不能直接解「腰 + 双臂、两个末端」的问题；MoveIt 2 默认的 KDL/LMA 插件在 `both_arms` 这类多链组上会报 *"Group is not a chain"*，需要支持多 tip 的插件（如 pick_ik、bio_ik）或自写插件。**Drake `InverseKinematics`、Pinocchio/pink/mink/placo（QP 微分 IK）、cuRobo（`link_names` 多末端）、PyRoki** 天然以整棵树为对象，适合共享腰部的联合 IK。
4. **双手持物的闭链规划**：OMPL 提供 Projection / Atlas / TangentBundle 约束空间；最近的研究（Cohn et al. 2023/ICRA 2024；Paro et al. 2026）表明 **leader–follower 参数化**（一臂自由采样、另一臂由 IK 求解以保持相对位姿）比通用投影/Atlas 更快、约束满足更严格。有可动腰时，建议把**腰 + leader 臂**作为自由变量，follower 臂由 IK 闭合。
5. **推荐栈（详见 §7）**：仿真/离线规划主干用 **Drake**（整体 IK + 运动学轨迹优化 + GCS + 约束规划）或 **cuRobo**（GPU 批量无碰撞 IK + 运动生成）；ROS 2 生态部署用 **MoveIt 2 + MTC + OMPL（约束规划）+ 多 tip IK 插件（pick_ik / bio_ik）**；任务层用 **PDDLStream**（或 MTC 的阶段化管道）；底层运动学与微分 IK 用 **Pinocchio + pink/placo**（或 MuJoCo + mink）。

---

## 1. 问题建模：带共享腰部的双臂 IK

### 1.1 运动学结构
- 机器人是一棵树：`base → waist(1~3 DOF) → {左臂 7 DOF → 左末端, 右臂 7 DOF → 右末端}`。
- 关节向量 `q = [q_w, q_L, q_R]`。左末端位姿 `T_L = FK_L(q_w, q_L)`，右末端 `T_R = FK_R(q_w, q_R)`。
- 雅可比按块组织：
  ```
  J = [ J_L,w  J_L,L   0    ]   ← 左末端 6 行
      [ J_R,w   0     J_R,R ]   ← 右末端 6 行
  ```
  腰部列 `J_*,w` **同时出现在两行**：这正是「耦合」来源——为左手移动腰，会改变右手位姿，除非右臂补偿。

### 1.2 常见任务/约束项（在 TAMP 中典型的组合）
| 类别 | 形式 | 说明 |
|---|---|---|
| 绝对位姿 | `T_L = T_L*`, `T_R = T_R*` | 抓取/放置目标，可用 Task Space Region（TSR，位姿区间）放松 |
| 相对位姿（闭链） | `T_L^{-1} T_R = T_rel*`（常数） | 双手持同一刚体；构型空间中是**零测度流形**（2n−6 维；两条 7-DOF 臂时为 8 维，见 Paro et al.） |
| 关节极限 / 速度极限 | 盒约束 | 7-DOF + 腰：极限比奇异更常成为瓶颈（TRAC-IK 的动机） |
| 自碰撞 / 臂-臂碰撞 / 环境碰撞 | 有符号距离 ≥ d_min | 双臂近距离协作时臂间碰撞是主要失败原因（SDAR 论文） |
| 冗余分解 | 零空间姿态、可操作度、关节居中、靠近种子 | 7-DOF 每臂 1 维自运动 + 腰部冗余 |
| 腰部偏好 | 腰部正则/低权重/限幅 | 腰通常质量大、慢、影响平衡与视野，应「能不动就少动」 |

### 1.3 冗余与可操作度
- 7-DOF 臂对给定末端位姿存在 1 维自运动流形；Cohn et al. 对 7-DOF 臂的自运动（连续 + 离散分支）有清晰讨论，并指出离散的「全局构型参数」（如 iiwa 有 8 个分支）在规划中需固定或显式切换。
- 可操作度（Yoshikawa，`sqrt(det(J Jᵀ))`）可作为代价：TRAC-IK 的 `Manip1~3` 求解模式、PyRoki 的 manipulability cost、placo 的 `ManipulabilityTask`、IKDiffuser 的引导采样都直接支持。

### 1.4 腰部带来的特殊问题
1. **工作空间扩展 vs. 耦合**：腰部转动/俯仰显著扩大双手可达空间（例如 PyRoki 论文中把移动底座位姿加入优化，Fetch 实验成功率从固定底座的 24% 提升到 100%——同样的思路适用于腰部）。但腰部移动会同时扰动两只手。
2. **优先级**：典型的严格优先级为「闭链/相对位姿约束（硬） > 主末端位姿 > 次末端位姿 > 腰部姿态正则 > 关节居中」。QP 类求解器（placo 的 `hard`/`soft`+权重、pink/mink 的任务权重 + 约束）可以直接表达。
3. **规划维度**：腰 + 双臂 ≈ 16 维；SDAR 论文指出双臂系统使自由度从 6–7 维翻倍到 12+ 维，即使 cuRobo 对随机起止构型也难以稳定成功——加上腰部只会更难，因此需要良好的 IK 种子、分层/解耦策略。

---

## 2. IK 求解方法分类与优缺点（以双臂+腰为视角）

| 方法 | 代表 | 优点 | 缺点 / 对共享腰部的影响 |
|---|---|---|---|
| 解析 / 闭式 | IKFast（OpenRAVE）、各臂专用几何解（iiwa、Panda 等） | 极快、可枚举所有分支、平滑可微（Cohn et al. 利用这一点参数化闭链流形） | 针对**单条串联链**；7-DOF 需要指定 free joint 离散化；腰 + 双臂整体无法直接生成。可用法：**固定/采样腰部角度 → 对每只臂用解析 IK** |
| 数值雅可比（伪逆 / DLS / 零空间投影） | KDL `ChainIkSolverVel_wdls`、Pinocchio CLIK 示例、KDL `TreeIkSolverVel_wdls`（多末端） | 简单、实时、零空间可加次任务 | 局部收敛、关节极限处易卡死（pink 文档明确说明微分 IK 是局部算法）；多末端时需手动加权/优先级 |
| SQP / 非线性优化 | TRAC-IK（KDL-RR + SQP 并行）、Drake `InverseKinematics`、CasADi+IPOPT、RelaxedIK/RangedIK | 关节极限处理好；可加碰撞、距离、姿态区间等约束 | TRAC-IK 仅单链；Drake/CasADi 可整树求解但较慢、需好的初值 |
| QP 任务优先级 / 全身微分 IK | pink（Pinocchio）、mink（MuJoCo）、placo、Isaac Lab Pink IK | 多任务 + 硬约束（关节/速度极限、碰撞、闭链等式）统一在一个 QP；非常适合「腰+双臂」 | 本质是局部速度级方法；用于离线规划时需迭代积分到收敛，可能停在局部极小 |
| 全局 / 采样 / 进化 | bio_ik（memetic）、pick_ik（global 模式含进化算法）、cuRobo 多种子、Drake `GlobalInverseKinematics`（混合整数凸优化） | 跳出局部极小；GlobalIK 可**证明不可行** | 计算代价高；GlobalIK 需 Gurobi 等 MIP 求解器 |
| GPU 批量 | cuRobo（37000 IK/s、7600 无碰撞 IK/s，相对 TRAC-IK 分别快 23×/80×，RTX 4090）、PyRoki（JAX，LM，CPU/GPU/TPU） | 适合 TAMP 中大批量抓取候选 / 可达性筛选 | 需要 GPU；cuRobo 需球体近似碰撞模型 |
| 学习型 | IKFlow（归一化流）、IKDiffuser（扩散，支持运动学树） | 快速产生多样解，用作**优化器的种子** | 精度仅毫米级，必须后接数值精修；需针对机器人训练 |

**学习型 IK 的关键结论（IKDiffuser, arXiv 2506.13087）**：其实验包含「Dual RM76 and Waist（17 DOF，双臂 + 腰）」平台；单独使用时位置误差约 4–5 mm，作为种子给 cuRobo 时该平台成功率 83.39% → 99.85%，给 Pink 时 69.38% → 99.04%；29-DOF Unitree G1 上 cuRobo 成功率 21.01% → 96.96%。作者明确结论：「生成 + 优化精修」是互补的，学习模型不能替代优化求解器。

---

## 3. IK 在任务与运动规划中的用法

### 3.1 目标/抓取采样（Goal sampling）
- **PDDLStream**（Garrett, Lozano-Pérez, Kaelbling，ICAPS 2020）：把 IK、抓取采样、碰撞检测、运动规划声明为黑盒 **stream**；其 PR2（双臂）示例中 `inverse-kinematics` 就是一个 stream，与 `sample-grasp`、`sample-pose`、`plan-base-motion` 并列。乐观规划只在需要时调用昂贵的 IK/运动规划。
- **MoveIt Task Constructor（MTC）**：`GenerateGraspPose`（围绕物体按 `angle_delta` 采样抓取）→ `ComputeIK`（每个候选求多个 IK 解，`max_ik_solutions`、最小解间距）→ 连接阶段规划。注意：MTC 维护者在 issue #390 中说明 **Cartesian 目标只支持单一末端**；双臂需每臂一个分支再用 `Merger` 合并（关节空间目标可直接用于组合组）。
- **GPU 批量筛选**：SDAR（arXiv 2512.08206，双 UR5e 桌面重排 TAMP）用 cuRobo 批量 IK 对多个子任务 × 多个抓取角度做可行性筛选（RTX 4090 单批约 256 个 IK，平均 20 批，约 1.7 s；IK 占规划时间约 9%）。MODAP（arXiv 2404.06758）指出 cuRobo 可能返回不理想的 IK 分支（如肘部低于末端），通过**指定 IK 种子**解决——这对有腰部的机器人同样重要（应给腰部一个「中立」种子）。

### 3.2 双手持物：闭链 / 相对位姿约束下的规划
1. **通用约束规划（OMPL）**：`ProjectedStateSpace`（牛顿法投影到约束流形）、`AtlasStateSpace`（切空间图表的分段线性近似）、`TangentBundleStateSpace`（惰性 atlas）。OMPL 文档建议先用 Projection 验证可行性（对参数不敏感）。理论综述：Kingston, Moll, Kavraki, *Sampling-Based Methods for Motion Planning with Constraints*（Annu. Rev. 2018）。
2. **TSR / CBiRRT**（Berenson et al.）：用 Task Space Regions 描述位姿区间约束，雅可比伪逆投影到约束流形。
3. **IK 参数化（leader–follower）**：
   - **Cohn, Shaw, Simchowitz, Tedrake — *Constrained Bimanual Planning with Analytic Inverse Kinematics*（arXiv 2309.08770，ICRA 2024）**：一臂自由，另一臂用解析 IK 跟随，使可行集变为正测度，可直接用 RRT/PRM、轨迹优化、GCS（并扩展 IRIS-NP 生成凸区域）。双 iiwa 实验中，基线方法（OMPL Atlas、约束轨迹优化）在离散点之间最大约束违反 6.62 cm / 3.22 cm，参数化方法保持在 0.001 cm 内。代价：IK-GCS 的 IRIS 区域离线构建约 5 小时，环境一变就失效。
   - **Paro, Petrović, Marković — *Fast Coordinated Bimanual Motion Planning With Hard Constraints*（arXiv 2608.20946，2026）**：不需要解析 IK、不需要离线预计算；leader 采样 + follower 数值 IK（以前一状态为种子保持同一分支）、约束感知插值、奇异附近拒绝采样、跟随臂加权的距离度量，基于 OMPL RRT-Connect，再用 TOPP-RA 时间参数化。在 Cohn et al. 的 iiwa 基准上比 IK-BiRRT 平均快 19.4×；在双 Kinova Gen3 实机上完成托盘搬运、长物体搬运。（代码作者称审稿后开源。）
   - **对可动腰的启示**：把 `[q_waist, q_leader]` 作为自由采样变量，follower 臂在**当前腰部角度**下用 IK 闭合相对位姿；或者把腰部也放入 follower 的 IK（变成「腰 + follower 臂」冗余 IK，但腰同时影响 leader，必须在整体 FK 下求解）。前者更简单、更稳定。
4. **轨迹优化直接加等式约束**：Drake `KinematicTrajectoryOptimization` / `InverseKinematics` 可添加两末端之间的相对位置/姿态约束（`AddPositionConstraint(frameB, p_BQ, frameA, ...)` 支持以另一机器人坐标系表达、`AddOrientationConstraint` 支持两帧之间夹角界），但 Cohn et al. 指出非凸等式约束只在好的初值下才收敛，且只在离散点上满足。

### 3.3 MoveIt 2 中的约束规划能力边界
- MoveIt 2 的 OMPL 约束规划（`enforce_constrained_state_space: true`）目前**只支持单个位置或单个姿态约束（含一位置+一姿态混合）**，作用于某一 link，相对于参考坐标系——**不直接支持两末端间的相对位姿约束**。MoveIt 维护者在 issue #3344 中也确认 MoveIt 核心缺少相对 TCP 位姿 IK 的基础设施，需自写插件。
- 因此双手持物闭链在 MoveIt 2 中通常需要：自定义 OMPL `Constraint`（相对位姿残差）+ 自定义状态采样器；或在外部（Drake / 自写 OMPL 程序）完成，再把轨迹交给 MoveIt 执行。

### 3.4 轨迹优化
| 方法/工具 | 要点 |
|---|---|
| TrajOpt（Tesseract：`trajopt_ifopt` + `trajopt_sqp`） | 序列凸优化；约束集含关节位置/速度/加速度/jerk、`CartPosConstraint`、`CartLineConstraint`、离散/连续碰撞、`InverseKinematicsConstraint`；默认 OSQP |
| CHOMP / STOMP（MoveIt 2 规划管线） | CHOMP 可用 OMPL 结果作初值做优化；STOMP 为随机轨迹优化 |
| Drake `KinematicTrajectoryOptimization` | B 样条路径 + 时长变量；可加位置/速度界、时长代价，以及任意 IK 类约束 |
| Drake GCS（Marcucci et al., Science Robotics 2023） | 在凸无碰撞区域图上做凸松弛 + 舍入，得到近全局最优轨迹；需离线构建 IRIS 区域 |
| cuRobo MotionGen | 并行几何规划 + L-BFGS 轨迹优化，平均约 50 ms（论文）；双臂通过 `link_names` + `link_poses` 同时规划两个末端；缺点：两臂**同步**到达（MODAP 指出不适合异步双臂任务） |
| PyRoki | JAX + LM，IK / 轨迹优化 / 重定向统一的代价组合；软约束（关节极限、碰撞为可微惩罚，不处理硬约束） |
| VAMP（Kavraki Lab） | CPU SIMD 向量化的采样规划（RRT-Connect、PRM 等），Panda RRT-Connect 中位数约 35 µs（单核）；已有 OMPL 集成教程，支持 Panda/UR5/Fetch/Baxter |

---

## 4. 开源工具与库（逐一说明「共享腰部」能力）

| 工具 | 语言 | 是什么 | 对「腰 + 双臂」的处理 | 链接 |
|---|---|---|---|---|
| **KDL**（Orocos） | C++ | 运动学/动力学库；`ChainIkSolverPos_LMA`、`ChainIkSolverVel_wdls` | 位置 IK 为**单链**；有 `TreeIkSolverVel_wdls`（多末端速度级）可做树 | https://github.com/orocos/orocos_kinematics_dynamics |
| **TRAC-IK** | C++（ROS） | KDL-RR + SQP 并行；`Speed/Distance/Manip1-3` 模式 | 构造函数 `TRAC_IK(base_link, tip_link, ...)` 内部 `tree.getChain()` → **严格单链、单末端**。变通：①把腰包含在每条链里（base→左手、base→右手），分别求解后**腰部值冲突**需协调（例如先固定腰再解两臂，或外层搜索腰角）；②换多末端求解器。README 注明 KDL 非线程安全，同进程不宜多实例；且 IK 不检查自碰撞 | https://github.com/traclabs/trac_ik |
| **IKFast** | Python 生成 C++ | OpenRAVE 自动生成解析 IK；7-DOF 需 `--freeindex` | 单链；可把腰关节设为 free joint（运行时给定），即「采样腰角 → 解析求每臂」 | https://www.openrave.org/docs/latest_stable/openravepy/ikfast/ |
| **MoveIt 2** | C++/Python | 规划框架：规划组、OMPL/CHOMP/STOMP/Pilz 管线、碰撞检测 | 规划组可包含 torso + 双臂（如 `torso_both_arms`），OMPL 在组的全部关节空间上联合规划（腰天然是一个关节变量）；**IK 插件**：KDL/LMA 对多链组报 "not a chain"，需用支持多 tip API 的插件 | https://moveit.picknik.ai/main/doc/examples/dual_arms/dual_arms_tutorial.html |
| **pick_ik** | C++（MoveIt 插件） | 梯度下降 + 可选进化算法（`mode: global`） | 实现 MoveIt **多 tip** IK 接口（每个 tip 一个位姿代价） | https://github.com/PickNikRobotics/pick_ik |
| **bio_ik** | C++（MoveIt 插件） | memetic（梯度 + 遗传 + 粒子群）；多目标 | 支持多末端位姿目标 → 可用于腰+双臂组 | https://github.com/TAMS-Group/bio_ik |
| **MoveIt Task Constructor** | C++/Python | 阶段化任务管道（生成器/传播器/连接器） | 每个阶段指定规划组；双臂 Cartesian 目标需两分支 + `Merger` | https://github.com/moveit/moveit_task_constructor |
| **OMPL** | C++/Python | 采样规划库 + 约束规划（Projection/Atlas/TangentBundle） | 状态空间可为任意关节向量（含腰）；闭链约束写成 `ompl::base::Constraint` | https://ompl.kavrakilab.org/core/constrainedPlanning.html |
| **Pinocchio** | C++/Python | 刚体运动学/动力学 + 解析导数 | 整棵树模型；任意 frame 雅可比；`buildReducedModel` 锁定关节（Unitree xr_teleoperate 用它锁定非手臂关节——若要让腰参与，则**不要锁腰**）；可配合 CasADi | https://github.com/stack-of-tasks/pinocchio |
| **pink** | Python | Pinocchio 上的 QP 微分 IK（加权任务 + 极限） | 多 `FrameTask` 共享腰部关节自然耦合；有 "Flying dual-arm UR3" 示例；文档明确为**局部**算法 | https://github.com/stephane-caron/pink |
| **mink** | Python | MuJoCo 上的 pink 移植；QP 微分 IK | 关节/速度极限、**任意 geom 对碰撞规避**、**闭链等式约束**（loop closure） | https://github.com/kevinzakka/mink |
| **placo** | C++/Python | Pinocchio + eiquadprog 的 QP 全身 IK/ID | `hard`/`soft` + 权重的任务优先级；有 `RelativeFrameTask`/`RelativePositionTask`（相对位姿）、`AvoidSelfCollisionsKinematicsConstraint`、`ManipulabilityTask`、`JointsTask` | https://github.com/rhoban/placo |
| **Drake** | C++/Python | 建模/优化工具箱 | `InverseKinematics` 以整个 `MultibodyPlant` 为变量，可加多个末端位置/姿态约束、`AddMinimumDistanceLowerBoundConstraint`（碰撞）、对子集用 `Joint::Lock`；`GlobalInverseKinematics`（MIP，可证不可行）；`KinematicTrajectoryOptimization`；GCS | https://drake.mit.edu/doxygen_cxx/classdrake_1_1multibody_1_1_inverse_kinematics.html |
| **cuRobo** | Python/CUDA | GPU 并行 IK、无碰撞 IK、运动生成、MPC | 机器人配置为整棵树（含腰）；`ee_link` + `link_names` 列出两只手，`plan_single(..., link_poses={...})` 同时约束两末端 | https://nvlabs.github.io/curobo/latest/getting-started/inverse_kinematics.html |
| **PyRoki** | Python（JAX） | 运动学优化工具箱（IK/轨迹优化/重定向） | 代价可组合，变量可扩展（论文中把移动底座位姿作为变量）；CPU/GPU/TPU | https://github.com/chungmin99/pyroki |
| **Tesseract / TrajOpt** | C++/Python | 工业运动规划框架 + TrajOpt SQP | 多约束集（含 IK 约束、连续碰撞） | https://github.com/tesseract-robotics/trajopt |
| **PDDLStream** | Python | TAMP 框架（PDDL + streams） | IK 作为 stream；有 PR2 双臂 PyBullet 示例与 Drake 示例 | https://github.com/caelan/pddlstream |
| **VAMP** | C++/Python | SIMD 加速采样规划 | 快速碰撞检测与 RRT-Connect；有 OMPL 集成 | https://github.com/KavrakiLab/vamp |
| RelaxedIK / RangedIK | Rust | 实时优化 IK（平滑、自碰撞、奇异规避；RangedIK 支持区间目标） | 偏实时运动生成，TAMP 中较少用 | https://github.com/uwgraphics/relaxed_ik_core |
| IKFlow / IKDiffuser | Python | 学习型 IK | 作为种子生成器；IKDiffuser 直接支持运动学树（含「双臂+腰」平台） | https://github.com/jstmn/ikflow ，https://arxiv.org/abs/2506.13087 |

---

## 5. 「共享腰部」的具体处理策略

### 5.1 IK 层面（三种模式，从简单到完整）
| 模式 | 做法 | 何时用 |
|---|---|---|
| A. 腰部先定、两臂分解 | 外层采样/搜索腰角 `q_w`（或按启发式：让腰朝向两目标中点），内层对每臂用解析/TRAC-IK 单链求解 | 求解器只支持单链（TRAC-IK/IKFast）；需要枚举多解做 TAMP 候选 |
| B. 整体联合 IK（推荐默认） | 在 `[q_w, q_L, q_R]` 上一次求解：两末端位姿任务 + 腰部正则（高代价/低权重） + 关节居中 + 碰撞约束 | Drake IK、Pinocchio+pink/placo、mink、cuRobo、PyRoki、pick_ik/bio_ik |
| C. 严格优先级 | 主臂/闭链约束为硬约束，次臂为软任务，腰部姿态为最低优先级正则；或零空间投影 `q̇ = J₁⁺ẋ₁ + N₁ J₂⁺(…)` | 当两只手的目标冲突（不可同时到达）时，需要明确「谁让步」 |

实践要点：
- **种子**：给腰部一个中立种子，并对腰部使用较大的正则权重；否则数值优化常倾向于「大幅扭腰」以降低末端误差（MODAP 关于 IK 分支/种子的经验同样适用）。
- **多解多样性**：TAMP 需要多个不同的 IK 解（不同肘部/腰部构型）供下游运动规划挑选。可用多随机种子（cuRobo `num_seeds`）、TRAC-IK `Distance` 模式、或学习型生成器（IKDiffuser）作为种子。
- **可达性**：先用 GPU 批量 IK 做可达性/抓取候选筛选（cuRobo 文档示例：用批量 IK 生成可达性地图）。

### 5.2 规划层面：腰部关节如何对待
| 策略 | 描述 | 优缺点 |
|---|---|---|
| 联合规划 | 在 16 维 `[q_w, q_L, q_R]` 上用 OMPL / cuRobo / TrajOpt 直接规划 | 最完整、能利用腰扩展工作空间；维度高、采样规划更慢 |
| 分阶段（sequential） | 先只动腰（双臂锁定在安全姿态）到合适朝向 → 再规划双臂（锁腰） | 简单稳健，维度降为 14/1；会损失「边转腰边伸手」的效率 |
| 主从/优先级 | 腰 + 一只臂为主动，另一只臂锁定或跟随 | 适合「一手固定物体、一手操作」的非对称任务 |
| 闭链（双手持物） | 自由变量 = 腰 + leader 臂；follower 臂由 IK 闭合相对位姿（Cohn 2023 / Paro 2026 的思路） | 约束满足最严格；follower 奇异附近需特殊处理（Paro et al. 的拒绝采样 + 加权度量） |

在 MoveIt 2 中的建议组定义（SRDF）：`torso`、`left_arm`、`right_arm`（单链，用于单臂 IK/规划，可选择是否含腰）、`torso_left_arm` / `torso_right_arm`（base→手，单链，TRAC-IK 可用）、`both_arms`、`torso_both_arms`（多链树，只能用多 tip IK 插件或关节目标）。注意：在 `torso_left_arm` 中解出的腰角会移动右臂——必须在**完整 RobotState** 下做碰撞检查，并据此决定右臂是否需要重新求解。

---

## 6. 近期研究（2023–2026，已核实）

| 工作 | 要点 | 链接 |
|---|---|---|
| cuRobo（NVIDIA, 2023） | GPU 并行 IK / 无碰撞 IK / 运动生成；37000 IK/s、7600 无碰撞 IK/s（RTX 4090）；规划平均约 50 ms | https://arxiv.org/abs/2310.17274 |
| Cohn et al., *Constrained Bimanual Planning with Analytic IK*（ICRA 2024） | 解析 IK 参数化闭链流形 → RRT/PRM/TrajOpt/GCS 可直接用 | https://arxiv.org/abs/2309.08770 |
| Paro et al., *Fast Coordinated Bimanual Motion Planning With Hard Constraints*（2026） | 无需解析 IK、无需离线预计算的 leader–follower + 约束感知插值；比 IK-BiRRT 快 19.4× | https://arxiv.org/abs/2608.20946 |
| MODAP（Gao et al., 2024） | 双臂桌面重排 TAMP：任务层采样 IK + cuRobo + TOPP-RA；比基线执行时间最多快约 40% | https://arxiv.org/abs/2404.06758 |
| SDAR（Zhang, Huang, Yu, 2025） | 依赖图任务规划 + GPU 批量 IK 选子任务 + cuRobo 同步双臂规划 + 回退策略；论文报告 100% 成功率（基线 85%），双 UR5e 实机 | https://arxiv.org/abs/2512.08206 |
| PyRoki（Kim, Yi et al., 2025） | JAX 运动学优化工具箱；IK-Beam 比 cuRobo 快 1.4–1.7×（Panda 基准） | https://arxiv.org/abs/2505.03728 |
| IKDiffuser（Zhang, Jiao, 2025） | 扩散式 IK，支持运动学树（含双臂+腰平台），作为优化器种子显著提高成功率 | https://arxiv.org/abs/2506.13087 |
| VAMP（Thomason, Kingston, Kavraki） | SIMD 向量化采样规划，微秒级 | https://github.com/KavrakiLab/vamp |
| GCS（Marcucci et al., Science Robotics 2023） | 凸集图上的运动规划，Drake 已实现 | https://www.science.org/doi/10.1126/scirobotics.adf7843 |
| IKFlow（Ames et al., RA-L 2022，较早但常作基线） | 归一化流生成多样 IK 解 | https://arxiv.org/abs/2111.08933 |

经典参考：TRAC-IK（Humanoids 2015）、PDDLStream（ICAPS 2020）、Kingston et al. 约束规划综述（2018）、Berenson et al. TSR/CBiRRT、Drake GlobalIK（Dai, Izatt, Tedrake, IJRR 2019）。

---

## 7. 推荐方案：双臂 + 可动腰的任务与运动规划（含碰撞规避与双臂协调约束）

### 7.1 推荐栈 A（研究/原型优先，Python）：**Drake 为主干 + PDDLStream 任务层 + cuRobo 批量 IK（可选）**
- **模型**：URDF/SDF 中保留腰部关节，`MultibodyPlant` 为整棵树。
- **IK（目标/抓取采样）**：Drake `InverseKinematics` 一次求解 `[q_w, q_L, q_R]`：两末端位置/姿态约束（可用区间表达 TSR）、`AddMinimumDistanceLowerBoundConstraint`（自碰撞+环境）、腰部二次代价（偏好中立）、关节居中代价；多随机初值（或用 IKDiffuser/cuRobo 批量结果做初值）获得多样解。不可行性判断可选 `GlobalInverseKinematics`。
- **双手持物**：Cohn et al. 的参数化（若有解析 IK，如 iiwa/Panda），或 Paro et al. 的数值 IK leader–follower 思路：自由变量 = 腰 + leader 臂；follower 用 Drake IK/Pinocchio 数值 IK（以前一状态为种子）闭合；用 OMPL RRT-Connect 规划，再用 `KinematicTrajectoryOptimization` 平滑（加相对位姿约束），最后 TOPP-RA 时间参数化并**沿轨迹校验相对位姿残差**。
- **自由运动（非持物）**：OMPL/VAMP 采样规划或 GCS（环境静态、可离线建区域时）→ 轨迹优化平滑。
- **任务层**：PDDLStream，stream 包括 `sample-grasp`、`ik(腰+双臂)`、`plan-free-motion`、`plan-closed-chain-motion`、`test-cfree`；PDDLStream 自带 Drake 示例可作为起点。

### 7.2 推荐栈 B（工程落地/ROS 2）：**MoveIt 2 + MTC + OMPL + 多 tip IK 插件 + cuRobo（可选加速）**
- **SRDF 组**：`torso`、`left_arm`、`right_arm`、`torso_left_arm`、`torso_right_arm`、`torso_both_arms`。
- **IK 插件**：单链组用 **TRAC-IK**（`Distance`/`Manip*` 模式）；`torso_both_arms` 用 **pick_ik**（`mode: global`）或 **bio_ik**（多 tip）。注意 KDL/LMA 无法用于多链组。
- **任务管道**：MTC——`CurrentState` → `GenerateGraspPose` → `ComputeIK`（每臂一分支，`Merger` 合并；或对 `torso_both_arms` 给关节目标）→ `Connect`（OMPL）→ 附着物体后继续。
- **闭链搬运**：MoveIt 2 的 OMPL 约束规划只支持单个位置/姿态约束，**不支持两末端相对位姿**；需自定义 OMPL `Constraint` 或在外部（栈 A 的方法）规划后交给 MoveIt 执行。
- **轨迹质量**：OMPL 初解 → CHOMP/STOMP 优化；或换用 Tesseract TrajOpt（`CartPosConstraint`、连续碰撞）。
- **加速**：若有 NVIDIA GPU，用 cuRobo 做批量无碰撞 IK（抓取候选筛选、可达性分析）和同步双臂运动生成（`link_names` 两只手）；注意 cuRobo 双臂规划是**同步到达**的，异步任务需按 MODAP/SDAR 的方式拆分子任务。

### 7.3 腰部处理的明确建议
1. **IK 一律在整棵树上联合求解（模式 B）**，腰部加正则/低权重；只有在用单链求解器（TRAC-IK/IKFast）时才用「外层采样腰角 + 内层单臂 IK」（模式 A）。
2. **规划默认联合规划腰+双臂**；若采样规划太慢或腰部需要大幅转动（换工位），采用**分阶段**：先单独规划腰部（双臂收拢锁定），再在固定腰下规划双臂；或在 TAMP 层把「转腰」作为一个独立动作。
3. **双手持物**：自由变量 = 腰 + leader 臂，follower 臂由 IK 闭合；leader 选在更拥挤一侧（Paro et al. 的建议：leader 自由采样、碰撞检测比 IK 便宜）。
4. **碰撞检查必须在完整状态下进行**：任何单臂 IK 解出的腰角都会移动另一只臂。
5. **种子与多样性**：TAMP 中每个抓取候选求多个 IK 解（不同腰/肘构型），由下游运动规划择优；GPU 批量或学习型种子可显著提高高维树结构的成功率（IKDiffuser 在「双臂+腰」平台上的结果）。

### 7.4 一句话选型
- **追求研究灵活性与约束表达能力** → Drake（IK + 轨迹优化 + GCS）+ OMPL + PDDLStream，底层可用 Pinocchio/placo 做微分 IK 精修。
- **追求 ROS 2 工程集成** → MoveIt 2 + MTC + pick_ik/bio_ik（腰+双臂组）+ TRAC-IK（单链组），闭链搬运单独实现。
- **追求速度/大批量候选** → cuRobo（或 PyRoki）做批量无碰撞 IK 与运动生成，作为上述任一栈的加速模块。

---

## 8. 参考链接（本次实际抓取/检索验证）
- TRAC-IK: https://github.com/traclabs/trac_ik ；论文 http://irl.cs.brown.edu/pubs/trac-ik.pdf
- KDL: https://github.com/orocos/orocos_kinematics_dynamics
- IKFast: https://www.openrave.org/docs/latest_stable/openravepy/ikfast/
- MoveIt 2 双臂: https://moveit.picknik.ai/main/doc/examples/dual_arms/dual_arms_tutorial.html
- MoveIt 2 OMPL 约束规划: https://moveit.picknik.ai/main/doc/how_to_guides/using_ompl_constrained_planning/ompl_constrained_planning.html
- MoveIt 相对 TCP IK 讨论: https://github.com/moveit/moveit2/issues/3344 ；"not a chain": https://github.com/moveit/moveit2/issues/1633
- MTC: https://moveit.github.io/moveit_task_constructor/ ；双臂 issue: https://github.com/moveit/moveit_task_constructor/issues/390
- pick_ik: https://github.com/PickNikRobotics/pick_ik ；bio_ik: https://github.com/TAMS-Group/bio_ik
- MoveIt CHOMP/STOMP/Pilz: https://moveit.picknik.ai/main/doc/how_to_guides/chomp_planner/chomp_planner_tutorial.html
- OMPL 约束规划: https://ompl.kavrakilab.org/core/constrainedPlanning.html
- Pinocchio: https://github.com/stack-of-tasks/pinocchio ；pink: https://github.com/stephane-caron/pink ；mink: https://github.com/kevinzakka/mink ；placo: https://github.com/rhoban/placo
- Drake IK: https://drake.mit.edu/doxygen_cxx/classdrake_1_1multibody_1_1_inverse_kinematics.html ；GlobalIK: https://drake.mit.edu/doxygen_cxx/classdrake_1_1multibody_1_1_global_inverse_kinematics.html ；KinematicTrajectoryOptimization: https://drake.mit.edu/doxygen_cxx/classdrake_1_1planning_1_1trajectory__optimization_1_1_kinematic_trajectory_optimization.html
- cuRobo: https://arxiv.org/abs/2310.17274 ；文档 https://nvlabs.github.io/curobo/latest/getting-started/inverse_kinematics.html ；双臂讨论 https://github.com/NVlabs/curobo/discussions/209
- PyRoki: https://github.com/chungmin99/pyroki ；https://arxiv.org/abs/2505.03728
- Tesseract TrajOpt: https://github.com/tesseract-robotics/trajopt
- PDDLStream: https://github.com/caelan/pddlstream
- VAMP: https://github.com/KavrakiLab/vamp
- GCS: https://www.science.org/doi/10.1126/scirobotics.adf7843
- TSR/CBiRRT: https://publications.ri.cmu.edu/task-space-regions-a-framework-for-pose-constrained-manipulation-planning
- Cohn et al. 2023: https://arxiv.org/abs/2309.08770 ；Paro et al. 2026: https://arxiv.org/abs/2608.20946
- MODAP: https://arxiv.org/abs/2404.06758 ；SDAR: https://arxiv.org/abs/2512.08206
- IKDiffuser: https://arxiv.org/abs/2506.13087 ；IKFlow: https://arxiv.org/abs/2111.08933
- RelaxedIK/RangedIK: https://github.com/uwgraphics/relaxed_ik_core
- 相对雅可比（双臂协同任务空间）: https://ar5iv.labs.arxiv.org/html/1905.01248

附：遥操作方向（已按要求弱化，仅供参考）——Unitree xr_teleoperate 用 Pinocchio+CasADi(IPOPT) 双臂 IK（https://github.com/unitreerobotics/xr_teleoperate ），Fourier F.A.R.T.S. 依赖 pin-pink 与 dex-retargeting（https://github.com/FFTAI/teleoperation ），Open-TeleVision（https://arxiv.org/abs/2407.01512 ）。
