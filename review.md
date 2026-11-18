这份交接与技术架构说明文档严格基于 single_arm_ik.py 与 ik_base.py 的真实执行逻辑整理，剥离了过时注释的干扰，并对代码中的隐藏陷阱、数学公式、参数默认值物理意义及迭代趋势影响进行了深度拆解。
1. 系统定位与模块架构概览
本套件是面向人形机器人腰臂协同（腰 2 自由度 + 单臂 7 自由度 = 9 DoF）与纯单臂（7 DoF）的高内聚数值逆运动学（IK）求解器。
1.1 双文件职责划分
 * ik_base.py（底层基础设施与外层裁判）：
   * 轻量级运动学链 (SerialChain)：纯 Python/NumPy 实现的 URDF 解析、正运动学（FK）、几何雅可比与 SEW 臂角（Shoulder-Elbow-Wrist Angle, \psi） 计算。
   * 三维几何与误差度量：提供三种末端姿态模式（full 6D 位姿、tool_dir 5D 工具轴指向、tool_cone 5D 锥角约束）的残差计算与多圈关节 2\pi k 等价分支对齐。
   * 外层候选解打分与排序 (score_candidate, select_best_candidate)：对内层收集上来的候选解进行物理/几何打分与 8 级字典序择优。
 * single_arm_ik.py（核心数值引擎与多阶段调度器）：
   * Pinocchio 刚体动力学集成：负责全机质量审计、精确静态重力矩 \tau_g 及其解析导数 \frac{\partial \tau_g}{\partial q}、整机水平质心（CoM）及雅可比计算。
   * 单种子数值迭代核心 (pinocchio_ik_one_seed)：基于加权阻尼最小二乘（W-DLS）+ 饱和零空间主动集钳制（SNS）+ 零空间多目标梯度下降的迭代求解器。
   * 五阶段多候选解搜索流水线 (collect_pinocchio_candidates_and_seeds)：结合任务空间种子库、腰部舒适窗、局部续算与多核并发（multiprocessing），在时间预算内搜索全局最优解。
   * 备用历史引擎：包含 SSIK（解析/半解析求解器包装）与 Direct-Psi（基于 scipy.optimize.least_squares 的固定臂角优化），默认关闭（USE_SSIK = False, USE_DIRECT_PSI = False），且在 9-DoF 含腰模式下强制禁用。
2. 接手必读：代码中的“注释陷阱”与真实逻辑偏差
在阅读或修改代码前，必须注意以下 6 处代码实际行为与注释/直觉严重不符的细节：
 * 内外层“腰部回正”权重的巨大倒挂与 sync_secondary_cost 补丁：
   * 现象：内层 DLS 零空间迭代中，腰部回中增益 WAIST_NULL_GAIN = 0.50（仅比手臂 NULLSPACE_GAIN = 0.30 略大）；但在外层 ik_base.py 的 score_candidate 打分时，腰部偏离惩罚权重高达 W_WAIST_RETURN = 2000.0（对比：手臂臂角权重 W_NOM = 50.0，平滑权重 W_SMOOTH = 8.0）。
   * 后果：如果 sync_secondary_cost = False（默认），内层迭代为了降低重力矩或避障会轻微动腰，但外层裁判会因为 2000.0 的极高权重直接枪毙所有动腰的解，选出腰部死锁但手臂姿态极扭曲的解。当开启 sync_secondary_cost = True 时，代码第 1572 行有一行硬编码补丁：params["waist_null_gain"] = 2.0，强行把内层腰部回中增益放大 4 倍以对齐外层偏好。
 * band_bounds 参数是未完工的死代码（Dead Code）：
   * 在 SingleArmIKSolver.pinocchio_ik_one_seed（第 2641 行）中传入了 band_bounds=(self.search_lower, self.search_upper)，一路传给了 _SecondaryObjective 的 band_lower 和 band_upper（第 846 行）。但查看 _SecondaryObjective.__init__ 内部，这两个参数根本没有被保存或使用。
 * W_LIMIT 的双重定义与隐式平方放大：
   * 外层打分常量 W_LIMIT = 10.0。但在 joint_limit_penalty 中，越界量已经除以 SAFE_MARGIN (0.02) 并平方：(\frac{0.02 - \text{margin}}{0.02})^2。这意味着当某个关节贴近限位（\text{margin}=0）时，单位弧度的等效刚度其实是 10.0 / 0.02^2 = 25000\text{ rad}^{-2}，远比表面看到的 10.0 剧烈得多。
 * fork 多进程并发池的平台兼容性隐患：
   * _ensure_parallel_pool（第 2070 行）硬编码了 mp.get_context("fork") 来共享 C++ pin_model 内存。在 Ubuntu Linux 上运行正常，但在 macOS 或 Windows 上使用多线程+fork 极易引发死锁或崩溃；此外，若外部挂载的 collision_manager 包含多线程锁或 GPU 上下文，fork 子进程会直接卡死。
 * S_JOINT / E_JOINT / W_JOINT 顶层常量的左臂硬编码：
   * ik_base.py 顶层定义了 S_JOINT = "LHand_Link2" 等左臂关节名作为函数默认参数。虽然 SingleArmIKSolver 初始化时会从 ARM_CONFIGS[side] 正确读取右臂关节名（RHand_Link2 等）注入 SerialChain，但如果你在外部直接调用 dpsi_dq() 或 sew_angle_from_points() 而不显式传参，右臂就会错误地去查左臂关节。
 * Pinocchio 与 SerialChain 的双轨 FK 校验：
   * 内层 DLS 迭代使用 C++ Pinocchio 算 FK，而候选解入池打分（score_candidate）使用纯 Python SerialChain 算 FK。在 strict 模式下，_check_load_candidate 又会用 Pinocchio 的 dls_task_terms 再复核一次位姿误差。两者在浮点精度上通常一致，但涉及到 base_link 与 urdf_root_link 坐标变换（_pin_uses_root_frame）时，务必保证两边的齐次变换矩阵严格同步。
3. 全局架构与求解流程全景
当你调用 solver.solve(T_target, ...) 时，数据在系统内部经历以下 4 个核心大阶段：
[输入 T_target, q_prev, joint_state]
       │
       ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 阶段 A：前置守卫与上下文冻结 (solve & _solve_impl)                      │
│  1. 获取非阻塞互斥锁 _solve_lock（保护 Pinocchio C++ Data 内存）        │
│  2. strict 模式：校验全身关节快照新鲜度 (<1.0s)、URDF 质量审计、冻结模板 │
│  3. 坐标系对齐 (local <-> world) 与三角不等式绝对臂展预检 (<0.001ms)    │
└─────────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 阶段 B：五阶段多种子 DLS 搜索流水线 (collect_pinocchio_candidates...)   │
│  判定工况：计算 FK(q_prev) 到目标的距离 -> 区分 [续算] vs [重新定位]    │
│   ├─ 1. continuation (局部续算)：仅小位移触发，在 q_prev±0.15rad 内快搜 │
│   ├─ 2. comfort (腰部舒适窗)：重定位触发，腰部限制在重力/朝向舒适窗内   │
│   ├─ 3. full (全局有效限位)：舒适窗无解或提前失去耐心时，放开限位回退   │
│   ├─ 4. unboxed (无框重搜)：跳变框 (<π) 内无解时，放开跳变框探底        │
│   └─ 5. jump_box (跳变框救援)：仅因跳变超限被拒时，在 q_prev±jump 内重搜│
│       │                                                                 │
│       └──> 每个种子调用【单种子数值引擎 pinocchio_ik_one_seed】：       │
│            主任务残差 e & 雅可比 A -> 无状态腰部释放逆度量 D ->         │
│            零空间复合负梯度 z -> SNS 主动集限位钳制 -> 回溯线搜索       │
└─────────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 阶段 C：硬约束过滤门控 (_finalize_result)                               │
│  1. H4 关节跳变过滤 (_apply_joint_jump_policy)：剔除单关节绕大圈候选    │
│  2. H1-H3 & 避障终检 (_filter_load_candidates)：                        │
│     复核位姿误差 (≤5e-4)、物理/虚拟限位、净空安全裕度、静态重力矩不超载 │
│  3. S1 载荷峰值容差带筛选：仅保留峰值负载率在 [min_peak, min_peak+0.02] │
└─────────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 阶段 D：软目标打分、8级排序与结果封装 (select_best_candidate)           │
│  1. 计算静态重力矩代价 W_TORQUE * mean(ratio^2) 并合入总代价 cost       │
│  2. 按 8 级字典序排列选出 best 解（及可选的 solutions 备选解列表）      │
│  3. 终检复核通过后更新 self.q_prev，打包返回 (q, Status, IKResult)      │
└─────────────────────────────────────────────────────────────────────────┘

4. 约束体系与四道硬门控（Hard Constraints）
求解器对“什么是合法的解”有极其严苛的分层定义。任何不满足硬约束的候选解（除非显式开启 allow_approximate=True）都会被直接丢弃：
4.1 硬约束清单
 * H1 末端位姿容差：
   * 位置欧氏误差 \Vert{}\Delta p\Vert{}_2 \le \text{POS\_TOL} (5 \times 10^{-4}\text{ m} = 0.5\text{ mm})。
   * 姿态违背量 \text{ori\_violation} \le \text{ORI\_TOL} (5 \times 10^{-4}\text{ rad} \approx 0.0286^\circ)。
 * H2 关节极限与安全裕度：
   * 关节角必须落在有效限位内：q \in [\text{lower}_{\text{eff}}, \text{upper}_{\text{eff}}]，其中有效限位为 URDF 物理限位、配置覆盖限位 position_bounds 与围绕 q_ready 的虚拟半窗 virtual_half_windows 的交集：
     
   * 在搜索阶段判定“可接受解（acceptable）”时，还要求所有可动关节距离有效限位边界的最小余量 \text{margin} = \min_i \min(q_i - l_i, u_i - q_i) \ge \text{SAFE\_MARGIN} (0.02\text{ rad} \approx 1.15^\circ)。若最终只能选出 \text{margin} < 0.02\text{ rad} 的解，状态会降级为 Status.NEAR_LIMIT。
 * H3 静态重力矩不超载（仅 load_mode="strict" 生效）：
   * 所有受守护关节（guard_joint_names，默认全模型所有关节）在当前构型下的静态重力矩绝对值不得超过实测标定的容许力矩：
     
 * H4 单关节防绕大圈跳变限制（joint_jump_policy）：
   * 候选解对齐到最接近上一帧 q_prev 的合法 2\pi k 周期分支后，单关节相对 q_prev 的最大位移不得超过阈值 max_joint_jump_rad（默认 \pi\text{ rad} = 180^\circ）：
     
 * H5 环境防碰撞净空硬门槛（当挂载 collision_manager 时生效）：
   * 不论 load_mode 是 "strict" 还是 "off"，只要挂载了碰撞管理器，最终解的连杆最小障碍物净空距离必须满足 d_{\min}(q) \ge \text{safety\_margin}。
4.2 为什么 strict 模式设计得如此严苛？
在 9-DoF 腰臂协同系统中，腰部俯仰关节（Pitch）一旦前倾，整个上半身（双臂+躯干+头部）的重心会产生巨大的倾覆力矩。如果非活跃关节（例如另一只手当前伸在前面还是缩在胸前）的状态未知，算出来的腰部重力矩就是完全错误的。因此 prepare_load_context 强制要求：
 * 必须传入新鲜的全身关节状态快照 JointStateSnapshot（默认超时阈值 load_state_max_age_s = 1.0s）；
 * 启动时 configure_model_metadata 会审计 URDF 所有连杆的质量与惯性张量特征值（三角不等式 \lambda_{\max} \le \lambda_1 + \lambda_2），任何 0 质量连杆若未在 load_massless_links 白名单中显式声明，直接阻断求解；
 * 求解完成准备返回前，还会调用 _stale_context_failure 检查求解耗时期间外部非活跃关节是否发生了移动（旋转关节容差 0.01 rad，移动关节 0.001 m），若移动则报 STALE_OR_CHANGED_CONTEXT 作废本次解。
5. 核心数学公式全解
5.1 主任务残差 e(q) 与解析雅可比 A(q)
代码在 _make_task_terms 中将位置误差除以特征长度 L_c（characteristic_length = 0.5 m），使位置残差无量纲化（0.5\text{ m} 的位置误差等价于 1\text{ rad} 的角度误差），从而与姿态残差同量纲。
设当前末端位姿为 T_{\text{cur}} = (R_c, p_c)，目标位姿为 T_{\text{tar}} = (R_t, p_t)，位置误差向量为 \Delta p = p_t - p_c。
模式 1：full（6D 完整位姿控制）
使用 Pinocchio 在末端局部坐标系（pin.LOCAL）下的李代数对数映射 \log_6：


无量纲化后的 6D 任务残差 e 与伴随修正后的任务雅可比 A 为：


注：引入 J_{\log_6}(\Delta T^{-1}) 是为了补偿李群曲率，使高斯-牛顿步在大角度偏差下依然指向真实的测地线方向。
模式 2：tool_dir（5D 位置 + 工具轴方向对齐，允许绕工具轴自转）
在世界对齐系（pin.LOCAL_WORLD_ALIGNED）下，设工具轴单位向量在当前系与目标系下的朝向分别为 a = R_c u_{\text{axis}} 与 b = R_t u_{\text{axis}}，两者夹角 \theta = \text{atan2}(\|a \times b\|_2, a^\top b)：
 * 常规情况（a^\top b \ge -1 + 10^{-8}）：
   
   
   其中 [a]_\times 为反对称矩阵，因为 \dot{a} = \omega \times a = -[a]_\times \omega = -[a]_\times J_\omega \dot{q}。
 * 180° 对径奇异死点逃逸（a^\top b < -1 + 10^{-8}）：
   此时 b = -a，方向残差 b - a = -2a 与雅可比行向量 -[a]_\times J_\omega 的列空间完全正交（(-[a]_\times J_\omega)^\top (-2a) = 0），常规梯度直接归零锁死！
   代码通过 _antipodal_tangent 对矩阵 M = J_{\text{dir}} - a(a^\top J_{\text{dir}}) 做 SVD 分解，提取最大奇异值对应的左奇异向量作为当前构型最容易转动的单位切向量 t \perp a，将残差替换为 e_{\text{dir}} = \theta \cdot t，从而平滑逃离 180° 死点。
模式 3：tool_cone（4D 位置 + 工具轴允许在半角 \theta_{\text{cone}} 的圆锥内自由摆动）
定义锥角违背量 v_{\text{cone}} = \max(0, \theta - \theta_{\text{cone}}) \in \mathbb{R}：


当 v_{\text{cone}} = 0（工具轴已在锥内）时，j_{\text{cone}} = 0，姿态完全释放不消耗自由度；当 v_{\text{cone}} > 0 时，令单位转轴 t = \frac{a \times b}{\|a \times b\|_2}，则 j_{\text{cone}}^\top = t^\top J_\omega。
5.2 无状态腰部自适应释放（逆度量矩阵 D）
为了实现“近处目标只动臂、远处目标或重载时才动腰，且去程与返程完全对称无状态”，代码定义了关节空间的对角逆度量矩阵 D = \text{diag}(d_1, \dots, d_n)（d_i 越小，代表该关节在最小二乘中越“昂贵”、越不愿意动）。
 * 手臂关节：d_i = 1.0。
 * 腰部关节：在近距离权重 w_{\text{near}} = 6.0（WAIST_TASK_WEIGHT）与远距离权重 w_{\text{far}} = 1.0（WAIST_TASK_WEIGHT_FAR）之间按释放系数 \alpha \in [0, 1] 插值：
   
 * 释放系数 \alpha 的三路取大机制：
   * 距离释放函数 f_{\text{rel}}(d)：设肩关节到目标的距离为 d，最大臂展为 R_{\max}，在 r_0 = 0.75（WAIST_RELEASE_START）到 r_1 = 0.94（WAIST_RELEASE_END）之间做三次 Hermite 平滑阶跃（Smoothstep）：
     
   * 返程解锁通道 r_{\text{ret}}：当机械臂从远点往近点退回时，虽然目标距离 d 已经变小（f_{\text{rel}}(d) \to 0），但此时腰部还停留在偏离 q_{\text{ref}} 的位置。如果立刻把腰锁死（d_{\text{waist}} = 1/6），腰部就退不回来了！因此引入返程解锁项（尺度 \theta_{\text{ret}} = 0.20\text{ rad}，WAIST_RETURN_RELEASE_RAD）：
     
   * 综合释放系数（若锁腰种子因重力矩超标触发了 waist_load_release=True，则直接令 \alpha = 1.0）：
     
5.3 饱和零空间（SNS）主动集钳制与度量一致投影
在每一轮迭代中，_sns_clamped_core 负责求解同时满足主任务、零空间次任务、关节限位与信赖域步长上限（\|\Delta q\|_2 \le \Delta q_{\max} = 0.5\text{ rad}）的步长 \Delta q。
设当前未被钳制在边界上的自由关节集合为 f，已被钳制在限位边界上的饱和关节位移向量为 \Delta q_{\text{clamped}}：
 * 扣除已钳制关节造成的残差偏移：
   
 * 构造加权阻尼海森矩阵 M_f（含随残差自适应增大的阻尼 \lambda_{\text{eff}}）：
   
   
   其中基础阻尼 \lambda_0 = 10^{-6}（PIN_IK_DAMPING）。
 * 主任务步与度量一致零空间步（通过 LAPACK dpotrf/dpotrs 进行上三角 Cholesky 分解求解，若非正定则回退至 _metric_svd_step）：
   
   
   为什么零空间投影算子写成 P_f = D_f - D_f A_f^\top M_f^{-1} A_f D_f？
   因为 P_f 是对称半正定矩阵！对于次任务负梯度 z = -\nabla H(q)，必然满足 z_f^\top \Delta q_{\text{null}, f} = w_{\text{null}} z_f^\top P_f z_f \ge 0，从代数上严格保证了零空间步永远是次任务代价 H(q) 的下降方向。
 * 零空间权重自适应衰减 w_{\text{null}}：
   
   
   其中衰减尺度 s_e = 0.06（NULLSPACE_ERR_SCALE）。当末端离目标很远（\|e\|_2 \gg 0.06）时，w_{\text{null}} \to 0，全力追踪主任务，防止零空间梯度干扰主任务收敛；当末端接近目标（\|e\|_2 \ll 0.06）时，w_{\text{null}} \to 1.0 \sim 1.5，全力在零空间内优化姿态与卸载力矩。
 * 正交信赖域截断与单关节逐个钳制循环：
   * 由于 \Delta q_{\text{clamped}} 位于已饱和坐标轴，而 \Delta q_f = \Delta q_{\text{task}, f} + \Delta q_{\text{null}, f} 位于自由坐标轴，两者正交：\|\Delta q_{\text{total}}\|_2^2 = \|\Delta q_{\text{clamped}}\|_2^2 + \|\Delta q_f\|_2^2。
   * 因此自由关节的剩余可用步长半径为 R_{\text{rem}} = \sqrt{\Delta q_{\max}^2 - \|\Delta q_{\text{clamped}}\|_2^2}。若 \|\Delta q_f\|_2 > R_{\text{rem}}，仅等比例缩放自由关节步长 \Delta q_f，绝不缩放已钳制在边界上的 \Delta q_{\text{clamped}}。
   * 检验试探点 q_{\text{trial}} = q + \Delta q_{\text{total}}：若仍有自由关节越界，挑出越界量最大的那 1 个关节 i^*，将其位移死死钉在边界上（\Delta q_{\text{clamped}}[i^*] = \text{bound}[i^*] - q[i^*]），移出自由集 f，进入下一轮内循环重算剩余关节。
5.4 零空间次任务复合代价函数 H(q) 与负梯度 z = -\nabla H(q)
在 _SecondaryObjective 中，次任务总代价 H(q) 由 5 个物理子项相加构成：
(1) 物理坐标回中项 H_{\text{posture}}
将关节拉向参考姿态 q_{\text{ref}}（默认 q_ready）：


其中手臂关节增益 k_{\text{ns},i} = 0.30（NULLSPACE_GAIN），腰部关节增益 k_{\text{ns},i} = 0.50（WAIST_NULL_GAIN）。
(2) 单侧安全裕度限位排斥项 H_{\text{limit}}
定义缓冲距 m = \max(\text{SAFE\_MARGIN}, 0.02) = 0.02\text{ rad}，下缓冲界 b_{\text{low}} = \min(q_{\text{ref}}, \text{lower} + m)，上缓冲界 b_{\text{high}} = \max(q_{\text{ref}}, \text{upper} - m)：


其中 JOINT_LIMIT_GAIN k_{\text{lim}} = 0.03。
精妙设计点：为什么用 \min(q_{\text{ref}}, \text{lower}+m)？因为如果参考姿态 q_{\text{ref}} 本身就贴近限位边界，不能让限位排斥力把处于 q_{\text{ref}} 的关节推离 q_{\text{ref}}。
(3) 静态重力矩卸载项 H_{\text{load}}（以 strict 模式 LoadPolicy 为例）
设守护关节数为 N_g，各关节静态重力矩为 \tau_i(q)，容许力矩为 C_i，归一化负载率为 r_i = \tau_i / C_i，软阈值为 s = 0.70（soft_ratio），超限部分为 e_i = \max(|r_i| - s, 0)：


利用 Pinocchio 的 computeGeneralizedGravityDerivatives 算出精确解析导数矩阵 \frac{\partial \tau}{\partial q} \in \mathbb{R}^{N_g \times n} 后，通过链式法则回传负梯度：


其中默认权重：k_{\text{abs}} = 0.04（或 0.05），k_{\text{exc}} = 0.50（超过 70% 负载率后惩罚力度暴增 10 倍以上），k_{\text{rel}} = 0.0。
(4) 整机质心（CoM）水平投影稳定项 H_{\text{com}}
包含根连杆固定质量 m_{\text{root}} 与可动子树质量 m_{\text{mov}} 的整机质心为：


设单位重力方向为 \hat{g}，水平面正交投影矩阵为 P_\perp = I_3 - \hat{g}\hat{g}^\top，则水平质心偏移代价为：


其中 COM_XY_GAIN k_{\text{com}} = 0.12（仅当 include_waist=True 时激活）。
5.5 腰部舒适窗的第一性原理正弦律推导 (derive_waist_comfort_bounds)
为什么腰部舒适窗不需要人工拍脑袋填角度，而是能传入 "auto" 自动算出来？代码基于刚体静力学第一性原理进行了极其优雅的推导：
 * 偏航轴（Yaw，转轴与重力向量夹角 < 30^\circ，即 |\text{axis} \cdot \hat{g}| \ge \cos 30^\circ）：
   转动偏航轴几乎不改变重力矩，但转角超过 90^\circ 会导致胸前相机和另一只手臂背离前方工作区。因此直接设为对称窗口：q_{\text{ready},i} \pm 90^\circ（WAIST_YAW_VIRTUAL_LIM）。
 * 俯仰/横滚轴（Pitch/Roll，重力敏感型关节）：
   当其余关节固定在 q_{\text{ready}}、仅单独转动某一个旋转腰关节 \theta = \theta_0 + \delta 时，其支撑的整个上半身子树绕该轴做刚体旋转，静态重力矩精确服从正弦律：
   
   
   代码只需在 \theta_0 和 \theta_0 + 90^\circ 两点各算一次重力矩 \tau_0 = \tau(\theta_0) 与 \tau_{90} = \tau(\theta_0 + \frac{\pi}{2})，即可直接恢复出正弦幅值（上半身放平时的最大重力矩）C_{\max} = \sqrt{\tau_0^2 + \tau_{90}^2} 与初始相位 x_0 = \text{atan2}(\tau_0, \tau_{90})（并在 \theta_0 + 45^\circ 处做一次精度校验）。
   随后从 \delta = 0 向正负两侧搜索，要求同时满足两条物理判据（s = 0.70 为 soft_ratio，A 为该关节容许力矩）：
   * 电机静态负荷不超软阈值：|\tau(\theta)| \le \max\big(s \cdot \min(C_{\max}, A), \; |\tau_0|\big)；
   * 上半身水平质心偏移受控：|\tau(\theta) - \tau_0| \le s \cdot C_{\max}（因为 \Delta \tau 正比于子树质心相对转轴的水平力臂变化量 \Delta x_{\text{com}}）。
     由此二分求解出的区间 [\theta_0 - \delta_{\text{lo}}, \theta_0 + \delta_{\text{hi}}] 即为该腰关节的物理舒适窗。
5.6 外层候选解综合评分与 8 级字典序择优 (ik_base.py)
当内层多个种子迭代收敛后，所有候选解进入外层 score_candidate 计算标量总代价 cost：
其中 \phi = \text{wrap\_to\_pi}(\psi(q) - \psi(q_{\text{ready}})) 为 SEW 肘部臂角偏离量。若开启了 sync_secondary_cost = True，还会额外加上 w_{\text{com}} H_{\text{com}} + w_{\text{col}} H_{\text{col}}。
最终 select_best_candidate 使用 candidate_sort_key 按 8 级优先级元组（从小到大） 选出最优解：
 * not pose_ok（位姿达标的排前面）
 * cost（综合加权代价最小）
 * phi_sort_value（臂角差 |\phi|，小于 PHI_TIE_TOL = 1e-5 视为 0 同档）
 * dq_prev（离上一帧 q_prev 欧氏距离最近）
 * -margin（最小关节限位余量最大）
 * dq_ready（离标准姿态 q_ready 最近）
 * pos_err（位置误差最小）
 * ori_err（姿态误差最小）
6. 核心超参数、默认值设置原因与迭代趋势影响表
理解每个参数为什么设这个默认值，以及调大/调小后会对迭代轨迹产生什么物理趋势影响，是调参排障的核心：
6.1 内层 DLS 数值与零空间参数 (single_arm_ik.py)
| 参数常量名 | 默认值 | 为什么这样设置（物理/数值原因） | 调大 / 调小对迭代过程的趋势影响 |
|---|---|---|---|
| PIN_IK_DAMPING (\lambda_0) | 1e-6 | 远离奇异点时保证 A D A^\top + \lambda I 严格正定可 Cholesky 分解，同时几乎不引入稳态位置偏差。 | 调大（如 1e-2）：奇异点附近步长更稳，但收敛速度显著变慢，易在达标前触发停滞检测；调小：接近奇异构型时条件数恶化，易触发 SVD 慢速回退。 |
| PIN_IK_MAX_STEP | 0.5 rad (~28.6°) | 雅可比矩阵只是非线性 FK 的一阶泰勒展开，超过 0.5 rad 后线性近似严重失真。 | 调大：大位移初期冲得快，但极易一步迈过头导致回溯线搜索频繁折半浪费算力；调小：轨迹极其细腻，但远距离目标需要更多迭代步数。 |
| PIN_IK_STALL_ITERS / RTOL | 12 步 / 2.5% | 锁腰种子遇到手臂够不到的远目标时，误差会卡在几厘米处原地蠕动。每 12 步检查若误差下降不足 2.5% 立即放弃该种子。 | 窗口调小/阈值调高：能极快淘汰死胡同种子，省下时间跑后续种子；但遇到狭长弯曲的误差峡谷（如避障绕行）可能误杀正在缓慢转弯的好种子。 |
| NULLSPACE_ERR_SCALE (s_e) | 0.06 | 对应无量纲化后约 3 cm 位置误差（0.06 \times 0.5\text{ m}）或 3.4° 姿态误差，作为“远场追目标”与“近场修姿态”的分水岭。 | 调大：主任务还没收敛时零空间力就过早介入，与主任务抢夺自由度，导致末端轨迹拐弯甚至不收敛；调小：只有末端完全到位后才开始修姿态。 |
| WAIST_TASK_WEIGHT / FAR | 6.0 / 1.0 | 近处目标让腰部动起来的代价是手臂的 6 倍，符合人类“够得着就不弯腰”的直觉；远处目标降为 1.0 与手臂平权协同。 | near 调大（如 20.0）：近距离绝对死锁腰部，但过渡段（0.75~0.94 臂展）释放时可能出现腰部响应迟滞；调小：稍微伸手腰就会跟着晃动。 |
| WAIST_RELEASE_START / END | 0.75 / 0.94 | 手臂伸展到 75% 臂展时肘部开始变直、操作度下降，此时平滑引入腰部；94% 时接近奇异边界，腰部完全释放。 | 区间前移（如 0.5~0.8）：机器人在中近距离就会积极用腰配合手臂，手臂肘部弯曲余量更大，但整机重心晃动变多。 |
| WAIST_RETURN_RELEASE_RAD | 0.20 rad (~11.5°) | 返程防卡死阈值：只要腰部偏离 q_ready 超过 11.5°，即使目标已回到近处，腰部依然保持释放状态以便零空间力将其拉回正位。 | 调得太小：腰部只要稍微偏一点就一直处于高灵活度状态；调得太大（或删掉该项）：远点抓完物体缩回胸前时，腰部会卡在半弯腰姿态回不来！ |
| NULLSPACE_GAIN / WAIST_NULL_GAIN | 0.30 / 0.50 | 零空间回中弹簧刚度。腰部设 0.50 比手臂 0.30 略强，优先保证躯干直立。 | 调大：姿态更接近 q_ready，但在姿态整理阶段（Settle）可能因步长过大导致线搜索无法保持位姿容差而提前终止。 |
| GRAVITY_TORQUE_GAIN / excess_weight | 0.04 / 0.50 | 正常负载（\le 70\%）下用 0.04 温和引导低力矩构型（如肘部下垂）；一旦超过 soft_ratio (0.70)，0.50 的超限刚度强力把关节推回安全区。 | GRAVITY_TORQUE_GAIN 调太大：会导致肘部拼命往重力方向掉，破坏标准臂角构型，甚至在回溯线搜索中与回中项打架震荡。 |
| CONTINUATION_MAX_STEP | 0.15 rad (~8.6°) | 伺服轨迹跟踪（续算模式）时，相邻两帧关节跳变不应超过 8.6°，且关闭姿态整理（CONTINUATION_POSTURE_SETTLE=False）实现零漂移死区保持。 | 调大：续算框变宽，更容易吸收剧烈的目标跳变，但也更容易在相邻路点间发生零空间自运动漂移。 |
6.2 外层候选解打分参数 (ik_base.py)
| 参数常量名 | 默认值 | 为什么这样设置（物理/工程原因） |
|---|---|---|
| W_WAIST_RETURN | 2000.0 | 最高优先级软约束：在满足位姿和负载的前提下，只要能不动腰就绝对不选动腰的分支（1° 的腰部偏离代价 \approx 0.61，相当于手臂关节移动 15.8^\circ 的平滑代价）。 |
| W_TORQUE | 500.0 | 次高优先级：强力压制高电机负载构型（均方负载率每增加 10%，代价增加 5.0）。 |
| W_NOM | 50.0 | 约束 7-DoF 手臂的冗余自由度（SEW 臂角 \psi），使其肘部朝向尽量贴合 q_ready 的自然下垂姿态。 |
| W_LIMIT | 10.0 | 配合 SAFE_MARGIN = 0.02 rad 惩罚贴近限位的候选解，防止选出卡在物理死角的解。 |
| W_SMOOTH | 8.0 | 在多个同族离散解之间，优先选择离上一帧 q_prev 欧氏距离更近的解，保证运动连续性。 |
7. 常见排障指南（Troubleshooting Dictionary）
当调用 q, status, res = solver.solve(...) 返回 q is None 时，直接查看 res.failure_reason、res.failure_detail 以及 solver.last_load_diagnostics：
 * 报错 LOAD_CONFIG_OR_STATE_INVALID（strict 模式最常见）：
   * 原因 A：未调用 solver.update_joint_state(msg) 喂入全身状态，或状态时间戳已超过 load_state_max_age_s（默认 1.0 秒），报 receipt_state_stale。
   * 原因 B：初始化 config 中未提供 load_allowable_efforts 字典（strict 模式绝不允许使用 URDF 默认力矩兜底）。
   * 原因 C：URDF 中存在质量为 0 的虚拟连杆（如 tcp_link、camera_optical_frame），未将其名称加入 config["load_massless_links"] 白名单。
   * 调试临时绕过：若在纯运动学仿真环境，初始化时传入 config={"load_mode": "off"} 即可关闭静态载荷门控。
 * 报错 OUTSIDE_WORKSPACE / WORKSPACE_REJECTED：
   * 原因：目标位置在 local 系下的模长超过了运动链所有连杆平移长度之和（三角不等式绝对上界 chain_max_reach）。请检查传入的 frame="local" 还是 "world" 是否搞反，或者目标坐标单位是否误传了毫米（mm）而非米（m）。
 * 报错 JOINT_JUMP_LIMIT：
   * 原因：找到了满足位姿和负载的解，但全都因为某个关节相对 q_prev 转动超过了 max_joint_jump_rad（默认 180^\circ）被 H4 门控拦截（常见于跨越奇异点或目标姿态翻转）。
   * 解决：插入过渡路点，或在配置中将 joint_jump_policy 设为 "prefer"（有不超限的优先用，没有时允许大跳变通过）。
 * 报错 NO_LOAD_SAFE_SOLUTION：
   * 原因：位姿可达（pose_valid_count > 0），但所有达标候选解要么触发了某个关节的静态重力矩超载（peak_ratio > 1.0），要么违反了障碍物安全净空（collision_ok == False）。查看 solver.last_load_diagnostics["rejected"] 列表，里面详细记录了每一个被拒候选解的超载关节名 overloaded_joints 与最小障碍物距离 collision_min_distance。
是否需要将这份交接文档导出为可下载的 Markdown 或 PDF 文件，或者针对多进程并发池（_parallel_pool）与 band_bounds 的遗留问题提供修复补丁？
