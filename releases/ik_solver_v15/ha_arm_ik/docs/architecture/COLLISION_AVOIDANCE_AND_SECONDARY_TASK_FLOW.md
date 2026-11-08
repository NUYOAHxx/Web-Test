# 防碰撞（MoveIt 2 / FCL）与零空间次任务协同架构规范文档

---

## 1. 概述与设计背景

本规范文档完整沉淀了 [`ha_arm_ik`](file:///home/parallels/Documents/Web-Test/releases/ik_solver_v15/ha_arm_ik) 中基于 MoveIt 2 SRDF / FCL（`hppfcl` / `coal`）的防碰撞扩展模块 [`collision_fcl.py`](file:///home/parallels/Documents/Web-Test/releases/ik_solver_v15/ha_arm_ik/collision_fcl.py) 与核心逆运动学引擎 [`single_arm_ik.py`](file:///home/parallels/Documents/Web-Test/releases/ik_solver_v15/ha_arm_ik/single_arm_ik.py) 的协同工作流程、单种子迭代内的零空间排斥数学模型、以及次任务（Secondary Objective）体系的内外层对齐机制。

---

## 2. 端到端防碰撞工作流（双模式全景框图）

系统支持两种防碰撞运行模式：
1. **外层非侵入式包装模式（[`CollisionAwareIKSolver`](file:///home/parallels/Documents/Web-Test/releases/ik_solver_v15/ha_arm_ik/collision_fcl.py#L612-L923)）**：不依赖主求解器内部改动，在主求解器输出初始解后，通过零空间碰撞排斥次任务与多分支探索完成避障；
2. **原生单种子内嵌模式（打入 [`single_arm_ik_collision_secondary_task.diff`](file:///home/parallels/Documents/Web-Test/releases/ik_solver_v15/diff/single_arm_ik_collision_secondary_task.diff)）**：将 `RobotCollisionManager` 直接挂载至 `SingleArmIKSolver`，在每个种子的每一步 DLS/SNS 迭代中实时计算碰撞排斥梯度 $\mathbf{z}_{\text{col}}$，并在外层选解时同步质心与防碰撞代价。

### 2.1 外层包装模式工作流框图 (`CollisionAwareIKSolver`)

```mermaid
flowchart TD
    classDef base fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,color:#1b5e20;
    classDef col fill:#e3f2fd,stroke:#1565c0,stroke-width:2px,color:#0d47a1;
    classDef evade fill:#fff3e0,stroke:#ef6c00,stroke-width:2px,color:#e65100;
    classDef out fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2px,color:#4a148c;

    A["输入目标位姿 T_target, 前一帧 q_prev"] --> B["1. 调用原求解器 base_solver.solve()<br/>产出 6D 位姿达标的初始解 q_sol"]:::base
    B --> C["2. 极速碰撞初检 col_mgr.check_collision(q_sol)<br/>(FCL 同步连杆位姿，耗时 ~6.8 µs)"]:::col
    C --> D{"是否发生碰撞或<br/>侵入安全裕度 (20mm)?"}
    
    D -- "否 (完全安全)" --> E["直接返回 (q_sol, SUCCESS)"]:::out
    
    D -- "是 (手肘/前臂碰障碍物或自碰撞)" --> F["3. 阶段 A：启动单种子零空间碰撞排斥滑移<br/>_run_nullspace_collision_settle(q_sol)<br/>(从 q_sol 出发，用见证点排斥梯度 z_col 连续推离 30 步)"]:::evade
    
    F --> G{"阶段 A 是否成功<br/>推离至安全区?"}
    G -- "是" --> H["返回避障精修解 (q_safe, SUCCESS)"]:::out
    
    G -- "否 (被障碍物卡在非凸死角)" --> I["4. 阶段 B：零空间多分支跨区搜索<br/>沿手臂 1D 零空间切向采样 8 个大跨度分支<br/>q_cand = q_sol + α·v_null (α ∈ [-0.85, +0.85])"]:::evade
    I --> J["每个分支执行：<br/>① 12 步 6D 全位姿闭环拉回 (消除位置+姿态漂移)<br/>② 10 步零空间碰撞排斥微调"]:::evade
    J --> K{"是否存在无碰撞且<br/>6D 达标的分支?"}
    K -- "是 (选净空最大者)" --> H
    K -- "否 (物理空间被彻底封死)" --> L["返回失败诊断 (q_sol, 'COLLISION')"]:::out
```

---

### 2.2 原生内嵌模式工作流框图（单种子内嵌 + 内外层次任务对齐）

当在 [`single_arm_ik.py`](file:///home/parallels/Documents/Web-Test/releases/ik_solver_v15/ha_arm_ik/single_arm_ik.py) 中挂载 `collision_manager` 并开启 `sync_secondary_cost=True` 时，防碰撞与水平质心直接贯穿**单种子内层迭代**与**多候选外层择优**全流程：

```mermaid
flowchart TD
    classDef seed fill:#fff8e1,stroke:#ffa000,stroke-width:2px,color:#ff6f00;
    classDef inner fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,color:#1b5e20;
    classDef sync fill:#e1f5fe,stroke:#0277bd,stroke-width:2px,color:#01579b;
    classDef filter fill:#ffebee,stroke:#c62828,stroke-width:2px,color:#b71c1c;

    S0["多阶段种子队列 (Continuation / Comfort / Full / JumpBox)"]:::seed --> S1

    subgraph INNER ["【内层：单种子 DLS/SNS 迭代 (pinocchio_ik_one_seed)】"]
        S1["主任务残差与雅可比 (_make_task_terms)<br/>计算 6D 位姿误差 e 与雅可比 J"]:::inner
        S2["五合一综合次任务求值 (_SecondaryObjective)<br/>z = z_posture + z_limit + z_torque + z_com + z_col<br/>H = H_posture + H_limit + H_torque + H_com + H_col"]:::inner
        S3["SNS 主动集限位钳制与零空间正交投影 (_sns_clamped_core)<br/>dq = dq_task + w_null · (D z - D Jᵀ A⁻¹ J D z)"]:::inner
        S4["回溯线搜索 + 达标后流形稳态整理 (Posture Settle 50 步)<br/>輸出收敛解 q 与次任务分量字典 diag['secondary_components']"]:::inner
        S1 --> S2 --> S3 --> S4
    end

    subgraph OUTER ["【外层：内外层代价同步与 Pareto 分档择优】"]
        O1["1. 基础候选打分 (score_candidate)<br/>计算 psi_cost + smooth_cost + limit_cost + waist_cost"]:::sync
        O2["2. 内外层次任务代价同步 (解决口径不一致问题)<br/>提取 diag['secondary_components'] 中的 com 与 collision 分量<br/>并入总代价：cost += w_com · c_com + w_collision · c_col"]:::sync
        O3["3. 硬约束门禁过滤 (_check_load_candidate)<br/>H1(6D位姿) ∩ H2(限位) ∩ H3(重力矩 ≤ 100%) ∩ H4(防跳变) ∩ H5(collision_ok)"]:::filter
        O4["4. 峰值载荷容差带分档 + 同档最优交付 (candidate_sort_key)"]:::sync
        O1 --> O2 --> O3 --> O4
    end

    S4 --> O1
```

---

## 3. 单种子每一步迭代内的防碰撞微观流程与数学原理

在单个种子的第 $k$ 步迭代（当前关节角 $\mathbf{q}_k \in \mathbb{R}^n$）中，防碰撞次任务按以下 **4 个微观步骤** 闭环执行：

```mermaid
flowchart TD
    classDef step1 fill:#e1f5fe,stroke:#0277bd,stroke-width:2px;
    classDef step2 fill:#fff8e1,stroke:#f9a825,stroke-width:2px;
    classDef step3 fill:#fce4ec,stroke:#c2185b,stroke-width:2px;
    classDef step4 fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px;

    subgraph MICRO ["单种子第 k 步防碰撞微观计算流"]
        M1["【步骤 1：6D 全位姿主任务残差与雅可比】<br/>e_pos = p_des - p_cur<br/>e_rot = -R_tar · log(R_tarᵀ R_cur)<br/>e_6D = [e_pos; e_rot] ∈ R⁶,  J(q_k) ∈ R⁶ˣⁿ"]:::step1

        M2["【步骤 2：FCL 几何同步与三维最近见证点测距】<br/>同步全连杆 SE(3) 位姿并调用 pin.computeDistances(geom_model, geom_data)<br/>对距离 d_AB < d_inf 的风险对提取：<br/>连杆 A 见证点 p_A、障碍物/连杆 B 见证点 p_B、单位接触法向 n (A → B)"]:::step2

        M3["【步骤 3：见证点线速度雅可比与 C¹ 连续排斥梯度】<br/>1. 见证点雅可比：J_pt(p_A) = J_trans,A - [p_A - o_A]× J_rot,A ∈ R³ˣⁿ<br/>2. 距离对关节角解析导数：∇_q d_AB = (-n)ᵀ J_pt(p_A) + (+n)ᵀ J_pt(p_B)<br/>3. C¹ 连续双层二次排斥势场负梯度：<br/>   z_col = -∇_q H_col(q_k) = ∑ grad_coeff(d_AB) · ∇_q d_AB"]:::step3

        M4["【步骤 4：度量一致零空间正交投影】<br/>A = J D Jᵀ + λI<br/>主任务步：dq_task = D Jᵀ A⁻¹ e_6D<br/>零空间步：dq_null = w_null · (D z - D Jᵀ A⁻¹ J D z)<br/>更新：q_{k+1} = q_k + dq_task + dq_null"]:::step4

        M1 --> M2 --> M3 --> M4
    end
```

### 3.1 关键数学公式说明

1. **为什么早期版本避障后姿态会漂移 $>6^\circ$，而现在姿态误差 $<0.001^\circ$？**
   - [`ik_base.py:pose_error(T_cur, T_tar)`](file:///home/parallels/Documents/Web-Test/releases/ik_solver_v15/ha_arm_ik/ik_base.py#L462-L467) 返回的旋转向量 $\mathbf{d}_{\text{rot}} = \log(R_{\text{tar}}^T R_{\text{cur}})$ 定义在**目标局部坐标系**下且方向为 $\text{cur} - \text{tar}$。
   - 而几何雅可比 $J_{\text{rot}}$ 定义在**基座坐标系**下。因此闭环消除姿态误差的基座系角速度残差必须经旋转矩阵映射并取反：
     $$\mathbf{e}_{\text{rot}} = -R_{\text{tar}} \cdot \log(R_{\text{tar}}^T R_{\text{cur}})$$
2. **$C^1$ 连续双层二次排斥势场 $H_{\text{col}}(\mathbf{q})$**：
   为消除在安全裕度边界 $d = d_{\text{safe}}$ 处的梯度断崖跳变，采用外层缓冲势垒（半径 $d_{\text{inf}}$）与内层高刚度势垒（半径 $d_{\text{safe}}$）光滑叠加：
   $$u_{\text{outer}} = \max\left(0, \, \frac{d_{\text{inf}} - d_{AB}}{d_{\text{inf}}}\right), \quad u_{\text{inner}} = \max\left(0, \, \frac{d_{\text{safe}} - d_{AB}}{d_{\text{safe}}}\right)$$
   $$H_{\text{col}}(d_{AB}) = \frac{1}{2} k_{\text{col}} u_{\text{outer}}^2 + \frac{1}{2} (2 k_{\text{col}}) u_{\text{inner}}^2$$
   $$z_{\text{col}} = -\nabla_q H_{\text{col}} = \left( \frac{k_{\text{col}}}{d_{\text{inf}}} u_{\text{outer}} + \frac{2 k_{\text{col}}}{d_{\text{safe}}} u_{\text{inner}} \right) \nabla_q d_{AB}$$
   在 $d_{AB} = d_{\text{safe}}$ 处，$u_{\text{inner}} = 0$，势能函数值与一阶导数均严格连续（$C^1$），彻底杜绝边界振颤。

---

## 4. 次任务（Secondary Objective）逻辑审计与内外层对齐修复

针对 [`single_arm_ik.py`](file:///home/parallels/Documents/Web-Test/releases/ik_solver_v15/ha_arm_ik/single_arm_ik.py) 内外层次任务体系的审计结论与处理方案如下：

| 审计项 | 现象描述 | 处理策略与代码落点 |
| :--- | :--- | :--- |
| **问题 1：排斥势场边界连续性** | 旧实现使用阶跃乘子 `boost = 3.0 if d < threshold else 1.0`，导致 $d = d_{\text{safe}}$ 处出现 3 倍梯度断崖。 | **已修复**：在 [`collision_fcl.py`](file:///home/parallels/Documents/Web-Test/releases/ik_solver_v15/ha_arm_ik/collision_fcl.py#L550-L565) 中升级为 $C^1$ 连续双层二次势场。 |
| **问题 2：`_SecondaryObjective` 中的 `relative_load` 项** | `relative_load`（维持参考力矩）与 `absolute_load`（卸载力矩）方向相反，但在默认配置下权重恒为 `0.0`（未激活）。 | **保持不变**：按约定保留原代码分支，不影响任何现有行为与回归测试。 |
| **问题 3：内层次任务 (`_SecondaryObjective`) 与外层选解 (`score_candidate`) 口径不一致** | ① 内层每步迭代计算了整机水平质心偏移代价 `parts["com"]`（及碰撞代价 `parts["collision"]`），但外层 `score_candidate` 打分时未将其计入总代价 `cost`，导致内层零空间整理的质心与避障成果在外层跨种子排序时被丢弃；<br>② 内层腰部回中增益（`0.50`）与外层腰回正权重（`W_WAIST_RETURN = 2000.0`）差距过大。 | **已解决并沉淀至补丁与源码**：在 [`single_arm_ik.py`](file:///home/parallels/Documents/Web-Test/releases/ik_solver_v15/ha_arm_ik/single_arm_ik.py#L4417-L4422) 与 [`collect_pinocchio_candidates_and_seeds`](file:///home/parallels/Documents/Web-Test/releases/ik_solver_v15/ha_arm_ik/single_arm_ik.py#L4725-L4737) 中新增 `sync_secondary_cost` 同步机制（当挂载 `collision_manager` 或显式开启时自动激活）：<br>1. 将 `diag["secondary_components"]` 中的 `com` 与 `collision` 代价按权重 `w_com * c_com + w_collision * c_col` 直接并入外层 `info["cost_no_torque"]` 与 `info["cost"]`；<br>2. 自动将内层默认腰部回中增益提升至 `2.0`，对齐内外层腰部回正偏好；<br>3. 在 `_check_load_candidate` 中增加 `collision_ok` 硬约束门禁。 |

