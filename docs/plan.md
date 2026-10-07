# Unitree G1 双臂逆运动学 (DualArmIKSolver) 架构设计与工程落地规划

> **设计定位**：针对 Unitree G1 人形机器人 16 DoF（2 DoF 腰部 + 双 7 DoF 手臂）系统，构建工业级双臂协同逆运动学求解模块（`DualArmIKSolver`）。  
> **核心任务**：输入左右末端 6D 目标位姿与当前/种子关节状态，在严格满足关节限位与位姿精度的前提下，求解平滑紧凑的 $q_{goal}$，并交付 MoveIt 2 / OMPL 进行无碰撞路径规划。  
> **工程原则**：分层解耦、职责正交、参数外置、算法内核与 MoveIt 插件适配器分离。

---

## 目录

- [一、系统边界与模块分工 (System Boundaries & Architecture)](#一系统边界与模块分工-system-boundaries--architecture)
  - [1.1 系统数据流全景图](#11-系统数据流全景图)
  - [1.2 模块权责边界划分](#12-模块权责边界划分)
  - [1.3 核心框架复用原则](#13-核心框架复用原则)
- [二、运动学数学问题严格形式化 (Kinematics Formulation)](#二运动学数学问题严格形式化-kinematics-formulation)
  - [2.1 联合关节状态向量与任务空间自由度](#21-联合关节状态向量与任务空间自由度)
  - [2.2 4-DoF 零空间冗余的物理与工程价值](#22-4-dof-零空间冗余的物理与工程价值)
  - [2.3 双臂腰部共享强耦合机理](#23-双臂腰部共享强耦合机理)
- [三、迭代加权二次规划 (QP) 逆运动学算法设计 (Iterative QP Position IK)](#三迭代加权二次规划-qp-逆运动学算法设计-iterative-qp-position-ik)
  - [3.1 局部泰勒展开与任务误差空间](#31-局部泰勒展开与任务误差空间)
  - [3.2 基础 QP 优化目标函数](#32-基础-qp-优化目标函数)
  - [3.3 种子状态跟踪 (Seed Tracking) 与构型突变规避](#33-种子状态跟踪-seed-tracking-与构型突变规避)
  - [3.4 关节物理限位与单步步长盒式硬约束](#34-关节物理限位与单步步长盒式硬约束)
  - [3.5 V1 标准二次规划形式推导](#35-v1-标准二次规划形式推导)
  - [3.6 空间位姿误差计算策略 (Pose Error in R12)](#36-空间位姿误差计算策略-pose-error-in-r12)
  - [3.7 任务权重矩阵与左右臂解耦配置](#37-任务权重矩阵与左右臂解耦配置)
  - [3.8 零空间次级优化项扩展 (Null-space Objectives)](#38-零空间次级优化项扩展-null-space-objectives)
- [四、碰撞检测与规划系统对接策略 (Collision & Planning Integration)](#四碰撞检测与规划系统对接策略-collision--planning-integration)
  - [4.1 V1 阶段碰撞检测边界](#41-v1-阶段碰撞检测边界)
  - [4.2 PlanningScene 外部闭环校验机制](#42-planningscene-外部闭环校验机制)
  - [4.3 IK 与 OMPL 的协同架构接口](#43-ik-与-ompl-的协同架构接口)
- [五、工程架构与核心 C++ 类设计 (Software Architecture & Class Design)](#五工程架构与核心-c-类设计-software-architecture--class-design)
  - [5.1 模块目录组织结构](#51-模块目录组织结构)
  - [5.2 核心类接口设计 (DualArmIKSolver)](#52-核心类接口设计-dualarmiksolver)
  - [5.3 参数化外部配置文件规范 (YAML)](#53-参数化外部配置文件规范-yaml)
  - [5.4 单次求解执行状态机与流水线](#54-单次求解执行状态机与流水线)
  - [5.5 步长收敛与阻尼衰减因子](#55-步长收敛与阻尼衰减因子)
  - [5.6 MoveIt Kinematics Plugin 适配器模式架构](#56-moveit-kinematics-plugin-适配器模式架构)
- [六、版本演进路线与开发实施计划 (Implementation Roadmap)](#六版本演进路线与开发实施计划-implementation-roadmap)
  - [6.1 四阶段版本演进路线 (V0 ~ V2)](#61-四阶段版本演进路线-v0--v2)
  - [6.2 V3 高级功能前瞻](#62-v3-高级功能前瞻)
  - [6.3 模块职责与边界速查表](#63-模块职责与边界速查表)
  - [6.4 实施步骤与关键排坑事项 (15 步实操清单)](#64-实施步骤与关键排坑事项-15-步实操清单)

---

## 一、系统边界与模块分工 (System Boundaries & Architecture)

### 1.1 系统数据流全景图

Unitree G1 机器人的双臂操作规划系统包含明确的管线层级划分：

```text
               ┌─────────────────────────────────────────┐
               │    笛卡尔目标指令 (Cartesian Targets)   │
               │  Left Target Pose  &  Right Target Pose │
               └────────────────────┬────────────────────┘
                                    │
                                    ▼
               ┌─────────────────────────────────────────┐
               │         Dual-Arm IK Solver 内核         │
               │  • 16 DoF 联合关节空间优化              │
               │  • 12D 笛卡尔任务空间加权跟踪           │
               │  • 盒式关节限位硬约束 (Box Constraints) │
               │  • 种子邻域保持 (Seed Tracking)         │
               └────────────────────┬────────────────────┘
                                    │
                                    ▼
                           目标关节构型 q_goal
                          (16×1 Joint Vector)
                                    │
                                    ▼
               ┌─────────────────────────────────────────┐
               │       MoveIt 2 / OMPL 运动规划器        │
               │  • 碰撞检测 (PlanningScene / FCL)       │
               │  • 连续路径搜索 (RRT-Connect / PRM)     │
               │  • q_start ───> q_goal 避障轨迹生成     │
               └────────────────────┬────────────────────┘
                                    │
                                    ▼
                         无碰撞关节空间轨迹
                         (Joint Trajectory)
                                    │
                                    ▼
               ┌─────────────────────────────────────────┐
               │        底盘与关节轨迹执行控制器         │
               │     (Joint Trajectory Controller)       │
               └─────────────────────────────────────────┘
```

---

### 1.2 模块权责边界划分

系统各构件的职责严格正交，坚决杜绝逻辑越界：

1. **`DualArmIKSolver`（目标生成器 / Goal Generator）**：
   - 负责：几何正/逆运动学计算、雅可比推导、关节物理限位裁决、零空间冗余构型分配，最终输出合法的 $q_{goal}$；
   - **不负责**：碰撞几何体之间的连续分离距离检测、全局无碰撞路径搜索、轨迹时间参数化。
2. **`OMPL`（路径生成器 / Path Generator）**：
   - 负责：从起始构型 $q_{start}$ 到目标构型 $q_{goal}$ 的构型空间流形搜索；
   - **不负责**：逆运动学位姿数值优化。
3. **`PlanningScene / FCL`（环境与自身碰撞判定器 / Collision Checker）**：
   - 负责：判定任意离散状态 $q$ 是否与自身或外部环境几何体发生干涉。
4. **MoveIt `RobotState`（机器人动力学与几何容器）**：
   - 负责：维护 URDF 树状运动学拓扑、提供标准正运动学（FK）、几何雅可比矩阵提取与物理限位参数读取。

---

### 1.3 核心框架复用原则

> [!NOTE]
> 严禁重复造轮子：本项目不自行从底层重写 URDF 解析、FCL 网格碰撞、RRT 算法及基础 FK 运算。  
> MoveIt 原生 `KinematicsBase` 接口不仅原生支持“多末端执行器（Multiple End-Effectors）同时给定目标位姿”的求解语义，且其核心规范明确要求常规 `getPositionIK()` 应优先返回最接近 Seed State 的局部解，这与 G1 人形机器人双臂 16 DoF 冗余系统完全契合。

---

## 二、运动学数学问题严格形式化 (Kinematics Formulation)

### 2.1 联合关节状态向量与任务空间自由度

Unitree G1 的上半身联合广义坐标向量定义为：

$$
q = \begin{bmatrix}
q_{wy} \\
q_{wp} \\
q_{L1} \\
\vdots \\
q_{L7} \\
q_{R1} \\
\vdots \\
q_{R7}
\end{bmatrix} \in \mathbb{R}^{16}
$$

分块组成如下：
- **腰部自由度 (Waist)**：2 DoF（偏航 Yaw $q_{wy}$、俯仰 Pitch $q_{wp}$）；
- **左侧机械臂 (Left Arm)**：7 DoF（肩、肘、腕部驱动关节 $q_{L1 \dots L7}$）；
- **右侧机械臂 (Right Arm)**：7 DoF（肩、肘、腕部驱动关节 $q_{R1 \dots R7}$）。

左右两侧末端执行器各自在三维空间中拥有 6 DoF 自由度（3 维平移 + 3 维旋转）：

$$
x = \begin{bmatrix} x_L \\ x_R \end{bmatrix} \in \mathbb{R}^{12}
$$

系统整体映射关系呈现显著的冗余特性：

$$
\dim(q) = 16 \longrightarrow \dim(x) = 12 \quad \Longrightarrow \quad n - m = 16 - 12 = 4\text{ DoF 冗余}
$$

---

### 2.2 4-DoF 零空间冗余的物理与工程价值

这 4 个冗余自由度是人形机器人操作算法设计的核心空间，用于承载以下次级优化指标：
1. **腰部端正偏好**：在手臂工作空间足够时保持躯干直立，避免不自然的躯干过度摇晃；
2. **肘部姿态调节 (Elbow Swivel Angle)**：保持舒适的人体仿生工效学构型；
3. **远离关节软限位**：引导关节远离机械硬死点，保留动态避险裕度；
4. **历史构型一致性**：抑制突发跳变，保证连续控制下的构型稳定性；
5. **运动学可操作度 (Manipulability)**：规避奇异位形；
6. **双臂自碰撞主动避让**：在零空间内拉开双臂骨架间距。

---

### 2.3 双臂腰部共享强耦合机理

> [!WARNING]
> **绝对禁止拆分为独立单臂求解**：  
> 绝对不能将腰部分离后分别调用左臂 IK 与右臂 IK。因为腰部的 $q_{wy}, q_{wp}$ 运动将同时刚性带动左臂基座与右臂基座发生空间位移和旋转。

系统正运动学显式耦合方程为：

$$
x_L = f_L(q_W, q_L), \qquad x_R = f_R(q_W, q_R)
$$

$$
\begin{bmatrix} x_L \\ x_R \end{bmatrix} = f(q_W, q_L, q_R)
$$

对应的空间全微分几何雅可比矩阵 $J(q) \in \mathbb{R}^{12 \times 16}$ 结构如下：

$$
J(q) = \begin{bmatrix}
J_L(q) \\
J_R(q)
\end{bmatrix}
=
\begin{bmatrix}
J_{L,W} & J_{L,arm} & 0_{6 \times 7} \\
J_{R,W} & 0_{6 \times 7} & J_{R,arm}
\end{bmatrix}
$$

其中：
- $J_{L,W} \in \mathbb{R}^{6 \times 2}$ 与 $J_{R,W} \in \mathbb{R}^{6 \times 2}$ 分别描述腰部运动对左右末端的影响；
- $J_{L,arm} \in \mathbb{R}^{6 \times 7}$ 与 $J_{R,arm} \in \mathbb{R}^{6 \times 7}$ 分别为左右臂独立关节对对应末端的影响；
- 矩阵中的互不干涉项严格为零分块 $0_{6 \times 7}$。

---

## 三、迭代加权二次规划 (QP) 逆运动学算法设计 (Iterative QP Position IK)

### 3.1 局部泰勒展开与任务误差空间

在当前迭代构型 $q_k$ 附近进行一阶局部线性化：

$$
x(q_k + \Delta q) \approx x(q_k) + J(q_k) \Delta q
$$

定义 12 维笛卡尔任务空间综合残差：

$$
e = x^* - x(q_k) = \begin{bmatrix} x_L^* - x_L(q_k) \\ x_R^* - x_R(q_k) \end{bmatrix} \in \mathbb{R}^{12}
$$

线性逼近目标即为：

$$
J(q_k) \Delta q \approx e
$$

---

### 3.2 基础 QP 优化目标函数

为防止雅可比矩阵奇异引发的关节增量发散，基础加权阻尼最小二乘目标表述为：

$$
\min_{\Delta q} \frac{1}{2} \|J \Delta q - e\|_W^2 + \frac{1}{2} \|\Delta q\|_R^2
$$

其中 $W \in \mathbb{R}^{12 \times 12}$ 为任务空间权重对角阵，$R \in \mathbb{R}^{16 \times 16}$ 为正则化阻尼权重矩阵。

---

### 3.3 种子状态跟踪 (Seed Tracking) 与构型突变规避

鉴于 16 DoF 系统的多解特性，求解器必须引入种子吸引势，优先收敛至与 $q_{seed}$ 距离最近的解流形：

$$
\min_{\Delta q} \frac{1}{2} \|J \Delta q - e\|_W^2 + \frac{1}{2} \|q_k + \Delta q - q_{seed}\|_R^2
$$

**物理效果与工程收益**：
- **第一优先级（高权重 $W$）**：左右双末端精准追踪目标位姿；
- **第二优先级（正则权重 $R$）**：尽量维持机械臂与腰部初始构型，抑制反肘、腰部无故剧烈偏航及左右臂构型跳变。

---

### 3.4 关节物理限位与单步步长盒式硬约束

相比经典 DLS 事后截断（Clamping）导致任务轨迹严重偏离，QP 原生支持盒式上下界硬约束（Box Constraints）：

$$
q_{min} \le q_k + \Delta q \le q_{max} \implies q_{min} - q_k \le \Delta q \le q_{max} - q_k
$$

同时为保证线性化成立并抑制超调，限制单步最大更新步长 $\Delta q_{max}$：

$$
-\Delta q_{max} \le \Delta q \le \Delta q_{max}
$$

合并得到 QP 的绝对上下界向量：

$$
lb = \max\left(q_{min} - q_k, \ -\Delta q_{max}\right)
$$

$$
ub = \min\left(q_{max} - q_k, \ \Delta q_{max}\right)
$$

在每一步迭代中，求解内核天然保证输出构型绝不越界。

---

### 3.5 V1 标准二次规划形式推导

将优化目标代数展开并整理为标准二次规划形式：

$$
\min_{\Delta q} \frac{1}{2} \Delta q^T H \Delta q + g^T \Delta q
$$

$$
\text{s.t.} \quad lb \le \Delta q \le ub
$$

#### Hessian 矩阵推导

$$
H = J^T W J + R \in \mathbb{R}^{16 \times 16}
$$

#### 线性梯度向量推导

$$
g = -J^T W e + R (q_k - q_{seed}) \in \mathbb{R}^{16}
$$

求解 QP 得到当前最佳单步搜索方向 $\Delta q$ 后，执行带松弛因子的状态更新：

$$
q_{k+1} = q_k + \alpha \Delta q \quad (\alpha \in (0, 1])
$$

---

### 3.6 空间位姿误差计算策略 (Pose Error in R12)

严禁直接采用欧拉角做减法，内部计算必须基于 $\mathcal{SO}(3)$ 旋转流形与四元数规范化：

$$
e = \begin{bmatrix} e_L \\ e_R \end{bmatrix} = \begin{bmatrix} e_p^L \\ e_R^L \\ e_p^R \\ e_R^R \end{bmatrix} \in \mathbb{R}^{12}
$$

- **位置误差 (3D)**：
  $$e_p = p_d - p$$
- **旋转误差 (3D)**：采用对数映射（Logarithmic Map）或四元数偏差轴角投影：
  $$e_R = 2 \left[ q_d \otimes q^{-1} \right]_{xyz}$$
  在计算前进行四元数半球对齐归一化处理（若 $q_d \cdot q < 0$，则取 $q_d \leftarrow -q_d$），彻底消除旋转误差不连续点。

---

### 3.7 任务权重矩阵与左右臂解耦配置

权重对角阵 $W$ 用于调控任务各自由度的惩罚刚度：

$$
W = \text{diag}\left(
  w_{p}^L, w_{p}^L, w_{p}^L, \quad
  w_{o}^L, w_{o}^L, w_{o}^L, \quad
  w_{p}^R, w_{p}^R, w_{p}^R, \quad
  w_{o}^R, w_{o}^R, w_{o}^R
\right)
$$

- **平移与旋转均衡**：通常位置权重设置高于旋转权重（如 $w_p = 1.0, w_o = 0.3 \sim 0.5$）；
- **任务不对称支持**：若单臂持物、另一臂仅做指示，可动态调高持物手臂任务权重。

---

### 3.8 零空间次级优化项扩展 (Null-space Objectives)

在 V1 稳定后，可将次级目标直接并入正则项 $R$ 中，例如加入关节中心吸引势能项：

$$
C_{joint}(q) = \sum_{i=1}^{16} \left( \frac{q_i - q_{i,center}}{q_{i,max} - q_{i,min}} \right)^2
$$

使梯度向量追加项为 $g_{joint} = k_c \nabla C_{joint}(q_k)$，使机械臂在冗余零空间内持续向舒适作业中心姿态靠拢。

---

## 四、碰撞检测与规划系统对接策略 (Collision & Planning Integration)

### 4.1 V1 阶段碰撞检测边界

> [!CAUTION]
> **切勿在 V1 将非凸网格碰撞检测直接塞入 QP 内核**：  
> FCL / PlanningScene 的底层算法输出的是离散布尔冲突或最近几何原语对距离，若要在 QP 中硬性加入避障，需要持续计算带符号距离场梯度 $\nabla d(q)$ 并构建局部半空间线性切平面。这极易导致非凸陷入局部死锁，使求解延迟由 2 ms 飙升至数十毫秒。

---

### 4.2 PlanningScene 外部闭环校验机制

V1 阶段采用“**先快速 IK，再环境过滤校验**”的健壮架构：

```text
 目标位姿 (Target) ───> [DualArmIKSolver 快速求解] ───> 候选关节解 q_goal
                                                              │
                                                              ▼
                                                 [PlanningScene::isStateValid()]
                                                              │
                             ┌────────────────────────────────┴────────────────────────────────┐
                             ▼                                                                 ▼
                        合法无碰撞                                                          发生碰撞
                             │                                                                 │
                             ▼                                                                 ▼
                   交付 OMPL 执行路径规划                                            启动二次恢复机制:
                                                                                   1. 扰动生成候选 Seed 重算
                                                                                   2. 调整腰部姿态偏好重算
```

---

### 4.3 IK 与 OMPL 的协同架构接口

逆运动学与路径规划各司其职：
- **`IK Solver` = 目标发生器 (Goal Generator)**：负责把笛卡尔需求转译为关节空间有效终点；
- **`OMPL` = 路径规划器 (Path Planner)**：负责搜索规避障碍的连续关节轨迹。

```cpp
// 典型业务调用链示例
IKResult result = ik_solver.solve(seed_state, left_target, right_target);
if (!result.success) {
    RCLCPP_WARN(logger, "IK 求解失败: %s", toString(result.failure_reason));
    return false;
}

// 检查该目标姿态在当前场景中是否发生碰撞
if (!planning_scene->isStateValid(result.solution, "upper_body")) {
    RCLCPP_WARN(logger, "IK 解处于碰撞区域，触发备用种子重新求解");
    // 触发重试策略...
    return false;
}

// 交付 MoveIt 2 启动路径规划
move_group.setJointValueTarget(result.solution);
move_group.plan(my_plan);
```

---

## 五、工程架构与核心 C++ 类设计 (Software Architecture & Class Design)

### 5.1 模块目录组织结构

建议将算法模块设计为自包含独立包，隔离核心数学计算与 ROS 2 插件层：

```text
dual_arm_ik/
├── CMakeLists.txt
├── package.xml
├── include/
│   └── dual_arm_ik/
│       ├── dual_arm_ik_solver.hpp    # 核心求解器主体类
│       ├── ik_config.hpp             # 求解器参数结构体
│       ├── ik_types.hpp              # 错误码、结果与中间量定义
│       ├── pose_error.hpp            # SE(3) 流形误差与雅可比转换
│       ├── qp_solver.hpp             # 抽象 QP 求解器封装 (OSQP / ProxQP / QPOases)
│       └── kinematics_adapter.hpp    # MoveIt RobotState 访问适配层
├── src/
│   ├── dual_arm_ik_solver.cpp
│   ├── pose_error.cpp
│   ├── qp_solver.cpp
│   └── kinematics_adapter.cpp
├── config/
│   └── dual_arm_ik.yaml             # 外部运行时调优参数
├── test/
│   ├── test_fk_consistency.cpp       # Level 1: FK 验证
│   ├── test_jacobian_diff.cpp        # Level 1: 数值雅可比有限差分
│   ├── test_pose_error.cpp           # Level 1: 姿态误差对偶性测试
│   └── test_dual_arm_benchmark.cpp   # Level 2: 综合 Benchmark 测试用例
└── plugin/
    └── dual_arm_kinematics_plugin.cpp # MoveIt KinematicsBase 插件接口实现
```

---

### 5.2 核心类接口设计 (DualArmIKSolver)

#### 错误码与结果结构体 (`ik_types.hpp`)

```cpp
#pragma once
#include <Eigen/Core>
#include <string>

enum class IKFailureReason : uint8_t {
    NONE = 0,
    MAX_ITERATIONS_REACHED,
    QP_SOLVER_FAILED,
    JOINT_LIMIT_VIOLATION,
    INVALID_TARGET_INPUT,
    SINGULAR_CONFIGURATION
};

struct IKResult {
    bool success{false};
    Eigen::VectorXd solution;              // 16×1 关节角
    
    double position_error_left{0.0};       // 米
    double orientation_error_left{0.0};    // 弧度
    double position_error_right{0.0};      // 米
    double orientation_error_right{0.0};   // 弧度
    double total_residual{0.0};
    
    int iterations{0};
    double solve_time_ms{0.0};
    IKFailureReason failure_reason{IKFailureReason::NONE};
};
```

#### 求解器主类定义 (`dual_arm_ik_solver.hpp`)

```cpp
#pragma once
#include "dual_arm_ik/ik_config.hpp"
#include "dual_arm_ik/ik_types.hpp"
#include <moveit/robot_model/robot_model.h>
#include <moveit/robot_state/robot_state.h>
#include <geometry_msgs/msg/pose.hpp>
#include <memory>

class DualArmIKSolver {
public:
    using Ptr = std::shared_ptr<DualArmIKSolver>;
    using ConstPtr = std::shared_ptr<const DualArmIKSolver>;

    DualArmIKSolver() = default;
    ~DualArmIKSolver() = default;

    bool initialize(
        const moveit::core::RobotModelConstPtr& robot_model,
        const IKConfig& config);

    IKResult solve(
        const moveit::core::RobotState& seed_state,
        const geometry_msgs::msg::Pose& left_target,
        const geometry_msgs::msg::Pose& right_target);

private:
    bool computeTaskResidual(
        const moveit::core::RobotState& state,
        const geometry_msgs::msg::Pose& left_target,
        const geometry_msgs::msg::Pose& right_target,
        Eigen::Vector<double, 12>& residual,
        IKResult& result);

    bool computeCombinedJacobian(
        const moveit::core::RobotState& state,
        Eigen::Matrix<double, 12, 16>& J_combined);

    bool solveSingleStepQP(
        const Eigen::Matrix<double, 12, 16>& J,
        const Eigen::Vector<double, 12>& residual,
        const Eigen::Vector<double, 16>& q_current,
        const Eigen::Vector<double, 16>& q_seed,
        Eigen::Vector<double, 16>& dq);

private:
    moveit::core::RobotModelConstPtr robot_model_;
    const moveit::core::JointModelGroup* joint_group_{nullptr};
    
    IKConfig config_;
    std::string left_tip_frame_;
    std::string right_tip_frame_;
    std::vector<std::string> joint_names_;
};
```

---

### 5.3 参数化外部配置文件规范 (YAML)

所有的调优参数均应在外部统一管理：

```yaml
dual_arm_ik:
  # 收敛终止准则
  max_iterations: 80
  position_tolerance: 0.001       # 1 mm
  orientation_tolerance: 0.0087   # 0.5 度 (弧度)

  # 步长与阻尼因子
  step_scale: 1.0                 # 默认更新阻尼 alpha
  regularization_damping: 0.01    # 对角线 Tikhonov 阻尼系数

  # 关节单步最大变化限幅 (rad)
  max_step_limits:
    waist_yaw: 0.08
    waist_pitch: 0.08
    arm_joints: 0.15

  # 任务权重 (12D)
  weights:
    position_left: 1.0
    orientation_left: 0.3
    position_right: 1.0
    orientation_right: 0.3

  # 种子姿态保持权重 (16D)
  seed_tracking_weights:
    waist: 0.8
    left_arm: 0.1
    right_arm: 0.1

  # 运动学骨架链配置
  planning_group: "upper_body"
  left_tip_link: "left_wrist_roll_link"
  right_tip_link: "right_wrist_roll_link"
```

---

### 5.4 单次求解执行状态机与流水线

在 `solve()` 函数内部执行 10 步闭环迭代流水线：

```text
 ┌────────────────────────────────────────────────────────────────────────┐
 │ Step 1: 初始化 q_k = q_seed，装载目标位姿                                │
 └───────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │ Step 2: 将 q_k 写入内部 RobotState，调用 update() 更新各连杆全局变换      │
 └───────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │ Step 3: 计算左右末端当前位姿与目标位姿的 12 维残差 e_k                      │
 └───────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │ Step 4: 收敛性检验: 是否 ||e_p|| < tol_p 且 ||e_R|| < tol_R ?           │
 └─────────────────┬────────────────────────────────────┬─────────────────┘
                   │ 是                                 │ 否
                   ▼                                    ▼
       ┌───────────────────────┐            ┌────────────────────────────┐
       │ 标记 SUCCESS 并返回解  │            │ Step 5: 计算联合几何雅可比  │
       └───────────────────────┘            │ J(q_k) ∈ R^{12×16}         │
                                            └─────────────┬──────────────┘
                                                          │
                                                          ▼
                                            ┌────────────────────────────┐
                                            │ Step 6: 组装 Hessian 矩阵  │
                                            │ H 与线性梯度向量 g          │
                                            └─────────────┬──────────────┘
                                                          │
                                                          ▼
                                            ┌────────────────────────────┐
                                            │ Step 7: 结合物理限位计算    │
                                            │ 动态上下界 lb 与 ub        │
                                            └─────────────┬──────────────┘
                                                          │
                                                          ▼
                                            ┌────────────────────────────┐
                                            │ Step 8: QP 求解器求解 Δq    │
                                            └─────────────┬──────────────┘
                                                          │
                                                          ▼
                                            ┌────────────────────────────┐
                                            │ Step 9: 步长更新:          │
                                            │ q_{k+1} = q_k + α · Δq     │
                                            └─────────────┬──────────────┘
                                                          │
                                                          ▼
                                            ┌────────────────────────────┐
                                            │ Step 10: 步数递增并回到    │
                                            │ Step 2 进行下一次循环       │
                                            └────────────────────────────┘
```

---

### 5.5 步长收敛与阻尼衰减因子

在某些靠近工作空间边界或局部非凸区域，满步长更新（$\alpha = 1.0$）可能引发残差振荡（Oscillation）。

可在内部预留简单的 Armijo 式回溯阻尼：若单步更新后误差反而明显恶化，则退缩采用半步长 $\alpha \leftarrow 0.5 \alpha$ 重新计算，保证算法的单调收敛性。

---

### 5.6 MoveIt Kinematics Plugin 适配器模式架构

算法设计严格解耦数学核心与 ROS 2 通信框架：

```text
                  MoveIt 2 运动规划系统
                            │
                            ▼
           DualArmKinematicsPlugin (ROS 2 插件适配层)
           • 继承于 kinematics::KinematicsBase
           • 实现 searchPositionIK() / getPositionIK()
           • 处理 ROS 消息转换与服务响应
                            │
                            ▼
           DualArmIKSolver (纯 C++ 数学求解内核)
           • 纯矩阵与优化运算，不绑定 ROS 2 节点生命周期
           • 依赖: Eigen3, QP Solver (OSQP / ProxQP)
```

---

## 六、版本演进路线与开发实施计划 (Implementation Roadmap)

### 6.1 四阶段版本演进路线 (V0 ~ V2)

```text
  V0: 运动学基石验证 ───> V1: 基础双臂 QP-IK ───> V1.5: MoveIt 插件封装 ───> V2: OMPL 规划全联调
  • Joint 拓扑排序       • 12×16 雅可比组装        • KinematicsBase 继承     • 端到端 E2E 规划
  • 解析/数值差分 J      • 盒式硬限位约束         • 参数 YAML 加载          • 场景障碍碰撞过滤
  • 位姿误差连续性       • 种子状态平滑保持       • ROS 2 插件动态加载      • 轨迹平滑性分析
```

- **V0 阶段（基石校验）**：
  - 确认 16 个关节在 URDF/SRDF 中的绝对命名与排序索引；
  - 验证双臂解析雅可比与数值差分雅可比在 $10^{-5}$ 容差下严格一致；
  - 验证四元数翻转处理机制在 $180^\circ$ 边界无跳变。
- **V1 阶段（求解器内核实现）**：
  - 建立标准 QP 求解管线（Hessian、梯度向量、盒式约束）；
  - 实现左右臂位姿加权残差计算与种子距离最小化；
  - 完成基准数据集测试，达到单次求解均值 $< 2.5\text{ ms}$。
- **V1.5 阶段（MoveIt 插件化封装）**：
  - 封装为 `kinematics::KinematicsBase` 动态共享库插件；
  - 跑通 MoveIt 原生可视化测试工具及多目标位姿接口。
- **V2 阶段（规划全链路联调）**：
  - 将生成的 $q_{goal}$ 接入 PlanningScene 碰撞验证；
  - 联调 OMPL RRT-Connect，评估端到端成功率与路径质量。

---

### 6.2 V3 高级功能前瞻

在 V1/V2 架构稳固的基础上，可在后续迭代中平滑扩展：
1. **Collision-aware IK**：引入带符号距离场梯度，将自碰撞与环境碰撞距离作为不等式约束直接加入 QP 内核；
2. **多候选解生成 (Multi-candidate IK)**：针对不同肘部仰角与腰部偏航角生成多样化构型候选，增加规划成功冗余；
3. **双臂相对位姿协同约束 (Relative Pose Constraint)**：支持双手协同抓取刚性工件，施加闭环约束：
   $$T_L^{-1} T_R = T_{LR}^*$$
4. **可操作度主动优化 (Manipulability Maximization)**：将 Yoshikawa 可操作度度量梯度作为零空间任务，主动躲避机械臂奇异构型。

---

### 6.3 模块职责与边界速查表

| 模块组件 | 职责与归属 | 选型与第一版落地策略 |
|---|---|---|
| **RobotModel** | 机器人几何与连杆运动学树 | 使用 MoveIt 原生解析，不重造轮子 |
| **RobotState** | 关节状态存储与 FK 运算 | 使用 MoveIt 原生 `RobotState` |
| **Jacobian** | $12 \times 16$ 联合几何雅可比 | 由 `RobotState::getJacobian()` 分块组装 |
| **Pose Error** | SE(3) 位姿误差向量 | 本项目自主实现四元数连续误差映射 |
| **QP Solver** | 单步不等式二次规划内核 | 选用成熟高效开源求解器（ProxQP / OSQP） |
| **Joint Limit** | 物理硬限位与步长限制 | 本项目在 QP 盒式约束中严格硬性施加 |
| **Seed Tracking** | 构型邻域保持与解连续性 | 本项目在二次型优化项中自主构建 |
| **Collision** | 单步构型无碰撞合法性判定 | 外部调用 `PlanningScene::isStateValid()` 校验 |
| **OMPL** | 关节空间无碰撞连续路径规划 | 使用 MoveIt 2 官方集成，接收 $q_{goal}$ 作为输入 |
| **IK Plugin** | 供 MoveIt 调用的标准化包装 | V1.5 阶段继承 `KinematicsBase` 实现导出 |

---

### 6.4 实施步骤与关键排坑事项 (15 步实操清单)

工程落地推荐严格按以下 15 步推进：

```text
 ① 检查并确认 SRDF group 包含腰部 2 关节与双臂 14 关节 (共 16 DoF)
    │
 ② 严格确认 16 个 Joint 的内存索引顺序并形成常量映射表 (避免错位排查数天)
    │
 ③ 确认基座坐标系 (Base Frame) 与腰部各级连杆变换树
    │
 ④ 确认左右末端执行器 Tip Frame 名称 (如 left/right_wrist_roll_link)
    │
 ⑤ 编写独立单测，校验解析雅可比与数值差分雅可比
    │
 ⑥ 编写独立单测，测试空间位姿误差在各类极限旋转下的连续性
    │
 ⑦ 接入 QP 求解器 (集成 ProxQP 或 OSQP)，编写轻量 Dummy 测试
    │
 ⑧ 实现单步 QP 求解，验证单步更新后目标残差单调下降
    │
 ⑨ 搭建迭代循环求解状态机，加入收敛条件与最大步数保护
    │
 ⑩ 注入真实关节限位与单步步长盒式截断
    │
 ⑪ 注入种子姿态惩罚权重，验证构型连续性与反肘抑制
    │
 ⑫ 完善输出结构体，记录迭代次数、耗时与精准失败原因码
    │
 ⑬ 对接 PlanningScene 完成外部碰撞过滤机制
    │
 ⑭ 封装为 MoveIt Kinematics Plugin
    │
 ⑮ 运行 Benchmark 数据集，产出 KPI 报告与消融对比图表
```

> [!IMPORTANT]
> **最关键避坑点**：  
> 必须优先落实第 ② 步！**16 个关节的实际内存排列次序必须彻底锁定**。若底层代码中的顺序（如先腰后臂，或先左臂后右臂再腰）与 MoveIt JointModelGroup 的顺序不一致，会导致雅可比列对应错位，QP 将把左臂指令推给腰部，产生完全无法收敛的灾难性后果。