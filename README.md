# 宇树科技 Unitree G1 人形机器人运动学与全身控制系统

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![ROS 2](https://img.shields.io/badge/ROS%202-Humble%20%7C%20Jazzy-orange?logo=ros)
![Pinocchio](https://img.shields.io/badge/Dynamics-Pinocchio%203.x-red)
![Pink / ProxQP](https://img.shields.io/badge/IK%20Solver-Inria%20Pink%20%2F%20ProxQP-brightgreen)
![MuJoCo](https://img.shields.io/badge/Physics-MuJoCo%2053--DoF-blueviolet)
![Three.js](https://img.shields.io/badge/Web%20Vis-Three.js%20r128-black)

</div>

---

## 目录

- [1. 项目概述](#1-项目概述)
- [2. 系统核心架构与特性](#2-系统核心架构与特性)
  - [2.1 10-DoF 躯干-手臂协同加权逆运动学 (Hybrid IK)](#21-10-dof-躯干-手臂协同加权逆运动学-hybrid-ik)
  - [2.2 MoveIt SRDF ACM 工业级全域自碰撞安全引擎](#22-moveit-srdf-acm-工业级全域自碰撞安全引擎)
  - [2.3 Web 数字孪生实时遥测监控大屏](#23-web-数字孪生实时遥测监控大屏)
  - [2.4 MuJoCo 53-DoF 全身物理仿真与策略部署](#24-mujoco-53-dof-全身物理仿真与策略部署)
  - [2.5 MoveIt 2 OMPL 工业级运动规划与空间避障](#25-moveit-2-ompl-工业级运动规划与空间避障)
- [3. 项目工程文件架构](#3-项目工程文件架构)
- [4. ROS 2 通信总线与标准接口规范](#4-ros-2-通信总线与标准接口规范)
- [5. 环境准备与依赖安装](#5-环境准备与依赖安装)
- [6. 快速上手与运行指南](#6-快速上手与运行指南)
  - [方式一：启动 Web 实时数字孪生监控仪表盘 (推荐)](#方式一启动-web-实时数字孪生监控仪表盘-推荐)
  - [方式二：ROS 2 完整调度启动 (Headless IK + Web 大屏 + TF 广播)](#方式二ros-2-完整调度启动-headless-ik--web-大屏--tf-广播)
  - [方式三：启动 RViz 3D 桌面端交互监控](#方式三启动-rviz-3d-桌面端交互监控)
  - [方式四：运行 MuJoCo 53-DoF 全身物理仿真](#方式四运行-mujoco-53-dof-全身物理仿真)
  - [方式五：启动 MoveIt 2 OMPL 运动规划流水线 (RRTConnect / RRT* / PRM)](#方式五启动-moveit-2-ompl-运动规划流水线-rrtconnect--rrt--prm)
- [7. 自动化测试与基准性能评测](#7-自动化测试与基准性能评测)
  - [7.1 自动化单元测试套件 (pytest)](#71-自动化单元测试套件-pytest)
  - [7.2 大规模随机逆解基准压力测试](#72-大规模随机逆解基准压力测试)
  - [7.3 点对点 (PTP) 轨迹规划与工作空间灵敏度分析](#73-点对点-ptp-轨迹规划与工作空间灵敏度分析)
  - [7.4 MoveIt 2 OMPL 全局运动规划与动态避障基准评测](#74-moveit-2-ompl-全局运动规划与动态避障基准评测)
- [8. 常见问题排查 (FAQ)](#8-常见问题排查-faq)

---

## 1. 项目概述

本项目是面向**宇树科技（Unitree）G1 人形机器人**的高性能运动学求解、安全防碰撞计算、全息数字孪生遥测与全身物理仿真的工业级全栈工程套件。

项目彻底重构了传统机器人开发中算法与可视化高度耦合的单体架构，采用**算展彻底解耦**（Headless 求解节点与被动展示大屏分离）的设计哲学：
1. **高性能核心算法库**：基于 Inria Pinocchio 刚体动力学库、Pink 流形优化器与 ProxQP 凸二次规划求解器，提供亚毫米级精度、微秒级求解的 10-DoF 躯干-手臂协同加权逆解；
2. **MoveIt SRDF 碰撞安全引擎**：复用工业级 MoveIt SRDF Allowed Collision Matrix (ACM) 规范，构建 28 对全机身物理干涉雷达与 ~40μs 极速降维二值门禁；
3. **Web 数字孪生高精度大屏**：基于 Three.js 与纯 CSS 现代深色仪表盘，通过 Server-Sent Events (SSE) 高频推流，零计算负载实时还原 3D 姿态、碰撞态势与算法诊断指标；
4. **MuJoCo 53-DoF 仿真支持**：支持双足运动强化学习策略部署、灵巧手开闭与三维 RRT* 规划。

<div align="center">

![01](resources/g1/images/01_mujoco_g1_53dof.png)
*图 1：Unitree G1 53-DoF MuJoCo 动力学仿真与双足站立姿态*

![02](resources/g1/images/02_mujoco_g1_53dof.png)
*图 2：G1 灵巧手与双臂末端操作作业示意*

</div>

---

## 2. 系统核心架构与特性

系统遵循松耦合、高复用、模块化工业架构设计，整体信息流与架构设计如下图所示：

```
                      ┌────────────────────────────────────────┐
                      │    Web 控制台 / 上层任务调度系统       │
                      └──────────────────┬─────────────────────┘
                                         │ HTTP POST /api/send_target
                                         ▼
                      ┌────────────────────────────────────────┐
                      │    Web Telemetry Hub (server.py)       │
                      │    工作空间物理安全包络防御校验 (Safe Reach)│
                      └──────────────────┬─────────────────────┘
                                         │ ROS 2: /g1/kinematics/target_pose
                                         ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│ Headless 核心求解节点 (core/solver/g1_ik_node.py)                                 │
│                                                                                  │
│ ┌───────────────────────────────────────┐  ┌───────────────────────────────────┐ │
│ │ 10-DoF 躯干-手臂协同加权求解器        │  │ MoveIt SRDF ACM 碰撞安全引擎      │ │
│ │ (core/solver/g1_hybrid_ik.py)         │  │ (core/collision/g1_collision.py)  │ │
│ │                                       │  │                                   │ │
│ │ • Stage 1: Lie-Manifold Configuration │  │ • MoveIt SRDF ACM 自动解析        │ │
│ │ • Stage 2: SE(3) FrameTask (位置+旋转)│  │ • 28 对物理风险干涉对拓扑        │ │
│ │ • Stage 3: 加权 PostureTask (动臂优先)│  │ • 10-DoF / 7-DoF 剪枝轻量化子模型 │ │
│ │ • Stage 4: Inria ProxQP 二次规划求解  │  │ • ~40μs 极速降维门禁判定          │ │
│ │ • Stage 5: 双阶段级联直立姿态恢复     │  │ • Pink SelfCollisionBarrier 兼容 │ │
│ └───────────────────────────────────────┘  └───────────────────────────────────┘ │
└────────────────────────────────────────┬─────────────────────────────────────────┘
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 │ 发布: /joint_states                           │ 发布: /g1/safety/collision_status
                 │ 发布: /g1/kinematics/actual_pose              │ 发布: /g1/kinematics/solver_metrics
                 ▼                                               ▼
┌─────────────────────────────────────────┐     ┌─────────────────────────────────────────┐
│     Web 实时遥测大屏 (Three.js)         │     │     ROS 2 RViz2 桌面端 3D 监控          │
│ • SSE 单向流式推流 (60Hz 刷新)          │     │ • robot_state_publisher 全机身 TF 广播 │
│ • 36 个官方 STL 视觉网格真实渲染        │     │ • 交互式 Marker / 质心投影 (CoM)        │
│ • 28 对安全雷达态势感知 / 收敛误差曲线  │     │ • 终端数据监控仪表盘                    │
└─────────────────────────────────────────┘     └─────────────────────────────────────────┘
```

### 2.1 10-DoF 躯干-手臂协同加权逆运动学 (Hybrid IK)

- **运动链构型**：3 自由度腰部（`waist_yaw`, `waist_roll`, `waist_pitch`）+ 7 自由度机械臂（`shoulder_pitch`, `shoulder_roll`, `shoulder_yaw`, `elbow`, `wrist_roll`, `wrist_pitch`, `wrist_yaw`），形成 10-DoF 冗余运动系统。
- **动臂优先原则 (Arm Priority)**：腰部关节赋予高代价权重（$w_{\text{waist}} \gg w_{\text{arm}}$），在手臂工作空间内仅依靠 7-DoF 手臂运动；当目标越出单臂极限包络时，腰部按需平滑介入协同完成大跨度作业。
- **级联直立姿态恢复机制 (Cascade Standing Return)**：在任务点从远端协同区域返回至单臂可达区域时，算法自动以双阶段级联模式触发腰部回正直立，防止躯干维持不必要的倾斜姿态。
- **Pink + ProxQP 工业标准求解流水线**：
  1. **Configuration**：在李群流形切空间中建立关节硬限位（0% 越界保证）；
  2. **FrameTask**：末端执行器 SE(3) 位置误差与李代数对数旋转测地线残差约束；
  3. **PostureTask**：各关节二次型运动代价加权与就绪预备姿态引力；
  4. **ProxQP Solver**：基于高阶原对偶凸二次规划算法进行微秒级约束最优化计算；
  5. **Safety Barrier**：自碰撞屏障函数与全域安全门禁检验。
- **关键性能指标**：平均求解耗时 1.5 ~ 2.5 ms，位置残差 < 0.5 mm，姿态残差 < 0.2°，限位越界率严格为 0.00%。

### 2.2 MoveIt SRDF ACM 工业级全域自碰撞安全引擎

- **工业标准兼容**：完全复用国际通用 MoveIt 语义机器人描述格式（`resources/g1/g1_29dof.srdf`）与 Allowed Collision Matrix (ACM)。
- **全域 28 对物理风险拓扑**：
  - **手臂与躯干/腰架/骨盆干涉**（`arm_torso`，12 对）：防止大臂内收打胸或手肘触碰腰部支架；
  - **双臂交叉碰撞干涉**（`arm_arm`，4 对）：防止双手胸前抱胸、交接物品时发生碰撞；
  - **手臂与头部干涉**（`arm_head`，4 对）：防止摸头、擦拭面部或举手碰撞；
  - **手臂与下肢大腿干涉**（`arm_leg`，8 对）：防止深蹲或极度弯腰时手掌接触大腿。
- **Pinocchio 几何模型动态减枝**：初始化时自动解析 SRDF ACM 禁用规则，剔除相邻固定连杆和无需检测的静态结构对。
- **极速降维门禁 (~40μs)**：利用 `pin.buildReducedModel` 提取轻量化碰撞几何图，单次检测开销仅 ~40 微秒，可直接嵌入 1kHz 控制循环。

### 2.3 Web 数字孪生实时遥测监控大屏

- **算展彻底解耦**：前端与 Web 服务端只负责遥测展示与指令分发，不进行任何 IK 计算；高负荷求解由后台 Headless ROS 2 节点负责，确保高负载下界面 60FPS 绝对流畅。
- **高保真 3D 渲染**：基于 Three.js 与 STLLoader 加载官方 36 个 STL 精确 CAD 外形网格，支持轨道相机控制（旋转、缩放、平移）、骨骼关节点可视化与末端轨迹跟踪。
- **全息监控报表**：
  - **算法收敛残差**：毫秒级实时刷新残差衰减折线图；
  - **关节状态仪表**：直观展示 29 个关节角、物理软硬件限位指示条与越界警戒；
  - **28 对碰撞雷达**：红/黄/绿三级实时显示各关键区域安全间隙（毫米级）；
  - **平衡与支撑质心**：实时投影机器人质心（CoM）与双足支撑多边形安全裕度。
- **防御性指令分发**：内置安全包络防御校验（拦截 NaN、越界值与不可达位置），提供一键预设姿态测试。

### 2.4 MuJoCo 53-DoF 全身物理仿真与策略部署

- 仿真模型基于 `resources/g1/scene53.xml`，完整包含 G1 机身 53 自由度刚体拓扑。
- 集成预训练强化学习策略权重（`simulation/policy/policy.pt`），支持双足稳定行走与速度向量跟踪。
- 支持三维灵巧手精细开闭、双臂联动与重力补偿。

### 2.5 MoveIt 2 OMPL 工业级运动规划与空间避障

- **官方 OMPL 规划管道无缝集成**：内置封装了 MoveIt 2 的完整 OMPL 运动规划流水线（`core/planning/moveit_ompl_planner.py`），提供高层零开销 Python 客户端接口 `MoveItOMPLPlanner`。
- **多规划组与算法支持**：
  - 支持 7-DoF（`left_arm`, `right_arm`）、10-DoF（`left_arm_torso`, `right_arm_torso`）及 14-DoF（`both_arms`）规划组；
  - 深度支持 **RRTConnect**（毫秒级极速求解）、**RRT\***（路径渐进最优）与 **PRM**（概率路线图全局重用）等多种采样规划算法。
- **动态碰撞障碍物注入与在线避障**：
  - 支持在规划场景中动态创建、更新与清除障碍物几何图元（如立方体 Box、圆柱体 Cylinder）；
  - OMPL 规划管道自动利用 SRDF 与连续碰撞干涉判定，生成平滑规避静态及动态障碍物的空间关节轨迹。
- **时间参数化与平滑插值**：
  - 自动经由 TOTG (Time-Optimal Trajectory Generation) 或五次多项式（Quintic Spline）插值，输出具备连续平滑速度与加速度的工业级安全可执行轨迹。
- **轻量独立三维规划**：保留极简 3D 笛卡尔空间 RRT* 规划器（`core/planning/rrt_star.py`），支持纯几何环境离线路径推演。

---

## 3. 项目工程文件架构

```
UNITREE-G1-ROBOT-MODEL/
├── core/                           # 核心算法与计算引擎
│   ├── kinematics/                 # 运动学模型与关节常量
│   │   ├── __init__.py             # 导出运动学 API
│   │   └── g1_model.py             # Pinocchio 模型封装、SRDF 加载与 10-DoF 轻量化子模型
│   ├── collision/                  # MoveIt SRDF ACM 碰撞安全检测引擎
│   │   ├── __init__.py             # 导出碰撞检测器
│   │   └── g1_collision.py         # 28 对全机身自碰撞检测与 ~40μs 极速降维门禁
│   ├── solver/                     # 逆运动学算法与 ROS 2 服务节点
│   │   ├── __init__.py             # 导出求解器
│   │   ├── g1_hybrid_ik.py         # Pink + ProxQP 10-DoF 协同加权逆解与级联直立姿态
│   │   └── g1_ik_node.py           # ROS 2 Headless 核心逆解服务节点
│   └── planning/                   # 空间路径与运动规划
│       ├── __init__.py             # 导出规划器 (MoveItOMPLPlanner, RRTStar3D)
│       ├── moveit_ompl_planner.py  # MoveIt 2 OMPL 工业级规划客户端封装
│       └── rrt_star.py             # 三维空间 RRT* 路径规划算法
├── visualizer/                     # 可视化与监控系统
│   ├── web/                        # Web 实时高精度数字孪生遥测看板
│   │   ├── server.py               # 遥测推流 (SSE)、HTTP 服务与指令分发
│   │   ├── index.html              # 现代化工业级深色看板界面
│   │   ├── app.js                  # Three.js 场景渲染与实时数据驱动
│   │   ├── style.css               # 玻璃拟态与响应式仪表盘样式
│   │   └── meshes -> ../../resources/g1/meshes # 3D STL CAD 网格符号链接
│   └── rviz/                       # ROS 2 RViz 3D 监控
│       ├── __init__.py             # 导出 Marker 构造函数
│       ├── markers.py              # RViz Marker 与几何图元构造工具
│       ├── visualize_10dof_rviz.py # 交互式 10-DoF 解算终端监控控制台
│       └── view_10dof_ik.rviz      # RViz2 预设可视化配置文件
├── simulation/                     # 物理仿真与强化学习策略
│   ├── __init__.py                 # 导出仿真配置
│   ├── deploy_mujoco53.py          # MuJoCo 53-DoF 全身仿真运行入口
│   └── policy/                     # 预训练策略权重目录
│       └── policy.pt               # 预训练双足强化学习控制策略
├── config/                         # 配置文件
│   ├── g1_53.yaml                  # MuJoCo 53-DoF 仿真参数与关节增益配置
│   └── moveit/                     # MoveIt 2 官方标准运动规划与控制器参数
│       ├── ompl_planning.yaml      # OMPL 规划算法 (RRTConnect, RRT*, PRM 等) 参数
│       ├── kinematics.yaml         # KDL / Kinematics 求解参数
│       ├── joint_limits.yaml       # 关节速度、加速度限位与安全系数
│       ├── initial_positions.yaml  # 机器人各规划组预设初始位姿
│       ├── moveit_controllers.yaml # MoveIt 控制器接口映射
│       └── moveit.rviz             # MoveIt 2 官方 RViz 预设界面配置
├── launch/                         # ROS 2 系统级调度启动入口
│   ├── g1_telemetry_system.launch.py # 标准数字孪生全系统启动脚本
│   └── g1_moveit_ompl.launch.py      # MoveIt 2 MoveGroup + OMPL 规划服务启动入口
├── resources/                      # 机器人描述资源
│   └── g1/
│       ├── g1_29dof.urdf           # 宇树 G1 29-DoF 机器人官方 URDF 模型
│       ├── g1_29dof.srdf           # MoveIt SRDF 碰撞矩阵 (ACM) 配置文件
│       ├── scene53.xml             # MuJoCo 53-DoF 全身物理仿真场景描述
│       ├── meshes/                 # 官方高精度 STL CAD 3D 视觉网格 (36 个)
│       └── images/                 # 项目展示图
├── scripts/                        # 统一操作与服务一键启动脚本
│   ├── run_web_dashboard.sh        # 启动 Web 数字孪生仪表盘 (默认端口 8080)
│   ├── run_telemetry_system.sh     # 启动 ROS 2 Headless IK + Web 遥测系统
│   ├── run_10dof_rviz.sh           # 启动 10-DoF RViz 3D 监控窗口与控制台
│   ├── run_mujoco.sh               # 启动 MuJoCo 53-DoF 全身物理仿真
│   └── run_moveit_ompl.sh          # 一键启动 MoveIt 2 OMPL 规划服务 (可选 --rviz)
├── benchmarks/                     # 性能压测与轨迹规划基准测试
│   ├── ik_benchmark.py             # 大规模随机逆解基准压力评测工具
│   ├── ptp_planning.py             # 点对点 (PTP) 轨迹规划与灵敏度分析工具
│   └── ompl_planning_demo.py       # MoveIt 2 OMPL 规划基准与动态避障演示脚本
├── tests/                          # 自动化单元测试套件 (pytest)
│   ├── __init__.py
│   ├── test_10dof_ik.py            # 10-DoF 协同逆解验证套件 (12 项测试)
│   ├── test_6dof_ik.py             # 6-DoF 全空间位姿逆解测试
│   ├── test_collision_engine.py    # MoveIt SRDF ACM 碰撞安全引擎测试
│   └── test_pinocchio_dynamics.py  # 动力学与正运动学等价性验证
├── pyproject.toml                  # Python 工程构建与依赖配置
└── requirements.txt                # Python 依赖清单
```

---

## 4. ROS 2 通信总线与标准接口规范

系统在 ROS 2 环境中提供了标准化的话题与数据总线定义，各模块间数据流清晰、规范：

| 话题名称 | 消息类型 (Type) | 发布者 (Publisher) | 订阅者 (Subscriber) | 说明 |
|---|---|---|---|---|
| `/g1/kinematics/target_pose` | `geometry_msgs/msg/PoseStamped` | Web Server / 用户控制脚本 | `g1_ik_solver` | 外部输入的 6D 空间末端笛卡尔目标位姿 |
| `/g1/kinematics/actual_pose` | `geometry_msgs/msg/PoseStamped` | `g1_ik_solver` | Web Server / RViz | 当前解算出的实际末端位姿反馈 |
| `/joint_states` | `sensor_msgs/msg/JointState` | `g1_ik_solver` | `robot_state_publisher`, Web Server, RViz | 全机身 29 关节当前角度广播与 TF 驱动 |
| `/g1/kinematics/solver_metrics` | `std_msgs/msg/String` (JSON) | `g1_ik_solver` | Web Server | 包含求解耗时、迭代步数、残差、直通命中率等算法度量 |
| `/g1/safety/collision_status` | `std_msgs/msg/String` (JSON) | `g1_ik_solver` | Web Server | 28 对碰撞检测结果、各区域毫米级安全净空与预警等级 |
| `/g1/ik/markers` | `visualization_msgs/msg/MarkerArray` | `g1_ik_solver` | RViz2 | 末端目标框、求解点、连杆射线与平衡投影 Marker |

---

## 5. 环境准备与依赖安装

### 5.1 系统要求
- **操作系统**：Ubuntu 22.04 LTS 或 24.04 LTS
- **Python 环境**：Python 3.10+ (推荐使用 Python 3.12 虚拟环境)
- **ROS 2 环境**（可选）：ROS 2 Jazzy 或 Humble（若未安装 ROS 2，系统亦可通过纯 Python 进程间管道自动降级独立运行）

### 5.2 依赖安装

推荐使用 `uv` 或原生 `venv` 创建隔离的虚拟环境：

```bash
# 1. 克隆代码仓库
git clone https://github.com/NUYOAHxx/Web-Test.git
cd UNITREE-G1-ROBOT-MODEL

# 2. 创建并激活 Python 虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 3. 安装项目依赖
pip install -r requirements.txt
```

核心依赖包括：
- `pinocchio`：刚体动力学计算库
- `pin-pink`：基于李群流形的机器人任务级逆运动学优化器
- `proxsuite`：Inria 工业级稀疏/稠密凸二次规划求解器 (ProxQP)
- `mujoco`：物理仿真引擎
- `numpy`, `scipy`, `pytest`, `pyyaml`

---

## 6. 快速上手与运行指南

### 方式一：启动 Web 实时数字孪生监控仪表盘 (推荐)

此命令为日常开发与调试推荐方式，支持一键拉起后台 Headless IK 求解服务节点与 Web 前端数字孪生看板：

```bash
bash scripts/run_web_dashboard.sh [PORT]
```
- 默认端口为 `8080`（例如指定端口：`bash scripts/run_web_dashboard.sh 9090`）。
- 启动后浏览器访问：**`http://localhost:8080`**。
- 可直接在 3D 视口中进行三维拖拽交互，或通过预设点控制机器人手臂并实时监控各关节状态与防碰撞安全雷达。

---

### 方式二：ROS 2 完整调度启动 (Headless IK + Web 大屏 + TF 广播)

在已配置好 ROS 2 环境的终端中，使用标准 ROS 2 Launch 调度一键启动整套系统：

```bash
bash scripts/run_telemetry_system.sh
```

或直接执行 ROS 2 Launch 命令：

```bash
ros2 launch launch/g1_telemetry_system.launch.py port:=8080 rviz:=false
```

启动组件包含：
1. `g1_ik_solver`：核心 10-DoF 协同求解与 28 对自碰撞安全门禁服务节点；
2. `robot_state_publisher`：广播全机身 TF 坐标树；
3. `g1_telemetry_hub`：Web 遥测监控大屏服务；
4. `rviz2`（可选）：传入 `rviz:=true` 可同步启动桌面端 RViz2 监控。

---

### 方式三：启动 RViz 3D 桌面端交互监控

若需要在 ROS 2 桌面环境中进行算法交互与调试，执行：

```bash
bash scripts/run_10dof_rviz.sh
```

脚本将自动检查并拉起 `robot_state_publisher`、载入 `visualizer/rviz/view_10dof_ik.rviz` 预设，并打开终端数据交互控制台。

---

### 方式四：运行 MuJoCo 53-DoF 全身物理仿真

运行 53 自由度 MuJoCo 物理仿真，加载官方场景与预训练双足平衡运动控制策略：

```bash
bash scripts/run_mujoco.sh
```

- 仿真配置文件位于 `config/g1_53.yaml`。
- 策略模型权重位于 `simulation/policy/policy.pt`。

---

### 方式五：启动 MoveIt 2 OMPL 运动规划流水线 (RRTConnect / RRT* / PRM)

在 ROS 2 环境中一键启动 MoveIt 2 `move_group` 运动规划服务与 OMPL 规划管道：

```bash
# 无头 Headless 模式 (推荐作为服务端或与其他节点联动时使用)
bash scripts/run_moveit_ompl.sh

# 带 RViz2 桌面端 3D 交互界面
bash scripts/run_moveit_ompl.sh --rviz
```

启动后提供标准 MoveIt 2 服务接口：
- `/plan_kinematic_path` (`moveit_msgs/srv/GetMotionPlan`)：运动学无碰撞轨迹规划；
- `/apply_planning_scene` (`moveit_msgs/srv/ApplyPlanningScene`)：动态添加/清除碰撞障碍物；
- `/compute_ik` & `/compute_fk`：运动学解算。

---

## 7. 自动化测试与基准性能评测

项目中提供完整的自动化单元测试、算法极限压测与轨迹规划诊断工具。

### 7.1 自动化单元测试套件 (pytest)

执行以下命令运行全量单元测试：

```bash
pytest tests/
```

测试套件包含 12 项高覆盖率测试，涵盖：
- `test_10dof_ik.py`：
  - 近距离点手臂优先与腰部能量惩罚机制；
  - 远距离大跨度作业中 10-DoF 躯干协同与收敛残差；
  - 级联直立姿态平滑恢复逻辑；
  - 极速冷启动与热启动求解一致性。
- `test_collision_engine.py`：
  - MoveIt SRDF ACM 28 对干涉矩阵拓扑完整性校验；
  - ~40μs 极速降维门禁判定与连续欧氏安全间隙计算；
  - 极端扭曲交叉姿态下的自干涉穿透精准拦截。
- `test_6dof_ik.py`：
  - 全空间 6-DoF 位置与 RPY 姿态欧拉角多轴约束逆解精度。
- `test_pinocchio_dynamics.py`：
  - 10-DoF 轻量裁剪子模型与全机身模型的正运动学/雅可比等价性。

---

### 7.2 大规模随机逆解基准压力测试

通过蒙特卡洛随机采样，对算法进行全局可达工作空间压力测试与多距离段对比：

```bash
python3 benchmarks/ik_benchmark.py --samples 500
```

实测输出报表示例：

```text
================================================================================
【测试 2】随机起点-终点对点规划对比评测 (left_arm, 测试对数: 500)
================================================================================
位移距离区间       | 样本量 | 纯单初猜局部 DLS            | G1 混合多级求解器 (Hybrid IK)
                  |        | 成功率   平均步数  平均耗时  | 成功率   平均步数  平均耗时   热启命中率
--------------------------------------------------------------------------------------------
超短距离 (< 5cm)  | 54     | 100.0%    2.1步   0.04ms    | 100.0%    1.0步   0.28ms   (100.0%直通)
中短距离 (5-15cm) | 128    |  98.4%    6.2步   0.11ms    | 100.0%    4.1步   0.72ms   ( 89.1%直通)
中长距离 (15-30cm)| 186    |  81.2%   12.4步   0.24ms    |  99.5%    8.7步   1.48ms   ( 58.6%直通)
极限跨度 (>= 30cm)| 132    |  47.7%   28.1步   0.52ms    |  98.5%   13.2步   2.16ms   ( 31.8%直通)
--------------------------------------------------------------------------------------------
```

---

### 7.3 点对点 (PTP) 轨迹规划与工作空间灵敏度分析

分析机械臂从起始姿态到目标姿态的奇异点与 Z 轴连续扫描边界：

```bash
# 连续扫描不同起始 Z 坐标并输出成功/失败边界表
python3 benchmarks/ptp_planning.py --sweep-z

# 生成轨迹规划对比分析曲线图
python3 benchmarks/ptp_planning.py --sweep-z --plot
```

---

### 7.4 MoveIt 2 OMPL 全局运动规划与动态避障基准评测

在后台启动 MoveIt 2 服务后（`bash scripts/run_moveit_ompl.sh`），执行以下演示与性能基准脚本：

```bash
python3 benchmarks/ompl_planning_demo.py
```

该基准涵盖 7-DoF、10-DoF 多自由度规划、动态碰撞障碍物注入在线绕行避障以及 RRTConnect / RRT* 算法对比评测：

```text
==============================================================================================================
                                G1 MoveIt 2 OMPL 运动规划性能基准报表
==============================================================================================================
测试场景                             规划组          算法        规划状态   规划耗时(ms) 路径路标点 轨迹总时长(s)
--------------------------------------------------------------------------------------------------------------
7-DoF 左臂自由空间快速规划           left_arm        RRTConnect  成功       29.89 ms     9 个点     0.74 s
10-DoF 躯干-手臂协同大目标规划       left_arm_torso  RRTConnect  成功       24.13 ms     10 个点    0.85 s
7-DoF 动态障碍物单臂绕行避障         left_arm        RRTConnect  成功       399.14 ms    19 个点    1.76 s
10-DoF 躯干协同大冗余极速避障        left_arm_torso  RRTConnect  成功       54.53 ms     15 个点    1.36 s
渐进最优路径优化评测                 left_arm        RRT*        成功       5023.39 ms   8 个点     0.67 s
==============================================================================================================
```

> **注**：
> 1. **双向扩展树极速收敛**：RRTConnect 算法在几十毫秒（自由空间 ~25ms，障碍物环境 ~50-400ms）内即可极速规划成功，完全满足工业节拍在线要求；
> 2. **10-DoF 协同避障优势**：面对路径障碍物，10-DoF 躯干协同规划耗时仅需 **54.53 ms**（相比单臂 7-DoF 的 399.14 ms 速度提升超 **7 倍**），充分体现了大冗余度解耦的优势；
> 3. **渐进最优算法**：RRT* 在充分优化时间预算下，能收敛出更少路标点（8 点）与更短轨迹耗时的最优几何路径。

---

## 8. 常见问题排查 (FAQ)

### Q1: 启动 Web 看板提示端口被占用 (Address already in use)？
脚本 `run_web_dashboard.sh` 已内置自动检测与端口释放机制。若需要手动释放端口，可执行：
```bash
fuser -k 8080/tcp
```

### Q2: 浏览器打开后 3D 模型显示不全或 404 (STL Loader Error)？
请检查 `visualizer/web/meshes` 软链接是否有效：
```bash
ls -la visualizer/web/meshes
# 正确链接应指向 ../../resources/g1/meshes
```
若失效，可手动重建：
```bash
rm -f visualizer/web/meshes
ln -s ../../resources/g1/meshes visualizer/web/meshes
```

### Q3: 为什么单臂求解和 10-DoF 协同求解结果不一致？
- 在目标点位于单臂极限工作空间（约半径 0.38m）内时，算法优先只动用 7-DoF 手臂，腰部保持直立；
- 当目标点超出单臂范围时，算法自动调度腰部 3-DoF（侧倾、旋转、俯仰）协同扩展可达空间；
- 当从协同点回到近处时，算法通过**级联直立姿态**引导腰部平滑复位至 0 度站立姿态。
