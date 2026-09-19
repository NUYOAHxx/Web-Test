# 宇树科技 G1 人形机器人控制与运动学系统

## 项目概述

本项目是基于宇树科技（Unitree）G1 人形机器人的控制与运动学全栈工程实现。系统涵盖 53 自由度 MuJoCo 物理仿真、10-DoF 躯干-手臂协同加权逆运动学（Weighted DLS IK）、28 对全机身自碰撞检测引擎、RRT* 三维空间路径规划，以及算展解耦的 ROS 2 与 Web 实时高精度数字孪生遥测监控大屏。

![01](resources%2Fg1%2Fimages%2F01_mujoco_g1_53dof.png)
![02](resources%2Fg1%2Fimages%2F02_mujoco_g1_53dof.png)

## 系统核心特性

- **10-DoF 躯干-手臂加权协同逆解**：支持 3-DoF 腰部与 7-DoF 手臂协同优化，优先动臂、按需弯腰转身，毫秒级收敛与动态收敛残差跟踪。
- **28 对几何自碰撞检测雷达**：基于 Pinocchio 与胶囊体/凸包几何构建 28 对安全碰撞对，实时预警与零空间避障约束。
- **Web 数字孪生实时遥测监控**：现代化工业级深色监控看板，实时展示 3D 骨骼位姿、收敛衰减曲线、关节角度仪表与自碰撞雷达。
- **53 自由度全身物理仿真**：支持双足行走、蹲起动作以及灵巧手精细开闭控制。
- **三维运动规划**：内置 3D RRT* 空间路径规划与五次多项式关节轨迹生成。

## 项目文件架构

```
UNITREE-G1-ROBOT-MODEL/
├── deploy/                     # 核心算法与业务实现模块
│   ├── collision/              # 28 对全机身几何自碰撞检测引擎 (g1_collision.py)
│   ├── config/                 # 机器人控制与关节配置 (g1_53.yaml)
│   ├── kinematics/             # 正运动学模型与关节常量 (g1_model.py)
│   ├── launch/                 # ROS 2 节点调度启动文件 (g1_telemetry_system.launch.py)
│   ├── planning/               # 空间路径规划算法 (rrt_star.py)
│   ├── simulation/             # MuJoCo 53-DoF 物理仿真与强化学习部署 (deploy_mujoco53.py)
│   ├── solver/                 # 10-DoF 逆解算法 (g1_hybrid_ik.py) 与 Headless 服务节点 (g1_ik_node.py)
│   └── visualizer/             # RViz 桌面可视化与 Web 遥测监控大屏 (web/)
├── output/                     # 诊断报表、轨迹对比图与测试输出
├── pre_train/                  # 预训练策略权重文件
│   └── g1/policy.pt
├── resources/                  # 机器人 URDF / MJCF 模型描述与 3D 网格资源
├── scripts/                    # 统一操作与服务启动脚本
│   ├── run_10dof_rviz.sh       # 启动 10-DoF RViz 3D 监控
│   ├── run_mujoco.sh           # 启动 MuJoCo 53-DoF 全身仿真
│   ├── run_telemetry_system.sh # 启动 ROS 2 Headless IK + Web 遥测系统
│   └── run_web_dashboard.sh    # 启动 Web 数字孪生仪表盘 (默认端口 8080)
├── tests/                      # 自动化单元测试、基准压力测试与 PTP 诊断工具
│   ├── test_10dof_ik.py        # 10-DoF 协同逆解验证套件
│   ├── test_ik_benchmark.py    # 大规模压力与位移距离基准评测
│   └── test_ptp_planning.py    # 点对点轨迹规划与奇异点敏感度分析
├── LICENSE                     # Apache 2.0 开源许可
├── README.md                   # 工程文档
├── pyproject.toml              # 项目依赖与包配置
└── requirements.txt            # Python 依赖清单
```

## 快速上手与运行指南

### 1. 环境准备
- Ubuntu Linux 22.04 / 24.04
- Python 3.10+ (推荐使用 `uv` 或虚拟环境)
- ROS 2 Jazzy / Humble (可选，用于完整 ROS 2 话题联动)

### 2. 核心系统启动

#### 方式一：启动 Web 实时高精度遥测监控仪表盘 (推荐)
```bash
bash scripts/run_web_dashboard.sh
# 浏览器访问 http://localhost:8080 即可进入数字孪生控制台
```

#### 方式二：ROS 2 完整调度启动 (Headless IK + Web 大屏 + TF 广播)
```bash
bash scripts/run_telemetry_system.sh
```

#### 方式三：启动 RViz 3D 桌面端交互监控
```bash
bash scripts/run_10dof_rviz.sh
```

#### 方式四：运行 MuJoCo 53-DoF 全身物理仿真
```bash
bash scripts/run_mujoco.sh
```

*(注：根目录下的 `run.sh`、`run_web_dashboard.sh` 等脚本提供便捷转发，可直接在根目录执行。)*

## 测试与诊断工具

项目中提供完备的算法测试与压力评估工具：

```bash
# 运行 10-DoF 协同逆解功能测试
python3 tests/test_10dof_ik.py

# 运行大规模随机逆解性能与收敛基准评测
python3 tests/test_ik_benchmark.py --samples 500

# 运行点对点 (PTP) 轨迹规划与 Z 轴扫描分析
python3 tests/test_ptp_planning.py --sweep-z
```

## 许可证

本项目遵循 [LICENSE](file:///home/parallels/Documents/myproj/UNITREE-G1-ROBOT-MODEL/LICENSE) 文件中指定的 Apache 2.0 许可条款。
