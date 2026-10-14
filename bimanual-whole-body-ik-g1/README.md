# 双臂机器人全身 IK（Whole-Body Bimanual IK）

一个可安装、可运行的 Python 项目，用于**双臂 + 可动躯干**的全身逆运动学。它区分两种不同任务：

1. **独立手目标**：左右手各自跟踪 6D 位姿，由一次联合优化求解；躯干自由度与两臂关节共同参与。
2. **共同搬运刚体**：由一个物体目标和两组固定抓取变换生成左右手目标，并将相对手位姿作为**硬闭链等式约束**，防止双手在搬运中相互挤压、拉开或扭转。

默认模型已改为 **Unitree G1 29-DoF rev_1_0 的 16-DoF 降阶模型**：固定 pelvis 基座、腰 yaw/pitch 两自由度、左右手臂各 7 自由度。腿部不参与 IK，`waist_roll_joint` 锁定为 0；末端 TCP 按 G1 rubber-hand 固定偏移并额外沿手掌前方偏移 0.08 m。关节轴、固定变换、位置限位与速度上限来自参考项目中的 Unitree URDF 降阶配置。模型参数见 `config/g1.json`，来源许可见 `assets/g1_description/LICENSE_unitree_ros`。

这是 **G1 的运动学模型，不是完整 G1 仿真/硬件栈**：没有复制完整 URDF 网格，也不含 G1 精确自碰撞、静力矩或动力学校验。球形障碍点采样和质心项仍是近似原型功能；固定 pelvis/腿部意味着不能求解腿部或浮动基座运动。

## 安装

需要 Python 3.10+。在项目根目录执行：

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\\Scripts\\activate
python -m pip install -e '.[dev]'
```

## 运行

```bash
# 同时运行两个示例并打印 JSON 结果
bimanual-ik-demo

# 单独运行一个模式，并可选保存结果
bimanual-ik-demo --mode independent --output out/independent.json
bimanual-ik-demo --mode carry --output out/carry.json

# 从源码运行
PYTHONPATH=src python -m bimanual_wbik.demo --mode both
python examples/quickstart.py

# 多目标姿态与连续轨迹基准（全量场景）
PYTHONPATH=src python -m bimanual_wbik.benchmark --output-dir results/g1-ik-benchmark

# 快速冒烟基准
PYTHONPATH=src python -m bimanual_wbik.benchmark --quick --output-dir results/g1-ik-benchmark-quick
```

测试：

```bash
python -m pytest
```

成功结果给出关节解、优化状态、左右手位置/姿态误差、闭链误差和迭代次数。CLI 对任一 IK 子任务失败会返回非零退出码。

基准工具覆盖独立双手的 10 组 G1 可达姿态、共同搬运的 8 组姿态，以及两种模式各 2 条连续轨迹。轨迹逐点以先前解作为 warm start。它输出 `ik_benchmark.json`（汇总及每帧完整轨迹）、`ik_benchmark_samples.csv`（便于表格/绘图分析）和 `ik_benchmark_report.md`。记录的指标包括成功率、求解耗时均值/中位数/P95/最大值、迭代次数、末端位置/姿态误差、闭链误差、关节限位余量、左右手目标/实际路径长度、各关节总行程、最大单步变化和离散加速度平滑度代理。G1 活动关节数据均为弧度；这是离线模型回归，不是实机实时性能或安全验证。

基准即使遇到失败也会保留完整逐样本结果和失败列表；只要任一目标失败，命令就以非零状态退出，便于 CI 发现回归。

## Python API

```python
from bimanual_wbik import GraspPair, WholeBodyIK, g1_robot
from bimanual_wbik.math3d import transform

robot = g1_robot()  # default_robot() 也返回同一 G1 模型
q = robot.nominal.copy()  # 硬件接入时替换为按 robot.names 排列的实测关节状态
ik = WholeBodyIK(robot)

# 任务 A：左右手各自跟踪目标
left_target = robot.fk(q, "left").copy()
right_target = robot.fk(q, "right").copy()
independent = ik.solve_independent(q, left_target, right_target)

# 任务 B：共同持有物体。抓取变换须由标定或抓取时的当前状态得到。
object_home = GraspPair.infer_object_pose(robot, q)  # 示例：两手中点作为物体参考
grasps = GraspPair.from_current(robot, q, object_home)
object_target = object_home @ transform(xyz=(0.04, 0.02, 0.02), rpy=(0.0, 0.0, 0.04))
carry = ik.solve_object_carry(q, object_target, grasps)
print(carry.as_dict(robot.names))
```

### 闭链公式

抓取变换定义为 `T_object_tool`，物体目标为 `T_world_object`。两手目标由同一个物体位姿生成：

```text
T_world_left  = T_world_object · T_object_left
T_world_right = T_world_object · T_object_right
```

优化期间施加非线性等式：

```text
T_left⁻¹ · T_right = T_object_left⁻¹ · T_object_right
```

该约束与物体绝对位姿跟踪分开处理，因此闭链不是一个可被低权重目标“牺牲掉”的软项。不可达目标、约束冲突或优化器失败会通过 `IKResult.success`/`message` 暴露；上层控制器应安全减速或停止。

### 全身约束和辅助项

- 每个关节都配置位置限位与速度上限。
- `dt=...` 限制单步关节位移不超过速度上限 × `dt`；也可用 `max_step` 指定每次 IK 调用的增量上限。
- `posture_reference` / `posture_weight` 让剩余冗余自由度偏向期望姿态；默认参考当前姿态。
- `SphereObstacle` 与机器人关节/末端采样点组成避障不等式约束。G1 降阶模型没有加载精确连杆网格，因此该功能只是点半径近似，不替代 mesh/胶囊体碰撞检测。
- 可选 `BalanceRegion` 把可配置的质心代理投影限制在凸形支撑多边形内；共同搬运可用 `payload_mass` 把物体质量纳入代理。它不是 G1 URDF 惯性计算、动力学或零力矩点校验。
- 末端雅可比采用世界坐标系几何雅可比，行序为 `[vx, vy, vz, wx, wy, wz]`。

示例：

```python
from bimanual_wbik import BalanceRegion, SphereObstacle

support = BalanceRegion(((-0.35, -0.25), (0.35, -0.25),
                         (0.35, 0.25), (-0.35, 0.25)), margin=0.02)
obstacles = [SphereObstacle(center=(0.45, 0.0, 0.9), radius=0.08)]
result = ik.solve_independent(q, left_target, right_target,
                              dt=0.02, obstacles=obstacles,
                              balance_region=support)
```

## 模型配置

`config/g1.json` 是可读的 G1 16-DoF 关节配置；`config/default_robot.json` 和 Python 包内默认文件保持同一配置，以兼容已有路径。可从 JSON 加载模型：

```python
from bimanual_wbik import RobotModel, WholeBodyIK
robot = RobotModel.from_json("config/g1.json")
ik = WholeBodyIK(robot)
```

约定：位置单位米、角度单位弧度；世界坐标即固定 pelvis 坐标。求解向量顺序为腰两关节、左臂 7 关节、右臂 7 关节，可通过 `robot.names`、`robot.lower`、`robot.upper` 和 `robot.velocity_limits` 查询。`robot.nominal` 是配置中的名义姿态。关节的 `origin_xyz`、`origin_rpy` 表示父链当前帧到关节帧的固定变换；`axis` 在关节帧表达。

为减少 SLSQP 目标函数的数值差分开销，末端位姿目标和闭链等式约束现使用模型几何雅可比与 SO(3) 对数映射解析导数；两侧 FK/雅可比共享一次腰部链遍历并缓存一次目标评估。外部障碍/平衡约束仍由 SciPy 对约束函数做数值差分。

## 工程边界

本项目提供的是**位置 IK / 运动学约束求解**，不含驱动器、ROS 2、实时调度、G1 精确碰撞网格、接触检测、静力矩/力矩控制或安全认证。闭链位置约束不等同于内力控制；共同搬运仍需独立的力/阻抗控制与载荷分配策略。真实部署前须核对具体 G1 硬件版本、URDF、标定和 TCP 偏移，在目标硬件和实时控制周期内完成限位、碰撞、急停和故障测试。求解输出是关节位置候选，不会直接连接或驱动机器人。
