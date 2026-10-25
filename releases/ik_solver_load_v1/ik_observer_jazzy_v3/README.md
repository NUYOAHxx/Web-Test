# ROS 2 Jazzy｜MoveIt 只读重心与关节负载观察器 v3

## 这版适合什么场景

其他程序已经负责 6D 目标、IK、MoveIt 规划与执行；本程序只读取当前状态，
把重心等 Marker 叠加到已经打开的 MoveIt RViz。

- 不调用 IK，不创建轨迹 action client，不发送执行命令。
- 不发布 `/joint_states`、`/tf` 或 `/tf_static`，不启动 RViz / robot_state_publisher。
- 业务输出仅 `/ha_ik_monitor/markers`（ROS 节点常规日志/参数事件除外）。
- 不要求本地 URDF 文件，优先读 `/robot_description`，
  并以目标节点的 GetParameters 服务作为后备。
- 不依赖 `ha_arm_ik`、Pinocchio 或 SciPy。静态模型计算仅使用 NumPy。
- 这是模型诊断工具，不是实机安全控制器，也不是电机负荷传感器。

## 1. 环境

使用已经能运行 MoveIt Jazzy 的 ROS 2 环境和匹配的 Python。
Jazzy 官方二进制通常对应 Ubuntu 24.04 / Python 3.12；若使用容器或自行构建，
以现有可正常导入 rclpy 的解释器为准，不要混用 Conda / 其他 Python 版本。

依赖：`numpy`、`rclpy`、`tf2_ros`、`sensor_msgs`、`std_msgs`、
`visualization_msgs`、`geometry_msgs`、`rcl_interfaces`。
已有 MoveIt/RViz 安装通常包含大部分 ROS 包。若在标准 Jazzy 系统中缺包，
可由用户自行安装 `python3-numpy`、`ros-jazzy-tf2-ros` 等相应软件包。
本次开发未在用户机器上执行任何安装操作。

## 2. 最短启动步骤

1. 保持原 MoveIt Demo / IK 程序和 RViz 正常运行。
2. 停止旧版 v2 自求解监控，避免同一 Marker 话题混入两套结果。
3. 在另一个终端进入解压目录，并 source 相同 ROS 和工作空间环境：

```bash
source /opt/ros/jazzy/setup.bash
# 如原程序需要，再 source 对应工作空间的 install/setup.bash
python3 ik_state_observer_jazzy.py --joint-labels
```

4. 在原 RViz 中选择 **Add → MarkerArray**，话题设为：

```text
/ha_ik_monitor/markers
```

保留原 Fixed Frame，只要它能通过现有 TF 连接到模型根坐标系。
不要再启动另一套 RobotModel / robot_state_publisher，不需要 InteractiveMarkers。

同一 ROS_DOMAIN_ID、正确命名空间和可通信的 RMW/QoS 是接入前提。

## 3. 确认输入名称

默认：

| 用途 | 默认值 |
| --- | --- |
| 当前关节状态 | `/joint_states` |
| 机器人描述话题 | `/robot_description` |
| 描述参数后备节点 | `/robot_state_publisher` |
| 参数名 | `robot_description` |
| 重力方向参考系 | `world` |
| 重力向量 | `[0,0,-9.80665]` m/s²，表达在上述参考系中 |

查看正在运行的图：

```bash
ros2 node list
ros2 topic list
ros2 topic info /joint_states -v
ros2 topic info /robot_description -v
ros2 param list /robot_state_publisher
```

如果机器人描述只存在于 `/move_group` 的参数中：

```bash
python3 ik_state_observer_jazzy.py \
  --description-node /move_group --joint-labels
```

如果用了命名空间，按实际名称配置，例如：

```bash
python3 ik_state_observer_jazzy.py \
  --joint-topic /robot/joint_states \
  --description-topic /robot/robot_description \
  --description-node /robot/robot_state_publisher
```

`--frame-prefix robot/` 仅在已有 TF 帧名确实使用该前缀时添加；
节点命名空间不等于 TF 帧名前缀，不能想当然地一起添加。

## 4. 会看到哪些信息

- 蓝色模型 CoM 球与状态文字。
- 关节静态重力矩相对 URDF effort 的颜色映射：低比例偏绿，高比例偏红；
  effort 或重力信息不可用时显示灰色。
- `--joint-labels`：关节名、模型重力矩、URDF effort 比例、真实读取到的限位范围与余量。
- `--show-link-com`：模型各有质量连杆的质心小球。
- `--print-joints`：终端逐关节位置、模型重力矩、输入 topic 中的 effort 和 URDF 范围。

```bash
python3 ik_state_observer_jazzy.py \
  --joint-labels --show-link-com --print-joints --rate 10 --console-rate 1
```

默认显示独立可动关节；`--joint-names` 只过滤显示，不减少完整模型状态需求。
例如：

```bash
--joint-names Hip_Link1,Body_Link,LHand_Link1,LHand_Link2,LHand_Link3,LHand_Link4,LHand_Link5,LHand_Link6,LHand_Link7
```

### 负荷解释

`gravity_model` 是抵消模型重力所需的静态广义力/力矩：
转动关节单位 N·m，移动关节单位 N。
本程序用 URDF 质量与局部质心、关节运动学和重力，通过子树质量一阶矩计算；
对 mimic 耦合用虚功关系归并到独立坐标。
不需要惯性张量来计算静态重力项，但不计算惯性、科氏力、摩擦或未建模外载荷。

`effort_topic_unverified` 是 JointState 发布者提供的原始字段，
可能为空、命令值、仿真值或实测值；程序不会自动称其为“实测力矩”。
URDF effort 限值也不一定是电机连续额定能力，不代表真实热裕量。

### 质量缺失

模型重心需要质量与局部质心，只有外观和 TF 不够。
几何连杆缺少/零质量时，默认禁用重心和重力矩估算并提示问题连杆。
`--allow-partial-mass` 可显式查看仅已建模质量部分的结果，但始终标记为 `PARTIAL_MODEL`，
不能将其当作整机重心。程序不会从 mesh 外观猜测质量。
没有正质量时，即使开启该选项也不会生成假重心。

## 5. 限位与用户文档对照

`limits_reference.json` 来自用户 Notion `robot model` 中的左臂+腰部限制表。
只做比较，不是硬件规格认证，不覆盖运行模型，也不裁剪订阅角度。

```bash
python3 ik_state_observer_jazzy.py \
  --limits-reference limits_reference.json --joint-labels
```

- 本版不使用 `q_ref ±90°/±150°`；它们属于旧 IK 的搜索窗口。
- 连续关节保留输入角度，不添加假的 `[-π,π]` 机械端点。
- `revolute` / `prismatic` 的有限边界来自当前运行模型的 `<limit>`；
  本版不自动合并 `safety_controller` 软限位。
- 超范围时保留输入并显示 `URDF_LIMIT_VIOLATION`，不篡改观测。
- 本版未读取 MoveIt 的全部规划限位参数或 `joint_limits.yaml`，
  因此标签明确是 URDF 限位，不称为 MoveIt 最终规划限位。

## 6. 重力与支撑平面

默认重力表达在 `world`。模型根系不是 world 时，程序通过现有 TF 旋转重力。
如果 `world` 不存在，CoM 仍可显示，但力矩为未知；需要选择实际存在且方向已知的重力参考系。
只有确认基座固定且 +Z 向上时，才可用该基座帧作为参考系。
不能为了消除报错随便改帧名或猜重力方向。

支撑投影默认关闭，因为未标定的“地面 Z=0”和矩形尺寸可能误导。
若现有 TF 的某帧确实定义了实际支撑平面：

```bash
--support-frame ACTUAL_SUPPORT_FRAME
```

如已核对实际尺寸，可再添加：

```bash
--support-half-extents 0.15 0.20
```

这里的 0.15/0.20 只是参数格式示例，不是机器人的已确认尺寸。
支撑帧 Z=0 为平面，+Z 向上；重力作用线与平面的交点用于投影。
矩形边界余量仅是静态模型几何指标，不是 ZMP、动态平衡或实机安全保证。

## 7. 时间、状态完整性与暂停

- 默认按名称读取完整 URDF 的所有独立可动关节，不能把未收到的关节填 0。
- 支持固定关节、转动、连续、移动和 mimic；floating / planar 显式拒绝，不猜状态。
- 默认状态超时 2 秒、关节时间偏差上限 0.5 秒；可用 `--state-timeout` / `--max-skew` 调整。
- 缺失、超时、明显时钟错配时，撤下旧估算 Marker 并显示等待原因。
- 零时间戳允许接入，但明确警告非时间对齐，使用接收新鲜度和最新 TF。
- 默认在状态时间查询 TF；`--tf-time latest` 只适用于明确接受时间近似的情况。
- 观察到 ROS 时钟回退时清空关节缓存；TF Buffer 也使用节点时钟。
- 如果原程序使用 `/clock`，监控同样设置：

```bash
python3 ik_state_observer_jazzy.py --joint-labels \
  --ros-args -p use_sim_time:=true
```

应用参数放在 `--ros-args` 之前。仿真时钟暂停时，ROS 定时器和 Marker 时间也可能暂停，
画面停留不应被当作持续收到新状态的证明。
`--rate` 是目标刷新率，不是已验证的实时性能。

## 8. 日志与运行模型导出

```bash
python3 ik_state_observer_jazzy.py \
  --csv logs/observer_01.csv \
  --save-description logs/runtime_robot_01.urdf \
  --joint-labels --print-joints
```

无需自己准备 URDF；`--save-description` 是可选地保存运行程序已经提供的描述，
便于之后核对模型。文件必须是新路径，不覆盖旧文件。
CSV 记录模型哈希、状态、姿态、CoM、模型重力矩、输入 effort、限位对照与状态时间；
旁侧 `.meta.json` 包含程序哈希与启动配置。导出和日志不是运行必需项。

## 9. 常见提示

| 提示 | 应检查什么 |
| --- | --- |
| WAIT_MODEL | 描述话题、参数节点、命名空间、QoS、ROS_DOMAIN_ID |
| WAIT_STATE / missing | 是否发布了完整独立关节名；不能只提供左臂而忽略其他会影响重心的关节 |
| stale / future timestamp | topic 是否持续更新、系统/仿真时钟是否匹配、超时设置是否合理 |
| MASS_UNAVAILABLE | 当前描述的 inertial/mass/origin；模型缺物理量时不能凭空修复 |
| GRAVITY_TF_UNKNOWN | 实际重力参考系、状态时刻 TF 与 frame_prefix |
| URDF_LIMIT_VIOLATION | 输入单位、运行 URDF、文档对照；程序不会裁剪角度 |
| MODEL_GRAVITY_OVER_URDF_EFFORT | 模型重力项超过 URDF effort，不等同于已测得过载 |
| RViz 提示 Frame 不存在 | Fixed Frame 与模型根系的 TF 连接、实际帧名前缀 |

如果 RViz 中只在拖动规划目标而未执行，当前 `/joint_states` 未必改变；
本程序观察的是当前发布状态，不是尚未执行的规划预览。

## 10. 本次验证与限制

在开发环境完成纯 Python/NumPy 测试，以及使用假的 ROS 消息与节点接口的流程测试。
实际测试环境为 Python 3.13 / NumPy 2.5，另有 Python 3.12 语法检查。
**没有 rclpy、ROS 2 Jazzy、真实 DDS/TF、用户完整模型或实机可供联调。**
模拟接口测试不是实际 ROS 集成测试，不代表真实延迟或物理精度已验证。

复验方法（当前目录）：

```bash
python3 -m py_compile ik_state_observer_jazzy.py
python3 -m unittest -v test_observer test_ros_interface_mock
python3 ik_state_observer_jazzy.py --help
```

测试覆盖树结构、FK、质心、静态重力与势能梯度、mimic 虚功归并、连续关节、
原始角度不裁剪、缺失/过期/乱序/零时间戳、日志、只创建 MarkerArray 业务发布者，
以及 topic / 参数后备接入、模型更新、TF 缺失、时钟回退和退出流程。
完整结果见 `offline_test_results.txt` 和 `validation_summary.json`。

接下来应在用户实际环境验证：描述获取、关节/帧名匹配、质量参数、RViz 对齐、
TF 时间同步和长时间运行；必要时再与 Pinocchio 或独立计算对照静态模型数值。