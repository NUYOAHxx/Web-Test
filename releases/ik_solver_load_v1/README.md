# IK Solver 负荷感知候选改进版 v1

**交付类型：新增独立 DLS 分支，不覆盖原 `solver.py` / `solver_dls.py`。**

原始依据是用户提供的 `ha_arm_ik` DLS、求解编排代码及 `robot model` 子页。原始文件保持不变。由于未取得完整 `ik_candidate.py` / `ik_common.py`，本版不猜测它们的内部行为，也不伪装成原 `IKResult`。

## 先看边界

- 只计算 IK 与**模型静态重力补偿力矩/力**，不发布关节状态、TF、控制命令或执行轨迹。
- 默认检查整个固定基座模型的独立可动关节，而不只检查左臂。
- 允许值必须显式填写。没有真实允许值，就不能声称实现了负荷门槛；代码不会用 `1.0` 或 URDF effort 自动顶替。
- 不包含碰撞检查、支撑域硬约束、惯性负荷、接触、摩擦、热保护或真实电机反馈。原来的 MoveIt 规划和控制保护必须保留。
- 通过模型门槛，只表示在给定完整状态、模型与重力下，候选终点的静态估计满足配置；不是实机安全证明。
- `NO_LOAD_SAFE_SOLUTION` 表示有限搜索没找到满足配置的候选，**不是数学证明不存在可行解**。
- 新类返回 `(q_or_None, status, report_dict)`，第三项不是旧 `IKResult`。调用方必须改用 `report["accepted"]`；不能假设旧 `is_valid_status()` 识别新状态。
- 本版没有移植 SSIK、direct-psi、原候选 SEW 偏好，也没有修改 MoveIt 插件。它对应原默认 Pinocchio DLS 路径的独立替代候选。

## 1. 实际改了什么

| 原实现/风险 | 本版实现 |
|---|---|
| 迭代中的重力矩项主要保持相对 q_ready 的力矩差 | 默认使用绝对归一化负荷 + 超软阈值惩罚；相对项保留但默认关闭 |
| 无效 effort 回退 1.0 | 必须显式传入每个受检查关节的正、有限允许力矩/力 |
| 候选主要使用平均负荷型指标 | 先逐关节硬检查，再比较峰值；相近峰值范围内兼顾整体负荷与连续性 |
| 锁腰成功可能过早退出 | 预算允许时比较解锁腰部候选；锁腰负荷高时降低后续腰部运动代价 |
| 非活动关节采用 neutral | 每次求解要求完整模型关节快照，非活动关节也用输入状态 |
| 连续关节按一个标量直接写 Pinocchio q | 正确映射 nq=2/nv=1 的 cos/sin，同时保留外部角度分支 |
| 加权零空间实现与注释不一致 | 加权 SVD，使用数值秩下的真实线性零空间基，避免阻尼伪投影泄漏 |
| full log6 误差的 Jacobian 待核对 | 使用一致的 Jlog6 修正，并做真实 Pinocchio 有限差分验证 |
| 方向任务位置 Jacobian 使用 WORLD | 使用末端原点的 LOCAL_WORLD_ALIGNED；方向锥外使用角度违背量 |
| 更新前诊断可能被当作最终诊断 | 每个 seed 返回前重算，最终接收前再次计算位姿、边界和负荷 |
| 某些物理异常被跳过 | 记录 seed 错误或返回失败，不把失效项伪装成启用 |
| 原 q_ref ±90° / ±150° 人为工作窗口 | 不再隐式应用；使用实际 URDF 边界，可显式收紧规划边界，不能扩大 |
| 求解过程中数据可能变旧 | 输入和返回前都检查状态年龄；有限预算搜索，明确终止原因 |

**原有“唯一零点/完全回归不漂移”不再作为承诺。**绝对卸载与固定参考姿态本来可能冲突；新的目标是保留末端有效性、满足已配置静态门槛，并在此基础上改善负荷和连续性。

## 2. 文件

```text
ha_arm_ik/load_policy.py          纯 NumPy：配置、状态、负荷代价/门槛、加权投影
ha_arm_ik/solver_load_aware.py    Pinocchio：模型、FK、DLS、多 seed、最终复核
examples/integration.py          接入现有 ROS 节点的骨架，不含执行
tests/                          合成 URDF、策略、真实 Pinocchio 数值/流程测试
run_validation.py               输出版本化测试记录
validation_pin*.json / .txt      实际执行证据及未测试范围
source_provenance.json           原始代码快照哈希与来源
manifest.sha256                  交付文件完整性校验
```

压缩包的空 `ha_arm_ik/__init__.py` 只是测试包骨架。**放入原工程时只复制两个新模块，不要覆盖原工程的 `__init__.py`。**

## 3. 使用前必须准备的三件事

### A. 完整机器人描述

无需重新寻找本地 URDF 源工程。可复用已有程序中的 `robot_description` XML，直接传给构造函数；也可用之前的 Jazzy 只读监控工具导出一份运行描述后使用。

描述必须包含与真实负载配置一致的质量、局部质心和惯性。MoveIt 规划场景里的 attached object 不会自动进入这里的动力学模型，末端载荷应显式建模。

本版审计每个 link 的质量；无质量/零质量的辅助坐标帧必须显式列入 `massless_links`。不能把缺少质量的真实结构随意列入白名单来“跑通”。

先查看模型和配置需求：

```bash
python3 -m ha_arm_ik.solver_load_aware \
  --urdf exported_robot.urdf \
  --base-link YOUR_IK_BASE_LINK \
  --ee-link YOUR_TCP_LINK \
  --inspect > model_requirements.json
```

`YOUR_*` 是待替换名称。检查输出的活动关节顺序、完整状态需求、质量缺失项、关节类型和 URDF 参考限位。

### B. 确认关节侧允许力矩/力

把 `model_requirements.json` 里的 `allowable_efforts` 字典单独保存为 `confirmed_allowable_efforts.json` 并填实值。模板故意为 `null`，不能直接运行。

- 转动关节单位 N·m；直线关节单位 N。
- 必须确认是关节输出侧，而不是未经减速比换算的电机轴额定值。
- 应结合连续工作能力、温度、驱动器限制与工程降额确定。
- URDF effort 只作为参考显示，不自动等价于允许持续工作值。
- 默认需要整个模型所有独立可动关节的允许值。显式选择子集时，必须包含所有 IK 活动关节；未检查的其他关节会列入报告。

算法的 `soft_ratio=0.7`、`early_stop_ratio=0.6`、`peak_tie=0.02` 都是可调优化偏好，**不是针对你的机器人的标定安全阈值**。硬门槛是绝对值不超过已配置允许值。

### C. 完整、可追踪新鲜度的 JointState

每次求解必须提供 `JointStateSnapshot`，包含完整模型所有独立可动关节。不会给缺失的另一只手臂、腰部或夹爪补零。

在 ROS 订阅回调中只捕获一次：

```python
snapshot = JointStateSnapshot.from_joint_state(
    msg,
    now_ros_ns=node.get_clock().now().nanoseconds,
)
```

`now_ros_ns` 必须与消息头使用同一 ROS 时钟，仿真时两端均需正确使用仿真时间。零时间戳会被标为未验证；`require_source_age=True` 时不接收。

如果话题拆成多个部分，需要在外部按关节名聚合并检查**每个关节**的新鲜度。不要给旧缓存重新加接收时间，不要把最近一个子消息的时间冒充全模型状态时间。

## 4. 接入原 Solver

在原工程可运行的环境中，保留原 Solver，通过适配入口复用其 URDF 路径、base/EE、活动关节顺序、参考姿态及重力设置：

```python
import json
from ha_arm_ik.load_policy import JointStateSnapshot, LoadPolicy
from ha_arm_ik.solver_load_aware import LoadAwareIKSolver, SolverConfig

with open("confirmed_allowable_efforts.json", encoding="utf-8") as f:
    confirmed_limits = json.load(f)

new_solver = LoadAwareIKSolver.from_existing_solver(
    old_solver,
    policy=LoadPolicy(allowable_efforts=confirmed_limits),
    config=SolverConfig(require_source_age=True),
    massless_links=confirmed_massless_frame_names,
    # 必要时显式传入核对后的 q_reference，避免继承旧代码已裁剪的连续关节参考角。
)

q, status, report = new_solver.solve(
    T_target,
    frame="local",
    snapshot=latest_complete_snapshot,
    allow_approximate=False,
)

if not report["accepted"]:
    # 不发送这个候选，不更新执行目标；按原系统的停止/失败策略处理。
    print(status, report.get("reason"), report.get("best_rejected"))
else:
    # q 的顺序是 new_solver.active_joint_names。
    # 交给现有 MoveIt 规划链路；这里不自动执行。
    pass
```

`confirmed_massless_frame_names`、`old_solver`、`T_target`、`latest_complete_snapshot` 是原工程中的实际对象，不是本包自动提供的值。`examples/integration.py` 给出更完整的回调骨架。

原 Solver 如果曾把 continuous 关节的 `q_ready` 裁剪到 ±π，适配器不会凭空恢复原意。应显式提供正确的参考角；新模块自身不会对真正的连续关节施加 ±π 硬限位。

### 不依赖原 Solver 初始化

也可直接使用运行中的机器人描述：

```python
new_solver = LoadAwareIKSolver(
    robot_description_xml,
    base_link=actual_base_link,
    ee_link=actual_tcp_link,
    q_reference=confirmed_reference_angles,
    policy=LoadPolicy(allowable_efforts=confirmed_limits),
    massless_links=confirmed_massless_frame_names,
    gravity_root=confirmed_gravity_vector_in_urdf_root,
    config=SolverConfig(require_source_age=True),
)
```

`frame="local"` 表示配置的 base link；每次根据完整输入状态计算其在根系中的变换。`frame="model"` 是 URDF 根坐标系。`"world"` 仅是根系的兼容别名，不会自动变换任意 TF 的 map/odom 目标。

## 5. 新算法口径

受检查关节的模型负荷：

```text
r_i = |tau_g_i(q)| / allowable_i
```

次任务包含：

1. 绝对归一化负荷的加权均方项。
2. 超过软阈值部分的平方惩罚。
3. 可选的参考力矩差异项，默认关闭。
4. 参考姿态回归、真实边界缓冲项。
5. 可选的参考 CoM 回拉，投影到与重力垂直的平面；它不是支撑域约束。

力矩梯度对所有受检查关节求值，再对 IK 活动速度列求导。因此腰部运动影响另一只手臂的静态负荷时，不会被“只提取活动关节负荷”直接漏掉。对固定根系 link 的质量也进行整体重心累计。

主任务使用阻尼 SVD 步，次任务使用加权数值零空间基。非线性 FK 仍可能变化，因此通过回溯检查实际误差；最终接收还会再次检查。没有用“零空间”字样替代实际任务验收。

多 seed 的最终规则：

1. 排除位姿无效、位置越界或任一已检查关节超过允许值的候选。
2. 找到最低峰值负荷；在 `peak_tie` 允许的小范围内优先兼顾连续性和整体负荷。
3. 返回前重算同一候选的位姿与负荷，重新检查状态年龄。
4. 失败始终返回 `q=None`，不会因 `allow_approximate=True` 绕过上述条件。

默认尝试锁腰及解锁候选。只有完成要求的比较且负荷达到偏好阈值时，才允许低负荷早停；**时间预算仍可能先到，因此这不是全局负荷最优保证**。

## 6. 状态与诊断

| 状态 | 含义 |
|---|---|
| `SUCCESS` | 当前候选通过位姿、边界、已配置静态负荷和状态新鲜度检查 |
| `NO_LOAD_SAFE_SOLUTION` | 已找到位姿有效候选，但本轮没有满足配置负荷门槛的候选 |
| `NO_VALID_POSE` | 本轮没有达到位姿要求的候选 |
| `TIME_BUDGET_EXCEEDED` | 预算耗尽且没有可接收候选 |
| `STALE_STATE` | 输入或返回时状态年龄不满足要求 |
| `PHYSICS_ERROR` | seed 的物理/数值计算失败，且没有可用候选 |
| `INVALID_INPUT_OR_MODEL` | 输入、模型或前置物理计算异常，查看 `reason` |
| `FINAL_RECHECK_FAILED` | 最终重新验收失败，不交付 q |
| `BUSY` | 同一实例正在求解；不得并发改写其 Data |

关键字段包括 `gravity_efforts`、`load_ratios`、`peak_joint`、`peak_ratio`、`overloaded_joints`、末端误差、CoM、各次任务代价、每个 seed 的异常/终止原因，以及是否达到预算。

预算是协作式检查，不能中断单次 C++ 调用，也不是实时截止保证。调用方应使用单 worker 与最新目标版本策略；旧目标结果必须在交给 MoveIt 前丢弃。不要在 JointState 回调里阻塞运行 IK。

## 7. 测试与仍未验证的范围

执行：

```bash
python3 run_validation.py
```

已纳入真实 Pinocchio 的数值验证：重力矩导数、完整次任务梯度、CoM Jacobian、SE(3) 对数误差 Jacobian、方向任务与锥外角度 Jacobian；以及连续关节、完整状态、质量、逐关节门槛、最终重检、超时/过期和锁腰策略等流程。

有一个合成平面机械臂用例验证：开启绝对负荷目标后，能在保持终点位置要求的条件下改善该合成用例的静态负荷。**不是对原版完整 Solver 或用户机器人的性能基准，不得将其改写成实机卸载收益。**

准确的测试数量、Python/NumPy/Pinocchio 版本见随包 `validation_pin*.json`。Python 3.12 做过语法解析，但这不等于 Jazzy/Python 3.12 运行联调。

尚未验证：

- 用户实际 URDF、质量/载荷、允许力矩、原工程完整调用方及旧 IKResult 适配。
- 用户 ROS 2 Jazzy、MoveIt、TF 和实时 JointState 联调。
- 规划轨迹中间状态的负荷、动态力矩和真实电机/驱动器状态。
- 全局最优、所有目标可达、唯一参考姿态、严格零漂移或实机安全。

本版会明确拒绝 mimic、floating、planar 和多自由度关节，而不是悄悄忽略。若实际模型包含这些结构，需要进一步实现并验证对应映射。

## 8. 推荐工程验收顺序

1. 只检查：确认模型、完整状态、坐标、关节顺序、允许力矩及质量白名单。
2. 只算不执行：对待命、近距、远伸、边界目标保存全部诊断。
3. 同一工况做原版/候选版对照：使用一致目标、完整状态、质量、种子与硬件，比较末端误差、峰值/整体负荷、连续性、拒绝率及耗时。
4. 接入现有 MoveIt 规划但不执行，检查整条轨迹的碰撞与负荷。
5. 动态负荷与实际反馈验证通过后，再按原机器人受控联调流程推进。

减慢轨迹不一定降低静态重力负荷；如果没有满足允许值的构型，需要改变目标、构型、负载或机器人工作条件，而不能只强迫求解器“返回一个 q”。