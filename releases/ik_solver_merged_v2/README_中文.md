# IK Solver v2｜原结构负荷感知合并版

**交付的是两个原路径、同名模块的候选更新，不再新增独立 Solver 类。**

- `ha_arm_ik/solver_dls.py`：负荷目标、完整状态映射、任务 Jacobian、加权零空间与步长回溯。
- `ha_arm_ik/solver.py`：原 `HAArmIKSolver` 入口、多初值编排、逐关节负荷门槛及最终复核。
- 无需导入上一版的 `solver_load_aware.py` 或 `load_policy.py`。

## 先看交付边界

1. 保留原构造函数参数、`solve()` 原位置参数、三元组返回形式及原 `make_result()` 工厂。新增状态输入是**仅限关键字**的 `joint_state`，也可先调用 `update_joint_state()` 后继续用旧调用方式。
2. **接口形状尽量兼容，不是零配置替换。** 默认严格模式需要已确认的允许力矩、模型质量信息和完整新鲜状态；缺失时不返回可用 q，不会退回假造的 effort=1.0。
3. 已在真实 Pinocchio 3.9.0 / 4.1.0 下分别通过同一套 **74 项测试**。模型为合成 URDF；缺失的 `ik_common.py`、`ik_candidate.py` 仅在测试中使用明确标记的替身。
4. 生产代码继续导入用户真实模块，**没有把测试替身当作用户源码交付**。尚未验证实际评分权重、枚举、IKResult 内部实现、用户 URDF、完整工程、ROS 2 Jazzy / MoveIt、Python 3.12 运行或实机。
5. 本版求解的是静态重力负荷感知 IK；不发布 ROS 控制消息、不执行轨迹、不做碰撞检查，也不是驱动器安全功能。

## 为什么不只改 solver_dls.py？

只改单初值迭代，外层仍可能提前接受锁腰解、用旧评分选中高负荷候选，或通过近似分支绕过要求。因此同时更新外层，但不改变用户原来的工程入口：

```text
HAArmIKSolver.solve()
  → 冻结完整模型状态、检查模型与配置
  → 原候选组合：DLS + 可选 SSIK/direct-psi
  → DLS 内绝对负荷次任务与 FK 回溯
  → 所有候选的独立位姿 / 位置边界 / 逐关节静态负荷验收
  → 合格候选先按峰值形成短名单，再保留原候选选择接口
  → 返回前重新计算，复核状态新鲜度与非活动关节变化
  → 原 make_result() → (q 或 None, 原 status, 原格式结果)
```

保留 SSIK/direct-psi 的源码不是声称它们完成了工程运行验证。含腰时关闭这些可选引擎的原策略不变。

## 本版具体改了什么？

### 1. 从“接近参考力矩”改为包含绝对卸载目标

对受检查关节定义 `r_i = |tau_g,i(q)| / tau_allow,i`。`tau_allow` 是经过确认、必要降额后的**关节输出侧**允许值；转动关节单位 N·m，直线关节单位 N。

次任务包含：绝对归一化负荷平方、超过软阈值的惩罚、姿态回归、边界缓冲及可选参考 CoM 回拉。相对参考力矩项仍可启用，但默认权重为 0。代价按受检查关节数归一化。

- 单关节超过配置允许值就拒绝；不能以平均负荷合格掩盖单个关节过载。
- 先找峰值接近最低的合格候选，再交回原 `select_best_candidate()` 保留 SEW、姿态与连续性偏好。
- 多初值、有限迭代和多个偏好之间存在折中；不保证全局最低峰值或唯一姿态。

### 2. 修正任务与零空间数学

- full：SE(3) 对数残差配套 `Jlog6`，最终位置验收使用普通位置距离。
- tool_dir / tool_cone：末端原点、世界轴对齐的 Jacobian；方向锥使用超锥角；精确反向时加入确定性逃逸方向。
- 加权 SVD：支持一般 SPD 逆度量；在选定数值秩下构造零空间次任务，再通过真实 FK 回溯保持主任务容差。
- 边界处冻结外推关节并缩放步长；不把它宣传为完整 SNS 最优算法。
- 不再依赖重力导数 API 是否顺带更新 `data.g`；显式计算当前重力矩。
- 最终日志与返回 q 重新对应，不复用更新前或其他候选的误差。

### 3. 完整状态与真实限位

- 使用完整状态快照，不给另一只手臂、腰部或夹爪补零。
- continuous 的 nq=2/nv=1 按 cos/sin 映射，保留参考角分支；不默认施加 ±π 硬限位。
- 删除原 DLS 的隐含 `q_ref ±90°/±150°` 工作窗口。那是算法窗口，不是厂家限位。
- `position_bounds` 只允许收紧实际模型边界；continuous 可显式增加工程工作边界。机械限位和规划限位仍需核对真实模型及 MoveIt 配置。
- 原始观测越界会报错，不裁剪以掩盖越界。内部初值投影属于数值搜索，不代表修改观测。
- 根/固定 link 质量纳入全模型 CoM；CoM 回拉是软目标，不等于支撑域或动态平衡硬约束。
- 新增 `gravity_root` 可显式设置 URDF 根系下的重力向量，单位 m/s²；缺省保留原安装坐标约定，需在工程核对。

## 安装：只替换两个文件

不要把整个交付目录直接覆盖到用户工程。特别是不要覆盖现有 `__init__.py`、`ik_common.py`、`ik_candidate.py`、`robot_model.py`。

1. 使用源码管理或复制备份当前 `ha_arm_ik/solver.py`、`ha_arm_ik/solver_dls.py`。
2. 对照包内 `baseline/` 和 `changes/`，检查是否与你当前本地版本一致。如果本地已有其他修改，应手动合并，不能盲目覆盖。
3. 配套替换交付包内 `ha_arm_ik/solver.py` 与 `ha_arm_ik/solver_dls.py`。
4. 生产接入不需要复制包内空 `ha_arm_ik/__init__.py`；它只用于离线测试。
5. 不要复制 `tests/stubs.py` 或 `tests/reference/` 到生产包。

安装后，在**用户真实工程根目录**执行（替换路径）：

```bash
python3 -m py_compile ha_arm_ik/solver.py ha_arm_ik/solver_dls.py
PYTHONPATH="$PWD" python3 /path/to/delivery/tools/check_user_imports.py
```

第二个命令不装载替身，只检查真实模块导入、签名和旧失败结果工厂；**不验证成功候选评分、模型质量或完整求解**。检查 JSON 中模块路径，确认没有误用交付包测试目录。

## 接入配置：先确认数值，再只算不执行

### A. 机器人描述来自已经运行的程序

原构造函数仍接收 URDF 文件路径。没有原始 URDF 工程时，可用上一版 ROS 2 Jazzy 只读监控程序从当前运行的 `robot_description` 导出 XML 到文件；不需要重新找原模型工程。

应导出当前实际模型，不能把 `ros2 param get` 的类型前缀或 YAML 包装直接当 XML。MoveIt attached object 不会自动纳入此静态模型；有外载需要先正确建模。不要用只有外观和 TF、没有质量信息的模型输出可信负荷结论。

### B. 生成故意留空的需求清单

```bash
python3 tools/inspect_urdf.py exported_robot.urdf > model_requirements.json
```

输出的 `config.load_allowable_efforts` 全部为 null，**不能直接运行**。填写依据明确的关节侧允许值；不能复制测试用例的数字，也不能不经核对把 URDF effort 当连续额定能力。

`massless_CANDIDATES_MUST_CONFIRM` 只是缺质量 link 的待核对列表；只把真正无质量的辅助帧填入 `load_massless_links`。真实结构漏质量时，应修模型，而不是加入白名单绕过检查。

### C. 最小接入

```python
import json
import numpy as np
from ha_arm_ik.solver import HAArmIKSolver  # 原入口不变
from ha_arm_ik.ik_candidate import is_valid_status
from ha_arm_ik.solver_dls import JointStateSnapshot

record = json.load(open("confirmed_model_requirements.json", encoding="utf-8"))
config = record["config"]  # 允许值及无质量辅助帧已人工核实
solver = HAArmIKSolver(
    urdf="exported_robot.urdf", side="left", include_waist=True, config=config,
)

# 在已有 /joint_states 回调中捕获一次；now_ros_ns 与消息头使用同一 ROS 时钟。
snapshot = JointStateSnapshot.from_joint_state(
    msg, now_ros_ns=node.get_clock().now().nanoseconds,
)
solver.update_joint_state(snapshot)

# 放到单个求解 worker；不要在订阅回调中阻塞。
# 此处 snapshot 是原样缓存的完整状态，不能给旧缓存重新盖时间戳。
q_observed = np.array([snapshot.positions[n] for n in solver.active_joint_names])
q, status, result = solver.solve(
    T_target, frame="local", q_prev=q_observed, update_prev=False,
    joint_state=snapshot, allow_approximate=False,
)
if q is None or not is_valid_status(status) or not solver.last_load_diagnostics["accepted"]:
    print(status, solver.last_load_diagnostics)
    # 不交付执行目标，进入用户原有失败处理。
else:
    # q 按 solver.active_joint_names 排列。
    # 这里只得到候选；仍需目标版本检查、最新状态检查和 MoveIt 规划/轨迹验收。
    pass
```

`msg`、`node`、`T_target` 来自实际工程，不是可复制运行的独立控制程序。包内 `examples/integration.py` 提供同样的只算接口骨架。

也可以只在回调里调用 `solver.update_joint_state(msg, now_ros_ns=...)`，之后继续使用原 `solver.solve(T_target, ...)`。但外部规划执行系统宜从当前观测设置 q_prev，避免把上一次接受的目标当成已经到达的实测状态。

部分 JointState 需要在外部按名称聚合，检查每个关节的年龄。不能把最新一个子消息的时间作为全部关节的时间。模拟时钟暂停/回跳和零时间戳应明确处理；本版收包新鲜度使用单调墙钟，采取保守拒绝策略，不把旧状态变成新状态。

## 常用配置与默认值

| 配置 | 默认 / 含义 |
| --- | --- |
| `load_mode` | `strict`。`off` 仅供明确的未受保护调试/对照，不作为执行前的临时绕过方案 |
| `load_allowable_efforts` | 必填，受检查关节的正有限允许值；不自动使用 URDF effort |
| `load_guard_joint_names` | 默认检查所有独立可动关节；显式子集必须包含全部 IK 活动关节，诊断列出未检查项 |
| `load_massless_links` | 默认空，仅已确认的无质量辅助帧 |
| `load_absolute_gain` | 默认 0.04，若显式提供旧 `dls_gravity_torque_gain`，单初值使用旧项值覆盖绝对增益；不要同时设置相互冲突的值 |
| `load_excess_gain` / `load_relative_gain` | 0.5 / 0；超软阈值惩罚 / 相对参考力矩偏好 |
| `load_soft_ratio` / `load_early_stop_ratio` | 0.7 / 0.6，均为优化偏好，不是硬件标定结果 |
| `load_peak_tie` | 0.02，峰值比率接近范围内保留原评分偏好 |
| `load_state_max_age_s` | 1.0，需结合输入速率和计算时延设定并验证 |
| `load_require_source_age` | 代码默认 False 以兼容已有状态对象；**ROS 接入配置建议显式 True**，验证消息头而不只收包时间 |
| `load_budget_s` | 0.5，协作式搜索预算；不承诺硬实时，不抢占正在进行的库调用/可选求解器 |
| `load_max_seeds` / `load_min_seeds` / `load_min_unlocked_seeds` | 10 / 3 / 2；预算允许才完成比较；预算不足会记录实际尝试数 |
| `load_context_tolerance_rad` / `load_context_tolerance_m` | 0.01 / 0.001；求解期间非活动关节变化检查阈值，工程需核实 |
| `position_bounds` | 按关节名给出更严格的实际/规划边界；不自动读取 MoveIt joint_limits.yaml |
| `q_ready` | 可显式覆盖参考向量，按活动关节排列；strict 下参考越界报错、不静默裁剪 |
| `gravity_root` | 可选根系重力向量；非零有限三维，m/s² |
| `dls_characteristic_length` | 0.5 m，用于平移/角度行尺度，不是新的机械限位 |

原 `dls_*` 调参入口保留。数学修正及代价归一化会改变原参数的实际效果，不能把旧调参效果视为自动继承。

## 返回值、失败与可观测性

仍返回 `(q, status, result)`，第三项经原 `make_result()` 构造，**不是独立版的 report 字典**。

为了不引入旧枚举没有的新值，新增负荷失败对外沿用原 `NO_VALID_POSE`；具体原因写入结果的失败说明和 `solver.last_load_diagnostics`。成功状态仍由原候选模块决定。

常见诊断码：

- `LOAD_CONFIG_OR_STATE_INVALID`：模型、允许值、完整状态或新鲜度不满足要求。
- `NO_LOAD_SAFE_SOLUTION`：已找到位姿合格候选，但有限搜索中没有通过配置静态负荷/边界门槛的候选；不是证明全局无解。
- `NO_VALID_POSE_OR_PHYSICS`：没有合格位姿/物理候选，查看各 seed 与拒绝记录。
- `STALE_OR_CHANGED_CONTEXT`：状态过期或非活动关节变化使冻结上下文失效。
- `FINAL_RECHECK_FAILED`：被选中的候选未通过返回前重算。
- `SOLVE_ERROR`：显式异常，保留异常类型与说明。
- `BUSY`：同实例已有求解；见该次失败结果说明，不覆盖正在运行求解的诊断。

`last_load_diagnostics` 是最近一次求解的侧通道，应在单 worker 返回时复制/记录。strict 下 `allow_approximate=True` 也不能越过位姿或负荷门槛。

## 已执行的测试与不能夸大的结论

| 类别 | 数量 | 实际覆盖 |
| --- | --- | --- |
| 真实 Pinocchio 数值回归 | 18 | 重力矩导数、完整 CoM/Jacobian、三类任务导数、整体次任务梯度、SPD 度量、连续/直线关节、边界、最终诊断、反向方向逃逸、合成臂卸载 |
| 状态/策略单元测试 | 25 | 显式允许值、逐关节拒绝、目标梯度、不可变快照、消息头与墙钟年龄、输入校验 |
| 旧入口与编排契约测试 | 31 | 原签名、旧工厂调用/替身结果类型、各引擎统一门槛、锁腰后比较、完整状态、坐标、最终重算、并发与清理 |

同一套 74 项在 Python 3.13.14 / Pinocchio 3.9.0 / NumPy 2.3.5 / SciPy 1.18.1，以及 Python 3.13.14 / Pinocchio 4.1.0 / NumPy 2.5.4 / SciPy 1.18.1 中全部通过。

`robot_model.py` 使用已提供的真实源码作为测试参考；`ik_common.py`、`ik_candidate.py` 的替身明确位于测试目录。**31 项契约测试不能证明真实候选模块、枚举、IKResult 内部字段或全部连续关节评分边界已兼容。**交付的生产模块仍调用真实原模块，需要在现场进一步检查，特别是连续关节无限边界与原评分工具的兼容性。

合成平面臂对照证明这版 DLS 代码能够在该用例保持位置要求并降低峰值/整体静态负荷；它不是用户机器人的收益或性能数据。两个 Pinocchio 版本各跑同一套，不计成 148 个独立测试场景。

复跑：

```bash
python3 run_validation.py
```

仅在独立验证目录、兼容测试环境中复跑。`requirements_tested_pin*.txt` 是实际测试环境快照，不是要求直接升级 ROS 系统依赖。Python 3.12 目前仅通过 AST 语法解析，不代表完成 Jazzy 的 3.12 运行验证。**本版没有修改 ROS 或系统依赖。**

## 工程验收与回滚

建议顺序：

1. 检查两个文件的真实导入、原候选评分与结果工厂，核对 URDF 质量/COM、实际限位、重力方向、外载和允许值。
2. 只算不执行，记录原版和合并版同一组目标、状态、种子、位姿误差、逐关节力矩/比率和耗时；失败要单独归档。
3. 再用 MoveIt 只规划，检查碰撞及**整条轨迹**的静态峰值和动态需求，而不只检查 IK 终点。
4. 加目标版本号与最新状态核对，丢弃过时求解结果；沿原系统的受控联调流程推进。

不支持的 mimic、floating、planar 和多自由度关节会明确拒绝。支撑域、动态惯性、接触、温升、驱动器保护和实测力矩仍需其他模型/反馈与系统验收。减速不能消除静态重力负荷。

回滚应恢复事先备份的两个文件及原配置。`baseline/` 仅对应本次取得的 Notion 源码快照；如果本地原版已有更新，不要用它覆盖较新的修改。交付未自动部署到用户电脑、未覆盖 Notion 原始代码页。

## 目录说明

- `ha_arm_ik/`：两个生产候选模块；空 init 仅供独立测试。
- `baseline/`：本次原始源码备份。
- `changes/`：逐文件统一 diff。
- `examples/`：只算不执行接入骨架。
- `tests/`：合成模型、真实 robot_model 参考源码和明确标注的缺失模块替身。
- `tools/`：模型需求检查、真实导入检查及可复现合并脚本/片段；不是自动安装器。
- `validation_pin*.json/.txt`：两版真实依赖环境的测试证据。
- `source_provenance.json`、`SHA256SUMS.txt`：来源与完整性清单。
