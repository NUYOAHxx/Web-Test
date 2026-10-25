# 双臂 16 维联合 IK（P1 + P2，dual-p2.1）—— `ha_arm_ik/dual_arm_ik.py`

dual-p2.1（随 merged-v15）：只给一只手目标时，另一只手默认 `"hold"`（`None` = `"hold"`）；不约束改为显式 `"free"`；
两只都是 `None` 报错。其余与 dual-p2 相同（求解逻辑未改，下面的性能数据仍有效）。

腰 2 + 左臂 7 + 右臂 7，左右手独立目标（模式一）。新文件，只依赖 numpy 和 Pinocchio 4.x，
不 import 任何单臂模块（solver / solver_dls / ik_common / ik_candidate / robot_model）；单臂文件一行未改。

## 1. 结论

| 指标（合成双臂 URDF，默认配置，预算 0.1 s，单进程） | 结果 |
|---|---|
| 可达目标 1000 组（q_prev = q_ready） | 1000/1000，中位 5.9 ms，p90 16.8，p99 26.3，最长 49.3 ms |
| 路点跟踪 600 步 | 600/600，中位 1.0 ms，最长 2.3 ms；每步关节变化 ≤ 0.053 rad |
| 所有用例（含失败、换姿、单手保持、不可达）最长 | 93.9 ms（≤ 100 ms） |
| 可证明的不可达（超臂展、两手间距超界） | 200/200 预检直接判出，≤ 0.8 ms |
| 测试 | `tests/test_dual_arm_ik.py` 40 个；全量 250 个（含单臂 210 个）全部通过（merged-v15 + dual-p2.1） |

## 2. 接口

```python
from ha_arm_ik.dual_arm_ik import DualArmIKSolver, DualArmLayout

s = DualArmIKSolver(urdf_path)            # layout 默认 = 真机命名（与 robot_model.py 同值，复制非引用）
q, status, res = s.solve(T_left, T_right, q_prev=None, update_prev=True, verbose=False,
                         allow_approximate=False, orientation_mode="full", tool_axis=None,
                         tool_cone_half_angle=None, *, joint_state=None, time_budget_s=None)
T_left, T_right = s.fk(q)                 # 4×4，URDF 根坐标系
s.reset_prev(q=None)
```

- `q`：16 维 `[腰 2 | 左 7 | 右 7]`，顺序见 `s.active_joint_names`；无解时 `None`。
- `T_left / T_right`：4×4 或 `pin.SE3`（URDF 根坐标系）/ `"hold"` / `"keep"` / `"free"` / `None`（= `"hold"`），见第 3 节。
- `orientation_mode / tool_axis / tool_cone_half_angle`：一个值（两手相同）或 `(左, 右)`；模式 `full / tool_dir / tool_cone`，含义与单臂相同。
- `q_prev`：不给时取 `joint_state` 里齐全的 16 个活动关节，否则取上次成功的解（初始 `q_ready`）。
- `res`（`DualIKResult`，dict）：`q_waist / q_left / q_right`，`left / right`（`pos_err, ori_err, pose_ok, orientation_mode`），`margin, margin_joint, max_jump, jump_joint, jump_ok, seed_kind, regime, seeds_used, iterations, elapsed_s`，失败时 `diagnosis`；另有 `res.failure_reason / res.failure_detail / res.per_seed`。
- 一个实例同一时刻只服务一个调用（并发调用直接返回 `SOLVER_BUSY`）；多线程请每线程一个实例（种子库进程内共享）。

## 3. 目标类型（每只手）

| 值 | 含义 |
|---|---|
| 4×4 / SE3 | 该手到这个位姿 |
| `"hold"` | 该手保持 `q_prev` 处的世界位姿（腰、该臂都可动） |
| `"keep"` | 该臂关节角保持 `q_prev`（腰动时手会跟着动） |
| `"free"` | 不约束，只回中 |
| `None` | 同 `"hold"`（默认）：只给一只手目标时，另一只手保持不动 |

- 至少一只手是位姿或 `"hold"`，否则 `ValueError`（如 `("keep", "free")`、`("free", "free")`）。
- 两只都是 `None` 报错（`ValueError`，多半是调用方漏传目标）；要两手都不动请显式传 `("hold", "hold")`，
  `q_prev` 已是解时原样返回。
- 另一只手要放开（只回中）时传 `"free"`。dual-p2 及以前 `None` 是不约束，升级时请把这类调用改成 `"free"`。

## 4. 状态与失败原因

`status` 与单臂同一词表：`SUCCESS / NEAR_LIMIT / APPROXIMATE / NO_VALID_POSE / OUTSIDE_WORKSPACE`。

| failure_reason | 何时 | 性质 |
|---|---|---|
| `LEFT/RIGHT/BOTH_UNREACHABLE`（OUTSIDE_WORKSPACE） | 目标位置对任何腰角都超出臂展 | 证明 |
| `NOT_JOINTLY_REACHABLE`（OUTSIDE_WORKSPACE） | 两目标间距 > 两手臂根间距 + 两臂臂展，或没有同一腰角让两手都在臂展内 | 证明 |
| `JOINT_JUMP_LIMIT` | 找到的解要转过 `max_joint_jump_rad`（默认 π，reject） | 找到了框外解，确定 |
| `LEFT/RIGHT/BOTH_UNREACHABLE`（NO_VALID_POSE） | 诊断：该手单独（腰 + 该臂）也解不出 | 搜索结果 |
| `NOT_JOINTLY_REACHABLE`（NO_VALID_POSE） | 诊断：两手各自可解，联合解没找到 | 搜索结果 |
| `TIME_BUDGET_EXCEEDED` | 预算用完，诊断没做完 | — |
| `SOLVER_BUSY` | 同一实例被并发调用 | — |

## 5. 判据（数值复制单臂，测试逐项核对）

位置 0.5 mm、姿态 0.5 mrad；限位裕度 0.02 rad（低于即 NEAR_LIMIT）；跳变上限 π（reject）；刚体校验容差；
DLS 步长 / 阻尼 / 停滞判据 / 零空间增益 / 腰部释放 / 姿态整理常数；续算判据 0.1 × 臂展 & 0.1 rad、续算框 0.15 rad、
续算单步 0.5 × 框。

## 6. 求解流程

1. 解析目标。
2. 可达性预检（< 1 ms，只判可证明的不可达）。
3. **续算**（路点跟踪，单臂 v9 规则）：每只手的目标都在 FK(q_prev) 附近时，从 q_prev 在 q_prev ± 0.15 rad 内求解，有解即采用（含 NEAR_LIMIT）。
4. **联合搜索**（到预算 70%）：种子 q_prev → q_ready → 组合种子库（腰网格 × 两臂各自最近样本）→ 随机扰动，第一个合格解即返回。迭代限位 = 内缩限位 ∩ 跳变框；种子库只取框内样本。跳变框起作用时框内搜到 55%，之后到 70% 不加框试 q_ready + 4 个库种子：解在框内即采用，在框外 → `JOINT_JUMP_LIMIT`。
5. 近限位补救（到 78%）：物理限位里从误差最小的终点续算，解出即 NEAR_LIMIT。
6. 失败诊断（到 95%，不加跳变框）：左手单独、右手单独能否解出；都能解出就主从分解拼种子再联合精化（能救回则成功）。

单种子迭代：两手 12 维误差堆叠，带权阻尼最小二乘 + 零空间回中（限幅）+ 限位饱和重解（SNS）+ 回溯线搜索 + 进度门限（第 30 / 60 步误差仍大于 0.1 / 0.02 换种子）。每次迭代约 110 µs。

## 7. 与单臂有意不同的地方

| 点 | 单臂 v14 | 双臂 | 原因 |
|---|---|---|---|
| 续算后的姿态整理 | 在 ±0.15 框内照常整理 | 不整理，回中只来自收敛过程中的零空间步（未达标时步长 ≤ 2 × 误差范数） | 目标不动、q_prev 已是解时单臂每次调用仍漂 0.10–0.15 rad（实测），伺服循环里等于自己在动；双臂原样返回，漂移随手的移动量走 |
| 跳变上限 | 先不加框搜，只找到超限解再框内重搜 | 框直接进迭代限位，框内找不到再不加框探测 | 默认路径不产生超限解；探测保证超限时报 `JOINT_JUMP_LIMIT` 而不是超时 |
| 失败诊断 | 无 | 单手可达性 + 主从分解 | 区分“哪只手到不了”与“两只手不能同时到” |

## 8. 配置（`config=`，未知键报错）

| 键 | 默认 | 说明 |
|---|---|---|
| `time_budget_s` | 0.1 | 一次 solve 的总时间（含诊断）；None = 不限 |
| `max_seeds` | 24 | 联合搜索种子上限 |
| `max_iter` | 150 | 单种子迭代上限 |
| `joint_jump_policy` | `"reject"` | `"prefer"`：超限解在没有合格解时也返回（`jump_ok=False`） |
| `max_joint_jump_rad` | π | 标量 / `{关节名: rad 或 None}` / None |
| `continuation_local` | True | 路点跟踪先求局部解 |
| `continuation_max_step` | 0.15 | 续算框半宽（rad） |
| `seed_library_size` | 16384 | 每臂样本数；0 = 不用库（建库约 0.3 s，进程内缓存） |
| `seed_library_seeds` | 8 | 联合搜索里的组合种子数 |
| `waist_grid` | (6, 11) | 组合种子的腰网格 |
| `posture_settle` | True | 常规搜索达标后在零空间整理姿态 |
| `diagnose_failures` | True | 失败时做诊断 |

## 9. 性能（`bench/dual/bench_dual.py`，结果 `bench/dual/result_pin4_1_0.txt/.json`）

单进程顺序执行，计时含 solve() 全部开销；Xeon 2.9 GHz，2 核。

| 用例 | N | 成功 | 中位 | p90 | p99 | 最长 (ms) | 失败原因 |
|---|---|---|---|---|---|---|---|
| B1 可达，q_prev = q_ready | 1000 | 100% | 5.9 | 16.8 | 26.3 | 49.3 | — |
| B2 大范围换姿，q_prev = 随机构型 | 300 | 96.7% | 10.3 | 25.7 | 70.9 | 90.2 | 跳变超限 9，NOT_JOINTLY_REACHABLE 1 |
| B3 路点跟踪（20 条 × 30 步） | 600 | 100% | 1.0 | 1.8 | 2.2 | 2.3 | — |
| B4 左手 hold、右手新目标 | 200 | 96.0% | 8.9 | 22.0 | 73.4 | 78.4 | 跳变超限 8 |
| B4 左臂 keep、右手新目标 | 200 | 97.0% | 5.8 | 11.8 | 64.0 | 67.7 | 跳变超限 6 |
| B5a 左手超出臂展（可证明） | 100 | 预检 100% | 0.6 | 0.6 | 0.8 | 0.8 | LEFT_UNREACHABLE |
| B5b 两手间距超界（可证明） | 100 | 预检 100% | 0.4 | 0.4 | 0.4 | 0.6 | NOT_JOINTLY_REACHABLE |
| B5c 两手目标来自两个独立构型 | 100 | 98% | 6.3 | 20.9 | 91.3 | 93.0 | NOT_JOINTLY_REACHABLE 2 |
| B5d 左手位置可达、姿态随机 | 100 | 96% | 6.1 | 19.4 | 92.7 | 93.9 | LEFT_UNREACHABLE 4 |

- B2 / B4 的失败是默认跳变上限 π 拦下的（参考解要转 > π，`prefer` 或调大上限即可解）。
- 诊断结论核查：10 个"搜索结果"类失败用 3 s 预算、400 个种子重解，9 个结论不变；B2 的 1 个 NOT_JOINTLY_REACHABLE 实为跳变超限（`prefer` + 400 种子才找到 3.68 rad 的解）。
- 跟踪：参考轨迹每步每关节 ≤ 0.020 rad，解的每步最大关节变化中位 0.028、最大 0.053 rad，没有 > 0.1 rad 的步。
- 对比原型（联合 + 种子组合，200 组）：100%，中位 7.0 ms，p90 106.6 ms → 现在 1000 组 100%，中位 5.9，p90 16.8，最长 49.3 ms。
- CPU 争用：种子库查询分块（低于 OpenBLAS 多线程门槛），两个 IK 进程并行时中位 5.7 ms、最长 52.8 ms、0 失败。仍建议 IK 进程设 `OPENBLAS_NUM_THREADS=1`。

## 10. 测试（`tests/test_dual_arm_ik.py`，40 个）

独立性（源码只 import numpy/pinocchio/标准库；子进程加载不拉入单臂与 scipy）· 判据与单臂常数一致 · 布局校验 ·
FK 对独立 Pinocchio · continuous 关节 2π 周期 · 雅可比有限差分（full / tool_dir / tool_cone 梯度）· 30 组随机可达 ·
小步跟踪 · 续算（不安定姿态起步每步 < 0.1 rad、已是解时原样返回、开关）· q_prev 略超限位 · hold / keep / free / None（= hold）/ 两只 None 报错 ·
姿态模式（每手不同）· joint_state · update_prev · 跨实例确定性 · 库缓存共享 · 跳变 reject / prefer / 按关节 ·
跳变超限在预算内判出 · 种子库只取跳变框内样本 · NEAR_LIMIT · 超臂展 / 间距超界立即失败 · 两手间距界与腰角无关 ·
诊断给 NOT_JOINTLY_REACHABLE · allow_approximate · 时间预算 · SOLVER_BUSY · 多线程 · 输入 / 配置校验。

运行：`python -m unittest tests.test_dual_arm_ik`；全量：`python run_validation.py`；性能：`python bench/dual/bench_dual.py [--quick]`。

## 11. 限制与后续

- 不做自碰 / 互碰检查（交给 MoveIt）。
- 只在合成 URDF 上验证；真机 URDF 若在 TCP 链上有不在 16 个活动关节里的可动关节，可达性预检自动关闭（`solver._precheck=False`），其余照常。
- P3（双手持同一物体的闭链约束、物体质量 / 负荷）未做；0.5 mm / 0.5 mrad 作为闭链验收阈值合适，内力 ≈ 抓持刚度 × 误差，真实误差来自标定与跟踪，需要柔顺控制。
