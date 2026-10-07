这个问题非常关键。对于你的 IK Solver，不能只看“能不能算出一个 q”，否则很容易出现一种假象：代码能运行、末端也大致到位，但实际上算法设计有问题，或者对后面的 OMPL 非常不友好。

我建议你把 benchmark 分成三层：

1. **数学正确性：这个 IK Solver 算得对不对**
2. **算法性能：这个 IK Solver 比其他方法好不好**
3. **系统有效性：这个 IK Solver 接给 OMPL 后到底有没有价值**

也就是说，不是简单地找一个“标准答案”对比，而是建立一套完整的 evaluation protocol。

---

# 一、第一原则：不要把“关节角误差”作为唯一 Ground Truth

你的 IK 是：

\[
q\in R^{16}\rightarrow
(T_L,T_R)\in SE(3)^2
\]

而且是冗余系统：

\[
16DOF > 12DOF
\]

所以同一个目标：

\[
(T_L^*,T_R^*)
\]

本身就可能对应很多个：

\[
q_1,q_2,q_3,\cdots
\]

因此：

> **不能简单地说“我的 q 和某个参考 q 不一样，所以我的 IK 错了”。**

两个不同的关节解，只要：

\[
FK_L(q)\approx T_L^*
\]

\[
FK_R(q)\approx T_R^*
\]

同时满足：

\[
q_{min}\le q\le q_{max}
\]

就可能都是正确的 IK 解。

这也是 MoveIt 的 `KinematicsBase` 强调 seed state 的原因：对于普通 `getPositionIK()`，solver 应返回接近 seed 的有效解，而不是随机重新寻找。官方接口还直接支持多末端 Pose 的 IK。[MoveIt](https://moveit.picknik.ai/main/api/html/classkinematics_1_1KinematicsBase.html?utm_source=chatgpt.com)

所以你的 benchmark 不能只设置一个“标准 q”。

---

# 二、我建议你建立 4 个 Benchmark Level

整个 benchmark 可以设计成：

```text
                    IK Benchmark
                         │
       ┌─────────────────┼─────────────────┐
       │                 │                 │
       ▼                 ▼                 ▼
   Level 1           Level 2           Level 3
数学正确性          算法性能           OMPL系统性能
       │                 │                 │
       ▼                 ▼                 ▼
   Jacobian          Success Rate      Planning Success
   FK/Error          Runtime           Planning Time
   Joint Limits      Iterations        Path Quality
                                         │
                                         ▼
                                    Level 4
                                  鲁棒性/极端情况
```

其中 Level 1 是必须做的，Level 2 是论文/技术报告最核心的，Level 3 才是证明你的 IK 真正适合机器人系统。

---

# 三、Level 1：先证明你的 IK 数学实现正确

这个阶段甚至不要拿其他 IK Solver 比。

先证明：

> **你的 Jacobian、FK、Pose Error、QP formulation 本身没有 bug。**

## 1. FK 正确性

随机生成：

\[
q_i\in[q_{min},q_{max}]
\]

然后：

```text
q
 ↓
RobotState
 ↓
FK
 ↓
T_left
T_right
```

再检查结果。

如果你有自己以前写的 FK，也可以：

```text
MoveIt RobotState FK
        vs
你的 FK
```

比较：

\[
||p_{moveit}-p_{your}||
\]

以及：

\[
R_{moveit}^{-1}R_{your}
\]

这个是第一层 sanity check。

---

# 四、Level 1 最重要：Jacobian Numerical Check

这个我强烈建议你做。

因为你的整个 QP IK 都依赖：

\[
J
\]

如果 Jacobian 错一个符号或者 joint order 错一个位置，IK 可能表现得非常诡异。

对于每一个关节：

\[
q_i\rightarrow q_i+\epsilon
\]

然后重新 FK：

\[
T(q_i+\epsilon)
\]

得到数值 Jacobian：

\[
J_{num}
\]

再和 MoveIt 得到的：

\[
J_{analytic}
\]

比较：

\[
E_J=
\frac{
||J_{analytic}-J_{num}||
}{
||J_{num}||
}
\]

最好同时看：

```text
max absolute error
mean absolute error
relative error
```

例如测试：

```text
100 个随机 q
```

最终：

```text
Jacobian relative error
< 1e-5
```

这类量级才比较令人放心，具体阈值要根据你的数值差分步长和姿态误差定义调整。

---

# 五、Level 1：Pose Error 验证

你的：

\[
e=
[e_L,e_R]
\]

也必须单独验证。

例如构造：

```text
T_current
T_target
```

然后人工制造：

```text
translation = [0.01, 0, 0]
rotation = 5°
```

检查：

```text
position error ≈ 10 mm
orientation error ≈ 5°
```

尤其要测试：

```text
0°
180°
接近 ±π
quaternion q 和 -q
```

否则姿态误差很容易出现 discontinuity。

---

# 六、Level 1：QP 单步验证

然后测试：

\[
J\Delta q\approx e
\]

给定：

```text
q
target pose
```

求：

\[
\Delta q
\]

检查：

\[
||J\Delta q-e||
\]

是否下降。

也就是：

```text
Before:
||e||

After:
||e_new||
```

应该绝大多数情况下：

\[
||e_{new}||<||e||
\]

如果经常出现：

\[
||e_{new}||>||e||
\]

说明：

- Jacobian
- Pose error
- QP sign
- frame
- update direction

至少有一个存在问题。

---

# 七、Level 2：真正的 IK Benchmark

这一层开始比较算法。

我建议你至少设置 3 个 solver：

### Baseline A：DLS / Levenberg-Marquardt

也就是经典：

\[
\Delta q
=
J^T
(JJ^T+\lambda^2I)^{-1}e
\]

它非常适合作为 baseline。

因为它是经典 numerical IK，而且与你的 QP IK 都属于 iterative Jacobian-based IK。

---

### Baseline B：MoveIt 现有 IK

如果你的 robot group 能够配置 KDL/LMA 等 solver，就拿它做 baseline。

MoveIt 的 `KinematicsBase` 本身就有多种 IK plugin 实现，例如 KDL、LMA、IKFast 等。[MoveIt](https://moveit.picknik.ai/main/api/html/classkinematics_1_1KinematicsBase.html?utm_source=chatgpt.com)

但是这里要注意：

**如果现有 solver 只能分别解决左右臂，而不能解决你这个“腰部共享 + 双臂联合”的 16DOF 问题，那么它只能作为参考，不应该当作严格的同任务 baseline。**

---

### Baseline C：你的 QP IK

也就是：

\[
\boxed{
\text{Dual-Arm QP Position IK}
}
\]

这是最终方法。

---

# 八、最重要的 Benchmark 表

最终你应该得到类似：

| 方法 | Success Rate | Pos Error | Rot Error | Mean Time | P95 Time | Iterations | Joint Limit | Seed Distance |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DLS | | | | | | | | |
| LMA/KDL | | | | | | | | |
| QP-IK | | | | | | | | |

这里面我认为最重要的是：

**Success Rate**

**Pose Error**

**Runtime**

**Seed Distance**

**Joint-limit violation**

然后才是其他指标。

---

# 九、Success Rate 怎么定义

不能简单：

```text
solver 返回 true = success
```

因为 solver 可能错误地返回 true。

应该自己重新 FK 验证。

定义：

\[
Success=
\begin{cases}
1,&
e_p^L<\epsilon_p
\land
e_R^L<\epsilon_R\\
&\land
e_p^R<\epsilon_p
\land
e_R^R<\epsilon_R\\
&\land
q\in[q_{min},q_{max}]\\
0,&otherwise
\end{cases}
\]

例如 benchmark 可以先采用：

```text
position:
≤ 1 mm

orientation:
≤ 0.5° / 1°
```

具体阈值最后根据你的任务需求定。

这才是真正的：

> IK Success Rate。

---

# 十、不要只测试“容易的 Pose”

你的 benchmark dataset 要分层。

我建议至少：

### Dataset A：Easy

目标 Pose 从一个正常工作空间随机生成。

例如：

```text
左右手都在身体前方
```

---

### Dataset B：Workspace Boundary

目标接近：

```text
左侧最大范围
右侧最大范围
上方
下方
远距离
```

这里可以测试 solver 的工作空间边界性能。

---

### Dataset C：High Redundancy

同一个 target：

```text
不同 seed
```

例如：

```text
seed 1
seed 2
seed 3
...
seed 100
```

观察得到的：

\[
q_{solution}
\]

是否合理。

这是你的 QP + seed tracking 特别应该展示的地方。

---

### Dataset D：Near Joint Limits

故意让初始状态或者目标状态接近：

\[
q_{min}
\]

或者：

\[
q_{max}
\]

测试：

```text
QP
vs
DLS
```

这里 QP 理论上应该体现优势：

> 约束不是 solver 外部 clamp，而是直接进入优化问题。

---

### Dataset E：Near Singular

这个非常重要。

故意选择：

```text
手臂接近奇异位形
```

测试：

```text
DLS
vs
QP
```

看：

- success rate
- iteration
- joint jump
- residual
- runtime

---

# 十一、你这个项目还必须测试“左右臂耦合”

这是你与普通单臂 IK 最大的区别。

应该设计一个专门的：

## Shared-Waist Benchmark

例如：

```text
左手目标
右手目标
```

然后比较：

```text
Independent Arm IK
        vs
Whole-body Dual-arm IK
```

这是一个非常有价值的实验。

因为 Independent IK：

```text
Left IK
Right IK
```

可能分别都成功。

但最终：

```text
waist
```

没有统一优化。

而你的：

```text
DualArm QP IK
```

会同时考虑：

\[
J_L
\]

和：

\[
J_R
\]

以及：

\[
q_W
\]

---

# 十二、可以设计一个非常漂亮的实验

固定：

\[
T_L^*,T_R^*
\]

然后改变：

\[
q_{waist}
\]

的 seed。

例如：

```text
Seed A:
waist = 0°

Seed B:
waist = 10°

Seed C:
waist = 20°

Seed D:
waist = -20°
```

然后观察：

```text
最终 q
末端误差
腰部变化
总 joint displacement
```

你应该能够证明：

> 在相同双臂目标下，solver 会根据 seed 选择不同但均有效的 IK solution，并且倾向于保持接近 seed。

这实际上和 MoveIt 对 `getPositionIK()` 的 seed-nearest 语义是一致的。[MoveIt](https://moveit.picknik.ai/main/api/html/classkinematics_1_1KinematicsBase.html?utm_source=chatgpt.com)

---

# 十三、Seed Continuity Benchmark 非常重要

这个实验我认为对你甚至比单纯 success rate 更重要。

模拟真实机器人：

```text
Target 1
Target 2
Target 3
Target 4
...
Target N
```

让目标 Pose 连续变化。

例如手向前移动：

```text
P1
P2
P3
...
P100
```

每一次：

\[
q_{seed}^{k}=q_{solution}^{k-1}
\]

然后统计：

\[
\Delta q_k
=
q_k-q_{k-1}
\]

重点观察：

\[
\max |\Delta q|
\]

以及：

\[
\sum ||\Delta q||
\]

---

如果你的 IK 好：

```text
Target trajectory
      ↓
q trajectory
```

应该是连续的。

如果出现：

```text
q:
0.2
0.21
0.22
3.1
0.23
```

这种跳变，即使 Pose error 很小，也说明 solver 不适合实际机器人。

---

# 十四、这也是为什么“Seed Distance”必须成为指标

定义：

\[
D_{seed}
=
||q_{solution}-q_{seed}||_W
\]

比较：

```text
DLS
QP
LMA
```

如果你的设计目标是：

> 在满足目标的情况下尽量保持当前姿态。

那么：

\[
D_{seed}
\]

就是一个核心指标。

不是越小越好到无限小，而是：

> 在满足 task constraint 的前提下尽可能小。

---

# 十五、Runtime 怎么 benchmark

不能只测：

```text
average time
```

因为 IK 在机器人系统中最怕的是 tail latency。

建议统计：

```text
Mean
Median
P90
P95
P99
Max
```

例如：

```text
QP IK

Mean: 1.8 ms
Median: 1.5 ms
P95: 3.2 ms
P99: 4.7 ms
Max: 8.9 ms
```

这比：

```text
Average = 1.8 ms
```

有意义得多。

MoveIt 自己也有专门的 IK benchmark 示例，通过大量 IK calls 测量求解时间，这可以作为你实现 benchmark runner 的参考。[MoveIt](https://moveit.picknik.ai/main/api/html/benchmark__ik_8cpp_source.html?utm_source=chatgpt.com)

---

# 十六、Iteration Count 也必须记录

每一次：

```text
iteration
```

都记录。

例如：

```text
Mean iterations
Median
P95
Max
```

你可能发现：

```text
Easy:
5 iterations

Near singular:
30 iterations
```

这个信息对于以后优化 solver 非常重要。

---

# 十七、还有一个指标：Condition Number

因为你的：

\[
J\in R^{12\times16}
\]

存在冗余。

可以计算：

\[
\sigma_{max}
\]

和：

\[
\sigma_{min}
\]

然后：

\[
\kappa(J)
=
\frac{\sigma_{max}}{\sigma_{min}}
\]

观察：

```text
condition number
vs
IK runtime
vs
success rate
```

你可能会发现：

```text
κ(J) ↑
       ↓
QP iterations ↑
       ↓
IK failure ↑
```

这会非常有工程价值。

---

# 十八、真正重要的 Level 3：IK → OMPL

这一层才是我认为你的项目最有价值的 benchmark。

因为你的 IK 最终不是为了“论文里算一个 q”。

你的实际系统：

```text
Pose target
 ↓
IK
 ↓
q_goal
 ↓
OMPL
 ↓
trajectory
```

所以应该测试：

\[
\boxed{
\text{IK Solver + OMPL}
}
\]

而不是只测试 IK。

---

# 十九、建立固定 Planning Benchmark

这是关键。

对于每一个 testcase：

```text
固定：
Robot
Environment
q_start
T_left_target
T_right_target
```

然后分别：

```text
Method A:
DLS → q_goal → OMPL

Method B:
LMA/KDL → q_goal → OMPL

Method C:
QP IK → q_goal → OMPL
```

**所有方法使用完全相同的 OMPL 配置。**

这点非常重要。

MoveIt 官方的 planner benchmarking 也是要求不同 planner 在相同 environment、start states、queries 和 goal states 下进行比较，并统计 planning time、path length、valid path 等指标。[MoveIt](https://moveit.picknik.ai/humble/doc/examples/benchmarking/benchmarking_tutorial.html?utm_source=chatgpt.com)

---

# 二十、这里有一个非常容易犯的 Benchmark 错误

比如：

```text
DLS:
q_goal_A

QP:
q_goal_B
```

然后：

```text
OMPL(q_start → q_goal_A)
OMPL(q_start → q_goal_B)
```

发现：

```text
QP planning time 更短
```

你不能直接说：

> QP 的 IK 更好。

因为：

\[
q_A\neq q_B
\]

两个目标状态本身可能就不一样。

因此你需要至少区分两个问题。

---

# 二十一、Experiment 1：固定 IK target，比较 IK

即：

```text
same T_left
same T_right
same seed
```

比较：

```text
IK algorithm
```

指标：

```text
success
error
runtime
iterations
seed distance
joint limit
```

这是纯 IK benchmark。

---

# 二十二、Experiment 2：真实系统 benchmark

让每个 IK solver 自己产生：

\[
q_{goal}
\]

然后：

```text
OMPL
```

统计：

```text
IK success rate
+
OMPL success rate
+
total time
+
planning time
+
path length
+
joint-space path length
+
trajectory smoothness
```

这是系统 benchmark。

---

# 二十三、最重要的最终指标：End-to-End Success Rate

我建议定义：

\[
Success_{E2E}
=
Success_{IK}
\times
Success_{Planning}
\]

也就是：

```text
目标 Pose
 ↓
IK 成功
 ↓
得到合法 q_goal
 ↓
OMPL 找到 collision-free path
 ↓
E2E Success
```

这个指标非常能说明问题。

比如：

| Solver | IK Success | OMPL Success | E2E |
|---|---:|---:|---:|
| DLS | 96% | 82% | 79% |
| LMA | 98% | 86% | 84% |
| QP | 97% | 94% | 91% |

即使 QP：

```text
IK Success ≈ DLS
```

它也可能：

```text
OMPL Success >> DLS
```

这时候你就真正证明了：

> QP 的 seed tracking、joint-limit handling 和冗余解选择，使其生成的 q_goal 更适合后续路径规划。

这比单纯说“我的 QP 收敛了”有价值得多。

---

# 二十四、Path Quality 也可以比较

对于 OMPL：

\[
L_q=
\sum_i
||q_{i+1}-q_i||
\]

即 joint-space path length。

还可以比较：

```text
Cartesian end-effector path length
```

以及：

```text
maximum joint velocity
maximum joint acceleration
```

如果你的 IK 产生的目标构型更合理，通常会间接改善：

```text
planning difficulty
path length
joint motion
```

不过这里要小心：

**不要把所有优势都归因于 IK。**

OMPL planner 本身也会影响结果。

---

# 二十五、我建议你的 benchmark dataset 最终这样设计

不要只随机 100 个 Pose。

我建议：

```text
Benchmark Set
│
├── B1 Random Reachable
│      1000 cases
│
├── B2 Workspace Boundary
│      200 cases
│
├── B3 Near Joint Limits
│      200 cases
│
├── B4 Near Singularities
│      200 cases
│
├── B5 Dual-arm Coupling
│      200 cases
│
├── B6 Seed Perturbation
│      100 targets × multiple seeds
│
├── B7 Continuous Motion
│      50 trajectories
│
└── B8 Collision / Planning
       200 cases
```

不一定非要这个数量。

重点是**场景类别完整**。

---

# 二十六、Random Pose 不能直接随机位置

这个很重要。

你不能：

```cpp
x = random(-2, 2);
y = random(-2, 2);
z = random(-2, 2);
```

然后说这是 random benchmark。

因为绝大部分可能根本不可达。

更合理的是：

```text
随机生成合法 q
       ↓
FK
       ↓
得到 T_left / T_right
       ↓
把这个 Pose 当作 ground-truth target
       ↓
随机生成另一个 seed
       ↓
IK
       ↓
检查能否重新找到目标
```

这有一个巨大好处：

> 你知道这个目标一定是由机器人产生的，因此是 reachable 的。

这实际上是你的 **Ground Truth Pose Dataset**。

---

# 二十七、我甚至建议你保存 Ground Truth q

例如：

```text
case_0001

q_gt
T_left_gt
T_right_gt
q_seed
```

然后：

```text
             q_gt
              │
              ▼
             FK
              │
              ▼
        T_left/right
              │
              ▼
           IK Solver
              │
              ▼
            q_sol
```

但注意：

**q_gt 不是唯一正确答案。**

所以最终不要计算：

\[
||q_{sol}-q_{gt}||
\]

作为主要 accuracy。

而应该计算：

\[
||FK(q_{sol})-T_{gt}||
\]

同时可以额外记录：

\[
||q_{sol}-q_{gt}||
\]

用于观察 solver 找到了与 ground truth 多接近的构型。

---

# 二十八、Benchmark 最终应该形成三张核心图

如果你最后要做论文、答辩或者技术汇报，我认为最值得展示的是：

### 图 1：IK Success Rate

```text
DLS
LMA
QP
```

不同场景：

```text
Random
Boundary
Joint Limit
Singular
Dual-arm
```

---

### 图 2：Runtime Distribution

例如：

```text
Mean / P95 / P99
```

而不是只给一个平均数。

---

### 图 3：IK → OMPL End-to-End

```text
             IK Success
                  │
                  ▼
             q_goal valid
                  │
                  ▼
          OMPL Planning Success
                  │
                  ▼
            E2E Success
```

然后比较：

```text
DLS
LMA
QP
```

这张图最能证明你的工程价值。

---

# 二十九、还有一个非常重要的 Benchmark：Ablation Study

因为你这个 QP IK 不是一个简单算法，而是：

\[
QP
+
Joint Limits
+
Seed Tracking
\]

所以必须证明每一个设计到底有没有用。

做：

```text
A: DLS

B: QP

C: QP + Joint Limits

D: QP + Seed Tracking

E: QP + Joint Limits + Seed Tracking
```

然后比较：

| Method | Success | Joint-limit violation | Seed distance | Runtime | OMPL success |
|---|---:|---:|---:|---:|---:|
| DLS | | | | | |
| QP | | | | | |
| QP + JL | | | | | |
| QP + Seed | | | | | |
| Full QP | | | | | |

这会非常清楚地回答：

> **为什么你的 Solver 要这样设计？**

而不是仅仅：

> “我用了 QP，所以效果比较好。”

---

# 三十、我给你定一个最终的 Benchmark 体系

如果现在让我直接给你的项目制定验收标准，我会这样定：

```text
                    IK Solver Validation
                           │
       ┌───────────────────┼───────────────────┐
       │                   │                   │
       ▼                   ▼                   ▼
  Correctness          IK Performance      System Performance
       │                   │                   │
       │                   │                   │
   FK Error            Success Rate       E2E Success
   Jacobian Error      Runtime            OMPL Success
   Pose Error          P95 Runtime        Planning Time
   Joint Limits        Iterations         Path Length
                       Seed Distance      Joint Motion
```

然后 Benchmark 方法：

```text
                Baselines
                   │
        ┌──────────┼──────────┐
        ▼          ▼          ▼
       DLS      MoveIt IK    QP-IK
```

实验集：

```text
Random Reachable
Workspace Boundary
Joint Limits
Singularity
Dual-arm Coupling
Seed Perturbation
Continuous Target
Collision / Planning
```

最终核心 KPI：

\[
\boxed{
Success\ Rate
}
\]

\[
\boxed{
Pose\ Residual
}
\]

\[
\boxed{
Runtime\ P95/P99
}
\]

\[
\boxed{
Seed\ Distance
}
\]

\[
\boxed{
Joint\ Limit\ Violation
}
\]

\[
\boxed{
OMPL\ Success\ Rate
}
\]

\[
\boxed{
End\text{-}to\text{-}End\ Success
}
\]

---

## 最后一个判断：什么才叫“这个 IK Solver 做得好”？

对你的机器人，我不会简单定义成：

> “比 DLS 快。”

真正合理的定义应该是：

> **在相同双臂目标、相同机器人模型和相同 seed 条件下，能够稳定、高成功率地找到满足双末端位姿约束的 16-DOF 关节解；同时满足关节限位，并保持与当前构型的连续性；最终生成的 q_goal 能够提高 MoveIt/OMPL 的规划成功率和整体执行质量。**

这才与你的实际系统目标完全一致。

而且这个 benchmark 方案还有一个好处：**它可以直接反过来指导我们实现 Solver。** 我们不是先把代码写完再想怎么证明它，而是先把测试指标定下来，然后让 `DualArmIKSolver` 从一开始就输出 `iterations / residual / seed_distance / runtime / failure_reason` 等 benchmark 所需数据。

下一步我建议直接把这套方案进一步落成一个 **\`IK Benchmark Specification\`**：把测试数据生成器、DLS baseline、QP-IK、MoveIt baseline、所有指标的精确定义、CSV 输出字段，以及最后画哪些图全部定义下来。这样你后面写代码时，就可以按照这个 specification 一项一项实现，而不是边写边猜怎么验证。