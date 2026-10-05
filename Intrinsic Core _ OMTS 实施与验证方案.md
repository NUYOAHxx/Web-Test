# Intrinsic Core / OMTS 实施与验证方案

## 版本基线与适用边界

**官方已证实：**Intrinsic Core 官方仓库将 Core 描述为面向工业机器人的开源本地 runtime、SDK 与硬件无关实时控制框架；仓库许可为 Apache-2.0，并明确声明该项目不是 Google 官方支持产品。[Core 仓库与 README](https://github.com/intrinsic-ai/intrinsic-core) Intrinsic 将 [OMTS（Open Machine Tending Solution）](https://github.com/intrinsic-ai/intrinsic-omts/blob/main/README.md)定位为基于 Core、兼容 ROS 的机床上下料参考应用，而非 Core 本身。它是参考方案，不能据此推定特定客户工位、设备或安全功能已获认证。Intrinsic 的 [Flowstate 产品页](https://www.intrinsic.ai/flowstate)描述的是另一个面向生产级机器人方案开发的环境并引导申请 Demo；不能把它与开源 Core 等同，也没有材料证明 OMTS 的 bridge bundle 对应哪个 Flowstate 服务版本。

**本项目基线建议：**为复现官方入门路径，首个集成候选版本固定为 Ubuntu 26.04 x86-64/amd64、Intrinsic Core `20260922.0`、OMTS `20260922.0`，并固定 ROS 2 Lyrical Luth、OMTS 仓库中的 Bazel 版本及所有下载物哈希。教程要求 Ubuntu 26.04；Core README 的 OS 文案却同时列 Ubuntu 24.04/26.04，并把“Ubuntu 22.04 LTS supported”放在有歧义的括号中，因此当前不能把 22.04 或 24.04 作为这条已明示教程路径的已证实兼容项。[版本化 Getting Started](https://github.com/intrinsic-ai/intrinsic-core/blob/20260922.0/developer_resources/learn/tutorials/getting_started.md)；[Core README 原文](https://raw.githubusercontent.com/intrinsic-ai/intrinsic-core/main/README.md)

Core README 列 ROS 2 Lyrical Luth；ROS wrapper 仓库 [sdk-ros](https://github.com/intrinsic-ai/sdk-ros)称默认支持 Lyrical，也支持 Jazzy；独立的 [ICON ROS 2 Control HWM](https://github.com/intrinsic-ai/icon-hwm-controller/blob/main/README.md)则以 Ubuntu 24.04 + ROS 2 Kilted 为目标，并当前限于位置控制机械臂、不支持数字/模拟 I/O。MoveIt 集成另有 Lyrical/Jazzy 依赖配置。上述分别属于不同组件声明，**不是 Core × Ubuntu × ROS 发行版的端到端兼容矩阵**，不得拼成整个平台的支持承诺。

| 组合/配置 | 官方资料可确认的内容 | 本项目处理方式 |
|---|---|---|
| Core + OMTS `20260922.0` | Getting Started 同时检出两仓库该 tag；OMTS 主干 Bazel 模块固定 Core archive 与 OMTS assets 为 `20260922.0` | 作为首个可重复基线；部署前核对对应 tag 的内容与模块锁定一致 |
| Ubuntu 26.04、x86-64/amd64 | 入门教程明确要求 Ubuntu 26.04；教程预构建物为 amd64；ARM 明确不支持 | 作为教程复现目标，不向 ARM 外推 |
| Ubuntu 22.04/24.04 | Core README 有含糊/冲突写法；教程不确认这两版 | 不纳入放行基线，除非官方书面澄清并通过 PoC |
| ROS 2 Lyrical/Jazzy/Kilted | 各组件仓库分别声明不同发行版；无统一矩阵 | 按实际 bridge/HWM/MoveIt 路径锁定，并逐项联调 |

硬件方面，Core 教程推荐 x86-64、6 核/12 线程约 4.9 GHz 或更高、32 GiB RAM（64 GiB 推荐）、NVMe 1 TB 且至少 100 GB 可用、2–3 个千兆以太网口；实时控制机器人要求 Intel CPU，AMD 可用于仿真，ARM 明确不支持。普通仿真可用集显；完整 OMTS 仿真加感知需要独立 NVIDIA GPU，教程建议 RTX 3060/4060 或更高。OMTS README 将 32 GiB 列为最低内存、64 GiB 推荐。此处保留官方“建议/条件/明确限制”的区别，不把建议配置误称所有场景的硬最低值。[Core Getting Started](https://github.com/intrinsic-ai/intrinsic-core/blob/20260922.0/developer_resources/learn/tutorials/getting_started.md)；[OMTS README](https://github.com/intrinsic-ai/intrinsic-omts/blob/main/README.md)

截至本报告核验日 **2026-10-03**，Core 与 OMTS Releases 页面所列最新 release 均为 `20260922.0`，但 Core main 仍在 10 月 1 日、OMTS main 仍在 10 月 2 日有提交；Getting Started 文档也在 release 次日修订过。Core `20260922.0` release API 的正文为空，不能由此推断具体变更或兼容政策。[Core Releases](https://github.com/intrinsic-ai/intrinsic-core/releases)；[OMTS Releases](https://github.com/intrinsic-ai/intrinsic-omts/releases)；[Core main 提交历史](https://github.com/intrinsic-ai/intrinsic-core/commits/main/)；[OMTS main 提交历史](https://github.com/intrinsic-ai/intrinsic-omts/commits/main/) 因而 main 的新文档/代码不自动等于可下载的新 runtime，复现时以 tag 与发布资产为基线，审阅 main 变化时单独记录 commit SHA 和差异。

还有一项可见历史变化需纳入版本审查：OMTS `20260922.0` tag 的 MODULE 使用固定 Core commit 的 `git_override`，并依赖 Git LFS；所查 OMTS main 改用 Core release archive `archive_override`，说明 archive 内含 materialized LFS assets。OMTS tag 与 main 的 FoundationPose 默认估计参数也不同：tag README 示例为 refinement 迭代 6、阈值 0.6/0.6，main 为 3、0.9/0.85。[tag MODULE](https://github.com/intrinsic-ai/intrinsic-omts/blob/20260922.0/MODULE.bazel)；[main MODULE](https://github.com/intrinsic-ai/intrinsic-omts/blob/main/MODULE.bazel)；[tag README](https://github.com/intrinsic-ai/intrinsic-omts/blob/20260922.0/README.md)；[main README](https://github.com/intrinsic-ai/intrinsic-omts/blob/main/README.md) 这些是仓库对照结果，不是 release narrative；不能未经复测就把 main 参数回灌给 tag 部署。

## 部署、构建与依赖获取

**官方已证实的单机教程路径：**主机在 Ubuntu 26.04 上以 k3s/Kubernetes 部署容器化 Core；教程不是独立 Docker 单容器、托管云或多节点集群部署教程。先安装 Git、Git LFS、GitHub CLI，完成 `gh auth login`，再按 `20260922.0` 检出 Core 和 OMTS。运行 Core 仓库的 `setup_k3s.sh`，按教程设置 containerd 访问权限；下载 Core release 中 `intrinsic-base-linux-amd64.tar` 并运行其启动程序，下载 `inctl-linux-amd64` 安装到 `/usr/local/bin`。需要感知包时还需 GPU 前置配置并运行 `setup_nvidia.sh`。教程也支持在另一台 PC 构建并指定部署地址，但默认示例使用 `localhost:17080`，不应将“可另机 build”扩写成集群拓扑保证。[固定版本教程](https://github.com/intrinsic-ai/intrinsic-core/blob/20260922.0/developer_resources/learn/tutorials/getting_started.md)

OMTS 构建安装 Bazelisk；OMTS 仓库 `.bazelversion` 固定 Bazel `8.8.0`，因此 Bazelisk 可执行文件下载地址使用 `latest` 不代表实际 Bazel 工具版本未锁定。教程给出 Ubuntu 26.04 下 `libxml2.so.16` 到 `libxml2.so.2` 的兼容符号链接处理；此类兼容修复应写入基线镜像自动化并记录，不要在不匹配系统上盲目创建。教程示例命令为：

```bash
bazel run //:omts_solution --config=lab_bb_01 -- \
  --address localhost:17080 --operation_mode=sim
```

文档称示例构建约 50 分钟，并预期看到 assets 处理、部署 application、wait for ready 等计时/状态。这是教程预期输出，不是本次报告实际执行结果，也不是构建时长保证。OMTS README 的较完整工作流还包括 `apply_scene_updates`、注册 FoundationPose estimator、运行 `//src:omts_app`；如照 README 主干而非 tag 操作，必须先确认相应命令/参数存在于所选版本。[OMTS README](https://github.com/intrinsic-ai/intrinsic-omts/blob/main/README.md)

依赖并非单一下载源。Core 源码与 release runtime/`inctl` 来自 GitHub；OMTS Bazel 会下载 Core archive、OMTS release 的 `flowstate_ros_bridge`、Hand-E 服务与控制 skill、Orbbec Gemini driver、RF-DETR segmentation 包；FoundationPose `refine_model.onnx` 和 `score_model.onnx` 从 NVIDIA NGC 固定版本获取。仓库中的 `models/*.glb` 使用 Git LFS。OMTS MODULE 中部分依赖与 assets 固定 SHA-256，Core archive 在 main MODULE 中列出的 SHA-256 为 `0009109250de1bec3b48abdd0a2ac36e7486447738810ebbb7fa216e2d5436f0`。[OMTS MODULE](https://raw.githubusercontent.com/intrinsic-ai/intrinsic-omts/main/MODULE.bazel)；[FoundationPose 下载定义](https://raw.githubusercontent.com/intrinsic-ai/intrinsic-omts/main/third_party/foundationpose/deps.bzl)

OMTS `.bazelrc` 配置优先尝试 Intrinsic 内容镜像，失败时允许回原始 URL，并设置 repository downloader retries=5；它不是完整离线模式，也不保证镜像覆盖所有依赖。官方明确称 `bazel test //tests/...` 为 hermetic/offline 且不需要运行集群或实体硬件，但没有承诺首次构建、模型下载或完整部署可离线。FoundationPose 权重不包含在 OMTS 的 Apache-2.0 许可中，README 指明其受 NVIDIA Open Model License 约束；需单独审阅模型使用/再分发条款。[OMTS README 与许可说明](https://github.com/intrinsic-ai/intrinsic-omts/blob/main/README.md)；[NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/) 官方公开材料未给出覆盖 OS、容器、预构建服务包、模型、固件的完整 SBOM；项目应自行生成 SPDX/CycloneDX 并做全物料许可审查。

## ROS 2 桥接与限制

**官方已证实：**ROS 集成不是一个通用“ROS 2 bridge”。`flowstate_ros_bridge` 用 World Bridge 将 Flowstate/Core 场景、机器人状态及传感器数据发布到 ROS；Executive Bridge 提供 ROS 向 Flowstate 发起请求的反向路径。Intrinsic 官方称其是旧单向 gateway 的重构双向机制，但这不意味着任意 ROS topic 自动透明转发。[Intrinsic ROS 集成说明](https://www.intrinsic.ai/blog/posts/building-stronger-connections-between-intrinsic-and-ros)；[sdk-ros](https://github.com/intrinsic-ai/sdk-ros)

桥接传感消息可包括 `sensor_msgs/JointState`、`geometry_msgs/WrenchStamped`、图像话题映射；Core 可视化教程用 `visualization_msgs/MarkerArray` 的 `/workcell_markers`。桥接配置支持机器人/力传感器 frame、关节名覆盖、服务及 Zenoh router 地址等字段。状态发布文档提及可高频约 300 Hz 或 throttle 到约 25 Hz；其示例输出 topic 名与配置默认值存在不完全一致，部署必须以实际配置和 `ros2 topic list/echo` 验证，不能仅照抄名字。[Bridge Protobuf 配置](https://github.com/intrinsic-ai/sdk-ros/blob/main/flowstate_ros_bridge/flowstate_ros_bridge.proto)；[状态/传感器说明](https://github.com/intrinsic-ai/sdk-ros/blob/main/flowstate_ros_bridge/docs/robot_state_sensor.md)

**控制通路边界尤其重要：**ROS 状态/场景 bridge 不等于实时伺服命令通道。ICON HWM 文档所示控制路径为 Realtime Control Service ↔ POSIX 共享内存/futex ↔ `icon_hwm_controller` ↔ ROS 2 Control `controller_manager`/`hardware_interface`，以 FlatBuffers 状态/命令流和锁步 tick 工作；命令接口主要是 joint position（可选 velocity feedforward），状态含 joint position/velocity。文档当前明确不支持数字/模拟 I/O。[ICON HWM README](https://github.com/intrinsic-ai/icon-hwm-controller/blob/main/README.md) OMTS 的机床 I/O 适配不可因此假设能经 ICON HWM 的 ROS 2 Control 接口实现。

Core RViz 教程示例要求 ROS Lyrical 桌面包、`rmw_zenoh_cpp`、本机 TCP `7447` 转发，Fixed Frame=`root`，Markers topic `/workcell_markers` 用 Transient Local durability；远程 GUI 另涉及 `17080`。独立 sdk-ros 的本地 LAN 示例使用 IPC 上 Zenoh router `17447`，端口情境不同，须按实际部署核对，不能混用。[Core Visualize the robot](https://raw.githubusercontent.com/intrinsic-ai/intrinsic-core/main/developer_resources/learn/tutorials/visualize_the_robot.md) RViz 是 belief world 视图，Gazebo 是模拟物理状态视图，两者都不是实体机械臂功能验收。官方教程仍将通用 ROS Connectivity（将 ROS 包/节点/topic 封装成 Core Skills）和 ROS 2 Control 接入教程列为 Coming Soon；MoveIt 集成仍在 active development，v0.0.2 release 标为 pre-release。[教程目录](https://github.com/intrinsic-ai/intrinsic-core/tree/main/developer_resources/learn/tutorials)；[MoveIt 项目](https://github.com/intrinsic-ai/intrinsic-moveit/releases)

**本项目 PoC 必查项：**固定 Core/bridge/ROS/RMW 版本、Zenoh 地址、ROS_DOMAIN_ID、frame 与 joint 命名、QoS；检查 topic 频率、时间戳新鲜度、TF 连通、断线重连和多机器人命名冲突。公开资料未定义完整 QoS 矩阵、`/clock`/`use_sim_time` 行为、相机编码/同步/吞吐上限、ROS namespace/remapping 规则，也未完整列明 Executive Bridge 的服务/动作取消、超时与错误语义。所有这些按目标组件版本台架验证，不能写成官方保证。

## 首个参考应用：单臂抓放

**设计建议（非官方验收承诺）：**以 OMTS 单臂应用结构为参考，首个单元先不接 CNC：一台 x86-64 实时 IPC、一个已支持的单臂控制资源、一只夹爪、一台 RGB-D 相机、一件尺寸/质量已知的刚性试件、带定位巢穴的料盘和一个放置窝。OMTS 示例配置包括 UR5e + Hand-E + Orbbec + CNC/Schunk vise，以及无 CNC 的 UR3e Lab BB-01；KR10 被标为占位/未接入 solution 配置，不应当作已验证机型。[OMTS 配置目录](https://github.com/intrinsic-ai/intrinsic-omts/tree/main/configs)；[OMTS 架构](https://github.com/intrinsic-ai/intrinsic-omts/blob/main/docs/ARCHITECTURE.md)

在现场测量并定义 `root`、机器人底座、相机、工具、料盘和放置窝坐标系；配置 `view`、`transit`、`pre_grasp`、`grasp`、`pre_place`、`place`、`safe_home` 帧。不要复用 OMTS 的场景数值作为本单元标定值。首轮应用状态机建议为：自检/就绪 → 定位 → 预抓取 → 抓取并确认 → 转运 → 放置并确认 → 完成；任何位姿无效、不可达、夹持不确定、接触超时或安全状态异常，都转入 HOLD/FAULT，禁止无条件自动续跑。

技能应拆分为相机定位、抓取帧生成、夹爪张开、预抓取运动、接近/接触、夹持确认、附着工件、撤离、转运、落位、松爪/脱附、回安全位。OMTS 当前提供 `cuboid_center` 抓取路径，但 `GraspPlannerInterface` 扩展点在所查主干没有具体实现；应先用固定工件/已定义姿态做可重复路径，再单独评估复杂抓取规划。[OMTS 架构文档](https://github.com/intrinsic-ai/intrinsic-omts/blob/main/docs/ARCHITECTURE.md) 接触和近距离运动只对必要的指定对象作局部碰撞排除，不得全局关闭碰撞检查；场景更新与机器人运动顺序执行，避免 world 资源锁冲突。

实机阶段将急停、防护门、扫描器等安全功能接入独立、经风险评估的安全回路。软件碰撞规划、普通数字 I/O、行为树停止按钮都不得替代安全额定功能。抓取后以夹爪状态或视觉确认持件；掉件、接触异常、IO 回读矛盾、保护停机时停止循环，由授权人员确认机器人与工件真实状态后再复位。

## 双臂协同抓放验证

**现状边界：**所查 Core README、OMTS 架构和单臂配置没有明确承诺同一 solution 支持双臂统一调度、双臂共享世界防碰撞或协同交接。下述完全是本项目 PoC 设计，不是已证实的 Core/OMTS 功能。双臂进入现场验证前，先由维护方或代码审查确认一个 solution 能实例化并寻址两套控制资源，以及实际设备驱动/实时控制器版本均适配。

PoC 分三步：先让两臂在物理分隔工作区各自独立、顺序完成抓放；再在各自包络不重叠时做并行；最后才试共同搬运/交接。共享 world 中显式建模两臂底座、工具、相机、工件、交接台、各自独占区、交叉区和全局安全退避位。交接必须有状态握手：接收臂确认夹持后，源臂才松爪；双方确认工件状态且路径清空后才进入后续动作。

先在仿真中证明共享世界同时包含两臂与工具，且规划/执行避免臂-臂、臂-工件及跨区碰撞；验证并行行为树不会触发资源锁冲突。再在实体围护、独立急停和低速人工监护下，从分区并行开始，逐步测试交接。注入一臂断连/暂停、夹持失败、互锁未到、网络中断等故障，检查另一臂不会执行危险后续动作。任一门槛失败则退回两套互不重叠的顺序单臂任务，不宣传双臂协同能力。

## 量化验收、观测与故障测试

下列数值均为**本项目建议的起始验收门槛**，不是官方性能保证；应按工件公差、节拍、安全风险和机器人厂商规格在 PoC 前批准。功能安全和停止距离由风险评估与适用法规/标准确定，不能用普通软件测试替代。

| 类别 | 可观察量/测试方法 | 建议通过门槛 |
|---|---|---|
| 部署与版本复现 | 记录 OS、Core/OMTS tag+SHA、Bazel、依赖哈希、容器 digest；干净环境部署 | 3 次独立部署 3/3 成功；各次均报告 Core pods 就绪、Solution ready；无未记录的手工改动 |
| 单臂功能 | 逐状态事件、感知结果、目标帧、抓/放确认；先 100 次短周期 | 100 次均无碰撞/掉件/错误状态推进；抓取与放置成功率分别统计，PoC 门槛各 ≥99%（不足时按故障分类整改，不以平均掩盖安全故障） |
| 生产节拍与耐久 | 按目标工艺连续至少 1,000 周期，记录每周期时间 | 有效完成率建议 ≥99.5%；记录 P50/P95/P99 周期时间，P95 不高于项目节拍上限；所有异常人工介入率单列，不计为成功 |
| 位姿/落位精度 | 独立量具测抓取、放置位置及姿态误差 | 误差 P95 ≤工艺公差的 50%，最大值不超工艺公差；如工艺公差尚未定义，不进入实机自动循环 |
| 实时性能 | 目标 IPC + 实际驱动，在感知/网络最大负载下测周期延迟/抖动/超期 | 先定义控制周期预算 T；百万周期 0 次 deadline miss，p99.99 ≤0.7T、最大值≤T。该门槛只是性能筛选；厂商实时能力与安全边界仍需单独确认 |
| 故障恢复 | 注入相机失效/遮挡、IK 无解、夹爪未夹、接触超时、I/O 超时、Zenoh 断连、进程崩溃、保护停机 | 每种故障至少 20 次；100% 转入预期 HOLD/FAULT、安全停止状态，无盲目重复动作；恢复必须有授权确认并重建 world/工件状态 |
| 双臂 PoC | 逐步并行、交接和断连注入；同步记录两臂状态与碰撞事件 | 先完成官方/代码确认的双控制资源能力门槛；无碰撞且每种单臂故障 20/20 次阻止危险续动，否则不放行双臂 |
| 复用性 | 在 UR5e 与 UR3e（设备确实可得且驱动可用时）迁移同一技能/应用，记录改动 | 业务逻辑复用率建议 ≥70%；仅改经批准的机器人/工具/frame/限位配置及薄适配层；全套回归通过。此项只证明这两种测试配置，不外推其他机型 |
| 遥测 | cycle ID、状态转移、错误码、传感器健康、IO 回读、时间戳与日志关联 | 关键事件完整率 ≥99.9%；时钟偏差 PoC 目标 ≤10 ms；断网缓存并恢复补传成功率 100%；安全动作不得依赖遥测可用 |
| 更新与回滚 | 预生产更新、故障注入回滚、配置/模型校验 | 3/3 成功回滚到已知良好版本，目标 15 分钟恢复软件基线；回滚后校准/配置校验 100% 且安全回路复验通过 |
| 供应链与许可 | 生成 SBOM、扫描、人工核验模型/依赖条款 | SPDX/CycloneDX 覆盖直接/传递依赖、OS 包、容器、预构建件、模型；未知许可证项为 0；未处置可利用 Critical/High 漏洞为 0，例外须书面限期批准 |

每个循环至少记录唯一 cycle/工件 ID、应用状态和时间、感知位姿/置信度、目标帧、规划结果/错误码、接触/力状态、夹爪命令及反馈、附着/脱附、IO 命令与回读、保护停机与人工复位原因。记录 Core、应用、模型、配置、固件版本及时间同步状态。日志/遥测服务的故障不应改变机器人安全行为；若网络断连导致运行不安全，需在本地控制策略中验证安全停止。

故障处理可参考官方教程列出的已知操作：Git LFS 模型缺失时 `git lfs pull`；Runtime/CAS 报错时检查 `app-intrinsic-base` pods 并重试部署；containerd socket 拒绝时按教程重启 artifacts-deployment pod；临时目录空间不足时设置有空间的 `TMPDIR`；教程对 `10.43.x` 服务连接拒绝建议检查本机 IP 变更并固定地址，必要时按其卸载/清理 k3s 步骤重建。[Getting Started 故障排查](https://github.com/intrinsic-ai/intrinsic-core/blob/20260922.0/developer_resources/learn/tutorials/getting_started.md) 这些仅对应文档所述错误，不代表全部故障的通用根因或恢复保证。

## 安全、网络、运维和回滚治理

版本发布物料清单应锁定：OS 镜像、内核与驱动、Core/OMTS tag 与 commit SHA、ROS 发行版及包、RMW/Zenoh、Bazel/Bazelisk、容器镜像 digest、机器人固件、相机/夹爪驱动、模型版本、所有 URL 与 SHA-256。将构建命令、配置、场景文件、安装脚本、哈希、SBOM 和人工审批纳入版本控制。Apache-2.0 只说明仓库许可，不覆盖 FoundationPose 权重或所有依赖；保留版权/许可文本、变更声明，针对二进制、模型、服务 bundle、固件分别做再分发及商业使用审查。

**安全与实时：**公开材料未给出 Core/OMTS 功能安全认证、PL/SIL 等级、特定单元急停保证或跨硬件控制周期/时延 SLA。应按工作站风险评估设计独立急停、门锁/扫描器和安全 PLC/回路；急停停止时间/距离以风险分析和适用标准/设备额定值为准。运行时测量 deadline、jitter、CPU/内存余量、负载及网络扰动；超出任何批准预算即停止试点放行。控制器配置、CPU affinity/实时优先级、共享内存权限与内核行为需要目标设备 PoC，不将官方“实时框架”描述当成数值性能保证。

**网络与遥测：**依赖下载涉及 GitHub、PyPI、Bazel 依赖源、容器镜像、NGC 等，需在构建环境为目标域名、代理和证书制定明确出站白名单。工作站网络按办公 IT、机器人控制、相机/设备、管理面隔离；仅开放经核验的协议/端口，生产控制网默认拒绝未授权入站和互联网出站。官方资料给出教程用端口/网络建议，但没有完整生产端口清单、数据出站清单、断网行为或遥测保留保证，因此由网络团队抓包验证实际流量、延迟、丢包、断网安全状态和恢复补传。

**运维更新：**设定值班责任、日志保留、配置/校准备份、补丁审批、供应链事件处理和维护窗口。每次升级先在同构预生产环境跑单元测试、仿真、故障注入与实机回归；变更 Core/ROS/模型/驱动时分别记录差异并逐阶段放行。保存已验收版本的镜像、二进制、模型、配置、构建环境和哈希，不能依赖 GitHub main 或可变 `latest` 在线恢复。定期演练备份恢复；回滚后重新检查版本一致性、坐标标定、工件/夹具真实状态、安全回路和首件质量。

**退出/回滚：**在 PoC 立项时定义退出触发条件，至少包括官方支持边界无法确认、关键驱动不兼容、严重安全/实时缺陷未解决、许可不允许目标使用、升级链不可维护或恢复演练失败。退出包须能从内部归档恢复；不得假设外部云、账号或在线依赖始终可用。具体数据格式与状态迁移需本项目 PoC，官方材料没有承诺退出工具或迁移服务。

**团队能力：**至少两名工程师能分别完成干净构建、诊断、回滚；指定机器人控制/设备集成、功能安全、网络安全和开源/模型许可负责人。单臂上线前由授权安全人员签署安全回路与风险评估，版本负责人签署物料清单，运维人员完成至少两轮断网/驱动失效/保护停机演练。若团队无法独立恢复现场到可确认的安全状态，不进入无人值守运行。

## 继续平台化或转自研 ROS 2 主干的决策门槛

建议以 8–12 周或至少 1,000 个代表性循环作为试点评估窗口。**继续使用 Core/OMTS 的门槛**：目标软硬件组合与维护方支持边界已书面确认或由可重复 PoC 覆盖；全部不可豁免的安全、实时、许可与网络硬门槛通过；更新、回滚、断网及值班恢复演练通过；应用需求中至少 70% 可由未分叉 Core/OMTS 能力或薄适配层实现；上表功能和质量门槛达标。这个 70% 是项目建议阈值，不是官方指标。

**启动自研 ROS 2 主干评估的门槛：**两轮限定期限的整改后仍不能满足安全/实时硬门槛，目标硬件或关键 ROS 集成路径长期不受支持，必要能力只能通过侵入式维护分叉实现，或平台升级/运维负担经工时核算持续超过自有集成代码维护量的 30%。进入评估不等于立即替换；先比较三年 TCO、人员储备、驱动/实时能力、许可、功能安全责任、升级责任、迁移与回滚风险，建立并行小规模 ROS 2 原型并用同一验收表对照。若任一方案无法提供合规安全回路、独立故障恢复或可审计物料基线，则两者都不放行生产。

以上验收阈值、单/双臂步骤和治理要求均为本项目建议/待 PoC，不代表 Intrinsic 官方保证。官方仓库对外说明项目处于早期开放阶段并非 Google 官方支持产品；其公开安全政策接受私下漏洞报告，但没有承诺确认/修复 SLA。[Core SECURITY.md](https://github.com/intrinsic-ai/intrinsic-core/blob/main/SECURITY.md) 因此需按实际发布、资产与具体工位完成独立安全、供应链和生产就绪审查。