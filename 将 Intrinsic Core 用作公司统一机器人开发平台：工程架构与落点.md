# 将 Intrinsic Core 用作公司统一机器人开发平台：工程架构与落点

Intrinsic Core 可作为公司机器人应用的本地运行时、SDK/API 与可组合机器人软件基础；Intrinsic OMTS（Open Machine Tending Solution）则提供一套可研究和定制的机器看护参考应用。工程上宜把 Core 当作平台依赖、OMTS 当作可选择性复用的应用骨架，并把自研能力分别放到 Skill、硬件模块/服务、Solution 配置或应用行为树等清晰边界内，而不是把整个 OMTS 当作通用平台复制。**【源码事实】**本报告所依据的 Core 与 OMTS 官方仓库均属于 `intrinsic-ai` 组织，并通过官网发布文章、README 互链核实；当前主分支调查快照分别为 Core `e87d7c6c60b7f9ad588ec2dbba4ad45016d9f82f`、OMTS `8b3887c87a94aaf721fd0c7dff5fee480f1d6d1e`。[R1][R2][R3]

## 版本基线：把主分支、发布包和教程锁定版本分开

**【源码事实】**截至 2026-10-03，Core 与 OMTS 已核验的最新公开 release/tag 均为 `20260922.0`；Core 另有 `20260921.0`。GitHub Compare API 显示 `20260922.0...main` 的 Core 主分支领先 258 个提交，OMTS 主分支领先 14 个提交。因此，当前 main 源码、发布 tag 和发布归档不是同一快照，报告中的源码分析不能自动套用到发布包。[R4][R5][R6][R7]

**【源码事实】**Core Getting Started 教程的安装示例要求 checkout 两个仓库的 `20260922.0` tag；与此同时，OMTS `main` 的 `MODULE.bazel` 通过 URL 与 SHA-256 固定依赖 Core `20260922.0` 发布归档。OMTS main 的行为因此受这个固定 Core 归档和 OMTS 自身补丁共同决定，而不是随着 Core main 自动更新。[R8][R9]

**【工程推断】**公司应为每个可复现构建记录 Core release/归档 SHA-256、OMTS tag 或 commit、OMTS 外部 bundle 版本与校验值、Bazel 配置及目标 cell。先从文档给出的同版本 tag 组合建立基线，再单独评估 main；除非完成构建、测试和运行验证，不应把 Core main 直接替换为 OMTS pin 的 Core release，也不应假设教程 tag 在全部目标操作系统和硬件上都能成功部署。**【未知】**本次没有执行构建、全量测试、部署或硬件验证，tag 组合的跨平台可构建性仍待公司环境实测。[R8][R9][R10]

操作系统要求也应当作为上线前确认项：**【源码事实】**Core README 与 Getting Started 对 Ubuntu 版本的表述不一致，README 提到 Ubuntu 24.04/26.04 并括注 Ubuntu 22.04 supported，而 Getting Started 第 8–10 行称需要 Ubuntu 26.04。本次材料不足以形成统一支持矩阵，应按所选 Core 版本向官方文档/支持渠道核实并在公司 CI 中固定验证环境。[R8][R11]

## 平台分层与代码归属

**【官方主张】**Core README 将项目描述为机器人本地 runtime、SDK 和控制框架，并把能力划分为 `intrinsic_runtime`、`intrinsic_control`、`intrinsic_motion_planning`、`intrinsic_perception`、`intrinsic_inference`、`intrinsic_sdk` 与 `intrinsic_apis`。README 和发布文章进一步宣称本地运行、ROS 兼容、硬件无关实时控制、数字孪生及从原型到生产等能力；这些产品定位应与下文逐项可见的源码机制区分，不能当成本次实测结论。[R1][R12][R13]

**【源码事实】**OMTS 将可部署的 `//:omts_solution` 与应用 `//src:omts_app` 分开：solution 声明硬件、场景、感知服务与 world updates；Python app 通过 gRPC 连接已部署 solution、读取 YAML 配置、组装 adapters 和 `bt.BehaviorTree`，再交给 `solution.executive.run(tree)` 执行。这个“平台 solution + 客户端应用”的边界是公司拆分部署资产与工艺流程代码时可直接参照的结构。[R14][R15][R16]

**【工程推断】**公司可在 Core 提供的 SDK/API 之上维护内部公共 Skill 与设备资产仓库，把客户/产线差异留在 cell 配置和应用行为树；多个工作流复用的能力不要从 OMTS app 复制成另一套私有执行/打包机制。OMTS 更适合作为具体单臂 machine-tending 应用的参考实现，只有其设备拓扑、工作流和许可证适配均符合需求时才 fork。两仓库 README 声明 Apache-2.0，但这不代表其他 Intrinsic 服务、预构建依赖或第三方模型资产均适用同一授权；交付前仍需逐项审查 LICENSE、资产许可及商业合同。[R17][R18]

## 构建、SDK/API 与本地连接方式

**【源码事实】**Core Getting Started 描述的本地路径包括准备 Ubuntu 主机、安装 k3s、运行 `intrinsic_runtime/setup_k3s.sh`、下载 release 的 `intrinsic-base-linux-amd64.tar` 并在本机解包运行，以及取得 `inctl`。Core 发布 `20260922.0` 页面列出 `inctl-linux-amd64`、`intrinsic-base-linux-amd64.tar` 和 `intrinsic-core.tar.gz`。这表明公开教程存在本地部署入口，但不是对其在本公司的执行成功证明。[R8][R19]

**【源码事实】**Core SDK 的 `intrinsic_sdk/intrinsic/solutions/deployments.py` 提供 `deployments.connect(address=...)`，得到代表已部署 solution 的 `Solution` 对象；对象可访问 `executive`、`skills`、`world`、`simulator` 等组件。OMTS `src/main.py` 默认连接地址为 `localhost:17080`，调用 `deployments.connect`，构造行为树后调用 `solution.executive.run(...)`。Core `Executive.run` 接收行为树/动作并提交执行。这条调用链是本地客户端连接已部署运行时的具体 SDK 使用样例。[R20][R21][R22]

**【工程推断】**公司可复用 SDK 的 `Solution` 接入与生命周期调用模式，但应把平台 API 客户端封装在内部应用公共库中：统一 endpoint、超时、日志、取消和错误映射；业务流程只依赖封装后的能力，避免每个应用自行拼接连接参数。超时、取消和生产级重连策略需按具体 SDK 版本及运行态验证；上述调用链本身没有证明公司所需的 SLA 或故障语义。

**【源码事实】**Core API 中既有本地运行时相关服务接口，例如 `PoseEstimationService`、`SimulationService`，也有标注 API-key auth、组织上下文及 frontend backend 的云端 solution/cluster discovery proto。两类 API 的存在不能说明任何 OMTS 本地流程都必须调用云端；OMTS 示例采用 localhost solution 地址，未在所查调用路径出现 cloud discovery 调用。[R23][R24][R21]

**【工程推断】**基础本地应用可按 Core 本地 runtime + SDK/API + solution 组织；云端 discovery、Flowstate 或企业服务则作为单独集成选项评估。**【未知】**本次没有核实 Flowstate 授权/订阅、云服务托管边界、区域可用性、企业 SLA、所有 API 的部署暴露与认证配置；不能据此做商业采购或生产支持承诺。[R13][R24]

## Skill、Asset 与 Solution：定义复用能力的三个边界

**【源码事实】**Core 的 `SkillManifest` proto 可描述 Skill ID、文档、参数 protobuf 完全限定名、返回消息类型、依赖、取消支持及关联资产等；Python `Skill` 接口将项目定义与执行接口结合，`execute` 接收参数和执行上下文，返回 protobuf 或 `None`。Core 的 `py_skill` 构建规则把 Python Skill、manifest、image 和 descriptor set 打包为 Skill bundle。示例教程 `custom_asset_creation_software.md` 展示 `say_skill.proto`、Python 实现、manifest、`py_skill` 构建与 `inctl skill install` 安装；教程样例位于外部 `sdk-examples`，不应误称为 OMTS 内已跟踪实现。[R25][R26][R27]

**【工程推断】**当一项能力需要稳定输入/输出契约、独立部署且跨多个流程复用时，应新建 Core SDK 风格 Skill：定义参数/结果 proto、实现 `Skill`、维护 manifest 与依赖，再通过官方构建规则生成 bundle。应明确 cancellation 与资源 footprint：接口文档指出支持取消要在 manifest 声明，并要求取消时安全、可恢复；默认 footprint 的 `lock_the_universe=True` 也意味着并发资源特性应被显式审查，而不是默认假设可并行。[R26][R27]

**【源码事实】**Core 的本地 Solution schema 将 Asset 引用与 Asset instance 分开：资产可以是本地资产或 catalog 引用；instance 结构另含实例名、backing asset 和配置文件路径。Core `intrinsic_asset_instance` 规则保存资产 ID/name/config。Core 教程还将 `inctl asset install`、服务实例创建和 world reset 描述为不同步骤；Skill 则按教程所述在 Process 中参数化调用，不像普通硬件/服务 Asset 那样直接作为 Solution instance 添加。[R28][R29][R30]

**【工程推断】**公司应把“资产包发布/安装”“Solution 声明包含什么资产”“某个 cell 上如何实例化资产”分别纳入版本控制。硬件、服务、场景模型作为 Solution assets；每个 workcell 的 instance 配置独立保存；Skill 用 manifest 定义能力契约并在流程中绑定参数。OMTS 的 `bazel/imported_asset.bzl` 可用于把供应商已打包 bundle、manifest 和描述符导入 Bazel solution，但导入预编译资产并不等于拥有其 Skill 源码或实现。[R31][R32]

**【未知】**本次检查的 Core/OMTS 源码和教程未确认通用 `inctl asset export` 命令或资产中心注册表实现；不能把 `asset install` 或 `imported_asset_bundle` 描述成导出功能。Core glossary 将 hosted Catalog 描述为 Intrinsic Enterprise 目录；具体可用性和商业边界须另行核对。[R30][R33]

## Runtime、ICON/HAL 与设备接入

**【官方主张】**Core README 将 `intrinsic_runtime` 描述为本地执行环境，将 ICON 描述为实时运动与硬件协调机制，并宣称 HAL 有助于减少硬件更换时重写驱动的工作量。Runtime 生命周期管理及硬件无关范围属于官方定位，以下公开代码只验证架构组件与特定实现存在，不验证所有设备和生产场景。[R12]

**【源码事实】**Core main 包含 ICON `MainLoop::Create`、`create_icon_main_loop.cc`、HAL `HardwareModuleManager`/`module_config.h`，以及 `joint_command.fbs`、`joint_state.fbs`、`io_controller.fbs`、`gripper.fbs` 等 HAL 数据契约；也有 Universal Robots 的 `ur_module.cc`/`rtde_ur_driver.cc`、KUKA RSI 模块以及 fake 模块。接口/序列化类型、模拟实现和具体设备模块是不同证据层级，某模块源码存在并不证明所有型号、固件或现场已经验证。[R34][R35][R36]

**【源码事实】**OMTS 的 `configs/omts/icon_config.textproto` 登记 UR 模块及其 ICON 时钟驱动，`ur_module_config.textproto` 配置 Universal Robots 与 `robot_ip`；`configs/README.md` 将 `kr_10` 明确标记为 stub/placeholder，且 OMTS 当前默认 solution 未接入该 KR10 配置。Core 另有 KUKA RSI HWM 源码与 bring-up 教程，但它不能替代 OMTS 中明确标记未接线的 KR10 cell。[R37][R38]

**【源码事实】**OMTS `src/hardware/robot.py` 通过已连接 Solution 的 SBL skills 构建机器人动作；这是应用层适配器，不是其 Python 代码直接实现 UR/KUKA 控制器通讯协议。Core `setup_k3s.sh` 和 `setup_realtime.sh` 是主机环境准备脚本，也不能直接视为统一应用启停控制器。[R39][R40]

**【工程推断】**若需接入新机器人，优先从 Core 的 ICON/HAL 设备模块及其 bring-up 教程复用接口和配置模式；若设备只能通过 ROS 2 `ros2_control` 暴露，则先验证后文的 ICON controller 路径。公司代码应把型号/固件特定通讯和安全适配落在设备驱动/HWM 或独立服务资产内，把机器人 frame、IP、控制器参数、限位与 cell 几何放在版本化实例配置中。**【未知】**本次未追踪完整 wire protocol、实时周期、异常恢复、认证/安全行为或具体硬件实机表现，必须在目标设备上做故障注入与安全验证。[R34][R35][R41]

## 行为树、ObjectWorld 与运动规划

**【源码事实】**Core 仓库包含行为树 proto、sequence 与 loop 执行规则；OMTS 的 `src/behaviors/machine_tending_bt.py` 用 `bt.Sequence` 按取料、装机、机床加工、卸料、回到进料位组织流程，再按配置循环。`src/main.py` 创建并运行该树。该例适合作为“流程编排调用 Skills”的参考，而非把行为树自身当作低层机器人控制器。[R42][R16][R21]

**【源码事实】**OMTS `src/hardware/robot.py` 将目标 frame 转为 `PoseEquality`/`MotionSegment`，可附加 collision settings；接触移动由 `move_to_contact` Skill 接收方向、接触力和 timeout。Core 的 `trajectory_planning/validation.cc` 对轨迹端点/采样点调用 configuration validation 进行碰撞检查。该代码证明请求构造和一种验证逻辑存在，但不证明任意连续轨迹、任意场景均无碰撞、不会死锁或满足实时期限。[R43][R44]

**【源码事实】**Core ObjectWorld 教程通过 ObjectWorldUpdates 定义对象关系和初始位置、创建/更新 frame、reparent object 及更新关节状态；OMTS world 工具将 scene、attachments、robot alignment、gripper、camera mount 等更新文件应用到运行中的部署，并可选择是否更新初始 world。OMTS 机器人适配器还构造 attach/detach 技能动作，用于抓取前后更新工件和机器人之间的关系。[R45][R46][R47]

**【工程推断】**公司可把 ObjectWorld 作为场景几何、命名坐标系和工件关系的共享模型；把工艺顺序、重试/恢复策略放在行为树；把规划约束和低层执行交给相应 Skill/服务。OMTS 的 `src/behaviors/`、`src/hardware/`、`src/core/config.py` 与 `configs/<cell>/` 是工艺型参考应用的优先 fork 点。若只是调整取放顺序、cell frame 或阈值，改行为树/配置；若形成跨流程可复用的独立动作契约，再提炼成 Skill。动态 world 更新的一致性、并发与恢复语义仍需目标运行时验证。[R42][R45][R46]

## 视觉与推理：把可见的 6D 位姿链与语义 AI 区分

**【源码事实】**Core 的 `estimate_pose_multi_view.py` 构造请求并调用 `PoseEstimationService`，对 ROI 与最少实例数过滤结果，取高分位姿并可更新 ObjectWorld；这证明了感知 Skill/服务调用链，而不表示该 Skill 内实现了 FoundationPose 神经网络。OMTS `src/hardware/vision.py` 调用 Core 的 `estimate_pose_multi_view`，再由 `dynamic_frame_calculator.py` 转换 camera frame 位姿、做高度检查并生成抓取 frame。[R48][R49][R50]

**【源码事实】**OMTS `src/foundationpose/BUILD` 将 FoundationPose 作为 Triton 后端的 MLModel 资产打包，`third_party/foundationpose/deps.bzl` 固定 NVIDIA Isaac ROS 源码并从 NGC 获取 ONNX 权重；注册工具调用 Core Train Service 注册 pose estimator。教程还说明启用 perception packages 需要 GPU 配置。由此可见应用需要模型资产、推理后端、相机资源及运行时服务配合，不能仅靠 OMTS Python 源码独立完成视觉推理。[R51][R52][R53][R54]

**【源码事实】**在所审查的 Core/OMTS 当前主分支、README、教程和配置中，未发现语言模型、自然语言驱动任务规划或通用语义推理的实现证据。**【工程推断】**可将已核实能力描述为几何/目标位姿估计和显式编排的视觉抓取链，不应称为开放词汇语义理解或 LLM 自主规划。**【未知】**关联闭源服务、Intrinsic Vision Model 或其他官方仓库是否提供更广泛 AI 功能，本次没有审计；此外 OMTS 把同一个 camera resource 传入 camera_1 到 camera_4，不能据此推断有四个独立相机。[R48][R49][R55]

**【工程推断】**若公司要复用现成的姿态估计流程，优先 fork OMTS `src/behaviors/pick.py`、`src/hardware/vision.py` 与 dynamic-frame calculator，并把相机标定、estimator ID、ROI、质量阈值和抓取 frame 放入显式配置。若需求是新的模型或跨业务推理服务，应单独定义模型资产及 Skill/API 边界；不要把服务调用编排代码误认成模型本身，也不要默认 `segmentation.tar.gz` 已在 OMTS 行为流程中使用。[R49][R50][R51][R56]

## ROS 桥接与 cell 配置

**【官方主张】**Core README 宣称 ROS 2 interoperability；但公开源码呈现的是多个范围不同的集成合同，而非已证实覆盖任意 ROS topic/service 的通用转换层。[R12][R57]

**【源码事实】**Core `ros2_bridge_tutorial.md` 的实际控制路径是将 `icon_hwm_controller` 做成同时实现 ROS 2 `ros2_control` Controller 与 ICON `HardwareModuleInterface` 的桥接，并通过 shared-memory IPC 连接 ICON。教程要求目标机器人准备 ROS package、Dockerfile、launch/controller 配置与 Bazel 构建，再生成 Intrinsic service asset；保留的 `joint_state_broadcaster`、`joint_trajectory_controller` 与 operational-state 管理是具体驱动集成要求。[R58]

**【源码事实】**其他 ROS 路径也有各自边界：`ros_proto_conversion.py` 明确覆盖 Point、Vector3、Quaternion、Pose 等几何类型；`RosImageSource` 是专用 ROS 相机图像路径，并对可处理的图像编码作出实现。OMTS 将 `flowstate_ros_bridge.bundle.tar` 作为预构建资产纳入 Solution，manifest 指向归档名；其桥接服务实现不在本次两个仓库源码中。Core RViz 教程要求本机 ROS 2 Lyrical 会话配置 `RMW_IMPLEMENTATION=rmw_zenoh_cpp`、Zenoh endpoint 指向 localhost:7447。这些分别是控制、类型转换、相机和可视化路径，不可互相推广为完整 ROS 兼容证明。[R59][R60][R61][R62]

**【工程推断】**设备集成方应分别维护 ROS driver/launch/controller 参数与容器打包，以及 Intrinsic 侧 HWM/service asset、ICON instance、robot/cell config 和场景更新；不要把 RViz 客户端变量当成生产网络、容器或安全配置。OMTS `configs/README.md`、`BUILD` 和 `src/core/config.py` 可作为配置分层参考：Bazel setup 选择 cell，`.textproto` 配 ICON/硬件服务，`.pbtxt` 配 ObjectWorld 更新，YAML 配应用/工艺参数。[R37][R63][R64]

**【源码事实】**OMTS 默认配置链明确接入 UR5e/UR3e 单臂工作单元；`kr_10` 被标记为 stub 且未接入默认 solution。**【未知】**Flowstate ROS Bridge bundle 的消息白名单、QoS、协议转换、安全机制和版本兼容性未在目标仓库公开；Core 的 ROS proto conversion 与相机代码也不能推出 ROS 消息全覆盖。需检查独立 `sdk-ros`、`icon-hwm-controller`、相机驱动代码及部署网络条件后再定集成合同。[R37][R38][R61][R65]

## 单臂抓放与双臂协同：证据边界必须严格

**【源码事实】**OMTS 的 `RobotConfig` 是单个 robot 与工具 frame 配置，`AppConfig.robot` 为一个 `RobotConfig`；`build_pick_from_infeed_subtree(robot, gripper, ...)` 接收一个 robot 和一个 gripper；总行为树接收一个 `robot`、一个 `gripper`，并将同一 robot 传给取料、装料、加工、卸料和回料子树。Lab BB-01 配置也指定一个 `ur_module` 和一个 `gripper`。这些源码共同证明 OMTS 所示流程是单臂抓放/机床上下料工作流。[R66][R67][R68]

**【源码事实】**Core 教程提到实时控制服务可管理一个或多个 robots 或其他实时设备（例如 gripper、digital outputs），并描述可更换机器人/夹爪。**【工程推断】**这支持“平台架构可能连接多个机器人/设备”的判断，但多个设备可接入不等价于双臂合作任务规划、两臂同步、共同持物、互相避碰或故障协同。禁止从硬件无关、通用多设备架构或单臂样例推出 Core/OMTS 已具备双臂协同能力。[R69]

本次对两仓库公开源码、教程和配置的审查**没有发现确凿的双臂/双手协同实现证据**，包括双臂任务 API、双机器人同步协调样例或双臂共同操作同一工件的测试。此结论是证据范围内“未发现”，并非证明 Intrinsic 其他仓库、闭源组件或未公开交付中绝无相关能力。**【未知】**目标机器人、控制器、两套 HWM 并行配置、跨臂碰撞场景以及协同技能是否能通过平台组合实现，都需专项设计与实验验证。[R66][R67][R68][R69]

**【工程推断】**公司若计划双臂项目，应单独设立技术验证而不是从 OMTS fork 后直接宣称支持：先确认两个机器人及同步时钟/状态语义、跨机器人 ObjectWorld 与碰撞规划支持、任务并发/互斥模型、共同持物和恢复动作、安全停机行为；再实现可审查的双臂测试工作流与硬件验收。以上是验证问题清单，不是这些能力已存在的事实。

## 从哪个示例 fork，以及公司代码放在哪里

下列选择基于已核实的仓库结构，属于**【工程推断】**而非官方强制目录规范：

- **已有 Core Skill 只需改参数/接入新流程：**优先复用 Core `SkillManifest`、Python `Skill` 接口与 `py_skill` 规则；新代码放公司 Skill package，维护 proto、实现、manifest、依赖与 bundle target。以 Core `custom_asset_creation_software.md` 的 `say_skill` 生命周期作最小示例；不要把该文档示例误当作 Core 仓库已跟踪源码。[R25][R26][R27]
- **需要自有硬件/服务/场景资产：**从 Core `custom_asset_creation_hardware.md` 的 BUILD、manifest、SDF、bundle 与 service instance 步骤复用；资产声明和实例配置分开。供应商已有预编译资产时可参考 OMTS `bazel/imported_asset.bzl`，但记录其外部版本、校验值和许可证，不复制其内部实现假设。[R28][R30][R31][R70]
- **需要完整 machine-tending 参考应用：**以 OMTS `src/behaviors/`、`src/hardware/`、`src/core/config.py`、`configs/<cell>/` 与 `docs/ARCHITECTURE.md` 为可选 fork 起点；把工艺顺序、应用级阈值与异常恢复留在 app/行为树，把共用原子能力抽成 Skill。先在官方支持的 UR cell 配置上运行验证；不要以未接入的 KR10 stub 为硬件支持证据。[R14][R15][R37][R38][R42]
- **需要新的 ICON 机器人接入：**先按 Core ROS2 bridge 教程判断是否已有适配的 `ros2_control` 路径，或从 Core 的 UR/KUKA HWM 与 `robot_bringup.md` 研究具体驱动、配置和时钟模块接口；新增型号实现与型号配置应独立版本化，避免将设备通信细节塞进 OMTS 业务树。[R35][R36][R41][R58]
- **需要新视觉流程：**复用 OMTS `vision.py`、`pick.py`、dynamic-frame calculator 的“捕获—姿态估计—坐标变换—构造抓取动作”结构；新模型独立作为 MLModel/服务资产并验证 GPU、数据与许可，不能仅改 OMTS Python 就视为模型能力已交付。[R48][R49][R50][R51]
- **需要新 cell，而不是新能力：**优先新增版本化 cell 配置、ObjectWorld 更新和 Solution asset/instance 选择；若某项能力应跨 cell 独立部署，则回到 Skill/Asset 边界建模，而非为每个 cell 复制逻辑。[R28][R37][R63][R64]

## Core-only、商业产品与待确认边界

**【源码事实】**官方 Core Getting Started 和 OMTS README 展示了可本地执行的基础路径：k3s/Core runtime、Bazel 构建 solution/app、本机地址连接 solution；OMTS 构建明确从 Core 发布归档和若干外部 driver/model/bridge bundle 获取依赖。**【工程推断】**就所查本地示例而言，Flowstate、云端集群发现或托管云 API 不是显式执行前置；但 OMTS perception/硬件路径仍可能需要发布包、外部资产、GPU 和运行中的 Core 服务，不能简化为“只 clone 源码即可运行”。[R8][R9][R21][R23][R24][R51]

**【官方主张】**Intrinsic 官方文章和产品页把 Core 视作可与 Flowstate、先进 AI 模型、工业级云服务协同的开源能力基础，并宣称可从原型走向生产；Flowstate 产品页描述其开发环境、仿真和实机测试能力。这些是官方产品主张，不是本次仓库构建/测试所验证的功能，也不意味着这些商业平台能力随 Core 源码一同交付。[R13][R71]

**【未知】**公开材料不足以确认 Flowstate 是否必须联网/托管、可否完全本地运行、具体许可和收费、模型服务交付方式、商业支持与 SLA，也不足以划定哪些能力属于 Core-only、哪些必须购买服务。商业选型需向官方取得合同、部署架构和功能矩阵；工程审计则应单独验证外部 bundle、模型、驱动和 Core release 的兼容性。[R13][R71]

**【源码事实】**OMTS 测试说明将其测试称为无需 live cluster/hardware 的 hermetic unit tests，命令为 `bazel test //tests/...`，并列出覆盖范围；CI 配置包含格式/lint 与 build/test job。**【工程推断】**公司可以复用该测试分层：先离线单测与静态检查，再做 solution/app 集成构建、模拟器/视觉资产验收，最后做真机与安全验收。**【未知】**本次未运行这些命令，也没有据此确认 21 个 target 当前 CI 全部通过或现场硬件性能。[R10][R72]

## References

以下编号在正文中稳定复用；源码链接尽量固定到调查时的 commit 或 release tag。`main` 链接用于便于浏览的官方当前文件，但会随上游变化；若行号/文件变化，应以本节注明的调查 SHA 为准。

- **[R1]** Intrinsic 官方发布文章（2026-09-22）：https://www.intrinsic.ai/blog/posts/introducing-intrinsic-core
- **[R2]** Core main README（调查 SHA `e87d7c6c60b7f9ad588ec2dbba4ad45016d9f82f`）：https://github.com/intrinsic-ai/intrinsic-core/blob/e87d7c6c60b7f9ad588ec2dbba4ad45016d9f82f/README.md
- **[R3]** OMTS main README（调查 SHA `8b3887c87a94aaf721fd0c7dff5fee480f1d6d1e`）：https://github.com/intrinsic-ai/intrinsic-omts/blob/8b3887c87a94aaf721fd0c7dff5fee480f1d6d1e/README.md
- **[R4]** Core release `20260922.0`：https://github.com/intrinsic-ai/intrinsic-core/releases/tag/20260922.0
- **[R5]** OMTS release `20260922.0`：https://github.com/intrinsic-ai/intrinsic-omts/releases/tag/20260922.0
- **[R6]** Core compare `20260922.0...main`：https://api.github.com/repos/intrinsic-ai/intrinsic-core/compare/20260922.0...main
- **[R7]** OMTS compare `20260922.0...main`：https://api.github.com/repos/intrinsic-ai/intrinsic-omts/compare/20260922.0...main
- **[R8]** Core Getting Started（版本、Ubuntu、k3s 与本地安装）：https://github.com/intrinsic-ai/intrinsic-core/blob/main/developer_resources/learn/tutorials/getting_started.md
- **[R9]** OMTS MODULE.bazel（Core 固定 release archive 与 SHA256）：https://github.com/intrinsic-ai/intrinsic-omts/blob/main/MODULE.bazel
- **[R10]** OMTS 测试说明：https://github.com/intrinsic-ai/intrinsic-omts/blob/main/tests/README.md；CI：https://github.com/intrinsic-ai/intrinsic-omts/blob/main/.github/workflows/ci.yml
- **[R11]** Core README（Ubuntu 前置条件）：https://github.com/intrinsic-ai/intrinsic-core/blob/main/README.md
- **[R12]** Core README 模块描述：https://github.com/intrinsic-ai/intrinsic-core/blob/main/README.md#L21-L32
- **[R13]** Intrinsic Core 产品页：https://www.intrinsic.ai/intrinsic-core
- **[R14]** OMTS 架构说明：https://github.com/intrinsic-ai/intrinsic-omts/blob/main/docs/ARCHITECTURE.md#L10-L45
- **[R15]** OMTS README 构建/运行与资产准备：https://github.com/intrinsic-ai/intrinsic-omts/blob/main/README.md#L128-L251
- **[R16]** OMTS behavior tree 实现：https://github.com/intrinsic-ai/intrinsic-omts/blob/main/src/behaviors/machine_tending_bt.py#L115-L164
- **[R17]** Core LICENSE：https://github.com/intrinsic-ai/intrinsic-core/blob/main/LICENSE
- **[R18]** OMTS LICENSE：https://github.com/intrinsic-ai/intrinsic-omts/blob/main/LICENSE
- **[R19]** Core `20260922.0` 发布资产：https://github.com/intrinsic-ai/intrinsic-core/releases/tag/20260922.0
- **[R20]** Core SDK deployments API：https://github.com/intrinsic-ai/intrinsic-core/blob/main/intrinsic_sdk/intrinsic/solutions/deployments.py#L82-L105
- **[R21]** OMTS app 连接与 executive 调用：https://github.com/intrinsic-ai/intrinsic-omts/blob/main/src/main.py#L41-L45；https://github.com/intrinsic-ai/intrinsic-omts/blob/main/src/main.py#L89-L165
- **[R22]** Core Executive.run：https://github.com/intrinsic-ai/intrinsic-core/blob/main/intrinsic_sdk/intrinsic/solutions/execution.py#L818-L845；Solution 访问器：https://github.com/intrinsic-ai/intrinsic-core/blob/main/intrinsic_sdk/intrinsic/solutions/deployments.py#L390-L426
- **[R23]** Core PoseEstimationService proto：https://github.com/intrinsic-ai/intrinsic-core/blob/main/intrinsic_apis/intrinsic/perception/proto/v1/pose_estimation_service.proto#L31-L69；SimulationService：https://github.com/intrinsic-ai/intrinsic-core/blob/main/intrinsic_apis/intrinsic/simulation/service/proto/v1/simulation_service.proto#L25-L40
- **[R24]** Cloud Solution Discovery proto：https://github.com/intrinsic-ai/intrinsic-core/blob/main/intrinsic_apis/intrinsic/frontend/cloud/api/v1/solutiondiscovery_api.proto#L77-L93；Cluster Discovery：https://github.com/intrinsic-ai/intrinsic-core/blob/main/intrinsic_apis/intrinsic/frontend/cloud/api/v1/clusterdiscovery_api.proto#L80-L93
- **[R25]** SkillManifest proto：https://github.com/intrinsic-ai/intrinsic-core/blob/main/intrinsic_apis/intrinsic/skills/proto/skill_manifest.proto#L79-L85；参数及返回等字段：https://github.com/intrinsic-ai/intrinsic-core/blob/main/intrinsic_apis/intrinsic/skills/proto/skill_manifest.proto#L189-L229
- **[R26]** Python Skill 接口：https://github.com/intrinsic-ai/intrinsic-core/blob/main/intrinsic_sdk/intrinsic/skills/python/skill_interface.py#L138-L185；execute/cancellation/footprint：https://github.com/intrinsic-ai/intrinsic-core/blob/main/intrinsic_sdk/intrinsic/skills/python/skill_interface.py#L246-L300
- **[R27]** `py_skill` 构建规则：https://github.com/intrinsic-ai/intrinsic-core/blob/main/intrinsic/skills/build_defs/skill.bzl#L305-L406；bundle 构建：https://github.com/intrinsic-ai/intrinsic-core/blob/main/intrinsic/skills/build_defs/skill.bzl#L513-L584；软件 Skill 教程：https://github.com/intrinsic-ai/intrinsic-core/blob/main/developer_resources/learn/tutorials/custom_asset_creation_software.md#L32-L116
- **[R28]** Core asset/instance schema：https://github.com/intrinsic-ai/intrinsic-core/blob/main/intrinsic_sdk/intrinsic/assets/build_defs/asset.proto#L39-L104；Asset v1 来源类型：https://github.com/intrinsic-ai/intrinsic-core/blob/main/intrinsic_sdk/intrinsic/assets/proto/v1/asset.proto#L24-L30
- **[R29]** `intrinsic_asset_instance` 构建规则：https://github.com/intrinsic-ai/intrinsic-core/blob/main/intrinsic_sdk/intrinsic/assets/build_defs/asset.bzl#L127-L201
- **[R30]** Core `swap_assets.md`（Skill 在 Process 中调用）：https://github.com/intrinsic-ai/intrinsic-core/blob/main/developer_resources/learn/tutorials/swap_assets.md#L9-L65；资产 glossary：https://github.com/intrinsic-ai/intrinsic-core/blob/main/developer_resources/learn/glossary/intrinsic_terms.md#L58-L94
- **[R31]** OMTS imported asset rule：https://github.com/intrinsic-ai/intrinsic-omts/blob/main/bazel/imported_asset.bzl#L1-L84
- **[R32]** OMTS BUILD 导入 Hand-E 与 Core Skills：https://github.com/intrinsic-ai/intrinsic-omts/blob/main/BUILD#L32-L50；https://github.com/intrinsic-ai/intrinsic-omts/blob/main/BUILD#L268-L300
- **[R33]** Core 硬件资产创建教程：https://github.com/intrinsic-ai/intrinsic-core/blob/main/developer_resources/learn/tutorials/custom_asset_creation_hardware.md#L134-L174
- **[R34]** ICON MainLoop：https://github.com/intrinsic-ai/intrinsic-core/blob/main/intrinsic_control/intrinsic/icon/server/main_loop.cc#L99-L140；创建主循环：https://github.com/intrinsic-ai/intrinsic-core/blob/main/intrinsic_control/intrinsic/icon/server/create_icon_main_loop.cc
- **[R35]** HAL manager 与配置接口：https://github.com/intrinsic-ai/intrinsic-core/tree/main/intrinsic_control/intrinsic/icon/hal
- **[R36]** Core UR module：https://github.com/intrinsic-ai/intrinsic-core/blob/main/intrinsic_control/intrinsic/icon/hardware_modules/universal_robots/ur_module.cc；RTDE driver：https://github.com/intrinsic-ai/intrinsic-core/blob/main/intrinsic_control/intrinsic/icon/hardware_modules/universal_robots/rtde_ur_driver.cc；KUKA RSI module：https://github.com/intrinsic-ai/intrinsic-core/blob/main/intrinsic_control/intrinsic/icon/hardware_modules/kuka_rsi/kuka_rsi_hardware_module.cc
- **[R37]** OMTS configs 目录说明：https://github.com/intrinsic-ai/intrinsic-omts/blob/main/configs/README.md#L1-L32；OMTS ICON 配置：https://github.com/intrinsic-ai/intrinsic-omts/blob/main/configs/omts/icon_config.textproto#L189-L193；UR 配置：https://github.com/intrinsic-ai/intrinsic-omts/blob/main/configs/omts/ur_module_config.textproto#L1-L5
- **[R38]** OMTS BUILD cell setup：https://github.com/intrinsic-ai/intrinsic-omts/blob/main/BUILD#L9-L30；KR10 stub 说明：https://github.com/intrinsic-ai/intrinsic-omts/blob/main/configs/README.md#L10-L13
- **[R39]** OMTS robot SBL adapter：https://github.com/intrinsic-ai/intrinsic-omts/blob/main/src/hardware/robot.py#L45-L72；技能任务构造：https://github.com/intrinsic-ai/intrinsic-omts/blob/main/src/hardware/robot.py#L97-L126
- **[R40]** Core k3s setup：https://github.com/intrinsic-ai/intrinsic-core/blob/main/intrinsic_runtime/setup_k3s.sh；realtime setup：https://github.com/intrinsic-ai/intrinsic-core/blob/main/intrinsic_runtime/setup_realtime.sh
- **[R41]** KUKA RSI bring-up 教程：https://github.com/intrinsic-ai/intrinsic-core/blob/main/developer_resources/learn/tutorials/interactive_tutorials/icon/robot_bringup.md#L217-L265
- **[R42]** Core behavior tree proto：https://github.com/intrinsic-ai/intrinsic-core/blob/main/intrinsic/executive/proto/behavior_tree.proto；loop 规则：https://github.com/intrinsic-ai/intrinsic-core/blob/main/intrinsic/executive/clips/behavior_tree/loop_node.clp#L15-L24；OMTS 流程树：[R16]
- **[R43]** OMTS Cartesian/运动任务构造：https://github.com/intrinsic-ai/intrinsic-omts/blob/main/src/hardware/robot.py#L317-L350；接触动作：https://github.com/intrinsic-ai/intrinsic-omts/blob/main/src/hardware/robot.py#L399-L421
- **[R44]** Core 轨迹碰撞验证：https://github.com/intrinsic-ai/intrinsic-core/blob/main/intrinsic_motion_planning/intrinsic/motion_planning/trajectory_planning/validation.cc#L40-L109
- **[R45]** Core cell customization/ObjectWorld 教程：https://github.com/intrinsic-ai/intrinsic-core/blob/main/developer_resources/learn/tutorials/cell_customization.md#L41-L58；运行中更新示例：https://github.com/intrinsic-ai/intrinsic-core/blob/main/developer_resources/learn/tutorials/cell_customization.md#L113-L121
- **[R46]** OMTS world update tool：https://github.com/intrinsic-ai/intrinsic-omts/blob/main/tools/world/apply_scene_updates.py#L29-L78
- **[R47]** OMTS attach/detach 任务：https://github.com/intrinsic-ai/intrinsic-omts/blob/main/src/hardware/robot.py#L423-L460
- **[R48]** Core multi-view pose skill（结果处理）：https://github.com/intrinsic-ai/intrinsic-core/blob/e87d7c6c60b7f9ad588ec2dbba4ad45016d9f82f/intrinsic_perception/intrinsic/perception/skills/multi_view/estimate_pose_multi_view.py#L53-L69；服务调用与过滤：https://github.com/intrinsic-ai/intrinsic-core/blob/e87d7c6c60b7f9ad588ec2dbba4ad45016d9f82f/intrinsic_perception/intrinsic/perception/skills/multi_view/estimate_pose_multi_view.py#L167-L243
- **[R49]** OMTS Vision adapter：https://github.com/intrinsic-ai/intrinsic-omts/blob/8b3887c87a94aaf721fd0c7dff5fee480f1d6d1e/src/hardware/vision.py#L171-L185
- **[R50]** OMTS pick 流程：https://github.com/intrinsic-ai/intrinsic-omts/blob/8b3887c87a94aaf721fd0c7dff5fee480f1d6d1e/src/behaviors/pick.py#L111-L140；动态 frame 变换：https://github.com/intrinsic-ai/intrinsic-omts/blob/8b3887c87a94aaf721fd0c7dff5fee480f1d6d1e/src/utils/dynamic_frame_calculator.py#L23-L77
- **[R51]** OMTS FoundationPose MLModel BUILD：https://github.com/intrinsic-ai/intrinsic-omts/blob/8b3887c87a94aaf721fd0c7dff5fee480f1d6d1e/src/foundationpose/BUILD#L15-L51
- **[R52]** FoundationPose NVIDIA/NGC 依赖：https://github.com/intrinsic-ai/intrinsic-omts/blob/8b3887c87a94aaf721fd0c7dff5fee480f1d6d1e/third_party/foundationpose/deps.bzl#L15-L47
- **[R53]** OMTS FoundationPose 实现边界说明：https://github.com/intrinsic-ai/intrinsic-omts/blob/8b3887c87a94aaf721fd0c7dff5fee480f1d6d1e/src/foundationpose/README.md#L1-L29
- **[R54]** OMTS pose estimator 注册工具：https://github.com/intrinsic-ai/intrinsic-omts/blob/8b3887c87a94aaf721fd0c7dff5fee480f1d6d1e/tools/pose_estimation/register_using_train_service.py#L296-L369；getting-started GPU 条件：https://github.com/intrinsic-ai/intrinsic-core/blob/main/developer_resources/learn/tutorials/getting_started.md#L89-L103
- **[R55]** Core README 的 AI/perception 宣称：https://github.com/intrinsic-ai/intrinsic-core/blob/main/README.md#L9-L28；OMTS README 感知示例：https://github.com/intrinsic-ai/intrinsic-omts/blob/main/README.md#L60-L70
- **[R56]** OMTS segmentation release/说明线索：https://github.com/intrinsic-ai/intrinsic-omts/blob/main/README.md#L128-L180
- **[R57]** Core ROS 兼容性主张：https://github.com/intrinsic-ai/intrinsic-core/blob/main/README.md
- **[R58]** Core ros2_control/ICON bridge 教程：https://github.com/intrinsic-ai/intrinsic-core/blob/main/developer_resources/learn/tutorials/interactive_tutorials/icon/ros2_bridge_tutorial.md#L10-L115；构建与部署步骤：https://github.com/intrinsic-ai/intrinsic-core/blob/main/developer_resources/learn/tutorials/interactive_tutorials/icon/ros2_bridge_tutorial.md#L116-L219
- **[R59]** ROS 几何 proto conversion：https://github.com/intrinsic-ai/intrinsic-core/blob/main/intrinsic_sdk/intrinsic/math/python/ros_proto_conversion.py#L20-L127
- **[R60]** ROS 相机源接口与编码处理：https://github.com/intrinsic-ai/intrinsic-core/blob/main/intrinsic_perception/intrinsic/perception/cameras/ros_image_source.h#L66-L145；https://github.com/intrinsic-ai/intrinsic-core/blob/main/intrinsic_perception/intrinsic/perception/cameras/ros_image_source.cc#L445-L555
- **[R61]** OMTS bridge manifest：https://github.com/intrinsic-ai/intrinsic-omts/blob/main/configs/common/flowstate_ros_bridge_manifest.textproto#L1-L21；外部预构建 bundle pin：https://github.com/intrinsic-ai/intrinsic-omts/blob/main/MODULE.bazel#L153-L164
- **[R62]** Core RViz/Zenoh 本机教程：https://github.com/intrinsic-ai/intrinsic-core/blob/main/developer_resources/learn/tutorials/visualize_the_robot.md#L29-L58
- **[R63]** OMTS cell 配置索引：[R37]；OMTS typed app config：https://github.com/intrinsic-ai/intrinsic-omts/blob/main/src/core/config.py#L175-L270
- **[R64]** OMTS Solution asset/instance BUILD 配置：[R38]；架构职责划分：[R14]
- **[R65]** OMTS 当前可选 setup 与硬件配置：[R38]；外部 SDK-ROS bridge 仓库（未在本报告审计实现）：https://github.com/intrinsic-ai/sdk-ros/tree/main/flowstate_ros_bridge
- **[R66]** OMTS RobotConfig/AppConfig：https://github.com/intrinsic-ai/intrinsic-omts/blob/main/src/core/config.py#L28-L42；https://github.com/intrinsic-ai/intrinsic-omts/blob/main/src/core/config.py#L247-L269
- **[R67]** OMTS 单臂 pick 子树：https://github.com/intrinsic-ai/intrinsic-omts/blob/main/src/behaviors/pick.py#L32-L58
- **[R68]** OMTS 单 robot/gripper machine tending tree：https://github.com/intrinsic-ai/intrinsic-omts/blob/main/src/behaviors/machine_tending_bt.py#L33-L42；https://github.com/intrinsic-ai/intrinsic-omts/blob/main/src/behaviors/machine_tending_bt.py#L79-L120；Lab BB-01 单臂配置：https://github.com/intrinsic-ai/intrinsic-omts/blob/main/configs/lab_bb_01/app_config.yaml#L1-L13
- **[R69]** Core `swap_assets.md` 多机器人/设备表述：https://github.com/intrinsic-ai/intrinsic-core/blob/main/developer_resources/learn/tutorials/swap_assets.md#L68-L78
- **[R70]** Core hardware asset creation tutorial：[R33]
- **[R71]** Intrinsic Flowstate 产品页：https://www.intrinsic.ai/flowstate
- **[R72]** OMTS 测试说明与 CI：[R10]
