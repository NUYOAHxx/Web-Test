# Intrinsic Core 引入评估：具身智能机器人研发公司的决策讨论稿

**资料核验截至：2026-10-02**。本文把已在 Intrinsic/Google、ROS、MoveIt、NVIDIA 官方页面或相应源码中核验的内容标为**事实**；适用性、成本、建议门槛和决策规则标为**判断/建议**。除非特别注明，网页和仓库链接是当前可见版本；`main`、`rolling` 等滚动分支可能后续变化，试点必须锁定具体 tag/commit 与二进制哈希。

## 建议结论

**判断：值得做有退出路径的小规模技术试点；当前证据不足以支持直接把 Intrinsic Core 作为自研机器人量产控制栈或全公司统一平台。** 试点应先回答三个门槛问题：一是公司的本体、控制器、驱动和传感器能否进入 Core 的实际 ROS 2/ICON 接口范围；二是目标软件版本与 Ubuntu/ROS 2 发行版能否在同一环境可重复构建和运行；三是运行时、数据流、安全边界和退出安排能否满足公司的要求。

**事实：** Intrinsic 于 2026-09-22 公布 Intrinsic Core，并称其为 Apache 2.0 开源、可本地运行、与 ROS 兼容的机器人能力集合；官方仓库列出 runtime、ICON 控制、运动规划、感知、推理、SDK、硬件及运动学组件。参见 [Intrinsic 公告](https://www.intrinsic.ai/blog/posts/introducing-intrinsic-core)、[Intrinsic Core 产品页](https://www.intrinsic.ai/intrinsic-core) 和 [intrinsic-core README](https://github.com/intrinsic-ai/intrinsic-core/blob/main/README.md)。

**事实：** 截至核验日，Core Releases 页面可见的版本只有 `20260921.0` 和 `20260922.0`，最新版本于 9 月 22 日发布。[GitHub Releases](https://github.com/intrinsic-ai/intrinsic-core/releases) 因此这里的判断对象是刚公开的早期代码与参考方案，而不是已经由公开资料证明具有多年维护记录、生产 SLA 或广泛客户验证的成熟产品。

**事实边界：** 仓库虽位于 `intrinsic-ai` 组织并由 Intrinsic 官方页面链接，但其 README 明确写明“不是 Google 官方支持产品”。公开来源不能据此推断 Google 对项目提供支持、担保或服务等级承诺。[Core README](https://github.com/intrinsic-ai/intrinsic-core/blob/main/README.md)

## 产品边界：它解决什么，不应假设什么

**事实：** Core 被描述为本地 runtime、SDK 与机器人能力的组合；安装指南采用 k3s 容器环境。官方能力清单包括 Gazebo 仿真、数字孪生、感知/位姿估计、运动与抓取规划、控制及相机标定。公开参考方案 Open Machine Tending Solution（OMTS）围绕 CNC 上下料，仓库的默认单元配置是 UR5e；另有 UR3e 实验室配置。[Intrinsic Core 产品页](https://www.intrinsic.ai/intrinsic-core)、[Core README](https://github.com/intrinsic-ai/intrinsic-core/blob/main/README.md)、[OMTS 仓库](https://github.com/intrinsic-ai/intrinsic-omts)

**判断：** 对已有 ROS 2/Linux 工程团队、应用接近工业机械臂的机床上下料、感知抓取、规划执行，并愿意按其运行时和资产模型组织软件的团队，Core 值得评估。若公司核心差异化在自研人形/移动双臂本体、全身运动、非标准执行器、力控/触觉闭环或专有总线，公开案例并未证明这些场景已被 Core 覆盖；这不是“肯定不支持”，而是必须逐型号、逐控制接口实测后才能认定。

**不要把以下几种东西混为一谈：**

- **Intrinsic Core：** 开源代码、预配置本地运行时和能力/接口组件。它减少从零拼装工作的可能性，但公开事实不等于适配本公司的硬件。
- **Flowstate：** Intrinsic 单独展示的开发/部署平台，提供图形化工作单元、技能和行为树界面，并展示仿真与真实工作单元切换。[Flowstate 产品页](https://www.intrinsic.ai/flowstate) 这与 Core 的开源许可、安装路径和平台合同不是同一事项。
- **Intrinsic Platform/IntrinsicOS 服务：** 若采购或使用这些商业服务，平台条款、账户、数据处理、费用和服务停止条款可能适用；不能把这些条款机械套用到仅使用 Apache-2.0 Core 的纯本地部署，也不能假设 Core-only 部署天然不受任何服务依赖影响。[Intrinsic Platform Terms](https://intrinsic.ai/platform-terms/)

## 技术兼容性与边界

| 领域 | 可验证事实 | 对本公司的含义 / 需验证事项 |
|---|---|---|
| **ROS 2** | Core README 列 ROS 2 Lyrical Luth；Core Getting Started 指南要求 Ubuntu 26.04。README 的操作系统说明同时出现 Ubuntu 24.04/26.04 和括号中的 Ubuntu 22.04，口径与 Getting Started 不一致。[README](https://github.com/intrinsic-ai/intrinsic-core/blob/main/README.md) [Getting Started](https://github.com/intrinsic-ai/intrinsic-core/blob/main/developer_resources/learn/tutorials/getting_started.md) | **判断：试点前必须向供应商确认“Core tag × Ubuntu × ROS 2”正式支持矩阵。** 不要在同一产品计划中把 Lyrical、Jazzy、Kilted 的独立仓库说明拼成一个已验证组合。对 ROS 1 也没有足够资料证明原生支持；未证实不等于确定不支持。 |
| **MoveIt 2** | Intrinsic 的 `intrinsic-moveit` 仓库提供 ROS 2 规划服务和 motion/grasp planning client skills；需要 `flowstate_ros_bridge` 同步 `/tf`、`/joint_states`，Core 连接示例使用 Zenoh。项目列出的主要硬件配置是 UR5e、Robotiq Hand-E 和 Orbbec 相机；README 称仍在积极开发。[intrinsic-moveit README](https://github.com/intrinsic-ai/intrinsic-moveit) | **判断：这是目前公开资料中最实在的规划集成路径，但属于明确的 ROS 2 服务/桥接和机器人配置，不代表 Core 内置所有 MoveIt 功能或任意本体可直接接入。** OMPL、STOMP、Pilz 出现在该仓库特定配置包中，不应说成所有 Core 应用的默认规划器。 |
| **ros2_control / 控制器** | Intrinsic 的 `icon-hwm-controller` 是 `ros2_control` 到 ICON 硬件模块的开源桥接。该项目 README 明确以 ROS 2 Kilted/Ubuntu 24.04 为目标，目前限于**位置控制机械臂**，不支持数字或模拟 I/O；列有 Universal Robots 和 FANUC 示例。它还声明不是 Google 官方支持产品。[ICON bridge README](https://github.com/intrinsic-ai/icon-hwm-controller) | **判断：不能把“有 ros2_control bridge”简化成支持所有控制器、速度/力矩模式、I/O、力传感器或自研驱动。** 若产品需要力控、关节力矩命令、触觉闭环、复合机器人或现场总线，需拿到逐项适配证据；不符合接口时要计入自研桥接、验证和维护成本。 |
| **ROS 控制栈分工** | ROS 2 `ros2_control` 官方文档把 Controller Manager、硬件抽象/Resource Manager 和 controllers 分开，控制循环读取硬件状态、更新活动控制器并写回硬件；MoveIt 文档说明其通常通过 `JointTrajectoryController` 等接口发送规划轨迹，但底层 controller 可以是另行实现的 ROS action。[ros2_control Kilted 架构](https://control.ros.org/kilted/doc/getting_started/getting_started.html) [MoveIt 控制器配置](https://moveit.picknik.ai/main/doc/examples/controller_configuration/controller_configuration_tutorial.html) | **判断：规划、任务编排、控制器和安全链是不同层。** 即便 MoveIt 规划成功，也不自动保证底层驱动可执行、满足周期时间或安全要求；不要让高级任务编排层承担应由专用控制环或机器人安全控制器承担的职责。 |
| **仿真与数字孪生** | Core 官方资料列 Gazebo 仿真和数字孪生；教程区分软件内部的“belief world”与 Gazebo 的“simulation world”，并说明状态可能不一致。OMTS 可先以 `operation_mode=sim` 运行。[Core 仿真教程](https://github.com/intrinsic-ai/intrinsic-core/tree/main/developer_resources/learn/tutorials) [Getting Started](https://github.com/intrinsic-ai/intrinsic-core/blob/main/developer_resources/learn/tutorials/getting_started.md) | **判断：可用于应用逻辑、状态机、场景和部分感知/规划调试；不能由“数字孪生”推断碰撞、柔顺性、接触、夹持力、力/触觉或相机噪声与实机等价。** Sim-to-real 必须作为单独验证阶段。 |
| **实机与硬件** | 入门指南称实时控制机器人硬件需 Intel CPU、x86-64，模拟可用 AMD，ARM 当前不支持；视觉/ML 工作负载需专用 NVIDIA GPU。建议配置含 2–3 个千兆网口。KUKA 替换资产教程要求改硬件配置，示例中控制频率从 UR 的 500 Hz 改为 KR10 的 250 Hz，且实机 IP、I/O 等需按机器人配置。[Getting Started](https://github.com/intrinsic-ai/intrinsic-core/blob/main/developer_resources/learn/tutorials/getting_started.md) [Swap Assets 教程](https://github.com/intrinsic-ai/intrinsic-core/blob/main/developer_resources/learn/tutorials/swap_assets.md) | **判断：所谓硬件无关应理解为有抽象层和可适配意图，不是即插即用兼容承诺。** 向供应商提交精确本体、控制柜/固件、驱动、末端执行器、传感器和网络配置，要求指出哪部分有可运行样例、哪部分需自行开发。 |
| **NVIDIA / 其他计算栈** | NVIDIA Isaac ROS 提供运行在 ROS 2 上的软件包；cuMotion 通过 MoveIt 2 插件提供串联机械臂规划，要求 URDF 和 XRDF 等机器人描述。[NVIDIA Isaac ROS Manipulation](https://nvidia-isaac-ros.github.io/concepts/manipulation/index.html) | **判断：Isaac ROS/cuMotion 是可与 ROS 2/MoveIt 组合的加速与感知/规划能力层，不是 Core 的同类完整运行时，也不是独立的安全控制系统。** 若公司已经大量依赖 NVIDIA GPU、CUDA 感知或 cuMotion，应比较“保留 ROS 2 主干、择取能力”与“引入 Core runtime/资产模型”两种增量成本。 |

**时间与实时性边界：** Core README 使用“real-time”“deterministic”等描述，但所查资料没有给出针对本公司硬件的最坏情况时延、抖动、控制周期和负载报告，也没有证据证明任何相关产品取得了适用于本项目的功能安全认证。ROS 2 官方实时编程指南也强调期限、调度和运行时工程条件；实时不是一个仅凭 ROS 接口或产品命名即可推定的性质。[Core README](https://github.com/intrinsic-ai/intrinsic-core/blob/main/README.md) [ROS 2 Real-Time Programming 源码文档](https://github.com/ros2/ros2_documentation/blob/rolling/source/Tutorials/Demos/Real-Time-Programming.rst)

## 集成代价与团队能力

**事实：** 官方 Quickstart 并非“安装一个 ROS package 即完成”：流程包含 Ubuntu 主机、k3s、Core 预构建包、`inctl`、Bazel/Bazelisk、OMTS 构建与部署；视觉服务还要配置 k3s 的 NVIDIA GPU 支持。[Getting Started](https://github.com/intrinsic-ai/intrinsic-core/blob/main/developer_resources/learn/tutorials/getting_started.md) 其 MoveIt 和 ICON 适配又要求 ROS 发行版、Zenoh/桥接、关节命名、URDF/几何、机器人 IP 和硬件配置吻合。[intrinsic-moveit](https://github.com/intrinsic-ai/intrinsic-moveit) [icon-hwm-controller](https://github.com/intrinsic-ai/icon-hwm-controller)

**判断：集成成本不应按开源许可费用估算。** 项目预算至少拆成以下工作包，再按本公司设备和基线评估人周；当前缺少本体、控制器、ROS 版本、现有驱动和团队情况，不给出伪精确工期或价格：

1. **主机与构建环境：** OS/CPU/GPU/网卡、k3s、容器镜像、依赖获取、固定版本与复现构建。
2. **ROS 互操作：** 发行版对齐、Zenoh/RMW、消息定义、TF/关节状态、服务/动作接口、QoS 与命名空间。
3. **硬件适配：** `ros2_control` 硬件组件或 ICON HWM、指令模式、反馈数据、I/O、夹爪、传感器、标定和故障恢复。
4. **应用移植：** 技能/服务封装、任务编排、行为树或状态机、异常路径及现有系统之间的接口。
5. **验证与安全：** 仿真模型误差、硬件在环、控制时延、故障注入、风险评估、安全 PLC/急停链和法规适用性审查。
6. **运维与退出：** 镜像/模型/SBOM、更新批准、备份、审计、网络隔离、日志、补丁和回滚演练。

**判断：相对工作量可先作条件化估算。** 若任务接近官方 UR5e/UR3e OMTS 且只是试跑参考应用，工程工作量相对低；若是已有 ROS 2 和 MoveIt、但型号/发行版/驱动需对接，属中等到较高；若本体是非位置控制、依赖模拟/数字 I/O、力矩/触觉、专有总线或当前未支持操作系统，则适配和验证往往成为主成本，项目应在“Core 的集成便利”与“继续自研可控”之间先做小实验。以上等级是本报告判断，不是供应商报价。

建议试点团队至少具备 ROS 2/Linux 与 C++/Bazel/容器构建能力；机器人驱动、实时系统和控制工程能力；MoveIt/URDF/标定及仿真调试能力；网络安全/系统运维；以及具备独立权限的功能安全/风险评估责任人。若团队缺少其中的硬件接口、实时或安全负责人，不宜直接进入实机阶段。

## 锁定、运维与安全风险

**代码许可与技术锁定。事实：** Core 仓库为 Apache 2.0。[LICENSE](https://github.com/intrinsic-ai/intrinsic-core/blob/main/LICENSE) 这有利于检查、修改和维护代码，但不自动覆盖第三方模型/驱动/资产、商标、商业服务、培训或支持。

**判断：** 代码层锁定可通过 fork、保留补丁和使用标准 ROS 2 消息来降低；平台层依赖可能更强。SDK、ICON/HAL、资产/技能模型、运行时打包、工具链和服务配置若进入核心产品，迁出时仍需替代运行时、重写适配与复核控制/安全行为。应把应用逻辑、机器人驱动、模型/配置、工艺数据与运行时做可替换分层。

**合同与退出风险。事实：** Intrinsic Platform Terms 针对其商业 Services 规定，服务可修改或停止、访问不保证不中断；组织管理员账户暂停/终止时，IntrinsicOS 和 Solutions 可能停止工作，即使部署在客户或终端客户控制的计算机上，内容也可能不可恢复删除。条款还限制 Platform Resources 的服务外使用，并规定费用/订阅在 Order 中确认。[Intrinsic Platform Terms](https://intrinsic.ai/platform-terms/) **范围提醒：这些条款针对平台服务，不能不加区分地套用到仅用开源 Core 的部署；相反，若方案依赖 Flowstate/IntrinsicOS，必须书面确认对应合同和离线连续性。**

**数据与遥测。事实：** Intrinsic 隐私政策称其产品/服务可能自动收集配置、活动日志、升级、技能或资产增删及机器人部署等使用信息，并使用云托管、安全、分析等服务提供商处理部分数据。[Intrinsic Privacy Policy](https://www.intrinsic.ai/legal/privacy) 政策并未在所核验内容中逐项说明这些收集如何适用于 Core-only 部署、是否默认启用或如何关闭；因此**不能仅凭“本地 runtime”批准气隙运行或敏感图像/点云不出域**。

**安全责任。事实：** 官方技术规格针对 Flowstate/相关服务明确说平台不替代机器人控制器的安全系统；客户负责工作单元风险评估、机器人安全配置，以及现场 IPC、网络和补丁等安全管理。[Intrinsic Technical Specifications](https://www.intrinsic.ai/legal/technical-specifications) 这不是 Core 获得安全认证的证据。Core `SECURITY.md` 提供通过邮件私下报漏洞的流程，但未给确认/修复时限或长期支持承诺。[Core SECURITY.md](https://github.com/intrinsic-ai/intrinsic-core/blob/main/SECURITY.md)

**运维建议（判断）：** 生产候选方案要做到版本锁定、可复现构建、签名/哈希校验、SBOM 与依赖许可清单、漏洞通知与补丁时限、备份/回滚、出站网络白名单、数据包捕获、账户与远程支持审计。不能在供应商解释其端口、云依赖和数据流前让 Core 获得无边界的生产网络或机器人命令权限。

## 方案选择矩阵

下表比较的是**决策路线**，不是基于缺失的本公司需求算出的分数。

| 方案 | 适合条件 | 主要收益 | 主要代价 / 风险 | 本次判断 |
|---|---|---|---|---|
| **Intrinsic Core** | 单元任务接近工业机械臂工作站；目标 Ubuntu/ROS 2/CPU/GPU 与已验证组合匹配；团队接受 ICON/资产/运行时模型 | 本地预配置运行时和参考应用，整合仿真、规划、感知等能力；Apache 2.0 开源 | 项目刚公开；版本矩阵和机器人支持需确认；新运行时/桥接要运维；代码许可不等于支持 SLA | 作为**限域试点候选**，暂不直接替代生产控制体系 |
| **自研 ROS 2 + ros2_control + MoveIt** | 公司已有强 ROS 2、驱动和控制团队；自研硬件/接口是产品核心；需完全掌握软件演进 | 组件与接口透明，能围绕本体需求自定义；避免把应用结构绑定到单一机器人平台 | 需自行整合生命周期、部署、感知、仿真、工具链、安全与持续维护；ROS 本身不是交付好的应用平台 | 若硬件差异化和长期控制权优先，通常是基线方案 |
| **ROS 2 主干 + 按需选组件（MoveIt、Isaac ROS/cuMotion 等）** | 已有机器人软件主干，只缺规划加速或特定感知能力；NVIDIA 硬件栈已可用 | 能局部增加能力，同时保持控制和应用编排由本公司持有 | 版本、GPU、URDF/XRDF、ROS 发行版、驱动及消息路径需逐项集成；组件不自动提供全栈部署或安全 | 适合以能力为单位做 A/B 技术评估 |
| **Flowstate/商业平台** | 需要图形化工作单元/流程工具、供应商服务与商业支持，并能接受其平台合同 | 官方页面展示图形化行为树、数字孪生、技能目录和仿真到实机的工作流程。[Flowstate](https://www.intrinsic.ai/flowstate) | 商业费用/订单、账户、内容、数据处理、服务连续性和迁出条款必须谈妥；不要与 Apache Core 混作一项 | 单独进行商务、安全和法务评估，不从 Core 试用推导平台条款已接受 |
| **既有状态机/行为树或自研任务编排层** | 公司已有可靠的状态管理、异常恢复与操作工具；只希望引入规划/感知/控制能力 | 能继续掌握任务逻辑与故障恢复；减少迁移整套应用编排平台的范围 | 本公司承担集成和工具维护；与所选执行引擎、ROS action/service、仿真/日志需打通 | 可作为 Core/ROS 组件之上的边界，避免把任务策略绑进底层控制环 |

**选型顺序建议（判断）：** 先用安全、实时、支持硬件、部署隔离和数据出域作硬门槛，而不是给所有方案做平均分。如果 Core 在目标硬件上未通过门槛，即使演示效果好也不应作为机器人控制栈；若只缺少一项能力，优先把 MoveIt、cuMotion、视觉或仿真做成可替换模块；只有当团队确实希望采用整套 Core 资产/runtime 模型，且试点证明节约的整合成本超过额外运维和退出成本时，再考虑扩大范围。

## 可回滚试点和逐步部署

这是建议方案，不是 Intrinsic 官方交付承诺。**试点范围建议只选一台非生产机械臂、一个任务、一个固定版本和一个隔离工作单元。** 保留当前驱动/控制栈作为唯一生产权威，Core 不可未经审查直接向生产机器人发指令。

| 阶段 | 做什么 | 晋级门槛 |
|---|---|---|
| **0. 资格与边界确认** | 冻结目标本体/控制器/固件/驱动/传感器清单；让供应商填完整 Core×OS×ROS 2 矩阵；确认 Apache Core、Flowstate、IntrinsicOS 的合同边界、数据流和安全责任。 | 版本矩阵和硬件接口无歧义；法务/安全/网络负责人批准试验环境；未确认不得进入实机控制。 |
| **1. 固定版本仿真** | 固定 release tag/commit 和容器/资产哈希，按 Getting Started 部署 Core 和 OMTS；保存构建日志、SBOM、配置与网络记录。 | 干净主机可重复部署；核心流程仿真可重复；所有依赖和出站流量可解释。 |
| **2. 自研机器人适配（影子/模拟）** | 用 mock hardware 或只读状态桥接验证 URDF/关节名、TF、状态与服务/action；命令输出先记录不下发给实体机器人。 | 接口映射完整，故障和丢消息有检测；未支持的控制模式和 I/O 有明确替代设计。 |
| **3. 隔离实机验证** | 仅在封闭测试单元，保留机器人厂商安全系统、急停和安全 PLC；在低速、空载或低风险条件下由具备资质人员监督。验证断网、急停、驱动掉线和恢复。 | 全部安全链独立验证通过；实测控制/通信指标满足本公司 deadline；任何非预期运动立即退回阶段 2。 |
| **4. 有限应用试运行** | 单任务、单班次或限定运行窗口；观察成功率、故障恢复、资源和运维工作量。禁止未经单独审批扩至产线或多个产品线。 | 验收指标达到预先冻结目标；漏洞、升级、监控、备份和退出流程负责人明确。 |
| **5. 量产决策** | 重新进行供应商支持、法律、网络安全、功能安全、生产质量与全生命周期评审。 | 管理层独立批准后再定规模；试点通过本身不构成量产批准。 |

### 回滚设计

- 所有 Core 镜像、配置、技能、模型、机器人描述和桥接代码用版本控制并保存可导出的副本；与现有控制软件、固件和参数集分开归档。
- 用物理/逻辑隔离保证 Core 可从机器人命令路径断开；明确谁能切换控制权。试点开始前演练回到冻结的既有栈。
- 预置可恢复的主机镜像/容器清单、网络规则、控制器参数和机器人侧配置；回滚不得依赖云账号可用或临时供应商协助。
- 若 Core 不可用，机器人回到厂商控制器和已验证的既有任务模式；不得让两个编排器同时拥有执行权。
- 记录每次更新的版本、哈希、批准人和测试结果。无法恢复至基线就是试点前置不合格。

### 建议验收指标

下列数值仅是**供会议讨论的内部试验提案**，不是厂商保证；最终阈值应由本公司任务危险分析、现栈基线和节拍要求批准并冻结。试点样本无法替代正式安全认证或量产可靠性论证。

| 指标 | 建议验收方式 | 建议门槛 / 判断依据 |
|---|---|---|
| **版本与部署可重复性** | 新主机按操作手册从固定版本重建，核对镜像/模型哈希、SBOM、配置差异 | 2 次独立部署成功；不得依赖未记录手工步骤；所有依赖/许可可盘点 |
| **仿真功能正确性** | 预先定义工作空间、工件、遮挡和异常条件；每类至少 100 次运行 | 正常场景任务成功率可先定 ≥98%；失败可分类、可重放；场景覆盖由任务团队批准 |
| **实机任务表现** | 安全批准后与现栈在相同工件/条件做对照，记录规划、执行、夹持和恢复 | 试验提案：至少 100 次代表性循环，零安全相关偏差；成功率不低于任务 SLA，且较现栈退化不超过 1 个百分点。对随机/高风险任务由安全与统计负责人另定样本数 |
| **控制及时性** | 在目标负载和网络下测量控制环周期、p99/p99.9 延迟、抖动、deadline miss；规划时延与控制时延分别统计 | 如果该环被设计为硬实时，使用风险分析给出的硬期限并要求零越限；建议 p99.9 使用不超过期限的 80% 作为余量起点。若低层闭环仍由机器人控制器负责，则在规划/指令接口上设定对应 SLA，不能把 ROS 通信统计等同为硬实时证明 |
| **稳定性和故障恢复** | 连续运行、断网、节点崩溃、相机/驱动掉线、控制器重启、资源耗尽注入 | 试验提案：8 小时运行无崩溃和非预期运动；所有故障按预期进入可识别状态；重新接管和恢复步骤可演练 |
| **安全边界** | 由安全负责人验证急停、保护停、安全 PLC、速度/空间限制、人员防护和控制权互锁 | 安全链不依赖 Core；不允许绕过安全控制器；任何非预期运动、急停失效或碰撞防护异常即停止实机试验 |
| **可退出性** | 禁网/撤销账户的模拟场景、导出应用资产、恢复既有镜像并运行既有任务 | Core/平台停用不影响恢复基线；应用逻辑、工艺数据及客户自有资产可按合同导出/保留 |

### 立即停止条件

出现任一情形，停止实机试验并回滚到阶段 2 或冻结的既有栈：

1. 急停、安全 PLC、机器人厂商安全功能被绕过、不可用或无法独立验证；出现任何非预期运动、碰撞或意外执行。
2. 对关键控制环发生 deadline 超限，或供应商/团队无法解释时延、抖动、掉线和失效状态。
3. 目标机器人实际依赖的命令/反馈/IO/力矩/传感器接口不在确认支持范围内，却必须由未验证的旁路适配才能运行。
4. 未经批准的外传、未知端口、远程访问、遥测、账户校验或云依赖；数据/配置/模型的用途、出境、保留或删除方式无法确认。
5. 镜像/依赖不能固定或复现；许可证、SBOM、漏洞响应和更新来源不清；安全漏洞在约定窗口内无人响应。
6. 条款显示账户暂停/终止可能让现场运行中断或内容不能导出，且供应商无法提供适用于本公司的连续运行和退出书面方案。
7. 试点超出预定工时、设备或风险预算，或连续两轮仍未达到已冻结指标；先复盘是否改为选择性引入 ROS/NVIDIA 等单项能力。

## 需向供应商、公司法务和安全团队确认的问题

### 向 Intrinsic / Google 相关团队

1. **支持范围：** 当前维护的 Core release、OS 与 ROS 2 发行版矩阵是什么？README 和 Getting Started 的 Ubuntu 22/24/26 口径为何不同？Core、MoveIt、ICON bridge、相机驱动的这些发行版是否已组合测试？
2. **自研本体：** 对本公司的本体型号、固件、控制器、驱动、总线、力/扭矩、触觉、夹爪、相机和数字/模拟 I/O，哪些有已运行示例、哪些经正式支持、哪些由客户自行集成？
3. **实时与性能：** ICON 控制频率、平台/CPU/内核限定、最坏情况时延/抖动、deadline 违例处理、失联安全状态、并发负载测试和可复现基准是什么？产品文字中的“real-time/deterministic”具体保证到哪一层？
4. **安全与认证：** 哪些产品和版本取得了哪些适用的机器人/功能安全认证？哪些安全功能由 Core 提供、哪些明确不提供？风险评估、最终工作单元认证和事故责任如何分配？
5. **本地运行与网络：** Core-only 是否能在无公网、无账号校验、无周期性 check-in 的现场完整运行？运行时的所有出站连接、域名/端口、遥测字段、默认状态和关闭方法是什么？相机图像、点云、工艺数据或模型权重是否可能上传？
6. **维护与安全响应：** 当前版本支持周期、LTS/EOL/EOS、补丁签名、漏洞分级和确认/修复 SLA、SBOM、第三方模型/资产/驱动许可清单、兼容性政策和弃用政策是什么？
7. **商业服务：** Core 开源部分、预构建物、Flowstate、IntrinsicOS、支持、培训和企业服务分别怎样收费？哪些能力要求账号/订阅/订单？是否可承诺离线持续运行、迁移协助和停服通知期？
8. **客户资产与支持责任：** 仓库写明非 Google 官方支持产品。哪个法律实体拥有、维护并支持 Core？哪些部分有技术支持 SLA？客户 fork、修改、商用、向下游交付与商标使用分别受什么条款约束？

### 向公司法务、采购、网络安全和功能安全负责人

- Core-only 部署、Flowstate 与 IntrinsicOS 是否分别适用 Apache 许可、Platform Terms、Technical Specifications 或单独 Order？确认服务外使用客户作品时移除 Platform Resources 的条款会否影响拟议架构。
- 客户代码、机器人模型、现场图像/点云、工艺和遥测的所有权、处理目的、存储地域、转委托方、保留期限、删除/导出机制及合规义务是什么？是否需要数据处理协议、跨境评估或保密附加条款？
- 服务可修改/终止、账号中止可能影响部署、内容可能被删除、服务不中断不保证、责任限额等条款对生产风险是否可接受？需要何种源代码/二进制托管、持续使用权、通知期、赔偿/保险或过渡支持？
- 当前项目适用哪些机器人安全、产品责任、网络安全和出口管制要求？谁签署风险评估、谁批准进入实机、谁负责安全链和事故报告？
- 若平台后续不再维护，团队是否有权永久使用、修改和分发所需代码/二进制？第三方组件、预训练模型、固件和模型资产的许可/专利/再分发权是否逐项确认？

## 会议决策建议

**建议会议只批准阶段 0–2 的预算和资源，不一次性批准生产采用。** 指定一名机器人控制负责人、一名平台/运维负责人、一名安全责任人和一名法务/采购接口人；冻结目标硬件与 ROS/OS 版本后，再由阶段门决定是否进入实机验证。若兼容、硬实时边界、数据出域、合同退出或独立安全链有一项无法确认，结论应是暂停或仅保留仿真/局部组件试用，而不是用“开源”“ROS 兼容”或演示效果填补证据空白。
