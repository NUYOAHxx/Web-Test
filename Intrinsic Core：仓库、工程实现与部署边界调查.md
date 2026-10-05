# Intrinsic Core：仓库、工程实现与部署边界调查

**调查截止：2026-10-02。** 本报告依据官方仓库、官方文档与公告，以及题述对源码、配置和公开 GitHub 记录的检视。为避免把产品文案说成代码保证，文中分别标明“源码可见”“官方文档称”或“公开记录显示”。除明确说明的数学检查外，本次没有构建、测试或部署项目；静态代码可说明实现路径，不等于运行验证。

## 仓库是什么，以及不是什么

截至调查日，公开仓库是 [intrinsic-ai/intrinsic-core](https://github.com/intrinsic-ai/intrinsic-core)，默认分支为 `main`、未归档；仓库 API 的 `pushed_at` 为 2026-10-01，main 头提交为 [`e669699712f5a67a15b90394207d6e25b457cf21`](https://github.com/intrinsic-ai/intrinsic-core/commit/e669699712f5a67a15b90394207d6e25b457cf21)，提交时间 2026-10-01。最新可见 GitHub Release 是 [`20260922.0`](https://github.com/intrinsic-ai/intrinsic-core/releases/tag/20260922.0)，发布于 2026-09-22；此前有 `20260921.0`。提交 SHA 是源码快照标识，不是产品版本号。仓库创建于 2026-09-08，主线头部晚于最新发布标签。

它不是单纯的 SDK 或示例库：根目录直接包含 `intrinsic_apis/`、`intrinsic_control/`、`intrinsic_hardware/`、`intrinsic_inference/`、`intrinsic_kinematics/`、`intrinsic_motion_planning/`、`intrinsic_perception/`、`intrinsic_runtime/`、`intrinsic_sdk/` 等源码树，也有 `incode/`、`developer_resources/`、`third_party/` 和 Bazel 配置。Intrinsic 官方 GitHub 组织将其描述为面向工业机器人的开放本地 runtime、SDK 与硬件无关实时控制框架；README 也称其包含 runtime、控制、运动规划、感知、推理、SDK、API、硬件和运动学模块（[仓库 README](https://github.com/intrinsic-ai/intrinsic-core#readme)）。这些是项目自述，以下实现切面只核实其中部分职责。

更准确的范围判断是：**Intrinsic Core 本身的多组件源码仓库，而不是 Intrinsic 商业平台全部产品的完整源码。** Intrinsic 2026-09-22 的[官方公告](https://www.intrinsic.ai/blog/posts/introducing-intrinsic-core)把开源内容称为平台的“core parts”；README 将 OMTS、MoveIt 集成、Inference 服务等作为相关的独立仓库，官方[产品页](https://www.intrinsic.ai/intrinsic-core)也把 GitHub 下载入口与 Flowstate、云服务等企业服务分别介绍。此处是依据公开组织方式作审慎边界判断；材料没有逐项声明哪些闭源代码未放入仓库，不能据此对未公开组件作断言。README 明确写明“这不是 Google 官方支持的产品”。

## 目录与关键实现路径

仓库以 C++ 源码（`.cc`、`.h`）为主，同时有 Go、Python、Protocol Buffers（`.proto`）和 Bazel/Starlark 构建文件；例如 Go HTTP gateway 位于 `intrinsic_runtime/httpjson/`，Python 推理管理位于 `intrinsic_inference/core/`，接口定义集中在 `intrinsic_apis/` 的 proto 树。根目录还含 `MODULE.bazel`、`go.mod`、`go.sum`、Python requirements 锁文件及 `.bazelrc`。这不是对各语言行数的统计，也不表示每种语言都承担同等规模或同等关键性的职责。

**技能到运动规划，再到轨迹参数化。** `incode/motion_planning/skills/move_robot.cc` 将机器人规格和运动规格整理后，通过 `MotionPlannerClient::PlanTrajectory` 请求规划并保存返回轨迹。`intrinsic_motion_planning/intrinsic/motion_planning/motion_planner/acceleration_limited_trajectory_parameterizer.cc` 则从 ObjectWorld 获取机器人 skeleton、构造运动学 chain，配置关节/笛卡尔约束，再调用路径细化和加速度受限轨迹生成逻辑（TOPP）。这证明检视到的技能层会调用规划服务，且该轨迹参数化路径会用到世界模型和运动学；不能据此推断所有规划器实现、规划性能或安全完整性均已审计。

**控制与硬件接口。** `intrinsic_control/intrinsic/icon/control/actions/joint_jogging_action.cc` 中，`Sense()` 轮询流式命令，`Control()` 综合速度覆盖、watchdog、关节限位和轨迹生成器计算 setpoint，再经关节位置接口写出。`intrinsic_hardware/intrinsic/hardware/gpio/icon_gpio_service.cc` 实现 GPIO 双向流写会话、信号 claim，并通过 ICON 会话创建和运行 ADIO actions；相应 `intrinsic_apis/intrinsic/hardware/gpio/v1/gpio_service.proto` 定义 RPC 合约。这里的实现样本说明控制逻辑与硬件服务不是只有文档或接口声明，但并不构成对所有驱动或实时行为的验证。

**运动学、感知、推理和 SDK。** `intrinsic_kinematics/intrinsic/kinematics/ik/chain_inverse_kinematics.cc` 按配置选择 Newton–Raphson、RT 或 QP 求解路径，处理 seed/hint，并对解的维度和关节限位进行检查。`intrinsic_perception/intrinsic/perception/calibration/camera_to_robot_calibration.cc` 通过 OpenCV 标定算法和 Ceres 位姿残差优化相机到机器人变换；实现拒绝未指定标定设置或少于三组位姿对。`intrinsic_inference/core/inference_runner.py` 等待 Triton gRPC 服务 ready 后在后台 reconcile 模型，`model_controller_triton.py` 用 Triton 的模型 repository load/unload RPC 管理模型。SDK 下有 geometry、skills、world、scene、simulation 等工具；例如 `intrinsic_sdk/intrinsic/geometry/api/compute_axis_aligned_bounding_box_3d.cc` 对 Mesh 顶点或 PointCloud 计算轴对齐包围盒。这些是具体实现切片，不代表对这些模块全部职责的穷尽说明。

**Runtime 的边界。** README 将 `intrinsic_runtime` 定位为本地执行引擎；实际检视到的源码包括 Go HTTP JSON gateway、k3s/Kubernetes 部署资源，以及 Bazel 将 runtime 发布包组装为 `intrinsic-base` 的目标。`intrinsic_runtime/httpjson/gateway.go` 注册 HTTP handlers、解析 protobuf `Any` 并启动 HTTP server；可选启用 WebRTC/PubSub。仅凭这些文件不能说仓库中已逐项展开整个生产运行时实现，也不能将发布归档等同于全源码重新构建后的所有服务。

## 工具链、构建、测试与可运行路径

主构建系统是 Bazel/Bzlmod：根 `.bazelversion` 固定 Bazel 8.8.1，`.bazelrc` 启用 Bzlmod、关闭 WORKSPACE、指定 C++20 和 Linux x86_64 host platform；`MODULE.bazel` 声明 C++、Go、Python、Rust、Node/Java、CUDA 等工具链依赖，并将 `requirements.txt` 用作面向 `linux_x86_64` 的 Python 锁文件。`requirements.txt` 含精确版本与哈希；`requirements.in` 是其输入之一。故这不是用某一个系统包管理器即可覆盖的单语言 SDK，依赖还通过 Bazel 模块、Go 模块及多个 Python 锁文件管理。

仓库的制品构建 Action `.github/actions/build-intrinsic-core-artifacts/action.yml` 实际调用 Bazel 构建 `//intrinsic_runtime:intrinsic_base` 与 `//intrinsic/tools/inctl:inctl_external`，通过 `bazel cquery` 取输出并打包成 Linux amd64 的 `intrinsic-base-linux-amd64.tar` 和 `inctl-linux-amd64`。可见 post-submit/release workflow 在 Ubuntu 24.04 runner 上执行制品构建；检视到的这些 workflow 步骤没有显式运行 `bazel test`。`AGENTS.md` 建议贡献者执行 `bazel build //...` 与 `bazel test //...`，这是本地开发建议而非已观察到的 CI 测试门禁。

仓库有真实测试和多语言服务示例。`intrinsic/assets/services/examples/calcserver/BUILD` 定义 C++、Go、Python calculator 服务、OCI images、Intrinsic service/solution，以及连接运行中 Core 环境的 Python E2E 测试；该 E2E 目标带 `manual` 标签，会部署后执行三种服务实现并断言结果。另有如 `//bazel:container_structure_test_runner_test` 的 Go 单测目标及 Python/C++ 测试文件。但本次**未尝试运行** `bazel build`、`bazel test` 或示例 E2E；因此没有全量构建成功、测试通过率或兼容性结论。

未执行完整构建/部署并非一次构建失败：当前工作机是 Ubuntu 24.04，未预装 `bazel`/`bazelisk`，可用内存约 23 GiB、工作区可用磁盘约 39 GB，且没有 `/dev/nvidia0`。这与 Getting Started 明确要求 Ubuntu 26.04、建议 32 GiB RAM 和至少 100 GB 空闲空间的部署条件不一致；教程还估计首次 OMTS 构建约需 50 分钟，带感知的完整方案需 NVIDIA GPU。加上 runtime 部署脚本会安装并修改 k3s、网络入口和实时主机配置，本次将验证范围限定为源码/配置静态检查和独立数学边界检查，不把未匹配条件的完整部署当作可验证测试。该工作机说明是本次执行环境的快照，不代表所有部分构建目标都不能在 Ubuntu 24.04 运行。

官方[仓库内 Getting Started](https://github.com/intrinsic-ai/intrinsic-core/blob/main/developer_resources/learn/tutorials/getting_started.md)给出的“从干净机器开始”流程，和 CI 从源码生成 Core 制品不是一回事。指南大致要求在 Ubuntu 上准备 git-lfs、GitHub CLI 并登录，克隆与 `20260922.0` 对应的 Core 和 OMTS，运行 `intrinsic_runtime/setup_k3s.sh` 配置 k3s，然后下载已构建的 Core runtime tar 与 `inctl`，再安装 Bazelisk，并用 OMTS 仓库的 `bazel run //:omts_solution --config=lab_bb_01 -- --address localhost:17080 --operation_mode=sim` 构建部署 OMTS。指南估计首次 OMTS build 约 50 分钟，也提示空间不足时设置 `TMPDIR`。这是官方描述的可运行路线，不是本次实机复现结果。

### OS、硬件与实时条件

指南将 Ubuntu 26.04 列为 prerequisite，建议 x86-64，称 ARM 暂不支持；实时机器人硬件要求 Intel CPU，仿真可用 AMD。它建议 6 核/12 线程、32 GiB RAM（推荐 64 GiB）、1 TB NVMe 且至少 100 GB 可用空间，并建议 2–3 个千兆网口；小型任务可能可在 16 GiB 上运行，但需要控制构建并停止 runtime。这些是教程中的推荐/声明，不是经本次硬件矩阵实测的硬门槛。指南还称仿真可用集成显卡，但完整 OMTS 感知需要专用 NVIDIA RTX 3060/4060 或更高配置，并提供 `intrinsic_runtime/setup_nvidia.sh` 配置路径。

OS 支持口径存在实际歧义：README 的 prerequisites 写 Ubuntu 24.04 或 26.04，并在括号提到 Ubuntu 22.04 supported；Getting Started 则明确要求 Ubuntu 26.04。Ubuntu 24.04 CI runner 只证明发布制品任务选择了该 runner，不证明完整 runtime 在该发行版上所有功能均受支持。报告不替官方消解这一矛盾。

实机实时设置不是普通容器启动即可获得。`intrinsic_runtime/setup_realtime.sh` 会安装 Ubuntu realtime 内核包、检查 BIOS SMT/超线程状态，禁止隔离 CPU 0 并要求留 housekeeping core，修改 GRUB/TuneD 配置；可能需要重启。`setup_k3s.sh` 安装 k3s、Helm、Istio 等组件，`setup_nvidia.sh` 依赖可访问的 k3s/Kubernetes 环境并部署 NVIDIA container toolkit/device plugin。部署依赖主机、网络和特定硬件条件，不应把“本地运行”理解为跨平台或无配置成本。

## 网络、云依赖、遥测与安全

Core 部署脚本从外部来源获取组件或 chart，例如 GitHub、Google Cloud Storage、GHCR 和 Artifact Registry；Getting Started 还要求 GitHub CLI 登录以取代码及 Release 包。因此安装与更新**有网络依赖**。这不等于运行中的 Core 必然要连 Intrinsic 云端：官方产品资料称 Core 可在本地硬件运行，并可与 Flowstate、企业服务和云服务组合；源码和安装路径不足以证明所有离线行为，也不足以证明强制云连接或默认上传数据。

`intrinsic_runtime/setup_k3s.sh` 将 Istio ingress Service 配为 LoadBalancer，配置 TCP 80、17080、7447；生成 Gateway 对 80/17080 使用 HTTP、主机为 `*`，7447 为 TCP。所检视的 Gateway 生成配置没有 TLS server 定义。实际是否能从局域网或更广范围访问，还取决于 k3s ServiceLB、主机防火墙和网络拓扑；不能只凭配置断言公网可达。多数应用服务是集群内 ClusterIP，但可经 Gateway 路由。

源码中存在 metrics 和 tracing。Go telemetry 可启用 Prometheus exporter；HTTP gateway 有 9101 `/metrics` 端口，Kubernetes ServiceMonitor 示例每 10 秒抓取。追踪可走本地 OTLP gRPC collector（地址 `oc-agent.app-intrinsic-base.svc.cluster.local:4317`，代码使用 `WithInsecure()`），或另行配置 Cloud Trace exporter。Gateway tracing CLI 默认关闭、采样率默认 0；不过不同 Helm values/profile 的 `opencensus_tracing` 值有差异：Core base 示例关闭，通用 `intrinsic/kubernetes/common-values.yaml` 却设为 true。故具体部署必须看最终渲染配置，不能把一个 profile 的默认值泛化为全部安装。

脚本给出 Istio access log 格式，其中可包含远端地址、路径、User-Agent、转发地址及字节数，但同一配置把 `accessLogFile` 设为空字符串；这是日志格式存在而该处文件日志未启用的证据，不足以证明其他应用日志、metrics 或 traces 不采集。开发者网站 [privacy 页面](https://developer.intrinsic.ai/privacy) 说明的是 Developer Community 网站收集账户/使用信息并使用 Google Analytics，不能外推为本地 Core runtime 的遥测政策。已检查材料没有给出完整 runtime 数据字段、保存期限或采集数据最终导出位置；对机器人、相机和工作单元数据是否默认上传或留存，**未在已检查文件中找到足以定论的说明**。

安全配置也需区分样例和整体保证。`intrinsic_runtime/kubernetes/intrinsic_base/app_values.yaml` 示例将 `robot_authentication` 设为 false、debug service 设为 false，并注释 debug service 不可用于生产；若干 workload 模板使用非 root UID/GID、只读根文件系统、关闭自动挂载 ServiceAccount token，HTTP Gateway 还禁用 privilege escalation 并丢弃 Linux capabilities。另一方面，检视到的 Proto Registry/Builder C++ 服务使用 `grpc::InsecureServerCredentials()`，一些网关路径是 HTTP。源码片段未显示这些具体服务自身的 TLS/身份验证，但外围代理、用户网络策略和其他部署 profile 可能提供额外控制；不能从样例断言整个产品均无认证，也不能假定样例默认配置适合生产。

安装脚本对若干 k3s、Helm、k9s、Istio 下载物固定版本并校验 SHA-256，部分 Helm chart 版本固定；基础 C++/Python OCI 镜像使用 distroless。这些是可见的供应链/容器硬化措施，不等于所有依赖均已锁定、容器均签名、或仓库完成了漏洞审计。仓库 `SECURITY.md` 要求通过 `security@intrinsic.ai` 私下报告漏洞并说明协调披露流程；未发现该文件承诺安全认证或特定响应时间。

## 公开活动、限制与已发现问题

主线最新提交时间为 2026-10-01，最近 100 条提交跨 2026-09-29 至 10-01，多条提交标题是 “Internal changes from Intrinsic”。这说明快照前主线活动频繁，但项目从 9 月初公开、可见 release 只有 `20260921.0` 与 `20260922.0` 两个早期标签；短期提交频繁不能证明发布稳定性、支持周期或现场可靠性。

GitHub issues API 当时返回 10 条 issue/PR 记录。问题 #4、#5、#10 处于 open；PR #2、#6、#7、#9 仍 open，其中 #9 为 draft；#8 已 closed，PR #1、#3 已 closed。这里的状态只代表检查时的公开记录，不评价维护者响应质量或问题严重级别。

有一项数学 API 问题通过当前 main 源码实际复现。`intrinsic_sdk/intrinsic/math/python/rotation3.py` 中 `check_rotation_matrix()` 检查上 3×3 子矩阵正交性，却没有检查行列式必须为 +1；对反射矩阵 `diag(-1,1,1)`，检查通过，而 `Rotation3.from_matrix()` 返回单位旋转，输入行列式为 -1、输出为 +1。公开 PR [#6](https://github.com/intrinsic-ai/intrinsic-core/pull/6) 当时仍 open，标题即为拒绝反射矩阵。这是针对克隆源码运行的明确复现，不应外推为其他数学 API 都存在同类问题。

另一个公开问题涉及四元数 proto round-trip。静态对照 `intrinsic_sdk/intrinsic/math/python/proto_conversion.py`、`quaternion.py` 与 `math_types.py` 可见：读取端用 `32 * float64 epsilon` 检查单位范数；写入端仅在 `quat.is_normalized()` 为 false 时才归一化，而该检查默认相对容差为 `1e-5`。因此略偏离单位范数（例如约 1.000001）的值可能被写端视为已归一化而保留，随后被更严格的读取检查拒绝；对应 issue [#4](https://github.com/intrinsic-ai/intrinsic-core/issues/4) 与 PR [#7](https://github.com/intrinsic-ai/intrinsic-core/pull/7) 当时均未关闭。本次未能独立完成端到端 round-trip 复现：导入 `data_types` 时环境缺少 `intrinsic.icon.proto` 模块。故此处是静态代码路径与公开 issue/PR 相互印证，不是运行成功的独立复现。

构建方面，issue [#8](https://github.com/intrinsic-ai/intrinsic-core/issues/8) 报告在 `20260922.0` 干净 Bazel 缓存构建 OMTS 时，tinygltf 归档 SHA-256 与锁定值不符。维护者回复称 main 已修复，教程与 release 修复将随后发布；调查时 main 的 `MODULE.bazel` 已切至 `tinygltf 2.9.6.bcr.1`，但最新可见 Release 仍为 `20260922.0`。issue [#10](https://github.com/intrinsic-ai/intrinsic-core/issues/10) 于 9 月 30 日报告依教程构建 OMTS 时遇到 dependency/checksum 错误，仍 open；其错误正文不完整（依赖截图），无评论。本次不能判断 #10 与 #8 是否同根因，也不能确认所用 release 是否已解决。

综合来看，公开仓库已经有真实多组件实现、发布物、安装说明和跨语言服务示例，且持续开发；同时公开历史极短，主线领先标签，发现过基础数值 API 和依赖校验问题，OS/硬件要求也存在文档口径差异。适合把它看作**仍在快速演进、应按确切 tag/硬件/系统组合验证的开源工程基线**；现有材料不足以推断长期稳定性或生产现场表现。

## 许可证、贡献与再分发

根目录 [`LICENSE`](https://github.com/intrinsic-ai/intrinsic-core/blob/main/LICENSE) 是 Apache License 2.0。该许可证允许复制、修改、衍生、再许可和分发，文本没有排除商业用途；分发时需向接收者提供许可证、显著标记被修改文件，并保留适用版权、专利、商标及署名声明；若分发作品带有适用 NOTICE，须依条款保留相关归属信息。许可提供贡献者范围内的专利授权，但包含专利诉讼触发终止条款；不授予商标权，并有“按现状”提供、免责声明和责任限制。以上是许可证文本的事实摘要，**不是法律意见**。

不能据根 LICENSE 推定仓库所有内容和外部依赖都仅受 Apache 2.0 管辖。递归仓库树中有多处第三方许可文件，但没有根级 `NOTICE`、DCO 或统一 `THIRD_PARTY_LICENSES` 清单；检视到 `third_party/abb_egm/LICENSE` 和 `third_party/universal_robots/LICENSE` 含保留版权/条件、二进制随附声明及不得未经许可借权利人名称背书等条款；`third_party/kuka/kr10_r1100_2/LICENSE`、`third_party/fanuc/crx_10ia/LICENSE` 和若干 ROS/导入库目录有 Apache 2.0 文本。`third_party/orbbec/` 有 Orbbec 版权声明下的 Apache 2.0 `LICENSE` 和单独 `NOTICE`。

Orbbec `NOTICE` 有一处值得在再分发审查中核实：它将 nlohmann/json 标为 BSD 2-Clause，但紧随的许可证正文标题与内容是 MIT；同一文件也列出 magic_enum 的 MIT 文本。本报告只记录文件内部矛盾，不替上游裁定实际适用许可证。依赖来源分散在 `MODULE.bazel`、`go.mod`/`go.sum`、根 `requirements.in`/`requirements.txt`、`intrinsic/production/external/requirements.txt`、`intrinsic_apis/MODULE.bazel` 等；版本和哈希有助于定位依赖，但没有逐个列出许可证。未对所有直接/传递依赖、机器人模型、Git LFS 对象、发布归档、容器镜像或运行时下载物做许可证审计，因此不能把本报告当作 SBOM 或分发合规结论。

贡献政策见 [`CONTRIBUTING.md`](https://github.com/intrinsic-ai/intrinsic-core/blob/main/CONTRIBUTING.md)：提交贡献须附 Google CLA；文档称贡献者/雇主保留版权，CLA 授予项目使用和再分发贡献的许可。公开文件未要求 DCO/Signed-off-by；只能说在检视的贡献说明与公开树中未找到该要求/根级 DCO，不能排除仓库外流程。商标另受 [`TRADEMARK.md`](https://github.com/intrinsic-ai/intrinsic-core/blob/main/TRADEMARK.md) 约束：Apache 2.0 不授权 Intrinsic 名称、标识或商标；改版再分发应移除品牌资产并使用独立名称，不得暗示官方背书。对商业使用、修改和再分发的具体项目，仍需逐项检查所包含第三方代码/资产及商标使用方式。

## 主要核查入口

报告中的链接直接指向相应官方页面或仓库相对路径，以便读者对照快照与当前内容：仓库 [README](https://github.com/intrinsic-ai/intrinsic-core#readme)、[Getting Started](https://github.com/intrinsic-ai/intrinsic-core/blob/main/developer_resources/learn/tutorials/getting_started.md)、[Intrinsic 官方公告](https://www.intrinsic.ai/blog/posts/introducing-intrinsic-core)、[产品页](https://www.intrinsic.ai/intrinsic-core) 与[发布页](https://github.com/intrinsic-ai/intrinsic-core/releases)。针对源码级结论，正文已尽量将精确路径置于对应段落；链接若指向 `main`，内容可能在调查截止日之后变化，应以文中固定的提交 SHA 与相应时间点为准。
