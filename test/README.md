# 机器人整机重力矩 3D 可视化系统 (通用模型与 Mesh 版)

专为机器人动力学测试、MoveIt 2 联调与空间重力矩直观理解设计的交互式 3D 可视化系统。系统采用真实三维笛卡尔右手系（Z-Up），以物理严谨的矢量数学模型为核心，实时解算并动态渲染重心相对于支点的空间几何关系、水平力臂及重力矩分量。

---

## 一、极简架构 (单文件应用，零外部框架依赖)

整个工程高度收敛，**前端所有 HTML、CSS 样式与 JavaScript 逻辑全部合并在单个 `index.html` 文件中**，开箱即用：

```
test/
├── index.html        # 🚀【单文件 Web 应用】包含 3D 渲染、STL/DAE 加载器、物理引擎、MoveIt 2 客户端与通用模型解析
├── moveit_bridge.py  # 🤖【通用 ROS 2 桥接服务】动态订阅 /robot_description 与 /joint_states，支持 --urdf 与 Mesh 伺服
├── libs/             # 📦【核心基础库】本地优先加载，离线完全可用 (无本地库时自动无缝降级走 CDN)
│   ├── three.min.js
│   ├── OrbitControls.js
│   ├── CSS2DRenderer.js
│   ├── STLLoader.js      # STL 3D 网格加载器
│   └── ColladaLoader.js  # Collada (DAE) 3D 模型加载器
└── README.md         # 📖 本说明文档
```

---

## 二、核心功能与最新优化

1. **通用任意机器人模型加载 (URDF + 3D Mesh 网格)**：
   - **零特定型号预设**：不绑定任何特定机器人，通用支持机械臂（UR、Franka Panda、KUKA）、双足人形机器人、四足机器狗、移动底盘等任意 URDF 模型。
   - **3D 网格文件无缝加载**：支持用户选择模型文件夹（包含 `.urdf` 与 `meshes/` 目录中的 `.stl` 或 `.dae` 文件）或批量多选文件，系统自动索引并调用 `STLLoader` / `ColladaLoader` 渲染出真实的 3D 机器人外观与材质。
   - **智能去噪骨架**：若未导入 Mesh 文件，自动提取长度 $> 2\text{cm}$ 的有效运动链，过滤零位固定虚关节，彻底杜绝多关节堆叠同心圆环与假底座。

2. **智能贴地站立对齐 (Auto-Grounding)**：
   - 针对根节点位于躯干/腰部的双足或四足机器人（下肢延伸至 $Z < 0$），系统自动计算全机空间包围盒最低点 $Z_{min}$。
   - 开启【自动贴地站立】后，机器人足底无缝贴合于 $Z = 0$ 地面网格，整机重心抬升至站立高度，支点可一键快捷对齐足底地面接触面 $(0, 0, 0)$。

3. **四象限防遮挡浮标机制 (彻底解决标签重叠)**：
   - 浮动数据标签（重心 G、重力 Fg、水平力臂 d、重力矩 $\tau$、支点 O）在空间及屏幕像素维度严格隔离于不同象限：
     - 东北象限 (Q1): 🟠 重心 G (CoM)
     - 西北象限 (Q2): 🔷 水平力臂 d
     - 西南象限 (Q3): 🔴 重力 Fg
     - 东南象限 (Q4): 🟣 重力矩 $\tau$
     - 正上方: 🔵 支点 O
   - 即使位矢 $|\vec{r}| \to 0$、力臂 $d \to 0$，所有标签各安其位，绝不重合遮挡。

4. **完整三维空间力学要素**：
   - **支点 $O$**：$(O_x, O_y, O_z)$，支持面板微调或快速贴地。
   - **整机重心 $G$（CoM）**：$(G_x, G_y, G_z)$ 与整机总质量 $m$。
   - **空间位矢 $\vec{r} = G - O$** 与长度 $|\vec{r}|$。
   - **重力向量 $\vec{F}_g = -mg\hat{k}$** 与重力作用线。
   - **水平力臂 $d = \sqrt{r_x^2 + r_y^2}$**（支点到重力作用线的垂直距离）。
   - **重力矩向量 $\vec{\tau} = \vec{r} \times \vec{F}_g = (-mg\cdot r_y, mg\cdot r_x, 0)$**：恒位于水平面内（$\tau_z \equiv 0$），垂直于水平力臂，支持分解 $\tau_x$ 与 $\tau_y$。
   - **空间角度解算**：方位角 $\phi = \text{atan2}(r_y, r_x)$ 与极角 $\theta = \arccos(-r_z / |\vec{r}|)$。

5. **Pinocchio 动力学监测与多 Tab 看板**：
   - **Pinocchio 深度集成**：桥接服务与 Deeplearning 环境原生接入 Pinocchio 4.1.0，高频解算广义重力向量 $\tau = g(q)$ 与各关节静载利用率 $\eta = \frac{|\tau_i|}{\tau_{max}} \times 100\%$。
   - **多 Tab 分类看板**：
     - **Tab 1: 🌐 空间合力矩**：三维空间位矢 $\vec{r}$、重力 $F_g$、水平力臂 $d$、合力矩 $\vec{\tau}$、方位角 $\phi$ 与极角 $\theta$。
     - **Tab 2: 🦾 关节负荷谱**：展示 Pinocchio 解算的所有关节静力矩、额定力矩极限、负荷率进度条及状态标签（🟢 安全 / 🟡 警戒 / 🔴 超载），支持名称检索与按负荷率/力矩绝对值即时排序。
     - **Tab 3: ⚖️ 稳定与功耗**：基座反作用旋量（$F_z, M_x, My, |M|$）、静态维持姿态铜损能耗代理（$\sum \tau_i^2$）、有效力矩 RMS（$\tau_{rms}$）及地面投影抗翻倒稳定裕度。
   - **3D 关节负荷热力图**：支持在 3D 场景中开启【🔥 关节负荷 3D 热力】，关节枢轴根据负荷率高亮着色（安全呈现翡翠绿光、警戒呈琥珀黄、超载呈赤红警示），模型外观与动力学状态一目了然。

6. **纯原生 MoveIt 2 (`demo.launch.py`) 联调**：
   - `moveit_bridge.py` 纯基于 Python 标准库、`pinocchio` 与 `rclpy`，无需 rosbridge 或额外第三方包。
   - 自动监听 `/robot_description` 与 `/joint_states`，实时更新连杆运动学姿态、各关节负荷谱与整机重心。
   - 内置轻量 Mesh HTTP 伺服服务（默认端口 9091），可自动将工作空间中的 CAD 网格直接流式推送到网页前端。

---

## 三、快速使用指南

### 1. 运行 Web 可视化前端
- **直接使用**：使用任意现代浏览器双击打开 `index.html`。
- **本地服务（推荐）**：
  ```bash
  python3 -m http.server 8080
  ```
  在浏览器访问 `http://localhost:8080`。

### 2. 启动 MoveIt / 本地模型桥接服务 (Pinocchio 动力学)
在您的 `Deeplearning` conda 环境中运行：
```bash
conda activate Deeplearning
cd test
python moveit_bridge.py --urdf g1_29dof.urdf
```
*(已在 Deeplearning 环境成功预装 Pinocchio 4.1.0，直接开箱即用)*

### 3. 本地直接拖拽加载 URDF 与 Mesh 模型
1. 打开网页左下角控制面板。
2. 点击 **【📁 选模型文件夹】**：选择您机器人的完整描述目录（包含 `.urdf` 与 `meshes/` 文件夹），前端将瞬间完成解析并高质感呈现整机 3D 实体模型、动力学负荷谱与重力矩！
3. 或者点击 **【📄 选 URDF+Mesh】**：手动多选 `.urdf` 描述文件以及对应的 `.stl` / `.dae` 几何网格文件。

### 4. 与 MoveIt 2 (`demo.launch.py`) 实时联调
1. 启动您的 MoveIt 2 演示节点（以您的机器人配置为准）：
   ```bash
   ros2 launch <your_robot>_moveit_config demo.launch.py
   ```
2. 启动桥接服务：
   ```bash
   conda activate Deeplearning
   python moveit_bridge.py
   ```
3. 打开 `index.html` 点击 **【🔗 连接 MoveIt 2】**，3D 机器人模型与空间重力矩、关节动力学负荷谱即刻实时呈现！
