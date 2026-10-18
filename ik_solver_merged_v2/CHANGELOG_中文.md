# v2 原结构合并版

- 将上一轮独立分支的负荷策略和模型映射并入原 solver_dls.py，不增加独立 Solver 类。
- 同步修改原 solver.py 的多初值编排和最终门槛；保留 HAArmIKSolver、原构造签名、solve 的位置参数以及 make_result 工厂。
- 相对力矩目标不再默认冒充绝对卸载；加入绝对归一化负荷、超软阈值惩罚和逐关节最终拒绝。
- 保留原候选偏好接口，但限制在位姿/边界/静态负荷合格、峰值接近最低的候选短名单中。
- 加权 SVD 数值零空间、Jlog6、末端原点世界对齐方向 Jacobian、方向反向逃逸、真实 FK 回溯。
- 全模型完整新鲜状态、continuous cos/sin 映射、正确的完整 CoM 和导数、显式重力向量配置。
- 移除隐含参考角 ±90°/±150° 工作窗口，不把连续关节 ±π 当机械限位。
- 失败沿用原 NO_VALID_POSE，具体新增原因通过 failure_detail / last_load_diagnostics 提供。
- 合并版专用 74 项测试在真实 Pinocchio 3.9.0 / 4.1.0 下分别通过；不复用上一版的 89 项作为本版证据。

原有 SSIK/direct-psi 源码保留，不代表完成其完整运行验证。两套未知原模块仅在离线测试中使用明确替身；实际用户工程、URDF、Jazzy/MoveIt、Python 3.12 运行和实机待验证。
