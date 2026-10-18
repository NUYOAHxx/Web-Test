# 与已提供 Solver 的对应关系

此文件描述独立候选分支的改动，不声称已修改用户原工程。

| 原函数/行为 | 新实现位置 |
|---|---|
| `_compute_nullspace_gradient_and_cost()` 相对力矩差异 | `LoadPolicy.cost_gradient()` + `LoadAwareIKSolver._secondary()` |
| `_solve_sns_clamped_step()` 的加权次任务 | `metric_step()` + `_bounded_direction()` |
| `pinocchio_ik_one_seed()` | `_one_seed()`：任务一致化、回溯与返回前重算 |
| `collect_pinocchio_candidates_and_seeds()` | `_seeds()` + `solve()`：负荷参与搜索、锁腰后比较解锁候选 |
| `_finalize_result()` / 未取得的 `ik_candidate.py` | `select_load_candidate()` + 最终 `_candidate()` 复核；不猜测旧评分实现 |
| `active_q_to_pinocchio_q()` neutral 填充 | `_full_q()`：完整输入状态 + continuous cos/sin |
| `evaluate_pose_physics()` | `evaluate_state()`：只读静态力矩与完整模型 CoM |
| `SerialChain.bounds()` 连续关节 ±π | 实际 URDF 类型分开处理；continuous 默认无机械角度硬边界 |

保留：多初值 DLS 思路、末端 full/tool_dir/tool_cone、参考姿态偏好、腰臂参与、可选 CoM 回拉。

未迁移：SSIK、direct-psi、SEW 最终候选偏好、原 IKResult/Status/FailureReason 类型，以及原距离驱动的腰部释放规则。

新文件不会自动成为原程序或 MoveIt 的求解器。需要显式切换调用入口，并根据 README 适配第三返回值与完整状态输入。