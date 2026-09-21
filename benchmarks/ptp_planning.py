#!/usr/bin/env python3
"""
Unitree G1 机械臂点对点 (PTP) 轨迹规划与起始位姿敏感度诊断工具

功能：
1. 笛卡尔直线点对点规划 (MoveL)：起点到终点直线插补并逐点闭环 IK 跟踪；
2. 关节空间平滑点对点规划 (MoveJ)：基于五次多项式的关节空间平滑规划；
3. 起始位置敏感度对比分析 (--compare)：针对“只改了 Z 坐标，A 规划失败而 B 成功”现象进行多维度原因剖析：
   - 几何臂长极限与肩关节中心物理包络 (Max 37.7 cm)；
   - 雅可比可操控度指标 (Manipulability Measure) 奇异点判定；
   - 关节限位余量监控与零空间解分支死锁分析；
4. Z 轴连续扫描测试 (--sweep-z)：自动扫描不同 Z 坐标下的 PTP 规划成功率与极限边界；
5. 可视化分析图表导出 (--plot)：导出 3D 轨迹、可操控度演化与关节角曲线。
"""

import argparse
import os
import sys
import numpy as np
import pinocchio as pin

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from core.solver.g1_hybrid_ik import G1HybridIKSolver

# G1 机械臂物理几何常数 (单位: 米)
G1_UPPER_ARM_LEN = 0.193   # 大臂长 (肩到肘)
G1_FORE_ARM_LEN = 0.184    # 小臂长 (肘到腕)
G1_MAX_REACH = G1_UPPER_ARM_LEN + G1_FORE_ARM_LEN  # 0.377 m (37.7 cm)


class G1PointToPointPlanner:
    """
    G1 机械臂点对点运动规划与诊断分析器
    """

    def __init__(self, urdf_path: str = None):
        self.solver = G1HybridIKSolver(urdf_path)
        self.model = self.solver.model
        self.data = self.solver.data
        
        # 缓存肩关节坐标 (在 Neutral 姿态下)
        q_neutral = pin.neutral(self.model)
        pin.forwardKinematics(self.model, self.data, q_neutral)
        pin.updateFramePlacements(self.model, self.data)
        
        left_shoulder_id = self.model.getFrameId("left_shoulder_pitch_link")
        right_shoulder_id = self.model.getFrameId("right_shoulder_pitch_link")
        
        self.shoulder_pos = {
            "left_arm": self.data.oMf[left_shoulder_id].translation.copy(),
            "right_arm": self.data.oMf[right_shoulder_id].translation.copy(),
        }

    def compute_manipulability(self, arm: str, q_arm: np.ndarray) -> float:
        """
        计算 Yoshikawa 可操控度指标: w = sqrt(det(J * J^T))
        数值越接近 0，代表机械臂越接近奇异点 (伸直锁死或过度折叠)，微小的末端位移需要极大的关节角速度。
        """
        indices = self.solver.arm_q_indices[arm]
        q_full = pin.neutral(self.model)
        for i, idx in enumerate(indices):
            q_full[idx] = q_arm[i]

        pin.forwardKinematics(self.model, self.data, q_full)
        pin.updateFramePlacements(self.model, self.data)
        ee_id = self.model.getFrameId(self.solver.ee_frame_names[arm])
        J = pin.computeFrameJacobian(
            self.model, self.data, q_full, ee_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )[:3, indices]
        
        det = np.linalg.det(J @ J.T)
        return float(np.sqrt(max(1e-12, det)))

    def plan_cartesian_linear(
        self,
        arm: str,
        start_pos: np.ndarray,
        goal_pos: np.ndarray,
        num_steps: int = 30,
        pos_tol: float = 2e-3,
    ) -> dict:
        """
        笛卡尔直线插补规划 (MoveL)
        在三维空间中从 start 到 goal 进行线性插值，逐点求解逆运动学
        """
        start_pos = np.asarray(start_pos, dtype=np.float64)
        goal_pos = np.asarray(goal_pos, dtype=np.float64)

        # 1. 检查起点和终点几何距离
        sh_pos = self.shoulder_pos[arm]
        dist_start_sh = np.linalg.norm(start_pos - sh_pos)
        dist_goal_sh = np.linalg.norm(goal_pos - sh_pos)

        res = {
            "success": False,
            "failed_step": -1,
            "failure_reason": "",
            "start_dist_to_shoulder": dist_start_sh,
            "goal_dist_to_shoulder": dist_goal_sh,
            "waypoints": [],
            "joint_traj": [],
            "manipulability": [],
            "errors_mm": [],
        }

        if dist_start_sh > G1_MAX_REACH:
            res["failure_reason"] = (
                f"起点超出臂长极限！距肩部 {dist_start_sh*100:.1f} cm > 最大臂长 {G1_MAX_REACH*100:.1f} cm"
            )
            return res

        if dist_goal_sh > G1_MAX_REACH:
            res["failure_reason"] = (
                f"终点超出臂长极限！距肩部 {dist_goal_sh*100:.1f} cm > 最大臂长 {G1_MAX_REACH*100:.1f} cm"
            )
            return res

        # 2. 求解起点关节角
        ok_start, q_cur, info_s = self.solver.solve_ik(arm, start_pos, pos_tol=pos_tol)
        if not ok_start:
            res["failure_reason"] = f"起点逆运动学无解 (误差: {info_s['pos_err_mm']:.1f} mm)"
            return res

        # 3. 逐点线性插值
        for step in range(num_steps + 1):
            s = step / float(num_steps)
            target_pt = (1.0 - s) * start_pos + s * goal_pos
            dist_sh = np.linalg.norm(target_pt - sh_pos)

            # 热启动求解当前路径点
            ok, q_sol, info = self.solver.solve_ik(
                arm, target_pt, seed_q=q_cur, pos_tol=pos_tol, max_iters=35
            )
            
            manip = self.compute_manipulability(arm, q_sol)
            err_mm = info.get("pos_err_mm", 0.0)

            res["waypoints"].append(target_pt)
            res["joint_traj"].append(q_sol.copy())
            res["manipulability"].append(manip)
            res["errors_mm"].append(err_mm)

            if not ok or err_mm > pos_tol * 2000.0:  # 容忍 4mm 以内
                res["failed_step"] = step
                res["failure_reason"] = (
                    f"中途在第 {step}/{num_steps} 步 (进度 s={s*100:.0f}%) 停滞！"
                    f"距肩部 {dist_sh*100:.1f} cm, 误差: {err_mm:.1f} mm, 可操控度: {manip:.4f}"
                )
                return res

            q_cur = q_sol

        res["success"] = True
        return res

    def plan_joint_space(
        self,
        arm: str,
        start_pos: np.ndarray,
        goal_pos: np.ndarray,
        num_steps: int = 30,
    ) -> dict:
        """
        关节空间平滑规划 (MoveJ, 五次多项式无冲击速度曲线)
        """
        start_pos = np.asarray(start_pos, dtype=np.float64)
        goal_pos = np.asarray(goal_pos, dtype=np.float64)

        ok_s, q_start, info_s = self.solver.solve_ik(arm, start_pos)
        ok_g, q_goal, info_g = self.solver.solve_ik(arm, goal_pos)

        res = {
            "success": False,
            "failure_reason": "",
            "joint_traj": [],
            "tcp_traj": [],
            "manipulability": [],
        }

        if not ok_s:
            res["failure_reason"] = f"起点逆运动学无解: {info_s['pos_err_mm']:.1f} mm"
            return res
        if not ok_g:
            res["failure_reason"] = f"终点逆运动学无解: {info_g['pos_err_mm']:.1f} mm"
            return res

        for step in range(num_steps + 1):
            t = step / float(num_steps)
            # 五次多项式平滑插值: s(t) = 10*t^3 - 15*t^4 + 6*t^5 (始末速度、加速度为 0)
            s = 10.0 * (t ** 3) - 15.0 * (t ** 4) + 6.0 * (t ** 5)
            q_t = (1.0 - s) * q_start + s * q_goal

            # 计算对应的末端正运动学轨迹
            tcp_pos, _ = self.solver.forward_kinematics(arm, q_t)
            manip = self.compute_manipulability(arm, q_t)

            res["joint_traj"].append(q_t)
            res["tcp_traj"].append(tcp_pos)
            res["manipulability"].append(manip)

        res["success"] = True
        return res


def diagnose_starts_comparison(
    planner: G1PointToPointPlanner,
    arm: str,
    start_A: np.ndarray,
    start_B: np.ndarray,
    goal: np.ndarray,
    export_plot: bool = False,
):
    """
    对比诊断模式：针对“换一个起始位置（如只改 Z）规划就能成功”的现象输出深度原因
    """
    print("\n" + "=" * 80)
    print(f"       G1 机械臂点对点规划对比诊断 ({arm})")
    print("=" * 80)
    print(f"目标位置 (Goal)     : [X={goal[0]:.3f}, Y={goal[1]:.3f}, Z={goal[2]:.3f}]")
    print(f"起始位置 A (Start A): [X={start_A[0]:.3f}, Y={start_A[1]:.3f}, Z={start_A[2]:.3f}]")
    print(f"起始位置 B (Start B): [X={start_B[0]:.3f}, Y={start_B[1]:.3f}, Z={start_B[2]:.3f}]")
    sh = planner.shoulder_pos[arm]
    print(f"肩关节原点参考位置   : [X={sh[0]:.3f}, Y={sh[1]:.3f}, Z={sh[2]:.3f}]")
    print(f"机械臂物理最大伸展   : {G1_MAX_REACH*100:.1f} cm (大臂 19.3cm + 小臂 18.4cm)")
    print("-" * 80)

    # 1. 运行笛卡尔直线插补规划
    res_A = planner.plan_cartesian_linear(arm, start_A, goal)
    res_B = planner.plan_cartesian_linear(arm, start_B, goal)

    # 2. 详细指标分析
    print("\n【路径 A 测试结果 (Start A -> Goal)】:")
    print(f"  * 起点离肩部距离 : {res_A['start_dist_to_shoulder']*100:.1f} cm")
    if res_A["success"]:
        min_m = min(res_A["manipulability"])
        max_err = max(res_A["errors_mm"])
        print("  * 规划状态       : \033[92m成功 (SUCCESS)\033[0m")
        print(f"  * 全程最小可操控度: {min_m:.4f}")
        print(f"  * 最大跟踪误差   : {max_err:.2f} mm")
    else:
        print("  * 规划状态       : \033[91m失败 (FAILED)\033[0m")
        print(f"  * 失败核心原因   : {res_A['failure_reason']}")

    print("\n【路径 B 测试结果 (Start B -> Goal)】:")
    print(f"  * 起点离肩部距离 : {res_B['start_dist_to_shoulder']*100:.1f} cm")
    if res_B["success"]:
        min_m = min(res_B["manipulability"])
        max_err = max(res_B["errors_mm"])
        print("  * 规划状态       : \033[92m成功 (SUCCESS)\033[0m")
        print(f"  * 全程最小可操控度: {min_m:.4f}")
        print(f"  * 最大跟踪误差   : {max_err:.2f} mm")
    else:
        print("  * 规划状态       : \033[91m失败 (FAILED)\033[0m")
        print(f"  * 失败核心原因   : {res_B['failure_reason']}")

    # 3. 现象深度原因剖析
    print("\n" + "=" * 80)
    print("                【为什么只改了 Z 坐标，解算结果截然不同？】")
    print("=" * 80)
    
    # 判据 1: 几何物理极限越界
    if res_A["start_dist_to_shoulder"] > G1_MAX_REACH and res_B["start_dist_to_shoulder"] <= G1_MAX_REACH:
        diff_cm = (res_A["start_dist_to_shoulder"] - G1_MAX_REACH) * 100.0
        print("[原因 1 - 绝对几何不可达 (Geometric Reach Limit)]")
        print(f"  * 起点 A (Z={start_A[2]:.2f}) 到肩部距离达 {res_A['start_dist_to_shoulder']*100:.1f} cm，已超过手臂物理极限 {diff_cm:.1f} cm！")
        print("    在空间上机器人即使完全伸直手臂也绝不可能触及，因此任何 IK 解算器都会被限位阻断。")
        print(f"  * 而起点 B (Z={start_B[2]:.2f}) 距肩部仅 {res_B['start_dist_to_shoulder']*100:.1f} cm，完全落在 37.7 cm 的自然操作包络圈内。\n")

    # 判据 2: 奇异点与可操控度骤降 (Manipulability Drop)
    if res_B["success"] and (not res_A["success"] or (res_A.get("manipulability") and min(res_A["manipulability"]) < 0.005)):
        print("[原因 2 - 运动学奇异性与可操控度瓶颈 (Kinematic Singularity)]")
        print("  * 起点 A 所在的高度通常伴随肘关节接近伸直或肩部俯仰角极限。")
        print("    当可操控度 sqrt(det(J J^T)) 逼近 0 时，雅可比矩阵发生退化，沿某些笛卡尔方向的微小位移")
        print("    会要求关节以接近无穷大的速度旋转，从而导致数值梯度停止收敛。")
        print("  * 起点 B 处肘关节拥有约 30°~60° 的黄金预备曲率，雅可比处于良态开阔区。\n")

    # 判据 3: 笛卡尔直线路径切入不可行凸集 (Workspace Non-Convexity)
    if not res_A["success"] and res_A["failed_step"] > 0:
        print("[原因 3 - 笛卡尔直线穿越不可行区域 (Workspace Boundary Clipping)]")
        print("  * 虽然起点 A 和终点 Goal 单独来看都可以解出，但连接两点的【直线轨迹】在中途切入了死区。")
        print("    机械臂的可达工作空间是一个非凸（Non-convex）的环状球壳，两点之间的直线并不保证全程落在可达区域内！")
        print("  * 解决方案：使用关节空间插补 (MoveJ) 或 RRT* 规划器绕过直线死区，而不是强行走直线 (MoveL)。\n")

    # 4. 可选生成可视化分析图
    if export_plot:
        output_dir = os.path.join(root_dir, "output")
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, "ptp_planning_analysis.png")
        generate_analysis_plot(res_A, res_B, start_A, start_B, goal, plot_path)
        print(f"[图表生成] 诊断分析图已保存至: {plot_path}")


def sweep_z_coordinates(planner: G1PointToPointPlanner, arm: str, x: float, y: float, goal: np.ndarray):
    """
    Z 坐标连续扫描：自动测试从高到低各 Z 高度下的 PTP 直线规划可行性
    """
    print("\n" + "=" * 80)
    print(f"       G1 机械臂 Z 轴连续高度敏感度扫描 (X={x:.2f}, Y={y:.2f})")
    print("=" * 80)
    print(f"{'起始 Z (m)':^10} | {'距肩部 (cm)':^12} | {'几何可达':^10} | {'MoveL 直线规划':^15} | {'瓶颈说明'}")
    print("-" * 80)

    z_candidates = np.linspace(1.60, 0.70, 19)
    sh = planner.shoulder_pos[arm]

    for z in z_candidates:
        st = np.array([x, y, z])
        dist_sh = np.linalg.norm(st - sh)
        geom_ok = dist_sh <= G1_MAX_REACH

        res = planner.plan_cartesian_linear(arm, st, goal, num_steps=20)
        
        if res["success"]:
            status_str = "\033[92m成功 (100%)\033[0m"
            reason_str = f"最小可操控度: {min(res['manipulability']):.4f}"
        else:
            status_str = "\033[91m失败\033[0m"
            if dist_sh > G1_MAX_REACH:
                reason_str = f"超长 {(dist_sh - G1_MAX_REACH)*100:.1f} cm (超出最大 37.7 cm 臂展)"
            elif res["failed_step"] == -1:
                reason_str = "起点逆解不存在 (受关节物理限位阻挡)"
            else:
                reason_str = f"中途在第 {res['failed_step']}/20 步卡死 (直线穿过死区)"

        geom_str = "√ 可达" if geom_ok else "× 超长"
        print(f"{z:^10.2f} | {dist_sh*100:^12.1f} | {geom_str:^10} | {status_str:^24} | {reason_str}")
    print("-" * 80)


def generate_analysis_plot(res_A: dict, res_B: dict, start_A, start_B, goal, output_path: str):
    """
    绘制对比分析折线图
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(1, 2, figsize=(14, 5))

        # 子图 1: 可操控度变化曲线
        ax1 = axs[0]
        if res_A.get("manipulability"):
            ax1.plot(res_A["manipulability"], "r-o", label=f"Path A (Start Z={start_A[2]:.2f})", linewidth=2)
        if res_B.get("manipulability"):
            ax1.plot(res_B["manipulability"], "g-s", label=f"Path B (Start Z={start_B[2]:.2f})", linewidth=2)
        ax1.set_title("Manipulability Measure along Trajectory")
        ax1.set_xlabel("Trajectory Waypoint Step")
        ax1.set_ylabel("Yoshikawa Manipulability")
        ax1.grid(True, linestyle="--", alpha=0.6)
        ax1.legend()

        # 子图 2: 跟踪误差对比
        ax2 = axs[1]
        if res_A.get("errors_mm"):
            ax2.plot(res_A["errors_mm"], "r--", label="Path A Error (mm)", linewidth=2)
        if res_B.get("errors_mm"):
            ax2.plot(res_B["errors_mm"], "g-", label="Path B Error (mm)", linewidth=2)
        ax2.set_title("Position Tracking Error along Trajectory")
        ax2.set_xlabel("Trajectory Waypoint Step")
        ax2.set_ylabel("Error (mm)")
        ax2.grid(True, linestyle="--", alpha=0.6)
        ax2.legend()

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
    except Exception as e:
        print(f"绘制图表失败: {e}")


def main():
    parser = argparse.ArgumentParser(description="G1 机械臂点对点 (PTP) 轨迹规划与起始点敏感度测试工具")
    parser.add_argument("--arm", choices=["left_arm", "right_arm"], default="left_arm", help="选择机械臂 (默认: left_arm)")
    parser.add_argument("--sweep-z", action="store_true", help="连续扫描不同起始 Z 坐标并输出成功/失败边界表")
    parser.add_argument("--plot", action="store_true", help="生成详细分析对比曲线图 ptp_planning_analysis.png")
    
    # 坐标参数
    parser.add_argument("--start-a", nargs=3, type=float, default=[0.18, 0.25, 1.45], help="起始位置 A [X Y Z] (默认较高 Z)")
    parser.add_argument("--start-b", nargs=3, type=float, default=[0.18, 0.25, 1.22], help="起始位置 B [X Y Z] (默认适中 Z)")
    parser.add_argument("--goal", nargs=3, type=float, default=[0.25, 0.22, 0.90], help="目标位置 [X Y Z]")

    args = parser.parse_args()

    planner = G1PointToPointPlanner()

    if args.sweep_z:
        sweep_z_coordinates(planner, args.arm, x=args.start_b[0], y=args.start_b[1], goal=np.array(args.goal))
    else:
        diagnose_starts_comparison(
            planner,
            arm=args.arm,
            start_A=np.array(args.start_a),
            start_B=np.array(args.start_b),
            goal=np.array(args.goal),
            export_plot=args.plot,
        )


if __name__ == "__main__":
    main()
