"""
收敛轨迹采集测试
================================================================================
守护 `convergence_trace` / `step_details` 这组诊断数据。

背景: 前端"收敛曲线"面板长期显示"待命", 排查后发现后端从未实现采集 ——
`solver_diagnostics` 里的两个字段是硬编码空数组。本文件把以下不变量固定下来:

  1. 轨迹非空, 且长度与 step_details 一致
  2. **序号连续** —— 收敛判定排在记录之前, 否则收敛那一步会缺失
  3. 曲线单调不增 —— 记录的是历史最优残差, 而非当前迭代点
  4. 曲线终点与最终报告的残差一致 (含抛光阶段)
  5. delta_mm 以"改善为正"为约定, 与前端渲染规则一致
  6. seed_switches / seed_names 齐备, 供曲线分区与 tooltip
  7. 关闭采集开关后不留残影, 求解结果本身不受影响
"""

import os
import sys

import numpy as np
import pytest

DIR_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if DIR_ROOT not in sys.path:
    sys.path.insert(0, DIR_ROOT)


@pytest.fixture(scope="module")
def solver():
    """求解器实例 (模块级复用, 加载 URDF 代价较高)。"""
    from core.solver.humanoid_pink_ik import HumanoidPinkIKSolver
    return HumanoidPinkIKSolver()


def _solve(solver, pos, rpy):
    """求解并返回 IKResult (solve_coordinated_ik 返回 4 元组)。"""
    result = solver.solve_coordinated_ik("left_arm", pos, target_rpy=rpy)[-1]
    return result


# 可达目标 (9-DoF 2腰+7臂 实测包络内)
REACHABLE = ([0.36, 0.26, 0.90], [10, 25, -5])
REACHABLE_2 = ([0.30, 0.24, 0.82], [0, 15, 0])


def test_trace_is_populated(solver):
    """轨迹与逐步明细必须非空 —— 这是前端面板能显示数据的前提。"""
    print("=" * 70)
    print("▶ 1. 轨迹数据非空")
    r = _solve(solver, *REACHABLE)

    trace = r["convergence_trace"]
    details = r["step_details"]

    assert isinstance(trace, list) and len(trace) > 0, "convergence_trace 不得为空数组"
    assert isinstance(details, list) and len(details) > 0, "step_details 不得为空数组"
    assert len(trace) == len(details), (
        f"曲线点数 {len(trace)} 与明细条数 {len(details)} 应一一对应"
    )
    print(f"  [OK] 曲线 {len(trace)} 点 / 明细 {len(details)} 条 / iters={r['iters']}")


def test_step_numbers_are_continuous(solver):
    """序号必须 1..N 连续。

    回归防护: 轨迹记录原本排在收敛判定之后, 求解在收敛那一步 break,
    导致序号断档 (实测出现过 1,3,4 —— 第 2 步缺失)。
    """
    print("▶ 2. 步骤序号连续")
    r = _solve(solver, *REACHABLE)
    steps = [d["step"] for d in r["step_details"]]

    assert steps == list(range(1, len(steps) + 1)), (
        f"步骤序号应从 1 连续递增, 实际: {steps[:12]}{'...' if len(steps) > 12 else ''}"
    )
    print(f"  [OK] 序号 1..{len(steps)} 连续无断档")


def test_curve_is_monotonic(solver):
    """曲线单调不增。

    曲线记录的是**历史最优残差**。若改成记录当前迭代点, 多种子切换处
    会出现向上跳变, 且抛光阶段 (从历史最优解出发) 与主循环尾段不在同一
    连续轨迹上, 读者会看到残差"先涨到 293mm 再瞬间归零"。
    """
    print("▶ 3. 曲线单调不增 (记录历史最优残差)")
    r = _solve(solver, *REACHABLE)
    trace = r["convergence_trace"]

    violations = [
        (i, trace[i], trace[i + 1])
        for i in range(len(trace) - 1)
        if trace[i + 1] > trace[i] + 1e-6
    ]
    assert not violations, f"曲线出现回退 (最优残差不应变大): {violations[:5]}"
    assert trace[-1] <= trace[0], "曲线终点不应高于起点"
    print(f"  [OK] {len(trace)} 点单调不增: {trace[0]:.2f}mm -> {trace[-1]:.4f}mm")


def test_curve_endpoint_matches_reported_residual(solver):
    """曲线终点必须与最终报告的残差一致。

    抛光阶段 (高斯-牛顿精修) 才是真正把残差压进容差的那一步;
    若不并入轨迹, 曲线终点会停在抛光前的较高残差处, 与报告值自相矛盾。
    """
    print("▶ 4. 曲线终点 == 报告残差")
    for pos, rpy in (REACHABLE, REACHABLE_2):
        r = _solve(solver, pos, rpy)
        trace = r["convergence_trace"]
        assert trace, "轨迹为空"
        assert abs(trace[-1] - r["pos_err_mm"]) < 0.01, (
            f"曲线终点 {trace[-1]}mm 与报告残差 {r['pos_err_mm']}mm 不一致"
        )
    print("  [OK] 两组目标下曲线终点均与报告残差吻合")


def test_delta_sign_convention(solver):
    """delta_mm 以"改善为正"为约定。

    前端渲染规则是正值 -> 绿色下箭头(变好)、负值 -> 红色上箭头(变差)。
    若后端改成"残差变化量"符号, 残差从 12.52mm 降到 0.00mm 的那一行
    会被标成红色"恶化", 方向完全颠倒。
    """
    print("▶ 5. delta_mm 改善为正")
    r = _solve(solver, *REACHABLE)
    details = r["step_details"]

    for d in details[1:]:
        prev_best = details[d["step"] - 2]["best_err_mm"]
        expected = prev_best - d["best_err_mm"]
        assert abs(d["delta_mm"] - expected) < 0.01, (
            f"第 {d['step']} 步 delta={d['delta_mm']}, "
            f"按改善量应为 {expected:.4f} (前优 {prev_best} - 本优 {d['best_err_mm']})"
        )
        assert d["delta_mm"] >= -1e-6, "历史最优残差不增, 故改善量不应为负"
    print(f"  [OK] {len(details) - 1} 步改善量符号与数值均自洽")


def test_seed_metadata_present(solver):
    """种子元数据齐备 (供曲线分区竖线与 tooltip)。"""
    print("▶ 6. 种子元数据")
    r = _solve(solver, *REACHABLE)

    switches = r.get("seed_switches")
    names = r.get("seed_names")
    assert isinstance(switches, list) and switches, "seed_switches 不得为空"
    assert switches == sorted(switches), "seed_switches 应递增"
    assert isinstance(names, list) and names, "seed_names 不得为空"

    # 每条明细的 seed_name 都应能在种子列表中找到 (或为抛光阶段标记)
    polish = {d["seed_name"] for d in r["step_details"] if d["seed_idx"] == -1}
    for d in r["step_details"]:
        assert d["seed_name"] in set(names) or d["seed_name"] in polish, (
            f"第 {d['step']} 步的 seed_name {d['seed_name']!r} 不在种子列表中"
        )
    print(f"  [OK] 种子切换点 {switches}, 种子名 {len(names)} 个, 抛光步标记 {polish or '无'}")


def test_step_detail_fields(solver):
    """逐步明细字段完整, 供前端表格直接消费。"""
    print("▶ 7. 逐步明细字段完整性")
    r = _solve(solver, *REACHABLE)
    required = {"step", "seed_idx", "seed_name", "pos_err_mm",
                "best_err_mm", "rot_err_deg", "delta_mm", "is_colliding", "status"}
    for d in r["step_details"]:
        missing = required - set(d)
        assert not missing, f"第 {d.get('step')} 步缺字段: {missing}"
        assert isinstance(d["is_colliding"], bool), "is_colliding 应为布尔"
        assert d["status"] in {"DLS", "CONVERGED", "COLLISION_PULSE", "SEED_INIT"}, (
            f"未知的步骤状态: {d['status']!r}"
        )
    print(f"  [OK] {len(r['step_details'])} 条明细字段齐全, 状态取值合法")


def test_trace_can_be_disabled(solver):
    """关闭采集开关后求解行为不变, 且不残留轨迹数据。

    批量求解 (基准测试 / 蒙特卡洛) 不需要逐步明细, 需要能关掉以省去
    每次迭代的 dict 构造开销。
    """
    print("▶ 8. 采集开关")
    pos, rpy = REACHABLE_2

    r_on = _solve(solver, pos, rpy)
    assert r_on["convergence_trace"], "开启时应产出轨迹"

    solver.trace_enabled = False
    try:
        r_off = _solve(solver, pos, rpy)
    finally:
        solver.trace_enabled = True

    assert r_off["convergence_trace"] == [], "关闭后不应产出轨迹"
    assert r_off["step_details"] == [], "关闭后不应产出明细"
    # 关闭采集不得影响求解结论
    assert r_off["success"] == r_on["success"], "采集开关不应改变求解成败"
    assert abs(r_off["pos_err_mm"] - r_on["pos_err_mm"]) < 1e-3, \
        "采集开关不应改变解的质量"
    print(f"  [OK] 关闭后无轨迹, 求解结论一致 (success={r_off['success']}, "
          f"残差={r_off['pos_err_mm']}mm)")


if __name__ == "__main__":
    print("=" * 70)
    print("[START] 收敛轨迹采集测试")
    print("=" * 70)
    from core.solver.humanoid_pink_ik import HumanoidPinkIKSolver
    _s = HumanoidPinkIKSolver()
    test_trace_is_populated(_s)
    test_step_numbers_are_continuous(_s)
    test_curve_is_monotonic(_s)
    test_curve_endpoint_matches_reported_residual(_s)
    test_delta_sign_convention(_s)
    test_seed_metadata_present(_s)
    test_step_detail_fields(_s)
    test_trace_can_be_disabled(_s)
    print("=" * 70)
    print("[SUCCESS] 收敛轨迹采集全部测试通过")
    print("=" * 70)
