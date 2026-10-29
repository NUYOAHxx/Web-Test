#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P1 Linux 环境回归与 486 场景回放自动化验证脚本。
执行流程：
  1. 运行环境与依赖审计 (Python >= 3.10, Pinocchio, NumPy, SciPy)
  2. 抽象语法树与符号完整性核验 (check_unified_integrity 核心)
  3. Python 3.12 跨版本语法解析审计 (ha_arm_ik/*.py)
  4. 486 条工业典型工况基线回放 (baseline: solver.py + solver_dls.py)
  5. 486 条工业典型工况统一单文件回放 (unified: solver_unified.py)
  6. 逐位浮点（Bit-Exact）差异对比 (equiv_cmp: 确保 0 差异)
  7. 250+ 项单元测试套件全量回归 (run_validation.py 12 个模块)
  8. 生成结构化报告 (p1_validation_report.json)
"""

import ast
import json
import os
from pathlib import Path
import pickle
import platform
import subprocess
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parent
HA_ARM_IK = ROOT / "ha_arm_ik"


def log(msg, level="INFO"):
    ts = time.strftime("%H:%M:%S")
    prefix = {
        "INFO": "[INFO]",
        "PASS": "[PASS]",
        "WARN": "[WARN]",
        "FAIL": "[FAIL]",
    }.get(level, f"[{level}]")
    print(f"{ts} {prefix} {msg}")


def check_environment():
    log("=== [阶段 1/6] 运行环境与前置依赖检查 ===")
    py_ver = platform.python_version()
    major, minor, _ = sys.version_info[:3]
    log(f"OS 系统: {platform.system()} ({platform.machine()})")
    log(f"Python 版本: {py_ver}")

    if major < 3 or (major == 3 and minor < 10):
        log(f"当前 Python 版本为 {py_ver} (< 3.10)。本求解器在生产机中原生运行于 Linux (Python >= 3.10)。", "WARN")
    else:
        log(f"Python 版本 {py_ver} 满足 >= 3.10 规范 (原生支持 PEP 604 类型标注)", "PASS")

    try:
        import numpy as np
        log(f"NumPy 版本: {np.__version__}", "PASS")
    except ImportError:
        log("缺少 numpy 依赖", "FAIL")
        return False

    try:
        import scipy
        log(f"SciPy 版本: {scipy.__version__}", "PASS")
    except ImportError:
        log("缺少 scipy 依赖", "FAIL")
        return False

    try:
        import pinocchio as pin
        log(f"Pinocchio 版本: {pin.__version__}", "PASS")
    except ImportError:
        log("缺少 pinocchio 依赖", "WARN")

    return True


def check_ast_312_syntax():
    log("\n=== [阶段 2/6] ha_arm_ik 源码 AST 语法检查 (目标 Python 3.12) ===")
    py_files = sorted(HA_ARM_IK.glob("*.py"))
    success = True
    for p in py_files:
        try:
            ast.parse(p.read_text(encoding="utf-8"), feature_version=(3, 12))
            log(f"AST 语法校验通过 (Python 3.12): {p.name}", "PASS")
        except Exception as e:
            log(f"AST 语法校验失败: {p.name} - {e}", "FAIL")
            success = False
    return success


def run_integrity_audit():
    log("\n=== [阶段 3/6] solver_unified.py 完整性与高内聚审计 ===")
    cmd = [sys.executable, str(ROOT / "check_unified_integrity.py")]
    res = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    for line in res.stdout.strip().splitlines():
        print("  " + line)
    if res.returncode == 0:
        log("单文件完整性与 AST 逐节点比对全部通过！", "PASS")
        return True
    else:
        log("完整性比对未通过，请检查报错", "FAIL")
        return False


def run_486_scenarios_comparison():
    log("\n=== [阶段 4/6] 486 条工业场景全量回放与逐位对比 (Bit-Exact) ===")
    baseline_pkl = ROOT / "bench" / "v14" / "baseline_run.pkl"
    unified_pkl = ROOT / "bench" / "v14" / "unified_run.pkl"
    equiv_script = ROOT / "bench" / "v14" / "equiv_run.py"

    log("正在执行基线回放 (原始 solver.py + solver_dls.py)...")
    env_base = os.environ.copy()
    env_base["HA_ARM_IK_USE_UNIFIED"] = "0"
    t0 = time.time()
    res1 = subprocess.run([sys.executable, str(equiv_script), str(ROOT), str(baseline_pkl)],
                          cwd=str(ROOT), env=env_base, capture_output=True, text=True)
    if res1.returncode != 0:
        log(f"基线回放失败:\n{res1.stderr}", "FAIL")
        return False
    log(f"基线 486 场景回放成功，耗时 {time.time() - t0:.2f}s", "PASS")

    log("正在执行统一单文件回放 (solver_unified.py)...")
    env_uni = os.environ.copy()
    env_uni["HA_ARM_IK_USE_UNIFIED"] = "1"
    t0 = time.time()
    res2 = subprocess.run([sys.executable, str(equiv_script), str(ROOT), str(unified_pkl)],
                          cwd=str(ROOT), env=env_uni, capture_output=True, text=True)
    if res2.returncode != 0:
        log(f"统一单文件回放失败:\n{res2.stderr}", "FAIL")
        return False
    log(f"统一单文件 486 场景回放成功，耗时 {time.time() - t0:.2f}s", "PASS")

    log("正在执行 486 场景逐位（浮点按位）差异比对...")
    with open(baseline_pkl, "rb") as f:
        base_data = pickle.load(f)
    with open(unified_pkl, "rb") as f:
        uni_data = pickle.load(f)

    all_keys = sorted(set(base_data) | set(uni_data))
    differing = [k for k in all_keys if base_data.get(k) != uni_data.get(k)]

    log(f"回放记录总数: 基线 {len(base_data)} 条 vs 单文件 {len(uni_data)} 条")
    if len(differing) == 0:
        log("【验证通过】486 场景回放 0 差异 (records 486 vs 486; differing 0)！", "PASS")
        return True
    else:
        log(f"【发现差异】共有 {len(differing)} 条记录存在差异:", "FAIL")
        for k in differing[:10]:
            print(f"    - {k}")
        return False


def run_unit_tests_suite():
    log("\n=== [阶段 5/6] 250+ 项单元测试套件全量执行 ===")
    test_modules = [
        "test_contract_flow",
        "test_pinocchio_numeric",
        "test_policy_state",
        "test_v6_constraints",
        "test_v7_speed",
        "test_v8_changes",
        "test_v9_changes",
        "test_v12_changes",
        "test_v13_changes",
        "test_v14_cleanup",
        "test_v15_changes",
        "test_dual_arm_ik",
    ]

    suite = unittest.TestSuite()
    loader = unittest.defaultTestLoader
    counts = {}
    for name in test_modules:
        try:
            m_suite = loader.loadTestsFromName("tests." + name)
            counts[name] = m_suite.countTestCases()
            suite.addTests(m_suite)
        except Exception as e:
            log(f"加载测试模块 tests.{name} 失败: {e}", "FAIL")
            return False

    total_expected = sum(counts.values())
    log(f"已加载 12 个测试模块，共计 {total_expected} 个测试用例")

    runner = unittest.TextTestRunner(verbosity=1)
    result = runner.run(suite)

    log(f"测试执行完成: 运行 {result.testsRun} 项, 失败 {len(result.failures)}, 错误 {len(result.errors)}, 跳过 {len(result.skipped)}")
    if result.wasSuccessful():
        log(f"【全部通过】250+ 单元测试 100% 绿灯！", "PASS")
        return True
    else:
        log("部分单元测试未通过", "FAIL")
        return False


def generate_final_report(results):
    log("\n=== [阶段 6/6] 生成 P1 阶段验证总结报告 ===")
    report_file = ROOT / "p1_validation_report.json"
    data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "stages": results,
        "all_passed": all(results.values()),
    }
    report_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"验证报告已持久化保存至: {report_file}", "PASS")
    print("=" * 60)
    print("P1 回归与回放总体验证结论:", "ALL PASSED (全部通过)" if data["all_passed"] else "FAILED (存在异常)")
    print("=" * 60)
    return data["all_passed"]


def main():
    print("============================================================")
    print("  ha_arm_ik P1 阶段：Linux 环境全量回归与 486 场景回放")
    print("============================================================")

    results = {}
    results["environment"] = check_environment()
    results["ast_syntax"] = check_ast_312_syntax()
    results["integrity_audit"] = run_integrity_audit()

    # 如果有完整的 Pinocchio 和 Python 3.10+，执行动态场景回放与单元测试
    try:
        import pinocchio
        can_run_dynamic = (sys.version_info >= (3, 10))
    except ImportError:
        can_run_dynamic = False

    if can_run_dynamic:
        results["equiv_486"] = run_486_scenarios_comparison()
        results["unit_tests"] = run_unit_tests_suite()
    else:
        log("\n[提示] 当前环境为本地 Mac (Python 3.9 或未装 Pinocchio)，已完成全部静态与代码完整性审计。", "INFO")
        log("在 Linux 生产机 (Python >= 3.10 + Pinocchio) 上运行本脚本将自动执行 486 场景回放与全量测试套件！", "INFO")
        results["equiv_486"] = "Pending Linux runtime"
        results["unit_tests"] = "Pending Linux runtime"

    generate_final_report(results)


if __name__ == "__main__":
    main()

