"""v9：用户已提供真实 ik_common.py / ik_candidate.py / robot_model.py（robot.py）源码。

测试全部改用真实模块（不再注入任何测试替身）。保留 install() / MAKE_RESULT_CALLS 名称，
仅为兼容旧测试的导入语句；install() 只负责导入真实模块并确认来源。v8 及以前的测试替身
留档于 baseline/tests_stubs_v8.py。
"""
from pathlib import Path

INSTALLED = False
MAKE_RESULT_CALLS = []  # 兼容旧导入；v9 起不再记录（改为直接检查返回的 IKResult）


def install():
    global INSTALLED
    if INSTALLED:
        return
    import ha_arm_ik.ik_common as common
    import ha_arm_ik.ik_candidate as candidate
    import ha_arm_ik.robot_model as robot_model

    pkg = Path(__file__).resolve().parents[1] / "ha_arm_ik"
    for module in (common, candidate, robot_model):
        if Path(module.__file__).resolve().parent != pkg:
            raise RuntimeError(f"{module.__name__} not loaded from {pkg}: {module.__file__}")
    INSTALLED = True
