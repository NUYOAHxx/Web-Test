"""v15: ik_common, ik_candidate, robot_model 已由 ik_base 统一替代。"""
from pathlib import Path

INSTALLED = False
MAKE_RESULT_CALLS = []


def install():
    global INSTALLED
    if INSTALLED:
        return
    import ha_arm_ik.ik_base as base

    pkg = Path(__file__).resolve().parents[1] / "ha_arm_ik"
    if Path(base.__file__).resolve().parent != pkg:
        raise RuntimeError(f"{base.__name__} not loaded from {pkg}: {base.__file__}")
    INSTALLED = True
