"""v15 与 v14 的逐位对照：以 v14 续算行为（solver.CONTINUATION_POSTURE_SETTLE = True）运行 bench/v14/equiv_run.py，
并忽略 v15 新增的诊断键（continuation["hold"]、search_diagnostics["continuation_hold"]）。

用法与 equiv_run.py 相同：equiv_run_v14mode.py <pkg_root> <out.pkl> [cases]
比较：python bench/v14/equiv_cmp.py <v14.pkl> <out.pkl> 10
"""
import os

_here = os.path.dirname(os.path.abspath(__file__))
_src = open(os.path.join(_here, "..", "v14", "equiv_run.py"), encoding="utf-8").read()
_old_ignore = 'IGNORE = {"elapsed_ms",'
_old_import = "from ha_arm_ik import solver as S  # noqa: E402\n"
assert _old_ignore in _src and _old_import in _src
_src = _src.replace(_old_ignore, 'IGNORE = {"hold", "continuation_hold", "elapsed_ms",')
_src = _src.replace(_old_import, _old_import + "S.CONTINUATION_POSTURE_SETTLE = True\n")
exec(compile(_src, os.path.join(_here, "..", "v14", "equiv_run.py"), "exec"), {"__name__": "__main__"})
