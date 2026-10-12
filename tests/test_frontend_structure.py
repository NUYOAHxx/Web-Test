"""
前端结构契约测试 (pytest 封装 + 静态结构校验)
================================================================================
前端此前完全没有自动化测试, 全部依赖人工在浏览器里点。重构后本文件把
可机器校验的契约固定下来, 与 tests/frontend_smoke.mjs (运行期行为) 互补。

覆盖:
  * 前端冒烟测试可执行且全通过 (node 环境缺失时跳过而非误报失败)
  * index.html 中每个脚本引用都真实存在
  * HTML 中的内联 onclick 已被清除 (在 ES modules 下会静默失效)
  * 所有 id 唯一, 且被 JS 引用的 id 都在 HTML 中定义
  * 页面不再引用已删除的单文件 app.js
"""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

WEB_DIR = Path(__file__).resolve().parent.parent / "visualizer" / "web"
SMOKE_TEST = Path(__file__).resolve().parent / "frontend_smoke.mjs"


# ══════════════════════════════════════════════════════════════
# 前端冒烟测试
# ══════════════════════════════════════════════════════════════

@pytest.mark.skipif(shutil.which("node") is None, reason="未安装 node, 跳过前端运行期测试")
def test_frontend_smoke():
    """前端模块冒烟测试必须全通过。

    这条用例守护的是"语法合法但运行期崩溃"的一类缺陷 —— 例如绘图函数
    引用了作用域外的变量, node --check 完全无法发现。
    """
    result = subprocess.run(
        ["node", str(SMOKE_TEST)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"前端冒烟测试未通过:\n{result.stdout}\n{result.stderr}"
    )
    assert "[FAIL]" not in result.stdout, f"存在失败用例:\n{result.stdout}"
    m = re.search(r"前端冒烟测试: (\d+)/(\d+) 通过", result.stdout)
    assert m, f"无法解析测试结果:\n{result.stdout}"
    passed, total = int(m.group(1)), int(m.group(2))
    assert passed == total, f"{passed}/{total} 通过"
    print(f"\n前端冒烟测试: {passed}/{total} 通过")


# ══════════════════════════════════════════════════════════════
# HTML 结构契约
# ══════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def html() -> str:
    return (WEB_DIR / "index.html").read_text(encoding="utf-8")


def test_no_inline_onclick(html):
    """不得存在内联事件处理器。

    内联 onclick 依赖全局函数名 (window.applyPreset)。前端改为 ES modules 后,
    模块作用域不再挂到 window, 所有内联处理器会静默失效 ——
    按钮可见、可点, 但什么也不发生, 控制台无任何报错。
    """
    hits = re.findall(r'on(?:click|change|input|mouseover)\s*=\s*"', html)
    assert not hits, (
        f"发现 {len(hits)} 处内联事件处理器, 在 ES modules 下会静默失效: "
        f"{re.findall(r'on\\w+\\s*=\\s*\"[^\"]*\"', html)}"
    )


def test_uses_module_entry(html):
    """入口必须是 type=module, 且不再引用旧的单文件 app.js。"""
    assert 'type="module"' in html, "入口应使用 ES module"
    assert "app.js" not in html, "旧单文件 app.js 已删除, 不应再被引用"
    assert 'src="js/main.js"' in html


def test_all_script_refs_exist(html):
    """HTML 引用的每个本地脚本都必须真实存在。"""
    refs = re.findall(r'<script[^>]*\ssrc="([^"]+)"', html)
    local = [r for r in refs if not r.startswith(("http://", "https://", "//"))]
    assert local, "应至少引用一个本地脚本"
    missing = [r for r in local if not (WEB_DIR / r).exists()]
    assert not missing, f"引用了不存在的脚本: {missing}"


def test_all_stylesheet_refs_exist(html):
    """HTML 引用的每个本地样式表都必须真实存在。"""
    refs = re.findall(r'<link[^>]*rel="stylesheet"[^>]*href="([^"]+)"', html)
    local = [r.split("?")[0] for r in refs if not r.startswith(("http://", "https://", "//"))]
    missing = [r for r in local if not (WEB_DIR / r).exists()]
    assert not missing, f"引用了不存在的样式表: {missing}"


def test_element_ids_unique(html):
    """id 必须全局唯一。

    重复 id 会让 getElementById 只返回第一个, 导致后续同名元素永远拿不到句柄,
    且表现为"部分面板不更新"这种极难定位的症状。
    """
    ids = re.findall(r'\bid="([^"]+)"', html)
    dupes = {i for i in ids if ids.count(i) > 1}
    assert not dupes, f"存在重复 id: {sorted(dupes)}"


def test_all_js_referenced_ids_defined():
    """JS 中以字面量引用的 id 必须在 HTML 中存在。

    只检查 `el("x")` / requireAll 中的字面量形式; 动态拼接的 id 由
    前端冒烟测试覆盖。
    """
    html_ids = set(re.findall(r'\bid="([^"]+)"', (WEB_DIR / "index.html").read_text(encoding="utf-8")))

    referenced = set()
    for js in (WEB_DIR / "js").rglob("*.js"):
        src = js.read_text(encoding="utf-8")
        referenced |= set(re.findall(r'\bel\(\s*"([A-Za-z][\w-]*)"', src))
        referenced |= set(re.findall(r'requireAll\(\s*\[([^\]]*)\]', src, re.S) and
                          re.findall(r'"([A-Za-z][\w-]*)"',
                                     " ".join(re.findall(r'requireAll\(\s*\[([^\]]*)\]', src, re.S))))

    missing = sorted(referenced - html_ids)
    assert not missing, (
        f"JS 引用了 HTML 中不存在的 id: {missing}\n"
        f"这会导致对应面板永远停留在占位符 (getElementById 返回 null 被静默跳过)"
    )


def test_required_elements_present(html):
    """关键面板的根元素必须存在, 否则整块区域空白。"""
    required = [
        "threeCanvasContainer", "viewportFallback",
        "sliderX", "sliderY", "sliderZ", "sliderRoll", "sliderPitch", "sliderYaw",
        "btnDispatchTarget", "presetGrid",
        "convergenceCanvas", "traceStepsBody", "pipelineStepper",
        "waistJointsList", "armJointsList",
        "minClearanceVal", "radarBadge", "logStream",
    ]
    ids = set(re.findall(r'\bid="([^"]+)"', html))
    missing = [r for r in required if r not in ids]
    assert not missing, f"缺少关键元素: {missing}"


def test_no_g1_references_in_web():
    """前端不得残留 G1 专有标识 (项目已去专有化, G1 仅为测试模型)。"""
    offenders = []
    for path in [WEB_DIR / "index.html", WEB_DIR / "style.css", *sorted((WEB_DIR / "js").rglob("*.js"))]:
        src = path.read_text(encoding="utf-8")
        for m in re.finditer(r"\bG1\b|\bg1_", src):
            line = src[:m.start()].count("\n") + 1
            text = src.splitlines()[line - 1].strip()
            offenders.append(f"{path.name}:{line}  {text}")
    assert not offenders, "前端存在 G1 专有标识:\n" + "\n".join(offenders)


# ══════════════════════════════════════════════════════════════
# CSS 契约
# ══════════════════════════════════════════════════════════════

def test_css_defines_severity_tokens():
    """严重度语义色必须集中定义为变量。

    这些 token 由 format.js 的判定结果直接消费 (var(--sev-ok) 等)。
    若散落为硬编码色值, 就无法保证"判定与展示同源", 也无法统一改主题。
    """
    css = (WEB_DIR / "style.css").read_text(encoding="utf-8")
    for token in ["--sev-ok", "--sev-warn", "--sev-alert"]:
        assert f"{token}:" in css, f"缺少严重度语义色变量 {token}"
    for token in ["--sev-ok-glow", "--sev-warn-glow", "--sev-alert-glow"]:
        assert f"{token}:" in css, f"缺少严重度发光变量 {token}"


def test_css_classes_used_by_js_exist():
    """JS 赋给元素的类名必须在 CSS 中有对应定义 (抽查关键语义类)。"""
    css = (WEB_DIR / "style.css").read_text(encoding="utf-8")
    for cls in [
        "val-green", "val-gold", "val-red",
        "clearance-val", "zone-card", "radar-status-badge",
        "joint-row", "joint-name", "joint-deg", "joint-margin", "bidi-bar-fill",
        "torque-bar-fill", "step-delta-good", "step-delta-bad",
        "trace-status-tag", "chip-step-badge", "pipeline-step-item",
    ]:
        assert f".{cls}" in css, f"CSS 缺少类定义 .{cls}"