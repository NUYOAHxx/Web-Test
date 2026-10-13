/**
 * IK 求解器诊断面板: 流水线状态指示 / 性能摘要 / 收敛曲线 / 计算步骤表。
 */

import { el, setText, setClass, setHidden, setHTML, escapeHtml } from "../dom.js";
import { fixed, solverStatus, toneByMagnitude, SOLVER } from "../format.js";
import { state } from "../state.js";
import { drawConvergence } from "./convergence-chart.js";

/** 当前激活的视图 */
let activeView = "curve";

/** 步骤表行 -> 迭代序号 的映射 (事件委托用) */
const ROW_INDEX_ATTR = "data-index";

/** 步骤动作 -> 徽章样式 */
const ACTION_BADGES = {
    CONVERGED: { cls: "converged", text: "DONE" },
    SEED_INIT: { cls: "seed", text: "SEED" },
    COLLISION_PULSE: { cls: "repulse", text: "APF" },
};
const DEFAULT_BADGE = { cls: "descent", text: "DLS" };

/**
 * 绑定面板交互 (视图切换 + 曲线悬停 + 表格悬停联动)。
 *
 * 步骤表的行悬停采用**事件委托**绑定在 tbody 上。
 * 旧实现每帧用 innerHTML 重建表格后对每一行重新 addEventListener,
 * 即 30Hz 下每秒分配约 3000 个监听器 —— 即使旧节点随之回收,
 * 这也是纯浪费, 且使悬停行为在重绘瞬间不稳定。
 */
export function initSolverPanel() {
    bindViewToggle();
    bindCanvasHover();
    bindTableHover();
    window.addEventListener("resize", () => redraw());
}

/* ──────────────────────── 视图切换 ──────────────────────── */

function bindViewToggle() {
    const btnCurve = el("btnTraceCurve");
    const btnSteps = el("btnTraceSteps");
    if (!btnCurve || !btnSteps) return;

    btnCurve.addEventListener("click", () => setView("curve"));
    btnSteps.addEventListener("click", () => setView("steps"));
}

function setView(view) {
    activeView = view;
    const isCurve = view === "curve";
    setClass(el("btnTraceCurve"), `trace-toggle-btn${isCurve ? " active" : ""}`);
    setClass(el("btnTraceSteps"), `trace-toggle-btn${isCurve ? "" : " active"}`);
    setHidden(el("traceCurveView"), !isCurve);
    setHidden(el("traceStepsView"), isCurve);
    // 页脚随视图切换即时响应。此前只在 renderSolver 里处理, 而它由 30Hz
    // 遥测帧驱动 —— 点击后要等下一帧才生效, 表现为"点了没反应"。
    setHidden(el("traceStepsFooter"), isCurve);
    if (isCurve) redraw();
    else if (state.trace.hover >= 0) highlightRow(state.trace.hover);
}

/* ──────────────────────── 曲线悬停 ──────────────────────── */

function bindCanvasHover() {
    const canvas = el("convergenceCanvas");
    if (!canvas) return;

    canvas.addEventListener("mousemove", (e) => {
        const diag = state.trace.data;
        const trace = diag?.convergence_trace;
        if (!trace || trace.length === 0) return;

        const rect = canvas.getBoundingClientRect();
        const plotW = rect.width - 32 - 12;   // 与 convergence-chart 的 PAD 保持一致
        const ratio = (e.clientX - rect.left - 32) / plotW;
        if (ratio < 0 || ratio > 1) {
            setHover(-1);
            return;
        }

        const idx = Math.max(0, Math.min(trace.length - 1, Math.round(ratio * (trace.length - 1))));
        setHover(idx, trace[idx], diag);
    });

    canvas.addEventListener("mouseleave", () => setHover(-1));
}

function setHover(index, value, diag) {
    state.trace.hover = index;

    const tip = el("traceTooltip");
    if (tip) {
        if (index < 0) {
            setHidden(tip, true);
        } else {
            const d = diag || state.trace.data;
            const seedIdx = resolveSeedIndex(d, index);
            const seedName = d?.seed_names?.[seedIdx] || `Seed #${seedIdx}`;
            setHidden(tip, false);
            setHTML(tip,
                `Step ${index + 1}: <strong>${fixed(value, 1)}mm</strong> ` +
                `<span class="tip-seed">(${escapeHtml(seedName)})</span>`);
            // 位置跟随指针对应的迭代点
            const canvas = el("convergenceCanvas");
            const rect = canvas.getBoundingClientRect();
            const plotW = rect.width - 44;
            const x = 32 + (index / Math.max(1, (d?.convergence_trace?.length || 2) - 1)) * plotW;
            tip.style.left = `${Math.min(x, rect.width - 120)}px`;
        }
    }

    highlightRow(index);
    redraw();
}

/** 由迭代序号反查所属种子序号 */
function resolveSeedIndex(diag, iteration) {
    const switches = diag?.seed_switches || [0];
    let idx = 0;
    for (let i = 0; i < switches.length; i++) {
        if (iteration >= switches[i]) idx = i;
    }
    return idx;
}

/* ──────────────────────── 表格悬停 (事件委托) ──────────────────────── */

function bindTableHover() {
    const tbody = el("traceStepsBody");
    if (!tbody) return;

    tbody.addEventListener("mouseover", (e) => {
        const row = e.target.closest(`tr[${ROW_INDEX_ATTR}]`);
        if (row) setHover(Number(row.getAttribute(ROW_INDEX_ATTR)));
    });
    tbody.addEventListener("mouseleave", () => setHover(-1));
}

function highlightRow(index) {
    const tbody = el("traceStepsBody");
    if (!tbody) return;
    tbody.querySelectorAll("tr.step-table-row").forEach((row, i) => {
        row.classList.toggle("step-row-highlight", i === index);
    });
}

/* ──────────────────────── 渲染 ──────────────────────── */

function redraw() {
    const diag = state.trace.data;
    if (!diag || activeView !== "curve") return;
    drawConvergence(el("convergenceCanvas"), diag.convergence_trace || [], {
        seedSwitches: diag.seed_switches || [0],
        hoverIndex: state.trace.hover,
    });
}

/**
 * 渲染一帧求解器诊断。
 * @param {object} diag solver_diagnostics
 */
export function renderSolver(diag) {
    if (!diag || typeof diag !== "object") return;
    state.trace.data = diag;

    const iters = Number(diag.iters) || 0;
    const posErr = Number(diag.pos_err_mm) || 0;
    const timeMs = Number(diag.time_ms) || 0;
    const seedUsed = Number(diag.seed_used) || 0;
    const seedName = diag.seed_name || `Seed #${seedUsed}`;
    const traceLen = diag.convergence_trace?.length || 0;
    const stepCount = diag.step_details?.length || 0;

    // 渲染指纹: 跳过内容未变化的重绘 (30Hz 推送下大多数帧无实质变化)
    const key = `${iters}_${timeMs.toFixed(1)}_${posErr.toFixed(2)}_${seedUsed}_${traceLen}_${stepCount}`;
    const changed = key !== state.trace.lastKey;
    state.trace.lastKey = key;

    setText(el("convergenceModeBadge"), (diag.mode || "—").replace(/_/g, " "));
    setText(el("traceTimeVal"), `${fixed(timeMs, 1)} ms`);
    setText(el("traceItersVal"), String(iters));
    setText(el("traceSeedVal"), seedName);

    const residue = el("traceResidueVal");
    setText(residue, `${fixed(posErr, 2)} mm`);
    setClass(residue, `stat-val val-${toneByMagnitude(posErr, SOLVER)}`);

    const st = solverStatus(iters, posErr);
    for (const node of [el("traceStatusTag"), el("traceStepsStatusTag")]) {
        setText(node, st.label);
        setClass(node, `trace-status-tag ${st.level === "ok" ? "safe" : st.level}`);
    }

    // 步骤表的页脚只在"计算步骤"视图下有意义 —— 曲线视图下它是冗余的第二处状态显示
    setHidden(el("traceStepsFooter"), activeView !== "steps");

    renderStepper(iters, posErr, diag.pipeline_stages);
    renderStepsTable(diag, stepCount, posErr);

    if (changed || state.trace.hover >= 0) redraw();
}

/**
 * 流水线状态指示器。
 *
 * 条目来自服务端下发的 `pipeline_stages`, 不在 HTML 里写死 ——
 * 写死的那五段 (Seed/FK/W-DLS/APF/Gate) 与求解器实际执行的 Pink QP 五段
 * 并不一致, 换求解器后更是完全对不上, 而界面不会有任何提示。
 *
 * @param {number} iters
 * @param {number} posErr
 * @param {Array} stages 服务端下发的阶段描述
 */
function renderStepper(iters, posErr, stages) {
    const stepper = el("pipelineStepper");
    if (!stepper) return;

    const list = Array.isArray(stages) && stages.length
        ? stages
        : [{ id: 1, name: "—", desc: "求解器未上报流水线阶段" }];

    // 阶段集合可能随求解器切换而变化, 故按签名重建, 避免重复追加
    const signature = list.map((s) => `${s.id}:${s.name}`).join("|");
    if (stepper.dataset.signature !== signature) {
        stepper.dataset.signature = signature;
        setHTML(stepper, list.map((s, i) => `
            <li class="pipeline-step-item" data-stage="${escapeHtml(String(s.id ?? i + 1))}"
                title="${escapeHtml(s.desc || s.fullName || "")}">
                <span class="step-num">${escapeHtml(String(s.id ?? i + 1))}</span>
                <span class="step-name">${escapeHtml(s.name ?? "")}</span>
            </li>`).join(""));
    }

    const items = stepper.querySelectorAll(".pipeline-step-item");
    if (iters === 0) {
        items.forEach((n, i) => setClass(n, `pipeline-step-item${i === 0 ? " active" : ""}`));
    } else if (posErr < SOLVER.GOOD) {
        items.forEach((n) => setClass(n, "pipeline-step-item converged"));
    } else {
        // 未收敛: 除最后一段 (抛光/门禁) 外均视为已走过。
        // 原先写死 `i < 4`, 隐含"永远五段"的前提, 阶段数变化即错位。
        const lastActive = Math.max(0, items.length - 1);
        items.forEach((n, i) => setClass(n, `pipeline-step-item${i < lastActive ? " active" : ""}`));
    }
}

/**
 * 逐步计算明细表。
 * @param {object} diag
 * @param {number} count
 * @param {number} finalErrMm
 */
function renderStepsTable(diag, count, finalErrMm) {
    setText(el("traceStepsBadge"), String(count));
    setText(el("stepsSummaryInfo"), count > 0
        ? `共 ${count} 步 · 末残差 ${fixed(finalErrMm, 2)} mm`
        : "待命 · 0 步");

    const tbody = el("traceStepsBody");
    if (!tbody) return;

    if (count === 0) {
        setHTML(tbody, `<tr><td colspan="6" class="steps-empty-placeholder">待命 · 等待下发 IK 指令</td></tr>`);
        return;
    }

    const html = diag.step_details.map((item, i) => {
        const step = item.step ?? (i + 1);
        const seedName = item.seed_name || `Seed #${item.seed_idx || 0}`;
        const rowActive = state.trace.hover === i ? " step-row-highlight" : "";
        const badge = ACTION_BADGES[item.status] || DEFAULT_BADGE;
        const isPolish = item.seed_idx === -1;
        return `<tr class="step-table-row${rowActive}" ${ROW_INDEX_ATTR}="${i}" title="${escapeHtml(item.action || "")}">
            <td><span class="step-idx">${step}</span></td>
            <td><span class="step-seed-name${isPolish ? " is-polish" : ""}">${escapeHtml(seedName)}</span></td>
            <td>${fixed(item.pos_err_mm ?? 0, 2)}</td>
            <td class="best-col">${fixed(item.best_err_mm ?? item.pos_err_mm ?? 0, 2)}</td>
            <td>${deltaCell(item.delta_mm)}</td>
            <td><span class="chip-step-badge ${badge.cls}">${badge.text}</span></td>
        </tr>`;
    }).join("");

    setHTML(tbody, html);
}

/**
 * 改善量单元格。
 *
 * 后端约定的符号: **改善为正** (本步让最优残差下降了多少毫米)。
 * 早期版本用的是"残差变化量"符号 (下降为负), 渲染规则因此完全相反 ——
 * 残差从 12.52mm 降到 0.00mm 的那一行被标成红色向上箭头"恶化"。
 * 现以"改善为正"为唯一约定, 正=绿下箭头, 负=红上箭头。
 *
 * @param {number} deltaMm 改善量 (mm)
 * @returns {string}
 */
function deltaCell(deltaMm) {
    const v = Number(deltaMm) || 0;
    if (v > 0.005) return `<span class="step-delta-good">↓${v.toFixed(2)}</span>`;
    if (v < -0.005) return `<span class="step-delta-bad">↑${Math.abs(v).toFixed(2)}</span>`;
    return `<span class="step-delta-neutral">—</span>`;
}