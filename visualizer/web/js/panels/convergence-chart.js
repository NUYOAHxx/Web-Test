/**
 * IK 残差收敛曲线 Canvas 渲染器。
 *
 * 职责严格限定为"把一条残差序列画成图": 不持有业务状态,
 * 不触碰 DOM 结构, 只按传入参数作画。
 * 面板状态 (当前数据 / 悬停索引) 由 solver.js 管理并透传。
 */

import { SOLVER } from "../format.js";

/** 画布内边距 */
const PAD = { left: 32, right: 12, top: 14, bottom: 20 };

/**
 * 绘制收敛曲线。
 * @param {HTMLCanvasElement} canvas
 * @param {number[]} trace 残差序列 (mm)
 * @param {object} opts
 * @param {number[]} opts.seedSwitches 多种子切换点 (迭代序号)
 * @param {number} opts.hoverIndex   悬停迭代序号, -1 表示无
 */
export function drawConvergence(canvas, trace, opts) {
    const { seedSwitches = [0], hoverIndex = -1 } = opts || {};
    const ctx = prepareCanvas(canvas);
    if (!ctx) return;

    const { width, height, dpr } = ctx.canvasMeta;
    const plotW = Math.max(10, width - PAD.left - PAD.right);
    const plotH = Math.max(10, height - PAD.top - PAD.bottom);

    // 空态提前返回: 先画网格再盖"待命"文字没有意义, 且会让空画布显得有内容。
    if (!Array.isArray(trace) || trace.length === 0) {
        drawStandby(ctx, width, height);
        return;
    }

    drawGrid(ctx, plotW, plotH);

    const maxErr = Math.max(SOLVER.TOL_MM * 8, Math.max(...trace) * 1.06);
    const xAt = (i) => PAD.left + (i / Math.max(1, trace.length - 1)) * plotW;
    const yAt = (v) => PAD.top + plotH - (clamp(v, 0, maxErr) / maxErr) * plotH;

    drawToleranceLine(ctx, yAt, plotW, plotH);
    drawAxisLabels(ctx, trace, maxErr, yAt, plotW, plotH);
    drawSeedDividers(ctx, seedSwitches, trace.length, xAt, plotH);
    drawAreaFill(ctx, trace, xAt, yAt, plotH);
    drawCurve(ctx, trace, xAt, yAt);
    drawEndpoints(ctx, trace, xAt, yAt);
    if (hoverIndex >= 0 && hoverIndex < trace.length) {
        drawHoverMarker(ctx, trace[hoverIndex], xAt(hoverIndex), yAt(trace[hoverIndex]), plotH);
    }

    void dpr;
}

/* ──────────────────────── 内部工具 ──────────────────────── */

function clamp(v, lo, hi) {
    return Math.max(lo, Math.min(hi, v));
}

/**
 * 按设备像素比调整画布位图尺寸并返回绘图上下文。
 *
 * 写入 canvas.width/height 会重置绘图状态并 (在某些布局场景下) 触发重排,
 * 因此先比对再写, 避免每帧无谓地赋值。
 *
 * @param {HTMLCanvasElement} canvas
 * @returns {CanvasRenderingContext2D|null}
 */
function prepareCanvas(canvas) {
    if (!canvas) return null;
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    const width = rect.width || 320;
    const height = rect.height || 110;

    const bw = Math.round(width * dpr);
    const bh = Math.round(height * dpr);
    if (canvas.width !== bw || canvas.height !== bh) {
        canvas.width = bw;
        canvas.height = bh;
    }

    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);
    ctx.canvasMeta = { width, height, dpr };
    return ctx;
}

function drawGrid(ctx, plotW, plotH) {
    ctx.strokeStyle = "rgba(255, 255, 255, 0.05)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (const f of [0, 0.5, 1]) {
        const y = PAD.top + plotH * f;
        ctx.moveTo(PAD.left, y);
        ctx.lineTo(PAD.left + plotW, y);
    }
    ctx.stroke();
}

/** 空态: 虚线框 + 居中提示, 让"暂无数据"读作一个明确的状态而非渲染失败 */
function drawStandby(ctx, width, height) {
    ctx.save();
    ctx.strokeStyle = "rgba(140, 165, 195, 0.14)";
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 4]);
    ctx.strokeRect(0.5, 0.5, width - 1, height - 1);
    ctx.setLineDash([]);

    ctx.fillStyle = "rgba(154, 168, 184, 0.75)";
    ctx.font = "10px 'JetBrains Mono', monospace";
    ctx.textAlign = "center";
    ctx.fillText("待命 · 下发 IK 指令后显示收敛曲线", width / 2, height / 2 + 3);
    ctx.restore();
}

function drawToleranceLine(ctx, yAt, plotW, plotH) {
    const tolY = yAt(SOLVER.TOL_MM);
    ctx.save();
    ctx.setLineDash([4, 3]);
    ctx.strokeStyle = "rgba(255, 184, 0, 0.7)";
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    ctx.moveTo(PAD.left, tolY);
    ctx.lineTo(PAD.left + plotW, tolY);
    ctx.stroke();
    ctx.restore();
}

function drawAxisLabels(ctx, trace, maxErr, yAt, plotW, plotH) {
    ctx.font = "9px 'JetBrains Mono', monospace";

    ctx.fillStyle = "rgba(255, 184, 0, 0.85)";
    ctx.textAlign = "right";
    ctx.fillText(`${SOLVER.TOL_MM}mm`, PAD.left - 4, yAt(SOLVER.TOL_MM) + 3);

    ctx.fillStyle = "rgba(140, 160, 180, 0.55)";
    ctx.fillText(`${Math.round(maxErr)}`, PAD.left - 4, PAD.top + 9);
    ctx.fillText("0", PAD.left - 4, PAD.top + plotH);

    ctx.textAlign = "left";
    ctx.fillText("it:1", PAD.left, PAD.top + plotH + PAD.bottom - 6);
    ctx.textAlign = "right";
    ctx.fillText(`it:${trace.length}`, PAD.left + plotW, PAD.top + plotH + PAD.bottom - 6);
}

function drawSeedDividers(ctx, switches, traceLen, xAt, plotH) {
    if (!Array.isArray(switches) || switches.length <= 1) return;
    ctx.save();
    ctx.setLineDash([2, 3]);
    switches.slice(1).forEach((idx, i) => {
        if (idx >= traceLen) return;
        const x = xAt(idx);
        ctx.strokeStyle = "rgba(0, 240, 255, 0.4)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(x, PAD.top);
        ctx.lineTo(x, PAD.top + plotH);
        ctx.stroke();

        ctx.fillStyle = "rgba(0, 240, 255, 0.75)";
        ctx.font = "8px 'JetBrains Mono', monospace";
        ctx.textAlign = "center";
        ctx.fillText(`S${i + 1}`, x, PAD.top - 4);
    });
    ctx.restore();
}

function drawAreaFill(ctx, trace, xAt, yAt, plotH) {
    const grad = ctx.createLinearGradient(0, PAD.top, 0, PAD.top + plotH);
    grad.addColorStop(0, "rgba(0, 240, 255, 0.22)");
    grad.addColorStop(0.7, "rgba(0, 255, 136, 0.08)");
    grad.addColorStop(1, "rgba(0, 255, 136, 0.01)");

    ctx.beginPath();
    ctx.moveTo(xAt(0), PAD.top + plotH);
    for (let i = 0; i < trace.length; i++) ctx.lineTo(xAt(i), yAt(trace[i]));
    ctx.lineTo(xAt(trace.length - 1), PAD.top + plotH);
    ctx.closePath();
    ctx.fillStyle = grad;
    ctx.fill();
}

function drawCurve(ctx, trace, xAt, yAt) {
    ctx.save();
    ctx.beginPath();
    trace.forEach((v, i) => {
        const x = xAt(i), y = yAt(v);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.strokeStyle = "#00f0ff";
    ctx.lineWidth = 2;
    ctx.shadowColor = "rgba(0, 240, 255, 0.6)";
    ctx.shadowBlur = 5;
    ctx.stroke();
    ctx.restore();
}

function drawEndpoints(ctx, trace, xAt, yAt) {
    dot(ctx, xAt(0), yAt(trace[0]), 2.5, "#00f0ff", 0);
    dot(ctx, xAt(trace.length - 1), yAt(trace[trace.length - 1]), 3.5, "#00ff88", 8);
}

function drawHoverMarker(ctx, value, x, y, plotH) {
    ctx.save();
    ctx.strokeStyle = "rgba(255, 255, 255, 0.4)";
    ctx.setLineDash([2, 2]);
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x, PAD.top);
    ctx.lineTo(x, PAD.top + plotH);
    ctx.stroke();
    ctx.restore();

    dot(ctx, x, y, 4.5, "#ffffff", 10);
}

function dot(ctx, x, y, r, color, blur) {
    ctx.save();
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fillStyle = color;
    if (blur) {
        ctx.shadowColor = color;
        ctx.shadowBlur = blur;
    }
    ctx.fill();
    ctx.restore();
}