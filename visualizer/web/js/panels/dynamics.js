/**
 * 质心与双足平衡动力学面板。
 *
 * 数据优先取 dynamics_telemetry (遥测侧完整计算), 回退到 solver_diagnostics
 * (求解时顺带计算的轻量版本), 兼容两种后端报文形态。
 */

import { el, setText, setClass, setStyle, setHTML, escapeHtml } from "../dom.js";
import { balanceStatus, BALANCE, fixed, signed, pct, toneByMagnitude, TORQUE } from "../format.js";

/**
 * 归一化两种后端报文形态。
 * @param {object} data
 * @returns {{com:number[]|null, marginMm:number, status:string, torques:object[], manip:object|null}}
 */
function normalize(data) {
    const dyn = data.dynamics_telemetry || {};
    const diag = data.solver_diagnostics || {};

    const bal = dyn.com_balance || (diag.com_pos ? {
        com_pos: diag.com_pos,
        margin_mm: diag.balance_margin_mm,
        status: diag.balance_status,
    } : null);

    return {
        com: bal ? bal.com_pos || diag.com_pos || null : null,
        marginMm: Number(bal ? bal.margin_mm : diag.balance_margin_mm),
        status: bal ? bal.status : diag.balance_status || null,
        torques: dyn.gravity_torques || [],
        manip: dyn.manipulability || diag.manipulability || null,
        peak: dyn.max_torque_joint || null,
    };
}

/**
 * 渲染动力学面板。
 * @param {object} data 遥测报文
 */
export function renderDynamics(data) {
    if (!data || typeof data !== "object") return;
    const n = normalize(data);

    if (n.com) {
        renderBalance(n.com, n.marginMm, n.status);
    }
    if (n.manip) renderManipulability(n.manip);
    if (n.peak) renderPeakTorque(n.peak);
    if (n.torques.length) renderTorqueList(n.torques);
}

function renderBalance(com, marginMm, rawStatus) {
    // 统一由裕度数值推导状态, 不直接信任后端给出的字符串
    const st = balanceStatus(marginMm);
    const tone = st.level === "ok" ? "safe" : st.level;

    // 顶栏芯片
    setText(el("comBalanceText"), `${st.status} (${fixed(marginMm, 1)}mm)`);
    setClass(el("comBalanceChip"), `status-chip chip-${tone}`);

    // 卡片
    setText(el("balanceBadge"), st.status);
    setClass(el("balanceBadge"), `balance-badge ${tone}`);
    // 三轴坐标压缩到两位小数且不留空格: 344px 侧栏的指标单元放不下
    // "+0.062, -0.010, +0.677 m" 这样的完整格式, 会截断掉第三轴高度值。
    // 精度损失对平衡评估无意义 —— 需要精确值时看 tooltip。
    const pos = [com[0], com[1], com[2]]
        .map((v) => (Number(v) >= 0 ? "+" : "") + (Number(v) || 0).toFixed(2))
        .join(" ");
    const posNode = el("comPosText");
    setText(posNode, pos);
    if (posNode) posNode.title = `${signed(com[0], 3)}, ${signed(com[1], 3)}, ${signed(com[2], 3)} m`;

    const marginNode = el("comMarginText");
    setText(marginNode, fixed(marginMm, 1));
    setClass(marginNode, `dyn-val-highlight font-mono val-${tone === "safe" ? "green" : tone}`);

    // 裕度刻度条
    const p = `${pct(marginMm, BALANCE.SCALE_MAX)}%`;
    setStyle(el("balanceFill"), "width", p);
    setStyle(el("balanceIndicator"), "left", p);
}

function renderManipulability(manip) {
    setText(el("manipulabilityVal"), fixed(manip.yoshikawa, 4));
    const node = el("singularityStatus");
    if (manip.is_singular) {
        setText(node, "⚠ 奇异点闭锁!");
        setStyle(node, "color", "var(--alert-red)");
    } else {
        setText(node, `非奇异 (σ_min: ${fixed(manip.min_singular_value, 2)})`);
        setStyle(node, "color", "var(--laser-green)");
    }
}

function renderPeakTorque(peak) {
    setText(el("peakTorqueVal"), `${signed(peak.torque_nm, 2)} N·m`);
    setText(el("peakTorqueJoint"), peak.name || "—");
}

/**
 * 重力补偿力矩列表 (折叠面板内)。
 * @param {object[]} torques
 */
function renderTorqueList(torques) {
    const maxAbs = Math.max(1.0, ...torques.map((t) => Math.abs(Number(t.torque_nm) || 0)));
    const rows = torques.map((t) => {
        const val = Number(t.torque_nm) || 0;
        const abs = Math.abs(val);
        // 力矩幅值是"越小越好", 用 toneByMagnitude 而非 severity
        const tone = toneByMagnitude(abs, TORQUE);
        const width = Math.min(100, Math.round((abs / maxAbs) * 100));
        const cls = tone === "red" ? "torque-bar-fill critical"
            : tone === "gold" ? "torque-bar-fill heavy"
            : "torque-bar-fill";
        const tip = `${t.name}: ${signed(val, 2)} N·m`;
        return `
            <div class="torque-item-row" title="${escapeHtml(tip)}">
                <span class="torque-label">${escapeHtml(t.label || t.name)}</span>
                <div class="torque-bar-container">
                    <div class="${cls}" style="width: ${width}%;"></div>
                </div>
                <span class="torque-val-text">${signed(val, 1)} N·m</span>
            </div>`;
    });
    setHTML(el("torquesList"), rows.join(""));
}