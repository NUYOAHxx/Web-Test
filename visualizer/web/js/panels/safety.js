/**
 * 碰撞安全监控面板。
 *
 * 关键设计: `clearanceStatus()` 在 format.js 中**只调用一次**,
 * 其结果同时驱动 大号数字 / 刻度标尺 / 状态徽章 / 顶栏安全芯片 / 碰撞告警框。
 *
 * 旧实现这四处各自独立判定且阈值互相矛盾 (10/22 与 5/20 混用),
 * 导致 8mm 间隙时界面显示 "数字标红 + 徽章 SECURE + 告警框隐藏"。
 */

import { el, setText, setClass, setHidden, setStyle, setHTML, escapeHtml } from "../dom.js";
import { clearanceStatus, CLEARANCE, fixed, pct } from "../format.js";

/** 分区卡片的键与展示名 */
const ZONES = [
    { id: "zoneTorso", key: "arm_torso", altKey: "torso_arm", name: "臂 – 躯干" },
    { id: "zoneArm", key: "arm_arm", altKey: null, name: "臂 – 臂" },
    { id: "zoneHead", key: "arm_head", altKey: "head_arm", name: "臂 – 头部" },
    { id: "zoneLeg", key: "arm_leg", altKey: "leg_arm", name: "臂 – 腿" },
];

/**
 * 渲染碰撞安全状态。
 * @param {object} data 遥测报文
 */
export function renderSafety(data) {
    // 面板自身保持防御性: main.js 虽有上游守卫, 但渲染函数被复用/测试时
    // 未必经过该路径。直接读 data.min_clearance_mm 会在 null 上抛异常,
    // 而这会发生在 30Hz 的遥测回调里 —— 一次抛错就中断整条渲染链。
    if (!data || typeof data !== "object") return;

    const clearance = data.min_clearance_mm;
    const colliding = Boolean(data.is_colliding);
    const pairs = data.colliding_pairs || [];

    // ← 全界面唯一的间隙安全判定
    const status = clearanceStatus(clearance, colliding);

    // 1. 大号数字与刻度标尺
    const valNode = el("minClearanceVal");
    if (valNode) {
        setText(valNode, fixed(clearance, 1));
        setClass(valNode, `clearance-val val-${toneName(status.level)}`);
        setStyle(valNode, "color", `var(--sev-${status.level})`);
        setStyle(valNode, "textShadow", `0 0 10px var(--sev-${status.level}-glow)`);
    }
    const marker = el("clearanceMarker");
    if (marker) {
        setStyle(marker, "left", `${pct(clearance, CLEARANCE.SCALE_MAX)}%`);
        setStyle(marker, "background", `var(--sev-${status.level})`);
        setStyle(marker, "boxShadow", `0 0 8px var(--sev-${status.level})`);
    }

    // 2. 状态徽章
    setText(el("radarBadge"), status.text);
    setClass(el("radarBadge"), `radar-status-badge ${toneName(status.level)}`);

    // 3. 顶栏全局安全芯片 (与徽章同源, 不会再出现说法不一)
    setText(el("globalCollisionText"), status.text);
    setClass(el("globalCollisionChip"), `status-chip chip-${toneName(status.level)}`);

    // 4. 碰撞告警清单
    const alertBox = el("collisionAlertBox");
    setHidden(alertBox, !colliding);
    if (colliding) {
        setHTML(el("alertPairsList"),
            pairs.map((p) => `<li>${escapeHtml(p[0])} ⚔ ${escapeHtml(p[1])} (${escapeHtml(p[2])})</li>`).join("")
        );
    }

    // 5. 四大区域切片
    const zones = data.zone_clearances || {};
    for (const z of ZONES) {
        const dist = zones[z.key] ?? (z.altKey ? zones[z.altKey] : undefined);
        renderZone(z.id, z.name, dist, status.level);
    }
}

/**
 * 单个分区卡片。
 * @param {string} id
 * @param {string} name
 * @param {number|undefined} distMm
 * @param {"alert"|"warn"|"ok"} globalLevel 全局等级, 分区数据缺失时兜底
 */
function renderZone(id, name, distMm, globalLevel) {
    const node = el(id);
    if (!node) return;

    const level = Number.isFinite(distMm)
        ? clearanceStatus(distMm, false).level
        : globalLevel;

    // 状态文字由卡片配色 (边框 + 底色) 表达, 这里只放数值。
    // 每格重复 "140.8mm · SECURE" 会让 4 张卡片产生冗余文本噪声,
    // 而真正的状态判定已在 clearanceStatus 中统一完成。
    const label = Number.isFinite(distMm) ? `${distMm.toFixed(1)}` : "—";

    setClass(node, `zone-card ${toneName(level)}`);
    setHTML(node,
        `<span class="zone-name">${escapeHtml(name)}</span>` +
        `<span class="zone-status">${escapeHtml(label)}<small>mm</small></span>`
    );
}

/**
 * 严重度 -> 徽章类名。
 * 顶层用 "safe" 而非 "ok", 与 CSS 既有类名保持一致。
 * @param {"alert"|"warn"|"ok"} level
 * @returns {string}
 */
function toneName(level) {
    return level === "ok" ? "safe" : level;
}