/**
 * 关节限位遥测仪表 (双向中心对称条)。
 *
 * 渲染策略: 每个关节只创建一次 DOM, 之后仅更新变化的字段。
 *
 * 旧实现虽然做了节点复用, 但把子元素句柄挂在 DOM 节点的自定义属性上
 * (card._labelTitle = ...), 这既污染了元素对象, 又在列表长度变化时
 * 无法回收。这里改为模块内 Map 显式持有句柄。
 */

import { el, setText, setClass, setStyle } from "../dom.js";
import { signed, severity, JOINT_MARGIN } from "../format.js";

/** 容器元素 -> (关节索引 -> 该关节的 DOM 句柄) */
const cardCache = new WeakMap();

/**
 * 渲染一组关节仪表。
 * @param {HTMLElement|null} container
 * @param {object[]} joints 关节遥测数组
 */
export function renderJoints(container, joints) {
    if (!container || !Array.isArray(joints)) return;

    let handles = cardCache.get(container);
    if (!handles) {
        handles = [];
        cardCache.set(container, handles);
    }

    // 关节数变化时补建 / 移除多余卡片
    while (handles.length < joints.length) {
        handles.push(createCard(container, handles.length));
    }
    while (handles.length > joints.length) {
        const card = handles.pop();
        card.root.remove();
    }

    joints.forEach((j, i) => updateCard(handles[i], j));
}

/**
 * 创建一个关节行并返回其内部句柄。
 *
 * 采用**单行紧凑布局** (标签 | 双向条 | 角度 | 余量) 而非多行卡片。
 *
 * 旧布局每关节 58px, 9 个关节占 673px, 直接把右栏的"质心与平衡"面板
 * 顶出视口 350px (实测 top=990px vs 栏高 636px) —— 关键安全信息需要滚动才可见。
 * 紧凑行降至 26px, 9 个关节 234px, 三块面板得以同屏可见。
 *
 * @param {HTMLElement} container
 */
function createCard(container) {
    const root = document.createElement("div");
    root.className = "joint-row";
    root.innerHTML = `
        <span class="joint-name"></span>
        <div class="bidi-bar-track">
            <div class="bidi-center-marker"></div>
            <div class="bidi-bar-fill"></div>
        </div>
        <span class="joint-deg"></span>
        <span class="joint-margin"></span>
    `;
    container.appendChild(root);

    return {
        root,
        name: root.querySelector(".joint-name"),
        deg: root.querySelector(".joint-deg"),
        margin: root.querySelector(".joint-margin"),
        barFill: root.querySelector(".bidi-bar-fill"),
    };
}

/**
 * 更新单个关节行。
 * @param {object} h createCard 返回的句柄
 * @param {object} j 关节遥测对象
 */
function updateCard(h, j) {
    const deg = Number(j.deg) || 0;
    const offsetPct = Number(j.offset_pct) || 0;

    // 距物理限位的最小余量
    const margin = Math.min(
        Math.abs(deg - (Number(j.min_deg) || 0)),
        Math.abs((Number(j.max_deg) || 0) - deg)
    );
    const level = severity(margin, JOINT_MARGIN);

    // ── 整行告警态 (用左侧色条而非整块染色, 避免 9 行里某一行过抢眼) ──
    h.root.dataset.level = level;

    // ── 名称 ──
    // 只显示标签本体。旧布局追加 " [q_w0]" 符号后, 形如 "waist_pitch q_w2"
    // 的字符串在 344px 栏内被截断成 "waist_pitch q_" —— 符号与名称本就是
    // 同一关节的两种写法, 并列显示没有信息增量。完整写法移入 tooltip。
    const name = j.label || j.name || "";
    const full = j.symbol ? `${name} (${j.symbol})` : name;
    setText(h.name, name);
    if (h.name.title !== full) h.name.title = full;
    h.root.title = `${full} · 限位 ${(Number(j.min_deg) || 0).toFixed(0)}° ~ ${(Number(j.max_deg) || 0).toFixed(0)}°`;

    // ── 数值 ──
    setClass(h.deg, `joint-deg text-${level}`);
    setText(h.deg, `${signed(deg, 1)}°`);

    // ── 余量 (接近限位时才值得占用视觉权重) ──
    setClass(h.margin, `joint-margin text-${level}`);
    setText(h.margin, level === "ok" ? `${margin.toFixed(0)}°` : `⚠ ${margin.toFixed(1)}°`);

    // ── 双向条: 中心为零点, 正负分别向两侧伸展 (0~100% 映射到半边 0~50%) ──
    const halfWidth = Math.min(50, Math.abs(offsetPct) * 0.5);
    const isPositive = offsetPct >= 0;
    setClass(h.barFill, `bidi-bar-fill ${isPositive ? "positive" : "negative"} lv-${level}`);
    setStyle(h.barFill, "left", isPositive ? "50%" : `${50 - halfWidth}%`);
    setStyle(h.barFill, "width", `${halfWidth}%`);
}

/**
 * 更新关节面板的副标题 (DoF 数量与当前臂)。
 * @param {number} waistCount
 * @param {number} armCount
 * @param {string} armLabel
 */
export function renderJointPanelMeta(waistCount, armCount, armLabel) {
    setText(el("waistDofLabel"), `${waistCount}-DoF`);
    setText(el("armSubLabel"), `${armCount}-DoF`);
    setText(el("armDofBadge"), `${waistCount + armCount}-DoF · ${armLabel}`);
}