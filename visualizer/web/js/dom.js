/**
 * DOM 访问与写入原语。
 *
 * 存在理由 (针对旧 app.js 的实际缺陷):
 *
 *  1. 旧代码混用两套访问方式 —— 235 处 `dom.xxx` 缓存引用 + 96 处散落的
 *     `document.getElementById(...)`, 同一个元素在两处各取一次。
 *  2. 旧代码在 100 多个位置写了 `if (dom.foo) {...}` 形式的空值守卫。
 *     这些守卫掩盖真实问题: 元素缺失时页面静默半残, 而非明确报错。
 *     这里改为**启动时一次性集中校验**, 缺失立即抛出并列出全部缺失 id。
 *  3. 遥测以 30Hz 推送, 旧代码每帧无条件写 ~40 处 textContent,
 *     即每秒约 1200 次 DOM 写入, 其中绝大多数是写入相同值(纯浪费)。
 *     `setText` / `setClass` 在值未变化时直接跳过。
 */

/** id -> Element 缓存 (模块级单例, 页面生命周期内不变) */
const cache = new Map();

/** 已告警过的缺失 id, 避免 30Hz 下重复刷屏 */
const warned = new Set();

/**
 * 按 id 取元素并缓存。
 *
 * 查不到时告警一次: 遥测以 30Hz 反复调用渲染函数, 若静默返回 null,
 * 下游 setText 会安静地跳过写入, 最终表现为"某列数据永远停在占位符"
 * 且控制台毫无线索 —— 这类故障极难排查, 因此这里主动暴露。
 *
 * @param {string} id
 * @returns {HTMLElement|null}
 */
export function el(id) {
    if (cache.has(id)) return cache.get(id);
    const node = document.getElementById(id);
    cache.set(id, node);
    if (!node && !warned.has(id)) {
        warned.add(id);
        console.warn(`[DOM] 未找到元素 #${id} —— 相关面板可能不会更新。`);
    }
    return node;
}

/**
 * 批量取元素并在任一缺失时抛出聚合错误。
 * 在应用启动时调用一次, 把"元素缺失"从运行期的静默半残变成启动期的硬失败。
 *
 * @param {string[]} ids
 * @returns {Record<string, HTMLElement>}
 */
export function requireAll(ids) {
    const out = {};
    const missing = [];
    for (const id of ids) {
        const node = el(id);
        if (node) out[id] = node;
        else missing.push(id);
    }
    if (missing.length) {
        throw new Error(
            `[DOM] 页面结构不完整, 缺失 ${missing.length} 个必需元素: ${missing.join(", ")}\n` +
            `请确认 index.html 与前端模块版本一致。`
        );
    }
    return out;
}

/** 清空缓存 (仅测试用) */
export function _resetCache() {
    cache.clear();
    warned.clear();
}

/* ────────────────────────── 写入原语 ────────────────────────── */

/**
 * 写入文本, 值未变化时跳过 DOM 操作。
 * @param {HTMLElement|null} node
 * @param {string} value
 */
export function setText(node, value) {
    if (!node) return;
    const s = String(value);
    if (node.textContent !== s) node.textContent = s;
}

/**
 * 整体替换 className, 未变化时跳过。
 * @param {HTMLElement|null} node
 * @param {string} cls
 */
export function setClass(node, cls) {
    if (!node) return;
    if (node.className !== cls) node.className = cls;
}

/**
 * 替换 innerHTML, 未变化时跳过。
 * 传入的 HTML 必须已转义 (参见 escapeHtml)。
 * @param {HTMLElement|null} node
 * @param {string} html
 */
export function setHTML(node, html) {
    if (!node) return;
    if (node.innerHTML !== html) node.innerHTML = html;
}

/**
 * 切换 hidden 属性。统一走属性而非 style.display, 避免与样式表的
 * display 规则互相覆盖 (旧代码在 style.display 与 class 之间摇摆)。
 * @param {HTMLElement|null} node
 * @param {boolean} hidden
 */
export function setHidden(node, hidden) {
    if (!node) return;
    if (node.hidden !== hidden) node.hidden = hidden;
}

/**
 * 设置 CSS 样式属性, 值未变化时跳过。
 * @param {HTMLElement|null} node
 * @param {string} prop
 * @param {string} value
 */
export function setStyle(node, prop, value) {
    if (!node) return;
    if (node.style[prop] !== value) node.style[prop] = value;
}

/* ────────────────────────── 转义 ────────────────────────── */

/**
 * HTML 文本转义。
 *
 * 遥测报文中的关节名、日志标题、告警配对名均来自外部数据, 旧代码直接拼进
 * innerHTML 未做转义。这里统一转义后再拼接。
 *
 * @param {unknown} value
 * @returns {string}
 */
export function escapeHtml(value) {
    if (value === null || value === undefined) return "";
    return String(value)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#39;");
}