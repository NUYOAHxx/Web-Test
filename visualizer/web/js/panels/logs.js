/**
 * 测控事件日志流。
 *
 * 修复的缺陷: 旧实现有两条独立的渲染路径 ——
 *   · `addLocalEventLog()` 前端事件 (指令下发/切臂等) 用 prepend 插入单条;
 *   · `renderActivityLogs()` 服务端事件每帧用 innerHTML 整体重绘。
 * 两者作用于**同一个容器**且互相不知情, 结果是: 用户刚下发一条指令、
 * 本地日志刚插入, 下一帧遥测到达就被服务端日志的 innerHTML 整体覆盖,
 * 该条记录凭空消失。已下发过什么在界面上留不下痕迹。
 *
 * 现在改为单一事件源: 本地事件与服务端事件进入同一个存储,
 * 由单一渲染函数合并输出, 两者不再互相覆盖。
 */

import { el, setHTML, escapeHtml, setText } from "../dom.js";
import { state } from "../state.js";

/** 服务端日志 (遥测报文下发) */
let serverLogs = [];
/** 前端本地事件 (指令下发 / 切臂 / 复位等) */
const localLogs = [];
/** 本地事件保留上限 */
const MAX_LOCAL = 25;
/** 渲染总量上限 */
const MAX_RENDER = 40;

/**
 * 记录一条前端本地事件。
 * @param {string} type  事件类型 (COMMAND / ERROR / CONFIG ...)
 * @param {string} title 标题
 * @param {string} desc  详情
 */
export function pushLocalLog(type, title, desc) {
    localLogs.unshift({
        time: new Date().toTimeString().split(" ")[0],
        type: String(type || "INFO").toUpperCase(),
        title,
        desc,
    });
    if (localLogs.length > MAX_LOCAL) localLogs.pop();
    render();
}

/**
 * 同步服务端日志。报文未变化时跳过重绘。
 * @param {object[]} logs
 */
export function syncServerLogs(logs) {
    if (!Array.isArray(logs) || logs.length === 0) return;
    if (logs.length === state.logs.lastCount && logs[logs.length - 1]?.title === state.logs.lastStamp) {
        return;
    }
    serverLogs = logs;
    state.logs.lastCount = logs.length;
    state.logs.lastStamp = logs[logs.length - 1]?.title || "";
    render();
}

function render() {
    // 服务端日志本身是按时间正序的数组, 展示时倒序 (最新在上)
    const merged = [
        ...localLogs,
        ...serverLogs.slice().reverse(),
    ].slice(0, MAX_RENDER);

    setHTML(el("logStream"), merged.map((item) => `
        <div class="log-item">
            <span class="log-time">${escapeHtml(item.time)}</span>
            <span class="log-badge ${escapeHtml((item.type || "INFO").toLowerCase())}">${escapeHtml(item.type || "INFO")}</span>
            <div class="log-content">
                <div class="log-title">${escapeHtml(item.title)}</div>
                <div class="log-desc">${escapeHtml(item.desc)}</div>
            </div>
        </div>`).join(""));

    // 计数徽标由本模块自行维护 —— 让调用方 (main.js) 记得每次刷新计数,
    // 等于把"日志渲染"和"计数"拆成两处必须同步的逻辑, 迟早会漏掉一处。
    setText(el("logCountBadge"), String(merged.length));
}

/**
 * 当前展示的条目总数 (本地 + 服务端)。
 * @returns {number}
 */
export function logEntryCount() {
    return Math.min(MAX_RENDER, localLogs.length + serverLogs.length);
}

/**
 * 清空本地事件 (切换操作臂等场景下避免跨臂混杂)。
 */
export function clearLocalLogs() {
    localLogs.length = 0;
    render();
}