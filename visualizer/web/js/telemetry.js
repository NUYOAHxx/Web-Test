/**
 * 遥测推流连接管理 (SSE)。
 *
 * 断线重连策略: 浏览器 EventSource 自带快速重试, 直接放任会在服务端重启期间
 * 打出每秒数十次的连接风暴。这里显式 close() 后按固定间隔重建, 并保证同一时刻
 * 最多存在一条连接。
 */

/** 重连间隔 (ms) */
const RECONNECT_DELAY = 2000;

/** @type {EventSource|null} */
let source = null;
/** @type {number|null} */
let reconnectTimer = null;
/** @type {(data:object)=>void} */
let onFrame = () => {};
/** @type {(online:boolean)=>void} */
let onLinkChange = () => {};

/**
 * 本地累积的完整遥测状态。
 *
 * 服务端为省带宽采用**增量推送**: 慢变量 (事件日志、收敛轨迹、连杆位姿)
 * 未发生变化时不在帧中出现。因此这里必须逐字段合并 —— 若直接用收到的帧
 * 覆盖, 未变化的那三类数据会变成 undefined, 对应面板随即空白。
 */
let accumulated = {};

/**
 * 把增量帧合并进累积状态。
 * solver_diagnostics 是嵌套对象, 需单独浅合并, 否则它内部的慢字段
 * (convergence_trace / step_details) 缺失会连带丢掉同帧的标量更新。
 *
 * @param {object} delta 服务端增量帧
 * @returns {object} 合并后的完整状态
 */
function mergeFrame(delta) {
    // 必须在浅拷贝**之前**取出上一轮的嵌套对象。
    //
    // `Object.assign({}, accumulated, delta)` 遇到同名的 solver_diagnostics 会
    // 直接用 delta 的那份**整体替换**掉, 于是紧接着的"合并"实际是拿 delta 和它
    // 自己合并, 上一轮的累积结果已被丢弃。表现为: 顶层字段 (事件日志) 正常刷新,
    // 而嵌套的收敛轨迹与逐步明细永远是空的 —— 增量推送下必现。
    const prevDiagnostics = accumulated.solver_diagnostics;

    accumulated = Object.assign({}, accumulated, delta);

    if (delta.solver_diagnostics) {
        accumulated.solver_diagnostics = Object.assign(
            {}, prevDiagnostics, delta.solver_diagnostics
        );
    }
    return accumulated;
}

/**
 * 启动遥测连接。
 * @param {(data:object)=>void} frameHandler 每帧**完整**状态回调
 * @param {(online:boolean)=>void} linkHandler 连接状态变化回调
 */
export function connectTelemetry(frameHandler, linkHandler) {
    onFrame = frameHandler;
    onLinkChange = linkHandler;
    open();
}

function open() {
    teardown();
    // 重连后服务端会重发一份完整首帧, 本地累积态作废, 避免混入旧数据
    accumulated = {};

    const es = new EventSource("/api/stream");
    source = es;

    es.onopen = () => onLinkChange(true);

    es.onmessage = (event) => {
        try {
            onFrame(mergeFrame(JSON.parse(event.data)));
        } catch (err) {
            console.error("[遥测] 报文解析失败", err);
        }
    };

    es.onerror = () => {
        onLinkChange(false);
        // 关闭后由定时器统一重建, 避免 EventSource 内置的快速重试风暴
        es.close();
        source = null;
        if (reconnectTimer === null) {
            reconnectTimer = setTimeout(() => {
                reconnectTimer = null;
                open();
            }, RECONNECT_DELAY);
        }
    };
}

function teardown() {
    if (source) {
        source.close();
        source = null;
    }
    if (reconnectTimer !== null) {
        clearTimeout(reconnectTimer);
        reconnectTimer = null;
    }
}

/** 主动断开 (页面卸载时) */
export function disconnectTelemetry() {
    teardown();
    onLinkChange(false);
}