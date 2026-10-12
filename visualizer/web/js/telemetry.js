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
 * 启动遥测连接。
 * @param {(data:object)=>void} frameHandler 每帧报文回调
 * @param {(online:boolean)=>void} linkHandler 连接状态变化回调
 */
export function connectTelemetry(frameHandler, linkHandler) {
    onFrame = frameHandler;
    onLinkChange = linkHandler;
    open();
}

function open() {
    teardown();

    const es = new EventSource("/api/stream");
    source = es;

    es.onopen = () => onLinkChange(true);

    es.onmessage = (event) => {
        try {
            onFrame(JSON.parse(event.data));
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