/**
 * 前端入口: 编排各模块, 并把一帧遥测报文分发给各面板。
 *
 * 与旧 app.js 的关键差别: 旧代码的 updateDashboardUI 是一个 288 行的巨函数,
 * 内部顺序执行"位姿表格 → 3D 标记 → CAD 模型 → 安全雷达 → 关节仪表 →
 * 动力学 → 日志 → 收敛曲线"共 8 组逻辑, 彼此耦合且无法单独测试。
 * 这里只保留**分发**, 具体的渲染逻辑全部下沉到各面板模块。
 */

import { el, requireAll, setText, setClass, setHidden } from "./dom.js";
import { state } from "./state.js";
import { fixed } from "./format.js";

import { connectTelemetry, disconnectTelemetry } from "./telemetry.js";

import {
    initCommandPanel,
    syncSlidersFromTelemetry,
    setDemoButtonState,
    resetStand,
    startDemo,
} from "./command.js";

import { initViewport, ViewportUnavailableError, resetCamera, toggleGrid, disposeViewport } from "./viewport/scene.js";
import { initOverlays, updatePoseMarkers, updateErrorLine, updateComMarkers, updateDemoOverlays } from "./viewport/overlays.js";
import { initRobotModel, updateVisuals, highlightLinks } from "./viewport/robot.js";

import { renderPose } from "./panels/pose.js";
import { renderSafety } from "./panels/safety.js";
import { renderJoints, renderJointPanelMeta } from "./panels/joints.js";
import { renderDynamics } from "./panels/dynamics.js";
import { renderSolver, initSolverPanel } from "./panels/solver.js";
import { syncServerLogs, pushLocalLog } from "./panels/logs.js";

/** 视口是否可用 (决定 3D 相关更新是否参与每帧分发) */
let viewportReady = false;

/** 页面元素完整性清单 —— 启动时一次性校验 */
const REQUIRED_IDS = [
    // 顶栏
    "connectionChip", "connectionStatus", "activeArmText",
    "globalCollisionChip", "globalCollisionText", "comBalanceChip", "comBalanceText",
    "btnCircleDemo", "btnResetStand",
    // 指令面板
    "btnSelectLeftArm", "btnSelectRightArm", "presetGrid",
    "sliderX", "sliderY", "sliderZ", "sliderRoll", "sliderPitch", "sliderYaw",
    "numX", "numY", "numZ", "numRoll", "numPitch", "numYaw",
    "btnDispatchTarget",
    // 位姿面板
    "tgtX", "tgtY", "tgtZ", "actX", "actY", "actZ",
    "deltaX", "deltaY", "deltaZ",
    "tgtRoll", "tgtPitch", "tgtYaw", "actRoll", "actPitch", "actYaw",
    "deltaRoll", "deltaPitch", "deltaYaw",
    "posErrNum", "rotErrNum", "streamHzNum", "packetCountText",
    // 视口
    "threeCanvasContainer", "viewportFallback", "btnResetCamera", "btnToggleGrid",
    "viewportMeshCount", "vpCollisionBadge", "vpCollisionCount",
    // 求解器
    "btnTraceCurve", "btnTraceSteps", "traceStepsBadge", "pipelineStepper",
    "convergenceModeBadge", "traceTimeVal", "traceItersVal", "traceSeedVal", "traceResidueVal",
    "traceCurveView", "traceStepsView", "convergenceCanvas", "traceTooltip",
    "traceStatusTag", "traceStepsBody", "stepsSummaryInfo", "traceStepsStatusTag",
    // 安全
    "minClearanceVal", "clearanceMarker", "radarBadge",
    "collisionAlertBox", "alertPairsList",
    "zoneTorso", "zoneArm", "zoneHead", "zoneLeg",
    // 关节
    "waistJointsList", "armJointsList", "armDofBadge", "waistDofLabel", "armSubLabel",
    // 动力学
    "balanceBadge", "comPosText", "comMarginText", "balanceFill", "balanceIndicator",
    "manipulabilityVal", "singularityStatus", "peakTorqueVal", "peakTorqueJoint",
    "torquesList",
    // 日志
    "logStream", "cardActivity", "logCountBadge",
];

/* ══════════════════════════ 启动 ══════════════════════════ */

function bootstrap() {
    try {
        requireAll(REQUIRED_IDS);
    } catch (err) {
        // 页面结构不完整时无法继续, 直接给出明确错误而不是半残运行
        console.error(err.message);
        showFatal(err.message);
        return;
    }

    initViewportOrFallback();
    initCommandPanel();
    initSolverPanel();
    bindGlobalActions();

    connectTelemetry(onTelemetryFrame, onLinkChange);

    window.addEventListener("beforeunload", () => {
        disconnectTelemetry();
        disposeViewport();
    });
}

/**
 * 初始化 3D 视口, 失败时降级但不影响纯数据面板。
 */
function initViewportOrFallback() {
    const container = el("threeCanvasContainer");
    try {
        initViewport(container);
        initOverlays();
        initRobotModel(container);
        viewportReady = true;
    } catch (err) {
        viewportReady = false;
        const reason = err instanceof ViewportUnavailableError
            ? err.message
            : "3D 场景初始化失败";
        console.warn(`[3D] 已降级为纯遥测模式: ${reason}`);
        showViewportFallback(reason);
    }
}

/**
 * 视口不可用提示。3D 停用, 其余面板照常工作。
 * @param {string} reason
 */
function showViewportFallback(reason) {
    const box = el("viewportFallback");
    setHidden(box, false);
    box.classList.add("is-active");
    const hint = el("viewportFallbackHint");
    if (hint) {
        hint.innerHTML =
            `${reason}, 3D 渲染已停用。<br />` +
            `遥测 / 关节 / 碰撞等全部数据面板仍正常工作。<br />` +
            `可尝试启用浏览器硬件加速后刷新页面。`;
    }
}

/**
 * 启动期致命错误提示。
 * @param {string} message
 */
function showFatal(message) {
    document.body.insertAdjacentHTML("afterbegin",
        `<div class="fatal-banner">前端启动失败 · ${message}</div>`);
}

function bindGlobalActions() {
    el("btnCircleDemo").addEventListener("click", startDemo);
    el("btnResetStand").addEventListener("click", resetStand);
    el("btnResetCamera").addEventListener("click", resetCamera);

    el("btnToggleGrid").addEventListener("click", () => {
        const visible = toggleGrid();
        pushLocalLog("VIEW", visible ? "显示地面网格" : "隐藏地面网格", "视口显示项切换");
    });

    // 重力力矩面板使用原生 <details>, HTML 初始为 open, 箭头须与之一致
    const drawer = el("torquesDrawer");
    if (drawer) {
        const syncArrow = () => setText(el("arrowTorques"), drawer.open ? "▲" : "▼");
        drawer.addEventListener("toggle", syncArrow);
        syncArrow();
    }
}

/* ══════════════════════ 遥测帧分发 ══════════════════════ */

/**
 * 链路状态变化。
 * @param {boolean} online
 */
function onLinkChange(online) {
    setText(el("connectionStatus"), online ? "DDS LINK: LIVE" : "DDS LINK: RECONNECTING");
    setClass(el("connectionChip"), `status-chip ${online ? "chip-online" : "chip-offline"}`);
}

/**
 * 一帧遥测报文: 分发给各面板。
 * @param {object} data
 */
function onTelemetryFrame(data) {
    if (!data) return;

    const cmd = data.cartesian_cmd_pose;
    const act = data.cartesian_actual_pose;
    const errMm = Number(data.euclidean_error_norm_mm) || 0;

    // 1. 位姿面板 (顺带回填指令滑块, 保持"面板所见 = 机器人所受")
    const poseResult = renderPose(data);
    syncSlidersFromTelemetry(poseResult.cmd, poseResult.cmdRpy);

    // 2. 安全监控 (间隙/碰撞的单一判定源)
    renderSafety(data);

    // 3. 关节仪表
    const waist = data.waist_telemetry || [];
    const arm = data.arm_telemetry || [];
    renderJoints(el("waistJointsList"), waist);
    renderJoints(el("armJointsList"), arm);
    renderJointPanelMeta(waist.length, arm.length, data.arm === "right_arm" ? "右臂" : "左臂");

    // 4. 动力学
    renderDynamics(data);

    // 5. 求解器诊断
    if (data.solver_diagnostics) renderSolver(data.solver_diagnostics);

    // 6. 日志流
    syncServerLogs(data.activity_logs);

    // 7. 3D 视口 (视口不可用时全部跳过)
    if (viewportReady) updateViewport(data, cmd, act, errMm);

    // 8. 演示按钮运行态
    const demoActive = Boolean(data.solver_diagnostics?.trajectory_active);
    setDemoButtonState(demoActive);
}

/**
 * 视口更新。仅在视口可用时调用。
 * @param {object} data
 * @param {number[]} cmd
 * @param {number[]} act
 * @param {number} errMm
 */
function updateViewport(data, cmd, act, errMm) {
    updatePoseMarkers(cmd, data.cartesian_cmd_quat, act, data.cartesian_actual_quat);
    updateErrorLine(cmd, act, errMm);

    const dyn = data.dynamics_telemetry || {};
    const bal = dyn.com_balance || data.solver_diagnostics || {};
    updateComMarkers(
        bal.com_pos || data.solver_diagnostics?.com_pos,
        Number(bal.margin_mm ?? data.solver_diagnostics?.balance_margin_mm) || 0,
        Boolean(data.is_colliding)
    );

    updateDemoOverlays(
        Boolean(data.solver_diagnostics?.trajectory_active),
        data.arm || state.arm,
        act
    );

    if (data.visuals) {
        updateVisuals(data.visuals);
        setText(el("viewportMeshCount"), `${state.meshCount}/36`);
    }

    // 碰撞零件在模型上标红。
    // 安全告警只写在侧栏列表里是不够的 —— 用户要能在 3D 模型上直接看到
    // 到底是哪两块零件在干涉, 否则必须自己在列表与模型之间来回对照。
    const hitLinks = [];
    for (const pair of data.colliding_pairs || []) {
        if (pair[0]) hitLinks.push(pair[0]);
        if (pair[1]) hitLinks.push(pair[1]);
    }
    highlightLinks(hitLinks);

    // 视口角标: 碰撞发生时在 3D 视图上给出计数提示 (零件已标红, 此处给数量)
    const nPairs = (data.colliding_pairs || []).length;
    setHidden(el("vpCollisionBadge"), nPairs === 0);
    if (nPairs) setText(el("vpCollisionCount"), String(nPairs));
}

/* ══════════════════════════ 引导 ══════════════════════════ */

if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", bootstrap);
} else {
    bootstrap();
}

// 导出固定值供调试使用
export { fixed };