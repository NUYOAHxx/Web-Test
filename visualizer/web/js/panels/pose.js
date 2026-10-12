/**
 * 空间位姿遥测面板: 6 轴 (XYZ + RPY) 的 目标 / 实测 / 残差 三列对照。
 */

import { el, setText } from "../dom.js";
import { at, fixed, signed, deg, mm, POS_ERR, ROT_ERR, toneByMagnitude } from "../format.js";

/**
 * 位置轴与姿态轴的 DOM id 映射。
 *
 * 显式列出 id 而非由轴名拼接 (如 `tgt${axis}`): 拼接会因大小写不一致
 * 产生静默失效的查询, 而 el() 缓存 null + setText 跳过写入,
 * 表现为该列永远停留在占位符且控制台无任何报错。
 */
const POS_AXES = [
    { ids: ["tgtX", "actX", "deltaX"] },
    { ids: ["tgtY", "actY", "deltaY"] },
    { ids: ["tgtZ", "actZ", "deltaZ"] },
];
const ROT_AXES = [
    { ids: ["tgtRoll", "actRoll", "deltaRoll"], index: 0 },
    { ids: ["tgtPitch", "actPitch", "deltaPitch"], index: 1 },
    { ids: ["tgtYaw", "actYaw", "deltaYaw"], index: 2 },
];

/**
 * 渲染一帧位姿遥测。
 *
 * @param {object} data 遥测报文
 * @param {HTMLElement|null} syncHook 供指令面板回填滑块的回调容器 (避免循环依赖)
 */
export function renderPose(data) {
    // 与 safety.js 同理: 渲染函数必须自身防御, 不能依赖调用方先判空
    if (!data || typeof data !== "object") return { cmd: null, cmdRpy: null };

    const cmd = data.cartesian_cmd_pose;
    const act = data.cartesian_actual_pose;
    const dPos = data.spatial_delta_mm;
    const dRot = data.spatial_delta_rpy_deg;
    const cmdRpy = data.cartesian_cmd_rpy_deg;
    const actRpy = data.cartesian_actual_rpy_deg;

    // 位置三轴
    POS_AXES.forEach((axis, i) => {
        setText(el(axis.ids[0]), formatSignedOrDash(at(cmd, i), 3));
        setText(el(axis.ids[1]), formatSignedOrDash(at(act, i), 3));
        setText(el(axis.ids[2]), formatDeltaOrDash(at(dPos, i), 1, " mm"));
    });

    // 姿态三轴
    for (const axis of ROT_AXES) {
        const k = axis.index;
        setText(el(axis.ids[0]), formatAngleOrDash(at(cmdRpy, k)));
        setText(el(axis.ids[1]), formatAngleOrDash(at(actRpy, k)));
        setText(el(axis.ids[2]), formatDeltaOrDash(at(dRot, k), 1, "°"));
    }

    renderErrorSummary(data);
    return { cmd, cmdRpy };
}

function renderErrorSummary(data) {
    const posErr = data.euclidean_error_norm_mm;
    const rotErr = data.rot_error_norm_deg;

    // 残差是"越小越好"的量, 必须用 toneByMagnitude。
    // 曾误用 severity()(语义为"低值 = 坏"), 导致 0.00mm 的完美收敛被标红。
    const posNode = el("posErrNum");
    if (posNode) {
        setText(posNode, fixed(posErr, 2));
        posNode.className = `metric-val val-${toneByMagnitude(posErr, POS_ERR)}`;
    }

    const rotNode = el("rotErrNum");
    if (rotNode) {
        setText(rotNode, fixed(rotErr, 2));
        rotNode.className = `metric-val val-${toneByMagnitude(rotErr, ROT_ERR)}`;
    }

    const link = data.telemetry_link || {};
    setText(el("streamHzNum"), fixed(link.dds_rate_hz, 1));
    setText(el("packetCountText"), String(link.ingress_packets ?? 0));
}

function formatSignedOrDash(v, digits) {
    return v === null ? "—" : signed(v, digits);
}

function formatAngleOrDash(v) {
    return v === null ? "—" : deg(v, 1);
}

function formatDeltaOrDash(v, digits, unit) {
    return v === null ? "—" : `${signed(v, digits)}${unit}`;
}