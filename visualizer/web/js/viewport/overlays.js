/**
 * 视口标记物: 目标球 / 末端实测球 / 残差连线 / 质心系统 / 支撑多边形 / 演示导轨 / 运动拖尾。
 *
 * 这些是"随遥测数据变化的可视化元素", 与 scene.js 的渲染基础设施分离。
 * 每个标记物提供独立的 update 方法, 主循环只负责派发数据。
 */

import { addToScene, onFrame } from "./scene.js";
import { clearanceStatus } from "../format.js";

/* 标记物句柄 */
let targetOrb = null;
let targetRing = null;
let actualOrb = null;
let errorLine = null;
let comOrb = null;
let comDropLine = null;
let comFloorDisc = null;
let supportPolygon = null;
let orbitRing = null;
let motionTrail = null;

/** 目标球上一次是否处于"越界"状态 —— 用于只在跨过边界时改色, 避免每帧写材质 */
let _targetWasOutside = false;

/** 拖尾历史点 (运动轨迹流光) */
const trailHistory = [];
const MAX_TRAIL = 80;

/** 演示轨迹导轨的分段数 */
const ORBIT_SEGMENTS = 64;

/** 质心状态色 */
const COM_COLORS = { ok: 0x00f0ff, warn: 0xffb800, alert: 0xff3366 };

/**
 * 构建全部标记物并加入场景。
 */
export function initOverlays() {
    initTargetOrb();
    initActualOrb();
    initErrorLine();
    initComSystem();
    initSupportPolygon();
    initOrbitRing();
    initMotionTrail();
    initReachBox();
    onFrame(animateOverlays);
}

/* ═════════════════════ 可达工作空间参照盒 ═════════════════════ */

/**
 * 线框盒: 标出该臂末端**实测可达**的工作空间范围。
 *
 * 最初的想法是画一个以世界原点为中心的正方体当距离参照。但任意大小的正方体
 * 没有物理含义 —— 1m 边长凭什么? 画小了显得手够不到, 画大了又失去参考价值。
 *
 * 改用服务端在启动时实测的可达包围盒 (model_info.reach_envelope, 与指令可达性
 * 判定同源), 于是这个盒子直接就是"手能够到的范围":
 *   - 目标球落在盒内 → 大概率收敛
 *   - 目标球落在盒外 → 画面上一眼可见地出界, 不必等看日志才发现
 * 这把"可达性"从一行提示文字变成了空间直觉。
 *
 * 注意包围盒是可达集的**超集**: 9-DoF 臂的可达区非凸, 盒内仍可能有够不着的
 * 角落。因此它只作参照, 不作判据 —— 判定仍由 KD 树完成。
 */
let reachBox = null;
let reachBoxSignature = "";

/** @type {{min:[number,number,number], max:[number,number,number], arm:string}|null} */
let reachBoxBounds = null;

function initReachBox() {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(24 * 3), 3));
    reachBox = addToScene(new THREE.LineSegments(
        geo,
        new THREE.LineBasicMaterial({
            color: 0x00c8e0, transparent: true, opacity: 0.34, depthWrite: false,
        })
    ));
    reachBox.renderOrder = 998;      // 压在机身之下, 不遮挡关节观察
    reachBox.frustumCulled = false;
    // 默认显示 —— 这正是用户要的距离参照。收到 model_info 前盒体尺寸为零,
    // 几何体退化成一个点, 不会造成可见干扰。
    reachBox.visible = true;
}

const _cornerIndex = [
    [0, 1], [1, 3], [3, 2], [2, 0],   // 底面 4 边
    [4, 5], [5, 7], [7, 6], [6, 4],   // 顶面 4 边
    [0, 4], [1, 5], [2, 6], [3, 7],   // 4 条竖边
];

/**
 * 按实测包围盒重绘线框。
 *
 * @param {object} envelope model_info.reach_envelope
 * @param {string} arm 当前操作臂
 * @returns {boolean} 盒��是否发生变化
 */
export function updateReachBox(envelope, arm) {
    if (!reachBox || !envelope) return false;
    const env = envelope[arm] || envelope.left_arm;
    if (!env || !env.x || !env.y || !env.z) return false;

    const sig = `${arm}|${env.x.min},${env.x.max},${env.y.min},${env.y.max},${env.z.min},${env.z.max}`;
    if (sig === reachBoxSignature) return false;
    reachBoxSignature = sig;

    const lo = [env.x.min, env.y.min, env.z.min];
    const hi = [env.x.max, env.y.max, env.z.max];
    const c = [
        [lo[0], lo[1], lo[2]], [hi[0], lo[1], lo[2]], [hi[0], hi[1], lo[2]], [lo[0], hi[1], lo[2]],
        [lo[0], lo[1], hi[2]], [hi[0], lo[1], hi[2]], [hi[0], hi[1], hi[2]], [lo[0], hi[1], hi[2]],
    ];
    const pos = reachBox.geometry.getAttribute("position");
    let k = 0;
    for (const [a, b] of _cornerIndex) {
        pos.setXYZ(k++, c[a][0], c[a][1], c[a][2]);
        pos.setXYZ(k++, c[b][0], c[b][1], c[b][2]);
    }
    pos.needsUpdate = true;
    reachBox.geometry.computeBoundingSphere();

    reachBoxBounds = { min: lo, max: hi, arm };
    return true;
}

/** 切换线框盒显隐 @returns {boolean} 切换后是否可见 */
export function toggleReachBox() {
    if (!reachBox) return false;
    reachBox.visible = !reachBox.visible;
    return reachBox.visible;
}

/** @returns {boolean} 线框盒当前是否可见 */
export function isReachBoxVisible() {
    return !!reachBox?.visible;
}

/** @returns {string|null} 线框盒尺寸描述 (米), 未就绪时为 null */
export function reachBoxSizeText() {
    if (!reachBoxBounds) return null;
    const { min, max } = reachBoxBounds;
    return `${(max[0] - min[0]).toFixed(2)} × ${(max[1] - min[1]).toFixed(2)} × ${(max[2] - min[2]).toFixed(2)} m`;
}

/**
 * 目标是否落在可达盒之外 —— 用于把目标球染成告警色。
 *
 * 越界只改**颜色**, 不改指令流: 大屏用于观察, 用户应能自行决定是否下发。
 *
 * @param {number[]} pos 目标世界坐标
 * @returns {boolean}
 */
export function isTargetOutsideReachBox(pos) {
    if (!reachBoxBounds || !Array.isArray(pos) || pos.length < 3) return false;
    const { min, max } = reachBoxBounds;
    for (let i = 0; i < 3; i++) {
        if (!(pos[i] >= min[i] && pos[i] <= max[i])) return true;
    }
    return false;
}

function initTargetOrb() {
    targetOrb = addToScene(new THREE.Mesh(
        new THREE.SphereGeometry(0.018, 24, 24),
        new THREE.MeshStandardMaterial({
            color: 0x00ff88, emissive: 0x00ff88, emissiveIntensity: 0.9, roughness: 0.2,
        })
    ));

    targetRing = new THREE.Mesh(
        new THREE.RingGeometry(0.024, 0.030, 32),
        new THREE.MeshBasicMaterial({ color: 0x00ff88, side: THREE.DoubleSide, transparent: true, opacity: 0.75 })
    );
    targetOrb.add(targetRing);

    const axes = new THREE.AxesHelper(0.05);
    axes.renderOrder = 999;
    targetOrb.add(axes);
}

function initActualOrb() {
    actualOrb = addToScene(new THREE.Mesh(
        new THREE.SphereGeometry(0.014, 24, 24),
        new THREE.MeshStandardMaterial({
            color: 0xffb800, emissive: 0xffb800, emissiveIntensity: 0.85, roughness: 0.25,
        })
    ));

    const axes = new THREE.AxesHelper(0.045);
    axes.renderOrder = 999;
    actualOrb.add(axes);
}

function initErrorLine() {
    errorLine = addToScene(new THREE.Line(
        new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]),
        new THREE.LineBasicMaterial({ color: 0xff3366, linewidth: 2, transparent: true, opacity: 0.85 })
    ));
    errorLine.visible = false;
}

function initComSystem() {
    comOrb = addToScene(new THREE.Mesh(
        new THREE.SphereGeometry(0.024, 24, 24),
        new THREE.MeshStandardMaterial({
            color: 0x00f0ff, emissive: 0x00f0ff, emissiveIntensity: 0.9, roughness: 0.2,
        })
    ));

    comDropLine = addToScene(new THREE.Line(
        new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]),
        new THREE.LineDashedMaterial({
            color: 0x00f0ff, dashSize: 0.02, gapSize: 0.015,
            linewidth: 2, transparent: true, opacity: 0.75,
        })
    ));
    comDropLine.computeLineDistances();

    comFloorDisc = addToScene(new THREE.Mesh(
        new THREE.RingGeometry(0.012, 0.032, 32),
        new THREE.MeshBasicMaterial({ color: 0x00f0ff, side: THREE.DoubleSide, transparent: true, opacity: 0.65 })
    ));
}

function initSupportPolygon() {
    // 双足地面支撑多边形线框 (沿机器人 X/Y 边界)
    const xs = [-0.09, 0.13];
    const ys = [-0.17, 0.17];
    const z = 0.002;
    const points = [
        new THREE.Vector3(xs[0], ys[0], z),
        new THREE.Vector3(xs[1], ys[0], z),
        new THREE.Vector3(xs[1], ys[1], z),
        new THREE.Vector3(xs[0], ys[1], z),
        new THREE.Vector3(xs[0], ys[0], z),
    ];
    supportPolygon = addToScene(new THREE.Line(
        new THREE.BufferGeometry().setFromPoints(points),
        new THREE.LineBasicMaterial({ color: 0x00ff88, linewidth: 2, transparent: true, opacity: 0.85 })
    ));
}

function initOrbitRing() {
    const points = [];
    for (let i = 0; i <= ORBIT_SEGMENTS; i++) {
        const th = (i / ORBIT_SEGMENTS) * Math.PI * 2;
        points.push(new THREE.Vector3(0.38 + 0.10 * Math.cos(th), 0.22, 0.82 + 0.10 * Math.sin(th)));
    }
    orbitRing = addToScene(new THREE.Line(
        new THREE.BufferGeometry().setFromPoints(points),
        new THREE.LineDashedMaterial({
            color: 0x00f0ff, dashSize: 0.015, gapSize: 0.01,
            linewidth: 2, transparent: true, opacity: 0.8,
        })
    ));
    orbitRing.computeLineDistances();
    orbitRing.visible = false;
}

function initMotionTrail() {
    const positions = new Float32Array(MAX_TRAIL * 3);
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geo.setDrawRange(0, 0);
    motionTrail = addToScene(new THREE.Line(
        geo,
        new THREE.LineBasicMaterial({ color: 0xffb800, linewidth: 2, transparent: true, opacity: 0.85 })
    ));
    motionTrail.visible = false;
}

/* ═════════════════════ 每帧动画 ═════════════════════ */

function animateOverlays() {
    if (targetRing) targetRing.rotation.z += 0.02;
}

/* ═════════════════════ 数据驱动更新 ═════════════════════ */

/**
 * 更新目标位姿球与末端实测球。
 * @param {number[]} cmdPos
 * @param {number[]|null} cmdQuat
 * @param {number[]} actPos
 * @param {number[]|null} actQuat
 */
export function updatePoseMarkers(cmdPos, cmdQuat, actPos, actQuat) {
    if (targetOrb && cmdPos) {
        targetOrb.position.set(cmdPos[0], cmdPos[1], cmdPos[2]);
        if (cmdQuat && cmdQuat.length === 4) {
            targetOrb.quaternion.set(cmdQuat[0], cmdQuat[1], cmdQuat[2], cmdQuat[3]);
        }
        // 目标落在可达盒之外时染成告警色 —— 空间上直接可见"出界了",
        // 不必等读日志。仍不阻止下发: 大屏用于观察。
        const outside = isTargetOutsideReachBox(cmdPos);
        if (outside !== _targetWasOutside) {
            _targetWasOutside = outside;
            const mat = targetOrb.material;
            mat.color.setHex(outside ? 0xff5566 : 0x00ff88);
            mat.emissive.setHex(outside ? 0xff5566 : 0x00ff88);
            targetRing.material.color.setHex(outside ? 0xff5566 : 0x00ff88);
        }
    }
    if (actualOrb && actPos) {
        actualOrb.position.set(actPos[0], actPos[1], actPos[2]);
        if (actQuat && actQuat.length === 4) {
            actualOrb.quaternion.set(actQuat[0], actQuat[1], actQuat[2], actQuat[3]);
        }
    }
}

/**
 * 更新残差连线。残差过小时隐藏, 避免在收敛良好时留下一条无意义红线。
 * @param {number[]} cmdPos
 * @param {number[]} actPos
 * @param {number} errMm
 */
export function updateErrorLine(cmdPos, actPos, errMm) {
    if (!errorLine) return;
    if (errMm < 3.0) {
        errorLine.visible = false;
        return;
    }
    errorLine.visible = true;
    const arr = errorLine.geometry.attributes.position.array;
    arr[0] = cmdPos[0]; arr[1] = cmdPos[1]; arr[2] = cmdPos[2];
    arr[3] = actPos[0]; arr[4] = actPos[1]; arr[5] = actPos[2];
    errorLine.geometry.attributes.position.needsUpdate = true;
}

/**
 * 更新质心球 / 垂准线 / 地面投影 / 支撑多边形配色。
 * @param {number[]} com
 * @param {number} marginMm
 * @param {boolean} isColliding
 */
export function updateComMarkers(com, marginMm, isColliding) {
    if (!com) return;
    const level = isColliding ? "alert" : marginMm >= 60 ? "ok" : marginMm >= 30 ? "warn" : "alert";
    const color = COM_COLORS[level];

    if (comOrb) {
        comOrb.position.set(com[0], com[1], com[2]);
        comOrb.material.color.setHex(color);
        comOrb.material.emissive.setHex(color);
    }
    if (comDropLine) {
        const arr = comDropLine.geometry.attributes.position.array;
        arr[0] = com[0]; arr[1] = com[1]; arr[2] = com[2];
        arr[3] = com[0]; arr[4] = com[1]; arr[5] = 0.002;
        comDropLine.geometry.attributes.position.needsUpdate = true;
        comDropLine.computeLineDistances();
        comDropLine.material.color.setHex(color);
    }
    if (comFloorDisc) {
        comFloorDisc.position.set(com[0], com[1], 0.002);
        comFloorDisc.material.color.setHex(color);
    }
    if (supportPolygon) {
        // 支撑域自身用绿/黄/红而非质心色, 与质心球区分
        supportPolygon.material.color.setHex(
            level === "ok" ? 0x00ff88 : level === "warn" ? 0xffb800 : 0xff3366
        );
    }
}

/**
 * 更新演示轨迹导轨与末端运动拖尾。
 * @param {boolean} demoActive
 * @param {string} arm
 * @param {number[]} actPos
 */
export function updateDemoOverlays(demoActive, arm, actPos) {
    const yCenter = arm === "right_arm" ? -0.22 : 0.22;

    if (orbitRing) {
        orbitRing.visible = demoActive;
        if (demoActive) {
            const arr = orbitRing.geometry.attributes.position.array;
            for (let i = 0; i <= ORBIT_SEGMENTS; i++) {
                const th = (i / ORBIT_SEGMENTS) * Math.PI * 2;
                arr[i * 3 + 0] = 0.38 + 0.10 * Math.cos(th);
                arr[i * 3 + 1] = yCenter;
                arr[i * 3 + 2] = 0.82 + 0.10 * Math.sin(th);
            }
            orbitRing.geometry.attributes.position.needsUpdate = true;
            orbitRing.computeLineDistances();
        }
    }

    if (!motionTrail) return;
    if (demoActive && actPos) {
        motionTrail.visible = true;
        trailHistory.push(new THREE.Vector3(actPos[0], actPos[1], actPos[2]));
        // 演示停止后不立刻清空, 而是逐点淡出, 形成"拖尾收束"的观感
        while (trailHistory.length > MAX_TRAIL) trailHistory.shift();
    } else if (trailHistory.length > 0) {
        trailHistory.shift();
    } else {
        motionTrail.visible = false;
    }

    const arr = motionTrail.geometry.attributes.position.array;
    for (let i = 0; i < trailHistory.length; i++) {
        arr[i * 3 + 0] = trailHistory[i].x;
        arr[i * 3 + 1] = trailHistory[i].y;
        arr[i * 3 + 2] = trailHistory[i].z;
    }
    motionTrail.geometry.attributes.position.needsUpdate = true;
    motionTrail.geometry.setDrawRange(0, trailHistory.length);
    if (trailHistory.length === 0) motionTrail.visible = false;
}

/** 视口是否可用 (标记物构建成功) */
export function overlaysReady() {
    return targetOrb !== null;
}