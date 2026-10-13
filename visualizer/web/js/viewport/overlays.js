/**
 * 视口标记物: 目标球 / 末端实测球 / 残差连线 / 质心系统 / 支撑多边形 / 演示导轨 / 运动拖尾。
 *
 * 这些是"随遥测数据变化的可视化元素", 与 scene.js 的渲染基础设施分离。
 * 每个标记物提供独立的 update 方法, 主循环只负责派发数据。
 */

import { addToScene } from "./scene.js";
import { clearanceStatus } from "../format.js";

/* 标记物句柄 */
let targetOrb = null;
let targetArrow = null;
let actualArrow = null;
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
    geo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(REACH_BOX_VERTS * 3), 3));
    reachBox = addToScene(new THREE.LineSegments(
        geo,
        new THREE.LineBasicMaterial({
            color: 0x00c8e0, transparent: true, opacity: 0.40, depthWrite: false,
        })
    ));
    reachBox.renderOrder = 998;      // 压在机身之下, 不遮挡关节观察
    reachBox.frustumCulled = false;
    // 默认显示 —— 这正是用户要的距离参照。收到 model_info 前盒体尺寸为零,
    // 几何体退化成一个点, 不会造成可见干扰。
    reachBox.visible = true;
}

/**
 * 范围盒的画法: 底面矩形 + 四角立柱, 不画顶面。
 *
 * 迭代过两版, 两次都因"看起来不像盒子"被否:
 *   · 12 条完整棱 —— 顶面线正好横穿机器人上半身, 把整体切割成几块;
 *   · 每角 1 段朝内斜线 (角括号) —— 8 段互不相连, 视线无法把它们重建成
 *     一个盒子, 实际观感是画面里飘着几根随机线段, 比棱线更糟。
 *
 * 现取中间态:
 *   · 底面矩形完整画出 —— 它把范围**锚在地面**上, 是"一个区域"最强的空间
 *     暗示, 也是判断目标是否出界最常参照的一圈;
 *   · 四角立柱给出高度感, 说明这是有体积的空间而非地面上的一个圈;
 *   · 不画顶面 —— 顶面线正是穿过头部和胸腔的那些, 去掉后视线豁然开朗,
 *     而底面 + 立柱已足以还原体积。
 *
 * 共 8 条线段 = 16 个顶点。
 */
const REACH_BOX_VERTS = 16;

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

    // 底面 4 个角 (z = 下界) 与顶面 4 个角 (z = 上界) 一一对应
    const b = [
        [lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]],
    ];
    const pos = reachBox.geometry.getAttribute("position");
    const put = (x, y, z) => pos.setXYZ(k++, x, y, z);
    let k = 0;

    // 底面矩形: 4 条边, 完整画出 —— 把范围锚在地面上
    for (let i = 0; i < 4; i++) {
        const p = b[i], q = b[(i + 1) % 4];
        put(p[0], p[1], lo[2]);
        put(q[0], q[1], lo[2]);
    }
    // 四角立柱: 从底面角点垂直连到顶面角点, 给出体积感
    for (const [x, y] of b) {
        put(x, y, lo[2]);
        put(x, y, hi[2]);
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

/* ═════════════════════ 目标 / 末端 标记物 ═════════════════════ */

/**
 * 位置与朝向的图元选择
 * -------------------------------------------------------------------
 * 旧实现给目标点叠了三个图元: 球 (r=0.018) + 同色圆环 + AxesHelper(0.05)。
 *
 *   · 球与环轮廓重合、颜色相同, 等于同一形状画了两遍, 只是让边缘"厚"了一圈;
 *   · AxesHelper 画的是三根 RGB 轴线, 即一个坐标系。坐标系回答不了用户真正
 *     想看的问题 —— "手要往哪个方向伸"。而且 0.05m 长的轴线在目标距离上只有
 *     几个像素, 颜色还被机身光照与雾吃掉, 基本读不出信息, 只是噪声。
 *
 * 现改为两个图元, 每个直接对应一个语义:
 *   · 位置 → 八面体。棱面结构使轮廓随视角变化, 比球更容易与背景区分, 且
 *     没有球面那种"一团糊"的观感。
 *   · 朝向 → 单根箭头, 沿末端 +Z (夹爪接近轴)。方向是单一信息, 箭头恰好
 *     只表达这一件事, 不会像坐标系那样把注意力分散到三个轴上。
 */

/** 目标点: 八面体 + 接近方向箭头 */
function initTargetOrb() {
    targetOrb = addToScene(new THREE.Mesh(
        new THREE.OctahedronGeometry(0.020),
        new THREE.MeshStandardMaterial({
            color: 0x00ff88, emissive: 0x00ff88, emissiveIntensity: 1.1,
            roughness: 0.25, metalness: 0.1,
        })
    ));
    targetOrb.renderOrder = 999;

    targetArrow = new THREE.ArrowHelper(
        new THREE.Vector3(0, 0, 1), new THREE.Vector3(0, 0, 0),
        0.085, 0x00ff88, 0.022, 0.013,
    );
    targetArrow.line.material.depthTest = false;
    targetArrow.cone.material.depthTest = false;
    targetOrb.add(targetArrow);
}

/** 末端实测点: 同构但更小更暗, 与目标形成主次 */
function initActualOrb() {
    actualOrb = addToScene(new THREE.Mesh(
        new THREE.OctahedronGeometry(0.013),
        new THREE.MeshStandardMaterial({
            color: 0xffb800, emissive: 0xffb800, emissiveIntensity: 0.8,
            roughness: 0.3, metalness: 0.1,
        })
    ));
    actualOrb.renderOrder = 999;

    actualArrow = new THREE.ArrowHelper(
        new THREE.Vector3(0, 0, 1), new THREE.Vector3(0, 0, 0),
        0.058, 0xffb800, 0.016, 0.010,
    );
    actualArrow.line.material.depthTest = false;
    actualArrow.cone.material.depthTest = false;
    actualOrb.add(actualArrow);
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

/* ── 每帧动画 ──
 *
 * 此前此处让目标圆环持续自转以吸引注意。该环已随"球 + 环 + 坐标系"一并移除。
 * 也**不能**把自转改到八面体上: updatePoseMarkers 每帧用四元数写入目标朝向,
 * 而 three.js 中 rotation 与 quaternion 互相联动, 叠加自转会污染朝向显示 ——
 * 箭头会指向与指令不符的方向。
 */

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
            const hex = outside ? 0xff5566 : 0x00ff88;
            targetOrb.material.color.setHex(hex);
            targetOrb.material.emissive.setHex(hex);
            targetArrow.setColor(hex);
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