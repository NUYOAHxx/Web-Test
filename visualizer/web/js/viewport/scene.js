/**
 * Three.js 渲染基础设施: 渲染器 / 相机 / 控制器 / 布光 / 地面 / 渲染循环。
 *
 * 与 overlays.js (目标球、质心球、拖尾等标记物) 与 robot.js (CAD 网格) 分离:
 * 本模块只管"怎么把画面画出来", 不关心画什么内容。
 */

/** 渲染器创建失败时抛出, 由 main.js 捕获并降级 */
export class ViewportUnavailableError extends Error {
    constructor(reason) {
        super(reason);
        this.name = "ViewportUnavailableError";
    }
}

let scene = null;
let camera = null;
let renderer = null;
let controls = null;
let gridHelper = null;
let resizeObserver = null;

/** 相机默认机位与观察目标 (复位视角时复用) */
const DEFAULT_CAMERA_POS = [2.05, -1.95, 1.15];
const DEFAULT_TARGET = [0.04, 0.0, 0.62];

/**
 * 初始化 3D 视口。
 *
 * WebGL 上下文可能创建失败 (无 GPU / 硬件加速关闭 / 远程桌面 / 无头环境)。
 * 旧代码在此处抛异常并中断整个启动流程, 导致纯数据面板也不再刷新。
 * 现在抛出专用错误类型, 由调用方决定降级, 且降级后数据面板照常工作。
 *
 * @param {HTMLElement} container
 * @returns {boolean} 是否成功创建视口
 */
export function initViewport(container) {
    // 机器人领域统一 Z 轴向上 (与 ROS 2 / URDF 约定一致)
    THREE.Object3D.DefaultUp.set(0, 0, 1);

    const width = container.clientWidth || 800;
    const height = container.clientHeight || 600;

    scene = new THREE.Scene();
    // 指数雾营造纵深。系数 0.08 在 ~2m 处已有可见衰减, 会把机器人整体压灰发暗。
    // 改为 0.028 —— 只在远端网格边缘起作用, 近处机身保持清晰。
    scene.fog = new THREE.FogExp2(0x0a1220, 0.028);

    camera = new THREE.PerspectiveCamera(45, width / height, 0.05, 50);
    camera.up.set(0, 0, 1);
    camera.position.set(...DEFAULT_CAMERA_POS);

    try {
        renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    } catch (err) {
        throw new ViewportUnavailableError("未能创建 WebGL 上下文");
    }
    if (!renderer || !renderer.getContext()) {
        throw new ViewportUnavailableError("WebGL 上下文不可用");
    }

    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    // 曝光必须与灯光总强度配套。旧配置灯光合计 3.9 又乘 1.35 曝光,
    // ACES 在高光区快速趋近纯白, 机身细节被整体压掉, 呈"惨白泛粉"。
    renderer.toneMappingExposure = 0.95;
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    container.appendChild(renderer.domElement);

    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.target.set(...DEFAULT_TARGET);
    controls.maxPolarAngle = Math.PI / 2 + 0.02;

    setupLights(scene);
    setupGround(scene);

    // 用 ResizeObserver 取代 window.resize:
    // 视口尺寸还受布局与折叠面板影响, 仅监听 window.resize 会漏掉这些场景。
    resizeObserver = new ResizeObserver(() => onResize(container));
    resizeObserver.observe(container);

    startRenderLoop();
    return true;
}

/**
 * 三点布光。
 *
 * 强度配比按"主光主导、补光仅提暗部、轮廓光只勾边"设计:
 *   环境 0.34 + 主光 1.05 + 补光 0.28 + 轮廓 0.42 ≈ 2.1 (旧配置为 3.9)
 *
 * 补光与轮廓光改为**低饱和冷色**而非纯青/纯紫。旧配置里 0x00f0ff 与 0xa566ff
 * 都是高饱和大角度铺在全身, 会把机身材质整体染色 —— 实际渲染中机器人
 * 呈现惨白泛粉, 而非中性金属/工程塑料质感。
 *
 * @param {THREE.Scene} s
 */
function setupLights(s) {
    // 环境光提供基础可读性, 略带冷色以贴合深蓝背景
    s.add(new THREE.AmbientLight(0x8a9bb4, 0.34));

    // 主光: 中性偏暖, 唯一投射阴影的光源, 定义主要形体
    const key = new THREE.DirectionalLight(0xfff4e8, 1.05);
    key.position.set(2.8, -2.4, 3.8);
    key.castShadow = true;
    s.add(key);

    // 补光: 仅抬起背光侧暗部, 强度必须明显低于主光, 否则立体感消失
    const fill = new THREE.DirectionalLight(0x8fb8d8, 0.28);
    fill.position.set(-2.8, -1.5, 2.2);
    s.add(fill);

    // 轮廓光: 从后上方勾出边缘, 分离主体与背景
    const rim = new THREE.DirectionalLight(0xb0a8d8, 0.42);
    rim.position.set(0.2, 3.2, 2.8);
    s.add(rim);
}

/**
 * 地面网格与原点坐标轴。
 * @param {THREE.Scene} s
 */
/**
 * 地面网格与原点坐标轴。
 *
 * 主线此前用 0x00f0ff 亮青, 在深色场景里形成高对比棋盘, 视觉上盖过机器人主体。
 * 现改为低亮度蓝灰, 让网格退为空间参照。
 *
 * @param {THREE.Scene} s
 */
function setupGround(s) {
    gridHelper = new THREE.GridHelper(3.5, 35, 0x2c4a63, 0x18242f);
    gridHelper.rotation.x = Math.PI / 2;   // GridHelper 默认在 XZ 平面, 旋到 XY
    gridHelper.position.set(0, 0, 0);
    s.add(gridHelper);

    const axes = new THREE.AxesHelper(0.25);
    axes.position.set(0, 0, 0.002);
    s.add(axes);
}

/**
 * 渲染循环。每帧的更新钩子由 overlays/robot 模块注册。
 * @type {Array<() => void>}
 */
const frameHooks = [];

/** rAF 句柄; disposeViewport 时必须取消, 否则循环会继续访问已释放的资源 */
let rafHandle = 0;
/** 循环是否仍在运行 */
let loopRunning = false;

/**
 * 注册每帧回调 (在渲染前执行)。
 * @param {() => void} fn
 */
export function onFrame(fn) {
    frameHooks.push(fn);
}

function startRenderLoop() {
    loopRunning = true;
    const loop = () => {
        if (!loopRunning) return;
        rafHandle = requestAnimationFrame(loop);
        for (const hook of frameHooks) hook();
        controls.update();
        renderer.render(scene, camera);
    };
    rafHandle = requestAnimationFrame(loop);
}

function onResize(container) {
    if (!renderer || !camera) return;
    const w = container.clientWidth;
    const h = container.clientHeight;
    if (!w || !h) return;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
}

/* ═════════════════════ 对外操作 ═════════════════════ */

/** @returns {THREE.Scene|null} */
export function getScene() { return scene; }

/** @returns {THREE.Camera|null} */
export function getCamera() { return camera; }

/** 把物体加入场景 */
export function addToScene(obj) {
    if (scene) scene.add(obj);
    return obj;
}

/** 相机复位到默认机位 */
export function resetCamera() {
    if (!camera || !controls) return;
    camera.position.set(...DEFAULT_CAMERA_POS);
    controls.target.set(...DEFAULT_TARGET);
    controls.update();
}

/** 切换地面网格显隐 @returns {boolean} 切换后的可见状态 */
export function toggleGrid() {
    if (!gridHelper) return false;
    gridHelper.visible = !gridHelper.visible;
    return gridHelper.visible;
}

/**
 * 释放渲染资源。
 *
 * 必须先停循环再置空 renderer —— 否则下一帧 requestAnimationFrame 回调
 * 会执行 `renderer.render(...)`, 而此时 renderer 已被置为 null,
 * 在页面卸载瞬间抛出 "Cannot read properties of null (reading 'render')"。
 * 该异常发生在微任务之外, 不会中断卸载流程, 但会污染控制台并掩盖真实问题。
 */
export function disposeViewport() {
    loopRunning = false;
    if (rafHandle) {
        cancelAnimationFrame(rafHandle);
        rafHandle = 0;
    }
    if (resizeObserver) {
        resizeObserver.disconnect();
        resizeObserver = null;
    }
    frameHooks.length = 0;
    if (renderer) {
        renderer.dispose();
        renderer = null;
    }
    scene = null;
    camera = null;
    controls = null;
}