/**
 * 机器人 CAD 网格 (STL) 的加载与位姿更新。
 *
 * 网格来自 URDF 的 visual 几何体, 由服务端通过 state.visuals 下发
 * (geomName -> {mesh, pos, quat})。
 *
 * 渲染策略:
 *  - 几何体按 mesh 文件名缓存 (同一文件可能被多个几何体引用)。
 *  - DOM 级缓存已构建的 Mesh, 避免每帧重建。
 *  - 加载失败的文件进入 failedMeshes, **不再重试** ——
 *    updateVisuals 随遥测以 30Hz 被调用, 若无此集合, 每个失败网格
 *    会在每一帧重新发起请求 (36 网格 × 30Hz ≈ 1080 次/秒),
 *    压垮服务端并把首个真实报错冲掉。
 */

import { addToScene } from "./scene.js";
import { state } from "../state.js";

/** 几何体名 -> THREE.Mesh */
const robotMeshes = new Map();
/** 网格文件名 -> BufferGeometry */
const geometryCache = new Map();
/** 正在加载中的网格文件名 */
const loadingMeshes = new Set();
/** 已确认加载失败的网格文件名 (不重试) */
const failedMeshes = new Set();

let stlLoader = null;
let robotGroup = null;
let root = null;

/** 需要深色材质的连杆 (按文件名小写匹配) */
const DARK_LINKS = new Set([
    "logo_link", "pelvis_contour_link",
    "waist_roll_link", "waist_roll_link_rev_1_0",
    "torso_link_rev_1_0", "torso_link_23dof_rev_1_0",
]);

let MAT_URDF_WHITE = null;
let MAT_URDF_DARK = null;
let MAT_HEAD_VISOR = null;
let MAT_RUBBER_HAND = null;
let MAT_LOGO = null;
let MAT_COLLISION = null;

/** 几何体名 -> 该网格的常态材质 (用于告警时还原) */
const normalMaterials = new Map();

/** 当前处于告警红光的网格集合 */
const highlightedMeshes = new Set();

/**
 * 材质库。
 *
 * 主壳体此前用 0xd8dee6 (接近白) 配 metalness 0.25, 在三点布光下直接高光溢出,
 * 关节缝隙与结构线全部消失, 机器人看起来像一团惨白的橡皮。
 *
 * 现改为**中灰哑光工程塑料**: 基色压暗、粗糙度提高以抑制高光斑,
 * 依赖布光而非基色本身塑造体积。金属件单独用高 metalness 表达。
 */
function buildMaterials() {
    // 主壳体: 浅灰哑光塑料
    MAT_URDF_WHITE = new THREE.MeshStandardMaterial({
        color: 0x9aa4b2, metalness: 0.08, roughness: 0.78,
    });
    // 深色结构件 / 内衬
    MAT_URDF_DARK = new THREE.MeshStandardMaterial({
        color: 0x39424f, metalness: 0.15, roughness: 0.72,
    });
    // 面罩玻璃
    MAT_HEAD_VISOR = new THREE.MeshStandardMaterial({
        color: 0x141a22, metalness: 0.55, roughness: 0.22,
    });
    // 橡胶手掌
    MAT_RUBBER_HAND = new THREE.MeshStandardMaterial({
        color: 0x23282f, metalness: 0.0, roughness: 0.95,
    });
    // 品牌标识: 唯一的自发光件, 作为点缀而非照明源
    MAT_LOGO = new THREE.MeshStandardMaterial({
        color: 0x0d7a94, emissive: 0x0d7a94, emissiveIntensity: 0.35,
        metalness: 0.2, roughness: 0.45,
    });
    // 碰撞告警材质: 自发光红, 在灰色机体上必须一眼可辨
    MAT_COLLISION = new THREE.MeshStandardMaterial({
        color: 0xff2d55, emissive: 0xff2d55, emissiveIntensity: 0.85,
        metalness: 0.0, roughness: 0.5,
    });
}

/**
 * 初始化机器人模型组。
 * @param {HTMLElement} rootEl 视口容器
 */
export function initRobotModel(rootEl) {
    buildMaterials();
    stlLoader = new THREE.STLLoader();
    robotGroup = new THREE.Group();
    root = rootEl;
    addToScene(robotGroup);
}

/**
 * 按连杆名选择材质。
 * @param {string} meshFile
 * @returns {THREE.Material}
 */
function materialFor(meshFile) {
    const lower = meshFile.toLowerCase();
    if (lower.startsWith("logo")) return MAT_LOGO;
    if (lower.includes("rubber_hand")) return MAT_RUBBER_HAND;
    if (lower.startsWith("head_link")) return MAT_HEAD_VISOR;
    const base = lower.replace(/\.stl$/, "");
    if (DARK_LINKS.has(base)) return MAT_URDF_DARK;
    return MAT_URDF_WHITE;
}

/**
 * 用遥测下发的几何体集合更新机器人姿态。
 * @param {Record<string, {mesh:string, pos:number[], quat:number[]}>} visuals
 */
export function updateVisuals(visuals) {
    if (!visuals || !robotGroup) return;

    for (const [geomName, info] of Object.entries(visuals)) {
        if (!info || !info.mesh) continue;
        const existing = robotMeshes.get(geomName);

        if (existing) {
            // 几何体已就位: 仅更新位姿 (每帧都要做的事, 开销极小)
            applyTransform(existing, info);
            continue;
        }

        const cached = geometryCache.get(info.mesh);
        if (cached) {
            robotMeshes.set(geomName, attach(cached, info, geomName));
            state.meshCount = robotMeshes.size;
            continue;
        }

        if (loadingMeshes.has(info.mesh) || failedMeshes.has(info.mesh)) continue;
        loadingMeshes.add(info.mesh);
        requestMesh(info.mesh, geomName, info);
    }
}

function requestMesh(meshFile, geomName, info) {
    stlLoader.load(
        `meshes/${meshFile}`,
        (geometry) => {
            geometry.computeVertexNormals();
            geometryCache.set(meshFile, geometry);
            loadingMeshes.delete(meshFile);
            // 加载期间几何体可能已被移除, 或同一文件被多个几何体引用
            if (!robotMeshes.has(geomName)) {
                robotMeshes.set(geomName, attach(geometry, info, geomName));
                state.meshCount = robotMeshes.size;
            }
            // 同一文件可能对应多个几何体, 补挂其余等待者
            syncPendingGeometryUsers(meshFile, geometry);
        },
        undefined,
        (err) => {
            loadingMeshes.delete(meshFile);
            failedMeshes.add(meshFile);
            // 仅在首个失败时上报一次, 避免每帧刷屏冲掉真实线索
            if (failedMeshes.size === 1) {
                console.error(
                    `[3D] 网格加载失败并已停止重试。首个失败: ${meshFile}。` +
                    `请确认 visualizer/web/meshes 符号链接指向有效的资源目录。`, err
                );
            }
        }
    );
}

/**
 * 把新加载的几何体挂到所有仍在等待它的几何体上。
 * @param {string} meshFile
 * @param {THREE.BufferGeometry} geometry
 */
function syncPendingGeometryUsers(meshFile, geometry) {
    // 当前实现中 updateVisuals 每帧都会被调用, 等待中的几何体会在下一帧
    // 从 geometryCache 命中并完成挂载, 因此此处无需额外处理。
    // 保留该函数以明确表达这一约定, 避免后续误加轮询逻辑。
    void meshFile; void geometry;
}

function attach(geometry, info, geomName) {
    const mat = materialFor(info.mesh);
    const mesh = new THREE.Mesh(geometry, mat);
    mesh.name = geomName;
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    // 记录常态材质: 碰撞告警解除后据此还原
    normalMaterials.set(geomName, mat);
    // 几何体可能是在告警状态下才加载完成的, 此时必须直接套用告警材质,
    // 否则会出现"碰撞已消失但新加载的零件仍是红的"或反之
    if (highlightedMeshes.has(geomName)) {
        mesh.material = MAT_COLLISION;
    }
    robotGroup.add(mesh);
    applyTransform(mesh, info);
    return mesh;
}

function applyTransform(mesh, info) {
    const p = info.pos;
    const q = info.quat;
    if (p && p.length === 3) mesh.position.set(p[0], p[1], p[2]);
    if (q && q.length === 4) mesh.quaternion.set(q[0], q[1], q[2], q[3]);
}

/**
 * 几何体名 -> 其所属 URDF link 名。
 *
 * visuals 的键形如 `left_shoulder_yaw_link_0`, 去掉末尾 `_序号` 即为 link 名。
 * 单独抽出为纯函数以便直接测试 —— 这段名称映射一旦出错, 碰撞告警会标红
 * 错误的零件, 而在 3D 场景里这种错误极难肉眼发现。
 *
 * @param {string} geomName
 * @returns {string}
 */
export function linkNameOf(geomName) {
    return String(geomName).replace(/_\d+$/, "");
}

/**
 * 高亮一组连杆为碰撞告警红。
 *
 * 后端上报的 colliding_pairs 给出的是 URDF **link 名**
 * (如 `left_shoulder_yaw_link`), 而 visuals 的键是**几何体名**
 * (如 `left_shoulder_yaw_link_0`)。两者按 `<link名>_<序号>` 对应,
 * 但同一 link 可能对应多个视觉几何体, 因此这里做前缀匹配而非精确查找。
 *
 * @param {string[]} linkNames 处于碰撞中的连杆名
 */
export function highlightLinks(linkNames) {
    const wanted = new Set((linkNames || []).map((n) => String(n)));
    if (wanted.size === 0 && highlightedMeshes.size === 0) return;

    const shouldHighlight = new Set();
    for (const geomName of robotMeshes.keys()) {
        if (wanted.has(linkNameOf(geomName)) || wanted.has(geomName)) {
            shouldHighlight.add(geomName);
        }
    }

    // 解除上一轮的高亮
    for (const geomName of highlightedMeshes) {
        if (shouldHighlight.has(geomName)) continue;
        const mesh = robotMeshes.get(geomName);
        const normal = normalMaterials.get(geomName);
        if (mesh && normal) mesh.material = normal;
    }
    // 应用本轮的高亮
    for (const geomName of shouldHighlight) {
        const mesh = robotMeshes.get(geomName);
        if (mesh) mesh.material = MAT_COLLISION;
    }
    highlightedMeshes.clear();
    for (const g of shouldHighlight) highlightedMeshes.add(g);
}

/**
 * 当前被高亮的连杆名 (供视口角标展示)。
 * @returns {string[]}
 */
export function highlightedLinkNames() {
    return [...highlightedMeshes].map((g) => g.replace(/_\d+$/, ""));
}

/** 视口是否可用 */
export function robotModelReady() {
    return robotGroup !== null && root !== null;
}