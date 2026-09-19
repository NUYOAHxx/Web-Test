/**
 * ==============================================================================
 * Unitree G1 实时高精度数字孪生与遥测监控大屏 - 前端交互与数据渲染引擎
 * (Pure Telemetry Visualizer & Command Dispatch Engine)
 * ==============================================================================
 */

// ── 全局 Three.js 与场景变量 ──
let scene, camera, renderer, controls;
let targetOrb, actualOrb, errorLine;
let robotCADGroup;
let currentArm = "left_arm";

// 当前选定的笛卡尔目标点缓存
const currentCmdCoord = {
    x: 0.35,
    y: 0.22,
    z: 0.85,
};

// ── DOM 元素缓存 ──
const dom = {
    // 顶部 HUD
    connectionChip: document.getElementById("connectionChip"),
    connectionStatus: document.getElementById("connectionStatus"),
    topicNameText: document.getElementById("topicNameText"),
    activeArmText: document.getElementById("activeArmText"),
    globalCollisionChip: document.getElementById("globalCollisionChip"),
    globalCollisionText: document.getElementById("globalCollisionText"),
    btnCircleDemo: document.getElementById("btnCircleDemo"),
    btnResetStand: document.getElementById("btnResetStand"),

    // 目标指令下发控制台
    btnSelectLeftArm: document.getElementById("btnSelectLeftArm"),
    btnSelectRightArm: document.getElementById("btnSelectRightArm"),
    sliderX: document.getElementById("sliderX"),
    sliderY: document.getElementById("sliderY"),
    sliderZ: document.getElementById("sliderZ"),
    valInputX: document.getElementById("valInputX"),
    valInputY: document.getElementById("valInputY"),
    valInputZ: document.getElementById("valInputZ"),
    btnDispatchTarget: document.getElementById("btnDispatchTarget"),

    // 空间位姿遥测
    tgtX: document.getElementById("tgtX"),
    tgtY: document.getElementById("tgtY"),
    tgtZ: document.getElementById("tgtZ"),
    actX: document.getElementById("actX"),
    actY: document.getElementById("actY"),
    actZ: document.getElementById("actZ"),
    deltaX: document.getElementById("deltaX"),
    deltaY: document.getElementById("deltaY"),
    deltaZ: document.getElementById("deltaZ"),
    posErrNum: document.getElementById("posErrNum"),
    posErrBar: document.getElementById("posErrBar"),
    streamHzNum: document.getElementById("streamHzNum"),
    packetCountText: document.getElementById("packetCountText"),
    footerStatusText: document.getElementById("footerStatusText"),
    logStream: document.getElementById("logStream"),

    // 安全雷达
    minClearanceVal: document.getElementById("minClearanceVal"),
    clearanceMarker: document.getElementById("clearanceMarker"),
    radarBadge: document.getElementById("radarBadge"),
    collisionAlertBox: document.getElementById("collisionAlertBox"),
    alertPairsList: document.getElementById("alertPairsList"),

    // 关节遥测列表
    waistJointsList: document.getElementById("waistJointsList"),
    armJointsList: document.getElementById("armJointsList"),
    armDofBadge: document.getElementById("armDofBadge"),

    // 视口辅助
    btnResetCamera: document.getElementById("btnResetCamera"),
    btnToggleGrid: document.getElementById("btnToggleGrid"),

    // IK 收敛动力学曲线
    cardConvergenceTrace: document.getElementById("cardConvergenceTrace"),
    convergenceModeBadge: document.getElementById("convergenceModeBadge"),
    traceTimeVal: document.getElementById("traceTimeVal"),
    traceItersVal: document.getElementById("traceItersVal"),
    traceSeedVal: document.getElementById("traceSeedVal"),
    traceResidueVal: document.getElementById("traceResidueVal"),
    convergenceCanvas: document.getElementById("convergenceCanvas"),
    traceTooltip: document.getElementById("traceTooltip"),
    traceStatusTag: document.getElementById("traceStatusTag"),
};

// ── 初始化启动入口 ──
window.addEventListener("DOMContentLoaded", () => {
    initThreeScene();
    initCommandPanel();
    initEventListeners();
    initConvergenceCanvasEvents();
    connectSSE();
});

// ─────────────────────────────────────────────────────────────
// 1. Three.js 场景与工业摄影级 3 点布光系统
// ─────────────────────────────────────────────────────────────
function initThreeScene() {
    // 强制全局 Z 轴向上 (完全契合工业机器人与 ROS 2 标准)
    THREE.Object3D.DefaultUp.set(0, 0, 1);

    const container = document.getElementById("threeCanvasContainer");
    const width = container.clientWidth;
    const height = container.clientHeight;

    scene = new THREE.Scene();
    scene.fog = new THREE.FogExp2(0x060910, 0.08);

    // 工业透视相机 (自然仰俯角，全局视野框入整机头部至脚踝)
    camera = new THREE.PerspectiveCamera(45, width / height, 0.05, 50);
    camera.up.set(0, 0, 1);
    camera.position.set(2.05, -1.95, 1.15);

    // ACES 色调映射 WebGL 渲染器
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.35;
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    container.appendChild(renderer.domElement);

    // 视角交互控制器
    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.target.set(0.04, 0.0, 0.62);
    controls.maxPolarAngle = Math.PI / 2 + 0.02;


    // 工业立体三点布光 (主光日光 + 辅光青光 + 轮廓光紫光)
    const ambientLight = new THREE.AmbientLight(0x708296, 0.6);
    scene.add(ambientLight);

    const keyLight = new THREE.DirectionalLight(0xffffff, 1.45);
    keyLight.position.set(2.8, -2.4, 3.8);
    keyLight.castShadow = true;
    scene.add(keyLight);

    const fillLight = new THREE.DirectionalLight(0x00f0ff, 0.85);
    fillLight.position.set(-2.8, -1.5, 2.2);
    scene.add(fillLight);

    const rimLight = new THREE.DirectionalLight(0xa566ff, 1.0);
    rimLight.position.set(0.2, 3.2, 2.8);
    scene.add(rimLight);

    // 水平地面网格 (XY 平面，Z=0)
    const gridHelper = new THREE.GridHelper(3.5, 35, 0x00f0ff, 0x141f2e);
    gridHelper.rotation.x = Math.PI / 2;
    gridHelper.position.set(0, 0, 0);
    scene.add(gridHelper);
    scene.gridHelper = gridHelper;

    // 空间微坐标轴辅助
    const axesHelper = new THREE.AxesHelper(0.25);
    axesHelper.position.set(0, 0, 0.002);
    scene.add(axesHelper);

    // 宇树 G1 机器人 36 连杆 CAD STL 装配体组
    robotCADGroup = new THREE.Group();
    scene.add(robotCADGroup);

    // 笛卡尔规划目标光球 (Target Orb: 荧光青绿)
    const targetGeo = new THREE.SphereGeometry(0.018, 24, 24);
    const targetMat = new THREE.MeshStandardMaterial({
        color: 0x00ff88,
        emissive: 0x00ff88,
        emissiveIntensity: 0.9,
        roughness: 0.2,
    });
    targetOrb = new THREE.Mesh(targetGeo, targetMat);
    scene.add(targetOrb);

    const ringGeo = new THREE.RingGeometry(0.024, 0.030, 32);
    const ringMat = new THREE.MeshBasicMaterial({ color: 0x00ff88, side: THREE.DoubleSide, transparent: true, opacity: 0.75 });
    const targetRing = new THREE.Mesh(ringGeo, ringMat);
    targetOrb.add(targetRing);

    // 实际末端手爪实测光球 (Actual FK Orb: 钛金金黄)
    const actualGeo = new THREE.SphereGeometry(0.014, 24, 24);
    const actualMat = new THREE.MeshStandardMaterial({
        color: 0xffb800,
        emissive: 0xffb800,
        emissiveIntensity: 0.85,
        roughness: 0.25,
    });
    actualOrb = new THREE.Mesh(actualGeo, actualMat);
    scene.add(actualOrb);

    // 空间跟踪残差连线 (Error Line: 激光红)
    const lineMat = new THREE.LineBasicMaterial({ color: 0xff3366, linewidth: 2, transparent: true, opacity: 0.85 });
    const lineGeo = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]);
    errorLine = new THREE.Line(lineGeo, lineMat);
    scene.add(errorLine);

    window.addEventListener("resize", onWindowResize);
    animate();
}

function onWindowResize() {
    const container = document.getElementById("threeCanvasContainer");
    if (!container) return;
    const w = container.clientWidth;
    const h = container.clientHeight;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
}

// ─────────────────────────────────────────────────────────────
// 2. 官方 CAD STL 3D 模型高保真 PBR 材质引擎 (修复黑脸与质感)
// ─────────────────────────────────────────────────────────────
const robotMeshes = {};
const stlGeometryCache = {};
const loadingMeshes = new Set();
let stlLoader = null;

// 1. 机甲外壳：细腻陶瓷珠光白，立体漫反射 (彻底告别死白)
const MAT_URDF_WHITE = new THREE.MeshStandardMaterial({
    color: 0xdce2ec,
    metalness: 0.18,
    roughness: 0.38,
});

// 2. 黑色机械骨架与暗部构件：深空石墨金属 (拒绝死黑剪影)
const MAT_URDF_DARK = new THREE.MeshStandardMaterial({
    color: 0x242832,
    metalness: 0.72,
    roughness: 0.28,
});

// 3. 头部面罩与传感器视窗：黑曜石深灰钛金属质感，反射环境立体光芒 (清晰勾勒弧面)
const MAT_HEAD_VISOR = new THREE.MeshStandardMaterial({
    color: 0x28303f,
    metalness: 0.65,
    roughness: 0.35,
});

// 4. 橡胶手爪：工业抓握防滑哑光黑
const MAT_RUBBER_HAND = new THREE.MeshStandardMaterial({
    color: 0x1a1d22,
    metalness: 0.05,
    roughness: 0.85,
});

// 5. 宇树胸前 Logo：发光青色霓虹
const MAT_LOGO_ACCENT = new THREE.MeshStandardMaterial({
    color: 0x00f0ff,
    emissive: 0x00f0ff,
    emissiveIntensity: 0.95,
    roughness: 0.2,
});

const URDF_DARK_MESHES = new Set([
    "pelvis.stl",
    "left_hip_pitch_link.stl",
    "right_hip_pitch_link.stl",
    "left_ankle_roll_link.stl",
    "right_ankle_roll_link.stl",
]);

function getMaterialForLink(meshFile) {
    const lower = meshFile.toLowerCase();
    if (lower.includes("logo")) return MAT_LOGO_ACCENT;
    if (lower.includes("rubber_hand")) return MAT_RUBBER_HAND;
    if (lower.includes("head")) return MAT_HEAD_VISOR;
    if (URDF_DARK_MESHES.has(lower)) return MAT_URDF_DARK;
    return MAT_URDF_WHITE;
}

function updateRobotCADModel(visuals) {
    if (!visuals || !robotCADGroup) return;
    if (!stlLoader) stlLoader = new THREE.STLLoader();

    for (const [geomName, info] of Object.entries(visuals)) {
        const meshFile = info.mesh;
        const pos = info.pos;
        const quat = info.quat;

        if (robotMeshes[geomName]) {
            const mesh = robotMeshes[geomName];
            mesh.position.set(pos[0], pos[1], pos[2]);
            mesh.quaternion.set(quat[0], quat[1], quat[2], quat[3]);
        } else {
            if (stlGeometryCache[meshFile]) {
                const mat = getMaterialForLink(meshFile);
                const mesh = new THREE.Mesh(stlGeometryCache[meshFile], mat);
                mesh.name = geomName;
                mesh.castShadow = true;
                mesh.receiveShadow = true;
                mesh.position.set(pos[0], pos[1], pos[2]);
                mesh.quaternion.set(quat[0], quat[1], quat[2], quat[3]);
                robotCADGroup.add(mesh);
                robotMeshes[geomName] = mesh;
            } else if (!loadingMeshes.has(meshFile)) {
                loadingMeshes.add(meshFile);
                stlLoader.load(
                    `meshes/${meshFile}`,
                    (geom) => {
                        geom.computeVertexNormals();
                        stlGeometryCache[meshFile] = geom;
                        loadingMeshes.delete(meshFile);
                        if (!robotMeshes[geomName]) {
                            const mat = getMaterialForLink(meshFile);
                            const mesh = new THREE.Mesh(geom, mat);
                            mesh.name = geomName;
                            mesh.castShadow = true;
                            mesh.receiveShadow = true;
                            mesh.position.set(pos[0], pos[1], pos[2]);
                            mesh.quaternion.set(quat[0], quat[1], quat[2], quat[3]);
                            robotCADGroup.add(mesh);
                            robotMeshes[geomName] = mesh;
                        }
                    },
                    undefined,
                    (err) => {
                        console.error(`加载网格失败: ${meshFile}`, err);
                        loadingMeshes.delete(meshFile);
                    }
                );
            }
        }
    }
}

function animate() {
    requestAnimationFrame(animate);
    controls.update();
    if (targetOrb) targetOrb.rotation.y += 0.015;
    renderer.render(scene, camera);
}

// ─────────────────────────────────────────────────────────────
// 3. 目标指令下发中枢交互逻辑
// ─────────────────────────────────────────────────────────────
function initCommandPanel() {
    // 左右臂选择切换
    dom.btnSelectLeftArm.addEventListener("click", () => setArmSelection("left_arm"));
    dom.btnSelectRightArm.addEventListener("click", () => setArmSelection("right_arm"));

    // 坐标滑块同步
    dom.sliderX.addEventListener("input", (e) => updateCoordFromSlider("x", parseFloat(e.target.value)));
    dom.sliderY.addEventListener("input", (e) => updateCoordFromSlider("y", parseFloat(e.target.value)));
    dom.sliderZ.addEventListener("input", (e) => updateCoordFromSlider("z", parseFloat(e.target.value)));

    // 立即下达目标指令主按钮
    dom.btnDispatchTarget.addEventListener("click", () => {
        dispatchTargetCommand(currentCmdCoord.x, currentCmdCoord.y, currentCmdCoord.z, "自定义目标指令");
    });
}

async function setArmSelection(arm) {
    currentArm = arm;
    if (arm === "left_arm") {
        dom.btnSelectLeftArm.classList.add("active");
        dom.btnSelectRightArm.classList.remove("active");
        dom.activeArmText.textContent = "LEFT ARM";
        if (dom.armDofBadge) dom.armDofBadge.textContent = "7-DoF · Left Arm";
        if (currentCmdCoord.y < 0) {
            currentCmdCoord.y = Math.abs(currentCmdCoord.y);
            dom.sliderY.value = currentCmdCoord.y;
            updateCoordFromSlider("y", currentCmdCoord.y);
        }
    } else {
        dom.btnSelectRightArm.classList.add("active");
        dom.btnSelectLeftArm.classList.remove("active");
        dom.activeArmText.textContent = "RIGHT ARM";
        if (dom.armDofBadge) dom.armDofBadge.textContent = "7-DoF · Right Arm";
        if (currentCmdCoord.y > 0) {
            currentCmdCoord.y = -Math.abs(currentCmdCoord.y);
            dom.sliderY.value = currentCmdCoord.y;
            updateCoordFromSlider("y", currentCmdCoord.y);
        }
    }

    try {
        await fetch("/api/set_arm", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ arm: arm })
        });
        addLocalEventLog("CONFIG", `切换操作臂: ${arm}`, `已切换为 ${arm === "left_arm" ? "左臂 (Left Arm)" : "右臂 (Right Arm)"}`);
    } catch (e) {
        console.error("切换臂通信异常", e);
    }
}

function updateCoordFromSlider(axis, val) {
    currentCmdCoord[axis] = val;
    if (axis === "x") dom.valInputX.textContent = `${val.toFixed(3)} m`;
    if (axis === "y") dom.valInputY.textContent = `${val.toFixed(3)} m`;
    if (axis === "z") dom.valInputZ.textContent = `${val.toFixed(3)} m`;
}

// 供界面按钮快捷步进调动 (±1cm, ±5cm)
window.stepCoord = function(axis, delta) {
    let cur = currentCmdCoord[axis] + delta;
    const slider = dom[`slider${axis.toUpperCase()}`];
    const min = parseFloat(slider.min);
    const max = parseFloat(slider.max);
    cur = Math.max(min, Math.min(max, cur));
    cur = Math.round(cur * 1000) / 1000;
    slider.value = cur;
    updateCoordFromSlider(axis, cur);
};

// 供工况预设矩阵一键选用 (自动根据执行臂镜像 Y 轴)
window.applyPreset = function(x, y, z, presetName) {
    let targetY = y;
    if (currentArm === "right_arm") {
        targetY = -y;
    }
    dom.sliderX.value = x;
    dom.sliderY.value = targetY;
    dom.sliderZ.value = z;
    updateCoordFromSlider("x", x);
    updateCoordFromSlider("y", targetY);
    updateCoordFromSlider("z", z);
    // 立即向后端下发该工况
    dispatchTargetCommand(x, targetY, z, presetName);
};

async function dispatchTargetCommand(x, y, z, presetName) {
    try {
        dom.btnDispatchTarget.style.opacity = "0.7";
        dom.btnDispatchTarget.style.pointerEvents = "none";

        const resp = await fetch("/api/send_target", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                x: x,
                y: y,
                z: z,
                arm: currentArm,
                preset_name: presetName,
            })
        });
        const res = await resp.json();
        if (res.status === "DISPATCHED") {
            addLocalEventLog("COMMAND", `下达目标指令: ${presetName}`, `[${currentArm}] X: ${x.toFixed(3)}m, Y: ${y.toFixed(3)}m, Z: ${z.toFixed(3)}m (交由 IK 引擎求解)`);
        }
    } catch (e) {
        console.error("目标下发异常", e);
    } finally {
        dom.btnDispatchTarget.style.opacity = "1";
        dom.btnDispatchTarget.style.pointerEvents = "auto";
    }
}

// ─────────────────────────────────────────────────────────────
// 4. 前端通用事件监听
// ─────────────────────────────────────────────────────────────
function initEventListeners() {
    dom.btnResetCamera.addEventListener("click", () => {
        camera.position.set(2.05, -1.95, 1.15);
        controls.target.set(0.04, 0.0, 0.62);
        controls.update();
    });


    dom.btnToggleGrid.addEventListener("click", () => {
        if (scene.gridHelper) scene.gridHelper.visible = !scene.gridHelper.visible;
    });

    dom.btnResetStand.addEventListener("click", async () => {
        try {
            await fetch("/api/reset_stand", { method: "POST" });
            const readyY = currentArm === "left_arm" ? 0.212 : -0.212;
            currentCmdCoord.x = 0.111;
            currentCmdCoord.y = readyY;
            currentCmdCoord.z = 0.751;
            dom.sliderX.value = 0.111;
            dom.sliderY.value = readyY;
            dom.sliderZ.value = 0.751;
            updateCoordFromSlider("x", 0.111);
            updateCoordFromSlider("y", readyY);
            updateCoordFromSlider("z", 0.751);
            addLocalEventLog("COMMAND", "下达复位指令", "已发布标准对称微屈直立预备就绪指令");
        } catch (e) {
            console.error("复位失败", e);
        }
    });

    dom.btnCircleDemo.addEventListener("click", async () => {
        try {
            await fetch("/api/demo_circle", { method: "POST" });
            addLocalEventLog("COMMAND", "启动轨迹演示", "正在执行连续空间圆周平滑协同运动");
        } catch (e) {
            console.error("演示失败", e);
        }
    });
}

function addLocalEventLog(type, title, desc) {
    const stream = dom.logStream;
    if (!stream) return;
    const item = document.createElement("div");
    item.className = "log-item";
    const timeStr = new Date().toTimeString().split(" ")[0];
    item.innerHTML = `
        <span class="log-time">${timeStr}</span>
        <span class="log-badge ${type.toLowerCase()}">${type}</span>
        <div class="log-content">
            <div class="log-title">${title}</div>
            <div class="log-desc">${desc}</div>
        </div>
    `;
    stream.prepend(item);
    if (stream.children.length > 30) stream.lastElementChild.remove();
}

// ─────────────────────────────────────────────────────────────
// 5. SSE 高频遥测推送流与仪表盘数据绑定
// ─────────────────────────────────────────────────────────────
function connectSSE() {
    const sse = new EventSource("/api/stream");

    sse.onopen = () => {
        dom.connectionStatus.textContent = "DDS LINK: LIVE";
        dom.connectionChip.className = "status-chip chip-online";
        if (dom.footerStatusText) {
            dom.footerStatusText.textContent = "Passive Telemetry: Listening on /joint_states (50Hz) · ROS 2 IK Channel Connected";
        }
    };

    sse.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            updateDashboardUI(data);
        } catch (err) {
            console.error("遥测数据解析错误", err);
        }
    };

    sse.onerror = () => {
        dom.connectionStatus.textContent = "DDS LINK: RECONNECTING";
        dom.connectionChip.className = "status-chip chip-offline";
        if (dom.footerStatusText) {
            dom.footerStatusText.textContent = "Passive Telemetry: Reconnecting to ROS 2 Telemetry Stream...";
        }
    };
}

function updateDashboardUI(data) {
    // 1. 笛卡尔空间位姿坐标渲染
    const target = data.cartesian_cmd_pose || data.target || [0.111, 0.212, 0.751];
    const actual = data.cartesian_actual_pose || data.actual || [0.111, 0.212, 0.751];
    const delta = data.spatial_delta_mm || data.delta_mm || [0.0, 0.0, 0.0];
    const errNorm = data.euclidean_error_norm_mm !== undefined ? data.euclidean_error_norm_mm : (data.pos_err_mm || 0.0);

    dom.tgtX.textContent = (target[0] >= 0 ? "+" : "") + target[0].toFixed(3);
    dom.tgtY.textContent = (target[1] >= 0 ? "+" : "") + target[1].toFixed(3);
    dom.tgtZ.textContent = (target[2] >= 0 ? "+" : "") + target[2].toFixed(3);

    dom.actX.textContent = (actual[0] >= 0 ? "+" : "") + actual[0].toFixed(3);
    dom.actY.textContent = (actual[1] >= 0 ? "+" : "") + actual[1].toFixed(3);
    dom.actZ.textContent = (actual[2] >= 0 ? "+" : "") + actual[2].toFixed(3);

    // 空间 3 轴增量残差
    dom.deltaX.textContent = `${(delta[0] >= 0 ? "+" : "") + delta[0].toFixed(1)} mm`;
    dom.deltaY.textContent = `${(delta[1] >= 0 ? "+" : "") + delta[1].toFixed(1)} mm`;
    dom.deltaZ.textContent = `${(delta[2] >= 0 ? "+" : "") + delta[2].toFixed(1)} mm`;

    // 跟踪残差范数与状态指标
    if (dom.posErrNum) {
        dom.posErrNum.textContent = errNorm.toFixed(2);
        if (errNorm < 2.0) {
            dom.posErrNum.className = "metric-val val-green";
        } else if (errNorm < 15.0) {
            dom.posErrNum.className = "metric-val val-gold";
        } else {
            dom.posErrNum.className = "metric-val val-red";
        }
    }
    if (dom.posErrBar) {
        const errBarPct = Math.min(100, (errNorm / 20.0) * 100);
        dom.posErrBar.style.width = `${errBarPct}%`;
        if (errNorm < 2.0) {
            dom.posErrBar.className = "progress-fill green";
        } else if (errNorm < 15.0) {
            dom.posErrBar.className = "progress-fill gold";
        } else {
            dom.posErrBar.className = "progress-fill red";
        }
    }

    // 2. 遥测 DDS 链路频率与报文数
    const link = data.telemetry_link || data.source || {};
    if (dom.streamHzNum) dom.streamHzNum.textContent = (link.dds_rate_hz || link.hz || 0).toFixed(1);
    if (dom.packetCountText) dom.packetCountText.textContent = `报文吞吐: ${link.ingress_packets || link.msg_count || 0} pkts`;

    // 3. 3D 光球与连线实时更新
    if (targetOrb) targetOrb.position.set(target[0], target[1], target[2]);
    if (actualOrb) actualOrb.position.set(actual[0], actual[1], actual[2]);

    if (errorLine) {
        if (errNorm < 3.0) {
            errorLine.visible = false;
        } else {
            errorLine.visible = true;
            const positions = errorLine.geometry.attributes.position.array;
            positions[0] = target[0]; positions[1] = target[1]; positions[2] = target[2];
            positions[3] = actual[0]; positions[4] = actual[1]; positions[5] = actual[2];
            errorLine.geometry.attributes.position.needsUpdate = true;
        }
    }

    // 4. 宇树 G1 官方 36 连杆 CAD STL 实时更新
    if (data.visuals) {
        updateRobotCADModel(data.visuals);
    }

    // 5. 全域 28 对安全碰撞雷达
    const minClearance = data.global_min_clearance_mm !== undefined ? data.global_min_clearance_mm : (data.min_clearance_mm || 35.0);
    dom.minClearanceVal.textContent = minClearance.toFixed(1);

    // 动态安全告警颜色
    if (minClearance < 10.0) {
        dom.minClearanceVal.style.color = "var(--alert-red)";
        dom.minClearanceVal.style.textShadow = "0 0 12px rgba(255, 51, 102, 0.7)";
        dom.clearanceMarker.style.background = "var(--alert-red)";
        dom.clearanceMarker.style.boxShadow = "0 0 10px var(--alert-red)";
    } else if (minClearance < 22.0) {
        dom.minClearanceVal.style.color = "var(--golden-yellow)";
        dom.minClearanceVal.style.textShadow = "0 0 10px rgba(255, 184, 0, 0.6)";
        dom.clearanceMarker.style.background = "var(--golden-yellow)";
        dom.clearanceMarker.style.boxShadow = "0 0 8px var(--golden-yellow)";
    } else {
        dom.minClearanceVal.style.color = "var(--laser-green)";
        dom.minClearanceVal.style.textShadow = "0 0 10px rgba(0, 255, 136, 0.5)";
        dom.clearanceMarker.style.background = "var(--laser-green)";
        dom.clearanceMarker.style.boxShadow = "0 0 8px var(--laser-green)";
    }

    // 净空数值着色与指示标尺 (0 ~ 50mm 映射为 0% ~ 100%)
    const markerPct = Math.max(0, Math.min(100, (minClearance / 50.0) * 100));
    dom.clearanceMarker.style.left = `${markerPct}%`;

    if (minClearance < 5.0) {
        dom.minClearanceVal.className = "clearance-val val-alert";
    } else if (minClearance < 20.0) {
        dom.minClearanceVal.className = "clearance-val val-warn";
    } else {
        dom.minClearanceVal.className = "clearance-val val-safe";
    }

    const isColliding = data.is_colliding || false;
    if (isColliding) {
        dom.radarBadge.textContent = "COLLISION";
        dom.radarBadge.className = "radar-status-badge alert";
        dom.globalCollisionChip.className = "status-chip chip-alert";
        dom.globalCollisionText.textContent = `COLLISION (${(data.colliding_pairs || []).length})`;
        dom.collisionAlertBox.style.display = "block";
        dom.alertPairsList.innerHTML = (data.colliding_pairs || []).map(p => `<li>${p[0]} ⚔️ ${p[1]} (${p[2]})</li>`).join("");
    } else if (minClearance < 5.0) {
        dom.radarBadge.textContent = "ALERT (<5mm)";
        dom.radarBadge.className = "radar-status-badge alert";
        dom.globalCollisionChip.className = "status-chip chip-alert";
        dom.globalCollisionText.textContent = `NEAR COLLISION (${minClearance.toFixed(1)}mm)`;
        dom.collisionAlertBox.style.display = "none";
    } else if (minClearance < 20.0) {
        dom.radarBadge.textContent = "CAUTION";
        dom.radarBadge.className = "radar-status-badge warn";
        dom.globalCollisionChip.className = "status-chip chip-gold";
        dom.globalCollisionText.textContent = "CAUTION (<20mm)";
        dom.collisionAlertBox.style.display = "none";
    } else {
        dom.radarBadge.textContent = "SECURE";
        dom.radarBadge.className = "radar-status-badge safe";
        dom.globalCollisionChip.className = "status-chip chip-safe";
        dom.globalCollisionText.textContent = "SECURE (0 Collisions)";
        dom.collisionAlertBox.style.display = "none";
    }

    // 4 大关键区域切片动态更新 (直接使用 arm_torso 等标准分类键)
    const zDists = data.zone_clearances || data.zone_clearance_mm || {};
    const dTorso = zDists.arm_torso !== undefined ? zDists.arm_torso : (zDists.torso_arm !== undefined ? zDists.torso_arm : 50.0);
    const dArm = zDists.arm_arm !== undefined ? zDists.arm_arm : 50.0;
    const dHead = zDists.arm_head !== undefined ? zDists.arm_head : (zDists.head_arm !== undefined ? zDists.head_arm : 50.0);
    const dLeg = zDists.arm_leg !== undefined ? zDists.arm_leg : (zDists.leg_arm !== undefined ? zDists.leg_arm : 50.0);

    updateZoneCard("zoneTorso", dTorso, "Arm - Torso", "12 pairs");
    updateZoneCard("zoneArm", dArm, "Inter-Arm", "4 pairs");
    updateZoneCard("zoneHead", dHead, "Arm - Head", "4 pairs");
    updateZoneCard("zoneLeg", dLeg, "Arm - Legs", "8 pairs");

    // 6. 航天级 10-DOF 双向中心对称关节遥测仪表渲染 (直接呈现原始关节名)
    renderAerospaceJoints(dom.waistJointsList, data.waist_joints || data.waist_telemetry || []);
    renderAerospaceJoints(dom.armJointsList, data.arm_joints || data.arm_telemetry || []);

    // 7. 测控日志流同步
    if (data.activity_logs && data.activity_logs.length > 0) {
        renderActivityLogs(data.activity_logs);
    }

    // 8. IK 求解收敛动力学曲线渲染
    if (data.solver_diagnostics) {
        renderConvergenceTrace(data.solver_diagnostics);
    }
}

function updateZoneCard(elemId, distMm, titleName, defaultCount) {
    const el = document.getElementById(elemId);
    if (!el) return;
    const dist = distMm !== undefined ? distMm : 50.0;
    const isAlert = dist < 5.0;
    const isWarn = dist >= 5.0 && dist < 20.0;

    let statusText = `${defaultCount} · ${dist.toFixed(1)}mm`;
    if (isAlert) {
        el.className = "zone-card alert";
        statusText += " · ALERT";
    } else if (isWarn) {
        el.className = "zone-card warn";
        statusText += " · CAUTION";
    } else {
        el.className = "zone-card safe";
        statusText += " · SECURE";
    }
    el.innerHTML = `
        <span class="zone-name">${titleName}</span>
        <span class="zone-status">${statusText}</span>
    `;
}


// 航天级双向中心对称仪表渲染器 (突出显示物理限位与余量告警)
function renderAerospaceJoints(container, joints) {
    if (!container || !joints) return;

    joints.forEach((j, idx) => {
        let card = container.children[idx];
        if (!card) {
            card = document.createElement("div");
            card.className = "joint-card";
            container.appendChild(card);
        }

        const isPositive = j.deg >= 0;
        const absOffset = Math.abs(j.offset_pct || 0);
        const barWidth = Math.min(50, absOffset * 0.5); // 0%~100% 映射为半边 0%~50%

        const degStr = (j.deg >= 0 ? "+" : "") + j.deg.toFixed(1) + "°";
        const radStr = (j.rad >= 0 ? "+" : "") + j.rad.toFixed(3) + " rad";
        const labelName = j.label || j.name;
        const symbol = j.symbol ? ` [${j.symbol}]` : "";

        // 关节物理限位安全余量分析 (Margin to physical limits)
        const marginMin = Math.abs(j.deg - j.min_deg);
        const marginMax = Math.abs(j.max_deg - j.deg);
        const margin = Math.min(marginMin, marginMax);
        const isAlert = margin < 6.0;
        const isWarn = !isAlert && margin < 15.0;

        card.className = "joint-card" + (isAlert ? " limit-alert" : isWarn ? " limit-warn" : "");

        const limitBadgeHtml = isAlert 
            ? `<span class="limit-status-badge alert">LIMIT ALERT</span>`
            : isWarn 
            ? `<span class="limit-status-badge warn">NEAR LIMIT</span>`
            : `<span class="joint-limit-tag">LIM [${j.min_deg.toFixed(0)}° ~ ${j.max_deg.toFixed(0)}°]</span>`;

        card.innerHTML = `
            <div class="joint-head">
                <div class="joint-title-wrap">
                    <span class="joint-label-title">${labelName}</span>
                    <span class="joint-symbol">${symbol}</span>
                    ${limitBadgeHtml}
                </div>
                <div class="joint-num-group">
                    <span class="joint-num-deg ${isAlert ? 'text-alert' : isWarn ? 'text-warn' : ''}">${degStr}</span>
                    <span class="joint-num-rad">${radStr}</span>
                </div>
            </div>
            <div class="joint-track-container">
                <div class="bidi-bar-track">
                    <div class="hard-stop-tick left" title="Hard limit min: ${j.min_deg}°"></div>
                    <div class="hard-stop-tick right" title="Hard limit max: ${j.max_deg}°"></div>
                    <div class="bidi-center-marker"></div>
                    <div class="bidi-bar-fill ${isPositive ? 'positive' : 'negative'} ${isAlert ? 'bar-limit-alert' : isWarn ? 'bar-limit-warn' : ''}" 
                         style="${isPositive ? `left: 50%; width: ${barWidth}%;` : `left: ${50 - barWidth}%; width: ${barWidth}%;`}">
                    </div>
                </div>
            </div>
            <div class="joint-foot">
                <span class="limit-bound min">MIN ${j.min_deg > 0 ? '+' : ''}${j.min_deg.toFixed(0)}°</span>
                <span class="margin-text ${isAlert ? 'text-alert' : isWarn ? 'text-warn' : ''}">
                    ${isAlert ? '🚨 MARGIN' : isWarn ? '⚠️ MARGIN' : 'MARGIN'}: ${margin.toFixed(1)}° (${(j.offset_pct >= 0 ? '+' : '') + j.offset_pct.toFixed(0)}%)
                </span>
                <span class="limit-bound max">MAX ${j.max_deg > 0 ? '+' : ''}${j.max_deg.toFixed(0)}°</span>
            </div>
        `;
    });
}

function renderActivityLogs(logs) {
    const stream = dom.logStream;
    if (!stream) return;
    stream.innerHTML = logs.slice().reverse().slice(0, 25).map(item => `
        <div class="log-item">
            <span class="log-time">${item.time}</span>
            <span class="log-badge ${(item.type || 'INFO').toLowerCase()}">${item.type || 'INFO'}</span>
            <div class="log-content">
                <div class="log-title">${item.title}</div>
                <div class="log-desc">${item.desc}</div>
            </div>
        </div>
    `).join("");
}

// ─────────────────────────────────────────────────────────────
// 8. IK 求解收敛动力学曲线交互与高质感 Canvas 渲染引擎
// ─────────────────────────────────────────────────────────────
let currentTraceData = null;
let hoverTraceIndex = -1;

function initConvergenceCanvasEvents() {
    const canvas = dom.convergenceCanvas;
    if (!canvas) return;

    canvas.addEventListener("mousemove", (e) => {
        if (!currentTraceData || !currentTraceData.convergence_trace || currentTraceData.convergence_trace.length === 0) {
            return;
        }
        const rect = canvas.getBoundingClientRect();
        const mouseX = e.clientX - rect.left;
        const padLeft = 32;
        const padRight = 12;
        const plotW = rect.width - padLeft - padRight;
        const trace = currentTraceData.convergence_trace;

        if (mouseX < padLeft || mouseX > padLeft + plotW) {
            hoverTraceIndex = -1;
            if (dom.traceTooltip) dom.traceTooltip.style.display = "none";
            drawTraceCanvas(currentTraceData);
            return;
        }

        const ratio = (mouseX - padLeft) / plotW;
        hoverTraceIndex = Math.round(ratio * (trace.length - 1));
        hoverTraceIndex = Math.max(0, Math.min(trace.length - 1, hoverTraceIndex));

        // 浮动 Tooltip 显示当前步残差与种子
        if (dom.traceTooltip) {
            const errVal = trace[hoverTraceIndex];
            const getX = padLeft + (hoverTraceIndex / Math.max(1, trace.length - 1)) * plotW;
            dom.traceTooltip.style.display = "block";
            dom.traceTooltip.style.left = `${getX}px`;

            let activeSeed = 0;
            const switches = currentTraceData.seed_switches || [0];
            const names = currentTraceData.seed_names || [];
            for (let i = 0; i < switches.length; i++) {
                if (hoverTraceIndex >= switches[i]) activeSeed = i;
            }
            const sName = names[activeSeed] || `Seed #${activeSeed}`;
            dom.traceTooltip.innerHTML = `Step ${hoverTraceIndex + 1}: <strong>${errVal.toFixed(1)}mm</strong> <span style="opacity:0.7">(${sName})</span>`;
        }

        drawTraceCanvas(currentTraceData);
    });

    canvas.addEventListener("mouseleave", () => {
        hoverTraceIndex = -1;
        if (dom.traceTooltip) dom.traceTooltip.style.display = "none";
        if (currentTraceData) drawTraceCanvas(currentTraceData);
    });

    window.addEventListener("resize", () => {
        if (currentTraceData) drawTraceCanvas(currentTraceData);
    });
}

function renderConvergenceTrace(diagnostics) {
    currentTraceData = diagnostics;
    const timeMs = diagnostics.time_ms !== undefined ? diagnostics.time_ms : 0.0;
    const iters = diagnostics.iters !== undefined ? diagnostics.iters : 0;
    const posErr = diagnostics.pos_err_mm !== undefined ? diagnostics.pos_err_mm : 0.0;
    const seedUsed = diagnostics.seed_used !== undefined ? diagnostics.seed_used : 0;
    const seedName = diagnostics.seed_name || (seedUsed >= 0 ? `Seed #${seedUsed}` : "None");
    const mode = diagnostics.mode || "10DOF_WEIGHTED_DLS";

    if (dom.convergenceModeBadge) dom.convergenceModeBadge.textContent = mode.replace(/_/g, " ");
    if (dom.traceTimeVal) dom.traceTimeVal.textContent = `${timeMs.toFixed(1)} ms`;
    if (dom.traceItersVal) dom.traceItersVal.textContent = `${iters}`;
    if (dom.traceSeedVal) dom.traceSeedVal.textContent = seedName;
    if (dom.traceResidueVal) {
        dom.traceResidueVal.textContent = `${posErr.toFixed(2)} mm`;
        dom.traceResidueVal.className = posErr < 2.5 ? "stat-val val-green" : (posErr < 15.0 ? "stat-val val-gold" : "stat-val val-red");
    }
    if (dom.traceStatusTag) {
        if (iters === 0) {
            dom.traceStatusTag.textContent = "STANDBY";
            dom.traceStatusTag.className = "trace-status-tag safe";
        } else if (posErr < 2.5) {
            dom.traceStatusTag.textContent = "CONVERGED";
            dom.traceStatusTag.className = "trace-status-tag safe";
        } else if (posErr < 15.0) {
            dom.traceStatusTag.textContent = "APPROXIMATE";
            dom.traceStatusTag.className = "trace-status-tag warn";
        } else {
            dom.traceStatusTag.textContent = "LOCAL MIN";
            dom.traceStatusTag.className = "trace-status-tag alert";
        }
    }

    drawTraceCanvas(diagnostics);
}

function drawTraceCanvas(diag) {
    const canvas = dom.convergenceCanvas;
    if (!canvas) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    const width = rect.width || 320;
    const height = rect.height || 110;

    if (canvas.width !== Math.round(width * dpr) || canvas.height !== Math.round(height * dpr)) {
        canvas.width = Math.round(width * dpr);
        canvas.height = Math.round(height * dpr);
    }

    const ctx = canvas.getContext("2d");
    ctx.save();
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, width, height);

    const padLeft = 32;
    const padRight = 12;
    const padTop = 14;
    const padBottom = 20;
    const plotW = Math.max(10, width - padLeft - padRight);
    const plotH = Math.max(10, height - padTop - padBottom);

    const trace = (diag && diag.convergence_trace) ? diag.convergence_trace : [];

    // 背景参考网格线
    ctx.strokeStyle = "rgba(255, 255, 255, 0.05)";
    ctx.lineWidth = 1.0;
    ctx.beginPath();
    ctx.moveTo(padLeft, padTop);
    ctx.lineTo(padLeft + plotW, padTop);
    ctx.moveTo(padLeft, padTop + plotH / 2);
    ctx.lineTo(padLeft + plotW, padTop + plotH / 2);
    ctx.moveTo(padLeft, padTop + plotH);
    ctx.lineTo(padLeft + plotW, padTop + plotH);
    ctx.stroke();

    if (!trace || trace.length === 0) {
        ctx.fillStyle = "rgba(100, 150, 190, 0.4)";
        ctx.font = "10px Inter, monospace";
        ctx.textAlign = "center";
        ctx.fillText("Standby (Ready for next IK dispatch)", width / 2, height / 2 + 3);
        ctx.restore();
        return;
    }

    const maxErr = Math.max(15.0, Math.max(...trace) * 1.06);

    const getX = (idx) => padLeft + (idx / Math.max(1, trace.length - 1)) * plotW;
    const getY = (val) => padTop + plotH - (Math.min(maxErr, Math.max(0, val)) / maxErr) * plotH;

    // 绘制容差门限线 (pos_tol = 2.0 mm)
    const tolY = getY(2.0);
    ctx.save();
    ctx.setLineDash([4, 3]);
    ctx.strokeStyle = "rgba(255, 184, 0, 0.7)";
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    ctx.moveTo(padLeft, tolY);
    ctx.lineTo(padLeft + plotW, tolY);
    ctx.stroke();
    ctx.restore();

    // 刻度文本
    ctx.fillStyle = "rgba(255, 184, 0, 0.85)";
    ctx.font = "9px 'JetBrains Mono', monospace";
    ctx.textAlign = "right";
    ctx.fillText("2mm", padLeft - 4, tolY + 3);

    ctx.fillStyle = "rgba(140, 160, 180, 0.5)";
    ctx.fillText(`${Math.round(maxErr)}m`, padLeft - 4, padTop + 9);
    ctx.fillText("0", padLeft - 4, padTop + plotH);

    // X 轴刻度
    ctx.textAlign = "left";
    ctx.fillText("it:1", padLeft, height - 4);
    ctx.textAlign = "right";
    ctx.fillText(`it:${trace.length}`, padLeft + plotW, height - 4);

    // 多种子划分虚线 (Multi-start Seed Dividers)
    const switches = diag.seed_switches || [0];
    if (switches.length > 1) {
        switches.slice(1).forEach((swIdx, i) => {
            if (swIdx < trace.length) {
                const sx = getX(swIdx);
                ctx.save();
                ctx.setLineDash([2, 3]);
                ctx.strokeStyle = "rgba(0, 240, 255, 0.4)";
                ctx.lineWidth = 1;
                ctx.beginPath();
                ctx.moveTo(sx, padTop);
                ctx.lineTo(sx, padTop + plotH);
                ctx.stroke();

                ctx.fillStyle = "rgba(0, 240, 255, 0.7)";
                ctx.font = "8px 'JetBrains Mono', monospace";
                ctx.textAlign = "center";
                ctx.fillText(`S${i + 1}`, sx, padTop - 3);
                ctx.restore();
            }
        });
    }

    // 渐变面积填充
    const areaGrad = ctx.createLinearGradient(0, padTop, 0, padTop + plotH);
    areaGrad.addColorStop(0, "rgba(0, 240, 255, 0.22)");
    areaGrad.addColorStop(0.7, "rgba(0, 255, 136, 0.08)");
    areaGrad.addColorStop(1, "rgba(0, 255, 136, 0.01)");

    ctx.beginPath();
    ctx.moveTo(getX(0), padTop + plotH);
    for (let i = 0; i < trace.length; i++) {
        ctx.lineTo(getX(i), getY(trace[i]));
    }
    ctx.lineTo(getX(trace.length - 1), padTop + plotH);
    ctx.closePath();
    ctx.fillStyle = areaGrad;
    ctx.fill();

    // 核心收敛折线
    ctx.save();
    ctx.beginPath();
    for (let i = 0; i < trace.length; i++) {
        const px = getX(i);
        const py = getY(trace[i]);
        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
    }
    ctx.strokeStyle = "#00f0ff";
    ctx.lineWidth = 2.0;
    ctx.shadowColor = "rgba(0, 240, 255, 0.6)";
    ctx.shadowBlur = 5;
    ctx.stroke();
    ctx.restore();

    // 起点与终点圆点
    const startX = getX(0);
    const startY = getY(trace[0]);
    ctx.beginPath();
    ctx.arc(startX, startY, 2.5, 0, Math.PI * 2);
    ctx.fillStyle = "#00f0ff";
    ctx.fill();

    const lastX = getX(trace.length - 1);
    const lastY = getY(trace[trace.length - 1]);
    ctx.beginPath();
    ctx.arc(lastX, lastY, 3.5, 0, Math.PI * 2);
    ctx.fillStyle = "#00ff88";
    ctx.shadowColor = "rgba(0, 255, 136, 0.8)";
    ctx.shadowBlur = 8;
    ctx.fill();

    // 悬停光标指示器 (Hover Inspection Marker)
    if (hoverTraceIndex >= 0 && hoverTraceIndex < trace.length) {
        const hx = getX(hoverTraceIndex);
        const hy = getY(trace[hoverTraceIndex]);

        ctx.save();
        ctx.strokeStyle = "rgba(255, 255, 255, 0.4)";
        ctx.setLineDash([2, 2]);
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(hx, padTop);
        ctx.lineTo(hx, padTop + plotH);
        ctx.stroke();

        ctx.beginPath();
        ctx.arc(hx, hy, 4.5, 0, Math.PI * 2);
        ctx.fillStyle = "#ffffff";
        ctx.shadowColor = "rgba(0, 240, 255, 0.9)";
        ctx.shadowBlur = 10;
        ctx.fill();
        ctx.restore();
    }

    ctx.restore();
}
