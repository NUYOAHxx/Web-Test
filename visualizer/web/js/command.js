/**
 * 目标指令下发面板: 操作臂选择 / 6D 位姿滑块 / 工况预设 / 指令下发。
 *
 * 重构要点:
 *  · 工况预设改为 **数据驱动 + data-preset 属性**, 不再用 HTML 内联 onclick。
 *    内联 onclick 依赖全局函数名, 在 ES modules 下模块作用域不再挂到 window,
 *    全部预设按钮会静默失效。
 *  · 右臂的 Y / Roll / Yaw 镜像逻辑集中在 state.js, 此处不再散落判断。
 */

import { el, requireAll, setText, setClass } from "./dom.js";
import { fixed, signed, rpyToQuat } from "./format.js";
import { state, mirrorY, mirrorRpy } from "./state.js";
import { buildTargetPayload, panelToWorldPosition, panelToWorldRpy } from "./target_payload.js";
import { pushLocalLog } from "./panels/logs.js";

/**
 * 滑块轴定义: [键, 滑块 id, 数值输入框 id, 小数位]
 *
 * 每根滑块配一个可键入的数值框。三种设定方式并存, 各管一段精度:
 *   · 拖动        粗调, 一次跨越整个量程
 *   · Shift + 拖动 细调, 1/5 步长, 用于逼近具体数值
 *   · 滚轮        精调, 每次 ±1 步长
 *   · 数值框      直接键入, 无级误差
 *
 * 为什么需要细调: 侧栏内每根滑块仅约 60px 宽, 0.4m 量程下每像素 6.7mm,
 * 纯拖动的分辨率远粗于一个自然步长 (5mm), 想停在具体数值上只能靠运气。
 */
const AXES = [
    ["x", "sliderX", "numX", 3],
    ["y", "sliderY", "numY", 3],
    ["z", "sliderZ", "numZ", 3],
    ["roll", "sliderRoll", "numRoll", 1],
    ["pitch", "sliderPitch", "numPitch", 1],
    ["yaw", "sliderYaw", "numYaw", 1],
];

/** 刻度位数: 位置米制保留 3 位, 姿态角度 1 位 */
const DIGITS = { x: 3, y: 3, z: 3, roll: 1, pitch: 1, yaw: 1 };

/**
 * 已应用的包络签名 —— 避免每帧重复写入滑块属性。
 *
 * `model_info` 在增量推送下会被合并进累积态并**每帧都存在**, 因此本函数
 * 实际以 60Hz 被调用。若不加签名判断, 每帧要写 12 个 DOM 元素的
 * min/max/step, 纯属无谓开销 —— 这些值对同一模型恒定不变。
 */
let appliedEnvelopeKey = "";

/**
 * 按服务端实测的可达包络设置滑块范围。
 *
 * 范围原先写死在 HTML 的 min/max 里, 与服务端的校验字典各存一份, 两处无关联:
 * 改一边不改另一边就会出现"滑块能选但后端拒绝", 换 URDF 后数值也不会变。
 * 现在 min/max 与后端校验读同一份 `model_info.reach_envelope`, 不存在分叉。
 *
 * 滑块总像素宽度不变, 因此量程拉长后每像素代表的距离变大; 步长按量程比例
 * 缩放, 保持"一次滚轮 ≈ 一个自然步长"的精度直觉。
 *
 * @param {object} info 服务端 model_info
 * @returns {boolean} 范围是否发生变化 (未变化时不做任何写入)
 */
export function applyReachEnvelope(info) {
    const envelopes = info?.reach_envelope;
    if (!envelopes) return false;

    // 右臂优先用右臂包络; 未下发该臂时退回左臂
    const env = envelopes[state.arm] || envelopes.left_arm;
    if (!env) return false;

    // 签名含臂名: 切换操作臂后包络不同, 必须重新应用
    const key = `${state.arm}|${AXES.map(([a]) => `${a}:${env[a]?.min}:${env[a]?.max}`).join(",")}`;
    if (key === appliedEnvelopeKey) return false;
    appliedEnvelopeKey = key;

    for (const [axis, sliderId, numId] of AXES) {
        const box = env[axis];
        if (!box) continue;
        const slider = el(sliderId);
        const num = el(numId);
        if (!slider || !num) continue;

        const lo = Number(box.min);
        const hi = Number(box.max);
        const span = Math.max(1e-6, hi - lo);

        slider.min = String(lo);
        slider.max = String(hi);
        // 姿态量程可达 300°, 位置仅约 0.5m; 步长按各自量程的 1/300 取,
        // 使两侧的"一格"手感一致
        num.min = String(lo);
        num.max = String(hi);
        slider.step = num.step = String(Math.max(1e-4, Math.round(span / 300 * 1e4) / 1e4));

        // 当前值若落在新范围外, 夹到边界, 避免滑块位置与数值框不一致
        const cur = Number(slider.value);
        if (!Number.isFinite(cur) || cur < lo || cur > hi) {
            const clamped = Math.min(hi, Math.max(lo, Number.isFinite(cur) ? cur : (lo + hi) / 2));
            slider.value = String(clamped);
            num.value = clamped.toFixed(DIGITS[axis]);
        }
    }
    return true;
}

/**
 * 工况预设。
 * 坐标按 **左臂** 约定给出 (y 为正), 右臂在应用时自动镜像。
 */
const PRESETS = {
    standby: { x: 0.111, y: 0.212, z: 0.751, rpy: [0, 0, 0], name: "待命位姿" },
    table: { x: 0.35, y: 0.22, z: 0.85, rpy: [0, 20, 0], name: "桌面取物" },
    far: { x: 0.45, y: 0.20, z: 0.85, rpy: [0, 0, 0], name: "远伸取物" },
    low: { x: 0.38, y: 0.18, z: 0.75, rpy: [0, 45, 0], name: "低位拾取" },
    side: { x: 0.25, y: 0.30, z: 0.82, rpy: [15, 0, 30], name: "侧向取物" },
    cross: { x: 0.28, y: -0.15, z: 0.88, rpy: [-20, 10, -15], name: "胸前交接" },
};

/**
 * 待发目标预览回调。由 main.js 注册, 在滑块变动时把位置交给视口渲染,
 * 使"先看见位姿、再决定是否下发"成为可能 —— 大屏是观测工具, 不该要求用户
 * 靠侧栏数字想象 3D 位置。
 * @type {((pos:number[]|null, quat:number[]|null)=>void)|null}
 */
let onPendingPreview = null;

/** 注册待发预览回调 @param {(pos:number[]|null, quat:number[]|null)=>void} fn */
export function setPendingPreviewHandler(fn) { onPendingPreview = fn; }


/**
 * 把当前面板上的待发位姿推给视口。
 *
 * 仅在**面板值与机器人实际收到的指令不一致**时才显示预览 —— 一旦下发成功
 * 或遥测回流一致, 预览自动消失, 避免与实心的"已下发目标"标记重影。
 *
 * 右臂时 Y 与部分姿态角需镜像, 否则预览会出现在错误的一侧。
 */
function notifyPendingPreview() {
    if (!onPendingPreview) return;
    const c = state.cmd;
    if (!state.cmdPending) {
        onPendingPreview(null, null);
        return;
    }
    // state.cmd 里姿态是 roll/pitch/yaw 三个独立字段, 不是 rpy 数组。
    // 换算复用 target_payload —— 预览与下发必须走同一条规则, 否则会出现
    // "预览画在一处、机器人走向另一处", 且无任何报错。
    onPendingPreview(
        panelToWorldPosition(c, state.arm),
        rpyToQuat(panelToWorldRpy([c.roll ?? 0, c.pitch ?? 0, c.yaw ?? 0], state.arm)),
    );
}

/**
 * 切臂回调。由 main.js 注册, 用于让滑块量程与视口线框盒同步换到新操作臂的
 * 实测可达范围 —— 两者由 main.js 的 applyReachBox() 统一刷新, 保证同源。
 * @type {(()=>void)|null}
 */
let onArmChanged = null;

/** 注册切臂回调 @param {()=>void} fn */
export function setArmChangeHandler(fn) { onArmChanged = fn; }

/** 标准直立预备位姿 */
const STAND_POSE = { x: 0.111, y: 0.212, z: 0.751, rpy: [0, 0, 0] };

/**
 * 初始化指令面板。
 */
export function initCommandPanel() {
    requireAll(AXES.flatMap(([, sliderId, numId]) => [sliderId, numId]));

    bindArmToggle();
    bindSliders();
    bindNumericInputs();
    bindPresets();
    bindDispatch();
    bindSliderInteractionGuard();

    updateArmUi();
}

function bindArmToggle() {
    el("btnSelectLeftArm").addEventListener("click", () => selectArm("left_arm"));
    el("btnSelectRightArm").addEventListener("click", () => selectArm("right_arm"));
}

function bindSliders() {
    for (const [key, sliderId, numId] of AXES) {
        const slider = el(sliderId);
        slider.addEventListener("input", (e) => {
            const v = parseFloat(e.target.value);
            if (Number.isFinite(v)) {
                setAxis(key, v);
                setPending(true);
            }
        });
        bindFineDrag(key, slider, numId);
        bindWheelAdjust(key, slider);
    }
}

/**
 * Shift + 拖动 = 细调 (1/5 步长)。
 *
 * 必须自己接管指针事件: 原生 range 控件只能按 step 量化, 无法提供
 * 比 step 更细的拖动精度, 因此按住 Shift 时 preventDefault 掉原生行为,
 * 改为按指针位置自行换算并量化到 1/5 步长。
 *
 * @param {string} key
 * @param {HTMLInputElement} slider
 * @param {string} numId
 */
function bindFineDrag(key, slider, numId) {
    slider.addEventListener("pointerdown", (e) => {
        if (!e.shiftKey || e.button !== 0) return;
        // 阻止原生拖动接管本次交互
        e.preventDefault();
        state.userDraggingSlider = true;
        slider.classList.add("fine-drag");

        const rect = slider.getBoundingClientRect();
        const min = parseFloat(slider.min);
        const max = parseFloat(slider.max);
        const coarse = parseFloat(slider.step) || 0.005;
        const fine = coarse / 5;

        const apply = (clientX) => {
            const ratio = (clientX - rect.left) / rect.width;
            const raw = min + Math.max(0, Math.min(1, ratio)) * (max - min);
            const snapped = Math.round(raw / fine) * fine;
            setAxis(key, Math.max(min, Math.min(max, Number(snapped.toFixed(6)))));
            setPending(true);
        };

        apply(e.clientX);

        const onMove = (ev) => apply(ev.clientX);
        const onUp = () => {
            document.removeEventListener("pointermove", onMove);
            document.removeEventListener("pointerup", onUp);
            slider.classList.remove("fine-drag");
            state.userDraggingSlider = false;
        };
        document.addEventListener("pointermove", onMove);
        document.addEventListener("pointerup", onUp);
    });
}

/**
 * 滚轮 = ±1 步长精调。
 *
 * 仅在"本次滚动确实会改变数值"时才阻止默认行为 —— 已经顶到量程端点时
 * 让滚轮正常滚页面, 避免在页面其他区域悬停时劫持滚动。
 *
 * @param {string} key
 * @param {HTMLInputElement} slider
 */
function bindWheelAdjust(key, slider) {
    slider.addEventListener("wheel", (e) => {
        const min = parseFloat(slider.min);
        const max = parseFloat(slider.max);
        const step = parseFloat(slider.step) || 0.005;
        const cur = state.cmd[key];
        const next = round6(cur + (e.deltaY > 0 ? -step : step));
        if (next < min || next > max) return;   // 端点处放行, 不劫持滚动
        e.preventDefault();
        setAxis(key, next);
        setPending(true);
    }, { passive: false });
}

function round6(v) {
    return Math.round(v * 1e6) / 1e6;
}

/**
 * 数值框直接输入。
 * 使用 "change" 而非 "input": 输入过程中 (如键入 0.38 的中间态 0.3)
 * 就下发会不断覆盖用户正在编辑的内容。
 */
function bindNumericInputs() {
    for (const [key, , numId] of AXES) {
        el(numId).addEventListener("change", (e) => {
            const raw = parseFloat(e.target.value);
            if (!Number.isFinite(raw)) {
                // 输入非法内容时回滚到当前值, 不让面板与实际下发值脱节
                e.target.value = state.cmd[key];
                return;
            }
            setAxis(key, raw);
            setPending(true);
        });
        // 上下方向键微调 (步长沿用滑块 step)
        el(numId).addEventListener("keydown", (e) => {
            if (e.key === "Enter") e.target.blur();
        });
    }
}

/**
 * 拖动/编辑期间抑制遥测回填, 否则服务端 30Hz 回流会把滑块和数值框抢走,
 * 表现为拖动时数值不断跳回。
 */
function bindSliderInteractionGuard() {
    for (const [, sliderId, numId] of AXES) {
        el(sliderId).addEventListener("pointerdown", () => { state.userDraggingSlider = true; });
        el(sliderId).addEventListener("keydown", () => { state.userDraggingSlider = true; });
        el(numId).addEventListener("focus", () => { state.userDraggingSlider = true; });
        el(numId).addEventListener("blur", () => { state.userDraggingSlider = false; });
    }
    document.addEventListener("pointerup", () => { state.userDraggingSlider = false; });
    document.addEventListener("keyup", () => { state.userDraggingSlider = false; });
}

function bindPresets() {
    const grid = el("presetGrid");
    if (!grid) return;
    grid.addEventListener("click", (e) => {
        const btn = e.target.closest("[data-preset]");
        if (!btn) return;
        const preset = PRESETS[btn.dataset.preset];
        if (preset) applyPreset(preset);
    });
}

function bindDispatch() {
    el("btnDispatchTarget").addEventListener("click", () => {
        dispatch(currentCmd(), "自定义 6D 目标");
    });
}

/* ──────────────────────── 状态操作 ──────────────────────── */

/**
 * 设定某个轴的值并同步滑块与数值框。
 * @param {string} key
 * @param {number} value
 */
export function setAxis(key, value) {
    state.cmd[key] = value;
    notifyPendingPreview();
    const axis = AXES.find((a) => a[0] === key);
    if (!axis) return;
    const [, sliderId, numId] = axis;
    const slider = el(sliderId);
    if (slider && slider.value !== String(value)) slider.value = value;
    const num = el(numId);
    // 正在编辑的数值框不覆写, 否则用户键入过程中会被服务端回流打断
    if (num && document.activeElement !== num && num.value !== String(value)) {
        num.value = Number(value).toFixed(axis[3]);
    }
}

/**
 * 以服务端下发的当前指令位姿回填滑块。
 *
 * 必要性: 滑块若只用 HTML 硬编码默认值, 页面刷新后会与机器人实际收到的指令
 * 脱节, 面板上会出现虚假的大偏差 (实测曾显示 66mm delta 而机器人其实静止)。
 *
 * 但必须让位于本地未下发的修改: 服务器值与面板待发值是两个不同的量,
 * 无条件回填会把用户刚编辑的内容抹掉 (见 state.cmdPending)。
 *
 * @param {number[]} targetPose 世界坐标系目标位置
 * @param {number[]} cmdRpy     世界坐标系目标姿态
 */
export function syncSlidersFromTelemetry(targetPose, cmdRpy) {
    if (!targetPose || !Number.isFinite(targetPose[0])) return;
    if (state.userDraggingSlider) return;

    const y = mirrorY(targetPose[1]);
    const [r, p, yaw] = mirrorRpy(cmdRpy || [0, 0, 0]);
    const incoming = [targetPose[0], y, targetPose[2], r, p, yaw];

    // 服务器值已与本地待发值一致 -> 确认收到, 清除待下发标记
    if (state.cmdPending) {
        const same = KEYS.every((k, i) => Math.abs(state.cmd[k] - incoming[i]) < 1e-6);
        if (same) setPending(false);
        else return;   // 服务器还没收到本次修改 -> 继续保留本地值
    }

    KEYS.forEach((k, i) => setAxis(k, incoming[i]));
}

/** 指令面板的轴顺序 */
const KEYS = ["x", "y", "z", "roll", "pitch", "yaw"];

/**
 * 标记/清除"有未下发修改"。
 * @param {boolean} pending
 */
function setPending(pending) {
    if (state.cmdPending === pending) return;
    state.cmdPending = pending;
    notifyPendingPreview();
    const btn = el("btnDispatchTarget");
    if (btn) btn.classList.toggle("has-pending", pending);
    setText(el("btnDispatchTarget"), pending ? "下发 6D 目标位姿 *" : "下发 6D 目标位姿");
}

/**
 * 应用工况预设并立即下发。
 * @param {{x:number,y:number,z:number,rpy:number[],name:string}} preset
 */
export function applyPreset(preset) {
    setPending(false);   // 预设即刻下发, 不存在"未下发"中间态

    setAxis("x", preset.x);
    setAxis("y", mirrorY(preset.y));
    setAxis("z", preset.z);
    setAxis("roll", preset.rpy[0]);
    setAxis("pitch", preset.rpy[1]);
    setAxis("yaw", preset.rpy[2]);

    // 镜像由 dispatch() 统一处理, 此处保持左臂约定原值
    dispatch({ x: preset.x, y: preset.y, z: preset.z, rpy: preset.rpy }, preset.name);
}

/**
 * 下发标准直立预备位姿。
 */
export async function resetStand() {
    const y = mirrorY(STAND_POSE.y);
    try {
        const resp = await fetch("/api/reset_stand", { method: "POST" });
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        setAxis("x", STAND_POSE.x);
        setAxis("y", y);
        setAxis("z", STAND_POSE.z);
        setAxis("roll", 0); setAxis("pitch", 0); setAxis("yaw", 0);
        setPending(false);
        pushLocalLog("COMMAND", "下发复位指令", "已发布标准对称微屈直立预备位姿");
    } catch (err) {
        pushLocalLog("ERROR", "复位指令失败", err.message || "无法连接到遥测控制中枢");
    }
}

/* ──────────────────────── 指令下发 ──────────────────────── */

/**
 * @returns {{x:number,y:number,z:number,rpy:number[]}}
 */
function currentCmd() {
    const c = state.cmd;
    return { x: c.x, y: c.y, z: c.z, rpy: [c.roll, c.pitch, c.yaw] };
}

/**
 * 向后端下发 6D 目标位姿。
 * @param {{x:number,y:number,z:number,rpy:number[]}} cmd
 * @param {string} presetName
 */
/**
 * 下发 6D 目标。
 *
 * 入参 `cmd` 一律是**左臂约定**的取值 —— 指令面板始终以左臂约定存储
 * (遥测回流经 syncSlidersFromTelemetry 的 mirrorY 换算), 预设也按左臂约定
 * 书写。右臂需要的符号翻转在此**唯一一处**完成。
 *
 * 此前镜像分散在调用方: 预设路径做了 mirrorY, 手动按钮路径没做。结果是
 * 右臂下手动下发的 Y (与 RPY 的 Roll/Yaw) 符号相反, 与面板显示的、
 * 与视口预览所画的都不一致, 且不报错。
 *
 * @param {{x:number,y:number,z:number,rpy?:number[]}} cmd 左臂约定
 * @param {string} presetName
 */
export async function dispatch(cmd, presetName) {
    const btn = el("btnDispatchTarget");
    if (btn) btn.disabled = true;

    const payload = buildTargetPayload(cmd, state.arm, presetName);

    try {
        const resp = await fetch("/api/send_target", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
        const res = await resp.json();
        const rpyInfo = payload.rpy ? ` RPY[${payload.rpy.map((v) => signed(v, 0)).join(", ")}]°` : "";

        if (resp.ok && res.status === "DISPATCHED") {
            // 服务器已接收, 本地值不再是"未下发修改"
            state.cmdPending = false;
            notifyPendingPreview();          // 已下发, 撤回预览标记
            el("btnDispatchTarget")?.classList.remove("has-pending");
            setText(el("btnDispatchTarget"), "下发 6D 目标位姿");
            pushLocalLog("COMMAND", `下发目标: ${presetName}`,
                `[${state.arm}] X ${signed(cmd.x, 3)}m Y ${signed(cmd.y, 3)}m Z ${signed(cmd.z, 3)}m${rpyInfo}`);

            // 可达性提示: 指令已照常下发, 这里只告知"可能不收敛"。
            // 大屏是观测工具 —— 不收敛本身正是要看的现象, 不应被拦在入口。
            const adv = res.reachability;
            if (adv && (adv.likely_reachable === false || adv.rpy_out_of_range)) {
                pushLocalLog("WARN", `目标可能不可达: ${presetName}`,
                    `${adv.note || adv.rpy_note || ""} —— 已下发, 请观察残差与收敛曲线`);
            }
        } else {
            // 仅畸形输入会走到这里 (NaN/Inf、rpy 长度非法、臂名未知)
            pushLocalLog("ERROR", `指令格式有误: ${presetName}`, res.error || `HTTP ${resp.status}`);
        }
    } catch (err) {
        pushLocalLog("ERROR", "指令下发通信失败", err.message || "无法连接到遥测控制中枢");
    } finally {
        if (btn) btn.disabled = false;
    }
}

/**
 * 启动轨迹演示。
 */
export async function startDemo() {
    try {
        const resp = await fetch("/api/demo_circle", { method: "POST" });
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        pushLocalLog("COMMAND", "启动轨迹演示", "正在执行连续空间圆周平滑协同运动");
    } catch (err) {
        pushLocalLog("ERROR", "轨迹演示启动失败", err.message || "无法连接到遥测控制中枢");
    }
}

/* ──────────────────────── 操作臂 ──────────────────────── */

/**
 * 切换操作臂。
 * @param {"left_arm"|"right_arm"} arm
 */
async function selectArm(arm) {
    const changed = state.arm !== arm;
    state.arm = arm;
    updateArmUi();

    if (changed) {
        // 切臂时 Y 的符号约定翻转, 同步滑块以免面板与实际下发值不一致
        setAxis("y", mirrorY(state.cmd.y));
        // 左右臂的可达范围不同, 滑块量程与视口线框盒都要跟着换
        onArmChanged?.();
    }

    try {
        const resp = await fetch("/api/set_arm", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ arm }),
        });
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        pushLocalLog("CONFIG", `切换操作臂: ${arm}`,
            `当前为 ${arm === "left_arm" ? "左臂 (Left Arm)" : "右臂 (Right Arm)"}`);
    } catch (err) {
        pushLocalLog("ERROR", "切换操作臂失败", err.message || "无法连接到遥测控制中枢");
    }
}

function updateArmUi() {
    const isLeft = state.arm === "left_arm";
    setClass(el("btnSelectLeftArm"), `arm-btn${isLeft ? " active" : ""}`);
    setClass(el("btnSelectRightArm"), `arm-btn${isLeft ? "" : " active"}`);
    setText(el("activeArmText"), isLeft ? "LEFT ARM" : "RIGHT ARM");
}

/* ──────────────────────── 工具 ──────────────────────── */

/** 顶栏演示按钮的运行态 */
export function setDemoButtonState(running) {
    const btn = el("btnCircleDemo");
    if (!btn) return;
    const isRunning = btn.classList.contains("running");
    if (running === isRunning) return;
    btn.classList.toggle("running", running);
    btn.disabled = running;
    setText(btn.querySelector(".btn-label"), running ? "演示执行中..." : "Trajectory Demo");
}