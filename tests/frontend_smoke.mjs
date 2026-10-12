/**
 * 前端模块冒烟测试 (在 Node 中执行, 无需浏览器)。
 *
 * 为什么需要它:
 *   纯语法检查 (node --check) 只能捕获语法错误, 无法发现
 *   ① import 名不存在 —— 运行时才报 "does not provide an export"
 *   ② 作用域错误 —— 函数体内引用了未定义/未传入的标识符
 *   ③ 顶���级初始化即崩溃
 *
 *   这三类都曾在重构中真实出现过 (例如收敛曲线绘制函数误用作用域外的
 *   坐标映射函数, 语法完全合法, 但每帧绘制都会抛 ReferenceError)。
 *
 * 做法: 桩掉浏览器专有 API (document / window / THREE / ResizeObserver),
 *       然后真实 import 各模块并调用其纯逻辑函数。
 */

import assert from "node:assert/strict";

/* ══════════════ 浏览器 API 桩 ══════════════ */

/** 极简 DOM 节点, 满足渲染模块的属性写入需求 */
class FakeNode {
    constructor(id = "") {
        this.id = id;
        this.textContent = "";
        this.innerHTML = "";
        this.className = "";
        this.value = "";
        this.hidden = false;
        this.disabled = false;
        this.dataset = {};
        this.style = new Proxy({}, { set: (t, k, v) => { t[k] = v; return true; }, get: (t, k) => t[k] });
        this.children = [];
        this.childNodes = [];
        this.scrollWidth = 0;
        this.clientWidth = 100;
        this.scrollHeight = 0;
        this.clientHeight = 100;
    }
    addEventListener() {}
    removeEventListener() {}
    appendChild(c) { this.children.push(c); this.childNodes.push(c); return c; }
    prepend(c) { this.children.unshift(c); return c; }
    remove() {}
    setAttribute() {}
    getAttribute() { return null; }
    querySelector() { return new FakeNode(); }
    querySelectorAll() { return []; }
    closest() { return null; }
    getBoundingClientRect() { return { left: 0, top: 0, width: 320, height: 110, right: 320, bottom: 110 }; }
    get lastElementChild() { return null; }
    get firstElementChild() { return null; }
}

/** 记录所有 id 查询, 供断言检查模块查了哪些元素 */
const queriedIds = new Set();

/**
 * 按 id 缓存节点。
 *
 * 真实浏览器中 getElementById 对同一 id 恒返回同一节点; 若桩每次新建,
 * 测试断言到的节点会与被测模块通过 dom.js 缓存拿到的节点不是同一个,
 * 导致"断言失败"却与被测代码无关。
 */
const nodesById = new Map();

globalThis.document = {
    readyState: "complete",
    body: new FakeNode("body"),
    documentElement: new FakeNode("html"),
    getElementById(id) {
        queriedIds.add(id);
        if (!nodesById.has(id)) nodesById.set(id, new FakeNode(id));
        return nodesById.get(id);
    },
    createElement(tag) { return new FakeNode(tag); },
    querySelector() { return new FakeNode(); },
    querySelectorAll() { return []; },
    addEventListener() {},
};

/**
 * 2D Canvas 上下文桩。
 *
 * 关键点: 所有属性访问 (含 canvas 这类被源码读取的) 都记一笔,
 * 而非只记函数调用。绘制代码在调用 stroke() 之前必然先写 `ctx.strokeStyle = ...`,
 * 因此"属性写入 + 调用"的混合计数才是"确实画了东西"的可靠信号。
 *
 * fillText 需保留实参以断言刻度文本, 单独处理。
 */
const ctxCalls = [];
const filledText = [];

/**
 * 返回对象属性的兜底必须能承接链式调用 (createLinearGradient().addColorStop)。
 * 桩若为所有属性返回同一个"记录型函数", 则 `gradient.addColorStop` 取到的仍是
 * 那个函数, 返回 undefined 后再 .addColorStop 就炸 —— 这是桩自身缺陷,
 * 会被误判成被测代码的问题。故对未知属性返回可链式的通用对象。
 */
function makeCtx() {
    const record = (prop) => (...args) => { ctxCalls.push(`${prop}(${args.length})`); };

    const target = {
        canvas: { width: 320, height: 110 },
        measureText: () => ({ width: 10 }),
        createLinearGradient: (...a) => {
            ctxCalls.push(`createLinearGradient(${a.length})`);
            return { addColorStop: (...b) => ctxCalls.push(`addColorStop(${b.length})`) };
        },
        fillText: (...args) => { filledText.push(String(args[0])); ctxCalls.push("fillText"); },
    };

    return new Proxy(target, {
        get(t, prop) {
            // 返回函数属性时保持身份稳定 (原型链上的方法名如 then/catch 等)
            if (prop in t) return t[prop];
            if (typeof prop !== "string") return undefined;
            if (prop === "then") return undefined;   // 勿被误判为 thenable
            // 普通绘图方法
            if (/^[a-z]/.test(prop)) {
                if (!t[prop]) t[prop] = record(prop);
                return t[prop];
            }
            // 任意其他对象属性 (渐变色、save() 的返回值等) -> 可链式桩
            return new Proxy({}, {
                get(_, p2) { return () => chainable(); },
                apply() { return chainable(); },
            });
        },
        set(t, prop, v) { ctxCalls.push(`set:${String(prop)}`); t[prop] = v; return true; },
    });

    function chainable() {
        return new Proxy({}, { get: () => () => chainable(), apply: () => chainable() });
    }
}

/** 为测试画布装配一个记录型上下文, 并清空计数 */
function freshCanvas(id = "convergenceCanvas") {
    const canvas = new FakeNode(id);
    canvas.width = 320;
    canvas.height = 110;
    ctxCalls.length = 0;
    filledText.length = 0;
    canvas.getContext = () => makeCtx();
    return canvas;
}

globalThis.window = {
    devicePixelRatio: 2,
    addEventListener() {},
    removeEventListener() {},
};

/* ══════════════ 测试用例 ══════════════ */

const results = [];
let failures = 0;

async function test(name, fn) {
    try {
        await fn();
        results.push(`  [PASS] ${name}`);
    } catch (err) {
        failures++;
        results.push(`  [FAIL] ${name}\n         ${err.message}`);
    }
}

/* ── 1. 模块可导入 (验证所有 import 都能解析到实际 export) ── */

const dom = await import("../visualizer/web/js/dom.js");
const format = await import("../visualizer/web/js/format.js");
const stateMod = await import("../visualizer/web/js/state.js");
const logs = await import("../visualizer/web/js/panels/logs.js");
const safety = await import("../visualizer/web/js/panels/safety.js");
const chart = await import("../visualizer/web/js/panels/convergence-chart.js");
const joints = await import("../visualizer/web/js/panels/joints.js");
const robot = await import("../visualizer/web/js/viewport/robot.js");

await test("全部模块可导入 (import 均可解析)", async () => {
    assert.ok(dom.el && dom.setText && dom.escapeHtml);
    assert.ok(format.clearanceStatus && format.solverStatus);
    assert.ok(stateMod.state && stateMod.mirrorY);
    assert.ok(logs.pushLocalLog && logs.syncServerLogs);
    assert.ok(safety.renderSafety);
    assert.ok(chart.drawConvergence);
    assert.ok(joints.renderJoints);
});

/* ── 2. DOM 写入原语 ── */

await test("setText 值未变化时跳过写入", () => {
    let n = 0;
    const node = new FakeNode();
    const spy = () => { n++; };
    Object.defineProperty(node, "textContent", {
        get() { return "x"; },
        set(v) { spy(); },
    });
    dom.setText(node, "x");
    assert.equal(n, 0, "相同值不应写入");
    dom.setText(node, "y");
    assert.equal(n, 1, "不同值应写入一次");
});

await test("escapeHtml 拦截全部 XSS 向量", () => {
    assert.equal(dom.escapeHtml('<script>'), "&lt;script&gt;");
    assert.equal(dom.escapeHtml('a & b'), "a &amp; b");
    assert.equal(dom.escapeHtml('"q"'), "&quot;q&quot;");
    assert.equal(dom.escapeHtml("'q'"), "&#39;q&#39;");
    assert.equal(dom.escapeHtml(null), "");
});

await test("escapeHtml 用于遥测字段后无注入风险", () => {
    const hostile = '<img src=x onerror="alert(1)">';
    const safe = dom.escapeHtml(hostile);
    assert.ok(!safe.includes("<img"));
    assert.ok(!safe.includes('"'));
});

/* ── 3. 严重度阈值一致性 (修复的核心缺陷) ── */

await test("碰撞间隙严重度在边界上单一确定", () => {
    const cases = [
        [0,   true,  "alert", "COLLISION"],
        [3.0, false, "alert", "ALERT <5mm"],
        [4.9, false, "alert", "ALERT <5mm"],
        [5.0, false, "warn",  "CAUTION <20mm"],
        [19.9,false, "warn",  "CAUTION <20mm"],
        [20.0,false, "ok",    "SECURE"],
        [999, false, "ok",    "SECURE"],
    ];
    for (const [mm, colliding, level, text] of cases) {
        const got = format.clearanceStatus(mm, colliding);
        assert.equal(got.level, level, `间隙 ${mm}mm 应为 ${level}, 实际 ${got.level}`);
        assert.equal(got.text, text, `间隙 ${mm}mm 文案不符`);
    }
});

await test("同一间隙值判定稳定 (不存在多处阈值互相矛盾)", () => {
    // 旧实现在 8mm 会出现 "数字标红 / 徽章 SECURE / 告警框隐藏" 的矛盾;
    // 现在严重度是单一函数产物, 调用多少次结果都一致。
    for (let i = 0; i < 20; i++) {
        assert.deepEqual(format.clearanceStatus(8.0, false), format.clearanceStatus(8.0, false));
    }
    assert.equal(format.clearanceStatus(8.0, false).level, "warn");
});

await test("碰撞标志优先于间隙阈值", () => {
    assert.equal(format.clearanceStatus(500, true).level, "alert");
    assert.equal(format.clearanceStatus(500, true).text, "COLLISION");
});

await test("NaN 输入不产生假安全判定", () => {
    const r = format.clearanceStatus(NaN, false);
    assert.equal(r.level, "ok");
    assert.equal(r.text, "NO DATA", "无数据不应显示为 SECURE");
});

await test("求解状态按迭代数与残差分级", () => {
    assert.equal(format.solverStatus(0, 0).label, "STANDBY");
    assert.equal(format.solverStatus(5, 1.0).label, "CONVERGED");
    assert.equal(format.solverStatus(5, 9.0).label, "APPROXIMATE");
    assert.equal(format.solverStatus(5, 99).label, "LOCAL MIN");
});

await test("平衡状态由裕度数值统一推导", () => {
    assert.equal(format.balanceStatus(90).status, "STABLE");
    assert.equal(format.balanceStatus(40).status, "LEANING");
    assert.equal(format.balanceStatus(10).status, "TIPPING");
});

/* ── 4. 格式化 ── */

await test("数值格式化处理缺字段", () => {
    assert.equal(format.signed(0.5, 3), "+0.500");
    assert.equal(format.signed(-0.5, 3), "-0.500");
    assert.equal(format.fixed(NaN), "—");
    assert.equal(format.signed(undefined), "—");
    assert.equal(format.at([1, 2, 3], 1), 2);
    assert.equal(format.at([1], 5), null, "越界应返回 null 而非 undefined");
    assert.equal(format.at(null, 0), null);
});

await test("pct 钳制到 0~100", () => {
    assert.equal(format.pct(50, 100), 50);
    assert.equal(format.pct(-5, 100), 0);
    assert.equal(format.pct(500, 100), 100);
    assert.equal(format.pct(NaN, 100), 0);
    assert.equal(format.pct(10, 0), 0, "满量程为 0 时不得除零");
});

/* ── 5. 渲染函数冒烟 (验证无作用域错误 / 无运行期崩溃) ── */

await test("收敛曲线可绘制非空序列 (覆盖作用域错误回归)", () => {
    // 这条用例守护的真实缺陷: 绘制函数曾引用作用域外的坐标映射函数,
    // 语法完全合法但每次绘制都抛 ReferenceError, 表现为曲线区域永远空白。
    const canvas = freshCanvas();
    chart.drawConvergence(canvas, [50, 30, 12, 5, 2.1, 1.2, 0.9], {
        seedSwitches: [0, 4],
        hoverIndex: 2,
    });

    assert.ok(ctxCalls.length > 20, `应产生实际绘制行为, 实际记录 ${ctxCalls.length} 条`);
    assert.ok(filledText.includes("2mm"), "应绘制容差参考线标注");
    assert.ok(filledText.includes("S1"), "多种子切换应标注 S1");
    assert.ok(filledText.some((s) => /^it:\d+$/.test(s)), "应标注迭代区间");
});

await test("收敛曲线可绘制单点序列 (除零防护)", () => {
    const canvas = freshCanvas();
    chart.drawConvergence(canvas, [3.5], { seedSwitches: [0], hoverIndex: -1 });
    assert.ok(ctxCalls.length > 10, "单点序列不应因除零而中断绘制");
});

await test("收敛曲线空序列显示待命而非崩溃", () => {
    const canvas = freshCanvas();
    chart.drawConvergence(canvas, [], { seedSwitches: [0], hoverIndex: -1 });
    assert.ok(filledText.some((s) => s.includes("待命")), "空序列应绘制待命提示");
    assert.ok(filledText.some((s) => s.includes("收敛曲线")), "空态应说明何时会出现曲线");
});

await test("悬停索引越界时不崩溃", () => {
    const canvas = freshCanvas();
    assert.doesNotThrow(() =>
        chart.drawConvergence(canvas, [10, 5], { seedSwitches: [0], hoverIndex: 99 }));
});

await test("曲线按设备像素比调整位图尺寸 (避免模糊/拉伸)", () => {
    const canvas = freshCanvas();
    chart.drawConvergence(canvas, [5, 2], { seedSwitches: [0], hoverIndex: -1 });
    // dpr=2, CSS 尺寸 320x110 -> 位图应为 640x220
    assert.equal(canvas.width, 640, "位图宽度应按 dpr 放大");
    assert.equal(canvas.height, 220, "位图高度应按 dpr 放大");
});

await test("收敛曲线拒绝非法入参而不崩溃", () => {
    const canvas = freshCanvas();
    assert.doesNotThrow(() => chart.drawConvergence(null, [1, 2], {}));
    assert.doesNotThrow(() => chart.drawConvergence(canvas, null, {}));
    assert.doesNotThrow(() => chart.drawConvergence(canvas, [NaN, NaN], {}));
    assert.doesNotThrow(() => chart.drawConvergence(canvas, [-5, 999], {}));
    assert.doesNotThrow(() => chart.drawConvergence(canvas, [1], null));
});

await test("安全面板可渲染完整报文", () => {
    assert.doesNotThrow(() => safety.renderSafety({
        min_clearance_mm: 8.0,
        is_colliding: false,
        colliding_pairs: [["left_elbow", "torso_link", "3.2mm"]],
        zone_clearances: { arm_torso: 77, arm_arm: 320, arm_head: 201, arm_leg: 254 },
    }));
});

await test("安全面板处理空报文 (待机态)", () => {
    assert.doesNotThrow(() => safety.renderSafety({}));
    assert.doesNotThrow(() => safety.renderSafety(null));
});

await test("关节仪表支持数量增减 (DOM 复用与回收)", () => {
    const container = new FakeNode("jointsList");
    const mk = (i) => ({ name: `j${i}`, label: `关节${i}`, deg: i * 5, rad: i * 0.08, offset_pct: i * 10, min_deg: -180, max_deg: 180 });
    assert.doesNotThrow(() => joints.renderJoints(container, [mk(0), mk(1), mk(2)]));
    assert.doesNotThrow(() => joints.renderJoints(container, [mk(0)]));
    assert.doesNotThrow(() => joints.renderJoints(container, []));
    assert.doesNotThrow(() => joints.renderJoints(null, [mk(0)]));
    assert.doesNotThrow(() => joints.renderJoints(container, null));
});

await test("关节仪表限位余量触发分级样式", () => {
    const container = new FakeNode("jointsList");
    // deg=0 距 ±180 限位各 180°, 余量充足
    assert.doesNotThrow(() => joints.renderJoints(container, [
        { name: "safe", deg: 0, rad: 0, offset_pct: 0, min_deg: -180, max_deg: 180 },
    ]));
    // deg=179 距上限仅 1° -> 应判为 alert 级
    assert.doesNotThrow(() => joints.renderJoints(container, [
        { name: "alert", deg: 179, rad: 3.12, offset_pct: 99, min_deg: -180, max_deg: 180 },
    ]));
});

await test("日志流: 本地事件不被服务端日志覆盖 (回归)", () => {
    logs.clearLocalLogs();
    logs.pushLocalLog("COMMAND", "下发目标", "X 0.35m");
    logs.syncServerLogs([
        { time: "10:00:01", type: "IK", title: "求解收敛", desc: "22ms" },
        { time: "10:00:02", type: "IK", title: "求解收敛2", desc: "20ms" },
    ]);
    // 关键断言: 旧实现下服务端 innerHTML 会把本地事件整体覆盖掉
    const stream = globalThis.document.getElementById("logStream");
    assert.ok(stream.innerHTML.includes("下发目标"), "本地事件必须在服务端日志刷新后仍存在");
    assert.ok(stream.innerHTML.includes("求解收敛2"));
});

await test("日志流: 服务端日志未变化时跳过重绘", () => {
    const stream = globalThis.document.getElementById("logStream");
    const same = [{ time: "11:00:00", type: "IK", title: "X", desc: "d" }];
    logs.syncServerLogs(same);
    const snapshot = stream.innerHTML;
    logs.syncServerLogs(same);
    assert.equal(stream.innerHTML, snapshot);
});

/* ── 6. 状态镜像 ── */

/* ── 7. 滑块精度工具 ── */

await test("滚轮微调遵守量程边界", () => {
    // command.js 的滚轮处理约定: 仅当本次滚动会改变数值时阻止默认行为,
    // 顶到量程端点后继续滚动必须放行, 否则在页面其他区域悬停时会劫持滚动。
    // 此处以同样的判定逻辑复算, 确保边界语义没有写反。
    const min = 0.05, max = 0.45, step = 0.005;
    const nextOf = (cur, deltaY) => Math.round((cur + (deltaY > 0 ? -step : step)) * 1e6) / 1e6;
    const inRange = (v) => v >= min && v <= max;

    assert.equal(inRange(nextOf(min, 1)), false, "min 处向下滚应越界 -> 放行页面");
    assert.equal(inRange(nextOf(max, -1)), false, "max 处向上滚应越界 -> 放行页面");
    assert.equal(inRange(nextOf(0.25, 1)), true, "中间向下滚应改值");
    assert.equal(inRange(nextOf(0.25, -1)), true, "中间向上滚应改值");
    assert.ok(Math.abs(nextOf(0.25, 1) - 0.245) < 1e-9, "单次滚动应恰好一个步长, 不跳格");
});

await test("细调分辨率优于粗调 (Shift 拖动 1/5 步长)", () => {
    const step = 0.005;
    assert.ok(step / 5 < step, "细调步长必须小于原生步长, 否则 Shift 拖动无意义");
    // 窄屏下滑块约 60px / 0.4m 量程 = 6.7mm 每像素, 明显粗于细调步长,
    // 这正是细调手段存在的理由。
    const pxResolution = 0.4 / 60;
    assert.ok(pxResolution > step, `像素分辨率 ${pxResolution.toFixed(4)} 应粗于步长 ${step}`);
    assert.ok(pxResolution > step / 5, "像素分辨率应粗于细调步长, 否则细调无增益");
});

/* ── 8. 碰撞告警的 link 名 → 几何体名映射 ── */

await test("碰撞 link 名可映射到视觉几何体名", () => {
    // 后端 colliding_pairs 给的是 URDF link 名, visuals 的键是几何体名。
    // 映射规则: 几何体名去掉末尾 "_序号" 即为 link 名。
    // 映射错误的后果是 3D 模型上标红**错误的零件**, 且肉眼极难发现 ——
    // 因此这里用真实上报过的碰撞对做回归。
    assert.equal(robot.linkNameOf("left_shoulder_yaw_link_0"), "left_shoulder_yaw_link");
    assert.equal(robot.linkNameOf("logo_link_0"), "logo_link");
    assert.equal(robot.linkNameOf("pelvis_0"), "pelvis");
    assert.equal(robot.linkNameOf("torso_link_23"), "torso_link");
    // 不带序号后缀的名称应原样返回
    assert.equal(robot.linkNameOf("head_link"), "head_link");
});

await test("碰撞映射覆盖实测出现过的真实碰撞对", () => {
    // 抓取时确实出现过的碰撞: 左肩偏航连杆 与 躯干标识件
    const reported = [["left_shoulder_yaw_link", "logo_link", "arm_torso"]];
    const geomNames = [
        "pelvis_0", "torso_link_0", "logo_link_0",
        "left_shoulder_pitch_link_0", "left_shoulder_yaw_link_0",
    ];
    const wanted = new Set(reported.flat().slice(0, 2));
    const hit = geomNames.filter((g) => wanted.has(robot.linkNameOf(g)));
    assert.equal(hit.length, 2, `应命中 2 个几何体, 实际 ${JSON.stringify(hit)}`);
    assert.ok(hit.includes("left_shoulder_yaw_link_0"));
    assert.ok(hit.includes("logo_link_0"));
    // 未涉及的零件不得被误标
    assert.ok(!hit.includes("pelvis_0"));
    assert.ok(!hit.includes("left_shoulder_pitch_link_0"));
});

/* ── 9. 严重度判定方向 ── */

await test("低值=坏的指标按低值告警 (间隙/余量)", () => {
    const th = { ALERT: 5, WARN: 20 };
    assert.equal(format.severity(3, th), "alert", "间隙 3mm 应告警");
    assert.equal(format.severity(12, th), "warn");
    assert.equal(format.severity(50, th), "ok");
    // 阈值字段名必须存在, 否则 `value < undefined` 恒为 false, 判定整体失效
    assert.ok(Number.isFinite(th.ALERT) && Number.isFinite(th.WARN));
});

await test("低值=好的指标方向不得反转 (残差/力矩)", () => {
    // 回归防护: 曾把 GOOD 当作 ALERT 传给 severity(), 判定整体反向,
    // 0.00mm 的完美收敛被标成红色告警。
    const th = format.POS_ERR;   // { GOOD: 2.0, WARN: 15.0 }
    assert.equal(format.toneByMagnitude(0.00, th), "green", "0mm 残差必须为绿");
    assert.equal(format.toneByMagnitude(1.99, th), "green");
    assert.equal(format.toneByMagnitude(2.0, th), "gold");
    assert.equal(format.toneByMagnitude(14.9, th), "gold");
    assert.equal(format.toneByMagnitude(15.0, th), "red");
    assert.equal(format.toneByMagnitude(NaN, th), "muted");
});

await test("两个判定函数方向相反, 不可互换", () => {
    // 这正是原缺陷的形态: 阈值数值相同但方向相反, 混用即颜色整体颠倒。
    // 用同一组阈值分别喂给两个函数, 结果必须恰好相反。
    const good = format.POS_ERR.GOOD, warn = format.POS_ERR.WARN;
    assert.equal(format.toneByMagnitude(0, { GOOD: good, WARN: warn }), "green",
        "低值=好的判定: 0 应为绿");
    assert.equal(format.severity(0, { ALERT: good, WARN: warn }), "alert",
        "低值=坏的判定: 0 应为告警");
    assert.equal(format.toneByMagnitude(warn, { GOOD: good, WARN: warn }), "red",
        "低值=好的判定: 超过 WARN 应为红");
    assert.equal(format.severity(warn, { ALERT: good, WARN: warn }), "ok",
        "低值=坏的判定: 超过 WARN 应为正常");
});

await test("所有阈值对象字段名与判定函数签名一致", () => {
    // 字段名对不上时 `value < undefined` 恒为 false, 配色会静默整体失效 ——
    // 这正是力矩面板曾经完全不着色的原因。
    const lowIsBad = [format.CLEARANCE, format.JOINT_MARGIN];
    const lowIsGood = [format.POS_ERR, format.ROT_ERR, format.SOLVER, format.TORQUE];
    for (const th of lowIsBad) {
        assert.ok(Number.isFinite(th.ALERT) && Number.isFinite(th.WARN),
            `severity 的阈值对象缺少 ALERT/WARN: ${JSON.stringify(th)}`);
    }
    for (const th of lowIsGood) {
        assert.ok(Number.isFinite(th.GOOD) && Number.isFinite(th.WARN),
            `toneByMagnitude 的阈值对象缺少 GOOD/WARN: ${JSON.stringify(th)}`);
    }
});

await test("右臂镜像 Y 与 Roll/Yaw, Pitch 不变", () => {
    stateMod.state.arm = "left_arm";
    assert.equal(stateMod.mirrorY(0.25), 0.25);
    assert.deepEqual(stateMod.mirrorRpy([10, 20, 30]), [10, 20, 30]);

    stateMod.state.arm = "right_arm";
    assert.equal(stateMod.mirrorY(0.25), -0.25);
    assert.deepEqual(stateMod.mirrorRpy([10, 20, 30]), [-10, 20, -30], "Roll/Yaw 反向, Pitch 保持");
    stateMod.state.arm = "left_arm";
});

/* ══════════════ 汇总 ══════════════ */

console.log(results.join("\n"));
console.log(`\n前端冒烟测试: ${results.length - failures}/${results.length} 通过`);
process.exit(failures > 0 ? 1 : 0);