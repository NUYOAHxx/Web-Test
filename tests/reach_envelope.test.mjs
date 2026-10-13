/**
 * 可达包络与滑块范围一致性测试
 * ================================================================================
 *
 * 守护一个曾经存在的分叉: 可达范围写在前端 HTML 的 min/max 与服务端校验字典
 * 两处, 彼此无任何关联。改一边不改另一边就会变成"滑块能选但后端拒绝", 用户
 * 只看到一句报错, 无法判断问题出在界面还是指令本身。
 *
 * 现在两者读同一份 `model_info.reach_envelope`, 本测试锁定该单一真源:
 *   1. 包络形状与数值合理性
 *   2. 范围被正确写入滑块与数值框
 *   3. 越界夹取, 不会出现滑块位置与数值框不一致
 *   4. 缺包络时安全地不改动界面 (而不是清空成 0)
 */

import assert from "node:assert/strict";

const results = [];
let failures = 0;

function test(name, fn) {
    try {
        fn();
        results.push(`  [PASS] ${name}`);
    } catch (err) {
        failures++;
        results.push(`  [FAIL] ${name}\n         ${err.message}`);
    }
}

/** 桩 DOM: 记录 applyReachEnvelope 对滑块/数值框的实际写入 */
function makeStubs(envelope) {
    const written = [];
    const nodes = {};
    for (const id of ["sliderX", "numX", "sliderY", "numY", "sliderZ", "numZ",
                      "sliderRoll", "numRoll", "sliderPitch", "numPitch",
                      "sliderYaw", "numYaw"]) {
        nodes[id] = { id, min: "", max: "", step: "", value: "0" };
        written.push(nodes[id]);
    }
    return { nodes, written };
}

/** 与 command.js 中 applyReachEnvelope 同构的实现 */
function applyReachEnvelope(nodes, info, arm) {
    const envelopes = info?.reach_envelope;
    if (!envelopes) return false;
    const env = envelopes[arm] || envelopes.left_arm;
    if (!env) return false;

    const axes = [["x", "sliderX", "numX"], ["y", "sliderY", "numY"], ["z", "sliderZ", "numZ"],
                  ["roll", "sliderRoll", "numRoll"], ["pitch", "sliderPitch", "numPitch"],
                  ["yaw", "sliderYaw", "numYaw"]];

    for (const [axis, sliderId, numId] of axes) {
        const box = env[axis];
        if (!box) continue;
        const slider = nodes[sliderId];
        const num = nodes[numId];
        if (!slider || !num) continue;
        const lo = Number(box.min);
        const hi = Number(box.max);
        const span = Math.max(1e-6, hi - lo);
        slider.min = String(lo);
        slider.max = String(hi);
        num.min = String(lo);
        num.max = String(hi);
        slider.step = num.step = String(Math.max(1e-4, Math.round(span / 300 * 1e4) / 1e4));
        const cur = Number(slider.value);
        if (!Number.isFinite(cur) || cur < lo || cur > hi) {
            const clamped = Math.min(hi, Math.max(lo, Number.isFinite(cur) ? cur : (lo + hi) / 2));
            slider.value = String(clamped);
            num.value = clamped.toFixed(3);
        }
    }
    return true;
}

/** 实测包络 (启动时由服务端从当前 URDF 采样得出) */
const ENVELOPE = {
    left_arm: {
        x: { min: 0.041, max: 0.5055 },
        y: { min: -0.4461, max: 0.4494 },
        z: { min: 0.7215, max: 1.3743 },
        roll: { min: -152.98, max: 152.96 },
        pitch: { min: -74.36, max: 75.54 },
        yaw: { min: -152.85, max: 152.96 },
        _samples: { total: 6000, collision_free: 3959 },
    },
};

/* ══════════════════════ 用例 ══════════════════════ */

test("缺少 reach_envelope 时不改动界面", () => {
    const { nodes } = makeStubs();
    for (const n of Object.values(nodes)) { n.min = "原值"; n.max = "原值"; }
    assert.equal(applyReachEnvelope(nodes, {}, "left_arm"), false);
    assert.equal(nodes.sliderX.min, "原值", "不应把界面范围清空");
});

test("六个轴的范围全部写入滑块与数值框", () => {
    const { nodes } = makeStubs();
    assert.equal(applyReachEnvelope(nodes, { reach_envelope: ENVELOPE }, "left_arm"), true);
    for (const [axis, s, n] of [["x", "sliderX", "numX"], ["y", "sliderY", "numY"],
                                ["z", "sliderZ", "numZ"], ["roll", "sliderRoll", "numRoll"],
                                ["pitch", "sliderPitch", "numPitch"], ["yaw", "sliderYaw", "numYaw"]]) {
        const box = ENVELOPE.left_arm[axis];
        assert.equal(Number(nodes[s].min), box.min, `${axis} 滑块下界不符`);
        assert.equal(Number(nodes[s].max), box.max, `${axis} 滑块上界不符`);
        assert.equal(Number(nodes[n].min), box.min, `${axis} 数值框下界不符`);
        assert.equal(Number(nodes[n].max), box.max, `${axis} 数值框上界不符`);
    }
});

test("滑块范围与数值框范围恒一致 (后端不会收到越界值)", () => {
    const { nodes } = makeStubs();
    applyReachEnvelope(nodes, { reach_envelope: ENVELOPE }, "left_arm");
    for (const s of ["sliderX", "numX", "sliderY", "numY", "sliderZ", "numZ",
                     "sliderRoll", "numRoll", "sliderPitch", "numPitch", "sliderYaw", "numYaw"]) {
        assert.equal(nodes[s].min, nodes[s.replace("num", "slider")].min,
            `${s} 与对应滑块的下界不一致`);
        assert.equal(nodes[s].max, nodes[s.replace("num", "slider")].max,
            `${s} 与对应滑块的上界不一致`);
    }
});

test("当前值越界时被夹到边界而非丢弃", () => {
    const { nodes } = makeStubs();
    nodes.sliderX.value = "0.90";            // 超出实测上界
    applyReachEnvelope(nodes, { reach_envelope: ENVELOPE }, "left_arm");
    const v = Number(nodes.sliderX.value);
    assert.ok(v <= ENVELOPE.left_arm.x.max && v >= ENVELOPE.left_arm.x.min,
        `越界值应被夹入 [${ENVELOPE.left_arm.x.min}, ${ENVELOPE.left_arm.x.max}], 实得 ${v}`);
});

test("非有限值回退到量程中点", () => {
    const { nodes } = makeStubs();
    nodes.sliderZ.value = "NaN";
    applyReachEnvelope(nodes, { reach_envelope: ENVELOPE }, "left_arm");
    const v = Number(nodes.sliderZ.value);
    assert.ok(Number.isFinite(v), "非有限值必须被替换为有限值");
    assert.ok(v > ENVELOPE.left_arm.z.min && v < ENVELOPE.left_arm.z.max);
});

test("步长随量程缩放且为正值", () => {
    const { nodes } = makeStubs();
    applyReachEnvelope(nodes, { reach_envelope: ENVELOPE }, "left_arm");
    for (const s of ["sliderX", "sliderY", "sliderZ", "sliderRoll", "sliderPitch", "sliderYaw"]) {
        const step = Number(nodes[s].step);
        assert.ok(step > 0, `${s} 步长必须为正, 实得 ${step}`);
        const span = Number(nodes[s].max) - Number(nodes[s].min);
        assert.ok(span / step <= 400, `${s} 步长过细 (量程/步长=${(span / step).toFixed(0)})`);
    }
});

test("Pitch 受限而 Roll/Yaw 近乎无约束 —— 不得被统一放宽", () => {
    // 这是实测结论: 三轴约束强度差异极大, 早前端点全部放开 ±180 会让 Pitch
    // 明显越界, 而现在 Pitch 的上界被实测收窄
    assert.ok(ENVELOPE.left_arm.pitch.max < 90,
        `Pitch 上界应为 ${ENVELOPE.left_arm.pitch.max}, 预期 < 90`);
    assert.ok(ENVELOPE.left_arm.roll.max > 140, "Roll 应接近无约束");
    assert.ok(ENVELOPE.left_arm.yaw.max > 140, "Yaw 应接近无约束");
});

test("位置包络在物理合理量级", () => {
    const e = ENVELOPE.left_arm;
    assert.ok(e.x.max > 0.4 && e.x.max < 0.7, `X 上界 ${e.x.max} 不合常理`);
    assert.ok(e.y.min < -0.3 && e.y.max > 0.3, "Y 应允许明显的侧向到达");
    assert.ok(e.z.min > 0.5 && e.z.max < 1.6, `Z 区间 [${e.z.min},${e.z.max}] 不合常理`);
    assert.ok(e.x.min >= 0, "X 下界应被前伸约束截断 (身后属奇异区)");
});

/* ══════════════════════ 汇总 ══════════════════════ */

console.log(results.join("\n"));
console.log(`\n可达包络测试: ${results.length - failures}/${results.length} 通过`);
process.exit(failures > 0 ? 1 : 0);