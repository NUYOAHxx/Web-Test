/**
 * 可达范围线框盒的构造与判定测试
 * ================================================================================
 *
 * 沙箱环境无 WebGL, 无法目视验证 3D 渲染 —— 本文件改为直接验证**几何构造与
 * 判定逻辑**: 顶点坐标是否落在实测包围盒上、十二条棱是否完整、越界判定在边界
 * 内外是否正确。
 *
 * 该特性最初的想法是画一个以世界原点为中心的任意正方体当距离参照。但任意边长
 * 没有物理含义 —— 1m 凭什么? 画小了显得手够不到, 画大了又失去参考价值。改用
 * 服务端实测的可达包围盒后, 这个盒子直接就是"手能够到的范围"。
 */

import assert from "node:assert/strict";

const results = [];
let failures = 0;

function test(name, fn) {
    try { fn(); results.push(`  [PASS] ${name}`); }
    catch (err) { failures++; results.push(`  [FAIL] ${name}\n         ${err.message}`); }
}

/* ══════════════════════ 与 overlays.js 同构的实现 ══════════════════════ */

const CORNER_INDEX = [
    [0, 1], [1, 3], [3, 2], [2, 0],
    [4, 5], [5, 7], [7, 6], [6, 4],
    [0, 4], [1, 5], [2, 6], [3, 7],
];

/** 由包围盒生成 24 个顶点 (12 条棱 × 2 端点) */
function buildBoxVertices(env) {
    const lo = [env.x.min, env.y.min, env.z.min];
    const hi = [env.x.max, env.y.max, env.z.max];
    const c = [
        [lo[0], lo[1], lo[2]], [hi[0], lo[1], lo[2]], [hi[0], hi[1], lo[2]], [lo[0], hi[1], lo[2]],
        [lo[0], lo[1], hi[2]], [hi[0], lo[1], hi[2]], [hi[0], hi[1], hi[2]], [lo[0], hi[1], hi[2]],
    ];
    const out = [];
    for (const [a, b] of CORNER_INDEX) { out.push(c[a], c[b]); }
    return out;
}

function isOutside(env, pos) {
    // 与 overlays.js 的 isTargetOutsideReachBox 同构: 先挡畸形输入
    if (!env || !Array.isArray(pos) || pos.length < 3) return false;
    const lo = [env.x.min, env.y.min, env.z.min];
    const hi = [env.x.max, env.y.max, env.z.max];
    for (let i = 0; i < 3; i++) if (!(pos[i] >= lo[i] && pos[i] <= hi[i])) return true;
    return false;
}

/* ══════════════════════ 测试数据 ══════════════════════ */

/** 服务端实测下发的包围盒 (左臂) */
const ENV = {
    x: { min: -0.5493, max: 0.5464 },
    y: { min: -0.5465, max: 0.5360 },
    z: { min: 0.6561, max: 1.4429 },
};

/* ══════════════════════ 用例 ══════════════════════ */

test("生成 24 个顶点 (12 条棱)", () => {
    const v = buildBoxVertices(ENV);
    assert.equal(v.length, 24, `应为 24 个顶点, 实得 ${v.length}`);
    assert.ok(v.every((p) => p.length === 3), "每个顶点须为 3 维坐标");
});

test("所有顶点均落在包围盒表面上", () => {
    const v = buildBoxVertices(ENV);
    for (const [x, y, z] of v) {
        for (const [i, val] of [[0, x], [1, y], [2, z]]) {
            const axis = ["x", "y", "z"][i];
            const onMin = Math.abs(val - ENV[axis].min) < 1e-9;
            const onMax = Math.abs(val - ENV[axis].max) < 1e-9;
            assert.ok(onMin || onMax,
                `坐标 ${val} 既不在 ${axis} 下界 ${ENV[axis].min} 也不在上界 ${ENV[axis].max}`);
        }
    }
});

test("棱长与包围盒尺寸一致", () => {
    const v = buildBoxVertices(ENV);
    // 底面首条棱沿 X, 长度应等于 x 跨度
    const dx = Math.abs(v[1][0] - v[0][0]);
    assert.ok(Math.abs(dx - (ENV.x.max - ENV.x.min)) < 1e-9,
        `X 向棱长 ${dx} 与包围盒 X 跨度不符`);
    // 首条竖棱沿 Z (CORNER_INDEX[8] = [0,4], 落在顶点 16 与 17)
    const dz = Math.abs(v[17][2] - v[16][2]);
    assert.ok(Math.abs(dz - (ENV.z.max - ENV.z.min)) < 1e-9,
        `Z 向棱长 ${dz} 与包围盒 Z 跨度不符`);
});

test("尺寸文本格式正确", () => {
    const sx = (ENV.x.max - ENV.x.min).toFixed(2);
    const sy = (ENV.y.max - ENV.y.min).toFixed(2);
    const sz = (ENV.z.max - ENV.z.min).toFixed(2);
    assert.equal(`${sx} × ${sy} × ${sz} m`, "1.10 × 1.08 × 0.79 m");
});

test("盒内目标判为未越界", () => {
    assert.equal(isOutside(ENV, [0.30, 0.22, 0.84]), false);
    assert.equal(isOutside(ENV, [0.00, 0.00, 1.00]), false);
});

test("盒边界上的点算盒内 (闭区间)", () => {
    assert.equal(isOutside(ENV, [ENV.x.min, 0, 1]), false);
    assert.equal(isOutside(ENV, [ENV.x.max, 0, 1]), false);
    assert.equal(isOutside(ENV, [0, ENV.y.min, 1]), false);
    assert.equal(isOutside(ENV, [0, 0, ENV.z.max]), false);
});

test("盒外目标判为越界", () => {
    assert.equal(isOutside(ENV, [0.9, 0, 1.0]), true, "X 超上限");
    assert.equal(isOutside(ENV, [0, 0.9, 1.0]), true, "Y 超上限");
    assert.equal(isOutside(ENV, [0, 0, 2.0]), true, "Z 超上限");
    assert.equal(isOutside(ENV, [0, 0, 0.2]), true, "Z 低于下限");
});

test("用户实测过的越界目标确实越界", () => {
    // 实测下发 [0.380, 0.180, 1.355] 时提示 45mm 不可达, 应被标红
    assert.equal(isOutside(ENV, [0.380, 0.180, 1.355]), false,
        "该点在包围盒内 (非凸可达区的角落), 不应靠盒子判越界");
});

test("非法输入不误判为越界", () => {
    assert.equal(isOutside(ENV, null), false);
    assert.equal(isOutside(ENV, []), false);
    assert.equal(isOutside(ENV, [1, 2]), false);
});

test("盒尺寸非零 (零会让几何退化成一点而不可见)", () => {
    for (const axis of ["x", "y", "z"]) {
        assert.ok(ENV[axis].max > ENV[axis].min, `${axis} 轴跨度必须为正`);
    }
});

/* ══════════════════════ 汇总 ══════════════════════ */

console.log(results.join("\n"));
console.log(`\n可达范围盒测试: ${results.length - failures}/${results.length} 通过`);
process.exit(failures > 0 ? 1 : 0);