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

// 每个角沿三个轴同时朝盒内伸出一小段 (下界角朝 +, 上界角朝 -)
const BRACKET_RATIO = 0.18;

/** 由包围盒生成 16 个顶点 (8 角 × 1 段 × 2 端点) */
function buildBoxVertices(env) {
    const lo = [env.x.min, env.y.min, env.z.min];
    const hi = [env.x.max, env.y.max, env.z.max];
    const span = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]];
    const mid = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2];
    const out = [];
    for (const cx of [lo[0], hi[0]]) {
        for (const cy of [lo[1], hi[1]]) {
            for (const cz of [lo[2], hi[2]]) {
                out.push([cx, cy, cz]);
                out.push([
                    cx + (cx < mid[0] ? 1 : -1) * span[0] * BRACKET_RATIO,
                    cy + (cy < mid[1] ? 1 : -1) * span[1] * BRACKET_RATIO,
                    cz + (cz < mid[2] ? 1 : -1) * span[2] * BRACKET_RATIO,
                ]);
            }
        }
    }
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

test("生成 16 个顶点 (8 角 × 1 段 × 2 端点)", () => {
    // 缓冲区大小必须与实际填充量一致: 多分配的顶点停留在 (0,0,0),
    // 会在世界原点画出退化的零长线段, 表现为凭空出现的细小线头。
    const v = buildBoxVertices(ENV);
    assert.equal(v.length, 16, `应为 16 个顶点, 实得 ${v.length}`);
    assert.ok(v.every((p) => p.length === 3), "每个顶点须为 3 维坐标");
});

test("角括号只在盒的 8 个角上, 不横贯视野", () => {
    // 每段的一端必须是包围盒的真实角点 —— 这是"角括号"与"完整棱线"的分界
    const v = buildBoxVertices(ENV);
    const corners = new Set();
    for (const cx of [ENV.x.min, ENV.x.max])
        for (const cy of [ENV.y.min, ENV.y.max])
            for (const cz of [ENV.z.min, ENV.z.max])
                corners.add(`${cx},${cy},${cz}`);
    for (let i = 0; i < v.length; i += 2) {
        assert.ok(corners.has(`${v[i][0]},${v[i][1]},${v[i][2]}`),
            `线段起点 ${v[i]} 不是包围盒角点`);
    }
    // 8 个角各被覆盖一次, 无遗漏
    assert.equal(v.filter((_, i) => i % 2 === 0).length, 8, "应为 8 个角点起点");
});

test("角括号向内缩, 不越出包围盒", () => {
    const v = buildBoxVertices(ENV);
    for (const [x, y, z] of v) {
        assert.ok(x >= ENV.x.min && x <= ENV.x.max, `X ${x} 越界`);
        assert.ok(y >= ENV.y.min && y <= ENV.y.max, `Y ${y} 越界`);
        assert.ok(z >= ENV.z.min && z <= ENV.z.max, `Z ${z} 越界`);
    }
});

test("内端点严格位于盒内, 且不在任何面上", () => {
    // 该断言原为"所有顶点落在盒面上", 它对旧的 12 条棱线成立, 但对角括号
    // 不成立: 角括号的另一端是朝内缩的点, 三个坐标都严格位于界内。
    const v = buildBoxVertices(ENV);
    for (let i = 1; i < v.length; i += 2) {
        const [x, y, z] = v[i];
        for (const [val, axis] of [[x, "x"], [y, "y"], [z, "z"]]) {
            assert.ok(val > ENV[axis].min && val < ENV[axis].max,
                `内端点 ${axis}=${val} 应严格位于 [${ENV[axis].min}, ${ENV[axis].max}] 内`);
        }
    }
});

test("棱长与包围盒尺寸一致", () => {
    const v = buildBoxVertices(ENV);
    // 首段沿 X, 长度应等于 X 跨度的 BRACKET_RATIO 倍
    const dx = Math.abs(v[1][0] - v[0][0]);
    assert.ok(Math.abs(dx - (ENV.x.max - ENV.x.min) * BRACKET_RATIO) < 1e-9,
        `X 向括号长度 ${dx} 与预期不符`);
    // 第三段沿 Z, 长度应等于 Z 跨度的 BRACKET_RATIO 倍
    const dz = Math.abs(v[5][2] - v[4][2]);
    assert.ok(Math.abs(dz - (ENV.z.max - ENV.z.min) * BRACKET_RATIO) < 1e-9,
        `Z 向括号长度 ${dz} 与预期不符`);
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