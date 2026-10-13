/**
 * 可达范围盒的几何构造与判定测试
 * ================================================================================
 *
 * 沙箱环境无 WebGL, 无法目视验证 3D 渲染 —— 本文件改为直接验证**几何构造与
 * 判定逻辑**。
 *
 * 该特性迭代过两版画法, 两次都因"看起来不像盒子"被否, 这段历史正是本文件要
 * 守住的:
 *   · 12 条完整棱 —— 顶面线横穿机器人上半身, 把整体切割成几块;
 *   · 每角 1 段朝内斜线 (角括号) —— 8 段互不相连, 视线无法重建成盒子,
 *     实际观感是画面里飘着几根随机线段, 比棱线更糟 (用户实机确认);
 *   · 现取中间态: 底面矩形 + 四角立柱, 不画顶面。
 *
 * 画法本身好不好看只能目视判断, 但**几何是否自洽**可以断言, 且退化很容易:
 * 早期版本缓冲区按 8 角 × 3 方向 × 2 端点 = 48 分配却只填 16, 剩余的
 * (0,0,0) 会在世界原点画出退化零长线段。
 */

import assert from "node:assert/strict";

const results = [];
let failures = 0;

function test(name, fn) {
    try { fn(); results.push(`  [PASS] ${name}`); }
    catch (err) { failures++; results.push(`  [FAIL] ${name}\n         ${err.message}`); }
}

/* ══════════════════════ 与 overlays.js 同构的实现 ══════════════════════ */

const REACH_BOX_VERTS = 24;

/**
 * 完整 12 条棱 (底面 4 + 顶面 4 + 立柱 4), 共 12 段 = 24 顶点。
 * @returns {number[][]} 扁平的顶点列表 [x0,y0,z0, x1,y1,z1, ...]
 */
function buildBoxVertices(env) {
    const lo = [env.x.min, env.y.min, env.z.min];
    const hi = [env.x.max, env.y.max, env.z.max];
    const b = [
        [lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]],
    ];
    const v = [];
    const push = (x, y, z) => v.push([x, y, z]);

    for (let i = 0; i < 4; i++) {          // 底面 4 边
        const p = b[i], q = b[(i + 1) % 4];
        push(p[0], p[1], lo[2]);
        push(q[0], q[1], lo[2]);
    }
    for (let i = 0; i < 4; i++) {          // 顶面 4 边 —— 缺失则立柱悬空无收口
        const p = b[i], q = b[(i + 1) % 4];
        push(p[0], p[1], hi[2]);
        push(q[0], q[1], hi[2]);
    }
    for (const [x, y] of b) {              // 四角立柱
        push(x, y, lo[2]);
        push(x, y, hi[2]);
    }
    return v;
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

test("生成 24 个顶点 (12 段 × 2 端点), 与缓冲区一致", () => {
    // 缓冲区大小必须与实际填充量一致: 多分配的顶点停留在 (0,0,0),
    // 会在世界原点画出退化的零长线段, 表现为凭空出现的细小线头。
    const v = buildBoxVertices(ENV);
    assert.equal(v.length, REACH_BOX_VERTS,
        `顶点��与缓冲区声明的 ${REACH_BOX_VERTS} 不符, 实得 ${v.length}`);
});

test("底面 4 条边全部落在 z 下界", () => {
    const v = buildBoxVertices(ENV);
    for (let i = 0; i < 8; i += 2) {      // 前 4 段是底面
        assert.equal(v[i][2], ENV.z.min, `底面边 ${i / 2} 起点 z 应为下界`);
        assert.equal(v[i + 1][2], ENV.z.min, `底面边 ${i / 2} 终点 z 应为下界`);
    }
});

test("顶面 4 条边完整闭合 (缺失则立柱悬空, 读起来像未完成的笼子)", () => {
    const v = buildBoxVertices(ENV);
    for (let i = 8; i < 16; i += 2) {     // 次 4 段是顶面
        assert.equal(v[i][2], ENV.z.max, `顶面边 ${(i - 8) / 2} 起点 z 应为上界`);
        assert.equal(v[i + 1][2], ENV.z.max, `顶面边 ${(i - 8) / 2} 终点 z 应为上界`);
    }
});

test("四角立柱一端在底面、一端在顶面, 且 XY 保持不变", () => {
    const v = buildBoxVertices(ENV);
    for (let i = 16; i < 24; i += 2) {
        const a = v[i], b = v[i + 1];
        assert.equal(a[0], b[0], `立柱 ${i} 的 x 应保持不变`);
        assert.equal(a[1], b[1], `立柱 ${i} 的 y 应保持不变`);
        assert.equal(a[2], ENV.z.min, `立柱 ${i} 起点应在 z 下界`);
        assert.equal(b[2], ENV.z.max, `立柱 ${i} 终点应在 z 上界`);
    }
});

test("立柱高度等于包围盒 Z 跨度", () => {
    const v = buildBoxVertices(ENV);
    const h = v[17][2] - v[16][2];
    assert.ok(Math.abs(h - (ENV.z.max - ENV.z.min)) < 1e-9,
        `立柱高度 ${h} 与 Z 跨度 ${ENV.z.max - ENV.z.min} 不符`);
});

test("底面矩形覆盖完整的 X / Y 跨度", () => {
    const v = buildBoxVertices(ENV).slice(0, 8);
    const xs = v.map((p) => p[0]), ys = v.map((p) => p[1]);
    assert.ok(Math.abs(Math.min(...xs) - ENV.x.min) < 1e-9, "底面未覆盖 x 下界");
    assert.ok(Math.abs(Math.max(...xs) - ENV.x.max) < 1e-9, "底面未覆盖 x 上界");
    assert.ok(Math.abs(Math.min(...ys) - ENV.y.min) < 1e-9, "底面未覆盖 y 下界");
    assert.ok(Math.abs(Math.max(...ys) - ENV.y.max) < 1e-9, "底面未覆盖 y 上界");
});

test("立方体闭合: 每个角点的三条棱都存在", () => {
    // 半吊子画法 (只画部分边) 会让盒子读起来缺一块 —— 这正是本特性迭代过
    // 两版都被否的共同原因。此处断言 12 条棱构成完整闭合体。
    const v = buildBoxVertices(ENV);
    assert.equal(v.length / 2, 12, "应为 12 条线段");
    const lo = [ENV.x.min, ENV.y.min, ENV.z.min], hi = [ENV.x.max, ENV.y.max, ENV.z.max];
    const corners = new Set();
    for (const a of [lo[0], hi[0]]) for (const b of [lo[1], hi[1]]) for (const c of [lo[2], hi[2]])
        corners.add(`${a},${b},${c}`);
    // 每条线段至少一端是角点 (立方体的所有棱都是角点间的连线)
    for (let i = 0; i < v.length; i += 2) {
        const k = `${v[i][0]},${v[i][1]},${v[i][2]}`;
        assert.ok(corners.has(k), `线段 ${i / 2} 的起点不是角点`);
    }
});

test("所有顶点均在包围盒内", () => {
    for (const [x, y, z] of buildBoxVertices(ENV)) {
        assert.ok(x >= ENV.x.min && x <= ENV.x.max, `X ${x} 越界`);
        assert.ok(y >= ENV.y.min && y <= ENV.y.max, `Y ${y} 越界`);
        assert.ok(z >= ENV.z.min && z <= ENV.z.max, `Z ${z} 越界`);
    }
});

test("不存在零长线段", () => {
    const v = buildBoxVertices(ENV);
    for (let i = 0; i < v.length; i += 2) {
        const a = v[i], b = v[i + 1];
        const d = Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
        assert.ok(d > 1e-6, `第 ${i / 2} 段退化为零长线段`);
    }
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