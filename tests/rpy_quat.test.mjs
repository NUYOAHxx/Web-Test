/**
 * RPY → 四元数换算的一致性测试
 * ================================================================================
 *
 * 待发目标预览的箭头必须指向后端将要解算的方向。若前端换算与后端约定不一致,
 * 箭头会指向与指令不符的方向 —— 那比不显示朝向更糟, 用户会据此判断错误。
 *
 * 后端统一使用 pin.rpy.rpyToMatrix(r, p, y), 即 R = Rz(yaw)·Ry(pitch)·Rx(roll)。
 * 本文件的基准值由 Python 侧直接调用 pinocchio 导出 (见提交说明)。
 */

import assert from "node:assert/strict";
import { rpyToQuat } from "../visualizer/web/js/format.js";

/** pin.rpy.rpyToMatrix 导出的基准四元数 [x, y, z, w] */
const REFERENCE = {
    "0,0,0": [0.0, 0.0, 0.0, 1.0],
    "90,0,0": [0.7071067811865475, 0.0, 0.0, 0.7071067811865476],
    "0,90,0": [0.0, 0.7071067811865475, 0.0, 0.7071067811865476],
    "0,0,90": [0.0, 0.0, 0.7071067811865475, 0.7071067811865476],
    "10,20,30": [0.03813457647485015, 0.18930785741199999, 0.23929833774473028, 0.9515485246437886],
    "-45,30,60": [-0.43967973954090955, 0.022260026714733823, 0.5319756951821668, 0.7233174113647117],
    "0,45,0": [0.0, 0.38268343236508984, 0.0, 0.9238795325112867],
};

let failures = 0;
const results = [];
for (const [key, ref] of Object.entries(REFERENCE)) {
    const rpy = key.split(",").map(Number);
    const got = rpyToQuat(rpy);
    const err = Math.max(...ref.map((v, i) => Math.abs(v - got[i])));
    const ok = err < 1e-6;
    if (!ok) failures++;
    results.push(`  [${ok ? "PASS" : "FAIL"}] RPY[${key}]  最大偏差 ${err.toExponential(2)}`);
}

const results2 = [];
function test(name, fn) {
    try { fn(); results2.push(`  [PASS] ${name}`); }
    catch (err) { failures++; results2.push(`  [FAIL] ${name}\n         ${err.message}`); }
}

test("零旋转为单位四元数", () => {
    const q = rpyToQuat([0, 0, 0]);
    assert.deepEqual(q.map((v) => Math.abs(v)), [0, 0, 0, 1]);
});

test("四元数模长恒为 1", () => {
    for (const rpy of [[0, 0, 0], [33, -47, 118], [-120, 80, -15]]) {
        const n = Math.hypot(...rpyToQuat(rpy));
        assert.ok(Math.abs(n - 1) < 1e-9, `RPY[${rpy}] 模长 ${n}`);
    }
});

test("缺省参数按 0 处理而非 NaN", () => {
    const q = rpyToQuat(undefined);
    assert.ok(q.every(Number.isFinite), "缺省输入不得产生 NaN");
    assert.ok(Math.abs(rpyToQuat([10])[0] - rpyToQuat([10, 0, 0])[0]) < 1e-12);
});

console.log(results.join("\n"));
console.log(results2.join("\n"));
console.log(`\nRPY 换算测试: ${results.length + results2.length - failures}/${results.length + results2.length} 通过`);
process.exit(failures > 0 ? 1 : 0);
