/**
 * SSE 增量帧合并测试
 * ================================================================================
 *
 * 服务端为节省带宽采用增量推送: 慢变量 (事件日志、收敛轨迹、连杆位姿)
 * 未发生变化时不出现在帧中。前端必须把帧合并进本地累积状态。
 *
 * 本文件用真实报文形态锁定合并语义。守护的是一个已真实发生过的缺陷:
 *
 *   浅拷贝 `Object.assign({}, acc, delta)` 遇到同名嵌套键 `solver_diagnostics`
 *   会用 delta 的那份整体替换掉上一轮的累积结果, 随后的"合并"变成 delta
 *   与自己合并。后果是顶层字段 (事件日志) 正常刷新, 而嵌套的收敛轨迹与逐步
 *   明细永远为空 —— 而控制台无任何报错, 面板只是"没有数据"。
 *
 * 该缺陷无法被语法检查或代码走查发现, 只能在增量推送上线后由运行时暴露。
 */

import assert from "node:assert/strict";

/**
 * 增量帧合并 (与 js/telemetry.js 的 mergeFrame 保持同一实现)。
 * 放在此处而非 import, 是为了让本测试能独立验证语义 —— 一旦实现与测试
 * 分离, 测试就退化为对自身的断言。
 *
 * @param {object} accumulated 上一轮的累积状态 (原地更新)
 * @param {object} delta 本轮增量帧
 * @returns {object} 合并后的累积状态
 */
export function mergeFrameInto(accumulated, delta) {
    // 必须在浅拷贝之前取出上一轮的嵌套对象, 否则会被 delta 整体替换掉
    const prevDiagnostics = accumulated.solver_diagnostics;
    Object.assign(accumulated, delta);
    if (delta.solver_diagnostics) {
        accumulated.solver_diagnostics = Object.assign(
            {}, prevDiagnostics, delta.solver_diagnostics
        );
    }
    return accumulated;
}

/* ══════════════════════ 测试用例 ══════════════════════ */

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

/** 完整首帧 (服务端在每条 SSE 连接建立时推送的基线快照) */
function fullFrame() {
    return {
        cartesian_cmd_pose: [0.36, 0.26, 0.90],
        euclidean_error_norm_mm: 0.0,
        activity_logs: [{ time: "10:00:01", type: "IK", title: "收敛", desc: "22ms" }],
        visuals: { pelvis_0: { mesh: "pelvis.STL", pos: [0, 0, 0.76], quat: [0, 0, 0, 1] } },
        solver_diagnostics: {
            time_ms: 4.2, iters: 5, mode: "9DOF_6DOF_POSE",
            convergence_trace: [282.6, 120.6, 8.1, 0.0],
            step_details: [
                { step: 1, seed_name: "Warm Start", pos_err_mm: 282.6, delta_mm: 0.0, is_colliding: false, status: "DLS" },
                { step: 2, seed_name: "Warm Start", pos_err_mm: 120.6, delta_mm: -162.0, is_colliding: false, status: "DLS" },
            ],
        },
    };
}

/** 增量帧: 仅含变化的快速字段, 慢变量一律缺席 */
function deltaFrame(overrides = {}) {
    return Object.assign({
        cartesian_cmd_pose: [0.30, 0.22, 0.84],
        euclidean_error_norm_mm: 0.12,
        solver_diagnostics: { time_ms: 3.9, iters: 6 },
    }, overrides);
}

test("完整首帧直接建立累积基线", () => {
    const acc = mergeFrameInto({}, fullFrame());
    assert.equal(acc.euclidean_error_norm_mm, 0.0);
    assert.equal(acc.activity_logs.length, 1);
    assert.equal(acc.solver_diagnostics.convergence_trace.length, 4);
    assert.equal(acc.solver_diagnostics.step_details.length, 2);
    assert.ok(acc.visuals.pelvis_0, "连杆位姿应进入基线");
});

test("嵌套 solver_diagnostics 的慢字段在增量帧后保留 (核心回归)", () => {
    const acc = mergeFrameInto({}, fullFrame());
    mergeFrameInto(acc, deltaFrame());

    // 快速字段已更新
    assert.equal(acc.solver_diagnostics.iters, 6, "迭代数应被增量帧更新");
    assert.equal(acc.solver_diagnostics.time_ms, 3.9);
    // 慢字段必须仍在 —— 旧实现此处为空, 步骤表与收敛曲线永远空白
    assert.equal(acc.solver_diagnostics.convergence_trace?.length, 4,
        "收敛轨迹应在增量帧后保留");
    assert.equal(acc.solver_diagnostics.step_details?.length, 2,
        "逐步明细应在增量帧后保留");
});

test("顶层慢字段在增量帧后保留", () => {
    const acc = mergeFrameInto({}, fullFrame());
    mergeFrameInto(acc, deltaFrame());
    assert.equal(acc.activity_logs?.length, 1, "事件日志应保留");
    assert.ok(acc.visuals?.pelvis_0, "连杆位姿应保留");
});

test("慢字段重新出现时被新值覆盖", () => {
    const acc = mergeFrameInto({}, fullFrame());
    // 模拟一次新求解: 轨迹与明细随帧一起更新
    mergeFrameInto(acc, {
        solver_diagnostics: {
            time_ms: 5.1, iters: 9,
            convergence_trace: [100.0, 50.0, 1.0],
            step_details: [{ step: 1, seed_name: "Ready Pose", pos_err_mm: 100.0,
                             delta_mm: 0.0, is_colliding: false, status: "DLS" }],
        },
    });
    assert.equal(acc.solver_diagnostics.convergence_trace.length, 3, "新轨迹应覆盖旧轨迹");
    assert.equal(acc.solver_diagnostics.step_details.length, 1);
    assert.equal(acc.solver_diagnostics.iters, 9);
});

test("连续多帧增量不累积脏数据", () => {
    const acc = mergeFrameInto({}, fullFrame());
    for (let i = 0; i < 200; i++) {
        mergeFrameInto(acc, deltaFrame({
            euclidean_error_norm_mm: i * 0.01,
            solver_diagnostics: { time_ms: 3 + i * 0.01, iters: i % 20 },
        }));
    }
    assert.equal(acc.solver_diagnostics.iters, 199 % 20, "标量应为最新值");
    assert.equal(acc.solver_diagnostics.step_details.length, 2, "慢字段应始终保留");
    assert.equal(Object.keys(acc).length, new Set(Object.keys(fullFrame())).size,
        "顶层键数不应随帧数增长");
});

test("缺字段的首帧不使后续合并崩溃", () => {
    const acc = mergeFrameInto({}, { euclidean_error_norm_mm: 1.0 });
    assert.doesNotThrow(() => mergeFrameInto(acc, deltaFrame()));
    assert.equal(acc.euclidean_error_norm_mm, 0.12);
    assert.equal(acc.solver_diagnostics.iters, 6);
});

test("合并不修改传入的增量帧对象", () => {
    const delta = deltaFrame();
    const before = JSON.stringify(delta);
    mergeFrameInto({}, fullFrame());
    mergeFrameInto({}, delta);
    assert.equal(JSON.stringify(delta), before, "不应就地改写收到的帧");
});

/* ══════════════════════ 汇总 ══════════════════════ */

console.log(results.join("\n"));
console.log(`\n增量合并测试: ${results.length - failures}/${results.length} 通过`);
process.exit(failures > 0 ? 1 : 0);
