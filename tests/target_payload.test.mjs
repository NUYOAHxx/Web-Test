/**
 * 目标 payload 与预览坐标的一致性测试
 * ================================================================================
 *
 * 这条不变量的性质: 面板存左臂约定, 视口与后端用世界坐标, 中间**恰好翻转一次**。
 *
 * 曾经破过: 镜像分散在调用方, 预设路径做了、手动按钮路径没做。右臂下手动
 * 下发的 Y 与 RPY 的 Roll/Yaw 符号相反 —— 机器人走向与面板显示、与视口
 * 预览都相反的位置, 而 HTTP 200、日志正常、界面无任何提示。
 *
 * 这类错误不会抛异常, 只能靠测试锁住。
 */

import assert from "node:assert/strict";
import {
    buildTargetPayload,
    panelToWorldPosition,
    panelToWorldRpy,
} from "../visualizer/web/js/target_payload.js";
import { mirrorYFor, mirrorRpyFor } from "../visualizer/web/js/state.js";

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

const ARMS = ["left_arm", "right_arm"];

test("左臂: 面板值即世界值, 镜像为恒等", () => {
    const p = buildTargetPayload({ x: 0.35, y: 0.22, z: 0.85 }, "left_arm", "t");
    assert.deepEqual([p.x, p.y, p.z], [0.35, 0.22, 0.85]);
});

test("右臂: Y 取反", () => {
    const p = buildTargetPayload({ x: 0.35, y: 0.22, z: 0.85 }, "right_arm", "t");
    assert.equal(p.y, -0.22);
});

test("右臂: Roll/Yaw 反向, Pitch 保持", () => {
    const p = buildTargetPayload(
        { x: 0, y: 0, z: 0, rpy: [10, 20, 30] }, "right_arm", "t");
    assert.deepEqual(p.rpy, [-10, 20, -30]);
});

/* ── 本次修复的核心断言 ── */

test("预览位置与下发位置一致 (两臂)", () => {
    // 曾经的 bug: 预览做了镜像, 手动下发没做 —— 两者给出不同的世界坐标
    const panel = { x: 0.35, y: 0.22, z: 0.85 };
    for (const arm of ARMS) {
        assert.deepEqual(
            panelToWorldPosition(panel, arm),
            [buildTargetPayload(panel, arm, "t").x,
             buildTargetPayload(panel, arm, "t").y,
             buildTargetPayload(panel, arm, "t").z],
            `${arm}: 预览与下发的世界坐标必须相同`,
        );
    }
});

test("预览姿态与下发姿态一致 (两臂)", () => {
    const panelRpy = [15, -30, 45];
    for (const arm of ARMS) {
        assert.deepEqual(
            panelToWorldRpy(panelRpy, arm),
            buildTargetPayload({ x: 0, y: 0, z: 0, rpy: panelRpy }, arm, "t").rpy,
            `${arm}: 预览与下发的姿态必须相同`,
        );
    }
});

/* ── 闭环: 遥测世界值 → 面板 → 世界值, 应当原样返回 ── */

test("闭环不变量: 世界值经面板往返后不变", () => {
    for (const arm of ARMS) {
        for (const worldY of [0.212, -0.212, 0, -0.47]) {
            const panelY = mirrorYFor(arm, worldY);            // syncSlidersFromTelemetry
            const back = buildTargetPayload(
                { x: 0.1, y: panelY, z: 0.8 }, arm, "t").y;   // dispatch
            assert.ok(Math.abs(back - worldY) < 1e-12,
                `${arm}: ${worldY} → 面板 ${panelY} → 世界 ${back}`);
        }
    }
});

test("闭环不变量: 姿态经面板往返后不变", () => {
    const worldRpy = [14.92, 38.93, 10.41];
    for (const arm of ARMS) {
        const panelRpy = mirrorRpyFor(arm, worldRpy);
        const back = buildTargetPayload(
            { x: 0, y: 0, z: 0, rpy: panelRpy }, arm, "t").rpy;
        for (let i = 0; i < 3; i++) {
            assert.ok(Math.abs(back[i] - worldRpy[i]) < 1e-12,
                `${arm}: 轴 ${i} 未还原 ${worldRpy[i]} → ${panelRpy[i]} → ${back[i]}`);
        }
    }
});

test("镜像是对合的 (镜像两次回到原值)", () => {
    for (const arm of ARMS) {
        assert.ok(Math.abs(mirrorYFor(arm, mirrorYFor(arm, 0.37)) - 0.37) < 1e-12);
        assert.deepEqual(mirrorRpyFor(arm, mirrorRpyFor(arm, [11, 22, 33])), [11, 22, 33]);
    }
});

/* ── payload 形状 ── */

test("缺少 rpy 时不下发 rpy 字段 (补零会伪装成零朝向)", () => {
    const p = buildTargetPayload({ x: 0, y: 0, z: 0 }, "right_arm", "t");
    assert.equal("rpy" in p, false);
});

test("rpy 长度不为 3 时不下发 (避免发出畸形姿态)", () => {
    assert.equal("rpy" in buildTargetPayload({ x: 0, y: 0, z: 0, rpy: [1, 2] }, "left_arm", "t"), false);
    assert.equal("rpy" in buildTargetPayload({ x: 0, y: 0, z: 0, rpy: [1, 2, 3, 4] }, "left_arm", "t"), false);
});

test("payload 带臂名与预设名", () => {
    const p = buildTargetPayload({ x: 0, y: 0, z: 0 }, "right_arm", "桌面取物");
    assert.equal(p.arm, "right_arm");
    assert.equal(p.preset_name, "桌面取物");
});

test("X 与 Z 不受臂别影响", () => {
    for (const arm of ARMS) {
        const p = buildTargetPayload({ x: 0.45, y: 0.3, z: 1.02 }, arm, "t");
        assert.equal(p.x, 0.45, `${arm}: X 不该被镜像`);
        assert.equal(p.z, 1.02, `${arm}: Z 不该被镜像`);
    }
});

console.log(results.join("\n"));
console.log(`\npayload 一致性测试: ${results.length - failures}/${results.length} 通过`);
process.exit(failures > 0 ? 1 : 0);