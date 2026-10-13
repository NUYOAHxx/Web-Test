/**
 * 目标指令 payload 构造 —— 全站唯一的"面板取值 → 下发报文"转换点。
 * ================================================================================
 *
 * 为什么单独抽出来
 * ----------------------------------------------------------------------------
 * 坐标系约定是本项目最容易悄悄出错、且**出错不抛异常**的一处:
 *
 *   · 指令面板 (滑块) 始终以**左臂约定**存储 —— 遥测回流经 mirrorY 换算后回填;
 *   · 视口 3D 标记与遥测表显示的是**世界坐标**;
 *   · 后端收到的必须是**世界坐标**, 它自己不做任何镜像 (只按臂名挑关节)。
 *
 * 于是从面板到世界存在**恰好一次**的符号翻转。此前镜像分散在多个调用方:
 * 预设路径做了 mirrorY, 手动按钮路径没做 —— 右臂下手动下发的 Y 与 RPY
 * 的 Roll/Yaw 符号相反, 机器人走向与面板显示、与视口预览都相反的位置,
 * 而日志、HTTP 状态、界面提示一切正常。
 *
 * 集中在此处后, "做几次镜像" 变成一个只有一行的答案, 且可被直接测试。
 *
 * 纯函数, 不读 DOM / 不读全局状态 —— 这是它能被锁住的前提。
 */

import { mirrorYFor, mirrorRpyFor } from "./state.js";

/**
 * 构造下发 6D 目标位姿的请求体。
 *
 * @param {{x:number,y:number,z:number,rpy?:number[]}} panelCmd **左臂约定**的面板取值
 * @param {string} arm "left_arm" | "right_arm"
 * @param {string} presetName 预设名 (自定义时传描述文本)
 * @returns {{x:number,y:number,z:number,arm:string,preset_name:string,rpy?:number[]}}
 *          世界坐标的报文
 */
export function buildTargetPayload(panelCmd, arm, presetName) {
    const payload = {
        x: panelCmd.x,
        y: mirrorYFor(arm, panelCmd.y),
        z: panelCmd.z,
        arm,
        preset_name: presetName,
    };
    // 姿态缺省时不补零 —— 补零会把"未指定朝向"伪装成"朝向为零",
    // 改变后端的约束强度, 行为不同于用户实际意图。
    if (Array.isArray(panelCmd.rpy) && panelCmd.rpy.length === 3) {
        payload.rpy = mirrorRpyFor(arm, panelCmd.rpy);
    }
    return payload;
}

/**
 * 面板左臂约定取值 → 视口世界坐标位置 (供待发预览标记使用)。
 *
 * 与 {@link buildTargetPayload} 共用同一套镜像规则。两者若各写一份, 迟早
 * 会分叉 —— 而分叉的后果正是"预览画在一处、机器人走向另一处"。
 *
 * @param {{x:number,y:number,z:number}} panelCmd
 * @param {string} arm
 * @returns {number[]} 世界坐标 [x, y, z]
 */
export function panelToWorldPosition(panelCmd, arm) {
    return [panelCmd.x, mirrorYFor(arm, panelCmd.y), panelCmd.z];
}

/**
 * 面板左臂约定 RPY → 视口世界坐标 RPY。
 *
 * @param {number[]} panelRpy [roll, pitch, yaw], 左臂约定
 * @param {string} arm
 * @returns {number[]} 世界坐标 RPY
 */
export function panelToWorldRpy(panelRpy, arm) {
    return mirrorRpyFor(arm, panelRpy);
}