/**
 * 前端共享可变状态。
 *
 * 旧 app.js 在全局作用域散落着 15 个 `let` 可变变量 (scene/camera/currentArm/
 * hoverTraceIndex/lastRenderedLogCount/...)。这些变量被十余个函数隐式读写,
 * 无法追踪谁会改什么, 也没有任何生命周期边界。
 *
 * 这里把它们收拢为一个显式对象: 谁持有引用、谁负责修改一目了然,
 * 且模块间传递的是这个对象而非隐式全局。
 */

export const state = {
    /** 当前操作臂: "left_arm" | "right_arm" */
    arm: "left_arm",

    /** 指令面板当前选定的 6D 目标位姿 (未经镜像处理的"自身坐标系"取值) */
    cmd: {
        x: 0.35,
        y: 0.22,
        z: 0.85,
        roll: 0.0,
        pitch: 0.0,
        yaw: 0.0,
    },

    /**
     * 用户是否正在操作指令滑块。
     * 拖动期间暂停遥测回填, 否则服务端 30Hz 回流会把滑块"抢走",
     * 表现为拖动时数值不断跳回。
     */
    userDraggingSlider: false,

    /**
     * 指令面板是否有**尚未下发的本地修改**。
     *
     * 这解决的是"拖完松手又弹回去"的问题。面板里滑块编辑的是**待下发指令**,
     * 而遥测每帧推送的是**服务器当前生效指令** —— 两者是不同的量。
     * 松手后 userDraggingSlider 清零, 下一帧遥测就把服务器上的旧值写回滑块,
     * 用户刚拖出来的数值凭空消失。
     *
     * 因此编辑后必须置位此标记, 遥测回填在标记存在时让位;
     * 直到指令真正下发(或与服务器值一致)才清除。
     */
    cmdPending: false,

    /** IK 收敛诊断视图的交互与渲染状态 */
    trace: {
        data: null,        // 最近一次 solver_diagnostics
        hover: -1,         // 当前悬停的迭代序号, -1 表示无
        lastKey: "",       // 上次渲染的指纹, 用于跳过无效重绘
    },

    /** 日志流增量渲染状态 */
    logs: {
        lastCount: 0,
        lastStamp: "",
    },

    /** 已成功加载的 CAD 网格数量 (视口角标展示) */
    meshCount: 0,
};

/**
 * 右臂时对 Y 与部分姿态角取镜像。
 *
 * 世界坐标系下左右臂关于 XZ 平面镜像, 而指令面板的 Y 滑块始终以
 * "左正右负"呈现, 因此切臂时需要对 Y / Roll / Yaw 取反。
 *
 * @param {number} v
 * @returns {number}
 */
export function mirrorY(v) {
    return state.arm === "right_arm" ? -v : v;
}

/**
 * 右臂姿态镜像: Roll 与 Yaw 反向, Pitch 不变。
 * @param {number[]} rpy
 * @returns {number[]}
 */
export function mirrorRpy(rpy) {
    if (state.arm !== "right_arm") return [rpy[0], rpy[1], rpy[2]];
    return [-rpy[0], rpy[1], -rpy[2]];
}