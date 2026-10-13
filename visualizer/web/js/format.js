/**
 * 数值格式化与**统一的告警阈值定义**。
 *
 * 存在理由 (针对旧 app.js 的真实缺陷):
 *
 * 旧代码对"最小碰撞间隙"的严重度判定散落在 4 处, 且阈值互相矛盾:
 *   · 间隙数字着色:   <10mm 红, <22mm 黄          (app.js 旧 1063 行)
 *   · 状态徽章:       <5mm ALERT, <20mm CAUTION   (旧 1100/1106 行)
 *   · 告警清单框:     仅 is_colliding 时显示       (旧 1098 行)
 *   · 分区卡片:       <5mm ALERT, <20mm CAUTION   (旧 1290 行)
 *
 * 结果是 8mm 间隙时界面自相矛盾: 大号数字已标红, 而状态徽章仍显示 SECURE,
 * 且碰撞告警框被隐藏。安全指示互相打架是比阈值不准更严重的问题。
 *
 * 现在全部阈值集中在本文件, 且**严重度只计算一次**再分发给各展示位,
 * 从结构上杜绝了"同一物理量在界面上有多种说法"。
 */

/* ═══════════════════ 告警阈值 (唯一真源) ═══════════════════ */

/** 最小碰撞间隙 (mm) */
export const CLEARANCE = {
    ALERT: 5.0,        // 低于此值: 告警
    WARN: 20.0,        // 低于此值: 注意
    SCALE_MAX: 50.0,   // 指示条满量程
};

/** 位置跟踪残差 (mm) */
export const POS_ERR = {
    GOOD: 2.0,
    WARN: 15.0,
    SCALE_MAX: 20.0,
};

/** 姿态跟踪残差 (°) */
export const ROT_ERR = {
    GOOD: 1.5,
    WARN: 5.0,
};

/**
 * 求解器收敛判定 (mm) —— 求解器侧容差比跟踪侧更宽, 因为它是中间量。
 *
 * 字段名用 GOOD/WARN 而非 CONVERGED/APPROX, 以便直接传给 toneByMagnitude。
 * 两者语义相同 (GOOD=收敛线, WARN=接近但未收敛), 只是命名对齐了判定方向。
 */
export const SOLVER = {
    GOOD: 2.5,        // 不超过此值判定收敛
    WARN: 15.0,       // 超过此值判定陷入局部极小
    TOL_MM: 2.0,      // 曲线上绘制的容差参考线
};

/** 双足平衡裕度 (mm) */
export const BALANCE = {
    LEANING: 30.0,
    STABLE: 60.0,
    SCALE_MAX: 120.0,
};

/** 关节距物理限位的余量 (°) */
export const JOINT_MARGIN = {
    ALERT: 6.0,
    WARN: 15.0,
};

/**
 * 重力补偿力矩幅值 (N·m) —— "越小越好"的量, 因此字段名用 GOOD/WARN 对齐
 * toneByMagnitude 的签名。
 *
 * 原字段名为 {HEAVY, CRITICAL}, 被误传给读取 {ALERT, WARN} 的 severity(),
 * 阈值全为 undefined, `value < undefined` 恒为 false —— 力矩配色整体失效,
 * 9.7N·m 与 0.0N·m 渲染完全一致。
 */
export const TORQUE = {
    /**
     * 以下两项是**利用率**阈值而非力矩绝对值: 判定量为
     * |τ| / 该关节的结构性上限 (上限由服务端按模型质量分布与限位标定)。
     *
     * 曾用绝对值 (4 / 8 N·m) 判断全部关节, 后果是:
     *   - 腰部俯仰承载躯干, 合法上限实测 12.9 N·m → 长期误报告警;
     *   - 腕关节恒 < 0.5 N·m → 任何工况都显示正常。
     * 阈值因此失去区分能力, 换模型后偏差更大。
     */
    GOOD: 0.45,     // 利用率低于此值视为正常负载
    WARN: 0.80,     // 超过此值视为接近结构上限
};

/* ═══════════════════ 严重度判定 ═══════════════════ */

/*
 * 判定方向必须显式化。
 *
 * 此前只有一个 severity(value, {ALERT, WARN}), 隐含约定是"低值 = 坏"
 * (适用于碰撞间隙、关节限位余量)。两类指标因此被误用:
 *
 *   ① 误差类指标 (残差/力矩) 的语义是"低值 = 好"。把 GOOD 当作 ALERT 传入后,
 *      判定整体反转 —— 0.00mm 的完美收敛被标成红色告警。
 *   ② 力矩处传入了 {HEAVY, CRITICAL} 而函数读的是 {ALERT, WARN},
 *      阈值全为 undefined, `value < undefined` 恒为 false, 配色功能整体失效。
 *
 * 现拆为两个方向明确的函数, 参数名自带语义, 从签名上杜绝再次误用。
 */

/**
 * 低值 = 坏 (间隙、限位余量、平衡裕度)。
 * @param {number} value
 * @param {{ALERT:number, WARN:number}} th 低于 ALERT 判告警, 低于 WARN 判注意
 * @returns {"alert"|"warn"|"ok"}
 */
export function severity(value, th) {
    if (!Number.isFinite(value)) return "ok";
    if (value < th.ALERT) return "alert";
    if (value < th.WARN) return "warn";
    return "ok";
}

/**
 * 低值 = 好 (残差、力矩幅值等"越小越好"的量)。
 * @param {number} value
 * @param {{GOOD:number, WARN:number}} th 不超过 GOOD 判良好, 不超过 WARN 判注意
 * @returns {"green"|"gold"|"red"}
 */
export function toneByMagnitude(value, th) {
    if (!Number.isFinite(value)) return "muted";
    if (value < th.GOOD) return "green";
    if (value < th.WARN) return "gold";
    return "red";
}

/**
 * 碰撞间隙的综合严重度。
 *
 * 这是界面**唯一**的间隙安全判定入口 —— 数字颜色、刻度标尺、状态徽章、
 * 顶栏安全芯片、碰撞告警框全部消费这一个结果。
 *
 * @param {number} clearanceMm
 * @param {boolean} isColliding 实际检测到碰撞 (优先于间隙阈值)
 * @returns {{level:"alert"|"warn"|"ok", text:string}}
 */
export function clearanceStatus(clearanceMm, isColliding) {
    if (isColliding) {
        return { level: "alert", text: "COLLISION" };
    }
    if (!Number.isFinite(clearanceMm)) {
        return { level: "ok", text: "NO DATA" };
    }
    if (clearanceMm < CLEARANCE.ALERT) {
        return { level: "alert", text: `ALERT <${CLEARANCE.ALERT}mm` };
    }
    if (clearanceMm < CLEARANCE.WARN) {
        return { level: "warn", text: `CAUTION <${CLEARANCE.WARN}mm` };
    }
    return { level: "ok", text: "SECURE" };
}

/**
 * 求解结果状态标签。
 * @param {number} iters
 * @param {number} posErrMm
 * @returns {{label:string, level:"alert"|"warn"|"ok"}}
 */
export function solverStatus(iters, posErrMm) {
    if (!iters) return { label: "STANDBY", level: "ok" };
    if (posErrMm < SOLVER.GOOD) return { label: "CONVERGED", level: "ok" };
    if (posErrMm < SOLVER.WARN) return { label: "APPROXIMATE", level: "warn" };
    return { label: "LOCAL MIN", level: "alert" };
}

/**
 * 平衡状态。
 * @param {number} marginMm
 * @returns {{status:string, level:"alert"|"warn"|"ok"}}
 */
export function balanceStatus(marginMm) {
    if (!Number.isFinite(marginMm)) return { status: "NO DATA", level: "ok" };
    if (marginMm >= BALANCE.STABLE) return { status: "STABLE", level: "ok" };
    if (marginMm >= BALANCE.LEANING) return { status: "LEANING", level: "warn" };
    return { status: "TIPPING", level: "alert" };
}

/**
 * 把数值钳到 0~100, 用于进度条 / 刻度条。
 * @param {number} value
 * @param {number} full 满量程
 * @returns {number}
 */
export function pct(value, full) {
    if (!Number.isFinite(value) || full <= 0) return 0;
    return Math.max(0, Math.min(100, (value / full) * 100));
}

/* ═══════════════════ 格式化 ═══════════════════ */

/**
 * 带符号定点数。
 * @param {number} v
 * @param {number} digits
 * @returns {string}
 */
export function signed(v, digits = 3) {
    if (!Number.isFinite(v)) return "—";
    return (v >= 0 ? "+" : "") + v.toFixed(digits);
}

/**
 * 不带符号的定点数。
 * @param {number} v
 * @param {number} digits
 * @returns {string}
 */
export function fixed(v, digits = 2) {
    return Number.isFinite(v) ? v.toFixed(digits) : "—";
}

/**
 * 位置残差 (mm), 固定 1 位。
 * @param {number} v
 * @returns {string}
 */
export function mm(v, digits = 1) {
    return Number.isFinite(v) ? `${signed(v, digits)} mm` : "—";
}

/**
 * 角度。
 * @param {number} v
 * @param {number} digits
 * @returns {string}
 */
export function deg(v, digits = 1) {
    return Number.isFinite(v) ? `${signed(v, digits)}°` : "—";
}

/**
 * 从数组安全取第 n 项, 越界返回 null 而不是抛错。
 * 遥测报文缺字段是常态 (待机时多数字段为空), 不应让渲染整体崩掉。
 * @param {ArrayLike<number>} arr
 * @param {number} idx
 * @returns {number|null}
 */
export function at(arr, idx) {
    if (!arr || idx >= arr.length) return null;
    const v = arr[idx];
    return Number.isFinite(v) ? v : null;
}

/**
 * 安全取对象字段。
 * @param {object} obj
 * @param {string} key
 * @param {*} fallback
 */
export function pick(obj, key, fallback = null) {
    if (!obj) return fallback;
    const v = obj[key];
    return v === undefined || v === null ? fallback : v;
}