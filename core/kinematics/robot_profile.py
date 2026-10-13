"""
机器人描述档案 (Robot Profile)
================================================================================

为什么需要这一层
--------------------------------------------------------------------------------
URDF 本身只描述几何与关节树, 不描述"这条链是腰、这条链是左臂、哪两个连杆
构成支撑足"。这些语义过去散落在代码各处硬编码:

- ``core/kinematics/humanoid_model.py`` 里写死 14 个关节名与 2 个末端 Frame;
- ``core/collision/humanoid_collision.py`` 里靠英文子串 (``"shoulder"``、
  ``"hip"``、``"left_"``) 猜测碰撞对属于哪个身体部位。

后果是换模型时行为不可预期, 且**不报错**。Pinocchio 的 ``getJointId()`` 对
不存在的名称返回 ``model.njoints`` 而非抛异常, 减一之后又恰好是 ``names`` 的
合法下标, 于是求解器会对着一条根本不存在的链求解, 输出看起来合理实则完全
错误的结果。

本模块把上述语义收敛为一份可替换的档案, 并提供**构造期强校验**:

- 档案可由 URDF 同目录的 ``<名字>.profile.json`` 提供, 换模型无需改码;
- 校验任一关节 / Frame / 连杆缺失, 立即抛出异常并列出模型中实际存在的名称,
  杜绝静默错位。

关节限位一律从 URDF 的 ``<limit>`` 读取, 不再由代码字典二次维护。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set

import numpy as np
import pinocchio as pin


class RobotProfileError(RuntimeError):
    """机器人档案与 URDF 不匹配时抛出。

    单独成类而非复用 FileNotFoundError: 换模型时的绝大多数失败都是
    "档案描述的关节在该 URDF 中不存在", 属于模型不匹配而非文件缺失,
    两者需要不同的排查方向。
    """


@dataclass
class RobotProfile:
    """描述"关节链语义"的一组声明。

    字段全部可选: 留空的部分会尝试从 URDF 结构自动推导 (见
    :func:`derive_missing`), 推导不出来再报错。
    """

    name: str = "humanoid"

    #: 腰部关节 (参与腰臂协同求解的躯干自由度)
    waist_joint_names: List[str] = field(default_factory=list)

    #: 机械臂关节, 从肩到腕依次排列
    left_arm_joint_names: List[str] = field(default_factory=list)
    right_arm_joint_names: List[str] = field(default_factory=list)

    #: 末端执行器 Frame
    left_ee_frame: str = ""
    right_ee_frame: str = ""

    #: 构成支撑多边形的连杆 (通常为左右脚)。留空则由 URDF 结构无法推断, 报错。
    support_links: List[str] = field(default_factory=list)

    #: 腿部关节。留空时由"支撑连杆的祖先链"自动推导。
    leg_joint_names: List[str] = field(default_factory=list)

    #: 头部关节。留空时由"不属于腰/臂/腿/支撑的其余关节"推导。
    head_joint_names: List[str] = field(default_factory=list)

    #: 关节限位覆盖 (仅在 URDF 未给出 <limit> 时生效)。
    #: 单位弧度, 形如 {"left_elbow_joint": [-1.0, 2.0]}。
    limit_overrides: Dict[str, Sequence[float]] = field(default_factory=dict)

    #: 头部关节。留空时由"不属于腰/臂/腿/支撑的其余关节"推导。
    head_joint_names: List[str] = field(default_factory=list)

    #: 头部连杆 (无独立关节者)。
    #:
    #: Pinocchio 会把无惯量的纯装饰连杆 (头部外壳、机身标牌、IMU 外壳等) 合并掉,
    #: 它们**不出现在** ``model.names`` 中, 因此无法靠关节回溯区分。于是
    #: ``head_link`` 只能回溯到躯干末端的 ``waist_pitch_joint``, 被归入躯干 ——
    #: 大屏的"臂-头"安全区会整块消失, 且不会有任何报错。
    #: 这类连杆只能按名字显式声明。
    head_links: List[str] = field(default_factory=list)

    #: 基座部位的归类名。根关节直挂的连杆 (pelvis、pelvis_contour、基座 IMU 等)
    #: 沿关节树上溯到根就断了, 无法归入任何已声明关节。
    #:
    #: 它们必须计入躯干: 骨盆是手臂最常发生干涉的部位之一, 若落进 "other",
    #: "臂-躯干"安全区会凭空少掉十几个碰撞对, 而碰撞计算照常进行 ——
    #: 表现为大屏一切正常, 实际却在漏报手臂扫到髋部的场景。
    base_part: str = "torso"

    #: 支撑多边形在 pelvis 坐标系下的显式覆盖 (米)。
    #: 形如 {"x_min": -0.09, "x_max": 0.13, "y_min": -0.17, "y_max": 0.17}
    support_polygon: Optional[Dict[str, float]] = None

    # ── 加载 ──────────────────────────────────────────────────────

    @classmethod
    def from_dict(cls, data: Dict, *, source: str = "<dict>") -> "RobotProfile":
        """由 JSON 字典构造档案。未知字段直接报错, 避免拼写错误被静默忽略。"""
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        unknown = set(data) - known
        if unknown:
            raise RobotProfileError(
                f"档案 {source} 含未知字段: {sorted(unknown)}\n"
                f"可用字段: {sorted(known)}"
            )
        return cls(**data)

    @classmethod
    def load_for_urdf(cls, urdf_path: str) -> "RobotProfile":
        """加载与某 URDF 配套的档案。

        查找规则: 与 URDF 同目录下 ``<basename>.profile.json``。找不到时返回
        :data:`DEFAULT_PROFILE`, 使默认模型无需任何额外文件。
        """
        sidecar = urdf_profile_path(urdf_path)
        if os.path.exists(sidecar):
            with open(sidecar, "r", encoding="utf-8") as f:
                return cls.from_dict(json.load(f), source=sidecar)
        return default_copy()


def urdf_profile_path(urdf_path: str) -> str:
    """给定 URDF 路径, 返回其配套档案的路径约定。"""
    base = os.path.splitext(os.path.basename(urdf_path))[0]
    return os.path.join(os.path.dirname(urdf_path), f"{base}.profile.json")


# ==============================================================================
# 默认档案 —— 本仓库内置模型的语义描述
# ==============================================================================

DEFAULT_PROFILE = RobotProfile(
    name="humanoid_29dof",
    waist_joint_names=["waist_yaw_joint", "waist_pitch_joint"],
    left_arm_joint_names=[
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
    ],
    right_arm_joint_names=[
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ],
    left_ee_frame="left_wrist_yaw_link",
    right_ee_frame="right_wrist_yaw_link",
    support_links=["left_ankle_roll_link", "right_ankle_roll_link"],
    # 头部外壳无独立关节, Pinocchio 已将其合并进躯干, 只能按连杆名声明,
    # 否则大屏的"臂-头"安全区会整块消失。
    head_links=["head_link"],
)


def default_copy() -> RobotProfile:
    """返回默认档案的深拷贝, 供构造器安全使用。"""
    return _deep_copy(DEFAULT_PROFILE)


def _profile_to_dict(p: RobotProfile) -> Dict:
    """按 dataclass 字段通用序列化 —— 手工列举字段的写法会在新增字段时静默漏掉,
    导致该字段取默认值且毫无提示 (曾经就这么漏掉过 head_links)。"""
    return {f: getattr(p, f) for f in RobotProfile.__dataclass_fields__}


# ==============================================================================
# 校验与自动推导
# ==============================================================================


def resolve(
    model: pin.Model,
    profile: RobotProfile,
    *,
    require_support: bool = True,
) -> RobotProfile:
    """校验档案并补全可推导字段, 返回可用的档案。

    校验失败一律抛 :class:`RobotProfileError`, 且异常信息中**列出模型实际
    存在的候选名称** —— 这是本模块存在的意义: 把"静默算出错误结果"变成
    "启动即失败并告诉你怎么改"。

    :param model: 已构建的 Pinocchio 模型
    :param profile: 待校验档案 (可含空字段)
    :param require_support: 是否要求支撑连杆可解析 (仅平衡评估需要)
    :return: 补全后的档案副本
    """
    p = _deep_copy(profile)

    # ── 1. 关节存在性 ──
    _require_joints(model, p.waist_joint_names, "腰部关节")
    _require_joints(model, p.left_arm_joint_names, "左臂关节")
    _require_joints(model, p.right_arm_joint_names, "右臂关节")

    if not p.waist_joint_names:
        raise RobotProfileError("档案未声明腰部关节, 无法构建腰臂协同求解链")
    if not p.left_arm_joint_names and not p.right_arm_joint_names:
        raise RobotProfileError("档案未声明任何机械臂关节, 无法构建 IK 求解链")

    # ── 2. Frame 存在性 ──
    _require_frames(model, [f for f in (p.left_ee_frame, p.right_ee_frame) if f], "末端 Frame")
    if not p.left_ee_frame and not p.right_ee_frame:
        raise RobotProfileError("档案未声明末端 Frame, 无法定义 IK 目标坐标系")

    # ── 3. 腿部关节: 未声明时按"支撑连杆的祖先链"推导 ──
    if not p.leg_joint_names:
        if p.support_links:
            p.leg_joint_names = _ancestor_chain(model, p.support_links)
        else:
            p.leg_joint_names = []

    # ── 4. 头部关节: 未声明时取剩余关节 ──
    if not p.head_joint_names:
        known = set(p.waist_joint_names) | set(p.left_arm_joint_names) \
            | set(p.right_arm_joint_names) | set(p.leg_joint_names) \
            | {joint_short_name(l) for l in p.support_links}
        p.head_joint_names = [
            model.names[i + 1] for i in range(model.njoints - 1)
            if model.names[i + 1] not in known
        ]

    # ── 5. 支撑连杆 ──
    if require_support:
        _require_links(model, p.support_links, "支撑连杆")
        if not p.support_links:
            raise RobotProfileError(
                "档案未声明 support_links, 无法推导支撑多边形。\n"
                "请在 profile.json 中填写左右脚连杆名, 例如:\n"
                '  {"support_links": ["left_ankle_roll_link", "right_ankle_roll_link"]}'
            )

    # ── 6. 限位覆盖项与头部连杆必须存在 ──
    _require_joints(model, list(p.limit_overrides), "限位覆盖中的关节")
    _require_links(model, p.head_links, "头部连杆")

    return p


def _require_joints(model: pin.Model, names: Sequence[str], role: str) -> None:
    missing = [n for n in names if not model.existJointName(n)]
    if missing:
        raise RobotProfileError(
            f"{role}在当前 URDF 中不存在: {missing}\n"
            f"当前模型共 {model.njoints - 1} 个关节, 实际名称示例: "
            f"{_sample(model.names[1:], 12)}"
        )


def _require_frames(model: pin.Model, names: Sequence[str], role: str) -> None:
    missing = [n for n in names if not model.existFrame(n)]
    if missing:
        candidates = _sample([model.frames[i].name for i in range(model.nframes)], 12)
        raise RobotProfileError(
            f"{role}在当前 URDF 中不存在: {missing}\n"
            f"当前模型共 {model.nframes} 个 Frame, 实际名称示例: {candidates}"
        )


def _require_links(model: pin.Model, links: Sequence[str], role: str) -> None:
    # 连杆名与 Frame 名不一定同名, 故两者都查
    missing = [n for n in links if not (model.existFrame(n) or model.existBodyName(n))]
    if missing:
        raise RobotProfileError(
            f"{role}在当前 URDF 中不存在: {missing}\n"
            f"当前模型连杆名称示例: {_sample(model.names[1:], 12)}"
        )


def _ancestor_chain(model: pin.Model, leaf_links: Sequence[str]) -> List[str]:
    """由支撑连杆向上回溯, 收集其全部祖先关节 (支撑关节本身除外)。

    这是"腿部关节"最可靠的通用定义: 双足机器人的腿, 正是从根到足的那段链。

    实现上必须沿 ``model.parents`` 逐级上溯。早期版本试图用"关节名把 _joint
    换成 _link"来定位父连杆, 但父关节本身驱动的正是当前连杆 —— 首轮即自匹配,
    循环立即终止, 结果只收集到踝关节一个, 其余腿部关节全部丢失。
    """
    out: List[str] = []
    seen: Set[str] = set(leaf_links)

    for link in leaf_links:
        if not model.existFrame(link):
            continue
        jid = model.frames[model.getFrameId(link)].parentJoint
        while jid > 0:
            jname = model.names[jid]
            if jname in seen:
                break
            seen.add(jname)
            out.append(jname)
            jid = int(model.parents[jid])
    return out


def joint_short_name(joint_name: str) -> str:
    """``left_hip_pitch_joint`` → ``left_hip_pitch_link``。

    Pinocchio 以关节名为准, 连杆名通常只是把 ``_joint`` 换成 ``_link``;
    在关节→连杆名转换不成立的模型上, 原始名称会被保留在集合中, 由
    :func:`_require_links` 负责报错。
    """
    return joint_name[:-6] + "_link" if joint_name.endswith("_joint") else joint_name


def _sample(items: Sequence[str], n: int) -> List[str]:
    return list(items[:n]) + (["…"] if len(items) > n else [])


def _deep_copy(profile: RobotProfile) -> RobotProfile:
    return RobotProfile.from_dict(json.loads(json.dumps(_profile_to_dict(profile))))


# ==============================================================================
# 关节限位 —— 唯一真源为 URDF 的 <limit>
# ==============================================================================

#: URDF 未给出 <limit> 时 Pinocchio 填的无穷大, 视为"缺失"
_UNBOUNDED = 1e19


def read_joint_limits(
    model: pin.Model,
    joint_names: Sequence[str],
    overrides: Optional[Dict[str, Sequence[float]]] = None,
) -> np.ndarray:
    """从 URDF 读取关节限位, 返回 ``(n, 2)`` 数组 ``[[lower, upper], ...]``。

    URDF 的 ``<limit>`` 是模型作者声明的唯一事实来源, 过去代码却另存一份
    字典, 导致两处可能分叉 —— 且分叉时不会报错, 只会算出错误的可行域。

    URDF 确实缺失限位时 (``lower``/``upper`` 为 ±inf), 依次回退到:

    1. 档案中的 ``limit_overrides``;
    2. 对称默认值 ``[-pi, pi]``。

    :raises RobotProfileError: 档案给出的覆盖值越界或格式非法
    """
    overrides = overrides or {}
    out = np.zeros((len(joint_names), 2), dtype=np.float64)

    # 先整体校验覆盖表: 即便某项当前用不上 (URDF 已给出限位), 非法值也必须报错。
    # 否则档案里的笔误会长期潜伏, 直到某天模型删掉了 <limit> 才突然生效 ——
    # 那时错误已经写进了求解结果。
    for _name, _ov in overrides.items():
        if len(_ov) != 2:
            raise RobotProfileError(
                f"关节 {_name} 的 limit_overrides 必须为 [lower, upper] 两元组, "
                f"实际得到 {list(_ov)!r}")
        if float(_ov[0]) > float(_ov[1]):
            raise RobotProfileError(
                f"关节 {_name} 的限位覆盖下界 {float(_ov[0]):.4f} 大于上界 "
                f"{float(_ov[1]):.4f}, 请修正 profile.json")

    for i, name in enumerate(joint_names):
        idx = model.getJointId(name) - 1
        lo = float(model.lowerPositionLimit[idx])
        hi = float(model.upperPositionLimit[idx])

        if not (abs(lo) >= _UNBOUNDED or abs(hi) >= _UNBOUNDED):
            out[i] = (lo, hi)
            continue

        ov = overrides.get(name)
        if ov is not None:
            # 格式与序关系已在上面整体校验过, 此处只需取值
            out[i] = (float(ov[0]), float(ov[1]))
        else:
            # URDF 未声明且无覆盖 —— 取对称默认, 范围足够宽以免误剪枝,
            # 实际可用域仍由 pink 的限位层结合 SRDF 判定。
            out[i] = (-np.pi, np.pi)

    return out


# ==============================================================================
# 支撑多边形 —— 由足部碰撞几何的 AABB 推导
# ==============================================================================


def derive_support_polygon(
    model: pin.Model,
    data: pin.Data,
    coll_model: pin.GeometryModel,
    support_links: Sequence[str],
) -> Dict[str, float]:
    """由支撑连杆的碰撞几何在参考姿态下的包围盒并集推导支撑多边形。

    过去该矩形是照抄某型号的双足尺寸写死的。脚掌大小不同的模型, 平衡裕度会
    系统性偏大或偏小, 且没有任何提示 —— 表现为"大屏上一切正常, 但机器人其实
    已经踩在支撑边界之外"。

    推导方式: 取支撑连杆子树内全部碰撞几何, 把各自局部包围盒的 8 个角点变换
    到世界坐标系, 再取 XY 平面的并集。

    :return: ``{"x_min", "x_max", "y_min", "y_max"}``, 单位米
    :raises RobotProfileError: 支撑连杆下没有可用碰撞几何
    """
    if not support_links:
        raise RobotProfileError("无法推导支撑多边形: 档案未声明 support_links")

    support_frame_ids: Set[int] = set()
    for link in support_links:
        if model.existFrame(link):
            support_frame_ids.add(model.getFrameId(link))
    if not support_frame_ids:
        raise RobotProfileError(f"支撑连杆在当前 URDF 中没有对应 Frame: {list(support_links)}")

    # 参考姿态下各连杆的世界位姿。
    # forwardKinematics 只填各关节的 placement, frame 的世界位姿在 data.oMf,
    # 必须再调 updateFramePlacements —— 漏掉它会让所有几何被当成在原点,
    # 左右脚叠在一起, 支撑多边形宽度退化成单只脚掌。
    pin.forwardKinematics(model, data, pin.neutral(model))
    pin.updateFramePlacements(model, data)

    lo = np.array([np.inf, np.inf])
    hi = np.array([-np.inf, -np.inf])
    used, skipped = 0, 0

    for g in coll_model.geometryObjects:
        # parentFrame 是整数下标。这里用范围判断而非 `in model.frames` ——
        # 后者在 Boost.Python 容器上语义不可靠, 会把合法的几何体一并跳过。
        if not isinstance(g.parentFrame, int) or not (0 <= g.parentFrame < model.nframes):
            skipped += 1
            continue
        if not _under_any(model, g.parentFrame, support_frame_ids):
            skipped += 1
            continue

        local = _local_aabb(g.geometry)
        if local is None:
            skipped += 1
            continue

        # GeometryObject.placement 是几何体相对所在 frame 的附加变换, 不可遗漏
        placement = data.oMf[g.parentFrame] * g.placement
        corners = np.array([
            placement.act(np.array([x, y, z]))
            for x in (local[0][0], local[1][0])
            for y in (local[0][1], local[1][1])
            for z in (local[0][2], local[1][2])
        ])
        lo = np.minimum(lo, corners[:, :2].min(axis=0))
        hi = np.maximum(hi, corners[:, :2].max(axis=0))
        used += 1

    if used == 0:
        raise RobotProfileError(
            f"支撑连杆 {list(support_links)} 子树下未找到任何可用碰撞几何 "
            f"(扫描 {len(coll_model.geometryObjects)} 个, 跳过 {skipped} 个), "
            "无法推导支撑多边形。\n"
            "请确认 URDF 为这些连杆定义了 <collision> 且类型为 mesh / box / "
            "sphere / cylinder / capsule。"
        )

    return {
        "x_min": round(float(lo[0]), 4),
        "x_max": round(float(hi[0]), 4),
        "y_min": round(float(lo[1]), 4),
        "y_max": round(float(hi[1]), 4),
    }


def _under_any(model: pin.Model, frame_id: int, roots: Set[int]) -> bool:
    """frame_id 是否位于 roots 中任一 frame 的子树内 (含自身)。

    下界判断一律用 ``0 <= idx < n`` 范围比较, 不能写 ``idx in model.frames`` ——
    Boost.Python 暴露的容器不支持成员判断, 该表达式恒为 False, 会让整段查找
    直接跳过, 表现为"明明命中根节点却判定为不在子树内"。
    """
    seen: Set[int] = set()
    cur = int(frame_id)
    while cur not in seen and 0 <= cur < model.nframes:
        if cur in roots:
            return True
        seen.add(cur)
        jid = model.frames[cur].parentJoint
        if jid <= 0:
            return False
        # 沿关节树逐级上溯: 当前连杆由 jid 驱动, 其父连杆由 jid 的父关节驱动。
        # 不能用"关节名换算连杆名"来定位 —— 那会回到当前连杆自身。
        parent_joint = int(model.parents[jid])
        if parent_joint <= 0:
            return False
        cur = _frame_of_joint(model, parent_joint)
        if cur is None:
            return False
    return False


def _frame_of_joint(model: pin.Model, joint_id: int) -> Optional[int]:
    """关节所驱动的连杆 frame 索引。"""
    target = model.names[joint_id]
    for i in range(model.nframes):
        if model.frames[i].name == target:
            return i
    return None


def _local_aabb(geometry) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """取几何体局部坐标系下的轴对齐包围盒 ``(min, max)``。

    逐类型处理, 因为 Pinocchio 的 ``aabb_local`` 需要显式构建凸包才能填充,
    而不同几何类型支持的构建接口并不一致。

    :return: 包围盒, 或该几何类型不受支持时返回 None
    """
    # 网格 (BVH): 直接取全部顶点
    if hasattr(geometry, "vertices") and hasattr(geometry, "num_vertices"):
        n = int(geometry.num_vertices())
        if n <= 0:
            return None
        pts = np.array([np.asarray(geometry.vertices(i), dtype=np.float64)
                        for i in range(n)])
        return pts.min(axis=0), pts.max(axis=0)

    # 球
    if hasattr(geometry, "radius") and not hasattr(geometry, "halfHeight"):
        # 本环境所用 Pinocchio 版本的 ShapeSphere 不暴露 center —— 球心即局部
        # 原点, 偏移由 GeometryObject.placement 携带, 故默认取原点。
        c = np.asarray(getattr(geometry, "center", np.zeros(3)), dtype=np.float64)
        r = float(geometry.radius)
        return c - r, c + r

    # 圆柱 / 胶囊 (沿 z 轴)
    if hasattr(geometry, "halfHeight") and hasattr(geometry, "radius"):
        c = np.asarray(getattr(geometry, "center", np.zeros(3)), dtype=np.float64)
        r = float(geometry.radius)
        h = float(geometry.halfHeight)
        return c - np.array([r, r, h]), c + np.array([r, r, h])

    return None
