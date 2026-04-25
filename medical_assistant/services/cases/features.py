from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class FeatureDefinition:
    feature_id: str
    label: str
    question: str
    positive_patterns: tuple[str, ...]


FEATURES: tuple[FeatureDefinition, ...] = (
    FeatureDefinition(
        "cold_sweet_short",
        "冷/甜刺激后短暂酸痛",
        "疼痛是否主要由冷水、冷风、甜食或酸饮触发，并且去掉刺激后很快缓解？请回答“是/不是/不确定”。",
        ("冷", "冰", "冷风", "凉", "甜", "酸饮", "酸一下", "刺一下", "很快消失", "短暂冷痛"),
    ),
    FeatureDefinition(
        "cold_hot_lingering",
        "冷热刺激都会痛且余痛较久",
        "冷的和热的是否都会诱发疼痛，而且痛感会拖一阵才缓解？请回答“是/不是/不确定”。",
        ("冷热都", "冷的热的", "热汤", "火锅", "冰饮", "余痛", "拖一阵", "持续时间", "冷热反复"),
    ),
    FeatureDefinition(
        "heat_night_spontaneous",
        "热痛/自发痛/夜间痛",
        "有没有不碰也会自己痛、夜里更痛，或者热刺激明显加重的情况？请回答“是/不是/不确定”。",
        ("热痛", "热刺激", "热汤", "夜", "痛醒", "自发痛", "不碰也", "跳痛", "平躺", "冷水反而舒服"),
    ),
    FeatureDefinition(
        "pulsing_pain",
        "搏动样跳痛",
        "疼痛是否像心跳一样一下一下跳着痛，尤其晚上明显？请回答“是/不是/不确定”。",
        ("搏动", "像心跳", "一下一下", "跳着", "跳痛"),
    ),
    FeatureDefinition(
        "food_impaction",
        "食物嵌塞后加重",
        "疼痛是否常在吃完东西塞住后加重，清出来后会缓解一些？请回答“是/不是/不确定”。",
        ("塞", "嵌塞", "卡东西", "塞东西", "肉丝", "菜叶", "剔出来", "挑出来", "牙线"),
    ),
    FeatureDefinition(
        "visible_cavity_or_black",
        "可见龋洞/发黑/残冠",
        "这颗牙是否能看到黑点、龋洞、缺损，或者已经烂掉一部分？请回答“是/不是/不确定”。",
        ("黑", "龋", "蛀", "牙洞", "洞", "烂", "坏牙", "残根", "残冠", "只剩", "缺损"),
    ),
    FeatureDefinition(
        "tooth_discoloration_or_dead_pulp",
        "牙齿变色/死髓或牙根深部不适",
        "这颗牙是否比旁边牙颜色更暗、发灰发黄，或怀疑牙神经已经坏死？请回答“是/不是/不确定”。",
        ("变色", "死髓", "灰", "发黄", "颜色", "牙根", "根部"),
    ),
    FeatureDefinition(
        "single_tooth_localized",
        "疼痛能定位到单颗牙",
        "你能否大致指出是某一颗牙在痛，而不是一片或整排都不舒服？请回答“是/不是/不确定”。",
        ("某颗", "一颗", "这颗", "固定", "能指出", "大概位置", "单颗"),
    ),
    FeatureDefinition(
        "bite_or_percussion_pain",
        "咬合/叩痛明显",
        "上下牙一碰、咬东西、敲到或按到那颗牙时，疼痛是否明显加重？请回答“是/不是/不确定”。",
        ("咬", "咬合", "咬痛", "叩痛", "敲", "碰一下", "先碰", "牙高", "顶起来", "按压痛", "咬硬"),
    ),
    FeatureDefinition(
        "gum_bump_or_pus",
        "牙龈鼓包/流脓/瘘管",
        "牙龈上是否有鼓包、小口、流脓，或者按压像有脓液？请回答“是/不是/不确定”。",
        ("鼓包", "脓", "流脓", "瘘", "小口", "冒东西", "液体感", "包", "挤"),
    ),
    FeatureDefinition(
        "face_swelling_fever",
        "面颊肿胀/发热/全身不适",
        "有没有脸肿、发热、明显乏力，或肿胀向外扩散的情况？请回答“是/不是/不确定”。",
        ("脸肿", "面颊", "半边脸", "发热", "低热", "体温", "乏力", "全身", "肿胀扩散", "热热的"),
    ),
    FeatureDefinition(
        "wisdom_back_gum",
        "最后面智齿区牙龈肿痛",
        "疼痛是否主要在最后面的智齿附近，像最里面那块牙龈或肉肿了？请回答“是/不是/不确定”。",
        ("智齿", "最后面", "最里面", "后面那块肉", "牙龈盖", "半露", "半萌", "冠周炎", "后牙区"),
    ),
    FeatureDefinition(
        "mouth_open_limited_swallow",
        "张口受限/吞咽牵扯痛",
        "有没有嘴张不大、吞咽牵扯痛，或张嘴说话刷牙时明显加重？请回答“是/不是/不确定”。",
        ("张口", "嘴张不大", "张嘴", "吞咽", "牵扯", "说话", "刷牙碰进去"),
    ),
    FeatureDefinition(
        "wisdom_impaction_pressure",
        "智齿顶压邻牙/阻生",
        "是否像最后面的牙在顶前面一颗大牙，或以前被说过智齿横着/阻生？请回答“是/不是/不确定”。",
        ("阻生", "横着", "斜着", "顶", "顶压", "邻牙", "前面那颗", "第二磨牙"),
    ),
    FeatureDefinition(
        "gum_bleeding_wide",
        "牙龈红肿/刷牙出血/范围较广",
        "是否是一片牙龈红肿、发胀或刷牙出血，而不是单颗牙深处痛？请回答“是/不是/不确定”。",
        ("牙龈炎", "牙周袋", "牙龈肿", "牙龈红", "红肿", "刷牙出血", "出血", "一片", "整口", "一嘴", "牙龈边缘"),
    ),
    FeatureDefinition(
        "tartar_or_bad_breath",
        "牙石/口臭/清洁困难",
        "是否有牙石、口臭异味，或长期清洁困难导致的牙龈不适？请回答“是/不是/不确定”。",
        ("牙石", "牙结石", "吸烟", "硬硬", "口臭", "异味", "臭味", "清洁差", "刷不到", "多年没洗牙", "卫生差"),
    ),
    FeatureDefinition(
        "loose_tooth",
        "牙齿松动或发飘",
        "这颗牙是否有松动、发飘，或咬东西时不稳的感觉？请回答“是/不是/不确定”。",
        ("松动", "晃", "发飘", "牙缝变大", "不稳", "踩在棉花"),
    ),
    FeatureDefinition(
        "gum_recession_root_sensitive",
        "牙龈退缩/牙根暴露敏感",
        "是否像牙根露出来或牙龈退缩，喝冷水、吸冷风或刷到根面时酸？请回答“是/不是/不确定”。",
        ("牙龈退缩", "牙龈萎缩", "牙本质", "磨耗", "风吹", "牙根露", "根面", "楔状", "小凹槽", "横刷", "牙颈部"),
    ),
    FeatureDefinition(
        "crack_or_sharp_bite_pain",
        "裂纹/咬到某一下瞬痛",
        "是否主要是咬到某个点或松口那一下突然尖锐疼，平时不碰未必一直痛？请回答“是/不是/不确定”。",
        ("隐裂", "裂", "裂纹", "松口", "电一下", "某一下", "瞬痛", "尖锐", "咬坚果", "啃排骨"),
    ),
    FeatureDefinition(
        "broken_edge_or_trauma",
        "崩裂/外伤/锐边敏感",
        "最近是否有牙崩了一角、被撞过、磕过，或出现锐边刺激？请回答“是/不是/不确定”。",
        ("崩", "崩裂", "锐边", "舌头碰", "外伤", "撞", "磕", "咬冰", "缺口"),
    ),
    FeatureDefinition(
        "bruxism_or_morning_soreness",
        "磨牙/紧咬/晨起酸痛",
        "是否早上起来整排牙酸、咬肌紧，或有夜磨牙、压力大紧咬的背景？请回答“是/不是/不确定”。",
        ("磨牙", "紧咬", "晨起", "早上", "咬肌", "下巴累", "压力", "整排", "整边"),
    ),
    FeatureDefinition(
        "post_treatment",
        "补牙/牙冠/根管/种植/正畸后出现",
        "这个疼痛是否发生在补牙、戴冠、根管、种植、正畸调整或临时冠脱落之后？请回答“是/不是/不确定”。",
        ("补牙后", "补完", "做过", "牙冠", "戴冠", "根管", "种植", "正畸", "牙套", "保持器", "临时冠", "牙桥", "治疗后"),
    ),
    FeatureDefinition(
        "child_or_erupting_tooth",
        "儿童/乳牙/换牙/萌出相关",
        "患者是否是儿童，或疼痛和乳牙、换牙、恒牙萌出、低龄萌牙有关？请回答“是/不是/不确定”。",
        ("儿童", "孩子", "小孩", "乳牙", "换牙", "恒牙萌出", "萌牙", "哭醒", "窝沟"),
    ),
    FeatureDefinition(
        "sinus_or_nasal_related",
        "鼻窦/鼻塞牵涉上牙痛",
        "是否伴随鼻塞、鼻窦不适，低头时上后牙胀痛更明显？请回答“是/不是/不确定”。",
        ("鼻", "鼻塞", "鼻窦", "低头", "上后牙", "上牙胀", "流涕"),
    ),
    FeatureDefinition(
        "jaw_muscle_joint_related",
        "颞下颌关节/咀嚼肌牵涉痛",
        "是否伴随张闭口关节不适、咀嚼肌酸紧，像关节或肌肉牵涉到后牙？请回答“是/不是/不确定”。",
        ("颞下颌", "关节", "咀嚼肌", "脸颊", "肌肉", "张闭口", "弹响"),
    ),
    FeatureDefinition(
        "neuralgic_or_electric",
        "电击样阵发痛/神经痛样",
        "疼痛是否像电击一样突然发作、持续很短、反复触发？请回答“是/不是/不确定”。",
        ("三叉神经", "电击", "刀割", "突然", "阵发", "触发点", "几秒"),
    ),
    FeatureDefinition(
        "ear_throat_heart_referred",
        "耳咽或心脏牵涉痛线索",
        "是否伴随耳部、咽喉或胸闷活动后下颌痛等非牙源性牵涉线索？请回答“是/不是/不确定”。",
        ("耳", "咽", "喉", "胸闷", "心脏", "活动后", "下颌", "放射"),
    ),
    FeatureDefinition(
        "soft_tissue_ulcer_burn",
        "口腔软组织破溃/烫伤/溃疡",
        "疼点是否更像牙龈、颊黏膜或口腔表面的破溃、烫伤、溃疡，而不是牙里面痛？请回答“是/不是/不确定”。",
        ("溃疡", "破", "破溃", "烫伤", "灼痛", "黏膜", "咬伤", "疱疹", "义齿", "磨伤", "擦伤"),
    ),
    FeatureDefinition(
        "unclear_multiple_teeth",
        "多颗/一侧/定位不清",
        "现在是否很难定位到某一颗牙，而是一侧、多颗、上下牙或整片区域都不舒服？请回答“是/不是/不确定”。",
        ("定位不清", "不知道哪颗", "一侧", "多颗", "上下牙", "整片", "整排", "说不清", "无法明确"),
    ),
)

FEATURE_BY_ID = {f.feature_id: f for f in FEATURES}

YES_WORDS = ("是", "对", "有", "会", "疼", "痛", "明显", "符合", "差不多", "是的", "嗯")
NO_WORDS = ("不是", "没有", "没", "不会", "不疼", "不痛", "无", "否", "不明显", "没什么")
UNCERTAIN_WORDS = ("不确定", "不知道", "不清楚", "说不准", "可能", "好像", "大概", "也许")


def all_features() -> tuple[FeatureDefinition, ...]:
    return FEATURES


def get_feature(feature_id: str) -> FeatureDefinition | None:
    return FEATURE_BY_ID.get(feature_id)


def _norm(text: str) -> str:
    return (text or "").lower().strip()


def _contains_any(text: str, patterns: Iterable[str]) -> bool:
    text = _norm(text)
    return any(p and p.lower() in text for p in patterns)


def extract_features(text: str) -> list[str]:
    """Return feature ids directly mentioned by text.

    This is intentionally rule-based for the demo: transparent, deterministic,
    and independent of external medical corpora.
    """

    found: list[str] = []
    for feature in FEATURES:
        if _contains_any(text, feature.positive_patterns):
            found.append(feature.feature_id)
    return found


def feature_labels(feature_ids: Iterable[str]) -> list[str]:
    labels: list[str] = []
    for fid in feature_ids:
        feature = get_feature(fid)
        if feature:
            labels.append(feature.label)
    return labels


def classify_answer(text: str, feature_id: str | None = None) -> str:
    """Classify a user's answer to the previous feature question.

    Returns: yes / no / uncertain / unrelated.
    """

    raw = _norm(text)
    if not raw:
        return "unrelated"

    # Strong explicit answers first.
    if any(w in raw for w in NO_WORDS):
        # "不是...但是有冷痛" should still be treated as no for the previous question;
        # update_memory will separately extract newly mentioned features.
        return "no"
    if raw in YES_WORDS or any(raw.startswith(w) for w in ("是", "对", "有", "会")):
        return "yes"
    if any(w in raw for w in UNCERTAIN_WORDS):
        return "uncertain"

    if feature_id:
        feature = get_feature(feature_id)
        if feature and _contains_any(raw, feature.positive_patterns):
            return "yes"

    # Short symptom-only replies like "疼" / "会" / "明显" are usually answers.
    if len(raw) <= 8 and any(w in raw for w in YES_WORDS):
        return "yes"

    return "unrelated"


def extract_search_terms(text: str) -> list[str]:
    """Small keyword extractor used by normalize fallback."""

    features = extract_features(text)
    terms: list[str] = []
    for fid in features:
        feature = get_feature(fid)
        if feature:
            terms.append(feature.label)
            terms.extend(feature.positive_patterns[:3])
    # Keep stable order and avoid noise.
    seen: set[str] = set()
    unique: list[str] = []
    for term in terms:
        if term and term not in seen:
            unique.append(term)
            seen.add(term)
    return unique[:8]
