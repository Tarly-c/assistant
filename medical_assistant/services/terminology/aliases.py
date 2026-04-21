from __future__ import annotations

NEGATION_PREFIXES = ["没有", "没", "无", "并无", "未见", "不是"]

LOCAL_STOP_TERMS = {
    "thing",
    "problem",
    "issue",
}

CHIEF_COMPLAINT_PATTERNS = [
    {"english": "headache", "patterns": ["头痛", "头疼", "脑袋疼", "头一阵一阵疼"]},
    {"english": "fever", "patterns": ["发烧", "发热", "高烧"]},
    {"english": "cough", "patterns": ["咳嗽", "老咳", "一直咳"]},
    {"english": "sore throat", "patterns": ["喉咙痛", "咽痛", "嗓子疼"]},
    {"english": "abdominal pain", "patterns": ["肚子痛", "腹痛", "胃痛", "肚子疼"]},
    {"english": "diarrhea", "patterns": ["拉肚子", "腹泻"]},
    {"english": "shortness of breath", "patterns": ["气短", "喘不过气", "呼吸困难"]},
    {"english": "chest pain", "patterns": ["胸痛", "胸口疼", "胸闷疼"]},
    {"english": "rash", "patterns": ["起疹子", "皮疹", "身上起红点"]},
]

FINDING_PATTERNS = [
    {"name": "nausea", "value": True, "patterns": ["恶心", "想吐"], "normalized_term": "nausea"},
    {"name": "vomiting", "value": True, "patterns": ["呕吐", "吐了"], "normalized_term": "vomiting"},
    {"name": "fever", "value": True, "patterns": ["发烧", "发热", "高烧"], "normalized_term": "fever"},
    {"name": "neck_stiffness", "value": True, "patterns": ["脖子硬", "颈部发硬", "脖子僵"], "normalized_term": "neck stiffness"},
    {"name": "shortness_of_breath", "value": True, "patterns": ["气短", "呼吸困难", "喘不过气"], "normalized_term": "shortness of breath"},
    {"name": "chest_pain", "value": True, "patterns": ["胸痛", "胸口疼"], "normalized_term": "chest pain"},
    {"name": "neurologic_deficit", "value": True, "patterns": ["肢体无力", "说话不清", "口齿不清", "半边没劲"], "normalized_term": "neurologic deficit"},
]

FACET_PATTERNS = [
    {"name": "pain_quality", "value": "throbbing", "patterns": ["一跳一跳", "跳着疼", "搏动"], "normalized_term": "throbbing pain"},
    {"name": "pain_quality", "value": "pressing", "patterns": ["紧箍", "压着疼", "闷胀"], "normalized_term": "pressing pain"},
    {"name": "laterality", "value": "unilateral", "patterns": ["一侧", "单侧"], "normalized_term": "unilateral"},
    {"name": "laterality", "value": "bilateral", "patterns": ["两侧", "整个头", "全头"], "normalized_term": "bilateral"},
    {"name": "sudden_onset", "value": True, "patterns": ["突然", "一下子", "猛地"], "normalized_term": "sudden onset"},
    {"name": "duration", "value": "2 days", "patterns": ["两天", "2天"], "normalized_term": "2 days"},
]

RED_FLAG_PATTERNS = [
    {"name": "sudden_onset", "patterns": ["突然", "一下子", "爆发"]},
    {"name": "neck_stiffness", "patterns": ["脖子硬", "颈部发硬", "脖子僵"]},
    {"name": "neurologic_deficit", "patterns": ["肢体无力", "说话不清", "抽搐", "意识模糊"]},
    {"name": "shortness_of_breath", "patterns": ["喘不过气", "呼吸困难"]},
    {"name": "chest_pain", "patterns": ["胸痛", "胸口疼"]},
]
