"""病例数据 + 预计算向量加载。"""
from __future__ import annotations
import json
from functools import lru_cache
from typing import Sequence
from medical_assistant.config import get_settings
from medical_assistant.schemas import CaseRecord, ScoredCase
from medical_assistant.text.split import clean


def _list_field(item: dict, *names: str) -> list[str]:
    out = []
    for name in names:
        val = item.get(name)
        if isinstance(val, str) and val.strip():
            out.append(val.strip())
        elif isinstance(val, list):
            out.extend(str(x).strip() for x in val if str(x).strip())
    seen = set()
    return [v for v in out if v not in seen and not seen.add(v)]  # type: ignore


def extra_texts(case: CaseRecord | ScoredCase) -> list[str]:
    texts = []
    for attr in ("title_en", "description_en", "aliases", "keywords"):
        val = getattr(case, attr, None)
        if isinstance(val, str) and val.strip():
            texts.append(val.strip())
        elif isinstance(val, (list, tuple)):
            texts.extend(str(x).strip() for x in val if str(x).strip())
    return texts


def full_text(case: CaseRecord | ScoredCase) -> str:
    """用于 embedding 的完整文本。"""
    parts = [case.title, case.description, case.treat]
    parts.extend(case.feature_tags or [])
    parts.extend(extra_texts(case))
    return "\n".join(clean(str(p)) for p in parts if clean(str(p)))


@lru_cache(maxsize=1)
def load_cases() -> list[CaseRecord]:
    path = get_settings().case_path
    if not path.exists():
        raise FileNotFoundError(f"Case data not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    cases = []
    for i, item in enumerate(raw, 1):
        if not isinstance(item, dict):
            continue
        cases.append(CaseRecord(
            case_id=str(item.get("case_id") or f"case_{i:03d}"),
            title=str(item.get("title", "")).strip(),
            description=str(item.get("description", "")).strip(),
            treat=str(item.get("treat") or item.get("treatment") or "").strip(),
            title_en=str(item.get("title_en") or "").strip(),
            description_en=str(item.get("description_en") or "").strip(),
            aliases=_list_field(item, "aliases", "synonyms"),
            keywords=_list_field(item, "keywords"),
            feature_tags=_list_field(item, "feature_tags"),
        ))
    return cases


@lru_cache(maxsize=1)
def _case_map() -> dict[str, CaseRecord]:
    return {c.case_id: c for c in load_cases()}


def get_case(case_id: str) -> CaseRecord | None:
    return _case_map().get(case_id)


def get_cases(ids: Sequence[str] | None = None) -> list[CaseRecord]:
    if ids is None:
        return load_cases()
    m = _case_map()
    return [m[cid] for cid in ids if cid in m]


def all_ids() -> list[str]:
    return [c.case_id for c in load_cases()]


# ── 预计算向量 ──

@lru_cache(maxsize=1)
def _load_vectors_raw() -> dict:
    path = get_settings().vectors_path
    if not path.exists():
        print(f"[WARN] Vectors not found: {path}")
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_sentence_vecs() -> dict[str, list[float]]:
    """case_id → 句子级 embedding 向量。"""
    raw = _load_vectors_raw()
    return {c["case_id"]: c["sentence_vec"]
            for c in raw.get("cases", []) if "sentence_vec" in c}


@lru_cache(maxsize=1)
def load_keyword_vecs() -> dict[str, dict[str, list[list[float]]]]:
    """case_id → {"positive": [[vec], ...], "negative": [[vec], ...]}。"""
    raw = _load_vectors_raw()
    return {c["case_id"]: c.get("keyword_vecs", {"positive": [], "negative": []})
            for c in raw.get("cases", [])}


@lru_cache(maxsize=1)
def load_feature_vecs() -> dict[str, list[float]]:
    """case_id → (K+M) 维特征向量。"""
    raw = _load_vectors_raw()
    return {c["case_id"]: c["feature_vec"]
            for c in raw.get("cases", []) if "feature_vec" in c}


@lru_cache(maxsize=1)
def load_meta() -> dict:
    """加载特征空间元信息。"""
    raw = _load_vectors_raw()
    return raw.get("meta", {})


@lru_cache(maxsize=1)
def load_clusters() -> dict:
    path = get_settings().clusters_path
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def cluster_label(dim: int) -> str:
    """获取特征维度的可读标签。"""
    data = load_clusters()
    meta = load_meta()
    K = meta.get("semantic_clusters", 0)
    if dim < K:
        sc = data.get("semantic_clusters", [])
        return sc[dim]["label"] if dim < len(sc) else f"semantic_{dim}"
    else:
        cc = data.get("concept_clusters", [])
        idx = dim - K
        return cc[idx]["name"] if idx < len(cc) else f"concept_{idx}"


def cluster_evidence(dim: int) -> list[str]:
    """获取特征维度的代表文本。"""
    data = load_clusters()
    meta = load_meta()
    K = meta.get("semantic_clusters", 0)
    if dim < K:
        sc = data.get("semantic_clusters", [])
        return sc[dim].get("texts", [])[:5] if dim < len(sc) else []
    else:
        cc = data.get("concept_clusters", [])
        idx = dim - K
        return cc[idx].get("examples", [])[:5] if idx < len(cc) else []
