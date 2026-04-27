"""★ 离线预计算全部向量 + 特征空间。

流程：
  1. 加载病例 → 批量 embed（句子级）
  2. LLM 概念抽取 → 每个病例 5-12 个 (term, role)
  3. Embed 所有 role → 聚类 → M 个概念维度 → LLM 命名
  4. Embed 所有 term → 存为关键词向量
  5. 句子窗口 embed → 聚类 → K 个语义簇
  6. 每个病例投影到 (K+M) 维 → 特征矩阵
  7. 保存
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from medical_assistant.config import get_settings
from medical_assistant.llm import call_structured, call_text
from medical_assistant.prompts import EXTRACT_CONCEPTS, NAME_CLUSTER
from medical_assistant.schemas import ExtractedConcepts
from medical_assistant.cases.store import load_cases, full_text, extra_texts
from medical_assistant.text.split import split_windows, clean
from medical_assistant.text.embed import embed_batch, embed_one, cosine, mean_vec


def _cluster(vecs: list[list[float]], threshold: float, max_n: int,
             ) -> list[dict[str, Any]]:
    """通用在线聚类。"""
    clusters: list[dict] = []
    for i, v in enumerate(vecs):
        bi, bs = -1, 0.0
        for j, cl in enumerate(clusters):
            s = cosine(v, cl["c"])
            if s > bs:
                bs, bi = s, j
        if bi >= 0 and bs >= threshold:
            cl = clusters[bi]
            cl["m"].append(i)
            cl["c"] = mean_vec([vecs[k] for k in cl["m"]])
        else:
            clusters.append({"c": list(v), "m": [i]})
        if len(clusters) > max_n * 2:
            clusters.sort(key=lambda c: len(c["m"]), reverse=True)
            clusters = clusters[:max_n]
    clusters.sort(key=lambda c: len(c["m"]), reverse=True)
    return clusters[:max_n]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true")
    p.add_argument("--skip-concepts", action="store_true",
                   help="跳过 LLM 概念抽取（调试用）")
    args = p.parse_args()

    cfg = get_settings()
    cases = load_cases()
    print(f"Loaded {len(cases)} cases from {cfg.case_path}")

    if args.check:
        for c in cases[:5]:
            print(f"  [{c.case_id}] {c.title}")
        return

    # ══════════════════════════════════════
    # Step 1: 句子级 embedding
    # ══════════════════════════════════════
    print("\n[Step 1] Sentence-level embedding...")
    case_texts = [full_text(c) for c in cases]
    case_sent_vecs = embed_batch(case_texts)
    dim = len(case_sent_vecs[0])
    print(f"  Done. dim={dim}, cases={len(cases)}")

    # ══════════════════════════════════════
    # Step 2: LLM 概念抽取
    # ══════════════════════════════════════
    # case_id → [{term, role, importance, negative}]
    case_concepts: dict[str, list[dict]] = {}

    if not args.skip_concepts:
        print(f"\n[Step 2] LLM concept extraction ({len(cases)} cases)...")
        for i, case in enumerate(cases):
            prompt = EXTRACT_CONCEPTS.format(title=case.title, description=case.description)
            result = call_structured(ExtractedConcepts, [
                {"role": "user", "content": prompt},
            ])
            concepts = []
            for c in result.concepts:
                if c.term and c.role:
                    concepts.append({
                        "term": c.term, "role": c.role,
                        "importance": c.importance or "medium",
                        "negative": bool(c.negative),
                    })
            case_concepts[case.case_id] = concepts
            if (i + 1) % 50 == 0:
                print(f"    {i+1}/{len(cases)} cases processed")
        print(f"  Done. Total concepts: {sum(len(v) for v in case_concepts.values())}")
    else:
        print("\n[Step 2] Skipped concept extraction (--skip-concepts)")
        for case in cases:
            case_concepts[case.case_id] = []

    # ══════════════════════════════════════
    # Step 3: Embed roles → 聚类 → M 概念维度
    # ══════════════════════════════════════
    all_roles: list[str] = []           # 所有 role 描述
    role_to_case: list[tuple[str, int]] = []  # (case_id, concept_idx_in_case)

    for cid, concepts in case_concepts.items():
        for ci, c in enumerate(concepts):
            all_roles.append(c["role"])
            role_to_case.append((cid, ci))

    concept_clusters_data: list[dict] = []
    if all_roles:
        print(f"\n[Step 3] Embedding {len(all_roles)} roles + clustering...")
        role_vecs = embed_batch(all_roles)

        raw_clusters = _cluster(role_vecs, cfg.concept_cluster_th, cfg.max_concept_dims)
        print(f"  Concept clusters: {len(raw_clusters)}")

        # 给每个簇命名
        for ci, cl in enumerate(raw_clusters):
            member_roles = [all_roles[k] for k in cl["m"][:8]]
            roles_text = "\n".join(f"- {r}" for r in member_roles)
            name = call_text([
                {"role": "user", "content": NAME_CLUSTER.format(roles=roles_text)},
            ])
            name = clean(name)[:20] or f"概念_{ci}"

            # 代表示例
            examples = []
            seen = set()
            for k in cl["m"]:
                cid, _ = role_to_case[k]
                if cid not in seen:
                    term = case_concepts[cid][role_to_case[k][1]]["term"]
                    examples.append(f"{term}({all_roles[k]})")
                    seen.add(cid)
                if len(examples) >= 5:
                    break

            concept_clusters_data.append({
                "id": ci, "name": name, "centroid": cl["c"],
                "member_count": len(cl["m"]),
                "examples": examples,
            })

        # 为每个概念分配维度
        role_to_dim = {}
        for ci, cl in enumerate(raw_clusters):
            for k in cl["m"]:
                role_to_dim[k] = ci
    else:
        print("\n[Step 3] No concepts to cluster")

    M = len(concept_clusters_data)
    print(f"  M = {M} concept dimensions")

    # ══════════════════════════════════════
    # Step 4: Embed terms → 关键词向量
    # ══════════════════════════════════════
    print(f"\n[Step 4] Embedding keyword terms...")
    all_terms: list[str] = []
    term_idx_map: list[tuple[str, int]] = []   # (case_id, concept_idx)

    for cid, concepts in case_concepts.items():
        for ci, c in enumerate(concepts):
            all_terms.append(c["term"])
            term_idx_map.append((cid, ci))

    term_vecs = embed_batch(all_terms) if all_terms else []
    print(f"  Done. {len(term_vecs)} term vectors")

    # 整理成 case_id → {positive: [vec], negative: [vec]}
    case_kw_vecs: dict[str, dict[str, list[list[float]]]] = {}
    for case in cases:
        case_kw_vecs[case.case_id] = {"positive": [], "negative": []}

    for idx, (tv) in enumerate(term_vecs):
        cid, ci = term_idx_map[idx]
        concepts = case_concepts.get(cid, [])
        if ci < len(concepts):
            neg = concepts[ci].get("negative", False)
            key = "negative" if neg else "positive"
            case_kw_vecs[cid][key].append(tv)

    # ══════════════════════════════════════
    # Step 5: 句子窗口 embed → K 个语义簇
    # ══════════════════════════════════════
    print(f"\n[Step 5] Sentence window embedding + clustering...")
    all_units: list[tuple[str, str]] = []
    for case in cases:
        for t in split_windows(case.title, case.description, extra=extra_texts(case)):
            all_units.append((case.case_id, t))
    print(f"  Text units: {len(all_units)}")

    unit_texts = [t for _, t in all_units]
    unit_vecs = embed_batch(unit_texts)

    raw_sem = _cluster(unit_vecs, cfg.semantic_cluster_th, cfg.max_semantic_clusters)
    K = len(raw_sem)
    print(f"  Semantic clusters: K = {K}")

    semantic_clusters_data = []
    for ci, cl in enumerate(raw_sem):
        members = cl["m"]
        centroid = cl["c"]
        best_idx = max(members, key=lambda k: cosine(unit_vecs[k], centroid))
        label = clean(unit_texts[best_idx])[:60]
        seen = set()
        texts = []
        for k in sorted(members, key=lambda k: cosine(unit_vecs[k], centroid), reverse=True):
            cid = all_units[k][0]
            if cid not in seen:
                texts.append(unit_texts[k])
                seen.add(cid)
            if len(texts) >= 5:
                break
        semantic_clusters_data.append({
            "id": ci, "label": label, "centroid": centroid,
            "member_count": len(members), "texts": texts,
        })

    # ══════════════════════════════════════
    # Step 6: 投影 → (K+M) 维特征矩阵
    # ══════════════════════════════════════
    print(f"\n[Step 6] Projecting to {K}+{M}={K+M} dim feature space...")

    # 按病例索引窗口向量
    case_unit_vecs: dict[str, list[list[float]]] = {}
    for (cid, _), uv in zip(all_units, unit_vecs):
        case_unit_vecs.setdefault(cid, []).append(uv)

    # 按病例索引概念 role 向量（带维度）
    case_role_vecs: dict[str, list[tuple[int, list[float]]]] = {}  # cid → [(dim, role_vec)]
    if all_roles:
        for ridx, rv in enumerate(role_vecs):
            cid, _ = role_to_case[ridx]
            d = role_to_dim.get(ridx, -1)
            if d >= 0:
                case_role_vecs.setdefault(cid, []).append((d, rv))

    case_data = []
    for ci, case in enumerate(cases):
        cid = case.case_id

        # K 维语义簇特征
        sem_fv = [0.0] * K
        for k in range(K):
            centroid = semantic_clusters_data[k]["centroid"]
            best_sim = 0.0
            for uv in case_unit_vecs.get(cid, []):
                s = cosine(uv, centroid)
                if s > best_sim:
                    best_sim = s
            sem_fv[k] = round(best_sim, 4)

        # M 维概念特征
        con_fv = [0.0] * M
        for d, rv in case_role_vecs.get(cid, []):
            if d < M:
                centroid = concept_clusters_data[d]["centroid"]
                s = cosine(rv, centroid)
                if s > con_fv[d]:
                    con_fv[d] = round(s, 4)

        feature_vec = sem_fv + con_fv

        case_data.append({
            "case_id": cid,
            "title": case.title,
            "sentence_vec": case_sent_vecs[ci],
            "feature_vec": feature_vec,
            "keyword_vecs": case_kw_vecs.get(cid, {"positive": [], "negative": []}),
        })

        if (ci + 1) % 200 == 0:
            print(f"    projected {ci+1}/{len(cases)}")

    # ══════════════════════════════════════
    # Step 7: 保存
    # ══════════════════════════════════════
    print(f"\n[Step 7] Saving...")

    vectors_out = {
        "meta": {
            "case_count": len(cases),
            "sentence_dim": dim,
            "semantic_clusters": K,
            "concept_dims": M,
            "total_features": K + M,
        },
        "cases": case_data,
    }
    cfg.vectors_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.vectors_path.write_text(
        json.dumps(vectors_out, ensure_ascii=False), encoding="utf-8",
    )
    print(f"  Vectors → {cfg.vectors_path}")

    clusters_out = {
        "semantic_clusters": semantic_clusters_data,
        "concept_clusters": concept_clusters_data,
    }
    cfg.clusters_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.clusters_path.write_text(
        json.dumps(clusters_out, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"  Clusters → {cfg.clusters_path}")

    # 简报
    print(f"\n{'='*60}")
    print(f"  Cases: {len(cases)}")
    print(f"  Sentence dim: {dim}")
    print(f"  Semantic clusters (K): {K}")
    print(f"  Concept dims (M): {M}")
    print(f"  Total features: {K + M}")
    print(f"  Total concepts extracted: {sum(len(v) for v in case_concepts.values())}")
    print(f"\nTop semantic clusters:")
    for sc in semantic_clusters_data[:8]:
        print(f"  [{sc['id']:>3}] n={sc['member_count']:>4} {sc['label']}")
    print(f"\nConcept dimensions:")
    for cc in concept_clusters_data[:15]:
        print(f"  [{cc['id']:>3}] n={cc['member_count']:>4} {cc['name']}: "
              f"{', '.join(cc['examples'][:3])}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
