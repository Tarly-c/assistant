"""离线预计算全部向量 + 特征空间。带断点续跑。"""
from __future__ import annotations
import argparse, json, math, time
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from medical_assistant.config import get_settings
from medical_assistant.llm import call_structured, call_text
from medical_assistant.prompts import EXTRACT_CONCEPTS, NAME_CLUSTER
from medical_assistant.schemas import ExtractedConcepts
from medical_assistant.cases.store import load_cases, full_text, extra_texts
from medical_assistant.text.split import split_windows, clean, norm
from medical_assistant.text.embed import embed_batch, embed_one, cosine, mean_vec

# ── 断点文件路径 ──
CKPT_DIR = Path("resources/.checkpoints")


def _ckpt(name: str) -> Path:
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    return CKPT_DIR / f"{name}.json"


def _save_ckpt(name: str, data: Any) -> None:
    path = _ckpt(name)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    size_mb = path.stat().st_size / 1024 / 1024
    print(f"  [checkpoint] saved {name} ({size_mb:.1f} MB)")


def _load_ckpt(name: str) -> Any | None:
    path = _ckpt(name)
    if not path.exists():
        return None
    print(f"  [checkpoint] loading {name}...")
    return json.loads(path.read_text(encoding="utf-8"))


def _cluster(vecs: list[list[float]], threshold: float, max_n: int) -> list[dict]:
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
    p.add_argument("--skip-concepts", action="store_true")
    p.add_argument("--clean", action="store_true", help="删除所有 checkpoint 重新开始")
    args = p.parse_args()

    cfg = get_settings()
    cases = load_cases()
    print(f"Loaded {len(cases)} cases from {cfg.case_path}")

    if args.check:
        for c in cases[:5]:
            print(f"  [{c.case_id}] {c.title}")
        return

    if args.clean:
        import shutil
        if CKPT_DIR.exists():
            shutil.rmtree(CKPT_DIR)
            print("Cleaned all checkpoints.")

    t0 = time.time()

    # ══════════════════════════════════════
    # Step 1: 句子级 embedding
    # ══════════════════════════════════════
    case_sent_vecs = _load_ckpt("step1_sent_vecs")
    if case_sent_vecs is None:
        print(f"\n[Step 1] Sentence-level embedding ({len(cases)} cases)...")
        case_texts = [full_text(c) for c in cases]
        case_sent_vecs = embed_batch(case_texts)
        _save_ckpt("step1_sent_vecs", case_sent_vecs)
    else:
        print(f"\n[Step 1] Loaded from checkpoint ({len(case_sent_vecs)} vecs)")
    dim = len(case_sent_vecs[0])
    print(f"  dim={dim}")

    # ══════════════════════════════════════
    # Step 2: LLM 概念抽取
    # ══════════════════════════════════════
    case_concepts: dict[str, list[dict]] = _load_ckpt("step2_concepts") or {}

    if not args.skip_concepts:
        already = len(case_concepts)
        todo = [c for c in cases if c.case_id not in case_concepts]
        if todo:
            print(f"\n[Step 2] LLM concept extraction ({len(todo)} remaining, {already} done)...")
            for i, case in enumerate(todo):
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

                # 每 50 条存一次 checkpoint
                if (i + 1) % 50 == 0:
                    _save_ckpt("step2_concepts", case_concepts)
                    print(f"    {already + i + 1}/{len(cases)} cases done")

            _save_ckpt("step2_concepts", case_concepts)
        else:
            print(f"\n[Step 2] All {len(cases)} cases already extracted from checkpoint")
    else:
        print(f"\n[Step 2] Skipped (--skip-concepts)")
        for case in cases:
            if case.case_id not in case_concepts:
                case_concepts[case.case_id] = []

    total_concepts = sum(len(v) for v in case_concepts.values())
    print(f"  Total concepts: {total_concepts}")

    # ══════════════════════════════════════
    # Step 3: Role embed → 聚类 → M 概念维度
    # ══════════════════════════════════════
    step3 = _load_ckpt("step3_concept_clusters")

    if step3 is None:
        all_roles: list[str] = []
        role_to_case: list[tuple[str, int]] = []
        for cid, concepts in case_concepts.items():
            for ci, c in enumerate(concepts):
                all_roles.append(c["role"])
                role_to_case.append((cid, ci))

        concept_clusters_data: list[dict] = []
        role_to_dim: dict[int, int] = {}

        if all_roles:
            print(f"\n[Step 3] Embedding {len(all_roles)} roles + clustering...")
            role_vecs = embed_batch(all_roles)
            raw_clusters = _cluster(role_vecs, cfg.concept_cluster_th, cfg.max_concept_dims)
            print(f"  Raw concept clusters: {len(raw_clusters)}")

            for ci, cl in enumerate(raw_clusters):
                member_roles = [all_roles[k] for k in cl["m"][:8]]
                roles_text = "\n".join(f"- {r}" for r in member_roles)
                name = call_text([
                    {"role": "user", "content": NAME_CLUSTER.format(roles=roles_text)},
                ])
                name = clean(name)[:20] or f"概念_{ci}"

                examples = []
                seen = set()
                for k in cl["m"]:
                    cid, ci_idx = role_to_case[k]
                    if cid not in seen:
                        term = case_concepts[cid][ci_idx]["term"]
                        examples.append(f"{term}({all_roles[k]})")
                        seen.add(cid)
                    if len(examples) >= 5:
                        break

                concept_clusters_data.append({
                    "id": ci, "name": name, "centroid": cl["c"],
                    "member_count": len(cl["m"]), "examples": examples,
                })

                for k in cl["m"]:
                    role_to_dim[k] = ci
        else:
            print(f"\n[Step 3] No concepts to cluster")
            role_vecs = []

        step3 = {
            "concept_clusters": concept_clusters_data,
            "role_to_dim": {str(k): v for k, v in role_to_dim.items()},
            "all_roles": all_roles,
            "role_to_case": role_to_case,
            "role_vecs": role_vecs,
        }
        _save_ckpt("step3_concept_clusters", step3)
    else:
        print(f"\n[Step 3] Loaded from checkpoint")
        concept_clusters_data = step3["concept_clusters"]
        role_to_dim = {int(k): v for k, v in step3["role_to_dim"].items()}
        all_roles = step3["all_roles"]
        role_to_case = [tuple(x) for x in step3["role_to_case"]]
        role_vecs = step3["role_vecs"]

    M = len(concept_clusters_data)
    print(f"  M = {M} concept dimensions")

    # ══════════════════════════════════════
    # Step 4: Term embed → 关键词向量
    # ══════════════════════════════════════
    step4 = _load_ckpt("step4_term_vecs")

    if step4 is None:
        print(f"\n[Step 4] Embedding keyword terms...")
        all_terms: list[str] = []
        term_idx_map: list[tuple[str, int]] = []
        for cid, concepts in case_concepts.items():
            for ci, c in enumerate(concepts):
                all_terms.append(c["term"])
                term_idx_map.append((cid, ci))

        term_vecs = embed_batch(all_terms) if all_terms else []
        step4 = {
            "all_terms": all_terms,
            "term_idx_map": term_idx_map,
            "term_vecs": term_vecs,
        }
        _save_ckpt("step4_term_vecs", step4)
    else:
        print(f"\n[Step 4] Loaded from checkpoint")
        all_terms = step4["all_terms"]
        term_idx_map = [tuple(x) for x in step4["term_idx_map"]]
        term_vecs = step4["term_vecs"]

    print(f"  {len(term_vecs)} term vectors")

    # 整理 case_id → {positive: [vec], negative: [vec]}
    case_kw_vecs: dict[str, dict[str, list[list[float]]]] = {}
    for case in cases:
        case_kw_vecs[case.case_id] = {"positive": [], "negative": []}
    for idx, tv in enumerate(term_vecs):
        cid, ci = term_idx_map[idx]
        concepts = case_concepts.get(cid, [])
        if ci < len(concepts):
            neg = concepts[ci].get("negative", False)
            key = "negative" if neg else "positive"
            case_kw_vecs[cid][key].append(tv)

    # ══════════════════════════════════════
    # Step 5: 句子窗口 embed → K 个语义簇
    # ══════════════════════════════════════
    step5 = _load_ckpt("step5_semantic_clusters")

    if step5 is None:
        print(f"\n[Step 5] Sentence window embedding + clustering...")

        all_units: list[tuple[str, str]] = []
        for case in cases:
            for t in split_windows(case.title, case.description, extra=extra_texts(case)):
                # ★ 预过滤坏文本
                cleaned = clean(t)
                if cleaned and len(norm(cleaned)) >= 2:
                    all_units.append((case.case_id, cleaned))

        print(f"  Text units: {len(all_units)} (after filtering)")
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

        step5 = {
            "semantic_clusters": semantic_clusters_data,
            "all_units": all_units,
            "unit_vecs": unit_vecs,
        }
        _save_ckpt("step5_semantic_clusters", step5)
    else:
        print(f"\n[Step 5] Loaded from checkpoint")
        semantic_clusters_data = step5["semantic_clusters"]
        all_units = [tuple(x) for x in step5["all_units"]]
        unit_vecs = step5["unit_vecs"]

    K = len(semantic_clusters_data)
    print(f"  K = {K} semantic clusters")

    # ══════════════════════════════════════
    # Step 6: 投影 → (K+M) 维特征矩阵
    # ══════════════════════════════════════
    print(f"\n[Step 6] Projecting to {K}+{M}={K+M} dim feature space...")

    # 按病例索引窗口向量
    case_unit_vecs: dict[str, list[list[float]]] = {}
    for (cid, _), uv in zip(all_units, unit_vecs):
        case_unit_vecs.setdefault(cid, []).append(uv)

    # 按病例索引概念 role 向量
    case_role_vecs: dict[str, list[tuple[int, list[float]]]] = {}
    if role_vecs:
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

        if (ci + 1) % 100 == 0:
            print(f"    projected {ci+1}/{len(cases)}")

    # ══════════════════════════════════════
    # Step 7: 保存最终输出
    # ══════════════════════════════════════
    print(f"\n[Step 7] Saving final output...")

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

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  Cases: {len(cases)}")
    print(f"  Sentence dim: {dim}")
    print(f"  Semantic clusters (K): {K}")
    print(f"  Concept dims (M): {M}")
    print(f"  Total features: {K + M}")
    print(f"  Total concepts: {total_concepts}")
    print(f"  Elapsed: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"\nTop semantic clusters:")
    for sc in semantic_clusters_data[:8]:
        print(f"  [{sc['id']:>3}] n={sc['member_count']:>4} {sc['label']}")
    print(f"\nConcept dimensions:")
    for cc in concept_clusters_data[:15]:
        print(f"  [{cc['id']:>3}] n={cc['member_count']:>4} {cc['name']}: "
              f"{', '.join(cc['examples'][:3])}")
    print(f"{'='*60}")

    # 清理 checkpoint（成功完成后）
    print(f"\nAll done. You can safely delete {CKPT_DIR} or keep for debugging.")


if __name__ == "__main__":
    main()
