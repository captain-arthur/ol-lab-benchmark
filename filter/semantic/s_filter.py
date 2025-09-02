#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
s_filter.py
- filter/semantic 하위 실험들의 공통 로직
- SBERT + Anchors + CrossEncoder 3단계 실험 프레임워크
"""

import os
import json
import time
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

import numpy as np
import requests
from sentence_transformers import SentenceTransformer, CrossEncoder

# =========================
# 유틸리티 함수
# =========================

def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def save_json(obj, path: str):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def timer_ms():
    t0 = time.time()
    return lambda: (time.time() - t0) * 1000.0

def ndcg_at_k(order, relevant, k):
    dcg = 0.0
    for i, d in enumerate(order[:k], start=1):
        if d in relevant:
            dcg += 1.0 / np.log2(i + 1)
    ideal = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 1) for i in range(1, ideal + 1))
    return (dcg / idcg) if idcg > 0 else 0.0

def pick_better_by_ndcg(order_a, order_b, relevant, k_focus=10):
    a = ndcg_at_k(order_a, relevant, k_focus)
    b = ndcg_at_k(order_b, relevant, k_focus)
    return (order_a, a) if a >= b else (order_b, b)

def is_confident(order_baseline, scores_this, order_this,
                 K_overlap=20, overlap_thr=0.6, margin_thr=0.03):
    top_b = set(order_baseline[:K_overlap])
    top_t = set(order_this[:K_overlap])
    overlap = len(top_b & top_t) / max(len(top_b), 1)
    if len(order_this) >= 2:
        s1 = scores_this[order_this[0]]
        s2 = scores_this[order_this[1]]
        margin = s1 - s2
    else:
        margin = 0.0
    return (overlap >= overlap_thr) or (margin >= margin_thr)

# =========================
# 캐시 관리
# =========================

def get_cache_file_path(cache_dir: str) -> str:
    return os.path.join(cache_dir, "anchors_cache.json")

def load_cache(query: str, cache_dir: str) -> Optional[List[str]]:
    cache_path = get_cache_file_path(cache_dir)
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache_list = json.load(f)
                for item in cache_list:
                    if item.get('query') == query:
                        expanded = item.get('expanded', {})
                        return expanded.get('anchors', [])
        except Exception:
            return None
    return None

def save_cache(query: str, anchors: List[str], cache_dir: str):
    cache_path = get_cache_file_path(cache_dir)
    ensure_dir(os.path.dirname(cache_path))
    
    cache_list = []
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache_list = json.load(f)
        except Exception:
            cache_list = []
    
    cache_list = [item for item in cache_list if item.get('query') != query]
    cache_list.append({
        'query': query,
        'expanded': {'anchors': anchors}
    })
    
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(cache_list, f, ensure_ascii=False, indent=2)

# =========================
# 데이터 처리
# =========================

def build_label_index(labels: List[int]) -> Dict[int, List[int]]:
    idx = defaultdict(list)
    for i, lab in enumerate(labels):
        idx[lab].append(i)
    return idx

def create_filtering_candidates(corpus_texts, corpus_labels, label_idx, query_label, n_negative=99):
    rng = np.random.default_rng(42)
    positives = label_idx[query_label]
    positive_doc = int(rng.choice(positives))
    
    negative_candidates = [i for i, lab in enumerate(corpus_labels) if lab != query_label]
    n_available = min(n_negative, len(negative_candidates))
    negative_docs = rng.choice(negative_candidates, size=n_available, replace=False)
    
    candidate_pool = [positive_doc] + negative_docs.tolist()
    return candidate_pool, positive_doc, negative_docs.tolist()

# =========================
# Ollama 앵커 생성
# =========================

def generate_anchors_ollama(q: str, num_anchors: int, ollama_host: str, 
                           ollama_port: int, ollama_model: str, cache_dir: str) -> List[str]:
    if num_anchors <= 0:
        return []

    cached_anchors = load_cache(q, cache_dir)
    if cached_anchors is not None:
        print(f"[Cache] Found cached anchors for query: {q[:50]}...")
        return cached_anchors[:num_anchors]

    url = f"http://{ollama_host}:{ollama_port}/api/generate"
    prompt = (
        f"Generate {num_anchors} short, distinct rephrasings of the query below. "
        f"Each anchor must express the same intent as the query in slightly different words. "
        f"No numbering, no extra text. One anchor per line.\n\n"
        f"Query: {q}\n"
    )
    payload = {"model": ollama_model, "prompt": prompt, "stream": False}

    try:
        resp = requests.post(url, json=payload, timeout=30)
        resp.raise_for_status()
        text = resp.json().get("response", "")
        anchors = [line.strip() for line in text.splitlines() if line.strip()]
        anchors = anchors[:num_anchors]
        seen, uniq = set(), []
        for a in anchors:
            if a not in seen:
                seen.add(a)
                uniq.append(a)
        
        if uniq:
            save_cache(q, uniq, cache_dir)
            print(f"[Cache] Saved anchors for query: {q[:50]}...")
        
        return uniq
    except Exception as e:
        print(f"[Error] Ollama call failed: {e}")
        return []

# =========================
# 임베딩/랭킹
# =========================

def embed_texts(model: SentenceTransformer, texts: List[str], batch_size: int, show_progress=False) -> np.ndarray:
    return model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=show_progress
    )

def rank_biencoder(q_emb: np.ndarray, doc_embs: np.ndarray) -> List[int]:
    sims = doc_embs @ q_emb
    return np.argsort(sims)[::-1].tolist()

def rank_with_anchors(q_emb: np.ndarray, anchor_embs: Optional[np.ndarray],
                      doc_embs: np.ndarray, anchor_mode: str = "weighted", alpha: float = 0.7) -> List[int]:
    base = doc_embs @ q_emb
    if anchor_embs is None or len(anchor_embs) == 0:
        scores = base
    else:
        max_anchor = None
        for a in anchor_embs:
            s = doc_embs @ a
            max_anchor = s if max_anchor is None else np.maximum(max_anchor, s)
        if anchor_mode == "max":
            scores = np.maximum(base, max_anchor)
        else:
            scores = alpha * base + (1.0 - alpha) * max_anchor
    return np.argsort(scores)[::-1].tolist()

# =========================
# CrossEncoder 관련
# =========================

def ce_score_pairs_cached(ce, q_idx: int, q_text: str, doc_indices: List[int], 
                         corpus_texts: List[str], ce_cache: dict) -> np.ndarray:
    need_pairs, need_ids = [], []
    for d in doc_indices:
        key = (q_idx, d)
        if key not in ce_cache:
            need_pairs.append([q_text, corpus_texts[d]])
            need_ids.append(d)
    if need_pairs:
        preds = ce.predict(need_pairs)
        for d, s in zip(need_ids, preds):
            ce_cache[(q_idx, d)] = float(s)
    return np.array([ce_cache[(q_idx, d)] for d in doc_indices], dtype=np.float32)

def calibrate_ce_threshold(ce: CrossEncoder, query_texts: List[str], corpus_texts: List[str],
                           sample_pairs: int = 400, base_thr: float = 0.5) -> Tuple[float, dict]:
    rng = np.random.default_rng(42)
    pairs = []
    n = min(len(query_texts), sample_pairs)
    for _ in range(n):
        q_idx = int(rng.integers(0, len(query_texts)))
        d_idx = int(rng.integers(0, len(corpus_texts)))
        if d_idx == q_idx:
            d_idx = (d_idx + 1) % len(corpus_texts)
        pairs.append([query_texts[q_idx], corpus_texts[d_idx]])

    if not pairs:
        return base_thr, {"used_base": True}

    scores = np.array(ce.predict(pairs), dtype=np.float32)
    p95 = float(np.percentile(scores, 95))
    thr = max(base_thr, p95)
    meta = {"mean": float(scores.mean()), "std": float(scores.std()), "p95": p95, 
            "base_thr": base_thr, "final_thr": thr, "n_pairs": len(scores)}
    print(f"[CE-Calib] p95={p95:.3f} -> CE thr={thr:.3f}")
    return thr, meta

def rerank_crossencoder(ce: CrossEncoder, q_idx: int, q: str, corpus_texts: List[str],
                       base_order: List[int], top_m: int, ce_cache: dict) -> List[int]:
    if top_m <= 0:
        return base_order
    head = base_order[:min(top_m, len(base_order))]
    head_scores = ce_score_pairs_cached(ce, q_idx, q, head, corpus_texts, ce_cache)
    head_reranked = [head[i] for i in np.argsort(head_scores)[::-1]]
    return head_reranked + list(base_order[len(head):])

# =========================
# 지표 계산
# =========================

def compute_metrics(ranked: List[int], relevant: set, k_list: List[int]) -> Dict[str, float]:
    out = {}
    for k in k_list:
        topk = ranked[:k]
        hit = sum(1 for d in topk if d in relevant)
        denom_rel = max(len(relevant), 1)
        out[f"R@{k}"] = hit / denom_rel
        out[f"P@{k}"] = hit / max(k, 1)
        out[f"NDCG@{k}"] = ndcg_at_k(ranked, relevant, k)
    
    rr = 0.0
    for rank, d in enumerate(ranked, start=1):
        if d in relevant:
            rr = 1.0 / rank
            break
    out["MRR"] = rr
    return out

def aggregate_metrics(all_metrics: Dict[str, List[float]]) -> Dict[str, float]:
    return {k: float(np.mean(v)) for k, v in all_metrics.items()}

def compute_filtering_metrics(sbert_scores, anchor_scores, ce_scores, 
                            candidate_pool, positive_doc, negative_docs,
                            thresholds, on_intent_ids=None):
    drop_decisions = []
    keep_decisions = []
    
    for i, doc_id in enumerate(candidate_pool):
        sbert_score = sbert_scores[i]
        anchor_score = anchor_scores[i]
        ce_score = ce_scores[i]
        
        should_drop = (sbert_score < thresholds["sbert"] and 
                      anchor_score < thresholds["anchor"] and 
                      ce_score < thresholds["ce"])
        
        if should_drop:
            drop_decisions.append(doc_id)
        else:
            keep_decisions.append(doc_id)
    
    total_negative = len(negative_docs)
    dropped_negative = len(set(drop_decisions) & set(negative_docs))
    drop_precision = dropped_negative / len(drop_decisions) if drop_decisions else 0.0
    drop_recall = dropped_negative / total_negative if total_negative > 0 else 0.0
    positive_kept = positive_doc in keep_decisions
    keep_recall = 1.0 if positive_kept else 0.0
    
    kept_set = set(keep_decisions)
    keep_any = 1.0 if (on_intent_ids is not None and len(kept_set & set(on_intent_ids)) > 0) else 0.0
    
    coverage = (len(drop_decisions) + len(keep_decisions)) / len(candidate_pool)
    
    return {
        "Drop_Precision": drop_precision,
        "Drop_Recall": drop_recall,
        "Keep_Recall": keep_recall,
        "Keep_Recall_Any": keep_any,
        "Coverage": coverage,
        "Dropped_Count": len(drop_decisions),
        "Kept_Count": len(keep_decisions)
    }

# =========================
# 핵심 실험 함수들
# =========================

def run_baseline_experiment(sbert, doc_embs, corpus_texts, corpus_labels, query_texts, query_labels,
                           label_idx, candidate_cache, k_list, max_queries, negative_samples,
                           drop_thresholds, out_dir, show_progress):
    """Baseline 실험 실행 (SBERT + Cosine)"""
    agg = defaultdict(list)
    filter_agg = defaultdict(list)
    t1 = timer_ms()
    
    for i, (q, lab) in enumerate(zip(query_texts, query_labels)):
        if i >= max_queries:
            break
        print(f"[Baseline] Processing query {i+1}/{min(max_queries, len(query_texts))}")
        
        relevant = set(label_idx[lab])
        
        # 필터링용 후보 풀 생성
        key = i
        if key not in candidate_cache:
            candidate_cache[key] = create_filtering_candidates(
                corpus_texts, corpus_labels, label_idx, lab, negative_samples
            )
        candidate_pool, positive_doc, negative_docs = candidate_cache[key]
        
        # SBERT 임베딩 및 점수 계산
        q_emb = embed_texts(sbert, [q], 1, show_progress)[0]
        order = rank_biencoder(q_emb, doc_embs)
        
        # 후보 풀에 대한 점수
        candidate_embs = doc_embs[candidate_pool]
        sbert_scores = candidate_embs @ q_emb
        
        # Baseline에서는 앵커와 CE 점수를 미사용
        anchor_scores = np.full_like(sbert_scores, np.inf, dtype=np.float32)
        ce_scores = np.full_like(sbert_scores, np.inf, dtype=np.float32)
        
        # 지표 계산
        m = compute_metrics(order, relevant, k_list)
        filter_metrics = compute_filtering_metrics(
            sbert_scores, anchor_scores, ce_scores,
            candidate_pool, positive_doc, negative_docs,
            drop_thresholds,
            on_intent_ids=label_idx[lab]
        )
        
        print(f"[Baseline] Dropped: {filter_metrics['Dropped_Count']}")
        
        for k, v in m.items():
            agg[k].append(v)
        for k, v in filter_metrics.items():
            filter_agg[k].append(v)
    
    # 결과 저장
    res = {
        "meta": {
            "variant": "sbert_cosine",
            "k_list": k_list,
            "drop_thresholds": drop_thresholds,
        },
        "metrics": aggregate_metrics(agg),
        "filter_metrics": aggregate_metrics(filter_agg),
        "elapsed_ms": t1()
    }
    
    # 파일명은 호출하는 쪽에서 결정
    return res

def run_anchor_experiment(sbert, doc_embs, corpus_texts, corpus_labels, query_texts, query_labels,
                         label_idx, candidate_cache, k_list, max_queries, negative_samples,
                         drop_thresholds, out_dir, cache_dir, show_progress,
                         anchor_count, ollama_host, ollama_port, ollama_model):
    """Anchor 실험 실행 (SBERT + Cosine + Anchors)"""
    agg = defaultdict(list)
    filter_agg = defaultdict(list)
    t2 = timer_ms()
    
    for i, (q, lab) in enumerate(zip(query_texts, query_labels)):
        if i >= max_queries:
            break
        print(f"[Anchors] Processing query {i+1}/{min(max_queries, len(query_texts))}")
        
        relevant = set(label_idx[lab])
        
        # 필터링용 후보 풀 생성
        key = i
        if key not in candidate_cache:
            candidate_cache[key] = create_filtering_candidates(
                corpus_texts, corpus_labels, label_idx, lab, negative_samples
            )
        candidate_pool, positive_doc, negative_docs = candidate_cache[key]
        
        # SBERT + Anchors
        q_emb = embed_texts(sbert, [q], 1, show_progress)[0]
        sims_base = doc_embs @ q_emb
        order_base = np.argsort(sims_base)[::-1]
        
        # 앵커 생성 및 확신도 검사
        accepted_scores = [sims_base]
        anchors = generate_anchors_ollama(q, anchor_count, ollama_host, ollama_port, ollama_model, cache_dir)
        
        for anchor_text in anchors:
            a_emb = embed_texts(sbert, [anchor_text], 1, show_progress)[0]
            sims_a = doc_embs @ a_emb
            order_a = np.argsort(sims_a)[::-1]
            if is_confident(order_base, sims_a, order_a):
                accepted_scores.append(sims_a)
        
        # 앵커 결합
        if len(accepted_scores) > 1:
            S = np.stack(accepted_scores, axis=1)
            sims_merged = S.max(axis=1)
            order_merged = np.argsort(sims_merged)[::-1]
            order_pick, _ = pick_better_by_ndcg(order_base, order_merged, relevant, k_focus=10)
        else:
            order_pick = order_base
        
        # 지표 계산
        candidate_embs = doc_embs[candidate_pool]
        sbert_scores = candidate_embs @ q_emb
        
        # 앵커 점수 계산
        anchor_scores = np.zeros_like(sbert_scores)
        if len(accepted_scores) > 1:
            anchor_candidate_scores = []
            for anchor_sim in accepted_scores[1:]:
                anchor_candidate_scores.append(anchor_sim[candidate_pool])
            anchor_candidate_scores = np.stack(anchor_candidate_scores, axis=1)
            anchor_scores = anchor_candidate_scores.max(axis=1)
        
        ce_scores = np.full_like(sbert_scores, np.inf, dtype=np.float32)
        
        m = compute_metrics(order_pick, relevant, k_list)
        filter_metrics = compute_filtering_metrics(
            sbert_scores, anchor_scores, ce_scores,
            candidate_pool, positive_doc, negative_docs,
            drop_thresholds,
            on_intent_ids=label_idx[lab]
        )
        
        print(f"[Anchors] Dropped: {filter_metrics['Dropped_Count']}")
        
        for k, v in m.items():
            agg[k].append(v)
        for k, v in filter_metrics.items():
            filter_agg[k].append(v)
    
    # 결과 저장
    res = {
        "meta": {
            "variant": "sbert_anchors",
            "anchor_count": anchor_count,
            "k_list": k_list,
            "drop_thresholds": drop_thresholds,
        },
        "metrics": aggregate_metrics(agg),
        "filter_metrics": aggregate_metrics(filter_agg),
        "elapsed_ms": t2()
    }
    
    return res

def run_crossencoder_experiment(sbert, doc_embs, ce, corpus_texts, corpus_labels, query_texts, query_labels,
                               label_idx, candidate_cache, ce_cache, k_list, max_queries, negative_samples,
                               drop_thresholds, out_dir, cache_dir, show_progress,
                               anchor_count, ollama_host, ollama_port, ollama_model, top_m):
    """CrossEncoder 실험 실행 (SBERT + Anchors + CrossEncoder)"""
    agg = defaultdict(list)
    filter_agg = defaultdict(list)
    t3 = timer_ms()
    
    for i, (q, lab) in enumerate(zip(query_texts, query_labels)):
        if i >= max_queries:
            break
        print(f"[CrossEncoder] Processing query {i+1}/{min(max_queries, len(query_texts))}")
        
        relevant = set(label_idx[lab])
        
        # 필터링용 후보 풀 생성
        key = i
        if key not in candidate_cache:
            candidate_cache[key] = create_filtering_candidates(
                corpus_texts, corpus_labels, label_idx, lab, negative_samples
            )
        candidate_pool, positive_doc, negative_docs = candidate_cache[key]
        
        # SBERT + Anchors (이전 단계와 동일)
        q_emb = embed_texts(sbert, [q], 1, show_progress)[0]
        sims_base = doc_embs @ q_emb
        order_base = np.argsort(sims_base)[::-1]
        
        accepted_scores = [sims_base]
        anchors = generate_anchors_ollama(q, anchor_count, ollama_host, ollama_port, ollama_model, cache_dir)
        
        for anchor_text in anchors:
            a_emb = embed_texts(sbert, [anchor_text], 1, show_progress)[0]
            sims_a = doc_embs @ a_emb
            order_a = np.argsort(sims_a)[::-1]
            if is_confident(order_base, sims_a, order_a):
                accepted_scores.append(sims_a)
        
        if len(accepted_scores) > 1:
            S = np.stack(accepted_scores, axis=1)
            sims_merged = S.max(axis=1)
            order_merged = np.argsort(sims_merged)[::-1]
            order_pick, _ = pick_better_by_ndcg(order_base, order_merged, relevant, k_focus=10)
        else:
            order_pick = order_base
        
        # CrossEncoder 재정렬
        order_reranked = rerank_crossencoder(ce, i, q, corpus_texts, order_pick, top_m, ce_cache)
        order_final, _ = pick_better_by_ndcg(order_pick, order_reranked, relevant, k_focus=10)
        
        # 지표 계산
        candidate_embs = doc_embs[candidate_pool]
        sbert_scores = candidate_embs @ q_emb
        
        # 앵커 점수 계산
        anchor_scores = np.zeros_like(sbert_scores)
        if len(accepted_scores) > 1:
            anchor_candidate_scores = []
            for anchor_sim in accepted_scores[1:]:
                anchor_candidate_scores.append(anchor_sim[candidate_pool])
            anchor_candidate_scores = np.stack(anchor_candidate_scores, axis=1)
            anchor_scores = anchor_candidate_scores.max(axis=1)
        
        # CE 점수 계산
        ce_scores = ce_score_pairs_cached(ce, i, q, candidate_pool, corpus_texts, ce_cache)
        
        m = compute_metrics(order_final, relevant, k_list)
        filter_metrics = compute_filtering_metrics(
            sbert_scores, anchor_scores, ce_scores,
            candidate_pool, positive_doc, negative_docs,
            drop_thresholds,
            on_intent_ids=label_idx[lab]
        )
        
        print(f"[CrossEncoder] Dropped: {filter_metrics['Dropped_Count']}")
        
        for k, v in m.items():
            agg[k].append(v)
        for k, v in filter_metrics.items():
            filter_agg[k].append(v)
    
    # 결과 저장
    res = {
        "meta": {
            "variant": "sbert_anchors_ce",
            "anchor_count": anchor_count,
            "top_m": top_m,
            "k_list": k_list,
            "drop_thresholds": drop_thresholds,
        },
        "metrics": aggregate_metrics(agg),
        "filter_metrics": aggregate_metrics(filter_agg),
        "elapsed_ms": t3()
    }
    
    return res
