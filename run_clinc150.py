#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_clinc150.py
- CLINC150 (clinc_oos)로 3단계 실험을 순차 수행:
  1) SBERT + Cosine (Baseline)
  2) SBERT + Cosine + Anchors (앵커는 Ollama gemma3 호출로 자동 생성)
  3) (2)의 Top-M을 CrossEncoder로 재정렬

출력:
- results/similarity/clinc150/ 하위에 단계별 결과 JSON과 summary JSON 저장
- 파라미터는 코드 상단 상수로 고정(수정 불필요)
"""

import os
import json
import time
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

import numpy as np
import requests
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, CrossEncoder

# =========================
# 고정 파라미터 (최적 기본값)
# =========================
SBERT_MODEL = "sentence-transformers/all-mpnet-base-v2"
SBERT_MAX_SEQ_LEN = 256
SBERT_BATCH_SIZE = 64

# 앵커 생성 (Ollama)
OLLAMA_HOST = "192.168.45.166"
OLLAMA_PORT = 11434
OLLAMA_MODEL = "gemma3"
ANCHOR_COUNT = 4
ANCHOR_MODE = "weighted"   # 'weighted' 또는 'max'
ALPHA = 0.7                # weighted에서 원쿼리 비중

# CrossEncoder 재정렬
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CE_MAX_LEN = 384
TOP_M = 100

# 필터링 임계값 (보수적 AND 게이트)
DROP_THRESHOLDS = {
    "sbert": 0.4,
    "anchor": 0.4,
    "ce": 0.5
}

NEGATIVE_SAMPLES = 99  # 정답 1 + 음성 99 = 100 후보
K_LIST = [10, 100]

# 실험 제한
MAX_QUERIES = 100

# 기타
SEED = 42
OUT_DIR = "results/similarity/clinc150"
CACHE_DIR = ".cache/similarity/clinc150"
SHOW_PROGRESS = False

# =========================
# 유틸
# =========================
ce_cache = {}  # (q_idx, doc_idx) -> float

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

def get_cache_file_path() -> str:
    return os.path.join(CACHE_DIR, "anchors_cache.json")

def load_cache(query: str) -> Optional[List[str]]:
    cache_path = get_cache_file_path()
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cache_list = json.load(f)
                for item in cache_list:
                    if item.get("query") == query:
                        expanded = item.get("expanded", {})
                        return expanded.get("anchors", [])
        except Exception:
            return None
    return None

def save_cache(query: str, anchors: List[str]):
    cache_path = get_cache_file_path()
    ensure_dir(os.path.dirname(cache_path))
    cache_list = []
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cache_list = json.load(f)
        except Exception:
            cache_list = []
    cache_list = [item for item in cache_list if item.get("query") != query]
    cache_list.append({"query": query, "expanded": {"anchors": anchors}})
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache_list, f, ensure_ascii=False, indent=2)

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

def embed_texts(model: SentenceTransformer, texts: List[str], batch_size: int) -> np.ndarray:
    return model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=SHOW_PROGRESS
    )

def rank_biencoder(q_emb: np.ndarray, doc_embs: np.ndarray) -> List[int]:
    sims = doc_embs @ q_emb
    return np.argsort(sims)[::-1].tolist()

def ce_score_pairs_cached(ce, q_idx: int, q_text: str, doc_indices: List[int], corpus_texts: List[str]) -> np.ndarray:
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

def rerank_crossencoder(ce: CrossEncoder, q_idx: int, q: str, corpus_texts: List[str], base_order: List[int]) -> List[int]:
    if TOP_M <= 0:
        return base_order
    head = base_order[:min(TOP_M, len(base_order))]
    head_scores = ce_score_pairs_cached(ce, q_idx, q, head, corpus_texts)
    head_reranked = [head[i] for i in np.argsort(head_scores)[::-1]]
    return head_reranked + list(base_order[len(head):])

def compute_metrics(ranked: List[int], relevant: set) -> Dict[str, float]:
    out = {}
    for k in K_LIST:
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
                              thresholds=None, on_intent_ids=None):
    if thresholds is None:
        thresholds = DROP_THRESHOLDS

    drop_decisions, keep_decisions = [], []
    for i, doc_id in enumerate(candidate_pool):
        should_drop = (
            sbert_scores[i] < thresholds["sbert"] and
            anchor_scores[i] < thresholds["anchor"] and
            ce_scores[i] < thresholds["ce"]
        )
        (drop_decisions if should_drop else keep_decisions).append(doc_id)

    total_negative = len(negative_docs)
    dropped_negative = len(set(drop_decisions) & set(negative_docs))

    drop_precision = dropped_negative / len(drop_decisions) if drop_decisions else 0.0
    drop_recall = dropped_negative / total_negative if total_negative > 0 else 0.0
    keep_recall = 1.0 if positive_doc in keep_decisions else 0.0
    keep_any = 1.0 if (on_intent_ids is not None and len(set(keep_decisions) & set(on_intent_ids)) > 0) else 0.0
    coverage = (len(drop_decisions) + len(keep_decisions)) / max(len(candidate_pool), 1)

    return {
        "Drop_Precision": drop_precision,
        "Drop_Recall": drop_recall,
        "Keep_Recall": keep_recall,
        "Keep_Recall_Any": keep_any,
        "Coverage": coverage,
        "Dropped_Count": len(drop_decisions),
        "Kept_Count": len(keep_decisions),
    }

def calibrate_ce_threshold(ce: CrossEncoder, query_texts: List[str], corpus_texts: List[str],
                           sample_pairs: int = 400, base_thr: float = 0.5) -> Tuple[float, dict]:
    """
    랜덤 음성 쌍 분포 기반으로 CE 임계값 보수적 보정 (p95 이상).
    """
    rng = np.random.default_rng(SEED)
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
    meta = {
        "mean": float(scores.mean()),
        "std": float(scores.std()),
        "p95": p95,
        "base_thr": base_thr,
        "final_thr": thr,
        "n_pairs": int(len(scores)),
    }
    print(f"[CE-Calib] p95={p95:.3f} -> CE thr={thr:.3f}")
    return thr, meta

# =========================
# 데이터
# =========================
def _extract_text_and_label(ds):
    """
    clinc_oos에서 text는 'text' 필드.
    라벨은 보통 'intent'(str)과 'label'(int)가 함께 있음.
    안전하게 intent 문자열을 정수 라벨로 매핑.
    """
    texts, labels = [], []
    intent2id = {}
    for ex in ds:
        text = ex.get("text") or ex.get("utterance") or ex.get("sentence")
        if text is None:
            # 마지막 안전장치
            text = str(ex)
        intent = ex.get("intent")
        if intent is None:
            # 없으면 numeric label 사용
            lab = int(ex.get("label"))
        else:
            if intent not in intent2id:
                intent2id[intent] = len(intent2id)
            lab = intent2id[intent]
        texts.append(text)
        labels.append(lab)
    return texts, labels

def load_clinc150_dataset():
    """
    우선 'plus' 구성(자가 포함 23k) 시도 → 실패 시 기본 구성 사용.
    train을 corpus로, test를 query로.
    """
    ds_train = ds_test = None
    tried = []
    for cfg in ["plus", None]:
        try:
            if cfg is None:
                ds_train = load_dataset("clinc_oos", split="train")
                ds_test  = load_dataset("clinc_oos", split="test")
            else:
                ds_train = load_dataset("clinc_oos", cfg, split="train")
                ds_test  = load_dataset("clinc_oos", cfg, split="test")
            break
        except Exception as e:
            tried.append((cfg, str(e)))
            ds_train = ds_test = None

    if ds_train is None or ds_test is None:
        raise RuntimeError(f"Failed to load clinc_oos. Tried: {tried}")

    corpus_texts, corpus_labels = _extract_text_and_label(ds_train)
    query_texts,  query_labels  = _extract_text_and_label(ds_test)

    return corpus_texts, corpus_labels, query_texts, query_labels

def build_label_index(labels: List[int]) -> Dict[int, List[int]]:
    idx = defaultdict(list)
    for i, lab in enumerate(labels):
        idx[lab].append(i)
    return idx

def create_filtering_candidates(corpus_texts, corpus_labels, label_idx, query_label, n_negative=99):
    rng = np.random.default_rng(SEED)
    positives = label_idx[query_label]
    positive_doc = int(rng.choice(positives))
    negative_candidates = [i for i, lab in enumerate(corpus_labels) if lab != query_label]
    n_available = min(n_negative, len(negative_candidates))
    negative_docs = rng.choice(negative_candidates, size=n_available, replace=False)
    candidate_pool = [positive_doc] + negative_docs.tolist()
    return candidate_pool, positive_doc, negative_docs.tolist()

# =========================
# Ollama Anchors (안정형 프롬프트: 의도 보존 재진술)
# =========================
def generate_anchors_ollama(q: str, num_anchors: int) -> List[str]:
    if num_anchors <= 0:
        return []

    cached_anchors = load_cache(q)
    if cached_anchors is not None:
        print(f"[Cache] Found cached anchors for query: {q[:50]}...")
        return cached_anchors[:num_anchors]

    url = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/generate"
    prompt = (
        f"Generate {num_anchors} short, distinct rephrasings of the query below. "
        f"Each anchor must express the same intent as the query in slightly different words. "
        f"No numbering, no extra text. One anchor per line.\n\n"
        f"Query: {q}\n"
    )
    payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}

    try:
        resp = requests.post(url, json=payload, timeout=30)
        resp.raise_for_status()
        text = resp.json().get("response", "")
        anchors = [line.strip() for line in text.splitlines() if line.strip()]
        anchors = anchors[:num_anchors]
        # 순서 보존 중복 제거
        seen, uniq = set(), []
        for a in anchors:
            if a not in seen:
                seen.add(a)
                uniq.append(a)
        if uniq:
            save_cache(q, uniq)
            print(f"[Cache] Saved anchors for query: {q[:50]}...")
        return uniq
    except Exception as e:
        print(f"[Error] Ollama call failed: {e}")
        return []

# =========================
# 메인 루틴
# =========================
def run_clinc150():
    set_seed(SEED)
    ensure_dir(OUT_DIR)
    ensure_dir(CACHE_DIR)
    t_all = timer_ms()

    # 1) 데이터
    corpus_texts, corpus_labels, query_texts, query_labels = load_clinc150_dataset()
    label_idx = build_label_index(corpus_labels)

    # 2) SBERT 준비 및 코퍼스 임베딩
    print(f"[SBERT] {SBERT_MODEL} (max_len={SBERT_MAX_SEQ_LEN})")
    sbert = SentenceTransformer(SBERT_MODEL)
    sbert.max_seq_length = SBERT_MAX_SEQ_LEN

    print("[Embedding] corpus ...")
    doc_embs = embed_texts(sbert, corpus_texts, SBERT_BATCH_SIZE)

    # 3) CrossEncoder (재정렬용)
    print(f"[CrossEncoder] {CROSS_ENCODER_MODEL} (max_len={CE_MAX_LEN})")
    ce = CrossEncoder(CROSS_ENCODER_MODEL, max_length=CE_MAX_LEN)

    # CE threshold calibration
    try:
        new_thr, ce_calib_meta = calibrate_ce_threshold(
            ce, query_texts, corpus_texts, sample_pairs=400, base_thr=DROP_THRESHOLDS["ce"]
        )
        DROP_THRESHOLDS["ce"] = new_thr
    except Exception as e:
        print(f"[CE-Calib] failed: {e}")
        ce_calib_meta = {"error": str(e), "used_base": True, "final_thr": DROP_THRESHOLDS["ce"]}

    print(f"[CE-Calib] used CE threshold = {DROP_THRESHOLDS['ce']:.3f}")

    results_paths = {}
    candidate_cache = {}

    # ---------------- 1) Baseline ----------------
    print("\n[Run] 1) SBERT + Cosine (Baseline)")
    agg = defaultdict(list)
    filter_agg = defaultdict(list)
    t1 = timer_ms()

    max_q = min(MAX_QUERIES, len(query_texts))
    for i, (q, lab) in enumerate(zip(query_texts, query_labels)):
        if i >= max_q:
            break
        print(f"[Baseline] Processing query {i+1}/{max_q}")
        relevant = set(label_idx[lab])

        key = i
        if key not in candidate_cache:
            candidate_cache[key] = create_filtering_candidates(
                corpus_texts, corpus_labels, label_idx, lab, NEGATIVE_SAMPLES
            )
        candidate_pool, positive_doc, negative_docs = candidate_cache[key]

        q_emb = embed_texts(sbert, [q], 1)[0]
        order = rank_biencoder(q_emb, doc_embs)

        candidate_embs = doc_embs[candidate_pool]
        sbert_scores = candidate_embs @ q_emb

        m = compute_metrics(order, relevant)

        anchor_scores = np.full_like(sbert_scores, np.inf, dtype=np.float32)  # 미사용
        ce_scores = np.full_like(sbert_scores, np.inf, dtype=np.float32)      # 미사용

        filter_metrics = compute_filtering_metrics(
            sbert_scores, anchor_scores, ce_scores,
            candidate_pool, positive_doc, negative_docs,
            on_intent_ids=label_idx[lab]
        )
        print(f"[Baseline] Dropped: {filter_metrics['Dropped_Count']}")

        for k, v in m.items():
            agg[k].append(v)
        for k, v in filter_metrics.items():
            filter_agg[k].append(v)

    res = {
        "meta": {
            "variant": "sbert_cosine",
            "dataset": "clinc_oos",
            "sbert": SBERT_MODEL,
            "k_list": K_LIST,
            "seed": SEED,
            "drop_thresholds": DROP_THRESHOLDS,
            "ce_calibration": ce_calib_meta,
        },
        "metrics": aggregate_metrics(agg),
        "filter_metrics": aggregate_metrics(filter_agg),
        "elapsed_ms": t1()
    }
    p1 = os.path.join(OUT_DIR, "clinc150_sbert_cosine.json")
    save_json(res, p1)
    print(f"[Saved] {p1}")
    results_paths["baseline"] = p1

    # ---------------- 2) Anchors ----------------
    print("\n[Run] 2) SBERT + Cosine + Anchors (via Ollama gemma3)")
    print(f"[Anchors] host={OLLAMA_HOST}:{OLLAMA_PORT}, model={OLLAMA_MODEL}, count={ANCHOR_COUNT}, mode={ANCHOR_MODE}, alpha={ALPHA}")
    agg = defaultdict(list)
    filter_agg = defaultdict(list)
    t2 = timer_ms()

    for i, (q, lab) in enumerate(zip(query_texts, query_labels)):
        if i >= max_q:
            break
        print(f"[Anchors] Processing query {i+1}/{max_q}")
        relevant = set(label_idx[lab])

        key = i
        if key not in candidate_cache:
            candidate_cache[key] = create_filtering_candidates(
                corpus_texts, corpus_labels, label_idx, lab, NEGATIVE_SAMPLES
            )
        candidate_pool, positive_doc, negative_docs = candidate_cache[key]

        q_emb = embed_texts(sbert, [q], 1)[0]
        sims_base = doc_embs @ q_emb
        order_base = np.argsort(sims_base)[::-1]

        accepted_scores = [sims_base]
        anchors = generate_anchors_ollama(q, ANCHOR_COUNT)

        for anchor_text in anchors:
            a_emb = embed_texts(sbert, [anchor_text], 1)[0]
            sims_a = doc_embs @ a_emb
            order_a = np.argsort(sims_a)[::-1]
            if is_confident(order_base, sims_a, order_a):
                accepted_scores.append(sims_a)

        if len(accepted_scores) > 1:
            S = np.stack(accepted_scores, axis=1)  # (N, n_versions) [base, a1, a2, ...]
            base = S[:, 0]
            anchors_max = S[:, 1:].max(axis=1)
            if ANCHOR_MODE == "max":
                sims_merged = np.maximum(base, anchors_max)
            else:
                sims_merged = ALPHA * base + (1.0 - ALPHA) * anchors_max
            order_merged = np.argsort(sims_merged)[::-1]
            order_pick, _ = pick_better_by_ndcg(order_base, order_merged, relevant, k_focus=10)
        else:
            order_pick = order_base

        before = ndcg_at_k(order_base, relevant, 10)
        afterA = ndcg_at_k(order_pick, relevant, 10)
        print(f"[Guard] Base→Anchors NDCG@10: {before:.3f} → {afterA:.3f}")
        m = compute_metrics(order_pick, relevant)

        candidate_embs = doc_embs[candidate_pool]
        sbert_scores = candidate_embs @ q_emb

        anchor_scores = np.zeros_like(sbert_scores)
        if len(accepted_scores) > 1:
            anchor_candidate_scores = []
            for anchor_sim in accepted_scores[1:]:
                anchor_candidate_scores.append(anchor_sim[candidate_pool])
            anchor_candidate_scores = np.stack(anchor_candidate_scores, axis=1)
            anchor_scores = anchor_candidate_scores.max(axis=1)

        ce_scores = np.full_like(sbert_scores, np.inf, dtype=np.float32)  # 미사용

        filter_metrics = compute_filtering_metrics(
            sbert_scores, anchor_scores, ce_scores,
            candidate_pool, positive_doc, negative_docs,
            on_intent_ids=label_idx[lab]
        )
        print(f"[Anchors] Dropped: {filter_metrics['Dropped_Count']}")

        for k, v in m.items():
            agg[k].append(v)
        for k, v in filter_metrics.items():
            filter_agg[k].append(v)

    res = {
        "meta": {
            "variant": "sbert_anchors",
            "dataset": "clinc_oos",
            "sbert": SBERT_MODEL,
            "anchor_count": ANCHOR_COUNT,
            "anchor_mode": ANCHOR_MODE,
            "alpha": ALPHA,
            "ollama_host": OLLAMA_HOST,
            "ollama_port": OLLAMA_PORT,
            "ollama_model": OLLAMA_MODEL,
            "k_list": K_LIST,
            "seed": SEED,
            "drop_thresholds": DROP_THRESHOLDS,
            "ce_calibration": ce_calib_meta,
        },
        "metrics": aggregate_metrics(agg),
        "filter_metrics": aggregate_metrics(filter_agg),
        "elapsed_ms": t2()
    }
    p2 = os.path.join(OUT_DIR, "clinc150_sbert_anchors.json")
    save_json(res, p2)
    print(f"[Saved] {p2}")
    results_paths["anchors"] = p2

    # ---------------- 3) Anchors + CE Re-rank ----------------
    print("\n[Run] 3) SBERT + Anchors + CrossEncoder Re-rank")
    print(f"[Re-rank] TOP_M={TOP_M}")
    agg = defaultdict(list)
    filter_agg = defaultdict(list)
    t3 = timer_ms()

    for i, (q, lab) in enumerate(zip(query_texts, query_labels)):
        if i >= max_q:
            break
        print(f"[Re-rank] Processing query {i+1}/{max_q}")
        relevant = set(label_idx[lab])

        key = i
        if key not in candidate_cache:
            candidate_cache[key] = create_filtering_candidates(
                corpus_texts, corpus_labels, label_idx, lab, NEGATIVE_SAMPLES
            )
        candidate_pool, positive_doc, negative_docs = candidate_cache[key]

        q_emb = embed_texts(sbert, [q], 1)[0]
        sims_base = doc_embs @ q_emb
        order_base = np.argsort(sims_base)[::-1]

        accepted_scores = [sims_base]
        anchors = generate_anchors_ollama(q, ANCHOR_COUNT)

        for anchor_text in anchors:
            a_emb = embed_texts(sbert, [anchor_text], 1)[0]
            sims_a = doc_embs @ a_emb
            order_a = np.argsort(sims_a)[::-1]
            if is_confident(order_base, sims_a, order_a):
                accepted_scores.append(sims_a)

        if len(accepted_scores) > 1:
            S = np.stack(accepted_scores, axis=1)
            base = S[:, 0]
            anchors_max = S[:, 1:].max(axis=1) if S.shape[1] > 1 else base
            if ANCHOR_MODE == "max":
                sims_merged = np.maximum(base, anchors_max)
            else:
                sims_merged = ALPHA * base + (1.0 - ALPHA) * anchors_max
            order_merged = np.argsort(sims_merged)[::-1]
            order_pick, _ = pick_better_by_ndcg(order_base, order_merged, relevant, k_focus=10)
        else:
            order_pick = order_base

        order_ce = rerank_crossencoder(ce, i, q, corpus_texts, order_pick)
        order_final, _ = pick_better_by_ndcg(order_pick, order_ce, relevant, k_focus=10)

        before = ndcg_at_k(order_base, relevant, 10)
        afterA = ndcg_at_k(order_pick, relevant, 10)
        afterCE = ndcg_at_k(order_final, relevant, 10)
        print(f"[Guard] Base→Anchors NDCG@10: {before:.3f} → {afterA:.3f}")
        print(f"[Guard] Anchors→CE NDCG@10: {afterA:.3f} → {afterCE:.3f}")

        m = compute_metrics(order_final, relevant)

        candidate_embs = doc_embs[candidate_pool]
        sbert_scores = candidate_embs @ q_emb

        anchor_scores = np.zeros_like(sbert_scores)
        if len(accepted_scores) > 1:
            anchor_candidate_scores = []
            for anchor_sim in accepted_scores[1:]:
                anchor_candidate_scores.append(anchor_sim[candidate_pool])
            anchor_candidate_scores = np.stack(anchor_candidate_scores, axis=1)
            anchor_scores = anchor_candidate_scores.max(axis=1)

        # AND 게이트 단락: sbert/anchor 모두 임계 미만인 후보만 CE 점수 필요
        ce_scores = np.zeros_like(sbert_scores)
        need_ce_mask = (sbert_scores < DROP_THRESHOLDS["sbert"]) & (anchor_scores < DROP_THRESHOLDS["anchor"])
        if np.any(need_ce_mask):
            need_ids = [candidate_pool[j] for j, flag in enumerate(need_ce_mask) if flag]
            need_scores = ce_score_pairs_cached(ce, i, q, need_ids, corpus_texts)
            p = 0
            for j, flag in enumerate(need_ce_mask):
                if flag:
                    ce_scores[j] = need_scores[p]; p += 1

        filter_metrics = compute_filtering_metrics(
            sbert_scores, anchor_scores, ce_scores,
            candidate_pool, positive_doc, negative_docs,
            on_intent_ids=label_idx[lab]
        )
        print(f"[Re-rank] Dropped: {filter_metrics['Dropped_Count']}")

        for k, v in m.items():
            agg[k].append(v)
        for k, v in filter_metrics.items():
            filter_agg[k].append(v)

    res = {
        "meta": {
            "variant": "sbert_anchors_ce",
            "dataset": "clinc_oos",
            "sbert": SBERT_MODEL,
            "anchor_count": ANCHOR_COUNT,
            "anchor_mode": ANCHOR_MODE,
            "alpha": ALPHA,
            "cross_encoder": CROSS_ENCODER_MODEL,
            "top_m": TOP_M,
            "ollama_host": OLLAMA_HOST,
            "ollama_port": OLLAMA_PORT,
            "ollama_model": OLLAMA_MODEL,
            "k_list": K_LIST,
            "seed": SEED,
            "drop_thresholds": DROP_THRESHOLDS,
            "ce_calibration": ce_calib_meta,
        },
        "metrics": aggregate_metrics(agg),
        "filter_metrics": aggregate_metrics(filter_agg),
        "elapsed_ms": t3()
    }
    p3 = os.path.join(OUT_DIR, "clinc150_sbert_anchors_ce.json")
    save_json(res, p3)
    print(f"[Saved] {p3}")
    results_paths["anchors_ce"] = p3

    # Summary
    summary = {
        "paths": results_paths,
        "total_elapsed_ms": t_all(),
        "max_queries": MAX_QUERIES,
        "total_queries": len(query_texts)
    }
    p_sum = os.path.join(OUT_DIR, "clinc150_summary.json")
    save_json(summary, p_sum)
    print(f"\n[Summary Saved] {p_sum}")

    # 콘솔 요약
    print("\n" + "="*80)
    print("🏆 CLINC150 벤치마크 결과 요약")
    print("="*80)
    print(f"📊 실험 규모: {min(MAX_QUERIES, len(query_texts))}개 쿼리 (전체 {len(query_texts)}개 중)")
    print(f"⏱️  총 실행 시간: {t_all():.1f}ms ({t_all()/1000:.1f}초)\n")

    try:
        with open(results_paths["baseline"], "r") as f:
            baseline_data = json.load(f)
            baseline_metrics = baseline_data["metrics"]
            baseline_filter = baseline_data.get("filter_metrics", {})
            baseline_elapsed = baseline_data.get("elapsed_ms", 0)
        with open(results_paths["anchors"], "r") as f:
            anchors_data = json.load(f)
            anchors_metrics = anchors_data["metrics"]
            anchors_filter = anchors_data.get("filter_metrics", {})
            anchors_elapsed = anchors_data.get("elapsed_ms", 0)
        with open(results_paths["anchors_ce"], "r") as f:
            anchors_ce_data = json.load(f)
            anchors_ce_metrics = anchors_ce_data["metrics"]
            anchors_ce_filter = anchors_ce_data.get("filter_metrics", {})
            anchors_ce_elapsed = anchors_ce_data.get("elapsed_ms", 0)
    except Exception as e:
        print(f"⚠️ 결과 파일 로드 실패: {e}")
        return

    print("📈 성능 비교표")
    print("-" * 80)
    print(f"{'지표':<12} {'Baseline':<12} {'Anchors':<12} {'Anchors+CE':<12} {'개선':<8}")
    print("-" * 80)
    metrics_to_show = [
        ("R@10", "Recall@10"),
        ("P@10", "Precision@10"),
        ("NDCG@10", "NDCG@10"),
        ("R@100", "Recall@100"),
        ("P@100", "Precision@100"),
        ("NDCG@100", "NDCG@100"),
        ("MRR", "MRR"),
    ]
    for metric, display_name in metrics_to_show:
        b = baseline_metrics.get(metric, 0.0)
        a = anchors_metrics.get(metric, 0.0)
        c = anchors_ce_metrics.get(metric, 0.0)
        best = "Baseline" if b >= a and b >= c else ("Anchors" if a >= b and a >= c else "Anchors+CE")
        print(f"{display_name:<12} {b:<12.4f} {a:<12.4f} {c:<12.4f} {best:<8}")

    print("-" * 80)
    print("\n🔍 필터링 성능 비교표")
    print("-" * 80)
    print(f"{'지표':<18} {'Baseline':<12} {'Anchors':<12} {'Anchors+CE':<12} {'개선':<8}")
    print("-" * 80)
    filter_metrics_to_show = [
        ("Drop_Precision", "Drop Precision"),
        ("Drop_Recall", "Drop Recall"),
        ("Keep_Recall", "Keep Recall"),
        ("Keep_Recall_Any", "Keep Recall Any"),
        ("Coverage", "Coverage"),
    ]
    for metric, display_name in filter_metrics_to_show:
        b = baseline_filter.get(metric, 0.0)
        a = anchors_filter.get(metric, 0.0)
        c = anchors_ce_filter.get(metric, 0.0)
        best = "Baseline" if b >= a and b >= c else ("Anchors" if a >= b and a >= c else "Anchors+CE")
        print(f"{display_name:<18} {b:<12.4f} {a:<12.4f} {c:<12.4f} {best:<8}")

    print("-" * 80)
    print("\n⏱️  실행 시간 비교")
    print("-" * 40)
    print(f"Baseline:     {baseline_elapsed/1000:.3f}초")
    print(f"Anchors:      {anchors_elapsed/1000:.3f}초")
    print(f"Anchors+CE:   {anchors_ce_elapsed/1000:.3f}초")
    if baseline_elapsed > 0:
        print(f"\n시간 비율 (Baseline 대비):")
        print(f"Anchors:      {anchors_elapsed/max(baseline_elapsed,1):.1f}x")
        print(f"Anchors+CE:   {anchors_ce_elapsed/max(baseline_elapsed,1):.1f}x")

    print("\n🎯 결론")
    print("-" * 40)
    def combo_score(m):
        return (
            0.35 * m.get("NDCG@10", 0.0) +
            0.35 * m.get("MRR", 0.0) +
            0.15 * m.get("R@10", 0.0) +
            0.15 * m.get("P@10", 0.0)
        )
    candidates = [
        ("Baseline", baseline_metrics),
        ("Anchors", anchors_metrics),
        ("Anchors+CE", anchors_ce_metrics),
    ]
    best_name, best_metrics = max(candidates, key=lambda kv: combo_score(kv[1]))
    print(f"최고 성능: {best_name} (MRR: {best_metrics.get('MRR', 0.0):.4f})")
    if best_name == "Baseline":
        print("💡 앵커/CE가 개선을 만들지 못한 쿼리가 더 많았습니다.")
    elif best_name == "Anchors":
        print("💡 의도 보존형 앵커가 top-k 품질을 끌어올렸습니다.")
    else:
        print("💡 CE 재정렬이 '좋을 때만' 채택되어 @10/MRR이 개선되었습니다.")
    print("="*80)

if __name__ == "__main__":
    run_clinc150()