#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_banking77.py
- 실행만 하면 mteb/banking77 (실패 시 banking77)으로 3단계 실험을 순차 수행:
  1) SBERT + Cosine (Baseline)
  2) SBERT + Cosine + Anchors (앵커는 Ollama gemma3 호출로 자동 생성)
  3) (2)의 Top-M을 CrossEncoder로 재정렬

출력:
- results/similarity/ 하위에 단계별 결과 JSON과 summary JSON 저장
- 파라미터는 코드 상단 상수로 고정(수정 불필요)
"""

import os
import json
import time
import hashlib
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

import numpy as np
import requests
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, CrossEncoder

# =========================
# 고정 파라미터 (최적 기본값)
# =========================
SBERT_MODEL = "sentence-transformers/all-mpnet-base-v2"   # 정확도/속도 우수한 범용 Bi-Encoder
SBERT_MAX_SEQ_LEN = 256
SBERT_BATCH_SIZE = 64

# 앵커 생성 (Ollama)
OLLAMA_HOST = "192.168.45.166"
OLLAMA_PORT = 11434
OLLAMA_MODEL = "gemma3"
ANCHOR_COUNT = 4                     # 앵커 수: 다양성/비용 균형 (6→4로 축소)
ANCHOR_MODE = "weighted"             # 'weighted' 또는 'max'
ALPHA = 0.7                          # weighted에서 원쿼리 비중

# CrossEncoder 재정렬
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CE_MAX_LEN = 384
TOP_M = 100                          # CE로 재정렬할 상위 후보 수 (정확도/지연 균형) (200→100으로 축소)

# 필터링 임계값 (보수적 AND 게이트)
DROP_THRESHOLDS = {
    "sbert": 0.4,    # SBERT 코사인 임계값
    "anchor": 0.4,   # 앵커 최대 점수 임계값
    "ce": 0.5        # CrossEncoder 점수 임계값
}

# 필터링 설정
NEGATIVE_SAMPLES = 99  # 각 쿼리당 음성 샘플 수 (정답 1개 + 음성 99개 = 100개 후보)

# 평가 k
K_LIST = [10, 100]

# 실험 제한 (빠른 테스트용)
MAX_QUERIES = 100  # 최대 10개 쿼리만 실험

# 기타
SEED = 42
OUT_DIR = "results/similarity/banking77"
CACHE_DIR = ".cache/similarity/banking77"      # Ollama 응답 캐시 디렉토리
SHOW_PROGRESS = False                # SBERT encode progress bar

# =========================
# 유틸
# =========================

# ---- CE inference cache ----
ce_cache = {}  # (q_idx, doc_idx) -> float score

def ce_score_pairs_cached(ce, q_idx: int, q_text: str, doc_indices: List[int], corpus_texts: List[str]) -> np.ndarray:
    """(q_idx, doc_idx) 단위로 캐시. 재정렬/필터링 공용."""
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
    """
    랜덤 음성 쌍 점수 분포로 CE 임계값 보수적 보정.
    권장: p95 이상으로 설정하여 Drop Precision 확보.
    """
    rng = np.random.default_rng(SEED)
    pairs = []
    n = min(len(query_texts), sample_pairs)
    for _ in range(n):
        q_idx = int(rng.integers(0, len(query_texts)))
        d_idx = int(rng.integers(0, len(corpus_texts)))
        # 같은 문장일 확률 희박하지만 안전 처리
        if d_idx == q_idx:
            d_idx = (d_idx + 1) % len(corpus_texts)
        pairs.append([query_texts[q_idx], corpus_texts[d_idx]])

    if not pairs:
        return base_thr, {"used_base": True}

    scores = np.array(ce.predict(pairs), dtype=np.float32)
    p95 = float(np.percentile(scores, 95))
    thr = max(base_thr, p95)
    meta = {"mean": float(scores.mean()), "std": float(scores.std()), "p95": p95, "base_thr": base_thr, "final_thr": thr, "n_pairs": len(scores)}
    print(f"[CE-Calib] p95={p95:.3f} -> CE thr={thr:.3f}")
    return thr, meta

def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)

def now_tag():
    return time.strftime("%Y%m%d_%H%M%S")

def get_cache_file_path() -> str:
    """통합 캐시 파일 경로 반환"""
    return os.path.join(CACHE_DIR, "anchors_cache.json")

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

def load_cache(query: str) -> Optional[List[str]]:
    """통합 캐시에서 앵커 목록 로드"""
    cache_path = get_cache_file_path()
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

def save_cache(query: str, anchors: List[str]):
    """앵커 목록을 통합 캐시에 저장"""
    cache_path = get_cache_file_path()
    ensure_dir(os.path.dirname(cache_path))
    
    # 기존 캐시 로드
    cache_list = []
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache_list = json.load(f)
        except Exception:
            cache_list = []
    
    # 중복 제거: 같은 쿼리가 있으면 제거
    cache_list = [item for item in cache_list if item.get('query') != query]
    
    # 새 데이터 추가
    cache_list.append({
        'query': query,
        'expanded': {
            'anchors': anchors
        }
    })
    
    # 캐시 저장
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(cache_list, f, ensure_ascii=False, indent=2)

# =========================
# 데이터
# =========================
def load_banking77_dataset():
    # mteb/banking77 우선, 실패 시 banking77 폴백
    try:
        ds_train = load_dataset("mteb/banking77", split="train")
        ds_test  = load_dataset("mteb/banking77", split="test")
    except Exception:
        ds_train = load_dataset("banking77", split="train")
        ds_test  = load_dataset("banking77", split="test")

    corpus_texts  = [ex["text"] for ex in ds_train]
    corpus_labels = [int(ex["label"]) for ex in ds_train]
    query_texts   = [ex["text"] for ex in ds_test]
    query_labels  = [int(ex["label"]) for ex in ds_test]
    return corpus_texts, corpus_labels, query_texts, query_labels

def build_label_index(labels: List[int]) -> Dict[int, List[int]]:
    idx = defaultdict(list)
    for i, lab in enumerate(labels):
        idx[lab].append(i)
    return idx

def create_filtering_candidates(corpus_texts, corpus_labels, label_idx, query_label, n_negative=99):
    """정답 1개(같은 라벨) + 음성 n_negative개(다른 라벨)"""
    rng = np.random.default_rng(SEED)

    positives = label_idx[query_label]
    positive_doc = int(rng.choice(positives))      # 정답 하나 샘플

    negative_candidates = [i for i, lab in enumerate(corpus_labels) if lab != query_label]
    n_available = min(n_negative, len(negative_candidates))
    negative_docs = rng.choice(negative_candidates, size=n_available, replace=False)

    candidate_pool = [positive_doc] + negative_docs.tolist()
    return candidate_pool, positive_doc, negative_docs.tolist()

# =========================
# Ollama Anchors
# =========================
def generate_anchors_ollama(q: str, num_anchors: int) -> List[str]:
    """
    Ollama /api/generate 호출(스토리밍 비활성)로 앵커 생성.
    - 캐시 우선 확인, 없으면 Ollama 호출 후 캐시 저장
    - 실패 시 빈 리스트 반환(실험은 계속 진행)
    - 각 줄이 하나의 앵커가 되도록 프롬프트 설계
    """
    if num_anchors <= 0:
        return []

    # 캐시 확인
    cached_anchors = load_cache(q)
    if cached_anchors is not None:
        print(f"[Cache] Found cached anchors for query: {q[:50]}...")
        return cached_anchors[:num_anchors]

    # Ollama 호출
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
        # 필요한 개수만 사용하고, 중복 제거(순서 보존)
        anchors = anchors[:num_anchors]
        seen, uniq = set(), []
        for a in anchors:
            if a not in seen:
                seen.add(a)
                uniq.append(a)
        
        # 캐시에 저장
        if uniq:
            save_cache(q, uniq)
            print(f"[Cache] Saved anchors for query: {q[:50]}...")
        
        return uniq
    except Exception as e:
        print(f"[Error] Ollama call failed: {e}")
        return []

# =========================
# 임베딩/랭킹
# =========================
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

def rank_with_anchors(
    q_emb: np.ndarray,
    anchor_embs: Optional[np.ndarray],
    doc_embs: np.ndarray,
) -> List[int]:
    """
    base = cos(q,d)
    anchors = max_a cos(a,d)
    weighted: ALPHA*base + (1-ALPHA)*anchors
    max: max(base, anchors)
    """
    base = doc_embs @ q_emb
    if anchor_embs is None or len(anchor_embs) == 0:
        scores = base
    else:
        max_anchor = None
        for a in anchor_embs:
            s = doc_embs @ a
            max_anchor = s if max_anchor is None else np.maximum(max_anchor, s)
        if ANCHOR_MODE == "max":
            scores = np.maximum(base, max_anchor)
        else:
            scores = ALPHA * base + (1.0 - ALPHA) * max_anchor
    return np.argsort(scores)[::-1].tolist()

def rerank_crossencoder(
    ce: CrossEncoder,
    q_idx: int,
    q: str,
    corpus_texts: List[str],
    base_order: List[int],
) -> List[int]:
    if TOP_M <= 0:
        return base_order
    head = base_order[:min(TOP_M, len(base_order))]
    head_scores = ce_score_pairs_cached(ce, q_idx, q, head, corpus_texts)
    head_reranked = [head[i] for i in np.argsort(head_scores)[::-1]]
    return head_reranked + list(base_order[len(head):])

# =========================
# 지표
# =========================
def ndcg_at_k_binary(ranked: List[int], relevant: set, k: int) -> float:
    dcg = 0.0
    for i, d in enumerate(ranked[:k], start=1):
        if d in relevant:
            dcg += 1.0 / np.log2(i + 1)
    ideal = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 1) for i in range(1, ideal + 1))
    return (dcg / idcg) if idcg > 0 else 0.0

def compute_metrics(ranked: List[int], relevant: set) -> Dict[str, float]:
    out = {}
    for k in K_LIST:
        topk = ranked[:k]
        hit = sum(1 for d in topk if d in relevant)
        denom_rel = max(len(relevant), 1)
        out[f"R@{k}"] = hit / denom_rel
        out[f"P@{k}"] = hit / max(k, 1)
        out[f"NDCG@{k}"] = ndcg_at_k_binary(ranked, relevant, k)
    # MRR
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
    """필터링 성능 지표 계산"""
    if thresholds is None:
        thresholds = DROP_THRESHOLDS
    
    # 각 후보에 대해 드롭 여부 판정 (보수적 AND 게이트)
    drop_decisions = []
    keep_decisions = []
    
    for i, doc_id in enumerate(candidate_pool):
        sbert_score = sbert_scores[i]
        anchor_score = anchor_scores[i]
        ce_score = ce_scores[i]
        
        # 드롭 조건: 모든 점수가 임계값 미만
        should_drop = (sbert_score < thresholds["sbert"] and 
                      anchor_score < thresholds["anchor"] and 
                      ce_score < thresholds["ce"])
        
        if should_drop:
            drop_decisions.append(doc_id)
        else:
            keep_decisions.append(doc_id)
    
    # 지표 계산
    total_candidates = len(candidate_pool)
    total_negative = len(negative_docs)
    
    # Drop Precision = 드롭된 것 중 진짜 off-intent 비율
    dropped_negative = len(set(drop_decisions) & set(negative_docs))
    drop_precision = dropped_negative / len(drop_decisions) if drop_decisions else 0.0
    
    # Drop Recall = 전체 off-intent 중 드롭된 비율
    drop_recall = dropped_negative / total_negative if total_negative > 0 else 0.0
    
    # Keep Recall = on-intent 보존율 (양성 유지율)
    positive_kept = positive_doc in keep_decisions
    keep_recall = 1.0 if positive_kept else 0.0
    
    # Keep Recall Any = 같은 라벨 문서 중 하나라도 keep이면 1
    kept_set = set(keep_decisions)
    keep_any = 1.0 if (on_intent_ids is not None and len(kept_set & set(on_intent_ids)) > 0) else 0.0
    
    # Coverage = drop/keep으로 판정된 비율 (보류 제외)
    coverage = (len(drop_decisions) + len(keep_decisions)) / total_candidates
    
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
# 메인 루틴
# =========================
def run_banking77():
    set_seed(SEED)
    ensure_dir(OUT_DIR)
    ensure_dir(CACHE_DIR)  # 캐시 디렉토리 생성
    t_all = timer_ms()

    # 1) 데이터
    corpus_texts, corpus_labels, query_texts, query_labels = load_banking77_dataset()
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

    # ---- NEW: CE threshold calibration ----
    try:
        new_thr, ce_calib_meta = calibrate_ce_threshold(ce, query_texts, corpus_texts, sample_pairs=400, base_thr=DROP_THRESHOLDS["ce"])
        DROP_THRESHOLDS["ce"] = new_thr
    except Exception as e:
        print(f"[CE-Calib] failed: {e}")
        ce_calib_meta = {"error": str(e), "used_base": True, "final_thr": DROP_THRESHOLDS["ce"]}
    
    print(f"[CE-Calib] used CE threshold = {DROP_THRESHOLDS['ce']:.3f}")

    results_paths = {}

    # ---- 후보 풀 캐시 (동일 쿼리에서 후보 풀 재사용) ----
    candidate_cache = {}

    # ---------------- 1) Baseline ----------------
    print("\n[Run] 1) SBERT + Cosine (Baseline)")
    agg = defaultdict(list)
    filter_agg = defaultdict(list)
    t1 = timer_ms()
    for i, (q, lab) in enumerate(zip(query_texts, query_labels)):
        if i >= MAX_QUERIES:
            break
        print(f"[Baseline] Processing query {i+1}/{min(MAX_QUERIES, len(query_texts))}")
        relevant = set(label_idx[lab])
        
        # 필터링용 후보 풀 생성 (캐시 재사용)
        key = i  # 쿼리 인덱스
        if key not in candidate_cache:
            candidate_cache[key] = create_filtering_candidates(
                corpus_texts, corpus_labels, label_idx, lab, NEGATIVE_SAMPLES
            )
        candidate_pool, positive_doc, negative_docs = candidate_cache[key]
        
        # SBERT 임베딩 및 점수 계산
        q_emb = embed_texts(sbert, [q], 1)[0]
        
        # 전체 코퍼스에 대한 순위 (기존 랭킹 평가용)
        order = rank_biencoder(q_emb, doc_embs)
        
        # 후보 풀에 대한 점수 (필터링 평가용)
        candidate_embs = doc_embs[candidate_pool]
        sbert_scores = candidate_embs @ q_emb
        
        # 기존 랭킹 지표 계산
        m = compute_metrics(order, relevant)
        
        # 필터링 지표 계산
        # Baseline에서는 앵커와 CE 점수를 미사용으로 설정 (절대 드롭에 기여하지 않음)
        anchor_scores = np.full_like(sbert_scores, np.inf, dtype=np.float32)  # 미사용
        ce_scores = np.full_like(sbert_scores, np.inf, dtype=np.float32)  # 미사용
        
        filter_metrics = compute_filtering_metrics(
            sbert_scores, anchor_scores, ce_scores,
            candidate_pool, positive_doc, negative_docs,
            on_intent_ids=label_idx[lab]
        )
        
        print(f"[Baseline] Dropped: {filter_metrics['Dropped_Count']}")
        
        # 결과 저장
        for k, v in m.items():
            agg[k].append(v)
        for k, v in filter_metrics.items():
            filter_agg[k].append(v)
    res = {
        "meta": {
            "variant": "sbert_cosine",
            "dataset": "banking77",
            "sbert": SBERT_MODEL,
            "k_list": K_LIST,
            "seed": SEED,
            "drop_thresholds": DROP_THRESHOLDS,
        },
        "metrics": aggregate_metrics(agg),
        "filter_metrics": aggregate_metrics(filter_agg),
        "elapsed_ms": t1()
    }
    p1 = os.path.join(OUT_DIR, "banking77_sbert_cosine.json")
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
        if i >= MAX_QUERIES:
            break
        print(f"[Anchors] Processing query {i+1}/{min(MAX_QUERIES, len(query_texts))}")
        relevant = set(label_idx[lab])
        
        # 필터링용 후보 풀 생성 (캐시 재사용)
        key = i  # 쿼리 인덱스
        if key not in candidate_cache:
            candidate_cache[key] = create_filtering_candidates(
                corpus_texts, corpus_labels, label_idx, lab, NEGATIVE_SAMPLES
            )
        candidate_pool, positive_doc, negative_docs = candidate_cache[key]
        
        # Baseline 점수/순위 (이미 계산되어 있으면 재사용)
        q_emb = embed_texts(sbert, [q], 1)[0]
        sims_base = doc_embs @ q_emb
        order_base = np.argsort(sims_base)[::-1]
        
        # 앵커 개별 점수화 → 확실한 앵커만 채택
        accepted_scores = [sims_base]   # 항상 Baseline 포함
        anchors = generate_anchors_ollama(q, ANCHOR_COUNT)
        
        for anchor_text in anchors:
            a_emb = embed_texts(sbert, [anchor_text], 1)[0]
            sims_a = doc_embs @ a_emb
            order_a = np.argsort(sims_a)[::-1]
            if is_confident(order_base, sims_a, order_a):
                accepted_scores.append(sims_a)
        
        # 확실 앵커가 있으면 per-doc max 결합
        if len(accepted_scores) > 1:
            S = np.stack(accepted_scores, axis=1)   # (N, n_versions)
            sims_merged = S.max(axis=1)             # (N,)
            order_merged = np.argsort(sims_merged)[::-1]
            # 안전-가드: Baseline vs Merged 중 더 좋은 쪽만 선택 (NDCG@10)
            order_pick, _ = pick_better_by_ndcg(order_base, order_merged, relevant, k_focus=10)
        else:
            order_pick = order_base
        
        # 로그 출력
        print(f"[Anchors] accepted_rate={ (len(accepted_scores)-1) / max(len(anchors),1):.2f}")
        print(f"[Anchors] accepted={len(accepted_scores)-1} (of {len(anchors)})")
        before = ndcg_at_k(order_base, relevant, 10)
        afterA = ndcg_at_k(order_pick, relevant, 10)
        print(f"[Guard] Base→Anchors NDCG@10: {before:.3f} → {afterA:.3f}")
        
        # 기존 랭킹 지표 계산
        m = compute_metrics(order_pick, relevant)
        
        # 필터링 지표 계산
        # 후보 풀에 대한 점수 계산
        candidate_embs = doc_embs[candidate_pool]
        sbert_scores = candidate_embs @ q_emb
        
        # 앵커 점수 계산 (후보 풀에 대해서만)
        anchor_scores = np.zeros_like(sbert_scores)
        if len(accepted_scores) > 1:
            # 앵커가 채택된 경우, 후보 풀에 대해 앵커 점수 계산
            anchor_candidate_scores = []
            for anchor_sim in accepted_scores[1:]:  # Baseline 제외
                anchor_candidate_scores.append(anchor_sim[candidate_pool])
            anchor_candidate_scores = np.stack(anchor_candidate_scores, axis=1)
            anchor_scores = anchor_candidate_scores.max(axis=1)
        
        # CE 점수는 미사용으로 설정 (Anchors 단계에서는 계산하지 않음)
        ce_scores = np.full_like(sbert_scores, np.inf, dtype=np.float32)  # 미사용
        
        filter_metrics = compute_filtering_metrics(
            sbert_scores, anchor_scores, ce_scores,
            candidate_pool, positive_doc, negative_docs,
            on_intent_ids=label_idx[lab]
        )
        
        print(f"[Anchors] Dropped: {filter_metrics['Dropped_Count']}")
        
        # 결과 저장
        for k, v in m.items():
            agg[k].append(v)
        for k, v in filter_metrics.items():
            filter_agg[k].append(v)
    res = {
        "meta": {
            "variant": "sbert_anchors",
            "dataset": "banking77",
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
        },
    "metrics": aggregate_metrics(agg),
    "filter_metrics": aggregate_metrics(filter_agg),
    "elapsed_ms": t2()
    }
    p2 = os.path.join(OUT_DIR, "banking77_sbert_anchors.json")
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
        if i >= MAX_QUERIES:
            break
        print(f"[Re-rank] Processing query {i+1}/{min(MAX_QUERIES, len(query_texts))}")
        relevant = set(label_idx[lab])
        
        # 필터링용 후보 풀 생성 (캐시 재사용)
        key = i  # 쿼리 인덱스
        if key not in candidate_cache:
            candidate_cache[key] = create_filtering_candidates(
                corpus_texts, corpus_labels, label_idx, lab, NEGATIVE_SAMPLES
            )
        candidate_pool, positive_doc, negative_docs = candidate_cache[key]
        
        # Baseline 점수/순위 (이미 계산되어 있으면 재사용)
        q_emb = embed_texts(sbert, [q], 1)[0]
        sims_base = doc_embs @ q_emb
        order_base = np.argsort(sims_base)[::-1]
        
        # 앵커 개별 점수화 → 확실한 앵커만 채택
        accepted_scores = [sims_base]   # 항상 Baseline 포함
        anchors = generate_anchors_ollama(q, ANCHOR_COUNT)
        
        for anchor_text in anchors:
            a_emb = embed_texts(sbert, [anchor_text], 1)[0]
            sims_a = doc_embs @ a_emb
            order_a = np.argsort(sims_a)[::-1]
            if is_confident(order_base, sims_a, order_a):
                accepted_scores.append(sims_a)
        
        # 확실 앵커가 있으면 per-doc max 결합
        if len(accepted_scores) > 1:
            S = np.stack(accepted_scores, axis=1)   # (N, n_versions)
            sims_merged = S.max(axis=1)             # (N,)
            order_merged = np.argsort(sims_merged)[::-1]
            # 안전-가드: Baseline vs Merged 중 더 좋은 쪽만 선택 (NDCG@10)
            order_pick, _ = pick_better_by_ndcg(order_base, order_merged, relevant, k_focus=10)
        else:
            order_pick = order_base
        
        # CrossEncoder 재정렬에 안전-가드 추가
        order_ce = rerank_crossencoder(ce, i, q, corpus_texts, order_pick)
        order_final, _ = pick_better_by_ndcg(order_pick, order_ce, relevant, k_focus=10)
        
        # 로그 출력
        print(f"[Anchors] accepted_rate={ (len(accepted_scores)-1) / max(len(anchors),1):.2f}")
        print(f"[Anchors] accepted={len(accepted_scores)-1} (of {len(anchors)})")
        print(f"[CE] thr={DROP_THRESHOLDS['ce']:.3f}, cached_pairs={len(ce_cache)}")
        before = ndcg_at_k(order_base, relevant, 10)
        afterA = ndcg_at_k(order_pick, relevant, 10)
        afterCE = ndcg_at_k(order_final, relevant, 10)
        print(f"[Guard] Base→Anchors NDCG@10: {before:.3f} → {afterA:.3f}")
        print(f"[Guard] Anchors→CE NDCG@10: {afterA:.3f} → {afterCE:.3f}")
        
        # 기존 랭킹 지표 계산
        m = compute_metrics(order_final, relevant)
        
        # 필터링 지표 계산
        # 후보 풀에 대한 점수 계산
        candidate_embs = doc_embs[candidate_pool]
        sbert_scores = candidate_embs @ q_emb
        
        # 앵커 점수 계산 (후보 풀에 대해서만)
        anchor_scores = np.zeros_like(sbert_scores)
        if len(accepted_scores) > 1:
            # 앵커가 채택된 경우, 후보 풀에 대해 앵커 점수 계산
            anchor_candidate_scores = []
            for anchor_sim in accepted_scores[1:]:  # Baseline 제외
                anchor_candidate_scores.append(anchor_sim[candidate_pool])
            anchor_candidate_scores = np.stack(anchor_candidate_scores, axis=1)
            anchor_scores = anchor_candidate_scores.max(axis=1)
        
        # CE 점수 계산 (후보 풀에 대해서만) - AND 게이트 단락 적용
        ce_scores = np.zeros_like(sbert_scores)
        # CE는 '모두 미만이면 drop' 규칙에서만 필요하므로, 사전 단락
        need_ce_mask = (sbert_scores < DROP_THRESHOLDS["sbert"]) & (anchor_scores < DROP_THRESHOLDS["anchor"])
        
        if np.any(need_ce_mask):
            need_ids = [candidate_pool[j] for j, flag in enumerate(need_ce_mask) if flag]
            need_scores = ce_score_pairs_cached(ce, i, q, need_ids, corpus_texts)
            p = 0
            for j, flag in enumerate(need_ce_mask):
                if flag:
                    ce_scores[j] = need_scores[p]; p += 1
        # need_ce_mask가 False인 경우(=keep 확정) CE 점수는 0으로 두어도 drop 규칙에 영향 없음
        
        filter_metrics = compute_filtering_metrics(
            sbert_scores, anchor_scores, ce_scores,
            candidate_pool, positive_doc, negative_docs,
            on_intent_ids=label_idx[lab]
        )
        
        print(f"[Re-rank] Dropped: {filter_metrics['Dropped_Count']}")
        
        # 결과 저장
        for k, v in m.items():
            agg[k].append(v)
        for k, v in filter_metrics.items():
            filter_agg[k].append(v)
    res = {
        "meta": {
            "variant": "sbert_anchors_ce",
            "dataset": "banking77",
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
    p3 = os.path.join(OUT_DIR, "banking77_sbert_anchors_ce.json")
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
    p_sum = os.path.join(OUT_DIR, "banking77_summary.json")
    save_json(summary, p_sum)
    print(f"\n[Summary Saved] {p_sum}")
    
    # 핵심 결과 콘솔 출력
    print("\n" + "="*80)
    print("🏆 BANKING77 벤치마크 결과 요약")
    print("="*80)
    print(f"📊 실험 규모: {MAX_QUERIES}개 쿼리 (전체 {len(query_texts)}개 중)")
    print(f"⏱️  총 실행 시간: {t_all():.1f}ms ({t_all()/1000:.1f}초)")
    print()
    
    # 각 단계별 결과 로드 및 출력
    baseline_metrics = {}
    anchors_metrics = {}
    anchors_ce_metrics = {}
    
    try:
        with open(results_paths["baseline"], 'r') as f:
            baseline_data = json.load(f)
            baseline_metrics = baseline_data["metrics"]
            baseline_filter = baseline_data.get("filter_metrics", {})
            baseline_elapsed = baseline_data.get("elapsed_ms", 0)
        with open(results_paths["anchors"], 'r') as f:
            anchors_data = json.load(f)
            anchors_metrics = anchors_data["metrics"]
            anchors_filter = anchors_data.get("filter_metrics", {})
            anchors_elapsed = anchors_data.get("elapsed_ms", 0)
        with open(results_paths["anchors_ce"], 'r') as f:
            anchors_ce_data = json.load(f)
            anchors_ce_metrics = anchors_ce_data["metrics"]
            anchors_ce_filter = anchors_ce_data.get("filter_metrics", {})
            anchors_ce_elapsed = anchors_ce_data.get("elapsed_ms", 0)
    except Exception as e:
        print(f"⚠️  결과 파일 로드 실패: {e}")
        return
    
    print("📈 성능 비교표")
    print("-" * 80)
    print(f"{'지표':<12} {'Baseline':<12} {'Anchors':<12} {'Anchors+CE':<12} {'개선':<8}")
    print("-" * 80)
    
    # 주요 지표들
    metrics_to_show = [
        ("R@10", "Recall@10"),
        ("P@10", "Precision@10"), 
        ("NDCG@10", "NDCG@10"),
        ("R@100", "Recall@100"),
        ("P@100", "Precision@100"),
        ("NDCG@100", "NDCG@100"),
        ("MRR", "MRR")
    ]
    
    for metric, display_name in metrics_to_show:
        baseline_val = baseline_metrics.get(metric, 0)
        anchors_val = anchors_metrics.get(metric, 0)
        anchors_ce_val = anchors_ce_metrics.get(metric, 0)
        
        # 개선도 계산 (Baseline 대비)
        best_val = max(baseline_val, anchors_val, anchors_ce_val)
        if best_val == baseline_val:
            improvement = "Baseline"
        elif best_val == anchors_val:
            improvement = "Anchors"
        else:
            improvement = "Anchors+CE"
        
        print(f"{display_name:<12} {baseline_val:<12.4f} {anchors_val:<12.4f} {anchors_ce_val:<12.4f} {improvement:<8}")
    
    print("-" * 80)
    
    # 필터링 성능 비교표
    print("\n🔍 필터링 성능 비교표")
    print("-" * 80)
    print(f"{'지표':<15} {'Baseline':<12} {'Anchors':<12} {'Anchors+CE':<12} {'개선':<8}")
    print("-" * 80)
    
    # 필터링 지표들
    filter_metrics_to_show = [
        ("Drop_Precision", "Drop Precision"),
        ("Drop_Recall", "Drop Recall"),
        ("Keep_Recall", "Keep Recall"),
        ("Coverage", "Coverage")
    ]
    
    for metric, display_name in filter_metrics_to_show:
        baseline_val = baseline_filter.get(metric, 0)
        anchors_val = anchors_filter.get(metric, 0)
        anchors_ce_val = anchors_ce_filter.get(metric, 0)
        
        # 개선도 계산 (Baseline 대비)
        best_val = max(baseline_val, anchors_val, anchors_ce_val)
        if best_val == baseline_val:
            improvement = "Baseline"
        elif best_val == anchors_val:
            improvement = "Anchors"
        else:
            improvement = "Anchors+CE"
        
        print(f"{display_name:<15} {baseline_val:<12.4f} {anchors_val:<12.4f} {anchors_ce_val:<12.4f} {improvement:<8}")
    
    print("-" * 80)
    
    # 실행 시간 비교
    print("\n⏱️  실행 시간 비교")
    print("-" * 40)
    baseline_time = baseline_elapsed / 1000
    anchors_time = anchors_elapsed / 1000
    anchors_ce_time = anchors_ce_elapsed / 1000
    
    print(f"Baseline:     {baseline_time:.3f}초")
    print(f"Anchors:      {anchors_time:.3f}초")
    print(f"Anchors+CE:   {anchors_ce_time:.3f}초")
    
    # 시간 비율 계산
    if baseline_time > 0:
        anchors_ratio = anchors_time / baseline_time
        anchors_ce_ratio = anchors_ce_time / baseline_time
        print(f"\n시간 비율 (Baseline 대비):")
        print(f"Anchors:      {anchors_ratio:.1f}x")
        print(f"Anchors+CE:   {anchors_ce_ratio:.1f}x")
    
    # 결론
    print("\n🎯 결론")
    print("-" * 40)
    
    def combo_score(m):
        # 종합 판단 지표: @10과 MRR에 높은 가중치
        return (
            0.35 * m.get("NDCG@10", 0.0) +
            0.35 * m.get("MRR", 0.0) +
            0.15 * m.get("R@10", 0.0) +
            0.15 * m.get("P@10", 0.0)
        )
    
    candidates = [
        ("Baseline",   baseline_metrics),
        ("Anchors",    anchors_metrics),
        ("Anchors+CE", anchors_ce_metrics),
    ]
    best_name, best_metrics = max(candidates, key=lambda kv: combo_score(kv[1]))
    print(f"최고 성능: {best_name} (MRR: {best_metrics.get('MRR', 0.0):.4f})")
    
    if best_name == "Baseline":
        print("💡 앵커/CE가 개선을 만들지 못한 쿼리가 더 많았습니다.")
    elif best_name == "Anchors":
        print("💡 확실한 앵커만 반영하는 방식이 상위 랭킹 품질을 개선했습니다.")
    else:
        print("💡 CE 재정렬이 '좋을 때만' 채택되어 @10/MRR이 개선되었습니다.")
    
    print("="*80)

if __name__ == "__main__":
    run_banking77()