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
ANCHOR_COUNT = 6                     # 앵커 수: 다양성/비용 균형
ANCHOR_MODE = "weighted"             # 'weighted' 또는 'max'
ALPHA = 0.7                          # weighted에서 원쿼리 비중

# CrossEncoder 재정렬
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CE_MAX_LEN = 384
TOP_M = 200                          # CE로 재정렬할 상위 후보 수 (정확도/지연 균형)

# 평가 k
K_LIST = [10, 100]

# 실험 제한 (빠른 테스트용)
MAX_QUERIES = 10  # 최대 10개 쿼리만 실험

# 기타
SEED = 42
OUT_DIR = "results/similarity"
CACHE_DIR = ".cache/similarity"      # Ollama 응답 캐시 디렉토리
SHOW_PROGRESS = False                # SBERT encode progress bar

# =========================
# 유틸
# =========================
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
        f"Generate {num_anchors} short, distinct anchor sentences to retrieve documents for the query below. "
        f"Each anchor must be on its own line, no numbering, no extra text.\n\n"
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
    q: str,
    corpus_texts: List[str],
    base_order: List[int],
) -> List[int]:
    if TOP_M <= 0:
        return base_order
    head = base_order[:TOP_M]
    pairs = [[q, corpus_texts[i]] for i in head]
    scores = ce.predict(pairs)
    head_reranked = [head[i] for i in np.argsort(scores)[::-1]]
    return head_reranked + base_order[TOP_M:]

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

    results_paths = {}

    # ---------------- 1) Baseline ----------------
    print("\n[Run] 1) SBERT + Cosine (Baseline)")
    agg = defaultdict(list)
    t1 = timer_ms()
    for i, (q, lab) in enumerate(zip(query_texts, query_labels)):
        if i >= MAX_QUERIES:
            break
        print(f"[Baseline] Processing query {i+1}/{min(MAX_QUERIES, len(query_texts))}")
        relevant = set(label_idx[lab])
        q_emb = embed_texts(sbert, [q], 1)[0]
        order = rank_biencoder(q_emb, doc_embs)
        m = compute_metrics(order, relevant)
        for k, v in m.items():
            agg[k].append(v)
    res = {
        "meta": {
            "variant": "sbert_cosine",
            "dataset": "banking77",
            "sbert": SBERT_MODEL,
            "k_list": K_LIST,
            "seed": SEED
        },
        "metrics": aggregate_metrics(agg),
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
    t2 = timer_ms()
    for i, (q, lab) in enumerate(zip(query_texts, query_labels)):
        if i >= MAX_QUERIES:
            break
        print(f"[Anchors] Processing query {i+1}/{min(MAX_QUERIES, len(query_texts))}")
        relevant = set(label_idx[lab])
        q_emb = embed_texts(sbert, [q], 1)[0]
        anchors = generate_anchors_ollama(q, ANCHOR_COUNT)
        anchor_embs = embed_texts(sbert, anchors, SBERT_BATCH_SIZE) if anchors else None
        order = rank_with_anchors(q_emb, anchor_embs, doc_embs)
        m = compute_metrics(order, relevant)
        for k, v in m.items():
            agg[k].append(v)
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
            "seed": SEED
        },
        "metrics": aggregate_metrics(agg),
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
    t3 = timer_ms()
    for i, (q, lab) in enumerate(zip(query_texts, query_labels)):
        if i >= MAX_QUERIES:
            break
        print(f"[Re-rank] Processing query {i+1}/{min(MAX_QUERIES, len(query_texts))}")
        relevant = set(label_idx[lab])
        q_emb = embed_texts(sbert, [q], 1)[0]
        anchors = generate_anchors_ollama(q, ANCHOR_COUNT)
        anchor_embs = embed_texts(sbert, anchors, SBERT_BATCH_SIZE) if anchors else None
        order0 = rank_with_anchors(q_emb, anchor_embs, doc_embs)
        order = rerank_crossencoder(ce, q, corpus_texts, order0)
        m = compute_metrics(order, relevant)
        for k, v in m.items():
            agg[k].append(v)
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
            "seed": SEED
        },
        "metrics": aggregate_metrics(agg),
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
            baseline_elapsed = baseline_data.get("elapsed_ms", 0)
        with open(results_paths["anchors"], 'r') as f:
            anchors_data = json.load(f)
            anchors_metrics = anchors_data["metrics"]
            anchors_elapsed = anchors_data.get("elapsed_ms", 0)
        with open(results_paths["anchors_ce"], 'r') as f:
            anchors_ce_data = json.load(f)
            anchors_ce_metrics = anchors_ce_data["metrics"]
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
    best_overall = "Baseline"
    best_score = 0
    
    # MRR을 기준으로 최고 성능 판단
    if anchors_metrics.get("MRR", 0) > baseline_metrics.get("MRR", 0):
        if anchors_ce_metrics.get("MRR", 0) > anchors_metrics.get("MRR", 0):
            best_overall = "Anchors+CE"
            best_score = anchors_ce_metrics.get("MRR", 0)
        else:
            best_overall = "Anchors"
            best_score = anchors_metrics.get("MRR", 0)
    else:
        best_overall = "Baseline"
        best_score = baseline_metrics.get("MRR", 0)
    
    print(f"최고 성능: {best_overall} (MRR: {best_score:.4f})")
    
    if best_overall == "Baseline":
        print("💡 앵커 기반 접근법이 이 데이터셋에서는 성능 향상을 가져오지 못했습니다.")
    else:
        print("💡 앵커 기반 접근법이 성능 향상을 가져왔습니다!")
    
    print("="*80)

if __name__ == "__main__":
    run_banking77()