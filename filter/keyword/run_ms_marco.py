# run_ms_marco.py - MS MARCO Passage Ranking Benchmark
import os
import json
import time
from typing import List, Dict, Any, Iterable, Tuple

import numpy as np
from datasets import load_dataset
from rank_bm25 import BM25Okapi

# 공통 모듈 import
from k_filter import (
    KeywordFilterConfig, ensure_dir, _key, tokenize_en, 
    build_semantic_data_ollama, semantic_rerank, MetricsAccumulator
)


# -----------------------------
# Configuration
# -----------------------------
EXPERIMENT_CONFIG = {
    "max_queries": 20,
    "split": "validation",        # "validation" or "train"
    "out_dir": "results/keyword/ms_marco"
}
try:
    EXPERIMENT_CONFIG["max_queries"] = int(os.getenv("OL_MAX_QUERIES", str(EXPERIMENT_CONFIG["max_queries"])))
except Exception:
    pass

# MS MARCO 전용 설정
def get_ms_marco_config() -> KeywordFilterConfig:
    """MS MARCO 데이터셋에 최적화된 설정"""
    return KeywordFilterConfig(
        # 기본 설정
        max_queries=EXPERIMENT_CONFIG["max_queries"],
        semantic_sample_rate=1.0,  # 실험 편의상 1.0
        
        # MS MARCO에 최적화된 파라미터
        df_thresh=0.95,  # MS MARCO 기본값
        idf_min=0.05,    # MS MARCO 기본값
        max_expanded=8,
        
        # 후보 기반 설정 (FiQA 스타일) - Recall 중심 튜닝
        top_r_pure=300,  # 200->300: 더 많은 후보 유지
        alpha_soft_bonus=0.3,  # 0.5->0.3: 보수적 보너스
        anchor_k=5,  # 3->5: 더 많은 앵커 보호
        rrf_k=40.0,  # 60->40: RRF 가중치 증가
        
        # 전체 문서 설정 (MS MARCO 스타일) - Recall 중심 튜닝
        top_r_sem=int(os.getenv("OL_TOP_R_SEM", "3000")),  # 2000->3000: 더 많은 후보 유지
        expanded_weight=float(os.getenv("OL_EXP_W", "0.25")),  # 0.35->0.25: 보수적 확장
        alpha_bonus=0.3,  # 0.45->0.3: 보수적 보너스
        # SAFE-DROP 완전 제거됨 - FN 최소화를 위해
        
        # 공통 안전장치 (Recall 중심 튜닝)
        guardrail_k=200,  # 100->200: 더 많은 관련 문서 보호
        enable_prf_fallback=True,
        
        # BM25 파라미터
        k1=1.5,
        b=0.75,
        
        # LLM 설정
        ollama_host=os.getenv("OLLAMA_HOST", "http://192.168.45.166:11434"),
        ollama_model=os.getenv("OLLAMA_MODEL", "gemma3"),
        ollama_timeout=15,
        ollama_retries=2,
    )


# -----------------------------
# Data: MS MARCO (streaming)
# -----------------------------
def iter_ms_marco_query_groups(split: str = "validation",
                               max_queries: int = 20) -> Iterable[Tuple[str, List[str], List[int]]]:
    """MS MARCO 데이터 스트리밍 로더"""
    try:
        ds = load_dataset("ms_marco", "v2.1", split=split, streaming=True)
    except Exception as e:
        raise RuntimeError(f"Failed to load ms_marco v2.1 split='{split}'. "
                           f"Check HF dataset availability/credentials. Error: {e}")
    count = 0
    for item in ds:
        query = item.get("query", "")
        passages = item.get("passages", {})
        if not passages or "passage_text" not in passages or "is_selected" not in passages:
            continue
        p_texts: List[str] = [ (t or "").strip() for t in passages["passage_text"] ]
        p_labels: List[int] = [1 if bool(b) else 0 for b in passages["is_selected"]]

        keep_texts, keep_labels = [], []
        for t, r in zip(p_texts, p_labels):
            if t:
                keep_texts.append(t)
                keep_labels.append(r)
        if not keep_texts:
            continue

        yield query, keep_texts, keep_labels
        count += 1
        if count >= max_queries:
            break


# -----------------------------
# Benchmark Execution
# -----------------------------
def run_benchmark(semantic: bool,
                  split: str = "validation",
                  max_queries: int = 20,
                  out_dir: str = "results") -> Dict[str, Any]:
    """MS MARCO 벤치마크 실행"""
    mode = "semantic" if semantic else "pure"
    save_dir = os.path.join(out_dir, mode)
    ensure_dir(save_dir)

    metrics_path = os.path.join(save_dir, "metrics.json")
    semantic_data_path = None
    cache_path = None
    cache: List[Dict[str, Any]] = []
    hits = misses = 0

    if semantic:
        semantic_data_path = os.path.join(save_dir, "semantic_data.jsonl")
        cache_dir = ".cache/keyword/ms_marco"
        ensure_dir(cache_dir)
        cache_path = os.path.join(cache_dir, "ollama_cache.json")

        # 기존 캐시 로드
        if os.path.exists(cache_path):
            try:
                loaded = json.load(open(cache_path, "r", encoding="utf-8"))
                if isinstance(loaded, list):
                    cache = loaded
            except Exception:
                cache = []

        # 기존 semantic_data.jsonl → 캐시 보강
        if os.path.exists(semantic_data_path):
            with open(semantic_data_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        row = json.loads(line)
                        q = row.get("query", "")
                        if not q:
                            continue
                        ek = []
                        if isinstance(row.get("expanded"), dict):
                            ek = row["expanded"].get("keywords", [])
                        if not ek:
                            ek = row.get("expanded_keywords", [])
                        cache.append({"query": q, "expanded": {"keywords": ek}})
                    except Exception:
                        continue

    # 설정 로드
    config = get_ms_marco_config()
    
    # 메트릭 누적
    metrics = MetricsAccumulator()
    t0 = time.time()
    n_queries = 0
    total_pairs = 0

    # 고정 시드
    np.random.seed(42)

    for q_idx, (query, passages, labels) in enumerate(iter_ms_marco_query_groups(split=split, max_queries=max_queries), start=1):
        n_queries += 1
        total_pairs += len(passages)

        # BM25 인덱스 구축 (쿼리별)
        docs_tokens = [tokenize_en(p) for p in passages]
        bm25 = BM25Okapi(docs_tokens, k1=config.k1, b=config.b)
        
        # BM25 base scores
        q_tokens = tokenize_en(query)
        base_scores = bm25.get_scores(q_tokens)

        # semantic 데이터 준비
        sem_data = None
        if semantic and (np.random.rand() < config.semantic_sample_rate):
            metrics.add_sem_attempt()
            # 캐시 조회
            k = _key(query)
            for item in cache:
                if _key(item.get("query", "")) == k:
                    sem_data = item
                    hits += 1
                    break
            if sem_data is None:
                misses += 1
                sem_data = build_semantic_data_ollama(
                    query,
                    host=config.ollama_host,
                    model=config.ollama_model,
                    timeout=config.ollama_timeout,
                    retries=config.ollama_retries
                )
                if sem_data:
                    cache.append(sem_data)
                    if cache_path is not None:
                        try:
                            json.dump(cache, open(cache_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
                        except Exception:
                            pass

        # 공통 모듈을 사용한 재랭킹
        final_order, final_scores, debug_info = semantic_rerank(
            query, passages, base_scores, sem_data, config
        )
        
        if not final_order:
            continue

        # 최종 라벨 추출
        final_labels = [int(labels[i]) for i in final_order]

        # semantic jsonl 저장
        if semantic and semantic_data_path:
            filtered = debug_info.get("filtered_terms", {})
            entry = {
                "qid": q_idx,
                "query": query,
                "expanded": {"keywords": filtered.get("expanded", [])},
                "recall_guardrail_applied": debug_info.get("recall_guardrail_applied", False),
                "pure_missing_cnt": debug_info.get("pure_missing_cnt", 0)
            }
            with open(semantic_data_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        # 메트릭 누적
        if debug_info.get("skipped_by_dfidf"):
            metrics.add_skipped_by_dfidf()
        metrics.add(final_labels, total_rel_all=sum(labels), sem_applied=debug_info.get("semantic_applied", False))

        # 샘플 로그
        if q_idx <= 3:
            sem_used = 'Y' if debug_info.get("semantic_applied", False) else ('-' if not sem_data else 'N')
            print(f"[{mode}] Q{q_idx} query='{query[:60]}' sem_used={sem_used} "
                  f"top_r={config.top_r_sem if semantic else config.top_r_pure} "
                  f"w_exp={config.expanded_weight}")

    elapsed = time.time() - t0
    result_metrics = metrics.result()

    # 메타 저장
    meta = {
        "model": "BM25 (+semantic rerank, recall-guardrail, PRF-fallback)" if semantic else "BM25 (pure)",
        "bm25_params": {"k1": config.k1, "b": config.b},
        "data": {
            "split": split,
            "max_queries": max_queries,
            "n_queries": n_queries,
            "total_pairs": total_pairs
        },
        "metrics": result_metrics,
        "timing_sec": elapsed,
        "semantic_sample_rate": config.semantic_sample_rate if semantic else 0.0,
        "top_r": config.top_r_sem if semantic else config.top_r_pure,
        "params": {
            "ALPHA": config.alpha_bonus,
            "DF_THRESH": config.df_thresh,
            "IDF_MIN": config.idf_min,
            "MAX_EXPANDED": config.max_expanded,
            "EXPANDED_WEIGHT": config.expanded_weight
        }
    }
    
    if metrics.queries_with_rel > 0:
        meta["recall_analysis"] = {
            "queries_with_rel": metrics.queries_with_rel,
            "total_rel_docs": metrics.total_rel_docs,
            "mean_rel_per_query": metrics.total_rel_docs / metrics.queries_with_rel,
            "queries_r10_zero": metrics.queries_r10_zero,
            "pct_r10_zero": (metrics.queries_r10_zero / metrics.queries_with_rel) * 100.0
        }
    
    if semantic and (metrics.sem_attempted > 0 or metrics.sem_applied > 0 or metrics.skipped_by_dfidf > 0):
        meta["semantic_debug"] = {
            "sem_attempted": metrics.sem_attempted,
            "sem_applied": metrics.sem_applied,
            "skipped_by_dfidf": metrics.skipped_by_dfidf
        }

    ensure_dir(save_dir)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # 캐시 저장
    if semantic and cache_path is not None:
        try:
            json.dump(cache, open(cache_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        except Exception:
            pass

    print(f"\n[{mode}] done.  n_queries={n_queries}, total_pairs={total_pairs}")
    print(f"  P@1={result_metrics['P@1']:.3f}, P@10={result_metrics['P@10']:.3f}, MRR@10={result_metrics['MRR@10']:.3f}")
    print(f"  nDCG@10={result_metrics['nDCG@10']:.3f}, R@10={result_metrics['R@10']:.3f}, R@100={result_metrics['R@100']:.3f}, time={elapsed:.1f}s")
    if metrics.queries_with_rel > 0:
        mean_rel_per_query = metrics.total_rel_docs / metrics.queries_with_rel
        pct_r10_zero = (metrics.queries_r10_zero / metrics.queries_with_rel) * 100
        print(f"  Recall Analysis: queries_with_rel={metrics.queries_with_rel}, mean_rel_per_query={mean_rel_per_query:.2f}")
        print(f"  R@10=0 cases: {metrics.queries_r10_zero}/{metrics.queries_with_rel} ({pct_r10_zero:.1f}%)")
    if semantic:
        dbg = meta.get("semantic_debug", {"sem_applied":0, "sem_attempted":0, "skipped_by_dfidf":0})
        print(f"  sem_applied={dbg['sem_applied']}/{dbg['sem_attempted']}, skipped_by_dfidf={dbg['skipped_by_dfidf']}")
    print(f" - Metrics : {metrics_path}")
    if semantic and semantic_data_path:
        print(f" - Semantic: {semantic_data_path}")
    return meta


def print_comparison_summary(pure_metrics: Dict[str, Any], semantic_metrics: Dict[str, Any]):
    """순수 BM25 vs 시맨틱 BM25 비교 요약"""
    print("\n" + "="*60)
    print("PURE vs SEMANTIC COMPARISON SUMMARY")
    print("="*60)
    pure_scores = pure_metrics.get("metrics", pure_metrics)
    sem_scores  = semantic_metrics.get("metrics", semantic_metrics)
    def line(k):
        a, b = pure_scores[k], sem_scores[k]
        return f"{a:.3f} → {b:.3f} ({'↑' if b>a else '↓'}{abs(b-a):.3f})"
    print(f"P@1     : {line('P@1')}")
    print(f"P@10    : {line('P@10')}")
    print(f"MRR@10  : {line('MRR@10')}")
    print(f"nDCG@10 : {line('nDCG@10')}")
    print(f"R@10    : {line('R@10')}")
    print(f"R@100   : {line('R@100')}")
    print("="*60)


# -----------------------------
# Top-level Runner
# -----------------------------
def run_ms_marco():
    """MS MARCO 벤치마크 실행"""
    print("=" * 60)
    print("MS MARCO Passage Ranking Benchmark")
    print("=" * 60)
    print("Configuration:")
    print(f"  - Max queries: {EXPERIMENT_CONFIG['max_queries']}")
    print(f"  - Split: {EXPERIMENT_CONFIG['split']}")
    print(f"  - Output directory: {EXPERIMENT_CONFIG['out_dir']}")
    print()
    
    try:
        print("Running Pure BM25...")
        pure_metrics = run_benchmark(
            semantic=False,
            split=EXPERIMENT_CONFIG["split"],
            max_queries=EXPERIMENT_CONFIG["max_queries"],
            out_dir=EXPERIMENT_CONFIG["out_dir"]
        )

        print("\nRunning Semantic BM25 (Recall-first + guardrail + PRF fallback)...")
        semantic_metrics = run_benchmark(
            semantic=True,
            split=EXPERIMENT_CONFIG["split"],
            max_queries=EXPERIMENT_CONFIG["max_queries"],
            out_dir=EXPERIMENT_CONFIG["out_dir"]
        )

        print_comparison_summary(pure_metrics, semantic_metrics)
        print("\nMS MARCO benchmark completed successfully!")
        return True
    except Exception as e:
        print(f"\n\nError during MS MARCO benchmark execution: {e}")
        return False


if __name__ == "__main__":
    run_ms_marco()