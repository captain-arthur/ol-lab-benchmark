# run_fiqa.py - FiQA Financial Question Answering Benchmark
import os
import json
import time
from typing import List, Dict, Any, Tuple

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
    "max_queries": 20,           # 기본값: 20
    "split": "test",             # mteb/fiqa는 test 사용
    "out_dir": "results/keyword/fiqa"
}

# 환경변수 오버라이드
try:
    EXPERIMENT_CONFIG["max_queries"] = int(os.getenv("OL_MAX_QUERIES", str(EXPERIMENT_CONFIG["max_queries"])))
except Exception:
    pass

# FiQA 전용 설정
def get_fiqa_config() -> KeywordFilterConfig:
    """FiQA 데이터셋에 최적화된 설정"""
    return KeywordFilterConfig(
        # 기본 설정
        max_queries=EXPERIMENT_CONFIG["max_queries"],
        semantic_sample_rate=float(os.getenv("OL_SEM_SAMPLE", "1.0")),
        
        # FiQA에 최적화된 파라미터
        df_thresh=float(os.getenv("OL_DF_THRESH", "0.80")),    # FiQA 기본값
        idf_min=float(os.getenv("OL_IDF_MIN", "0.1")),         # FiQA 기본값
        max_expanded=int(os.getenv("OL_MAX_EXPANDED", "8")),
        
        # 후보 기반 설정 (FiQA 스타일)
        top_r_pure=200,
        alpha_soft_bonus=float(os.getenv("OL_ALPHA", "0.5")),
        anchor_k=int(os.getenv("OL_ANCHOR_K", "3")),
        rrf_k=float(os.getenv("OL_RRF_K", "60.0")),
        
        # 전체 문서 설정 (MS MARCO 스타일)
        top_r_sem=int(os.getenv("OL_TOP_R_SEM", "1000")),
        expanded_weight=0.35,
        alpha_bonus=0.45,
        enable_safe_drop=False,  # FiQA에서는 SAFE-DROP 비활성화
        
        # 공통 안전장치
        guardrail_k=int(os.getenv("OL_GUARDRAIL_K", "100")),
        enable_prf_fallback=True,
        
        # BM25 파라미터
        k1=float(os.getenv("OL_BM25_K1", "1.5")),
        b=float(os.getenv("OL_BM25_B", "0.75")),
        
        # LLM 설정
        ollama_host=os.getenv("OLLAMA_HOST", "http://192.168.45.166:11434"),
        ollama_model=os.getenv("OLLAMA_MODEL", "gemma3"),
        ollama_timeout=int(os.getenv("OLLAMA_TIMEOUT", "12")),
        ollama_retries=int(os.getenv("OLLAMA_RETRIES", "1")),
    )


# -----------------------------
# FiQA Data Loading
# -----------------------------
def load_fiqa_data(split: str = "test") -> Tuple[List[str], List[Dict[str, Any]], Dict[str, List[int]]]:
    """Returns (documents, queries, qrels_map[query_id] -> list of doc_idx)"""
    print(f"Loading FiQA dataset (split: {split})...")
    queries_ds = load_dataset("mteb/fiqa", "queries", split="queries")
    corpus_ds  = load_dataset("mteb/fiqa", "corpus",  split="corpus")
    qrels_ds   = load_dataset("mteb/fiqa",            split=split)

    documents: List[str] = []
    doc_id_to_idx: Dict[str, int] = {}
    for i, ex in enumerate(corpus_ds):
        t = (ex.get("title") or "").strip()
        body = (ex.get("text") or "").strip()
        doc = (t + " " + body).strip() if t else body
        documents.append(doc)
        doc_id_to_idx[ex["_id"]] = i

    queries: List[Dict[str, Any]] = []
    for ex in queries_ds:
        qid = ex["_id"]
        qtx = ex["text"]
        if qid and qtx:
            queries.append({"query_id": qid, "query": qtx})

    qrels_map: Dict[str, List[int]] = {}
    for ex in qrels_ds:
        qid = ex["query-id"]
        cid = ex["corpus-id"]
        score = ex.get("score", 1)
        if score and score > 0 and cid in doc_id_to_idx:
            qrels_map.setdefault(qid, []).append(doc_id_to_idx[cid])

    # 쿼리 중 정답 없는 케이스 제거(메트릭 분모 안정화)
    queries = [q for q in queries if q["query_id"] in qrels_map and len(qrels_map[q["query_id"]]) > 0]

    print(f"Loaded {len(documents)} docs, {len(queries)} queries (with qrels)")
    return documents, queries, qrels_map


# -----------------------------
# Benchmark Execution
# -----------------------------
def run_benchmark(semantic: bool = False,
                  split: str = "test",
                  max_queries: int = 20,
                  out_dir: str = "results") -> Dict[str, Any]:

    ensure_dir(out_dir)
    documents, queries, qrels_map = load_fiqa_data(split)

    # 쿼리 제한
    if max_queries > 0:
        queries = queries[:max_queries]

    n_queries = len(queries)
    print(f"Running benchmark with {n_queries} queries...")

    # 전코퍼스 BM25 (1회)
    print("Building BM25 index...")
    tokenized_docs = [tokenize_en(doc) for doc in documents]
    bm25 = BM25Okapi(tokenized_docs, k1=1.5, b=0.75)

    # 설정 로드
    config = get_fiqa_config()

    # 캐시 초기화/프리로드
    cache: List[Dict[str, Any]] = []
    cache_path = None
    semantic_data_path = None
    mode = "semantic" if semantic else "pure"
    save_dir = os.path.join(out_dir, mode)
    ensure_dir(save_dir)

    metrics_path = os.path.join(save_dir, "metrics.json")

    if semantic:
        semantic_data_path = os.path.join(save_dir, "semantic_data.jsonl")
        cache_dir = ".cache/keyword/fiqa"
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

    # 메트릭 누적
    metrics = MetricsAccumulator()
    total_pairs = 0
    hits = 0
    misses = 0

    np.random.seed(42)
    t0 = time.time()

    for qi, q in enumerate(queries, 1):
        qid, qtext = q["query_id"], q["query"]
        rel_list = qrels_map.get(qid, [])
        rel_set = set(rel_list)
        if len(rel_set) == 0:
            continue

        # BM25 base scores (전코퍼스 1회)
        q_tokens = tokenize_en(qtext)
        base_scores = bm25.get_scores(q_tokens)

        # semantic 데이터 준비
        sem_data = None
        if semantic and (np.random.rand() < config.semantic_sample_rate):
            metrics.add_sem_attempt()
            k = _key(qtext)
            for item in cache:
                if _key(item.get("query", "")) == k:
                    sem_data = item
                    hits += 1
                    break
            if sem_data is None:
                misses += 1
                sem_data = build_semantic_data_ollama(
                    qtext, 
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
        final_order, final_scores, debug = semantic_rerank(
            qtext, documents, base_scores, sem_data, config
        )

        # 순위별 관련성 계산
        ranked_rel_bin: List[int] = []
        for r in range(min(1000, len(final_order))):
            di = final_order[r]
            rel = 1 if di in rel_set else 0
            ranked_rel_bin.append(rel)

        # 메트릭 누적
        if ranked_rel_bin:
            metrics.add(ranked_rel_bin, len(rel_set), debug.get("semantic_applied", False))
            if debug.get("skipped_by_dfidf", False):
                metrics.add_skipped_by_dfidf()

        total_pairs += min(len(final_order), 1000)

        # semantic jsonl 저장
        if semantic and sem_data and semantic_data_path:
            filtered = debug.get("filtered_terms", {})
            with open(semantic_data_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "query": qtext,
                    "expanded": {"keywords": filtered.get("expanded", [])},
                }, ensure_ascii=False) + "\n")

        # 샘플 로그
        if qi <= 3:
            sem_used = 'Y' if debug.get("semantic_applied", False) else ('-' if not sem_data else 'N')
            print(f"[{mode}] Q{qi} qid={qid} sem_used={sem_used} | rels={len(rel_set)}")

    elapsed = time.time() - t0
    result_metrics = metrics.result()

    # 메타 저장
    meta = {
        "dataset": "mteb/fiqa",
        "split": split,
        "mode": mode,
        "n_queries": n_queries,
        "total_pairs": total_pairs,
        "elapsed_time": elapsed,
        "metrics": result_metrics,
        "config": {
            "max_queries": max_queries,
            "top_r_pure": config.top_r_pure,
            "top_r_sem": config.top_r_sem if semantic else None,
            "k1": config.k1, "b": config.b,
            "anchor_k": config.anchor_k, "rrf_k": config.rrf_k, "guardrail_k": config.guardrail_k
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
    if semantic and cache_path:
        try:
            json.dump(cache, open(cache_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        except Exception:
            pass

    # 요약 출력
    print(f"\n[{mode}] done. n_queries={n_queries}, total_pairs={total_pairs}")
    print(f"  P@1={result_metrics['P@1']:.3f}, P@10={result_metrics['P@10']:.3f}, MRR@10={result_metrics['MRR@10']:.3f}")
    print(f"  nDCG@10={result_metrics['nDCG@10']:.3f}, R@10={result_metrics['R@10']:.3f}, R@100={result_metrics['R@100']:.3f}, time={elapsed:.1f}s")
    if "recall_analysis" in meta:
        ra = meta["recall_analysis"]
        print(f"  Recall Analysis: queries_with_rel={ra['queries_with_rel']}, mean_rel_per_query={ra['mean_rel_per_query']:.2f}")
        print(f"  R@10=0 cases: {ra['queries_r10_zero']}/{ra['queries_with_rel']} ({ra['pct_r10_zero']:.1f}%)")
    if semantic and "semantic_debug" in meta:
        dbg = meta["semantic_debug"]
        print(f"  sem_applied={dbg['sem_applied']}/{dbg['sem_attempted']}, skipped_by_dfidf={dbg['skipped_by_dfidf']}")
        print(f"  cache_hits={hits}, cache_misses={misses}, cache_size={len(cache)}")
    print(f" - Metrics : {metrics_path}")

    return meta


def print_comparison_summary(pure_metrics: Dict[str, Any], semantic_metrics: Dict[str, Any]):
    """순수 BM25 vs 시맨틱 BM25 비교 요약"""
    print("\n" + "="*60)
    print("PURE vs SEMANTIC COMPARISON SUMMARY")
    print("="*60)
    pure_scores = pure_metrics.get("metrics", pure_metrics)
    sem_scores  = semantic_metrics.get("metrics", semantic_metrics)
    def fmt(k):
        a, b = pure_scores[k], sem_scores[k]
        arrow = "↑" if b > a else "↓"
        return f"{a:.3f} → {b:.3f} ({arrow}{abs(b-a):.3f})"
    print(f"P@1     : {fmt('P@1')}")
    print(f"P@10    : {fmt('P@10')}")
    print(f"MRR@10  : {fmt('MRR@10')}")
    print(f"nDCG@10 : {fmt('nDCG@10')}")
    print(f"R@10    : {fmt('R@10')}")
    if "semantic_debug" in semantic_metrics:
        debug = semantic_metrics["semantic_debug"]
        print(f"\nSemantic Rerank Stats: attempted={debug['sem_attempted']}, applied={debug['sem_applied']}, skipped_by_dfidf={debug['skipped_by_dfidf']}")
    print("="*60)


# -----------------------------
# Runner
# -----------------------------
def run_fiqa():
    """FiQA 벤치마크 실행"""
    print("=" * 60)
    print("FiQA Financial Question Answering Benchmark")
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

        print("\nRunning Semantic BM25 (Rerank)...")
        semantic_metrics = run_benchmark(
            semantic=True,
            split=EXPERIMENT_CONFIG["split"],
            max_queries=EXPERIMENT_CONFIG["max_queries"],
            out_dir=EXPERIMENT_CONFIG["out_dir"]
        )

        print_comparison_summary(pure_metrics, semantic_metrics)
        print("\nFiQA benchmark completed successfully!")
        return True

    except Exception as e:
        print(f"\n\nError during FiQA benchmark execution: {e}")
        return False


if __name__ == "__main__":
    run_fiqa()