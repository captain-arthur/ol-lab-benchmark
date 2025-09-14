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
    build_semantic_data_ollama, semantic_rerank, MetricsAccumulator,
    build_global_stats
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
        
        # 후보 기반 설정 (FiQA 스타일) - Recall 중심 튜닝
        top_r_pure=300,  # 200->300: 더 많은 후보 유지
        alpha_soft_bonus=float(os.getenv("OL_ALPHA", "0.3")),  # 0.5->0.3: 보수적 보너스
        anchor_k=int(os.getenv("OL_ANCHOR_K", "5")),  # 3->5: 더 많은 앵커 보호
        rrf_k=float(os.getenv("OL_RRF_K", "40.0")),  # 60->40: RRF 가중치 증가
        
        # 전체 문서 설정 (MS MARCO 스타일) - Recall 중심 튜닝
        top_r_sem=int(os.getenv("OL_TOP_R_SEM", "3000")),  # 1000->3000: 더 많은 후보 유지
        expanded_weight=0.25,  # 0.35->0.25: 보수적 확장
        alpha_bonus=0.3,  # 0.45->0.3: 보수적 보너스
        # SAFE-DROP 완전 제거됨 - FN 최소화를 위해
        
        # 공통 안전장치 (Recall 중심 튜닝)
        guardrail_k=int(os.getenv("OL_GUARDRAIL_K", "200")),  # 100->200: 더 많은 관련 문서 보호
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
                  out_dir: str = "results",
                  expansion_mode: str = "filtered") -> Dict[str, Any]:

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

    # 전역 통계 계산 (개선된 필터링을 위해, 샘플링으로 성능 최적화)
    print("Building global statistics...")
    # 전체 문서의 20% 샘플링으로 통계 계산 (성능 최적화)
    sample_size = min(10000, len(documents) // 5)  # 최대 10,000개 또는 전체의 20%
    if sample_size < len(documents):
        import random
        random.seed(42)  # 재현 가능한 샘플링
        sampled_docs = random.sample(documents, sample_size)
        print(f"Using {sample_size} sampled documents for global statistics")
    else:
        sampled_docs = documents
        print(f"Using all {len(documents)} documents for global statistics")
    
    df_global, idf_global, N_global = build_global_stats(sampled_docs, tokenize_en)

    # 설정 로드
    config = get_fiqa_config()

    # 캐시 초기화/프리로드 (성능 최적화)
    cache: List[Dict[str, Any]] = []
    cache_dict: Dict[str, Dict[str, Any]] = {}  # 빠른 조회를 위한 딕셔너리
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

        # 기존 캐시 로드 (성능 최적화)
        if os.path.exists(cache_path):
            try:
                loaded = json.load(open(cache_path, "r", encoding="utf-8"))
                if isinstance(loaded, list):
                    cache = loaded
                    # 빠른 조회를 위한 딕셔너리 구축 (품질 우선순위)
                    for item in cache:
                        k = _key(item.get("query", ""))
                        if k not in cache_dict:
                            # 첫 번째 엔트리
                            cache_dict[k] = item
                        else:
                            # 기존 엔트리와 품질 비교
                            existing = cache_dict[k]
                            existing_keywords = existing.get("expanded", {}).get("keywords", [])
                            new_keywords = item.get("expanded", {}).get("keywords", [])
                            
                            # 더 많은 키워드를 가진 엔트리 우선
                            if len(new_keywords) > len(existing_keywords):
                                cache_dict[k] = item
                    print(f"Loaded {len(cache)} cached semantic expansions")
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
        
        # 진행상황 로깅
        print(f"\n[{qi}/{n_queries}] Processing query: {qtext[:60]}{'...' if len(qtext) > 60 else ''}")
        print(f"  Query ID: {qid}, Relevant docs: {len(rel_set)}")

        # BM25 base scores (전코퍼스 1회)
        q_tokens = tokenize_en(qtext)
        base_scores = bm25.get_scores(q_tokens)

        # semantic 데이터 준비 (개선된 캐시 로직)
        sem_data = None
        if semantic and (np.random.rand() < config.semantic_sample_rate):
            metrics.add_sem_attempt()
            k = _key(qtext)
            
            # 캐시 조회 (O(1) 딕셔너리 조회)
            if k in cache_dict:
                item = cache_dict[k]
                # 품질 검증: 확장 키워드가 충분한지 확인
                expanded = item.get("expanded", {})
                keywords = expanded.get("keywords", []) if isinstance(expanded, dict) else []
                if len(keywords) >= 3:  # 최소 3개 이상의 확장 키워드 필요
                    sem_data = item
                    hits += 1
                    print(f"  ✓ Cache HIT: {len(keywords)} keywords")
                else:
                    print(f"  ⚠ Cache HIT but low quality: {len(keywords)} keywords")
            else:
                print(f"  ✗ Cache MISS: calling LLM...")
            
            if sem_data is None:
                misses += 1
                sem_data = build_semantic_data_ollama(
                    qtext, 
                    host=config.ollama_host,
                    model=config.ollama_model,
                    timeout=config.ollama_timeout,
                    retries=config.ollama_retries
                )
                
                # 품질 검증 후 캐시에 추가
                if sem_data:
                    expanded = sem_data.get("expanded", {})
                    keywords = expanded.get("keywords", []) if isinstance(expanded, dict) else []
                    if len(keywords) >= 3:  # 품질이 좋은 경우만 캐시에 추가
                        # 중복 체크 후 추가 (O(1) 딕셔너리 조회)
                        k_new = _key(sem_data.get("query", ""))
                        if k_new not in cache_dict:
                            cache.append(sem_data)
                            cache_dict[k_new] = sem_data
                            print(f"  ✓ Added to cache: {len(keywords)} keywords")
                            if cache_path is not None:
                                try:
                                    json.dump(cache, open(cache_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
                                except Exception:
                                    pass
                        else:
                            print(f"  ⚠ Duplicate detected, not cached")
                    else:
                        print(f"  ⚠ Low quality LLM response: {len(keywords)} keywords, not cached")
                else:
                    print(f"  ✗ LLM call failed")
        else:
            print(f"  → Pure BM25 mode (no semantic expansion)")

        # 공통 모듈을 사용한 재랭킹 (전역 통계 전달)
        print(f"  → Reranking with {'semantic' if sem_data else 'pure'} BM25...")
        final_order, final_scores, debug = semantic_rerank(
            qtext, documents, base_scores, sem_data, config, df_global, idf_global, N_global, expansion_mode
        )
        
        # 디버그 정보 출력
        if debug.get("semantic_applied"):
            filtered_terms = debug.get("filtered_terms", {})
            expanded_tokens = filtered_terms.get("expanded_tokens", [])
            print(f"  → Used {len(expanded_tokens)} filtered tokens: {expanded_tokens[:5]}{'...' if len(expanded_tokens) > 5 else ''}")
        else:
            print(f"  → No semantic expansion applied")

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
        
        # 쿼리별 결과 요약
        rel_found = sum(ranked_rel_bin)
        print(f"  → Found {rel_found}/{len(rel_set)} relevant docs")

        # 진행상황 출력 (5개마다)
        if qi % 5 == 0:
            elapsed = time.time() - t0
            cache_hit_rate = hits / (hits + misses) * 100 if (hits + misses) > 0 else 0
            print(f"  📊 Progress: {qi}/{n_queries} queries, {elapsed:.1f}s elapsed, Cache: {cache_hit_rate:.1f}% hit rate")

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


# -----------------------------
# Runner
# -----------------------------
def run_three_way_comparison(max_queries: int = 20):
    """3-way 비교: 순수 BM25 vs 무조건적 확장 vs 정제된 확장"""
    print("=" * 60)
    print("FiQA 3-Way Comparison")
    print("=" * 60)
    print("Configuration:")
    print(f"  - Max queries: {max_queries}")
    print(f"  - Split: {EXPERIMENT_CONFIG['split']}")
    print(f"  - Output directory: {EXPERIMENT_CONFIG['out_dir']}")
    print()

    try:
        results = {}
        
        # 1. Pure BM25 (baseline)
        print("1. Running Pure BM25 (Baseline)...")
        results["pure"] = run_benchmark(
            semantic=False,
            split=EXPERIMENT_CONFIG["split"],
            max_queries=max_queries,
            out_dir=EXPERIMENT_CONFIG["out_dir"]
        )

        # 2. BM25 + 무조건적 키워드 확장
        print("\n2. Running BM25 + 무조건적 키워드 확장...")
        results["unfiltered"] = run_benchmark(
            semantic=True,
            split=EXPERIMENT_CONFIG["split"],
            max_queries=max_queries,
            out_dir=EXPERIMENT_CONFIG["out_dir"],
            expansion_mode="unfiltered"
        )

        # 3. BM25 + 의미 제약·정제된 확장 (제안 기법)
        print("\n3. Running BM25 + 의미 제약·정제된 확장 (제안 기법)...")
        results["filtered"] = run_benchmark(
            semantic=True,
            split=EXPERIMENT_CONFIG["split"],
            max_queries=max_queries,
            out_dir=EXPERIMENT_CONFIG["out_dir"],
            expansion_mode="filtered"
        )

        # 4. 3-way Comparison
        print("\n" + "=" * 60)
        print("3-WAY COMPARISON SUMMARY")
        print("=" * 60)
        
        pure_metrics = results["pure"]["metrics"]
        unfiltered_metrics = results["unfiltered"]["metrics"]
        filtered_metrics = results["filtered"]["metrics"]
        
        print(f"P@1     : {pure_metrics['P@1']:.3f} → {unfiltered_metrics['P@1']:.3f} → {filtered_metrics['P@1']:.3f}")
        print(f"P@10    : {pure_metrics['P@10']:.3f} → {unfiltered_metrics['P@10']:.3f} → {filtered_metrics['P@10']:.3f}")
        print(f"MRR@10  : {pure_metrics['MRR@10']:.3f} → {unfiltered_metrics['MRR@10']:.3f} → {filtered_metrics['MRR@10']:.3f}")
        print(f"nDCG@10 : {pure_metrics['nDCG@10']:.3f} → {unfiltered_metrics['nDCG@10']:.3f} → {filtered_metrics['nDCG@10']:.3f}")
        print(f"R@10    : {pure_metrics['R@10']:.3f} → {unfiltered_metrics['R@10']:.3f} → {filtered_metrics['R@10']:.3f}")
        
        print(f"\nImprovements over Pure BM25:")
        print(f"  무조건적 확장: R@10 {unfiltered_metrics['R@10'] - pure_metrics['R@10']:+.3f}")
        print(f"  정제된 확장:   R@10 {filtered_metrics['R@10'] - pure_metrics['R@10']:+.3f}")
        print("=" * 60)
        
        # 3-way 비교 결과를 consolidated_results.json으로 저장
        consolidated_results = {
            "experiment_info": {
                "dataset": "mteb/fiqa",
                "split": "test",
                "n_queries": max_queries,
                "total_pairs": max_queries * 1000,  # 대략적 추정
                "experiment_date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "description": {
                    "pure": "순수 BM25: 기본적인 정보 검색 모델",
                    "simple": "단순 키워드 확장 BM25: 동의어 기반 확장, 정제 없음",
                    "semantic": "의미적 확장 BM25: LLM 기반 확장, DF/IDF 정제, FN 최소화"
                }
            },
            "core_metrics": {
                "R@100": {
                    "pure": pure_metrics['R@100'],
                    "simple": unfiltered_metrics['R@100'],
                    "semantic": filtered_metrics['R@100'],
                    "improvement_pure_to_semantic": filtered_metrics['R@100'] - pure_metrics['R@100'],
                    "improvement_simple_to_semantic": filtered_metrics['R@100'] - unfiltered_metrics['R@100']
                },
                "R@10": {
                    "pure": pure_metrics['R@10'],
                    "simple": unfiltered_metrics['R@10'],
                    "semantic": filtered_metrics['R@10'],
                    "improvement_pure_to_semantic": filtered_metrics['R@10'] - pure_metrics['R@10'],
                    "improvement_simple_to_semantic": filtered_metrics['R@10'] - unfiltered_metrics['R@10']
                }
            },
            "auxiliary_metrics": {
                "P@10": {
                    "pure": pure_metrics['P@10'],
                    "simple": unfiltered_metrics['P@10'],
                    "semantic": filtered_metrics['P@10']
                },
                "nDCG@10": {
                    "pure": pure_metrics['nDCG@10'],
                    "simple": unfiltered_metrics['nDCG@10'],
                    "semantic": filtered_metrics['nDCG@10']
                },
                "MRR@10": {
                    "pure": pure_metrics['MRR@10'],
                    "simple": unfiltered_metrics['MRR@10'],
                    "semantic": filtered_metrics['MRR@10']
                }
            },
            "reference_metrics": {
                "MAP@100": {
                    "pure": pure_metrics['MAP@100'],
                    "simple": unfiltered_metrics['MAP@100'],
                    "semantic": filtered_metrics['MAP@100']
                },
                "nDCG@100": {
                    "pure": pure_metrics['nDCG@100'],
                    "simple": unfiltered_metrics['nDCG@100'],
                    "semantic": filtered_metrics['nDCG@100']
                }
            },
            "detailed_results": {
                "pure": pure_metrics,
                "simple": unfiltered_metrics,
                "semantic": filtered_metrics
            }
        }
        
        # consolidated_results.json 저장
        consolidated_path = os.path.join(EXPERIMENT_CONFIG["out_dir"], "consolidated_results.json")
        ensure_dir(EXPERIMENT_CONFIG["out_dir"])
        with open(consolidated_path, "w", encoding="utf-8") as f:
            json.dump(consolidated_results, f, ensure_ascii=False, indent=2)
        
        print(f"\nConsolidated results saved to: {consolidated_path}")
        
        return results

    except Exception as e:
        print(f"\n\nError during 3-way comparison: {e}")
        return None


if __name__ == "__main__":
    # 환경변수 기반 설정
    max_queries = EXPERIMENT_CONFIG["max_queries"]
    
    # 항상 3-way 비교 실행
    run_three_way_comparison(max_queries)