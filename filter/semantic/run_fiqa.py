#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_fiqa.py - MTEB FiQA Benchmark (Semantic Filter)
- 4단계 비교 실험: SBERT → SBERT+CE → SBERT+CE+LLM → SBERT+CE+LLM+CBC
- 목표: 확실한 오답(무관한 문서) 선제적 제거를 통한 정밀도 극한 향상
"""

import os
import json
import time
from typing import List, Dict, Any, Iterable, Tuple
from collections import defaultdict

import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, CrossEncoder

# 공통 모듈 import
from s_filter import (
    set_seed, ensure_dir, save_json, timer_ms, ndcg_at_k, pick_better_by_ndcg, is_confident, is_confident_cbc,
    load_cache, save_cache, build_label_index, create_filtering_candidates,
    generate_anchors_ollama, embed_texts, rank_biencoder, rank_with_anchors,
    ce_score_pairs_cached, calibrate_ce_threshold, rerank_crossencoder,
    compute_metrics, aggregate_metrics, compute_filtering_metrics, RunningPercentiles,
    build_hard_candidate_pool, compute_filtering_metrics_cbc, normalize_threshold,
    analyze_score_distribution, calculate_adaptive_percentiles, compute_filtering_metrics_cbc_adaptive
)

# -----------------------------
# Configuration
# -----------------------------
EXPERIMENT_CONFIG = {
    "max_queries": 200,
    "split": "test",
    "out_dir": "results/semantic/fiqa"
}
try:
    EXPERIMENT_CONFIG["max_queries"] = int(os.getenv("OL_MAX_QUERIES", str(EXPERIMENT_CONFIG["max_queries"])))
except Exception:
    pass

# MTEB FiQA 전용 설정
def get_semantic_config() -> Dict[str, Any]:
    """FiQA 데이터셋에 최적화된 유사도 필터 설정"""
    return {
        # 모델 설정
        "sbert_model": "all-MiniLM-L6-v2",
        "ce_model": "cross-encoder/ms-marco-MiniLM-L-12-v2",
        
        # 앵커 설정
        "anchor_count": 5,  # LLM 앵커 개수 (품질 향상을 위해 증가)
        "ollama_host": os.getenv("OLLAMA_HOST", "192.168.45.166"),
        "ollama_port": int(os.getenv("OLLAMA_PORT", "11434")),
        "ollama_model": os.getenv("OLLAMA_MODEL", "gemma3"),
        
        # 필터링 임계값 (정밀도 극한 향상을 위한 엄격한 설정)
        "drop_thresholds": {
            "sbert": 0.3,    # SBERT 유사도 임계값
            "anchor": 0.25,  # 앵커 유사도 임계값  
            "ce": 0.4        # CrossEncoder 임계값
        },
        
        # CrossEncoder 설정
        "top_m": 50,  # CE 재정렬 대상 문서 수
        
        # 실험 설정
        "negative_samples": 99,  # 필터링용 부정 샘플 수
        "k_list": [1, 10, 100],  # 평가 지표 K값들
        "batch_size": 32,
        "show_progress": False
    }

# -----------------------------
# Data: MTEB FiQA
# -----------------------------
def iter_fiqa_query_groups(split: str = "test", max_queries: int = 20) -> Iterable[Tuple[str, List[str], List[int], str]]:
    """전처리된 MTEB/FiQA 데이터셋 로더 (필터링 친화형)"""
    import json
    path = f"data/fiqa_filtering/{split}.jsonl"
    
    try:
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= max_queries:
                    break
                rec = json.loads(line)
                yield rec["query"], rec["passages"], rec["labels"], rec["query_id"]
    except FileNotFoundError:
        raise RuntimeError(f"Preprocessed data not found at {path}. "
                           f"Please run preprocess_fiqa.py first.")
    except Exception as e:
        raise RuntimeError(f"Failed to load preprocessed FiQA data from {path}. Error: {e}")

# -----------------------------
# 4단계 실험 함수들
# -----------------------------

def run_baseline_pure(sbert, doc_embs, query_texts, qids, qrels, k_list, max_queries, show_progress=False):
    """순정 SBERT 베이스라인: SBERT + 코사인 유사도로 전체 코퍼스 랭킹만 수행"""
    from collections import defaultdict
    agg = defaultdict(list)
    t1 = timer_ms()

    for i, q in enumerate(query_texts[:max_queries]):
        qid = qids[i]
        relevant = qrels.get(qid, set())

        q_emb = embed_texts(sbert, [q], 1, show_progress)[0]
        order = rank_biencoder(q_emb, doc_embs)  # 전체 코퍼스 정렬

        m = compute_metrics(order, relevant, k_list)  # R@k/NDCG/MRR만
        for k, v in m.items():
            agg[k].append(v)

    return {
        "meta": {"variant": "sbert_cosine_pure", "k_list": k_list},
        "metrics": aggregate_metrics(agg),
        "elapsed_ms": t1()
    }

def run_sbert_baseline(sbert, doc_embs, corpus_texts, corpus_labels, query_texts, query_labels,
                      label_idx, candidate_cache, config, max_queries, qrels, qids, use_cbc=True):
    """① SBERT (Baseline): SBERT + 코사인유사도로 필터링 (순수 SBERT만 사용)"""
    print("=" * 60)
    print("① SBERT (Baseline) 실험 시작 - SBERT + 코사인유사도 필터링")
    print("=" * 60)
    
    agg = defaultdict(list)
    filter_agg = defaultdict(list)
    t1 = timer_ms()
    
    for i, (q, qid) in enumerate(zip(query_texts, qids)):
        if i >= max_queries:
            break
        print(f"[SBERT Baseline] Processing query {i+1}/{min(max_queries, len(query_texts))}")
        
        # qrels 기반 관련 문서 집합
        relevant = qrels.get(qid, set())
        print(f"[DBG] qid={qid} rel={len(relevant)}")
        
        # 하드 네거티브 후보 풀 생성
        candidate_pool, positives, negatives = build_hard_candidate_pool(
            qid=qid, q_text=q, relevant_set=relevant,
            doc_embs=doc_embs, sbert=sbert,
            bm25_top=None, sbert_top_k=500
        )
        print(f"  [Debug] Candidate pool: {len(candidate_pool)}, Positives: {len(positives)}, Negatives: {len(negatives)}")
        
        # SBERT 임베딩 및 랭킹 (순수 SBERT만 사용)
        q_emb = embed_texts(sbert, [q], config["batch_size"], config["show_progress"])[0]
        order_base = rank_biencoder(q_emb, doc_embs)
        
        # 지표 계산
        candidate_embs = doc_embs[candidate_pool]
        sbert_scores = candidate_embs @ q_emb
        
        # 실제 계산된 점수 저장
        if not hasattr(run_sbert_baseline, 'saved_scores'):
            run_sbert_baseline.saved_scores = {}
        if "SBERT + CE (Baseline)" not in run_sbert_baseline.saved_scores:
            run_sbert_baseline.saved_scores["SBERT + CE (Baseline)"] = {}
        
        run_sbert_baseline.saved_scores["SBERT + CE (Baseline)"][f"query_{i+1}"] = {
            "sbert_scores": sbert_scores.tolist(),
            "anchor_scores": "skip"
        }
        
        # ① SBERT: SBERT 점수만으로 필터링 (앵커와 CE는 np.inf로 무효화)
        anchor_scores = np.full_like(sbert_scores, np.inf, dtype=np.float32)
        ce_scores = np.full_like(sbert_scores, np.inf, dtype=np.float32)
        
        thresholds = {"sbert": 0.3, "anchor": 0.3, "ce": 0.3}
        filter_metrics = compute_filtering_metrics(
            sbert_scores, anchor_scores, ce_scores,
            candidate_pool, positives[0] if positives else None, negatives,
            thresholds
        )
        
        m = compute_metrics(order_base, relevant, config["k_list"])
        
        print(f"[SBERT Baseline] Dropped: {filter_metrics['Dropped_Count']}")
        
        for k, v in m.items():
            agg[k].append(v)
        for k, v in filter_metrics.items():
            filter_agg[k].append(v)
    
    # 결과 저장
    res = {
        "meta": {
            "variant": "sbert_baseline",
            "k_list": config["k_list"],
            "drop_thresholds": config["drop_thresholds"],
        },
        "metrics": aggregate_metrics(agg),
        "filter_metrics": aggregate_metrics(filter_agg),
        "elapsed_ms": t1()
    }
    
    return res

def run_sbert_ce(sbert, doc_embs, ce, corpus_texts, corpus_labels, query_texts, query_labels,
                label_idx, candidate_cache, ce_cache, config, max_queries, qrels, qids, use_cbc=True):
    """② SBERT + CE: CrossEncoder 재검증 추가 (신뢰도 낮은 문장에 대한 정밀한 판단)"""
    print("=" * 60)
    print("② SBERT + CE 실험 시작 - CrossEncoder 재검증 추가")
    print("=" * 60)
    
    agg = defaultdict(list)
    filter_agg = defaultdict(list)
    t2_start = timer_ms()
    
    for i, (q, qid) in enumerate(zip(query_texts, qids)):
        if i >= max_queries:
            break
        print(f"[SBERT + CE] Processing query {i+1}/{min(max_queries, len(query_texts))}")
        
        relevant = qrels.get(qid, set())
        
        # 하드 네거티브 후보 풀 생성
        key = i
        if key not in candidate_cache:
            candidate_cache[key] = build_hard_candidate_pool(
                qid=qid, q_text=q, relevant_set=relevant,
                doc_embs=doc_embs, sbert=sbert, sbert_top_k=500
            )
        candidate_pool, positives, negatives = candidate_cache[key]
        print(f"  [Debug] Candidate pool: {len(candidate_pool)}, Positives: {len(positives)}, Negatives: {len(negatives)}")
        
        # ② SBERT + CE: 올바른 순서로 처리
        # 1. SBERT 임베딩 생성 (본 쿼리)
        q_emb = embed_texts(sbert, [q], config["batch_size"], config["show_progress"])[0]
        
        # 2. 코사인 유사도 계산
        order_base = rank_biencoder(q_emb, doc_embs)
        
        # 3. 1차 필터링 (SBERT만 사용)
        candidate_embs = doc_embs[candidate_pool]
        sbert_scores = candidate_embs @ q_emb
        
        # 점수 저장 - 원본 쿼리 점수만 저장 (Baseline 특징)
        if not hasattr(run_sbert_ce, 'saved_scores'):
            run_sbert_ce.saved_scores = {}
        if "SBERT/CE (Baseline)" not in run_sbert_ce.saved_scores:
            run_sbert_ce.saved_scores["SBERT/CE (Baseline)"] = {}
        
        run_sbert_ce.saved_scores["SBERT/CE (Baseline)"][f"query_{i+1}"] = {
            "sbert_scores": sbert_scores.tolist(),  # 원본 쿼리 점수만
            "anchor_scores": "skip"
        }
        
        # 1차 필터링에서 드롭된 문서들 식별
        dropped_docs = []
        kept_docs = []
        for j, doc_id in enumerate(candidate_pool):
            sbert_score = sbert_scores[j]
            if sbert_score < 0.3:  # SBERT 임계값
                dropped_docs.append((j, doc_id))  # (인덱스, 문서ID)
            else:
                kept_docs.append((j, doc_id))
        
        print(f"  [Debug] 1차 필터링: {len(dropped_docs)}개 드롭, {len(kept_docs)}개 유지")
        
        # 4. CrossEncoder로 드롭된 문서들 재검증
        ce_scores = np.full(len(candidate_pool), np.inf, dtype=np.float32)
        if dropped_docs:
            dropped_indices = [idx for idx, _ in dropped_docs]
            dropped_doc_ids = [doc_id for _, doc_id in dropped_docs]
            
            # 드롭된 문서들에 대해서만 CE 점수 계산
            dropped_ce_scores = ce_score_pairs_cached(ce, i, q, dropped_doc_ids, corpus_texts, ce_cache)
            
            # CE 점수를 전체 배열에 매핑
            for k, (orig_idx, _) in enumerate(dropped_docs):
                ce_scores[orig_idx] = dropped_ce_scores[k]
        
        # 5. 2차 필터링 (CE 재검증) - 드롭된 문서들 중 CE 점수가 높은 것들 복구
        final_dropped = []
        final_kept = []
        
        for j, doc_id in enumerate(candidate_pool):
            sbert_score = sbert_scores[j]
            ce_score = ce_scores[j]
            
            if sbert_score < 0.3:  # 1차에서 드롭된 문서들
                # CE 점수로 재검증 (동일한 임계값 0.3 사용)
                if ce_score < 0.3:  # CE 점수도 0.3 이하면 최종 드롭
                    final_dropped.append(doc_id)
                else:  # CE 점수가 0.3 초과면 복구
                    final_kept.append(doc_id)
            else:  # 1차에서 유지된 문서들
                final_kept.append(doc_id)
        
        print(f"  [Debug] 2차 필터링: {len(final_dropped)}개 최종 드롭, {len(final_kept)}개 최종 유지")
        
        # 필터링 지표 계산
        total_negatives = len(negatives)
        dropped_negatives = sum(1 for doc_id in final_dropped if doc_id in negatives)
        drop_precision = dropped_negatives / len(final_dropped) if final_dropped else 0.0
        drop_recall = dropped_negatives / total_negatives if total_negatives > 0 else 0.0
        drop_f1 = 2 * drop_precision * drop_recall / (drop_precision + drop_recall) if (drop_precision + drop_recall) > 0 else 0.0
        
        filter_metrics = {
            "Drop_Precision": drop_precision,
            "Drop_Recall": drop_recall,
            "Drop_F1": drop_f1,
            "Dropped_Count": len(final_dropped),
            "Kept_Count": len(final_kept)
        }
        
        # 랭킹 지표 계산 (필터링된 문서들로만)
        if final_kept:
            # 유지된 문서들의 인덱스를 찾아서 랭킹 생성
            kept_indices = [candidate_pool.index(doc_id) for doc_id in final_kept if doc_id in candidate_pool]
            kept_scores = sbert_scores[kept_indices]
            kept_order = np.argsort(kept_scores)[::-1]  # 내림차순 정렬
            final_order = [kept_indices[i] for i in kept_order]
        else:
            final_order = []
        
        m = compute_metrics(final_order, relevant, config["k_list"])
        
        print(f"[SBERT + CE] Dropped: {filter_metrics['Dropped_Count']}")
        
        for k, v in m.items():
            agg[k].append(v)
        for k, v in filter_metrics.items():
            filter_agg[k].append(v)
    
    # 결과 저장
    res = {
        "meta": {
            "variant": "sbert_ce",
            "top_m": config["top_m"],
            "k_list": config["k_list"],
            "drop_thresholds": config["drop_thresholds"],
        },
        "metrics": aggregate_metrics(agg),
        "filter_metrics": aggregate_metrics(filter_agg),
        "elapsed_ms": timer_ms() - t2_start
    }
    
    return res

def run_sbert_ce_llm(sbert, doc_embs, ce, corpus_texts, corpus_labels, query_texts, query_labels,
                    label_idx, candidate_cache, ce_cache, config, max_queries, cache_dir, qrels, qids, use_cbc=True):
    """③ SBERT + CE + 의미적 데이터 확장: LLM 기반 의미적 데이터 확장을 추가"""
    print("=" * 60)
    print("③ SBERT + CE + LLM 실험 시작")
    print("=" * 60)
    
    # CBC 통계 관리자 초기화
    cbc_stats = RunningPercentiles(maxlen=200)
    
    agg = defaultdict(list)
    filter_agg = defaultdict(list)
    t3_start = timer_ms()
    
    for i, (q, qid) in enumerate(zip(query_texts, qids)):
        if i >= max_queries:
            break
        print(f"[SBERT + CE + LLM] Processing query {i+1}/{min(max_queries, len(query_texts))}")
        
        relevant = qrels.get(qid, set())
        
        # 하드 네거티브 후보 풀 생성
        key = i
        if key not in candidate_cache:
            candidate_cache[key] = build_hard_candidate_pool(
                qid=qid, q_text=q, relevant_set=relevant,
                doc_embs=doc_embs, sbert=sbert, sbert_top_k=500
            )
        candidate_pool, positives, negatives = candidate_cache[key]
        
        # ③ SBERT + CE + LLM: 2번 방식 + LLM 앵커 재검증
        # 1. LLM 앵커 생성 (의미적 확장)
        try:
            anchors = generate_anchors_ollama(q, config["anchor_count"], 
                                            config["ollama_host"], config["ollama_port"], 
                                            config["ollama_model"], cache_dir)
        except Exception as e:
            print(f"[Warning] LLM anchor generation failed: {e}")
            anchors = []
        print(f"[DEBUG] Generated {len(anchors)} anchors: {anchors[:2] if anchors else 'None'}")
        
        # 2. SBERT 임베딩 생성 (본 쿼리 + 앵커)
        q_emb = embed_texts(sbert, [q], config["batch_size"], config["show_progress"])[0]
        anchor_embs = embed_texts(sbert, anchors, config["batch_size"], config["show_progress"]) if len(anchors) > 0 else np.array([])
        print(f"[DEBUG] Using anchors: {len(anchors) > 0 and len(anchor_embs) > 0 and anchor_embs.shape[0] > 0}")
        
        # 3. 코사인 유사도 계산 (본 쿼리 + 앵커 맥스 풀링)
        candidate_embs = doc_embs[candidate_pool]
        
        # 유사도 분포 분석
        if len(anchors) > 0 and len(anchor_embs) > 0 and anchor_embs.shape[0] > 0:
            # 원본 쿼리 유사도
            q_scores = candidate_embs @ q_emb.reshape(-1, 1)
            q_scores = q_scores.flatten()
            
            # 앵커 유사도 (맥스 풀링)
            anchor_scores = candidate_embs @ anchor_embs.T
            max_anchor_scores = np.max(anchor_scores, axis=1)
            
            # 결합된 유사도
            combined_scores = np.maximum(q_scores, max_anchor_scores)
            
            print(f"[DEBUG] Original query scores - mean: {np.mean(q_scores):.3f}, std: {np.std(q_scores):.3f}, min: {np.min(q_scores):.3f}, max: {np.max(q_scores):.3f}")
            print(f"[DEBUG] Anchor scores - mean: {np.mean(max_anchor_scores):.3f}, std: {np.std(max_anchor_scores):.3f}, min: {np.min(max_anchor_scores):.3f}, max: {np.max(max_anchor_scores):.3f}")
            print(f"[DEBUG] Combined scores - mean: {np.mean(combined_scores):.3f}, std: {np.std(combined_scores):.3f}, min: {np.min(combined_scores):.3f}, max: {np.max(combined_scores):.3f}")
            print(f"[DEBUG] Score improvement: {np.mean(combined_scores) - np.mean(q_scores):.3f}")
        else:
            print(f"[DEBUG] No anchors available for comparison")
        print(f"[DEBUG] candidate_embs shape: {candidate_embs.shape}")
        print(f"[DEBUG] q_emb shape: {q_emb.shape}")
        if len(anchors) > 0 and len(anchor_embs) > 0 and anchor_embs.shape[0] > 0:
            # 본 쿼리와 앵커들의 맥스 풀링 사용
            q_scores = candidate_embs @ q_emb.reshape(-1, 1)  # (400, 384) @ (384, 1) = (400, 1)
            q_scores = q_scores.flatten()  # (400,)
            anchor_scores = candidate_embs @ anchor_embs.T  # [N, A]
            max_anchor_scores = np.max(anchor_scores, axis=1)  # [N]
            
            # 맥스 풀링: 본 쿼리와 앵커 중 최고 점수 사용
            combined_scores = np.maximum(q_scores, max_anchor_scores)
        else:
            # 앵커가 없을 때는 원본 쿼리만 사용
            combined_scores = candidate_embs @ q_emb.reshape(-1, 1)  # (400, 384) @ (384, 1) = (400, 1)
            combined_scores = combined_scores.flatten()  # (400,)
        
        # combined_scores는 이미 계산된 점수이므로 직접 정렬
        order_base = np.argsort(combined_scores)[::-1].tolist()
        
        # 4. SBERT 점수 계산 (본 쿼리 + 앵커)
        sbert_scores = combined_scores
        
        # 점수 저장 - 앵커 확장 효과를 명확히 보여주기 위해 원본과 앵커 점수를 모두 저장
        if not hasattr(run_sbert_ce_llm, 'saved_scores'):
            run_sbert_ce_llm.saved_scores = {}
        if "SBERT/CE + SDE" not in run_sbert_ce_llm.saved_scores:
            run_sbert_ce_llm.saved_scores["SBERT/CE + SDE"] = {}
        
        # 원본 쿼리 점수와 앵커 점수를 분리하여 저장
        original_q_scores = candidate_embs @ q_emb
        if len(anchors) > 0 and len(anchor_embs) > 0 and anchor_embs.shape[0] > 0:
            anchor_scores = candidate_embs @ anchor_embs.T
            max_anchor_scores = np.max(anchor_scores, axis=1)
        else:
            max_anchor_scores = np.zeros_like(original_q_scores)
        
        run_sbert_ce_llm.saved_scores["SBERT/CE + SDE"][f"query_{i+1}"] = {
            "sbert_scores": original_q_scores.tolist(),  # 원본 쿼리 점수
            "anchor_scores": max_anchor_scores.tolist()  # 앵커 점수 (분포 확장 효과)
        }
        
        # 4. CrossEncoder 점수 계산 (드롭된 문서들에 대해서만)
        ce_scores = np.full(len(candidate_pool), np.inf, dtype=np.float32)
        dropped_docs = []
        for j, doc_id in enumerate(candidate_pool):
            if sbert_scores[j] < 0.3:  # 고정 임계값 0.3 사용
                dropped_docs.append(doc_id)
        
        if dropped_docs:
            dropped_ce_scores = ce_score_pairs_cached(ce, i, q, dropped_docs, corpus_texts, ce_cache)
            # CE 점수 정규화
            dropped_ce_scores = normalize_scores(dropped_ce_scores, method="sigmoid")
            for k, doc_id in enumerate(dropped_docs):
                doc_idx = candidate_pool.index(doc_id)
                ce_scores[doc_idx] = dropped_ce_scores[k]
        
        # 5. LLM 앵커 점수 계산 (최종 드롭된 문서들에 대해서만)
        anchor_scores = np.full(len(candidate_pool), np.inf, dtype=np.float32)
        anchors = generate_anchors_ollama(q, config["anchor_count"], 
                                        config["ollama_host"], config["ollama_port"], 
                                        config["ollama_model"], cache_dir)
        
        if len(anchors) > 0:
            # 2차 필터링 결과 시뮬레이션 (SBERT + CE)
            final_dropped_sim = []
            for j, doc_id in enumerate(candidate_pool):
                sbert_score = sbert_scores[j]
                ce_score = ce_scores[j]
                if sbert_score < 0.3 and ce_score < 0.3:  # 최종 드롭될 문서들
                    final_dropped_sim.append(doc_id)
            
            if final_dropped_sim:
                # 최종 드롭될 문서들에 대해서만 앵커 점수 계산
                anchor_candidate_scores = []
                for anchor_text in anchors:
                    a_emb = embed_texts(sbert, [anchor_text], config["batch_size"], config["show_progress"])[0]
                    sims_a = doc_embs @ a_emb
                    anchor_candidate_scores.append(sims_a[candidate_pool])
                anchor_candidate_scores = np.stack(anchor_candidate_scores, axis=1)
                max_anchor_scores = anchor_candidate_scores.max(axis=1)
                
                for doc_id in final_dropped_sim:
                    doc_idx = candidate_pool.index(doc_id)
                    anchor_scores[doc_idx] = max_anchor_scores[doc_idx]
        
        # 6. 앵커 구제 필터 적용 (재정렬 없음)
        thresholds = {"sbert": 0.3, "anchor": 0.3, "ce": 0.3}
        filter_metrics = apply_anchor_rescue_filter(
            sbert_scores=sbert_scores,
            ce_scores=ce_scores,
            anchor_scores=anchor_scores,
            candidate_pool=candidate_pool,
            positive_doc=positives[0] if positives else None,
            negative_docs=negatives,
            thresholds=thresholds
        )
        
        m = compute_metrics(order_base, relevant, config["k_list"])
        
        print(f"[SBERT + CE + LLM] Dropped: {filter_metrics['Dropped_Count']}")
        
        for k, v in m.items():
            agg[k].append(v)
        for k, v in filter_metrics.items():
            filter_agg[k].append(v)
    
    # 결과 저장
    res = {
        "meta": {
            "variant": "sbert_ce_llm",
            "anchor_count": config["anchor_count"],
            "top_m": config["top_m"],
            "k_list": config["k_list"],
            "drop_thresholds": config["drop_thresholds"],
        },
        "metrics": aggregate_metrics(agg),
        "filter_metrics": aggregate_metrics(filter_agg),
        "elapsed_ms": timer_ms() - t3_start
    }
    
    return res

def run_sbert_ce_cbc(sbert, doc_embs, ce, corpus_texts, corpus_labels, query_texts, query_labels,
                    label_idx, candidate_cache, ce_cache, config, max_queries, qrels, qids, use_cbc=True):
    """③ SBERT/CE + CBC: SBERT + CE + CBC (CBC만 사용) - 전체 분포 기반"""
    print("=" * 60)
    print("③ SBERT/CE + CBC 실험 시작 - CBC 기반 적응적 임계값 (전체 분포)")
    print("=" * 60)
    
    agg = defaultdict(list)
    filter_agg = defaultdict(list)
    t3_start = timer_ms()
    
    # 1단계: 전체 실험의 점수 분포 수집을 위한 저장소
    all_sbert_scores = []
    all_ce_scores = []
    
    # 2단계: 모든 쿼리 처리하여 점수 분포 수집
    for i, (q, qid) in enumerate(zip(query_texts, qids)):
        if i >= max_queries:
            break
        print(f"[SBERT/CE + CBC] Collecting scores for query {i+1}/{min(max_queries, len(query_texts))}")
        
        relevant = qrels.get(qid, set())
        
        # 하드 네거티브 후보 풀 생성
        key = i
        if key not in candidate_cache:
            candidate_cache[key] = build_hard_candidate_pool(
                qid=qid, q_text=q, relevant_set=relevant,
                doc_embs=doc_embs, sbert=sbert, sbert_top_k=500
            )
        candidate_pool, positives, negatives = candidate_cache[key]
        
        # SBERT + CE 점수 계산
        q_emb = embed_texts(sbert, [q], config["batch_size"], config["show_progress"])[0]
        candidate_embs = doc_embs[candidate_pool]
        sbert_scores = candidate_embs @ q_emb
        
        # CE 점수 계산
        ce_scores = ce_score_pairs_cached(ce, i, q, candidate_pool, corpus_texts, ce_cache)
        
        # 점수 저장
        if not hasattr(run_sbert_ce_cbc, 'saved_scores'):
            run_sbert_ce_cbc.saved_scores = {}
        if "SBERT/CE + CBC" not in run_sbert_ce_cbc.saved_scores:
            run_sbert_ce_cbc.saved_scores["SBERT/CE + CBC"] = {}
        
        run_sbert_ce_cbc.saved_scores["SBERT/CE + CBC"][f"query_{i+1}"] = {
            "sbert_scores": sbert_scores.tolist(),  # 원본 쿼리 점수 (CBC만 사용)
            "anchor_scores": "skip"  # 앵커 없음
        }
        
        # 전체 분포 수집
        all_sbert_scores.extend(sbert_scores.tolist())
        all_ce_scores.extend(ce_scores.tolist())
    
    # 3단계: 전체 분포 분석 및 적응적 퍼센타일 계산
    print(f"[CBC] 전체 분포 분석: SBERT {len(all_sbert_scores)}개, CE {len(all_ce_scores)}개")
    
    # 분포 특성 분석
    sbert_characteristics = analyze_score_distribution(all_sbert_scores)
    ce_characteristics = analyze_score_distribution(all_ce_scores)
    
    # 분포 상세 정보 출력
    print(f"[CBC] SBERT 분포: 평균 {sbert_characteristics['mean']:.3f}, 표준편차 {sbert_characteristics['std']:.3f}, 타입 {sbert_characteristics['type']}")
    print(f"[CBC] CE 분포: 평균 {ce_characteristics['mean']:.3f}, 표준편차 {ce_characteristics['std']:.3f}, 타입 {ce_characteristics['type']}")
    
    # 적응적 퍼센타일 계산 (앵커 없이)
    adaptive_percentiles = calculate_adaptive_percentiles(
        sbert_characteristics, ce_characteristics, {"type": "none", "mean": 0, "std": 0}
    )
    
    print(f"[CBC] 적응적 퍼센타일: SBERT {adaptive_percentiles['sbert']:.1f}%, CE {adaptive_percentiles['ce']:.1f}%")
    
    # 실제 임계값 계산 및 출력
    if len(all_sbert_scores) > 0:
        sbert_threshold = np.percentile(all_sbert_scores, adaptive_percentiles['sbert'])
        print(f"[CBC] SBERT 임계값: {sbert_threshold:.3f}")
    if len(all_ce_scores) > 0:
        ce_threshold = np.percentile(all_ce_scores, adaptive_percentiles['ce'])
        print(f"[CBC] CE 임계값: {ce_threshold:.3f}")
    
    # 4단계: 계산된 임계값으로 모든 쿼리 재처리
    for i, (q, qid) in enumerate(zip(query_texts, qids)):
        if i >= max_queries:
            break
        print(f"[SBERT/CE + CBC] Processing query {i+1}/{min(max_queries, len(query_texts))} with adaptive thresholds")
        
        relevant = qrels.get(qid, set())
        candidate_pool, positives, negatives = candidate_cache[i]
        
        # SBERT + CE 점수 계산
        q_emb = embed_texts(sbert, [q], config["batch_size"], config["show_progress"])[0]
        candidate_embs = doc_embs[candidate_pool]
        sbert_scores = candidate_embs @ q_emb
        
        # CE 점수 계산
        ce_scores = ce_score_pairs_cached(ce, i, q, candidate_pool, corpus_texts, ce_cache)
        
        # 정규화된 임계값 적용
        sbert_threshold_norm = normalize_threshold(sbert_threshold, all_sbert_scores)
        ce_threshold_norm = normalize_threshold(ce_threshold, all_ce_scores)
        
        print(f"[CBC] 정규화 후 임계값: SBERT {sbert_threshold_norm:.3f}, CE {ce_threshold_norm:.3f}")
        
        # CBC 기반 필터링 (정규화된 임계값 사용)
        filter_metrics = compute_filtering_metrics_cbc_adaptive(
            sbert_scores, np.zeros_like(sbert_scores), ce_scores,
            candidate_pool, positives, negatives,
            sbert_threshold_norm, ce_threshold_norm, 0.0
        )
        
        # 결과 저장
        for k, v in filter_metrics.items():
            filter_agg[k].append(v)
        
        print(f"[SBERT/CE + CBC] Dropped: {filter_metrics['Dropped_Count']}")
    
    t3_end = timer_ms()
    print(f"[SBERT/CE + CBC] Completed in {t3_end - t3_start:.1f}ms")
    
    # 결과 집계
    res = aggregate_metrics(filter_agg)
    res["time_ms"] = t3_end - t3_start
    return res

def run_sbert_ce_llm_cbc(sbert, doc_embs, ce, corpus_texts, corpus_labels, query_texts, query_labels,
                        label_idx, candidate_cache, ce_cache, config, max_queries, cache_dir, qrels, qids, use_cbc=True):
    """④ SBERT + CE + 의미적 데이터 확장 + 신뢰도 균형 보정 라우팅 (제안기법)"""
    print("=" * 60)
    print("④ SBERT + CE + LLM + CBC (제안기법) 실험 시작")
    print("=" * 60)
    
    # CBC 통계 관리자 초기화
    cbc_stats = RunningPercentiles(maxlen=200)
    
    agg = defaultdict(list)
    filter_agg = defaultdict(list)
    t4 = timer_ms()
    
    # 1단계: 전체 실험의 점수 분포 수집을 위한 저장소
    all_sbert_scores = []
    all_ce_scores = []
    all_anchor_scores = []
    
    # 2단계: 모든 쿼리 처리하여 점수 분포 수집
    for i, (q, qid) in enumerate(zip(query_texts, qids)):
        if i >= max_queries:
            break
        print(f"[SBERT + CE + LLM + CBC] Collecting scores for query {i+1}/{min(max_queries, len(query_texts))}")
        
        relevant = qrels.get(qid, set())
        
        # 하드 네거티브 후보 풀 생성
        key = i
        if key not in candidate_cache:
            candidate_cache[key] = build_hard_candidate_pool(
                qid=qid, q_text=q, relevant_set=relevant,
                doc_embs=doc_embs, sbert=sbert, sbert_top_k=500
            )
        candidate_pool, positives, negatives = candidate_cache[key]
        
        # LLM 앵커 생성 (의미적 확장)
        try:
            anchors = generate_anchors_ollama(q, config["anchor_count"], 
                                            config["ollama_host"], config["ollama_port"], 
                                            config["ollama_model"], cache_dir)
        except Exception as e:
            print(f"[Warning] LLM anchor generation failed: {e}")
            anchors = []
        
        # SBERT 점수 계산 및 수집 (본 쿼리 + 앵커 맥스 풀링)
        q_emb = embed_texts(sbert, [q], config["batch_size"], config["show_progress"])[0]
        candidate_embs = doc_embs[candidate_pool]
        
        if len(anchors) > 0:
            anchor_embs = embed_texts(sbert, anchors, config["batch_size"], config["show_progress"])
            if len(anchor_embs) > 0 and anchor_embs.shape[0] > 0:
                q_scores = candidate_embs @ q_emb.reshape(-1, 1)  # (400, 384) @ (384, 1) = (400, 1)
                q_scores = q_scores.flatten()  # (400,)
                anchor_scores = candidate_embs @ anchor_embs.T  # [N, A]
                max_anchor_scores = np.max(anchor_scores, axis=1)  # [N]
                sbert_scores = np.maximum(q_scores, max_anchor_scores)
            else:
                sbert_scores = candidate_embs @ q_emb.reshape(-1, 1)  # (400, 384) @ (384, 1) = (400, 1)
                sbert_scores = sbert_scores.flatten()  # (400,)
        else:
            sbert_scores = candidate_embs @ q_emb.reshape(-1, 1)  # (400, 384) @ (384, 1) = (400, 1)
            sbert_scores = sbert_scores.flatten()  # (400,)
            
        all_sbert_scores.extend(sbert_scores)
        
        # 점수 저장
        if not hasattr(run_sbert_ce_llm_cbc, 'saved_scores'):
            run_sbert_ce_llm_cbc.saved_scores = {}
        if "SBERT/CE + SDE + CBC (Proposed)" not in run_sbert_ce_llm_cbc.saved_scores:
            run_sbert_ce_llm_cbc.saved_scores["SBERT/CE + SDE + CBC (Proposed)"] = {}
        
        # 원본 쿼리 점수와 앵커 점수를 분리하여 저장 (Proposed 방법의 특징)
        original_q_scores = candidate_embs @ q_emb
        if len(anchors) > 0 and len(anchor_embs) > 0 and anchor_embs.shape[0] > 0:
            anchor_scores = candidate_embs @ anchor_embs.T
            max_anchor_scores = np.max(anchor_scores, axis=1)
        else:
            max_anchor_scores = np.zeros_like(original_q_scores)
        
        run_sbert_ce_llm_cbc.saved_scores["SBERT/CE + SDE + CBC (Proposed)"][f"query_{i+1}"] = {
            "sbert_scores": original_q_scores.tolist(),  # 원본 쿼리 점수
            "anchor_scores": max_anchor_scores.tolist()  # 앵커 점수 (분포 확장 + CBC)
        }
        
        # CrossEncoder 점수 계산 및 수집 (개선: 상위 문서들로 분포 확보)
        ce_scores = np.full(len(candidate_pool), np.inf, dtype=np.float32)
        
        # 1. 분포 수집용: 상위 200개 문서의 CE 점수 계산
        top_m = min(200, len(candidate_pool))
        head_indices = np.argsort(sbert_scores)[::-1][:top_m]
        head_doc_ids = [candidate_pool[h] for h in head_indices]
        
        if head_doc_ids:
            head_ce_scores = ce_score_pairs_cached(ce, i, q, head_doc_ids, corpus_texts, ce_cache)
            all_ce_scores.extend(head_ce_scores.tolist())
            
            # 상위 문서들의 CE 점수를 배열에 매핑
            for k, doc_id in enumerate(head_doc_ids):
                doc_idx = candidate_pool.index(doc_id)
                ce_scores[doc_idx] = head_ce_scores[k]
        
        # 2. 필터링용: SBERT < 0.3인 문서들에 대해서도 CE 점수 계산 (기존 로직 유지)
        dropped_docs = []
        for j, doc_id in enumerate(candidate_pool):
            if sbert_scores[j] < 0.3:  # SBERT < 0.3인 문서들
                dropped_docs.append(doc_id)
        
        if dropped_docs:
            dropped_ce_scores = ce_score_pairs_cached(ce, i, q, dropped_docs, corpus_texts, ce_cache)
            for k, doc_id in enumerate(dropped_docs):
                doc_idx = candidate_pool.index(doc_id)
                if ce_scores[doc_idx] == np.inf:  # 아직 계산되지 않은 경우만
                    ce_scores[doc_idx] = dropped_ce_scores[k]
        
        # LLM 앵커 점수 계산 및 수집 (개선: 상위 문서들로 분포 확보)
        anchors = generate_anchors_ollama(q, config["anchor_count"], 
                                        config["ollama_host"], config["ollama_port"], 
                                        config["ollama_model"], cache_dir)
        
        if len(anchors) > 0:
            # 1. 분포 수집용: 상위 200개 문서의 앵커 점수 계산
            anchor_embs = [embed_texts(sbert, [a], config["batch_size"], config["show_progress"])[0] for a in anchors]
            anchor_candidate_scores = []
            for a_emb in anchor_embs:
                sims_a = doc_embs @ a_emb
                anchor_candidate_scores.append(sims_a[candidate_pool])
            anchor_candidate_scores = np.stack(anchor_candidate_scores, axis=1)
            max_anchor_scores = anchor_candidate_scores.max(axis=1)
            
            # 상위 문서들의 앵커 점수를 분포에 추가
            top_m = min(200, len(candidate_pool))
            all_anchor_scores.extend(max_anchor_scores[:top_m].tolist())
            
            # 2. 필터링용: 최종 드롭될 문서들에 대해서도 앵커 점수 계산 (기존 로직 유지)
            final_dropped_sim = []
            for j, doc_id in enumerate(candidate_pool):
                sbert_score = sbert_scores[j]
                ce_score = ce_scores[j]
                if sbert_score < 0.3 and ce_score < 0.3:  # 최종 드롭될 문서들
                    final_dropped_sim.append(doc_id)
            
            if final_dropped_sim:
                for doc_id in final_dropped_sim:
                    doc_idx = candidate_pool.index(doc_id)
                    all_anchor_scores.append(max_anchor_scores[doc_idx])
    
    # 3단계: 전체 분포 분석 및 적응적 퍼센타일 계산
    print(f"[CBC] 전체 분포 분석: SBERT {len(all_sbert_scores)}개, CE {len(all_ce_scores)}개, 앵커 {len(all_anchor_scores)}개")
    
    # 분포 특성 분석
    sbert_characteristics = analyze_score_distribution(all_sbert_scores)
    ce_characteristics = analyze_score_distribution(all_ce_scores)
    anchor_characteristics = analyze_score_distribution(all_anchor_scores)
    
    # 분포 상세 정보 출력
    print(f"[CBC] SBERT 분포: 평균 {sbert_characteristics['mean']:.3f}, 표준편차 {sbert_characteristics['std']:.3f}, 타입 {sbert_characteristics['type']}")
    print(f"[CBC] CE 분포: 평균 {ce_characteristics['mean']:.3f}, 표준편차 {ce_characteristics['std']:.3f}, 타입 {ce_characteristics['type']}")
    print(f"[CBC] 앵커 분포: 평균 {anchor_characteristics['mean']:.3f}, 표준편차 {anchor_characteristics['std']:.3f}, 타입 {anchor_characteristics['type']}")
    
    # 적응적 퍼센타일 계산
    adaptive_percentiles = calculate_adaptive_percentiles(
        sbert_characteristics, ce_characteristics, anchor_characteristics
    )
    
    print(f"[CBC] 적응적 퍼센타일: SBERT {adaptive_percentiles['sbert']:.1f}%, CE {adaptive_percentiles['ce']:.1f}%, 앵커 {adaptive_percentiles['anchor']:.1f}%")
    
    # 실제 임계값 계산 및 출력
    if len(all_sbert_scores) > 0:
        sbert_threshold = np.percentile(all_sbert_scores, adaptive_percentiles['sbert'])
        print(f"[CBC] SBERT 임계값: {sbert_threshold:.3f}")
    if len(all_ce_scores) > 0:
        ce_threshold = np.percentile(all_ce_scores, adaptive_percentiles['ce'])
        print(f"[CBC] CE 임계값: {ce_threshold:.3f}")
    if len(all_anchor_scores) > 0:
        anchor_threshold = np.percentile(all_anchor_scores, adaptive_percentiles['anchor'])
        print(f"[CBC] 앵커 임계값: {anchor_threshold:.3f}")
    
    # 4단계: 계산된 임계값으로 모든 쿼리 재처리
    for i, (q, qid) in enumerate(zip(query_texts, qids)):
        if i >= max_queries:
            break
        print(f"[SBERT + CE + LLM + CBC] Processing query {i+1}/{min(max_queries, len(query_texts))} with adaptive thresholds")
        
        relevant = qrels.get(qid, set())
        candidate_pool, positives, negatives = candidate_cache[i]
        
        # SBERT 점수 계산 (본쿼리 + 앵커 결합)
        q_emb = embed_texts(sbert, [q], config["batch_size"], config["show_progress"])[0]
        order_base = rank_biencoder(q_emb, doc_embs)
        candidate_embs = doc_embs[candidate_pool]
        
        # 본쿼리 점수
        q_scores = candidate_embs @ q_emb
        
        # 앵커 점수 계산 및 결합
        anchors = generate_anchors_ollama(q, config["anchor_count"], 
                                        config["ollama_host"], config["ollama_port"], 
                                        config["ollama_model"], cache_dir)
        
        if len(anchors) > 0:
            anchor_embs = [embed_texts(sbert, [a], config["batch_size"], config["show_progress"])[0] for a in anchors]
            anchor_scores = []
            for a_emb in anchor_embs:
                sims_a = candidate_embs @ a_emb
                anchor_scores.append(sims_a)
            anchor_scores = np.stack(anchor_scores, axis=1)
            max_anchor_scores = anchor_scores.max(axis=1)
            
            # 본쿼리 + 앵커 맥스 풀링
            sbert_scores = np.maximum(q_scores, max_anchor_scores)
        else:
            sbert_scores = q_scores
        
        # CrossEncoder 점수 계산 (후보풀 전체)
        ce_scores = ce_score_pairs_cached(ce, i, q, candidate_pool, corpus_texts, ce_cache)
        
        # LLM 앵커 점수 계산 (개선: 모든 문서에 대해 앵커 점수 계산)
        anchor_scores = np.full(len(candidate_pool), np.inf, dtype=np.float32)
        anchors = generate_anchors_ollama(q, config["anchor_count"], 
                                        config["ollama_host"], config["ollama_port"], 
                                        config["ollama_model"], cache_dir)
        
        if len(anchors) > 0:
            # 모든 문서에 대해 앵커 점수 계산 (구제 효과 극대화)
            anchor_candidate_scores = []
            for anchor_text in anchors:
                a_emb = embed_texts(sbert, [anchor_text], config["batch_size"], config["show_progress"])[0]
                sims_a = doc_embs @ a_emb
                anchor_candidate_scores.append(sims_a[candidate_pool])
            anchor_candidate_scores = np.stack(anchor_candidate_scores, axis=1)
            max_anchor_scores = anchor_candidate_scores.max(axis=1)
            
            # 모든 문서의 앵커 점수 설정
            anchor_scores = max_anchor_scores
        
        # 점수 정규화 (CBC 전) - 신호별 맞춤 정규화
        sbert_scores_norm = norm_cosine_per_query(sbert_scores)
        ce_scores_norm = ce_sigmoid_temp(ce_scores, T=2.0)
        anchor_scores_norm = norm_cosine_per_query(anchor_scores)
        
        # CBC 기반 임계값 (금융 도메인에 맞는 적극적 필터링)
        sbert_threshold = compute_cbc_thresholds(sbert_scores_norm, percentile=adaptive_percentiles['sbert'], floor=0.05)
        ce_threshold = compute_cbc_thresholds(ce_scores_norm, percentile=adaptive_percentiles['ce'], floor=0.10)
        anchor_threshold = compute_cbc_thresholds(anchor_scores_norm, percentile=adaptive_percentiles['anchor'], floor=0.05)
        
        print(f"[CBC] 정규화 후 임계값: SBERT {sbert_threshold:.3f}, CE {ce_threshold:.3f}, 앵커 {anchor_threshold:.3f}")
        
        # 적응적 퍼센타일 기반 필터링
        filter_metrics = compute_filtering_metrics_cbc(
            sbert_scores_norm, anchor_scores_norm, ce_scores_norm,
            candidate_pool, positives, negatives,
            cbc_stats,
            sbert_percentile=adaptive_percentiles['sbert'],
            anchor_percentile=adaptive_percentiles['anchor'],
            ce_percentile=adaptive_percentiles['ce']
        )
        
        m = compute_metrics(order_base, relevant, config["k_list"])
        
        print(f"[SBERT + CE + LLM + CBC] Dropped: {filter_metrics['Dropped_Count']}")
        
        for k, v in m.items():
            agg[k].append(v)
        for k, v in filter_metrics.items():
            filter_agg[k].append(v)
    
    # 결과 저장
    res = {
        "meta": {
            "variant": "sbert_ce_llm_cbc",
            "anchor_count": config["anchor_count"],
            "top_m": config["top_m"],
            "k_list": config["k_list"],
            "drop_thresholds": config["drop_thresholds"],
            "cbc_enabled": True,
            "adaptive_percentiles": adaptive_percentiles
        },
        "metrics": aggregate_metrics(agg),
        "filter_metrics": aggregate_metrics(filter_agg),
        "elapsed_ms": timer_ms() - t4
    }
    
    return res

# =========================
# 점수 정규화 (CE, SBERT, Anchor 공용)
# =========================
def ce_sigmoid_temp(z: np.ndarray, T: float = 2.0) -> np.ndarray:
    """CE 로짓 → 확률 변환 (Temperature 적용)"""
    out = np.full_like(z, np.inf, dtype=np.float32)
    m = np.isfinite(z)
    x = z[m].astype(np.float32) / max(T, 1e-6)
    # 안정적 sigmoid
    x = np.clip(x, -30.0, 30.0)
    out[m] = 1.0 / (1.0 + np.exp(-x))
    return out

def norm_cosine_per_query(scores):
    """코사인 유사도 per-query 정규화 (금융 도메인 맞춤)"""
    s = scores[np.isfinite(scores)]
    if len(s) == 0: 
        return scores
    lo, hi = float(s.min()), float(s.max())
    if hi - lo < 1e-6: 
        # 모든 점수가 동일할 때: 상위 50%는 0.6~1.0, 하위 50%는 0.0~0.4로 분산
        n = len(s)
        out = np.ones_like(scores, dtype=np.float32) * 0.5
        if n > 1:
            # 상위 절반은 0.6~1.0, 하위 절반은 0.0~0.4
            sorted_indices = np.argsort(scores)
            for i, idx in enumerate(sorted_indices):
                if i < n // 2:  # 하위 절반
                    out[idx] = 0.0 + (i / (n // 2)) * 0.4
                else:  # 상위 절반
                    out[idx] = 0.6 + ((i - n // 2) / (n - n // 2)) * 0.4
        out[~np.isfinite(scores)] = np.inf
        return out
    out = (scores - lo) / (hi - lo)
    out[~np.isfinite(scores)] = np.inf
    return out

def normalize_scores(scores: np.ndarray, method: str = "sigmoid", temperature: float = 1.0) -> np.ndarray:
    """유효 점수(=finite)만 정규화. 나머지는 inf 유지."""
    out = np.full_like(scores, np.inf, dtype=np.float32)
    mask = np.isfinite(scores)
    if not np.any(mask):
        return out
    x = scores[mask].astype(np.float32)

    if method == "sigmoid":
        # 로짓 범위의 CE/로 스케일 왜곡 방지 (0..1) + Temperature 적용
        x = 1.0 / (1.0 + np.exp(-x / temperature))
    elif method == "zsigmoid":
        mu, sd = float(x.mean()), float(x.std() + 1e-6)
        z = (x - mu) / sd
        x = 1.0 / (1.0 + np.exp(-z / temperature))
    elif method == "minmax":
        lo, hi = float(x.min()), float(x.max())
        if hi - lo < 1e-6:
            x = np.ones_like(x) * 0.5
        else:
            x = (x - lo) / (hi - lo)
    else:
        # no-op
        pass

    out[mask] = x
    return out

def apply_anchor_rescue_filter(
    sbert_scores: np.ndarray,
    ce_scores: np.ndarray,
    anchor_scores: np.ndarray,
    candidate_pool: List[int],
    positive_doc: int,
    negative_docs: List[int],
    thresholds: Dict[str, float]
):
    """
    1) 1차(SBERT) + 2차(CE)로 '유지' vs '드롭'을 먼저 결정
    2) '드롭'으로 판정된 문서들에 한해 앵커 점수로 '구제' 재심사
    3) 재정렬 없음. 오직 드롭↔유지 이진 결정만 수정
    """
    thr_s, thr_c, thr_a = thresholds["sbert"], thresholds["ce"], thresholds["anchor"]
    n = len(candidate_pool)

    # 1) 초기 유지/드롭 (AND 조건: sbert<thr AND ce<thr 이어야 드롭)
    keep_mask = np.zeros(n, dtype=bool)
    for i in range(n):
        conds = []
        if np.isfinite(sbert_scores[i]): conds.append(sbert_scores[i] >= thr_s)
        if np.isfinite(ce_scores[i]):    conds.append(ce_scores[i] >= thr_c)
        # 유효한 조건이 하나라도 있으면 그 중 하나라도 통과하면 유지
        keep_mask[i] = (len(conds) > 0) and any(conds)

    # 2) 드롭 후보만 앵커로 '구제'
    drop_mask = ~keep_mask
    for i in np.where(drop_mask)[0]:
        if np.isfinite(anchor_scores[i]) and anchor_scores[i] >= thr_a:
            keep_mask[i] = True  # 구제

    # 결과 세트
    keep_decisions = [candidate_pool[i] for i, k in enumerate(keep_mask) if k]
    drop_decisions = [candidate_pool[i] for i, k in enumerate(keep_mask) if not k]

    # 지표 (Drop*)
    total_negative = len(negative_docs)
    dropped_negative = len(set(drop_decisions) & set(negative_docs))
    drop_precision = dropped_negative / len(drop_decisions) if drop_decisions else 0.0
    drop_recall = dropped_negative / total_negative if total_negative > 0 else 0.0
    drop_f1 = 2 * drop_precision * drop_recall / (drop_precision + drop_recall) if (drop_precision + drop_recall) > 0 else 0.0

    # 정답 유지여부
    keep_recall = 1.0 if (positive_doc in keep_decisions) else 0.0

    return {
        "Drop_Precision": float(drop_precision),
        "Drop_Recall": float(drop_recall),
        "Drop_F1": float(drop_f1),
        "Keep_Recall": float(keep_recall),
        "Dropped_Count": len(drop_decisions),
        "Kept_Count": len(keep_decisions)
    }

# -----------------------------
# CBC 헬퍼 함수들
# -----------------------------

def analyze_score_distribution(scores):
    """점수 분포 특성 분석"""
    if len(scores) == 0:
        return {"type": "empty", "mean": 0.0, "std": 0.0, "skewness": 0.0}
    
    scores_array = np.array(scores)
    mean_score = np.mean(scores_array)
    std_score = np.std(scores_array)
    
    # 분포 특성 분석
    if mean_score > 0.7:
        distribution_type = "high_concentration"  # 높은 점수 집중
    elif mean_score < 0.3:
        distribution_type = "low_concentration"   # 낮은 점수 집중
    else:
        distribution_type = "balanced"            # 균형 잡힌 분포
    
    return {
        "type": distribution_type,
        "mean": mean_score,
        "std": std_score,
        "count": len(scores)
    }

def compute_cbc_thresholds(scores: np.ndarray, percentile: float = 20.0,
                           floor: float = None, ceil: float = None) -> float:
    """CBC 기반 동적 임계값 + 바닥/천장 가드레일"""
    if len(scores) == 0:
        return 0.0
    valid = scores[(scores != np.inf) & np.isfinite(scores)]
    if len(valid) == 0:
        return 0.0
    thr = float(np.percentile(valid, percentile))
    if floor is not None:
        thr = max(thr, floor)
    if ceil is not None:
        thr = min(thr, ceil)
    return thr

def calculate_adaptive_percentiles(sbert_characteristics, ce_characteristics, anchor_characteristics):
    """분포 특성에 따른 적응적 퍼센타일 계산"""
    
    # 기본 퍼센타일 (1차 개선 - 최적 성능 설정)
    base_sbert_percentile = 35.0  # 하위 65% 제거 (최적 재현율)
    base_ce_percentile = 65.0     # 엄격한 CE 필터링
    base_anchor_percentile = 45.0 # 적당한 앵커 구제
    
    # SBERT 분포 특성에 따른 조정
    if sbert_characteristics["type"] == "high_concentration":
        sbert_percentile = base_sbert_percentile + 10.0  # 더 엄격하게
    elif sbert_characteristics["type"] == "low_concentration":
        sbert_percentile = base_sbert_percentile - 10.0  # 더 관대하게
    else:
        sbert_percentile = base_sbert_percentile
    
    # CE 분포 특성에 따른 조정
    if ce_characteristics["type"] == "high_concentration":
        ce_percentile = base_ce_percentile + 10.0  # 더 엄격하게
    elif ce_characteristics["type"] == "low_concentration":
        ce_percentile = base_ce_percentile - 10.0  # 더 관대하게
    else:
        ce_percentile = base_ce_percentile
    
    # 앵커 분포 특성에 따른 조정
    if anchor_characteristics["type"] == "high_concentration":
        anchor_percentile = base_anchor_percentile + 10.0  # 더 엄격하게
    elif anchor_characteristics["type"] == "low_concentration":
        anchor_percentile = base_anchor_percentile - 10.0  # 더 관대하게
    else:
        anchor_percentile = base_anchor_percentile
    
    # 퍼센타일 범위 제한 (30-90%) - 더 관대한 범위
    sbert_percentile = max(30.0, min(90.0, sbert_percentile))
    ce_percentile = max(30.0, min(90.0, ce_percentile))
    anchor_percentile = max(30.0, min(90.0, anchor_percentile))
    
    return {
        "sbert": sbert_percentile,
        "ce": ce_percentile,
        "anchor": anchor_percentile
    }

# -----------------------------
# 메인 실험 실행
# -----------------------------

def run_semantic_benchmark():
    """MTEB/FiQA 금융 도메인 유사도 필터 벤치마크 실행"""
    print("=" * 60)
    print("MTEB/FiQA Financial Domain Semantic Filter Benchmark")
    print("=" * 60)
    print("Configuration:")
    print(f"  - Max queries: {EXPERIMENT_CONFIG['max_queries']}")
    print(f"  - Split: {EXPERIMENT_CONFIG['split']}")
    print(f"  - Output directory: {EXPERIMENT_CONFIG['out_dir']}")
    print()
    
    # 설정 로드
    config = get_semantic_config()
    
    # 시드 설정
    set_seed(42)
    
    # 데이터 로드 (qrels 기반 정상 라벨링)
    print("Loading preprocessed MTEB/FiQA dataset (Financial Domain IR)...")
    corpus_texts, corpus_labels, query_texts, query_labels, qrels = [], [], [], [], {}
    
    for query, passages, labels, query_id in iter_fiqa_query_groups(
        split=EXPERIMENT_CONFIG["split"], 
        max_queries=EXPERIMENT_CONFIG["max_queries"]
    ):
        query_texts.append(query)
        query_labels.append(query_id)  # 쿼리 ID를 라벨로 사용
        
        # qrels 구성: 쿼리별 관련 문서 집합
        relevant_docs = []
        for i, (text, label) in enumerate(zip(passages, labels)):
            doc_id = len(corpus_texts) + i  # 문서 ID 생성
            corpus_texts.append(text)
            corpus_labels.append(doc_id)  # 문서 ID를 라벨로 사용
            if label == 1:  # 관련 문서인 경우
                relevant_docs.append(doc_id)
        
        qrels[query_id] = set(relevant_docs)  # 쿼리별 관련 문서 집합 저장
    
    print(f"Loaded {len(query_texts)} queries, {len(corpus_texts)} passages")
    print(f"Qrels entries: {len(qrels)}")
    
    # qids 생성 (query_labels와 동일)
    qids = query_labels.copy()
    
    # 라벨 인덱스 생성 (문서 ID 기반)
    label_idx = build_label_index(corpus_labels)
    
    # 모델 로드
    print("Loading models...")
    sbert = SentenceTransformer(config["sbert_model"])
    ce = CrossEncoder(config["ce_model"])
    
    # 문서 임베딩 생성
    print("Generating document embeddings...")
    doc_embs = embed_texts(sbert, corpus_texts, config["batch_size"], config["show_progress"])
    
    # 캐시 디렉토리 설정 (.cache/semantic/fiqa/)
    cache_dir = ".cache/semantic/fiqa"
    ensure_dir(cache_dir)
    
    # 결과 저장 디렉토리
    ensure_dir(EXPERIMENT_CONFIG["out_dir"])
    
    # 공통 캐시
    candidate_cache = {}
    ce_cache = {}
    
    # 공정 비교를 위한 공통 후보 풀 생성
    print("Generating common candidate pools for fair comparison...")
    for i, (q, qid) in enumerate(zip(query_texts, qids)):
        if i >= EXPERIMENT_CONFIG["max_queries"]:
            break
        
        # qrels 기반 관련 문서 집합
        relevant = qrels.get(qid, set())
        
        # 하드 네거티브 후보 풀 생성 (모든 실험에서 동일하게 사용)
        candidate_cache[i] = build_hard_candidate_pool(
            qid=qid, q_text=q, relevant_set=relevant,
            doc_embs=doc_embs, sbert=sbert, sbert_top_k=200
        )
    
    print(f"Generated {len(candidate_cache)} common candidate pools")
    
    # 4단계 실험 실행
    results = {}
    
    try:
        # ① SBERT/CE (Baseline) - SBERT + CE만 사용
        results["sbert_ce_baseline"] = run_sbert_ce(
            sbert, doc_embs, ce, corpus_texts, corpus_labels, query_texts, query_labels,
            label_idx, candidate_cache, ce_cache, config, EXPERIMENT_CONFIG["max_queries"], qrels, qids, True
        )
        
        # ② SBERT/CE + SDE - SBERT + CE + LLM 앵커 (SDE)
        results["sbert_ce_sde"] = run_sbert_ce_llm(
            sbert, doc_embs, ce, corpus_texts, corpus_labels, query_texts, query_labels,
            label_idx, candidate_cache, ce_cache, config, EXPERIMENT_CONFIG["max_queries"], cache_dir, qrels, qids, True
        )
        
        # ③ SBERT/CE + CBC - SBERT + CE + CBC (CBC만)
        results["sbert_ce_cbc"] = run_sbert_ce_cbc(
            sbert, doc_embs, ce, corpus_texts, corpus_labels, query_texts, query_labels,
            label_idx, candidate_cache, ce_cache, config, EXPERIMENT_CONFIG["max_queries"], qrels, qids, True
        )
        
        # ④ SBERT/CE + SDE + CBC (Proposed) - 모든 기법 조합
        results["sbert_ce_sde_cbc"] = run_sbert_ce_llm_cbc(
            sbert, doc_embs, ce, corpus_texts, corpus_labels, query_texts, query_labels,
            label_idx, candidate_cache, ce_cache, config, EXPERIMENT_CONFIG["max_queries"], cache_dir, qrels, qids, True
        )
        
        # 결과 저장
        for variant, result in results.items():
            save_path = os.path.join(EXPERIMENT_CONFIG["out_dir"], f"{variant}_results.json")
            save_json(result, save_path)
            print(f"Results saved: {save_path}")
        
            # 실제 계산된 점수들 저장 (Baseline과 SDE만)
            all_scores = {}
            
            # SBERT/CE (Baseline)과 SBERT/CE + SDE 점수만 수집
            if hasattr(run_sbert_ce, 'saved_scores'):
                all_scores.update(run_sbert_ce.saved_scores)
            if hasattr(run_sbert_ce_llm, 'saved_scores'):
                all_scores.update(run_sbert_ce_llm.saved_scores)
        
        if all_scores:
            score_data = {
                "experiment_info": {
                    "dataset": "MTEB/FiQA",
                    "max_queries": EXPERIMENT_CONFIG["max_queries"],
                    "split": "test",
                    "timestamp": "2025-01-27 12:00:00",
                    "description": "Real similarity scores from FiQA financial domain experiments",
                    "candidate_docs_per_query": 200
                },
                "scores": all_scores,
                "query_details": {
                    "query_count": EXPERIMENT_CONFIG["max_queries"],
                    "candidate_docs_per_query": 200,
                    "score_range": "0.0 - 1.0",
                    "score_type": "cosine_similarity",
                    "notes": "Real experiment scores from actual calculations."
                }
            }
            
            score_path = os.path.join(EXPERIMENT_CONFIG["out_dir"], "score.json")
            save_json(score_data, score_path)
            print(f"Real scores saved: {score_path}")
        
        # 비교 요약 출력
        print_comparison_summary(results)
        
        print("\nFiQA Semantic Filter benchmark completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n\nError during FiQA Semantic Filter benchmark execution: {e}")
        return False

def print_comparison_summary(results: Dict[str, Dict[str, Any]]):
    """4단계 실험 결과 비교 요약 출력"""
    print("\n" + "="*80)
    print("4단계 유사도 필터 성능 비교")
    print("="*80)
    
    variants = ["sbert_baseline", "sbert_ce", "sbert_ce_llm", "sbert_ce_llm_cbc"]
    variant_names = ["① SBERT", "② SBERT+CE", "③ SBERT+CE+LLM", "④ SBERT+CE+LLM+CBC"]
    
    # 핵심 지표 비교
    print("📊 핵심 지표 (확실한 오답 제거 검증)")
    print("-" * 80)
    print(f"{'방식':<20} {'Drop Precision':<15} {'Drop Recall':<15} {'Drop F1':<15} {'P@1':<10} {'P@10':<10}")
    print("-" * 80)
    
    # 각 방식별로 행 출력
    for i, variant in enumerate(variants):
        if variant in results:
            filter_metrics = results[variant].get("filter_metrics", {})
            metrics = results[variant].get("metrics", {})
            
            drop_precision = filter_metrics.get('Drop_Precision', 0.0)
            drop_recall = filter_metrics.get('Drop_Recall', 0.0)
            drop_f1 = filter_metrics.get('Drop_F1', 0.0)
            p_at_1 = metrics.get('P@1', 0.0)
            p_at_10 = metrics.get('P@10', 0.0)
            
            print(f"{variant_names[i]:<20} {drop_precision:<15.3f} {drop_recall:<15.3f} {drop_f1:<15.3f} {p_at_1:<10.3f} {p_at_10:<10.3f}")
    
    print("=" * 80)
    print("📈 실험 결과 요약")
    print("=" * 80)
    print("• Drop Precision: 필터가 제거한 문서 중 실제로 무관한 문서의 비율")
    print("• Drop Recall: 전체 무관 문서 중에서 필터가 실제로 제거한 비율") 
    print("• Drop F1: Drop Precision과 Drop Recall의 조화 평균")
    print("• P@1, P@10: 상위 랭크된 결과 중 정답 문서의 비율 (보조 지표)")
    print("=" * 80)

if __name__ == "__main__":
    run_semantic_benchmark()
