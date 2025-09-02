#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_clinc150.py
- s_filter.py 공통 모듈을 사용하여 리팩토링된 버전
- mteb/clinc150 (실패 시 clinc150)으로 3단계 실험을 순차 수행
"""

import os
from collections import defaultdict
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, CrossEncoder

# 공통 모듈 import
from .s_filter import *

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
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # 범용 모델
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
MAX_QUERIES = 100  # 최대 100개 쿼리만 실험

# 기타
SEED = 42
OUT_DIR = "results/similarity/clinc150"
CACHE_DIR = ".cache/similarity/clinc150"      # Ollama 응답 캐시 디렉토리
SHOW_PROGRESS = False                # SBERT encode progress bar

# =========================
# 데이터셋 로딩
# =========================

def load_clinc150_dataset():
    """CLINC150 데이터셋 로딩"""
    try:
        # MTEB에서 로딩 시도
        dataset = load_dataset("mteb/clinc150")
        print("✅ MTEB/CLINC150 데이터셋 로딩 성공")
    except Exception as e:
        print(f"⚠️   MTEB/CLINC150 로딩 실패: {e}")
        try:
            # 직접 로딩 시도
            dataset = load_dataset("clinc150")
            print("✅ CLINC150 데이터셋 로딩 성공")
        except Exception as e2:
            print(f"❌ CLINC150 데이터셋 로딩 실패: {e2}")
            raise
    
    # 데이터 추출
    train_data = dataset["train"]
    test_data = dataset["test"]
    
    # 텍스트와 라벨 분리
    corpus_texts = []
    corpus_labels = []
    query_texts = []
    query_labels = []
    
    # 학습 데이터를 코퍼스로 사용
    for item in train_data:
        text = item["text"]
        label = item["intent"]  # intent를 라벨로 사용
        corpus_texts.append(text)
        corpus_labels.append(label)
    
    # 테스트 데이터를 쿼리로 사용
    for item in test_data:
        text = item["text"]
        label = item["intent"]  # intent를 라벨로 사용
        query_texts.append(text)
        query_labels.append(label)
    
    print(f"📊 코퍼스: {len(corpus_texts)}개 문장")
    print(f"🔍 쿼리: {len(query_texts)}개 문장")
    print(f"🏷️  라벨 종류: {len(set(corpus_labels))}개")
    
    return corpus_texts, corpus_labels, query_texts, query_labels

# =========================
# 실험 실행
# =========================

def run_clinc150():
    set_seed(SEED)
    ensure_dir(OUT_DIR)
    ensure_dir(CACHE_DIR)
    t_all = timer_ms()

    corpus_texts, corpus_labels, query_texts, query_labels = load_clinc150_dataset()
    label_idx = build_label_index(corpus_labels)

    sbert = SentenceTransformer(SBERT_MODEL)
    sbert.max_seq_length = SBERT_MAX_SEQ_LEN
    doc_embs = embed_texts(sbert, corpus_texts, SBERT_BATCH_SIZE, SHOW_PROGRESS)

    ce = CrossEncoder(CROSS_ENCODER_MODEL, max_length=CE_MAX_LEN)

    # CE threshold calibration
    print("\n🔧 CrossEncoder 임계값 보정 중...")
    ce_threshold, ce_calib_meta = calibrate_ce_threshold(
        ce, query_texts, corpus_texts, sample_pairs=400, base_thr=0.5
    )
    DROP_THRESHOLDS["ce"] = ce_threshold
    print(f"✅ CE 임계값 보정 완료: {ce_threshold:.3f}")

    results_paths = {}
    candidate_cache = {}
    ce_cache = {}

    # 1) Baseline
    baseline_results = run_baseline_experiment(
        sbert, doc_embs, corpus_texts, corpus_labels, query_texts, query_labels,
        label_idx, candidate_cache, K_LIST, MAX_QUERIES, NEGATIVE_SAMPLES,
        DROP_THRESHOLDS, OUT_DIR, SHOW_PROGRESS
    )
    # 결과를 파일로 저장
    baseline_path = os.path.join(OUT_DIR, "clinc150_sbert_cosine.json")
    save_json(baseline_results, baseline_path)
    print(f"[Saved] {baseline_path}")
    results_paths["baseline"] = baseline_path

    # 2) Anchors
    anchor_results = run_anchor_experiment(
        sbert, doc_embs, corpus_texts, corpus_labels, query_texts, query_labels,
        label_idx, candidate_cache, K_LIST, MAX_QUERIES, NEGATIVE_SAMPLES,
        DROP_THRESHOLDS, OUT_DIR, CACHE_DIR, SHOW_PROGRESS,
        ANCHOR_COUNT, OLLAMA_HOST, OLLAMA_PORT, OLLAMA_MODEL
    )
    # 결과를 파일로 저장
    anchor_path = os.path.join(OUT_DIR, "clinc150_sbert_anchors.json")
    save_json(anchor_results, anchor_path)
    print(f"[Saved] {anchor_path}")
    results_paths["anchors"] = anchor_path

    # 3) CrossEncoder
    ce_results = run_crossencoder_experiment(
        sbert, doc_embs, ce, corpus_texts, corpus_labels, query_texts, query_labels,
        label_idx, candidate_cache, ce_cache, K_LIST, MAX_QUERIES, NEGATIVE_SAMPLES,
        DROP_THRESHOLDS, OUT_DIR, CACHE_DIR, SHOW_PROGRESS,
        ANCHOR_COUNT, OLLAMA_HOST, OLLAMA_PORT, OLLAMA_MODEL, TOP_M
    )
    # 결과를 파일로 저장
    ce_path = os.path.join(OUT_DIR, "clinc150_sbert_anchors_ce.json")
    save_json(ce_results, ce_path)
    print(f"[Saved] {ce_path}")
    results_paths["crossencoder"] = ce_path

    # Summary
    summary = {
        "paths": results_paths,
        "total_elapsed_ms": t_all(),
        "max_queries": MAX_QUERIES,
        "total_queries": len(query_texts),
        "parameters": {
            "sbert_model": SBERT_MODEL,
            "ollama_host": OLLAMA_HOST,
            "ollama_port": OLLAMA_PORT,
            "ollama_model": OLLAMA_MODEL,
            "k_list": K_LIST,
            "seed": SEED,
            "drop_thresholds": DROP_THRESHOLDS,
            "ce_calibration": ce_calib_meta,
        }
    }
    p_sum = os.path.join(OUT_DIR, "clinc150_summary.json")
    save_json(summary, p_sum)
    print(f"\n[Summary Saved] {p_sum}")
    
    # 핵심 결과 콘솔 출력
    print("\n" + "="*80)
    print("🏆 CLINC150 벤치마크 결과 요약")
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
        with open(results_paths["crossencoder"], 'r') as f:
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
    run_clinc150()
