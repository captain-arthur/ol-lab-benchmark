#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_banking77_refactored.py
- s_filter.py 공통 모듈을 사용하여 리팩토링된 버전
- mteb/banking77 (실패 시 banking77)으로 3단계 실험을 순차 수행
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
SBERT_MODEL = "sentence-transformers/all-mpnet-base-v2"
SBERT_MAX_SEQ_LEN = 256
SBERT_BATCH_SIZE = 64

# 앵커 생성 (Ollama)
OLLAMA_HOST = "192.168.45.166"
OLLAMA_PORT = 11434
OLLAMA_MODEL = "gemma3"
ANCHOR_COUNT = 4
ANCHOR_MODE = "weighted"
ALPHA = 0.7

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

# 필터링 설정
NEGATIVE_SAMPLES = 99
K_LIST = [10, 100]
MAX_QUERIES = 100

# 기타
SEED = 42
OUT_DIR = "results/similarity/banking77"
CACHE_DIR = ".cache/similarity/banking77"
SHOW_PROGRESS = False

# =========================
# 데이터셋 로딩
# =========================

def load_banking77_dataset():
    """Banking77 데이터셋 로드"""
    try:
        ds_train = load_dataset("mteb/banking77", split="train")
        ds_test = load_dataset("mteb/banking77", split="test")
    except Exception:
        ds_train = load_dataset("banking77", split="train")
        ds_test = load_dataset("banking77", split="test")

    corpus_texts = [ex["text"] for ex in ds_train]
    corpus_labels = [int(ex["label"]) for ex in ds_train]
    query_texts = [ex["text"] for ex in ds_test]
    query_labels = [int(ex["label"]) for ex in ds_test]
    return corpus_texts, corpus_labels, query_texts, query_labels

# =========================
# 실험 실행
# =========================

def run_banking77():
    """Banking77 실험 실행"""
    set_seed(SEED)
    ensure_dir(OUT_DIR)
    ensure_dir(CACHE_DIR)
    t_all = timer_ms()

    # 1) 데이터
    corpus_texts, corpus_labels, query_texts, query_labels = load_banking77_dataset()
    label_idx = build_label_index(corpus_labels)

    # 2) SBERT 준비 및 코퍼스 임베딩
    print(f"[SBERT] {SBERT_MODEL} (max_len={SBERT_MAX_SEQ_LEN})")
    sbert = SentenceTransformer(SBERT_MODEL)
    sbert.max_seq_length = SBERT_MAX_SEQ_LEN

    print("[Embedding] corpus ...")
    doc_embs = embed_texts(sbert, corpus_texts, SBERT_BATCH_SIZE, SHOW_PROGRESS)

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
    ce_cache = {}

    # 1) Baseline
    print("\n[Run] 1) SBERT + Cosine (Baseline)")
    baseline_results = run_baseline_experiment(
        sbert, doc_embs, corpus_texts, corpus_labels, query_texts, query_labels,
        label_idx, candidate_cache, K_LIST, MAX_QUERIES, NEGATIVE_SAMPLES,
        DROP_THRESHOLDS, OUT_DIR, SHOW_PROGRESS
    )
    # 결과를 파일로 저장
    baseline_path = os.path.join(OUT_DIR, "banking77_sbert_cosine.json")
    save_json(baseline_results, baseline_path)
    print(f"[Saved] {baseline_path}")
    results_paths["baseline"] = baseline_path

    # 2) Anchors
    print("\n[Run] 2) SBERT + Cosine + Anchors")
    anchor_results = run_anchor_experiment(
        sbert, doc_embs, corpus_texts, corpus_labels, query_texts, query_labels,
        label_idx, candidate_cache, K_LIST, MAX_QUERIES, NEGATIVE_SAMPLES,
        DROP_THRESHOLDS, OUT_DIR, CACHE_DIR, SHOW_PROGRESS,
        ANCHOR_COUNT, OLLAMA_HOST, OLLAMA_PORT, OLLAMA_MODEL
    )
    # 결과를 파일로 저장
    anchor_path = os.path.join(OUT_DIR, "banking77_sbert_anchors.json")
    save_json(anchor_results, anchor_path)
    print(f"[Saved] {anchor_path}")
    results_paths["anchors"] = anchor_path

    # 3) CrossEncoder
    print("\n[Run] 3) CrossEncoder Reranking")
    ce_results = run_crossencoder_experiment(
        sbert, doc_embs, ce, corpus_texts, corpus_labels, query_texts, query_labels,
        label_idx, candidate_cache, ce_cache, K_LIST, MAX_QUERIES, NEGATIVE_SAMPLES,
        DROP_THRESHOLDS, OUT_DIR, CACHE_DIR, SHOW_PROGRESS,
        ANCHOR_COUNT, OLLAMA_HOST, OLLAMA_PORT, OLLAMA_MODEL, TOP_M
    )
    # 결과를 파일로 저장
    ce_path = os.path.join(OUT_DIR, "banking77_sbert_anchors_ce.json")
    save_json(ce_results, ce_path)
    print(f"[Saved] {ce_path}")
    results_paths["crossencoder"] = ce_path

    # 전체 요약
    summary = {
        "dataset": "banking77",
        "config": {
            "sbert_model": SBERT_MODEL,
            "cross_encoder_model": CROSS_ENCODER_MODEL,
            "anchor_count": ANCHOR_COUNT,
            "top_m": TOP_M,
            "drop_thresholds": DROP_THRESHOLDS,
            "k_list": K_LIST,
            "max_queries": MAX_QUERIES,
            "seed": SEED
        },
        "results_paths": results_paths,
        "total_elapsed_ms": t_all(),
        "ce_calibration": ce_calib_meta
    }
    
    summary_path = os.path.join(OUT_DIR, "banking77_summary.json")
    save_json(summary, summary_path)
    print(f"\n[Summary] Saved to {summary_path}")
    
    return summary

# run_baseline_experiment 함수는 s_filter.py에서 import하여 사용

# run_anchor_experiment 함수는 s_filter.py에서 import하여 사용

# run_crossencoder_experiment 함수는 s_filter.py에서 import하여 사용

# =========================
# 메인
# =========================

if __name__ == "__main__":
    run_banking77()
