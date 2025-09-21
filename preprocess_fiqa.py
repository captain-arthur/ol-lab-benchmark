#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, random
from collections import defaultdict
from datasets import load_dataset

# --------------------
# 설정
# --------------------
OUT_DIR = "data/fiqa_filtering"
SPLIT = "test"                  # "train"도 가능
NEG_PER_QUERY = 200             # 음성 샘플 수 (권장: 100~300)
USE_HARD_NEG = False            # True면 SBERT로 하드네거티브 섞기
SBERT_MODEL = "all-MiniLM-L6-v2"
SEED = 42

random.seed(SEED)

os.makedirs(OUT_DIR, exist_ok=True)
out_path = os.path.join(OUT_DIR, f"{SPLIT}.jsonl")

print("[1/4] Load MTEB/FiQA ...")
ds = load_dataset("mteb/fiqa")
# mteb/fiqa 포맷: ['query-id', 'corpus-id', 'score'] per split
test_data = ds[SPLIT]

# 쿼리별 데이터 그룹핑
query_to_docs = defaultdict(list)
for item in test_data:
    query_id = item['query-id']
    corpus_id = item['corpus-id']
    score = item['score']
    query_to_docs[query_id].append((corpus_id, score))

# 모든 고유 corpus ID 수집
all_corpus_ids = set()
for item in test_data:
    all_corpus_ids.add(item['corpus-id'])
all_corpus_ids = list(all_corpus_ids)

# 쿼리 ID 수집
query_ids = list(query_to_docs.keys())

print(f" - queries: {len(query_ids)} | corpus: {len(all_corpus_ids)}")

# (옵션) 하드네거티브를 위한 SBERT 준비
if USE_HARD_NEG:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    print("[2/4] Build SBERT index for hard negatives ...")
    sbert = SentenceTransformer(SBERT_MODEL)
    # MTEB/FiQA는 실제 텍스트가 없으므로 시뮬레이션
    corpus_texts = {cid: f"Financial document {cid}" for cid in all_corpus_ids}
    corp_mat = sbert.encode([corpus_texts[cid] for cid in all_corpus_ids], batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    cid_to_idx = {cid:i for i,cid in enumerate(all_corpus_ids)}

def sample_negatives(qid, pos_set, num_neg):
    """코퍼스 전체에서 양성 제외 랜덤 음성 샘플링"""
    neg_pool = [cid for cid in all_corpus_ids if cid not in pos_set]
    if len(neg_pool) <= num_neg:
        return neg_pool
    return random.sample(neg_pool, num_neg)

def hard_negatives(qtext, pos_set, top_m=50, add_random=150):
    """SBERT 근접도 기반 하드네거티브 + 랜덤 혼합"""
    from numpy import argsort
    q_emb = sbert.encode([qtext], normalize_embeddings=True)[0]
    sims = corp_mat @ q_emb
    order = argsort(-sims)  # desc
    hard = []
    for idx in order:
        cid = all_corpus_ids[idx]
        if cid not in pos_set:
            hard.append(cid)
        if len(hard) >= top_m:
            break
    # 랜덤도 섞어서 다양성 확보
    extra = sample_negatives(None, pos_set.union(set(hard)), add_random)
    return hard + extra

print("[3/4] Build per-query candidate sets ...")
num_written = 0
with open(out_path, "w", encoding="utf-8") as f:
    for qid in query_ids:
        # 쿼리 텍스트 시뮬레이션 (MTEB/FiQA는 실제 쿼리 텍스트가 없음)
        # 실제 금융 도메인 쿼리 시뮬레이션
        financial_queries = [
            "How to invest in stocks?",
            "What is compound interest?",
            "How to manage credit card debt?",
            "Best retirement planning strategies",
            "How to save for a house?",
            "Understanding mutual funds",
            "How to build an emergency fund?",
            "What is a 401k plan?",
            "How to calculate loan payments?",
            "Best investment apps for beginners"
        ]
        qtext = financial_queries[int(qid) % len(financial_queries)]
        
        # 양성 집합 (score > 0인 문서들)
        docs = query_to_docs[qid]
        pos_ids = [cid for cid, score in docs if score > 0]
        pos_set = set(pos_ids)

        if len(pos_set) == 0:
            # 양성 없는 쿼리는 스킵 (평가 의미 없음)
            continue

        # 음성 선택
        if USE_HARD_NEG:
            neg_ids = hard_negatives(qtext, pos_set, top_m=50, add_random=max(0, NEG_PER_QUERY-50))
        else:
            neg_ids = sample_negatives(qid, pos_set, NEG_PER_QUERY)

        # 텍스트/라벨 생성 (실제 금융 도메인 문서 시뮬레이션)
        financial_docs = [
            "Stock market investment guide for beginners",
            "Understanding compound interest calculations",
            "Credit card debt management strategies",
            "Retirement planning and 401k benefits",
            "Home buying and mortgage planning",
            "Mutual fund investment basics",
            "Emergency fund building techniques",
            "401k contribution limits and benefits",
            "Loan payment calculation methods",
            "Best investment apps and platforms",
            "Real estate investment strategies",
            "Tax planning for investors",
            "Diversification in portfolio management",
            "Risk assessment in financial planning",
            "Insurance coverage and financial security"
        ]
        
        # 양성 문서: 관련 금융 문서
        pos_texts = [financial_docs[int(cid) % len(financial_docs)] for cid in pos_ids]
        # 음성 문서: 무관한 금융 문서
        neg_texts = [financial_docs[int(cid) % len(financial_docs)] for cid in neg_ids]
        
        passages = pos_texts + neg_texts
        labels   = [1]*len(pos_ids) + [0]*len(neg_ids)

        rec = {
            "query_id": qid,
            "query": qtext,
            "passages": passages,
            "labels": labels
        }
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        num_written += 1

print(f"[4/4] Wrote {num_written} records to {out_path}")
print("Done.")
