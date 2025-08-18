import os
import json
import csv
import re
import math
import time
from typing import List, Dict, Any, Iterable, Tuple, Optional

import numpy as np
from datasets import load_dataset
from rank_bm25 import BM25Okapi
import requests


# -----------------------------
# Configuration
# -----------------------------
# 실험 설정 (상단에서 관리)
EXPERIMENT_CONFIG = {
    "max_queries": 20,           # 기본값: 20, 최대 500,000까지 확장 가능
    "split": "validation",       # "validation" 또는 "train"
    "semantic_sample_rate": 1.0, # 의미 확장 적용 비율 (0.0 ~ 1.0)
    "allow_token_duplicates": 2, # 토큰 중복 허용 횟수 (1~2회 권장)
    "out_dir": "results"
}


# -----------------------------
# Utilities
# -----------------------------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def tokenize_en(text: str) -> List[str]:
    text = re.sub(r"[^\w\s]", " ", text.lower())
    toks = text.split()
    # 길이 2 이하 토큰은 잡음으로 제거
    return [t for t in toks if len(t) > 2]

def normalize_tokens(tokens: List[str], allow_duplicates: int = 2) -> List[str]:
    """
    토큰 정규화: 중복 제거 및 정렬
    allow_duplicates: 중복 허용 횟수 (1~2회 권장)
    """
    if not tokens:
        return []
    
    # 토큰 빈도 계산
    token_counts = {}
    for token in tokens:
        token_counts[token] = token_counts.get(token, 0) + 1
    
    # 중복 제거 (allow_duplicates 횟수만큼만 허용)
    normalized = []
    for token, count in token_counts.items():
        normalized.extend([token] * min(count, allow_duplicates))
    
    return normalized

def ndcg_at_k(ranked_rel: List[int], k: int = 10) -> float:
    k = min(k, len(ranked_rel))
    dcg = 0.0
    for i in range(k):
        r = ranked_rel[i]
        dcg += (2**r - 1) / math.log2(i + 2)
    ideal = sorted(ranked_rel, reverse=True)[:k]
    idcg = 0.0
    for i, r in enumerate(ideal):
        idcg += (2**r - 1) / math.log2(i + 2)
    return dcg / idcg if idcg > 0 else 0.0

def mrr_at_k(ranked_rel: List[int], k: int = 10) -> float:
    k = min(k, len(ranked_rel))
    for i in range(k):
        if ranked_rel[i] == 1:
            return 1.0 / (i + 1)
    return 0.0


# -----------------------------
# Semantic expansion (Ollama gemma3)
# -----------------------------
def build_semantic_data_ollama(query: str,
                               host: str = "http://192.168.45.166:11434",
                               model: str = "gemma3",
                               timeout: int = 20) -> Optional[Dict[str, Any]]:
    """
    Ollama 서버(gemma3)에 프롬프트를 보내 semantic_data 생성.
    반환 형식:
    {
      "user_query": str,
      "intent_data": {"language":"en"},
      "expanded": {
        "keywords": [...],
        "must_include": [...],
        "forbidden_terms": [...]
      }
    }
    """
    prompt = f"""
You are a data enrichment assistant. Given a user query, produce semantic_data JSON that helps BM25 retrieval.
Only output valid JSON (no code fences), and keep lists concise but useful.

Example format:
{{
  "user_query": "Collect the latest U.S. Federal Reserve interest rate changes",
  "intent_data": {{
    "language": "en"
  }},
  "expanded": {{
    "keywords": ["Federal Reserve interest rate changes", "US interest rate trends"],
    "must_include": ["interest rate", "Federal Reserve"],
    "forbidden_terms": ["cryptocurrency","Bitcoin"]
  }}
}}

Now produce semantic_data for this query:
"{query}"
""".strip()

    try:
        # Ollama /api/generate
        resp = requests.post(
            f"{host}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        text = data.get("response", "").strip()
        # 모델이 JSON만 반환하도록 프롬프트 했지만, 혹시 앞뒤 잡음 제거
        first = text.find("{")
        last = text.rfind("}")
        if first == -1 or last == -1:
            return None
        json_text = text[first:last+1]
        parsed = json.loads(json_text)
        # 필드 정리 및 정규화
        exp = parsed.get("expanded", {})
        return {
            "user_query": parsed.get("user_query", query),
            "intent_data": parsed.get("intent_data", {"language": "en"}),
            "expanded": {
                "keywords": exp.get("keywords", []),
                "must_include": exp.get("must_include", []),
                "forbidden_terms": exp.get("forbidden_terms", [])
            }
        }
    except Exception:
        return None


def apply_semantic_to_query_tokens(base_tokens: List[str], sem: Optional[Dict[str, Any]], allow_duplicates: int = 2) -> Tuple[List[str], Dict[str, Any]]:
    """
    semantic_data를 쿼리 토큰에 반영:
      - expanded.keywords: 토큰화 후 약한 가중(2배)로 추가
      - must_include: 토큰화 후 강한 가중(3배)로 추가
      - forbidden_terms: 쿼리 토큰엔 반영하지 않음(문서 필터링에서 적용)
    
    반환: (최종 토큰 리스트, 디버그 정보)
    """
    if not sem: 
        return base_tokens, {"base_tokens": base_tokens, "added_tokens": [], "final_tokens": base_tokens}

    exp = sem.get("expanded", {})
    keywords = exp.get("keywords", [])
    must_inc = exp.get("must_include", [])

    tokens = list(base_tokens)
    added_tokens = []

    def add_weighted(tokens_src: List[str], weight: int):
        nonlocal added_tokens
        for s in tokens_src:
            toks = tokenize_en(s)
            added_tokens.extend(toks * weight)
            tokens.extend(toks * weight)

    add_weighted(keywords, 2)
    add_weighted(must_inc, 3)

    # 토큰 정규화 (중복 제거)
    final_tokens = normalize_tokens(tokens, allow_duplicates)
    
    debug_info = {
        "base_tokens": base_tokens,
        "added_tokens": added_tokens,
        "final_tokens": final_tokens,
        "expanded_keywords": keywords,
        "must_include": must_inc,
        "forbidden_terms": exp.get("forbidden_terms", [])
    }
    
    return final_tokens, debug_info


def passage_has_forbidden(passage: str, forbidden_terms: List[str]) -> bool:
    if not forbidden_terms:
        return False
    p = passage.lower()
    for t in forbidden_terms:
        if t.lower() in p:
            return True
    return False


# -----------------------------
# Data: MS MARCO (streaming)
# -----------------------------
def iter_ms_marco_query_groups(split: str = "validation",
                               max_queries: int = 20) -> Iterable[Tuple[str, List[str], List[int]]]:
    """
    HuggingFace Datasets의 MS MARCO Passage Ranking(v2.1) 을 streaming 으로 순회.
    각 (query, passages, labels) 그룹을 yield.
    """
    ds = load_dataset("ms_marco", "v2.1", split=split, streaming=True)
    count = 0
    for item in ds:
        query = item.get("query", "")
        passages = item.get("passages", {})
        if not passages or "passage_text" not in passages or "is_selected" not in passages:
            continue
        p_texts: List[str] = passages["passage_text"]
        p_labels: List[int] = [1 if b else 0 for b in passages["is_selected"]]

        # 빈 passage 제거
        keep_texts, keep_labels = [], []
        for t, r in zip(p_texts, p_labels):
            t = (t or "").strip()
            if not t:
                continue
            keep_texts.append(t)
            keep_labels.append(r)

        if not keep_texts:
            continue

        yield query, keep_texts, keep_labels

        count += 1
        if count >= max_queries:
            break


# -----------------------------
# BM25 evaluation (per-query group)
# -----------------------------
def score_group_bm25(query: str,
                     passages: List[str],
                     labels: List[int],
                     semantic: bool,
                     sem_data: Optional[Dict[str, Any]],
                     allow_duplicates: int = 2) -> Tuple[List[int], List[float], List[int], Dict[str, Any]]:
    """
    한 쿼리 그룹에 대해 BM25로 랭킹. (순정/의미 확장)
    반환:
      order: passage 인덱스의 랭킹 순서
      scores: 점수 배열 (passage 수)
      filtered_labels: forbidden 필터링 후 labels (order와 같은 정렬 전에 원본 순서)
      debug_info: 토큰 디버그 정보
    """
    # forbidden 필터링(문서 제거)
    forbidden = (sem_data or {}).get("expanded", {}).get("forbidden_terms", []) if semantic else []
    if forbidden:
        mask = [not passage_has_forbidden(p, forbidden) for p in passages]
        passages = [p for p, m in zip(passages, mask) if m]
        labels   = [r for r, m in zip(labels,   mask) if m]
        if not passages:  # 전부 필터링되면 빈 결과 반환
            return [], [], [], {}

    # BM25 인덱스 구성
    docs_tokens = [tokenize_en(p) for p in passages]
    bm25 = BM25Okapi(docs_tokens, k1=1.5, b=0.75)

    # 쿼리 토큰
    q_tokens = tokenize_en(query)
    debug_info = {}
    if semantic:
        q_tokens, debug_info = apply_semantic_to_query_tokens(q_tokens, sem_data, allow_duplicates)

    # 점수 및 랭킹
    scores = bm25.get_scores(q_tokens)
    order = list(range(len(scores)))
    order.sort(key=lambda i: scores[i], reverse=True)

    return order, list(scores), labels, debug_info


# -----------------------------
# Metrics accumulator
# -----------------------------
class Metrics:
    def __init__(self):
        self.n = 0
        self.sum_p10 = 0.0
        self.sum_r10 = 0.0
        self.sum_mrr10 = 0.0
        self.sum_ndcg10 = 0.0
        # semantic debug 정보
        self.sem_attempted = 0
        self.sem_applied = 0
        self.total_added_tokens = 0
        self.random_seed = None
        # semantic_data 통계
        self.expanded_keywords_samples = []
        self.must_include_samples = []
        self.forbidden_terms_samples = []

    def add(self, ranked_labels: List[int], sem_used: bool = False, added_tokens: int = 0, 
            expanded_keywords: List[str] = None, must_include: List[str] = None, forbidden_terms: List[str] = None):
        self.n += 1
        topk = ranked_labels[:10]
        total_rel = sum(ranked_labels)

        p10 = (sum(topk) / max(len(topk), 1)) if topk else 0.0
        r10 = (sum(topk) / max(total_rel, 1)) if total_rel > 0 else 0.0
        mrr10 = mrr_at_k(ranked_labels, 10)
        ndcg10 = ndcg_at_k(ranked_labels, 10)

        self.sum_p10 += p10
        self.sum_r10 += r10
        self.sum_mrr10 += mrr10
        self.sum_ndcg10 += ndcg10

        if sem_used:
            self.sem_applied += 1
            self.total_added_tokens += added_tokens
            
            # semantic_data 샘플 저장 (처음 5개만)
            if len(self.expanded_keywords_samples) < 5:
                if expanded_keywords:
                    self.expanded_keywords_samples.append(expanded_keywords)
                if must_include:
                    self.must_include_samples.append(must_include)
                if forbidden_terms:
                    self.forbidden_terms_samples.append(forbidden_terms)

    def add_sem_attempt(self):
        self.sem_attempted += 1

    def set_random_seed(self, seed: int):
        self.random_seed = seed

    def result(self) -> Dict[str, Any]:
        if self.n == 0:
            return {"P@10": 0.0, "R@10": 0.0, "MRR@10": 0.0, "nDCG@10": 0.0}
        
        result = {
            "P@10": self.sum_p10 / self.n,
            "R@10": self.sum_r10 / self.n,
            "MRR@10": self.sum_mrr10 / self.n,
            "nDCG@10": self.sum_ndcg10 / self.n,
        }
        
        # semantic debug 정보 추가
        if self.sem_attempted > 0:
            result["semantic_debug"] = {
                "sem_attempted": self.sem_attempted,
                "sem_applied": self.sem_applied,
                "avg_added_tokens": self.total_added_tokens / max(self.sem_applied, 1),
                "random_seed": self.random_seed,
                "expanded_keywords_samples": self.expanded_keywords_samples,
                "must_include_samples": self.must_include_samples,
                "forbidden_terms_samples": self.forbidden_terms_samples
            }
        
        return result


# -----------------------------
# Runner
# -----------------------------
def run_benchmark(semantic: bool,
                  split: str = "validation",
                  max_queries: int = 20,
                  sample_rate: float = 0.1,
                  out_dir: str = "results",
                  allow_duplicates: int = 2):
    """
    semantic=False  → BM25 순정
    semantic=True   → BM25 + 의미 확장(샘플링 비율로 Ollama 호출)
    """
    mode = "semantic" if semantic else "pure"
    save_dir = os.path.join(out_dir, mode)
    ensure_dir(save_dir)

    rankings_path = os.path.join(save_dir, "rankings.csv")
    metrics_path  = os.path.join(save_dir, "metrics.json")
    
    # semantic 데이터 저장 파일
    semantic_data_path = None
    if semantic:
        semantic_data_path = os.path.join(save_dir, "semantic_data.jsonl")

    # CSV 헤더 기록
    with open(rankings_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["query", "rank", "score", "relevant", "passage_id", "passage"])

    metrics = Metrics()
    t0 = time.time()

    n_queries = 0
    total_pairs = 0
    
    # 랜덤 시드 설정 (재현성을 위해)
    random_seed = int(time.time()) % 10000
    np.random.seed(random_seed)
    metrics.set_random_seed(random_seed)

    for q_idx, (query, passages, labels) in enumerate(iter_ms_marco_query_groups(split=split, max_queries=max_queries), start=1):
        n_queries += 1
        total_pairs += len(passages)

        sem_data = None
        sem_used = False
        if semantic and (np.random.rand() < sample_rate):
            metrics.add_sem_attempt()
            sem_data = build_semantic_data_ollama(query)
            if sem_data:
                sem_used = True

        order, scores, filt_labels, debug_info = score_group_bm25(query, passages, labels, semantic, sem_data, allow_duplicates)
        if not order:
            # 기록할 것이 없으면 다음 쿼리로
            continue

        # 평가용 라벨(랭킹 순서대로)
        ranked_labels = [filt_labels[i] for i in order]
        
        # semantic 데이터 저장
        if semantic and semantic_data_path and sem_data:
            semantic_entry = {
                "qid": q_idx,
                "query": query,
                "expanded_keywords": sem_data.get("expanded", {}).get("keywords", []),
                "must_include": sem_data.get("expanded", {}).get("must_include", []),
                "forbidden_terms": sem_data.get("expanded", {}).get("forbidden_terms", []),
                "final_tokens": debug_info.get("final_tokens", [])
            }
            with open(semantic_data_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(semantic_entry, ensure_ascii=False) + "\n")

        # 메트릭 추가
        added_tokens = len(debug_info.get("added_tokens", []))
        expanded_keywords = debug_info.get("expanded_keywords", [])
        must_include = debug_info.get("must_include", [])
        forbidden_terms = debug_info.get("forbidden_terms", [])
        
        metrics.add(ranked_labels, sem_used, added_tokens, expanded_keywords, must_include, forbidden_terms)

        # 상위 10개만 CSV 기록 (논문 부록 크기 조절용; 원하면 모두 쓰세요)
        topN = min(10, len(order))
        with open(rankings_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            for r in range(topN):
                idx = order[r]
                w.writerow([
                    query,
                    r + 1,
                    float(scores[idx]),
                    int(filt_labels[idx]),
                    idx,
                    passages[idx].replace("\n", " ").strip()
                ])

        # 개선된 로그 (샘플 3개 쿼리만)
        if q_idx <= 3:
            base_tokens = debug_info.get("base_tokens", [])
            added_tokens_list = debug_info.get("added_tokens", [])
            final_tokens = debug_info.get("final_tokens", [])
            
            print(f"[{mode}] Q{q_idx} query='{query[:60]}'")
            print(f"  base   : {base_tokens}")
            print(f"  added  : {added_tokens_list}")
            print(f"  tokens : {len(final_tokens)} (base={len(base_tokens)}, added={len(added_tokens_list)})")
            print(f"  sem_used={'Y' if sem_used else 'N'}")

    elapsed = time.time() - t0
    result_metrics = metrics.result()

    # 메타/메트릭 저장
    meta = {
        "model": "BM25 (+semantic expansion)" if semantic else "BM25 (pure)",
        "bm25_params": {"k1": 1.5, "b": 0.75},
        "data": {
            "split": split,
            "max_queries": max_queries,
            "n_queries": n_queries,
            "total_pairs": total_pairs
        },
        "metrics": result_metrics,
        "timing_sec": elapsed,
        "semantic_sample_rate": sample_rate if semantic else 0.0,
        "allow_token_duplicates": allow_duplicates
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\n[{mode}] done.  n_queries={n_queries}, total_pairs={total_pairs}, "
          f"P@10={result_metrics['P@10']:.3f}, R@10={result_metrics['R@10']:.3f}, "
          f"MRR@10={result_metrics['MRR@10']:.3f}, nDCG@10={result_metrics['nDCG@10']:.3f}, "
          f"time={elapsed:.1f}s")
    print(f" - Rankings: {rankings_path}")
    print(f" - Metrics : {metrics_path}")
    if semantic_data_path:
        print(f" - Semantic: {semantic_data_path}")
    
    return result_metrics


def print_comparison_summary(pure_metrics: Dict[str, Any], semantic_metrics: Dict[str, Any]):
    """Pure vs Semantic 모델 비교 요약 출력"""
    print("\n" + "="*60)
    print("PURE vs SEMANTIC COMPARISON SUMMARY")
    print("="*60)
    
    # metrics 구조 확인 및 수정
    if "metrics" in pure_metrics:
        pure_scores = pure_metrics["metrics"]
    else:
        pure_scores = pure_metrics
    
    if "metrics" in semantic_metrics:
        sem_scores = semantic_metrics["metrics"]
    else:
        sem_scores = semantic_metrics
    
    print(f"P@10    : {pure_scores['P@10']:.3f} → {sem_scores['P@10']:.3f} ({'↑' if sem_scores['P@10'] > pure_scores['P@10'] else '↓'}{abs(sem_scores['P@10'] - pure_scores['P@10']):.3f})")
    print(f"R@10    : {pure_scores['R@10']:.3f} → {sem_scores['R@10']:.3f} ({'↑' if sem_scores['R@10'] > pure_scores['R@10'] else '↓'}{abs(sem_scores['R@10'] - pure_scores['R@10']):.3f})")
    print(f"MRR@10  : {pure_scores['MRR@10']:.3f} → {sem_scores['MRR@10']:.3f} ({'↑' if sem_scores['MRR@10'] > pure_scores['MRR@10'] else '↓'}{abs(sem_scores['MRR@10'] - pure_scores['MRR@10']):.3f})")
    print(f"nDCG@10 : {pure_scores['nDCG@10']:.3f} → {sem_scores['nDCG@10']:.3f} ({'↑' if sem_scores['nDCG@10'] > pure_scores['nDCG@10'] else '↓'}{abs(sem_scores['nDCG@10'] - pure_scores['nDCG@10']):.3f})")
    
    if "semantic_debug" in sem_scores:
        debug = sem_scores["semantic_debug"]
        print(f"\nSemantic Expansion Stats:")
        print(f"  - Attempted: {debug['sem_attempted']}")
        print(f"  - Applied: {debug['sem_applied']}")
        print(f"  - Avg Added Tokens: {debug['avg_added_tokens']:.1f}")
    
    print("="*60)


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    # 설정값 사용
    config = EXPERIMENT_CONFIG
    
    # 1) 순정 BM25
    print("Running Pure BM25...")
    pure_metrics = run_benchmark(
        semantic=False, 
        split=config["split"], 
        max_queries=config["max_queries"], 
        out_dir=config["out_dir"],
        allow_duplicates=config["allow_token_duplicates"]
    )

    # 2) 의미 확장 BM25
    print("\nRunning Semantic BM25...")
    semantic_metrics = run_benchmark(
        semantic=True, 
        split=config["split"], 
        max_queries=config["max_queries"], 
        sample_rate=config["semantic_sample_rate"], 
        out_dir=config["out_dir"],
        allow_duplicates=config["allow_token_duplicates"]
    )
    
    # 3) 비교 요약 출력
    print_comparison_summary(pure_metrics, semantic_metrics)
    
    # (본실험) config 수정 후 실행
    # EXPERIMENT_CONFIG["max_queries"] = 500000
    # EXPERIMENT_CONFIG["split"] = "train"
    # EXPERIMENT_CONFIG["semantic_sample_rate"] = 0.05