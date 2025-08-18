import os
import json
import csv
import re
import math
import time
from typing import List, Dict, Any, Iterable, Tuple, Optional, Set

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
    "out_dir": "results"
}

# 환경변수로 실험 규모를 손쉽게 바꾸기 위한 오버라이드 (sanity check/배치 실행용)
try:
    EXPERIMENT_CONFIG["max_queries"] = int(os.getenv("OL_MAX_QUERIES", str(EXPERIMENT_CONFIG["max_queries"])))
except Exception:
    pass

# --- Semantic rerank config --- [semantic-rerank]
TOP_R_PURE = 200                # 순정 BM25 상위 후보 수
TOP_R_SEM = 1000                # semantic BM25 상위 후보 수 - Recall↑ 위해 확장
ALPHA = 0.5                    # expanded_keywords soft boost
BETA  = 1.2                    # must_include soft boost (ALPHA보다 큼)
GAMMA = 1.8                    # forbidden_terms soft penalty
DELTA = 0.6                    # anchors(구절) 일치 보너스
EXPANDED_WEIGHT = 0.4          # 확장 쿼리 가중합 계수 (Recall↑ 위해 상향)
DF_THRESH = 0.80               # 확장 커버리지↑ 위해 완화 (노이즈는 EXPANDED_WEIGHT로 제어)
IDF_MIN = 0.1                  # idf < IDF_MIN 이면 제거
MAX_EXPANDED = 8               # expanded_keywords 상한
MAX_MUST = 3                   # must_include 상한
SEMANTIC_SAMPLE_RATE = 1.0     # 의미확장 적용 비율 (예: 0.5면 절반의 쿼리만)
PHRASE_WINDOW = None           # 단순 substring 매칭이면 None
MUST_WEIGHT = 0.2              # must_include 소프트 가중 (후보 선정 이전) - Recall↑ 위해 상향
DELTA_PRE = 0.3                # anchors 사전 보너스 (후보 선정 이전)

# 환경변수로 semantic 후보폭을 조절 (sanity check 시 축소 가능)
try:
    TOP_R_SEM = int(os.getenv("OL_TOP_R_SEM", str(TOP_R_SEM)))
except Exception:
    pass

# BM25 파라미터 [semantic-rerank]
K1 = 1.5
B = 0.75


# -----------------------------
# Utilities
# -----------------------------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def _key(q: str) -> str:
    """캐시 키 정규화: 공백 정리 및 소문자화"""
    return re.sub(r"\s+", " ", (q or "").strip().lower())

def tokenize_en(text: str) -> List[str]:
    text = re.sub(r"[^\w\s]", " ", text.lower())
    toks = text.split()
    # 길이 2 이하 토큰은 잡음으로 제거
    return [t for t in toks if len(t) > 2]

def normalize_terms(terms: List[str]) -> List[str]:
    """간단한 정규화: 소문자화 및 중복 제거"""
    normalized = []
    seen = set()
    for term in terms:
        term = term.lower().strip()
        if term and term not in seen:
            normalized.append(term)
            seen.add(term)
    return normalized

def dedup(terms: List[str]) -> List[str]:
    """중복 제거"""
    return list(dict.fromkeys(terms))

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

def map_at_k(ranked_rel: List[int], k: int = 100) -> float:
    k = min(k, len(ranked_rel))
    if sum(ranked_rel) == 0:
        return 0.0
    
    precision_sum = 0.0
    relevant_count = 0
    
    for i in range(k):
        if ranked_rel[i] == 1:
            relevant_count += 1
            precision_sum += relevant_count / (i + 1)
    
    return precision_sum / sum(ranked_rel)


# -----------------------------
# Index/Statistics Preparation [semantic-rerank]
# -----------------------------
def build_idf_dict_from_df(df_dict: Dict[str, int], N: int) -> Dict[str, float]:
    """DF 사전으로부터 BM25와 동일식으로 IDF 계산"""
    idf_dict: Dict[str, float] = {}
    for token, df in df_dict.items():
        # BM25Okapi와 동일 공식
        idf = math.log((N - df + 0.5) / (df + 0.5))
        idf_dict[token] = idf
    return idf_dict

def build_doc_token_index(passages: List[str], tokenizer) -> Dict[int, Tuple[Set[str], str]]:
    """문서 토큰셋/원문 캐시 구축"""
    doc_index = {}
    for doc_id, passage in enumerate(passages):
        tokens = set(tokenizer(passage))
        doc_index[doc_id] = (tokens, passage)
    return doc_index

def build_df_dict(passages: List[str], tokenizer) -> Dict[str, int]:
    """문서 빈도(DF) 사전 구축"""
    df_dict = {}
    for passage in passages:
        tokens = set(tokenizer(passage))
        for token in tokens:
            df_dict[token] = df_dict.get(token, 0) + 1
    return df_dict


# -----------------------------
# Semantic Expansion (Ollama gemma3) [semantic-rerank]
# -----------------------------
def build_semantic_data_ollama(query: str,
                               host: str = "http://192.168.45.166:11434",
                               model: str = "gemma3",
                               timeout: int = 15,
                               retries: int = 2) -> Optional[Dict[str, Any]]:
    """
    Ollama 서버(gemma3)에 프롬프트를 보내 semantic_data 생성.
    반환 형식:
    {
      "user_query": str,
      "intent_data": {"language":"en"},
      "expanded": {
        "keywords": [...],
        "must_include": [...],
        "forbidden_terms": [...],
        "anchors": [...]
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
    "forbidden_terms": ["cryptocurrency","Bitcoin"],
    "anchors": ["Federal Reserve", "interest rate changes"]
  }}
}}

Now produce semantic_data for this query:
"{query}"
""".strip()

    for attempt in range(retries + 1):
        try:
            # Ollama /api/generate (temperature=0으로 고정)
            resp = requests.post(
                f"{host}/api/generate",
                json={"model": model, "prompt": prompt, "stream": False, "temperature": 0},
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
                    "forbidden_terms": exp.get("forbidden_terms", []),
                    "anchors": exp.get("anchors", [])
                }
            }
        except Exception:
            if attempt == retries:
                return None
            time.sleep(0.5 * (attempt + 1))  # 짧은 백오프


def filter_semantic_terms(terms: List[str], df_dict: Dict[str, int], N: int, idf: Dict[str, float], is_must: bool = False) -> List[str]:
    """의미확장 데이터 전처리 필터(DF/IDF) & 상한 적용 [semantic-rerank]"""
    kept = []
    for t in normalize_terms(terms):
        if (df_dict.get(t, 0) / N) <= DF_THRESH and idf.get(t, 0.0) >= IDF_MIN:
            kept.append(t)
    limit = MAX_MUST if is_must else MAX_EXPANDED
    return dedup(kept)[:limit]


def soft_semantic_score(doc_tokens: Set[str], doc_text: str, idf: Dict[str, float], 
                       expanded: List[str], must_inc: List[str], forbid: List[str], anchors: List[str]) -> float:
    """재랭크 점수식 구현 (중복 토큰 주입 금지) [semantic-rerank]"""
    s = 0.0
    for t in expanded:
        if t in doc_tokens: 
            s += ALPHA * idf.get(t, 0.0)
    for t in must_inc:
        if t in doc_tokens: 
            s += BETA * idf.get(t, 0.0)
    for t in forbid:
        if t in doc_tokens: 
            s -= GAMMA * idf.get(t, 0.0)
    if anchors:
        for ph in anchors:
            if ph and ph.lower() in doc_text.lower():
                s += DELTA
    return s


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
# BM25 evaluation (per-query group) [semantic-rerank]
# -----------------------------
def score_group_bm25_rerank(query: str,
                           passages: List[str],
                           labels: List[int],
                           semantic: bool,
                           sem_data: Optional[Dict[str, Any]],
                           df_dict: Dict[str, int],
                           idf: Dict[str, float],
                           doc_index: Dict[int, Tuple[Set[str], str]]) -> Tuple[List[int], List[float], List[int], Dict[str, Any]]:
    """
    한 쿼리 그룹에 대해 BM25 + 재랭크로 랭킹. [semantic-rerank]
    반환:
      order: passage 인덱스의 랭킹 순서
      scores: 점수 배열 (passage 수)
      labels: 원본 labels
      debug_info: 디버그 정보
    """
    N = len(passages)
    
    # 1차: 순정 BM25로 후보 뽑기 (semantic 여부에 따라 다른 TOP_R 적용)
    docs_tokens = [tokenize_en(p) for p in passages]
    bm25 = BM25Okapi(docs_tokens, k1=K1, b=B)
    
    # 쿼리 토큰 (순정 토크나이저 결과만 사용)
    q_tokens = tokenize_en(query)
    base_scores = bm25.get_scores(q_tokens)
    
    # semantic 모드일 때만 확장 쿼리 가중합 적용
    if semantic and sem_data:
        exp = sem_data.get("expanded", {})
        expanded = filter_semantic_terms(exp.get("keywords", []), df_dict, N, idf, False)
        must_inc = filter_semantic_terms(exp.get("must_include", []), df_dict, N, idf, True)
        forbidden = normalize_terms(exp.get("forbidden_terms", []))
        anchors = normalize_terms(exp.get("anchors", []))

        # 확장된 쿼리 생성 (중복 제거)
        q_tokens_expanded = list(set(q_tokens + must_inc + expanded))
        if q_tokens_expanded:
            alt_scores = bm25.get_scores(q_tokens_expanded)
            base_scores = [base + alt * EXPANDED_WEIGHT for base, alt in zip(base_scores, alt_scores)]

        # must_include-only 보정(소프트) → 후보 선정 이전
        if must_inc:
            must_scores = bm25.get_scores(must_inc)
            base_scores = [s + MUST_WEIGHT * ms for s, ms in zip(base_scores, must_scores)]

        # anchors 사전 보너스(소프트) → 후보 선정 이전
        if anchors:
            low_anchors = [a for a in anchors if a]
            if low_anchors:
                for i, (_, doc_text) in doc_index.items():
                    lt = doc_text.lower()
                    if any(a in lt for a in low_anchors):
                        base_scores[i] += DELTA_PRE
    
    # TOP_R 후보 추출 (semantic 여부에 따라 다름)
    top_r = TOP_R_SEM if semantic else TOP_R_PURE
    candidates = list(range(len(base_scores)))
    candidates.sort(key=lambda i: base_scores[i], reverse=True)
    top_candidates = candidates[:top_r]
    
    debug_info: Dict[str, Any] = {"semantic_applied": False, "filtered_terms": {}, "skipped_by_dfidf": False}
    
    # 2차: semantic=True이고 샘플링에 걸린 쿼리만 재랭크 적용
    if semantic and sem_data:
        # 위에서 계산한 expanded/must_inc/forbidden/anchors를 재사용
        # (없다면 안전하게 다시 계산)
        try:
            expanded
        except NameError:
            exp = sem_data.get("expanded", {})
            expanded = filter_semantic_terms(exp.get("keywords", []), df_dict, N, idf, False)
            must_inc = filter_semantic_terms(exp.get("must_include", []), df_dict, N, idf, True)
            forbidden = normalize_terms(exp.get("forbidden_terms", []))
            anchors = normalize_terms(exp.get("anchors", []))
        
        debug_info["filtered_terms"] = {
            "expanded": expanded,
            "must_include": must_inc,
            "forbidden": forbidden,
            "anchors": anchors
        }
        
        # DF/IDF 필터로 적용할 게 없으면 스킵
        if not expanded and not must_inc and not anchors and not forbidden:
            debug_info["skipped_by_dfidf"] = True
        else:
            debug_info["semantic_applied"] = True
            # 각 후보 문서에 soft_semantic_score를 더해 재정렬 (하드 필터 아님)
            reranked_candidates = []
            for doc_id in top_candidates:
                doc_tokens, doc_text = doc_index[doc_id]
                semantic_bonus = soft_semantic_score(doc_tokens, doc_text, idf, expanded, must_inc, forbidden, anchors)
                final_score = base_scores[doc_id] + semantic_bonus
                reranked_candidates.append((doc_id, final_score))
            
            # 재랭크된 후보들 정렬
            reranked_candidates.sort(key=lambda x: x[1], reverse=True)
            top_candidates = [doc_id for doc_id, _ in reranked_candidates]
    
    # 재랭크 직후에 추가
    final_score_map = {}
    if semantic and sem_data and reranked_candidates:
        final_score_map = {doc_id: s for doc_id, s in reranked_candidates}
    
    # 3차: 최종 top-100에서 메트릭 계산
    final_order = top_candidates[:100]  # top-100으로 제한
    # 최종 점수는 재랭크 점수 우선, 없으면 base_scores
    final_scores = [
        (final_score_map[i] if i in final_score_map else float(base_scores[i]))
        for i in final_order
    ]
    final_labels = [labels[i] for i in final_order]
    
    return final_order, final_scores, final_labels, debug_info


# -----------------------------
# Metrics accumulator [semantic-rerank]
# -----------------------------
class Metrics:
    def __init__(self):
        self.n = 0
        self.sum_p1 = 0.0
        self.sum_p10 = 0.0
        self.sum_r10 = 0.0
        self.sum_mrr10 = 0.0
        self.sum_ndcg10 = 0.0
        self.sum_ndcg100 = 0.0
        self.sum_map100 = 0.0
        # semantic debug 정보
        self.sem_attempted = 0
        self.sem_applied = 0
        self.skipped_by_dfidf = 0

    def add(self, ranked_labels: List[int], total_rel_all: int, sem_applied: bool = False):
        self.n += 1
        topk = ranked_labels[:10]
        # 주의: recall의 분모는 그룹 전체의 관련 문서 수여야 함
        total_rel = max(int(total_rel_all), 0)

        p1 = ranked_labels[0] if ranked_labels else 0.0
        p10 = (sum(topk) / max(len(topk), 1)) if topk else 0.0
        r10 = (sum(topk) / max(total_rel, 1)) if total_rel > 0 else 0.0
        mrr10 = mrr_at_k(ranked_labels, 10)
        ndcg10 = ndcg_at_k(ranked_labels, 10)
        ndcg100 = ndcg_at_k(ranked_labels, 100)
        map100 = map_at_k(ranked_labels, 100)

        self.sum_p1 += p1
        self.sum_p10 += p10
        self.sum_r10 += r10
        self.sum_mrr10 += mrr10
        self.sum_ndcg10 += ndcg10
        self.sum_ndcg100 += ndcg100
        self.sum_map100 += map100

        if sem_applied:
            self.sem_applied += 1

    def add_sem_attempt(self):
        self.sem_attempted += 1

    def add_skipped_by_dfidf(self):
        self.skipped_by_dfidf += 1

    def result(self) -> Dict[str, float]:
        if self.n == 0:
            return {
                "P@1": 0.0, "P@10": 0.0, "R@10": 0.0, "MRR@10": 0.0, 
                "nDCG@10": 0.0, "nDCG@100": 0.0, "MAP@100": 0.0
            }
        
        return {
            "P@1": self.sum_p1 / self.n,
            "P@10": self.sum_p10 / self.n,
            "R@10": self.sum_r10 / self.n,
            "MRR@10": self.sum_mrr10 / self.n,
            "nDCG@10": self.sum_ndcg10 / self.n,
            "nDCG@100": self.sum_ndcg100 / self.n,
            "MAP@100": self.sum_map100 / self.n,
        }


# -----------------------------
# Runner [semantic-rerank]
# -----------------------------
def run_benchmark(semantic: bool,
                  split: str = "validation",
                  max_queries: int = 20,
                  out_dir: str = "results"):
    """
    semantic=False  → BM25 순정
    semantic=True   → BM25 + 의미 확장 재랭크
    """
    mode = "semantic" if semantic else "pure"
    save_dir = os.path.join(out_dir, mode)
    ensure_dir(save_dir)

    rankings_path = os.path.join(save_dir, "rankings.csv")
    metrics_path  = os.path.join(save_dir, "metrics.json")
    
    # semantic 데이터 저장 파일
    semantic_data_path = None
    cache_path = None
    cache: Dict[str, Any] = {}
    hits = misses = 0
    save_every = 20  # 20개마다 중간 저장
    
    if semantic:
        semantic_data_path = os.path.join(save_dir, "semantic_data.jsonl")
        # 캐시는 .cache 폴더에 저장
        cache_dir = os.path.join(out_dir, ".cache")
        ensure_dir(cache_dir)
        cache_path = os.path.join(cache_dir, "ollama_cache.json")
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as cf:
                    cache = json.load(cf)
            except Exception:
                cache = {}
        
        # semantic_data.jsonl → 캐시로 프리로드
        if os.path.exists(semantic_data_path):
            try:
                with open(semantic_data_path, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            row = json.loads(line)
                        except Exception:
                            continue
                        q = row.get("query", "")
                        if not q:
                            continue
                        k = _key(q)
                        if k not in cache:
                            # jsonl에는 필터링된 필드만 있으니 캐시 구조에 맞춰 래핑
                            cache[k] = {
                                "user_query": q,
                                "intent_data": {"language": "en"},
                                "expanded": {
                                    "keywords": row.get("expanded_keywords", []),
                                    "must_include": row.get("must_include", []),
                                    "forbidden_terms": row.get("forbidden_terms", []),
                                    "anchors": row.get("anchors", []),
                                }
                            }
            except Exception:
                pass

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

    for q_idx, (query, passages, labels) in enumerate(iter_ms_marco_query_groups(split=split, max_queries=max_queries), start=1):
        n_queries += 1
        total_pairs += len(passages)

        # 통계/인덱스 준비
        df_dict = build_df_dict(passages, tokenize_en)
        idf = build_idf_dict_from_df(df_dict, len(passages))
        doc_index = build_doc_token_index(passages, tokenize_en)

        sem_data = None
        debug_info: Dict[str, Any] = {}
        if semantic and (np.random.rand() < SEMANTIC_SAMPLE_RATE):
            metrics.add_sem_attempt()
            # 캐시 우선 (정규화된 키 사용)
            key = _key(query)
            if key in cache:
                hits += 1
                sem_data = cache[key]
            else:
                misses += 1
                sem_data = build_semantic_data_ollama(query)
                if sem_data:
                    cache[key] = sem_data

        order, scores, final_labels, debug_info = score_group_bm25_rerank(
            query, passages, labels, semantic, sem_data, df_dict, idf, doc_index
        )
        
        if not order:
            continue

        # semantic 데이터 저장
        if semantic and semantic_data_path and sem_data:
            filtered = debug_info.get("filtered_terms", {})
            semantic_entry = {
                "qid": q_idx,
                "query": query,
                "expanded_keywords": filtered.get("expanded", []),
                "must_include": filtered.get("must_include", []),
                "forbidden_terms": filtered.get("forbidden", []),
                "anchors": filtered.get("anchors", [])
            }
            with open(semantic_data_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(semantic_entry, ensure_ascii=False) + "\n")

        # 메트릭 추가 및 스킵 집계
        if debug_info.get("skipped_by_dfidf"):
            metrics.add_skipped_by_dfidf()
        metrics.add(final_labels, total_rel_all=sum(labels), sem_applied=debug_info.get("semantic_applied", False))

        # 중간 저장 (주기적으로)
        if semantic and (q_idx % save_every == 0) and cache_path is not None:
            try:
                with open(cache_path, "w", encoding="utf-8") as cf:
                    json.dump(cache, cf, ensure_ascii=False, indent=2)
            except Exception:
                pass

        # 상위 10개만 CSV 기록
        topN = min(10, len(order))
        with open(rankings_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            for r in range(topN):
                idx = order[r]
                w.writerow([
                    query,
                    r + 1,
                    float(scores[r]),
                    int(final_labels[r]),
                    idx,
                    passages[idx].replace("\n", " ").strip()
                ])

        # 간단 로그 (샘플 3개 쿼리만)
        if q_idx <= 3:
            sem_used = 'Y' if debug_info.get("semantic_applied", False) else ('-' if not sem_data else 'N')
            print(f"[{mode}] Q{q_idx} query='{query[:60]}' sem_used={sem_used} "
                  f"top_r={TOP_R_SEM if semantic else TOP_R_PURE} "
                  f"w_exp={EXPANDED_WEIGHT} w_must={MUST_WEIGHT}")

    elapsed = time.time() - t0
    result_metrics = metrics.result()

    # 메타/메트릭 저장 (샘플 문자열 제거)
    meta = {
        "model": "BM25 (+semantic rerank)" if semantic else "BM25 (pure)",
        "bm25_params": {"k1": K1, "b": B},
        "data": {
            "split": split,
            "max_queries": max_queries,
            "n_queries": n_queries,
            "total_pairs": total_pairs
        },
        "metrics": result_metrics,
        "timing_sec": elapsed,
        "semantic_sample_rate": SEMANTIC_SAMPLE_RATE if semantic else 0.0,
        "top_r": TOP_R_SEM if semantic else TOP_R_PURE,
        "params": {
            "ALPHA": ALPHA,
            "BETA": BETA,
            "GAMMA": GAMMA,
            "DELTA": DELTA,
            "DF_THRESH": DF_THRESH,
            "IDF_MIN": IDF_MIN,
            "MAX_EXPANDED": MAX_EXPANDED,
            "MAX_MUST": MAX_MUST,
            "EXPANDED_WEIGHT": EXPANDED_WEIGHT,
            "MUST_WEIGHT": MUST_WEIGHT,
            "DELTA_PRE": DELTA_PRE
        }
    }
    
    # semantic debug 정보 추가 (숫자만)
    if semantic and (metrics.sem_attempted > 0 or metrics.sem_applied > 0 or metrics.skipped_by_dfidf > 0):
        meta["semantic_debug"] = {
            "sem_attempted": metrics.sem_attempted,
            "sem_applied": metrics.sem_applied,
            "skipped_by_dfidf": metrics.skipped_by_dfidf
        }
    
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # 캐시 저장
    if semantic and cache_path is not None:
        try:
            with open(cache_path, "w", encoding="utf-8") as cf:
                json.dump(cache, cf, ensure_ascii=False, indent=2)
        except Exception:
            pass

    # 콘솔 요약 (간단)
    print(f"\n[{mode}] done.  n_queries={n_queries}, total_pairs={total_pairs}")
    print(f"  P@1={result_metrics['P@1']:.3f}, P@10={result_metrics['P@10']:.3f}, MRR@10={result_metrics['MRR@10']:.3f}")
    print(f"  nDCG@10={result_metrics['nDCG@10']:.3f}, R@10={result_metrics['R@10']:.3f}, time={elapsed:.1f}s")
    if semantic:
        dbg = meta.get("semantic_debug", {"sem_applied":0, "sem_attempted":0, "skipped_by_dfidf":0})
        print(f"  sem_applied={dbg['sem_applied']}/{dbg['sem_attempted']}, skipped_by_dfidf={dbg['skipped_by_dfidf']}")
        print(f"  cache_hits={hits}, cache_misses={misses}, cache_size={len(cache)}")
    print(f" - Rankings: {rankings_path}")
    print(f" - Metrics : {metrics_path}")
    if semantic and semantic_data_path:
        print(f" - Semantic: {semantic_data_path}")
    
    return meta


def print_comparison_summary(pure_metrics: Dict[str, Any], semantic_metrics: Dict[str, Any]):
    """Pure vs Semantic 모델 비교 요약 출력 [semantic-rerank]"""
    print("\n" + "="*60)
    print("PURE vs SEMANTIC COMPARISON SUMMARY")
    print("="*60)
    
    pure_scores = pure_metrics.get("metrics", pure_metrics)
    sem_scores = semantic_metrics.get("metrics", semantic_metrics)
    
    print(f"P@1     : {pure_scores['P@1']:.3f} → {sem_scores['P@1']:.3f} ({'↑' if sem_scores['P@1'] > pure_scores['P@1'] else '↓'}{abs(sem_scores['P@1'] - pure_scores['P@1']):.3f})")
    print(f"P@10    : {pure_scores['P@10']:.3f} → {sem_scores['P@10']:.3f} ({'↑' if sem_scores['P@10'] > pure_scores['P@10'] else '↓'}{abs(sem_scores['P@10'] - pure_scores['P@10']):.3f})")
    print(f"MRR@10  : {pure_scores['MRR@10']:.3f} → {sem_scores['MRR@10']:.3f} ({'↑' if sem_scores['MRR@10'] > pure_scores['MRR@10'] else '↓'}{abs(sem_scores['MRR@10'] - pure_scores['MRR@10']):.3f})")
    print(f"nDCG@10 : {pure_scores['nDCG@10']:.3f} → {sem_scores['nDCG@10']:.3f} ({'↑' if sem_scores['nDCG@10'] > pure_scores['nDCG@10'] else '↓'}{abs(sem_scores['nDCG@10'] - pure_scores['nDCG@10']):.3f})")
    print(f"R@10    : {pure_scores['R@10']:.3f} → {sem_scores['R@10']:.3f} ({'↑' if sem_scores['R@10'] > pure_scores['R@10'] else '↓'}{abs(sem_scores['R@10'] - pure_scores['R@10']):.3f})")
    
    if "semantic_debug" in semantic_metrics:
        debug = semantic_metrics["semantic_debug"]
        print(f"\nSemantic Rerank Stats:")
        print(f"  - Attempted: {debug['sem_attempted']}")
        print(f"  - Applied: {debug['sem_applied']}")
        print(f"  - Skipped by DF/IDF: {debug['skipped_by_dfidf']}")
    
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
        out_dir=config["out_dir"]
    )

    # 2) 의미 확장 BM25 (재랭크)
    print("\nRunning Semantic BM25 (Rerank)...")
    semantic_metrics = run_benchmark(
        semantic=True, 
        split=config["split"], 
        max_queries=config["max_queries"], 
        out_dir=config["out_dir"]
    )
    
    # 3) 비교 요약 출력
    print_comparison_summary(pure_metrics, semantic_metrics)
    
    # (본실험) config 수정 후 실행
    # EXPERIMENT_CONFIG["max_queries"] = 500000
    # EXPERIMENT_CONFIG["split"] = "train"
    # SEMANTIC_SAMPLE_RATE = 0.05