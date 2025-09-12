# k_filter.py - 공통 키워드 필터링 모듈
import os
import json
import re
import math
import time
from typing import List, Dict, Any, Tuple, Optional, Set, Iterable

import numpy as np
from rank_bm25 import BM25Okapi
import requests


# -----------------------------
# Configuration
# -----------------------------
class KeywordFilterConfig:
    """키워드 필터링 설정 클래스"""
    
    def __init__(self, **kwargs):
        # 기본 설정
        self.max_queries = kwargs.get("max_queries", 20)
        self.semantic_sample_rate = kwargs.get("semantic_sample_rate", 1.0)
        
        # 확장 키워드 설정
        self.max_expanded = kwargs.get("max_expanded", 8)
        self.df_thresh = kwargs.get("df_thresh", 0.95)  # MS MARCO 방식 기본값
        self.idf_min = kwargs.get("idf_min", 0.05)      # MS MARCO 방식 기본값
        
        # 재랭킹 방식 설정
        self.adaptive_mode = kwargs.get("adaptive_mode", True)
        self.candidate_threshold = kwargs.get("candidate_threshold", 1000)
        
        # 후보 기반 설정 (FiQA 스타일) - Recall 중심 튜닝
        self.top_r_pure = kwargs.get("top_r_pure", 300)  # 200->300: 더 많은 후보 유지
        self.alpha_soft_bonus = kwargs.get("alpha_soft_bonus", 0.3)  # 0.5->0.3: 보수적 보너스
        self.anchor_k = kwargs.get("anchor_k", 5)  # 3->5: 더 많은 앵커 보호
        self.rrf_k = kwargs.get("rrf_k", 40.0)  # 60->40: RRF 가중치 증가
        
        # 전체 문서 설정 (MS MARCO 스타일) - Recall 중심 튜닝
        self.top_r_sem = kwargs.get("top_r_sem", 3000)  # 2000->3000: 더 많은 후보 유지
        self.expanded_weight = kwargs.get("expanded_weight", 0.25)  # 0.35->0.25: 보수적 확장
        self.alpha_bonus = kwargs.get("alpha_bonus", 0.3)  # 0.45->0.3: 보수적 보너스
        # SAFE-DROP 제거됨 - FN 최소화를 위해
        
        # 공통 안전장치 (Recall 중심 튜닝)
        self.guardrail_k = kwargs.get("guardrail_k", 200)  # 100->200: 더 많은 관련 문서 보호
        self.enable_prf_fallback = kwargs.get("enable_prf_fallback", True)
        
        # BM25 파라미터
        self.k1 = kwargs.get("k1", 1.5)
        self.b = kwargs.get("b", 0.75)
        
        # LLM 설정
        self.ollama_host = kwargs.get("ollama_host", "http://192.168.45.166:11434")
        self.ollama_model = kwargs.get("ollama_model", "gemma3")
        self.ollama_timeout = kwargs.get("ollama_timeout", 15)
        self.ollama_retries = kwargs.get("ollama_retries", 2)


# -----------------------------
# Utilities
# -----------------------------
def ensure_dir(p: str):
    """디렉토리 생성"""
    os.makedirs(p, exist_ok=True)

def _key(q: str) -> str:
    """쿼리 정규화 키 생성"""
    return re.sub(r"\s+", " ", (q or "").strip().lower())

def tokenize_en(text: str) -> List[str]:
    """영어 텍스트 토큰화"""
    text = re.sub(r"[^\w\s]", " ", (text or "").lower())
    toks = text.split()
    return [t for t in toks if len(t) > 2]

def normalize_terms(terms: List[str]) -> List[str]:
    """용어 정규화 및 중복 제거"""
    out, seen = [], set()
    for term in terms or []:
        t = (term or "").lower().strip()
        if t and t not in seen:
            out.append(t)
            seen.add(t)
    return out

def dedup(seq: List[str]) -> List[str]:
    """시퀀스 중복 제거 (순서 유지)"""
    return list(dict.fromkeys(seq or []))

def terms_to_tokens(terms: List[str]) -> List[str]:
    """문구 리스트를 토큰 리스트로 평탄화"""
    toks: List[str] = []
    for phrase in normalize_terms(terms or []):
        toks.extend(tokenize_en(phrase))
    return dedup(toks)


# -----------------------------
# Metrics
# -----------------------------
def ndcg_at_k(ranked_rel: List[int], k: int = 10) -> float:
    """nDCG@k 계산"""
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
    """MRR@k 계산"""
    k = min(k, len(ranked_rel))
    for i in range(k):
        if ranked_rel[i] == 1:
            return 1.0 / (i + 1)
    return 0.0

def map_at_k(ranked_rel: List[int], k: int = 100) -> float:
    """MAP@k 계산"""
    k = min(k, len(ranked_rel))
    if sum(ranked_rel) == 0:
        return 0.0
    precision_sum, rel_count = 0.0, 0
    for i in range(k):
        if ranked_rel[i] == 1:
            rel_count += 1
            precision_sum += rel_count / (i + 1)
    return precision_sum / max(1, sum(ranked_rel))


# -----------------------------
# Index/Statistics
# -----------------------------
def build_idf_dict_from_df(df_dict: Dict[str, int], N: int) -> Dict[str, float]:
    """DF 딕셔너리로부터 IDF 딕셔너리 생성"""
    idf_dict: Dict[str, float] = {}
    for token, df in df_dict.items():
        idf = math.log((N - df + 0.5) / (df + 0.5))
        idf_dict[token] = idf
    return idf_dict

def build_doc_token_index(passages: List[str], tokenizer=tokenize_en) -> Dict[int, Tuple[Set[str], str]]:
    """문서 토큰 인덱스 구축"""
    doc_index = {}
    for doc_id, passage in enumerate(passages):
        tokens = set(tokenizer(passage))
        doc_index[doc_id] = (tokens, passage or "")
    return doc_index

def build_df_dict(passages: List[str], tokenizer=tokenize_en) -> Dict[str, int]:
    """문서 빈도(DF) 딕셔너리 구축"""
    df_dict = {}
    for passage in passages:
        tokens = set(tokenizer(passage))
        for token in tokens:
            df_dict[token] = df_dict.get(token, 0) + 1
    return df_dict


# -----------------------------
# Semantic Expansion
# -----------------------------
def build_semantic_data_ollama(query: str,
                               host: str = "http://192.168.45.166:11434",
                               model: str = "gemma3",
                               timeout: int = 15,
                               retries: int = 2) -> Optional[Dict[str, Any]]:
    """LLM을 통한 시맨틱 확장 데이터 생성"""
    prompt = f"""
Return ONLY a valid JSON object (no code fences). You expand the user query for information retrieval.

Schema:
{{
  "user_query": "<original query>",
  "expanded": {{
    "keywords": ["6-10 short distinct phrases, 2-4 words each, lowercase"]
  }}
}}

Rules:
- expanded.keywords must have 6-10 items.
- Each phrase <=4 words, distinct, lowercase.
- No filler words (e.g., "and", "from"), no duplicates.
- No additional fields or commentary.

Now output JSON for:
"{query}"
""".strip()

    for attempt in range(retries + 1):
        try:
            resp = requests.post(
                f"{host}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0}
                },
                timeout=timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            text = (data.get("response") or "").strip()
            
            # 안전한 JSON 추출
            first, last = text.find("{"), text.rfind("}")
            if first == -1 or last == -1:
                continue
            obj = json.loads(text[first:last+1])
            expanded = obj.get("expanded", {}) if isinstance(obj.get("expanded"), dict) else {}
            return {
                "query": obj.get("user_query", query),
                "expanded": {"keywords": expanded.get("keywords", []) or []}
            }
        except Exception:
            if attempt == retries:
                return None
            time.sleep(0.4 * (attempt + 1))
    return None


# -----------------------------
# Query Expansion Utils
# -----------------------------
def filter_semantic_terms(terms: List[str], df_dict: Dict[str, int], N: int, 
                         idf: Dict[str, float], config: KeywordFilterConfig) -> List[str]:
    """확장 키워드 필터링"""
    kept = []
    for t in normalize_terms(terms):
        if (df_dict.get(t, 0) / max(1, N)) <= config.df_thresh and idf.get(t, 0.0) >= config.idf_min:
            kept.append(t)
    return dedup(kept)[:config.max_expanded]

def extract_expanded(sem_data: Optional[Dict[str, Any]],
                     df_dict: Dict[str, int], N: int, idf: Dict[str, float],
                     config: KeywordFilterConfig) -> List[str]:
    """시맨틱 데이터에서 확장 키워드 추출"""
    if not sem_data:
        return []
    ek = []
    if isinstance(sem_data.get("expanded"), dict):
        ek = sem_data["expanded"].get("keywords", [])
    if not ek:
        ek = sem_data.get("expanded_keywords", [])  # legacy fallback
    return filter_semantic_terms(ek, df_dict, N, idf, config)

def soft_semantic_bonus_tokens(doc_tokens: Set[str], idf: Dict[str, float], 
                              expanded_terms: List[str], config: KeywordFilterConfig) -> float:
    """확장 키워드 소프트 보너스 계산"""
    s = 0.0
    exp_tokens = set()
    for phrase in expanded_terms:
        exp_tokens.update(tokenize_en(phrase))
    for t in exp_tokens:
        if t in doc_tokens:
            s += config.alpha_bonus * idf.get(t, 0.0)
    return s


# -----------------------------
# PRF Fallback
# -----------------------------
def prf_rm3_terms(passages: List[str], base_scores, tokenizer=tokenize_en,
                  top_m: int = 20, top_terms: int = 8, min_len: int = 3) -> List[str]:
    """PRF(RM3-lite) fallback 확장 키워드 생성"""
    cand_idx = list(range(len(passages)))
    cand_idx.sort(key=lambda i: float(base_scores[i]), reverse=True)
    cand_idx = cand_idx[:min(top_m, len(cand_idx))]

    df = {}
    for i in cand_idx:
        toks = set([t for t in tokenizer(passages[i]) if len(t) >= min_len])
        for t in toks:
            df[t] = df.get(t, 0) + 1

    return [t for t, _ in sorted(df.items(), key=lambda x: x[1], reverse=True)[:top_terms]]


# -----------------------------
# Core Reranking Functions
# -----------------------------
def candidate_based_rerank(query: str, documents: List[str], base_scores: np.ndarray,
                          semantic_data: Optional[Dict[str, Any]], config: KeywordFilterConfig) -> Tuple[List[int], List[float], Dict[str, Any]]:
    """후보 기반 재랭킹 (FiQA 스타일)"""
    N = len(documents)
    
    # 상위 후보 선택
    cand_idx = list(range(N))
    cand_idx.sort(key=lambda i: base_scores[i], reverse=True)
    cand_idx = cand_idx[:min(config.top_r_pure, N)]
    
    # 후보 기반 DF/IDF
    cand_texts = [documents[i] for i in cand_idx]
    df_dict = build_df_dict(cand_texts, tokenize_en)
    idf = build_idf_dict_from_df(df_dict, len(cand_texts))
    doc_index = build_doc_token_index(cand_texts, tokenize_en)
    
    # 확장 키워드 추출
    expanded = extract_expanded(semantic_data, df_dict, len(cand_texts), idf, config)
    
    # PRF fallback
    if not expanded and config.enable_prf_fallback:
        prf_terms = prf_rm3_terms(cand_texts, np.array([base_scores[i] for i in cand_idx]))
        expanded = filter_semantic_terms(prf_terms, df_dict, len(cand_texts), idf, config)
    
    # 소프트 보너스 재랭킹
    pairs: List[Tuple[int, float]] = []
    for loc, (_tokset, _passage) in doc_index.items():
        s = float(base_scores[cand_idx[loc]])
        for t in expanded:
            if t in _tokset:
                s += config.alpha_soft_bonus * idf.get(t, 0.0)
        pairs.append((loc, s))
    
    # 정렬
    pairs.sort(key=lambda x: x[1], reverse=True)
    
    # RRF 융합
    base_rank_map = {
        loc: rank for rank, loc in enumerate(
            sorted(range(len(cand_idx)),
                   key=lambda i: base_scores[cand_idx[i]], reverse=True),
            start=1
        )
    }
    tmp = []
    for loc, s_sem in pairs:
        r_base = base_rank_map[loc]
        s_rrf = s_sem + 1.0 / (config.rrf_k + r_base)
        tmp.append((loc, s_rrf))
    tmp.sort(key=lambda x: x[1], reverse=True)
    
    # 앵커 보호
    anchor_locs = sorted(range(len(cand_idx)),
                         key=lambda i: base_scores[cand_idx[i]], reverse=True)[:config.anchor_k]
    anchor_set = set(anchor_locs)
    tail = [p for p in tmp if p[0] not in anchor_set]
    final_pairs = [(loc, float(base_scores[cand_idx[loc]])) for loc in anchor_locs] + tail
    
    final_local_order = [loc for loc, _ in final_pairs]
    final_global_order = [cand_idx[loc] for loc in final_local_order]
    final_scores = [float(base_scores[i]) for i in final_global_order]
    
    debug_info = {
        "semantic_applied": bool(expanded),
        "filtered_terms": {"expanded": expanded},
        "skipped_by_dfidf": not bool(expanded),
        "mode": "candidate_based"
    }
    
    return final_global_order, final_scores, debug_info


def full_document_rerank(query: str, documents: List[str], base_scores: np.ndarray,
                        semantic_data: Optional[Dict[str, Any]], config: KeywordFilterConfig) -> Tuple[List[int], List[float], Dict[str, Any]]:
    """전체 문서 기반 재랭킹 (MS MARCO 스타일)"""
    N = len(documents)
    
    # 전체 문서 DF/IDF
    df_dict = build_df_dict(documents, tokenize_en)
    idf = build_idf_dict_from_df(df_dict, N)
    doc_index = build_doc_token_index(documents, tokenize_en)
    
    # BM25 인덱스
    docs_tokens = [tokenize_en(p) for p in documents]
    bm25 = BM25Okapi(docs_tokens, k1=config.k1, b=config.b)
    
    # 순정 BM25 점수
    q_tokens = tokenize_en(query)
    pure_scores = bm25.get_scores(q_tokens)
    base_scores = list(pure_scores)
    
    # 확장 키워드 추출
    expanded = extract_expanded(semantic_data, df_dict, N, idf, config)
    
    # PRF fallback
    if not expanded and config.enable_prf_fallback:
        prf_terms = prf_rm3_terms(documents, pure_scores, tokenize_en)
        expanded = filter_semantic_terms(prf_terms, df_dict, N, idf, config)
    
    # 확장 쿼리 혼합
    if expanded:
        q_tokens_expanded = list(set(q_tokens + expanded))
        alt_scores = bm25.get_scores(q_tokens_expanded)
        base_scores = [float(b) + config.expanded_weight * float(a)
                       for b, a in zip(base_scores, alt_scores)]
    
    # 후보 추출
    top_r = config.top_r_sem if expanded else config.top_r_pure
    cand_mixed = list(range(len(base_scores)))
    cand_mixed.sort(key=lambda i: base_scores[i], reverse=True)
    cand_mixed = cand_mixed[:top_r]
    
    cand_pure = list(range(len(pure_scores)))
    cand_pure.sort(key=lambda i: pure_scores[i], reverse=True)
    cand_pure = cand_pure[:config.top_r_pure]
    
    top_candidates = dedup(cand_mixed + cand_pure)
    
    # SAFE-DROP 제거됨 - FN 최소화를 위해 모든 후보 유지
    
    # 재랭킹
    reranked_candidates: List[Tuple[int, float]] = []
    final_score_map: Dict[int, float] = {}
    
    if expanded:
        for doc_id in top_candidates:
            doc_tokens, _ = doc_index[doc_id]
            bonus = soft_semantic_bonus_tokens(doc_tokens, idf, expanded, config)
            final_score = float(base_scores[doc_id]) + bonus
            reranked_candidates.append((doc_id, final_score))
        reranked_candidates.sort(key=lambda x: x[1], reverse=True)
        top_candidates = [doc_id for doc_id, _ in reranked_candidates]
        final_score_map = {doc_id: float(s) for doc_id, s in reranked_candidates}
    else:
        top_candidates.sort(key=lambda i: base_scores[i], reverse=True)
    
    # Recall 가드레일
    pure_top100 = cand_pure[:config.guardrail_k]
    final_order = top_candidates[:config.guardrail_k]
    missing = [d for d in pure_top100 if d not in final_order]
    if missing:
        room = max(0, config.guardrail_k - len(final_order))
        final_order = (final_order + missing[:room])[:config.guardrail_k]
    
    final_scores = [float(final_score_map.get(i, base_scores[i])) for i in final_order]
    
    debug_info = {
        "semantic_applied": bool(expanded),
        "filtered_terms": {"expanded": expanded},
        "skipped_by_dfidf": not bool(expanded),
        "mode": "full_document"
    }
    
    return final_order, final_scores, debug_info


# -----------------------------
# Main Reranking Function
# -----------------------------
def semantic_rerank(query: str, documents: List[str], base_scores: np.ndarray,
                   semantic_data: Optional[Dict[str, Any]], config: KeywordFilterConfig) -> Tuple[List[int], List[float], Dict[str, Any]]:
    """통합 시맨틱 재랭킹 함수"""
    
    # 적응형 방식 선택
    if config.adaptive_mode and len(documents) <= config.candidate_threshold:
        return candidate_based_rerank(query, documents, base_scores, semantic_data, config)
    else:
        return full_document_rerank(query, documents, base_scores, semantic_data, config)


# -----------------------------
# Metrics Accumulator
# -----------------------------
class MetricsAccumulator:
    """메트릭 누적 클래스"""
    
    def __init__(self):
        self.n = 0
        self.sum_p1 = 0.0
        self.sum_p10 = 0.0
        self.sum_r10 = 0.0
        self.sum_r100 = 0.0
        self.sum_mrr10 = 0.0
        self.sum_ndcg10 = 0.0
        self.sum_ndcg100 = 0.0
        self.sum_map100 = 0.0
        self.sem_attempted = 0
        self.sem_applied = 0
        self.skipped_by_dfidf = 0
        self.queries_with_rel = 0
        self.total_rel_docs = 0
        self.queries_r10_zero = 0
    
    def add(self, ranked_labels: List[int], total_rel_all: int, sem_applied: bool = False):
        """메트릭 추가"""
        self.n += 1
        topk = ranked_labels[:10]
        top100 = ranked_labels[:100]
        total_rel = max(int(total_rel_all), 0)
        
        p1 = float(ranked_labels[0]) if ranked_labels else 0.0
        p10 = (sum(topk) / max(len(topk), 1)) if topk else 0.0
        r10 = (sum(topk) / max(total_rel, 1)) if total_rel > 0 else 0.0
        r100 = (sum(top100) / max(total_rel, 1)) if total_rel > 0 else 0.0
        mrr10 = mrr_at_k(ranked_labels, 10)
        ndcg10 = ndcg_at_k(ranked_labels, 10)
        ndcg100 = ndcg_at_k(ranked_labels, 100)
        map100 = map_at_k(ranked_labels, 100)
        
        self.sum_p1 += p1
        self.sum_p10 += p10
        self.sum_r10 += r10
        self.sum_r100 += r100
        self.sum_mrr10 += mrr10
        self.sum_ndcg10 += ndcg10
        self.sum_ndcg100 += ndcg100
        self.sum_map100 += map100
        
        if total_rel > 0:
            self.queries_with_rel += 1
            self.total_rel_docs += total_rel
            if r10 == 0.0:
                self.queries_r10_zero += 1
        
        if sem_applied:
            self.sem_applied += 1
    
    def add_sem_attempt(self):
        """시맨틱 시도 추가"""
        self.sem_attempted += 1
    
    def add_skipped_by_dfidf(self):
        """DF/IDF 스킵 추가"""
        self.skipped_by_dfidf += 1
    
    def result(self) -> Dict[str, float]:
        """결과 메트릭 반환"""
        if self.n == 0:
            return {"P@1":0.0, "P@10":0.0, "R@10":0.0, "R@100":0.0, "MRR@10":0.0, "nDCG@10":0.0, "nDCG@100":0.0, "MAP@100":0.0}
        return {
            "P@1": self.sum_p1 / self.n,
            "P@10": self.sum_p10 / self.n,
            "R@10": self.sum_r10 / self.n,
            "R@100": self.sum_r100 / self.n,
            "MRR@10": self.sum_mrr10 / self.n,
            "nDCG@10": self.sum_ndcg10 / self.n,
            "nDCG@100": self.sum_ndcg100 / self.n,
            "MAP@100": self.sum_map100 / self.n,
        }
