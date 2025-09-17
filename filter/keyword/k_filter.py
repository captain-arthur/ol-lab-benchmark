# k_filter.py - 공통 키워드 필터링 모듈
import os
import json
import re
import math
import time
import hashlib
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
        
        # 확장 키워드 설정 (Precision 최적화 - 매우 엄격)
        self.max_expanded = kwargs.get("max_expanded", 3)  # 5->3: 더 엄격한 확장
        self.max_precision_tokens = kwargs.get("max_precision_tokens", 3)  # 5->3: 더 엄격한 제한
        self.df_thresh = kwargs.get("df_thresh", 0.85)  # 0.90->0.85: 더 엄격한 필터링
        self.idf_min = kwargs.get("idf_min", 0.15)      # 0.10->0.15: 더 엄격한 필터링
        self.quality_threshold = kwargs.get("quality_threshold", 0.08)  # 0.05->0.08: 더 높은 품질 요구
        
        # 재랭킹 방식 설정
        self.adaptive_mode = kwargs.get("adaptive_mode", True)
        self.candidate_threshold = kwargs.get("candidate_threshold", 1000)
        
        # 후보 기반 설정 (FiQA 스타일) - Precision 중심 튜닝 (매우 엄격)
        self.top_r_pure = kwargs.get("top_r_pure", 100)  # 200->100: 더 엄격한 후보 선택
        self.alpha_soft_bonus = kwargs.get("alpha_soft_bonus", 0.2)  # 0.5->0.2: 매우 보수적 확장
        self.anchor_k = kwargs.get("anchor_k", 0)  # 1->0: BM25 1위 고정 해제
        self.rrf_k = kwargs.get("rrf_k", 1_000_000_000)  # 3.0->1B: RRF 영향 ≈ 0으로 무력화
        
        # 전체 문서 설정 (MS MARCO 스타일) - Precision 중심 튜닝 (매우 엄격)
        self.top_r_sem = kwargs.get("top_r_sem", 1000)  # 2000->1000: 더 엄격한 후보 선택
        self.expanded_weight = kwargs.get("expanded_weight", 0.05)  # 0.15->0.05: 매우 보수적 확장
        self.alpha_bonus = kwargs.get("alpha_bonus", 0.1)  # 0.2->0.1: 매우 보수적 보너스
        
        # 공통 안전장치 (Precision 중심 튜닝 - 매우 엄격)
        self.guardrail_k = kwargs.get("guardrail_k", 50)  # 150->50: 매우 엄격한 가드레일
        self.enable_prf_fallback = kwargs.get("enable_prf_fallback", True)
        
        # P@1 최적화를 위한 새로운 파라미터
        self.confidence_percentile = kwargs.get("confidence_percentile", 95.0)  # 신뢰도 퍼센타일
        self.min_confidence_threshold = kwargs.get("min_confidence_threshold", 0.05)  # 최소 신뢰도 임계값
        self.boost_multiplier = kwargs.get("boost_multiplier", 3.0)  # P@1을 위한 부스트 배수
        self.expand_candidates = kwargs.get("expand_candidates", True)  # 후보 확장 여부
        self.candidate_expansion_factor = kwargs.get("candidate_expansion_factor", 4.0)  # 2.0 -> 4.0: 후보폭 확장
        self.top1_head_k = kwargs.get("top1_head_k", 12)  # answer-shape 승격이 볼 헤드 크기

        # answer-shape 승격기 임계값
        self.answer_shape_promote_tau = kwargs.get("answer_shape_promote_tau", 2.5)      # best - current >= tau 이면 승격
        self.answer_shape_strong = kwargs.get("answer_shape_strong", 4.0)                # strong 기준 (절대 승격용)
        self.answer_shape_min_for_current = kwargs.get("answer_shape_min_for_current", 1.0)  # current 가 이 미만이면 약함
        
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

def _norm_key(s: str) -> str:
    """키 정규화 함수"""
    # 공백 축약 + 소문자
    return re.sub(r"\s+", " ", (s or "").strip().lower())

def _semexp_cache_key(model: str, prompt: str, query: str) -> str:
    """시맨틱 확장 캐시 키 생성"""
    pv = PROMPT_VERSION + "|" + prompt  # 프롬프트 문자열 자체를 반영
    phash = hashlib.sha1(pv.encode("utf-8")).hexdigest()[:12]
    return f"semexp:{model}:{phash}:{_norm_key(query)}"

# 파일 캐시 시스템 (간단한 구조)
def get_semexp_cache(model: str, prompt: str, query: str, cache_path: str = None):
    """시맨틱 확장 캐시 조회"""
    if not cache_path or not os.path.exists(cache_path):
        return None
    
    try:
        with open(cache_path, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)
        
        # 쿼리 정규화
        normalized_query = _norm_key(query)
        
        # 배열에서 해당 쿼리 찾기
        for entry in cache_data:
            if _norm_key(entry.get("query", "")) == normalized_query:
                return entry
        
        return None
    except Exception:
        return None

def set_semexp_cache(model: str, prompt: str, query: str, value: Dict[str, Any], cache_path: str = None):
    """시맨틱 확장 캐시 저장 - 6~10개 보장인 경우만 저장"""
    if not cache_path:
        return
    
    kws = (value or {}).get("expanded", {}).get("keywords", [])
    
    if isinstance(kws, list) and 6 <= len(kws) <= 10:
        # 기존 캐시 로드
        cache_data = []
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                    if not isinstance(cache_data, list):
                        cache_data = []
            except Exception:
                cache_data = []
        
        # 쿼리 정규화
        normalized_query = _norm_key(query)
        
        # 중복 체크 및 업데이트
        found = False
        for i, entry in enumerate(cache_data):
            if _norm_key(entry.get("query", "")) == normalized_query:
                cache_data[i] = value  # 업데이트
                found = True
                break
        
        if not found:
            cache_data.append(value)  # 새 항목 추가
        
        # 파일에 저장
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

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
# Precision-oriented Drop Logic
# -----------------------------
def _idf_top_threshold(idf_dict: Dict[str, float], pct: float = 0.9) -> float:
    """IDF 상위 퍼센타일 임계값 계산"""
    vals = sorted(idf_dict.values())
    if not vals:
        return 0.0
    return vals[int(len(vals) * pct)]

def _count_overlap(tokens: Set[str], q_tokens: Set[str]) -> int:
    """토큰 집합 간 겹침 개수 계산"""
    return len(tokens & q_tokens)

def apply_precision_drop_logic(
    query: str, 
    documents: List[str], 
    final_global_order: List[int], 
    final_scores: List[float],
    expanded_tokens: List[str],
    debug_info: Dict[str, Any],
    doc_index: Dict[int, Set[str]] = None,
    cand_idx: List[int] = None
) -> Tuple[List[int], List[float]]:
    """Precision 최적화를 위한 조건부 Drop 로직"""
    
    # 로컬 IDF 계산 (전역 IDF가 없는 경우를 위해)
    local_idf = build_idf_dict_from_df(
        build_df_dict(documents, tokenize_en), 
        len(documents)
    )
    
    # IDF 상위 90% 임계값
    idf_cut = _idf_top_threshold(local_idf, pct=0.9)
    
    # 쿼리 핵심 토큰 (IDF 상위 90%)
    q_tokens = set(tokenize_en(query))
    q_core = {t for t in q_tokens if local_idf.get(t, 0.0) >= idf_cut}
    
    # 하위 퍼센타일 컷 (하위 40%는 DROP 후보)
    cut_rank = int(len(final_global_order) * 0.6)
    
    keep_ids, keep_scores = [], []
    expanded_set = set(expanded_tokens) if isinstance(expanded_tokens, list) else set()
    
    debug_info.setdefault("dropped", [])
    
    for rank, doc_id in enumerate(final_global_order):
        # 최적화된 토큰 조회 (O(1) 접근)
        if doc_index is not None:
            doc_tokens = doc_index.get(doc_id) or set(tokenize_en(documents[doc_id]))
        else:
            doc_tokens, _ = build_doc_token_index([documents[doc_id]], tokenize_en)[0]
        
        # Drop 조건 체크
        cond_expanded_match = len(doc_tokens & expanded_set) == 0
        cond_core_low = _count_overlap(doc_tokens, q_core) <= 0
        cond_tail = rank >= cut_rank
        
        # 3조건 모두 만족하면 DROP
        if cond_expanded_match and cond_core_low and cond_tail:
            debug_info["dropped"].append({
                "doc_id": int(doc_id),
                "rank": int(rank),
                "reason": "no_expanded_match+no_core_overlap+tail",
                "doc_preview": documents[doc_id][:100] + "..." if len(documents[doc_id]) > 100 else documents[doc_id]
            })
            continue
        
        keep_ids.append(doc_id)
        keep_scores.append(final_scores[rank])
    
    return keep_ids, keep_scores


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

# 억제 단어(전역)
DENYLIST = {
    "a","an","the","and","or","of","to","in","on","for","with","from","by","as","at",
    "is","are","was","were","be","been","being","have","has","had","do","does","did",
    "can","could","should","would","may","might","will","your","you","this","that",
    "these","those","which","not","no","yes","it","its","we","they","i"
}

def build_global_stats(passages: List[str], tokenizer=tokenize_en):
    """코퍼스 전역 DF/IDF 준비 함수"""
    df_global = build_df_dict(passages, tokenizer)
    N_global = len(passages)
    idf_global = build_idf_dict_from_df(df_global, N_global)
    return df_global, idf_global, N_global

def quantile_threshold(values: List[float], q: float, default: float) -> float:
    """분위수 기반 임계값 계산"""
    if not values: 
        return default
    arr = np.array(values, dtype=float)
    return float(np.quantile(arr, q))

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

# 프롬프트 버전 관리
PROMPT_VERSION = "semexp_v3_strict_6to10_finance"

def _prompt_semexp(query: str) -> str:
    """극단 정밀 모드: LLM 산출물을 확장 키워드 그 이상으로 활용"""
    return f"""You are a financial IR expert and a BM25-aware retrieval engineer.
Goal: Produce a JSON that maximizes Precision@1 by separating the TRUE answer docs from look-alikes.

Rules:
- Think contrastively: what appears ONLY in correct docs and often in wrong docs.
- All n-grams lowercase, 2–5 tokens, no stopwords.
- Return EXACTLY one valid JSON object, no prose.

Fill:
- must_have_ngrams: decisive phrases that uniquely occur in correct docs.
- anti_ngrams: phrases typical in confusing but wrong contexts.
- anchor_phrases: phrases helpful for phrase/proximity boosting.
- constraints: jurisdiction/date_range/unit/entity if implied.
- contrastive_signals: positive_terms vs hard_negative_terms.
- query_variants: 3–5 semantic rewrites preserving intent; keep narrow.
- expanded.keywords: 6–10 precision-optimized terms.
- entailment_checklist: 3–6 yes/no verifiable claims in the target doc.

Query: "{query}"
Output JSON schema:
{{
  "user_query": "{query}",
  "intent": "definition | comparison | computation | entity_lookup | regulation | how_to",
  "answer_type": "number | date | entity | span | passage",
  "must_have_ngrams": ["...","..."],
  "anti_ngrams": ["...","..."],
  "anchor_phrases": ["...","..."],
  "constraints": {{
    "jurisdiction": "US | EU | ...",
    "date_range": {{"from": "YYYY", "to": "YYYY"}},
    "unit": "bps | USD | % | ...",
    "entity": ["Fed","ECB","Apple Inc."]
  }},
  "contrastive_signals": {{
    "positive_terms": ["...","..."],
    "hard_negative_terms": ["...","..."]
  }},
  "query_variants": ["...","...","..."],
  "expanded": {{ "keywords": ["...","...","...","...","...","..."] }},
  "entailment_checklist": [
    "텍스트에 X가 명시되어 있다",
    "수치 단위가 Y로 표기된다",
    "주어가 Z 기관/엔티티다"
  ]
}}
""".strip()

def build_semantic_data_ollama(query: str,
                               host: str = "http://192.168.45.166:11434",
                               model: str = "gemma3",
                               timeout: int = 15,
                               retries: int = 2) -> Optional[Dict[str, Any]]:
    """LLM을 통한 시맨틱 확장 데이터 생성 - 강화된 프롬프트와 엄격한 검증"""
    
    prompt = _prompt_semexp(query)
    
    for attempt in range(retries + 1):
        try:
            resp = requests.post(
                f"{host}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0,  # 일관성 최대화
                        "top_p": 1.0,
                        "repeat_penalty": 1.1
                    }
                },
                timeout=timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            text = (data.get("response") or "").strip()
            
            # 엄격한 JSON 추출 및 검증
            parsed = _extract_and_validate_json_strict(text, query)
            if parsed:  # ✅ 6~10개 충족
                return parsed
                
        except Exception:
            pass
        
        # 재시도 대기
        if attempt < retries:
            time.sleep(0.4 * (attempt + 1))
    
    # 모든 시도 실패 → Fallback
    return _generate_fallback_keywords(query)

def get_or_build_semantic(query: str, cfg: KeywordFilterConfig, cache_path: str = None) -> Optional[Dict[str, Any]]:
    """캐시를 활용한 시맨틱 확장 데이터 생성"""
    prompt = _prompt_semexp(query)
    cached = get_semexp_cache(cfg.ollama_model, prompt, query, cache_path)
    if cached:
        print(f"  ✓ Cache HIT: {len(cached.get('expanded', {}).get('keywords', []))} keywords")
        return cached
    
    print(f"  ✗ Cache MISS: calling LLM...")
    result = build_semantic_data_ollama(query, cfg.ollama_host, cfg.ollama_model, cfg.ollama_timeout, cfg.ollama_retries)
    if result:
        set_semexp_cache(cfg.ollama_model, prompt, query, result, cache_path)  # ✅ 6~10개 조건 만족 시에만 저장
        print(f"  ✓ Added to cache: {len(result.get('expanded', {}).get('keywords', []))} keywords")
    return result

def _extract_and_validate_json_strict(text: str, original_query: str) -> Optional[Dict[str, Any]]:
    """엄격한 JSON 추출 및 검증 - 6~10개 키워드 강제"""
    # 후보 JSON 블록들을 최대한 뽑아내어 하나라도 유효하면 통과
    candidates = []
    try:
        # 1) 첫 { ~ 마지막 } 블록
        a, b = text.find("{"), text.rfind("}")
        if a != -1 and b != -1 and b > a:
            candidates.append(text[a:b+1])
        # 2) ```json 블록
        p = text.find("```json")
        if p != -1:
            p2 = text.find("```", p+7)
            if p2 != -1:
                candidates.append(text[p+7:p2].strip())
        # 3) 일반 ``` 블록
        p = text.find("```")
        if p != -1:
            p2 = text.find("```", p+3)
            if p2 != -1:
                candidates.append(text[p+3:p2].strip())
    except Exception:
        pass

    for cand in candidates:
        try:
            obj = json.loads(cand)
            kws = obj.get("expanded", {}).get("keywords", [])
            if not isinstance(kws, list): 
                continue

            # 정제: 소문자/트림/2~4어절/중복 제거
            cleaned = []
            seen = set()
            for kw in kws:
                if not isinstance(kw, str): 
                    continue
                s = " ".join(kw.lower().split())
                wc = len(s.split())
                if 2 <= wc <= 4 and s and s not in seen:
                    cleaned.append(s); seen.add(s)

            if 6 <= len(cleaned) <= 10:
                # 새로운 스키마 필드들 기본값 설정
                result = {
                    "query": obj.get("user_query", original_query),
                    "expanded": {"keywords": cleaned},
                    "must_have_ngrams": obj.get("must_have_ngrams", []),
                    "anti_ngrams": obj.get("anti_ngrams", []),
                    "anchor_phrases": obj.get("anchor_phrases", []),
                    "constraints": obj.get("constraints", {}),
                    "contrastive_signals": obj.get("contrastive_signals", {"positive_terms": [], "hard_negative_terms": []}),
                    "query_variants": obj.get("query_variants", []),
                    "entailment_checklist": obj.get("entailment_checklist", [])
                }
                return result
        except Exception:
            continue
    return None

def _extract_and_validate_json(text: str, original_query: str) -> Optional[Dict[str, Any]]:
    """JSON 추출 및 검증 - 더 강력한 파싱"""
    try:
        # 여러 방법으로 JSON 추출 시도
        json_candidates = []
        
        # 방법 1: 첫 번째 { 부터 마지막 } 까지
        first_brace = text.find("{")
        last_brace = text.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            json_candidates.append(text[first_brace:last_brace+1])
        
        # 방법 2: ```json 블록 내부
        json_start = text.find("```json")
        if json_start != -1:
            json_start += 7
            json_end = text.find("```", json_start)
            if json_end != -1:
                json_candidates.append(text[json_start:json_end].strip())
        
        # 방법 3: ``` 블록 내부
        code_start = text.find("```")
        if code_start != -1:
            code_start += 3
            code_end = text.find("```", code_start)
            if code_end != -1:
                json_candidates.append(text[code_start:code_end].strip())
        
        # 각 후보를 시도
        for candidate in json_candidates:
            try:
                obj = json.loads(candidate)
                keywords = obj.get("expanded", {}).get("keywords", [])
                
                # 키워드 검증
                if isinstance(keywords, list) and len(keywords) >= 6:
                    # 키워드 정제
                    cleaned_keywords = []
                    for kw in keywords:
                        if isinstance(kw, str):
                            cleaned = kw.strip().lower()
                            if 2 <= len(cleaned.split()) <= 4 and cleaned:
                                cleaned_keywords.append(cleaned)
                    
                    if len(cleaned_keywords) >= 6:
                        return {
                            "query": obj.get("user_query", original_query),
                            "expanded": {"keywords": cleaned_keywords[:10]}  # 최대 10개로 제한
                        }
            except json.JSONDecodeError:
                continue
                
    except Exception:
        pass
    
    return None

def _generate_fallback_keywords(query: str) -> Dict[str, Any]:
    """LLM 실패 시 fallback 키워드 생성 - 6~10개 보장"""
    # 쿼리에서 핵심 단어 추출
    words = query.lower().split()
    
    # 불용어 제거
    stopwords = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "will", "would", "could", "should", "may", "might", "can", "what", "how", "when", "where", "why", "which", "who"}
    
    filtered_words = [w for w in words if w not in stopwords and len(w) > 2]
    
    # 2-4어절 키워드 생성
    fallback_keywords = []
    
    # 2어절 조합 (우선순위)
    for i in range(len(filtered_words) - 1):
        if len(fallback_keywords) >= 10:
            break
        phrase = f"{filtered_words[i]} {filtered_words[i+1]}"
        fallback_keywords.append(phrase)
    
    # 3어절 조합
    for i in range(len(filtered_words) - 2):
        if len(fallback_keywords) >= 10:
            break
        phrase = f"{filtered_words[i]} {filtered_words[i+1]} {filtered_words[i+2]}"
        fallback_keywords.append(phrase)
    
    # 4어절 조합
    for i in range(len(filtered_words) - 3):
        if len(fallback_keywords) >= 10:
            break
        phrase = f"{filtered_words[i]} {filtered_words[i+1]} {filtered_words[i+2]} {filtered_words[i+3]}"
        fallback_keywords.append(phrase)
    
    # 최소 6개 보장 (부족한 경우 보충)
    while len(fallback_keywords) < 6 and filtered_words:
        word = filtered_words[0] if filtered_words else "analysis"
        fallback_keywords.extend([
            f"{word} analysis",
            f"{word} evaluation", 
            f"{word} strategy",
            f"{word} management",
            f"{word} planning",
            f"{word} optimization"
        ])
        break
    
    # 최대 10개로 제한
    return {
        "query": query,
        "expanded": {"keywords": fallback_keywords[:10]}
    }


# -----------------------------
# Precision-Optimized Keyword Filtering
# -----------------------------

def evaluate_filtering_stage(
    stage_name: str,
    input_docs: List[str], 
    output_docs: List[int],
    relevant_docs: Set[int],
    input_to_global_mapping: List[int] = None
) -> Dict[str, float]:
    """
    각 단계의 Keep/Drop 기준 혼동행렬 기반 지표:
      - Keep을 Positive로 간주 (파이프라인을 통과시킴)
      - Drop을 Negative로 간주
      - input_to_global_mapping: 로컬 인덱스를 글로벌 인덱스로 매핑
    """
    input_set = set(range(len(input_docs)))
    keep_set = set(output_docs)
    drop_set = input_set - keep_set

    # 글로벌 인덱스 매핑이 있으면 로컬 인덱스로 변환
    if input_to_global_mapping:
        # 글로벌 relevant_docs를 로컬 인덱스로 변환
        local_relevant_docs = set()
        for global_idx in relevant_docs:
            if global_idx in input_to_global_mapping:
                local_idx = input_to_global_mapping.index(global_idx)
                local_relevant_docs.add(local_idx)
        rel = local_relevant_docs
    else:
        # 매핑이 없으면 기존 방식 (전체 문서 기준)
        rel = set(relevant_docs)
    
    nonrel = input_set - rel

    # 혼동행렬
    TP = len(keep_set & rel)      # 관련인데 Keep
    FP = len(keep_set & nonrel)   # 무관인데 Keep
    FN = len(drop_set & rel)      # 관련인데 Drop
    TN = len(drop_set & nonrel)   # 무관인데 Drop

    # Keep 기준 지표
    keep_precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    keep_recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    keep_fpr       = FP / (FP + TN) if (FP + TN) > 0 else 0.0  # 무관인데 통과

    # Drop 기준 지표 (본문 서술과 일치: Drop Precision = Drop 중 무관 비율)
    drop_precision = TN / (TN + FN) if (TN + FN) > 0 else 0.0  # Drop된 것 중 무관
    drop_recall    = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    drop_fnr       = FN / (TP + FN) if (TP + FN) > 0 else 0.0  # 관련을 Drop한 비율

    # F-β 지표 (단계별 운영점 설명력 강화)
    # Precision 우선 단계: β=0.5, Recall 우선 단계: β=2.0
    keep_fbeta_05 = (1 + 0.5**2) * (keep_precision * keep_recall) / (0.5**2 * keep_precision + keep_recall) if (0.5**2 * keep_precision + keep_recall) > 0 else 0.0
    keep_fbeta_20 = (1 + 2.0**2) * (keep_precision * keep_recall) / (2.0**2 * keep_precision + keep_recall) if (2.0**2 * keep_precision + keep_recall) > 0 else 0.0
    drop_fbeta_05 = (1 + 0.5**2) * (drop_precision * drop_recall) / (0.5**2 * drop_precision + drop_recall) if (0.5**2 * drop_precision + drop_recall) > 0 else 0.0
    drop_fbeta_20 = (1 + 2.0**2) * (drop_precision * drop_recall) / (2.0**2 * drop_precision + drop_recall) if (2.0**2 * drop_precision + drop_recall) > 0 else 0.0

    return {
        "stage": stage_name,
        "input_count": len(input_docs),
        "output_count": len(output_docs),
        "dropped_count": len(drop_set),

        # Keep 관점 (교수님이 직관적으로 보는 '통과시킨 것의 품질/손실')
        "keep_precision": keep_precision,
        "keep_recall": keep_recall,
        "keep_fpr": keep_fpr,

        # Drop 관점 (본문에 맞춘 '잘 버렸는가/잘못 버렸는가')
        "drop_precision": drop_precision,
        "drop_recall": drop_recall,
        "drop_fnr": drop_fnr,

        # F-β 지표 (단계별 운영점 설명력)
        "keep_fbeta_05": keep_fbeta_05,  # Precision 우선 (β=0.5)
        "keep_fbeta_20": keep_fbeta_20,  # Recall 우선 (β=2.0)
        "drop_fbeta_05": drop_fbeta_05,  # Precision 우선 (β=0.5)
        "drop_fbeta_20": drop_fbeta_20,  # Recall 우선 (β=2.0)

        # 혼동행렬 원시값(부록/재현성)
        "TP": TP, "FP": FP, "FN": FN, "TN": TN
    }

def strict_keyword_matching(documents: List[str], query: str, semantic_data: Dict[str, Any]) -> List[int]:
    """
    토큰 단위의 엄격 매칭.
    - 쿼리 토큰 + (확장 키워드 중 상위 3개를 토큰화) 를 합쳐 core_tokens 구성
    - 문서별로 core_tokens 매칭 비율 평가, 길이 기반 동적 임계값 적용
    """
    import math
    
    # 1) 핵심 토큰 구성
    core_tokens = tokenize_en(query)

    if semantic_data and isinstance(semantic_data.get("expanded"), dict):
        for phrase in (semantic_data["expanded"].get("keywords") or [])[:3]:
            core_tokens.extend(tokenize_en(phrase))

    # 토큰 중복 제거 및 필터링
    def dedup(tokens):
        seen = set()
        result = []
        for token in tokens:
            if token not in seen and len(token) > 2:
                seen.add(token)
                result.append(token)
        return result
    
    core_tokens = dedup(core_tokens)
    if not core_tokens:
        return []

    # 동적 임계값: 짧은 쿼리는 높은 매칭률, 긴 쿼리는 60~70%로 완화 (과잉 드롭 완화)
    n = len(core_tokens)
    if n <= 4:
        min_hits = n  # 전부 포함
    elif n <= 8:
        min_hits = math.ceil(n * 0.70)  # 0.75 → 0.70
    else:
        min_hits = math.ceil(n * 0.60)  # 0.65 → 0.60

    # 2) 문서 스코어링
    results = []
    for i, doc in enumerate(documents):
        doc_tokens = set(tokenize_en(doc))
        hits = sum(1 for t in core_tokens if t in doc_tokens)
        if hits >= min_hits:
            results.append((i, hits))

    # 매칭 히트수 기준 내림차순
    results.sort(key=lambda x: x[1], reverse=True)
    return [i for i, _ in results]

def verify_precision(candidate_docs: List[int], documents: List[str], query: str) -> List[int]:
    """
    과잉제거 방지:
      - 짧은 쿼리(<=4)는 100% 근접, 중간은 70~80%, 긴 쿼리는 60~70%
      - 너무 짧은 문서 제외(노이즈), 너무 긴 문서 상한을 1500 단어로 완화
    """
    import math
    
    q_tokens = tokenize_en(query)
    n = len(q_tokens)
    if n <= 4:
        ratio = 0.85  # 0.9 → 0.85
    elif n <= 8:
        ratio = 0.70  # 0.75 → 0.70
    else:
        ratio = 0.60  # 0.65 → 0.60

    verified = []
    for doc_id in candidate_docs:
        doc_text = documents[doc_id]
        doc_tokens = set(tokenize_en(doc_text))
        q_tokens_set = set(q_tokens)
        
        # 토큰 집합 교집합으로 매칭 (서브스트링 매칭 대신)
        matched_tokens = q_tokens_set & doc_tokens
        matched = len(matched_tokens)

        if matched < max(1, math.ceil(n * ratio)):
            continue

        words = len(doc_text.split())
        if words < 10:
            continue
        if words > 1500:  # 500 → 1500으로 완화
            continue

        verified.append(doc_id)
    return verified

def adjust_recall(precision_docs: List[int], documents: List[str], query: str, semantic_data: Dict[str, Any]) -> List[int]:
    """
    후보가 너무 적을 때만 단계적으로 완화:
      1) 70% 기준으로 최대 5개
      2) 여전히 10개 미만이면 60% 기준으로 추가 최대 5개
    """
    import math
    
    target_min = 15  # 10 → 15로 확대
    out = list(precision_docs)
    if len(out) >= target_min:
        return out

    q_tokens = tokenize_en(query)
    def add_by_ratio(ratio: float, limit: int):
        nonlocal out
        add = []
        for i, doc in enumerate(documents):
            if i in out:
                continue
            hits = sum(1 for t in q_tokens if t in doc.lower())
            if hits >= math.ceil(len(q_tokens) * ratio):
                add.append(i)
            if len(add) >= limit:
                break
        out.extend(add)

    add_by_ratio(0.70, 8)  # 5 → 8로 확대
    if len(out) < target_min:
        add_by_ratio(0.60, 8)  # 5 → 8로 확대

    return out

def precision_optimized_keyword_filter(
    documents: List[str],
    query: str,
    semantic_data: Dict[str, Any],
    config: KeywordFilterConfig,
    relevant_docs: Set[int]
) -> Tuple[List[int], Dict[str, Any]]:
    """정밀도 우선 키워드 필터링"""
    
    stage_metrics = []
    
    # 1단계: 매우 엄격한 키워드 매칭
    strict_matches = strict_keyword_matching(documents, query, semantic_data)
    
    # 빈 후보 보호 장치: Stage 1 결과가 0이면 BM25 상위 20개를 강제로 통과
    safety_guard_applied = False
    if len(strict_matches) == 0:
        safety_guard_applied = True
        # 간단한 BM25 스타일 점수로 상위 20개 선택
        query_tokens = set(tokenize_en(query))
        doc_scores = []
        for i, doc in enumerate(documents):
            doc_tokens = set(tokenize_en(doc))
            score = len(query_tokens & doc_tokens)
            doc_scores.append((i, score))
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        strict_matches = [i for i, score in doc_scores[:20] if score > 0]
        
        # 안전장치 모니터링: 무관 문서 비율 체크 (내부 로그만)
        if strict_matches:
            safety_guard_nonrel_count = sum(1 for i in strict_matches if i not in relevant_docs)
            safety_guard_nonrel_ratio = safety_guard_nonrel_count / len(strict_matches)
            # 무관 문서 비율이 너무 높으면 내부적으로만 기록
            if safety_guard_nonrel_ratio > 0.8:
                pass  # 출력 제거, 내부 메트릭으로만 추적
    
    # Stage 1 지표 계산 (전체 문서 기준 - 실제 Drop 결정 시점)
    metrics_stage1 = evaluate_filtering_stage("strict_matching", documents, strict_matches, relevant_docs)
    stage_metrics.append(metrics_stage1)
    
    # 2단계: 정밀도 검증 (False Positive 제거)
    precision_verified = verify_precision(strict_matches, documents, query)
    
    # Stage 2 지표 계산 (Stage 1의 Keep 집합을 입력으로, 실제 Drop 결정 시점)
    if strict_matches:
        stage1_keep_docs = [documents[i] for i in strict_matches]
        # precision_verified는 strict_matches 내의 로컬 인덱스
        metrics_stage2 = evaluate_filtering_stage("precision_verification", stage1_keep_docs, precision_verified, relevant_docs, strict_matches)
    else:
        metrics_stage2 = evaluate_filtering_stage("precision_verification", [], [], relevant_docs)
    stage_metrics.append(metrics_stage2)
    
    # 3단계: 재현율 보정 (너무 많이 버렸는지 확인)
    recall_adjusted = adjust_recall(precision_verified, documents, query, semantic_data)
    
    # Stage 3 지표 계산 (Stage 2의 Keep 집합을 입력으로, 실제 Drop 결정 시점)
    if precision_verified:
        stage2_keep_docs = [documents[i] for i in precision_verified]
        # recall_adjusted는 precision_verified 내의 로컬 인덱스
        metrics_stage3 = evaluate_filtering_stage("recall_adjustment", stage2_keep_docs, recall_adjusted, relevant_docs, precision_verified)
    else:
        metrics_stage3 = evaluate_filtering_stage("recall_adjustment", [], [], relevant_docs)
    stage_metrics.append(metrics_stage3)
    
    # 최종 Drop Precision 계산 (전체 파이프라인 기준)
    # 전체 문서에서 최종 Keep된 문서를 제외한 나머지가 Drop
    final_keep_docs = recall_adjusted
    final_metrics = evaluate_filtering_stage("final_pipeline", documents, final_keep_docs, relevant_docs)
    stage_metrics.append(final_metrics)
    
    return recall_adjusted, {
        "strict_matches": len(strict_matches),
        "precision_verified": len(precision_verified), 
        "final_output": len(recall_adjusted),
        "stage_1": strict_matches,
        "stage_2": precision_verified,
        "stage_3": recall_adjusted,
        "stage_metrics": stage_metrics,
        "safety_guard_applied": safety_guard_applied
    }

# -----------------------------
# Answer Shape Scoring
# -----------------------------

def _has_number_unit_windows(text: str, units: List[str], window_tokens: int = 12) -> Tuple[int, float]:
    """
    숫자+단위 패턴 카운트와, 토큰 윈도우 기반 근접도 점수(가중)를 반환.
    근접도가 높을수록 점수↑ (token window가 작을수록 가중↑)
    """
    t = (text or "").lower()
    nums = [(m.start(), m.group(0)) for m in re.finditer(r'\b\d[\d,.\s]*\b', t)]
    hits = 0
    score = 0.0
    # 단위 정규화
    unit_set = set(u.lower() for u in (units or []))
    unit_alias = {
        "%": ["%", " percent", " percentage", " per cent"],
        "bps": ["bps", "bp", "basis points"],
        "usd": ["usd", "$", " us$"]
    }
    # 확장
    for u in list(unit_set):
        for a in unit_alias.get(u, []):
            unit_set.add(a)

    toks = tokenize_en(t)  # 토큰열
    for i, tok in enumerate(toks):
        # 숫자 토큰과 단위 토큰이 윈도우 내 함께 있으면 히트
        if re.match(r'^\d', tok):
            rng = toks[max(0, i - window_tokens): min(len(toks), i + window_tokens + 1)]
            if any(any(u.strip() and u in " ".join(rng) for u in unit_set)):
                hits += 1
                # 더 가까운 윈도우일수록 가중↑
                score += 1.0 + (12.0 / max(4.0, window_tokens))
    return hits, score

def _entity_proximity_score(text: str, entities: List[str], window_tokens: int = 12) -> float:
    """엔티티와 숫자/단위가 가까이 나타나는지 근접 점수."""
    t = (text or "").lower()
    toks = tokenize_en(t)
    idxs_by = {}
    for i, tok in enumerate(toks):
        idxs_by.setdefault(tok, []).append(i)
    def _positions(phrase: str) -> List[int]:
        pts = []
        p = tokenize_en(phrase)
        if not p: return pts
        for i in range(len(toks) - len(p) + 1):
            if toks[i:i+len(p)] == p:
                pts.append(i)
        return pts

    # 숫자/단위 위치(단순 탐색)
    num_pos = [i for i, tok in enumerate(toks) if re.match(r'^\d', tok)]
    unit_pos = [i for i, tok in enumerate(toks) if tok in {"percent", "percentage", "bps", "bp", "usd"}]
    score = 0.0
    for ent in (entities or []):
        pos = _positions(ent)
        for p in pos:
            if any(abs(p - n) <= window_tokens for n in num_pos) and any(abs(p - u) <= window_tokens for u in unit_pos):
                score += 2.0  # 엔티티-숫자-단위 삼자 근접
            elif any(abs(p - n) <= window_tokens for n in num_pos) or any(abs(p - u) <= window_tokens for u in unit_pos):
                score += 1.0
    return score

def _date_range_score(text: str, date_range: Dict[str, Any]) -> float:
    """constraints.date_range와 텍스트의 연도 매칭 정도."""
    if not date_range: 
        return 0.0
    t = (text or "").lower()
    years = [int(y) for y in re.findall(r'\b(19|20)\d{2}\b', t)]
    y_from = int(date_range.get("from") or 0)
    y_to = int(date_range.get("to") or 9999)
    if not years: 
        return 0.0
    in_range = sum(1 for y in years if y_from <= y <= y_to)
    return 1.0 * in_range

def _intent_cues_score(text: str, intent: str, entities: List[str]) -> float:
    """intent별 결정적 cue 가중."""
    t = (text or "").lower()
    s = 0.0
    if intent == "definition":
        # 'X is/means/defined as' 형태
        for e in (entities or []):
            e = (e or "").lower()
            if e and (e + " is ") in t or (e + " means ") in t or ("defined as" in t):
                s += 2.0
    elif intent == "regulation":
        # 'section/article/rule' 등 규정 문맥
        if any(k in t for k in ["section ", " article ", " rule ", " regulation ", " compliance "]):
            s += 1.5
    elif intent == "computation":
        # 공식/수식 단서
        if any(k in t for k in ["formula", "calculated as", "computed as", " = "]):
            s += 1.0
    return s

def answer_shape_score(doc_text: str, semantic_data: Dict[str, Any], query: str) -> Tuple[float, Dict[str, float]]:
    """
    문서가 '답 형태'를 갖췄는지 점수화.
    가중치 설계는 P@1 승격을 위해 공격적으로 설정.
    """
    txt = (doc_text or "")
    
    # semantic_data가 None이거나 필드가 없을 수 있으므로 안전하게 처리
    if not semantic_data:
        semantic_data = {}
    
    cons = semantic_data.get("constraints", {}) or {}
    entities = cons.get("entity", []) or []
    unit = cons.get("unit")
    units = []
    if unit: 
        units.append(str(unit))
    
    # 숫자+단위 윈도우
    unit_hits, unit_window = _has_number_unit_windows(txt, units, window_tokens=12)
    # 엔티티-숫자/단위 근접
    ent_prox = _entity_proximity_score(txt, entities, window_tokens=12)
    # 기간 일치
    dr_score = _date_range_score(txt, cons.get("date_range") or {})
    # intent cue
    intent = semantic_data.get("intent") or ""
    intent_s = _intent_cues_score(txt, intent, entities)

    # answer_type 별 기본 보정
    atype = semantic_data.get("answer_type") or ""
    type_bonus = 0.0
    if atype == "number" and (unit_hits > 0 or unit_window > 0):
        type_bonus += 1.5
    if atype == "date" and dr_score > 0:
        type_bonus += 1.0
    if atype == "entity" and entities and any(e.lower() in txt.lower() for e in (entities or [])):
        type_bonus += 1.0

    # 종합 점수 (가중치 튜닝: P@1 위주)
    score = (
        1.8 * unit_hits +
        1.2 * unit_window +
        2.0 * ent_prox +
        1.2 * dr_score +
        1.0 * intent_s +
        type_bonus
    )
    feats = {
        "unit_hits": float(unit_hits),
        "unit_window": float(unit_window),
        "entity_proximity": float(ent_prox),
        "date_range": float(dr_score),
        "intent_cues": float(intent_s),
        "type_bonus": float(type_bonus)
    }
    return score, feats

# -----------------------------
# Query Expansion Utils
# -----------------------------
def filter_semantic_terms_tokens(
    phrases: List[str],
    df_global: Dict[str,int],
    idf_global: Dict[str,float],
    N_global: int,
    config: Optional[KeywordFilterConfig] = None,
    **kwargs
) -> List[str]:
    """확장어 필터: Precision 최적화 모드 - False Positive 최소화 강화"""
    
    # 1단계: 구문에서 토큰 추출 및 기본 정제
    tokens = []
    for ph in phrases or []:
        for t in tokenize_en(ph):
            if len(t) >= 3 and t not in DENYLIST:
                tokens.append(t)
    tokens = dedup(tokens)

    if not tokens:
        return []

    # 2단계: 정밀도 중심 품질 점수 계산
    scored_tokens = []
    for token in tokens:
        df_count = df_global.get(token, 0)
        idf_score = idf_global.get(token, 0.0)
        
        # 더 엄격한 필터링 조건
        df_ratio = df_count / N_global if N_global > 0 else 0
        
        # 조건 1: 너무 일반적인 단어 제거 (85% 이상 문서에 나타남)
        if df_ratio > 0.85:
            continue
            
        # 조건 2: 너무 희귀한 단어 제거 (IDF < 0.15)
        if idf_score < 0.15:
            continue
            
        # 조건 3: 토큰 길이 체크 (너무 짧거나 긴 토큰 제거)
        if len(token) < 3 or len(token) > 15:
            continue
        
        # 정밀도 중심 품질 점수 계산
        base_quality = idf_score * (1.0 - df_ratio)
        
        # 추가 품질 지표
        length_bonus = min(0.1, (len(token) - 3) * 0.02) if len(token) > 3 else 0.0
        noise_penalty = 0.0
        
        # 숫자나 특수문자 포함 시 페널티
        if any(c.isdigit() for c in token) or any(c in ".,!?;:" for c in token):
            noise_penalty = 0.1
        
        # 최종 품질 점수
        quality_score = base_quality + length_bonus - noise_penalty
        
        # 높은 품질 임계값 적용
        quality_threshold = config.quality_threshold if config else 0.08
        if quality_score >= quality_threshold:
            scored_tokens.append((token, quality_score, idf_score, df_ratio))
    
    # 3단계: 품질 점수 기준 정렬 및 선택
    scored_tokens.sort(key=lambda x: x[1], reverse=True)  # 품질 점수 기준 내림차순
    
    # 최대 토큰 수 제한 (정밀도 보장)
    max_tokens = config.max_precision_tokens if config else 3
    kept = [token for token, _, _, _ in scored_tokens[:max_tokens]]
    
    # 4단계: 최소 보장 로직 (정밀도 유지하면서 최소 1개는 보장)
    if not kept and tokens:
        # 모든 토큰 중에서 가장 나은 것 선택
        best_token = None
        best_score = -1
        
        for token in tokens:
            df_count = df_global.get(token, 0)
            idf_score = idf_global.get(token, 0.0)
            df_ratio = df_count / N_global if N_global > 0 else 0
            
            # 최소한의 조건만 적용
            if df_ratio <= 0.95 and idf_score >= 0.05:  # 매우 관대한 조건
                quality_score = idf_score * (1.0 - df_ratio)
                if quality_score > best_score:
                    best_score = quality_score
                    best_token = token
        
        if best_token:
            kept = [best_token]
    
    return kept


def filter_semantic_terms_tokens_unfiltered(
    phrases: List[str],
    df_global: Dict[str,int],
    idf_global: Dict[str,float],
    N_global: int,
    **kwargs
) -> List[str]:
    """확장어 필터: 정제 없이 무조건적 사용 (②번 방식)"""
    tokens = []
    for ph in phrases or []:
        for t in tokenize_en(ph):
            if len(t) >= 2:  # 최소 길이만 체크, DENYLIST 무시
                tokens.append(t)
    tokens = dedup(tokens)

    # 정제 없이 그대로 사용
    return tokens

def filter_semantic_terms(terms: List[str], df_dict: Dict[str, int], N: int, 
                         idf: Dict[str, float], config: KeywordFilterConfig) -> List[str]:
    """기존 호환성을 위한 래퍼 함수"""
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

def soft_semantic_bonus_tokens_capped(
    doc_tokens: Set[str],
    idf_global: Dict[str,float],
    expanded_tokens: List[str],
    per_token_cap: float = 0.6,   # 토큰당 최대 기여
    per_doc_cap: float = 1.5      # 문서당 총 보너스 상한
) -> float:
    """보너스 계산: 토큰 기반 + 문서당 상한(cap)"""
    bonus = 0.0
    for t in expanded_tokens:
        if t in doc_tokens:
            bonus += min(per_token_cap, idf_global.get(t, 0.0))
            if bonus >= per_doc_cap:
                return per_doc_cap
    return bonus

def percentile_confidence_scoring(query_tokens: Set[str], expanded_tokens: List[str], 
                                document_tokens: Set[str], idf_dict: Dict[str, float],
                                confidence_percentile: float = 95.0) -> float:
    """퍼센타일 기반 신뢰도 점수 계산 - 원본 쿼리 + 확장 키워드 모두 평가"""
    
    # 1. 원본 쿼리 + 확장 키워드 모두 평가
    all_query_terms = query_tokens.union(set(expanded_tokens))
    
    # 2. 문서와의 매칭 점수 계산
    matching_scores = []
    for term in all_query_terms:
        if term in document_tokens:
            matching_scores.append(idf_dict.get(term, 0.0))
    
    if not matching_scores:
        return 0.0
    
    # 3. 퍼센타일 방식으로 신뢰도 측정
    matching_scores.sort(reverse=True)
    percentile_idx = int(len(matching_scores) * (confidence_percentile / 100.0))
    percentile_idx = min(percentile_idx, len(matching_scores) - 1)
    
    # 4. 신뢰도 균형보정 (상위 퍼센타일의 가중 평균)
    if len(matching_scores) >= 3:
        # 상위 3개 점수의 가중 평균
        weights = [0.5, 0.3, 0.2]
        confidence_score = sum(score * weight for score, weight in zip(matching_scores[:3], weights))
    else:
        # 점수가 적으면 단순 평균
        confidence_score = sum(matching_scores) / len(matching_scores)
    
    return confidence_score

def simple_drop_logic(confidence_score: float, min_confidence_threshold: float = 0.1) -> bool:
    """단순한 Drop 로직 - 정답에 대한 확신이 없으면 Drop"""
    return confidence_score < min_confidence_threshold

def hybrid_confidence_gate(query_tokens: Set[str], expanded_phrases: List[str], 
                          documents: List[str], final_global_order: List[int], 
                          final_scores: List[float], idf_dict: Dict[str, float],
                          semantic_data: Dict[str, Any],
                          top_k: int = 20, percentile_threshold: float = 80.0,
                          margin_threshold: float = None, absolute_threshold: float = None) -> Dict[str, Any]:
    """하이브리드 신뢰도 게이트: 퍼센타일 + 마진 + 절대 임계값"""
    
    if len(final_global_order) < 2:
        return {"action": "keep", "reason": "insufficient_documents"}
    
    # 1. 상위 K개 문서의 신뢰도 점수 계산
    top_k_docs = final_global_order[:min(top_k, len(final_global_order))]
    confidence_scores = []
    
    for doc_idx in top_k_docs:
        doc_tokens = set(tokenize_en(documents[doc_idx]))
        expanded_tokens = []
        for phrase in expanded_phrases:
            expanded_tokens.extend(tokenize_en(phrase))
        
        confidence_score = percentile_confidence_scoring(
            query_tokens, expanded_tokens, doc_tokens, idf_dict, 95.0
        )
        confidence_scores.append(confidence_score)
    
    # 2. 퍼센타일 정규화
    confidence_scores.sort(reverse=True)
    percentile_idx = int(len(confidence_scores) * (percentile_threshold / 100.0))
    percentile_idx = min(percentile_idx, len(confidence_scores) - 1)
    percentile_threshold_value = confidence_scores[percentile_idx]
    
    # 3. 상대 마진 규칙 (top-1 vs top-2)
    if margin_threshold is None:
        # 상위 20의 표준편차 0.5σ로 자동 설정
        if len(confidence_scores) >= 2:
            margin_threshold = np.std(confidence_scores[:min(20, len(confidence_scores))]) * 0.5
        else:
            margin_threshold = 0.1
    
    # 4. 절대 임계값 (개발셋 상위20 평균의 30~40% 분위)
    if absolute_threshold is None:
        if len(confidence_scores) >= 5:
            absolute_threshold = np.percentile(confidence_scores[:min(20, len(confidence_scores))], 35)
        else:
            absolute_threshold = 0.05
    
    # 5. 하이브리드 결정 규칙
    top1_confidence = confidence_scores[0] if confidence_scores else 0.0
    top2_confidence = confidence_scores[1] if len(confidence_scores) > 1 else 0.0
    margin = top1_confidence - top2_confidence
    
    # 규칙 A: 퍼센타일 규칙
    rule_a_fail = top1_confidence < percentile_threshold_value
    
    # 규칙 B: 마진 규칙
    rule_b_fail = margin < margin_threshold
    
    # 규칙 C: 절대 임계값 규칙
    rule_c_fail = top1_confidence < absolute_threshold
    
    # 하이브리드 결정
    if rule_b_fail:  # 우선순위 1: 마진
        action = "demote"
        reason = "margin_rule"
        trigger_rule = "B"
    elif rule_a_fail:  # 우선순위 2: 퍼센타일
        action = "demote"
        reason = "percentile_rule"
        trigger_rule = "A"
    elif rule_c_fail:  # 우선순위 3: 절대 임계값
        action = "demote"
        reason = "absolute_threshold"
        trigger_rule = "C"
    else:
        action = "keep"
        reason = "all_rules_passed"
        trigger_rule = "none"
    
    # 6. entailment_checklist 체크 (새로운 규칙 D)
    checklist = semantic_data.get("entailment_checklist", [])
    checklist_hit_ratio = 0.0
    if checklist and final_global_order:
        top_doc_text = documents[final_global_order[0]]
        hits = 0
        for item in checklist:
            # 간단한 패턴 매칭 (향후 LLM verifier로 확장 가능)
            if any(word.lower() in top_doc_text.lower() for word in item.split() if len(word) > 3):
                hits += 1
        checklist_hit_ratio = hits / len(checklist) if checklist else 0.0
    
    # 규칙 D: 체크리스트 히트율
    rule_d_fail = checklist_hit_ratio < 0.3  # 30% 미만이면 실패
    
    # 하이브리드 결정 (체크리스트 규칙 추가)
    if rule_b_fail:  # 우선순위 1: 마진
        action = "demote"
        reason = "margin_rule"
        trigger_rule = "B"
    elif rule_d_fail:  # 우선순위 2: 체크리스트
        action = "demote"
        reason = "checklist_rule"
        trigger_rule = "D"
    elif rule_a_fail:  # 우선순위 3: 퍼센타일
        action = "demote"
        reason = "percentile_rule"
        trigger_rule = "A"
    elif rule_c_fail:  # 우선순위 4: 절대 임계값
        action = "demote"
        reason = "absolute_threshold"
        trigger_rule = "C"
    else:
        action = "keep"
        reason = "all_rules_passed"
        trigger_rule = "none"
    
    return {
        "action": action,
        "reason": reason,
        "trigger_rule": trigger_rule,
        "top1_confidence": top1_confidence,
        "top2_confidence": top2_confidence,
        "margin": margin,
        "percentile_threshold": percentile_threshold_value,
        "margin_threshold": margin_threshold,
        "absolute_threshold": absolute_threshold,
        "checklist_hit_ratio": checklist_hit_ratio,
        "rule_a_fail": rule_a_fail,
        "rule_b_fail": rule_b_fail,
        "rule_c_fail": rule_c_fail,
        "rule_d_fail": rule_d_fail
    }

def rank_boost_for_p1(query_tokens: Set[str], expanded_tokens: List[str], 
                     document_tokens: Set[str], idf_dict: Dict[str, float],
                     base_score: float, boost_multiplier: float = 3.0) -> float:
    """P@1을 위한 강력한 순위 부스트 - 1위 변경 목표"""
    
    # 1. 원본 쿼리 + 확장 키워드 모두 평가
    all_query_terms = query_tokens.union(set(expanded_tokens))
    
    # 2. 문서와의 매칭 점수 계산
    matching_scores = []
    for term in all_query_terms:
        if term in document_tokens:
            matching_scores.append(idf_dict.get(term, 0.0))
    
    if not matching_scores:
        return base_score
    
    # 3. 강력한 부스트 계산 (1위 변경 목표)
    matching_scores.sort(reverse=True)
    
    # 상위 매칭 점수들의 가중 합 (더 공격적)
    if len(matching_scores) >= 3:
        weights = [0.6, 0.3, 0.1]  # 더 공격적인 가중치
        boost_score = sum(score * weight for score, weight in zip(matching_scores[:3], weights))
    else:
        boost_score = sum(matching_scores) / len(matching_scores)
    
    # 4. 강력한 부스트 적용 (1위 변경을 위한 배수)
    boosted_score = base_score + (boost_score * boost_multiplier)
    
    return boosted_score

def phrase_proximity_bonus(query_tokens: Set[str], expanded_phrases: List[str], 
                          document_text: str, idf_dict: Dict[str, float],
                          phrase_weight: float = 2.0, proximity_weight: float = 1.5) -> float:
    """프레이즈 매칭과 근접성 보너스 계산"""
    
    bonus = 0.0
    doc_tokens = tokenize_en(document_text)
    
    # 1. 프레이즈 매칭 보너스
    for phrase in expanded_phrases:
        phrase_tokens = tokenize_en(phrase)
        if len(phrase_tokens) >= 2:  # 2그램 이상만
            # 문서에서 프레이즈가 연속으로 나타나는지 확인
            for i in range(len(doc_tokens) - len(phrase_tokens) + 1):
                if doc_tokens[i:i+len(phrase_tokens)] == phrase_tokens:
                    # 프레이즈 매칭 보너스
                    phrase_idf = sum(idf_dict.get(token, 0.0) for token in phrase_tokens)
                    bonus += phrase_weight * phrase_idf
                    break
    
    # 2. 근접성 보너스 (쿼리 핵심어와 확장어가 가까이 있을 때)
    query_core_tokens = [t for t in query_tokens if idf_dict.get(t, 0.0) > 0.1]
    expanded_tokens = []
    for phrase in expanded_phrases:
        expanded_tokens.extend(tokenize_en(phrase))
    
    for i, token in enumerate(doc_tokens):
        if token in query_core_tokens:
            # 주변 12토큰 내에서 확장어 찾기
            start = max(0, i - 6)
            end = min(len(doc_tokens), i + 7)
            nearby_tokens = doc_tokens[start:end]
            
            for exp_token in expanded_tokens:
                if exp_token in nearby_tokens:
                    # 근접성 보너스
                    proximity_bonus = proximity_weight * idf_dict.get(exp_token, 0.0)
                    bonus += proximity_bonus
    
    return bonus

def _semantic_decisiveness(doc_text: str, semantic_data: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
    """문서의 결정적 시그널 강도를 측정 (expanded.keywords 기반)"""
    txt = (doc_text or "").lower()
    
    # 기본 필드들 (LLM이 생성하지 않을 수 있음)
    must = semantic_data.get("must_have_ngrams", []) or []
    anchors = semantic_data.get("anchor_phrases", []) or []
    pos = (semantic_data.get("contrastive_signals", {}) or {}).get("positive_terms", []) or []
    checklist = semantic_data.get("entailment_checklist", []) or []
    
    # expanded.keywords를 대체 시그널로 사용
    expanded_keywords = []
    if isinstance(semantic_data.get("expanded"), dict):
        expanded_keywords = semantic_data["expanded"].get("keywords", []) or []
    elif isinstance(semantic_data.get("expanded_keywords"), list):
        expanded_keywords = semantic_data["expanded_keywords"] or []

    # 하드 필터 통과 여부 (anti/constraints/hard-negative)
    passed, _ = hard_filter_document(doc_text, semantic_data)

    def _hits(phrases):
        h = 0
        for ph in phrases:
            ph = (ph or "").lower().strip()
            if ph and ph in txt:
                h += 1
        return h

    must_hits = _hits(must)
    anchor_hits = _hits(anchors)
    pos_hits = _hits(pos)
    expanded_hits = _hits(expanded_keywords)

    checklist_hits = 0
    for item in checklist:
        if any(w for w in item.split() if w.lower() in txt):
            checklist_hits += 1
    checklist_ratio = (checklist_hits / len(checklist)) if checklist else 0.0

    # 결정성 점수: expanded.keywords를 주요 시그널로 사용
    score = (
        10.0 * (1 if must_hits > 0 else 0) +     # must-have 있으면 사실상 결정
        6.0  * (1 if passed else 0) +
        2.0  * anchor_hits +
        1.5  * pos_hits +
        4.0  * checklist_ratio +
        3.0  * expanded_hits  # expanded.keywords 매칭 보너스
    )
    return score, {
        "must": float(must_hits),
        "constraints_passed": 1.0 if passed else 0.0,
        "anchor_hits": float(anchor_hits),
        "pos_hits": float(pos_hits),
        "checklist_ratio": float(checklist_ratio),
        "expanded_hits": float(expanded_hits),
    }

def semantic_top1_override(
    documents: List[str],
    final_global_order: List[int],
    final_scores: List[float],
    semantic_data: Optional[Dict[str, Any]],
    head_k: int = 5
) -> Tuple[List[int], List[float], Dict[str, Any]]:
    """결정적 시그널 기반으로 top-1을 '승격'하는 마지막 단계."""
    info = {"applied": False}
    if not semantic_data or not final_global_order:
        return final_global_order, final_scores, info

    head = final_global_order[:min(head_k, len(final_global_order))]
    feats = []
    for doc_id in head:
        s, f = _semantic_decisiveness(documents[doc_id], semantic_data)
        feats.append((doc_id, s, f))

    # 현재 1위와 최고의 결정성 후보 비교
    cur_id = head[0]
    cur_s, cur_f = next((s, f) for d, s, f in feats if d == cur_id)
    best_id, best_s, best_f = max(feats, key=lambda x: x[1])

    # 승격 조건 (더 관대하게, P@1 개선을 위해)
    promote = False
    # (1) must-have 존재: best에 있고 current엔 없으면 무조건 승격
    if best_f["must"] > 0 and cur_f["must"] == 0:
        promote = True
    # (2) 제약 준수: best 통과, current 실패면 승격
    elif best_f["constraints_passed"] > cur_f["constraints_passed"]:
        promote = True
    # (3) expanded.keywords 매칭: best가 current보다 매칭하면 승격 (더 관대하게)
    elif best_f["expanded_hits"] > cur_f["expanded_hits"]:
        promote = True
    # (4) 종합 결정성 점수 차이가 충분 (>= 1.0, 매우 관대하게)
    elif (best_s - cur_s) >= 1.0:
        promote = True
    # (5) 최후의 수단: best가 current보다 조금이라도 높으면 승격
    elif best_s > cur_s:
        promote = True

    if promote and (best_id != cur_id):
        # best를 맨 앞으로, 나머지 순서는 유지
        new_order = [best_id] + [d for d in final_global_order if d != best_id]
        # 점수는 기존 배열 순서대로 재배치
        id2score = {d: s for d, s in zip(final_global_order, final_scores)}
        new_scores = [id2score[d] for d in new_order]
        info = {
            "applied": True,
            "prev_top": cur_id,
            "new_top": best_id,
            "prev_feat": cur_f,
            "new_feat": best_f,
            "prev_score": cur_s,
            "new_score": best_s
        }
        return new_order, new_scores, info

    # 디버그 정보 추가
    info["debug"] = {
        "head_docs": head,
        "feats": feats,
        "cur_id": cur_id,
        "cur_score": cur_s,
        "best_id": best_id,
        "best_score": best_s,
        "promote": promote,
        "same_id": best_id == cur_id
    }
    return final_global_order, final_scores, info

def top1_answer_shape_promoter(
    documents: List[str],
    final_global_order: List[int],
    final_scores: List[float],
    semantic_data: Optional[Dict[str, Any]],
    query: str,
    head_k: int = 12,
    tau: float = 2.5,
    strong: float = 4.0,
    cur_min: float = 1.0
) -> Tuple[List[int], List[float], Dict[str, Any]]:
    info = {"applied": False}
    
    # 더 엄격한 입력 검증
    if not semantic_data or not final_global_order or not documents:
        return final_global_order or [], final_scores or [], info
    
    if not isinstance(final_global_order, list) or not isinstance(final_scores, list):
        return final_global_order or [], final_scores or [], info
    
    if len(final_global_order) == 0:
        return final_global_order, final_scores, info

    try:
        head = final_global_order[:min(head_k, len(final_global_order))]
        stats = []
        for doc_id in head:
            if doc_id < len(documents):  # 인덱스 범위 검증
                s, f = answer_shape_score(documents[doc_id], semantic_data, query)
                stats.append((doc_id, s, f))
        
        if not stats:
            return final_global_order, final_scores, info

        cur_id = head[0]
        cur_s, cur_f = next((s, f) for d, s, f in stats if d == cur_id)
        best_id, best_s, best_f = max(stats, key=lambda x: x[1])

        promote = False
        # 규칙 1: best가 강함(strong)이고 current가 약함(cur_min 미만) → 무조건 승격
        if best_s >= strong and cur_s < cur_min:
            promote = True
        # 규칙 2: best - current >= tau → 승격
        elif (best_s - cur_s) >= tau:
            promote = True

        if promote and (best_id != cur_id):
            new_order = [best_id] + [d for d in final_global_order if d != best_id]
            id2score = {d: s for d, s in zip(final_global_order, final_scores)}
            new_scores = [id2score[d] for d in new_order]
            info = {
                "applied": True,
                "prev_top": cur_id,
                "new_top": best_id,
                "prev_shape": cur_s,
                "new_shape": best_s,
                "feats_prev": cur_f,
                "feats_new": best_f
            }
            return new_order, new_scores, info

        info.update({
            "applied": False,
            "cur_top": cur_id, "cur_shape": cur_s,
            "best_id": best_id, "best_shape": best_s
        })
        return final_global_order, final_scores, info
    
    except Exception as e:
        info["error"] = str(e)
        return final_global_order, final_scores, info

def hard_filter_document(document_text: str, semantic_data: Dict[str, Any]) -> Tuple[bool, str]:
    """극단 정밀 모드: 하드 필터 (anti_ngrams/constraints)"""
    
    doc_raw = document_text or ""
    doc_lower = doc_raw.lower()
    
    # 1. anti_ngrams 체크 (혼동어구가 있으면 즉시 Drop)
    anti_ngrams = semantic_data.get("anti_ngrams", [])
    if anti_ngrams:
        for anti_ngram in anti_ngrams:
            if anti_ngram and anti_ngram.lower() in doc_lower:
                return False, f"anti_ngram_detected: {anti_ngram}"
    
    # 2. constraints 체크
    constraints = semantic_data.get("constraints", {}) or {}
    
    # 단위 체크 (다양한 표기 허용)
    unit = constraints.get("unit")
    if unit:
        u = str(unit).lower()
        if u not in doc_lower:
            # 흔한 변형(%, bps -> basis points 등) 보완
            variants = {
                "%": [" percent", "percentage", " per cent"],
                "bps": ["basis points", "bp"],
                "usd": ["$", " us$"],
            }
            cand = variants.get(u, [])
            if not any(v in doc_lower for v in cand):
                return False, f"unit_constraint_failed: {unit}"
    
    # 관할권 체크
    jurisdiction = constraints.get("jurisdiction")
    if jurisdiction and str(jurisdiction).lower() not in doc_lower:
        return False, f"jurisdiction_constraint_failed: {jurisdiction}"
    
    # 엔티티 체크
    entities = constraints.get("entity", []) or []
    if entities:
        if not any(str(e).lower() in doc_lower for e in entities):
            return False, f"entity_constraint_failed: {entities}"
    
    # 3. hard_negative_terms 체크
    contrastive_signals = semantic_data.get("contrastive_signals", {})
    hard_negative_terms = contrastive_signals.get("hard_negative_terms", [])
    if hard_negative_terms:
        for term in hard_negative_terms:
            if term and str(term).lower() in doc_lower:
                return False, f"hard_negative_term_detected: {term}"
    
    return True, "passed_all_filters"

def enhanced_rank_boost_for_p1(query_tokens: Set[str], semantic_data: Dict[str, Any], 
                              document_tokens: Set[str], document_text: str, 
                              idf_dict: Dict[str, float], base_score: float, 
                              boost_multiplier: float = 3.0) -> float:
    """극단 정밀 모드: P@1을 위한 강화된 순위 부스트 (must_have_ngrams/anchor_phrases 포함)"""
    
    # 1. 기존 토큰 기반 부스트
    expanded_phrases = semantic_data.get("expanded", {}).get("keywords", [])
    expanded_tokens = []
    for phrase in expanded_phrases:
        expanded_tokens.extend(tokenize_en(phrase))
    
    token_boost = rank_boost_for_p1(query_tokens, expanded_tokens, document_tokens, 
                                   idf_dict, base_score, boost_multiplier)
    
    # 2. 프레이즈 + 근접성 보너스
    phrase_bonus = phrase_proximity_bonus(query_tokens, expanded_phrases, 
                                         document_text, idf_dict)
    
    # 3. must_have_ngrams 보너스 (정답 문서만의 결정적 패턴)
    must_have_ngrams = semantic_data.get("must_have_ngrams", [])
    must_have_bonus = 0.0
    if must_have_ngrams:
        doc_lower = document_text.lower()
        for ngram in must_have_ngrams:
            if ngram.lower() in doc_lower:
                # 결정적 패턴이 있으면 큰 보너스
                ngram_tokens = tokenize_en(ngram)
                ngram_idf = sum(idf_dict.get(token, 0.0) for token in ngram_tokens)
                must_have_bonus += 8.0 * ngram_idf  # 8배 가중치 (5.0->8.0)
    
    # 4. anchor_phrases 보너스 (프레이즈 매칭 강화)
    anchor_phrases = semantic_data.get("anchor_phrases", [])
    anchor_bonus = 0.0
    if anchor_phrases:
        doc_lower = document_text.lower()
        for phrase in anchor_phrases:
            if phrase.lower() in doc_lower:
                # 앵커 프레이즈가 있으면 2배 가중치
                phrase_tokens = tokenize_en(phrase)
                phrase_idf = sum(idf_dict.get(token, 0.0) for token in phrase_tokens)
                anchor_bonus += 2.0 * phrase_idf  # 2배 가중치
    
    # 5. positive_terms 보너스
    contrastive_signals = semantic_data.get("contrastive_signals", {})
    positive_terms = contrastive_signals.get("positive_terms", [])
    positive_bonus = 0.0
    if positive_terms:
        doc_lower = document_text.lower()
        for term in positive_terms:
            if term.lower() in doc_lower:
                term_tokens = tokenize_en(term)
                term_idf = sum(idf_dict.get(token, 0.0) for token in term_tokens)
                positive_bonus += 1.5 * term_idf  # 1.5배 가중치
    
    # 6. 최종 점수 = 토큰 부스트 + 프레이즈 보너스 + 결정적 패턴 보너스
    final_score = token_boost + phrase_bonus + must_have_bonus + anchor_bonus + positive_bonus
    
    return final_score

def soft_semantic_bonus_tokens(doc_tokens: Set[str], idf: Dict[str, float], 
                              expanded_terms: List[str], config: KeywordFilterConfig) -> float:
    """기존 호환성을 위한 래퍼 함수"""
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
                          semantic_data: Optional[Dict[str, Any]], config: KeywordFilterConfig,
                          df_global: Dict[str,int] = None, idf_global: Dict[str,float] = None, N_global: int = None,
                          expansion_mode: str = "filtered") -> Tuple[List[int], List[float], Dict[str, Any]]:
    """후보 기반 재랭킹 (FiQA 스타일)"""
    N = len(documents)
    
    # 상위 후보 선택 (P@1 최적화를 위한 확장)
    cand_idx = list(range(N))
    cand_idx.sort(key=lambda i: base_scores[i], reverse=True)
    
    # P@1 최적화: 후보 확장으로 정답이 후보에 포함될 확률 증가
    if config.expand_candidates:
        expanded_candidate_count = min(int(config.top_r_pure * config.candidate_expansion_factor), N)
        cand_idx = cand_idx[:expanded_candidate_count]
    else:
        cand_idx = cand_idx[:min(config.top_r_pure, N)]
    
    # 후보 토큰 인덱스
    cand_texts = [documents[i] for i in cand_idx]
    doc_index_local = build_doc_token_index(cand_texts, tokenize_en)  # key: local idx
    doc_tokens_by_global = { cand_idx[loc]: doc_index_local[loc][0] for loc in range(len(cand_idx)) }

    # 전역 통계 사용 여부 확인
    if df_global is not None and idf_global is not None and N_global is not None:
        # 확장어 추출 (직접 추출, 필터링 없음)
        expanded_phrases = []
        if semantic_data and isinstance(semantic_data.get("expanded"), dict):
            expanded_phrases = semantic_data["expanded"].get("keywords", [])
        if not expanded_phrases:
            expanded_phrases = semantic_data.get("expanded_keywords", []) if semantic_data else []
        # 확장 모드에 따른 토큰 처리
        if expansion_mode == "unfiltered":
            # ②번 방식: 정제 없이 무조건적 사용
            expanded_tokens = filter_semantic_terms_tokens_unfiltered(
                expanded_phrases, df_global, idf_global, N_global
            )
        else:
            # ③번 방식: 정제된 확장 (기본값)
            expanded_tokens = filter_semantic_terms_tokens(
                expanded_phrases, df_global, idf_global, N_global, config
            )
        
        # PRF fallback (확장 토큰이 너무 적을 때만)
        if not expanded_tokens and config.enable_prf_fallback:
            prf_terms = prf_rm3_terms(cand_texts, np.array([base_scores[i] for i in cand_idx]))
            if expansion_mode == "unfiltered":
                expanded_tokens = filter_semantic_terms_tokens_unfiltered(
                    prf_terms, df_global, idf_global, N_global
                )
            else:
                expanded_tokens = filter_semantic_terms_tokens(
                    prf_terms, df_global, idf_global, N_global
                )
    else:
        # 기존 방식 (호환성)
        df_dict = build_df_dict(cand_texts, tokenize_en)
        idf = build_idf_dict_from_df(df_dict, len(cand_texts))
        expanded = extract_expanded(semantic_data, df_dict, len(cand_texts), idf, config)
        
        # PRF fallback
        if not expanded and config.enable_prf_fallback:
            prf_terms = prf_rm3_terms(cand_texts, np.array([base_scores[i] for i in cand_idx]))
            expanded = filter_semantic_terms(prf_terms, df_dict, len(cand_texts), idf, config)
        expanded_tokens = expanded
    
    # P@1 최적화를 위한 강력한 부스트 재랭킹
    query_tokens = set(tokenize_en(query))
    pairs: List[Tuple[int, float]] = []
    
    for loc, (_tokset, _passage) in doc_index_local.items():
        base_score = float(base_scores[cand_idx[loc]])
        
        if expanded_tokens and df_global is not None and idf_global is not None:
            # 극단 정밀 모드: 하드 필터 먼저 적용
            if semantic_data:
                passed_filter, filter_reason = hard_filter_document(_passage, semantic_data)
                if not passed_filter:
                    # 하드 필터에 걸리면 매우 낮은 점수로 강등
                    final_score = base_score * 0.1
                else:
                    # 강화된 순위 부스트 적용 (새로운 스키마 포함)
                    final_score = enhanced_rank_boost_for_p1(
                        query_tokens, semantic_data, _tokset, _passage, idf_global, 
                        base_score, config.boost_multiplier
                    )
            else:
                # 기존 방식 (호환성)
                expanded_phrases = []
                if semantic_data and isinstance(semantic_data.get("expanded"), dict):
                    expanded_phrases = semantic_data["expanded"].get("keywords", [])
                if not expanded_phrases:
                    expanded_phrases = semantic_data.get("expanded_keywords", []) if semantic_data else []
                
                final_score = enhanced_rank_boost_for_p1(
                    query_tokens, {"expanded": {"keywords": expanded_phrases}}, _tokset, _passage, idf_global, 
                    base_score, config.boost_multiplier
                )
        else:
            # 확장 토큰이 없으면 원본 점수 유지
            final_score = base_score
            
        pairs.append((loc, final_score))
    
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
    
    # 앵커 보호 (anchor_k=0이면 앵커 보호 없음)
    if config.anchor_k > 0:
        anchor_locs = sorted(range(len(cand_idx)),
                             key=lambda i: base_scores[cand_idx[i]], reverse=True)[:config.anchor_k]
        anchor_set = set(anchor_locs)
        tail = [p for p in tmp if p[0] not in anchor_set]
        final_pairs = [(loc, float(base_scores[cand_idx[loc]])) for loc in anchor_locs] + tail
    else:
        # anchor_k=0: 부스트 점수로만 정렬 (1위 변경 허용)
        final_pairs = tmp
    
    final_local_order = [loc for loc, _ in final_pairs]
    final_global_order = [cand_idx[loc] for loc in final_local_order]
    # 부스트 점수를 유지 (base_scores로 덮어쓰지 않음)
    final_scores = [score for _, score in final_pairs]
    
    # P@1 진단을 위한 로깅
    candidate_recall = 0
    if hasattr(config, 'relevant_doc_ids') and config.relevant_doc_ids:
        relevant_in_candidates = sum(1 for doc_id in config.relevant_doc_ids if doc_id in cand_idx)
        candidate_recall = relevant_in_candidates / len(config.relevant_doc_ids)
    
    debug_info = {
        "semantic_applied": bool(expanded_tokens),
        "filtered_terms": {"expanded_tokens": expanded_tokens},
        "skipped_by_dfidf": not bool(expanded_tokens),
        "mode": "candidate_based_global_stats" if df_global is not None else "candidate_based",
        "p1_diagnostics": {
            "candidate_recall": candidate_recall,
            "candidate_count": len(cand_idx),
            "expanded_tokens_count": len(expanded_tokens) if expanded_tokens else 0,
            "boost_multiplier": config.boost_multiplier,
            "anchor_k": config.anchor_k
        }
    }
    
    # --- [NEW] Precision-oriented conditional DROP ---
    if expanded_tokens:  # 확장 토큰이 있을 때만 Drop 로직 적용
        original_count = len(final_global_order)
        
        # 최종 1위 직전: 하이브리드 신뢰도 게이트 적용
        if final_global_order and df_global is not None and idf_global is not None:
            # 확장 프레이즈 추출
            expanded_phrases = []
            if semantic_data and isinstance(semantic_data.get("expanded"), dict):
                expanded_phrases = semantic_data["expanded"].get("keywords", [])
            if not expanded_phrases:
                expanded_phrases = semantic_data.get("expanded_keywords", []) if semantic_data else []
            
            # 하이브리드 게이트 적용
            gate_result = hybrid_confidence_gate(
                query_tokens, expanded_phrases, documents, final_global_order, 
                final_scores, idf_global, semantic_data, top_k=20, percentile_threshold=80.0
            )
            
            if gate_result["action"] == "demote" and len(final_global_order) > 1:
                # 1위를 맨 뒤로 보내고 2등을 1등으로 승격
                demoted_doc = final_global_order[0]
                demoted_score = final_scores[0]
                final_global_order = final_global_order[1:] + [demoted_doc]
                final_scores = final_scores[1:] + [demoted_score]
                debug_info["hybrid_confidence_gate"] = {
                    **gate_result,
                    "demoted_top_doc": True,
                    "new_top_doc": final_global_order[0]
                }
            else:
                debug_info["hybrid_confidence_gate"] = {
                    **gate_result,
                    "demoted_top_doc": False
                }
        
        # 기존 Precision Drop 로직 (글로벌 토큰 매핑 사용)
        final_global_order, final_scores = apply_precision_drop_logic(
            query, documents, final_global_order, final_scores, 
            expanded_tokens, debug_info, doc_tokens_by_global, None
        )
        dropped_count = original_count - len(final_global_order)
        debug_info["precision_drop"] = {
            "original_count": original_count,
            "dropped_count": dropped_count,
            "final_count": len(final_global_order)
        }
    
    # 결정적 시그널 기반 top-1 승격 오버라이드
    final_global_order, final_scores, ov_info = semantic_top1_override(
        documents, final_global_order, final_scores, semantic_data, head_k=5
    )
    debug_info["top1_override"] = ov_info
    
    # answer-shape 승격기
    final_global_order, final_scores, as_info = top1_answer_shape_promoter(
        documents, final_global_order, final_scores, semantic_data, query,
        head_k=config.top1_head_k,
        tau=config.answer_shape_promote_tau,
        strong=config.answer_shape_strong,
        cur_min=config.answer_shape_min_for_current
    )
    debug_info["top1_answer_shape_promoter"] = as_info
    
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
    
    # 확장 키워드 추출 (직접 추출, 필터링 없음)
    expanded_phrases = []
    if semantic_data and isinstance(semantic_data.get("expanded"), dict):
        expanded_phrases = semantic_data["expanded"].get("keywords", [])
    if not expanded_phrases:
        expanded_phrases = semantic_data.get("expanded_keywords", []) if semantic_data else []
    
    # 토큰화 및 필터링
    expanded = []
    for phrase in expanded_phrases:
        for t in tokenize_en(phrase):
            if len(t) >= 3 and t not in DENYLIST:
                expanded.append(t)
    expanded = dedup(expanded)
    
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
        "filtered_terms": {"expanded_tokens": expanded},
        "skipped_by_dfidf": not bool(expanded),
        "mode": "full_document"
    }
    
    # 결정적 시그널 기반 top-1 승격 오버라이드
    final_order, final_scores, ov_info = semantic_top1_override(
        documents, final_order, final_scores, semantic_data, head_k=5
    )
    debug_info["top1_override"] = ov_info
    
    # answer-shape 승격기
    final_order, final_scores, as_info = top1_answer_shape_promoter(
        documents, final_order, final_scores, semantic_data, query,
        head_k=config.top1_head_k,
        tau=config.answer_shape_promote_tau,
        strong=config.answer_shape_strong,
        cur_min=config.answer_shape_min_for_current
    )
    debug_info["top1_answer_shape_promoter"] = as_info
    
    return final_order, final_scores, debug_info


# -----------------------------
# Main Reranking Function
# -----------------------------
def semantic_rerank(query: str, documents: List[str], base_scores: np.ndarray,
                   semantic_data: Optional[Dict[str, Any]], config: KeywordFilterConfig,
                   df_global: Dict[str,int] = None, idf_global: Dict[str,float] = None, N_global: int = None,
                   expansion_mode: str = "filtered") -> Tuple[List[int], List[float], Dict[str, Any]]:
    """통합 시맨틱 재랭킹 함수"""
    
    # 적응형 방식 선택
    if config.adaptive_mode and len(documents) <= config.candidate_threshold:
        return candidate_based_rerank(query, documents, base_scores, semantic_data, config, df_global, idf_global, N_global, expansion_mode)
    else:
        return full_document_rerank(query, documents, base_scores, semantic_data, config)


# -----------------------------
# Metrics Accumulator
# -----------------------------
class MetricsAccumulator:
    """메트릭 누적 클래스"""
    
    def __init__(self):
        # 랭킹 지표
        self.n = 0
        self.sum_p1 = 0.0
        self.sum_p10 = 0.0
        self.sum_p100 = 0.0
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
        
        # 필터 지표 (단계별 누적)
        self.stage_metrics_accumulator = {
            "strict_matching": [],
            "precision_verification": [],
            "recall_adjustment": [],
            "final_pipeline": [],
            "llm_filter": []
        }
    
    def add(self, ranked_labels: List[int], total_rel_all: int, sem_applied: bool = False):
        """메트릭 추가"""
        self.n += 1
        topk = ranked_labels[:10]
        top100 = ranked_labels[:100]
        total_rel = max(int(total_rel_all), 0)
        
        p1 = float(ranked_labels[0]) if ranked_labels else 0.0
        p10 = (sum(topk) / max(len(topk), 1)) if topk else 0.0
        p100 = (sum(top100) / max(len(top100), 1)) if top100 else 0.0
        r10 = (sum(topk) / max(total_rel, 1)) if total_rel > 0 else 0.0
        r100 = (sum(top100) / max(total_rel, 1)) if total_rel > 0 else 0.0
        mrr10 = mrr_at_k(ranked_labels, 10)
        ndcg10 = ndcg_at_k(ranked_labels, 10)
        ndcg100 = ndcg_at_k(ranked_labels, 100)
        map100 = map_at_k(ranked_labels, 100)
        
        self.sum_p1 += p1
        self.sum_p10 += p10
        self.sum_p100 += p100
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
    
    def add_filter_metrics(self, stage_metrics: List[Dict[str, float]]):
        """필터 지표 추가"""
        for stage_metric in stage_metrics:
            stage_name = stage_metric.get("stage", "unknown")
            if stage_name in self.stage_metrics_accumulator:
                self.stage_metrics_accumulator[stage_name].append(stage_metric)
    
    def result_filter_metrics(self) -> Dict[str, Dict[str, float]]:
        """필터 지표 결과 반환"""
        result = {}
        for stage_name, metrics_list in self.stage_metrics_accumulator.items():
            if not metrics_list:
                result[stage_name] = {
                    "keep_precision": 0.0, "keep_recall": 0.0, "keep_fpr": 0.0,
                    "drop_precision": 0.0, "drop_recall": 0.0, "drop_fnr": 0.0,
                    "keep_fbeta_05": 0.0, "keep_fbeta_20": 0.0,
                    "drop_fbeta_05": 0.0, "drop_fbeta_20": 0.0
                }
                continue
            
            # 평균 계산
            n = len(metrics_list)
            result[stage_name] = {
                "keep_precision": sum(m.get("keep_precision", 0.0) for m in metrics_list) / n,
                "keep_recall": sum(m.get("keep_recall", 0.0) for m in metrics_list) / n,
                "keep_fpr": sum(m.get("keep_fpr", 0.0) for m in metrics_list) / n,
                "drop_precision": sum(m.get("drop_precision", 0.0) for m in metrics_list) / n,
                "drop_recall": sum(m.get("drop_recall", 0.0) for m in metrics_list) / n,
                "drop_fnr": sum(m.get("drop_fnr", 0.0) for m in metrics_list) / n,
                "keep_fbeta_05": sum(m.get("keep_fbeta_05", 0.0) for m in metrics_list) / n,
                "keep_fbeta_20": sum(m.get("keep_fbeta_20", 0.0) for m in metrics_list) / n,
                "drop_fbeta_05": sum(m.get("drop_fbeta_05", 0.0) for m in metrics_list) / n,
                "drop_fbeta_20": sum(m.get("drop_fbeta_20", 0.0) for m in metrics_list) / n,
            }
        return result

    def result(self) -> Dict[str, float]:
        """결과 메트릭 반환"""
        if self.n == 0:
            return {"P@1":0.0, "P@10":0.0, "P@100":0.0, "R@10":0.0, "R@100":0.0, "MRR@10":0.0, "nDCG@10":0.0, "nDCG@100":0.0, "MAP@100":0.0}
        return {
            "P@1": self.sum_p1 / self.n,
            "P@10": self.sum_p10 / self.n,
            "P@100": self.sum_p100 / self.n,
            "R@10": self.sum_r10 / self.n,
            "R@100": self.sum_r100 / self.n,
            "MRR@10": self.sum_mrr10 / self.n,
            "nDCG@10": self.sum_ndcg10 / self.n,
            "nDCG@100": self.sum_ndcg100 / self.n,
            "MAP@100": self.sum_map100 / self.n,
        }
