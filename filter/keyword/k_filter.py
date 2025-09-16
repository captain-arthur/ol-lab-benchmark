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
        self.anchor_k = kwargs.get("anchor_k", 1)  # 3->1: 더 엄격한 앵커 보호
        self.rrf_k = kwargs.get("rrf_k", 3.0)  # 10.0->3.0: 매우 보수적 확장어 효과
        
        # 전체 문서 설정 (MS MARCO 스타일) - Precision 중심 튜닝 (매우 엄격)
        self.top_r_sem = kwargs.get("top_r_sem", 1000)  # 2000->1000: 더 엄격한 후보 선택
        self.expanded_weight = kwargs.get("expanded_weight", 0.05)  # 0.15->0.05: 매우 보수적 확장
        self.alpha_bonus = kwargs.get("alpha_bonus", 0.1)  # 0.2->0.1: 매우 보수적 보너스
        
        # 공통 안전장치 (Precision 중심 튜닝 - 매우 엄격)
        self.guardrail_k = kwargs.get("guardrail_k", 50)  # 150->50: 매우 엄격한 가드레일
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
    debug_info: Dict[str, Any]
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
    """강화된 시맨틱 확장 프롬프트"""
    return f"""You are a strict query-expansion assistant for financial IR.

RETURN FORMAT (MANDATORY):
- Return ONLY a single valid JSON object. No prose, no markdown.
- Schema:
{{
  "user_query": "{query}",
  "expanded": {{
    "keywords": ["kw1", "kw2", "kw3", "kw4", "kw5", "kw6"]
  }}
}}

HARD CONSTRAINTS (ALL MUST HOLD):
1) The array "keywords" MUST contain AT LEAST 6 and AT MOST 10 items
   — never fewer than 6, never more than 10.
2) Each keyword MUST be 2–4 words, all lowercase, distinct, no duplicates.
3) Use finance/technical terms tied to the intent of the query; avoid generic words.
4) Exclude stopwords (e.g., "and", "the", "for", "with", "from").

SELF-CORRECTION RULE:
- If your first draft violates ANY constraint, rewrite and output a corrected JSON that satisfies ALL constraints.

Query: "{query}"
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
                return {
                    "query": obj.get("user_query", original_query),
                    "expanded": {"keywords": cleaned}
                }
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
    
    # 상위 후보 선택
    cand_idx = list(range(N))
    cand_idx.sort(key=lambda i: base_scores[i], reverse=True)
    cand_idx = cand_idx[:min(config.top_r_pure, N)]
    
    # 후보 토큰 인덱스
    cand_texts = [documents[i] for i in cand_idx]
    doc_index = build_doc_token_index(cand_texts, tokenize_en)

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
    
    # 소프트 보너스 재랭킹 (전역 IDF, 보너스 capped)
    pairs: List[Tuple[int, float]] = []
    for loc, (_tokset, _passage) in doc_index.items():
        s = float(base_scores[cand_idx[loc]])
        if expanded_tokens and df_global is not None and idf_global is not None:
            s += config.alpha_soft_bonus * soft_semantic_bonus_tokens_capped(
                _tokset, idf_global, expanded_tokens,
                per_token_cap=0.6, per_doc_cap=1.5
            )
        elif expanded_tokens:
            # 기존 방식 (호환성)
            for t in expanded_tokens:
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
        "semantic_applied": bool(expanded_tokens),
        "filtered_terms": {"expanded_tokens": expanded_tokens},
        "skipped_by_dfidf": not bool(expanded_tokens),
        "mode": "candidate_based_global_stats" if df_global is not None else "candidate_based"
    }
    
    # --- [NEW] Precision-oriented conditional DROP ---
    if expanded_tokens:  # 확장 토큰이 있을 때만 Drop 로직 적용
        original_count = len(final_global_order)
        final_global_order, final_scores = apply_precision_drop_logic(
            query, documents, final_global_order, final_scores, 
            expanded_tokens, debug_info
        )
        dropped_count = original_count - len(final_global_order)
        debug_info["precision_drop"] = {
            "original_count": original_count,
            "dropped_count": dropped_count,
            "final_count": len(final_global_order)
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
