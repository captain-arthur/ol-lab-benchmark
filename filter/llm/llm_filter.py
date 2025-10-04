#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
llm_filter.py
- LLM 기반 필터링 실험을 위한 핵심 모듈
- 4가지 실험 방식 비교: LLM Only, LLM+SDE, LLM+CBC, LLM+SDE+CBC
- MS MARCO 데이터셋 기반 유사도 점수 평가
"""

import os
import json
import time
import hashlib
import re
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
from tqdm import tqdm

# ===== Output: force line-buffered =====
import sys
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except:
    pass

# ===============================
# Configuration
# ===============================
@dataclass
class LLMFilterConfig:
    """LLM 필터링 실험 설정"""
    
        # 기본 설정
    max_queries: int = 20
    max_passages: int = 100
        
        # LLM 설정
    api_mode: str = "ollama"  # "ollama" 또는 "gemini"
    ollama_url: str = "http://192.168.45.166:11434"
    model_name: str = "gemma3:latest"
    temperature: float = 0.0
    top_p: float = 1.0
        
        # Gemini API 설정
    gemini_api_key: Optional[str] = None
    gemini_model: str = "gemini-2.5-flash"
        
        # 평가 설정
    sim_threshold: float = 0.5  # 유사도 임계값
    
    # SDE 설정
    sde_k: int = 2  # 확장 쿼리 개수 (고정)
    
    # CBC 설정
    cbc_enabled: bool = False
    cbc_cache_path: str = ".cache/cbc_stats.json"
    
    # 캐시 설정
    cache_dir: str = ".cache"
    results_dir: str = "results/llm/ms_marco"

# ===============================
# LLM Client Classes
# ===============================
class Ollama:
    """Ollama 클라이언트"""
    def __init__(self, url: str, temperature: float = 0.0, top_p: float = 1.0):
        self.url = url.rstrip('/')
        self.temperature = temperature
        self.top_p = top_p

    def generate(self, model: str, prompt: str, **kwargs) -> str:
        """텍스트 생성"""
        try:
            import requests
            
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                }
            }
            payload.update(kwargs)
            
            response = requests.post(f"{self.url}/api/generate", json=payload, timeout=120)
            response.raise_for_status()
            
            result = response.json()
            return result.get("response", "").strip()
            
        except Exception as e:
            print(f"❌ Ollama API 오류: {e}")
            return ""

class GeminiAPI:
    """Google Gemini API 클라이언트"""
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash", temperature: float = 0.0):
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
    
    def generate(self, prompt: str, **kwargs) -> str:
        """텍스트 생성"""
        try:
            import google.generativeai as genai
            
            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel(self.model_name)
            
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.temperature,
                )
            )
            
            return response.text.strip()
            
        except Exception as e:
            print(f"❌ Gemini API 오류: {e}")
            return ""

# ===============================
# Cache Management
# ===============================
class ResponseCache:
    """LLM 응답 캐시 관리"""
    
    def __init__(self, cache_path: str):
        self.cache_path = cache_path
        self.cache = self._load_cache()
    
    def _load_cache(self) -> Dict[str, str]:
        """캐시 로드"""
        if os.path.exists(self.cache_path):
            try:
                cache = {}
                with open(self.cache_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            item = json.loads(line.strip())
                            cache[item['key']] = item['response']
                return cache
            except:
                return {}
        return {}
    
    def _save_cache(self):
        """캐시 저장"""
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with open(self.cache_path, 'w', encoding='utf-8') as f:
            for key, response in self.cache.items():
                item = {'key': key, 'response': response}
                json.dump(item, f, ensure_ascii=False)
                f.write('\n')
    
    def get(self, key: str) -> Optional[str]:
        """캐시에서 조회"""
        return self.cache.get(key)
    
    def set(self, key: str, response: str):
        """캐시에 저장"""
        self.cache[key] = response
        self._save_cache()

class CBCStats:
    """CBC 통계 관리 - 분포 기반 동적 percentile 적용"""
    
    def __init__(self, cache_path: str):
        self.cache_path = cache_path
        self.stats = self._load_stats()
    
    def _load_stats(self) -> Dict[str, List[float]]:
        """통계 로드"""
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {'overlap': [], 'margin': []}
        return {'overlap': [], 'margin': []}
    
    def _save_stats(self):
        """통계 저장"""
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with open(self.cache_path, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, ensure_ascii=False, indent=2)
    
    def update(self, overlap: float, margin: float):
        """통계 업데이트"""
        self.stats['overlap'].append(overlap)
        self.stats['margin'].append(margin)
        self._save_stats()
    
    def analyze_distribution(self, metric: str) -> Dict[str, Any]:
        """분포 특성 분석"""
        values = self.stats.get(metric, [])
        if len(values) < 3:
            return {"type": "insufficient", "mean": 0.5, "std": 0.1}
        
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        # 분포 특성에 따른 분류
        if std_val < 0.1:
            dist_type = "low_variance"  # 낮은 분산
        elif std_val > 0.3:
            dist_type = "high_variance"  # 높은 분산
        else:
            dist_type = "balanced"  # 균형
        
        return {
            "type": dist_type,
            "mean": mean_val,
            "std": std_val,
            "count": len(values)
        }
    
    def get_adaptive_threshold(self, metric: str) -> float:
        """분포 특성에 따른 적응적 임계값 계산 (더 적극적 필터링)"""
        values = self.stats.get(metric, [])
        if not values:
            return 0.7  # 더 높은 기본값
        
        # 분포 특성 분석
        dist_info = self.analyze_distribution(metric)
        
        # 더 적극적인 percentile 사용
        if dist_info["type"] == "low_variance":
            # 낮은 분산: 매우 엄격한 기준
            percentile = 90
        elif dist_info["type"] == "high_variance":
            # 높은 분산: 여전히 엄격한 기준
            percentile = 80
        else:
            # 균형: 높은 기준
            percentile = 85
        
        # 동적 percentile 적용
        sorted_values = sorted(values)
        idx = int(len(sorted_values) * percentile / 100)
        idx = min(idx, len(sorted_values) - 1)
        threshold = sorted_values[idx]
        
        # 더 높은 최소 임계값으로 조정
        threshold = max(0.5, min(0.9, threshold))
        
        return threshold

# ===============================
# Data Loading
# ===============================
def load_ms_marco_data(max_queries: int = 20) -> List[Dict[str, Any]]:
    """MS MARCO 데이터셋 로드 (s_filter.py 방식과 동일하게 수정)"""
    try:
        from datasets import load_dataset
        
        print(f"🔄 MS MARCO 데이터셋 로드 중... (최대 {max_queries}개 쿼리)")
        
        # validation split 사용 (s_filter.py와 동일)
        dataset = load_dataset('ms_marco', 'v1.1', split='validation')
        
        processed_data = []
        for i in range(min(max_queries, len(dataset))):
            sample = dataset[i]
            
            passages_dict = sample['passages']
            passage_texts = passages_dict['passage_text']
            is_selected = passages_dict['is_selected']
            
            # 관련성 레이블을 binary로 변환 (0: 무관, 1: 관련)
            relevance_labels = [1 if label == 1 else 0 for label in is_selected]
            
            processed_data.append({
                'query_id': str(sample['query_id']),
                'query': sample['query'],
                'passages': passage_texts,
                'relevance_labels': relevance_labels,
                'answers': sample['answers'],
                'query_type': sample.get('query_type', 'unknown')
            })
        
        print(f"✅ MS MARCO 데이터 로드 완료: {len(processed_data)}개 쿼리")
        return processed_data
        
    except Exception as e:
        print(f"❌ MS MARCO 데이터 로드 실패: {e}")
        return []

# ===============================
# Prompt Generation
# ===============================
def create_similarity_prompt(query: str, document: str) -> str:
    """유사도 점수 평가 프롬프트 (점수만 반환)"""
    return f"""Rate relevance between query and document. Respond with only a number between 0.0 and 1.0.

Query: {query}
Document: {document}

Score:"""

def create_sde_prompt(query: str) -> str:
    """SDE (Semantic Data Expansion) 프롬프트"""
    return f"""Generate 3 different ways to ask the same question. Do NOT repeat the original query.

Original Query: "{query}"

Rules:
1) Create 3 different paraphrases of the original query
2) Each paraphrase must be different from the original
3) Make each paraphrase 1 sentence, ≤20 tokens
4) Focus on different ways to express the same intent

Output (3 lines, one paraphrase per line):"""

# ===============================
# LLM Interaction
# ===============================
def llm_cached_call(client, cache: ResponseCache, model_name: str, prompt: str, **kwargs) -> Tuple[str, Dict[str, Any]]:
    """캐시된 LLM 호출"""
    # 캐시 키 생성
    cache_key = hashlib.md5(f"{model_name}:{prompt}".encode()).hexdigest()
    
    # 캐시에서 조회
    cached_response = cache.get(cache_key)
    if cached_response is not None:
        return cached_response, {"cached": True}
    
    # LLM 호출
    if hasattr(client, 'generate'):
        if isinstance(client, Ollama):
            response = client.generate(model_name, prompt, **kwargs)
        else:  # GeminiAPI
            response = client.generate(prompt, **kwargs)
    else:
        response = ""
    
    # 캐시에 저장
    cache.set(cache_key, response)
    
    return response, {"cached": False}

# ===============================
# Response Parsing
# ===============================
def parse_similarity_score(response: str) -> float:
    """유사도 점수 파싱 (개선된 에러 핸들링)"""
    try:
        # 응답에서 숫자 추출 (0.0-1.0 범위)
        score_match = re.search(r'(\d+\.?\d*)', response.strip())
        if score_match:
            score = float(score_match.group(1))
            # 범위 정규화
            if score > 1.0:
                score = score / 10.0 if score <= 10.0 else score / 100.0
            # 소수점 2자리까지 세분화
            return round(max(0.0, min(1.0, score)), 2)
    except Exception as e:
        print(f"⚠️ LLM 응답 파싱 오류: {e}, 응답: {response[:100]}...")
        # 로그 기록을 위한 추가 정보
        import logging
        logging.warning(f"LLM parsing error: {e}, response: {response}")
    
    # 기본값 반환 (중간값)
    return 0.50

def parse_advanced_sde_response(response: str, original_query: str) -> List[str]:
    """고도화된 SDE 응답 파싱 (CBC 시너지 최적화)"""
    paraphrases = []
    original_lower = original_query.lower().strip()
    
    # 라인 기반 파싱 (더 안정적)
    lines = response.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        
        # 따옴표로 시작하는 문장 찾기
        if line.startswith('"') and line.endswith('"'):
            sentence = line[1:-1].strip()  # 따옴표 제거
            
            # 품질 검증
            if (sentence.lower().strip() != original_lower and 
                len(sentence) > 20 and  # 최소 길이
                len(sentence) < 200 and
                not sentence.startswith('**') and  # 헤더 제외
                not sentence.startswith('*') and   # 마크다운 제외
                'Rationale:' not in sentence and   # 설명 제외
                'transaction volume' not in sentence.lower() and  # 설명 제외
                'POS data' not in sentence.lower() and  # 설명 제외
                'inventory turnover' not in sentence.lower()):    # 설명 제외
                
                paraphrases.append(sentence)
    
    return paraphrases[:5]  # 최대 5개 (다양한 관점)

def parse_sde_response(response: str, original_query: str) -> List[str]:
    """기존 SDE 응답 파싱 (하위 호환성)"""
    paraphrases = []
    original_lower = original_query.lower().strip()
    
    for line in response.strip().split('\n'):
        line = line.strip()
        if line and not line.startswith('#'):
            # 본 쿼리와 중복되지 않는지 확인 (대소문자 무시)
            if line.lower().strip() != original_lower:
                paraphrases.append(line)
    
    return paraphrases[:3]  # 최대 3개

# ===============================
# SDE Cache Management
# ===============================
def get_sde_cache_file_path(cache_dir: str) -> str:
    """SDE 캐시 파일 경로 반환"""
    return os.path.join(cache_dir, "sde_cache.json")

def load_sde_cache(query: str, cache_dir: str) -> Optional[List[str]]:
    """SDE 캐시에서 paraphrases 로드"""
    cache_path = get_sde_cache_file_path(cache_dir)
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache_list = json.load(f)
                for item in cache_list:
                    if item.get('query') == query:
                        expanded = item.get('expanded', {})
                        return expanded.get('paraphrases', [])
        except Exception:
            pass
    return None

def save_sde_cache(query: str, paraphrases: List[str], cache_dir: str):
    """SDE 캐시에 paraphrases 저장"""
    cache_path = get_sde_cache_file_path(cache_dir)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    
    cache_list = []
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache_list = json.load(f)
        except Exception:
            cache_list = []
    
    # 기존 항목 제거 (중복 방지)
    cache_list = [item for item in cache_list if item.get('query') != query]
    
    # 새 항목 추가
    cache_list.append({
        'query': query,
        'expanded': {
            'paraphrases': paraphrases
        }
    })
    
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(cache_list, f, ensure_ascii=False, indent=2)

# ===============================
# SDE (Semantic Data Expansion)
# ===============================
def log_sde_shift(original_scores, sde_scores, thr):
    """SDE 효과를 수치로 측정하는 로깅 함수"""
    import numpy as np
    o = np.array(original_scores); s = np.array(sde_scores)
    delta = s - o
    # 1) 평균/표준편차
    print(f"[SDE] Δmean={delta.mean():.4f}, Δstd={delta.std():.4f}")
    # 2) 임계값 크로싱 수
    cross_up = np.sum((o < thr) & (s >= thr))
    cross_dn = np.sum((o >= thr) & (s < thr))
    print(f"[SDE] crossings: up={cross_up}, down={cross_dn} @thr={thr:.2f}")
    # 3) 코사인 유사도(순위보다는 점수 변화량 관찰)
    cos = (o@s)/(np.linalg.norm(o)+1e-8)/(np.linalg.norm(s)+1e-8)
    print(f"[SDE] cos(original, sde)={cos:.4f}")
def diverse_enough(p, original):
    """의미 다양성 검사"""
    # 1) 레벤슈타인 유사도(낮을수록 OK)
    import difflib
    ratio = difflib.SequenceMatcher(a=p.lower(), b=original.lower()).ratio()
    if ratio > 0.85:  # 너무 비슷하면 제외
        return False
    # 2) 토큰 Jaccard(겹치지 않을수록 OK)
    s1, s2 = set(p.lower().split()), set(original.lower().split())
    jacc = len(s1&s2)/max(1, len(s1|s2))
    return jacc < 0.75

def filter_paraphrases(paraphrases: List[str], original_query: str, max_count: int) -> List[str]:
    """paraphrases 품질 필터링 (의미 다양성 추가)"""
    seen = set()
    out = []
    original_words = len(original_query.split())
    
    for p in paraphrases:
        p_l = p.lower().strip()
        p_words = len(p.split())
        
        # 원본과 동일한지 확인
        if p_l == original_query.lower().strip(): 
            continue
            
        # 길이 필터 (±100% 범위로 완화)
        if not (max(3, original_words * 0.5) <= p_words <= original_words * 2.0 + 2):
            continue
            
        # 과도한 구체화 제거
        if any(bad in p_l for bad in ["last fiscal year", "per day", "united states", "during the", "in the"]):
            continue
            
        # 의미 다양성 검사 추가
        if not diverse_enough(p, original_query):
            continue
            
        # 중복 제거
        if p_l in seen: 
            continue
            
        seen.add(p_l)
        out.append(p)
        
        # 최대 개수 제한
        if len(out) >= max_count:
            break
            
    return out

def generate_anchors_ollama(client, model_name: str, query: str, num_anchors: int, config: LLMFilterConfig) -> List[str]:
    """s_filter.py와 동일한 방식으로 앵커 생성"""
    if num_anchors <= 0:
        return []
    
    cache_key = f"anchors_{hashlib.md5(query.encode()).hexdigest()}"
    
    # 캐시에서 조회
    cached_paraphrases = load_sde_cache(query, config.cache_dir)
    if cached_paraphrases:
        print(f"🎯 [Cache] Found cached paraphrases for query: {query[:50]}...")
        return cached_paraphrases[:num_anchors]
    
    print(f"🔄 [Cache] Generating {num_anchors} anchors for query: {query[:50]}...")
    
    # 차별화된 paraphrase 생성 프롬프트 (SDE 효과 증대)
    prompt = (
        f"Generate {num_anchors} paraphrases of: '{query}'\n"
        f"Goal: Create semantically distinct variations that capture different aspects of the same information need.\n"
        f"\n"
        f"Rules:\n"
        f"1. Keep similar length (original ±30%)\n"
        f"2. Use COMPLETELY different word choices and structures\n"
        f"3. Preserve core intent but approach from different angles\n"
        f"4. Include domain-specific synonyms and technical terms\n"
        f"5. Make each paraphrase as different as possible from the original\n"
        f"\n"
        f"Examples for '{query}':\n"
        f"- Focus on different aspects: sales → revenue → income → earnings\n"
        f"- Use different question structures: direct → indirect → descriptive\n"
        f"- Vary specificity: general → specific → technical\n"
        f"- Change perspective: store-level → company-level → industry-level\n"
        f"\n"
        f"Write only the paraphrases, one per line. No explanations.\n"
    )
    
    # LLM 호출
    if isinstance(client, Ollama):
        response = client.generate(model_name, prompt)
    else:  # GeminiAPI
        response = client.generate(prompt)
    
    # 응답 파싱 및 품질 필터링
    anchors_raw = [line.strip() for line in response.splitlines() if line.strip()]
    anchors = filter_paraphrases(anchors_raw, query, config.sde_k)
    
    if anchors:
        # 캐시에 저장
        save_sde_cache(query, anchors, config.cache_dir)
        print(f"💾 [Cache] Saved {len(anchors)} paraphrases for query: {query[:50]}...")

    return anchors

# ===============================
# SDE Cache (SDE paraphrases만 캐시)
# ===============================
def get_sde_cache(cache_dir: str) -> str:
    """SDE 캐시 파일 경로 반환"""
    return os.path.join(cache_dir, "sde_cache.json")

# ===============================
# Similarity Scoring
# ===============================
def score_similarity(client, model_name: str, query: str, document: str) -> float:
    """단일 쿼리-문서 유사도 점수 계산 (캐시 없음)"""
    prompt = create_similarity_prompt(query, document)
    
    # LLM 직접 호출 (캐시 없음)
    if isinstance(client, Ollama):
        response = client.generate(model_name, prompt)
    else:  # GeminiAPI
        response = client.generate(prompt)
    
    return parse_similarity_score(response)

def score_documents(client, model_name: str, query: str, documents: List[str]) -> List[float]:
    """쿼리에 대한 모든 문서 점수 계산 (캐시 없음)"""
    scores = []
    for document in documents:
        score = score_similarity(client, model_name, query, document)
        scores.append(score)
    return scores

# ===============================
# CBC (Confidence-Based Calibration)
# ===============================
def cbc_threshold_per_query(scores: List[float]) -> float:
    """쿼리 단위 CBC 임계값 계산 (고정 임계값 0.95 수준의 높은 성능)"""
    if not scores:
        return 0.95
    
    std = np.std(scores)
    # 매우 적극적인 percentile 사용 (고정 임계값 0.95 수준)
    pct = 0.8  # 80th percentile (매우 높은 임계값)
    
    if std < 0.1:
        pct = 0.75  # 75th percentile
    elif std > 0.3:
        pct = 0.85  # 85th percentile
    
    thr = np.quantile(scores, pct)
    # 고정 임계값 0.95 수준으로 조정
    return float(np.clip(thr, 0.9, 0.98))

def apply_cbc_filtering(scores: List[float], threshold: float = 0.5) -> List[Tuple[float, bool]]:
    """CBC 기반 필터링 적용 (고정 임계값 0.95 수준의 높은 성능)"""
    cbc_threshold = cbc_threshold_per_query(scores)
    # 고정 임계값 0.95 수준의 높은 성능을 위해 더 높은 임계값 사용
    aggressive_threshold = max(cbc_threshold, 0.95)  # 최소 0.95 이상
    return [(float(score), score >= aggressive_threshold) for score in scores]

# ===============================
# Metrics Calculation (s_filter.py 방식으로 수정)
# ===============================
def compute_drop_from_predictions(predictions: List[bool], ground_truth: List[int]) -> Dict[str, float]:
    """predictions 기반으로 Drop 지표를 직접 계산"""
    dropped = [i for i, keep in enumerate(predictions) if not keep]
    dropped_pos = sum(1 for i in dropped if ground_truth[i] == 1)
    dropped_neg = sum(1 for i in dropped if ground_truth[i] == 0)
    total_neg = sum(1 for g in ground_truth if g == 0)

    drop_precision = (dropped_neg / len(dropped)) if dropped else 0.0
    drop_recall = (dropped_neg / total_neg) if total_neg else 0.0
    drop_f1 = (2*drop_precision*drop_recall/(drop_precision+drop_recall)
               if (drop_precision+drop_recall)>0 else 0.0)
    
    return {
        "Dropped_Count": len(dropped), 
        "Dropped_Positive_Count": dropped_pos,
        "Dropped_Negative_Count": dropped_neg,
        "Drop_Precision": drop_precision, 
        "Drop_Recall": drop_recall, 
        "Drop_F1": drop_f1
    }

def calculate_metrics(predictions: List[bool], ground_truth: List[int]) -> Dict[str, float]:
    """평가 메트릭 계산"""
    tp = sum(1 for p, g in zip(predictions, ground_truth) if p and g == 1)
    fp = sum(1 for p, g in zip(predictions, ground_truth) if p and g == 0)
    fn = sum(1 for p, g in zip(predictions, ground_truth) if not p and g == 1)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn
    }

def compute_filtering_metrics_llm(llm_scores: List[float], candidate_pool: List[int], 
                                 positive_docs: List[int], negative_docs: List[int],
                                 threshold: float = 0.3) -> Dict[str, float]:
    """LLM 점수 기반 필터링 메트릭 계산 (s_filter.py 방식)"""
    drop_decisions = []
    keep_decisions = []
    
    for i, doc_id in enumerate(candidate_pool):
        llm_score = llm_scores[i]
        
        # LLM 점수가 임계값 미만이면 드롭
        if llm_score < threshold:
            drop_decisions.append(doc_id)
        else:
            keep_decisions.append(doc_id)
    
    # Drop Precision/Recall 계산 (s_filter.py 방식)
    dropped_positives = [d for d in drop_decisions if d in positive_docs]
    dropped_negatives = [d for d in drop_decisions if d in negative_docs]
    
    total_positives = len(positive_docs)
    total_negatives = len(negative_docs)
    
    drop_precision = len(dropped_negatives) / len(drop_decisions) if drop_decisions else 0.0
    drop_recall = len(dropped_negatives) / total_negatives if total_negatives > 0 else 0.0
    drop_f1 = 2 * drop_precision * drop_recall / (drop_precision + drop_recall) if (drop_precision + drop_recall) > 0 else 0.0
    
    return {
        'Dropped_Count': len(drop_decisions),
        'Dropped_Positive_Count': len(dropped_positives),
        'Dropped_Negative_Count': len(dropped_negatives),
        'Drop_Precision': drop_precision,
        'Drop_Recall': drop_recall,
        'Drop_F1': drop_f1
    }

# ===============================
# Optimized Experiments (통합 점수 측정)
# ===============================
def generate_all_similarity_scores(data: List[Dict[str, Any]], config: LLMFilterConfig) -> Dict[str, Dict[str, List[float]]]:
    """모든 쿼리-문서 쌍의 유사도 점수를 한 번에 생성"""
    print("\n🔄 모든 쿼리-문서 유사도 점수 생성 중...")
    
    # 클라이언트 초기화
    if config.api_mode == "gemini":
        client = GeminiAPI(config.gemini_api_key, config.gemini_model, config.temperature)
        model_name = config.gemini_model
    else:
        client = Ollama(config.ollama_url, config.temperature, config.top_p)
        model_name = config.model_name
    
    all_scores = {}
    
    for item in tqdm(data, desc="유사도 점수 생성"):
        query = item['query']
        passages = item['passages'][:config.max_passages]
        
        # 앵커 생성 (s_filter.py와 동일)
        anchors = generate_anchors_ollama(client, model_name, query, config.sde_k, config)
        
        query_scores = {}
        # 원본 쿼리 점수도 포함
        original_scores = score_documents(client, model_name, query, passages)
        query_scores[query] = original_scores
        
        # 앵커 점수
        for anchor in anchors:
            scores = score_documents(client, model_name, anchor, passages)
            query_scores[anchor] = scores
        
        all_scores[query] = query_scores
    
    return all_scores

def run_experiments_with_scores(data: List[Dict[str, Any]], all_scores: Dict[str, Dict[str, List[float]]], config: LLMFilterConfig) -> List[Dict[str, Any]]:
    """생성된 점수를 기반으로 모든 실험 실행 (s_filter.py 방식으로 수정)"""
    experiments = []
    
    # CBC 통계 초기화
    cbc_stats = CBCStats(config.cbc_cache_path)
    
    for item in tqdm(data, desc="실험 실행"):
        query = item['query']
        passages = item['passages']
        relevance_labels = item['relevance_labels']
        
        # s_filter.py 방식: 하드 네거티브 후보 풀 생성 시뮬레이션
        # 실제로는 SBERT가 필요하지만, LLM 방식에서는 단순화
        candidate_pool = list(range(len(passages)))
        positive_docs = [i for i, label in enumerate(relevance_labels) if label == 1]
        negative_docs = [i for i, label in enumerate(relevance_labels) if label == 0]
        
        query_scores = all_scores[query]
        
        # 실험별 결과 저장
        experiment_results = {
            'query': query,
            'experiments': {}
        }
        
        # 1. LLM Only (Baseline)
        # 쿼리별 점수 구조: all_scores[query][paraphrase] = scores
        if query not in all_scores:
            print(f"⚠️ 쿼리 점수를 찾을 수 없음: {query}")
            continue
            
        query_scores = all_scores[query]
        
        # 원본 쿼리 점수 찾기 (첫 번째 paraphrase가 원본)
        baseline_scores = None
        for paraphrase, scores in query_scores.items():
            if paraphrase == query:  # 원본 쿼리와 정확히 일치
                baseline_scores = scores
                break
        
        if baseline_scores is None:
            # 원본 쿼리가 없으면 첫 번째 paraphrase 사용
            if query_scores:
                baseline_scores = list(query_scores.values())[0]
            else:
                print(f"❌ 쿼리 점수가 비어있음: {query}")
                continue
        # Baseline - predictions 기반으로 Drop 지표 계산
        baseline_predictions = [score >= config.sim_threshold for score in baseline_scores]
        baseline_filter_metrics = compute_drop_from_predictions(baseline_predictions, relevance_labels)
        experiment_results['experiments']['baseline'] = {
            'scores': baseline_scores,
            'filter_metrics': baseline_filter_metrics
        }
        
        # 2. LLM + SDE (차등 융합 방식)
        # 원본 쿼리 점수
        original_scores = baseline_scores
        
        # 앵커 점수들 (원본 제외)
        anchor_scores = [scores for k, scores in query_scores.items() if k != query]
        
        if len(anchor_scores) > 0:
            # 개선된 Score Fusion 방식 (원본 + paraphrase 신호 합산)
            # 1단계: 앵커들 중 최고 점수
            anchor_max_scores = [np.max([scores[i] for scores in anchor_scores]) for i in range(len(original_scores))]
            
            # 2단계: 차등 융합 + 경계 민감도 방식 (Threshold-aware Boost)
            alpha = 0.7  # 원본 비중
            beta = 0.3   # 앵커 비중
            margin = 0.05  # 앵커 승리시 추가 보너스
            thr = config.sim_threshold
            band = 0.05  # 경계 주변 밴드
            
            sde_final_scores = []
            for i in range(len(original_scores)):
                fused = alpha * original_scores[i] + beta * anchor_max_scores[i]
                
                # A. 경계 민감 가중치(Threshold-aware Boost)
                if abs(original_scores[i] - thr) <= band and (anchor_max_scores[i] - original_scores[i]) > 0.01:
                    fused = min(1.0, fused + 0.07)  # 경계 근처만 강하게 밀어줌
                
                # B. 앵커 우승 "확실성" 보너스(Top-2 margin)
                anchor_sorted = sorted([scores[i] for scores in anchor_scores], reverse=True)
                margin_top2 = anchor_sorted[0] - (anchor_sorted[1] if len(anchor_sorted) > 1 else 0.0)
                if anchor_max_scores[i] > original_scores[i] + 0.02 and margin_top2 > 0.03:
                    fused = min(1.0, fused + 0.05)
                
                sde_final_scores.append(fused)
        else:
            sde_final_scores = original_scores
        
        # SDE 효과 로깅
        log_sde_shift(original_scores, sde_final_scores, config.sim_threshold)
        
        # SDE - predictions 기반으로 Drop 지표 계산
        sde_predictions = [score >= config.sim_threshold for score in sde_final_scores]
        sde_filter_metrics = compute_drop_from_predictions(sde_predictions, relevance_labels)
        experiment_results['experiments']['sde'] = {
            'scores': sde_final_scores,
            'filter_metrics': sde_filter_metrics
        }
        
        # 3. LLM + CBC (쿼리별 분포 기반 임계값)
        cbc_threshold = max(cbc_threshold_per_query(baseline_scores), 0.6)
        # CBC - predictions 기반으로 Drop 지표 계산
        cbc_predictions = [score >= cbc_threshold for score in baseline_scores]
        cbc_filter_metrics = compute_drop_from_predictions(cbc_predictions, relevance_labels)
        experiment_results['experiments']['cbc'] = {
            'scores': baseline_scores,
            'filter_metrics': cbc_filter_metrics
        }
        
        # 4. LLM + SDE + CBC (SDE 점수 분포 기반 CBC 임계값)
        sde_cbc_threshold = max(cbc_threshold_per_query(sde_final_scores), 0.6)
        # SDE+CBC 효과 로깅
        log_sde_shift(original_scores, sde_final_scores, sde_cbc_threshold)
        # SDE+CBC - predictions 기반으로 Drop 지표 계산
        sde_cbc_predictions = [score >= sde_cbc_threshold for score in sde_final_scores]
        sde_cbc_filter_metrics = compute_drop_from_predictions(sde_cbc_predictions, relevance_labels)
        experiment_results['experiments']['sde_cbc'] = {
            'scores': sde_final_scores,
            'filter_metrics': sde_cbc_filter_metrics
        }
        
        # CBC 통계 업데이트
        if len(baseline_scores) > 1:
            sorted_indices = sorted(range(len(baseline_scores)), key=lambda i: baseline_scores[i], reverse=True)
            top_half = set(sorted_indices[:len(sorted_indices)//2])
            ground_truth_top = set(i for i, gt in enumerate(relevance_labels) if gt == 1)
            overlap = len(top_half & ground_truth_top) / max(len(ground_truth_top), 1)
            margin = baseline_scores[sorted_indices[0]] - baseline_scores[sorted_indices[1]] if len(baseline_scores) > 1 else 0.0
            cbc_stats.update(overlap, margin)
        
        experiments.append(experiment_results)
    
    return experiments

def run_optimized_experiments(data: List[Dict[str, Any]], config: LLMFilterConfig) -> List[Dict[str, Any]]:
    """최적화된 실험 실행 (s_filter.py 방식으로 수정)"""
    print("\n🚀 최적화된 LLM 필터링 실험")
    print("="*60)
    
    # 1단계: 모든 유사도 점수 생성
    all_scores = generate_all_similarity_scores(data, config)
    
    # 2단계: 실험별 필터링 및 평가
    experiment_results = run_experiments_with_scores(data, all_scores, config)
    
    # 3단계: 결과 집계 (s_filter.py 방식)
    final_results = []
    
    for exp_type in ['baseline', 'sde', 'cbc', 'sde_cbc']:
        # Drop Precision/Recall/F1 계산 (s_filter.py 방식)
        exp_drop_precision = [r['experiments'][exp_type]['filter_metrics']['Drop_Precision'] for r in experiment_results]
        exp_drop_recall = [r['experiments'][exp_type]['filter_metrics']['Drop_Recall'] for r in experiment_results]
        exp_drop_f1 = [r['experiments'][exp_type]['filter_metrics']['Drop_F1'] for r in experiment_results]
        
        overall_metrics = {
            'drop_precision': np.mean(exp_drop_precision),
            'drop_recall': np.mean(exp_drop_recall),
            'drop_f1': np.mean(exp_drop_f1)
        }
        
        final_results.append({
            'experiment_type': f'LLM_{exp_type.upper()}',
            'overall_metrics': overall_metrics,
            'results': experiment_results
        })
        
        print(f"✅ {exp_type.upper()}: Drop_P={overall_metrics['drop_precision']:.1%}, Drop_R={overall_metrics['drop_recall']:.1%}, Drop_F1={overall_metrics['drop_f1']:.1%}")
    
    return final_results

def run_llm_baseline_experiment(data: List[Dict[str, Any]], config: LLMFilterConfig) -> Dict[str, Any]:
    """1. LLM Only (Baseline) 실험"""
    print("\n" + "="*60)
    print("🚀 LLM Only (Baseline) 실험")
    print("="*60)
    
    # 클라이언트 초기화
    if config.api_mode == "gemini":
        client = GeminiAPI(config.gemini_api_key, config.gemini_model, config.temperature)
        model_name = config.gemini_model
    else:
        client = Ollama(config.ollama_url, config.temperature, config.top_p)
        model_name = config.model_name
    
    cache = ResponseCache(os.path.join(config.cache_dir, "llm_responses_cache.jsonl"))
    
    results = []
    total_latency = 0
    
    for item in tqdm(data, desc="LLM Baseline"):
        query = item['query']
        passages = item['passages'][:config.max_passages]
        ground_truth = [int(gt) for gt in item['relevance_labels'][:config.max_passages]]
        
        start_time = time.time()
        
        # 단일 쿼리로 모든 문서 점수 계산
        scores = score_documents(client, model_name, query, passages)
        
        # 임계값 기반 예측
        predictions = [bool(score >= config.sim_threshold) for score in scores]
        
        latency = time.time() - start_time
        total_latency += latency
        
        # 메트릭 계산
        metrics = calculate_metrics(predictions, ground_truth)
        
        # Drop 메트릭 계산
        candidate_pool = list(range(len(passages)))
        positive_docs = [i for i, v in enumerate(ground_truth) if v == 1]
        negative_docs = [i for i, v in enumerate(ground_truth) if v == 0]
        
        drop_metrics = compute_filtering_metrics_llm(
            scores, candidate_pool, positive_docs, negative_docs, config.sim_threshold
        )
        
        results.append({
            'query': query,
            'scores': scores,
            'predictions': predictions,
            'ground_truth': ground_truth,
            'metrics': metrics,
            'drop_metrics': drop_metrics,
            'latency': latency
        })
    
    # 전체 메트릭 계산
    all_precision = [r['metrics']['precision'] for r in results]
    all_recall = [r['metrics']['recall'] for r in results]
    all_f1 = [r['metrics']['f1'] for r in results]
    
    overall_metrics = {
        'precision': np.mean(all_precision),
        'recall': np.mean(all_recall),
        'f1': np.mean(all_f1),
        'total_latency': total_latency,
        'avg_latency': total_latency / len(results)
    }
    
    print(f"✅ LLM Baseline 완료")
    print(f"📊 Precision: {overall_metrics['precision']:.3f}")
    print(f"📊 Recall: {overall_metrics['recall']:.3f}")
    print(f"📊 F1: {overall_metrics['f1']:.3f}")
    print(f"📊 총 시간: {total_latency:.1f}초")
    
    return {
        'experiment_type': 'LLM_BASELINE',
        'overall_metrics': overall_metrics,
        'results': results
    }

def run_llm_sde_experiment(data: List[Dict[str, Any]], config: LLMFilterConfig) -> Dict[str, Any]:
    """2. LLM + SDE 실험 (수정된 버전)"""
    print("\n" + "="*60)
    print("🚀 LLM + SDE 실험")
    print("="*60)

    # 클라이언트 초기화
    if config.api_mode == "gemini":
        client = GeminiAPI(config.gemini_api_key, config.gemini_model, config.temperature)
        model_name = config.gemini_model
    else:
        client = Ollama(config.ollama_url, config.temperature, config.top_p)
        model_name = config.model_name

    sde_cache = get_sde_cache(config.cache_dir)
    cache = ResponseCache(os.path.join(config.cache_dir, "llm_responses_cache.jsonl"))

    results = []
    total_latency = 0.0
    
    for item in tqdm(data, desc="LLM + SDE"):
        query = item['query']
        passages = item['passages'][:config.max_passages]
        ground_truth = [int(x) for x in item['relevance_labels'][:config.max_passages]]

        start_time = time.time()
        
        # 원본 쿼리 점수 계산
        original_scores = score_documents(client, model_name, query, passages)
        
        # 앵커 생성 (s_filter.py와 동일)
        anchors = generate_anchors_ollama(client, model_name, query, config.sde_k, config)

        # 각 앵커로 점수 계산
        anchor_scores = [score_documents(client, model_name, anchor, passages) for anchor in anchors]
        
        # 개선된 Score Fusion 방식
        if anchor_scores:
            # 앵커들 중 최고 점수
            anchor_max_scores = [np.max([s[i] for s in anchor_scores]) for i in range(len(passages))]
            
            # 차등 융합 + 경계 민감도 방식
            alpha = 0.7  # 원본 비중
            beta = 0.3   # 앵커 비중
            margin = 0.05  # 앵커 승리시 추가 보너스
            
            agg_scores = []
            for i in range(len(passages)):
                fused = alpha * original_scores[i] + beta * anchor_max_scores[i]
                if anchor_max_scores[i] > original_scores[i] + 0.02:  # 앵커가 의미 있게 우위일 때
                    fused = min(1.0, fused + margin)
                agg_scores.append(fused)
        else:
            agg_scores = original_scores

        # 임계값 기반 예측
        predictions = [bool(x >= config.sim_threshold) for x in agg_scores]
        
        latency = time.time() - start_time
        total_latency += latency

        # 메트릭 계산
        metrics = calculate_metrics(predictions, ground_truth)
        
        # Drop 메트릭 계산
        candidate_pool = list(range(len(passages)))
        positive_docs = [i for i, v in enumerate(ground_truth) if v == 1]
        negative_docs = [i for i, v in enumerate(ground_truth) if v == 0]
        
        drop_metrics = compute_filtering_metrics_llm(
            agg_scores, candidate_pool, positive_docs, negative_docs, config.sim_threshold
        )
        
        results.append({
            'query': query,
            'anchors': anchors,
            'agg_scores': agg_scores,
            'predictions': predictions,
            'ground_truth': ground_truth,
            'metrics': metrics,
            'drop_metrics': drop_metrics,
            'latency': latency
        })

    # 전체 메트릭 계산
    all_precision = [r['metrics']['precision'] for r in results]
    all_recall = [r['metrics']['recall'] for r in results]
    all_f1 = [r['metrics']['f1'] for r in results]
    
    overall_metrics = {
        'precision': np.mean(all_precision),
        'recall': np.mean(all_recall),
        'f1': np.mean(all_f1),
        'total_latency': total_latency,
        'avg_latency': total_latency / len(results)
    }
    
    print(f"✅ LLM + SDE 완료")
    print(f"📊 Precision: {overall_metrics['precision']:.3f}")
    print(f"📊 Recall: {overall_metrics['recall']:.3f}")
    print(f"📊 F1: {overall_metrics['f1']:.3f}")
    print(f"📊 총 시간: {total_latency:.1f}초")
    
    return {
        'experiment_type': 'LLM_SDE',
        'overall_metrics': overall_metrics,
        'results': results
    }


def run_llm_cbc_experiment(data: List[Dict[str, Any]], config: LLMFilterConfig) -> Dict[str, Any]:
    """3. LLM + CBC 실험"""
    print("\n" + "="*60)
    print("🚀 LLM + CBC 실험")
    print("="*60)
    
    # 클라이언트 초기화
    if config.api_mode == "gemini":
        client = GeminiAPI(config.gemini_api_key, config.gemini_model, config.temperature)
        model_name = config.gemini_model
    else:
        client = Ollama(config.ollama_url, config.temperature, config.top_p)
        model_name = config.model_name
    
    # LLM 점수는 캐시하지 않음 (실시간 계산)
    cbc_stats = CBCStats(config.cbc_cache_path)
    
    results = []
    total_latency = 0
    
    for item in tqdm(data, desc="LLM + CBC"):
        query = item['query']
        passages = item['passages'][:config.max_passages]
        ground_truth = [int(gt) for gt in item['relevance_labels'][:config.max_passages]]
        
        start_time = time.time()
        
        # 단일 쿼리로 모든 문서 점수 계산
        scores = score_documents(client, model_name, query, passages)
        
        # CBC 필터링 적용 (수정된 시그니처)
        cbc_results = apply_cbc_filtering(scores, config.sim_threshold)
        predictions = [bool(keep) for _, keep in cbc_results]
        
        # CBC 통계 업데이트 (간단한 overlap 계산)
        if len(scores) > 1:
            sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            top_half = set(sorted_indices[:len(sorted_indices)//2])
            ground_truth_top = set(i for i, gt in enumerate(ground_truth) if gt == 1)
            overlap = len(top_half & ground_truth_top) / max(len(ground_truth_top), 1)
            margin = scores[sorted_indices[0]] - scores[sorted_indices[1]] if len(scores) > 1 else 0.0
            cbc_stats.update(overlap, margin)
        
        latency = time.time() - start_time
        total_latency += latency
        
        # 메트릭 계산
        metrics = calculate_metrics(predictions, ground_truth)
        
        # Drop 메트릭 계산 (실제 CBC predictions 사용)
        drop_metrics = compute_drop_from_predictions(predictions, ground_truth)
        
        results.append({
            'query': query,
            'scores': scores,
            'cbc_results': cbc_results,
            'predictions': predictions,
            'ground_truth': ground_truth,
            'metrics': metrics,
            'drop_metrics': drop_metrics,
            'latency': latency
        })
    
    # 전체 메트릭 계산
    all_precision = [r['metrics']['precision'] for r in results]
    all_recall = [r['metrics']['recall'] for r in results]
    all_f1 = [r['metrics']['f1'] for r in results]
        
    overall_metrics = {
        'precision': np.mean(all_precision),
        'recall': np.mean(all_recall),
        'f1': np.mean(all_f1),
        'total_latency': total_latency,
        'avg_latency': total_latency / len(results)
    }
    
    print(f"✅ LLM + CBC 완료")
    print(f"📊 Precision: {overall_metrics['precision']:.3f}")
    print(f"📊 Recall: {overall_metrics['recall']:.3f}")
    print(f"📊 F1: {overall_metrics['f1']:.3f}")
    print(f"📊 총 시간: {total_latency:.1f}초")
        
    return {
        'experiment_type': 'LLM_CBC',
            'overall_metrics': overall_metrics,
        'results': results
    }

def run_llm_sde_cbc_experiment(data: List[Dict[str, Any]], config: LLMFilterConfig) -> Dict[str, Any]:
    """4. LLM + SDE + CBC (Proposed) 실험"""
    print("\n" + "="*60)
    print("🚀 LLM + SDE + CBC (Proposed) 실험")
    print("="*60)
    
    # 클라이언트 초기화
    if config.api_mode == "gemini":
        client = GeminiAPI(config.gemini_api_key, config.gemini_model, config.temperature)
        model_name = config.gemini_model
    else:
        client = Ollama(config.ollama_url, config.temperature, config.top_p)
        model_name = config.model_name
    
    # LLM 점수는 캐시하지 않음 (실시간 계산)
    cbc_stats = CBCStats(config.cbc_cache_path)
    
    results = []
    total_latency = 0
    
    for item in tqdm(data, desc="LLM + SDE + CBC"):
        query = item['query']
        passages = item['passages'][:config.max_passages]
        ground_truth = [int(gt) for gt in item['relevance_labels'][:config.max_passages]]
        
        start_time = time.time()
        
        # 원본 쿼리 점수 계산
        original_scores = score_documents(client, model_name, query, passages)
        
        # 앵커 생성 (s_filter.py와 동일)
        anchors = generate_anchors_ollama(client, model_name, query, config.sde_k, config)
        
        # 각 앵커로 문서 점수 계산
        anchor_scores = []
        for anchor in anchors:
            scores = score_documents(client, model_name, anchor, passages)
            anchor_scores.append(scores)
        
        # 개선된 Score Fusion 방식
        if anchor_scores and len(anchor_scores) > 0:
            # 앵커들 중 최고 점수
            anchor_max_scores = [np.max([s[i] for s in anchor_scores]) for i in range(len(passages))]
            
            # 차등 융합 + 경계 민감도 방식
            alpha = 0.7  # 원본 비중
            beta = 0.3   # 앵커 비중
            margin = 0.05  # 앵커 승리시 추가 보너스
            
            avg_scores = []
            for i in range(len(passages)):
                fused = alpha * original_scores[i] + beta * anchor_max_scores[i]
                if anchor_max_scores[i] > original_scores[i] + 0.02:  # 앵커가 의미 있게 우위일 때
                    fused = min(1.0, fused + margin)
                avg_scores.append(fused)
        else:
            avg_scores = original_scores
        
        # CBC 필터링 적용 (수정된 시그니처)
        cbc_results = apply_cbc_filtering(avg_scores, config.sim_threshold)
        predictions = [bool(keep) for _, keep in cbc_results]
        
        # CBC 통계 업데이트
        if len(avg_scores) > 1:
            sorted_indices = sorted(range(len(avg_scores)), key=lambda i: avg_scores[i], reverse=True)
            top_half = set(sorted_indices[:len(sorted_indices)//2])
            ground_truth_top = set(i for i, gt in enumerate(ground_truth) if gt == 1)
            overlap = len(top_half & ground_truth_top) / max(len(ground_truth_top), 1)
            margin = avg_scores[sorted_indices[0]] - avg_scores[sorted_indices[1]] if len(avg_scores) > 1 else 0.0
            cbc_stats.update(overlap, margin)
        
        latency = time.time() - start_time
        total_latency += latency
        
        # 메트릭 계산
        metrics = calculate_metrics(predictions, ground_truth)
        
        # Drop 메트릭 계산
        candidate_pool = list(range(len(passages)))
        positive_docs = [i for i, v in enumerate(ground_truth) if v == 1]
        negative_docs = [i for i, v in enumerate(ground_truth) if v == 0]
        
        drop_metrics = compute_filtering_metrics_llm(
            avg_scores, candidate_pool, positive_docs, negative_docs, config.sim_threshold
        )
        
        results.append({
            'query': query,
            'anchors': anchors,
            'avg_scores': avg_scores,
            'cbc_results': cbc_results,
            'predictions': predictions,
            'ground_truth': ground_truth,
            'metrics': metrics,
            'drop_metrics': drop_metrics,
            'latency': latency
        })
    
    # 전체 메트릭 계산
    all_precision = [r['metrics']['precision'] for r in results]
    all_recall = [r['metrics']['recall'] for r in results]
    all_f1 = [r['metrics']['f1'] for r in results]
        
    overall_metrics = {
        'precision': np.mean(all_precision),
        'recall': np.mean(all_recall),
        'f1': np.mean(all_f1),
        'total_latency': total_latency,
        'avg_latency': total_latency / len(results)
    }
    
    print(f"✅ LLM + SDE + CBC 완료")
    print(f"📊 Precision: {overall_metrics['precision']:.3f}")
    print(f"📊 Recall: {overall_metrics['recall']:.3f}")
    print(f"📊 F1: {overall_metrics['f1']:.3f}")
    print(f"📊 총 시간: {total_latency:.1f}초")
        
    return {
        'experiment_type': 'LLM_SDE_CBC',
        'overall_metrics': overall_metrics,
        'results': results
    }

# ===============================
# Main Execution
# ===============================
def main():
    """메인 실행 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="LLM 기반 필터링 실험")
    parser.add_argument('--max_queries', type=int, default=5, help='최대 쿼리 수')
    parser.add_argument('--max_passages', type=int, default=100, help='최대 문서 수')
    parser.add_argument('--api_mode', type=str, default='ollama', choices=['ollama', 'gemini'], help='API 모드')
    parser.add_argument('--model_name', type=str, default='gemma3:latest', help='모델 이름')
    parser.add_argument('--sim_threshold', type=float, default=0.5, help='유사도 임계값')
    parser.add_argument('--sde_k', type=int, default=3, help='SDE 앵커 개수')
    parser.add_argument('--cbc_enabled', action='store_true', help='CBC 활성화')
    
    args = parser.parse_args()
    
    # 설정 생성
    config = LLMFilterConfig(
        max_queries=args.max_queries,
        max_passages=args.max_passages,
        api_mode=args.api_mode,
        model_name=args.model_name,
        sim_threshold=args.sim_threshold,
        sde_k=args.sde_k,
        cbc_enabled=args.cbc_enabled
    )
    
    # 데이터 로드
    data = load_ms_marco_data(config.max_queries)
    if not data:
        print("❌ 데이터 로드 실패")
        return
    
    print(f"\n🚀 LLM 필터링 실험 시작")
    print(f"📊 쿼리 수: {len(data)}")
    print(f"📊 문서 수: {config.max_passages}")
    print(f"📊 API 모드: {config.api_mode}")
    print(f"📊 임계값: {config.sim_threshold}")
    
    # 실험 실행
    experiments = []
    
    # 1. LLM Baseline
    baseline_result = run_llm_baseline_experiment(data, config)
    experiments.append(baseline_result)
    
    # 2. LLM + SDE
    sde_result = run_llm_sde_experiment(data, config)
    experiments.append(sde_result)
    
    # 3. LLM + CBC
    cbc_result = run_llm_cbc_experiment(data, config)
    experiments.append(cbc_result)
    
    # 4. LLM + SDE + CBC
    sde_cbc_result = run_llm_sde_cbc_experiment(data, config)
    experiments.append(sde_cbc_result)
    
    # 결과 저장
    os.makedirs(config.results_dir, exist_ok=True)
    results_path = os.path.join(config.results_dir, "llm_experiments_results.json")
    
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(experiments, f, ensure_ascii=False, indent=2)
    
    # 결과 요약
    print("\n" + "="*80)
    print("📊 실험 결과 요약")
    print("="*80)
    
    for exp in experiments:
        exp_type = exp['experiment_type']
        metrics = exp['overall_metrics']
        print(f"{exp_type:15} | P: {metrics['precision']:.3f} | R: {metrics['recall']:.3f} | F1: {metrics['f1']:.3f} | Time: {metrics['total_latency']:.1f}s")
    
    print(f"\n💾 결과 저장: {results_path}")
    print("✅ 모든 실험 완료!")

if __name__ == "__main__":
    main()
