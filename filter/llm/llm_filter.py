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
    sde_k: int = 3  # 확장 쿼리 개수
    
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
    """CBC 통계 관리"""
    
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
    
    def get_threshold(self, metric: str, percentile: int = 80) -> float:
        """퍼센타일 기반 임계값 계산"""
        values = self.stats.get(metric, [])
        if not values:
            return 0.5  # 기본값
        
        sorted_values = sorted(values)
        idx = int(len(sorted_values) * percentile / 100)
        idx = min(idx, len(sorted_values) - 1)
        return sorted_values[idx]

# ===============================
# Data Loading
# ===============================
def load_ms_marco_data(max_queries: int = 20) -> List[Dict[str, Any]]:
    """MS MARCO 데이터셋 로드"""
    try:
        from datasets import load_dataset
        
        print(f"🔄 MS MARCO 데이터셋 로드 중... (최대 {max_queries}개 쿼리)")
        
        dataset = load_dataset('ms_marco', 'v1.1')
        train_data = dataset['train']
        
        processed_data = []
        for i in range(min(max_queries, len(train_data))):
            sample = train_data[i]
            
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
    """유사도 점수 파싱 (소수점 2자리까지 세분화)"""
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
    except:
        pass
    return 0.50  # 기본값 (소수점 2자리)

def parse_sde_response(response: str, original_query: str) -> List[str]:
    """SDE 응답 파싱 (본 쿼리와 중복 방지)"""
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
def generate_sde_paraphrases(client, cache: ResponseCache, model_name: str, query: str, config: LLMFilterConfig) -> List[str]:
    """SDE paraphrases 생성 (캐시 활용, 본 쿼리와 중복 방지)"""
    # 캐시에서 조회
    cached_paraphrases = load_sde_cache(query, config.cache_dir)
    if cached_paraphrases is not None:
        print(f"🎯 [SDE Cache] Found cached paraphrases for query: {query[:50]}...")
        return cached_paraphrases[:config.sde_k]
    
    # 캐시에 없으면 LLM으로 생성
    print(f"🔄 [SDE Cache] Generating paraphrases for query: {query[:50]}...")
    prompt = create_sde_prompt(query)
    response, _ = llm_cached_call(client, cache, model_name, prompt)
    paraphrases = parse_sde_response(response, query)
    
    # 원본 쿼리 추가 (첫 번째 paraphrase로)
    paraphrases = [query] + paraphrases[:config.sde_k-1]
    
    # 캐시에 저장
    save_sde_cache(query, paraphrases, config.cache_dir)
    print(f"💾 [SDE Cache] Saved paraphrases for query: {query[:50]}...")
    
    return paraphrases

# ===============================
# SDE Cache (SDE paraphrases만 캐시)
# ===============================
def get_sde_cache(cache_dir: str) -> ResponseCache:
    """SDE paraphrases 생성용 캐시만 반환"""
    return ResponseCache(os.path.join(cache_dir, "sde_paraphrases_cache.jsonl"))

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
def apply_cbc_filtering(scores: List[float], cbc_stats: CBCStats, threshold: float = 0.5) -> List[Tuple[float, bool]]:
    """CBC 기반 필터링 적용"""
    if not cbc_stats.stats['overlap']:
        # 통계가 없으면 기본 임계값 사용
        return [(float(score), bool(score >= threshold)) for score in scores]
    
    # CBC 임계값 계산
    cbc_threshold = cbc_stats.get_threshold('overlap', 80) * 0.8  # 보수적 적용
    
    results = []
    for score in scores:
        # 신뢰도 기반 보정
        if score >= cbc_threshold:
            results.append((float(score), True))
                else:
            results.append((float(score), False))
    
    return results

# ===============================
# Metrics Calculation
# ===============================
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
        
        # SDE paraphrases 생성
        paraphrases = generate_sde_paraphrases(client, get_sde_cache(config.cache_dir), model_name, query, config)
        
        query_scores = {}
        for paraphrase in paraphrases:
            scores = score_documents(client, model_name, paraphrase, passages)
            query_scores[paraphrase] = scores
        
        all_scores[query] = query_scores
    
    return all_scores

def run_experiments_with_scores(data: List[Dict[str, Any]], all_scores: Dict[str, Dict[str, List[float]]], config: LLMFilterConfig) -> List[Dict[str, Any]]:
    """생성된 점수를 기반으로 모든 실험 실행"""
    experiments = []
    
    # CBC 통계 초기화
    cbc_stats = CBCStats(config.cbc_cache_path)
    
    for item in tqdm(data, desc="실험 실행"):
        query = item['query']
        ground_truth = [int(gt) for gt in item['relevance_labels'][:config.max_passages]]
        
        query_scores = all_scores[query]
        
        # 실험별 결과 저장
        experiment_results = {
            'query': query,
            'ground_truth': ground_truth,
            'experiments': {}
        }
        
        # 1. LLM Only (Baseline)
        baseline_scores = query_scores[query]  # 본 쿼리만
        baseline_predictions = [bool(score >= config.sim_threshold) for score in baseline_scores]
        baseline_metrics = calculate_metrics(baseline_predictions, ground_truth)
        experiment_results['experiments']['baseline'] = {
            'scores': baseline_scores,
            'predictions': baseline_predictions,
            'metrics': baseline_metrics
        }
        
        # 2. LLM + SDE
        all_query_scores = list(query_scores.values())
        sde_avg_scores = [np.mean([scores[i] for scores in all_query_scores]) for i in range(len(baseline_scores))]
        sde_predictions = [bool(score >= config.sim_threshold) for score in sde_avg_scores]
        sde_metrics = calculate_metrics(sde_predictions, ground_truth)
        experiment_results['experiments']['sde'] = {
            'scores': sde_avg_scores,
            'predictions': sde_predictions,
            'metrics': sde_metrics
        }
        
        # 3. LLM + CBC
        cbc_threshold = cbc_stats.get_threshold('overlap', 80) * 0.8 if cbc_stats.stats['overlap'] else config.sim_threshold
        cbc_predictions = [bool(score >= cbc_threshold) for score in baseline_scores]
        cbc_metrics = calculate_metrics(cbc_predictions, ground_truth)
        experiment_results['experiments']['cbc'] = {
            'scores': baseline_scores,
            'predictions': cbc_predictions,
            'metrics': cbc_metrics
        }
        
        # 4. LLM + SDE + CBC
        sde_cbc_predictions = [bool(score >= cbc_threshold) for score in sde_avg_scores]
        sde_cbc_metrics = calculate_metrics(sde_cbc_predictions, ground_truth)
        experiment_results['experiments']['sde_cbc'] = {
            'scores': sde_avg_scores,
            'predictions': sde_cbc_predictions,
            'metrics': sde_cbc_metrics
        }
        
        # CBC 통계 업데이트
        if len(baseline_scores) > 1:
            sorted_indices = sorted(range(len(baseline_scores)), key=lambda i: baseline_scores[i], reverse=True)
            top_half = set(sorted_indices[:len(sorted_indices)//2])
            ground_truth_top = set(i for i, gt in enumerate(ground_truth) if gt == 1)
            overlap = len(top_half & ground_truth_top) / max(len(ground_truth_top), 1)
            margin = baseline_scores[sorted_indices[0]] - baseline_scores[sorted_indices[1]]
            cbc_stats.update(overlap, margin)
        
        experiments.append(experiment_results)
    
    return experiments

def run_optimized_experiments(data: List[Dict[str, Any]], config: LLMFilterConfig) -> List[Dict[str, Any]]:
    """최적화된 실험 실행"""
    print("\n🚀 최적화된 LLM 필터링 실험")
    print("="*60)
    
    # 1단계: 모든 유사도 점수 생성
    all_scores = generate_all_similarity_scores(data, config)
    
    # 2단계: 실험별 필터링 및 평가
    experiment_results = run_experiments_with_scores(data, all_scores, config)
    
    # 3단계: 결과 집계
    final_results = []
    
    for exp_type in ['baseline', 'sde', 'cbc', 'sde_cbc']:
        exp_precision = [r['experiments'][exp_type]['metrics']['precision'] for r in experiment_results]
        exp_recall = [r['experiments'][exp_type]['metrics']['recall'] for r in experiment_results]
        exp_f1 = [r['experiments'][exp_type]['metrics']['f1'] for r in experiment_results]
        
        overall_metrics = {
            'precision': np.mean(exp_precision),
            'recall': np.mean(exp_recall),
            'f1': np.mean(exp_f1)
        }
        
        final_results.append({
            'experiment_type': f'LLM_{exp_type.upper()}',
            'overall_metrics': overall_metrics,
            'results': experiment_results
        })
        
        print(f"✅ {exp_type.upper()}: P={overall_metrics['precision']:.3f}, R={overall_metrics['recall']:.3f}, F1={overall_metrics['f1']:.3f}")
    
    return final_results

def _old_run_llm_sde_experiment(data: List[Dict[str, Any]], config: LLMFilterConfig) -> Dict[str, Any]:
    """2. LLM + SDE 실험"""
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
    
    # LLM 점수는 캐시하지 않음 (실시간 계산)
    
    results = []
    total_latency = 0
    
    for item in tqdm(data, desc="LLM + SDE"):
        query = item['query']
        passages = item['passages'][:config.max_passages]
        ground_truth = [int(gt) for gt in item['relevance_labels'][:config.max_passages]]
        
        start_time = time.time()
        
        # SDE paraphrases 생성
        paraphrases = generate_sde_paraphrases(client, cache, model_name, query, config)
        
        # 각 paraphrase로 문서 점수 계산
            all_scores = []
        for paraphrase in paraphrases:
            scores = score_documents(client, cache, model_name, paraphrase, passages)
                all_scores.append(scores)
            
        # 점수 집계 (평균)
        avg_scores = [np.mean([scores[i] for scores in all_scores]) for i in range(len(passages))]
        
        # 임계값 기반 예측
        predictions = [bool(score >= config.sim_threshold) for score in avg_scores]
        
        latency = time.time() - start_time
        total_latency += latency
        
        # 메트릭 계산
        metrics = calculate_metrics(predictions, ground_truth)
        
        results.append({
            'query': query,
            'paraphrases': paraphrases,
            'all_scores': all_scores,
            'avg_scores': avg_scores,
            'predictions': predictions,
            'ground_truth': ground_truth,
            'metrics': metrics,
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
        
        # CBC 필터링 적용
        cbc_results = apply_cbc_filtering(scores, cbc_stats, config.sim_threshold)
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
        
        results.append({
            'query': query,
            'scores': scores,
            'cbc_results': cbc_results,
            'predictions': predictions,
            'ground_truth': ground_truth,
            'metrics': metrics,
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
        
        # SDE paraphrases 생성
        paraphrases = generate_sde_paraphrases(client, get_sde_cache(config.cache_dir), model_name, query, config)
        
        # 각 paraphrase로 문서 점수 계산
        all_scores = []
        for paraphrase in paraphrases:
            scores = score_documents(client, model_name, paraphrase, passages)
            all_scores.append(scores)
        
        # 점수 집계 (평균)
        avg_scores = [np.mean([scores[i] for scores in all_scores]) for i in range(len(passages))]
        
        # CBC 필터링 적용
        cbc_results = apply_cbc_filtering(avg_scores, cbc_stats, config.sim_threshold)
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
        
        results.append({
            'query': query,
            'paraphrases': paraphrases,
            'all_scores': all_scores,
            'avg_scores': avg_scores,
            'cbc_results': cbc_results,
            'predictions': predictions,
            'ground_truth': ground_truth,
            'metrics': metrics,
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
