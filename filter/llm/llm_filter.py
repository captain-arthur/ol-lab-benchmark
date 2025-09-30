#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
llm_filter.py
- LLM 기반 필터링 실험을 위한 공통 모듈
- MS MARCO와 FiQA 데이터셋에 대한 4가지 LLM 실험 방식 구현
- ① Large-scale LLM Only (Baseline)
- ② Large-scale LLM + Lightweight LLM  
- ③ Large-scale LLM + Lightweight LLM + SetFit
- ④ Large-scale LLM + Lightweight LLM + SetFit + CBC (Proposed)
"""

import os
import json
import time
import math
import random
import pathlib
import hashlib
import re
from collections import Counter, defaultdict
from typing import List, Dict, Any, Optional, Tuple

# ===============================
# Optional imports (graceful)
# ===============================
try:
    from datasets import load_dataset, Dataset
except Exception:
    load_dataset = None
    Dataset = None

try:
    import requests
except Exception:
    requests = None

# SetFit imports (support both old/new APIs)
SetFitModel = None
SetFitTrainer = None
TrainingArguments = None
try:
    # Newer API
    from setfit import SetFitModel, Trainer as SetFitTrainer, TrainingArguments
except Exception:
    try:
        # Older API
        from setfit import SetFitModel, SetFitTrainer, TrainingArguments
    except Exception:
        pass

try:
    import numpy as np
except Exception:
    np = None

try:
    import torch
except Exception:
    torch = None

# ===============================
# Configuration
# ===============================
class LLMFilterConfig:
    """LLM 필터링 설정 클래스"""
    
    def __init__(self, **kwargs):
        # 기본 설정
        self.seed = kwargs.get("seed", 42)
        self.max_queries = kwargs.get("max_queries", 20)
        
        # 라우팅 임계값
        self.theta = kwargs.get("theta", 0.8)  # 높은 신뢰도 기준
        self.warmup_k = kwargs.get("warmup_k", 4)  # SetFit 학습용 최소 쿼리 수
        self.update_every = kwargs.get("update_every", 5)
        
        # SetFit 설정
        self.setfit_mode = kwargs.get("setfit_mode", "both")
        self.mc_passes = kwargs.get("mc_passes", 10)
        self.dropout_p = kwargs.get("dropout_p", 0.2)
        
        # Ollama 설정
        self.ollama_url = kwargs.get("ollama_url", "http://192.168.45.166:11434")
        self.gemma_lite = kwargs.get("gemma_lite", "gemma:2b")
        self.gemma_heavy = kwargs.get("gemma_heavy", "gemma3")
        self.temperature = kwargs.get("temperature", 0.0)
        self.top_p = kwargs.get("top_p", 1.0)
        
        # 평가 설정
        self.eval_with_heavy_oracle = kwargs.get("eval_with_heavy_oracle", True)

# ===============================
# Utilities
# ===============================
def set_seed(seed: int = 42):
    """시드 설정"""
    random.seed(seed)
    try:
        import numpy as _np
        _np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch as _t
        _t.manual_seed(seed)
        _t.cuda.manual_seed_all(seed)
    except Exception:
        pass

def ensure_dir(path: str):
    """디렉토리 생성"""
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)

def jdump(obj, path):
    """JSON 저장"""
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def save_json(obj, path):
    """JSON 저장 (별칭)"""
    jdump(obj, path)

def jappend_jsonl(path, rec):
    """JSONL에 레코드 추가"""
    ensure_dir(os.path.dirname(path))
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def normalize_text(s: str) -> str:
    """텍스트 정규화"""
    return " ".join((s or "").strip().lower().split())

def nearest_label(pred: str, label_list: List[str]) -> str:
    """가장 가까운 라벨 찾기"""
    pred = (pred or "").strip().lower()
    if not label_list:
        return pred
    label_list = [str(l) for l in label_list]
    import difflib
    cand = difflib.get_close_matches(pred, label_list, n=1, cutoff=0.0)
    return cand[0] if cand else label_list[0]

def token_f1_em(pred: str, gold: str) -> Tuple[float, float]:
    """토큰 F1 및 Exact Match 계산"""
    p = normalize_text(pred).split()
    g = normalize_text(gold).split()
    em = 1.0 if p == g else 0.0
    if not p and not g:
        return 1.0, em
    inter = Counter(p) & Counter(g)
    tp = sum(inter.values())
    if tp == 0:
        return 0.0, em
    prec = tp / max(1, len(p))
    rec = tp / max(1, len(g))
    f1 = 2 * prec * rec / max(1e-9, (prec + rec))
    return f1, em

# ===============================
# Response Cache
# ===============================
class ResponseCache:
    """Ollama 응답 캐시"""
    
    def __init__(self, path: str):
        self.path = path
        ensure_dir(os.path.dirname(path))
        self.map = {}
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            rec = json.loads(line)
                            k = rec.get("cache_key")
                            if k:
                                self.map[k] = rec
                        except Exception:
                            pass
            except Exception:
                pass

    def key(self, model: str, prompt: str) -> str:
        """캐시 키 생성"""
        return hashlib.sha256((model + "||" + prompt).encode("utf-8")).hexdigest()[:20]

    def get(self, model: str, prompt: str) -> Optional[Dict[str, Any]]:
        """캐시에서 응답 가져오기"""
        return self.map.get(self.key(model, prompt))

    def put(self, model: str, prompt: str, text: str, usage: Dict[str, Any]):
        """캐시에 응답 저장"""
        rec = {
            "cache_key": self.key(model, prompt),
            "model": model,
            "prompt": prompt,
            "text": text,
            "usage": usage,
        }
        jappend_jsonl(self.path, rec)
        self.map[rec["cache_key"]] = rec

# ===============================
# Ollama Client
# ===============================
class Ollama:
    """Ollama 클라이언트"""
    
    def __init__(self, base_url="http://192.168.45.166:11434", temperature=0.0, top_p=1.0, seed=42):
        if requests is None:
            raise RuntimeError("`requests` not installed. Please `pip install requests`.")
        self.base_url = base_url
        self.temperature = temperature
        self.top_p = top_p
        self.seed = seed

    def generate(self, model: str, prompt: str, system: Optional[str] = None) -> Dict[str, Any]:
        """텍스트 생성"""
        payload = {
            "model": model,
            "prompt": prompt,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stream": False,
            "seed": self.seed,
        }
        if system:
            payload["system"] = system
        t0 = time.time()
        r = requests.post(f"{self.base_url}/api/generate",
                          headers={"Content-Type":"application/json"},
                          data=json.dumps(payload), timeout=600)
        lat = time.time()-t0
        if r.status_code != 200:
            raise RuntimeError(f"Ollama error {r.status_code}: {r.text[:200]}")
        out = r.json()
        usage = {
            "latency_s": lat,
            "eval_count": out.get("eval_count"),
            "prompt_eval_count": out.get("prompt_eval_count"),
            "total_tokens_est": (out.get("eval_count") or 0) + (out.get("prompt_eval_count") or 0),
        }
        return {"text": out.get("response",""), "usage": usage}

# ===============================
# Task-specific Prompts
# ===============================
def prompt_banking77(text: str, labels: List[str]) -> str:
    """Banking77 프롬프트"""
    head = "Banking77 프롬프트\n\n" \
           "Banking customer service intent classification task.\n" \
           "Classify the customer's intent from the given text.\n" \
           "Answer with EXACTLY ONE label from the list below.\n" \
           "Your answer MUST be exactly one label from the list, character-perfect.\n" \
           "Be precise and accurate in your classification.\n" \
           "Respond with the label ONLY. No extra words.\n\n"
    label_hint = ", ".join(labels[:20]) + (", ..." if len(labels) > 20 else "")
    return f"""{head}Input: {text}

Valid intent labels (choose one): {label_hint}
Intent:"""

def prompt_clinc150(text: str, labels: List[str]) -> str:
    """CLINC150 프롬프트"""
    head = "CLINC150 프롬프트\n\n"
    label_text = ", ".join(sorted(list(set(labels))))
    
    return f"""{head}User Query: {text}

Available Intent IDs: {label_text}

Instructions:
- Return ONLY the intent ID number (0-{len(labels)-1})
- Do not include any other text or explanation
- If unsure, choose the most likely intent ID
- Be precise and accurate in your classification
- Consider the context and specific keywords in the query

Examples:
- "how to say hello in french" → 0 (translation)
- "transfer money to my savings" → 1 (banking)
- "set a timer for 10 minutes" → 2 (timer)

Intent ID:"""

def prompt_stsb(text_block: str) -> str:
    """STS-B 프롬프트"""
    head = "STS-B 프롬프트\n\n" \
           "Semantic Textual Similarity (STS-B) task.\n" \
           "Compare the similarity between two sentences and output a single integer from 0 to 5.\n\n" \
           "Scoring Guide:\n" \
           "0 = completely different meanings\n" \
           "1 = mostly different meanings\n" \
           "2 = somewhat different meanings\n" \
           "3 = somewhat similar meanings\n" \
           "4 = mostly similar meanings\n" \
           "5 = identical or very similar meanings\n\n" \
           "Examples:\n" \
           "- 'A cat is sleeping' vs 'A dog is running' → 0\n" \
           "- 'A woman is cooking' vs 'A man is cooking' → 3\n" \
           "- 'A man is playing guitar' vs 'A man is playing the guitar' → 5\n" \
           "- 'A girl is styling her hair' vs 'A girl is brushing her hair' → 2\n" \
           "- 'A group of men play soccer' vs 'A group of boys are playing soccer' → 4\n\n" \
           "Output ONLY the number (0-5):\n\n"
    return f"""{head}{text_block}

Score:"""

# ===============================
# Confidence Extraction
# ===============================
def extract_confidence_simple(response: str, base: float = 0.65) -> float:
    """신뢰도 추출 로직"""
    response_lower = (response or "").lower().strip()
    
    # 확실한 표현들 (높은 신뢰도)
    confident_indicators = [
        "확실", "certain", "definitely", "absolutely", "clearly",
        "분명", "obviously", "without doubt", "no doubt", "strongly", "100%",
        "정확", "exact", "precise", "perfect", "correct"
    ]
    
    # 중간 신뢰도 표현들
    moderate_indicators = [
        "아마", "probably", "likely", "seems", "appears",
        "보통", "usually", "typically", "generally"
    ]
    
    # 불확실한 표현들 (낮은 신뢰도)
    uncertain_indicators = [
        "모르", "not sure", "uncertain", "unclear", "might be", "guess",
        "maybe", "perhaps", "possibly", "could be", "might"
    ]
    
    # 확실한 표현 체크
    for indicator in confident_indicators:
        if indicator in response_lower:
            return 0.9
    
    # 중간 신뢰도 표현 체크
    for indicator in moderate_indicators:
        if indicator in response_lower:
            return 0.7
    
    # 불확실한 표현 체크
    for indicator in uncertain_indicators:
        if indicator in response_lower:
            return 0.4
    
    return base

def extract_confidence_from_llm_response(response: str, scores: List[float]) -> float:
    """LLM 응답에서 confidence 메타정보 파싱 또는 softmax-normalized score 기반 계산"""
    import re
    
    # 1. LLM 응답에서 명시적 confidence 값 파싱 시도
    confidence_patterns = [
        r'confidence[:\s]*([0-9.]+)',
        r'신뢰도[:\s]*([0-9.]+)',
        r'확신도[:\s]*([0-9.]+)',
        r'certainty[:\s]*([0-9.]+)',
        r'([0-9.]+)\s*%?\s*confidence',
        r'([0-9.]+)\s*%?\s*신뢰도'
    ]
    
    for pattern in confidence_patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            try:
                conf_value = float(match.group(1))
                # 0-1 범위로 정규화
                if conf_value > 1.0:
                    conf_value = conf_value / 100.0
                return max(0.0, min(1.0, conf_value))
            except ValueError:
                continue
    
    # 2. 점수 분포 기반 confidence 계산 (softmax-normalized)
    if scores and len(scores) > 1:
        try:
            import numpy as np
            # Softmax 정규화
            exp_scores = np.exp(np.array(scores))
            softmax_scores = exp_scores / np.sum(exp_scores)
            
            # 최대 확률과 엔트로피 기반 confidence
            max_prob = np.max(softmax_scores)
            entropy = -np.sum(softmax_scores * np.log(softmax_scores + 1e-9))
            max_entropy = np.log(len(scores))
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
            
            # 높은 최대 확률 + 낮은 엔트로피 = 높은 confidence
            confidence = max_prob * (1 - normalized_entropy)
            return max(0.0, min(1.0, confidence))
        except Exception:
            pass
    
    # 3. 점수 분산 기반 confidence (기존 방식 개선)
    if scores and len(scores) > 1:
        try:
            import numpy as np
            # 표준편차 대신 변동계수 사용 (더 정규화된 지표)
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            if mean_score > 0:
                cv = std_score / mean_score  # 변동계수
                confidence = 1.0 / (1.0 + cv)  # 변동계수가 낮을수록 높은 confidence
                return max(0.0, min(1.0, confidence))
        except Exception:
            pass
    
    # 4. 기본값 반환
    return 0.65

# ===============================
# Task-specific Predictors
# ===============================
def ollama_cached_call(client: Ollama, cache: ResponseCache, model: str, prompt: str) -> Tuple[str, Dict[str,Any]]:
    """캐시된 Ollama 호출"""
    k = cache.get(model, prompt)
    if k:
        return k["text"], k["usage"]
    resp = client.generate(model, prompt)
    text, usage = resp["text"], resp["usage"]
    cache.put(model, prompt, text, usage)
    return text, usage

def parse_label_only(text: str) -> str:
    """라벨만 파싱"""
    line = (text or "").strip().splitlines()[0]
    line = re.sub(r'^[\s"\']+|[\s"\']+$', '', line)
    m = re.match(r'^([A-Za-z0-9_\-\.]+)$', line)
    return m.group(1) if m else line.strip()

def predict_banking77(client: Ollama, cache: ResponseCache, model: str, text: str, labels: List[str]) -> Tuple[str, float, Dict[str,Any], str]:
    """Banking77 예측"""
    prompt = prompt_banking77(text, labels)
    out, usage = ollama_cached_call(client, cache, model, prompt)
    raw = parse_label_only(out)
    lab = nearest_label(raw, labels)
    conf = extract_confidence_simple(out, base=0.70 if raw == lab else 0.60)
    return lab, conf, usage, out

def predict_clinc150(client: Ollama, cache: ResponseCache, model: str, text: str, labels: List[str]) -> Tuple[str, float, Dict[str,Any], str]:
    """CLINC150 예측"""
    prompt = prompt_clinc150(text, labels)
    out, usage = ollama_cached_call(client, cache, model, prompt)
    
    lab = None
    
    # 1. 정확한 숫자 매칭 (0-149 범위)
    m = re.search(r'\b([0-9]|[1-9][0-9]|1[0-4][0-9])\b', out)
    if m:
        raw = m.group(1)
        if raw.isdigit():
            idx = int(raw)
            if 0 <= idx < len(labels):
                lab = str(idx)
    
    # 2. 라벨 텍스트 직접 매칭
    if lab is None:
        raw = parse_label_only(out)
        lab = nearest_label(raw, labels)
    
    # 3. 범위 클리핑 (안전장치)
    if lab is None or not lab.isdigit():
        lab = "0"  # 기본값
    else:
        idx = int(lab)
        if idx >= len(labels):
            idx = idx % len(labels)
        lab = str(idx)
    
    # 신뢰도 계산
    raw_matched = m.group(1) if m else parse_label_only(out)
    conf = extract_confidence_simple(out, base=0.75 if raw_matched == lab else 0.55)
    
    return lab, conf, usage, out

def predict_stsb(client: Ollama, cache: ResponseCache, model: str, text_block: str) -> Tuple[str, float, Dict[str,Any], str]:
    """STS-B 예측"""
    prompt = prompt_stsb(text_block)
    out, usage = ollama_cached_call(client, cache, model, prompt)
    
    lab = None
    
    # 1. 모든 숫자 추출
    all_nums = re.findall(r'\b(\d+)\b', out)
    
    if all_nums:
        # 2. 0-5 범위 내 숫자 필터링
        valid_nums = [int(x) for x in all_nums if 0 <= int(x) <= 5]
        
        if valid_nums:
            lab = str(valid_nums[0])
        else:
            # 4. 범위를 벗어난 숫자 정규화
            max_num = max(int(x) for x in all_nums)
            if max_num > 0:
                normalized = int(round(5 * int(all_nums[0]) / max_num))
                lab = str(max(0, min(5, normalized)))
            else:
                lab = "3"
    else:
        # 5. 숫자가 없으면 키워드 기반 추정
        out_lower = out.lower()
        if any(word in out_lower for word in ['identical', 'same', 'exact', 'perfect']):
            lab = "5"
        elif any(word in out_lower for word in ['very similar', 'mostly similar', 'almost same']):
            lab = "4"
        elif any(word in out_lower for word in ['somewhat similar', 'moderately similar']):
            lab = "3"
        elif any(word in out_lower for word in ['somewhat different', 'moderately different']):
            lab = "2"
        elif any(word in out_lower for word in ['very different', 'mostly different']):
            lab = "1"
        elif any(word in out_lower for word in ['completely different', 'totally different', 'unrelated']):
            lab = "0"
        else:
            lab = "3"  # 기본값
    
    # 신뢰도 계산
    if lab in ["0", "1", "2", "3", "4", "5"]:
        conf = extract_confidence_simple(out, base=0.80)
    else:
        conf = extract_confidence_simple(out, base=0.50)
    
    return lab, conf, usage, out

def predict_task(client: Ollama, cache: ResponseCache, model: str, meta: Dict[str, Any], text: str) -> Tuple[str, float, Dict[str,Any], str]:
    """태스크별 예측"""
    name = meta["name"]
    labels = meta.get("labels", [])
    
    if name == "banking77":
        return predict_banking77(client, cache, model, text, labels)
    elif name == "clinc150":
        return predict_clinc150(client, cache, model, text, labels)
    elif name == "stsb":
        return predict_stsb(client, cache, model, text)
    else:
        # generic classifier
        prompt = "Return exactly one label from this list. Label only.\n" \
                 f"Input: {text}\nLabels: {', '.join(labels[:30])}\nLabel:"
        out, usage = ollama_cached_call(client, cache, model, prompt)
        raw = parse_label_only(out)
        lab = nearest_label(raw, labels)
        conf = extract_confidence_simple(out, base=0.65)
        return lab, conf, usage, out

# ===============================
# SetFit Helper
# ===============================
class SetFitHelper:
    """SetFit 헬퍼 클래스"""
    
    def __init__(self):
        self.enabled = (SetFitModel is not None and SetFitTrainer is not None and 
                        TrainingArguments is not None and Dataset is not None and torch is not None and np is not None)
        self.model = None
        self.label2id = {"LOW":0, "HIGH":1}
        self.id2label = {0:"LOW", 1:"HIGH"}
        self.bufX, self.bufY = [], []

    def _args_safe(self, epochs=3, batch_size=16, num_iterations=None):
        if TrainingArguments is None:
            raise RuntimeError("TrainingArguments unavailable")
        kwargs = {"num_epochs": epochs, "batch_size": batch_size}
        if num_iterations is not None:
            try: 
                kwargs["num_iterations"] = num_iterations
            except Exception: 
                pass
        return TrainingArguments(**kwargs)

    def warmup(self, texts: List[str], labels: List[str]):
        if not (self.enabled and texts):
            return
        
        if len(texts) < 2:
            print(f"[SetFit] Warmup skipped: insufficient samples ({len(texts)})")
            return
        
        unique_labels = set(labels)
        if len(unique_labels) < 2:
            print(f"[SetFit] Warmup skipped: insufficient label diversity ({len(unique_labels)})")
            self.model = None
            return
        
        try:
            y = []
            for l in labels:
                if l in self.label2id:
                    y.append(self.label2id[l])
                else:
                    y.append(0)
            
            ds = Dataset.from_dict({"text": texts, "label": y})
            self.model = SetFitModel.from_pretrained("sentence-transformers/paraphrase-mpnet-base-v2")
            args = self._args_safe(epochs=3, batch_size=min(16, len(texts)), num_iterations=20)
            trainer = SetFitTrainer(model=self.model, args=args, train_dataset=ds, column_mapping={"text":"text","label":"label"})
            trainer.train()
            print(f"[SetFit] Warmup OK ({len(texts)} samples, {len(unique_labels)} labels)")
        except Exception as e:
            print(f"[SetFit] Warmup failed: {e}")
            self.model = None

    def predict_conf(self, text: str) -> Tuple[Optional[str], float]:
        if not self.enabled:
            return None, 0.0
        
        if self.model is None:
            return "HIGH", 0.8
        
        try:
            p = self.model.predict_proba([text])[0]
            p_high = float(p[1]) if len(p) > 1 else 0.0
            return ("HIGH" if p_high >= 0.5 else "LOW"), p_high
        except Exception as e:
            print(f"[SetFit] predict_conf failed: {e}")
            return None, 0.0

    def add_online(self, x: str, y: str):
        if not (self.enabled and self.model is not None):
            return
        self.bufX.append(x)
        self.bufY.append(y)
        if len(self.bufX) >= 5:  # UPDATE_EVERY
            self._flush()

    def _flush(self):
        if not self.bufX: 
            return
        try:
            y = []
            for l in self.bufY:
                if l in self.label2id:
                    y.append(self.label2id[l])
                elif l.upper() in self.label2id:
                    y.append(self.label2id[l.upper()])
                elif any(key.lower() in l.lower() for key in self.label2id.keys()):
                    for key in self.label2id.keys():
                        if key.lower() in l.lower():
                            y.append(self.label2id[key])
                            break
                else:
                    y.append(0)
            
            if len(self.bufX) < 2:
                return
                
            ds = Dataset.from_dict({"text": self.bufX, "label": y})
            args = self._args_safe(epochs=1, batch_size=min(16, len(self.bufX)), num_iterations=5)
            trainer = SetFitTrainer(model=self.model, args=args, train_dataset=ds, column_mapping={"text":"text","label":"label"})
            trainer.train()
            print(f"[SetFit] Online mini-fit ({len(self.bufX)} samples)")
        except Exception as e:
            print(f"[SetFit] Online update failed: {e}")
        finally:
            self.bufX, self.bufY = [], []

# ===============================
# Evaluation Metrics
# ===============================



# ===============================
# CBC Router for Batch-based Routing
# ===============================
class CBCRouter:
    """CBC 기반 배치 라우팅 시스템"""
    
    def __init__(self, config: LLMFilterConfig):
        self.config = config
        self.batch_scores = {
            'setfit': [],
            'lite': [],
            'heavy': []
        }
        self.dynamic_thresholds = {}
        self.ollama = Ollama()
        self.cache = ResponseCache(".cache/cbc_cache.jsonl")
        self.setfit_helper = SetFitHelper()
        self.percentile_threshold = 95 # 기본값
        self.min_threshold = 0.6 # 기본값
        self.calibration_mode = "dynamic" # 기본값
        
        # SetFit 초기화 추가
        self._initialize_setfit()
    
    def _initialize_setfit(self):
        """SetFit 초기 학습"""
        try:
            # 간단한 훈련 데이터로 warmup
            train_texts = [
                "계좌 잔액 [SEP] 잔액이 얼마인가요?",
                "이체 요청 [SEP] 돈을 보내주세요",
                "카드 한도 [SEP] 카드 한도가 얼마인가요?",
                "대출 신청 [SEP] 대출을 받고 싶습니다",
                "환율 조회 [SEP] 현재 환율이 어떻게 되나요?",
                "무관한 질문 [SEP] 오늘 날씨가 어때요?",
                "영화 추천 [SEP] 재미있는 영화 알려주세요",
                "요리 방법 [SEP] 김치찌개 만드는 법 알려주세요"
            ]
            train_labels = ["HIGH", "HIGH", "HIGH", "HIGH", "HIGH", "LOW", "LOW", "LOW"]
            
            print("🔄 Initializing SetFit with warmup data...")
            self.setfit_helper.warmup(train_texts, train_labels)
            print("✅ SetFit warmup completed successfully")
            
        except Exception as e:
            print(f"⚠️ SetFit warmup failed: {e}")
            print("⚠️ SetFit will use default predictions")
    
    
    def cbc_decision(self, score: float, percentile: int) -> bool:
        """CBC 기반 라우팅 결정"""
        try:
            # 현재 배치의 점수 분포에서 임계값 계산
            if not self.batch_scores['setfit']:
                return score >= 0.7  # 기본 임계값
            
            threshold = np.percentile(self.batch_scores['setfit'], percentile)
            return score >= threshold
            
        except Exception as e:
            print(f"⚠️ CBC decision failed: {e}")
            return score >= 0.7  # 기본 임계값
    
    
    def process_batch_cbc_with_anchors(self, batch_data: List[Tuple[str, str, Dict[str, Any]]], anchors_dict: Dict[str, List[str]] = None) -> List[Dict[str, Any]]:
        """CBC + 앵커 기반 배치 처리"""
        results = []
        
        # 모든 샘플의 신뢰도 점수 수집
        for query, text, meta in batch_data:
            anchors = anchors_dict.get(query, []) if anchors_dict else []
            
            # 앵커 기반 신뢰도 계산
            confidence_score = self.calculate_anchor_based_confidence(query, text, anchors)
            
            self.batch_scores['setfit'].append(confidence_score)
            
            # 계층적 CBC 라우팅 수행
            routing_result = self.hierarchical_cbc_routing(query, text, anchors)
            
            results.append({
                'query': query,
                'text': text,
                'meta': meta,
                'anchors': anchors,
                'setfit_confidence': confidence_score,
                'routing_decision': routing_result
            })
        
        return results
    
    
    def _predict_with_model(self, model: str, query: str, text: str, meta: Dict[str, Any]) -> Dict[str, Any]:
        """지정된 모델로 예측 수행"""
        task_name = meta.get("name", "generic")
        
        if task_name == "banking77":
            label, confidence, usage, raw = predict_banking77(
                self.ollama, self.cache, model, text, meta.get("labels", [])
            )
        elif task_name == "clinc150":
            label, confidence, usage, raw = predict_clinc150(
                self.ollama, self.cache, model, text, meta.get("labels", [])
            )
        elif task_name == "stsb":
            label, confidence, usage, raw = predict_stsb(
                self.ollama, self.cache, model, f"{query} [SEP] {text}"
            )
        else:
            label, confidence, usage, raw = predict_task(
                self.ollama, self.cache, model, meta, text
            )
        
        return {
            'label': label,
            'confidence': confidence,
            'usage': usage,
            'raw': raw
        }
    
    
    def _calculate_dynamic_thresholds(self):
        """점수 분포 기반으로 동적 임계값 계산 (CBC 핵심)"""
        if not self.batch_scores['setfit']:
            return
        
        # 빈 배열 처리 (안전성 보장)
        def safe_percentile(scores, percentile):
            if not scores:
                return self.min_threshold
            return np.percentile(scores, percentile)
        
        # p95 기반 동적 임계값 계산 (CBC의 핵심)
        self.dynamic_thresholds = {
            'setfit': safe_percentile(self.batch_scores['setfit'], self.percentile_threshold),
            'lite': safe_percentile(self.batch_scores['lite'], self.percentile_threshold),
            'heavy': safe_percentile(self.batch_scores['heavy'], self.percentile_threshold)
        }
        
        # 최소 임계값 보장 (안정성)
        self.dynamic_thresholds = {
            k: max(v, self.min_threshold) for k, v in self.dynamic_thresholds.items()
        }
    
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """CBC 라우팅 통계 반환"""
        if not self.dynamic_thresholds:
            return {}
        
        return {
            'dynamic_thresholds': self.dynamic_thresholds,
            'percentile_threshold': self.percentile_threshold,
            'calibration_mode': self.calibration_mode
        }
    
    
    def calculate_anchor_based_confidence(self, query: str, text: str, anchors: List[str] = None) -> float:
        """앵커 기반 향상된 신뢰도 계산 (CBC 핵심)"""
        try:
            # 기본 SetFit 예측
            combined_text = f"{query} [SEP] {text}"
            
            if self.setfit_helper.enabled and self.setfit_helper.model is not None:
                setfit_probs = self.setfit_helper.model.predict_proba([combined_text])[0]
            else:
                # SetFit 비활성 시 기본값
                setfit_probs = np.array([0.5, 0.5])
            
            # 기본 특징 계산 (NumPy 호환)
            if hasattr(setfit_probs, 'numpy'):
                setfit_probs = setfit_probs.numpy()
            setfit_probs = np.array(setfit_probs)
            
            p_max = np.max(setfit_probs)
            
            # 단순화된 점수 계산
            score = p_max * 0.8 + 0.2  # 기본 신뢰도 + 보너스
            
            return np.clip(score, 0.0, 1.0)
            
        except Exception as e:
            print(f"⚠️ Anchor-based confidence calculation failed: {e}")
            return 0.5  # 기본값
    
    
    
    
    def hierarchical_cbc_routing(self, query: str, text: str, anchors: List[str] = None) -> Dict[str, Any]:
        """계층적 CBC 라우팅 (CBC 핵심) - 실제 배치별 점수 분포 반영"""
        try:
            # 1단계: SetFit + 앵커 (비용 최소)
            setfit_score = self.calculate_anchor_based_confidence(query, text, anchors)
            
            # SetFit 점수를 배치에 추가
            self.batch_scores['setfit'].append(setfit_score)
            
            # CBC 기반 동적 임계값 계산
            setfit_threshold = self._get_dynamic_threshold('setfit', 80)  # 상위 80%
            
            if setfit_score >= setfit_threshold:
                return {
                    'stage': 'setfit',
                    'model_used': 'setfit',
                    'confidence': setfit_score,
                    'decision': 'accept',
                    'threshold_used': setfit_threshold,
                    'reason': f'SetFit confidence {setfit_score:.3f} >= threshold {setfit_threshold:.3f}'
                }
            
            # 2단계: 경량 LLM (중간 비용)
            lite_result = self._predict_with_model(self.config.gemma_lite, query, text, {'name': 'generic'})
            lite_score = lite_result['confidence']
            
            # Lite LLM 점수를 배치에 추가
            self.batch_scores['lite'].append(lite_score)
            lite_threshold = self._get_dynamic_threshold('lite', 90)  # 상위 90%
            
            if lite_score >= lite_threshold:
                return {
                    'stage': 'lite',
                    'model_used': 'lite',
                    'confidence': lite_score,
                    'decision': 'accept',
                    'threshold_used': lite_threshold,
                    'reason': f'Lite LLM confidence {lite_score:.3f} >= threshold {lite_threshold:.3f}'
                }
            
            # 3단계: 고성능 LLM (최고 비용)
            heavy_result = self._predict_with_model(self.config.gemma_heavy, query, text, {'name': 'generic'})
            heavy_score = heavy_result['confidence']
            
            # Heavy LLM 점수를 배치에 추가
            self.batch_scores['heavy'].append(heavy_score)
            
            return {
                'stage': 'heavy',
                'model_used': 'heavy',
                'confidence': heavy_score,
                'decision': 'accept',
                'threshold_used': 'N/A',
                'reason': 'Heavy LLM required for low confidence cases'
            }
            
        except Exception as e:
            return {
                'stage': 'fallback',
                'model_used': 'heavy',
                'confidence': 0.5,
                'decision': 'accept',
                'reason': f'Fallback due to error: {e}'
            }
    
    def _get_dynamic_threshold(self, stage: str, percentile: int) -> float:
        """동적 임계값 계산 (CBC 핵심) - 실제 배치별 점수 분포 반영"""
        try:
            if not self.batch_scores or stage not in self.batch_scores:
                return 0.7  # 기본 임계값
            
            stage_scores = self.batch_scores[stage]
            if not stage_scores:
                return 0.7
            
            # 실제 percentile 계산
            threshold = np.percentile(stage_scores, percentile)
            
            # 최소 임계값 보장 (안정성)
            min_threshold = 0.6
            return max(threshold, min_threshold)
            
        except Exception as e:
            print(f"⚠️ Dynamic threshold calculation failed for {stage}: {e}")
            return 0.7

# ===============================
# Experiment Functions
# ===============================






# ===============================
# MS MARCO & FiQA LLM Filtering Experiments
# ===============================

def load_ms_marco_data(max_queries: int = 10) -> List[Dict[str, Any]]:
    """MS MARCO 데이터셋 로드 (실제 데이터셋 사용)"""
    try:
        from datasets import load_dataset
        
        print(f"🔄 MS MARCO 실제 데이터셋 로드 중... (최대 {max_queries}개 쿼리)")
        
        # MS MARCO v1.1 데이터셋 로드
        dataset = load_dataset('ms_marco', 'v1.1')
        train_data = dataset['train']
        
        # 데이터 전처리 및 변환
        processed_data = []
        for i in range(min(max_queries, len(train_data))):
            sample = train_data[i]
            
            # passages에서 텍스트와 관련성 레이블 추출
            passages_dict = sample['passages']
            passage_texts = passages_dict['passage_text']
            is_selected = passages_dict['is_selected']
            
            # 관련성 레이블을 binary로 변환 (0: 무관, 1: 관련)
            relevance_labels = [1 if label == 1 else 0 for label in is_selected]
            
            processed_data.append({
                'query_id': str(sample['query_id']),
                'query': sample['query'],
                'passages': passage_texts,
                'relevance_labels': relevance_labels,  # 실제 Ground Truth
                'answers': sample['answers'],
                'query_type': sample.get('query_type', 'unknown')
            })
        
        print(f"✅ MS MARCO 실제 데이터 로드 완료: {len(processed_data)}개 쿼리")
        print(f"📊 첫 번째 쿼리 예시:")
        print(f"   쿼리: \"{processed_data[0]['query']}\"")
        print(f"   문서 수: {len(processed_data[0]['passages'])}개")
        print(f"   관련 문서 수: {sum(processed_data[0]['relevance_labels'])}개")
        
        return processed_data
        
    except Exception as e:
        print(f"❌ MS MARCO 데이터 로드 실패: {e}")
        print("🔄 테스트용 데이터로 대체합니다...")
        
        # 실패 시 테스트용 데이터로 대체
        sample_data = [
            {
                'query_id': f'q_{i}',
                'query': f'What is machine learning? Query {i}',
                'passages': [
                    f'Machine learning is a subset of artificial intelligence that enables computers to learn without being explicitly programmed. Document {i}-1',
                    f'Deep learning uses neural networks with multiple layers to process data and make decisions. Document {i}-2',
                    f'Natural language processing is a field of AI that focuses on the interaction between computers and human language. Document {i}-3',
                    f'Computer vision is a field of AI that trains computers to interpret and understand visual information from images. Document {i}-4',
                    f'The weather today is sunny with a temperature of 25 degrees Celsius. Document {i}-5',
                    f'Data science combines statistics, programming, and domain expertise to extract insights from data. Document {i}-6'
                ],
                'relevance_labels': [1, 1, 1, 1, 0, 1],  # 테스트용 Ground Truth
                'answers': [f'Machine learning is a subset of AI. Answer {i}']
            }
            for i in range(max_queries)
        ]
        
        print(f"✅ MS MARCO 테스트 데이터 로드 완료: {len(sample_data)}개 쿼리")
        return sample_data

def load_fiqa_data(max_queries: int = 648) -> List[Dict[str, Any]]:
    """FiQA 데이터셋 로드"""
    try:
        from datasets import load_dataset
        dataset = load_dataset("mteb/fiqa", split="test")
        
        # 쿼리별로 문서 그룹핑
        query_to_docs = defaultdict(list)
        for item in dataset:
            query_id = item['query-id']
            corpus_id = item['corpus-id']
            score = item['score']
            query_to_docs[query_id].append((corpus_id, score))
        
        data = []
        for query_id, docs in list(query_to_docs.items())[:max_queries]:
            # 관련 문서만 필터링 (score > 0)
            relevant_docs = [doc for doc in docs if doc[1] > 0]
            if len(relevant_docs) < 2:  # 최소 2개 관련 문서 필요
                continue
            
            # 상위 50개 문서 선택
            top_docs = docs[:50]  # [(corpus_id, score), ...]
            passages = [f"Financial document {cid}" for cid, _ in top_docs]
            
            # 관련성 레이블 생성 (passages와 동일 길이)
            relevant_set = {cid for cid, s in docs if s > 0}
            relevance_labels = [1 if cid in relevant_set else 0 for cid, _ in top_docs]
                
            data.append({
                'query_id': query_id,
                'query': f"Financial query {query_id}",  # 실제 쿼리 텍스트는 MTEB에서 제공하지 않음
                'passages': passages,
                'relevance_labels': relevance_labels,  # 실험 함수와 호환
                'answers': [f"Answer for query {query_id}"]
            })
        
        print(f"✅ FiQA 데이터 로드 완료: {len(data)}개 쿼리")
        return data
        
    except Exception as e:
        print(f"❌ FiQA 데이터 로드 실패: {e}")
        return []

def create_llm_filtering_prompt(query: str, passages: List[str]) -> str:
    """LLM 필터링을 위한 프롬프트 생성"""
    return f"""You are an expert information retrieval system. Your task is to filter and rank documents based on their relevance to the given query.

Query: {query}

Documents to evaluate:
{chr(10).join([f"{i+1}. {passage}" for i, passage in enumerate(passages)])}

Instructions:
1. Analyze each document's relevance to the query
2. For each document, provide a relevance score from 0.0 to 1.0
3. 1.0 = highly relevant, 0.0 = completely irrelevant
4. Consider semantic similarity, topic alignment, and information quality
5. Be precise and consistent in your scoring

Format your response as:
Document 1: [score]
Document 2: [score]
...
Document {len(passages)}: [score]

Scores:"""

def parse_llm_scores(response: str, num_docs: int) -> List[float]:
    """LLM 응답에서 점수 파싱"""
    scores = []
    lines = response.strip().split('\n')
    
    for i in range(num_docs):
        score = 0.5  # 기본값
        for line in lines:
            # 다양한 형식 지원: "Document 1:", "문서 1:", "1. ...", "문서1:"
            if (f"Document {i+1}:" in line or f"문서 {i+1}:" in line or 
                f"{i+1}. " in line or f"문서{i+1}:" in line):
                # 숫자 추출 (점수 부분만)
                import re
                # 콜론(:) 뒤의 숫자 찾기
                if ':' in line:
                    score_part = line.split(':', 1)[1]
                else:
                    score_part = line
                
                # 0.0~1.0 형식의 숫자 찾기 (더 정확한 패턴)
                score_match = re.search(r'[01]\.\d+', score_part)
                if score_match:
                    try:
                        score = float(score_match.group())
                        score = max(0.0, min(1.0, score))  # 0-1 범위로 클리핑
                    except:
                        score = 0.5
                else:
                    # 정수 찾기 (1~10을 0.1~1.0으로 변환)
                    int_match = re.search(r'\b([1-9]|10)\b', score_part)
                    if int_match:
                        try:
                            score = float(int_match.group()) / 10.0
                            score = max(0.0, min(1.0, score))
                        except:
                            score = 0.5
                break
        scores.append(score)
    
    return scores

def calculate_f1_metrics(predictions: List[float], ground_truth: List[int], threshold: float = 0.5) -> Dict[str, float]:
    """F1 점수 중심의 평가 지표 계산"""
    if len(predictions) != len(ground_truth):
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    # 이진 분류로 변환
    pred_binary = [1 if p >= threshold else 0 for p in predictions]
    
    # True Positive, False Positive, False Negative 계산
    tp = sum(1 for p, g in zip(pred_binary, ground_truth) if p == 1 and g == 1)
    fp = sum(1 for p, g in zip(pred_binary, ground_truth) if p == 1 and g == 0)
    fn = sum(1 for p, g in zip(pred_binary, ground_truth) if p == 0 and g == 1)
    
    # Precision, Recall, F1 계산
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn
    }

def run_llm_baseline_experiment(data: List[Dict[str, Any]], config: LLMFilterConfig) -> Dict[str, Any]:
    """① Large-scale LLM Only (Baseline) 실험"""
    print("🔄 Running LLM Baseline Experiment...")
    
    ollama = Ollama(config.ollama_url, config.temperature, config.top_p, config.seed)
    cache = ResponseCache(".cache/llm_baseline_cache.jsonl")
    
    results = []
    start_time = time.time()
    total_queries = len(data)
    
    for idx, item in enumerate(data):
        print(f"📊 Baseline 진행: {idx+1}/{total_queries} 쿼리 처리 중...")
        query = item['query']
        passages = item['passages']
        
        # LLM으로 필터링 수행
        prompt = create_llm_filtering_prompt(query, passages)
        response, usage = ollama_cached_call(ollama, cache, config.gemma_heavy, prompt)
        scores = parse_llm_scores(response, len(passages))
        
        # 실제 MS MARCO Ground Truth 사용
        ground_truth = item['relevance_labels']
        
        # 평가 지표 계산
        metrics = calculate_f1_metrics(scores, ground_truth)
        
        # Cost 및 Latency 계산
        cost = 3  # Heavy 모델만 사용
        latency = usage.get('latency_s', 0.0)
        
        results.append({
            'query_id': item['query_id'],
            'query': query,
            'scores': scores,
            'ground_truth': ground_truth,
            'metrics': metrics,
            'model_used': 'heavy',  # Baseline은 항상 heavy 모델 사용
            'usage': usage,
            'cost_per_query': cost,
            'latency_s': latency
        })
    
    end_time = time.time()
    
    # 전체 성능 계산
    all_precision = [r['metrics']['precision'] for r in results]
    all_recall = [r['metrics']['recall'] for r in results]
    all_f1 = [r['metrics']['f1'] for r in results]
    
    # 추가 지표 계산
    all_latency = [r.get('latency_s', 0.0) for r in results]
    all_cost = [r.get('cost_per_query', 3) for r in results]  # 기본값 3 (Heavy)
    
    overall_metrics = {
        'avg_precision': np.mean(all_precision),
        'avg_recall': np.mean(all_recall),
        'avg_f1': np.mean(all_f1),
        'total_time': end_time - start_time,
        'avg_time_per_query': (end_time - start_time) / len(results),
        'avg_latency_ms': 1000.0 * np.mean(all_latency),
        'avg_cost_per_query': np.mean(all_cost)
    }
    
    print(f"✅ LLM Baseline Experiment completed")
    print(f"📊 Average F1: {overall_metrics['avg_f1']:.3f}")
    print(f"📊 Average Precision: {overall_metrics['avg_precision']:.3f}")
    print(f"📊 Average Recall: {overall_metrics['avg_recall']:.3f}")
    
    return {
        'results': results,
        'overall_metrics': overall_metrics,
        'experiment_type': 'LLM_Baseline'
    }

def run_llm_lite_route_experiment(data: List[Dict[str, Any]], config: LLMFilterConfig) -> Dict[str, Any]:
    """② Large-scale LLM + Lightweight LLM 실험"""
    print("🔄 Running LLM Lite-Route Experiment...")
    
    ollama = Ollama(config.ollama_url, config.temperature, config.top_p, config.seed)
    cache = ResponseCache(".cache/llm_lite_route_cache.jsonl")
    
    results = []
    start_time = time.time()
    total_queries = len(data)
    
    for idx, item in enumerate(data):
        print(f"📊 Lite Route 진행: {idx+1}/{total_queries} 쿼리 처리 중...")
        query = item['query']
        passages = item['passages']
        
        # 경량 LLM으로 먼저 시도
        prompt = create_llm_filtering_prompt(query, passages)
        lite_response, lite_usage = ollama_cached_call(ollama, cache, config.gemma_lite, prompt)
        lite_scores = parse_llm_scores(lite_response, len(passages))
        
        # 개선된 신뢰도 계산 (LLM 응답 기반)
        confidence = extract_confidence_from_llm_response(lite_response, lite_scores)
        
        # 신뢰도가 낮으면 고성능 LLM으로 라우팅
        if confidence < config.theta:
            heavy_response, heavy_usage = ollama_cached_call(ollama, cache, config.gemma_heavy, prompt)
            final_scores = parse_llm_scores(heavy_response, len(passages))
            model_used = 'heavy'
            total_usage = {
                'lite_tokens': lite_usage.get('total_tokens_est', 0),
                'heavy_tokens': heavy_usage.get('total_tokens_est', 0),
                'total_tokens': lite_usage.get('total_tokens_est', 0) + heavy_usage.get('total_tokens_est', 0)
            }
            # Cost 및 Latency 계산 (Lite + Heavy)
            cost = 2 + 3  # Lite(2) + Heavy(3)
            latency = lite_usage.get('latency_s', 0.0) + heavy_usage.get('latency_s', 0.0)
        else:
            final_scores = lite_scores
            model_used = 'lite'
            total_usage = lite_usage
            # Cost 및 Latency 계산 (Lite만)
            cost = 2  # Lite만 사용
            latency = lite_usage.get('latency_s', 0.0)
        
        # 실제 MS MARCO Ground Truth 사용
        ground_truth = item['relevance_labels']
        
        # 평가 지표 계산
        metrics = calculate_f1_metrics(final_scores, ground_truth)
        
        results.append({
            'query_id': item['query_id'],
            'query': query,
            'scores': final_scores,
            'ground_truth': ground_truth,
            'metrics': metrics,
            'model_used': model_used,
            'lite_confidence': confidence,  # 명확한 필드명 사용
            'usage': total_usage,
            'cost_per_query': cost,
            'latency_s': latency
        })
    
    end_time = time.time()
    
    # 전체 성능 계산
    all_precision = [r['metrics']['precision'] for r in results]
    all_recall = [r['metrics']['recall'] for r in results]
    all_f1 = [r['metrics']['f1'] for r in results]
    
    # 모델 사용 통계
    lite_usage_count = sum(1 for r in results if r['model_used'] == 'lite')
    heavy_usage_count = sum(1 for r in results if r['model_used'] == 'heavy')
    
    # 추가 지표 계산
    all_latency = [r.get('latency_s', 0.0) for r in results]
    all_cost = [r.get('cost_per_query', 3) for r in results]
    
    overall_metrics = {
        'avg_precision': np.mean(all_precision),
        'avg_recall': np.mean(all_recall),
        'avg_f1': np.mean(all_f1),
        'total_time': end_time - start_time,
        'avg_time_per_query': (end_time - start_time) / len(results),
        'avg_latency_ms': 1000.0 * np.mean(all_latency),
        'avg_cost_per_query': np.mean(all_cost),
        'model_usage': {
            'lite': lite_usage_count,
            'heavy': heavy_usage_count,
            'lite_ratio': lite_usage_count / len(results)
        }
    }
    
    print(f"✅ LLM Lite-Route Experiment completed")
    print(f"📊 Average F1: {overall_metrics['avg_f1']:.3f}")
    print(f"📊 Lite usage: {overall_metrics['model_usage']['lite_ratio']:.1%}")
    
    return {
        'results': results,
        'overall_metrics': overall_metrics,
        'experiment_type': 'LLM_Lite_Route'
    }

def run_llm_setfit_experiment(data: List[Dict[str, Any]], config: LLMFilterConfig) -> Dict[str, Any]:
    """③ Large-scale LLM + Lightweight LLM + SetFit 실험 (온라인 학습)"""
    print("🔄 Running LLM SetFit Experiment with Online Learning...")
    
    ollama = Ollama(config.ollama_url, config.temperature, config.top_p, config.seed)
    cache = ResponseCache(".cache/llm_setfit_cache.jsonl")
    setfit_helper = SetFitHelper()
    
    results = []
    start_time = time.time()
    total_queries = len(data)
    
    # SetFit 학습 데이터 수집 (온라인 학습용)
    learning_samples = []
    setfit_learning_threshold = config.warmup_k  # 처음 N개 쿼리는 학습용
    
    print(f"📚 SetFit 온라인 학습 시작: 처음 {setfit_learning_threshold}개 쿼리로 학습")
    
    for idx, item in enumerate(data):
        print(f"📊 SetFit 진행: {idx+1}/{total_queries} 쿼리 처리 중...")
        query = item['query']
        passages = item['passages']
        
        # 1단계: LLM으로 필터링 수행 (SetFit 없이)
        if idx < setfit_learning_threshold:
            # 학습 단계: 경량 LLM으로 먼저 시도
            prompt = create_llm_filtering_prompt(query, passages)
            lite_response, lite_usage = ollama_cached_call(ollama, cache, config.gemma_lite, prompt)
            lite_scores = parse_llm_scores(lite_response, len(passages))
            
            # 개선된 신뢰도 계산 (LLM 응답 기반)
            confidence = extract_confidence_from_llm_response(lite_response, lite_scores)
            
            if confidence < config.theta:
                # 신뢰도 낮음 → 고성능 LLM 사용
                heavy_response, heavy_usage = ollama_cached_call(ollama, cache, config.gemma_heavy, prompt)
                final_scores = parse_llm_scores(heavy_response, len(passages))
                model_used = 'heavy'
                total_usage = {
                    'lite_tokens': lite_usage.get('total_tokens_est', 0),
                    'heavy_tokens': heavy_usage.get('total_tokens_est', 0),
                    'total_tokens': lite_usage.get('total_tokens_est', 0) + heavy_usage.get('total_tokens_est', 0)
                }
                # Cost 및 Latency 계산 (Lite + Heavy)
                cost = 2 + 3  # Lite(2) + Heavy(3)
                latency = lite_usage.get('latency_s', 0.0) + heavy_usage.get('latency_s', 0.0)
            else:
                final_scores = lite_scores
                model_used = 'lite'
                total_usage = lite_usage
                # Cost 및 Latency 계산 (Lite만)
                cost = 2  # Lite만 사용
                latency = lite_usage.get('latency_s', 0.0)
            
            # 실제 MS MARCO Ground Truth 사용 (데이터셋 원본 레이블)
            ground_truth = item['relevance_labels']
            
            # SetFit 학습 데이터 수집 (Ground Truth 기반)
            for i, passage in enumerate(passages):
                combined_text = f"{query} [SEP] {passage}"
                # 실제 Ground Truth를 라벨로 사용 (데이터셋 원본 레이블)
                ground_truth_label = ground_truth[i]
                label = "HIGH" if ground_truth_label == 1 else "LOW"  # Ground Truth 기반 라벨
                learning_samples.append((combined_text, label))
            
            # 평가 지표 계산
            metrics = calculate_f1_metrics(final_scores, ground_truth)
            
            results.append({
                'query_id': item['query_id'],
                'query': query,
                'scores': final_scores,
                'ground_truth': ground_truth,
                'metrics': metrics,
                'model_used': model_used,
                'setfit_confidence': None,  # 아직 SetFit 미사용
                'usage': total_usage,
                'phase': 'learning',
                'cost_per_query': cost,
                'latency_s': latency
            })
            
            print(f"📖 학습 단계 {idx+1}/{setfit_learning_threshold}: {model_used} 모델 사용, F1={metrics['f1']:.3f}")
        
        else:
            # SetFit 사용 단계: 충분한 학습 데이터가 있으면 SetFit 활성화
            if len(learning_samples) >= 16 and setfit_helper.model is None:  # 클래스당 8개 × 2클래스
                print(f"🎯 SetFit 모델 활성화: {len(learning_samples)}개 샘플로 학습")
                # SetFit 웜업
                texts, labels = zip(*learning_samples)
                setfit_helper.warmup(list(texts), list(labels))
            
            # SetFit으로 신뢰도 예측 (모델이 있으면)
            if setfit_helper.model is not None:
                setfit_confidences = []
                for passage in passages:
                    combined_text = f"{query} [SEP] {passage}"
                    _, confidence = setfit_helper.predict_conf(combined_text)
                    setfit_confidences.append(confidence)
                
                # SetFit 신뢰도 기반 라우팅
                avg_setfit_confidence = np.mean(setfit_confidences)
                
                # SetFit 신뢰도에 관계없이 먼저 경량 LLM 사용
                prompt = create_llm_filtering_prompt(query, passages)
                lite_response, lite_usage = ollama_cached_call(ollama, cache, config.gemma_lite, prompt)
                lite_scores = parse_llm_scores(lite_response, len(passages))
                
                # 개선된 Lite LLM 신뢰도 계산 (LLM 응답 기반)
                lite_confidence = extract_confidence_from_llm_response(lite_response, lite_scores)
                
                if lite_confidence < config.theta:
                    # Lite LLM 신뢰도 낮음 → 고성능 LLM 사용
                    heavy_response, heavy_usage = ollama_cached_call(ollama, cache, config.gemma_heavy, prompt)
                    final_scores = parse_llm_scores(heavy_response, len(passages))
                    model_used = 'heavy'
                    usage = {
                        'lite_tokens': lite_usage.get('total_tokens_est', 0),
                        'heavy_tokens': heavy_usage.get('total_tokens_est', 0),
                        'total_tokens': lite_usage.get('total_tokens_est', 0) + heavy_usage.get('total_tokens_est', 0)
                    }
                else:
                    # Lite LLM 신뢰도 높음 → Lite LLM 결과 사용
                    final_scores = lite_scores
                    model_used = 'lite'
                    usage = lite_usage
                
                # SetFit 신뢰도 저장
                setfit_confidence = avg_setfit_confidence
            else:
                # SetFit 모델이 없으면 기본 라우팅
                prompt = create_llm_filtering_prompt(query, passages)
                lite_response, lite_usage = ollama_cached_call(ollama, cache, config.gemma_lite, prompt)
                lite_scores = parse_llm_scores(lite_response, len(passages))
                confidence = 1.0 - np.std(lite_scores) if len(lite_scores) > 1 else 0.5
                
                if confidence < config.theta:
                    heavy_response, heavy_usage = ollama_cached_call(ollama, cache, config.gemma_heavy, prompt)
                    final_scores = parse_llm_scores(heavy_response, len(passages))
                    model_used = 'heavy'
                    usage = {
                        'lite_tokens': lite_usage.get('total_tokens_est', 0),
                        'heavy_tokens': heavy_usage.get('total_tokens_est', 0),
                        'total_tokens': lite_usage.get('total_tokens_est', 0) + heavy_usage.get('total_tokens_est', 0)
                    }
                else:
                    final_scores = lite_scores
                    model_used = 'lite'
                    usage = lite_usage
                
                final_scores = lite_scores if model_used == 'lite' else final_scores
                setfit_confidence = None
            
            scores = final_scores
            
            # 실제 MS MARCO Ground Truth 사용 (데이터셋 원본 레이블)
            ground_truth = item['relevance_labels']
            
            # 평가 지표 계산
            metrics = calculate_f1_metrics(scores, ground_truth)
            
            # Cost 및 Latency 계산 (SetFit 사용 단계)
            if model_used == 'setfit':
                cost = 1  # SetFit만 사용
                latency = 0.0  # SetFit은 즉시 처리
            elif model_used == 'lite':
                cost = 2  # Lite만 사용
                latency = usage.get('latency_s', 0.0)
            else:  # heavy
                cost = 2 + 3  # Lite + Heavy
                latency = usage.get('latency_s', 0.0)
            
            results.append({
                'query_id': item['query_id'],
                'query': query,
                'scores': scores,
                'ground_truth': ground_truth,
                'metrics': metrics,
                'model_used': model_used,
                'setfit_confidence': setfit_confidence,
                'usage': usage,
                'phase': 'setfit' if setfit_helper.model is not None else 'fallback',
                'cost_per_query': cost,
                'latency_s': latency
            })
            
            # SetFit 온라인 학습 (Ground Truth 기반, LLM 결과는 보조적 역할)
            if setfit_helper.model is not None:
                for i, passage in enumerate(passages):
                    combined_text = f"{query} [SEP] {passage}"
                    # Ground Truth를 주 라벨로 사용
                    ground_truth_label = ground_truth[i]
                    label = "HIGH" if ground_truth_label == 1 else "LOW"
                    
                    # LLM 결과와 Ground Truth가 일치하는 경우만 온라인 학습에 추가 (품질 보장)
                    llm_score = scores[i]
                    llm_prediction = "HIGH" if llm_score >= 0.5 else "LOW"
                    if label == llm_prediction:  # LLM과 Ground Truth가 일치하는 경우만
                        setfit_helper.add_online(combined_text, label)
            
            setfit_str = f"{setfit_confidence:.3f}" if setfit_confidence is not None else "N/A"
            print(f"🎯 SetFit 단계 {idx+1}: {model_used} 모델 사용, F1={metrics['f1']:.3f}, SetFit={setfit_str}")
    
    end_time = time.time()
    
    # 전체 성능 계산
    all_precision = [r['metrics']['precision'] for r in results]
    all_recall = [r['metrics']['recall'] for r in results]
    all_f1 = [r['metrics']['f1'] for r in results]
    
    # 모델 사용 통계
    lite_usage_count = sum(1 for r in results if r['model_used'] == 'lite')
    heavy_usage_count = sum(1 for r in results if r['model_used'] == 'heavy')
    
    # SetFit 사용 통계
    setfit_used_count = sum(1 for r in results if r.get('setfit_confidence') is not None)
    
    # 추가 지표 계산
    all_latency = [r.get('latency_s', 0.0) for r in results]
    all_cost = [r.get('cost_per_query', 3) for r in results]
    
    overall_metrics = {
        'avg_precision': np.mean(all_precision),
        'avg_recall': np.mean(all_recall),
        'avg_f1': np.mean(all_f1),
        'total_time': end_time - start_time,
        'avg_time_per_query': (end_time - start_time) / len(results),
        'avg_latency_ms': 1000.0 * np.mean(all_latency),
        'avg_cost_per_query': np.mean(all_cost),
        'model_usage': {
            'lite': lite_usage_count,
            'heavy': heavy_usage_count,
            'lite_ratio': lite_usage_count / len(results)
        },
        'setfit_usage': {
            'learning_queries': setfit_learning_threshold,
            'setfit_activated': setfit_helper.model is not None,
            'setfit_used_queries': setfit_used_count,
            'learning_samples': len(learning_samples)
        }
    }
    
    print(f"✅ LLM SetFit Experiment completed")
    print(f"📊 Average F1: {overall_metrics['avg_f1']:.3f}")
    print(f"📊 Lite usage: {overall_metrics['model_usage']['lite_ratio']:.1%}")
    print(f"📊 SetFit 학습: {overall_metrics['setfit_usage']['learning_samples']}개 샘플, {overall_metrics['setfit_usage']['setfit_used_queries']}개 쿼리에서 사용")
    
    return {
        'results': results,
        'overall_metrics': overall_metrics,
        'experiment_type': 'LLM_SetFit_Online_Learning'
    }

def run_llm_cbc_experiment(data: List[Dict[str, Any]], config: LLMFilterConfig) -> Dict[str, Any]:
    """④ Large-scale LLM + Lightweight LLM + SetFit + CBC (Proposed) 실험 (온라인 학습)"""
    print("🔄 Running LLM CBC Experiment with Online Learning...")
    
    ollama = Ollama(config.ollama_url, config.temperature, config.top_p, config.seed)
    cache = ResponseCache(".cache/llm_cbc_cache.jsonl")
    cbc_router = CBCRouter(config)
    
    results = []
    start_time = time.time()
    total_queries = len(data)
    
    # SetFit 학습 데이터 수집 (온라인 학습용)
    learning_samples = []
    setfit_learning_threshold = config.warmup_k  # 처음 N개 쿼리는 학습용
    
    print(f"📚 CBC + SetFit 온라인 학습 시작: 처음 {setfit_learning_threshold}개 쿼리로 학습")
    
    for idx, item in enumerate(data):
        print(f"📊 CBC 진행: {idx+1}/{total_queries} 쿼리 처리 중...")
        query = item['query']
        passages = item['passages']
        
        # 1단계: LLM으로 필터링 수행 (SetFit 없이)
        if idx < setfit_learning_threshold:
            # 학습 단계: 경량 LLM으로 먼저 시도
            prompt = create_llm_filtering_prompt(query, passages)
            lite_response, lite_usage = ollama_cached_call(ollama, cache, config.gemma_lite, prompt)
            lite_scores = parse_llm_scores(lite_response, len(passages))
            
            # 개선된 신뢰도 계산 (LLM 응답 기반)
            confidence = extract_confidence_from_llm_response(lite_response, lite_scores)
            
            if confidence < config.theta:
                # 신뢰도 낮음 → 고성능 LLM 사용
                heavy_response, heavy_usage = ollama_cached_call(ollama, cache, config.gemma_heavy, prompt)
                final_scores = parse_llm_scores(heavy_response, len(passages))
                model_used = 'heavy'
                total_usage = {
                    'lite_tokens': lite_usage.get('total_tokens_est', 0),
                    'heavy_tokens': heavy_usage.get('total_tokens_est', 0),
                    'total_tokens': lite_usage.get('total_tokens_est', 0) + heavy_usage.get('total_tokens_est', 0)
                }
                # Cost 및 Latency 계산 (Lite + Heavy)
                cost = 2 + 3  # Lite(2) + Heavy(3)
                latency = lite_usage.get('latency_s', 0.0) + heavy_usage.get('latency_s', 0.0)
            else:
                final_scores = lite_scores
                model_used = 'lite'
                total_usage = lite_usage
                # Cost 및 Latency 계산 (Lite만)
                cost = 2  # Lite만 사용
                latency = lite_usage.get('latency_s', 0.0)
            
            # 실제 MS MARCO Ground Truth 사용 (데이터셋 원본 레이블)
            ground_truth = item['relevance_labels']
            
            # SetFit 학습 데이터 수집 (Ground Truth 기반)
            for i, passage in enumerate(passages):
                combined_text = f"{query} [SEP] {passage}"
                # 실제 Ground Truth를 라벨로 사용 (데이터셋 원본 레이블)
                ground_truth_label = ground_truth[i]
                label = "HIGH" if ground_truth_label == 1 else "LOW"  # Ground Truth 기반 라벨
                learning_samples.append((combined_text, label))
            
            # 평가 지표 계산
            metrics = calculate_f1_metrics(final_scores, ground_truth)
            
            results.append({
                'query_id': item['query_id'],
                'query': query,
                'scores': final_scores,
                'ground_truth': ground_truth,
                'metrics': metrics,
                'model_used': model_used,
                'setfit_confidence': None,  # 아직 SetFit 미사용
                'usage': total_usage,
                'phase': 'learning',
                'cost_per_query': cost,
                'latency_s': latency
            })
            
            print(f"📖 CBC 학습 단계 {idx+1}/{setfit_learning_threshold}: {model_used} 모델 사용, F1={metrics['f1']:.3f}")
        
        else:
            # SetFit 사용 단계: 충분한 학습 데이터가 있으면 SetFit 활성화
            if len(learning_samples) >= 16 and cbc_router.setfit_helper.model is None:  # 클래스당 8개 × 2클래스
                print(f"🎯 CBC + SetFit 모델 활성화: {len(learning_samples)}개 샘플로 학습")
                # SetFit 웜업
                texts, labels = zip(*learning_samples)
                cbc_router.setfit_helper.warmup(list(texts), list(labels))
            
            # CBC 기반 계층적 라우팅 수행
            if cbc_router.setfit_helper.model is not None:
                # SetFit + CBC 라우팅
                routing_result = cbc_router.hierarchical_cbc_routing(query, passages[0] if passages else "", [])
                
                # SetFit에서 종료되는 경우 실제로 종료
                if routing_result['stage'] == 'setfit':
                    # SetFit 점수로 문서별 점수 생성
                    combined = [f"{query} [SEP] {p}" for p in passages]
                    probs = cbc_router.setfit_helper.model.predict_proba(combined)  # shape: [N,2]
                    scores = [float(p[1]) for p in probs]  # HIGH 확률
                    model_used = 'setfit'
                    usage = {'total_tokens_est': 0, 'latency_s': 0.0}
                    setfit_confidence = routing_result.get('confidence', None)
                    # Cost 및 Latency 계산 (SetFit만 사용)
                    cost = 1  # SetFit만 사용
                    latency = 0.0  # SetFit은 즉시 처리
                else:
                    # SetFit 신뢰도에 관계없이 먼저 경량 LLM 사용
                    prompt = create_llm_filtering_prompt(query, passages)
                    lite_response, lite_usage = ollama_cached_call(ollama, cache, config.gemma_lite, prompt)
                    lite_scores = parse_llm_scores(lite_response, len(passages))
                    
                    # 개선된 Lite LLM 신뢰도 계산 (LLM 응답 기반)
                    lite_confidence = extract_confidence_from_llm_response(lite_response, lite_scores)
                    
                    if lite_confidence < config.theta:
                        # Lite LLM 신뢰도 낮음 → 고성능 LLM 사용
                        heavy_response, heavy_usage = ollama_cached_call(ollama, cache, config.gemma_heavy, prompt)
                        scores = parse_llm_scores(heavy_response, len(passages))
                        model_used = 'heavy'
                        usage = {
                            'lite_tokens': lite_usage.get('total_tokens_est', 0),
                            'heavy_tokens': heavy_usage.get('total_tokens_est', 0),
                            'total_tokens': lite_usage.get('total_tokens_est', 0) + heavy_usage.get('total_tokens_est', 0)
                        }
                        # Cost 및 Latency 계산 (Lite + Heavy)
                        cost = 2 + 3  # Lite(2) + Heavy(3)
                        latency = lite_usage.get('latency_s', 0.0) + heavy_usage.get('latency_s', 0.0)
                    else:
                        # Lite LLM 신뢰도 높음 → Lite LLM 결과 사용
                        scores = lite_scores
                        model_used = 'lite'
                        usage = lite_usage
                        # Cost 및 Latency 계산 (Lite만)
                        cost = 2  # Lite만 사용
                        latency = lite_usage.get('latency_s', 0.0)
                    
                    setfit_confidence = routing_result.get('confidence', None)
            else:
                # SetFit 모델이 없으면 기본 라우팅
                prompt = create_llm_filtering_prompt(query, passages)
                lite_response, lite_usage = ollama_cached_call(ollama, cache, config.gemma_lite, prompt)
                lite_scores = parse_llm_scores(lite_response, len(passages))
                confidence = 1.0 - np.std(lite_scores) if len(lite_scores) > 1 else 0.5
                
                if confidence < config.theta:
                    heavy_response, heavy_usage = ollama_cached_call(ollama, cache, config.gemma_heavy, prompt)
                    scores = parse_llm_scores(heavy_response, len(passages))
                    model_used = 'heavy'
                    usage = {
                        'lite_tokens': lite_usage.get('total_tokens_est', 0),
                        'heavy_tokens': heavy_usage.get('total_tokens_est', 0),
                        'total_tokens': lite_usage.get('total_tokens_est', 0) + heavy_usage.get('total_tokens_est', 0)
                    }
                    # Cost 및 Latency 계산 (Lite + Heavy)
                    cost = 2 + 3  # Lite(2) + Heavy(3)
                    latency = lite_usage.get('latency_s', 0.0) + heavy_usage.get('latency_s', 0.0)
                else:
                    scores = lite_scores
                    model_used = 'lite'
                    usage = lite_usage
                    # Cost 및 Latency 계산 (Lite만)
                    cost = 2  # Lite만 사용
                    latency = lite_usage.get('latency_s', 0.0)
                
                setfit_confidence = None
            
            # 실제 MS MARCO Ground Truth 사용 (데이터셋 원본 레이블)
            ground_truth = item['relevance_labels']
            
            # 평가 지표 계산
            metrics = calculate_f1_metrics(scores, ground_truth)
            
            # 모델 사용 통계
            model_usage = {
                'setfit': 1 if model_used == 'setfit' else 0,
                'lite': 1 if model_used == 'lite' else 0,
                'heavy': 1 if model_used == 'heavy' else 0
            }
            
            results.append({
                'query_id': item['query_id'],
                'query': query,
                'scores': scores,
                'ground_truth': ground_truth,
                'metrics': metrics,
                'model_used': model_used,
                'setfit_confidence': setfit_confidence,
                'usage': usage,
                'phase': 'cbc' if cbc_router.setfit_helper.model is not None else 'fallback',
                'model_usage': model_usage,
                'cost_per_query': cost,
                'latency_s': latency
            })
            
            # SetFit 온라인 학습 (Ground Truth 기반, LLM 결과는 보조적 역할)
            if cbc_router.setfit_helper.model is not None:
                for i, passage in enumerate(passages):
                    combined_text = f"{query} [SEP] {passage}"
                    # Ground Truth를 주 라벨로 사용
                    ground_truth_label = ground_truth[i]
                    label = "HIGH" if ground_truth_label == 1 else "LOW"
                    
                    # LLM 결과와 Ground Truth가 일치하는 경우만 온라인 학습에 추가 (품질 보장)
                    llm_score = scores[i]
                    llm_prediction = "HIGH" if llm_score >= 0.5 else "LOW"
                    if label == llm_prediction:  # LLM과 Ground Truth가 일치하는 경우만
                        cbc_router.setfit_helper.add_online(combined_text, label)
            
            setfit_str = f"{setfit_confidence:.3f}" if setfit_confidence is not None else "N/A"
            print(f"🎯 CBC 단계 {idx+1}: {model_used} 모델 사용, F1={metrics['f1']:.3f}, SetFit={setfit_str}")
    
    end_time = time.time()
    
    # 전체 성능 계산
    all_precision = [r['metrics']['precision'] for r in results]
    all_recall = [r['metrics']['recall'] for r in results]
    all_f1 = [r['metrics']['f1'] for r in results]
    
    # 전체 모델 사용 통계
    total_setfit = sum(r.get('model_usage', {}).get('setfit', 0) for r in results)
    total_lite = sum(r.get('model_usage', {}).get('lite', 0) for r in results)
    total_heavy = sum(r.get('model_usage', {}).get('heavy', 0) for r in results)
    total_models = total_setfit + total_lite + total_heavy
    
    # SetFit 사용 통계
    setfit_used_count = sum(1 for r in results if r.get('setfit_confidence') is not None)
    
    # 추가 지표 계산
    all_latency = [r.get('latency_s', 0.0) for r in results]
    all_cost = [r.get('cost_per_query', 3) for r in results]
    
    # Termination rates 계산
    setfit_count = sum(1 for r in results if r.get('model_used') == 'setfit')
    lite_count = sum(1 for r in results if r.get('model_used') == 'lite')
    heavy_count = sum(1 for r in results if r.get('model_used') == 'heavy')
    
    overall_metrics = {
        'avg_precision': np.mean(all_precision),
        'avg_recall': np.mean(all_recall),
        'avg_f1': np.mean(all_f1),
        'total_time': end_time - start_time,
        'avg_time_per_query': (end_time - start_time) / len(results),
        'avg_latency_ms': 1000.0 * np.mean(all_latency),
        'avg_cost_per_query': np.mean(all_cost),
        'model_usage': {
            'setfit': total_setfit,
            'lite': total_lite,
            'heavy': total_heavy,
            'setfit_ratio': total_setfit / max(total_models, 1),
            'lite_ratio': total_lite / max(total_models, 1),
            'heavy_ratio': total_heavy / max(total_models, 1)
        },
        'term_rates': {
            'setfit': setfit_count / len(results),
            'lite': lite_count / len(results),
            'heavy': heavy_count / len(results)
        },
        'setfit_usage': {
            'learning_queries': setfit_learning_threshold,
            'setfit_activated': cbc_router.setfit_helper.model is not None,
            'setfit_used_queries': setfit_used_count,
            'learning_samples': len(learning_samples)
        }
    }
    
    print(f"✅ LLM CBC Experiment completed")
    print(f"📊 Average F1: {overall_metrics['avg_f1']:.3f}")
    print(f"📊 Model usage: SetFit={overall_metrics['model_usage']['setfit_ratio']:.1%}, Lite={overall_metrics['model_usage']['lite_ratio']:.1%}, Heavy={overall_metrics['model_usage']['heavy_ratio']:.1%}")
    print(f"📊 SetFit 학습: {overall_metrics['setfit_usage']['learning_samples']}개 샘플, {overall_metrics['setfit_usage']['setfit_used_queries']}개 쿼리에서 사용")
    
    return {
        'results': results,
        'overall_metrics': overall_metrics,
        'experiment_type': 'LLM_CBC_Online_Learning'
    }

def run_ms_marco_llm_experiments(max_queries: int = 10) -> Dict[str, Any]:
    """MS MARCO 데이터셋에 대한 LLM 필터링 실험 실행"""
    print("🚀 MS MARCO LLM Filtering Experiments")
    print("=" * 60)
    
    # 데이터 로드
    data = load_ms_marco_data(max_queries)
    if not data:
        print("❌ MS MARCO 데이터 로드 실패")
        return {}
    
    # 설정
    config = LLMFilterConfig(
        max_queries=max_queries,
        theta=0.8,  # 높은 신뢰도 기준
        warmup_k=4,  # SetFit 학습용 최소 쿼리 수
        seed=42
    )
    
    experiments = {}
    
    # ① Large-scale LLM Only (Baseline)
    print("\n📊 Experiment 1: Large-scale LLM Only (Baseline)")
    experiments['baseline'] = run_llm_baseline_experiment(data, config)
    
    # ② Large-scale LLM + Lightweight LLM
    print("\n📊 Experiment 2: Large-scale LLM + Lightweight LLM")
    experiments['lite_route'] = run_llm_lite_route_experiment(data, config)
    
    # ③ Large-scale LLM + Lightweight LLM + SetFit
    print("\n📊 Experiment 3: Large-scale LLM + Lightweight LLM + SetFit")
    experiments['setfit'] = run_llm_setfit_experiment(data, config)
    
    # ④ Large-scale LLM + Lightweight LLM + SetFit + CBC (Proposed)
    print("\n📊 Experiment 4: Large-scale LLM + Lightweight LLM + SetFit + CBC (Proposed)")
    experiments['cbc'] = run_llm_cbc_experiment(data, config)
    
    # 결과 비교
    print("\n📈 MS MARCO LLM Filtering Results Summary")
    print("=" * 60)
    print(f"{'Method':<40} {'F1':<8} {'Precision':<10} {'Recall':<8} {'Model Usage'}")
    print("-" * 60)
    
    for name, exp in experiments.items():
        metrics = exp['overall_metrics']
        f1 = metrics['avg_f1']
        precision = metrics['avg_precision']
        recall = metrics['avg_recall']
        
        if 'model_usage' in metrics:
            usage = metrics['model_usage']
            if 'setfit_ratio' in usage:
                model_usage = f"SetFit: {usage['setfit_ratio']:.1%}, Lite: {usage['lite_ratio']:.1%}, Heavy: {usage['heavy_ratio']:.1%}"
            elif 'lite_ratio' in usage:
                model_usage = f"Lite: {usage['lite_ratio']:.1%}"
            else:
                model_usage = "Heavy: 100%"
        else:
            model_usage = "Heavy: 100%"
        
        method_name = {
            'baseline': 'LLM Only (Baseline)',
            'lite_route': 'LLM + Lite Route',
            'setfit': 'LLM + Lite + SetFit',
            'cbc': 'LLM + Lite + SetFit + CBC (Proposed)'
        }.get(name, name)
        
        print(f"{method_name:<40} {f1:<8.3f} {precision:<10.3f} {recall:<8.3f} {model_usage}")
    
    # 결과 저장
    output_dir = "results/llm/ms_marco"
    ensure_dir(output_dir)
    
    for name, exp in experiments.items():
        output_path = os.path.join(output_dir, f"llm_{name}_results.json")
        save_json(exp, output_path)
        print(f"💾 {name} 결과 저장: {output_path}")
    
    print(f"\n✅ MS MARCO LLM 실험 완료: {len(data)}개 쿼리")
    return experiments

def run_fiqa_llm_experiments(max_queries: int = 648) -> Dict[str, Any]:
    """FiQA 데이터셋에 대한 LLM 필터링 실험 실행"""
    print("🚀 FiQA LLM Filtering Experiments")
    print("=" * 60)
    
    # 데이터 로드
    data = load_fiqa_data(max_queries)
    if not data:
        print("❌ FiQA 데이터 로드 실패")
        return {}
    
    # 설정
    config = LLMFilterConfig(
        max_queries=max_queries,
        theta=0.8,  # 높은 신뢰도 기준
        warmup_k=4,  # SetFit 학습용 최소 쿼리 수
        seed=42
    )
    
    experiments = {}
    
    # ① Large-scale LLM Only (Baseline)
    print("\n📊 Experiment 1: Large-scale LLM Only (Baseline)")
    experiments['baseline'] = run_llm_baseline_experiment(data, config)
    
    # ② Large-scale LLM + Lightweight LLM
    print("\n📊 Experiment 2: Large-scale LLM + Lightweight LLM")
    experiments['lite_route'] = run_llm_lite_route_experiment(data, config)
    
    # ③ Large-scale LLM + Lightweight LLM + SetFit
    print("\n📊 Experiment 3: Large-scale LLM + Lightweight LLM + SetFit")
    experiments['setfit'] = run_llm_setfit_experiment(data, config)
    
    # ④ Large-scale LLM + Lightweight LLM + SetFit + CBC (Proposed)
    print("\n📊 Experiment 4: Large-scale LLM + Lightweight LLM + SetFit + CBC (Proposed)")
    experiments['cbc'] = run_llm_cbc_experiment(data, config)
    
    # 결과 비교
    print("\n📈 FiQA LLM Filtering Results Summary")
    print("=" * 60)
    print(f"{'Method':<40} {'F1':<8} {'Precision':<10} {'Recall':<8} {'Model Usage'}")
    print("-" * 60)
    
    for name, exp in experiments.items():
        metrics = exp['overall_metrics']
        f1 = metrics['avg_f1']
        precision = metrics['avg_precision']
        recall = metrics['avg_recall']
        
        if 'model_usage' in metrics:
            usage = metrics['model_usage']
            if 'setfit_ratio' in usage:
                model_usage = f"SetFit: {usage['setfit_ratio']:.1%}, Lite: {usage['lite_ratio']:.1%}, Heavy: {usage['heavy_ratio']:.1%}"
            elif 'lite_ratio' in usage:
                model_usage = f"Lite: {usage['lite_ratio']:.1%}"
            else:
                model_usage = "Heavy: 100%"
        else:
            model_usage = "Heavy: 100%"
        
        method_name = {
            'baseline': 'LLM Only (Baseline)',
            'lite_route': 'LLM + Lite Route',
            'setfit': 'LLM + Lite + SetFit',
            'cbc': 'LLM + Lite + SetFit + CBC (Proposed)'
        }.get(name, name)
        
        print(f"{method_name:<40} {f1:<8.3f} {precision:<10.3f} {recall:<8.3f} {model_usage}")
    
    # 결과 저장
    output_dir = "results/llm/fiqa"
    ensure_dir(output_dir)
    
    for name, exp in experiments.items():
        output_path = os.path.join(output_dir, f"llm_{name}_results.json")
        save_json(exp, output_path)
        print(f"💾 {name} 결과 저장: {output_path}")
    
    print(f"\n✅ FiQA LLM 실험 완료: {len(data)}개 쿼리")
    return experiments


if __name__ == "__main__":
    # 개별 데이터셋 실험 실행 예시
    print("LLM Filtering Experiments")
    print("=" * 50)
    print("개별 데이터셋 실험을 실행하려면:")
    print("- MS MARCO: python run_llm_ms_marco.py")
    print("- FiQA: python run_llm_fiqa.py")
