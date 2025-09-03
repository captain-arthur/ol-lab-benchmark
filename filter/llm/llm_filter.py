#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
llm_filter.py
- LLM 필터링을 위한 공통 모듈
- Ollama 클라이언트, 캐시, 검증기, SetFit 헬퍼 등 공통 기능 제공
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
        self.theta = kwargs.get("theta", 0.65)
        self.warmup_k = kwargs.get("warmup_k", 10)
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
def pearson_spearman(preds: List[float], golds: List[float]) -> Dict[str, float]:
    """Pearson과 Spearman 상관계수 계산"""
    try:
        from scipy.stats import pearsonr, spearmanr
        
        if len(preds) != len(golds) or len(preds) < 2:
            return {"pearson": 0.0, "spearman": 0.0}
        
        pred_vals = [float(p) if isinstance(p, str) else p for p in preds]
        gold_vals = [float(g) if isinstance(g, str) else g for g in golds]
        
        pearson_corr, _ = pearsonr(pred_vals, gold_vals)
        spearman_corr, _ = spearmanr(pred_vals, gold_vals)
        
        return {
            "pearson": float(pearson_corr) if not np.isnan(pearson_corr) else 0.0,
            "spearman": float(spearman_corr) if not np.isnan(spearman_corr) else 0.0
        }
    except Exception as e:
        print(f"상관계수 계산 실패: {e}")
        return {"pearson": 0.0, "spearman": 0.0}

def expected_calibration_error(confs: List[float], corrects: List[int], n_bins=10) -> float:
    """Expected Calibration Error"""
    if not confs or np is None:
        return 0.0
    bins = np.linspace(0, 1, n_bins+1)
    ece = 0.0
    confs = np.array(confs)
    corrects = np.array(corrects)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        m = (confs >= lo) & (confs < hi if i < n_bins-1 else confs <= hi)
        if not m.any(): 
            continue
        acc = corrects[m].mean()
        conf_avg = confs[m].mean()
        ece += (m.mean()) * abs(acc - conf_avg)
    return float(ece)

def routing_quality(decisions: List[bool], needed_heavy: List[bool]) -> Dict[str,float]:
    """라우팅 품질 평가"""
    if len(decisions) != len(needed_heavy) or not decisions:
        return {"routing_accuracy":0.0,"fp_rate":0.0,"fn_rate":0.0,"precision":0.0,"recall":0.0}
    
    TP = sum(1 for d,need in zip(decisions, needed_heavy) if d and need)
    TN = sum(1 for d,need in zip(decisions, needed_heavy) if (not d) and (not need))
    FP = sum(1 for d,need in zip(decisions, needed_heavy) if d and (not need))
    FN = sum(1 for d,need in zip(decisions, needed_heavy) if (not d) and need)
    
    total = len(decisions)
    prec = TP / max(1, TP+FP)
    rec  = TP / max(1, TP+FN)
    
    return {
        "routing_accuracy": (TP+TN)/total,
        "fp_rate": FP/total,
        "fn_rate": FN/total,
        "precision": prec,
        "recall": rec
    }

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
    
    def hierarchical_cbc_routing(self, query: str, text: str, anchors: List[str] = None) -> Dict[str, Any]:
        """계층적 CBC 라우팅 (Phase 2)"""
        try:
            # 1단계: SetFit + 앵커 (비용 최소)
            setfit_score = self.calculate_anchor_based_confidence(query, text, anchors)
            
            if self.cbc_decision(setfit_score, 80):  # 상위 80%
                return {
                    'stage': 'setfit',
                    'model_used': 'setfit',
                    'confidence': setfit_score,
                    'decision': 'accept',
                    'reason': 'SetFit + Anchor confidence high enough'
                }
            
            # 2단계: 경량 LLM (중간 비용)
            lite_result = self._predict_with_model(self.config.gemma_lite, query, text, {'name': 'generic'})
            lite_score = lite_result['confidence']
            
            if self.cbc_decision(lite_score, 90):  # 상위 90%
                return {
                    'stage': 'lite',
                    'model_used': 'lite',
                    'confidence': lite_score,
                    'decision': 'accept',
                    'reason': 'Lite LLM confidence high enough'
                }
            
            # 3단계: 고성능 LLM (최고 비용, 정말 필요한 경우만)
            heavy_result = self._predict_with_model(self.config.gemma_heavy, query, text, {'name': 'generic'})
            heavy_score = heavy_result['confidence']
            
            return {
                'stage': 'heavy',
                'model_used': 'heavy',
                'confidence': heavy_score,
                'decision': 'accept',
                'reason': 'Heavy LLM required for low confidence cases'
            }
            
        except Exception as e:
            print(f"⚠️ Hierarchical CBC routing failed: {e}")
            return {
                'stage': 'fallback',
                'model_used': 'heavy',
                'confidence': 0.5,
                'decision': 'accept',
                'reason': 'Fallback due to error'
            }
    
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
    
    def process_batch(self, batch_data: List[Tuple[str, str, Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """배치 단위로 처리하여 CBC 기반 라우팅 수행"""
        results = []
        
        # 1단계: 모든 샘플에 대해 점수만 계산 (라우팅 없음)
        for query, text, meta in batch_data:
            scores = self._calculate_all_scores(query, text, meta)
            self.batch_scores['setfit'].append(scores['setfit'])
            self.batch_scores['lite'].append(scores['lite'])
            self.batch_scores['heavy'].append(scores['heavy'])
            results.append({
                'query': query,
                'text': text,
                'meta': meta,
                'scores': scores,
                'final_decision': None
            })
        
        # 2단계: 점수 분포 분석하여 동적 임계값 계산
        self._calculate_dynamic_thresholds()
        
        # 3단계: 동적 임계값 기반으로 최종 라우팅 결정
        for i, result in enumerate(results):
            result['final_decision'] = self._make_routing_decision(
                result['scores'], i
            )
        
        return results
    
    def process_batch_cbc_with_anchors(self, batch_data: List[Tuple[str, str, Dict[str, Any]]], anchors_dict: Dict[str, List[str]] = None) -> List[Dict[str, Any]]:
        """CBC + 앵커 기반 배치 처리"""
        results = []
        
        print(f"🔄 CBC: Processing {len(batch_data)} samples with anchor-based confidence...")
        
        # 1단계: 모든 샘플의 신뢰도 점수 수집
        for query, text, meta in batch_data:
            anchors = anchors_dict.get(query, []) if anchors_dict else []
            
            # 앵커 기반 신뢰도 계산
            confidence_score = self.calculate_anchor_based_confidence(query, text, anchors)
            
            self.batch_scores['setfit'].append(confidence_score)
            
            results.append({
                'query': query,
                'text': text,
                'meta': meta,
                'anchors': anchors,
                'setfit_confidence': confidence_score,
                'routing_decision': None
            })
        
        # 2단계: 동적 임계값 계산
        self._calculate_dynamic_thresholds()
        
        # 3단계: 계층적 CBC 라우팅 수행
        for i, result in enumerate(results):
            routing_result = self.hierarchical_cbc_routing(
                result['query'], 
                result['text'], 
                result['anchors']
            )
            result['routing_decision'] = routing_result
        
        print(f"✅ CBC: Batch processing completed")
        return results
    
    def _calculate_all_scores(self, query: str, text: str, meta: Dict[str, Any]) -> Dict[str, float]:
        """SetFit, 경량 LLM, 고성능 LLM의 점수 모두 계산"""
        scores = {}
        
        # SetFit 점수 계산 (API 수정됨)
        try:
            # 쿼리와 텍스트를 결합하여 SetFit에 전달
            combined_text = f"{query} [SEP] {text}"
            setfit_result = self.setfit_helper.predict_conf(combined_text)
            setfit_score = setfit_result[1]  # p_high 확률값만 사용
            scores['setfit'] = setfit_score
        except Exception as e:
            print(f"⚠️ SetFit score calculation failed: {e}")
            scores['setfit'] = 0.5  # 기본값
        
        # 경량 LLM 점수 계산
        try:
            lite_result = self._predict_with_model(self.config.gemma_lite, query, text, meta)
            scores['lite'] = lite_result['confidence']
        except:
            scores['lite'] = 0.5  # 기본값
        
        # 고성능 LLM 점수 계산
        try:
            heavy_result = self._predict_with_model(self.config.gemma_heavy, query, text, meta)
            scores['heavy'] = heavy_result['confidence']
        except:
            scores['heavy'] = 0.5  # 기본값
        
        return scores
    
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
    
    def _explain_routing_decision(self, scores: Dict[str, float], decision: Dict[str, Any]) -> str:
        """라우팅 결정에 대한 설명 생성"""
        setfit_score = scores['setfit']
        lite_score = scores['lite']
        
        if decision['routing_strategy'] == 'SetFit+Lite_Confident':
            return f"SetFit({setfit_score:.3f}) >= {self.dynamic_thresholds['setfit']:.3f} AND Lite({lite_score:.3f}) >= {self.dynamic_thresholds['lite']:.3f} → Use Lite"
        elif decision['routing_strategy'] == 'SetFit_Confident_But_Lite_Uncertain':
            return f"SetFit({setfit_score:.3f}) >= {self.dynamic_thresholds['setfit']:.3f} BUT Lite({lite_score:.3f}) < {self.dynamic_thresholds['lite']:.3f} → Use Heavy"
        else:
            return f"SetFit({setfit_score:.3f}) < {self.dynamic_thresholds['setfit']:.3f} → Use Heavy"
    
    def _calculate_dynamic_thresholds(self):
        """점수 분포 기반으로 동적 임계값 계산 (CBC 핵심)"""
        if not self.batch_scores['setfit']:
            return
        
        print(f"📊 CBC: Score distributions - SetFit: {len(self.batch_scores['setfit'])}, Lite: {len(self.batch_scores['lite'])}, Heavy: {len(self.batch_scores['heavy'])}")
        
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
        
        print(f"🎯 CBC: Dynamic thresholds calculated - SetFit: {self.dynamic_thresholds['setfit']:.3f}, Lite: {self.dynamic_thresholds['lite']:.3f}, Heavy: {self.dynamic_thresholds['heavy']:.3f}")
    
    def _make_routing_decision(self, scores: Dict[str, float], index: int) -> Dict[str, Any]:
        """CBC 기반 최종 라우팅 결정"""
        setfit_score = scores['setfit']
        lite_score = scores['lite']
        heavy_score = scores['heavy']
        
        # SetFit 신뢰도가 동적 임계값을 넘으면 경량 LLM 결과 사용
        if setfit_score >= self.dynamic_thresholds['setfit']:
            if lite_score >= self.dynamic_thresholds['lite']:
                return {
                    'model_used': 'lite',
                    'final_score': lite_score,
                    'decision': 'accept',
                    'confidence_level': 'high',
                    'routing_strategy': 'SetFit+Lite_Confident'
                }
            else:
                return {
                    'model_used': 'heavy',
                    'final_score': heavy_score,
                    'decision': 'accept',
                    'confidence_level': 'medium',
                    'routing_strategy': 'SetFit_Confident_But_Lite_Uncertain'
                }
        
        # SetFit 신뢰도가 낮으면 고성능 LLM 결과 사용
        else:
            return {
                'model_used': 'heavy',
                'final_score': heavy_score,
                'decision': 'accept',
                'confidence_level': 'low',
                'routing_strategy': 'SetFit_Uncertain_Use_Heavy'
            }
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """CBC 라우팅 통계 반환"""
        if not self.dynamic_thresholds:
            return {}
        
        return {
            'dynamic_thresholds': self.dynamic_thresholds,
            'percentile_threshold': self.percentile_threshold,
            'calibration_mode': self.calibration_mode,
            'score_distributions': {
                'setfit': {
                    'mean': np.mean(self.batch_scores['setfit']),
                    'std': np.std(self.batch_scores['setfit']),
                    'p95': np.percentile(self.batch_scores['setfit'], 95),
                    'p90': np.percentile(self.batch_scores['setfit'], 90),
                    'p75': np.percentile(self.batch_scores['setfit'], 75)
                },
                'lite': {
                    'mean': np.mean(self.batch_scores['lite']),
                    'std': np.std(self.batch_scores['lite']),
                    'p95': np.percentile(self.batch_scores['lite'], 95),
                    'p90': np.percentile(self.batch_scores['lite'], 90),
                    'p75': np.percentile(self.batch_scores['lite'], 75)
                },
                'heavy': {
                    'mean': np.mean(self.batch_scores['heavy']),
                    'std': np.std(self.batch_scores['heavy']),
                    'p95': np.percentile(self.batch_scores['heavy'], 95),
                    'p90': np.percentile(self.batch_scores['heavy'], 90),
                    'p75': np.percentile(self.batch_scores['heavy'], 75)
                }
            },
            'calibration_analysis': {
                'setfit_overconfidence': self._analyze_overconfidence('setfit'),
                'lite_overconfidence': self._analyze_overconfidence('lite'),
                'heavy_overconfidence': self._analyze_overconfidence('heavy')
            }
        }
    
    def _analyze_overconfidence(self, model_type: str) -> Dict[str, float]:
        """과신도 분석 (CBC의 calibration mismatch 해결 효과 측정)"""
        if not self.batch_scores[model_type]:
            return {}
        
        scores = self.batch_scores[model_type]
        mean_score = np.mean(scores)
        p95_score = np.percentile(scores, 95)
        
        # 과신도 지표: 평균이 p95에 가까우면 과신도 낮음
        overconfidence = max(0, mean_score - p95_score + 0.1)
        
        return {
            'mean': mean_score,
            'p95': p95_score,
            'overconfidence_score': overconfidence,
            'calibration_quality': 'good' if overconfidence < 0.1 else 'poor'
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
            margin = p_max - np.sort(setfit_probs)[-2] if len(setfit_probs) > 1 else 0.0
            entropy = -np.sum(setfit_probs * np.log(setfit_probs + 1e-10))
            
            # 앵커 일치도 계산 (핵심!)
            if anchors:
                agreement = self._calculate_anchor_agreement(query, text, anchors)
            else:
                agreement = self._calculate_simple_agreement(query, text)
            
            # 추가 피처
            length_ratio = min(len(text) / max(len(query), 1), 3.0)
            keyword_overlap = self._calculate_keyword_overlap(query, text)
            
            # 종합 점수 (가중치 기반)
            score = (0.4*p_max + 0.2*margin - 0.15*entropy + 
                     0.15*agreement + 0.05*(length_ratio/3.0) + 0.05*keyword_overlap)
            
            return np.clip(score, 0.0, 1.0)
            
        except Exception as e:
            print(f"⚠️ Anchor-based confidence calculation failed: {e}")
            return 0.5  # 기본값
    
    def _calculate_anchor_agreement(self, query: str, text: str, anchors: List[str]) -> float:
        """앵커와의 일치도 계산"""
        try:
            if not anchors:
                return 0.5
            
            query_words = set(query.lower().split())
            text_words = set(text.lower().split())
            
            agreements = []
            for anchor in anchors:
                anchor_words = set(anchor.lower().split())
                
                # 쿼리-앵커 일치도
                query_anchor = len(query_words & anchor_words) / max(len(query_words), 1)
                
                # 텍스트-앵커 일치도  
                text_anchor = len(text_words & anchor_words) / max(len(text_words), 1)
                
                # 앵커별 종합 일치도
                anchor_agreement = (query_anchor + text_anchor) / 2
                agreements.append(anchor_agreement)
            
            # 전체 앵커의 평균 일치도
            return np.mean(agreements) if agreements else 0.5
            
        except Exception as e:
            print(f"⚠️ Anchor agreement calculation failed: {e}")
            return 0.5
    
    def _calculate_simple_agreement(self, query: str, text: str) -> float:
        """앵커 없을 때 간단한 일치도 계산"""
        try:
            query_words = set(query.lower().split())
            text_words = set(text.lower().split())
            
            if not query_words or not text_words:
                return 0.0
            
            # 쿼리-텍스트 직접 일치도
            overlap = len(query_words & text_words)
            union = len(query_words | text_words)
            
            return overlap / union if union > 0 else 0.0
            
        except Exception as e:
            print(f"⚠️ Simple agreement calculation failed: {e}")
            return 0.5
    
    def _calculate_keyword_overlap(self, query: str, text: str) -> float:
        """키워드 중복도 계산"""
        try:
            query_words = set(query.lower().split())
            text_words = set(text.lower().split())
            
            if not query_words or not text_words:
                return 0.0
            
            overlap = len(query_words & text_words)
            total = len(query_words)
            
            return overlap / total if total > 0 else 0.0
            
        except Exception as e:
            return 0.0
    
    def hierarchical_cbc_routing(self, query: str, text: str, anchors: List[str] = None) -> Dict[str, Any]:
        """계층적 CBC 라우팅 (CBC 핵심)"""
        routing_log = []
        
        try:
            # 1단계: SetFit + 앵커 (비용 최소)
            setfit_score = self.calculate_anchor_based_confidence(query, text, anchors)
            routing_log.append(f"SetFit+Anchor score: {setfit_score:.3f}")
            
            # 동적 임계값 계산 (현재 배치 기준)
            setfit_threshold = self._get_dynamic_threshold('setfit', 80)
            
            if setfit_score >= setfit_threshold:
                routing_log.append(f"SetFit score ({setfit_score:.3f}) >= threshold ({setfit_threshold:.3f}) → Accept")
                return {
                    'stage': 'setfit',
                    'model_used': 'setfit',
                    'confidence': setfit_score,
                    'decision': 'accept',
                    'routing_log': routing_log
                }
            
            # 2단계: 경량 LLM (중간 비용)
            routing_log.append(f"SetFit score ({setfit_score:.3f}) < threshold ({setfit_threshold:.3f}) → Route to Lite LLM")
            lite_result = self._predict_with_model(self.config.gemma_lite, query, text, {'name': 'generic'})
            lite_score = lite_result['confidence']
            routing_log.append(f"Lite LLM score: {lite_score:.3f}")
            
            lite_threshold = self._get_dynamic_threshold('lite', 90)
            
            if lite_score >= lite_threshold:
                routing_log.append(f"Lite score ({lite_score:.3f}) >= threshold ({lite_threshold:.3f}) → Accept")
                return {
                    'stage': 'lite',
                    'model_used': 'lite',
                    'confidence': lite_score,
                    'decision': 'accept',
                    'routing_log': routing_log
                }
            
            # 3단계: 고성능 LLM (최고 비용, 정말 필요한 경우만)
            routing_log.append(f"Lite score ({lite_score:.3f}) < threshold ({lite_threshold:.3f}) → Route to Heavy LLM")
            heavy_result = self._predict_with_model(self.config.gemma_heavy, query, text, {'name': 'generic'})
            heavy_score = heavy_result['confidence']
            routing_log.append(f"Heavy LLM score: {heavy_score:.3f}")
            
            return {
                'stage': 'heavy',
                'model_used': 'heavy',
                'confidence': heavy_score,
                'decision': 'accept',
                'routing_log': routing_log
            }
            
        except Exception as e:
            routing_log.append(f"Error: {e}")
            return {
                'stage': 'fallback',
                'model_used': 'heavy',
                'confidence': 0.5,
                'decision': 'accept',
                'reason': 'Fallback due to error'
            }
    
    def _get_dynamic_threshold(self, stage: str, percentile: int) -> float:
        """동적 임계값 계산 (CBC 핵심)"""
        try:
            if not self.batch_scores:
                return 0.7  # 기본 임계값
            
            stage_scores = [s[stage] for s in self.batch_scores if stage in s]
            if not stage_scores:
                return 0.7
            
            threshold = np.percentile(stage_scores, percentile)
            return max(threshold, 0.6)  # 최소 임계값 보장
            
        except Exception:
            return 0.7

# ===============================
# Experiment Functions
# ===============================
def run_cbc_experiment(queries: List[str], texts: List[str], labels: List[str], 
                       meta: Dict[str, Any], config: LLMFilterConfig) -> Dict[str, Any]:
    """CBC 기반 배치 라우팅 실험"""
    print("🔄 Running CBC Experiment...")
    
    # CBC 라우터 초기화
    cbc_router = CBCRouter(config)
    
    # 배치 데이터 준비
    batch_data = []
    for query, text, label in zip(queries, texts, labels):
        batch_data.append((query, text, {**meta, 'label': label}))
    
    # CBC 기반 배치 처리
    start_time = time.time()
    results = cbc_router.process_batch(batch_data)
    end_time = time.time()
    
    # 결과 분석
    routing_stats = cbc_router.get_routing_stats()
    
    # 성능 메트릭 계산
    metrics = {
        'total_time': end_time - start_time,
        'avg_time_per_sample': (end_time - start_time) / len(results),
        'routing_stats': routing_stats,
        'model_usage': {
            'lite': sum(1 for r in results if r['final_decision']['model_used'] == 'lite'),
            'heavy': sum(1 for r in results if r['final_decision']['model_used'] == 'heavy')
        },
        'decisions': {
            'accept': sum(1 for r in results if r['final_decision']['decision'] == 'accept'),
            'total': len(results)
        }
    }
    
    print(f"✅ CBC Experiment completed in {metrics['total_time']:.2f}s")
    print(f"📊 Model usage: Lite={metrics['model_usage']['lite']}, Heavy={metrics['model_usage']['heavy']}")
    print(f"🎯 Dynamic thresholds: {routing_stats.get('dynamic_thresholds', {})}")
    
    return {
        'results': results,
        'metrics': metrics,
        'routing_stats': routing_stats
    }

def run_setfit_experiment(queries: List[str], texts: List[str], labels: List[str], 
                         meta: Dict[str, Any], config: LLMFilterConfig) -> Dict[str, Any]:
    """SetFit 기반 라우팅 실험 (CBC와 비교용)"""
    print("🔄 Running SetFit Experiment...")
    
    # SetFit 헬퍼 초기화
    setfit_helper = SetFitHelper()
    
    # 경량/중량 모델 초기화
    ollama = Ollama()
    cache = ResponseCache()
    
    results = []
    start_time = time.time()
    
    for query, text, label in zip(queries, texts, labels):
        # SetFit 신뢰도 예측
        setfit_confidence = setfit_helper.predict_conf(query)[1]
        
        # 신뢰도 기반 라우팅
        if setfit_confidence >= config.theta:
            # SetFit 신뢰도 높음 → 경량 모델 사용
            model = config.gemma_lite
            result = predict_task(ollama, cache, model, meta, text)
        else:
            # SetFit 신뢰도 낮음 → 고성능 모델 사용
            model = config.gemma_heavy
            result = predict_task(ollama, cache, model, meta, text)
        
        results.append({
            'query': query,
            'text': text,
            'label': label,
            'setfit_confidence': setfit_confidence,
            'model_used': model,
            'prediction': result[0],
            'confidence': result[1],
            'usage': result[2]
        })
    
    end_time = time.time()
    
    # 성능 메트릭 계산
    metrics = {
        'total_time': end_time - start_time,
        'avg_time_per_sample': (end_time - start_time) / len(results),
        'model_usage': {
            'lite': sum(1 for r in results if r['model_used'] == config.gemma_lite),
            'heavy': sum(1 for r in results if r['model_used'] == config.gemma_heavy)
        }
    }
    
    print(f"✅ SetFit Experiment completed in {metrics['total_time']:.2f}s")
    print(f"📊 Model usage: Lite={metrics['model_usage']['lite']}, Heavy={metrics['model_usage']['heavy']}")
    
    return {
        'results': results,
        'metrics': metrics
    }

def run_lite_then_route_experiment(queries: List[str], texts: List[str], labels: List[str], 
                                  meta: Dict[str, Any], config: LLMFilterConfig) -> Dict[str, Any]:
    """경량 LLM + 신뢰도 기반 라우팅 실험"""
    print("🔄 Running Lite-then-Route Experiment...")
    
    ollama = Ollama()
    cache = ResponseCache()
    
    results = []
    start_time = time.time()
    
    for query, text, label in zip(queries, texts, labels):
        # 경량 모델로 먼저 시도
        lite_result = predict_task(ollama, cache, config.gemma_lite, meta, text)
        lite_confidence = lite_result[1]
        
        # 신뢰도가 낮으면 고성능 모델로 라우팅
        if lite_confidence < config.theta:
            heavy_result = predict_task(ollama, cache, config.gemma_heavy, meta, text)
            final_result = heavy_result
            model_used = config.gemma_heavy
        else:
            final_result = lite_result
            model_used = config.gemma_lite
        
        results.append({
            'query': query,
            'text': text,
            'label': label,
            'lite_confidence': lite_confidence,
            'model_used': model_used,
            'prediction': final_result[0],
            'confidence': final_result[1],
            'usage': final_result[2]
        })
    
    end_time = time.time()
    
    metrics = {
        'total_time': end_time - start_time,
        'avg_time_per_sample': (end_time - start_time) / len(results),
        'model_usage': {
            'lite': sum(1 for r in results if r['model_used'] == config.gemma_lite),
            'heavy': sum(1 for r in results if r['model_used'] == config.gemma_heavy)
        }
    }
    
    print(f"✅ Lite-then-Route Experiment completed in {metrics['total_time']:.2f}s")
    print(f"📊 Model usage: Lite={metrics['model_usage']['lite']}, Heavy={metrics['model_usage']['heavy']}")
    
    return {
        'results': results,
        'metrics': metrics
    }

def run_heavy_only_experiment(queries: List[str], texts: List[str], labels: List[str], 
                             meta: Dict[str, Any], config: LLMFilterConfig) -> Dict[str, Any]:
    """고성능 LLM만 사용하는 실험 (베이스라인)"""
    print("🔄 Running Heavy-Only Experiment...")
    
    ollama = Ollama()
    cache = ResponseCache()
    
    results = []
    start_time = time.time()
    
    for query, text, label in zip(queries, texts, labels):
        result = predict_task(ollama, cache, config.gemma_heavy, meta, text)
        
        results.append({
            'query': query,
            'text': text,
            'label': label,
            'model_used': config.gemma_heavy,
            'prediction': result[0],
            'confidence': result[1],
            'usage': result[2]
        })
    
    end_time = time.time()
    
    metrics = {
        'total_time': end_time - start_time,
        'avg_time_per_sample': (end_time - start_time) / len(results),
        'model_usage': {
            'lite': 0,
            'heavy': len(results)
        }
    }
    
    print(f"✅ Heavy-Only Experiment completed in {metrics['total_time']:.2f}s")
    print(f"📊 Model usage: Lite=0, Heavy={metrics['model_usage']['heavy']}")
    
    return {
        'results': results,
        'metrics': metrics
    }

def run_cbc_anchor_experiment(queries: List[str], texts: List[str], labels: List[str], 
                              meta: Dict[str, Any], config: LLMFilterConfig, 
                              anchors_dict: Dict[str, List[str]] = None) -> Dict[str, Any]:
    """CBC + 앵커 일치도 실험 (통합된 최종 버전)"""
    print("🔄 Running CBC + Anchor Agreement Experiment...")
    print("📚 Research Focus: SetFit + Anchor Agreement + Distribution-based Calibration")
    
    # CBC 앵커 라우터 초기화
    cbc_router = CBCRouter(config)
    
    # 배치 데이터 준비
    batch_data = []
    for query, text, label in zip(queries, texts, labels):
        batch_data.append((query, text, {**meta, 'label': label}))
    
    # CBC 기반 배치 처리
    start_time = time.time()
    results = cbc_router.process_batch_cbc_with_anchors(batch_data, anchors_dict)
    end_time = time.time()
    
    # 성능 메트릭 계산
    metrics = {
        'total_time': end_time - start_time,
        'avg_time_per_sample': (end_time - start_time) / len(results),
        'model_usage': {
            'setfit': sum(1 for r in results if r.get('routing_decision', {}).get('stage') == 'setfit'),
            'lite': sum(1 for r in results if r.get('routing_decision', {}).get('stage') == 'lite'),
            'heavy': sum(1 for r in results if r.get('routing_decision', {}).get('stage') == 'heavy')
        },
        'routing_efficiency': {
            'setfit_acceptance_rate': sum(1 for r in results if r.get('routing_decision', {}).get('stage') == 'setfit') / len(results),
            'lite_acceptance_rate': sum(1 for r in results if r.get('routing_decision', {}).get('stage') == 'lite') / len(results),
            'heavy_usage_rate': sum(1 for r in results if r.get('routing_decision', {}).get('stage') == 'heavy') / len(results)
        },
        'dynamic_thresholds': cbc_router.dynamic_thresholds,
        'confidence_weights': {
            'p_max': 0.4,
            'margin': 0.2,
            'entropy': -0.15,
            'agreement': 0.15,
            'length': 0.05,
            'keyword': 0.05
        }
    }
    
    print(f"✅ CBC + Anchor Experiment completed in {metrics['total_time']:.2f}s")
    print(f"📊 Stage usage: SetFit={metrics['model_usage']['setfit']}, Lite={metrics['model_usage']['lite']}, Heavy={metrics['model_usage']['heavy']}")
    print(f"🎯 Acceptance rates: SetFit={metrics['routing_efficiency']['setfit_acceptance_rate']:.2%}, Lite={metrics['routing_efficiency']['lite_acceptance_rate']:.2%}")
    
    return {
        'results': results,
        'metrics': metrics
    }

def generate_sample_anchors(query: str) -> List[str]:
    """샘플 앵커 생성 (데모용)"""
    # 간단한 키워드 기반 앵커 생성
    keywords = query.lower().split()
    anchors = []
    
    if any(word in keywords for word in ['계좌', '잔액', 'balance']):
        anchors.extend(['잔액 조회', '계좌 정보', '잔고 확인'])
    
    if any(word in keywords for word in ['이체', '송금', 'transfer']):
        anchors.extend(['돈 보내기', '계좌 이체', '송금 서비스'])
    
    if any(word in keywords for word in ['카드', 'card']):
        anchors.extend(['신용카드', '카드 정보', '카드 서비스'])
    
    if any(word in keywords for word in ['대출', 'loan']):
        anchors.extend(['대출 신청', '대출 상담', '대출 조건'])
    
    if any(word in keywords for word in ['투자', 'investment']):
        anchors.extend(['투자 상담', '투자 상품', '투자 전략'])
    
    return anchors[:3]  # 최대 3개 앵커
