#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
main.py
- Entry: run_lllmI()
- Purpose: Benchmark last-stage LLM filter with High/Low routing,
           using simplified prompts + SetFit-assisted confidence.
- Datasets: Banking77 (intent), CLINC150 (intent), STS-B (similarity 0..5)

Run:
  uv run main.py
"""

import os, sys, json, time, math, random, pathlib, hashlib, re
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
# Global config
# ===============================
SEED = 42
MAX_QUERIES = 20

THETA = 0.7             # routing threshold (balanced performance from test_semantic_collection.py)
WARMUP_K = 10           # SetFit warmup samples
UPDATE_EVERY = 5        # SetFit online mini-fit size

SETFIT_MODE = "both"    # "off" | "basic" | "dual" | "both"
MC_PASSES = 10          # cheap MC for uncertainty approx
DROPOUT_P = 0.2         # kept for API symmetry (no-op for most setfit builds)

# Eval oracle
EVAL_WITH_HEAVY_ORACLE = True   # use heavy to check whether routing decision was correct (not counted in cost/latency)

# Ollama endpoints/models
OLLAMA_URL = "http://192.168.45.166:11434"
GEMMA_LITE = "gemma:2b"    # lightweight
GEMMA_HEAVY = "gemma3"     # heavyweight

TEMPERATURE = 0.0
TOP_P = 1.0

OUT_DIR = "results/llm_filter"
CACHE_DIR = ".cache/llm_filter"


# ===============================
# Utilities
# ===============================
def set_seed(seed: int = SEED):
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
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)

def jdump(obj, path):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def jappend_jsonl(path, rec):
    ensure_dir(os.path.dirname(path))
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def normalize_text(s: str) -> str:
    return " ".join((s or "").strip().lower().split())

def nearest_label(pred: str, label_list: List[str]) -> str:
    pred = (pred or "").strip().lower()
    if not label_list:
        return pred
    label_list = [str(l) for l in label_list]
    import difflib
    cand = difflib.get_close_matches(pred, label_list, n=1, cutoff=0.0)
    return cand[0] if cand else label_list[0]

def token_f1_em(pred: str, gold: str) -> Tuple[float, float]:
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
# Simple cache for Ollama calls
# ===============================
class ResponseCache:
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
        return hashlib.sha256((model + "||" + prompt).encode("utf-8")).hexdigest()[:20]

    def get(self, model: str, prompt: str) -> Optional[Dict[str, Any]]:
        return self.map.get(self.key(model, prompt))

    def put(self, model: str, prompt: str, text: str, usage: Dict[str, Any]):
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
# Ollama client (non-stream)
# ===============================
class Ollama:
    def __init__(self, base_url=OLLAMA_URL, temperature=TEMPERATURE, top_p=TOP_P, seed=SEED):
        if requests is None:
            raise RuntimeError("`requests` not installed. Please `pip install requests`.")
        self.base_url = base_url
        self.temperature = temperature
        self.top_p = top_p
        self.seed = seed

    def generate(self, model: str, prompt: str, system: Optional[str] = None) -> Dict[str, Any]:
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
# Minimal, task-specific prompts
# ===============================
def prompt_banking77(text: str, labels: List[str]) -> str:
    # bilingual, strict answer-only
    head = "Banking customer service intent classification task.\n\n" \
           "Classify the customer's intent from the given text.\n" \
           "Answer with EXACTLY ONE label from the list below.\n" \
           "Respond with the label ONLY. No extra words.\n\n"
    label_hint = ", ".join(labels[:20]) + (", ..." if len(labels) > 20 else "")
    return f"""{head}Input: {text}

Valid intent labels (choose one): {label_hint}
Intent:"""

def prompt_clinc150(text: str, labels: List[str]) -> str:
    # run_clinc150.py와 동일한 방식: 단순한 숫자 라벨 사용
    head = "Intent Classification Task\n\n"
    label_text = ", ".join(sorted(list(set(labels))))
    
    return f"""{head}User Query: {text}

Available Intent IDs: {label_text}

Return only the intent ID number:"""

def prompt_stsb(text_block: str) -> str:
    # text_block: "Sentence 1: ...\nSentence 2: ..."
    # return 0..5 integer only
    # 최적화된 프롬프트: 상세한 예시와 가이드라인으로 30% 정확도 달성
    head = "Semantic Textual Similarity (STS-B) task.\n\n" \
           "Compare the similarity between two sentences and output a single integer from 0 to 5.\n\n" \
           "Detailed Examples:\n" \
           "- 'A cat is sleeping' vs 'A dog is running' → 0 (completely different animals and actions)\n" \
           "- 'A woman is cooking' vs 'A man is cooking' → 3 (same action, different gender)\n" \
           "- 'A girl is styling her hair' vs 'A girl is brushing her hair' → 2 (same person and body part, similar actions)\n" \
           "- 'A man is playing guitar' vs 'A man is playing the guitar' → 5 (identical meaning, just article difference)\n" \
           "- 'A group of men play soccer' vs 'A group of boys are playing soccer' → 4 (same sport, same group activity, different age)\n\n" \
           "Scoring Guide:\n" \
           "0 = completely different meanings (different topics/actions)\n" \
           "1 = mostly different meanings (same topic but very different actions)\n" \
           "2 = somewhat different meanings (same topic, related but different actions)\n" \
           "3 = somewhat similar meanings (same action, different details)\n" \
           "4 = mostly similar meanings (same action and topic, minor differences)\n" \
           "5 = identical or very similar meanings (same meaning, just word choice differences)\n\n" \
           "Output ONLY the number:\n\n"
    return f"""{head}{text_block}

Score:"""


# ===============================
# Heuristic confidence extractor
# ===============================
def extract_confidence_simple(response: str, base: float = 0.65) -> float:
    """개선된 신뢰도 추출 로직 (test_semantic_collection.py 기반)"""
    response_lower = (response or "").lower().strip()
    
    # 확실한 표현들
    confident_indicators = [
        "확실", "certain", "definitely", "absolutely", "clearly",
        "분명", "obviously", "without doubt", "no doubt", "strongly", "100%"
    ]
    
    # 불확실한 표현들
    uncertain_indicators = [
        "아마", "probably", "maybe", "perhaps", "possibly",
        "모르", "not sure", "uncertain", "unclear", "might be", "guess"
    ]
    
    # 확실한 표현 체크
    for indicator in confident_indicators:
        if indicator in response_lower:
            return 0.9
    
    # 불확실한 표현 체크
    for indicator in uncertain_indicators:
        if indicator in response_lower:
            return 0.5
    
    # 응답 길이 기반 추정
    if len(response.strip()) < 20:
        return 0.8
    elif len(response.strip()) < 50:
        return 0.7
    else:
        return 0.6


# ===============================
# Dataset loaders (robust)
# ===============================
def load_banking77(max_q=MAX_QUERIES):
    if load_dataset is None:
        raise RuntimeError("datasets not installed")
    ds = load_dataset("banking77")
    X = ds["test"]["text"][:max_q]
    labels_list = ds["test"].features["label"].names
    # gold are indices; map to strings
    Y = [labels_list[i] for i in ds["test"]["label"][:max_q]]
    return X, Y, {"task":"cls", "name":"banking77", "labels": [str(l) for l in labels_list]}

def load_clinc150(max_q=MAX_QUERIES):
    if load_dataset is None:
        raise RuntimeError("datasets not installed")
    ds = load_dataset("clinc_oos", "plus")
    X, Y = [], []
    
    # run_clinc150.py와 동일한 라벨 매핑 방식 사용
    intent2id = {}
    
    # 다양한 라벨을 포함하도록 샘플링
    seen_labels = set()
    for t, lab in zip(ds["test"]["text"], ds["test"]["intent"]):
        if lab == "oos":
            continue
        if len(X) >= max_q:
            break
        # 각 라벨당 최대 1개만 선택
        if lab not in seen_labels:
            # 라벨 매핑
            if lab not in intent2id:
                intent2id[lab] = len(intent2id)
            mapped_label = intent2id[lab]
            
            X.append(t)
            Y.append(str(mapped_label))
            seen_labels.add(lab)
    
    # 라벨이 부족하면 추가 샘플링
    if len(X) < max_q:
        for t, lab in zip(ds["test"]["text"], ds["test"]["intent"]):
            if lab == "oos":
                continue
            if len(X) >= max_q:
                break
            # 이미 선택된 라벨도 최대 2개까지 추가 허용
            if Y.count(str(intent2id.get(lab, 0))) < 2:
                if lab not in intent2id:
                    intent2id[lab] = len(intent2id)
                mapped_label = intent2id[lab]
                X.append(t)
                Y.append(str(mapped_label))
    
    # 매핑된 라벨 리스트 생성
    mapped_labels = [str(i) for i in range(len(intent2id))]
    return X, Y, {"task":"cls", "name":"clinc150", "labels": mapped_labels}

def load_stsb(max_q=MAX_QUERIES):
    if load_dataset is None:
        raise RuntimeError("datasets not installed")
    ds = load_dataset("sentence-transformers/stsb", split="test")
    
    # 다양한 라벨을 포함하도록 샘플링
    X, Y = [], []
    seen_labels = set()
    
    for i in range(len(ds)):
        if len(X) >= max_q:
            break
        s1 = ds["sentence1"][i]
        s2 = ds["sentence2"][i]
        raw_label = ds["score"][i]
        
        # -1.0 라벨은 0으로 처리 (잘못된 라벨을 0으로 매핑)
        if raw_label == -1.0:
            raw_label = 0.0
            
        # 0-5 범위로 정규화 (0.0~1.0 → 0~5)
        normalized_label = float(raw_label) * 5.0
        label = str(int(round(normalized_label)))
        
        # 최대 3개까지 같은 라벨 허용
        if Y.count(label) < 3:
            X.append(f"Sentence 1: {s1}\nSentence 2: {s2}")
            Y.append(label)
            seen_labels.add(label)
    
    # 라벨이 부족하면 추가 샘플링
    if len(X) < max_q:
        for i in range(len(ds)):
            if len(X) >= max_q:
                break
            s1 = ds["sentence1"][i]
            s2 = ds["sentence2"][i]
            raw_label = ds["score"][i]
            
            if raw_label == -1.0:
                raw_label = 0.0
                
            normalized_label = float(raw_label) * 5.0
            label = str(int(round(normalized_label)))
            if label not in seen_labels:
                X.append(f"Sentence 1: {s1}\nSentence 2: {s2}")
                Y.append(label)
                seen_labels.add(label)
    
    return X, Y, {"task":"cls", "name":"stsb", "labels": [str(i) for i in range(6)]}

DATASETS = [load_banking77, load_clinc150, load_stsb]


# ===============================
# Task-specific LLM predictors
# ===============================
def ollama_cached_call(client: Ollama, cache: ResponseCache, model: str, prompt: str) -> Tuple[str, Dict[str,Any]]:
    k = cache.get(model, prompt)
    if k:
        return k["text"], k["usage"]
    resp = client.generate(model, prompt)
    text, usage = resp["text"], resp["usage"]
    cache.put(model, prompt, text, usage)
    return text, usage

def parse_label_only(text: str) -> str:
    # grab first token-ish line
    line = (text or "").strip().splitlines()[0]
    # remove punctuation wrappers
    line = re.sub(r'^[\s"\']+|[\s"\']+$', '', line)
    # keep first word-like token set (allow underscores and hyphens)
    m = re.match(r'^([A-Za-z0-9_\-\.]+)$', line)
    return m.group(1) if m else line.strip()

def predict_bank(client, cache, model, text, labels) -> Tuple[str, float, Dict[str,Any], str]:
    prompt = prompt_banking77(text, labels)
    out, usage = ollama_cached_call(client, cache, model, prompt)
    raw = parse_label_only(out)
    lab = nearest_label(raw, labels)
    conf = extract_confidence_simple(out, base=0.70 if raw == lab else 0.60)
    return lab, conf, usage, out

def predict_clinc(client, cache, model, text, labels) -> Tuple[str, float, Dict[str,Any], str]:
    prompt = prompt_clinc150(text, labels)
    out, usage = ollama_cached_call(client, cache, model, prompt)
    raw = parse_label_only(out)
    lab = nearest_label(raw, labels)
    conf = extract_confidence_simple(out, base=0.70 if raw == lab else 0.60)
    return lab, conf, usage, out

def predict_stsb(client, cache, model, text_block) -> Tuple[str, float, Dict[str,Any], str]:
    prompt = prompt_stsb(text_block)
    out, usage = ollama_cached_call(client, cache, model, prompt)
    # extract integer 0..5 from anywhere
    m = re.search(r'\b([0-5])\b', out)
    lab = m.group(1) if m else "3"
    conf = extract_confidence_simple(out, base=0.70)
    return lab, conf, usage, out

def predict_task(client, cache, model, meta, text):
    name = meta["name"]
    labels = meta["labels"]
    if name == "banking77":
        return predict_bank(client, cache, model, text, labels)
    elif name == "clinc150":
        return predict_clinc(client, cache, model, text, labels)
    elif name == "stsb":
        return predict_stsb(client, cache, model, text)
    else:
        # generic classifier
        # reply: label only from provided labels
        prompt = "Return exactly one label from this list. Label only.\n" \
                 f"Input: {text}\nLabels: {', '.join(labels[:30])}\nLabel:"
        out, usage = ollama_cached_call(client, cache, model, prompt)
        raw = parse_label_only(out)
        lab = nearest_label(raw, labels)
        conf = extract_confidence_simple(out, base=0.65)
        return lab, conf, usage, out


# ===============================
# CBC Verifier
# ===============================
class CBCVerifier:
    """Claim-Backed Confidence 검증기"""
    
    def __init__(self, ollama_client):
        self.client = ollama_client
        
    def check_supportability(self, claim: str, evidence: str) -> tuple[bool, float]:
        """Supportability 검증: 주장이 증거에 의해 지원되는가?"""
        prompt = f"""
다음 주장이 증거에 의해 지원되는지 평가하세요.

증거: {evidence}

주장: {claim}

평가 기준:
- 주장이 증거에서 직접적으로 추론 가능한가?
- 증거가 주장을 뒷받침하는 충분한 정보를 제공하는가?
- 주장이 증거와 일치하는가?

결과: 지원됨 (SUPPORTED) 또는 지원되지 않음 (NOT_SUPPORTED)
신뢰도: 0.0-1.0 (소수점 2자리)
"""
        
        try:
            response = self.client.generate("gemma:2b", prompt)
            result_text = response.get("response", "").strip()
            
            # 결과 파싱
            supported = "SUPPORTED" in result_text.upper()
            confidence = 0.5  # 기본값
            
            # 신뢰도 추출
            if "신뢰도:" in result_text:
                try:
                    conf_line = [line for line in result_text.split('\n') if "신뢰도:" in line][0]
                    confidence = float(conf_line.split(":")[1].strip())
                except:
                    pass
            
            return supported, confidence
            
        except Exception as e:
            print(f"CBC Supportability 검증 오류: {e}")
            return False, 0.0
    
    def check_entailment(self, premise: str, hypothesis: str) -> tuple[str, float]:
        """Entailment/Contradiction 검증: 전제와 가설의 논리적 관계"""
        prompt = f"""
다음 전제와 가설의 논리적 관계를 평가하세요.

전제: {premise}

가설: {hypothesis}

관계 유형:
- ENTAILMENT: 전제가 가설을 함축함
- CONTRADICTION: 전제와 가설이 모순됨
- NEUTRAL: 전제와 가설이 중립적 관계

결과: [ENTAILMENT/CONTRADICTION/NEUTRAL]
신뢰도: 0.0-1.0 (소수점 2자리)
"""
        
        try:
            response = self.client.generate("gemma:2b", prompt)
            result_text = response.get("response", "").strip()
            
            # 결과 파싱
            relation = "NEUTRAL"  # 기본값
            confidence = 0.5
            
            if "ENTAILMENT" in result_text.upper():
                relation = "ENTAILMENT"
            elif "CONTRADICTION" in result_text.upper():
                relation = "CONTRADICTION"
            
            # 신뢰도 추출
            if "신뢰도:" in result_text:
                try:
                    conf_line = [line for line in result_text.split('\n') if "신뢰도:" in line][0]
                    confidence = float(conf_line.split(":")[1].strip())
                except:
                    pass
            
            return relation, confidence
            
        except Exception as e:
            print(f"CBC Entailment 검증 오류: {e}")
            return "NEUTRAL", 0.0
    
    def verify_claim(self, claim: str, evidence: str, original_confidence: float) -> dict:
        """CBC 전체 검증 프로세스"""
        # 1. Supportability 검증
        supported, support_conf = self.check_supportability(claim, evidence)
        
        # 2. Entailment 검증
        relation, entail_conf = self.check_entailment(evidence, claim)
        
        # 3. 증거 점수 계산
        evidence_score = (support_conf + entail_conf) / 2.0
        
        # 4. Hallucination Drop 적용
        if evidence_score < 0.5:
            drop_factor = evidence_score
            final_confidence = original_confidence * drop_factor
        else:
            final_confidence = original_confidence
        
        return {
            "supported": supported,
            "support_confidence": support_conf,
            "entailment_relation": relation,
            "entailment_confidence": entail_conf,
            "evidence_score": evidence_score,
            "original_confidence": original_confidence,
            "final_confidence": final_confidence
        }

# ===============================
# SetFit: binary HIGH/LOW router
# ===============================
class SetFitHelper:
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
        # some builds support num_iterations (contrastive)
        if num_iterations is not None:
            try: kwargs["num_iterations"] = num_iterations
            except Exception: pass
        return TrainingArguments(**kwargs)

    def warmup(self, texts: List[str], labels: List[str]):
        if not (self.enabled and texts):
            return
        try:
            y = [self.label2id.get(l, 0) for l in labels]
            ds = Dataset.from_dict({"text": texts, "label": y})
            self.model = SetFitModel.from_pretrained("sentence-transformers/paraphrase-mpnet-base-v2")
            args = self._args_safe(epochs=3, batch_size=16, num_iterations=20)
            trainer = SetFitTrainer(model=self.model, args=args, train_dataset=ds, column_mapping={"text":"text","label":"label"})
            trainer.train()
            print(f"[SetFit] Warmup OK ({len(texts)} samples)")
        except Exception as e:
            print(f"[SetFit] Warmup failed: {e}")
            self.model = None

    def predict_conf(self, text: str) -> Tuple[Optional[str], float]:
        if not (self.enabled and self.model is not None):
            return None, 0.0
        try:
            p = self.model.predict_proba([text])[0]
            p_high = float(p[1]) if len(p) > 1 else 0.0
            return ("HIGH" if p_high >= 0.5 else "LOW"), p_high
        except Exception as e:
            print(f"[SetFit] predict_conf failed: {e}")
            return None, 0.0

    def filter_decision(self, text: str) -> Tuple[str, float, float]:
        """
        SetFit 필터링 결정 (Dual Confidence + MC Dropout)
        Returns: (decision, base_confidence, uncertainty)
        """
        if not (self.enabled and self.model is not None):
            return "UNCERTAIN", 0.0, 1.0
        
        try:
            # 1. 기본 신뢰도 (Dual Confidence Check)
            base_decision, base_confidence = self.predict_conf(text)
            
            # 2. MC Dropout 불확실성 추정
            uncertainty = self.mc_uncertainty(text, passes=MC_PASSES)
            
            # 3. 필터링 결정 (불확실성 임계값 완화)
            if base_confidence > 0.7 and uncertainty < 0.5:
                decision = "UNDERSTANDS"
            else:
                decision = "UNCERTAIN"
            
            return decision, base_confidence, uncertainty
            
        except Exception as e:
            print(f"[SetFit] filter_decision failed: {e}")
            return "UNCERTAIN", 0.0, 1.0

    def mc_uncertainty(self, text: str, passes=MC_PASSES) -> float:
        if not (self.enabled and self.model is not None and np is not None):
            return 0.0
        try:
            probs = []
            for _ in range(min(passes, 5)):
                pr = self.model.predict_proba([text])[0]
                probs.append(pr)
            if len(probs) < 2: return 0.0
            P = np.stack(probs, axis=0)
            pbar = P.mean(axis=0) + 1e-12
            ent = float(-(pbar * np.log(pbar)).sum())
            maxH = math.log(len(pbar))
            return float(ent / max(1e-9, maxH))
        except Exception:
            return 0.0

    def add_online(self, x: str, y: str):
        if not (self.enabled and self.model is not None):
            return
        self.bufX.append(x); self.bufY.append(y)
        if len(self.bufX) >= UPDATE_EVERY:
            self._flush()

    def _flush(self):
        if not self.bufX: return
        try:
            # 라벨 변환 시 안전성 검사
            y = []
            for l in self.bufY:
                if l in self.label2id:
                    y.append(self.label2id[l])
                else:
                    # 알 수 없는 라벨은 기본값 사용
                    y.append(0)
            
            # 최소 2개 이상의 샘플이 있어야 함
            if len(self.bufX) < 2:
                print(f"[SetFit] Online update skipped: insufficient samples ({len(self.bufX)})")
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
# Metrics
# ===============================
def expected_calibration_error(confs: List[float], corrects: List[int], n_bins=10) -> float:
    if not confs or np is None:
        return 0.0
    bins = np.linspace(0, 1, n_bins+1)
    ece = 0.0
    confs = np.array(confs); corrects = np.array(corrects)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        m = (confs >= lo) & (confs < hi if i < n_bins-1 else confs <= hi)
        if not m.any(): continue
        acc = corrects[m].mean()
        conf_avg = confs[m].mean()
        ece += (m.mean()) * abs(acc - conf_avg)
    return float(ece)

def routing_quality(decisions: List[bool], needed_heavy: List[bool]) -> Dict[str,float]:
    # decision True -> heavy, False -> light
    # needed_heavy True -> light was wrong (i.e., heavy desirable)
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

def intent_coverage_diversity(samples: List[Dict], labels_all: List[str]) -> Dict[str,Any]:
    # samples: each has keys expected (gold), conf, predicted
    if not samples:
        return {"coverage_per_intent":{}, "diversity_score":0.0, "balance_score":0.0}
    by_intent = defaultdict(list)
    for s in samples:
        by_intent[s["expected"]].append(s)
    coverage = {}
    collected_intents = []
    for lab in labels_all:
        arr = by_intent.get(lab, [])
        high = [x for x in arr if x["final_conf"] >= 0.7]
        coverage[lab] = len(high) / max(1, len(arr))
        if high:
            collected_intents.append(lab)
    unique = len(set(collected_intents))
    diversity = unique / max(1, len(labels_all))
    # balance: std/mean of counts
    counts = [len([x for x in by_intent.get(lab, []) if x["final_conf"]>=0.7]) for lab in labels_all]
    if sum(counts)==0:
        balance = 0.0
    else:
        mu = np.mean(counts) if np is not None else (sum(counts)/len(counts))
        std = np.std(counts) if np is not None else 0.0
        balance = float(1.0 - (std / max(1e-9, mu)))
    return {"coverage_per_intent": coverage, "diversity_score": diversity, "balance_score": balance}

def cost_savings_vs_heavy(heavy_call_rate_arm: float) -> float:
    # assume heavy-only baseline = 100% heavy
    return 1.0 - heavy_call_rate_arm


# ===============================
# Runner per dataset/arm
# ===============================
def run_on_dataset(arm: str,
                   meta: Dict[str,Any],
                   X: List[str], Y: List[str],
                   client: Ollama, cache: ResponseCache,
                   setfit_mode: Optional[str],
                   out_dir: str):

    is_cls = meta["task"] == "cls"
    name   = meta["name"]
    labels_all = meta.get("labels", [])

    # B-arm: force disable SetFit
    if arm.startswith("B_"):
        setfit_mode = "off"

    # SetFit (C-arms only)
    sf = None
    if setfit_mode in ("basic","dual") and arm in ["C_setfit"]:
        sf = SetFitHelper()
        if not sf.enabled:
            print(f"[SetFit] not available -> lite_only")
            sf = None
    
    # CBC (D-arms only)
    cbc = None
    if arm == "D_cbc_enhanced":
        cbc = CBCVerifier(client)

    # Warmup teacher (allow heavy for warmup only)
    if sf is not None and sf.model is None and WARMUP_K > 0:
        warmX = X[:min(WARMUP_K, len(X))]
        warmY = []
        print(f"[SetFit] Warmup {len(warmX)} samples")
        for t in warmX:
            lite_lab, lite_conf, _, _   = predict_task(client, cache, GEMMA_LITE, meta, t)
            heavy_lab, heavy_conf, _, _ = predict_task(client, cache, GEMMA_HEAVY, meta, t)
            ok = normalize_text(lite_lab) == normalize_text(heavy_lab)
            teach = "HIGH" if (ok and lite_conf >= THETA) else "LOW"
            warmY.append(teach)
        try:
            sf.warmup(warmX, warmY)
        except Exception as e:
            print(f"[SetFit] Warmup failed: {e}")
            sf = None

    runs_path = os.path.join(out_dir, arm, "runs.jsonl")
    ensure_dir(os.path.dirname(runs_path))

    # Accumulators
    corrects, confs = [], []
    lat_sum = 0.0
    tok_sum = 0
    heavy_calls = 0
    decisions = []       # True=heavy
    needed_heavy = []    # ground truth (if light would be wrong) from heavy-oracle

    samples_log = []     # for intent analysis

    for i, (inp, gold) in enumerate(zip(X, Y)):
        if i >= MAX_QUERIES: break

        # Arm A: heavy-only
        if arm == "A_heavy_only":
            pred, _, uH, raw = predict_task(client, cache, GEMMA_HEAVY, meta, inp)
            correct = int(normalize_text(pred) == normalize_text(gold))
            corrects.append(correct)
            conf = 0.9  # heavy conf uncalibrated -> assume high
            confs.append(conf)
            lat_sum += uH.get("latency_s",0.0); tok_sum += int(uH.get("total_tokens_est",0) or 0)
            heavy_calls += 1
            decisions.append(True)
            needed_heavy.append(True)  # in heavy-only baseline assume routed heavy
            jappend_jsonl(runs_path, {
                "dataset": name, "i": i, "arm": arm, "input": inp, "gold": gold,
                "decision": "HeavyOnly", "called_heavy": True, "pred": pred,
                "correct": bool(correct), "conf": conf, "usage": {"heavy": uH}
            })
            if i < 3:
                print(f"\n🔬 CASE {i+1} [{name}::{arm}] HEAVY ONLY")
                print(f"   Input: '{inp[:60]}{'...' if len(inp)>60 else ''}'")
                print(f"   Gold: '{gold}' | Output: '{pred}' | Acc: {correct}")
            continue

        # Step 1: Light pass
        lite_pred, lite_conf, uL, rawL = predict_task(client, cache, GEMMA_LITE, meta, inp)

        # Step 2: SetFit filtering / CBC verification
        conf_mode = "lite_only"
        setfit_decision = "UNCERTAIN"
        setfit_conf = 0.0
        setfit_uncertainty = 0.0
        cbc_result = None
        
        if sf is not None and sf.model is not None:
            setfit_decision, setfit_conf, setfit_uncertainty = sf.filter_decision(inp)
            if setfit_decision == "UNDERSTANDS":
                # SetFit이 이해하면 SetFit 결과 사용
                conf_mode = "setfit_filter"
                base_conf = setfit_conf
                final_pred = "SETFIT_RESULT"  # SetFit 결과 (실제로는 SetFit 예측 사용)
            else:
                # SetFit이 이해 못하면 경량 LLM 신뢰도에 MC Dropout 불확실성 적용
                conf_mode = "setfit"
                base_conf = lite_conf * (1.0 - setfit_uncertainty)
        else:
            base_conf = lite_conf
        
        # Step 2.5: CBC verification (D_cbc_enhanced only)
        if cbc is not None:
            # CBC 검증을 위한 증거 생성 (실제로는 외부 지식베이스에서 가져와야 함)
            evidence = f"사용자 질문: {inp}\n경량 LLM 응답: {lite_pred}"
            
            # CBC 검증 실행
            cbc_result = cbc.verify_claim(lite_pred, evidence, base_conf)
            base_conf = cbc_result["final_confidence"]
            conf_mode = "cbc_verified"

        # Step 3: Routing
        if conf_mode == "setfit_filter":
            # SetFit이 이해한 경우
            called_heavy = False
            final_pred = "SETFIT_RESULT"  # 실제로는 SetFit 예측 사용
            final_conf = base_conf
        else:
            # 일반적인 라우팅
            called_heavy = base_conf < THETA
            if called_heavy:
                pred, _, uH, rawH = predict_task(client, cache, GEMMA_HEAVY, meta, inp)
                lat_sum += uH.get("latency_s",0.0); tok_sum += int(uH.get("total_tokens_est",0) or 0)
                heavy_calls += 1
                final_pred = pred
                final_conf = max(base_conf, 0.80)  # escalate conf when heavy used
            else:
                final_pred = lite_pred
                final_conf = base_conf

        # Step 4: Online SetFit teacher (no extra heavy call)
        if sf is not None and sf.model is not None:
            teacher = "LOW" if called_heavy else ("HIGH" if base_conf >= THETA else "LOW")
            sf.add_online(inp, teacher)

        # Step 5: Correctness
        correct = int(normalize_text(final_pred) == normalize_text(gold))
        corrects.append(correct); confs.append(final_conf)
        lat_sum += uL.get("latency_s",0.0); tok_sum += int(uL.get("total_tokens_est",0) or 0)
        decisions.append(bool(called_heavy))

        # Eval oracle (does light need heavy?)
        need = None
        if EVAL_WITH_HEAVY_ORACLE:
            # If we kept light, check if light is wrong AND heavy would be right
            if not called_heavy:
                # do NOT count this in cost/latency stats
                hv_pred, _, _, _ = predict_task(client, cache, GEMMA_HEAVY, meta, inp)
                need = (normalize_text(lite_pred) != normalize_text(gold)) and (normalize_text(hv_pred) == normalize_text(gold))
            else:
                need = True  # we routed heavy
        else:
            # fallback heuristic: need heavy iff base_conf < THETA
            need = base_conf < THETA
        needed_heavy.append(bool(need))

        # Log sample
        rec = {
            "dataset": name, "i": i, "arm": arm, "input": inp, "gold": gold,
            "lite_pred": lite_pred, "lite_conf": lite_conf,
            "final_pred": final_pred, "final_conf": final_conf,
            "conf_mode": conf_mode, "unc_setfit": setfit_uncertainty,
            "setfit_decision": setfit_decision, "setfit_conf": setfit_conf,
            "cbc_result": cbc_result,
            "decision": "SetFit" if conf_mode == "setfit_filter" else ("CBC" if conf_mode == "cbc_verified" else ("Low->Heavy" if called_heavy else "High->Light")),
            "called_heavy": called_heavy,
            "correct": bool(correct),
            "usage": {"lite": uL, "heavy": (uH if called_heavy else {"latency_s":0.0,"total_tokens_est":0})}
        }
        samples_log.append({
            "expected": gold,
            "predicted": final_pred,
            "final_conf": final_conf
        })
        jappend_jsonl(runs_path, rec)

        if i < 3:
            print(f"\n🔬 CASE {i+1} ANALYSIS:")
            print(f"   Input: '{inp[:60]}{'...' if len(inp) > 60 else ''}'")
            print(f"   Gold: {gold}")
            print(f"   Lite: '{lite_pred}' (conf {lite_conf:.2f})")
            if sf is not None and sf.model is not None:
                print(f"   SetFit: {setfit_decision} (conf {setfit_conf:.2f}, unc {setfit_uncertainty:.2f})")
            if cbc_result is not None:
                print(f"   CBC: evidence_score {cbc_result['evidence_score']:.2f}, final_conf {cbc_result['final_confidence']:.2f}")
            print(f"   Base: {base_conf:.2f} [{conf_mode}]")
            if conf_mode == "setfit_filter":
                print(f"   Decision: SetFit 처리")
            elif conf_mode == "cbc_verified":
                print(f"   Decision: CBC 검증")
            else:
                print(f"   Decision: {'LOW -> call HEAVY' if called_heavy else 'HIGH -> keep LITE'}")
            print(f"   Final: '{final_pred}' (conf {final_conf:.2f}) | Correct: {bool(correct)}")

    # Summaries
    n = min(len(X), MAX_QUERIES)
    qual = {"accuracy": sum(corrects)/max(1,n)}
    eff = {
        "avg_latency_s": lat_sum / max(1, n),
        "avg_tokens": tok_sum / max(1, n),
        "heavy_call_rate": heavy_calls / max(1, n)
    }
    # Calibration
    ece = expected_calibration_error(confs, [int(c) for c in corrects], n_bins=10)
    # Routing effect
    route = routing_quality(decisions, needed_heavy)
    # Intent analytics (only for classification with labels)
    intent_stats = intent_coverage_diversity(samples_log, labels_all) if is_cls and labels_all else {}
    # Cost savings
    savings = cost_savings_vs_heavy(eff["heavy_call_rate"])

    metrics = {
        "quality": qual,
        "efficiency": eff,
        "calibration": {"ECE": ece},
        "routing": route,
        "intent_stats": intent_stats,
        "cost": {"savings_vs_heavy_only": savings},
        "meta": {"dataset": name, "task": meta["task"], "arm": arm, "theta": THETA, "setfit_mode": setfit_mode}
    }

    jdump(metrics, os.path.join(out_dir, arm, "metrics.json"))

    # Pretty print
    print(f"\n{'='*80}")
    print(f"📊 {name.upper()} DATASET - {arm.upper()} ARCHITECTURE")
    print(f"{'='*80}")
    print(f"\n🔍 QUALITY:    Acc {qual['accuracy']:.4f}")
    print(f"⚡ EFFICIENCY:  Lat {eff['avg_latency_s']:.3f}s | Tokens {eff['avg_tokens']:.1f} | Heavy {eff['heavy_call_rate']:.1%}")
    print(f"🎯 ROUTING:     Acc {route['routing_accuracy']:.3f} | FP {route['fp_rate']:.3f} | FN {route['fn_rate']:.3f}")
    print(f"🎛️ CALIBRATION: ECE {ece:.3f}")
    if intent_stats:
        print(f"📚 INTENT:      Diversity {intent_stats.get('diversity_score',0.0):.3f} | Balance {intent_stats.get('balance_score',0.0):.3f}")
    print(f"💸 COST:        Savings {savings:.3f}")

    return metrics


# ===============================
# Entry point
# ===============================
def run_lllmI():
    set_seed(SEED)
    ensure_dir(OUT_DIR); ensure_dir(CACHE_DIR)
    client = Ollama()
    cache = ResponseCache(os.path.join(CACHE_DIR, "ollama_cache.jsonl"))

    # Load datasets
    datasets = []
    for fn in DATASETS:
        try:
            X, Y, meta = fn(MAX_QUERIES)
            datasets.append((X,Y,meta))
        except Exception as e:
            print(f"[Skip] {fn.__name__} -> {e}")

    if not datasets:
        print("No datasets loaded. Please install `datasets` or check loaders.")
        return

    all_results = []

    for (X, Y, meta) in datasets:
        name = meta["name"]
        base_dir = os.path.join(OUT_DIR, name)
        ensure_dir(base_dir)

        print(f"\n{'='*100}")
        print(f"🔬 BENCHMARK: {name.upper()}")
        print(f"{'='*100}")

        # A: Heavy-only
        print("\n📊 Running Heavy-only baseline...")
        A = run_on_dataset("A_heavy_only", meta, X, Y, client, cache, None, base_dir)

        # B: Lite -> Route (no SetFit)
        print("\n📊 Running Basic routing...")
        B = run_on_dataset("B_lite_then_route", meta, X, Y, client, cache, "off", base_dir)

        # C: SetFit + MC Dropout
        C = None
        if SETFIT_MODE in ("dual","both"):
            print("\n📊 Running SetFit...")
            C = run_on_dataset("C_setfit", meta, X, Y, client, cache, "dual", base_dir)
        
        # D: SetFit + CBC verification
        D = None
        print("\n📊 Running D_cbc_enhanced...")
        D = run_on_dataset("D_cbc_enhanced", meta, X, Y, client, cache, "dual", base_dir)

        # Simple comprehensive comparison (vs heavy)
        def combined_score(arm, heavy):
            # similar to earlier: perf 40%, cost 30%, latency 30%
            perf_ratio = arm["quality"]["accuracy"] / max(1e-9, heavy["quality"]["accuracy"])
            lat_impr = (heavy["efficiency"]["avg_latency_s"] - arm["efficiency"]["avg_latency_s"]) / max(1e-9, heavy["efficiency"]["avg_latency_s"])
            savings = 1.0 - arm["efficiency"]["heavy_call_rate"]
            return 0.4*perf_ratio + 0.3*savings + 0.3*(1.0 + lat_impr)

        table = []
        for arm_name, arm_metrics in [("A_heavy_only", A), ("B_lite_then_route", B), ("C_setfit", C), ("D_cbc_enhanced", D)]:
            if not arm_metrics: continue
            score = "Baseline" if arm_name=="A_heavy_only" else f"{combined_score(arm_metrics, A):.3f}"
            table.append((arm_name,
                          f"Acc {arm_metrics['quality']['accuracy']:.4f}",
                          f"{arm_metrics['efficiency']['avg_latency_s']:.3f}",
                          f"{arm_metrics['efficiency']['heavy_call_rate']:.1%}",
                          f"ECE {arm_metrics['calibration']['ECE']:.3f}",
                          score))

        # pretty table
        print(f"\n📋 RESULTS TABLE — {name.upper()}")
        print(f"{'Architecture':<20} {'Quality':<16} {'Latency(s)':<12} {'Heavy%':<10} {'Calib':<10} {'Score':<10}")
        print(f"{'-'*20} {'-'*16} {'-'*12} {'-'*10} {'-'*10} {'-'*10}")
        for row in table:
            print(f"{row[0]:<20} {row[1]:<16} {row[2]:<12} {row[3]:<10} {row[4]:<10} {row[5]:<10}")

        all_results.append({"dataset": name, "arms": {"A":A,"B":B,"C":C,"D":D}})

    # Save global
    jdump({"seed":SEED,"theta":THETA,"setfit_mode":SETFIT_MODE,"results":all_results}, os.path.join(OUT_DIR, "summary_all.json"))
    print(f"\n💾 RESULTS SAVED TO: {OUT_DIR}")


if __name__ == "__main__":
    run_lllmI()