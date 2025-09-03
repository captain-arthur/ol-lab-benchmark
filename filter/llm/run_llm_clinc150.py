#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_llm_clinc150.py
- CLINC150 데이터셋을 위한 LLM 필터링 벤치마크
- 공통 모듈 llm_filter.py를 사용하여 CLINC150 전용 로직 구현
"""

import os
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict

# 공통 모듈 import
from .llm_filter import (
    LLMFilterConfig, set_seed, ensure_dir, jdump, jappend_jsonl,
    normalize_text, Ollama, ResponseCache, predict_task, SetFitHelper,
    pearson_spearman, expected_calibration_error, routing_quality
)

# ===============================
# Configuration
# ===============================
EXPERIMENT_CONFIG = {
    "max_queries": 20,
    "out_dir": "results/llm_filter/clinc150",
    "cache_dir": ".cache/llm_filter/clinc150"
}

# 환경변수 오버라이드
try:
    EXPERIMENT_CONFIG["max_queries"] = int(os.getenv("OL_MAX_QUERIES", str(EXPERIMENT_CONFIG["max_queries"])))
except Exception:
    pass

def get_clinc150_config() -> LLMFilterConfig:
    """CLINC150 데이터셋에 최적화된 설정"""
    return LLMFilterConfig(
        max_queries=EXPERIMENT_CONFIG["max_queries"],
        theta=float(os.getenv("OL_THETA", "0.65")),
        warmup_k=int(os.getenv("OL_WARMUP_K", "10")),
        update_every=int(os.getenv("OL_UPDATE_EVERY", "5")),
        setfit_mode=os.getenv("OL_SETFIT_MODE", "both"),
        ollama_url=os.getenv("OLLAMA_URL", "http://192.168.45.166:11434"),
        gemma_lite=os.getenv("OLLAMA_MODEL_LITE", "gemma:2b"),
        gemma_heavy=os.getenv("OLLAMA_MODEL_HEAVY", "gemma3"),
    )

# ===============================
# Data Loading
# ===============================
def load_clinc150_dataset(max_q: int = 20) -> Tuple[List[str], List[str], Dict[str, Any]]:
    """CLINC150 데이터셋 로드"""
    try:
        from datasets import load_dataset
    except ImportError:
        raise RuntimeError("datasets not installed. Please install with: pip install datasets")
    
    # run_clinc150.py의 검증된 방식 사용
    ds_test = None
    tried = []
    for cfg in ["plus", None]:
        try:
            if cfg is None:
                ds_test = load_dataset("clinc_oos", split="test")
            else:
                ds_test = load_dataset("clinc_oos", cfg, split="test")
            break
        except Exception as e:
            tried.append((cfg, str(e)))
            ds_test = None

    if ds_test is None:
        raise RuntimeError(f"Failed to load clinc_oos. Tried: {tried}")

    # 검증된 방식으로 텍스트와 라벨 추출 (라벨 다양성 강제 보장)
    X, Y = [], []
    intent2id = {}
    label_counts = {}
    
    # 1단계: 모든 고유 intent 수집
    for ex in ds_test:
        intent = ex.get("intent")
        if intent and intent != "oos":
            if intent not in intent2id:
                intent2id[intent] = len(intent2id)
    
    # 2단계: 라벨별로 균등하게 샘플링
    max_per_label = max(1, max_q // len(intent2id)) if intent2id else 1
    
    for ex in ds_test:
        if len(X) >= max_q:
            break
            
        text = ex.get("text") or ex.get("utterance") or ex.get("sentence")
        if text is None:
            text = str(ex)
        intent = ex.get("intent")
        
        if intent == "oos":
            continue
            
        if intent is None:
            # 없으면 numeric label 사용
            lab = int(ex.get("label"))
        else:
            if intent not in intent2id:
                intent2id[intent] = len(intent2id)
            lab = intent2id[intent]
        
        # 라벨별 개수 제한
        lab_str = str(lab)
        if lab_str not in label_counts:
            label_counts[lab_str] = 0
        
        if label_counts[lab_str] < max_per_label:
            X.append(text)
            Y.append(lab_str)
            label_counts[lab_str] += 1
    
    # 라벨 리스트 생성
    mapped_labels = [str(i) for i in range(len(intent2id))]
    
    meta = {
        "task": "cls",
        "name": "clinc150",
        "labels": mapped_labels
    }
    
    print(f"[CLINC150] Loaded {len(X)} samples with {len(set(Y))} unique labels")
    return X, Y, meta

# ===============================
# Experiment Runner
# ===============================
def run_clinc150_experiment():
    """CLINC150 실험 실행"""
    config = get_clinc150_config()
    set_seed(config.seed)
    
    # 디렉토리 생성
    ensure_dir(EXPERIMENT_CONFIG["out_dir"])
    ensure_dir(EXPERIMENT_CONFIG["cache_dir"])
    
    # 데이터 로드
    X, Y, meta = load_clinc150_dataset(config.max_queries)
    
    # 클라이언트 초기화
    client = Ollama(
        base_url=config.ollama_url,
        temperature=config.temperature,
        top_p=config.top_p,
        seed=config.seed
    )
    cache = ResponseCache(os.path.join(EXPERIMENT_CONFIG["cache_dir"], "ollama_cache.jsonl"))
    
    print(f"\n{'='*80}")
    print(f"🔬 CLINC150 LLM FILTERING BENCHMARK")
    print(f"{'='*80}")
    print(f"Config: theta={config.theta}, setfit_mode={config.setfit_mode}")
    print(f"Models: lite={config.gemma_lite}, heavy={config.gemma_heavy}")
    
    # 실험 실행
    results = {}
    
    # A: Heavy-only baseline
    print("\n📊 Running Heavy-only baseline...")
    results["A_heavy_only"] = run_arm("A_heavy_only", meta, X, Y, client, cache, None, config)
    
    # B: Lite -> Route (no SetFit)
    print("\n📊 Running Basic routing...")
    results["B_lite_then_route"] = run_arm("B_lite_then_route", meta, X, Y, client, cache, "off", config)
    
    # C: SetFit + MC Dropout
    if config.setfit_mode in ("dual", "both"):
        print("\n📊 Running SetFit...")
        results["C_setfit"] = run_arm("C_setfit", meta, X, Y, client, cache, "dual", config)
    
    # 결과 요약
    print_results_summary(results, meta)
    
    # 결과 저장
    save_results(results, meta)
    
    return results

def run_arm(arm: str, meta: Dict[str, Any], X: List[str], Y: List[str], 
            client: Ollama, cache: ResponseCache, setfit_mode: Optional[str], 
            config: LLMFilterConfig) -> Dict[str, Any]:
    """개별 실험 아머 실행"""
    
    is_cls = meta["task"] == "cls"
    name = meta["name"]
    labels_all = meta.get("labels", [])
    
    # B-arm: force disable SetFit
    if arm.startswith("B_"):
        setfit_mode = "off"
    
    # SetFit (C-arms only)
    sf = None
    if setfit_mode in ("basic", "dual") and arm in ["C_setfit"]:
        sf = SetFitHelper()
        if not sf.enabled:
            print(f"[SetFit] not available -> lite_only")
            sf = None
    
    # Warmup teacher (allow heavy for warmup only)
    if sf is not None and sf.model is None and config.warmup_k > 0:
        warmX = X[:min(config.warmup_k, len(X))]
        warmY = []
        print(f"[SetFit] Warmup {len(warmX)} samples")
        for t in warmX:
            lite_lab, lite_conf, _, _ = predict_task(client, cache, config.gemma_lite, meta, t)
            heavy_lab, heavy_conf, _, _ = predict_task(client, cache, config.gemma_heavy, meta, t)
            ok = normalize_text(lite_lab) == normalize_text(heavy_lab)
            teach = "HIGH" if (ok and lite_conf >= config.theta) else "LOW"
            warmY.append(teach)
        try:
            sf.warmup(warmX, warmY)
        except Exception as e:
            print(f"[SetFit] Warmup failed: {e}")
            sf = None
    
    runs_path = os.path.join(EXPERIMENT_CONFIG["out_dir"], arm, "runs.jsonl")
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
        if i >= config.max_queries: 
            break
        
        # Arm A: heavy-only
        if arm == "A_heavy_only":
            pred, _, uH, raw = predict_task(client, cache, config.gemma_heavy, meta, inp)
            correct = int(normalize_text(pred) == normalize_text(gold))
            corrects.append(correct)
            conf = 0.9  # heavy conf uncalibrated -> assume high
            confs.append(conf)
            lat_sum += uH.get("latency_s", 0.0)
            tok_sum += int(uH.get("total_tokens_est", 0) or 0)
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
        lite_pred, lite_conf, uL, rawL = predict_task(client, cache, config.gemma_lite, meta, inp)
        
        # Step 2: SetFit filtering
        conf_mode = "lite_only"
        setfit_decision = "UNCERTAIN"
        setfit_conf = 0.0
        setfit_uncertainty = 0.0
        
        if sf is not None and sf.model is not None:
            setfit_decision, setfit_conf, setfit_uncertainty = sf.predict_conf(inp), 0.0, 0.0
            if setfit_decision == "HIGH":
                conf_mode = "setfit_filter"
                base_conf = max(lite_conf, setfit_conf)
                final_pred = lite_pred
            else:
                conf_mode = "setfit"
                base_conf = lite_conf * 0.8  # uncertainty penalty
        else:
            base_conf = lite_conf
        
        # Step 3: Routing
        if conf_mode == "setfit_filter":
            called_heavy = False
            final_conf = base_conf
        else:
            called_heavy = base_conf < config.theta
            if called_heavy:
                pred, _, uH, rawH = predict_task(client, cache, config.gemma_heavy, meta, inp)
                lat_sum += uH.get("latency_s", 0.0)
                tok_sum += int(uH.get("total_tokens_est", 0) or 0)
                heavy_calls += 1
                final_pred = pred
                final_conf = max(base_conf, 0.80)
            else:
                final_pred = lite_pred
                final_conf = base_conf
        
        # Step 4: Online SetFit teacher
        if sf is not None and sf.model is not None:
            teacher = "LOW" if called_heavy else ("HIGH" if base_conf >= config.theta else "LOW")
            sf.add_online(inp, teacher)
        
        # Step 5: Correctness
        correct = int(normalize_text(str(final_pred)) == normalize_text(str(gold)))
        corrects.append(correct)
        confs.append(final_conf)
        lat_sum += uL.get("latency_s", 0.0)
        tok_sum += int(uL.get("total_tokens_est", 0) or 0)
        decisions.append(bool(called_heavy))
        
        # Eval oracle (does light need heavy?)
        if config.eval_with_heavy_oracle:
            if called_heavy:
                hv_pred = final_pred
                need = (normalize_text(str(lite_pred)) != normalize_text(str(gold))) and \
                       (normalize_text(str(hv_pred)) == normalize_text(str(gold)))
            else:
                hv_pred, _, _, _ = predict_task(client, cache, config.gemma_heavy, meta, inp)
                need = (normalize_text(str(lite_pred)) != normalize_text(str(gold))) and \
                       (normalize_text(str(hv_pred)) == normalize_text(str(gold)))
        else:
            need = base_conf < config.theta
        needed_heavy.append(bool(need))
        
        # Log sample
        rec = {
            "dataset": name, "i": i, "arm": arm, "input": inp, "gold": gold,
            "lite_pred": lite_pred, "lite_conf": lite_conf,
            "final_pred": final_pred, "final_conf": final_conf,
            "conf_mode": conf_mode, "unc_setfit": setfit_uncertainty,
            "setfit_decision": setfit_decision, "setfit_conf": setfit_conf,
            "decision": "SetFit" if conf_mode == "setfit_filter" else ("Low->Heavy" if called_heavy else "High->Light"),
            "called_heavy": called_heavy,
            "correct": bool(correct),
            "raw_lite": rawL, "raw_heavy": rawH if called_heavy else None,
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
                print(f"   SetFit: {setfit_decision} (conf {setfit_conf:.2f})")
            print(f"   Base: {base_conf:.2f} [{conf_mode}]")
            print(f"   Decision: {'LOW -> call HEAVY' if called_heavy else 'HIGH -> keep LITE'}")
            print(f"   Final: '{final_pred}' (conf {final_conf:.2f}) | Correct: {bool(correct)}")
    
    # Summaries
    n = min(len(X), config.max_queries)
    
    # 분류 태스크: 정확도 계산
    qual = {"accuracy": sum(corrects)/max(1, n)}
    
    eff = {
        "avg_latency_s": lat_sum / max(1, n),
        "avg_tokens": tok_sum / max(1, n),
        "heavy_call_rate": heavy_calls / max(1, n)
    }
    
    # Calibration
    ece = expected_calibration_error(confs, [int(c) for c in corrects], n_bins=10)
    
    # Routing effect
    route = routing_quality(decisions, needed_heavy)
    
    # Intent analytics
    intent_stats = intent_coverage_diversity(samples_log, labels_all) if is_cls and labels_all else {}
    
    # Cost savings
    savings = 1.0 - eff["heavy_call_rate"]
    
    metrics = {
        "quality": qual,
        "efficiency": eff,
        "calibration": {"ECE": ece},
        "routing": route,
        "intent_stats": intent_stats,
        "cost": {"savings_vs_heavy_only": savings},
        "meta": {"dataset": name, "task": meta["task"], "arm": arm, "theta": config.theta, "setfit_mode": setfit_mode}
    }
    
    jdump(metrics, os.path.join(EXPERIMENT_CONFIG["out_dir"], arm, "metrics.json"))
    
    # Pretty print
    print(f"\n{'='*80}")
    print(f"📊 {name.upper()} DATASET - {arm.upper()} ARCHITECTURE")
    print(f"{'='*80}")
    print(f"\n🔍 QUALITY:    Acc {qual['accuracy']:.4f}")
    print(f"⚡ EFFICIENCY:  Lat {eff['avg_latency_s']:.3f}s | Tokens {eff['avg_tokens']:.1f} | Heavy {eff['heavy_call_rate']:.1%}")
    print(f"🎯 ROUTING:     Acc {route['routing_accuracy']:.3f} | FP {route['fp_rate']:.3f} | FN {route['fn_rate']:.3f} (θ={config.theta})")
    print(f"🎛️ CALIBRATION: ECE {ece:.3f}")
    if intent_stats:
        print(f"📚 INTENT:      Diversity {intent_stats.get('diversity_score',0.0):.3f} | Balance {intent_stats.get('balance_score',0.0):.3f}")
    print(f"💸 COST:        Savings {savings:.3f}")
    
    return metrics

def intent_coverage_diversity(samples: List[Dict], labels_all: List[str]) -> Dict[str, Any]:
    """의도별 커버리지 및 다양성 계산"""
    if not samples:
        return {"coverage_per_intent": {}, "diversity_score": 0.0, "balance_score": 0.0}
    
    if len(samples) < 10:
        print(f"[Warning] Intent diversity calculation skipped: insufficient samples ({len(samples)} < 10)")
        return {"coverage_per_intent": {}, "diversity_score": 0.0, "balance_score": 0.0, "warning": "insufficient_samples"}
    
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
    if sum(counts) == 0:
        balance = 0.0
    else:
        mu = sum(counts) / len(counts)
        std = (sum((c - mu) ** 2 for c in counts) / len(counts)) ** 0.5
        balance = float(1.0 - (std / max(1e-9, mu))) if mu > 0 else 0.0
    
    return {"coverage_per_intent": coverage, "diversity_score": diversity, "balance_score": balance}

def print_results_summary(results: Dict[str, Any], meta: Dict[str, Any]):
    """결과 요약 출력"""
    print(f"\n📋 RESULTS SUMMARY — {meta['name'].upper()}")
    print(f"{'Architecture':<20} {'Quality':<16} {'Latency(s)':<12} {'Heavy%':<10} {'Calib':<10}")
    print(f"{'-'*20} {'-'*16} {'-'*12} {'-'*10} {'-'*10}")
    
    for arm_name, arm_metrics in results.items():
        if not arm_metrics:
            continue
        
        quality_str = f"Acc {arm_metrics['quality']['accuracy']:.4f}"
        table_row = (arm_name,
                     quality_str,
                     f"{arm_metrics['efficiency']['avg_latency_s']:.3f}",
                     f"{arm_metrics['efficiency']['heavy_call_rate']:.1%}",
                     f"ECE {arm_metrics['calibration']['ECE']:.3f}")
        
        print(f"{table_row[0]:<20} {table_row[1]:<16} {table_row[2]:<12} {table_row[3]:<10} {table_row[4]:<10}")

def save_results(results: Dict[str, Any], meta: Dict[str, Any]):
    """결과 저장"""
    summary_path = os.path.join(EXPERIMENT_CONFIG["out_dir"], "summary.json")
    summary = {
        "dataset": meta["name"],
        "task": meta["task"],
        "results": results
    }
    jdump(summary, summary_path)
    print(f"\n💾 RESULTS SAVED TO: {EXPERIMENT_CONFIG['out_dir']}")

if __name__ == "__main__":
    run_clinc150_experiment()
