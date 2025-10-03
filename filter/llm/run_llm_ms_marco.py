#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_llm_ms_marco.py
- MS MARCO 데이터셋에 대한 LLM 필터링 실험 실행
- 4가지 실험 방식 비교: LLM Only, LLM+SDE, LLM+CBC, LLM+SDE+CBC
"""

import sys
import os
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# ===== Output: force line-buffered + optional tqdm =====
try:
    # Python 3.7+ 에서 라인 단위로 바로바로 flush
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

from filter.llm.llm_filter import (
    load_ms_marco_data, 
    LLMFilterConfig, 
    run_optimized_experiments
)

def main():
    """MS MARCO LLM 실험 실행 (4가지 방식 비교)"""
    parser = argparse.ArgumentParser(description='MS MARCO LLM Filtering Experiments')
    parser.add_argument('queries', type=int, nargs='?', default=1, help='Number of queries to process (default: 1)')
    
    args = parser.parse_args()
    
    print("🚀 MS MARCO LLM 필터링 실험 (4가지 방식 비교)", flush=True)
    print("=" * 60, flush=True)
    print(f"📊 처리할 쿼리 수: {args.queries}", flush=True)
    print(f"📊 쿼리당 문서 수: 10 (고정)", flush=True)
    print(f"🔒 유사도 임계값: 0.3 (고정)", flush=True)
    print(f"🤖 API 모드: ollama (고정)", flush=True)
    print(f"🎯 실행할 실험: baseline, sde, cbc, sde_cbc (모든 실험)", flush=True)
    print("=" * 60, flush=True)
    
    try:
        # 데이터 로드
        data = load_ms_marco_data(args.queries)
        if not data:
            print("❌ MS MARCO 데이터 로드 실패", flush=True)
            return
        
        # 설정 (모든 파라미터 고정)
        config = LLMFilterConfig(
            max_queries=args.queries,
            max_passages=10,  # 고정
            api_mode='ollama',  # 고정
            model_name='gemma3:latest',  # 고정
            sim_threshold=0.3,  # 고정
            sde_k=3,  # 고정
            temperature=0.0,
            top_p=1.0
        )
        
        # 최적화된 실험 실행 (4가지 방식 통합)
        results = run_optimized_experiments(data, config)
        
        # 결과 저장 (간단한 메트릭만)
        output_dir = "results/llm/ms_marco"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "llm_experiments_summary.json")
        
        # 간단한 요약만 저장
        summary = []
        for result in results:
            summary.append({
                'experiment_type': result['experiment_type'],
                'precision': result['overall_metrics']['precision'],
                'recall': result['overall_metrics']['recall'],
                'f1': result['overall_metrics']['f1'],
                'total_latency': result['overall_metrics']['total_latency']
            })
        
        with open(output_path, 'w', encoding='utf-8') as f:
            import json
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 결과 요약 저장: {output_path}", flush=True)
        
        # 결과 요약
        print("\n" + "=" * 80, flush=True)
        print("📊 실험 결과 비교", flush=True)
        print("=" * 80, flush=True)
        print(f"{'실험':<20} | {'Precision':<10} | {'Recall':<10} | {'F1':<10} | {'시간(s)':<10}", flush=True)
        print("-" * 80, flush=True)
        
        for result in results:
            exp_type = result['experiment_type']
            metrics = result['overall_metrics']
            print(f"{exp_type:<20} | {metrics['precision']:<10.3f} | {metrics['recall']:<10.3f} | {metrics['f1']:<10.3f} | {metrics['total_latency']:<10.1f}", flush=True)
        
        print("=" * 80, flush=True)
        print("\n✅ 모든 MS MARCO LLM 실험 완료!", flush=True)
            
    except Exception as e:
        print(f"❌ MS MARCO LLM 실험 중 오류 발생: {e}", flush=True)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
