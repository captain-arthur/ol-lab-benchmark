#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_llm_ms_marco.py
- MS MARCO 데이터셋에 대한 LLM 필터링 실험 실행
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

from filter.llm.llm_filter import run_ms_marco_llm_experiments

def main():
    """MS MARCO LLM 실험 실행"""
    parser = argparse.ArgumentParser(description='MS MARCO LLM Filtering Experiments')
    parser.add_argument('--max_queries', type=int, default=20, help='Maximum number of queries to process (default: 20)')
    
    args = parser.parse_args()
    
    print("🚀 MS MARCO LLM Filtering Experiments", flush=True)
    print("=" * 60, flush=True)
    print(f"📊 Processing {args.max_queries} queries", flush=True)
    
    try:
        # MS MARCO 실험 실행
        results = run_ms_marco_llm_experiments(max_queries=args.max_queries)
        
        if results:
            print("\n✅ MS MARCO LLM 실험 완료!", flush=True)
            print(f"📊 총 {len(results)} 개의 실험 방식 테스트 완료", flush=True)
            
            # 최고 성능 방법 찾기
            best_f1 = 0
            best_method = ""
            for name, exp in results.items():
                f1 = exp['overall_metrics']['avg_f1']
                if f1 > best_f1:
                    best_f1 = f1
                    best_method = name
            
            method_names = {
                'baseline': 'LLM Only (Baseline)',
                'lite_route': 'LLM + Lite Route',
                'setfit': 'LLM + Lite + SetFit',
                'cbc': 'LLM + Lite + SetFit + CBC (Proposed)'
            }
            
            print(f"🏆 최고 성능: {method_names.get(best_method, best_method)} (F1: {best_f1:.3f})", flush=True)
        else:
            print("❌ MS MARCO LLM 실험 실패", flush=True)
            
    except Exception as e:
        print(f"❌ MS MARCO LLM 실험 중 오류 발생: {e}", flush=True)

if __name__ == "__main__":
    main()
