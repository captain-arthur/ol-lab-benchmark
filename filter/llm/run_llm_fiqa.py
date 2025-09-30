#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_llm_fiqa.py
- FiQA 데이터셋에 대한 LLM 필터링 실험 실행
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from filter.llm.llm_filter import run_fiqa_llm_experiments

def main():
    """FiQA LLM 실험 실행"""
    print("🚀 FiQA LLM Filtering Experiments")
    print("=" * 60)
    
    try:
        # FiQA 실험 실행
        results = run_fiqa_llm_experiments(max_queries=648)
        
        if results:
            print("\n✅ FiQA LLM 실험 완료!")
            print(f"📊 총 {len(results)} 개의 실험 방식 테스트 완료")
            
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
            
            print(f"🏆 최고 성능: {method_names.get(best_method, best_method)} (F1: {best_f1:.3f})")
        else:
            print("❌ FiQA LLM 실험 실패")
            
    except Exception as e:
        print(f"❌ FiQA LLM 실험 중 오류 발생: {e}")

if __name__ == "__main__":
    main()
