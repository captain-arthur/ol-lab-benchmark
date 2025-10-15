#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_llm_fiqa.py
- FiQA 데이터셋에 대한 LLM 필터링 실험 실행
- 개선된 SDE + CBC 방법 적용
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from filter.llm.llm_filter import run_fiqa_llm_experiments

def main():
    """FiQA LLM 실험 실행"""
    print("🚀 FiQA LLM Filtering Experiments (개선된 SDE + CBC)")
    print("=" * 60)
    
    try:
        # FiQA 실험 실행 (2개 쿼리로 빠른 테스트)
        results = run_fiqa_llm_experiments(max_queries=2)
        
        if results:
            print("\n✅ FiQA LLM 실험 완료!")
            print(f"📊 총 {len(results)} 개의 실험 방식 테스트 완료")
            
            # 최고 성능 방법 찾기
            best_f1 = 0
            best_method = ""
            for result in results:
                exp_type = result['experiment_type']
                f1 = result['overall_metrics']['drop_f1']
                if f1 > best_f1:
                    best_f1 = f1
                    best_method = exp_type
            
            method_names = {
                'LLM_BASELINE': 'LLM Only (Baseline)',
                'LLM_SDE': 'LLM + SDE',
                'LLM_CBC': 'LLM + CBC',
                'LLM_SDE_CBC': 'LLM + SDE + CBC (Proposed)'
            }
            
            print(f"🏆 최고 성능: {method_names.get(best_method, best_method)} (Drop_F1: {best_f1:.1%})")
        else:
            print("❌ FiQA LLM 실험 실패")
            
    except Exception as e:
        print(f"❌ FiQA LLM 실험 중 오류 발생: {e}")

if __name__ == "__main__":
    main()
