#!/usr/bin/env python3
"""
실험 결과 표 생성기
"""

import json
import os

def create_results_table():
    """실험 결과 표 생성"""
    
    # 결과 파일 경로
    results_dir = "results/semantic/fiqa"
    files = {
        "SBERT/CE (Baseline)": "sbert_ce_baseline_results.json",
        "SBERT/CE + SDE": "sbert_ce_sde_results.json", 
        "SBERT/CE + CBC": "sbert_ce_cbc_results.json",
        "SBERT/CE + SDE + CBC (Proposed)": "sbert_ce_sde_cbc_results.json"
    }
    
    print("=" * 80)
    print("📊 4단계 유사도 필터 성능 비교 (20개 쿼리 실험)")
    print("=" * 80)
    print("📈 핵심 지표 (확실한 오답 제거 검증)")
    print("-" * 80)
    print(f"{'방식':<35} {'Drop Precision':<15} {'Drop Recall':<12} {'Drop F1':<10} {'P@1':<8} {'P@10':<8}")
    print("-" * 80)
    
    for method, filename in files.items():
        filepath = os.path.join(results_dir, filename)
        
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # 필터 메트릭 추출
            filter_metrics = data.get('filter_metrics', {})
            metrics = data.get('metrics', {})
            
            drop_precision = filter_metrics.get('Drop_Precision', 0.0)
            drop_recall = filter_metrics.get('Drop_Recall', 0.0)
            drop_f1 = filter_metrics.get('Drop_F1', 0.0)
            p_at_1 = metrics.get('P@1', 0.0)
            p_at_10 = metrics.get('P@10', 0.0)
            
            print(f"{method:<35} {drop_precision:<15.3f} {drop_recall:<12.3f} {drop_f1:<10.3f} {p_at_1:<8.3f} {p_at_10:<8.3f}")
        else:
            print(f"{method:<35} {'N/A':<15} {'N/A':<12} {'N/A':<10} {'N/A':<8} {'N/A':<8}")
    
    print("-" * 80)
    print("📈 실험 결과 요약")
    print("=" * 80)
    print("• Drop Precision: 필터가 제거한 문서 중 실제로 무관한 문서의 비율")
    print("• Drop Recall: 전체 무관 문서 중에서 필터가 실제로 제거한 비율") 
    print("• Drop F1: Drop Precision과 Drop Recall의 조화 평균")
    print("• P@1, P@10: 상위 랭크된 결과 중 정답 문서의 비율 (보조 지표)")
    print("=" * 80)

if __name__ == "__main__":
    create_results_table()
