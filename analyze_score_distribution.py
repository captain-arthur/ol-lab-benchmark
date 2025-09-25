#!/usr/bin/env python3
"""
점수 분포도 분석기
SBERT/CE vs SBERT/CE + SDE 간의 분포 차이 분석
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import json

def analyze_score_distribution():
    """점수 분포도 분석"""
    
    # CSV 파일 읽기
    df = pd.read_csv('results/semantic/fiqa/score.csv')
    
    print("=" * 80)
    print("📊 점수 분포도 분석: SBERT/CE vs SBERT/CE + SDE")
    print("=" * 80)
    
    # 1. SBERT/CE (Baseline) 분석
    baseline_data = df[df['method'] == 'SBERT/CE (Baseline)']
    baseline_scores = []
    for scores_str in baseline_data['scores_array']:
        scores = eval(scores_str)  # 문자열을 리스트로 변환
        baseline_scores.extend(scores)
    
    # 2. SBERT/CE + SDE 분석
    sde_data = df[df['method'] == 'SBERT/CE + SDE']
    sde_scores = []
    for scores_str in sde_data['scores_array']:
        scores = eval(scores_str)
        sde_scores.extend(scores)
    
    # 3. 통계 분석
    print("\n📈 기본 통계:")
    print("-" * 50)
    print(f"SBERT/CE (Baseline):")
    print(f"  - 총 점수 개수: {len(baseline_scores)}")
    print(f"  - 평균: {np.mean(baseline_scores):.3f}")
    print(f"  - 표준편차: {np.std(baseline_scores):.3f}")
    print(f"  - 최솟값: {np.min(baseline_scores):.3f}")
    print(f"  - 최댓값: {np.max(baseline_scores):.3f}")
    print(f"  - 범위: {np.max(baseline_scores) - np.min(baseline_scores):.3f}")
    
    print(f"\nSBERT/CE + SDE:")
    print(f"  - 총 점수 개수: {len(sde_scores)}")
    print(f"  - 평균: {np.mean(sde_scores):.3f}")
    print(f"  - 표준편차: {np.std(sde_scores):.3f}")
    print(f"  - 최솟값: {np.min(sde_scores):.3f}")
    print(f"  - 최댓값: {np.max(sde_scores):.3f}")
    print(f"  - 범위: {np.max(sde_scores) - np.min(sde_scores):.3f}")
    
    # 4. 분포 비교
    print("\n🔍 분포 비교:")
    print("-" * 50)
    std_diff = np.std(sde_scores) - np.std(baseline_scores)
    range_diff = (np.max(sde_scores) - np.min(sde_scores)) - (np.max(baseline_scores) - np.min(baseline_scores))
    
    print(f"표준편차 차이: {std_diff:+.3f} ({'확장' if std_diff > 0 else '축소'})")
    print(f"범위 차이: {range_diff:+.3f} ({'확장' if range_diff > 0 else '축소'})")
    
    # 5. 점수 구간별 분포
    print("\n📊 점수 구간별 분포:")
    print("-" * 50)
    
    def get_score_ranges(scores):
        ranges = {
            "0.0-0.2": 0, "0.2-0.4": 0, "0.4-0.6": 0, 
            "0.6-0.8": 0, "0.8-1.0": 0
        }
        for score in scores:
            if 0.0 <= score < 0.2:
                ranges["0.0-0.2"] += 1
            elif 0.2 <= score < 0.4:
                ranges["0.2-0.4"] += 1
            elif 0.4 <= score < 0.6:
                ranges["0.4-0.6"] += 1
            elif 0.6 <= score < 0.8:
                ranges["0.6-0.8"] += 1
            elif 0.8 <= score <= 1.0:
                ranges["0.8-1.0"] += 1
        return ranges
    
    baseline_ranges = get_score_ranges(baseline_scores)
    sde_ranges = get_score_ranges(sde_scores)
    
    print("구간별 점수 개수:")
    for range_name in baseline_ranges.keys():
        baseline_count = baseline_ranges[range_name]
        sde_count = sde_ranges[range_name]
        diff = sde_count - baseline_count
        print(f"  {range_name}: Baseline {baseline_count:3d} → SDE {sde_count:3d} ({diff:+3d})")
    
    # 6. 쿼리별 상세 분석
    print("\n🔬 쿼리별 상세 분석:")
    print("-" * 50)
    
    for i in range(1, 6):  # 5개 쿼리
        baseline_query = baseline_data[baseline_data['query_number'] == i]
        sde_query = sde_data[sde_data['query_number'] == i]
        
        if not baseline_query.empty and not sde_query.empty:
            baseline_scores_query = eval(baseline_query.iloc[0]['scores_array'])
            sde_scores_query = eval(sde_query.iloc[0]['scores_array'])
            
            print(f"Query {i}:")
            print(f"  Baseline: {baseline_scores_query} (std: {np.std(baseline_scores_query):.3f})")
            print(f"  SDE:      {sde_scores_query} (std: {np.std(sde_scores_query):.3f})")
            print(f"  차이:     {np.std(sde_scores_query) - np.std(baseline_scores_query):+.3f}")
    
    # 7. 결론
    print("\n🎯 결론:")
    print("-" * 50)
    
    if abs(std_diff) < 0.01 and abs(range_diff) < 0.01:
        print("❌ SDE가 분포를 확장하지 못했습니다!")
        print("   - 표준편차와 범위가 거의 동일")
        print("   - 앵커가 원본 쿼리보다 낮은 유사도를 가짐")
        print("   - Max pooling으로 인해 원본 점수가 그대로 유지됨")
    elif std_diff > 0.01 or range_diff > 0.01:
        print("✅ SDE가 분포를 확장했습니다!")
        print(f"   - 표준편차 증가: {std_diff:+.3f}")
        print(f"   - 범위 확장: {range_diff:+.3f}")
    else:
        print("⚠️  SDE의 효과가 미미합니다.")
    
    return baseline_scores, sde_scores

if __name__ == "__main__":
    baseline_scores, sde_scores = analyze_score_distribution()
