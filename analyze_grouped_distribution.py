#!/usr/bin/env python3
"""
그룹별 점수 분포도 분석 - Baseline vs SDE Combined
"""

import pandas as pd
import numpy as np
import ast

def analyze_grouped_distribution(csv_path="results/semantic/fiqa/score.csv"):
    """그룹별 CSV 파일에서 Baseline과 SDE Combined의 점수 분포를 분석합니다."""
    try:
        scores_df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}")
        return

    print("=" * 80)
    print("📊 그룹별 점수 분포도 분석: Baseline vs SDE Combined")
    print("=" * 80)

    # Baseline 분석 (sbert만)
    baseline_data = scores_df[(scores_df['method'] == "SBERT/CE (Baseline)") & (scores_df['score_type'] == 'sbert')]
    if len(baseline_data) == 0:
        print("❌ Baseline 데이터를 찾을 수 없습니다.")
        return
    
    baseline_scores = ast.literal_eval(baseline_data['scores_array'].iloc[0])
    baseline_stats = {
        "total_count": len(baseline_scores),
        "mean": np.mean(baseline_scores),
        "std": np.std(baseline_scores),
        "min": np.min(baseline_scores),
        "max": np.max(baseline_scores),
        "range": np.max(baseline_scores) - np.min(baseline_scores)
    }

    # SDE Combined 분석
    sde_combined_data = scores_df[(scores_df['method'] == "SBERT/CE + SDE") & (scores_df['score_type'] == 'combined')]
    if len(sde_combined_data) == 0:
        print("❌ SDE Combined 데이터를 찾을 수 없습니다.")
        return
    
    sde_combined_scores = ast.literal_eval(sde_combined_data['scores_array'].iloc[0])
    sde_combined_stats = {
        "total_count": len(sde_combined_scores),
        "mean": np.mean(sde_combined_scores),
        "std": np.std(sde_combined_scores),
        "min": np.min(sde_combined_scores),
        "max": np.max(sde_combined_scores),
        "range": np.max(sde_combined_scores) - np.min(sde_combined_scores)
    }

    print("\n📈 기본 통계:")
    print("--------------------------------------------------")
    print("SBERT/CE (Baseline):")
    print(f"  - 총 점수 개수: {baseline_stats['total_count']}")
    print(f"  - 평균: {baseline_stats['mean']:.3f}")
    print(f"  - 표준편차: {baseline_stats['std']:.3f}")
    print(f"  - 최솟값: {baseline_stats['min']:.3f}")
    print(f"  - 최댓값: {baseline_stats['max']:.3f}")
    print(f"  - 범위: {baseline_stats['range']:.3f}")

    print("\nSBERT/CE + SDE (Combined):")
    print(f"  - 총 점수 개수: {sde_combined_stats['total_count']}")
    print(f"  - 평균: {sde_combined_stats['mean']:.3f}")
    print(f"  - 표준편차: {sde_combined_stats['std']:.3f}")
    print(f"  - 최솟값: {sde_combined_stats['min']:.3f}")
    print(f"  - 최댓값: {sde_combined_stats['max']:.3f}")
    print(f"  - 범위: {sde_combined_stats['range']:.3f}")

    print("\n🔍 분포 비교:")
    print("--------------------------------------------------")
    std_diff = sde_combined_stats['std'] - baseline_stats['std']
    range_diff = sde_combined_stats['range'] - baseline_stats['range']
    print(f"표준편차 차이: {std_diff:.3f} ({'확장' if std_diff > 0 else '축소'})")
    print(f"범위 차이: {range_diff:.3f} ({'확장' if range_diff > 0 else '축소'})")

    # 구간별 분포 분석
    print("\n📊 점수 구간별 분포:")
    print("--------------------------------------------------")
    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    baseline_hist, _ = np.histogram(baseline_scores, bins=bins)
    sde_hist, _ = np.histogram(sde_combined_scores, bins=bins)
    
    bins_labels = ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]
    for i, label in enumerate(bins_labels):
        diff = sde_hist[i] - baseline_hist[i]
        print(f"  {label}: Baseline {baseline_hist[i]:>3} → SDE {sde_hist[i]:>3} ({diff:>+3})")

    # 분포 확장 효과 분석
    print("\n🎯 SDE 앵커 확장 효과 분석:")
    print("--------------------------------------------------")
    
    # SDE의 sbert와 anchor 점수도 개별 분석
    sde_sbert_data = scores_df[(scores_df['method'] == "SBERT/CE + SDE") & (scores_df['score_type'] == 'sbert')]
    sde_anchor_data = scores_df[(scores_df['method'] == "SBERT/CE + SDE") & (scores_df['score_type'] == 'anchor')]
    
    if len(sde_sbert_data) > 0 and len(sde_anchor_data) > 0:
        sde_sbert_scores = ast.literal_eval(sde_sbert_data['scores_array'].iloc[0])
        sde_anchor_scores = ast.literal_eval(sde_anchor_data['scores_array'].iloc[0])
        
        print(f"원본 쿼리 점수 (SBERT): {len(sde_sbert_scores)}개, 범위: {np.min(sde_sbert_scores):.3f} ~ {np.max(sde_sbert_scores):.3f}")
        print(f"앵커 점수: {len(sde_anchor_scores)}개, 범위: {np.min(sde_anchor_scores):.3f} ~ {np.max(sde_anchor_scores):.3f}")
        print(f"합쳐진 점수: {len(sde_combined_scores)}개, 범위: {np.min(sde_combined_scores):.3f} ~ {np.max(sde_combined_scores):.3f}")
        
        # 앵커 확장 효과 계산
        original_range = np.max(sde_sbert_scores) - np.min(sde_sbert_scores)
        combined_range = np.max(sde_combined_scores) - np.min(sde_combined_scores)
        expansion_effect = combined_range - original_range
        
        print(f"\n앵커 확장 효과:")
        print(f"  - 원본 범위: {original_range:.3f}")
        print(f"  - 확장된 범위: {combined_range:.3f}")
        print(f"  - 확장 효과: {expansion_effect:.3f}")

    print("\n🎯 결론:")
    print("--------------------------------------------------")
    if sde_combined_stats['range'] > baseline_stats['range']:
        print("✅ SDE가 분포를 확장했습니다!")
        print(f"   - 표준편차 변화: {std_diff:+.3f}")
        print(f"   - 범위 확장: {range_diff:+.3f}")
        print(f"   - 총 점수 증가: {sde_combined_stats['total_count'] - baseline_stats['total_count']}개")
    else:
        print("⚠️  SDE의 분포 확장 효과가 미미합니다.")
    
    print(f"\n📊 최종 비교:")
    print(f"  - Baseline: {baseline_stats['total_count']}개 점수, 범위 {baseline_stats['range']:.3f}")
    print(f"  - SDE Combined: {sde_combined_stats['total_count']}개 점수, 범위 {sde_combined_stats['range']:.3f}")

if __name__ == "__main__":
    analyze_grouped_distribution()
