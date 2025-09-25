#!/usr/bin/env python3
"""
그룹별 점수 저장기 - 각 방법별로 하나의 배열로 합쳐서 저장
"""

import json
import pandas as pd
from pathlib import Path
import numpy as np

def save_scores_grouped(json_file_path=None, output_csv_path="results/semantic/fiqa/score.csv", target_length=1000):
    """
    JSON 점수 데이터를 그룹별로 합쳐서 CSV 배열 형태로 저장
    각 방법별로 하나의 배열로 합치기
    모든 배열을 target_length개로 맞춤
    """
    if json_file_path is None:
        json_file_path = "results/semantic/fiqa/score.json"
    
    print("🚀 그룹별 점수 저장기")
    print("=" * 50)
    print(f"📁 JSON 파일 읽기: {json_file_path}")
    
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ 파일을 찾을 수 없습니다: {json_file_path}")
        return None
    except Exception as e:
        print(f"❌ 파일 읽기 오류: {e}")
        return None
    
    print("🔄 그룹별 점수 배열 합치기...")
    
    grouped_data = []
    
    for method_name, method_data in data['scores'].items():
        print(f"📊 {method_name} 처리 중...")
        
        # 각 방법별로 sbert와 anchor 점수를 모두 수집
        all_sbert_scores = []
        all_anchor_scores = []
        
        for query_id, query_data in method_data.items():
            if query_id.startswith('query_'):
                # SBERT 점수 수집
                if 'sbert_scores' in query_data and query_data['sbert_scores'] != 'skip':
                    all_sbert_scores.extend(query_data['sbert_scores'])
                
                # 앵커 점수 수집 (있는 경우)
                if 'anchor_scores' in query_data and query_data['anchor_scores'] != 'skip':
                    all_anchor_scores.extend(query_data['anchor_scores'])
        
        # SBERT 점수 배열 저장 (1000개로 맞춤)
        if all_sbert_scores:
            sbert_scores_rounded = [round(score, 2) for score in all_sbert_scores]
            
            # 1000개로 맞추기 (반복 또는 샘플링)
            if len(sbert_scores_rounded) < target_length:
                # 부족하면 반복해서 채우기
                repeat_times = (target_length // len(sbert_scores_rounded)) + 1
                sbert_scores_rounded = (sbert_scores_rounded * repeat_times)[:target_length]
            elif len(sbert_scores_rounded) > target_length:
                # 초과하면 랜덤 샘플링
                np.random.seed(42)  # 재현 가능한 결과
                sbert_scores_rounded = np.random.choice(sbert_scores_rounded, target_length, replace=False).tolist()
            
            unique_scores = len(set(sbert_scores_rounded))
            duplicate_rate = (len(sbert_scores_rounded) - unique_scores) / len(sbert_scores_rounded) * 100
            
            print(f"  - SBERT 점수: {len(sbert_scores_rounded)}개, 고유값: {unique_scores}개, 중복률: {duplicate_rate:.1f}%")
            
            grouped_data.append({
                'method': method_name,
                'score_type': 'sbert',
                'scores_array': sbert_scores_rounded,
                'array_length': len(sbert_scores_rounded),
                'unique_count': unique_scores,
                'duplicate_rate': round(duplicate_rate, 1),
                'min_score': round(min(sbert_scores_rounded), 2),
                'max_score': round(max(sbert_scores_rounded), 2),
                'mean_score': round(sum(sbert_scores_rounded) / len(sbert_scores_rounded), 2)
            })
        
        # 앵커 점수 배열 저장 (1000개로 맞춤)
        if all_anchor_scores:
            anchor_scores_rounded = [round(score, 2) for score in all_anchor_scores]
            
            # 1000개로 맞추기 (반복 또는 샘플링)
            if len(anchor_scores_rounded) < target_length:
                # 부족하면 반복해서 채우기
                repeat_times = (target_length // len(anchor_scores_rounded)) + 1
                anchor_scores_rounded = (anchor_scores_rounded * repeat_times)[:target_length]
            elif len(anchor_scores_rounded) > target_length:
                # 초과하면 랜덤 샘플링
                np.random.seed(42)  # 재현 가능한 결과
                anchor_scores_rounded = np.random.choice(anchor_scores_rounded, target_length, replace=False).tolist()
            
            unique_scores = len(set(anchor_scores_rounded))
            duplicate_rate = (len(anchor_scores_rounded) - unique_scores) / len(anchor_scores_rounded) * 100
            
            print(f"  - 앵커 점수: {len(anchor_scores_rounded)}개, 고유값: {unique_scores}개, 중복률: {duplicate_rate:.1f}%")
            
            grouped_data.append({
                'method': method_name,
                'score_type': 'anchor',
                'scores_array': anchor_scores_rounded,
                'array_length': len(anchor_scores_rounded),
                'unique_count': unique_scores,
                'duplicate_rate': round(duplicate_rate, 1),
                'min_score': round(min(anchor_scores_rounded), 2),
                'max_score': round(max(anchor_scores_rounded), 2),
                'mean_score': round(sum(anchor_scores_rounded) / len(anchor_scores_rounded), 2)
            })
        
        # SBERT/CE + SDE의 경우 sbert와 anchor를 하나로 합치기 (1000개로 맞춤)
        if method_name == "SBERT/CE + SDE" and all_sbert_scores and all_anchor_scores:
            combined_scores = all_sbert_scores + all_anchor_scores
            combined_scores_rounded = [round(score, 2) for score in combined_scores]
            
            # 1000개로 맞추기 (반복 또는 샘플링)
            if len(combined_scores_rounded) < target_length:
                # 부족하면 반복해서 채우기
                repeat_times = (target_length // len(combined_scores_rounded)) + 1
                combined_scores_rounded = (combined_scores_rounded * repeat_times)[:target_length]
            elif len(combined_scores_rounded) > target_length:
                # 초과하면 랜덤 샘플링
                np.random.seed(42)  # 재현 가능한 결과
                combined_scores_rounded = np.random.choice(combined_scores_rounded, target_length, replace=False).tolist()
            
            unique_scores = len(set(combined_scores_rounded))
            duplicate_rate = (len(combined_scores_rounded) - unique_scores) / len(combined_scores_rounded) * 100
            
            print(f"  - 합쳐진 점수: {len(combined_scores_rounded)}개, 고유값: {unique_scores}개, 중복률: {duplicate_rate:.1f}%")
            
            grouped_data.append({
                'method': method_name,
                'score_type': 'combined',
                'scores_array': combined_scores_rounded,
                'array_length': len(combined_scores_rounded),
                'unique_count': unique_scores,
                'duplicate_rate': round(duplicate_rate, 1),
                'min_score': round(min(combined_scores_rounded), 2),
                'max_score': round(max(combined_scores_rounded), 2),
                'mean_score': round(sum(combined_scores_rounded) / len(combined_scores_rounded), 2)
            })
    
    # DataFrame 생성
    df = pd.DataFrame(grouped_data)
    
    # 출력 디렉토리 생성
    output_path = Path(output_csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # CSV 저장
    df.to_csv(output_csv_path, index=False, encoding='utf-8')
    
    print(f"\n✅ 총 {len(grouped_data)} 개의 그룹별 점수 배열이 저장되었습니다.")
    print(f"📁 저장 위치: {output_csv_path}")
    print(f"📊 파일 크기: {Path(output_csv_path).stat().st_size / 1024:.1f} KB")
    print(f"📋 컬럼: {', '.join(df.columns.tolist())}")
    
    # 통계 정보
    print(f"\n📈 데이터 통계:")
    print(f"  - 방법 수: {df['method'].nunique()}")
    print(f"  - 점수 타입: {df['score_type'].unique().tolist()}")
    print(f"  - 평균 배열 길이: {df['array_length'].mean():.0f}")
    print(f"  - 평균 중복률: {df['duplicate_rate'].mean():.1f}%")
    
    # 미리보기
    print(f"\n👀 그룹별 데이터 미리보기:")
    for i, row in df.iterrows():
        scores_preview = str(row['scores_array'])[:200] + "..." if len(str(row['scores_array'])) > 200 else str(row['scores_array'])
        print(f"📊 {row['method']} ({row['score_type']})")
        print(f"   배열 길이: {row['array_length']}")
        print(f"   고유값 개수: {row['unique_count']}")
        print(f"   중복률: {row['duplicate_rate']}%")
        print(f"   점수 범위: {row['min_score']} ~ {row['max_score']}")
        print(f"   평균: {row['mean_score']}")
        print(f"   배열: {scores_preview}")
        print()
    
    print("✅ 저장 완료!")
    return df

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="그룹별 점수 저장기")
    parser.add_argument("--input", default="results/semantic/fiqa/score.json", help="입력 JSON 파일 경로")
    parser.add_argument("--output", default="results/semantic/fiqa/score.csv", help="출력 CSV 파일 경로")
    parser.add_argument("--target-length", type=int, default=1000, help="목표 배열 길이 (기본값: 1000)")
    
    args = parser.parse_args()
    
    save_scores_grouped(args.input, args.output, args.target_length)
