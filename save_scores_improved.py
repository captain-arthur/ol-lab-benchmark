#!/usr/bin/env python3
"""
개선된 점수 저장기 - 중복 문제 해결 및 올바른 경로에 저장
"""

import json
import pandas as pd
from pathlib import Path
import numpy as np

def save_scores_improved(json_file_path=None, output_csv_path="results/semantic/fiqa/score.csv"):
    """
    JSON 점수 데이터를 개선된 CSV 배열 형태로 저장
    중복 문제 해결 및 올바른 경로에 저장
    """
    if json_file_path is None:
        json_file_path = "results/semantic/fiqa/score.json"
    
    print("🚀 개선된 점수 저장기")
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
    
    print("🔄 점수 배열 데이터 처리 중...")
    
    all_arrays = []
    
    for method_name, method_data in data['scores'].items():
        for query_id, query_data in method_data.items():
            if query_id.startswith('query_'):
                query_num = int(query_id.split('_')[1])
                
                # SBERT 점수 처리
                if 'sbert_scores' in query_data and query_data['sbert_scores'] != 'skip':
                    sbert_scores = query_data['sbert_scores']
                    
                    # 중복 문제 진단
                    unique_scores = len(set(sbert_scores))
                    duplicate_rate = (len(sbert_scores) - unique_scores) / len(sbert_scores) * 100
                    
                    print(f"  📊 {method_name} - 쿼리 {query_num}:")
                    print(f"    - 배열 길이: {len(sbert_scores)}")
                    print(f"    - 고유값 개수: {unique_scores}")
                    print(f"    - 중복률: {duplicate_rate:.1f}%")
                    
                    if duplicate_rate > 90:
                        print(f"    ⚠️  높은 중복률 감지! 이는 문서 임베딩 중복 문제일 수 있습니다.")
                    
                    # 소수점 2자리로 반올림
                    sbert_scores_rounded = [round(score, 2) for score in sbert_scores]
                    
                    all_arrays.append({
                        'method': method_name,
                        'query_number': query_num,
                        'score_type': 'sbert',
                        'scores_array': sbert_scores_rounded,
                        'array_length': len(sbert_scores_rounded),
                        'unique_count': unique_scores,
                        'duplicate_rate': round(duplicate_rate, 1),
                        'min_score': round(min(sbert_scores_rounded), 2),
                        'max_score': round(max(sbert_scores_rounded), 2),
                        'mean_score': round(sum(sbert_scores_rounded) / len(sbert_scores_rounded), 2)
                    })
                
                # 앵커 점수 처리 (있는 경우)
                if 'anchor_scores' in query_data and query_data['anchor_scores'] != 'skip':
                    anchor_scores = query_data['anchor_scores']
                    anchor_scores_rounded = [round(score, 2) for score in anchor_scores]
                    
                    all_arrays.append({
                        'method': method_name,
                        'query_number': query_num,
                        'score_type': 'anchor',
                        'scores_array': anchor_scores_rounded,
                        'array_length': len(anchor_scores_rounded),
                        'unique_count': len(set(anchor_scores_rounded)),
                        'duplicate_rate': round((len(anchor_scores_rounded) - len(set(anchor_scores_rounded))) / len(anchor_scores_rounded) * 100, 1),
                        'min_score': round(min(anchor_scores_rounded), 2),
                        'max_score': round(max(anchor_scores_rounded), 2),
                        'mean_score': round(sum(anchor_scores_rounded) / len(anchor_scores_rounded), 2)
                    })
    
    # DataFrame 생성
    df = pd.DataFrame(all_arrays)
    
    # 출력 디렉토리 생성
    output_path = Path(output_csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # CSV 저장
    df.to_csv(output_csv_path, index=False, encoding='utf-8')
    
    print(f"\n✅ 총 {len(all_arrays)} 개의 점수 배열이 저장되었습니다.")
    print(f"📁 저장 위치: {output_csv_path}")
    print(f"📊 파일 크기: {Path(output_csv_path).stat().st_size / 1024:.1f} KB")
    print(f"📋 컬럼: {', '.join(df.columns.tolist())}")
    
    # 통계 정보
    print(f"\n📈 데이터 통계:")
    print(f"  - 방법 수: {df['method'].nunique()}")
    print(f"  - 쿼리 수: {df['query_number'].nunique()}")
    print(f"  - 점수 타입: {df['score_type'].unique().tolist()}")
    print(f"  - 평균 배열 길이: {df['array_length'].mean():.0f}")
    print(f"  - 평균 중복률: {df['duplicate_rate'].mean():.1f}%")
    
    # 중복 문제 요약
    high_duplicate = df[df['duplicate_rate'] > 90]
    if len(high_duplicate) > 0:
        print(f"\n⚠️  높은 중복률 쿼리들:")
        for _, row in high_duplicate.iterrows():
            print(f"  - {row['method']} 쿼리 {row['query_number']}: {row['duplicate_rate']}% 중복")
    
    # 미리보기
    print(f"\n👀 데이터 미리보기:")
    for i, row in df.head(3).iterrows():
        scores_preview = str(row['scores_array'])[:100] + "..." if len(str(row['scores_array'])) > 100 else str(row['scores_array'])
        print(f"📊 {row['method']} - 쿼리 {row['query_number']} ({row['score_type']})")
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
    
    parser = argparse.ArgumentParser(description="개선된 점수 저장기")
    parser.add_argument("--input", default="results/semantic/fiqa/score.json", help="입력 JSON 파일 경로")
    parser.add_argument("--output", default="results/semantic/fiqa/score.csv", help="출력 CSV 파일 경로")
    
    args = parser.parse_args()
    
    save_scores_improved(args.input, args.output)
