#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
compare_models.py
- Ollama vs Gemini 2.5 모델 성능 비교
"""

import subprocess
import json
import os
import sys
from datetime import datetime

def run_experiment(api_mode, max_queries=5, threshold=0.6):
    """실험 실행 및 결과 반환"""
    print(f"\n🧪 {api_mode.upper()} 실험 시작 (쿼리 {max_queries}개, 임계값 {threshold})")
    print("=" * 60)
    
    cmd = [
        "uv", "run", "filter/llm/run_llm_ms_marco.py",
        "--max_queries", str(max_queries),
        "--api_mode", api_mode,
        "--threshold", str(threshold)
    ]
    
    # Gemini API 키 설정
    if api_mode == "gemini":
        env = os.environ.copy()
        env["GOOGLE_API_KEY"] = "AIzaSyC5uQZK0RrnAN3rEHJnkMdXs6Il2wCol9I"
        result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    else:
        result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ {api_mode} 실험 실패:")
        print(result.stderr)
        return None
    
    # 결과 파일에서 메트릭 읽기
    try:
        with open("results/llm/ms_marco/llm_baseline_test_results.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            return {
                "api_mode": api_mode,
                "precision": data.get("precision", 0),
                "recall": data.get("recall", 0),
                "f1_score": data.get("f1_score", 0),
                "avg_latency": data.get("avg_latency", 0),
                "avg_cost": data.get("avg_cost", 0),
                "threshold_used": data.get("threshold_used", threshold)
            }
    except Exception as e:
        print(f"❌ 결과 파일 읽기 실패: {e}")
        return None

def main():
    print("🚀 Ollama vs Gemini 2.5 모델 성능 비교")
    print("=" * 60)
    print(f"📅 시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 실험 설정
    max_queries = 5
    threshold = 0.6
    
    # Ollama 실험
    ollama_results = run_experiment("ollama", max_queries, threshold)
    
    # Gemini 실험
    gemini_results = run_experiment("gemini", max_queries, threshold)
    
    # 결과 비교
    print("\n" + "=" * 60)
    print("📊 성능 비교 결과")
    print("=" * 60)
    
    if ollama_results and gemini_results:
        print(f"{'메트릭':<20} {'Ollama':<15} {'Gemini 2.5':<15} {'승자':<10}")
        print("-" * 60)
        
        # F1 Score 비교
        ollama_f1 = ollama_results["f1_score"]
        gemini_f1 = gemini_results["f1_score"]
        f1_winner = "Ollama" if ollama_f1 > gemini_f1 else "Gemini 2.5" if gemini_f1 > ollama_f1 else "동점"
        print(f"{'F1 Score':<20} {ollama_f1:<15.3f} {gemini_f1:<15.3f} {f1_winner:<10}")
        
        # Precision 비교
        ollama_prec = ollama_results["precision"]
        gemini_prec = gemini_results["precision"]
        prec_winner = "Ollama" if ollama_prec > gemini_prec else "Gemini 2.5" if gemini_prec > ollama_prec else "동점"
        print(f"{'Precision':<20} {ollama_prec:<15.3f} {gemini_prec:<15.3f} {prec_winner:<10}")
        
        # Recall 비교
        ollama_recall = ollama_results["recall"]
        gemini_recall = gemini_results["recall"]
        recall_winner = "Ollama" if ollama_recall > gemini_recall else "Gemini 2.5" if gemini_recall > ollama_recall else "동점"
        print(f"{'Recall':<20} {ollama_recall:<15.3f} {gemini_recall:<15.3f} {recall_winner:<10}")
        
        # Latency 비교
        ollama_lat = ollama_results["avg_latency"]
        gemini_lat = gemini_results["avg_latency"]
        lat_winner = "Ollama" if ollama_lat < gemini_lat else "Gemini 2.5" if gemini_lat < ollama_lat else "동점"
        print(f"{'Avg Latency (s)':<20} {ollama_lat:<15.3f} {gemini_lat:<15.3f} {lat_winner:<10}")
        
        # Cost 비교
        ollama_cost = ollama_results["avg_cost"]
        gemini_cost = gemini_results["avg_cost"]
        cost_winner = "Ollama" if ollama_cost < gemini_cost else "Gemini 2.5" if gemini_cost < ollama_cost else "동점"
        print(f"{'Avg Cost':<20} {ollama_cost:<15.3f} {gemini_cost:<15.3f} {cost_winner:<10}")
        
        # 종합 평가
        print("\n" + "=" * 60)
        print("🏆 종합 평가")
        print("=" * 60)
        
        # 가중치 점수 계산 (F1: 40%, Precision: 30%, Recall: 20%, Latency: 10%)
        ollama_score = (ollama_f1 * 0.4 + ollama_prec * 0.3 + ollama_recall * 0.2 + (1.0 / max(ollama_lat, 0.1)) * 0.1)
        gemini_score = (gemini_f1 * 0.4 + gemini_prec * 0.3 + gemini_recall * 0.2 + (1.0 / max(gemini_lat, 0.1)) * 0.1)
        
        print(f"Ollama 종합 점수: {ollama_score:.3f}")
        print(f"Gemini 2.5 종합 점수: {gemini_score:.3f}")
        
        if ollama_score > gemini_score:
            print("🥇 승자: Ollama (gemma3:latest)")
        elif gemini_score > ollama_score:
            print("🥇 승자: Gemini 2.5 Flash")
        else:
            print("🤝 동점")
            
        # 상세 분석
        print("\n📈 상세 분석:")
        if ollama_f1 > gemini_f1:
            print(f"• F1 Score: Ollama가 {ollama_f1 - gemini_f1:.3f} 높음")
        else:
            print(f"• F1 Score: Gemini 2.5가 {gemini_f1 - ollama_f1:.3f} 높음")
            
        if ollama_lat < gemini_lat:
            print(f"• 속도: Ollama가 {gemini_lat - ollama_lat:.1f}초 빠름")
        else:
            print(f"• 속도: Gemini 2.5가 {ollama_lat - gemini_lat:.1f}초 빠름")
            
        # 결과 저장
        comparison_results = {
            "timestamp": datetime.now().isoformat(),
            "experiment_config": {
                "max_queries": max_queries,
                "threshold": threshold
            },
            "ollama_results": ollama_results,
            "gemini_results": gemini_results,
            "comparison": {
                "ollama_score": ollama_score,
                "gemini_score": gemini_score,
                "winner": "Ollama" if ollama_score > gemini_score else "Gemini 2.5" if gemini_score > ollama_score else "Tie"
            }
        }
        
        with open("results/llm/ms_marco/model_comparison_results.json", "w", encoding="utf-8") as f:
            json.dump(comparison_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 비교 결과 저장: results/llm/ms_marco/model_comparison_results.json")
        
    else:
        print("❌ 실험 결과를 비교할 수 없습니다.")
        if not ollama_results:
            print("• Ollama 실험 실패")
        if not gemini_results:
            print("• Gemini 실험 실패")
    
    print(f"\n✅ 비교 완료: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()