# 🔧 개선 프로세스 규칙 (IMPROVEMENT RULES)

## 📋 개선 프로세스 체계

### **1단계: 문제 진단 및 테스트 파일 생성**
```bash
# 1. 문제가 있는 데이터셋/기능 식별
# 2. test_[dataset/feature]_debug.py 파일 생성
# 3. 다양한 접근 방식 테스트
# 4. 결과 분석 및 최적 방법 선택
```

### **2단계: 단계별 개선 적용**
```bash
# 1. test_*.py로 검증된 로직을 run_llm.py에 적용
# 2. 각 단계마다 uv run main.py로 테스트
# 3. 성능 개선 확인 후 다음 단계 진행
# 4. 문제 발생 시 이전 단계로 롤백
```

### **3단계: 결과 검증 및 문서화**
```bash
# 1. 최종 성능 측정
# 2. 개선 사항 문서화
# 3. 다음 개선 영역 식별
```

## 🎯 테스트 파일 명명 규칙

### **데이터셋별 테스트**
- `test_[dataset]_debug.py` - 기본 문제 진단
- `test_[dataset]_mapping.py` - 라벨 매핑 개선
- `test_[dataset]_prompt.py` - 프롬프트 최적화
- `test_[dataset]_routing.py` - 라우팅 전략 개선

### **기능별 테스트**
- `test_confidence_extraction.py` - 신뢰도 추출 개선
- `test_threshold_optimization.py` - 임계값 최적화
- `test_setfit_integration.py` - SetFit 통합 개선

## 📊 테스트 파일 템플릿

### **기본 구조**
```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
[데이터셋/기능] [문제/개선사항] 테스트
"""

import time
import json
from typing import List, Tuple, Dict, Any

class Ollama:
    def __init__(self, base_url="http://192.168.45.166:11434"):
        self.base_url = base_url
        
    def generate(self, model: str, prompt: str) -> Dict[str, Any]:
        import requests
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.0,
                        "top_p": 1.0
                    }
                },
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Ollama API 오류: {e}")
            return {"response": "", "usage": {}}

def test_[feature]():
    """[기능] 테스트"""
    print("🔬 [데이터셋/기능] [문제/개선사항] 테스트")
    print("=" * 60)
    
    # 1. 다양한 접근 방식 정의
    approaches = [
        {
            "name": "기본 방식",
            "description": "현재 사용 중인 방식"
        },
        {
            "name": "개선 방식 1",
            "description": "첫 번째 개선 접근"
        },
        {
            "name": "개선 방식 2", 
            "description": "두 번째 개선 접근"
        }
    ]
    
    # 2. 각 방식별 테스트
    results = {}
    for approach in approaches:
        print(f"\n📊 {approach['name']} 테스트")
        print("-" * 40)
        
        # 테스트 로직 구현
        # ...
        
        # 결과 저장
        results[approach["name"]] = {
            "accuracy": accuracy,
            "latency": latency,
            "details": details
        }
    
    # 3. 결과 저장
    with open("[dataset]_[feature]_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 4. 최적 방식 선택
    best_approach = max(results.keys(), key=lambda k: results[k]["accuracy"])
    print(f"\n🏆 최적 방식: {best_approach}")

if __name__ == "__main__":
    test_[feature]()
```

## 🔄 개선 사이클

### **Cycle 1: 데이터셋별 개선**
1. `test_[dataset]_debug.py` - 문제 진단
2. `test_[dataset]_mapping.py` - 라벨 매핑 개선
3. `test_[dataset]_prompt.py` - 프롬프트 최적화
4. `run_llm.py`에 적용 및 테스트

### **Cycle 2: 기능별 개선**
1. `test_confidence_extraction.py` - 신뢰도 추출 개선
2. `test_threshold_optimization.py` - 임계값 최적화
3. `test_setfit_integration.py` - SetFit 통합 개선
4. `run_llm.py`에 적용 및 테스트

### **Cycle 3: 통합 최적화**
1. 전체 시스템 성능 분석
2. 병목 지점 식별
3. 종합적 개선 적용
4. 최종 성능 검증

## 📈 성공 지표

### **정확도 개선**
- 목표: 각 데이터셋별 10% 이상 정확도 향상
- 측정: Before vs After 비교

### **효율성 개선**
- 목표: 비용 절약 80% 이상 유지
- 측정: Heavy LLM 호출률 감소

### **다양성 개선**
- 목표: 다양성 점수 0.5 이상
- 측정: 라벨 분포 균등성

## 🚨 문제 해결 가이드

### **정확도가 낮은 경우**
1. 프롬프트 최적화
2. 라벨 매핑 개선
3. 데이터 전처리 검토
4. 모델 파라미터 조정

### **라우팅이 비효율적인 경우**
1. 신뢰도 추출 로직 개선
2. 임계값 최적화
3. SetFit 통합 검토
4. 라우팅 전략 수정

### **다양성이 낮은 경우**
1. 샘플링 전략 개선
2. 라벨 분포 균등화
3. 데이터셋 로더 수정
4. 평가 메트릭 조정

## 📝 문서화 규칙

### **테스트 결과 문서화**
```markdown
## [데이터셋/기능] 개선 결과

### 문제 진단
- 문제점: [설명]
- 원인: [분석]
- 영향: [성능 지표]

### 개선 방법
- 접근법: [설명]
- 구현: [코드 변경사항]
- 검증: [테스트 결과]

### 결과
- 정확도: [Before] → [After]
- 효율성: [Before] → [After]
- 다양성: [Before] → [After]

### 다음 단계
- [ ] 추가 개선 영역
- [ ] 통합 테스트
- [ ] 문서 업데이트
```

## 🎯 우선순위

### **High Priority**
1. CLINC150 정확도 개선 (현재 10%)
2. STS-B 정확도 개선 (현재 0%)
3. Banking77 정확도 개선 (현재 30%)

### **Medium Priority**
1. 신뢰도 추출 로직 최적화
2. 임계값 동적 조정
3. SetFit 통합 안정화

### **Low Priority**
1. 성능 메트릭 추가
2. 시각화 개선
3. 문서화 완성

---

**마지막 업데이트**: 2024-12-19
**버전**: 1.0
**담당자**: AI Assistant
