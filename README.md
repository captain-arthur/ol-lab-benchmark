# OL-Lab Benchmark

LLM 필터링 및 라우팅 벤치마크 시스템

## 기능

- **다중 아키텍처 비교**: Heavy-only, Lite-then-route, SetFit, CBC-enhanced
- **다양한 태스크 지원**: 분류, 회귀, 검색
- **완전한 성능 지표**: Accuracy, ECE, Routing Quality, Search Metrics (P@1, R@10, MRR@10, nDCG@10, MAP@100)
- **비용 효율성 분석**: 토큰 사용량, 지연시간, 비용 절약율

## 실행

```bash
uv run main.py
```

## 결과

결과는 `results/llm_filter/` 디렉토리에 저장됩니다:
- `summary_all.json`: 전체 결과 요약
- 각 데이터셋별 상세 결과

## 아키텍처

- **A (Heavy-only)**: Heavy 모델만 사용
- **B (Lite-then-route)**: Lite 모델 + 라우팅
- **C (SetFit)**: SetFit + 라우팅  
- **D (CBC-enhanced)**: CBC 검증 + 라우팅
