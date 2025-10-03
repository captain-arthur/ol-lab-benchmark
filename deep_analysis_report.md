# 🔍 심층 분석 보고서: Ollama vs Gemini 2.5 Flash

## 📊 실험 개요

**쿼리**: "what is rba"  
**임계값**: 0.6  
**문서 수**: 10개  
**실제 관련 문서**: 1개 (문서 6번)  

---

## 🎯 실험 결과 요약

| 모델 | F1 Score | Precision | Recall | Latency | 
|------|----------|-----------|--------|---------|
| **Ollama (gemma3:latest)** | 33.3% | 20.0% | 100% | 94.8초 |
| **Gemini 2.5 Flash** | 18.2% | 10.0% | 100% | 28.5초 |

---

## 📝 문서 내용 분석

### 쿼리: "what is rba"

**실제 관련 문서 (Ground Truth)**: 문서 6번만 관련 (1개)

**문서별 내용**:
1. **RBA 스캔들** - 호주 중앙은행의 부패 스캔들
2. **RBA 정의** - 호주 중앙은행의 역사와 기능 설명
3. **Microsoft RBA** - IT 시스템 관련 RBA
4. **Rebuildable Atomizer (RBA)** - 전자담배 부품
5. **Results-Based Accountability (RBA)** - 사회복지 프로그램 방법론
6. **Results-Based Accountability (RBA)** - 사회복지 프로그램 방법론 (중복)
7. **Results-Based Accountability (RBA)** - 사회복지 프로그램 방법론 (중복)
8. **Risk-based Authentication (RBA)** - 보안 인증 방법
9. **Rebuildable Atomizer (RBA)** - 전자담배 부품 (중복)
10. **RBA 디지털 컨설팅** - 기술 컨설팅 회사

---

## 🤖 모델별 유사도 점수 분석

### Ollama (gemma3:latest) 응답
```
Document 1: [0.6]  - RBA 스캔들
Document 2: [0.8]  - RBA 정의 (정답!)
Document 3: [0.2]  - Microsoft RBA
Document 4: [0.2]  - 전자담배 RBA
Document 5: [0.6]  - 사회복지 RBA
Document 6: [0.8]  - 사회복지 RBA (중복)
Document 7: [0.6]  - 사회복지 RBA (중복)
Document 8: [0.2]  - 보안 RBA
Document 9: [0.4]  - 전자담배 RBA (중복)
Document 10: [0.2] - 컨설팅 RBA
```

### Gemini 2.5 Flash 응답
```
Document 1: 0.7   - RBA 스캔들
Document 2: 1.0   - RBA 정의 (정답!)
Document 3: 0.4   - Microsoft RBA
Document 4: 1.0   - 전자담배 RBA
Document 5: 1.0   - 사회복지 RBA
Document 6: 1.0   - 사회복지 RBA (중복)
Document 7: 0.9   - 사회복지 RBA (중복)
Document 8: 1.0   - 보안 RBA
Document 9: 1.0   - 전자담배 RBA (중복)
Document 10: 0.4  - 컨설팅 RBA
```

---

## 🔍 성능 차이의 정확한 원인 분석

### 1. **임계값 0.6 기준 Keep/Drop 결정**

#### Ollama 결정:
- **Keep**: 문서 1, 2, 5, 6, 7 (5개)
- **Drop**: 문서 3, 4, 8, 9, 10 (5개)
- **결과**: TP=1, FP=4, FN=0
- **Precision**: 1/5 = 20%, **Recall**: 1/1 = 100%

#### Gemini 결정:
- **Keep**: 문서 1, 2, 4, 5, 6, 7, 8, 9 (8개)
- **Drop**: 문서 3, 10 (2개)
- **결과**: TP=1, FP=7, FN=0
- **Precision**: 1/8 = 12.5%, **Recall**: 1/1 = 100%

### 2. **핵심 차이점**

#### **Ollama의 장점**:
- ✅ **더 보수적인 점수**: 대부분 문서에 0.2-0.8 점수 부여
- ✅ **명확한 구분**: 관련/무관 문서를 더 명확히 구분
- ✅ **낮은 False Positive**: 임계값 0.6에서 4개만 Keep

#### **Gemini의 문제점**:
- ❌ **과도한 높은 점수**: 대부분 문서에 0.9-1.0 점수 부여
- ❌ **구분 능력 부족**: 관련/무관 문서를 잘 구분하지 못함
- ❌ **높은 False Positive**: 임계값 0.6에서 8개를 Keep

### 3. **프롬프트 분석**

두 모델 모두 동일한 프롬프트를 사용했지만, **프롬프트의 지시사항을 다르게 해석**:

#### 프롬프트 핵심 지시사항:
```
- Be EXTREMELY CONSERVATIVE with high scores (0.8+)
- Only give 0.8+ if the document DIRECTLY and COMPLETELY answers the query
- When in doubt, give a LOWER score
```

#### 해석 차이:
- **Ollama**: 지시사항을 잘 따름 → 보수적인 점수 부여
- **Gemini**: 지시사항을 무시함 → 거의 모든 문서에 높은 점수 부여

---

## 🎯 결론 및 원인

### **성능 차이의 정확한 원인**:

1. **점수 분포의 차이**:
   - **Ollama**: 0.2-0.8 범위의 보수적 점수 분포
   - **Gemini**: 0.4-1.0 범위의 관대한 점수 분포

2. **프롬프트 준수도**:
   - **Ollama**: "Be EXTREMELY CONSERVATIVE" 지시를 잘 따름
   - **Gemini**: 보수적 점수 부여 지시를 무시함

3. **임계값 민감도**:
   - **임계값 0.6**: Ollama에게는 적절, Gemini에게는 너무 낮음
   - **Gemini 최적 임계값**: 약 0.9-1.0 수준

### **실제 성능**:
- **Ollama**: 더 정확한 문서 필터링 (Precision 20% vs 12.5%)
- **Gemini**: 더 빠른 응답 속도 (28.5초 vs 94.8초)

### **권장사항**:
1. **Gemini 사용 시**: 임계값을 0.9 이상으로 조정 필요
2. **Ollama 사용 시**: 현재 임계값 0.6이 적절함
3. **프롬프트 개선**: Gemini에 더 강한 보수적 지시 필요

---

## 📈 추가 실험 제안

1. **임계값 최적화**: Gemini용 임계값 0.9, 0.95 테스트
2. **프롬프트 개선**: "EXTREMELY CONSERVATIVE" 강조
3. **더 많은 쿼리**: 1개 쿼리로는 일반화 어려움
4. **다른 데이터셋**: FiQA 데이터셋으로 검증
