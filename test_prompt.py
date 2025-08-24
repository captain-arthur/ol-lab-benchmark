import json
import re
import time
from typing import Dict, Any, Optional
import requests


def build_semantic_data_ollama(query: str,
                               host: str = "http://192.168.45.166:11434",
                               model: str = "gemma3",
                               timeout: int = 15,
                               retries: int = 2) -> Optional[Dict[str, Any]]:
    """
    Ollama 서버(gemma3)에 프롬프트를 보내 semantic_data 생성.
    반환 형식:
    {
      "user_query": str,
      "intent_data": {"language":"en"},
      "expanded": {
        "keywords": [...],
        "must_include": [...],
        "forbidden_terms": [...],
      }
    }
    """
    prompt = f"""
You are a domain-adapted query expansion assistant for financial IR (BM25 + soft rerank).
Return **ONLY** a single valid JSON object (no code fences, no prose).

## Objective
Maximize **recall** for BM25 by proposing broad yet relevant expansions.
`expanded.keywords` is the most important field: it MUST contain multiple useful variants.

## Output schema (strict)
{{
  "user_query": "<original query>",
  "intent_data": {{"language": "en"}},
  "expanded": {{
    "keywords": ["<MUST be 4–8 distinct terms/phrases>"],
    "must_include": ["<0–1 essential term>"],
    "forbidden_terms": ["<0–1 off-domain term>"]
  }}
}}

## Rules
- `expanded.keywords`:
  - Absolutely required: **4–8 items** (never fewer, never more).
  - Each ≤ 4 words, short & meaningful.
  - Prefer synonyms, financial domain variants, and broader category terms.
  - Deduplicate, lowercase, no trivial forms (no just plural/singular).
- `must_include`:
  - At most 1 item, only if essential to preserve intent.
- `forbidden_terms`:
  - At most 1 item, only if clearly misleading/off-domain.
- No other fields or commentary.
- Always output valid JSON that matches the schema.

## Good example
{{
  "user_query": "impact of us federal reserve interest rate hikes",
  "intent_data": {{"language": "en"}},
  "expanded": {{
    "keywords": ["federal reserve rate hikes", "us interest rate changes", "monetary policy tightening", "fed funds rate increases", "us central bank policy", "interest rate announcements"],
    "must_include": [],
    "forbidden_terms": []
  }}
}}

Now produce the JSON for this query:
"{query}"
""".strip()

    for attempt in range(retries + 1):
        try:
            # Ollama /api/generate (temperature=0으로 고정)
            resp = requests.post(
                f"{host}/api/generate",
                json={"model": model, "prompt": prompt, "stream": False, "temperature": 0},
                timeout=timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            text = data.get("response", "").strip()
            # 모델이 JSON만 반환하도록 프롬프트 했지만, 혹시 앞뒤 잡음 제거
            first = text.find("{")
            last = text.rfind("}")
            if first == -1 or last == -1:
                return None
            json_text = text[first:last+1]
            parsed = json.loads(json_text)
            # 필드 정리 및 정규화
            exp = parsed.get("expanded", {})
            return {
                "user_query": parsed.get("user_query", query),
                "intent_data": parsed.get("intent_data", {"language": "en"}),
                "expanded": {
                    "keywords": exp.get("keywords", []),
                    "must_include": exp.get("must_include", []),
                    "forbidden_terms": exp.get("forbidden_terms", []),
                }
            }
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt == retries:
                return None
            time.sleep(0.5 * (attempt + 1))  # 짧은 백오프


def test_semantic_data_generation():
    """
    의미적 데이터 생성을 테스트하는 독립 실행 가능한 함수
    """
    print("=" * 80)
    print("의미적 데이터 생성 테스트 시작")
    print("=" * 80)
    
    # 테스트할 쿼리들
    test_queries = [
        "What are the risks of investing in cryptocurrency?",
        "How to analyze stock market trends",
        "Best retirement planning strategies",
        "Federal Reserve interest rate changes",
        "What is a corporation?",
        "Why did Rachel Carson write Silent Spring?",
        "How to make chocolate chip cookies",
        "What causes climate change"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n[{i}/{len(test_queries)}] 테스트 쿼리: {query}")
        print("-" * 60)
        
        try:
            # Ollama 서버에 요청
            result = build_semantic_data_ollama(query)
            
            if result:
                print("✅ 성공적으로 생성됨:")
                print(f"  - User Query: {result['user_query']}")
                print(f"  - Keywords: {result['expanded']['keywords']}")
                print(f"  - Must Include: {result['expanded']['must_include']}")
                print(f"  - Forbidden Terms: {result['expanded']['forbidden_terms']}")
                
                # 품질 검사
                keywords_count = len(result['expanded']['keywords'])
                must_include_count = len(result['expanded']['must_include'])
                forbidden_count = len(result['expanded']['forbidden_terms'])
                
                print(f"\n📊 품질 지표:")
                print(f"  - Keywords 개수: {keywords_count} (목표: 4-8)")
                print(f"  - Must Include 개수: {must_include_count} (목표: 0-1)")
                print(f"  - Forbidden Terms 개수: {forbidden_count} (목표: 0-1)")
                
                # 품질 평가
                quality_score = 0
                if 4 <= keywords_count <= 8:
                    quality_score += 3
                    print("  ✅ Keywords 개수 적절")
                else:
                    print(f"  ❌ Keywords 개수 부적절 (현재: {keywords_count})")
                
                if must_include_count <= 1:
                    quality_score += 1
                    print("  ✅ Must Include 개수 적절")
                else:
                    print(f"  ❌ Must Include 개수 부적절 (현재: {must_include_count})")
                
                if forbidden_count <= 1:
                    quality_score += 1
                    print("  ✅ Forbidden Terms 개수 적절")
                else:
                    print(f"  ❌ Forbidden Terms 개수 부적절 (현재: {forbidden_count})")
                
                print(f"  📈 전체 품질 점수: {quality_score}/5")
                
            else:
                print("❌ 생성 실패: Ollama 서버 응답 없음")
                
        except Exception as e:
            print(f"❌ 오류 발생: {e}")
        
        print("-" * 60)
    
    print("\n" + "=" * 80)
    print("테스트 완료")
    print("=" * 80)


# pytest 테스트 함수
def test_semantic_data_generation_pytest():
    """
    pytest에서 실행할 수 있는 테스트 함수
    """
    # 간단한 테스트 쿼리로 빠른 검증
    test_query = "What are the risks of investing in cryptocurrency?"
    
    result = build_semantic_data_ollama(test_query)
    
    # 기본 검증
    assert result is not None, "Ollama 서버에서 응답을 받지 못했습니다"
    assert "expanded" in result, "결과에 'expanded' 필드가 없습니다"
    assert "keywords" in result["expanded"], "결과에 'keywords' 필드가 없습니다"
    assert "must_include" in result["expanded"], "결과에 'must_include' 필드가 없습니다"
    assert "forbidden_terms" in result["expanded"], "결과에 'forbidden_terms' 필드가 없습니다"
    
    # 품질 검증
    keywords = result["expanded"]["keywords"]
    must_include = result["expanded"]["must_include"]
    forbidden_terms = result["expanded"]["forbidden_terms"]
    
    # Keywords는 4-8개여야 함
    assert 4 <= len(keywords) <= 8, f"Keywords 개수가 부적절합니다: {len(keywords)} (목표: 4-8)"
    
    # Must include는 0-1개여야 함
    assert len(must_include) <= 1, f"Must include 개수가 부적절합니다: {len(must_include)} (목표: 0-1)"
    
    # Forbidden terms는 0-1개여야 함
    assert len(forbidden_terms) <= 1, f"Forbidden terms 개수가 부적절합니다: {len(forbidden_terms)} (목표: 0-1)"
    
    print(f"✅ pytest 테스트 통과!")
    print(f"  - Keywords: {keywords}")
    print(f"  - Must Include: {must_include}")
    print(f"  - Forbidden Terms: {forbidden_terms}")


if __name__ == "__main__":
    # 독립 실행 시 테스트 함수 호출
    test_semantic_data_generation()
