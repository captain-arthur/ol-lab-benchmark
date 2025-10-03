#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
debug_parsing.py
- Gemini 응답 파싱 디버깅
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from filter.llm.llm_filter import parse_llm_similarity_scores

# 실제 Gemini 응답
response = """Document 1: [0.8]
Document 2: [1.0]
Document 3: [0.6]
Document 4: [0.2]
Document 5: [0.2]
Document 6: [0.2]
Document 7: [0.2]
Document 8: [0.2]
Document 9: [0.2]
Document 10: [0.2]"""

print("Gemini 응답:")
print(response)
print("\n파싱 결과:")

parsed = parse_llm_similarity_scores(response, 10)
for i, result in enumerate(parsed):
    print(f"Document {i+1}: {result['similarity_score']}")
