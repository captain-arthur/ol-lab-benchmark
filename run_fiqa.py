# run_fiqa.py
import os
import json
import re
import math
import time
from typing import List, Dict, Any, Tuple, Optional, Set

import numpy as np
from datasets import load_dataset
from rank_bm25 import BM25Okapi
import requests


# -----------------------------
# Configuration
# -----------------------------
EXPERIMENT_CONFIG = {
    "max_queries": 20,           # 기본값: 20
    "split": "test",             # mteb/fiqa는 test 사용
    "out_dir": "results/keyword/fiqa"
}

# 환경변수 오버라이드
try:
    EXPERIMENT_CONFIG["max_queries"] = int(os.getenv("OL_MAX_QUERIES", str(EXPERIMENT_CONFIG["max_queries"])))
except Exception:
    pass

# --- Semantic rerank config --- [semantic-rerank]
TOP_R_PURE = 200                                # 순정 BM25 상위 후보 수
TOP_R_SEM = int(os.getenv("OL_TOP_R_SEM", "1000"))  # semantic 후보 수
ALPHA = float(os.getenv("OL_ALPHA", "0.5"))     # expanded soft boost (IDF 가중 가산)

# (과거 필드; 현재 미사용)
# EXPANDED_WEIGHT = 0.4

DF_THRESH = float(os.getenv("OL_DF_THRESH", "0.80"))    # DF 필터 임계
IDF_MIN   = float(os.getenv("OL_IDF_MIN", "0.1"))       # IDF 필터 임계
MAX_EXPANDED = int(os.getenv("OL_MAX_EXPANDED", "8"))

SEMANTIC_SAMPLE_RATE = float(os.getenv("OL_SEM_SAMPLE", "1.0"))  # 의미확장 적용 비율 (0~1)

# BM25 파라미터
K1 = float(os.getenv("OL_BM25_K1", "1.5"))
B  = float(os.getenv("OL_BM25_B",  "0.75"))

# 앵커/융합 파라미터
ANCHOR_K = int(os.getenv("OL_ANCHOR_K", "3"))      # BM25 상위 k개는 순서 보호
RRF_K    = float(os.getenv("OL_RRF_K", "60.0"))    # RRF 결합 상수 (작을수록 재랭크 영향↑)

# 가드레일: 최소한 순정 BM25 global top-100은 포함
GUARDRAIL_K = int(os.getenv("OL_GUARDRAIL_K", "100"))

# Ollama ENV
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://192.168.45.166:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "12"))
OLLAMA_RETRIES = int(os.getenv("OLLAMA_RETRIES", "1"))


# -----------------------------
# Utilities
# -----------------------------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def _key(q: str) -> str:
    return re.sub(r"\s+", " ", (q or "").strip().lower())

def tokenize_en(text: str) -> List[str]:
    text = re.sub(r"[^\w\s]", " ", (text or "").lower())
    toks = text.split()
    return [t for t in toks if len(t) > 2]

def normalize_terms(terms: List[str]) -> List[str]:
    out, seen = [], set()
    for term in terms or []:
        t = (term or "").lower().strip()
        if t and t not in seen:
            out.append(t)
            seen.add(t)
    return out

def dedup(seq: List[str]) -> List[str]:
    return list(dict.fromkeys(seq or []))

def ndcg_at_k(ranked_rel: List[int], k: int = 10) -> float:
    k = min(k, len(ranked_rel))
    dcg = 0.0
    for i in range(k):
        r = ranked_rel[i]
        dcg += (2**r - 1) / math.log2(i + 2)
    ideal = sorted(ranked_rel, reverse=True)[:k]
    idcg = 0.0
    for i, r in enumerate(ideal):
        idcg += (2**r - 1) / math.log2(i + 2)
    return dcg / idcg if idcg > 0 else 0.0

def mrr_at_k(ranked_rel: List[int], k: int = 10) -> float:
    k = min(k, len(ranked_rel))
    for i in range(k):
        if ranked_rel[i] == 1:
            return 1.0 / (i + 1)
    return 0.0

def map_at_k(ranked_rel: List[int], k: int = 100) -> float:
    k = min(k, len(ranked_rel))
    if sum(ranked_rel) == 0:
        return 0.0
    precision_sum, rel_count = 0.0, 0
    for i in range(k):
        if ranked_rel[i] == 1:
            rel_count += 1
            precision_sum += rel_count / (i + 1)
    return precision_sum / max(1, sum(ranked_rel))

def build_idf_dict_from_df(df_dict: Dict[str, int], N: int) -> Dict[str, float]:
    idf_dict: Dict[str, float] = {}
    for token, df in df_dict.items():
        idf = math.log((N - df + 0.5) / (df + 0.5))
        idf_dict[token] = idf
    return idf_dict

def build_doc_token_index(passages: List[str]) -> Dict[int, Tuple[Set[str], str]]:
    doc_index = {}
    for i, p in enumerate(passages):
        doc_index[i] = (set(tokenize_en(p)), p or "")
    return doc_index

def build_df_dict(passages: List[str]) -> Dict[str, int]:
    df: Dict[str, int] = {}
    for p in passages:
        for t in set(tokenize_en(p)):
            df[t] = df.get(t, 0) + 1
    return df


# -----------------------------
# Semantic Expansion (Ollama gemma3)
# -----------------------------
def build_semantic_data_ollama(query: str,
                               host: str = OLLAMA_HOST,
                               model: str = OLLAMA_MODEL,
                               timeout: int = OLLAMA_TIMEOUT,
                               retries: int = OLLAMA_RETRIES) -> Optional[Dict[str, Any]]:
    prompt = f"""
Return ONLY a valid JSON object (no code fences). You expand the user query for information retrieval.

Schema:
{{
  "user_query": "<original query>",
  "expanded": {{
    "keywords": ["6-10 short distinct phrases, 2-4 words each, lowercase"]
  }}
}}

Rules:
- expanded.keywords must have 6-10 items.
- Each phrase <=4 words, distinct, lowercase.
- No filler words (e.g., "and", "from"), no duplicates.
- No additional fields or commentary.

Now output JSON for:
"{query}"
""".strip()

    for attempt in range(retries + 1):
        try:
            resp = requests.post(
                f"{host}/api/generate",
                json={"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0.1}},
                timeout=timeout
            )
            resp.raise_for_status()
            data = resp.json()
            text = (data.get("response") or "").strip()

            # 안전한 JSON 추출: 첫 '{' ~ 마지막 '}'
            first, last = text.find("{"), text.rfind("}")
            if first == -1 or last == -1:
                continue
            obj = json.loads(text[first:last+1])
            expanded = obj.get("expanded", {}) if isinstance(obj.get("expanded"), dict) else {}
            return {
                "query": obj.get("user_query", query),
                "expanded": {"keywords": expanded.get("keywords", []) or []}
            }
        except Exception:
            if attempt == retries:
                return None
            time.sleep(0.5 * (attempt + 1))
    return None


# -----------------------------
# FiQA Data Loading (mteb/fiqa)
# -----------------------------
def load_fiqa_data(split: str = "test") -> Tuple[List[str], List[Dict[str, Any]], Dict[str, List[int]]]:
    """Returns (documents, queries, qrels_map[query_id] -> list of doc_idx)"""
    print(f"Loading FiQA dataset (split: {split})...")
    queries_ds = load_dataset("mteb/fiqa", "queries", split="queries")
    corpus_ds  = load_dataset("mteb/fiqa", "corpus",  split="corpus")
    qrels_ds   = load_dataset("mteb/fiqa",            split=split)

    documents: List[str] = []
    doc_id_to_idx: Dict[str, int] = {}
    for i, ex in enumerate(corpus_ds):
        t = (ex.get("title") or "").strip()
        body = (ex.get("text") or "").strip()
        doc = (t + " " + body).strip() if t else body
        documents.append(doc)
        doc_id_to_idx[ex["_id"]] = i

    queries: List[Dict[str, Any]] = []
    for ex in queries_ds:
        qid = ex["_id"]
        qtx = ex["text"]
        if qid and qtx:
            queries.append({"query_id": qid, "query": qtx})

    qrels_map: Dict[str, List[int]] = {}
    for ex in qrels_ds:
        qid = ex["query-id"]
        cid = ex["corpus-id"]
        score = ex.get("score", 1)
        if score and score > 0 and cid in doc_id_to_idx:
            qrels_map.setdefault(qid, []).append(doc_id_to_idx[cid])

    # 쿼리 중 정답 없는 케이스 제거(메트릭 분모 안정화)
    queries = [q for q in queries if q["query_id"] in qrels_map and len(qrels_map[q["query_id"]]) > 0]

    print(f"Loaded {len(documents)} docs, {len(queries)} queries (with qrels)")
    return documents, queries, qrels_map


# -----------------------------
# Helpers for semantic terms (phrase → token)
# -----------------------------
def terms_to_tokens(terms: List[str]) -> List[str]:
    """문구 리스트를 토큰 리스트로 평탄화 + dedup"""
    toks: List[str] = []
    for phrase in normalize_terms(terms or []):
        toks.extend(tokenize_en(phrase))
    return dedup(toks)

def _filter_sem_tokens(tokens: List[str],
                       df_dict: Dict[str, int],
                       N: int,
                       idf: Dict[str, float],
                       limit: int) -> List[str]:
    kept = []
    for t in tokens:
        if (df_dict.get(t, 0) / max(N, 1)) <= DF_THRESH and idf.get(t, 0.0) >= IDF_MIN:
            kept.append(t)
    return dedup(kept)[:limit]

# ---- PRF (RM3-lite) fallback: LLM 없이 확장어 추출 ----
def prf_rm3_terms(texts: List[str], base_scores_for_cands: np.ndarray,
                  top_m: int = 20, top_terms: int = 8, min_len: int = 3) -> List[str]:
    idx = list(range(len(texts)))
    idx.sort(key=lambda i: float(base_scores_for_cands[i]), reverse=True)
    idx = idx[:min(top_m, len(idx))]
    df: Dict[str, int] = {}
    for i in idx:
        toks = set([t for t in tokenize_en(texts[i]) if len(t) >= min_len])
        for t in toks:
            df[t] = df.get(t, 0) + 1
    return [t for t, _ in sorted(df.items(), key=lambda x: x[1], reverse=True)[:top_terms]]


# -----------------------------
# Core: per-query scoring & rerank (candidate-level)
# -----------------------------
def rerank_on_candidates(query_text: str,
                         bm25: BM25Okapi,
                         documents: List[str],
                         base_scores: np.ndarray,
                         top_r: int,
                         sem_data: Optional[Dict[str, Any]],
                         allow_semantics: bool) -> Tuple[List[int], List[float], Dict[str, Any]]:
    """
    1) BM25로 전코퍼스 base_scores에서 상위 top_r 후보 인덱스 선택
    2) semantic/PRF 데이터가 있으면 후보 텍스트만으로 DF/IDF, 토큰셋 구성 후 soft 보너스 적용
    3) 후보 내에서만 재정렬 → 최종 순서/점수 반환
    + 가드레일: 전코퍼스 순정 BM25 top-100을 최종 결과에 포함 보장
    """
    N = len(documents)
    cand_idx = list(range(N))
    cand_idx.sort(key=lambda i: base_scores[i], reverse=True)
    cand_idx = cand_idx[:min(top_r, N)]

    # pure 모드 등, 재랭크 비허용 시 BM25 cand 그대로 반환
    if not allow_semantics:
        final_local_order = list(range(len(cand_idx)))
        final_global_order = [cand_idx[i] for i in final_local_order]

        # --- Recall guardrail: ensure pure BM25 global top-100 are present ---
        pure_global = list(range(len(base_scores)))
        pure_global.sort(key=lambda i: base_scores[i], reverse=True)
        pure_topk_global = pure_global[:GUARDRAIL_K]
        pure_set = set(pure_topk_global)

        # 유니온
        seen = set(final_global_order)
        for i in pure_topk_global:
            if i not in seen:
                final_global_order.append(i); seen.add(i)

        # 자르기 전에 BM25 top-100 포함을 강제
        if len(final_global_order) > GUARDRAIL_K:
            cur = final_global_order[:GUARDRAIL_K]
            missing = [i for i in pure_topk_global if i not in set(cur)]
            if missing:
                j = GUARDRAIL_K - 1
                while missing and j >= 0:
                    if cur[j] not in pure_set:
                        cur[j] = missing.pop(0)
                    j -= 1
            final_global_order = cur

        final_scores_local = [float(base_scores[i]) for i in final_global_order]
        return final_global_order, final_scores_local, {
            "semantic_applied": False, "filtered_terms": {}, "skipped_by_dfidf": False
        }

    debug = {"semantic_applied": False, "filtered_terms": {}, "skipped_by_dfidf": False}

    # 후보 텍스트/토큰 기반으로만 DF/IDF·토큰셋 생성
    cand_texts = [documents[i] for i in cand_idx]
    df_dict = build_df_dict(cand_texts)
    idf = build_idf_dict_from_df(df_dict, len(cand_texts))
    doc_index = build_doc_token_index(cand_texts)

    # --- 확장어: 문구 → 토큰화 후 필터 ---
    expanded: List[str] = []
    if sem_data:
        exp = sem_data.get("expanded", {}) or {}
        expanded_tokens_in = terms_to_tokens(exp.get("keywords", []))
        expanded = _filter_sem_tokens(expanded_tokens_in, df_dict, len(cand_texts), idf, limit=MAX_EXPANDED)

    # ---- PRF fallback: LLM 미스/무효 시 ----
    if not expanded:
        prf_terms = prf_rm3_terms(cand_texts, np.array([base_scores[i] for i in cand_idx]),
                                  top_m=20, top_terms=MAX_EXPANDED)
        prf_terms = _filter_sem_tokens(prf_terms, df_dict, len(cand_texts), idf, limit=MAX_EXPANDED)
        if prf_terms:
            expanded = prf_terms
            debug["filtered_terms"] = {"expanded": expanded, "fallback": "prf"}
        else:
            debug["skipped_by_dfidf"] = True
            # 최종 순서는 BM25 cand 순서 그대로
            final_local_order = list(range(len(cand_idx)))
            final_global_order = [cand_idx[i] for i in final_local_order]

            # 가드레일 적용(포함 보장 방식)
            pure_global = list(range(len(base_scores)))
            pure_global.sort(key=lambda i: base_scores[i], reverse=True)
            pure_topk_global = pure_global[:GUARDRAIL_K]
            pure_set = set(pure_topk_global)

            seen = set(final_global_order)
            for i in pure_topk_global:
                if i not in seen:
                    final_global_order.append(i); seen.add(i)

            if len(final_global_order) > GUARDRAIL_K:
                cur = final_global_order[:GUARDRAIL_K]
                missing = [i for i in pure_topk_global if i not in set(cur)]
                if missing:
                    j = GUARDRAIL_K - 1
                    while missing and j >= 0:
                        if cur[j] not in pure_set:
                            cur[j] = missing.pop(0)
                        j -= 1
                final_global_order = cur

            final_scores_local = [float(base_scores[i]) for i in final_global_order]
            return final_global_order, final_scores_local, debug

    debug["semantic_applied"] = True
    if "filtered_terms" not in debug or not debug["filtered_terms"]:
        debug["filtered_terms"] = {"expanded": expanded}

    # soft bonus 계산 (가산형, 비음수)
    pairs: List[Tuple[int, float]] = []
    for loc, (_tokset, _passage) in doc_index.items():
        s = float(base_scores[cand_idx[loc]])
        for t in expanded:
            if t in _tokset:
                s += ALPHA * idf.get(t, 0.0)
        pairs.append((loc, s))

    # 1차: semantic 가감 점수로 정렬
    pairs.sort(key=lambda x: x[1], reverse=True)

    # 2차: RRF 융합(베이스 순위와 가벼운 결합) — 과도한 순위 변동 완충
    base_rank_map = {
        loc: rank for rank, loc in enumerate(
            sorted(range(len(cand_idx)),
                   key=lambda i: base_scores[cand_idx[i]], reverse=True),
            start=1
        )
    }
    tmp = []
    for loc, s_sem in pairs:
        r_base = base_rank_map[loc]
        s_rrf = s_sem + 1.0 / (RRF_K + r_base)
        tmp.append((loc, s_rrf))
    tmp.sort(key=lambda x: x[1], reverse=True)

    # 3차: 베이스 앵커 보호 — 상위 ANCHOR_K는 원래 BM25 순서를 그대로 유지
    anchor_locs = sorted(range(len(cand_idx)),
                         key=lambda i: base_scores[cand_idx[i]], reverse=True)[:ANCHOR_K]
    anchor_set = set(anchor_locs)
    tail = [p for p in tmp if p[0] not in anchor_set]
    final_pairs = [(loc, float(base_scores[cand_idx[loc]])) for loc in anchor_locs] + tail

    final_local_order = [loc for loc, _ in final_pairs]
    final_global_order = [cand_idx[loc] for loc in final_local_order]

    # --- Recall guardrail: ensure pure BM25 global top-100 are present ---
    pure_global = list(range(len(base_scores)))
    pure_global.sort(key=lambda i: base_scores[i], reverse=True)
    pure_topk_global = pure_global[:GUARDRAIL_K]
    pure_set = set(pure_topk_global)

    # 유니온
    seen = set(final_global_order)
    for i in pure_topk_global:
        if i not in seen:
            final_global_order.append(i)
            seen.add(i)

    # 길면 자르되, BM25 top-100 포함 보장 방식으로 자르기
    if len(final_global_order) > GUARDRAIL_K:
        cur = final_global_order[:GUARDRAIL_K]
        missing = [i for i in pure_topk_global if i not in set(cur)]
        if missing:
            j = GUARDRAIL_K - 1
            while missing and j >= 0:
                if cur[j] not in pure_set:
                    cur[j] = missing.pop(0)
                j -= 1
        final_global_order = cur

    final_scores_local = [float(base_scores[i]) for i in final_global_order]
    return final_global_order, final_scores_local, debug


# -----------------------------
# Benchmark Execution
# -----------------------------
def run_benchmark(semantic: bool = False,
                  split: str = "test",
                  max_queries: int = 20,
                  out_dir: str = "results") -> Dict[str, Any]:

    ensure_dir(out_dir)
    documents, queries, qrels_map = load_fiqa_data(split)

    # 쿼리 제한
    if max_queries > 0:
        queries = queries[:max_queries]

    n_queries = len(queries)
    print(f"Running benchmark with {n_queries} queries...")

    # 전코퍼스 BM25 (1회)
    print("Building BM25 index...")
    tokenized_docs = [tokenize_en(doc) for doc in documents]
    bm25 = BM25Okapi(tokenized_docs, k1=K1, b=B)

    # 캐시 초기화/프리로드
    cache: List[Dict[str, Any]] = []
    cache_path = None
    semantic_data_path = None
    mode = "semantic" if semantic else "pure"
    save_dir = os.path.join(out_dir, mode)
    ensure_dir(save_dir)

    metrics_path  = os.path.join(save_dir, "metrics.json")

    if semantic:
        semantic_data_path = os.path.join(save_dir, "semantic_data.jsonl")
        cache_dir = ".cache/keyword/fiqa"
        ensure_dir(cache_dir)
        cache_path = os.path.join(cache_dir, "ollama_cache.json")

        # 기존 캐시 로드 (리스트 스키마 권장)
        if os.path.exists(cache_path):
            try:
                loaded = json.load(open(cache_path, "r", encoding="utf-8"))
                if isinstance(loaded, list):
                    cache = loaded
            except Exception:
                cache = []

        # 기존 semantic_data.jsonl → 캐시 보강
        if os.path.exists(semantic_data_path):
            with open(semantic_data_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        row = json.loads(line)
                        q = row.get("query", "")
                        if not q:
                            continue
                        ek = []
                        if isinstance(row.get("expanded"), dict):
                            ek = row["expanded"].get("keywords", [])
                        if not ek:
                            ek = row.get("expanded_keywords", [])
                        cache.append({"query": q, "expanded": {"keywords": ek}})
                    except Exception:
                        continue

    # 메트릭 누적
    metrics = {"P@1": 0.0, "P@10": 0.0, "MRR@10": 0.0, "nDCG@10": 0.0, "R@10": 0.0, "R@100": 0.0, "MAP@100": 0.0}
    total_pairs = 0
    queries_with_rel = 0
    total_rel_docs = 0
    queries_r10_zero = 0
    sem_attempted = 0
    sem_applied = 0
    skipped_by_dfidf = 0
    hits = 0
    misses = 0

    np.random.seed(42)
    t0 = time.time()

    for qi, q in enumerate(queries, 1):
        qid, qtext = q["query_id"], q["query"]
        rel_list = qrels_map.get(qid, [])
        rel_set = set(rel_list)
        if len(rel_set) == 0:
            continue
        queries_with_rel += 1
        total_rel_docs += len(rel_set)

        # BM25 base scores (전코퍼스 1회)
        q_tokens = tokenize_en(qtext)
        base_scores = bm25.get_scores(q_tokens)

        # semantic 데이터 준비
        sem_data = None
        if semantic and (np.random.rand() < SEMANTIC_SAMPLE_RATE):
            sem_attempted += 1
            k = _key(qtext)
            for item in cache:
                if _key(item.get("query", "")) == k:
                    sem_data = item
                    hits += 1
                    break
            if sem_data is None:
                misses += 1
                sem_data = build_semantic_data_ollama(qtext)
                if sem_data:
                    cache.append(sem_data)
                    if cache_path is not None:
                        try:
                            json.dump(cache, open(cache_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
                        except Exception:
                            pass

        # 후보 재랭크
        allow_sem = bool(semantic and sem_data)  # pure 모드 or sem_data 없음 → 재랭크 비허용
        top_r = TOP_R_SEM if allow_sem else TOP_R_PURE
        final_order, final_scores, debug = rerank_on_candidates(
            qtext, bm25, documents, base_scores, top_r, sem_data, allow_semantics=allow_sem
        )

        # semantic 적용 통계 업데이트
        if semantic:
            if debug.get("semantic_applied", False):
                sem_applied += 1
            if debug.get("skipped_by_dfidf", False):
                skipped_by_dfidf += 1

        # 순위별 관련성 계산
        ranked_rel_bin: List[int] = []
        for r in range(min(1000, len(final_order))):  # 메트릭용 최대 1000까지 안전
            di = final_order[r]
            rel = 1 if di in rel_set else 0
            ranked_rel_bin.append(rel)

        # 메트릭 누적
        if ranked_rel_bin:
            metrics["P@1"]     += 1.0 if ranked_rel_bin[0] > 0 else 0.0
            metrics["P@10"]    += sum(ranked_rel_bin[:10]) / 10.0
            metrics["MRR@10"]  += mrr_at_k(ranked_rel_bin, 10)
            metrics["nDCG@10"] += ndcg_at_k(ranked_rel_bin, 10)
            rel_in_top10  = sum(ranked_rel_bin[:10])
            rel_in_top100 = sum(ranked_rel_bin[:100])
            metrics["R@10"]    += rel_in_top10  / len(rel_set)
            metrics["R@100"]   += rel_in_top100 / len(rel_set)
            metrics["MAP@100"] += map_at_k(ranked_rel_bin, 100)
            if rel_in_top10 == 0:
                queries_r10_zero += 1

        total_pairs += min(len(final_order), 1000)

        # semantic jsonl 저장(필터 결과 — 실제 사용된 토큰 기준)
        if semantic and sem_data and semantic_data_path:
            filtered = debug.get("filtered_terms", {})
            with open(semantic_data_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "query": qtext,
                    "expanded": {"keywords": filtered.get("expanded", [])},
                }, ensure_ascii=False) + "\n")

        # 샘플 로그
        if qi <= 3:
            sem_used = 'Y' if debug.get("semantic_applied", False) else ('-' if not sem_data else 'N')
            print(f"[{mode}] Q{qi} qid={qid} sem_used={sem_used} top_r={top_r} | rels={len(rel_set)}")

    # 평균화
    if queries_with_rel > 0:
        for k in metrics.keys():
            metrics[k] /= queries_with_rel

    elapsed = time.time() - t0

    # 메타 저장
    meta = {
        "dataset": "mteb/fiqa",
        "split": split,
        "mode": mode,
        "n_queries": n_queries,
        "total_pairs": total_pairs,
        "elapsed_time": elapsed,
        "metrics": metrics,
        "config": {
            "max_queries": max_queries,
            "top_r_pure": TOP_R_PURE,
            "top_r_sem": TOP_R_SEM if semantic else None,
            "k1": K1, "b": B,
            "anchor_k": ANCHOR_K, "rrf_k": RRF_K, "guardrail_k": GUARDRAIL_K
        }
    }

    if queries_with_rel > 0:
        meta["recall_analysis"] = {
            "queries_with_rel": queries_with_rel,
            "total_rel_docs": total_rel_docs,
            "mean_rel_per_query": total_rel_docs / queries_with_rel,
            "queries_r10_zero": queries_r10_zero,
            "pct_r10_zero": (queries_r10_zero / queries_with_rel) * 100.0
        }

    if semantic and (sem_attempted > 0 or sem_applied > 0 or skipped_by_dfidf > 0):
        meta["semantic_debug"] = {
            "sem_attempted": sem_attempted,
            "sem_applied": sem_applied,
            "skipped_by_dfidf": skipped_by_dfidf
        }

    ensure_dir(save_dir)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # 캐시 저장
    if semantic and cache_path:
        try:
            json.dump(cache, open(cache_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        except Exception:
            pass

    # 요약 출력
    print(f"\n[{mode}] done. n_queries={n_queries}, total_pairs={total_pairs}")
    print(f"  P@1={metrics['P@1']:.3f}, P@10={metrics['P@10']:.3f}, MRR@10={metrics['MRR@10']:.3f}")
    print(f"  nDCG@10={metrics['nDCG@10']:.3f}, R@10={metrics['R@10']:.3f}, R@100={metrics['R@100']:.3f}, time={elapsed:.1f}s")
    if "recall_analysis" in meta:
        ra = meta["recall_analysis"]
        print(f"  Recall Analysis: queries_with_rel={ra['queries_with_rel']}, mean_rel_per_query={ra['mean_rel_per_query']:.2f}")
        print(f"  R@10=0 cases: {ra['queries_r10_zero']}/{ra['queries_with_rel']} ({ra['pct_r10_zero']:.1f}%)")
    if semantic and "semantic_debug" in meta:
        dbg = meta["semantic_debug"]
        print(f"  sem_applied={dbg['sem_applied']}/{sem_attempted}, skipped_by_dfidf={dbg['skipped_by_dfidf']}")
        print(f"  cache_hits={hits}, cache_misses={misses}, cache_size={len(cache)}")
    print(f" - Metrics : {metrics_path}")

    return meta


def print_comparison_summary(pure_metrics: Dict[str, Any], semantic_metrics: Dict[str, Any]):
    print("\n" + "="*60)
    print("PURE vs SEMANTIC COMPARISON SUMMARY")
    print("="*60)
    pure_scores = pure_metrics.get("metrics", pure_metrics)
    sem_scores  = semantic_metrics.get("metrics", semantic_metrics)
    def fmt(k):
        a, b = pure_scores[k], sem_scores[k]
        arrow = "↑" if b > a else "↓"
        return f"{a:.3f} → {b:.3f} ({arrow}{abs(b-a):.3f})"
    print(f"P@1     : {fmt('P@1')}")
    print(f"P@10    : {fmt('P@10')}")
    print(f"MRR@10  : {fmt('MRR@10')}")
    print(f"nDCG@10 : {fmt('nDCG@10')}")
    print(f"R@10    : {fmt('R@10')}")
    if "semantic_debug" in semantic_metrics:
        debug = semantic_metrics["semantic_debug"]
        print(f"\nSemantic Rerank Stats: attempted={debug['sem_attempted']}, applied={debug['sem_applied']}, skipped_by_dfidf={debug['skipped_by_dfidf']}")
    print("="*60)


# -----------------------------
# Runner
# -----------------------------
def run_fiqa():
    print("=" * 60)
    print("FiQA Financial Question Answering Benchmark")
    print("=" * 60)
    print("Configuration:")
    print(f"  - Max queries: {EXPERIMENT_CONFIG['max_queries']}")
    print(f"  - Split: {EXPERIMENT_CONFIG['split']}")
    print(f"  - Output directory: {EXPERIMENT_CONFIG['out_dir']}")
    print()

    try:
        print("Running Pure BM25...")
        pure_metrics = run_benchmark(
            semantic=False,
            split=EXPERIMENT_CONFIG["split"],
            max_queries=EXPERIMENT_CONFIG["max_queries"],
            out_dir=EXPERIMENT_CONFIG["out_dir"]
        )

        print("\nRunning Semantic BM25 (Rerank)...")
        semantic_metrics = run_benchmark(
            semantic=True,
            split=EXPERIMENT_CONFIG["split"],
            max_queries=EXPERIMENT_CONFIG["max_queries"],
            out_dir=EXPERIMENT_CONFIG["out_dir"]
        )

        print_comparison_summary(pure_metrics, semantic_metrics)
        print("\nFiQA benchmark completed successfully!")
        return True

    except Exception as e:
        print(f"\n\nError during FiQA benchmark execution: {e}")
        return False


if __name__ == "__main__":
    run_fiqa()