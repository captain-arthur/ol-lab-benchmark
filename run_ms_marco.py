# run_ms_marco_keyword_recall.py
import os
import json
import re
import math
import time
from typing import List, Dict, Any, Iterable, Tuple, Optional, Set

import numpy as np
from datasets import load_dataset
from rank_bm25 import BM25Okapi
import requests

# -----------------------------
# Configuration (recall-first + safe-drop)
# -----------------------------
EXPERIMENT_CONFIG = {
    "max_queries": 20,
    "split": "validation",        # "validation" or "train"
    "out_dir": "results/keyword/ms_marco"
}
try:
    EXPERIMENT_CONFIG["max_queries"] = int(os.getenv("OL_MAX_QUERIES", str(EXPERIMENT_CONFIG["max_queries"])))
except Exception:
    pass

# 후보폭 확대 (리콜↑)
TOP_R_PURE = 200
TOP_R_SEM = int(os.getenv("OL_TOP_R_SEM", "2000"))  # 1000 -> 2000(기본), 필요시 4000

# 확장 쿼리 혼합 가중 (안정적, 가산만)
EXPANDED_WEIGHT = float(os.getenv("OL_EXP_W", "0.35"))  # 0.25~0.45 튠 권장
ALPHA = 0.45  # 확장 키워드 hit 보너스 (비음수, 가산만)

# 확장 키워드 필터 (확장어 "선택"용 — 문서 드롭엔 쓰지 않음)
DF_THRESH = 0.95
IDF_MIN = 0.05
MAX_EXPANDED = 8
SEMANTIC_SAMPLE_RATE = 1.0  # 실험 편의상 1.0, 캐시/LLM 비용 고려해 조절

# SAFE-DROP 임계 (보수적)
SAFE_DROP_QUANTILE = float(os.getenv("OL_SAFE_DROP_Q", "0.25"))  # q25

# BM25 params
K1 = 1.5
B = 0.75

# LLM 확장 호출(선택)
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://192.168.45.166:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3")

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

# -----------------------------
# Index/Statistics
# -----------------------------
def build_idf_dict_from_df(df_dict: Dict[str, int], N: int) -> Dict[str, float]:
    idf_dict: Dict[str, float] = {}
    for token, df in df_dict.items():
        idf = math.log((N - df + 0.5) / (df + 0.5))
        idf_dict[token] = idf
    return idf_dict

def build_doc_token_index(passages: List[str], tokenizer) -> Dict[int, Tuple[Set[str], str]]:
    doc_index = {}
    for doc_id, passage in enumerate(passages):
        tokens = set(tokenizer(passage))
        doc_index[doc_id] = (tokens, passage or "")
    return doc_index

def build_df_dict(passages: List[str], tokenizer) -> Dict[str, int]:
    df_dict = {}
    for passage in passages:
        tokens = set(tokenizer(passage))
        for token in tokens:
            df_dict[token] = df_dict.get(token, 0) + 1
    return df_dict

# -----------------------------
# Semantic Expansion (Ollama, optional)
# -----------------------------
def build_semantic_data_ollama(query: str,
                               host: str = OLLAMA_HOST,
                               model: str = OLLAMA_MODEL,
                               timeout: int = 15,
                               retries: int = 2) -> Optional[Dict[str, Any]]:
    """
    LLM로 확장 키워드 JSON을 받아 표준 스키마로 반환:
    {"query": str, "expanded": {"keywords": [...]}}
    """
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
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0}
                },
                timeout=timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            text = (data.get("response") or "").strip()
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
            time.sleep(0.4 * (attempt + 1))
    return None

# -----------------------------
# Query expansion utils
# -----------------------------
def filter_semantic_terms(terms: List[str], df_dict: Dict[str, int], N: int, idf: Dict[str, float]) -> List[str]:
    """
    확장어 '선택'을 위한 DF/IDF 필터. 문서 드롭에는 절대 사용하지 않음.
    """
    kept = []
    for t in normalize_terms(terms):
        if (df_dict.get(t, 0) / max(1, N)) <= DF_THRESH and idf.get(t, 0.0) >= IDF_MIN:
            kept.append(t)
    return dedup(kept)[:MAX_EXPANDED]

def extract_expanded(sem_data: Optional[Dict[str, Any]],
                     df_dict: Dict[str, int], N: int, idf: Dict[str, float]) -> List[str]:
    """
    어떤 스키마로 들어와도 expanded 키워드만 표준화해 반환.
    - 선호: {"expanded": {"keywords": [...]}}
    - fallback: {"expanded_keywords": [...]} (legacy)
    """
    if not sem_data:
        return []
    ek = []
    if isinstance(sem_data.get("expanded"), dict):
        ek = sem_data["expanded"].get("keywords", [])
    if not ek:
        ek = sem_data.get("expanded_keywords", [])  # legacy fallback
    return filter_semantic_terms(ek, df_dict, N, idf)

def soft_semantic_bonus_tokens(doc_tokens: Set[str], idf: Dict[str, float], expanded_terms: List[str]) -> float:
    """
    확장어(문구)를 토큰화하여 문서 토큰과 교집합을 보며 비음수 가산 보너스를 부여
    """
    s = 0.0
    exp_tokens = set()
    for phrase in expanded_terms:
        exp_tokens.update(tokenize_en(phrase))
    for t in exp_tokens:
        if t in doc_tokens:
            s += ALPHA * idf.get(t, 0.0)
    return s

# ---- PRF (RM3-lite) fallback: LLM 없이 확장어 추출 ----
def prf_rm3_terms(passages: List[str], base_scores, tokenizer,
                  top_m: int = 20, top_terms: int = 8, min_len: int = 3) -> List[str]:
    cand_idx = list(range(len(passages)))
    cand_idx.sort(key=lambda i: float(base_scores[i]), reverse=True)
    cand_idx = cand_idx[:min(top_m, len(cand_idx))]

    df = {}
    for i in cand_idx:
        toks = set([t for t in tokenizer(passages[i]) if len(t) >= min_len])
        for t in toks:
            df[t] = df.get(t, 0) + 1

    return [t for t, _ in sorted(df.items(), key=lambda x: x[1], reverse=True)[:top_terms]]

# -----------------------------
# Data: MS MARCO (streaming)
# -----------------------------
def iter_ms_marco_query_groups(split: str = "validation",
                               max_queries: int = 20) -> Iterable[Tuple[str, List[str], List[int]]]:
    try:
        ds = load_dataset("ms_marco", "v2.1", split=split, streaming=True)
    except Exception as e:
        raise RuntimeError(f"Failed to load ms_marco v2.1 split='{split}'. "
                           f"Check HF dataset availability/credentials. Error: {e}")
    count = 0
    for item in ds:
        query = item.get("query", "")
        passages = item.get("passages", {})
        if not passages or "passage_text" not in passages or "is_selected" not in passages:
            continue
        p_texts: List[str] = [ (t or "").strip() for t in passages["passage_text"] ]
        p_labels: List[int] = [1 if bool(b) else 0 for b in passages["is_selected"]]

        keep_texts, keep_labels = [], []
        for t, r in zip(p_texts, p_labels):
            if t:
                keep_texts.append(t)
                keep_labels.append(r)
        if not keep_texts:
            continue

        yield query, keep_texts, keep_labels
        count += 1
        if count >= max_queries:
            break

# -----------------------------
# Scoring (per query group)
# -----------------------------
def score_group_bm25_rerank(query: str,
                            passages: List[str],
                            labels: List[int],
                            semantic: bool,
                            sem_data: Optional[Dict[str, Any]],
                            df_dict: Dict[str, int],
                            idf: Dict[str, float],
                            doc_index: Dict[int, Tuple[Set[str], str]]) -> Tuple[List[int], List[float], List[int], Dict[str, Any]]:
    N = len(passages)
    docs_tokens = [tokenize_en(p) for p in passages]
    bm25 = BM25Okapi(docs_tokens, k1=K1, b=B)

    # --- 순정 BM25 점수(guardrail/앵커용) ---
    q_tokens = tokenize_en(query)
    pure_scores = bm25.get_scores(q_tokens)    # 순정 스코어는 보존
    base_scores = list(pure_scores)            # 혼합/재랭크용 가변 스코어

    # === 확장어 확보: (1) LLM (2) 실패시 PRF fallback ===
    expanded: List[str] = []
    if semantic:
        if sem_data:
            expanded = extract_expanded(sem_data, df_dict, N, idf)
        if not expanded:
            prf_terms = prf_rm3_terms(passages, pure_scores, tokenize_en, top_m=20, top_terms=8)
            expanded = filter_semantic_terms(prf_terms, df_dict, N, idf)

    # 확장 쿼리 혼합 (소프트 가산)
    if semantic and expanded:
        q_tokens_expanded = list(set(q_tokens + expanded))
        alt_scores = bm25.get_scores(q_tokens_expanded)
        base_scores = [float(b) + EXPANDED_WEIGHT * float(a)
                       for b, a in zip(base_scores, alt_scores)]

    # 후보 추출: 혼합 Top-R  ∪  순정 Top-R(=앵커)  → 리콜 보장
    top_r = TOP_R_SEM if semantic else TOP_R_PURE
    cand_mixed = list(range(len(base_scores)))
    cand_mixed.sort(key=lambda i: base_scores[i], reverse=True)
    cand_mixed = cand_mixed[:top_r]

    cand_pure = list(range(len(pure_scores)))
    cand_pure.sort(key=lambda i: pure_scores[i], reverse=True)
    cand_pure = cand_pure[:TOP_R_PURE]  # 앵커 (절대 드롭하지 않음)

    top_candidates = dedup(cand_mixed + cand_pure)

    # ===== SAFE DROP (보수적 AND 게이트) =====
    # 확장 키워드를 토큰 단위로 변환
    expanded_token_set = set()
    for phrase in expanded:
        expanded_token_set.update(tokenize_en(phrase))

    # 확장 토큰이 없으면 SAFE-DROP을 건너뜀 (초보수적)
    if not (semantic and len(expanded_token_set) > 0):
        # SAFE-DROP 스킵
        debug_info: Dict[str, Any] = {
            "semantic_applied": bool(expanded),
            "filtered_terms": {"expanded": expanded},
            "skipped_by_dfidf": not bool(expanded),
            "safe_drop_q": SAFE_DROP_QUANTILE,
            "safe_drop_q25": None,
            "safe_dropped": 0
        }
    else:
        # 기존 SAFE-DROP 로직 수행
        # 조건: (1) 앵커 아님  (2) 순정 BM25 < q25  (3) 쿼리토큰 겹침 0  (4) 확장키워드 겹침 0
        q25 = float(np.quantile(pure_scores, SAFE_DROP_QUANTILE)) if len(pure_scores) > 0 else -1e9
        query_token_set = set(q_tokens)

        def has_overlap(doc_tokens: Set[str], probe: Set[str]) -> bool:
            # 토큰 직접 매치 (키워드는 토큰으로 이미 들어옴)
            return len(doc_tokens & probe) > 0

        kept_after_safe, dropped_safe = [], []
        for doc_id in top_candidates:
            if doc_id in cand_pure:
                kept_after_safe.append(doc_id)  # 앵커는 무조건 보존
                continue
            doc_tokens, _ = doc_index[doc_id]
            cond_low_bm25 = pure_scores[doc_id] < q25
            cond_no_query  = not has_overlap(doc_tokens, query_token_set)
            cond_no_expand = not (expanded_token_set and has_overlap(doc_tokens, expanded_token_set))
            if cond_low_bm25 and cond_no_query and cond_no_expand:
                dropped_safe.append(doc_id)
            else:
                kept_after_safe.append(doc_id)

        top_candidates = kept_after_safe

        debug_info: Dict[str, Any] = {
            "semantic_applied": bool(expanded),
            "filtered_terms": {"expanded": expanded},
            "skipped_by_dfidf": not bool(expanded),
            "safe_drop_q": SAFE_DROP_QUANTILE,
            "safe_drop_q25": q25,
            "safe_dropped": len(dropped_safe)
        }

    # === 재랭크(드롭 없음, 순수 가산만) ===
    reranked_candidates: List[Tuple[int, float]] = []
    final_score_map: Dict[int, float] = {}
    if semantic and expanded:
        for doc_id in top_candidates:
            doc_tokens, _ = doc_index[doc_id]
            bonus = soft_semantic_bonus_tokens(doc_tokens, idf, expanded)  # 비음수
            final_score = float(base_scores[doc_id]) + bonus
            reranked_candidates.append((doc_id, final_score))
        reranked_candidates.sort(key=lambda x: x[1], reverse=True)
        top_candidates = [doc_id for doc_id, _ in reranked_candidates]
        final_score_map = {doc_id: float(s) for doc_id, s in reranked_candidates}
    else:
        # semantic 미적용이면 혼합 없이 base_scores 기준
        top_candidates.sort(key=lambda i: base_scores[i], reverse=True)

    # === Recall guardrail: 순정 BM25 top-100 반드시 포함 ===
    pure_top100 = cand_pure[:100]
    final_order = top_candidates[:100]
    missing = [d for d in pure_top100 if d not in final_order]
    if missing:
        room = max(0, 100 - len(final_order))
        final_order = (final_order + missing[:room])[:100]
        debug_info["recall_guardrail_applied"] = True
        debug_info["pure_missing_cnt"] = len(missing)
    else:
        debug_info["recall_guardrail_applied"] = False
        debug_info["pure_missing_cnt"] = 0

    final_scores = [float(final_score_map.get(i, base_scores[i])) for i in final_order]
    final_labels = [int(labels[i]) for i in final_order]
    return final_order, final_scores, final_labels, debug_info

# -----------------------------
# Metrics accumulator
# -----------------------------
class Metrics:
    def __init__(self):
        self.n = 0
        self.sum_p1 = 0.0
        self.sum_p10 = 0.0
        self.sum_r10 = 0.0
        self.sum_r100 = 0.0
        self.sum_mrr10 = 0.0
        self.sum_ndcg10 = 0.0
        self.sum_ndcg100 = 0.0
        self.sum_map100 = 0.0
        self.sem_attempted = 0
        self.sem_applied = 0
        self.skipped_by_dfidf = 0
        self.queries_with_rel = 0
        self.total_rel_docs = 0
        self.queries_r10_zero = 0

    def add(self, ranked_labels: List[int], total_rel_all: int, sem_applied: bool = False):
        self.n += 1
        topk = ranked_labels[:10]
        top100 = ranked_labels[:100]
        total_rel = max(int(total_rel_all), 0)

        p1 = float(ranked_labels[0]) if ranked_labels else 0.0
        p10 = (sum(topk) / max(len(topk), 1)) if topk else 0.0
        r10 = (sum(topk) / max(total_rel, 1)) if total_rel > 0 else 0.0
        r100 = (sum(top100) / max(total_rel, 1)) if total_rel > 0 else 0.0
        mrr10 = mrr_at_k(ranked_labels, 10)
        ndcg10 = ndcg_at_k(ranked_labels, 10)
        ndcg100 = ndcg_at_k(ranked_labels, 100)
        map100 = map_at_k(ranked_labels, 100)

        self.sum_p1 += p1
        self.sum_p10 += p10
        self.sum_r10 += r10
        self.sum_r100 += r100
        self.sum_mrr10 += mrr10
        self.sum_ndcg10 += ndcg10
        self.sum_ndcg100 += ndcg100
        self.sum_map100 += map100

        if total_rel > 0:
            self.queries_with_rel += 1
            self.total_rel_docs += total_rel
            if r10 == 0.0:
                self.queries_r10_zero += 1

        if sem_applied:
            self.sem_applied += 1

    def add_sem_attempt(self):
        self.sem_attempted += 1

    def add_skipped_by_dfidf(self):
        self.skipped_by_dfidf += 1

    def result(self) -> Dict[str, float]:
        if self.n == 0:
            return {"P@1":0.0, "P@10":0.0, "R@10":0.0, "R@100":0.0, "MRR@10":0.0, "nDCG@10":0.0, "nDCG@100":0.0, "MAP@100":0.0}
        return {
            "P@1": self.sum_p1 / self.n,
            "P@10": self.sum_p10 / self.n,
            "R@10": self.sum_r10 / self.n,
            "R@100": self.sum_r100 / self.n,
            "MRR@10": self.sum_mrr10 / self.n,
            "nDCG@10": self.sum_ndcg10 / self.n,
            "nDCG@100": self.sum_ndcg100 / self.n,
            "MAP@100": self.sum_map100 / self.n,
        }

# -----------------------------
# Runner
# -----------------------------
def run_benchmark(semantic: bool,
                  split: str = "validation",
                  max_queries: int = 20,
                  out_dir: str = "results"):
    mode = "semantic" if semantic else "pure"
    save_dir = os.path.join(out_dir, mode)
    ensure_dir(save_dir)

    metrics_path  = os.path.join(save_dir, "metrics.json")
    semantic_data_path = None
    cache_path = None
    cache: List[Dict[str, Any]] = []
    hits = misses = 0

    if semantic:
        semantic_data_path = os.path.join(save_dir, "semantic_data.jsonl")
        cache_dir = ".cache/keyword/ms_marco"
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

    metrics = Metrics()
    t0 = time.time()
    n_queries = 0
    total_pairs = 0

    # 고정 시드
    np.random.seed(42)

    for q_idx, (query, passages, labels) in enumerate(iter_ms_marco_query_groups(split=split, max_queries=max_queries), start=1):
        n_queries += 1
        total_pairs += len(passages)

        df_dict = build_df_dict(passages, tokenize_en)
        idf = build_idf_dict_from_df(df_dict, len(passages))
        doc_index = build_doc_token_index(passages, tokenize_en)

        sem_data = None
        if semantic and (np.random.rand() < SEMANTIC_SAMPLE_RATE):
            metrics.add_sem_attempt()
            # 캐시 조회
            k = _key(query)
            for item in cache:
                if _key(item.get("query", "")) == k:
                    sem_data = item
                    hits += 1
                    break
            if sem_data is None:
                misses += 1
                sem_data = build_semantic_data_ollama(query)
                if sem_data:
                    cache.append(sem_data)
                    if cache_path is not None:
                        try:
                            json.dump(cache, open(cache_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
                        except Exception:
                            pass

        order, scores, final_labels, debug_info = score_group_bm25_rerank(
            query, passages, labels, semantic, sem_data, df_dict, idf, doc_index
        )
        if not order:
            continue

        # semantic jsonl 저장 (expanded-only 기록 + safe-drop 통계)
        if semantic and semantic_data_path:
            filtered = debug_info.get("filtered_terms", {})
            entry = {
                "qid": q_idx,
                "query": query,
                "expanded": {"keywords": filtered.get("expanded", [])},
                "recall_guardrail_applied": debug_info.get("recall_guardrail_applied", False),
                "pure_missing_cnt": debug_info.get("pure_missing_cnt", 0),
                "safe_drop_q": debug_info.get("safe_drop_q", None),
                "safe_drop_q25": debug_info.get("safe_drop_q25", None),
                "safe_dropped": debug_info.get("safe_dropped", 0)
            }
            with open(semantic_data_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        # 메트릭 누적
        if debug_info.get("skipped_by_dfidf"):
            metrics.add_skipped_by_dfidf()
        metrics.add(final_labels, total_rel_all=sum(labels), sem_applied=debug_info.get("semantic_applied", False))

        # 샘플 로그
        if q_idx <= 3:
            sem_used = 'Y' if debug_info.get("semantic_applied", False) else ('-' if not sem_data else 'N')
            print(f"[{mode}] Q{q_idx} query='{query[:60]}' sem_used={sem_used} "
                  f"top_r={TOP_R_SEM if semantic else TOP_R_PURE} "
                  f"w_exp={EXPANDED_WEIGHT} safe_dropped={debug_info.get('safe_dropped',0)}")

    elapsed = time.time() - t0
    result_metrics = metrics.result()

    meta = {
        "model": "BM25 (+semantic rerank, safe-drop, recall-guardrail, PRF-fallback)" if semantic else "BM25 (pure)",
        "bm25_params": {"k1": K1, "b": B},
        "data": {
            "split": split,
            "max_queries": max_queries,
            "n_queries": n_queries,
            "total_pairs": total_pairs
        },
        "metrics": result_metrics,
        "timing_sec": elapsed,
        "semantic_sample_rate": SEMANTIC_SAMPLE_RATE if semantic else 0.0,
        "top_r": TOP_R_SEM if semantic else TOP_R_PURE,
        "params": {
            "ALPHA": ALPHA,
            "DF_THRESH": DF_THRESH,
            "IDF_MIN": IDF_MIN,
            "MAX_EXPANDED": MAX_EXPANDED,
            "EXPANDED_WEIGHT": EXPANDED_WEIGHT,
            "SAFE_DROP_QUANTILE": SAFE_DROP_QUANTILE
        }
    }
    if metrics.queries_with_rel > 0:
        meta["recall_analysis"] = {
            "queries_with_rel": metrics.queries_with_rel,
            "total_rel_docs": metrics.total_rel_docs,
            "mean_rel_per_query": metrics.total_rel_docs / metrics.queries_with_rel,
            "queries_r10_zero": metrics.queries_r10_zero,
            "pct_r10_zero": (metrics.queries_r10_zero / metrics.queries_with_rel) * 100.0
        }
    if semantic and (metrics.sem_attempted > 0 or metrics.sem_applied > 0 or metrics.skipped_by_dfidf > 0):
        meta["semantic_debug"] = {
            "sem_attempted": metrics.sem_attempted,
            "sem_applied": metrics.sem_applied,
            "skipped_by_dfidf": metrics.skipped_by_dfidf
        }

    ensure_dir(save_dir)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # 캐시 저장
    if semantic and cache_path is not None:
        try:
            json.dump(cache, open(cache_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        except Exception:
            pass

    print(f"\n[{mode}] done.  n_queries={n_queries}, total_pairs={total_pairs}")
    print(f"  P@1={result_metrics['P@1']:.3f}, P@10={result_metrics['P@10']:.3f}, MRR@10={result_metrics['MRR@10']:.3f}")
    print(f"  nDCG@10={result_metrics['nDCG@10']:.3f}, R@10={result_metrics['R@10']:.3f}, R@100={result_metrics['R@100']:.3f}, time={elapsed:.1f}s")
    if metrics.queries_with_rel > 0:
        mean_rel_per_query = metrics.total_rel_docs / metrics.queries_with_rel
        pct_r10_zero = (metrics.queries_r10_zero / metrics.queries_with_rel) * 100
        print(f"  Recall Analysis: queries_with_rel={metrics.queries_with_rel}, mean_rel_per_query={mean_rel_per_query:.2f}")
        print(f"  R@10=0 cases: {metrics.queries_r10_zero}/{metrics.queries_with_rel} ({pct_r10_zero:.1f}%)")
    if semantic:
        dbg = meta.get("semantic_debug", {"sem_applied":0, "sem_attempted":0, "skipped_by_dfidf":0})
        print(f"  sem_applied={dbg['sem_applied']}/{dbg['sem_attempted']}, skipped_by_dfidf={dbg['skipped_by_dfidf']}")
    print(f" - Metrics : {metrics_path}")
    if semantic and semantic_data_path:
        print(f" - Semantic: {semantic_data_path}")
    return meta

def print_comparison_summary(pure_metrics: Dict[str, Any], semantic_metrics: Dict[str, Any]):
    print("\n" + "="*60)
    print("PURE vs SEMANTIC COMPARISON SUMMARY")
    print("="*60)
    pure_scores = pure_metrics.get("metrics", pure_metrics)
    sem_scores  = semantic_metrics.get("metrics", semantic_metrics)
    def line(k):
        a, b = pure_scores[k], sem_scores[k]
        return f"{a:.3f} → {b:.3f} ({'↑' if b>a else '↓'}{abs(b-a):.3f})"
    print(f"P@1     : {line('P@1')}")
    print(f"P@10    : {line('P@10')}")
    print(f"MRR@10  : {line('MRR@10')}")
    print(f"nDCG@10 : {line('nDCG@10')}")
    print(f"R@10    : {line('R@10')}")
    print(f"R@100   : {line('R@100')}")
    print("="*60)

# -----------------------------
# Top-level Runner
# -----------------------------
def run_ms_marco():
    print("=" * 60)
    print("MS MARCO Passage Ranking Benchmark")
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

        print("\nRunning Semantic BM25 (Recall-first + SAFE-DROP + guardrail + PRF fallback)...")
        semantic_metrics = run_benchmark(
            semantic=True,
            split=EXPERIMENT_CONFIG["split"],
            max_queries=EXPERIMENT_CONFIG["max_queries"],
            out_dir=EXPERIMENT_CONFIG["out_dir"]
        )

        print_comparison_summary(pure_metrics, semantic_metrics)
        print("\nMS MARCO benchmark completed successfully!")
        return True
    except Exception as e:
        print(f"\n\nError during MS MARCO benchmark execution: {e}")
        return False

if __name__ == "__main__":
    run_ms_marco()