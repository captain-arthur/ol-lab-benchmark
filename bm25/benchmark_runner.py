import os
import json
import re
import math
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from gsheet.gsheet import GoogleSheetManager
from rank_bm25 import BM25Okapi
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import nltk
from nltk.corpus import stopwords
import numpy as np


class BenchmarkRunner:
    """벤치마크 실행 클래스"""
    
    def __init__(self, spreadsheet_id: str):
        """
        BenchmarkRunner 초기화
        
        Args:
            spreadsheet_id: 구글 시트 ID
        """
        self.spreadsheet_id = spreadsheet_id
        self.sheet_manager = GoogleSheetManager(spreadsheet_id)
        
        # NLTK 데이터 다운로드
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        
        self.stop_words = set(stopwords.words('english'))
    
    def copy_dataset_to_experiment_sheet(self, source_sheet: str, target_sheet: str) -> bool:
        """
        데이터셋을 실험 결과 시트로 복사
        
        Args:
            source_sheet: 원본 시트 이름
            target_sheet: 대상 시트 이름
            
        Returns:
            성공 여부
        """
        print(f"데이터셋 복사 중: {source_sheet} → {target_sheet}")
        
        try:
            # 원본 데이터 가져오기
            source_range = f"{source_sheet}!A:Z"
            source_data = self.sheet_manager.get_sheet_data(source_range)
            
            if not source_data:
                print(f"원본 시트에서 데이터를 찾을 수 없습니다: {source_sheet}")
                return False
            
            print(f"원본 데이터 로드 완료: {len(source_data)}행")
            
            # 대상 시트 초기화 (기존 데이터 삭제)
            try:
                # 대상 시트의 모든 데이터 삭제
                self.sheet_manager.service.spreadsheets().values().clear(
                    spreadsheetId=self.spreadsheet_id,
                    range=f"{target_sheet}!A:Z"
                ).execute()
                print(f"대상 시트 초기화 완료: {target_sheet}")
            except Exception as e:
                print(f"시트 초기화 중 오류 (무시): {e}")
            
            if len(source_data) > 2:
                # 헤더와 데이터 복사 (업로드 시간은 현재 시간으로 업데이트)
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # 새로운 업로드 시간 행 생성
                new_time_row = [f"업로드 시간: {current_time}"]
                
                # 빈 행 (구분용)
                empty_row = []
                
                # 헤더와 데이터는 원본에서 가져오기 (업로드 시간 제외)
                data_to_copy = [new_time_row, empty_row] + source_data[2:]  # 새로운 시간 + 빈 행 + 헤더 + 데이터
                
                # 대상 시트에 데이터를 한 번에 업로드 (덮어쓰기)
                try:
                    body = {'values': data_to_copy}
                    result = self.sheet_manager.service.spreadsheets().values().update(
                        spreadsheetId=self.spreadsheet_id,
                        range=f"{target_sheet}!A1",
                        valueInputOption='RAW',
                        body=body
                    ).execute()
                    
                    updated_rows = result.get('updatedRows', 0)
                    print(f"데이터 복사 완료: {updated_rows}행 (복사 시간: {current_time})")
                    return True
                    
                except Exception as e:
                    print(f"데이터 복사 중 오류: {e}")
                    return False
            else:
                print("복사할 데이터가 없습니다.")
                return False
                
        except Exception as e:
            print(f"데이터 복사 중 오류 발생: {e}")
            return False
    
    def tokenize_en(self, text: str) -> List[str]:
        """영어 텍스트 토큰화 (개선된 버전)"""
        # 더 정교한 토크나이징
        text = re.sub(r"[^\w\s]", " ", text.lower())
        toks = text.split()
        # 불용어 제거 및 길이 필터링 (길이 조건 완화)
        return [t for t in toks if t not in self.stop_words and len(t) > 1]

    def tokenize_ko_simple(self, text: str) -> List[str]:
        """한국어 텍스트 토큰화 (간단한 fallback)"""
        toks = re.findall(r"[가-힣]+|[A-Za-z0-9]+", text)
        return [t.lower() for t in toks if len(t) > 1]

    def preprocess_text(self, text: str, language: str = "en") -> List[str]:
        """
        언어별 텍스트 전처리 (토큰화, 불용어 제거)
        
        Args:
            text: 전처리할 텍스트
            language: 언어 ("en" 또는 "ko")
            
        Returns:
            토큰 리스트
        """
        if language == "ko":
            return self.tokenize_ko_simple(text)
        return self.tokenize_en(text)
    
    def _mrr_at_k(self, ranked_rel: List[int], k: int=10) -> float:
        """Mean Reciprocal Rank at K"""
        k = min(k, len(ranked_rel))
        for i in range(k):
            if ranked_rel[i] == 1:
                return 1.0/(i+1)
        return 0.0

    def _ndcg_at_k(self, ranked_rel: List[int], k: int=10) -> float:
        """Normalized Discounted Cumulative Gain at K"""
        import math
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

    def _recall_at_k(self, ranked_rel: List[int], total_rel: int, k: int=10) -> float:
        """Recall at K"""
        if total_rel == 0: 
            return 0.0
        return sum(ranked_rel[:min(k,len(ranked_rel))]) / total_rel

    def bm25_params(self, language: str) -> Dict[str, float]:
        """언어별 BM25 하이퍼파라미터"""
        if language == "en":
            return dict(k1=1.5, b=0.75)
        elif language == "ko":
            # 한국어는 형태소 토큰 기준 문서 길이 편차가 커지는 경우가 있어 b를 다소 낮춤
            return dict(k1=1.2, b=0.65)
        return dict(k1=1.5, b=0.75)  # 기본값

    def build_bm25_index(self, passages: List[str], language: str = "en") -> BM25Okapi:
        """
        BM25 인덱스 구축
        
        Args:
            passages: 문서 리스트
            language: 언어 ("en" 또는 "ko")
            
        Returns:
            BM25 인덱스
        """
        print("BM25 인덱스 구축 중...")
        
        # 문서 전처리
        processed_passages = []
        for passage in passages:
            tokens = self.preprocess_text(passage, language)
            processed_passages.append(tokens)
        
        # BM25 인덱스 생성 (언어별 파라미터 적용)
        params = self.bm25_params(language)
        bm25 = BM25Okapi(processed_passages, **params)
        
        print(f"BM25 인덱스 구축 완료: {len(passages)}개 문서 (언어: {language}, 파라미터: {params})")
        return bm25
    
    def _bm25_rank_for_group(self, query: str, passages: List[str], language: str="en") -> Tuple[List[float], List[int]]:
        """쿼리 그룹 전용 BM25 랭킹 계산"""
        # 쿼리 그룹 전용 인덱스 구성
        corpus_tokens = [self.preprocess_text(p, language) for p in passages]
        params = self.bm25_params(language)
        bm25 = BM25Okapi(corpus_tokens, **params)
        q_tokens = self.preprocess_text(query, language)
        scores = bm25.get_scores(q_tokens)  # len == len(passages)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        

        
        return scores, order

    def run_bm25_benchmark(self, data: List[List[Any]], language: str="en") -> Tuple[List[float], Dict[str, float], List[Tuple[int, float]]]:
        """
        BM25 랭킹 벤치마크 실행 (쿼리별 그룹 평가)
        
        Args:
            data: 데이터 리스트 (헤더 포함)
            language: 언어 ("en" 또는 "ko")
            
        Returns:
            all_scores: (행 순서) bm25_score 리스트
            metrics: 순위 기반 지표 집계
            row_scores: [(sheet_row_idx, bm25_score), ...] 시트 업데이트 매핑용
        """
        print("BM25 랭킹 벤치마크 실행 중...")

        # 헤더 탐색
        headers = None
        header_row_idx = -1
        for i, row in enumerate(data):
            if {'query','passage','relevant'}.issubset(set(row)):
                headers = row
                header_row_idx = i
                break
        if headers is None:
            print("필수 컬럼(query, passage, relevant)을 찾지 못했습니다.")
            return [], {}, []

        q_idx = headers.index('query')
        p_idx = headers.index('passage')
        r_idx = headers.index('relevant')

        # 시트의 데이터 영역
        rows = data[header_row_idx+1:]

        # 같은 query별로 그룹핑(원본 행 인덱스 포함)
        from collections import defaultdict
        groups = defaultdict(list)  # query -> list of (row_idx_in_sheet, passage, relevant)
        for local_i, row in enumerate(rows):
            if len(row) <= max(q_idx, p_idx, r_idx): 
                continue
            q, p, rel = row[q_idx], row[p_idx], row[r_idx]
            if q is None or p is None: 
                continue
            try:
                rel = int(rel)
            except:
                rel = 0
            sheet_row_idx = header_row_idx + 1 + local_i  # 0-based → 시트 내 상대 위치
            groups[q].append((sheet_row_idx, p, rel))

        # 쿼리별 랭킹 및 지표 집계
        Kvals = (10, 50, 100)
        sum_metrics = {f"P@{k}":0.0 for k in Kvals}
        sum_metrics.update({f"R@{k}":0.0 for k in Kvals})
        sum_metrics.update({f"MRR@{k}":0.0 for k in Kvals})
        sum_metrics.update({f"nDCG@{k}":0.0 for k in Kvals})
        n_queries = 0

        # 점수 결과를 "행 인덱스"와 함께 저장(시트 업데이트용)
        row_scores: List[Tuple[int,float]] = []

        for q, items in groups.items():
            passages = [p for _, p, _ in items]
            labels   = [rel for _, _, rel in items]
            if not passages:
                continue

            scores, order = self._bm25_rank_for_group(q, passages, language=language)
            ranked_rel = [labels[i] for i in order]
            total_rel  = sum(labels)

            # 지표
            for K in Kvals:
                # Precision@K
                topk = ranked_rel[:min(K, len(ranked_rel))]
                sum_metrics[f"P@{K}"] += (sum(topk)/max(len(topk),1))
                # Recall@K
                sum_metrics[f"R@{K}"] += self._recall_at_k(ranked_rel, total_rel, K)
                # MRR@K
                sum_metrics[f"MRR@{K}"] += self._mrr_at_k(ranked_rel, K)
                # nDCG@K
                sum_metrics[f"nDCG@{K}"] += self._ndcg_at_k(ranked_rel, K)

            n_queries += 1

            # 점수를 원래 행에 매핑해 저장
            for rank_pos, idx in enumerate(order):
                sheet_row_idx, _, _ = items[idx]
                score = float(scores[idx])
                row_scores.append((sheet_row_idx, score))
                if n_queries <= 2:  # 처음 2개 쿼리만 디버깅
                    print(f"    쿼리 {n_queries}, 순위 {rank_pos+1}: 행 {sheet_row_idx}, 점수 {score}")

        if n_queries == 0:
            print("평가 가능한 쿼리가 없습니다.")
            return [], {}, []

        # 평균 지표 계산
        metrics = {k: v/n_queries for k, v in sum_metrics.items()}
        metrics['n_queries'] = n_queries  # 쿼리 수 추가
        metrics['model'] = f'BM25 ({language})'  # 모델 정보 추가

        # 행 순서대로 점수 배열 만들기(시트 업데이트 편의)
        max_row_idx = max((i for i, _ in row_scores), default=-1)
        all_scores = [0.0]*(max_row_idx+1)  # 기본값 0.0
        for ridx, sc in row_scores:
            all_scores[ridx] = sc

        print("BM25 랭킹 벤치마크 완료")
        return all_scores, metrics, row_scores
    
    def run_bm25(self, queries: List[str], passages: List[str], ground_truth: List[int], language: str = "en") -> Tuple[List[float], Dict[str, float]]:
        """
        기존 BM25 벤치마크 실행
        
        Args:
            queries: 쿼리 리스트
            passages: 문서 리스트
            ground_truth: 정답 레이블
            language: 언어 ("en" 또는 "ko")
            
        Returns:
            점수 리스트와 성능 지표
        """
        print("기존 BM25 벤치마크 실행 중...")
        
        # BM25 인덱스 구축
        bm25 = self.build_bm25_index(passages, language)
        
        # 각 쿼리에 대해 BM25 점수 계산
        scores = []
        predictions = []
        
        for i, query in enumerate(queries):
            # 쿼리 전처리
            query_tokens = self.preprocess_text(query, language)
            
            # BM25 점수 계산
            doc_scores = bm25.get_scores(query_tokens)
            
            # 현재 문서의 점수 (자기 자신에 대한 점수)
            current_score = doc_scores[i]
            scores.append(current_score)
            
            # 임계값 기반 예측 (점수가 높으면 관련성 있음)
            # 더 낮은 임계값 설정: 평균 - 0.5 * 표준편차
            threshold = np.mean(doc_scores) - 0.5 * np.std(doc_scores)
            prediction = 1 if current_score > threshold else 0
            predictions.append(prediction)
            
            # 디버깅 정보 출력 (처음 3개만)
            if i < 3:
                print(f"  샘플 {i+1}: 점수={current_score:.4f}, 임계값={threshold:.4f}, 예측={prediction}, 실제={ground_truth[i]}")
        
        # 성능 지표 계산
        print(f"예측 결과: {predictions}")
        print(f"실제 정답: {ground_truth}")
        print(f"예측 분포: 0={predictions.count(0)}개, 1={predictions.count(1)}개")
        print(f"정답 분포: 0={ground_truth.count(0)}개, 1={ground_truth.count(1)}개")
        
        # 정확도는 항상 계산 가능
        accuracy = accuracy_score(ground_truth, predictions)
        
        if len(set(ground_truth)) > 1:  # 클래스가 2개 이상인 경우
            precision = precision_score(ground_truth, predictions, zero_division=0)
            recall = recall_score(ground_truth, predictions, zero_division=0)
            f1 = f1_score(ground_truth, predictions, zero_division=0)
        else:
            print("경고: 모든 정답이 동일합니다. Precision, Recall, F1 계산이 불가능합니다.")
            precision = recall = f1 = 0.0
        
        performance_metrics = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'accuracy': accuracy,
            'total_samples': len(queries),
            'model': 'BM25'
        }
        
        print("기존 BM25 벤치마크 완료")
        return scores, performance_metrics
    
    def update_sheet_with_bm25_results(self, sheet_name: str, row_scores: List[Tuple[int,float]]) -> bool:
        """
        구글 시트에 BM25 결과 추가
        
        Args:
            sheet_name: 시트 이름
            row_scores: [(sheet_row_idx, bm25_score), ...]
            sheet_row_idx는 0-based로, 구글시트 실제 행 번호는 +1 필요.
            
        Returns:
            성공 여부
        """
        print(f"BM25 결과를 시트에 추가 중: {sheet_name}")
        try:
            range_name = f"{sheet_name}!A:Z"
            existing = self.sheet_manager.get_sheet_data(range_name)
            if not existing:
                print("기존 데이터를 찾을 수 없습니다.")
                return False

            # 헤더 위치
            header_row_idx = -1
            for i, row in enumerate(existing):
                if 'query' in row and 'passage' in row and 'relevant' in row:
                    header_row_idx = i
                    break
            if header_row_idx == -1:
                print("헤더를 찾을 수 없습니다.")
                return False

            headers = existing[header_row_idx].copy()
            if 'bm25_score' not in headers:
                headers.append('bm25_score')
                header_range = f"{sheet_name}!A{header_row_idx+1}:{chr(ord('A')+len(headers)-1)}{header_row_idx+1}"
                self.sheet_manager.service.spreadsheets().values().update(
                    spreadsheetId=self.spreadsheet_id,
                    range=header_range,
                    valueInputOption='RAW',
                    body={'values':[headers]}
                ).execute()

            col_idx = len(headers)  # 1-based 컬럼
            updates = []
            print(f"업데이트할 점수 수: {len(row_scores)}")
            for sheet_row_idx, score in row_scores:
                # 시트 실제 행 번호
                row_no = sheet_row_idx + 1
                col_letter = chr(ord('A') + col_idx - 1)
                updates.append({
                    'range': f"{sheet_name}!{col_letter}{row_no}",
                    'values': [[float(score)]]
                })
                

                
                if len(updates) <= 3:  # 처음 3개만 디버깅 출력
                    print(f"  행 {row_no}, 컬럼 {col_letter}: {score}")
            if updates:
                batch_body = {'valueInputOption':'RAW', 'data': updates}
                self.sheet_manager.service.spreadsheets().values().batchUpdate(
                    spreadsheetId=self.spreadsheet_id,
                    body=batch_body
                ).execute()

            print(f"배치 업데이트 완료: {len(updates)}개 셀")
            return True

        except Exception as e:
            print(f"BM25 결과 추가 중 오류: {e}")
            return False
    
    def run_benchmark_experiment(self, experiment_name: str, 
                                model_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        BM25 벤치마크 실험 실행
        
        Args:
            experiment_name: 실험 이름
            model_config: 모델 설정
            
        Returns:
            실험 결과
        """
        print(f"BM25 벤치마크 실험 시작: {experiment_name}")
        
        # 실험 시작 시간
        start_time = datetime.now()
        
        # 실험 결과 초기화
        experiment_results = {
            'experiment_name': experiment_name,
            'start_time': start_time.strftime("%Y-%m-%d %H:%M:%S"),
            'model_config': model_config or {},
            'results': {}
        }
        
        try:
            # 실험 데이터 가져오기
            sheet_name = "MS MARCO 데이터셋 실험결과"
            range_name = f"{sheet_name}!A:Z"
            data = self.sheet_manager.get_sheet_data(range_name)
            
            print(f"데이터 구조 확인: {len(data)}행")
            if data:
                print(f"첫 번째 행: {data[0]}")
                print(f"두 번째 행: {data[1] if len(data) > 1 else 'N/A'}")
                print(f"세 번째 행: {data[2] if len(data) > 2 else 'N/A'}")
            
            if not data or len(data) < 3:
                print("실험 데이터가 부족합니다.")
                experiment_results['error'] = "데이터 부족"
                return experiment_results
            
            # BM25 벤치마크 실행
            bm25_scores, performance_metrics, row_scores = self.run_bm25_benchmark(data, language="en")
            
            if not bm25_scores:
                print("BM25 벤치마크 실행 실패")
                experiment_results['error'] = "BM25 실행 실패"
                return experiment_results
            
            # BM25 결과를 시트에 추가
            update_success = self.update_sheet_with_bm25_results(sheet_name, row_scores)
            
            if not update_success:
                print("BM25 결과 시트 업데이트 실패")
                experiment_results['error'] = "결과 업데이트 실패"
                return experiment_results
            
            # 실험 종료 시간
            end_time = datetime.now()
            experiment_results['end_time'] = end_time.strftime("%Y-%m-%d %H:%M:%S")
            experiment_results['duration'] = (end_time - start_time).total_seconds()
            
            # 성능 지표 저장
            experiment_results['results'] = performance_metrics
            experiment_results['results']['processing_time'] = experiment_results['duration']
            
            # 쿼리 수 계산 - run_bm25_benchmark에서 반환된 metrics에서 가져오기
            if 'n_queries' in performance_metrics:
                experiment_results['results']['total_queries'] = performance_metrics['n_queries']
            else:
                # fallback: 데이터에서 직접 계산
                headers = None
                for i, row in enumerate(data):
                    if {'query','passage','relevant'}.issubset(set(row)):
                        headers = row
                        break
                if headers:
                    q_idx = headers.index('query')
                    rows = data[i+1:]
                    unique_queries = set()
                    for row in rows:
                        if len(row) > q_idx and row[q_idx]:
                            unique_queries.add(row[q_idx])
                    experiment_results['results']['total_queries'] = len(unique_queries)
                else:
                    experiment_results['results']['total_queries'] = 0
            
            # 콘솔에 결과 요약 출력
            self.print_benchmark_summary(experiment_results, bm25_scores)
            
            print(f"BM25 벤치마크 실험 완료: {experiment_name}")
            return experiment_results
            
        except Exception as e:
            print(f"BM25 벤치마크 실험 중 오류 발생: {e}")
            experiment_results['error'] = str(e)
            return experiment_results
    
    def print_benchmark_summary(self, experiment_results: Dict[str, Any], bm25_scores: List[float]):
        """
        벤치마크 결과 요약을 콘솔에 출력
        
        Args:
            experiment_results: 실험 결과
            bm25_scores: BM25 점수 리스트
        """
        print("\n" + "="*60)
        print("BM25 벤치마크 결과 요약")
        print("="*60)
        
        print(f"실험 이름: {experiment_results['experiment_name']}")
        print(f"시작 시간: {experiment_results['start_time']}")
        print(f"종료 시간: {experiment_results.get('end_time', 'N/A')}")
        print(f"소요 시간: {experiment_results.get('duration', 'N/A'):.2f}초")
        
        print("\n성능 지표:")
        results = experiment_results.get('results', {})
        print(f"  모델: {results.get('model', 'Unknown')}")
        
        # 순위 기반 지표 출력
        for k in [10, 50, 100]:
            print(f"  Precision@{k}: {results.get(f'P@{k}', 0):.4f}")
            print(f"  Recall@{k}: {results.get(f'R@{k}', 0):.4f}")
            print(f"  MRR@{k}: {results.get(f'MRR@{k}', 0):.4f}")
            print(f"  nDCG@{k}: {results.get(f'nDCG@{k}', 0):.4f}")
        
        print(f"  총 쿼리 수: {results.get('total_queries', 0)}")
        
        print("\n모델 점수 통계:")
        if bm25_scores:
            scores = np.array(bm25_scores)
            print(f"  평균 점수: {np.mean(scores):.4f}")
            print(f"  표준편차: {np.std(scores):.4f}")
            print(f"  최소 점수: {np.min(scores):.4f}")
            print(f"  최대 점수: {np.max(scores):.4f}")
            print(f"  중간값: {np.median(scores):.4f}")
        
        print("="*60)
    
    def save_experiment_results(self, results: Dict[str, Any], 
                               sheet_name: str = "MS MARCO 데이터셋 실험결과") -> bool:
        """
        실험 결과를 구글 시트에 저장
        
        Args:
            results: 실험 결과
            sheet_name: 시트 이름
            
        Returns:
            성공 여부
        """
        print(f"실험 결과 저장 중: {sheet_name}")
        
        try:
            # 기존 데이터 삭제
            try:
                self.sheet_manager.service.spreadsheets().values().clear(
                    spreadsheetId=self.spreadsheet_id,
                    range=f"{sheet_name}!A:Z"
                ).execute()
                print(f"기존 데이터 삭제 완료: {sheet_name}")
            except Exception as e:
                print(f"기존 데이터 삭제 중 오류 (무시): {e}")
            
            # 결과를 행 형태로 변환
            result_rows = []
            
            # 실험 정보
            result_rows.append([f"실험 이름: {results['experiment_name']}"])
            result_rows.append([f"시작 시간: {results['start_time']}"])
            result_rows.append([f"종료 시간: {results.get('end_time', 'N/A')}"])
            result_rows.append([f"소요 시간: {results.get('duration', 'N/A')}초"])
            result_rows.append([])  # 빈 행
            
            # 모델 설정
            if results.get('model_config'):
                result_rows.append(["모델 설정:"])
                for key, value in results['model_config'].items():
                    result_rows.append([f"  {key}: {value}"])
                result_rows.append([])  # 빈 행
            
            # 성능 결과
            if results.get('results'):
                result_rows.append(["성능 결과:"])
                for key, value in results['results'].items():
                    result_rows.append([f"  {key}: {value}"])
            
            # 오류 정보
            if results.get('error'):
                result_rows.append([])  # 빈 행
                result_rows.append([f"오류: {results['error']}"])
            
            # 구글 시트에 업로드 (덮어쓰기)
            range_name = f"{sheet_name}!A1"
            body = {'values': result_rows}
            result = self.sheet_manager.service.spreadsheets().values().update(
                spreadsheetId=self.spreadsheet_id,
                range=range_name,
                valueInputOption='RAW',
                body=body
            ).execute()
            
            updated_rows = result.get('updatedRows', 0)
            print(f"실험 결과 저장 완료: {updated_rows}행")
            return True
                
        except Exception as e:
            print(f"실험 결과 저장 중 오류 발생: {e}")
            return False
    

    
    def run_full_benchmark_pipeline(self, experiment_name: str = "BM25 Ranking Benchmark",
                                   model_config: Dict[str, Any] = None) -> bool:
        """
        전체 벤치마크 파이프라인 실행
        
        Args:
            experiment_name: 실험 이름
            model_config: 모델 설정
            
        Returns:
            성공 여부
        """
        print("=== 벤치마크 파이프라인 시작 ===")
        
        try:
            # 1. 데이터셋 복사
            copy_success = self.copy_dataset_to_experiment_sheet(
                source_sheet="MS MARCO 데이터셋",  # 실제 시트 이름으로 변경 필요할 수 있음
                target_sheet="MS MARCO 데이터셋 실험결과"
            )
            
            if not copy_success:
                print("데이터셋 복사 실패")
                return False
            
            # 2. 벤치마크 실험 실행
            experiment_results = self.run_benchmark_experiment(experiment_name, model_config)
            
            # 3. 실험 결과를 "결과" 시트에 저장
            save_success = self.save_experiment_results(experiment_results, "결과")
            
            if save_success:
                print("=== 벤치마크 파이프라인 완료 ===")
                return True
            else:
                print("실험 결과 저장 실패")
                return False
                
        except Exception as e:
            print(f"벤치마크 파이프라인 중 오류 발생: {e}")
            return False


def main():
    """벤치마크 실행 테스트"""
    SPREADSHEET_ID = "1yCmOp7__YWKB4k4pUWSF7DHfqvaR2-WVLnPkz5AfzN4"
    
    # 벤치마크 러너 생성
    benchmark_runner = BenchmarkRunner(SPREADSHEET_ID)
    
    # 전체 파이프라인 실행
    success = benchmark_runner.run_full_benchmark_pipeline(
        experiment_name="BM25 Ranking Benchmark"
    )
    
    if success:
        print("벤치마크 파이프라인이 성공적으로 완료되었습니다.")
    else:
        print("벤치마크 파이프라인 실행에 실패했습니다.")


if __name__ == "__main__":
    main()
