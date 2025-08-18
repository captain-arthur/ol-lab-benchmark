import os
import json
from datasets import load_dataset
import pandas as pd
from typing import List, Dict, Any


class MSMarcoDataLoader:
    """MS MARCO 데이터셋 로더"""
    
    def __init__(self, cache_dir: str = "./data"):
        """
        MSMarcoDataLoader 초기화
        
        Args:
            cache_dir: 데이터 캐시 디렉토리
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def load_ms_marco_passage_ranking(self, split: str = "train", max_samples: int = 1000) -> List[Dict[str, Any]]:
        """
        MS MARCO Passage Ranking 데이터셋 로드
        
        Args:
            split: 데이터 분할 ("train", "validation", "test")
            max_samples: 최대 생성할 행 수 (query-passage 쌍)
            
        Returns:
            처리된 데이터 리스트
        """
        print(f"MS MARCO Passage Ranking 데이터셋 로딩 중... (split: {split})")
        
        try:
            # 데이터셋 로드
            dataset = load_dataset("ms_marco", "v2.1", split=split, cache_dir=self.cache_dir)
            
            print(f"총 {len(dataset)}개 쿼리 로드됨")
            
            # 데이터 처리
            processed_data = []
            query_count = 0
            
            for i, item in enumerate(dataset):
                if i % 100 == 0:
                    print(f"쿼리 처리 중... {i}/{len(dataset)} (생성된 행: {len(processed_data)})")
                
                # 필요한 필드 추출
                query = item.get('query', '')
                passages = item.get('passages', {})
                answers = item.get('answers', [])
                
                # passages에서 텍스트와 관련성 정보 추출
                if passages and 'passage_text' in passages and 'is_selected' in passages:
                    passage_texts = passages['passage_text']
                    is_selected = passages['is_selected']
                    
                    # 각 passage에 대해 데이터 생성
                    for j, (passage_text, is_rel) in enumerate(zip(passage_texts, is_selected)):
                        if passage_text.strip():  # 빈 텍스트가 아닌 경우만
                            processed_data.append({
                                'query': query,
                                'passage': passage_text,
                                'relevant': '1' if is_rel else '0',
                                'passage_id': str(j),
                                'answers': '; '.join(answers) if answers else ''
                            })
                            
                            # 목표 샘플 수에 도달하면 중단
                            if len(processed_data) >= max_samples:
                                break
                    
                    query_count += 1
                
                # 목표 샘플 수에 도달하면 중단
                if len(processed_data) >= max_samples:
                    break
            
            print(f"처리 완료: {len(processed_data)}개 행 (쿼리 {query_count}개에서 생성)")
            return processed_data
            
        except Exception as e:
            print(f"데이터셋 로딩 오류: {e}")
            return []
    
    def load_ms_marco_qa(self, split: str = "train", max_samples: int = 1000) -> List[Dict[str, Any]]:
        """
        MS MARCO QA 데이터셋 로드
        
        Args:
            split: 데이터 분할 ("train", "validation", "test")
            max_samples: 최대 샘플 수
            
        Returns:
            처리된 데이터 리스트
        """
        print(f"MS MARCO QA 데이터셋 로딩 중... (split: {split})")
        
        try:
            # 데이터셋 로드
            dataset = load_dataset("ms_marco", "v1.1", split=split, cache_dir=self.cache_dir)
            
            print(f"총 {len(dataset)}개 샘플 로드됨")
            
            # 샘플 수 제한
            if max_samples and len(dataset) > max_samples:
                dataset = dataset.select(range(max_samples))
                print(f"처리할 샘플 수: {len(dataset)}개")
            
            # 데이터 처리
            processed_data = []
            for i, item in enumerate(dataset):
                if i % 100 == 0:
                    print(f"처리 중... {i}/{len(dataset)}")
                
                query = item.get('query', '')
                passages = item.get('passages', {})
                answers = item.get('answers', [])
                well_formed_answers = item.get('wellFormedAnswers', [])
                
                # passages에서 텍스트와 관련성 정보 추출
                if passages and 'passage_text' in passages and 'is_selected' in passages:
                    passage_texts = passages['passage_text']
                    is_selected = passages['is_selected']
                    
                    # 각 passage에 대해 데이터 생성
                    for j, (passage_text, is_rel) in enumerate(zip(passage_texts, is_selected)):
                        if passage_text.strip():  # 빈 텍스트가 아닌 경우만
                            processed_data.append({
                                'query': query,
                                'passage': passage_text,
                                'relevant': '1' if is_rel else '0',
                                'passage_id': j,
                                'answers': '; '.join(answers) if answers else '',
                                'well_formed_answers': '; '.join(well_formed_answers) if well_formed_answers else ''
                            })
                
                # 메모리 관리를 위해 중간에 저장
                if len(processed_data) >= max_samples:
                    break
            
            print(f"처리 완료: {len(processed_data)}개 샘플")
            return processed_data
            
        except Exception as e:
            print(f"데이터셋 로딩 오류: {e}")
            return []
    
    def save_to_json(self, data: List[Dict[str, Any]], filename: str):
        """
        데이터를 JSON 파일로 저장
        
        Args:
            data: 저장할 데이터
            filename: 파일명
        """
        filepath = os.path.join(self.cache_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"데이터 저장 완료: {filepath}")
    
    def load_from_json(self, filename: str) -> List[Dict[str, Any]]:
        """
        JSON 파일에서 데이터 로드
        
        Args:
            filename: 파일명
            
        Returns:
            로드된 데이터
        """
        filepath = os.path.join(self.cache_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"데이터 로드 완료: {filepath} ({len(data)}개 샘플)")
            return data
        else:
            print(f"파일이 존재하지 않습니다: {filepath}")
            return []


def main():
    """데이터 로더 테스트"""
    loader = MSMarcoDataLoader()
    
    # MS MARCO Passage Ranking 데이터 로드 (작은 샘플로 테스트)
    print("=== MS MARCO Passage Ranking 데이터 로드 ===")
    passage_data = loader.load_ms_marco_passage_ranking(split="train", max_samples=100)
    
    if passage_data:
        # JSON으로 저장
        loader.save_to_json(passage_data, "ms_marco_passage_sample.json")
        
        # 샘플 출력
        print("\n=== 샘플 데이터 ===")
        for i, item in enumerate(passage_data[:3]):
            print(f"\n샘플 {i+1}:")
            print(f"Query: {item['query']}")
            print(f"Passage: {item['passage'][:100]}...")
            print(f"Relevant: {item['relevant']}")
    
    # MS MARCO QA 데이터 로드 (작은 샘플로 테스트)
    print("\n=== MS MARCO QA 데이터 로드 ===")
    qa_data = loader.load_ms_marco_qa(split="train", max_samples=100)
    
    if qa_data:
        # JSON으로 저장
        loader.save_to_json(qa_data, "ms_marco_qa_sample.json")
        
        # 샘플 출력
        print("\n=== 샘플 데이터 ===")
        for i, item in enumerate(qa_data[:3]):
            print(f"\n샘플 {i+1}:")
            print(f"Query: {item['query']}")
            print(f"Passage: {item['passage'][:100]}...")
            print(f"Relevant: {item['relevant']}")
            print(f"Answers: {item.get('answers', 'N/A')}")


if __name__ == "__main__":
    main()
