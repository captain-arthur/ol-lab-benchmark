from gsheet.gsheet import GoogleSheetManager
from bm25.data_loader import MSMarcoDataLoader
from bm25.benchmark_runner import BenchmarkRunner


def main():
    
    SPREADSHEET_ID = "1yCmOp7__YWKB4k4pUWSF7DHfqvaR2-WVLnPkz5AfzN4"
    
    try:
        sheet_manager = GoogleSheetManager(SPREADSHEET_ID)
        
        # MS MARCO 데이터셋 로드
        print("MS MARCO 데이터셋 로딩 중...")
        data_loader = MSMarcoDataLoader()
        
        # MS MARCO test 데이터셋 최대 개수: 101,092개
        # 원하는 샘플 수를 설정하세요 (예: 1000, 5000, 10000, 50000, 101092)
        desired_samples = 100  # 여기서 원하는 개수를 설정하세요
        
        print("새로운 데이터 다운로드 중...")
        ms_marco_data = data_loader.load_ms_marco_passage_ranking(
            split="validation", 
            max_samples=desired_samples
        )
        
        if not ms_marco_data:
            print("데이터 로딩에 실패했습니다.")
            return
        
        # MS MARCO 데이터 업로드 (중복 체크 포함)
        success = sheet_manager.upload_data_with_timestamp(
            sheet_name="MS MARCO 데이터셋",
            data=ms_marco_data,
            key_columns=["query", "passage"]  # query와 passage 조합으로 중복 체크
        )
        
        if success:
            print("데이터 업로드가 완료되었습니다.")
        else:
            print("데이터 업로드에 실패했습니다.")
            
    except Exception as e:
        print(f"오류 발생: {e}")


def run_benchmark():
    """벤치마크 실행 함수"""
    print("\n=== 벤치마크 실행 ===")
    
    SPREADSHEET_ID = "1yCmOp7__YWKB4k4pUWSF7DHfqvaR2-WVLnPkz5AfzN4"
    
    try:
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
            
    except Exception as e:
        print(f"벤치마크 실행 중 오류 발생: {e}")


if __name__ == "__main__":
    # 데이터셋 업로드
    main()
    
    # 벤치마크 실행 (선택사항)
    run_benchmark()
