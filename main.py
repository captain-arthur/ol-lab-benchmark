# 기존 모듈들 (주석 처리)
# from filter.keyword.run_ms_marco import run_ms_marco
# from filter.keyword.run_fiqa import run_fiqa
# from filter.semantic.run_banking77 import run_banking77
# from filter.semantic.run_stsb import run_stsb
# from filter.semantic.run_clinc150 import run_clinc150

# 새로운 리팩토링된 LLM 필터링 모듈들
from filter.llm.run_llm_banking77 import run_banking77_experiment
from filter.llm.run_llm_clinc150 import run_clinc150_experiment
from filter.llm.run_llm_stsb import run_stsb_experiment

def main():
    # # ms marco 실험 실행 (주석 처리)
    # run_ms_marco()
    
    # # fiqa 실험 실행 (주석 처리)
    # run_fiqa()
    
    # # banking77 실험 실행 (기존 semantic, 주석 처리)
    # run_banking77()
    
    # # stsb 실험 실행 (기존 semantic, 주석 처리)
    # run_stsb()
    
    # # clinc150 실험 실행 (기존 semantic, 주석 처리)
    # run_clinc150()
    
    # 새로운 LLM 필터링 실험들
    print("🔬 LLM Filtering Benchmarks")
    print("=" * 50)
    
    # Banking77 LLM 필터링
    print("\n📊 Running Banking77 LLM Filtering...")
    run_banking77_experiment()
    
    # CLINC150 LLM 필터링
    print("\n📊 Running CLINC150 LLM Filtering...")
    run_clinc150_experiment()
    
    # STS-B LLM 필터링
    print("\n📊 Running STS-B LLM Filtering...")
    run_stsb_experiment()

if __name__ == "__main__":
    main()