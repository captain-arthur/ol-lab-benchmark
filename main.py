from filter.keyword.run_ms_marco import run_ms_marco
from filter.keyword.run_fiqa import run_fiqa
from filter.semantic.run_banking77 import run_banking77
from filter.semantic.run_stsb import run_stsb
from filter.semantic.run_clinc150 import run_clinc150
from filter.llm.run_llm import run_lllmI

def main():
    # # ms marco 실험 실행
    # run_ms_marco()
    
    # # fiqa 실험 실행
    # run_fiqa()
    
    # # banking77 실험 실행
    # run_banking77()
    
    # stsb 실험 실행
    # run_stsb()
    
    # clinc150 실험 실행
    # run_clinc150()
    
    # llm 실험 실행
    run_lllmI()

if __name__ == "__main__":
    main()