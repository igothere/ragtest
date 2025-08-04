#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG 요약 기반 청킹 예제 스크립트

이 스크립트는 LLM을 활용한 요약 기반 청킹 기능을 테스트하는 예제입니다.

사용법:
1. OLLAMA 설정:
   export OLLAMA_ENDPOINT="https://api.hamonize.com/ollama/api/chat"
   export OLLAMA_MODEL="airun-chat:latest"

2. 요약 기반 청킹 활성화:
   export USE_SUMMARIZATION_CHUNKING="true"

3. 스크립트 실행:
   python rag.py <고유파일명> <원본파일명>

예시:
   python rag.py document.pdf "중요한 문서.pdf"
"""

import os
import subprocess
import sys

def setup_environment():
    """환경 변수 설정 및 확인"""
    print("🔧 환경 설정 확인...")
    
    # OLLAMA 설정 확인
    ollama_endpoint = os.getenv("OLLAMA_ENDPOINT")
    ollama_model = os.getenv("OLLAMA_MODEL")
    
    if not ollama_endpoint:
        print("❌ OLLAMA_ENDPOINT 환경변수가 설정되지 않았습니다.")
        print("   다음 명령으로 설정하세요:")
        print('   export OLLAMA_ENDPOINT="https://api.hamonize.com/ollama/api/chat"')
        return False
    else:
        print(f"✅ OLLAMA 엔드포인트 확인됨: {ollama_endpoint}")
    
    if not ollama_model:
        print("❌ OLLAMA_MODEL 환경변수가 설정되지 않았습니다.")
        print("   다음 명령으로 설정하세요:")
        print('   export OLLAMA_MODEL="airun-chat:latest"')
        return False
    else:
        print(f"✅ OLLAMA 모델 확인됨: {ollama_model}")
    
    # 요약 청킹 모드 확인
    use_summarization = os.getenv("USE_SUMMARIZATION_CHUNKING", "false").lower()
    if use_summarization == "true":
        print("✅ 요약 기반 청킹 모드 활성화됨")
    else:
        print("⚠️ 요약 기반 청킹 모드가 비활성화되어 있습니다.")
        print("   다음 명령으로 활성화하세요:")
        print("   export USE_SUMMARIZATION_CHUNKING='true'")
        
        # 사용자에게 활성화 여부 묻기
        response = input("지금 활성화하시겠습니까? (y/n): ").lower()
        if response == 'y':
            os.environ["USE_SUMMARIZATION_CHUNKING"] = "true"
            print("✅ 요약 기반 청킹 모드 활성화됨")
        else:
            print("📝 기본 청킹 모드로 진행합니다.")
    
    return True

def run_rag_with_summarization(unique_filename, original_filename):
    """요약 기반 청킹으로 RAG 실행"""
    print(f"\n🚀 요약 기반 RAG 처리 시작...")
    print(f"   파일: {original_filename}")
    print(f"   고유명: {unique_filename}")
    
    try:
        # rag.py 실행
        result = subprocess.run([
            sys.executable, "rag.py", unique_filename, original_filename
        ], capture_output=True, text=True, encoding='utf-8')
        
        if result.returncode == 0:
            print("✅ RAG 처리 성공!")
            print("\n📊 처리 결과:")
            print(result.stdout)
        else:
            print("❌ RAG 처리 실패!")
            print("\n🔍 오류 내용:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ 실행 중 오류 발생: {e}")
        return False
    
    return True

def main():
    print("=" * 60)
    print("🧠 RAG 요약 기반 청킹 예제")
    print("=" * 60)
    
    # 환경 설정 확인
    if not setup_environment():
        sys.exit(1)
    
    # 명령행 인자 확인
    if len(sys.argv) < 3:
        print("\n📋 사용법:")
        print("   python rag_summarization_example.py <고유파일명> <원본파일명>")
        print("\n📝 예시:")
        print("   python rag_summarization_example.py doc.pdf '중요한문서.pdf'")
        sys.exit(1)
    
    unique_filename = sys.argv[1]
    original_filename = sys.argv[2]
    
    # 파일 존재 확인
    file_path = os.path.join("./docs", unique_filename)
    if not os.path.exists(file_path):
        print(f"❌ 파일을 찾을 수 없습니다: {file_path}")
        sys.exit(1)
    
    # RAG 처리 실행
    success = run_rag_with_summarization(unique_filename, original_filename)
    
    if success:
        print("\n🎉 모든 작업이 완료되었습니다!")
        print("\n💡 요약 기반 청킹의 장점:")
        print("   - 의미적으로 연관된 내용이 같은 청크에 포함됨")
        print("   - 주제 전환점에서 자연스럽게 분할됨")
        print("   - 검색 정확도 향상 기대")
        print("   - OLLAMA 모델을 사용하여 로컬 환경에서 처리")
    else:
        print("\n❌ 작업 중 오류가 발생했습니다.")
        sys.exit(1)

if __name__ == "__main__":
    main()