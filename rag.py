# rag.py

import os
import sys
import psycopg2
import traceback
import re # 정규 표현식을 위해 추가
from sentence_transformers import SentenceTransformer
from pdfminer.high_level import extract_text
import requests
from typing import List, Tuple
import json

# ... (설정 부분은 동일) ...
TARGET_DIR = "./docs"
MODEL_NAME = "nlpai-lab/KURE-v1"
DB_CONFIG = {
    "host": "localhost", "port": "5432", "dbname": "ragtest",
    "user": "eden", "password": "qwer123"
}
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# LLM 설정 (OLLAMA 설정)
OLLAMA_ENDPOINT = os.getenv("OLLAMA_ENDPOINT", "https://api.hamonize.com/ollama/api/chat")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "airun-chat:latest")

# -------------------------------------------------
# 텍스트 정규화 함수 (새로 추가)
# -------------------------------------------------
def normalize_text(text: str) -> str:
    """텍스트 정규화를 위한 함수"""
    if not text:
        return ""
    
    # 0. NULL 바이트 및 제어 문자 제거 (PostgreSQL 오류 방지)
    text = text.replace('\x00', '')
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    
    # 1. 소문자로 변환
    text = text.lower()
    
    # 2. 불필요한 기호 제거 (한글, 영문, 숫자, 기본 구두점 제외)
    # 한글, 영문, 숫자, 공백, 그리고 문장 구분을 위한 온점(.)만 남깁니다.
    # text = re.sub(r'[^가-힣a-z0-9\s\.]', ' ', text)
    text = re.sub(r'[^가-힣a-z0-9\s\./_-]', ' ', text)
    
    # 3. 여러 개의 공백을 하나의 공백으로 축소
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 4. (선택사항) 불용어(Stopword) 제거
    # 성능에 큰 영향을 주지 않거나, 오히려 문맥을 해칠 수 있어 최근에는 잘 사용하지 않기도 합니다.
    # 필요 시 아래 주석을 해제하고, 불용어 사전을 정의하여 사용하세요.
    # stopwords = ['은', '는', '이', '가', '을', '를', '의', '에', '과', '와', '도']
    # words = text.split()
    # words = [word for word in words if word not in stopwords]
    # text = ' '.join(words)
    
    return text

# -------------------------------------------------
# LLM 기반 요약 및 청킹 함수들
# -------------------------------------------------

def call_ollama_api(messages: List[dict], max_tokens: int = 150) -> str:
    """OLLAMA API를 호출하는 함수"""
    try:
        payload = {
            "model": OLLAMA_MODEL,
            "messages": messages,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": 0.3
            }
        }
        
        response = requests.post(
            OLLAMA_ENDPOINT,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            return result.get("message", {}).get("content", "").strip()
        else:
            print(f"⚠️ OLLAMA API 오류: {response.status_code} - {response.text}")
            return ""
            
    except Exception as e:
        print(f"⚠️ OLLAMA API 호출 중 오류 발생: {e}")
        return ""

def get_paragraph_summary(text: str, max_tokens: int = 150) -> str:
    """단락의 요약을 생성하는 함수"""
    if not OLLAMA_ENDPOINT or not OLLAMA_MODEL:
        print("⚠️ OLLAMA 설정이 되지 않아 요약 기능을 사용할 수 없습니다.")
        return ""
    
    messages = [
        {
            "role": "system", 
            "content": "당신은 텍스트 요약 전문가입니다. 주어진 텍스트의 핵심 내용을 간결하게 요약해주세요. 한국어로 답변하세요."
        },
        {
            "role": "user", 
            "content": f"다음 텍스트를 2-3문장으로 요약해주세요:\n\n{text[:2000]}"  # 토큰 제한을 위해 2000자로 제한
        }
    ]
    
    return call_ollama_api(messages, max_tokens)

def find_topic_boundaries(text: str, paragraph_size: int = 1000) -> List[Tuple[int, str]]:
    """LLM을 사용하여 주제 전환점을 찾는 함수"""
    if not OLLAMA_ENDPOINT or not OLLAMA_MODEL:
        print("⚠️ OLLAMA 설정이 되지 않아 주제 분석 기능을 사용할 수 없습니다.")
        return [(0, "전체 문서")]
    
    # 텍스트를 단락 단위로 분할
    paragraphs = []
    for i in range(0, len(text), paragraph_size):
        paragraph = text[i:i + paragraph_size]
        if paragraph.strip():
            paragraphs.append((i, paragraph))
    
    if len(paragraphs) <= 1:
        return [(0, "전체 문서")]
    
    boundaries = [(0, "문서 시작")]
    
    try:
        # 각 단락의 요약 생성
        summaries = []
        for i, (pos, paragraph) in enumerate(paragraphs):
            print(f"  📝 단락 {i+1}/{len(paragraphs)} 요약 생성 중...")
            summary = get_paragraph_summary(paragraph)
            summaries.append((pos, summary))
        
        # 연속된 요약들을 비교하여 주제 전환점 찾기
        for i in range(1, len(summaries)):
            prev_summary = summaries[i-1][1]
            curr_summary = summaries[i][1]
            
            if prev_summary and curr_summary:
                # LLM에게 두 요약이 다른 주제인지 판단 요청
                is_different_topic = check_topic_change(prev_summary, curr_summary)
                if is_different_topic:
                    boundaries.append((summaries[i][0], f"주제 전환: {curr_summary[:50]}..."))
                    print(f"  🔄 주제 전환점 발견: 위치 {summaries[i][0]}")
        
    except Exception as e:
        print(f"⚠️ 주제 경계 분석 중 오류 발생: {e}")
        return [(0, "전체 문서")]
    
    return boundaries

def check_topic_change(prev_summary: str, curr_summary: str) -> bool:
    """두 요약 사이에 주제 변화가 있는지 LLM으로 판단"""
    messages = [
        {
            "role": "system",
            "content": "당신은 텍스트 분석 전문가입니다. 두 텍스트 요약이 서로 다른 주제를 다루는지 판단해주세요. 'YES' 또는 'NO'로만 답변하세요."
        },
        {
            "role": "user",
            "content": f"다음 두 요약이 서로 다른 주제를 다루나요?\n\n이전 요약: {prev_summary}\n\n현재 요약: {curr_summary}\n\n다른 주제라면 YES, 같은 주제라면 NO로 답변하세요."
        }
    ]
    
    try:
        answer = call_ollama_api(messages, max_tokens=10)
        return answer.upper() == "YES"
    except Exception as e:
        print(f"⚠️ 주제 변화 판단 중 오류 발생: {e}")
        return False

def summarization_aware_chunking(text: str, min_chunk_size: int = 300, max_chunk_size: int = 800) -> List[str]:
    """요약 기반 청킹을 수행하는 메인 함수"""
    print("🧠 LLM 기반 요약 청킹 시작...")
    
    if not OLLAMA_ENDPOINT or not OLLAMA_MODEL:
        print("⚠️ OLLAMA 설정이 없어 기본 청킹 방식을 사용합니다.")
        return chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
    
    # 주제 경계점 찾기
    boundaries = find_topic_boundaries(text)
    print(f"  📍 {len(boundaries)}개의 주제 경계점 발견")
    
    chunks = []
    
    for i in range(len(boundaries)):
        start_pos = boundaries[i][0]
        end_pos = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(text)
        
        section_text = text[start_pos:end_pos].strip()
        if not section_text:
            continue
        
        # 섹션이 너무 크면 추가로 분할
        if len(section_text) <= max_chunk_size:
            chunks.append(section_text)
            print(f"  ✂️ 청크 생성: {len(section_text)}자 (주제: {boundaries[i][1][:30]}...)")
        else:
            # 큰 섹션은 기본 방식으로 추가 분할
            sub_chunks = chunk_text(section_text, max_chunk_size, CHUNK_OVERLAP)
            chunks.extend(sub_chunks)
            print(f"  ✂️ 대형 섹션을 {len(sub_chunks)}개 청크로 분할")
    
    # 너무 작은 청크들은 인접 청크와 병합
    merged_chunks = []
    current_chunk = ""
    
    for chunk in chunks:
        if len(current_chunk + chunk) <= max_chunk_size:
            current_chunk += (" " + chunk if current_chunk else chunk)
        else:
            if current_chunk:
                merged_chunks.append(current_chunk)
            current_chunk = chunk
    
    if current_chunk:
        merged_chunks.append(current_chunk)
    
    print(f"  🎯 최종 {len(merged_chunks)}개 청크 생성 완료")
    return merged_chunks

# -------------------------------------------------

# ... (chunk_text 함수는 동일) ...
def chunk_text(text, chunk_size, overlap):
    if not text: return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

# DB 삽입 함수 수정: original_filename 인자 추가
def insert_chunk_to_db(cursor, title, content, embedding, source_file, original_filename, chunk_index, total_chunks):
    # 모든 문자열 필드에서 NULL 바이트 제거
    def clean_string(s):
        if s is None:
            return None
        return str(s).replace('\x00', '').replace('\r', '').replace('\n\n\n', '\n\n')
    
    title = clean_string(title)
    content = clean_string(content)
    source_file = clean_string(source_file)
    original_filename = clean_string(original_filename)
    
    embedding_str = "[" + ",".join(map(str, embedding)) + "]"
    sql = """
        INSERT INTO documents (title, content, embedding, source_file, original_filename, chunk_index, total_chunks)
        VALUES (%s, %s, %s::vector, %s, %s, %s, %s)
    """
    try:
        print(f"    💾 DB 삽입 시도: 청크 {chunk_index}/{total_chunks}")
        cursor.execute(sql, (title, content, embedding_str, source_file, original_filename, chunk_index, total_chunks))
        print(f"    ✅ DB 삽입 성공")
    except Exception as e:
        print(f"    ❌ DB 삽입 오류 발생: {e}")
        print(f"    디버그 정보: title 길이={len(title) if title else 0}, content 길이={len(content) if content else 0}")
        raise

# 파일 처리 함수 수정: original_filename 인자 추가
def process_file(file_path, unique_filename, original_filename, model, cursor):
    """단일 파일을 처리하여 청크 단위로 DB에 저장하는 함수"""
    print(f"\n📄 '{original_filename}' (저장명: {unique_filename}) 파일 처리 시작...")
    try:
        # 파일 확장자에 따라 내용 추출
        ext = os.path.splitext(unique_filename)[1].lower()
        text = ""
        if ext == '.pdf':
            text = extract_text(file_path)
        elif ext in ['.txt', '.md']:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        else:
            print(f"  ⚠️ 지원하지 않는 파일 형식: {unique_filename}")
            return False

        if not text or not text.strip():
            print(f"  ⚠️ 내용이 없는 파일입니다: {original_filename}")
            return True

        # --- 여기가 핵심 수정 부분 ---
        # 텍스트를 청킹하기 전에 정규화를 수행합니다.
        print("  - 텍스트 정규화 수행 중...")
        normalized_text = normalize_text(text)
        print(f"  - 정규화 완료 (원본: {len(text)}자 -> 정규화: {len(normalized_text)}자)")
        
        # 요약 기반 청킹 또는 기본 청킹 선택
        use_summarization_chunking = os.getenv("USE_SUMMARIZATION_CHUNKING", "false").lower() == "true"
        
        if use_summarization_chunking and OLLAMA_ENDPOINT and OLLAMA_MODEL:
            print("  🧠 OLLAMA 기반 요약 청킹 사용")
            chunks = summarization_aware_chunking(normalized_text)
        else:
            print("  ✂️ 기본 청킹 방식 사용")
            chunks = chunk_text(normalized_text, CHUNK_SIZE, CHUNK_OVERLAP)
        
        total_chunks = len(chunks)
        print(f"  ✅ {total_chunks}개의 청크로 분할 완료")
        # ---------------------------

        # 각 청크를 임베딩하고 DB에 저장
        for i, chunk in enumerate(chunks):
            # 청크 자체는 이미 정규화되었으므로 바로 임베딩
            if not chunk: continue # 빈 청크는 건너뛰기
            
            chunk_index = i + 1
            print(f"  🔄 청크 {chunk_index}/{total_chunks} 처리 중...")
            
            embedding = model.encode(chunk).tolist()
            
            chunk_title = f"{original_filename}_chunk_{chunk_index}"
            insert_chunk_to_db(cursor, chunk_title, chunk, embedding, unique_filename, original_filename, chunk_index, total_chunks)

        print(f"  ✅ 파일 처리 성공: {original_filename}")
        return True

    except Exception as e:
        print(f"  ❌ 파일 처리 중 심각한 오류 발생: {original_filename} - {e}")
        traceback.print_exc()
        return False

def main():
    print("==============================================")
    print("🚀 RAG 임베딩 스크립트 실행")
    print("==============================================")
    
    # 인자 2개를 받도록 수정
    if len(sys.argv) < 3:
        print("❌ 사용법: python rag.py <고유 파일명> <원본 파일명>")
        sys.exit(1)
    
    unique_filename = sys.argv[1]
    original_filename = sys.argv[2]
    print(f"🎯 처리 대상 파일: {original_filename} (고유명: {unique_filename})")

    model = None
    conn = None
    try:
        print("🤖 임베딩 모델 로딩...")
        model = SentenceTransformer(MODEL_NAME)
        print("✅ 모델 로딩 완료")

        print("🐘 데이터베이스 연결 시도...")
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        print("✅ 데이터베이스 연결 성공")

        # 파일 경로는 고유 파일명으로 확인
        file_path = os.path.join(TARGET_DIR, unique_filename)
        print(f"🔍 파일 경로 확인: {file_path}")
        if not os.path.isfile(file_path):
            print(f"❌ 파일을 찾을 수 없습니다!")
            sys.exit(1)

        # process_file 호출 시 두 파일명 모두 전달
        if process_file(file_path, unique_filename, original_filename, model, cur):
            conn.commit()
            print(f"\n🎉 최종 커밋 완료: '{original_filename}' 데이터가 DB에 저장되었습니다.")
        else:
            conn.rollback()
            print(f"\n⚠️ 롤백 완료: '{original_filename}' 처리 중 오류가 발생하여 DB 변경사항이 취소되었습니다.")
            sys.exit(1)

    except Exception as e:
        print("\n❌ 스크립트 실행 중 최상위 오류 발생!")
        traceback.print_exc()
        if conn:
            conn.rollback()
        sys.exit(1)
    finally:
        if conn:
            conn.close()
            print("\n🔌 데이터베이스 연결이 종료되었습니다.")
    
    print("🏁 모든 작업이 정상적으로 완료되었습니다.")

if __name__ == "__main__":
    main()