# rag.py

import os
import sys
import psycopg2
import traceback
import re # 정규 표현식을 위해 추가
from sentence_transformers import SentenceTransformer
from pdfminer.high_level import extract_text
from hlm_processors import process_file_to_hlm_chunks

# ... (설정 부분은 동일) ...
TARGET_DIR = "/home/eden/rag/docs"
MODEL_NAME = "nlpai-lab/KURE-v1"
DB_CONFIG = {
    "host": "localhost", "port": "5432", "dbname": "ragtest",
    "user": "eden", "password": "qwer123"
}
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# -------------------------------------------------
# 텍스트 정규화 함수 (새로 추가)
# -------------------------------------------------
def normalize_text(text: str) -> str:
    """텍스트 정규화를 위한 함수"""
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
        raise

# 파일 처리 함수 수정: original_filename 인자 추가
def process_file(file_path, unique_filename, original_filename, model, cursor):
    """단일 파일을 처리하여 청크 단위로 DB에 저장하는 함수"""
    print(f"\n📄 '{original_filename}' (저장명: {unique_filename}) 파일 처리 시작...")
    try:
        # 파일 확장자에 따라 내용 추출
        # ext = os.path.splitext(unique_filename)[1].lower()
        text = ""
        # if ext == '.pdf':
        #     text = extract_text(file_path)
        # elif ext in ['.txt', '.md']:
        #     with open(file_path, 'r', encoding='utf-8') as f:
        #         text = f.read()
        # else:
        #     print(f"  ⚠️ 지원하지 않는 파일 형식: {unique_filename}")
        #     return False

        # if not text or not text.strip():
        #     print(f"  ⚠️ 내용이 없는 파일입니다: {original_filename}")
        #     return True
        print("🧠 HLM 구조 분석 + 청킹 수행 중...")
        text = process_file_to_hlm_chunks(file_path)
        total_chunks = len(text)
        print(f"  ✂️ {total_chunks}개의 HLM 청크로 분할 완료")

        if total_chunks == 0:
            print(f"  ⚠️ 내용이 없는 파일입니다: {original_filename}")
            return True

        # --- 여기가 핵심 수정 부분 ---
        # 텍스트를 청킹하기 전에 정규화를 수행합니다.
        print("  - 텍스트 정규화 수행 중...")
        # normalized_text = normalize_text(text)
        # print(f"  - 정규화 완료 (원본: {len(text)}자 -> 정규화: {len(normalized_text)}자)")
        # ---------------------------

        # 정규화된 텍스트를 기반으로 청킹
        chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
        total_chunks = len(chunks)
        print(f"  ✂️ {total_chunks}개의 청크로 분할 완료")

        # 각 청크를 임베딩하고 DB에 저장
        for i, chunk in enumerate(chunks):
            # 청크 자체는 이미 정규화되었으므로 바로 임베딩
            if not chunk: continue # 빈 청크는 건너뛰기
            
            chunk_index = i + 1
            print(f"  🔄 청크 {chunk_index}/{total_chunks} 처리 중...")
            
            embedding = model.encode(chunk).tolist()
            
            chunk_title = f"{original_filename}_chunk_{chunk_index}"
            insert_chunk_to_db(cursor, chunk["title"], chunk["content"], embedding, unique_filename, original_filename, chunk_index, total_chunks)

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