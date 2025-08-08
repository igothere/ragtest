#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
표 구조를 유지하는 RAG 시스템

기존 rag.py를 확장하여 PDF 표를 구조화된 형태로 처리
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import sys
import psycopg2
import traceback
import re
import requests
import json
import pandas as pd
import pdfplumber
from typing import List, Tuple, Dict, Any
from model_manager import ModelManager, get_model_with_fallback

# 설정
TARGET_DIR = "./docs"
MODEL_NAME = "nlpai-lab/KURE-v1"
DB_CONFIG = {
    "host": "localhost", "port": "5432", "dbname": "ragtest",
    "user": "eden", "password": "qwer123"
}
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# OLLAMA 설정
OLLAMA_ENDPOINT = os.getenv("OLLAMA_ENDPOINT", "https://api.hamonize.com/ollama/api/chat")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "airun-chat:latest")

def extract_pdf_with_tables(file_path: str) -> List[Dict[str, Any]]:
    """PDF에서 텍스트 + 표를 묶어 하나의 청크로 추출"""
    results = []

    try:
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                print(f"  📄 페이지 {page_num} 분석 중...")

                tables = page.extract_tables()
                markdown_tables = []
                searchable_table_texts = []

                if tables:
                    for table_idx, table in enumerate(tables):
                        if table and len(table) > 1:
                            try:
                                df = pd.DataFrame(table[1:], columns=table[0])
                                df = df.dropna(how='all').dropna(axis=1, how='all')
                                df = df.applymap(lambda x: str(x).replace('\x00', '') if pd.notna(x) else x)

                                if not df.empty:
                                    markdown = df.to_markdown(index=False).replace('\x00', '')
                                    markdown_tables.append(f"[표 {table_idx + 1}]\n{markdown}")
                                    searchable_table_texts.append(create_table_searchable_text(df))
                                    print(f"    ✅ 표 발견: {len(df)}행 × {len(df.columns)}열")
                            except Exception as e:
                                print(f"    ⚠️ 표 처리 오류: {e}")

                text = page.extract_text()
                clean_text = ''
                if text and text.strip():
                    clean_text = text.replace('\x00', '')
                    clean_text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', clean_text).strip()
                    print(f"    📝 텍스트: {len(clean_text)}자")

                # 텍스트 + 표 합치기
                full_content = ''
                embedding_text = ''

                if clean_text:
                    full_content += clean_text + "\n\n"
                    embedding_text += clean_text + "\n\n"
                if markdown_tables:
                    full_content += "\n\n".join(markdown_tables)
                    embedding_text += "\n\n".join(searchable_table_texts)

                if full_content.strip():
                    results.append({
                        'type': 'mixed',
                        'title': f'페이지 {page_num} 텍스트+표',
                        'content': full_content.strip(),
                        'embedding_text': embedding_text.strip(),
                        'metadata': {
                            'page': page_num,
                            'table_count': len(markdown_tables),
                            'has_tables': bool(markdown_tables)
                        }
                    })

    except Exception as e:
        print(f"  ❌ PDF 처리 중 오류: {e}")
        from pdfminer.high_level import extract_text
        text = extract_text(file_path)
        if text:
            results.append({
                'type': 'text',
                'title': '폴백 텍스트',
                'content': text,
                'embedding_text': text,
                'metadata': {'fallback': True}
            })

    return results

# def extract_pdf_with_tables(file_path: str) -> List[Dict[str, Any]]:
#     """PDF에서 표와 텍스트를 구분하여 추출"""
#     results = []
    
#     try:
#         with pdfplumber.open(file_path) as pdf:
#             for page_num, page in enumerate(pdf.pages, 1):
#                 print(f"  📄 페이지 {page_num} 분석 중...")
                
#                 # 표 추출
#                 tables = page.extract_tables()
                
#                 if tables:
#                     for table_idx, table in enumerate(tables):
#                         if table and len(table) > 1:
#                             try:
#                                 # 표를 DataFrame으로 변환
#                                 df = pd.DataFrame(table[1:], columns=table[0])
#                                 df = df.dropna(how='all').dropna(axis=1, how='all')
                                
#                                 # DataFrame의 모든 값에서 NULL 바이트 제거
#                                 df = df.applymap(lambda x: str(x).replace('\x00', '') if pd.notna(x) else x)
                                
#                                 if not df.empty:
#                                     # 마크다운 변환 시에도 NULL 바이트 제거
#                                     markdown_content = df.to_markdown(index=False).replace('\x00', '')
                                    
#                                     table_info = {
#                                         'type': 'table',
#                                         'page': page_num,
#                                         'table_index': table_idx + 1,
#                                         'content': markdown_content,
#                                         'searchable_text': create_table_searchable_text(df),
#                                         'metadata': {
#                                             'rows': len(df),
#                                             'columns': len(df.columns),
#                                             'column_names': [str(col).replace('\x00', '') for col in df.columns.tolist()]
#                                         }
#                                     }
#                                     results.append(table_info)
#                                     print(f"    ✅ 표 발견: {len(df)}행 × {len(df.columns)}열")
#                             except Exception as e:
#                                 print(f"    ⚠️ 표 처리 오류: {e}")
                
#                 # 텍스트 추출 (표 영역 제외는 복잡하므로 전체 텍스트 사용)
#                 text = page.extract_text()
#                 if text and text.strip():
#                     # NULL 바이트 및 제어 문자 제거
#                     clean_text = text.replace('\x00', '')
#                     clean_text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', clean_text)
#                     clean_text = clean_text.strip()
                    
#                     if clean_text:
#                         # 표가 있는 페이지의 텍스트는 표 정보와 함께 저장
#                         has_tables = len([t for t in tables if t]) > 0
#                         text_info = {
#                             'type': 'text',
#                             'page': page_num,
#                             'content': clean_text,
#                             'searchable_text': clean_text,
#                             'metadata': {
#                                 'has_tables': has_tables,
#                                 'table_count': len([t for t in tables if t])
#                             }
#                         }
#                         results.append(text_info)
#                         print(f"    📝 텍스트: {len(clean_text)}자")
    
#     except Exception as e:
#         print(f"  ❌ PDF 처리 중 오류: {e}")
#         # 실패 시 기존 방식으로 폴백
#         from pdfminer.high_level import extract_text
#         text = extract_text(file_path)
#         if text:
#             results.append({
#                 'type': 'text',
#                 'page': 1,
#                 'content': text,
#                 'searchable_text': text,
#                 'metadata': {'fallback': True}
#             })
    
#     return results

def create_table_searchable_text(df: pd.DataFrame) -> str:
    """표를 검색 가능한 텍스트로 변환"""
    # 컬럼명 + 모든 셀 값을 텍스트로 결합
    column_text = "컬럼: " + ", ".join(str(col).replace('\x00', '') for col in df.columns.tolist())
    
    # 각 행의 데이터를 자연어 형태로 변환
    row_texts = []
    for _, row in df.iterrows():
        row_items = []
        for col, val in row.items():
            if pd.notna(val) and str(val).strip():
                # NULL 바이트 제거
                clean_col = str(col).replace('\x00', '')
                clean_val = str(val).replace('\x00', '')
                row_items.append(f"{clean_col}: {clean_val}")
        if row_items:
            row_texts.append(" | ".join(row_items))
    
    result = f"[표 데이터] {column_text}\n" + "\n".join(row_texts)
    # 최종적으로 NULL 바이트 제거
    return result.replace('\x00', '')

def create_structured_chunks(extracted_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """추출된 데이터를 청크로 변환"""
    chunks = []
    
    for item in extracted_data:
        if item['type'] == 'table':
            # 표는 하나의 청크로 (크기에 따라 분할 가능)
            chunk = {
                'type': 'table',
                'title': f"페이지 {item['page']} 표 {item['table_index']}",
                'content': f"[표 {item['table_index']}]\n{item['content']}",
                'embedding_text': item['searchable_text'],
                'metadata': item['metadata']
            }
            chunks.append(chunk)
            
        elif item['type'] == 'text':
            # 텍스트는 기존 방식으로 청킹
            text_chunks = chunk_text(item['content'], CHUNK_SIZE, CHUNK_OVERLAP)
            for i, text_chunk in enumerate(text_chunks):
                chunk = {
                    'type': 'text',
                    'title': f"페이지 {item['page']} 텍스트 {i+1}",
                    'content': text_chunk,
                    'embedding_text': text_chunk,
                    'metadata': {'page': item['page'], 'chunk_index': i+1}
                }
                chunks.append(chunk)
    
    return chunks

from transformers import AutoTokenizer

# 모델 토크나이저 로딩
tokenizer = AutoTokenizer.from_pretrained("nlpai-lab/KURE-v1")

def chunk_text(text, chunk_size=480, overlap=40):
    """KURE-v1 기준 토큰 단위 청크 분할"""
    if not text:
        return []

    input_ids = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    start = 0
    end = chunk_size

    while start < len(input_ids):
        chunk_ids = input_ids[start:end]
        chunk_text = tokenizer.decode(chunk_ids)
        chunks.append(chunk_text)

        # 다음 청크로 이동 (overlap 고려)
        start = end - overlap
        end = start + chunk_size

    return chunks


def normalize_text(text: str) -> str:
    """텍스트 정규화 함수 (NULL 바이트 제거 포함)"""
    if not text:
        return ""
    
    # NULL 바이트 제거 (PostgreSQL 오류 방지)
    text = text.replace('\x00', '')
    
    # 기타 제어 문자 제거
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    
    text = text.lower()
    text = re.sub(r'[^가-힣a-z0-9\s\./_-]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def insert_chunk_to_db(cursor, title, content, embedding, source_file, original_filename, chunk_index, total_chunks, chunk_type='text', metadata=None):
    """DB 삽입 함수 (표 메타데이터 포함)"""
    # 모든 문자열 필드에서 NULL 바이트 제거
    def clean_string(s):
        if s is None:
            return None
        return str(s).replace('\x00', '').replace('\r', '').replace('\n\n\n', '\n\n')
    
    title = clean_string(title)
    content = clean_string(content)
    source_file = clean_string(source_file)
    original_filename = clean_string(original_filename)
    chunk_type = clean_string(chunk_type)
    
    embedding_str = "[" + ",".join(map(str, embedding)) + "]"
    metadata_json = json.dumps(metadata) if metadata else None
    if metadata_json:
        metadata_json = clean_string(metadata_json)
    
    sql = """
        INSERT INTO documents (title, content, embedding, source_file, original_filename, chunk_index, total_chunks, chunk_type, metadata)
        VALUES (%s, %s, %s::vector, %s, %s, %s, %s, %s, %s)
    """
    try:
        print(f"    💾 DB 삽입 시도: {chunk_type} 청크 {chunk_index}/{total_chunks}")
        cursor.execute(sql, (title, content, embedding_str, source_file, original_filename, chunk_index, total_chunks, chunk_type, metadata_json))
        print(f"    ✅ DB 삽입 성공")
    except Exception as e:
        print(f"    ❌ DB 삽입 오류 발생: {e}")
        print(f"    디버그 정보: title 길이={len(title) if title else 0}, content 길이={len(content) if content else 0}")
        raise

def process_file_with_tables(file_path, unique_filename, original_filename, model, cursor):
    """표 구조를 유지하면서 파일 처리"""
    print(f"\n📄 '{original_filename}' 파일 처리 시작 (표 구조 유지)...")
    
    try:
        # PDF에서 표와 텍스트 추출
        print("  🔍 PDF에서 표와 텍스트 추출 중...")
        extracted_data = extract_pdf_with_tables(file_path)
        
        if not extracted_data:
            print(f"  ⚠️ 추출된 내용이 없습니다: {original_filename}")
            return True
        
        # 구조화된 청크 생성
        # print("  🧩 구조화된 청크 생성 중...")
        # chunks = create_structured_chunks(extracted_data)
        # total_chunks = len(chunks)
        
        # print(f"  ✂️ {total_chunks}개의 청크로 분할 완료")

         # 구조화된 청크 생성
        print("  🧩 텍스트+표 통합 청크 생성 중...")
        chunks = extracted_data
        total_chunks = len(chunks)
        print(f"  ✂️ {total_chunks}개의 청크로 분할 완료")
        
        # 각 청크를 임베딩하고 DB에 저장
        for i, chunk in enumerate(chunks):
            if not chunk['embedding_text'].strip():
                continue
            
            chunk_index = i + 1
            print(f"  🔄 청크 {chunk_index}/{total_chunks} 처리 중... ({chunk['type']})")
            
            # 임베딩용 텍스트 정규화
            normalized_text = normalize_text(chunk['embedding_text'])
            embedding = model.encode(normalized_text).tolist()
            
            insert_chunk_to_db(
                cursor, 
                chunk['title'], 
                chunk['content'], 
                embedding, 
                unique_filename, 
                original_filename, 
                chunk_index, 
                total_chunks,
                chunk['type'],
                chunk['metadata']
            )
        
        print(f"  ✅ 파일 처리 성공: {original_filename}")
        return True
        
    except Exception as e:
        print(f"  ❌ 파일 처리 중 오류 발생: {original_filename} - {e}")
        traceback.print_exc()
        return False

def main():
    print("==============================================")
    print("🚀 RAG 임베딩 스크립트 실행 (표 구조 유지)")
    print("==============================================")
    
    if len(sys.argv) < 3:
        print("❌ 사용법: python rag_with_tables.py <고유 파일명> <원본 파일명>")
        sys.exit(1)
    
    unique_filename = sys.argv[1]
    original_filename = sys.argv[2]
    print(f"🎯 처리 대상 파일: {original_filename} (고유명: {unique_filename})")

    model = None
    conn = None
    try:
        print("🤖 공유 임베딩 모델 로딩...")
        model = get_model_with_fallback()
        print("✅ 모델 로딩 완료")

        print("🐘 데이터베이스 연결 시도...")
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        print("✅ 데이터베이스 연결 성공")

        file_path = os.path.join(TARGET_DIR, unique_filename)
        print(f"🔍 파일 경로 확인: {file_path}")
        if not os.path.isfile(file_path):
            print(f"❌ 파일을 찾을 수 없습니다!")
            sys.exit(1)

        if process_file_with_tables(file_path, unique_filename, original_filename, model, cur):
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