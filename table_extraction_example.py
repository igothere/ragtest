#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDF 표 추출 및 구조화된 청킹 예제

필요한 패키지:
pip install tabula-py camelot-py[cv] pdfplumber

각 라이브러리의 특징:
- tabula-py: Java 기반, 정확도 높음
- camelot: Python 네이티브, 설정 옵션 많음  
- pdfplumber: 가벼움, 텍스트와 표 동시 처리 가능
"""

import pandas as pd
import pdfplumber
import json
from typing import List, Dict, Any
import re

def extract_tables_with_pdfplumber(pdf_path: str) -> List[Dict[str, Any]]:
    """pdfplumber를 사용하여 PDF에서 표와 텍스트를 추출"""
    results = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            print(f"📄 페이지 {page_num} 처리 중...")
            
            # 표 추출
            tables = page.extract_tables()
            
            if tables:
                for table_idx, table in enumerate(tables):
                    if table and len(table) > 1:  # 헤더 + 최소 1행 이상
                        # 표를 DataFrame으로 변환
                        df = pd.DataFrame(table[1:], columns=table[0])
                        
                        # 빈 행/열 제거
                        df = df.dropna(how='all').dropna(axis=1, how='all')
                        
                        if not df.empty:
                            table_info = {
                                'type': 'table',
                                'page': page_num,
                                'table_index': table_idx + 1,
                                'data': df.to_dict('records'),
                                'columns': df.columns.tolist(),
                                'markdown': df.to_markdown(index=False),
                                'csv': df.to_csv(index=False),
                                'summary': f"페이지 {page_num}의 표 {table_idx + 1}: {len(df)}행 × {len(df.columns)}열"
                            }
                            results.append(table_info)
                            print(f"  ✅ 표 발견: {len(df)}행 × {len(df.columns)}열")
            
            # 표가 아닌 텍스트 추출
            text = page.extract_text()
            if text and text.strip():
                # 표 영역을 제외한 텍스트 (근사치)
                text_info = {
                    'type': 'text',
                    'page': page_num,
                    'content': text.strip(),
                    'summary': f"페이지 {page_num}의 텍스트: {len(text)}자"
                }
                results.append(text_info)
                print(f"  📝 텍스트: {len(text)}자")
    
    return results

def create_structured_chunks(extracted_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """추출된 데이터를 구조화된 청크로 변환"""
    chunks = []
    
    for item in extracted_data:
        if item['type'] == 'table':
            # 표는 여러 방식으로 청킹 가능
            
            # 방법 1: 전체 표를 하나의 청크로
            table_chunk = {
                'type': 'table',
                'content': item['markdown'],  # 마크다운 형식
                'metadata': {
                    'page': item['page'],
                    'table_index': item['table_index'],
                    'rows': len(item['data']),
                    'columns': len(item['columns']),
                    'column_names': item['columns']
                },
                'searchable_text': f"{item['summary']} 컬럼: {', '.join(item['columns'])} " + 
                                 ' '.join([str(cell) for row in item['data'] for cell in row.values() if cell]),
                'structured_data': item['data']
            }
            chunks.append(table_chunk)
            
            # 방법 2: 표가 크면 행 단위로 분할
            if len(item['data']) > 10:  # 10행 이상이면 분할
                for i in range(0, len(item['data']), 5):  # 5행씩 분할
                    sub_data = item['data'][i:i+5]
                    sub_df = pd.DataFrame(sub_data)
                    
                    sub_chunk = {
                        'type': 'table_section',
                        'content': sub_df.to_markdown(index=False),
                        'metadata': {
                            'page': item['page'],
                            'table_index': item['table_index'],
                            'section': f"{i+1}-{min(i+5, len(item['data']))}행",
                            'columns': item['columns']
                        },
                        'searchable_text': f"표 섹션 ({i+1}-{min(i+5, len(item['data']))}행) " +
                                         ' '.join([str(cell) for row in sub_data for cell in row.values() if cell]),
                        'structured_data': sub_data
                    }
                    chunks.append(sub_chunk)
        
        elif item['type'] == 'text':
            # 텍스트는 기존 방식으로 청킹
            text_chunk = {
                'type': 'text',
                'content': item['content'],
                'metadata': {
                    'page': item['page']
                },
                'searchable_text': item['content']
            }
            chunks.append(text_chunk)
    
    return chunks

def format_chunk_for_embedding(chunk: Dict[str, Any]) -> str:
    """청크를 임베딩용 텍스트로 변환"""
    if chunk['type'] in ['table', 'table_section']:
        # 표 데이터는 검색 가능한 텍스트 + 구조 정보 조합
        return f"[표 데이터] {chunk['searchable_text']}\n\n{chunk['content']}"
    else:
        return chunk['searchable_text']

def main():
    pdf_path = "./docs/73eb14fc-da45-4b1e-a29f-c42d10500155.pdf"
    
    print("🔍 PDF에서 표와 텍스트 추출 중...")
    extracted_data = extract_tables_with_pdfplumber(pdf_path)
    
    print(f"\n📊 추출 결과:")
    tables_count = len([item for item in extracted_data if item['type'] == 'table'])
    text_count = len([item for item in extracted_data if item['type'] == 'text'])
    print(f"  - 표: {tables_count}개")
    print(f"  - 텍스트 섹션: {text_count}개")
    
    print("\n🧩 구조화된 청크 생성 중...")
    chunks = create_structured_chunks(extracted_data)
    
    print(f"\n✅ 최종 청크: {len(chunks)}개")
    for i, chunk in enumerate(chunks, 1):
        print(f"  {i}. {chunk['type']} - {len(chunk['content'])}자")
        if chunk['type'] in ['table', 'table_section']:
            print(f"     메타데이터: {chunk['metadata']}")
    
    # 임베딩용 텍스트 생성 예시
    print("\n📝 임베딩용 텍스트 예시:")
    for i, chunk in enumerate(chunks[:3], 1):  # 처음 3개만 출력
        embedding_text = format_chunk_for_embedding(chunk)
        print(f"\n--- 청크 {i} ---")
        print(embedding_text[:200] + "..." if len(embedding_text) > 200 else embedding_text)

if __name__ == "__main__":
    main()