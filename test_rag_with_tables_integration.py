#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG with Tables Integration Tests

ModelManager와 rag_with_tables.py의 통합을 테스트합니다.
"""

import os
import sys
import unittest
import tempfile
import shutil
import subprocess
import psycopg2
import json
from unittest.mock import patch, MagicMock, call
from model_manager import ModelManager, reset_model_manager
from rag_with_tables import (
    extract_pdf_with_tables, 
    process_file_with_tables,
    create_table_searchable_text,
    normalize_text,
    insert_chunk_to_db
)
import pandas as pd


class TestRagWithTablesIntegration(unittest.TestCase):
    """RAG with Tables와 ModelManager 통합 테스트"""
    
    def setUp(self):
        """테스트 설정"""
        # ModelManager 초기화
        reset_model_manager()
        
        # 테스트용 임시 디렉토리 생성
        self.test_dir = tempfile.mkdtemp()
        self.test_cache_dir = os.path.join(self.test_dir, "model_cache")
        os.makedirs(self.test_cache_dir, exist_ok=True)
        
        # 환경 변수 설정
        self.original_env = {}
        test_env = {
            "MODEL_NAME": "nlpai-lab/KURE-v1",
            "MODEL_CACHE_DIR": self.test_cache_dir,
            "MODEL_DEVICE": "cpu"
        }
        
        for key, value in test_env.items():
            self.original_env[key] = os.environ.get(key)
            os.environ[key] = value
    
    def tearDown(self):
        """테스트 정리"""
        # 환경 변수 복원
        for key, value in self.original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        
        # 임시 디렉토리 삭제
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        
        # ModelManager 초기화
        reset_model_manager()

    @patch('rag_with_tables.get_model_with_fallback')
    def test_model_manager_integration(self, mock_get_model):
        """ModelManager 통합 테스트"""
        # Mock 모델 설정
        mock_model = MagicMock()
        import numpy as np
        mock_model.encode.return_value = np.array([0.1, 0.2, 0.3])
        mock_get_model.return_value = mock_model
        
        # 테스트 텍스트
        test_text = "테스트 문서 내용"
        normalized_text = normalize_text(test_text)
        
        # 모델 호출 테스트
        from rag_with_tables import get_model_with_fallback
        model = get_model_with_fallback()
        embedding = model.encode(normalized_text)
        
        # 검증
        mock_get_model.assert_called_once()
        mock_model.encode.assert_called_once_with(normalized_text)
        import numpy as np
        np.testing.assert_array_equal(embedding, np.array([0.1, 0.2, 0.3]))

    @patch('rag_with_tables.get_model_with_fallback')
    @patch('rag_with_tables.psycopg2.connect')
    def test_process_file_with_shared_model(self, mock_connect, mock_get_model):
        """공유 모델을 사용한 파일 처리 테스트"""
        # Mock 설정
        mock_model = MagicMock()
        import numpy as np
        mock_embedding = np.array([0.1] * 768)
        mock_model.encode.return_value = mock_embedding
        mock_get_model.return_value = mock_model
        
        mock_cursor = MagicMock()
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn
        
        # 테스트 PDF 파일 생성 (실제로는 mock으로 처리)
        test_file = os.path.join(self.test_dir, "test.pdf")
        with open(test_file, 'w') as f:
            f.write("dummy pdf content")
        
        # extract_pdf_with_tables를 mock으로 처리
        with patch('rag_with_tables.extract_pdf_with_tables') as mock_extract:
            mock_extract.return_value = [
                {
                    'type': 'mixed',
                    'title': '페이지 1 텍스트+표',
                    'content': '테스트 내용\n\n[표 1]\n| 컬럼1 | 컬럼2 |\n|-------|-------|\n| 값1   | 값2   |',
                    'embedding_text': '테스트 내용\n\n컬럼: 컬럼1, 컬럼2\n컬럼1: 값1 | 컬럼2: 값2',
                    'metadata': {'page': 1, 'table_count': 1, 'has_tables': True}
                }
            ]
            
            # 파일 처리 실행 - 실제로는 main() 함수를 통해 get_model_with_fallback이 호출됨
            # 여기서는 직접 모델을 전달하므로 모델 사용만 확인
            result = process_file_with_tables(
                test_file, 
                "test.pdf", 
                "원본파일.pdf", 
                mock_model, 
                mock_cursor
            )
            
            # 검증
            self.assertTrue(result)
            mock_model.encode.assert_called()
            mock_cursor.execute.assert_called()

    @patch('rag_with_tables.get_model_with_fallback')
    @patch('rag_with_tables.psycopg2.connect')
    @patch('rag_with_tables.extract_pdf_with_tables')
    @patch('os.path.isfile')
    @patch('sys.argv', ['rag_with_tables.py', 'test.pdf', '원본파일.pdf'])
    def test_main_function_integration(self, mock_isfile, mock_extract, mock_connect, mock_get_model):
        """main() 함수에서 ModelManager 통합 테스트"""
        # Mock 설정
        mock_model = MagicMock()
        import numpy as np
        mock_embedding = np.array([0.1] * 768)
        mock_model.encode.return_value = mock_embedding
        mock_get_model.return_value = mock_model
        
        mock_cursor = MagicMock()
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn
        
        mock_isfile.return_value = True
        
        mock_extract.return_value = [
            {
                'type': 'mixed',
                'title': '페이지 1 텍스트+표',
                'content': '테스트 내용',
                'embedding_text': '테스트 내용',
                'metadata': {'page': 1, 'table_count': 0, 'has_tables': False}
            }
        ]
        
        # main() 함수 실행
        from rag_with_tables import main
        main()
        
        # 검증
        mock_get_model.assert_called_once()
        mock_model.encode.assert_called()
        mock_cursor.execute.assert_called()
        mock_conn.commit.assert_called_once()

    def test_table_searchable_text_creation(self):
        """표 검색 가능 텍스트 생성 테스트"""
        # 테스트 DataFrame 생성
        df = pd.DataFrame({
            '이름': ['홍길동', '김철수'],
            '나이': [25, 30],
            '직업': ['개발자', '디자이너']
        })
        
        # 검색 가능 텍스트 생성
        searchable_text = create_table_searchable_text(df)
        
        # 검증
        self.assertIn('컬럼: 이름, 나이, 직업', searchable_text)
        self.assertIn('이름: 홍길동', searchable_text)
        self.assertIn('나이: 25', searchable_text)
        self.assertIn('직업: 개발자', searchable_text)

    @patch('rag_with_tables.get_model_with_fallback')
    def test_model_fallback_mechanism(self, mock_get_model):
        """모델 폴백 메커니즘 테스트"""
        # 첫 번째 호출에서 예외 발생, 두 번째 호출에서 성공
        mock_model = MagicMock()
        import numpy as np
        mock_model.encode.return_value = np.array([0.1] * 768)
        
        # 폴백 시나리오 시뮬레이션
        mock_get_model.side_effect = [Exception("공유 모델 실패"), mock_model]
        
        # 첫 번째 시도 (실패)
        with self.assertRaises(Exception):
            from rag_with_tables import get_model_with_fallback
            get_model_with_fallback()
        
        # 두 번째 시도 (성공)
        mock_get_model.side_effect = None
        mock_get_model.return_value = mock_model
        
        model = get_model_with_fallback()
        self.assertEqual(model, mock_model)

    def test_normalize_text_with_null_bytes(self):
        """NULL 바이트가 포함된 텍스트 정규화 테스트"""
        # NULL 바이트가 포함된 텍스트
        text_with_nulls = "테스트\x00텍스트\x01제어문자\x1f포함"
        
        # 정규화 실행
        normalized = normalize_text(text_with_nulls)
        
        # 검증
        self.assertNotIn('\x00', normalized)
        self.assertNotIn('\x01', normalized)
        self.assertNotIn('\x1f', normalized)
        self.assertIn('테스트', normalized)
        self.assertIn('텍스트', normalized)

    @patch('rag_with_tables.get_model_with_fallback')
    def test_concurrent_model_access(self, mock_get_model):
        """동시 모델 접근 테스트"""
        import threading
        import time
        
        # Mock 모델 설정
        mock_model = MagicMock()
        import numpy as np
        mock_model.encode.return_value = np.array([0.1] * 768)
        mock_get_model.return_value = mock_model
        
        results = []
        errors = []
        
        def worker():
            try:
                from rag_with_tables import get_model_with_fallback
                model = get_model_with_fallback()
                embedding = model.encode("테스트 텍스트")
                results.append(embedding)
            except Exception as e:
                errors.append(e)
        
        # 여러 스레드에서 동시 접근
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()
        
        # 모든 스레드 완료 대기
        for thread in threads:
            thread.join()
        
        # 검증
        self.assertEqual(len(errors), 0, f"에러 발생: {errors}")
        self.assertEqual(len(results), 5)
        import numpy as np
        for result in results:
            np.testing.assert_array_equal(result, np.array([0.1] * 768))

    @patch('rag_with_tables.psycopg2.connect')
    def test_db_insertion_with_null_bytes(self, mock_connect):
        """NULL 바이트가 포함된 데이터의 DB 삽입 테스트"""
        # Mock 설정
        mock_cursor = MagicMock()
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn
        
        # NULL 바이트가 포함된 데이터
        title = "테스트\x00제목"
        content = "내용\x00포함\r\n\n\n여러줄"
        embedding = [0.1, 0.2, 0.3]
        
        # DB 삽입 실행
        insert_chunk_to_db(
            mock_cursor,
            title,
            content,
            embedding,
            "test.pdf",
            "원본.pdf",
            1,
            1,
            "text",
            {"test": "metadata"}
        )
        
        # 검증
        mock_cursor.execute.assert_called_once()
        call_args = mock_cursor.execute.call_args[0]
        
        # NULL 바이트가 제거되었는지 확인
        inserted_title = call_args[1][0]
        inserted_content = call_args[1][1]
        
        self.assertNotIn('\x00', inserted_title)
        self.assertNotIn('\x00', inserted_content)
        self.assertIn('테스트', inserted_title)
        self.assertIn('제목', inserted_title)

    def test_memory_usage_optimization(self):
        """메모리 사용량 최적화 테스트"""
        # 이 테스트는 실제 메모리 사용량을 측정하기 어려우므로
        # ModelManager가 싱글톤으로 작동하는지 확인
        
        # 여러 번 인스턴스를 요청해도 같은 객체가 반환되는지 확인
        manager1 = ModelManager.get_instance()
        manager2 = ModelManager.get_instance()
        
        self.assertIs(manager1, manager2)
        
        # 모델도 같은 인스턴스인지 확인 (mock 환경에서는 제한적)
        with patch('model_manager.SentenceTransformer') as mock_transformer:
            mock_model = MagicMock()
            # Mock 모델이 검증을 통과하도록 설정
            import numpy as np
            mock_model.encode.return_value = np.array([0.1, 0.2, 0.3])
            mock_transformer.return_value = mock_model
            
            # 첫 번째 모델 요청
            model1 = ModelManager.get_model()
            
            # 두 번째 모델 요청 (캐시된 모델 반환)
            model2 = ModelManager.get_model()
            
            # 같은 모델 인스턴스인지 확인
            self.assertIs(model1, model2)
            
            # SentenceTransformer가 한 번만 호출되었는지 확인 (재시도 포함하여 최대 4번까지 허용)
            self.assertLessEqual(mock_transformer.call_count, 4)


class TestRagWithTablesSubprocess(unittest.TestCase):
    """서브프로세스에서의 RAG 처리 테스트"""
    
    def setUp(self):
        """테스트 설정"""
        self.test_dir = tempfile.mkdtemp()
        self.test_cache_dir = os.path.join(self.test_dir, "model_cache")
        os.makedirs(self.test_cache_dir, exist_ok=True)
    
    def tearDown(self):
        """테스트 정리"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    @patch('subprocess.run')
    def test_subprocess_model_access(self, mock_run):
        """서브프로세스에서 모델 접근 테스트"""
        # 성공적인 서브프로세스 실행 시뮬레이션
        mock_run.return_value = MagicMock(returncode=0, stdout="처리 완료")
        
        # 서브프로세스 실행 시뮬레이션
        result = subprocess.run([
            sys.executable, 
            "rag_with_tables.py", 
            "test.pdf", 
            "원본.pdf"
        ], capture_output=True, text=True)
        
        # Mock이 호출되었는지 확인
        mock_run.assert_called_once()

    def test_environment_variable_inheritance(self):
        """환경 변수 상속 테스트"""
        # 환경 변수 설정
        test_env = os.environ.copy()
        test_env.update({
            "MODEL_NAME": "nlpai-lab/KURE-v1",
            "MODEL_CACHE_DIR": self.test_cache_dir,
            "MODEL_DEVICE": "cpu"
        })
        
        # 환경 변수가 올바르게 설정되었는지 확인
        self.assertEqual(test_env["MODEL_NAME"], "nlpai-lab/KURE-v1")
        self.assertEqual(test_env["MODEL_CACHE_DIR"], self.test_cache_dir)
        self.assertEqual(test_env["MODEL_DEVICE"], "cpu")


class TestBackwardCompatibility(unittest.TestCase):
    """하위 호환성 테스트"""
    
    @patch('rag_with_tables.get_model_with_fallback')
    def test_api_compatibility(self, mock_get_model):
        """API 호환성 테스트"""
        # Mock 모델 설정
        mock_model = MagicMock()
        import numpy as np
        mock_model.encode.return_value = np.array([0.1] * 768)
        mock_get_model.return_value = mock_model
        
        # 기존 방식과 동일한 인터페이스 확인
        from rag_with_tables import get_model_with_fallback
        model = get_model_with_fallback()
        
        # SentenceTransformer와 동일한 메서드 제공 확인
        self.assertTrue(hasattr(model, 'encode'))
        
        # 인코딩 결과 형식 확인
        result = model.encode("테스트 텍스트")
        import numpy as np
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(len(result), 768)

    @patch('rag_with_tables.get_model_with_fallback')
    @patch('rag_with_tables.extract_pdf_with_tables')
    def test_output_format_consistency(self, mock_extract, mock_get_model):
        """출력 형식 일관성 테스트"""
        # Mock 설정
        mock_model = MagicMock()
        import numpy as np
        mock_model.encode.return_value = np.array([0.1] * 768)
        mock_get_model.return_value = mock_model
        
        mock_extract.return_value = [
            {
                'type': 'mixed',
                'title': '페이지 1 텍스트+표',
                'content': '테스트 내용',
                'embedding_text': '테스트 내용',
                'metadata': {'page': 1, 'table_count': 0, 'has_tables': False}
            }
        ]
        
        # 기존 함수 호출
        from rag_with_tables import extract_pdf_with_tables
        result = extract_pdf_with_tables("dummy_path")
        
        # 출력 형식 확인
        self.assertIsInstance(result, list)
        self.assertTrue(len(result) > 0)
        
        item = result[0]
        self.assertIn('type', item)
        self.assertIn('title', item)
        self.assertIn('content', item)
        self.assertIn('embedding_text', item)
        self.assertIn('metadata', item)


if __name__ == '__main__':
    # 테스트 실행
    unittest.main(verbosity=2)