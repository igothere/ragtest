#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API Server Integration Tests with ModelManager

이 모듈은 ModelManager와 통합된 API 서버의 기능을 테스트합니다.
"""

import unittest
import json
import tempfile
import os
import sys
import threading
import time
from unittest.mock import patch, MagicMock, Mock
from io import BytesIO

# 테스트 환경 설정
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Flask 앱 임포트 전에 테스트 환경 설정
os.environ['MODEL_CACHE_DIR'] = tempfile.mkdtemp()
os.environ['MODEL_DEVICE'] = 'cpu'

from api_server import app
from model_manager import ModelManager, reset_model_manager


class TestAPIServerIntegration(unittest.TestCase):
    """API 서버와 ModelManager 통합 테스트"""
    
    @classmethod
    def setUpClass(cls):
        """테스트 클래스 설정"""
        # Flask 테스트 클라이언트 설정
        app.config['TESTING'] = True
        app.config['UPLOAD_FOLDER'] = tempfile.mkdtemp()
        cls.client = app.test_client()
        cls.app_context = app.app_context()
        cls.app_context.push()
    
    @classmethod
    def tearDownClass(cls):
        """테스트 클래스 정리"""
        cls.app_context.pop()
    
    def setUp(self):
        """각 테스트 전 설정"""
        # ModelManager 초기화
        reset_model_manager()
        
        # 테스트용 임시 디렉토리
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """각 테스트 후 정리"""
        # ModelManager 초기화
        reset_model_manager()
        
        # 임시 파일 정리
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @patch('api_server.psycopg2.connect')
    @patch('api_server.requests.post')
    def test_chat_endpoint_with_shared_model(self, mock_requests, mock_db):
        """공유 모델을 사용한 채팅 엔드포인트 테스트"""
        # Mock 설정
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_db.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        
        # 검색 결과 Mock
        mock_cursor.fetchall.return_value = [
            (1, "테스트 문서 내용", "test.pdf", 1, 0.95)
        ]
        
        # Ollama API 응답 Mock
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "message": {"content": "테스트 답변입니다."}
        }
        mock_response.raise_for_status.return_value = None
        mock_requests.return_value = mock_response
        
        # ModelManager Mock
        with patch('model_manager.SentenceTransformer') as mock_transformer:
            mock_model = MagicMock()
            # numpy array처럼 동작하도록 Mock 설정
            import numpy as np
            mock_embedding = np.array([0.1] * 768)
            mock_model.encode.return_value = mock_embedding
            mock_model.device = "cpu"
            mock_transformer.return_value = mock_model
            
            # 채팅 요청
            response = self.client.post('/chat', 
                json={'question': '테스트 질문입니다.'},
                content_type='application/json'
            )
            
            # 응답 검증
            self.assertEqual(response.status_code, 200)
            data = json.loads(response.data)
            self.assertIn('answer', data)
            self.assertIn('sources', data)
            self.assertEqual(data['answer'], '테스트 답변입니다.')
            
            # 모델이 호출되었는지 확인 (검증용 1회 + 실제 사용 1회)
            self.assertEqual(mock_model.encode.call_count, 2)
            # 마지막 호출이 실제 질문 인코딩인지 확인
            last_call = mock_model.encode.call_args_list[-1]
            self.assertEqual(last_call[0][0], '테스트 질문입니다.')

    def test_model_status_endpoint(self):
        """모델 상태 엔드포인트 테스트"""
        with patch('model_manager.SentenceTransformer') as mock_transformer:
            mock_model = MagicMock()
            mock_model.device = "cpu"
            # 모델 검증을 위한 encode 메소드 Mock
            import numpy as np
            mock_model.encode.return_value = np.array([[0.1] * 768])  # 2D array로 반환
            mock_transformer.return_value = mock_model
            
            # ModelManager 인스턴스 생성 및 모델 로드
            manager = ModelManager.get_instance()
            manager.get_model()
            
            # 상태 엔드포인트 호출
            response = self.client.get('/model/status')
            
            # 응답 검증
            self.assertEqual(response.status_code, 200)
            data = json.loads(response.data)
            self.assertEqual(data['status'], 'loaded')
            self.assertIn('model_name', data)
            self.assertIn('load_time', data)
            self.assertIn('memory_usage_mb', data)
            self.assertIn('device', data)
            self.assertIn('config', data)

    def test_model_status_endpoint_not_loaded(self):
        """모델이 로드되지 않은 상태의 상태 엔드포인트 테스트"""
        # 모델을 로드하지 않은 상태에서 상태 확인
        response = self.client.get('/model/status')
        
        # 응답 검증
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'not_loaded')
        self.assertIn('message', data)

    def test_model_reload_endpoint(self):
        """모델 재로드 엔드포인트 테스트"""
        with patch('model_manager.SentenceTransformer') as mock_transformer:
            mock_model = MagicMock()
            mock_model.device = "cpu"
            # 모델 검증을 위한 encode 메소드 Mock
            import numpy as np
            mock_model.encode.return_value = np.array([[0.1] * 768])  # 2D array로 반환
            mock_transformer.return_value = mock_model
            
            # 먼저 모델 로드
            manager = ModelManager.get_instance()
            manager.get_model()
            
            # 재로드 엔드포인트 호출
            response = self.client.post('/model/reload')
            
            # 응답 검증
            self.assertEqual(response.status_code, 200)
            data = json.loads(response.data)
            self.assertEqual(data['status'], 'success')
            self.assertIn('message', data)
            self.assertIn('model_info', data)

    def test_model_reload_endpoint_failure(self):
        """모델 재로드 실패 테스트"""
        with patch('model_manager.ModelManager.reload_model') as mock_reload:
            mock_reload.return_value = False
            
            # 재로드 엔드포인트 호출
            response = self.client.post('/model/reload')
            
            # 응답 검증
            self.assertEqual(response.status_code, 500)
            data = json.loads(response.data)
            self.assertEqual(data['status'], 'error')
            self.assertIn('message', data)

    @patch('api_server.psycopg2.connect')
    def test_chat_endpoint_model_access_error(self, mock_db):
        """모델 접근 에러 시 채팅 엔드포인트 테스트"""
        # ModelManager에서 에러 발생하도록 Mock
        with patch('api_server.get_model_with_fallback') as mock_get_model:
            from model_manager import ModelAccessError
            mock_get_model.side_effect = ModelAccessError("모델 접근 실패")
            
            # 채팅 요청
            response = self.client.post('/chat', 
                json={'question': '테스트 질문입니다.'},
                content_type='application/json'
            )
            
            # 에러 응답 검증
            self.assertEqual(response.status_code, 500)
            data = json.loads(response.data)
            self.assertIn('error', data)
            self.assertIn('모델에 접근할 수 없습니다', data['error'])

    @patch('api_server.psycopg2.connect')
    def test_chat_endpoint_missing_question(self, mock_db):
        """질문이 없는 채팅 요청 테스트"""
        response = self.client.post('/chat', 
            json={},
            content_type='application/json'
        )
        
        # 에러 응답 검증
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)
        self.assertEqual(data['error'], '질문이 없습니다.')

    @patch('api_server.psycopg2.connect')
    @patch('api_server.requests.post')
    def test_chat_endpoint_no_search_results(self, mock_requests, mock_db):
        """검색 결과가 없는 경우 테스트"""
        # Mock 설정
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_db.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        
        # 빈 검색 결과
        mock_cursor.fetchall.return_value = []
        
        # ModelManager Mock
        with patch('model_manager.SentenceTransformer') as mock_transformer:
            mock_model = MagicMock()
            # numpy array처럼 동작하도록 Mock 설정
            import numpy as np
            mock_embedding = np.array([0.1] * 768)
            mock_model.encode.return_value = mock_embedding
            mock_model.device = "cpu"
            mock_transformer.return_value = mock_model
            
            # 채팅 요청
            response = self.client.post('/chat', 
                json={'question': '테스트 질문입니다.'},
                content_type='application/json'
            )
            
            # 응답 검증
            self.assertEqual(response.status_code, 200)
            data = json.loads(response.data)
            self.assertEqual(data['answer'], '관련된 정보를 문서에서 찾을 수 없습니다.')
            self.assertEqual(data['sources'], [])

    def test_concurrent_chat_requests(self):
        """동시 채팅 요청 테스트"""
        with patch('api_server.psycopg2.connect') as mock_db, \
             patch('api_server.requests.post') as mock_requests:
            
            # Mock 설정
            mock_conn = MagicMock()
            mock_cursor = MagicMock()
            mock_db.return_value = mock_conn
            mock_conn.cursor.return_value = mock_cursor
            mock_cursor.fetchall.return_value = [
                (1, "테스트 문서", "test.pdf", 1, 0.95)
            ]
            
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "message": {"content": "테스트 답변"}
            }
            mock_response.raise_for_status.return_value = None
            mock_requests.return_value = mock_response
            
            # ModelManager Mock
            with patch('model_manager.SentenceTransformer') as mock_transformer:
                mock_model = MagicMock()
                # numpy array처럼 동작하도록 Mock 설정
                import numpy as np
                mock_embedding = np.array([0.1] * 768)
                mock_model.encode.return_value = mock_embedding
                mock_model.device = "cpu"
                mock_transformer.return_value = mock_model
                
                # 동시 요청 함수
                def make_request(question_id):
                    response = self.client.post('/chat', 
                        json={'question': f'테스트 질문 {question_id}'},
                        content_type='application/json'
                    )
                    return response.status_code == 200
                
                # 여러 스레드에서 동시 요청
                threads = []
                results = []
                
                for i in range(5):
                    thread = threading.Thread(
                        target=lambda i=i: results.append(make_request(i))
                    )
                    threads.append(thread)
                    thread.start()
                
                # 모든 스레드 완료 대기
                for thread in threads:
                    thread.join()
                
                # 모든 요청이 성공했는지 확인
                self.assertTrue(all(results))
                self.assertEqual(len(results), 5)

    def test_upload_endpoint_backward_compatibility(self):
        """업로드 엔드포인트 하위 호환성 테스트"""
        # 테스트 파일 생성
        test_content = b"Test document content"
        
        with patch('subprocess.run') as mock_subprocess:
            mock_subprocess.return_value = MagicMock(
                stdout="처리 완료",
                returncode=0
            )
            
            # 파일 업로드 요청
            response = self.client.post('/upload',
                data={
                    'file': (BytesIO(test_content), 'test.txt')
                },
                content_type='multipart/form-data'
            )
            
            # 응답 검증 (기존 기능 유지)
            self.assertEqual(response.status_code, 200)
            data = json.loads(response.data)
            self.assertIn('message', data)
            self.assertIn('처리 성공', data['message'])

    def test_model_manager_singleton_behavior(self):
        """ModelManager 싱글톤 동작 테스트"""
        with patch('model_manager.SentenceTransformer') as mock_transformer:
            mock_model = MagicMock()
            mock_model.device = "cpu"
            # 모델 검증을 위한 encode 메소드 Mock
            import numpy as np
            mock_model.encode.return_value = np.array([[0.1] * 768])  # 2D array로 반환
            mock_transformer.return_value = mock_model
            
            # 여러 번 인스턴스 요청
            manager1 = ModelManager.get_instance()
            manager2 = ModelManager.get_instance()
            
            # 같은 인스턴스인지 확인
            self.assertIs(manager1, manager2)
            
            # 모델도 같은 인스턴스인지 확인
            model1 = manager1.get_model()
            model2 = manager2.get_model()
            self.assertIs(model1, model2)
            
            # SentenceTransformer가 한 번만 호출되었는지 확인
            self.assertEqual(mock_transformer.call_count, 1)


class TestAPIServerPerformance(unittest.TestCase):
    """API 서버 성능 테스트"""
    
    def setUp(self):
        """테스트 설정"""
        reset_model_manager()
        app.config['TESTING'] = True
        self.client = app.test_client()
        self.app_context = app.app_context()
        self.app_context.push()
    
    def tearDown(self):
        """테스트 정리"""
        self.app_context.pop()
        reset_model_manager()

    def test_model_loading_time_improvement(self):
        """모델 로딩 시간 개선 테스트"""
        with patch('model_manager.SentenceTransformer') as mock_transformer:
            mock_model = MagicMock()
            mock_model.device = "cpu"
            # 모델 검증을 위한 encode 메소드 Mock
            import numpy as np
            mock_model.encode.return_value = np.array([[0.1] * 768])  # 2D array로 반환
            mock_transformer.return_value = mock_model
            
            # 첫 번째 모델 로딩 시간 측정
            start_time = time.time()
            manager = ModelManager.get_instance()
            model1 = manager.get_model()
            first_load_time = time.time() - start_time
            
            # 두 번째 모델 접근 시간 측정 (이미 로드됨)
            start_time = time.time()
            model2 = manager.get_model()
            second_access_time = time.time() - start_time
            
            # 두 번째 접근이 훨씬 빨라야 함
            self.assertLess(second_access_time, first_load_time / 10)
            
            # 같은 모델 인스턴스인지 확인
            self.assertIs(model1, model2)

    def test_memory_usage_optimization(self):
        """메모리 사용량 최적화 테스트"""
        with patch('model_manager.SentenceTransformer') as mock_transformer:
            mock_model = MagicMock()
            mock_model.device = "cpu"
            # 모델 검증을 위한 encode 메소드 Mock
            import numpy as np
            mock_model.encode.return_value = np.array([[0.1] * 768])  # 2D array로 반환
            mock_transformer.return_value = mock_model
            
            # 여러 번 모델 요청
            manager = ModelManager.get_instance()
            models = []
            for _ in range(5):
                models.append(manager.get_model())
            
            # 모든 모델이 같은 인스턴스인지 확인 (메모리 절약)
            for model in models[1:]:
                self.assertIs(models[0], model)
            
            # SentenceTransformer가 한 번만 생성되었는지 확인
            self.assertEqual(mock_transformer.call_count, 1)


if __name__ == '__main__':
    # 테스트 실행
    unittest.main(verbosity=2)