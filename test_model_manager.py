#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ModelManager 단위 테스트

ModelManager 클래스의 싱글톤 패턴, 스레드 안전성, 모델 로딩 등을 테스트합니다.
"""

import unittest
import threading
import time
import os
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock, call
from model_manager import (
    ModelManager, ModelConfig, ModelStatus, 
    get_model, get_model_with_fallback, get_model_status, is_model_loaded, reset_model_manager,
    ModelLoadingError, ModelConfigurationError, ModelAccessError
)


class TestModelConfig(unittest.TestCase):
    """ModelConfig 클래스 테스트"""
    
    def test_default_config(self):
        """기본 설정 테스트"""
        config = ModelConfig()
        self.assertEqual(config.name, "nlpai-lab/KURE-v1")
        self.assertEqual(config.cache_dir, "./model_cache")
        self.assertEqual(config.device, "auto")
        self.assertFalse(config.trust_remote_code)
    
    def test_custom_config(self):
        """커스텀 설정 테스트"""
        config = ModelConfig(
            name="test-model",
            cache_dir="/tmp/test",
            device="cpu",
            trust_remote_code=True
        )
        self.assertEqual(config.name, "test-model")
        self.assertEqual(config.cache_dir, "/tmp/test")
        self.assertEqual(config.device, "cpu")
        self.assertTrue(config.trust_remote_code)
    
    def test_from_env(self):
        """환경 변수에서 설정 로드 테스트"""
        with patch.dict(os.environ, {
            'MODEL_NAME': 'env-model',
            'MODEL_CACHE_DIR': '/env/cache',
            'MODEL_DEVICE': 'cuda',
            'MODEL_TRUST_REMOTE_CODE': 'true'
        }):
            config = ModelConfig.from_env()
            self.assertEqual(config.name, 'env-model')
            self.assertEqual(config.cache_dir, '/env/cache')
            self.assertEqual(config.device, 'cuda')
            self.assertTrue(config.trust_remote_code)
    
    def test_from_env_defaults(self):
        """환경 변수가 없을 때 기본값 사용 테스트"""
        with patch.dict(os.environ, {}, clear=True):
            config = ModelConfig.from_env()
            self.assertEqual(config.name, "nlpai-lab/KURE-v1")
            self.assertEqual(config.cache_dir, "./model_cache")
            self.assertEqual(config.device, "auto")
            self.assertFalse(config.trust_remote_code)


class TestModelStatus(unittest.TestCase):
    """ModelStatus 클래스 테스트"""
    
    def test_model_status_creation(self):
        """ModelStatus 생성 테스트"""
        status = ModelStatus(
            is_loaded=True,
            model_name="test-model",
            load_time=1.5,
            memory_usage=1024,
            device="cpu"
        )
        self.assertTrue(status.is_loaded)
        self.assertEqual(status.model_name, "test-model")
        self.assertEqual(status.load_time, 1.5)
        self.assertEqual(status.memory_usage, 1024)
        self.assertEqual(status.device, "cpu")
        self.assertIsNone(status.error_message)
    
    def test_model_status_with_error(self):
        """에러가 있는 ModelStatus 테스트"""
        status = ModelStatus(
            is_loaded=False,
            model_name="test-model",
            load_time=0.0,
            memory_usage=0,
            device="unknown",
            error_message="로딩 실패"
        )
        self.assertFalse(status.is_loaded)
        self.assertEqual(status.error_message, "로딩 실패")


class TestModelManager(unittest.TestCase):
    """ModelManager 클래스 테스트"""
    
    def setUp(self):
        """테스트 전 설정"""
        # 싱글톤 인스턴스 초기화
        reset_model_manager()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """테스트 후 정리"""
        # 싱글톤 인스턴스 초기화
        reset_model_manager()
        # 임시 디렉토리 정리
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_singleton_pattern(self):
        """싱글톤 패턴 테스트"""
        instance1 = ModelManager.get_instance()
        instance2 = ModelManager.get_instance()
        self.assertIs(instance1, instance2)
    
    def test_direct_instantiation_prevention(self):
        """직접 인스턴스화 방지 테스트"""
        ModelManager.get_instance()  # 먼저 인스턴스 생성
        with self.assertRaises(RuntimeError):
            ModelManager()
    
    def test_thread_safety_singleton(self):
        """싱글톤 스레드 안전성 테스트"""
        instances = []
        
        def create_instance():
            instances.append(ModelManager.get_instance())
        
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=create_instance)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # 모든 인스턴스가 동일해야 함
        first_instance = instances[0]
        for instance in instances:
            self.assertIs(instance, first_instance)
    
    @patch('model_manager.SentenceTransformer')
    def test_model_loading_success(self, mock_sentence_transformer):
        """모델 로딩 성공 테스트"""
        # Mock 모델 설정
        mock_model = Mock()
        mock_model.device = "cpu"
        mock_model.parameters.return_value = [Mock(numel=lambda: 1000000)]
        mock_sentence_transformer.return_value = mock_model
        
        # 임시 캐시 디렉토리 설정
        with patch.dict(os.environ, {'MODEL_CACHE_DIR': self.temp_dir}):
            manager = ModelManager.get_instance()
            model = manager.get_model()
            
            self.assertIsNotNone(model)
            self.assertTrue(manager.is_model_loaded())
            
            status = manager.get_status()
            self.assertIsNotNone(status)
            self.assertTrue(status.is_loaded)
            self.assertIsNone(status.error_message)
    
    @patch('model_manager.SentenceTransformer')
    def test_model_loading_failure(self, mock_sentence_transformer):
        """모델 로딩 실패 테스트"""
        # Mock에서 예외 발생
        mock_sentence_transformer.side_effect = Exception("모델 로딩 실패")
        
        manager = ModelManager.get_instance()
        
        with self.assertRaises(RuntimeError):
            manager.get_model()
        
        self.assertFalse(manager.is_model_loaded())
        
        status = manager.get_status()
        self.assertIsNotNone(status)
        self.assertFalse(status.is_loaded)
        self.assertIsNotNone(status.error_message)
    
    @patch('model_manager.SentenceTransformer')
    def test_thread_safety_model_loading(self, mock_sentence_transformer):
        """모델 로딩 스레드 안전성 테스트"""
        # Mock 모델 설정 (로딩 시간 시뮬레이션)
        mock_model = Mock()
        mock_model.device = "cpu"
        mock_model.parameters.return_value = [Mock(numel=lambda: 1000000)]
        
        def slow_loading(*args, **kwargs):
            time.sleep(0.1)  # 로딩 시간 시뮬레이션
            return mock_model
        
        mock_sentence_transformer.side_effect = slow_loading
        
        models = []
        
        def get_model_thread():
            models.append(ModelManager.get_model())
        
        # 여러 스레드에서 동시에 모델 요청
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=get_model_thread)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # 모든 스레드가 동일한 모델 인스턴스를 받아야 함
        first_model = models[0]
        for model in models:
            self.assertIs(model, first_model)
        
        # SentenceTransformer는 한 번만 호출되어야 함
        self.assertEqual(mock_sentence_transformer.call_count, 1)
    
    @patch('model_manager.SentenceTransformer')
    def test_model_reload(self, mock_sentence_transformer):
        """모델 재로드 테스트"""
        # 첫 번째 모델
        mock_model1 = Mock()
        mock_model1.device = "cpu"
        mock_model1.parameters.return_value = [Mock(numel=lambda: 1000000)]
        
        # 두 번째 모델
        mock_model2 = Mock()
        mock_model2.device = "cpu"
        mock_model2.parameters.return_value = [Mock(numel=lambda: 1000000)]
        
        mock_sentence_transformer.side_effect = [mock_model1, mock_model2]
        
        manager = ModelManager.get_instance()
        
        # 첫 번째 로드
        model1 = manager.get_model()
        self.assertIs(model1, mock_model1)
        
        # 재로드
        success = manager.reload_model()
        self.assertTrue(success)
        
        model2 = manager.get_model()
        self.assertIs(model2, mock_model2)
        self.assertIsNot(model1, model2)
    
    @patch('model_manager.SentenceTransformer')
    def test_model_reload_failure(self, mock_sentence_transformer):
        """모델 재로드 실패 테스트"""
        # 첫 번째 로드는 성공
        mock_model = Mock()
        mock_model.device = "cpu"
        mock_model.parameters.return_value = [Mock(numel=lambda: 1000000)]
        
        # 두 번째 로드는 실패
        mock_sentence_transformer.side_effect = [mock_model, Exception("재로드 실패")]
        
        manager = ModelManager.get_instance()
        
        # 첫 번째 로드
        model1 = manager.get_model()
        self.assertIsNotNone(model1)
        
        # 재로드 실패
        success = manager.reload_model()
        self.assertFalse(success)
    
    def test_config_update(self):
        """설정 업데이트 테스트"""
        manager = ModelManager.get_instance()
        original_config = manager.get_config()
        
        # 설정 업데이트
        manager.update_config(device="cuda", cache_dir="/new/cache")
        
        updated_config = manager.get_config()
        self.assertEqual(updated_config.device, "cuda")
        self.assertEqual(updated_config.cache_dir, "/new/cache")
        self.assertEqual(updated_config.name, original_config.name)  # 변경되지 않은 값
    
    def test_config_update_invalid_key(self):
        """잘못된 키로 설정 업데이트 테스트"""
        manager = ModelManager.get_instance()
        
        # 잘못된 키로 업데이트 (에러가 발생하지 않아야 함)
        manager.update_config(invalid_key="value")
        
        # 설정이 변경되지 않았는지 확인
        config = manager.get_config()
        self.assertFalse(hasattr(config, 'invalid_key'))
    
    @patch('model_manager.os.makedirs')
    @patch('model_manager.os.path.exists')
    @patch('model_manager.SentenceTransformer')
    def test_cache_directory_creation(self, mock_sentence_transformer, mock_exists, mock_makedirs):
        """캐시 디렉토리 생성 테스트"""
        mock_exists.return_value = False
        mock_model = Mock()
        mock_model.device = "cpu"
        mock_model.parameters.return_value = [Mock(numel=lambda: 1000000)]
        mock_sentence_transformer.return_value = mock_model
        
        manager = ModelManager.get_instance()
        manager.get_model()
        
        # makedirs가 호출되었는지 확인
        mock_makedirs.assert_called_once()


class TestGlobalFunctions(unittest.TestCase):
    """전역 함수 테스트"""
    
    def setUp(self):
        """테스트 전 설정"""
        reset_model_manager()
    
    def tearDown(self):
        """테스트 후 정리"""
        reset_model_manager()
    
    @patch('model_manager.SentenceTransformer')
    def test_get_model_function(self, mock_sentence_transformer):
        """get_model 전역 함수 테스트"""
        mock_model = Mock()
        mock_model.device = "cpu"
        mock_model.parameters.return_value = [Mock(numel=lambda: 1000000)]
        mock_sentence_transformer.return_value = mock_model
        
        model = get_model()
        self.assertIsNotNone(model)
    
    def test_is_model_loaded_function(self):
        """is_model_loaded 전역 함수 테스트"""
        # 초기에는 로드되지 않음
        self.assertFalse(is_model_loaded())
    
    def test_get_model_status_function(self):
        """get_model_status 전역 함수 테스트"""
        # 초기에는 상태가 없음
        status = get_model_status()
        self.assertIsNone(status)


class TestMemoryEstimation(unittest.TestCase):
    """메모리 사용량 추정 테스트"""
    
    def setUp(self):
        """테스트 전 설정"""
        reset_model_manager()
    
    def tearDown(self):
        """테스트 후 정리"""
        reset_model_manager()
    
    @patch('model_manager.SentenceTransformer')
    def test_memory_estimation_with_parameters(self, mock_sentence_transformer):
        """파라미터 기반 메모리 추정 테스트"""
        mock_model = Mock()
        mock_model.device = "cpu"
        
        # 1M 파라미터 시뮬레이션
        mock_param = Mock()
        mock_param.numel.return_value = 1000000
        mock_model.parameters.return_value = [mock_param]
        mock_sentence_transformer.return_value = mock_model
        
        manager = ModelManager.get_instance()
        manager.get_model()
        
        status = manager.get_status()
        # 1M 파라미터 * 4바이트 + 오버헤드 = 약 104MB
        self.assertGreater(status.memory_usage, 100)
    
    @patch('model_manager.SentenceTransformer')
    def test_memory_estimation_fallback(self, mock_sentence_transformer):
        """메모리 추정 폴백 테스트"""
        mock_model = Mock()
        mock_model.device = "cpu"
        
        # parameters 메서드가 없는 경우
        del mock_model.parameters
        mock_sentence_transformer.return_value = mock_model
        
        manager = ModelManager.get_instance()
        manager.get_model()
        
        status = manager.get_status()
        # 기본값 사용
        self.assertEqual(status.memory_usage, 1024)


class TestErrorHandlingAndFallback(unittest.TestCase):
    """에러 처리 및 폴백 메커니즘 테스트"""
    
    def setUp(self):
        """테스트 전 설정"""
        reset_model_manager()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """테스트 후 정리"""
        reset_model_manager()
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    @patch('model_manager.SentenceTransformer')
    def test_model_loading_with_retry_success(self, mock_sentence_transformer):
        """재시도 후 성공하는 모델 로딩 테스트"""
        mock_model = Mock()
        mock_model.device = "cpu"
        mock_model.parameters.return_value = [Mock(numel=lambda: 1000000)]
        mock_model.encode.return_value = [[0.1, 0.2, 0.3]]
        
        # 첫 번째 시도는 실패, 두 번째 시도는 성공
        mock_sentence_transformer.side_effect = [Exception("첫 번째 실패"), mock_model]
        
        with patch.dict(os.environ, {'MODEL_CACHE_DIR': self.temp_dir}):
            manager = ModelManager.get_instance()
            manager.set_retry_config(max_attempts=2, delay=0.1)
            
            model = manager.get_model()
            self.assertIsNotNone(model)
            self.assertTrue(manager.is_model_loaded())
            
            # 두 번 호출되었는지 확인 (첫 번째 실패, 두 번째 성공)
            self.assertEqual(mock_sentence_transformer.call_count, 2)
    
    @patch('model_manager.SentenceTransformer')
    def test_model_loading_with_fallback_success(self, mock_sentence_transformer):
        """공유 모델 실패 후 폴백 성공 테스트"""
        mock_fallback_model = Mock()
        mock_fallback_model.device = "cpu"
        mock_fallback_model.parameters.return_value = [Mock(numel=lambda: 1000000)]
        mock_fallback_model.encode.return_value = [[0.1, 0.2, 0.3]]
        
        # 공유 모델은 모든 시도에서 실패, 폴백 모델은 성공
        def side_effect(*args, **kwargs):
            if kwargs.get('device') == 'cpu' and kwargs.get('trust_remote_code') == False:
                return mock_fallback_model  # 폴백 모델
            else:
                raise Exception("공유 모델 로딩 실패")
        
        mock_sentence_transformer.side_effect = side_effect
        
        with patch.dict(os.environ, {'MODEL_CACHE_DIR': self.temp_dir}):
            manager = ModelManager.get_instance()
            manager.set_retry_config(max_attempts=2, delay=0.1)
            
            model = manager.get_model()
            self.assertIsNotNone(model)
            self.assertTrue(manager.is_model_loaded())
            
            status = manager.get_status()
            self.assertTrue(status.fallback_used)
            self.assertEqual(status.retry_count, 2)
    
    @patch('model_manager.SentenceTransformer')
    def test_model_loading_complete_failure(self, mock_sentence_transformer):
        """공유 모델과 폴백 모델 모두 실패 테스트"""
        mock_sentence_transformer.side_effect = Exception("모든 로딩 실패")
        
        with patch.dict(os.environ, {'MODEL_CACHE_DIR': self.temp_dir}):
            manager = ModelManager.get_instance()
            manager.set_retry_config(max_attempts=2, delay=0.1)
            
            with self.assertRaises(ModelAccessError):
                manager.get_model()
            
            self.assertFalse(manager.is_model_loaded())
            
            status = manager.get_status()
            self.assertIsNotNone(status)
            self.assertFalse(status.is_loaded)
            self.assertIsNotNone(status.error_message)
    
    @patch('model_manager.SentenceTransformer')
    def test_configuration_validation_errors(self, mock_sentence_transformer):
        """설정 검증 에러 테스트"""
        # 모든 SentenceTransformer 호출이 실패하도록 설정
        mock_sentence_transformer.side_effect = Exception("설정 에러로 인한 실패")
        
        manager = ModelManager.get_instance()
        
        # 빈 모델 이름
        manager.update_config(name="")
        with self.assertRaises(ModelAccessError):
            manager.get_model()
        
        # 새 인스턴스로 테스트
        reset_model_manager()
        manager = ModelManager.get_instance()
        
        # 빈 캐시 디렉토리
        manager.update_config(name="test-model", cache_dir="")
        with self.assertRaises(ModelAccessError):
            manager.get_model()
    
    @patch('model_manager.os.makedirs')
    def test_cache_directory_creation_failure(self, mock_makedirs):
        """캐시 디렉토리 생성 실패 테스트"""
        mock_makedirs.side_effect = PermissionError("권한 없음")
        
        manager = ModelManager.get_instance()
        manager.update_config(cache_dir="/invalid/path")
        
        with self.assertRaises(ModelAccessError):
            manager.get_model()
    
    @patch('model_manager.SentenceTransformer')
    def test_model_validation_failure(self, mock_sentence_transformer):
        """로드된 모델 검증 실패 테스트"""
        # 모든 모델 로딩이 검증 실패하도록 설정
        mock_model = Mock()
        mock_model.device = "cpu"
        mock_model.parameters.return_value = [Mock(numel=lambda: 1000000)]
        mock_model.encode.side_effect = Exception("인코딩 실패")
        mock_sentence_transformer.return_value = mock_model
        
        with patch.dict(os.environ, {'MODEL_CACHE_DIR': self.temp_dir}):
            manager = ModelManager.get_instance()
            # 폴백도 실패하도록 설정
            manager.set_fallback_enabled(False)
            
            with self.assertRaises(ModelAccessError):
                manager.get_model()
    
    @patch('model_manager.SentenceTransformer')
    def test_fallback_disabled(self, mock_sentence_transformer):
        """폴백 비활성화 테스트"""
        mock_sentence_transformer.side_effect = Exception("로딩 실패")
        
        with patch.dict(os.environ, {'MODEL_CACHE_DIR': self.temp_dir}):
            manager = ModelManager.get_instance()
            manager.set_fallback_enabled(False)
            manager.set_retry_config(max_attempts=1, delay=0.1)
            
            with self.assertRaises(ModelAccessError):
                manager.get_model()
            
            status = manager.get_status()
            self.assertFalse(status.fallback_used)
    
    @patch('model_manager.SentenceTransformer')
    def test_singleton_fallback_loading(self, mock_sentence_transformer):
        """싱글톤 폴백 모델 로딩 테스트"""
        mock_fallback_model = Mock()
        mock_fallback_model.device = "cpu"
        mock_sentence_transformer.return_value = mock_fallback_model
        
        with patch.dict(os.environ, {'MODEL_CACHE_DIR': self.temp_dir}):
            manager = ModelManager.get_instance()
            
            # 싱글톤 폴백 로딩 시도
            success = manager._try_load_fallback_to_singleton()
            
            self.assertTrue(success)
            self.assertIsNotNone(manager._model)
            self.assertIs(manager._model, mock_fallback_model)
            
            # CPU 디바이스와 trust_remote_code=False로 호출되었는지 확인
            mock_sentence_transformer.assert_called_with(
                "nlpai-lab/KURE-v1",
                cache_folder=self.temp_dir,
                device="cpu",
                trust_remote_code=False
            )
    
    @patch('model_manager.SentenceTransformer')
    def test_singleton_fallback_loading_failure(self, mock_sentence_transformer):
        """싱글톤 폴백 모델 로딩 실패 테스트"""
        mock_sentence_transformer.side_effect = Exception("폴백 실패")
        
        manager = ModelManager.get_instance()
        success = manager._try_load_fallback_to_singleton()
        
        self.assertFalse(success)
        self.assertIsNone(manager._model)
    
    def test_error_state_clearing(self):
        """에러 상태 초기화 테스트"""
        manager = ModelManager.get_instance()
        
        # 에러 상태 설정
        manager._status = ModelStatus(
            is_loaded=False,
            model_name="test",
            load_time=0.0,
            memory_usage=0,
            device="unknown",
            error_message="테스트 에러"
        )
        
        # 에러 상태 초기화
        manager.clear_error_state()
        
        self.assertIsNone(manager.get_status())
        self.assertFalse(manager.is_model_loaded())
    
    def test_retry_config_validation(self):
        """재시도 설정 검증 테스트"""
        manager = ModelManager.get_instance()
        
        # 유효한 설정
        manager.set_retry_config(3, 1.5)
        self.assertEqual(manager._max_retry_attempts, 3)
        self.assertEqual(manager._retry_delay, 1.5)
        
        # 잘못된 최대 시도 횟수
        with self.assertRaises(ValueError):
            manager.set_retry_config(0, 1.0)
        
        # 잘못된 지연 시간
        with self.assertRaises(ValueError):
            manager.set_retry_config(3, -1.0)
    
    @patch('model_manager.SentenceTransformer')
    def test_concurrent_access_with_errors(self, mock_sentence_transformer):
        """에러 상황에서의 동시 접근 테스트"""
        mock_model = Mock()
        mock_model.device = "cpu"
        mock_model.parameters.return_value = [Mock(numel=lambda: 1000000)]
        mock_model.encode.return_value = [[0.1, 0.2, 0.3]]
        
        # 처음 몇 번은 실패, 나중에 성공
        call_count = 0
        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise Exception(f"실패 {call_count}")
            return mock_model
        
        mock_sentence_transformer.side_effect = side_effect
        
        with patch.dict(os.environ, {'MODEL_CACHE_DIR': self.temp_dir}):
            manager = ModelManager.get_instance()
            manager.set_retry_config(max_attempts=3, delay=0.1)
            
            results = []
            errors = []
            
            def get_model_thread():
                try:
                    model = manager.get_model()
                    results.append(model)
                except Exception as e:
                    errors.append(e)
            
            # 여러 스레드에서 동시에 모델 요청
            threads = []
            for _ in range(5):
                thread = threading.Thread(target=get_model_thread)
                threads.append(thread)
                thread.start()
            
            for thread in threads:
                thread.join()
            
            # 모든 스레드가 같은 모델을 받거나 같은 에러를 받아야 함
            if results:
                first_model = results[0]
                for model in results:
                    self.assertIs(model, first_model)


class TestGlobalFallbackFunctions(unittest.TestCase):
    """전역 폴백 함수 테스트"""
    
    def setUp(self):
        """테스트 전 설정"""
        reset_model_manager()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """테스트 후 정리"""
        reset_model_manager()
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    @patch('model_manager.SentenceTransformer')
    def test_get_model_with_fallback_success(self, mock_sentence_transformer):
        """get_model_with_fallback 성공 테스트"""
        mock_model = Mock()
        mock_model.device = "cpu"
        mock_model.parameters.return_value = [Mock(numel=lambda: 1000000)]
        mock_model.encode.return_value = [[0.1, 0.2, 0.3]]
        mock_sentence_transformer.return_value = mock_model
        
        with patch.dict(os.environ, {'MODEL_CACHE_DIR': self.temp_dir}):
            model = get_model_with_fallback()
            self.assertIsNotNone(model)
    
    @patch('model_manager.SentenceTransformer')
    def test_get_model_with_fallback_uses_fallback(self, mock_sentence_transformer):
        """get_model_with_fallback 폴백 사용 테스트"""
        mock_fallback_model = Mock()
        mock_fallback_model.device = "cpu"
        
        def side_effect(*args, **kwargs):
            if kwargs.get('device') == 'cpu' and kwargs.get('trust_remote_code') == False:
                return mock_fallback_model  # 개별 폴백 모델
            else:
                raise Exception("공유 모델 실패")
        
        mock_sentence_transformer.side_effect = side_effect
        
        with patch.dict(os.environ, {'MODEL_CACHE_DIR': self.temp_dir}):
            model = get_model_with_fallback()
            self.assertIsNotNone(model)
            self.assertIs(model, mock_fallback_model)
    
    @patch('model_manager.SentenceTransformer')
    def test_get_model_with_fallback_complete_failure(self, mock_sentence_transformer):
        """get_model_with_fallback 완전 실패 테스트"""
        mock_sentence_transformer.side_effect = Exception("모든 실패")
        
        with patch.dict(os.environ, {'MODEL_CACHE_DIR': self.temp_dir}):
            with self.assertRaises(ModelAccessError):
                get_model_with_fallback()


class TestLoggingAndMonitoring(unittest.TestCase):
    """로깅 및 모니터링 테스트"""
    
    def setUp(self):
        """테스트 전 설정"""
        reset_model_manager()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """테스트 후 정리"""
        reset_model_manager()
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    @patch('model_manager.logger')
    @patch('model_manager.SentenceTransformer')
    def test_error_logging(self, mock_sentence_transformer, mock_logger):
        """에러 로깅 테스트"""
        mock_sentence_transformer.side_effect = Exception("테스트 에러")
        
        with patch.dict(os.environ, {'MODEL_CACHE_DIR': self.temp_dir}):
            manager = ModelManager.get_instance()
            manager.set_retry_config(max_attempts=1, delay=0.1)
            
            with self.assertRaises(ModelAccessError):
                manager.get_model()
            
            # 에러 로그가 기록되었는지 확인
            mock_logger.error.assert_called()
            error_calls = [call for call in mock_logger.error.call_args_list 
                          if "테스트 에러" in str(call)]
            self.assertTrue(len(error_calls) > 0)
    
    @patch('model_manager.logger')
    @patch('model_manager.SentenceTransformer')
    def test_lifecycle_logging(self, mock_sentence_transformer, mock_logger):
        """모델 생명주기 로깅 테스트"""
        mock_model = Mock()
        mock_model.device = "cpu"
        mock_model.parameters.return_value = [Mock(numel=lambda: 1000000)]
        mock_model.encode.return_value = [[0.1, 0.2, 0.3]]
        mock_sentence_transformer.return_value = mock_model
        
        with patch.dict(os.environ, {'MODEL_CACHE_DIR': self.temp_dir}):
            manager = ModelManager.get_instance()
            manager.get_model()
            
            # 생명주기 로그가 기록되었는지 확인
            info_calls = [str(call) for call in mock_logger.info.call_args_list]
            
            # 로딩 시작 로그
            start_logs = [log for log in info_calls if "로딩 시작" in log]
            self.assertTrue(len(start_logs) > 0)
            
            # 로딩 완료 로그
            complete_logs = [log for log in info_calls if "로딩 완료" in log]
            self.assertTrue(len(complete_logs) > 0)


if __name__ == '__main__':
    # 테스트 실행
    unittest.main(verbosity=2)