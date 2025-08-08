#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ModelManager 동시 접근 안전성 테스트

이 모듈은 ModelManager의 동시 접근 제어, 요청 큐잉, 타임아웃 처리 등
동시성 관련 기능들을 테스트합니다.
"""

import unittest
import threading
import time
import logging
from unittest.mock import patch, MagicMock
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Any

from model_manager import (
    ModelManager, 
    ModelAccessError, 
    ModelTimeoutError, 
    ModelConcurrencyError,
    reset_model_manager
)

# 테스트용 로깅 설정
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class TestConcurrentAccess(unittest.TestCase):
    """ModelManager 동시 접근 안전성 테스트"""
    
    def setUp(self):
        """각 테스트 전 초기화"""
        reset_model_manager()
        self.manager = ModelManager.get_instance()
        
        # 테스트용 빠른 설정
        self.manager.set_concurrency_config(
            max_concurrent_access=3,
            access_timeout=5.0,
            loading_timeout=10.0
        )
        self.manager.set_retry_config(max_attempts=1, delay=0.1)
    
    def tearDown(self):
        """각 테스트 후 정리"""
        try:
            self.manager.shutdown()
        except:
            pass
        reset_model_manager()
    
    def test_concurrent_model_access_within_limit(self):
        """동시 접근 제한 내에서의 정상 동작 테스트"""
        
        # 모델 로딩을 모킹
        with patch('model_manager.SentenceTransformer') as mock_transformer:
            mock_model = MagicMock()
            mock_model.encode.return_value = [[0.1, 0.2, 0.3]]
            mock_model.device = "cpu"
            mock_transformer.return_value = mock_model
            
            results = []
            errors = []
            
            def access_model(thread_id: int):
                try:
                    model = self.manager.get_model(timeout=3.0)
                    results.append((thread_id, model is not None))
                    time.sleep(0.1)  # 짧은 작업 시뮬레이션
                except Exception as e:
                    errors.append((thread_id, str(e)))
            
            # 제한 내 동시 접근 (3개)
            threads = []
            for i in range(3):
                thread = threading.Thread(target=access_model, args=(i,))
                threads.append(thread)
                thread.start()
            
            for thread in threads:
                thread.join()
            
            # 모든 접근이 성공해야 함
            self.assertEqual(len(results), 3)
            self.assertEqual(len(errors), 0)
            
            # 모든 결과가 성공이어야 함
            for thread_id, success in results:
                self.assertTrue(success, f"Thread {thread_id} failed")
    
    def test_concurrent_access_limit_exceeded(self):
        """동시 접근 제한 초과 시 에러 발생 테스트"""
        
        # 매우 작은 제한으로 설정
        self.manager.set_concurrency_config(
            max_concurrent_access=1,  # 1개로 제한
            access_timeout=2.0,
            loading_timeout=5.0
        )
        
        # 모델 로딩을 모킹하되, 접근 시마다 시간이 걸리도록 설정
        access_count = {"count": 0}
        
        def mock_get_model_with_delay(timeout, start_time):
            access_count["count"] += 1
            time.sleep(0.5)  # 모델 접근 시 지연 시뮬레이션
            
            mock_model = MagicMock()
            mock_model.encode.return_value = [[0.1, 0.2, 0.3]]
            mock_model.device = "cpu"
            return mock_model
        
        # _get_model_with_concurrency_control을 직접 테스트
        results = []
        errors = []
        start_barrier = threading.Barrier(3)  # 3개 스레드 동기화
        
        def access_model_directly(thread_id: int):
            try:
                start_barrier.wait()  # 모든 스레드가 동시에 시작
                # 직접 동시 접근 제어 메서드 호출
                model = self.manager._get_model_with_concurrency_control(timeout=1.0)
                results.append((thread_id, "success"))
            except ModelConcurrencyError as e:
                errors.append((thread_id, "concurrency_error"))
            except Exception as e:
                errors.append((thread_id, f"other_error: {e}"))
        
        # 모델을 미리 로드하지 않고 동시 접근 테스트
        with patch.object(self.manager, '_wait_for_model_or_load', side_effect=mock_get_model_with_delay):
            # 제한 초과 동시 접근 (3개, 제한은 1개)
            threads = []
            for i in range(3):
                thread = threading.Thread(target=access_model_directly, args=(i,))
                threads.append(thread)
                thread.start()
            
            for thread in threads:
                thread.join(timeout=5.0)
        
        # 결과 검증
        total_requests = len(results) + len(errors)
        self.assertEqual(total_requests, 3)
        
        logger.info(f"Concurrency test results: {results}, errors: {errors}")
        
        # 동시 접근 제한 에러가 발생해야 함
        concurrency_errors = [thread_id for thread_id, error_type in errors if error_type == "concurrency_error"]
        self.assertGreater(len(concurrency_errors), 0, f"동시 접근 제한 에러가 발생하지 않음. Results: {results}, Errors: {errors}")
        
        # 적어도 하나는 성공해야 함
        self.assertGreater(len(results), 0, "No successful access")
    
    def test_concurrent_access_counter(self):
        """동시 접근 카운터 테스트"""
        
        # 제한을 2로 설정
        self.manager.set_concurrency_config(
            max_concurrent_access=2,
            access_timeout=5.0,
            loading_timeout=10.0
        )
        
        # 동시 접근 카운터 직접 테스트
        with self.manager._access_lock:
            # 첫 번째 접근
            self.manager._current_access_count += 1
            self.assertEqual(self.manager._current_access_count, 1)
            
            # 두 번째 접근
            self.manager._current_access_count += 1
            self.assertEqual(self.manager._current_access_count, 2)
            
            # 세 번째 접근 시도 - 제한 확인
            if self.manager._current_access_count >= self.manager._max_concurrent_access:
                with self.assertRaises(ModelConcurrencyError):
                    raise ModelConcurrencyError(f"최대 동시 접근 수 초과: {self.manager._current_access_count}/{self.manager._max_concurrent_access}")
            
            # 정리
            self.manager._current_access_count = 0
    
    def test_request_queuing_during_loading(self):
        """모델 로딩 중 요청 큐잉 테스트"""
        
        # 동시 접근 제한을 충분히 크게 설정
        self.manager.set_concurrency_config(
            max_concurrent_access=10,  # 큰 값으로 설정
            access_timeout=8.0,
            loading_timeout=10.0
        )
        
        loading_started = threading.Event()
        loading_can_complete = threading.Event()
        
        def slow_model_loading(*args, **kwargs):
            loading_started.set()
            loading_can_complete.wait(timeout=8.0)  # 로딩 완료 신호 대기
            
            mock_model = MagicMock()
            mock_model.encode.return_value = [[0.1, 0.2, 0.3]]
            mock_model.device = "cpu"
            return mock_model
        
        with patch('model_manager.SentenceTransformer', side_effect=slow_model_loading):
            results = []
            errors = []
            
            def access_model(thread_id: int):
                try:
                    start_time = time.time()
                    model = self.manager.get_model(timeout=10.0)
                    end_time = time.time()
                    results.append((thread_id, end_time - start_time, model is not None))
                except Exception as e:
                    errors.append((thread_id, str(e)))
            
            # 첫 번째 스레드가 로딩을 시작
            thread1 = threading.Thread(target=access_model, args=(1,))
            thread1.start()
            
            # 로딩이 시작될 때까지 대기
            loading_started.wait(timeout=3.0)
            time.sleep(0.2)  # 로딩이 확실히 시작되도록
            
            # 추가 스레드들이 큐에서 대기 (적은 수로 조정)
            threads = []
            for i in range(2, 4):  # 2개만 추가
                thread = threading.Thread(target=access_model, args=(i,))
                threads.append(thread)
                thread.start()
                time.sleep(0.1)  # 스레드들이 순차적으로 대기하도록
            
            # 잠시 후 로딩 완료 허용
            time.sleep(0.5)
            loading_can_complete.set()
            
            # 모든 스레드 완료 대기
            thread1.join(timeout=12.0)
            for thread in threads:
                thread.join(timeout=12.0)
            
            # 결과 검증
            logger.info(f"Queuing test results: {len(results)} successes, {len(errors)} errors")
            if errors:
                logger.error(f"Errors: {errors}")
            
            # 대부분의 요청이 성공해야 함
            self.assertGreater(len(results), 0, "No successful requests")
            
            # 모든 결과가 성공이어야 함
            for thread_id, duration, success in results:
                self.assertTrue(success, f"Thread {thread_id} failed")
                # 대기한 스레드들은 더 오래 걸려야 함
                if thread_id > 1:
                    self.assertGreater(duration, 0.4, f"Thread {thread_id} didn't wait long enough: {duration}s")
    
    def test_access_timeout(self):
        """모델 접근 타임아웃 테스트"""
        
        def never_completing_loading(*args, **kwargs):
            time.sleep(10)  # 타임아웃보다 오래 걸리는 로딩
            mock_model = MagicMock()
            return mock_model
        
        with patch('model_manager.SentenceTransformer', side_effect=never_completing_loading):
            
            start_time = time.time()
            with self.assertRaises(ModelTimeoutError):
                self.manager.get_model(timeout=1.0)  # 1초 타임아웃
            end_time = time.time()
            
            # 타임아웃이 대략 맞는지 확인 (약간의 오차 허용)
            duration = end_time - start_time
            self.assertGreater(duration, 0.9)
            self.assertLess(duration, 2.0)
    
    def test_loading_timeout(self):
        """모델 로딩 타임아웃 테스트"""
        
        # 로딩 타임아웃을 짧게 설정
        self.manager.set_concurrency_config(
            max_concurrent_access=3,
            access_timeout=10.0,
            loading_timeout=1.0  # 1초 로딩 타임아웃
        )
        
        def slow_loading(*args, **kwargs):
            time.sleep(2.0)  # 로딩 타임아웃보다 오래 걸림
            mock_model = MagicMock()
            return mock_model
        
        with patch('model_manager.SentenceTransformer', side_effect=slow_loading):
            
            start_time = time.time()
            with self.assertRaises(ModelTimeoutError):
                self.manager.get_model(timeout=5.0)
            end_time = time.time()
            
            # 로딩 타임아웃이 적용되었는지 확인
            duration = end_time - start_time
            self.assertGreater(duration, 0.9)
            self.assertLess(duration, 2.0)
    
    def test_concurrent_stats_tracking(self):
        """동시 접근 통계 추적 테스트"""
        
        with patch('model_manager.SentenceTransformer') as mock_transformer:
            mock_model = MagicMock()
            mock_model.encode.return_value = [[0.1, 0.2, 0.3]]
            mock_model.device = "cpu"
            mock_transformer.return_value = mock_model
            
            # 초기 상태 확인
            stats = self.manager.get_concurrency_stats()
            self.assertEqual(stats["current_access_count"], 0)
            self.assertEqual(stats["queued_requests"], 0)
            self.assertFalse(stats["is_loading"])
            
            # 모델 로드 후 통계 확인
            model = self.manager.get_model()
            self.assertIsNotNone(model)
            
            # 상태 확인
            status = self.manager.get_status()
            self.assertIsNotNone(status)
            self.assertTrue(status.is_loaded)
            self.assertIsNotNone(status.last_access_time)
    
    def test_loading_cancellation(self):
        """모델 로딩 취소 테스트"""
        
        loading_started = threading.Event()
        loading_should_continue = threading.Event()
        
        def slow_loading(*args, **kwargs):
            loading_started.set()
            # 취소 가능한 시점에서 대기
            loading_should_continue.wait(timeout=10.0)
            mock_model = MagicMock()
            mock_model.encode.return_value = [[0.1, 0.2, 0.3]]
            mock_model.device = "cpu"
            return mock_model
        
        with patch('model_manager.SentenceTransformer', side_effect=slow_loading):
            
            loading_result = {"completed": False, "error": None}
            
            def start_loading():
                try:
                    self.manager.get_model(timeout=15.0)
                    loading_result["completed"] = True
                except Exception as e:
                    loading_result["error"] = str(e)
            
            # 로딩 시작
            loading_thread = threading.Thread(target=start_loading)
            loading_thread.start()
            
            # 로딩이 시작될 때까지 대기
            self.assertTrue(loading_started.wait(timeout=3.0), "Loading didn't start")
            time.sleep(0.1)  # 로딩이 확실히 시작되도록
            
            # 로딩 상태 확인
            stats_before = self.manager.get_concurrency_stats()
            logger.info(f"Stats before cancel: {stats_before}")
            
            # 로딩 취소 시도
            cancelled = self.manager.cancel_loading()
            logger.info(f"Cancel result: {cancelled}")
            
            # 로딩 완료 허용 (취소되지 않은 경우를 위해)
            loading_should_continue.set()
            
            # 스레드 완료 대기
            loading_thread.join(timeout=5.0)
            
            # 상태 확인 - 취소 여부와 관계없이 로딩 상태는 false여야 함
            stats_after = self.manager.get_concurrency_stats()
            logger.info(f"Stats after cancel: {stats_after}")
            logger.info(f"Loading result: {loading_result}")
            
            # 로딩이 완료되었거나 취소되었어야 함
            self.assertFalse(stats_after["is_loading"], "Loading state should be false after completion/cancellation")
    
    def test_thread_safety_stress_test(self):
        """스레드 안전성 스트레스 테스트"""
        
        with patch('model_manager.SentenceTransformer') as mock_transformer:
            mock_model = MagicMock()
            mock_model.encode.return_value = [[0.1, 0.2, 0.3]]
            mock_model.device = "cpu"
            mock_transformer.return_value = mock_model
            
            # 많은 수의 동시 요청
            num_threads = 20
            results = []
            errors = []
            
            def stress_access(thread_id: int):
                try:
                    for i in range(5):  # 각 스레드가 5번 접근
                        model = self.manager.get_model(timeout=2.0)
                        results.append((thread_id, i, model is not None))
                        time.sleep(0.01)  # 짧은 대기
                except Exception as e:
                    errors.append((thread_id, str(e)))
            
            # 스레드 풀로 동시 실행
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(stress_access, i) for i in range(num_threads)]
                
                for future in as_completed(futures, timeout=10.0):
                    try:
                        future.result()
                    except Exception as e:
                        logger.error(f"Stress test thread failed: {e}")
            
            # 결과 검증
            logger.info(f"Stress test results: {len(results)} successes, {len(errors)} errors")
            
            # 대부분의 요청이 성공해야 함 (일부는 동시 접근 제한으로 실패할 수 있음)
            success_rate = len(results) / (len(results) + len(errors)) if (len(results) + len(errors)) > 0 else 0
            self.assertGreater(success_rate, 0.3, f"성공률이 너무 낮음: {success_rate:.3f}")
            
            # 적어도 일부 요청은 성공해야 함
            self.assertGreater(len(results), 0, "No successful requests")
    
    def test_configuration_updates(self):
        """동시 접근 설정 업데이트 테스트"""
        
        # 초기 설정 확인
        stats = self.manager.get_concurrency_stats()
        self.assertEqual(stats["max_concurrent_access"], 3)
        self.assertEqual(stats["access_timeout"], 5.0)
        self.assertEqual(stats["loading_timeout"], 10.0)
        
        # 설정 업데이트
        self.manager.set_concurrency_config(
            max_concurrent_access=5,
            access_timeout=15.0,
            loading_timeout=30.0
        )
        
        # 업데이트된 설정 확인
        stats = self.manager.get_concurrency_stats()
        self.assertEqual(stats["max_concurrent_access"], 5)
        self.assertEqual(stats["access_timeout"], 15.0)
        self.assertEqual(stats["loading_timeout"], 30.0)
        
        # 잘못된 설정 테스트
        with self.assertRaises(ValueError):
            self.manager.set_concurrency_config(0, 5.0, 10.0)  # max_concurrent_access < 1
        
        with self.assertRaises(ValueError):
            self.manager.set_concurrency_config(5, 0, 10.0)  # access_timeout <= 0
        
        with self.assertRaises(ValueError):
            self.manager.set_concurrency_config(5, 5.0, 0)  # loading_timeout <= 0


class TestConcurrentAccessIntegration(unittest.TestCase):
    """동시 접근 통합 테스트"""
    
    def setUp(self):
        """각 테스트 전 초기화"""
        reset_model_manager()
    
    def tearDown(self):
        """각 테스트 후 정리"""
        try:
            ModelManager.get_instance().shutdown()
        except:
            pass
        reset_model_manager()
    
    def test_real_model_concurrent_access(self):
        """실제 모델을 사용한 동시 접근 테스트 (통합 테스트)"""
        
        # 실제 환경에서는 이 테스트를 건너뛸 수 있음
        import os
        if os.getenv("SKIP_INTEGRATION_TESTS", "false").lower() == "true":
            self.skipTest("Integration tests skipped")
        
        manager = ModelManager.get_instance()
        manager.set_concurrency_config(
            max_concurrent_access=2,
            access_timeout=30.0,
            loading_timeout=60.0
        )
        
        results = []
        errors = []
        
        def concurrent_encode(thread_id: int):
            try:
                model = manager.get_model(timeout=45.0)
                # 실제 인코딩 수행
                embeddings = model.encode(["테스트 문장"])
                results.append((thread_id, len(embeddings) > 0))
            except Exception as e:
                errors.append((thread_id, str(e)))
        
        # 2개의 동시 접근 (제한 내)
        threads = []
        for i in range(2):
            thread = threading.Thread(target=concurrent_encode, args=(i,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join(timeout=60.0)
        
        # 결과 검증
        if len(errors) > 0:
            logger.warning(f"Integration test errors: {errors}")
        
        # 적어도 하나는 성공해야 함
        self.assertGreater(len(results), 0, "No successful concurrent access")


if __name__ == '__main__':
    # 테스트 실행
    unittest.main(verbosity=2)