#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
싱글톤 수정 사항 테스트

중복 모델 로딩이 방지되는지 확인합니다.
"""

import time
import threading
from model_manager import ModelManager, get_model_with_fallback, reset_model_manager

def test_singleton_behavior():
    """싱글톤 동작 테스트"""
    print("🔍 싱글톤 동작 테스트 시작")
    
    # 초기화
    reset_model_manager()
    
    try:
        # 첫 번째 모델 요청
        print("1️⃣ 첫 번째 모델 요청...")
        model1 = get_model_with_fallback(timeout=60.0)
        print(f"   ✅ 첫 번째 모델 로드 성공: {type(model1)}")
        print(f"   📍 모델 디바이스: {model1.device}")
        
        # 두 번째 모델 요청 (같은 인스턴스여야 함)
        print("2️⃣ 두 번째 모델 요청...")
        model2 = get_model_with_fallback(timeout=10.0)
        print(f"   ✅ 두 번째 모델 로드 성공: {type(model2)}")
        
        # 같은 인스턴스인지 확인
        is_same = model1 is model2
        print(f"   🔄 같은 인스턴스인가? {is_same}")
        
        if is_same:
            print("   ✅ 싱글톤 패턴 정상 작동!")
        else:
            print("   ❌ 싱글톤 패턴 실패 - 다른 인스턴스!")
            
        # 상태 확인
        manager = ModelManager.get_instance()
        status = manager.get_status()
        if status:
            print(f"   📊 모델 상태:")
            print(f"      - 로드됨: {status.is_loaded}")
            print(f"      - 모델명: {status.model_name}")
            print(f"      - 디바이스: {status.device}")
            print(f"      - 폴백 사용: {status.fallback_used}")
            print(f"      - 메모리 사용량: {status.memory_usage}MB")
        
        return is_same
        
    except Exception as e:
        print(f"   ❌ 테스트 실패: {e}")
        return False

def test_concurrent_access():
    """동시 접근 테스트"""
    print("\n🔍 동시 접근 테스트 시작")
    
    # 초기화
    reset_model_manager()
    
    models = []
    errors = []
    
    def get_model_thread(thread_id):
        try:
            print(f"   🧵 스레드 {thread_id} 시작")
            model = get_model_with_fallback(timeout=60.0)
            models.append((thread_id, model))
            print(f"   ✅ 스레드 {thread_id} 완료")
        except Exception as e:
            errors.append((thread_id, str(e)))
            print(f"   ❌ 스레드 {thread_id} 실패: {e}")
    
    # 3개 스레드로 동시 접근
    threads = []
    for i in range(3):
        thread = threading.Thread(target=get_model_thread, args=(i,))
        threads.append(thread)
        thread.start()
    
    # 모든 스레드 완료 대기
    for thread in threads:
        thread.join(timeout=70.0)
    
    print(f"   📊 결과: {len(models)}개 성공, {len(errors)}개 실패")
    
    if len(models) > 0:
        # 모든 모델이 같은 인스턴스인지 확인
        first_model = models[0][1]
        all_same = all(model is first_model for _, model in models)
        print(f"   🔄 모든 모델이 같은 인스턴스인가? {all_same}")
        
        if all_same:
            print("   ✅ 동시 접근에서도 싱글톤 패턴 정상 작동!")
        else:
            print("   ❌ 동시 접근에서 싱글톤 패턴 실패!")
            
        return all_same
    else:
        print("   ❌ 모든 스레드 실패")
        return False

if __name__ == "__main__":
    print("🚀 싱글톤 수정 사항 테스트")
    print("=" * 50)
    
    # 기본 싱글톤 테스트
    singleton_ok = test_singleton_behavior()
    
    # 동시 접근 테스트
    concurrent_ok = test_concurrent_access()
    
    print("\n" + "=" * 50)
    print("📋 최종 결과:")
    print(f"   싱글톤 동작: {'✅ 성공' if singleton_ok else '❌ 실패'}")
    print(f"   동시 접근: {'✅ 성공' if concurrent_ok else '❌ 실패'}")
    
    if singleton_ok and concurrent_ok:
        print("🎉 모든 테스트 통과! 중복 모델 로딩 문제가 해결되었습니다.")
    else:
        print("⚠️  일부 테스트 실패. 추가 수정이 필요합니다.")