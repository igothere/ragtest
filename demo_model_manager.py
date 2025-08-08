#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ModelManager 데모 스크립트

ModelManager의 기본 사용법을 보여주는 데모입니다.
"""

import time
from model_manager import ModelManager, get_model, get_model_status, is_model_loaded

def main():
    print("==============================================")
    print("🚀 ModelManager 데모 시작")
    print("==============================================")
    
    # 1. 초기 상태 확인
    print("\n1. 초기 상태 확인")
    print(f"   모델 로드 상태: {is_model_loaded()}")
    print(f"   모델 상태: {get_model_status()}")
    
    # 2. 첫 번째 모델 로드 (시간 측정)
    print("\n2. 첫 번째 모델 로드")
    start_time = time.time()
    try:
        model1 = get_model()
        load_time = time.time() - start_time
        print(f"   ✅ 모델 로드 성공 ({load_time:.2f}초)")
        print(f"   모델 타입: {type(model1)}")
        print(f"   모델 로드 상태: {is_model_loaded()}")
        
        # 상태 정보 출력
        status = get_model_status()
        if status:
            print(f"   모델명: {status.model_name}")
            print(f"   로드 시간: {status.load_time:.2f}초")
            print(f"   메모리 사용량: ~{status.memory_usage}MB")
            print(f"   디바이스: {status.device}")
    except Exception as e:
        print(f"   ❌ 모델 로드 실패: {e}")
        return
    
    # 3. 두 번째 모델 접근 (캐시된 모델 사용)
    print("\n3. 두 번째 모델 접근 (캐시된 모델)")
    start_time = time.time()
    model2 = get_model()
    access_time = time.time() - start_time
    print(f"   ✅ 모델 접근 완료 ({access_time:.4f}초)")
    print(f"   동일한 인스턴스: {model1 is model2}")
    
    # 4. 모델 사용 예제
    print("\n4. 모델 사용 예제")
    try:
        test_text = "안녕하세요, 이것은 테스트 문장입니다."
        print(f"   입력 텍스트: {test_text}")
        
        start_time = time.time()
        embedding = model1.encode(test_text)
        encode_time = time.time() - start_time
        
        print(f"   ✅ 임베딩 생성 완료 ({encode_time:.4f}초)")
        print(f"   임베딩 차원: {embedding.shape}")
        print(f"   임베딩 타입: {type(embedding)}")
    except Exception as e:
        print(f"   ❌ 임베딩 생성 실패: {e}")
    
    # 5. ModelManager 인스턴스 직접 사용
    print("\n5. ModelManager 인스턴스 직접 사용")
    manager = ModelManager.get_instance()
    config = manager.get_config()
    print(f"   모델명: {config.name}")
    print(f"   캐시 디렉토리: {config.cache_dir}")
    print(f"   디바이스: {config.device}")
    print(f"   신뢰할 수 있는 원격 코드: {config.trust_remote_code}")
    
    # 6. 설정 업데이트 예제
    print("\n6. 설정 업데이트 예제")
    print("   현재 디바이스 설정:", config.device)
    manager.update_config(device="cpu")
    updated_config = manager.get_config()
    print("   업데이트된 디바이스 설정:", updated_config.device)
    
    print("\n==============================================")
    print("🎉 ModelManager 데모 완료")
    print("==============================================")

if __name__ == "__main__":
    main()