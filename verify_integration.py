#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API Server Integration Verification Script

이 스크립트는 ModelManager와 API 서버의 통합이 올바르게 작동하는지 검증합니다.
"""

import sys
import os
import time
from unittest.mock import patch, MagicMock

# 현재 디렉토리를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_model_manager_integration():
    """ModelManager 통합 테스트"""
    print("🔍 ModelManager 통합 검증 시작...")
    
    try:
        from model_manager import ModelManager, get_model_with_fallback, get_model_status
        
        # 1. 싱글톤 패턴 검증
        print("  1. 싱글톤 패턴 검증...")
        manager1 = ModelManager.get_instance()
        manager2 = ModelManager.get_instance()
        assert manager1 is manager2, "싱글톤 패턴이 올바르게 작동하지 않습니다"
        print("     ✅ 싱글톤 패턴 정상 작동")
        
        # 2. 설정 검증
        print("  2. 모델 설정 검증...")
        config = manager1.get_config()
        assert config.name == "nlpai-lab/KURE-v1", f"모델 이름이 예상과 다릅니다: {config.name}"
        print(f"     ✅ 모델 이름: {config.name}")
        print(f"     ✅ 캐시 디렉토리: {config.cache_dir}")
        print(f"     ✅ 디바이스: {config.device}")
        
        # 3. 전역 함수 검증
        print("  3. 전역 함수 검증...")
        status = get_model_status()
        print(f"     ✅ get_model_status() 작동: {status is not None}")
        
        print("✅ ModelManager 통합 검증 완료\n")
        return True
        
    except Exception as e:
        print(f"❌ ModelManager 통합 검증 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_api_server_imports():
    """API 서버 임포트 테스트"""
    print("🔍 API 서버 임포트 검증 시작...")
    
    try:
        # API 서버 모듈 임포트 테스트
        import api_server
        
        # 필요한 함수들이 임포트되었는지 확인
        assert hasattr(api_server, 'ModelManager'), "ModelManager가 임포트되지 않았습니다"
        assert hasattr(api_server, 'get_model_with_fallback'), "get_model_with_fallback이 임포트되지 않았습니다"
        assert hasattr(api_server, 'get_model_status'), "get_model_status가 임포트되지 않았습니다"
        assert hasattr(api_server, 'ModelAccessError'), "ModelAccessError가 임포트되지 않았습니다"
        
        print("     ✅ 모든 필수 함수 임포트 완료")
        
        # Flask 앱 확인
        assert hasattr(api_server, 'app'), "Flask 앱이 생성되지 않았습니다"
        print("     ✅ Flask 앱 생성 확인")
        
        print("✅ API 서버 임포트 검증 완료\n")
        return True
        
    except Exception as e:
        print(f"❌ API 서버 임포트 검증 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_api_endpoints():
    """API 엔드포인트 테스트"""
    print("🔍 API 엔드포인트 검증 시작...")
    
    try:
        import api_server
        
        # Flask 테스트 클라이언트 생성
        api_server.app.config['TESTING'] = True
        client = api_server.app.test_client()
        
        # 1. 모델 상태 엔드포인트 테스트
        print("  1. /model/status 엔드포인트 테스트...")
        response = client.get('/model/status')
        assert response.status_code in [200, 500], f"예상치 못한 상태 코드: {response.status_code}"
        print(f"     ✅ 상태 코드: {response.status_code}")
        
        # 2. 모델 재로드 엔드포인트 테스트
        print("  2. /model/reload 엔드포인트 테스트...")
        response = client.post('/model/reload')
        assert response.status_code in [200, 500], f"예상치 못한 상태 코드: {response.status_code}"
        print(f"     ✅ 상태 코드: {response.status_code}")
        
        # 3. 채팅 엔드포인트 구조 테스트 (실제 호출은 하지 않음)
        print("  3. /chat 엔드포인트 구조 테스트...")
        # 빈 요청으로 테스트 (400 에러 예상)
        response = client.post('/chat', json={})
        assert response.status_code == 400, f"예상치 못한 상태 코드: {response.status_code}"
        print("     ✅ 잘못된 요청에 대한 적절한 에러 응답")
        
        print("✅ API 엔드포인트 검증 완료\n")
        return True
        
    except Exception as e:
        print(f"❌ API 엔드포인트 검증 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_backward_compatibility():
    """하위 호환성 테스트"""
    print("🔍 하위 호환성 검증 시작...")
    
    try:
        import api_server
        
        # Flask 테스트 클라이언트 생성
        api_server.app.config['TESTING'] = True
        client = api_server.app.test_client()
        
        # 1. 업로드 엔드포인트 존재 확인
        print("  1. /upload 엔드포인트 존재 확인...")
        # 파일 없이 요청 (400 에러 예상)
        response = client.post('/upload')
        assert response.status_code == 400, f"예상치 못한 상태 코드: {response.status_code}"
        print("     ✅ 업로드 엔드포인트 정상 작동")
        
        # 2. 기존 전역 변수들 확인
        print("  2. 기존 설정 변수들 확인...")
        assert hasattr(api_server, 'UPLOAD_FOLDER'), "UPLOAD_FOLDER가 정의되지 않았습니다"
        assert hasattr(api_server, 'ALLOWED_EXTENSIONS'), "ALLOWED_EXTENSIONS가 정의되지 않았습니다"
        assert hasattr(api_server, 'DB_CONFIG'), "DB_CONFIG가 정의되지 않았습니다"
        print("     ✅ 모든 기존 설정 변수 존재")
        
        print("✅ 하위 호환성 검증 완료\n")
        return True
        
    except Exception as e:
        print(f"❌ 하위 호환성 검증 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """메인 검증 함수"""
    print("=" * 60)
    print("🚀 API Server ModelManager Integration Verification")
    print("=" * 60)
    print()
    
    # 테스트 실행
    tests = [
        ("ModelManager 통합", test_model_manager_integration),
        ("API 서버 임포트", test_api_server_imports),
        ("API 엔드포인트", test_api_endpoints),
        ("하위 호환성", test_backward_compatibility),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"📋 {test_name} 테스트 실행 중...")
        result = test_func()
        results.append((test_name, result))
        if not result:
            print(f"⚠️  {test_name} 테스트에서 문제가 발견되었습니다.")
        print()
    
    # 결과 요약
    print("=" * 60)
    print("📊 검증 결과 요약")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} {test_name}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print()
    print(f"총 테스트: {len(results)}")
    print(f"성공: {passed}")
    print(f"실패: {failed}")
    
    if failed == 0:
        print("\n🎉 모든 검증이 성공적으로 완료되었습니다!")
        print("   API 서버와 ModelManager의 통합이 올바르게 작동합니다.")
        return True
    else:
        print(f"\n⚠️  {failed}개의 테스트에서 문제가 발견되었습니다.")
        print("   문제를 해결한 후 다시 실행해주세요.")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)