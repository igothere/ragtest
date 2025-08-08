# 설계 문서

## 개요

이 설계 문서는 RAG 시스템에서 SentenceTransformer 모델 로딩을 최적화하기 위한 솔루션을 설명합니다. 모델을 한 번만 로딩하고 여러 프로세스에서 공유하는 중앙화된 모델 매니저를 구현하여, api_server.py와 rag_with_tables.py에서 발생하는 중복 모델 로딩으로 인한 메모리 비효율성과 시작 시간 문제를 해결합니다.

## 아키텍처

### 현재 아키텍처의 문제점
- **중복 로딩**: SentenceTransformer("nlpai-lab/KURE-v1")가 api_server.py와 rag_with_tables.py에서 독립적으로 로딩됨
- **메모리 낭비**: 각 프로세스가 약 1GB 모델의 자체 복사본을 메모리에 유지
- **시작 지연**: 각 스크립트가 모델 로딩 시간(5-10초)을 경험
- **유지보수 오버헤드**: 모델 설정이 여러 파일에 분산됨

### 제안된 아키텍처

다음 구성 요소를 가진 **싱글톤 모델 매니저** 패턴을 구현합니다:

1. **ModelManager 클래스**: 모델 생명주기를 처리하는 중앙화된 싱글톤
2. **공유 모델 인스턴스**: 프로세스 간 접근 가능한 단일 메모리 내 모델 인스턴스
3. **설정 중앙화**: 모델 설정을 위한 단일 지점
4. **에러 처리 및 폴백**: 공유 모델 실패 시 우아한 성능 저하

## 구성 요소 및 인터페이스

### 1. ModelManager 클래스 (`model_manager.py`)

```python
class ModelManager:
    """공유 SentenceTransformer 모델 관리를 위한 싱글톤 클래스"""
    
    _instance = None
    _model = None
    _model_name = "nlpai-lab/KURE-v1"
    
    @classmethod
    def get_instance(cls) -> 'ModelManager'
    
    @classmethod
    def get_model(cls) -> SentenceTransformer
    
    def _load_model(self) -> SentenceTransformer
    
    def is_model_loaded(self) -> bool
    
    def reload_model(self) -> bool
```

### 2. 통합 지점

#### API 서버 통합
- 직접 모델 로딩을 `ModelManager.get_model()`로 교체
- 기존 채팅 엔드포인트 기능 유지
- 모니터링을 위한 모델 상태 엔드포인트 추가

#### RAG 처리 통합  
- `rag_with_tables.py`에서 직접 모델 로딩 교체
- 기존 문서 처리 워크플로우 유지
- 처리 전 모델 검증 추가

### 3. 설정 관리

#### 환경 변수
```bash
MODEL_NAME=nlpai-lab/KURE-v1
MODEL_CACHE_DIR=./model_cache
MODEL_DEVICE=auto  # auto, cpu, cuda
```

#### 설정 파일 (`config/model_config.py`)
```python
MODEL_CONFIG = {
    "name": "nlpai-lab/KURE-v1",
    "cache_dir": "./model_cache",
    "device": "auto",
    "trust_remote_code": False
}
```

## 데이터 모델

### ModelStatus
```python
@dataclass
class ModelStatus:
    is_loaded: bool
    model_name: str
    load_time: float
    memory_usage: int  # MB
    device: str
    error_message: Optional[str] = None
```

### ModelConfig
```python
@dataclass
class ModelConfig:
    name: str = "nlpai-lab/KURE-v1"
    cache_dir: str = "./model_cache"
    device: str = "auto"
    trust_remote_code: bool = False
```

## 에러 처리

### 에러 시나리오 및 대응

1. **모델 로딩 실패**
   - 상세한 에러 정보 로깅
   - 적절한 HTTP 상태 코드 반환
   - 필요시 개별 모델 로딩으로 폴백 제공

2. **메모리 부족**
   - 모델 로딩 중 메모리 사용량 모니터링
   - 우아한 성능 저하 구현
   - 필요시 모델 캐시 정리

3. **동시 접근 문제**
   - 락을 사용한 스레드 안전 모델 접근
   - 모델 로딩 중 요청 큐잉
   - 장시간 실행 작업에 대한 타임아웃 처리

### 폴백 전략
```python
def get_model_with_fallback():
    try:
        return ModelManager.get_model()
    except Exception as e:
        logger.warning(f"공유 모델 실패: {e}, 개별 로딩으로 폴백")
        return SentenceTransformer("nlpai-lab/KURE-v1")
```

## 테스트 전략

### 단위 테스트
1. **ModelManager 테스트**
   - 싱글톤 패턴 검증
   - 모델 로딩 성공/실패 시나리오
   - 스레드 안전성 검증
   - 메모리 누수 감지

2. **통합 테스트**
   - 공유 모델을 사용한 API 서버
   - 공유 모델을 사용한 문서 처리
   - 동시 접근 테스트
   - 폴백 메커니즘 검증

### 성능 테스트
1. **메모리 사용량 비교**
   - 이전: 2배 모델 메모리 사용량
   - 이후: 1배 모델 메모리 사용량
   - 동시 작업 중 메모리 모니터링

2. **시작 시간 측정**
   - 초기 모델 로딩 시간
   - 후속 접근 시간 (약 0ms여야 함)
   - 전체 시스템 시작 개선

3. **처리량 테스트**
   - 문서 처리 속도
   - 채팅 응답 시간
   - 동시 요청 처리

### 부하 테스트
1. **동시 문서 처리**
   - 여러 파일 동시 처리
   - 높은 부하 하에서 모델 접근
   - 리소스 사용률 모니터링

2. **API 스트레스 테스트**
   - 다중 채팅 요청
   - 모델 공유 안정성
   - 응답 시간 일관성

## 구현 단계

### 1단계: 핵심 모델 매니저
- ModelManager 싱글톤 클래스 생성
- 기본 모델 로딩 및 공유 구현
- 설정 관리 추가
- 핵심 기능에 대한 단위 테스트

### 2단계: API 서버 통합
- ModelManager를 사용하도록 api_server.py 수정
- 채팅 엔드포인트 구현 업데이트
- 모델 상태 모니터링 엔드포인트 추가
- API 기능에 대한 통합 테스트

### 3단계: 문서 처리 통합
- ModelManager를 사용하도록 rag_with_tables.py 수정
- 서브프로세스 모델 접근 패턴 업데이트
- 하위 호환성 보장
- 문서 처리에 대한 통합 테스트

### 4단계: 에러 처리 및 모니터링
- 포괄적인 에러 처리 구현
- 폴백 메커니즘 추가
- 성능 모니터링 및 로깅
- 부하 테스트 및 최적화

## 마이그레이션 전략

### 하위 호환성
- 기존 API 계약 유지
- 현재 기능 보존
- 기능 플래그를 통한 점진적 롤아웃
- 쉬운 롤백 메커니즘

### 배포 단계
1. 통합 없이 ModelManager 배포
2. 기능 플래그와 함께 api_server.py 업데이트
3. 기능 플래그와 함께 rag_with_tables.py 업데이트
4. 기본적으로 공유 모델 활성화
5. 개별 모델 로딩 코드 제거

## 모니터링 및 관찰 가능성

### 추적할 메트릭
- 모델 로딩 시간
- 메모리 사용량 감소
- 요청 처리 시간
- 에러율 및 유형
- 동시 접근 패턴

### 로깅 전략
- 모델 생명주기 이벤트
- 성능 메트릭
- 에러 조건
- 리소스 사용률

### 헬스 체크
- 모델 가용성 엔드포인트
- 메모리 사용량 모니터링
- 성능 저하 알림