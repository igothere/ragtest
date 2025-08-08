#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
모델 설정 파일

ModelManager에서 사용할 기본 설정을 정의합니다.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import logging

@dataclass
class ModelConfig:
    """Model configuration dataclass for centralized configuration management"""
    name: str = "nlpai-lab/KURE-v1"
    cache_dir: str = "./model_cache"
    device: str = "auto"
    trust_remote_code: bool = False
    max_memory_usage_mb: int = 2048
    model_load_timeout: int = 300
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        self.validate()
    
    def validate(self) -> None:
        """Validate the configuration parameters"""
        if not self.name or not isinstance(self.name, str):
            raise ValueError("Model name must be a non-empty string")
        
        if not self.cache_dir or not isinstance(self.cache_dir, str):
            raise ValueError("Cache directory must be a non-empty string")
        
        valid_devices = ["auto", "cpu", "cuda", "mps"]
        if self.device not in valid_devices:
            raise ValueError(f"Device must be one of {valid_devices}, got: {self.device}")
        
        if not isinstance(self.trust_remote_code, bool):
            raise ValueError("trust_remote_code must be a boolean")
        
        if not isinstance(self.max_memory_usage_mb, int) or self.max_memory_usage_mb <= 0:
            raise ValueError("max_memory_usage_mb must be a positive integer")
        
        if not isinstance(self.model_load_timeout, int) or self.model_load_timeout <= 0:
            raise ValueError("model_load_timeout must be a positive integer")
        
        if self.log_level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            raise ValueError("log_level must be a valid logging level")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "name": self.name,
            "cache_dir": self.cache_dir,
            "device": self.device,
            "trust_remote_code": self.trust_remote_code,
            "max_memory_usage_mb": self.max_memory_usage_mb,
            "model_load_timeout": self.model_load_timeout,
            "log_level": self.log_level,
            "log_format": self.log_format
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ModelConfig':
        """Create ModelConfig from dictionary"""
        return cls(**config_dict)
    
    @classmethod
    def from_env(cls) -> 'ModelConfig':
        """Create ModelConfig from environment variables"""
        return cls(
            name=os.getenv("MODEL_NAME", "nlpai-lab/KURE-v1"),
            cache_dir=os.getenv("MODEL_CACHE_DIR", "./model_cache"),
            device=os.getenv("MODEL_DEVICE", "auto"),
            trust_remote_code=os.getenv("MODEL_TRUST_REMOTE_CODE", "false").lower() == "true",
            max_memory_usage_mb=int(os.getenv("MODEL_MAX_MEMORY_MB", "2048")),
            model_load_timeout=int(os.getenv("MODEL_LOAD_TIMEOUT", "300")),
            log_level=os.getenv("MODEL_LOG_LEVEL", "INFO"),
            log_format=os.getenv("MODEL_LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )

@dataclass
class ModelSettings:
    """Legacy model settings class for backward compatibility"""
    # 기본 모델 설정
    DEFAULT_MODEL_NAME = "nlpai-lab/KURE-v1"
    DEFAULT_CACHE_DIR = "./model_cache"
    DEFAULT_DEVICE = "auto"
    DEFAULT_TRUST_REMOTE_CODE = False
    
    # 성능 관련 설정
    MAX_MEMORY_USAGE_MB = 2048  # 최대 메모리 사용량 (MB)
    MODEL_LOAD_TIMEOUT = 300    # 모델 로딩 타임아웃 (초)
    
    # 로깅 설정
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

def get_model_config_from_env() -> ModelConfig:
    """
    환경 변수에서 모델 설정을 가져옵니다.
    
    Returns:
        ModelConfig: 모델 설정 객체
    """
    return ModelConfig.from_env()

def get_model_config_dict_from_env() -> dict:
    """
    환경 변수에서 모델 설정을 딕셔너리로 가져옵니다. (Legacy support)
    
    Returns:
        dict: 모델 설정 딕셔너리
    """
    return {
        "name": os.getenv("MODEL_NAME", ModelSettings.DEFAULT_MODEL_NAME),
        "cache_dir": os.getenv("MODEL_CACHE_DIR", ModelSettings.DEFAULT_CACHE_DIR),
        "device": os.getenv("MODEL_DEVICE", ModelSettings.DEFAULT_DEVICE),
        "trust_remote_code": os.getenv("MODEL_TRUST_REMOTE_CODE", "false").lower() == "true"
    }

def validate_model_config(config) -> bool:
    """
    모델 설정의 유효성을 검사합니다.
    
    Args:
        config (ModelConfig or dict): 검사할 설정 객체 또는 딕셔너리
        
    Returns:
        bool: 설정이 유효한지 여부
    """
    try:
        if isinstance(config, dict):
            # Dictionary validation for backward compatibility
            required_keys = ["name", "cache_dir", "device", "trust_remote_code"]
            
            # 필수 키 확인
            for key in required_keys:
                if key not in config:
                    print(f"❌ 필수 설정 키 누락: {key}")
                    return False
            
            # 모델명 확인
            if not config["name"] or not isinstance(config["name"], str):
                print("❌ 유효하지 않은 모델명")
                return False
            
            # 캐시 디렉토리 확인
            if not config["cache_dir"] or not isinstance(config["cache_dir"], str):
                print("❌ 유효하지 않은 캐시 디렉토리")
                return False
            
            # 디바이스 설정 확인
            valid_devices = ["auto", "cpu", "cuda", "mps"]
            if config["device"] not in valid_devices:
                print(f"❌ 유효하지 않은 디바이스 설정: {config['device']}")
                print(f"   지원되는 디바이스: {valid_devices}")
                return False
            
            # trust_remote_code 확인
            if not isinstance(config["trust_remote_code"], bool):
                print("❌ trust_remote_code는 boolean 값이어야 합니다")
                return False
            
            return True
        
        elif isinstance(config, ModelConfig):
            # ModelConfig validation
            config.validate()
            return True
        
        else:
            print("❌ 설정은 ModelConfig 객체 또는 딕셔너리여야 합니다")
            return False
            
    except ValueError as e:
        print(f"❌ 설정 검증 실패: {e}")
        return False
    except Exception as e:
        print(f"❌ 예상치 못한 오류: {e}")
        return False

def create_cache_directory(cache_dir: str) -> bool:
    """
    캐시 디렉토리를 생성합니다.
    
    Args:
        cache_dir (str): 생성할 캐시 디렉토리 경로
        
    Returns:
        bool: 생성 성공 여부
    """
    try:
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
            print(f"✅ 캐시 디렉토리 생성: {cache_dir}")
        else:
            print(f"✅ 캐시 디렉토리 존재: {cache_dir}")
        return True
    except Exception as e:
        print(f"❌ 캐시 디렉토리 생성 실패: {e}")
        return False

# 환경별 설정 프리셋
DEVELOPMENT_CONFIG = ModelConfig(
    name="nlpai-lab/KURE-v1",
    cache_dir="./dev_model_cache",
    device="cpu",
    trust_remote_code=False
)

PRODUCTION_CONFIG = ModelConfig(
    name="nlpai-lab/KURE-v1",
    cache_dir="/opt/model_cache",
    device="auto",
    trust_remote_code=False
)

TEST_CONFIG = ModelConfig(
    name="nlpai-lab/KURE-v1",
    cache_dir="./test_model_cache",
    device="cpu",
    trust_remote_code=False
)

# Legacy dictionary configs for backward compatibility
DEVELOPMENT_CONFIG_DICT = {
    "name": "nlpai-lab/KURE-v1",
    "cache_dir": "./dev_model_cache",
    "device": "cpu",
    "trust_remote_code": False
}

PRODUCTION_CONFIG_DICT = {
    "name": "nlpai-lab/KURE-v1",
    "cache_dir": "/opt/model_cache",
    "device": "auto",
    "trust_remote_code": False
}

TEST_CONFIG_DICT = {
    "name": "nlpai-lab/KURE-v1",
    "cache_dir": "./test_model_cache",
    "device": "cpu",
    "trust_remote_code": False
}

def get_config_for_environment(env: str = "development") -> ModelConfig:
    """
    환경에 따른 설정을 반환합니다.
    
    Args:
        env (str): 환경 이름 ("development", "production", "test")
        
    Returns:
        ModelConfig: 환경별 설정 객체
    """
    configs = {
        "development": DEVELOPMENT_CONFIG,
        "production": PRODUCTION_CONFIG,
        "test": TEST_CONFIG
    }
    
    return configs.get(env, DEVELOPMENT_CONFIG)

def get_config_dict_for_environment(env: str = "development") -> dict:
    """
    환경에 따른 설정을 딕셔너리로 반환합니다. (Legacy support)
    
    Args:
        env (str): 환경 이름 ("development", "production", "test")
        
    Returns:
        dict: 환경별 설정 딕셔너리
    """
    configs = {
        "development": DEVELOPMENT_CONFIG_DICT,
        "production": PRODUCTION_CONFIG_DICT,
        "test": TEST_CONFIG_DICT
    }
    
    return configs.get(env, DEVELOPMENT_CONFIG_DICT)