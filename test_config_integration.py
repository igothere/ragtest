#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration integration tests

Tests the configuration system in realistic usage scenarios.
"""

import os
import tempfile
from unittest.mock import patch
from config import ModelConfig, get_model_config_from_env, get_config_for_environment

def test_real_world_environment_loading():
    """Test loading configuration in a real-world scenario"""
    print("=== Real-world Environment Loading Test ===")
    
    # Simulate production environment variables
    production_env = {
        "MODEL_NAME": "nlpai-lab/KURE-v1",
        "MODEL_CACHE_DIR": "/opt/models/cache",
        "MODEL_DEVICE": "cuda",
        "MODEL_TRUST_REMOTE_CODE": "false",
        "MODEL_MAX_MEMORY_MB": "4096",
        "MODEL_LOAD_TIMEOUT": "600",
        "MODEL_LOG_LEVEL": "WARNING"
    }
    
    with patch.dict(os.environ, production_env, clear=True):
        config = get_model_config_from_env()
        
        print(f"✅ Loaded production config: {config.name}")
        print(f"✅ Cache directory: {config.cache_dir}")
        print(f"✅ Device: {config.device}")
        print(f"✅ Memory limit: {config.max_memory_usage_mb}MB")
        print(f"✅ Timeout: {config.model_load_timeout}s")
        print(f"✅ Log level: {config.log_level}")
        
        # Verify all values are correctly loaded
        assert config.name == "nlpai-lab/KURE-v1"
        assert config.cache_dir == "/opt/models/cache"
        assert config.device == "cuda"
        assert config.trust_remote_code == False
        assert config.max_memory_usage_mb == 4096
        assert config.model_load_timeout == 600
        assert config.log_level == "WARNING"
        
        print("✅ All production environment values loaded correctly")
    
    print()

def test_partial_environment_override():
    """Test partial environment variable override with defaults"""
    print("=== Partial Environment Override Test ===")
    
    # Only set some environment variables
    partial_env = {
        "MODEL_DEVICE": "cpu",
        "MODEL_LOG_LEVEL": "DEBUG"
    }
    
    with patch.dict(os.environ, partial_env, clear=True):
        config = get_model_config_from_env()
        
        print(f"✅ Model name (default): {config.name}")
        print(f"✅ Device (override): {config.device}")
        print(f"✅ Cache dir (default): {config.cache_dir}")
        print(f"✅ Log level (override): {config.log_level}")
        
        # Verify defaults and overrides
        assert config.name == "nlpai-lab/KURE-v1"  # Default
        assert config.device == "cpu"  # Override
        assert config.cache_dir == "./model_cache"  # Default
        assert config.log_level == "DEBUG"  # Override
        assert config.trust_remote_code == False  # Default
        
        print("✅ Partial override with defaults working correctly")
    
    print()

def test_environment_specific_configs():
    """Test environment-specific configuration presets"""
    print("=== Environment-Specific Configuration Test ===")
    
    environments = ["development", "production", "test"]
    
    for env_name in environments:
        config = get_config_for_environment(env_name)
        
        print(f"✅ {env_name.capitalize()} environment:")
        print(f"   Model: {config.name}")
        print(f"   Cache: {config.cache_dir}")
        print(f"   Device: {config.device}")
        
        # Verify environment-specific settings
        if env_name == "development":
            assert config.cache_dir == "./dev_model_cache"
            assert config.device == "cpu"
        elif env_name == "production":
            assert config.cache_dir == "/opt/model_cache"
            assert config.device == "auto"
        elif env_name == "test":
            assert config.cache_dir == "./test_model_cache"
            assert config.device == "cpu"
        
        # All environments should use the same model
        assert config.name == "nlpai-lab/KURE-v1"
    
    print("✅ All environment-specific configurations working correctly")
    print()

def test_config_validation_scenarios():
    """Test configuration validation in various scenarios"""
    print("=== Configuration Validation Scenarios ===")
    
    # Test 1: Valid configuration
    try:
        valid_config = ModelConfig(
            name="test-model",
            cache_dir="/tmp/test",
            device="cuda",
            trust_remote_code=False
        )
        print("✅ Valid configuration created successfully")
    except Exception as e:
        print(f"❌ Valid configuration failed: {e}")
        return False
    
    # Test 2: Invalid device
    try:
        ModelConfig(device="invalid_device")
        print("❌ Invalid device should have failed")
        return False
    except ValueError:
        print("✅ Invalid device correctly rejected")
    
    # Test 3: Invalid memory setting
    try:
        ModelConfig(max_memory_usage_mb=-100)
        print("❌ Negative memory should have failed")
        return False
    except ValueError:
        print("✅ Negative memory correctly rejected")
    
    # Test 4: Invalid log level
    try:
        ModelConfig(log_level="INVALID_LEVEL")
        print("❌ Invalid log level should have failed")
        return False
    except ValueError:
        print("✅ Invalid log level correctly rejected")
    
    print("✅ All validation scenarios working correctly")
    print()

def test_config_serialization_roundtrip():
    """Test configuration serialization and deserialization"""
    print("=== Configuration Serialization Roundtrip Test ===")
    
    # Create original configuration
    original_config = ModelConfig(
        name="roundtrip-test",
        cache_dir="/tmp/roundtrip",
        device="mps",
        trust_remote_code=True,
        max_memory_usage_mb=1024,
        model_load_timeout=120,
        log_level="DEBUG"
    )
    
    # Serialize to dictionary
    config_dict = original_config.to_dict()
    print(f"✅ Serialized to dict with {len(config_dict)} keys")
    
    # Deserialize back to ModelConfig
    restored_config = ModelConfig.from_dict(config_dict)
    print(f"✅ Deserialized back to ModelConfig")
    
    # Verify all fields match
    assert original_config.name == restored_config.name
    assert original_config.cache_dir == restored_config.cache_dir
    assert original_config.device == restored_config.device
    assert original_config.trust_remote_code == restored_config.trust_remote_code
    assert original_config.max_memory_usage_mb == restored_config.max_memory_usage_mb
    assert original_config.model_load_timeout == restored_config.model_load_timeout
    assert original_config.log_level == restored_config.log_level
    
    print("✅ Serialization roundtrip successful - all fields match")
    print()

def main():
    """Run all integration tests"""
    print("Configuration System Integration Tests")
    print("=" * 50)
    print()
    
    try:
        test_real_world_environment_loading()
        test_partial_environment_override()
        test_environment_specific_configs()
        test_config_validation_scenarios()
        test_config_serialization_roundtrip()
        
        print("🎉 All integration tests passed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)