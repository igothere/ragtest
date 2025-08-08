#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration system tests

Tests for ModelConfig dataclass and configuration loading functionality.
"""

import os
import pytest
import tempfile
from unittest.mock import patch
from config import (
    ModelConfig, 
    get_model_config_from_env,
    get_model_config_dict_from_env,
    validate_model_config,
    get_config_for_environment,
    get_config_dict_for_environment,
    create_cache_directory
)


class TestModelConfig:
    """Test cases for ModelConfig dataclass"""
    
    def test_default_config_creation(self):
        """Test creating ModelConfig with default values"""
        config = ModelConfig()
        
        assert config.name == "nlpai-lab/KURE-v1"
        assert config.cache_dir == "./model_cache"
        assert config.device == "auto"
        assert config.trust_remote_code == False
        assert config.max_memory_usage_mb == 2048
        assert config.model_load_timeout == 300
        assert config.log_level == "INFO"
    
    def test_custom_config_creation(self):
        """Test creating ModelConfig with custom values"""
        config = ModelConfig(
            name="custom-model",
            cache_dir="/custom/cache",
            device="cpu",
            trust_remote_code=True,
            max_memory_usage_mb=1024,
            model_load_timeout=600,
            log_level="DEBUG"
        )
        
        assert config.name == "custom-model"
        assert config.cache_dir == "/custom/cache"
        assert config.device == "cpu"
        assert config.trust_remote_code == True
        assert config.max_memory_usage_mb == 1024
        assert config.model_load_timeout == 600
        assert config.log_level == "DEBUG"
    
    def test_config_validation_valid(self):
        """Test validation with valid configuration"""
        config = ModelConfig(
            name="test-model",
            cache_dir="/test/cache",
            device="cuda",
            trust_remote_code=False
        )
        # Should not raise any exception
        config.validate()
    
    def test_config_validation_invalid_name(self):
        """Test validation with invalid model name"""
        with pytest.raises(ValueError, match="Model name must be a non-empty string"):
            ModelConfig(name="")
        
        with pytest.raises(ValueError, match="Model name must be a non-empty string"):
            ModelConfig(name=None)
    
    def test_config_validation_invalid_cache_dir(self):
        """Test validation with invalid cache directory"""
        with pytest.raises(ValueError, match="Cache directory must be a non-empty string"):
            ModelConfig(cache_dir="")
        
        with pytest.raises(ValueError, match="Cache directory must be a non-empty string"):
            ModelConfig(cache_dir=None)
    
    def test_config_validation_invalid_device(self):
        """Test validation with invalid device"""
        with pytest.raises(ValueError, match="Device must be one of"):
            ModelConfig(device="invalid")
    
    def test_config_validation_invalid_trust_remote_code(self):
        """Test validation with invalid trust_remote_code"""
        with pytest.raises(ValueError, match="trust_remote_code must be a boolean"):
            ModelConfig(trust_remote_code="true")
    
    def test_config_validation_invalid_memory_usage(self):
        """Test validation with invalid memory usage"""
        with pytest.raises(ValueError, match="max_memory_usage_mb must be a positive integer"):
            ModelConfig(max_memory_usage_mb=0)
        
        with pytest.raises(ValueError, match="max_memory_usage_mb must be a positive integer"):
            ModelConfig(max_memory_usage_mb=-1)
    
    def test_config_validation_invalid_timeout(self):
        """Test validation with invalid timeout"""
        with pytest.raises(ValueError, match="model_load_timeout must be a positive integer"):
            ModelConfig(model_load_timeout=0)
        
        with pytest.raises(ValueError, match="model_load_timeout must be a positive integer"):
            ModelConfig(model_load_timeout=-1)
    
    def test_config_validation_invalid_log_level(self):
        """Test validation with invalid log level"""
        with pytest.raises(ValueError, match="log_level must be a valid logging level"):
            ModelConfig(log_level="INVALID")
    
    def test_to_dict(self):
        """Test converting ModelConfig to dictionary"""
        config = ModelConfig(
            name="test-model",
            cache_dir="/test/cache",
            device="cpu"
        )
        
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict["name"] == "test-model"
        assert config_dict["cache_dir"] == "/test/cache"
        assert config_dict["device"] == "cpu"
        assert "trust_remote_code" in config_dict
        assert "max_memory_usage_mb" in config_dict
    
    def test_from_dict(self):
        """Test creating ModelConfig from dictionary"""
        config_dict = {
            "name": "dict-model",
            "cache_dir": "/dict/cache",
            "device": "cuda",
            "trust_remote_code": True,
            "max_memory_usage_mb": 1024,
            "model_load_timeout": 600,
            "log_level": "DEBUG",
            "log_format": "custom format"
        }
        
        config = ModelConfig.from_dict(config_dict)
        
        assert config.name == "dict-model"
        assert config.cache_dir == "/dict/cache"
        assert config.device == "cuda"
        assert config.trust_remote_code == True
        assert config.max_memory_usage_mb == 1024
        assert config.model_load_timeout == 600
        assert config.log_level == "DEBUG"
        assert config.log_format == "custom format"


class TestEnvironmentConfiguration:
    """Test cases for environment variable configuration loading"""
    
    def test_from_env_defaults(self):
        """Test loading configuration from environment with defaults"""
        with patch.dict(os.environ, {}, clear=True):
            config = ModelConfig.from_env()
            
            assert config.name == "nlpai-lab/KURE-v1"
            assert config.cache_dir == "./model_cache"
            assert config.device == "auto"
            assert config.trust_remote_code == False
            assert config.max_memory_usage_mb == 2048
            assert config.model_load_timeout == 300
            assert config.log_level == "INFO"
    
    def test_from_env_custom_values(self):
        """Test loading configuration from environment with custom values"""
        env_vars = {
            "MODEL_NAME": "custom-env-model",
            "MODEL_CACHE_DIR": "/env/cache",
            "MODEL_DEVICE": "cpu",
            "MODEL_TRUST_REMOTE_CODE": "true",
            "MODEL_MAX_MEMORY_MB": "1024",
            "MODEL_LOAD_TIMEOUT": "600",
            "MODEL_LOG_LEVEL": "DEBUG",
            "MODEL_LOG_FORMAT": "custom env format"
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            config = ModelConfig.from_env()
            
            assert config.name == "custom-env-model"
            assert config.cache_dir == "/env/cache"
            assert config.device == "cpu"
            assert config.trust_remote_code == True
            assert config.max_memory_usage_mb == 1024
            assert config.model_load_timeout == 600
            assert config.log_level == "DEBUG"
            assert config.log_format == "custom env format"
    
    def test_get_model_config_from_env(self):
        """Test get_model_config_from_env function"""
        env_vars = {
            "MODEL_NAME": "function-test-model",
            "MODEL_DEVICE": "cuda"
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            config = get_model_config_from_env()
            
            assert isinstance(config, ModelConfig)
            assert config.name == "function-test-model"
            assert config.device == "cuda"
    
    def test_get_model_config_dict_from_env(self):
        """Test get_model_config_dict_from_env function for backward compatibility"""
        env_vars = {
            "MODEL_NAME": "dict-test-model",
            "MODEL_DEVICE": "mps"
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            config_dict = get_model_config_dict_from_env()
            
            assert isinstance(config_dict, dict)
            assert config_dict["name"] == "dict-test-model"
            assert config_dict["device"] == "mps"


class TestConfigurationValidation:
    """Test cases for configuration validation"""
    
    def test_validate_model_config_with_modelconfig(self):
        """Test validation with ModelConfig object"""
        config = ModelConfig(name="test-model")
        assert validate_model_config(config) == True
    
    def test_validate_model_config_with_dict(self):
        """Test validation with dictionary (backward compatibility)"""
        config_dict = {
            "name": "test-model",
            "cache_dir": "./test_cache",
            "device": "cpu",
            "trust_remote_code": False
        }
        assert validate_model_config(config_dict) == True
    
    def test_validate_model_config_invalid_dict(self):
        """Test validation with invalid dictionary"""
        config_dict = {
            "name": "",  # Invalid empty name
            "cache_dir": "./test_cache",
            "device": "cpu",
            "trust_remote_code": False
        }
        assert validate_model_config(config_dict) == False
    
    def test_validate_model_config_missing_keys(self):
        """Test validation with missing required keys"""
        config_dict = {
            "name": "test-model",
            # Missing required keys
        }
        assert validate_model_config(config_dict) == False
    
    def test_validate_model_config_invalid_type(self):
        """Test validation with invalid configuration type"""
        assert validate_model_config("invalid") == False
        assert validate_model_config(123) == False


class TestEnvironmentConfigurations:
    """Test cases for environment-specific configurations"""
    
    def test_get_config_for_environment_development(self):
        """Test getting development environment configuration"""
        config = get_config_for_environment("development")
        
        assert isinstance(config, ModelConfig)
        assert config.cache_dir == "./dev_model_cache"
        assert config.device == "cpu"
    
    def test_get_config_for_environment_production(self):
        """Test getting production environment configuration"""
        config = get_config_for_environment("production")
        
        assert isinstance(config, ModelConfig)
        assert config.cache_dir == "/opt/model_cache"
        assert config.device == "auto"
    
    def test_get_config_for_environment_test(self):
        """Test getting test environment configuration"""
        config = get_config_for_environment("test")
        
        assert isinstance(config, ModelConfig)
        assert config.cache_dir == "./test_model_cache"
        assert config.device == "cpu"
    
    def test_get_config_for_environment_default(self):
        """Test getting configuration for unknown environment (should default to development)"""
        config = get_config_for_environment("unknown")
        
        assert isinstance(config, ModelConfig)
        assert config.cache_dir == "./dev_model_cache"
        assert config.device == "cpu"
    
    def test_get_config_dict_for_environment(self):
        """Test getting environment configuration as dictionary (backward compatibility)"""
        config_dict = get_config_dict_for_environment("production")
        
        assert isinstance(config_dict, dict)
        assert config_dict["cache_dir"] == "/opt/model_cache"
        assert config_dict["device"] == "auto"


class TestCacheDirectoryCreation:
    """Test cases for cache directory creation"""
    
    def test_create_cache_directory_success(self):
        """Test successful cache directory creation"""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = os.path.join(temp_dir, "test_cache")
            
            result = create_cache_directory(cache_dir)
            
            assert result == True
            assert os.path.exists(cache_dir)
            assert os.path.isdir(cache_dir)
    
    def test_create_cache_directory_existing(self):
        """Test cache directory creation when directory already exists"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Directory already exists
            result = create_cache_directory(temp_dir)
            
            assert result == True
            assert os.path.exists(temp_dir)
    
    def test_create_cache_directory_failure(self):
        """Test cache directory creation failure"""
        # Try to create directory in non-existent parent
        invalid_path = "/non/existent/path/cache"
        
        result = create_cache_directory(invalid_path)
        
        assert result == False


if __name__ == "__main__":
    # Run tests if script is executed directly
    pytest.main([__file__, "-v"])