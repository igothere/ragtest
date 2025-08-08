#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration system demonstration

Shows how to use the ModelConfig dataclass and configuration loading functionality.
"""

import os
from config import (
    ModelConfig,
    get_model_config_from_env,
    get_config_for_environment,
    validate_model_config,
    create_cache_directory
)

def demo_basic_config():
    """Demonstrate basic ModelConfig usage"""
    print("=== Basic ModelConfig Usage ===")
    
    # Create default configuration
    config = ModelConfig()
    print(f"Default config: {config}")
    print(f"Model name: {config.name}")
    print(f"Cache dir: {config.cache_dir}")
    print(f"Device: {config.device}")
    print()
    
    # Create custom configuration
    custom_config = ModelConfig(
        name="custom-model",
        cache_dir="/tmp/custom_cache",
        device="cpu",
        trust_remote_code=True
    )
    print(f"Custom config: {custom_config}")
    print()

def demo_environment_config():
    """Demonstrate environment variable configuration"""
    print("=== Environment Variable Configuration ===")
    
    # Set some environment variables
    os.environ["MODEL_NAME"] = "demo-model"
    os.environ["MODEL_DEVICE"] = "cuda"
    os.environ["MODEL_CACHE_DIR"] = "/tmp/demo_cache"
    
    # Load configuration from environment
    env_config = get_model_config_from_env()
    print(f"Config from environment: {env_config}")
    print(f"Model name from env: {env_config.name}")
    print(f"Device from env: {env_config.device}")
    print()
    
    # Clean up environment variables
    del os.environ["MODEL_NAME"]
    del os.environ["MODEL_DEVICE"]
    del os.environ["MODEL_CACHE_DIR"]

def demo_environment_presets():
    """Demonstrate environment-specific configuration presets"""
    print("=== Environment-Specific Presets ===")
    
    environments = ["development", "production", "test"]
    
    for env in environments:
        config = get_config_for_environment(env)
        print(f"{env.capitalize()} config:")
        print(f"  Cache dir: {config.cache_dir}")
        print(f"  Device: {config.device}")
        print(f"  Model: {config.name}")
        print()

def demo_config_validation():
    """Demonstrate configuration validation"""
    print("=== Configuration Validation ===")
    
    # Valid configuration
    valid_config = ModelConfig(name="test-model", device="cpu")
    print(f"Valid config validation: {validate_model_config(valid_config)}")
    
    # Test validation with dictionary (backward compatibility)
    valid_dict = {
        "name": "dict-model",
        "cache_dir": "./cache",
        "device": "auto",
        "trust_remote_code": False
    }
    print(f"Valid dict validation: {validate_model_config(valid_dict)}")
    
    # Invalid configuration (will raise exception)
    try:
        invalid_config = ModelConfig(name="", device="invalid")
    except ValueError as e:
        print(f"Invalid config caught: {e}")
    
    print()

def demo_config_serialization():
    """Demonstrate configuration serialization"""
    print("=== Configuration Serialization ===")
    
    # Create configuration
    config = ModelConfig(
        name="serialization-test",
        cache_dir="/tmp/serial_cache",
        device="mps",
        trust_remote_code=True
    )
    
    # Convert to dictionary
    config_dict = config.to_dict()
    print(f"Config as dict: {config_dict}")
    
    # Create from dictionary
    restored_config = ModelConfig.from_dict(config_dict)
    print(f"Restored config: {restored_config}")
    print(f"Configs match: {config.to_dict() == restored_config.to_dict()}")
    print()

def demo_cache_directory():
    """Demonstrate cache directory creation"""
    print("=== Cache Directory Management ===")
    
    # Create a test cache directory
    test_cache_dir = "./demo_cache_test"
    
    print(f"Creating cache directory: {test_cache_dir}")
    success = create_cache_directory(test_cache_dir)
    print(f"Creation successful: {success}")
    
    if success and os.path.exists(test_cache_dir):
        print(f"Directory exists: {os.path.exists(test_cache_dir)}")
        # Clean up
        os.rmdir(test_cache_dir)
        print("Cleaned up test directory")
    
    print()

def main():
    """Run all configuration demonstrations"""
    print("ModelConfig System Demonstration")
    print("=" * 50)
    print()
    
    demo_basic_config()
    demo_environment_config()
    demo_environment_presets()
    demo_config_validation()
    demo_config_serialization()
    demo_cache_directory()
    
    print("Configuration system demonstration complete!")

if __name__ == "__main__":
    main()