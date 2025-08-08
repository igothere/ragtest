# Design Document

## Overview

This design document outlines a solution to optimize SentenceTransformer model loading in the RAG system by implementing a centralized model manager that loads the model once and shares it across multiple processes. The solution addresses memory inefficiency and startup time issues caused by duplicate model loading in api_server.py and rag_with_tables.py.

## Architecture

### Current Architecture Issues
- **Duplicate Loading**: SentenceTransformer("nlpai-lab/KURE-v1") is loaded independently in both api_server.py and rag_with_tables.py
- **Memory Waste**: Each process maintains its own copy of the ~1GB model in memory
- **Startup Delay**: Each script experiences model loading time (5-10 seconds)
- **Maintenance Overhead**: Model configuration scattered across multiple files

### Proposed Architecture

The solution implements a **Singleton Model Manager** pattern with the following components:

1. **ModelManager Class**: Centralized singleton that handles model lifecycle
2. **Shared Model Instance**: Single in-memory model instance accessible across processes
3. **Configuration Centralization**: Single point for model configuration
4. **Error Handling & Fallback**: Graceful degradation when shared model fails

## Components and Interfaces

### 1. ModelManager Class (`model_manager.py`)

```python
class ModelManager:
    """Singleton class for managing shared SentenceTransformer model"""
    
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

### 2. Integration Points

#### API Server Integration
- Replace direct model loading with `ModelManager.get_model()`
- Maintain existing chat endpoint functionality
- Add model status endpoint for monitoring

#### RAG Processing Integration  
- Replace direct model loading in `rag_with_tables.py`
- Maintain existing document processing workflow
- Add model validation before processing

### 3. Configuration Management

#### Environment Variables
```bash
MODEL_NAME=nlpai-lab/KURE-v1
MODEL_CACHE_DIR=./model_cache
MODEL_DEVICE=auto  # auto, cpu, cuda
```

#### Configuration File (`config/model_config.py`)
```python
MODEL_CONFIG = {
    "name": "nlpai-lab/KURE-v1",
    "cache_dir": "./model_cache",
    "device": "auto",
    "trust_remote_code": False
}
```

## Data Models

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

## Error Handling

### Error Scenarios & Responses

1. **Model Loading Failure**
   - Log detailed error information
   - Return appropriate HTTP status codes
   - Provide fallback to individual model loading if needed

2. **Memory Exhaustion**
   - Monitor memory usage during model loading
   - Implement graceful degradation
   - Clear model cache if necessary

3. **Concurrent Access Issues**
   - Thread-safe model access using locks
   - Queue requests during model loading
   - Timeout handling for long-running operations

### Fallback Strategy
```python
def get_model_with_fallback():
    try:
        return ModelManager.get_model()
    except Exception as e:
        logger.warning(f"Shared model failed: {e}, falling back to individual loading")
        return SentenceTransformer("nlpai-lab/KURE-v1")
```

## Testing Strategy

### Unit Tests
1. **ModelManager Tests**
   - Singleton pattern validation
   - Model loading success/failure scenarios
   - Thread safety verification
   - Memory leak detection

2. **Integration Tests**
   - API server with shared model
   - Document processing with shared model
   - Concurrent access testing
   - Fallback mechanism validation

### Performance Tests
1. **Memory Usage Comparison**
   - Before: 2x model memory usage
   - After: 1x model memory usage
   - Memory monitoring during concurrent operations

2. **Startup Time Measurement**
   - Initial model loading time
   - Subsequent access time (should be ~0ms)
   - Overall system startup improvement

3. **Throughput Testing**
   - Document processing speed
   - Chat response time
   - Concurrent request handling

### Load Testing
1. **Concurrent Document Processing**
   - Multiple files processed simultaneously
   - Model access under high load
   - Resource utilization monitoring

2. **API Stress Testing**
   - Multiple chat requests
   - Model sharing stability
   - Response time consistency

## Implementation Phases

### Phase 1: Core Model Manager
- Create ModelManager singleton class
- Implement basic model loading and sharing
- Add configuration management
- Unit tests for core functionality

### Phase 2: API Server Integration
- Modify api_server.py to use ModelManager
- Update chat endpoint implementation
- Add model status monitoring endpoint
- Integration tests for API functionality

### Phase 3: Document Processing Integration
- Modify rag_with_tables.py to use ModelManager
- Update subprocess model access pattern
- Ensure backward compatibility
- Integration tests for document processing

### Phase 4: Error Handling & Monitoring
- Implement comprehensive error handling
- Add fallback mechanisms
- Performance monitoring and logging
- Load testing and optimization

## Migration Strategy

### Backward Compatibility
- Maintain existing API contracts
- Preserve current functionality
- Gradual rollout with feature flags
- Easy rollback mechanism

### Deployment Steps
1. Deploy ModelManager without integration
2. Update api_server.py with feature flag
3. Update rag_with_tables.py with feature flag
4. Enable shared model by default
5. Remove individual model loading code

## Monitoring and Observability

### Metrics to Track
- Model loading time
- Memory usage reduction
- Request processing time
- Error rates and types
- Concurrent access patterns

### Logging Strategy
- Model lifecycle events
- Performance metrics
- Error conditions
- Resource utilization

### Health Checks
- Model availability endpoint
- Memory usage monitoring
- Performance degradation alerts