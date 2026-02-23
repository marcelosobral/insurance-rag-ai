import os
import pytest
import tempfile
import json
import numpy as np
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient

# Test configuration
TEST_OPENAI_API_KEY = "test-key-12345"
TEST_MODEL = "gpt-4o-mini"

@pytest.fixture(scope="session")
def test_env():
    """Set up test environment variables"""
    os.environ["OPENAI_API_KEY"] = TEST_OPENAI_API_KEY
    os.environ["LLM_MODEL"] = TEST_MODEL
    yield
    # Cleanup
    if "OPENAI_API_KEY" in os.environ:
        del os.environ["OPENAI_API_KEY"]
    if "LLM_MODEL" in os.environ:
        del os.environ["LLM_MODEL"]

@pytest.fixture(autouse=True)
def reset_embeddings_model():
    """Reset global embeddings model between tests to prevent mock leakage."""
    from app import embeddings
    embeddings._model = None
    yield
    embeddings._model = None

@pytest.fixture
def temp_vector_store():
    """Create temporary vector store directory"""
    with tempfile.TemporaryDirectory() as tmpdir:
        vector_store_path = os.path.join(tmpdir, "vector_store")
        os.makedirs(vector_store_path)
        
        # Mock the paths
        with patch('app.rag.INDEX_PATH', os.path.join(vector_store_path, "index.faiss")), \
             patch('app.rag.META_PATH', os.path.join(vector_store_path, "metadata.npy")), \
             patch('app.cache.CACHE_PATH', os.path.join(vector_store_path, "cache.json")):
            yield vector_store_path

@pytest.fixture
def sample_documents():
    """Sample test documents"""
    return [
        {
            "text": "This policy covers bodily injury up to $500,000 per occurrence.",
            "source": "test_policy_1.txt"
        },
        {
            "text": "Exclusions include intentional damage and criminal acts.",
            "source": "test_policy_1.txt"
        },
        {
            "text": "Health insurance covers preventive care at 100% with no deductible.",
            "source": "health_policy.txt"
        }
    ]

@pytest.fixture
def mock_openai_response():
    """Mock OpenAI API response"""
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = "Based on the policy, coverage is $500,000 per occurrence."
    return mock_response

@pytest.fixture
def client(test_env):
    """FastAPI test client"""
    from app.main import app
    return TestClient(app)

@pytest.fixture
def sample_embeddings():
    """Sample embeddings for testing"""
    return np.random.rand(3, 384).astype('float32')  # 384 is the dimension for all-MiniLM-L6-v2