import pytest
import numpy as np
from unittest.mock import patch, Mock
from app.embeddings import get_model, embed_texts, embed_query


class TestEmbeddings:
    """Test suite for embeddings module"""

    @pytest.mark.unit
    def test_get_model_singleton(self):
        """Test that get_model returns the same instance"""
        with patch('app.embeddings.SentenceTransformer') as mock_st:
            mock_model = Mock()
            mock_st.return_value = mock_model
            
            model1 = get_model()
            model2 = get_model()
            
            assert model1 is model2
            mock_st.assert_called_once_with("all-MiniLM-L6-v2")

    @pytest.mark.unit
    def test_embed_texts(self):
        """Test embedding multiple texts"""
        texts = ["Sample insurance policy text", "Another policy clause"]
        
        with patch('app.embeddings.get_model') as mock_get_model:
            mock_model = Mock()
            mock_embeddings = np.random.rand(2, 384).astype('float32')
            mock_model.encode.return_value = mock_embeddings
            mock_get_model.return_value = mock_model
            
            result = embed_texts(texts)
            
            mock_model.encode.assert_called_once_with(texts, convert_to_numpy=True)
            assert isinstance(result, np.ndarray)
            assert result.shape == (2, 384)

    @pytest.mark.unit
    def test_embed_query(self):
        """Test embedding a single query"""
        query = "What is covered by this policy?"
        
        with patch('app.embeddings.get_model') as mock_get_model:
            mock_model = Mock()
            mock_embeddings = np.random.rand(1, 384).astype('float32')
            mock_model.encode.return_value = mock_embeddings
            mock_get_model.return_value = mock_model
            
            result = embed_query(query)
            
            mock_model.encode.assert_called_once_with([query], convert_to_numpy=True)
            assert isinstance(result, np.ndarray)
            assert result.shape == (384,)

    @pytest.mark.unit
    def test_embed_query_empty_string(self):
        """Test embedding empty string"""
        with patch('app.embeddings.get_model') as mock_get_model:
            mock_model = Mock()
            mock_embeddings = np.random.rand(1, 384).astype('float32')
            mock_model.encode.return_value = mock_embeddings
            mock_get_model.return_value = mock_model
            
            result = embed_query("")
            
            assert isinstance(result, np.ndarray)
            assert result.shape == (384,)

    @pytest.mark.integration
    @pytest.mark.slow
    def test_real_embeddings(self):
        """Integration test with real model (slow)"""
        # This test actually loads the model - mark as slow
        text = "Insurance policy covers liability up to $1,000,000"
        
        # Test real embedding
        result = embed_query(text)
        
        assert isinstance(result, np.ndarray)
        assert len(result.shape) == 1
        assert result.shape[0] > 0  # Should have some dimensions
        assert not np.allclose(result, 0)  # Should not be all zeros