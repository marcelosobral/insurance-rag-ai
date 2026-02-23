from unittest.mock import Mock, patch

import faiss
import numpy as np
import pytest

from app.rag import generate_answer, load_index, retrieve


class TestRAG:
    """Test suite for RAG module"""

    @pytest.fixture
    def mock_index_and_metadata(self, sample_embeddings, sample_documents):
        """Create mock FAISS index and metadata"""
        # Create a real FAISS index for testing
        dimension = sample_embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(sample_embeddings)

        metadata = np.array(
            [
                {"source": doc["source"], "text": doc["text"]}
                for doc in sample_documents
            ],
            dtype=object,
        )

        return index, metadata

    @pytest.mark.unit
    def test_load_index(self, temp_vector_store, mock_index_and_metadata):
        """Test loading FAISS index and metadata"""
        mock_index, mock_metadata = mock_index_and_metadata

        with patch("os.path.exists", return_value=True), patch(
            "faiss.read_index", return_value=mock_index
        ), patch("numpy.load", return_value=mock_metadata):

            index, metadata = load_index()

            assert index is mock_index
            assert np.array_equal(metadata, mock_metadata)

    @pytest.mark.unit
    def test_load_index_missing_files(self):
        """Test load_index raises when files are missing"""
        with patch("os.path.exists", return_value=False):
            with pytest.raises(FileNotFoundError):
                load_index()

    @pytest.mark.unit
    def test_retrieve_success(
        self, temp_vector_store, mock_index_and_metadata, sample_embeddings
    ):
        """Test successful retrieval"""
        mock_index, mock_metadata = mock_index_and_metadata
        query_embedding = sample_embeddings[0]  # Use first embedding as query

        with patch(
            "app.rag.load_index", return_value=(mock_index, mock_metadata)
        ), patch("app.rag.embed_query", return_value=query_embedding):

            result = retrieve("test query", top_k=2)

            assert "results" in result
            assert "avg_score" in result
            assert len(result["results"]) >= 1  # Should find at least one match

            for r in result["results"]:
                assert "source" in r
                assert "text" in r
                assert "score" in r

    @pytest.mark.unit
    def test_retrieve_no_results(self, temp_vector_store):
        """Test retrieval when no good matches found"""
        # Create empty index
        dimension = 384
        empty_index = faiss.IndexFlatL2(dimension)
        empty_metadata = np.array([], dtype=object)

        with patch(
            "app.rag.load_index", return_value=(empty_index, empty_metadata)
        ), patch(
            "app.rag.embed_query",
            return_value=np.random.rand(dimension).astype("float32"),
        ):

            result = retrieve("test query")

            assert result["results"] == []
            assert result["avg_score"] == 0.0

    @pytest.mark.unit
    def test_generate_answer_cache_hit(self, temp_vector_store, mock_openai_response):
        """Test generate_answer with cache hit"""
        cached_response = {
            "answer": "Cached answer",
            "sources": ["cached_policy.txt"],
            "confidence": 0.95,
        }

        with patch("app.rag.check_cache", return_value=(cached_response, True)):
            result = generate_answer("test query")

            assert result["answer"] == "Cached answer"
            assert result["cache_hit"] is True
            assert "latency_seconds" in result

    @pytest.mark.unit
    def test_generate_answer_cache_miss(
        self, temp_vector_store, mock_openai_response, sample_documents
    ):
        """Test generate_answer with cache miss"""
        # Mock retrieval results
        retrieval_result = {
            "results": [
                {"text": doc["text"], "source": doc["source"], "score": 0.1}
                for doc in sample_documents
            ],
            "avg_score": 0.1,
        }

        with patch("app.rag.check_cache", return_value=(None, False)), patch(
            "app.rag.retrieve", return_value=retrieval_result
        ), patch(
            "app.rag.client.chat.completions.create", return_value=mock_openai_response
        ), patch(
            "app.rag.add_to_cache"
        ) as mock_add_cache:

            result = generate_answer("test query")

            assert result["cache_hit"] is False
            assert "answer" in result
            assert "sources" in result
            assert "confidence" in result
            assert "latency_seconds" in result

            # Verify cache was updated
            mock_add_cache.assert_called_once()

    @pytest.mark.unit
    def test_confidence_calculation(self, temp_vector_store):
        """Test confidence score calculation"""
        # Test with different average scores
        test_cases = [
            (0.0, 1.0),  # Perfect match -> high confidence
            (1.0, 0.5),  # Average match -> medium confidence
            (10.0, 0.09),  # Poor match -> low confidence
        ]

        for avg_score, expected_min_conf in test_cases:
            retrieval_result = {
                "results": [{"text": "test", "source": "test.txt", "score": avg_score}],
                "avg_score": avg_score,
            }

            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = "Test answer"

            with patch("app.rag.check_cache", return_value=(None, False)), patch(
                "app.rag.retrieve", return_value=retrieval_result
            ), patch(
                "app.rag.client.chat.completions.create", return_value=mock_response
            ), patch(
                "app.rag.add_to_cache"
            ):

                result = generate_answer("test query")
                confidence = result["confidence"]

                assert 0.0 <= confidence <= 1.0
                if avg_score == 0.0:
                    assert confidence == 1.0  # Perfect score
                else:
                    assert confidence >= expected_min_conf

    @pytest.mark.integration
    def test_full_rag_pipeline(
        self, temp_vector_store, sample_embeddings, sample_documents
    ):
        """Integration test for the full RAG pipeline"""
        # Create real index with sample data
        dimension = sample_embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(sample_embeddings)

        metadata = np.array(
            [
                {"source": doc["source"], "text": doc["text"]}
                for doc in sample_documents
            ],
            dtype=object,
        )

        # Mock OpenAI response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = (
            "Based on the policy, coverage includes bodily injury protection."
        )

        with patch("app.rag.load_index", return_value=(index, metadata)), patch(
            "app.rag.embed_query", return_value=sample_embeddings[0]
        ), patch("app.rag.check_cache", return_value=(None, False)), patch(
            "app.rag.client.chat.completions.create", return_value=mock_response
        ), patch(
            "app.rag.add_to_cache"
        ):

            result = generate_answer("What does the policy cover?")

            # Verify complete response structure
            assert "answer" in result
            assert "sources" in result
            assert "confidence" in result
            assert "cache_hit" in result
            assert "latency_seconds" in result

            assert result["cache_hit"] is False
            assert isinstance(result["sources"], list)
            assert len(result["sources"]) > 0
            assert 0.0 <= result["confidence"] <= 1.0
            assert result["latency_seconds"] >= 0
