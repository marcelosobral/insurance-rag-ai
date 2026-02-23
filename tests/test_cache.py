import json
import os
from unittest.mock import Mock, mock_open, patch

import numpy as np
import pytest

from app.cache import (add_to_cache, check_cache, cosine_similarity,
                       load_cache, save_cache)


class TestCache:
    """Test suite for cache module"""

    @pytest.mark.unit
    def test_cosine_similarity(self):
        """Test cosine similarity calculation"""
        a = np.array([1, 0, 0])
        b = np.array([1, 0, 0])

        similarity = cosine_similarity(a, b)
        assert abs(similarity - 1.0) < 1e-6

        a = np.array([1, 0, 0])
        b = np.array([0, 1, 0])

        similarity = cosine_similarity(a, b)
        assert abs(similarity - 0.0) < 1e-6

    @pytest.mark.unit
    def test_load_cache_file_not_exists(self):
        """Test loading cache when file doesn't exist"""
        with patch("os.path.exists", return_value=False):
            cache = load_cache()
            assert cache == []

    @pytest.mark.unit
    def test_load_cache_file_exists(self):
        """Test loading cache when file exists"""
        test_cache = [
            {
                "query": "test query",
                "embedding": [0.1, 0.2, 0.3],
                "response": {"answer": "test answer"},
            }
        ]

        with patch("os.path.exists", return_value=True), patch(
            "builtins.open", mock_open(read_data=json.dumps(test_cache))
        ):
            cache = load_cache()
            assert cache == test_cache

    @pytest.mark.unit
    def test_save_cache(self):
        """Test saving cache to file"""
        test_cache = [{"query": "test", "response": "answer"}]

        mock_file = mock_open()
        with patch("builtins.open", mock_file):
            save_cache(test_cache)

        mock_file.assert_called_once()
        handle = mock_file()
        written_data = "".join(call.args[0] for call in handle.write.call_args_list)
        assert json.loads(written_data) == test_cache

    @pytest.mark.unit
    def test_check_cache_hit(self, temp_vector_store):
        """Test cache hit scenario"""
        query = "What is covered?"
        query_embedding = np.array([0.1, 0.2, 0.3])

        test_cache = [
            {
                "query": "Similar query",
                "embedding": [0.1, 0.2, 0.3],  # Same embedding = high similarity
                "response": {"answer": "Cached answer", "sources": ["policy.txt"]},
            }
        ]

        with patch("app.cache.load_cache", return_value=test_cache), patch(
            "app.cache.embed_query", return_value=query_embedding
        ):

            result, hit = check_cache(query, threshold=0.9)

            assert hit is True
            assert result == {"answer": "Cached answer", "sources": ["policy.txt"]}

    @pytest.mark.unit
    def test_check_cache_miss(self, temp_vector_store):
        """Test cache miss scenario"""
        query = "What is covered?"
        query_embedding = np.array([1.0, 0.0, 0.0])

        test_cache = [
            {
                "query": "Different query",
                "embedding": [0.0, 1.0, 0.0],  # Different embedding = low similarity
                "response": {"answer": "Cached answer"},
            }
        ]

        with patch("app.cache.load_cache", return_value=test_cache), patch(
            "app.cache.embed_query", return_value=query_embedding
        ):

            result, hit = check_cache(query, threshold=0.9)

            assert hit is False
            assert result is None

    @pytest.mark.unit
    def test_add_to_cache(self, temp_vector_store):
        """Test adding new item to cache"""
        query = "New query"
        response = {"answer": "New answer", "sources": ["new_policy.txt"]}
        embedding = [0.5, 0.6, 0.7]

        existing_cache = [{"query": "old", "embedding": [0.1, 0.2], "response": {}}]

        with patch("app.cache.load_cache", return_value=existing_cache), patch(
            "app.cache.embed_query", return_value=np.array(embedding)
        ), patch("app.cache.save_cache") as mock_save:

            add_to_cache(query, response)

            # Verify save_cache was called with updated cache
            mock_save.assert_called_once()
            saved_cache = mock_save.call_args[0][0]

            assert len(saved_cache) == 2
            assert saved_cache[1]["query"] == query
            assert saved_cache[1]["response"] == response
            assert saved_cache[1]["embedding"] == embedding

    @pytest.mark.integration
    def test_cache_integration(self, temp_vector_store):
        """Integration test for full cache workflow"""
        with patch("app.cache.embed_query") as mock_embed:
            # Test embedding returns
            mock_embed.side_effect = [
                np.array([1.0, 0.0, 0.0]),  # First check_cache
                np.array([1.0, 0.0, 0.0]),  # add_to_cache
                np.array([1.0, 0.0, 0.0]),  # Second check_cache
            ]

            query1 = "What is the coverage limit?"
            response1 = {"answer": "Coverage is $1M", "sources": ["policy.txt"]}

            # First call should miss and add to cache
            result, hit = check_cache(query1)
            assert hit is False

            add_to_cache(query1, response1)

            # Second call should hit cache
            result, hit = check_cache(query1)
            assert hit is True
            assert result == response1
