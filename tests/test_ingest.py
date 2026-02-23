import os
import tempfile
from unittest.mock import patch

import numpy as np
import pytest

from app.ingest import build_index, chunk_text, load_documents


class TestIngest:
    """Test suite for data ingestion module"""

    @pytest.mark.unit
    def test_chunk_text_basic(self):
        """Test basic text chunking functionality"""
        text = "A" * 1500  # 1500 character string
        chunks = chunk_text(text, chunk_size=500)

        assert len(chunks) == 3
        assert len(chunks[0]) == 500
        assert len(chunks[1]) == 500
        assert len(chunks[2]) == 500

    @pytest.mark.unit
    def test_chunk_text_short_text(self):
        """Test chunking text shorter than chunk size"""
        text = "Short text"
        chunks = chunk_text(text, chunk_size=500)

        assert len(chunks) == 1
        assert chunks[0] == "Short text"

    @pytest.mark.unit
    def test_chunk_text_exact_size(self):
        """Test chunking text that is exactly chunk size"""
        text = "A" * 500  # Exactly 500 characters
        chunks = chunk_text(text, chunk_size=500)

        assert len(chunks) == 1
        assert chunks[0] == text

    @pytest.mark.unit
    def test_chunk_text_custom_size(self):
        """Test chunking with custom chunk size"""
        text = "ABCDEFGHIJ"  # 10 characters
        chunks = chunk_text(text, chunk_size=3)

        assert len(chunks) == 4
        assert chunks[0] == "ABC"
        assert chunks[1] == "DEF"
        assert chunks[2] == "GHI"
        assert chunks[3] == "J"

    @pytest.mark.unit
    def test_load_documents_empty_directory(self):
        """Test loading documents from empty directory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("app.ingest.DATA_DIR", tmpdir):
                docs, metadata = load_documents()

                assert docs == []
                assert metadata == []

    @pytest.mark.unit
    def test_load_documents_with_files(self):
        """Test loading documents from directory with files"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            test_file1 = os.path.join(tmpdir, "policy1.txt")
            test_file2 = os.path.join(tmpdir, "policy2.txt")
            non_txt_file = os.path.join(tmpdir, "readme.md")

            with open(test_file1, "w") as f:
                f.write("Policy 1 content with liability coverage.")

            with open(test_file2, "w") as f:
                f.write("Policy 2 content with health benefits.")

            with open(non_txt_file, "w") as f:
                f.write("This should be ignored")

            with patch("app.ingest.DATA_DIR", tmpdir):
                docs, metadata = load_documents()

                # Should only process .txt files
                assert len(docs) == 2
                assert len(metadata) == 2

                # Check content (order-agnostic)
                expected_texts = [
                    "Policy 1 content with liability coverage.",
                    "Policy 2 content with health benefits.",
                ]
                for expected_text in expected_texts:
                    assert any(expected_text in doc for doc in docs)

                # Check metadata (order-agnostic)
                sources = {meta["source"] for meta in metadata}
                assert sources == {"policy1.txt", "policy2.txt"}
                for meta in metadata:
                    assert meta["text"] in docs

    @pytest.mark.unit
    def test_load_documents_with_chunking(self):
        """Test loading documents with text chunking"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a large test file that will be chunked
            test_file = os.path.join(tmpdir, "large_policy.txt")
            large_content = "A" * 1500  # 1500 characters, will create 3 chunks

            with open(test_file, "w") as f:
                f.write(large_content)

            with patch("app.ingest.DATA_DIR", tmpdir):
                docs, metadata = load_documents()

                # Should create 3 chunks from the large file
                assert len(docs) == 3
                assert len(metadata) == 3

                # All chunks should reference the same source
                for meta in metadata:
                    assert meta["source"] == "large_policy.txt"

    @pytest.mark.unit
    def test_build_index_success(self, temp_vector_store):
        """Test successful index building"""
        sample_docs = [
            "Insurance policy covers liability",
            "Health benefits include preventive care",
            "Deductible amount is $500",
        ]

        sample_metadata = [
            {"source": "policy1.txt", "text": sample_docs[0]},
            {"source": "policy2.txt", "text": sample_docs[1]},
            {"source": "policy3.txt", "text": sample_docs[2]},
        ]

        sample_embeddings = np.random.rand(3, 384).astype("float32")

        with patch(
            "app.ingest.load_documents", return_value=(sample_docs, sample_metadata)
        ), patch("app.ingest.embed_texts", return_value=sample_embeddings), patch(
            "faiss.write_index"
        ) as mock_write_index, patch(
            "numpy.save"
        ) as mock_save, patch(
            "os.makedirs"
        ) as mock_makedirs:

            build_index()

            # Verify index was created and saved
            mock_write_index.assert_called_once()
            mock_save.assert_called_once()
            mock_makedirs.assert_called_once_with("vector_store", exist_ok=True)

    @pytest.mark.unit
    def test_build_index_empty_documents(self, temp_vector_store):
        """Test building index with no documents"""
        with patch("app.ingest.load_documents", return_value=([], [])), patch(
            "app.ingest.embed_texts", return_value=np.array([]).reshape(0, 384)
        ), patch("faiss.write_index") as mock_write_index, patch(
            "numpy.save"
        ) as mock_save:

            build_index()

            # Should still create empty index
            mock_write_index.assert_called_once()
            mock_save.assert_called_once()

    @pytest.mark.integration
    def test_full_ingestion_pipeline(self):
        """Integration test for the full ingestion pipeline"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test data directory and vector store
            data_dir = os.path.join(tmpdir, "data")
            vector_store_dir = os.path.join(tmpdir, "vector_store")
            os.makedirs(data_dir)
            os.makedirs(vector_store_dir)

            # Create test policy file
            test_file = os.path.join(data_dir, "test_policy.txt")
            with open(test_file, "w") as f:
                f.write(
                    "This policy provides comprehensive coverage for "
                    "bodily injury and property damage."
                )

            index_path = os.path.join(vector_store_dir, "index.faiss")
            meta_path = os.path.join(vector_store_dir, "metadata.npy")

            with patch("app.ingest.DATA_DIR", data_dir), patch(
                "app.ingest.INDEX_PATH", index_path
            ), patch("app.ingest.META_PATH", meta_path):

                # Run the full ingestion
                build_index()

                # Verify files were created
                assert os.path.exists(index_path)
                assert os.path.exists(meta_path)

                # Verify metadata content
                metadata = np.load(meta_path, allow_pickle=True)
                assert len(metadata) == 1
                assert metadata[0]["source"] == "test_policy.txt"
                assert "comprehensive coverage" in metadata[0]["text"]

    @pytest.mark.unit
    def test_build_index_file_permissions(self, temp_vector_store):
        """Test index building with file permission errors"""
        sample_docs = ["Test document"]
        sample_metadata = [{"source": "test.txt", "text": "Test document"}]
        sample_embeddings = np.random.rand(1, 384).astype("float32")

        with patch(
            "app.ingest.load_documents", return_value=(sample_docs, sample_metadata)
        ), patch("app.ingest.embed_texts", return_value=sample_embeddings), patch(
            "faiss.write_index", side_effect=PermissionError("Access denied")
        ):

            with pytest.raises(PermissionError):
                build_index()
