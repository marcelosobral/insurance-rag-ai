from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient


class TestAPI:
    """Test suite for FastAPI endpoints"""

    def test_health_endpoint(self, client):
        """Test health check endpoint"""
        response = client.get("/")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    @pytest.mark.unit
    def test_query_endpoint_success(self, client):
        """Test successful query endpoint"""
        mock_result = {
            "answer": "The policy covers bodily injury up to $1,000,000.",
            "sources": ["sample_policy.txt"],
            "confidence": 0.85,
            "cache_hit": False,
            "latency_seconds": 0.45,
        }

        with patch("app.main.generate_answer", return_value=mock_result):
            response = client.post(
                "/query", json={"question": "What does the policy cover?"}
            )

            assert response.status_code == 200
            data = response.json()

            assert data["answer"] == mock_result["answer"]
            assert data["sources"] == mock_result["sources"]
            assert data["confidence"] == mock_result["confidence"]
            assert data["cache_hit"] == mock_result["cache_hit"]
            assert data["latency_seconds"] == mock_result["latency_seconds"]

    @pytest.mark.unit
    def test_query_endpoint_missing_question(self, client):
        """Test query endpoint with missing question field"""
        response = client.post("/query", json={})

        assert response.status_code == 422  # Validation error

    @pytest.mark.unit
    def test_query_endpoint_invalid_json(self, client):
        """Test query endpoint with invalid JSON"""
        response = client.post(
            "/query",
            content="invalid json",
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code == 422

    @pytest.mark.unit
    def test_query_endpoint_empty_question(self, client):
        """Test query endpoint with empty question"""
        mock_result = {
            "answer": "Please provide a valid question.",
            "sources": [],
            "confidence": 0.0,
            "cache_hit": False,
            "latency_seconds": 0.01,
        }

        with patch("app.main.generate_answer", return_value=mock_result):
            response = client.post("/query", json={"question": ""})

            assert response.status_code == 200

    @pytest.mark.unit
    def test_startup_event(self):
        """Test startup event loads models"""
        with patch("app.embeddings.get_model") as mock_get_model:
            import asyncio

            from app.main import lifespan

            async def run_lifespan():
                async with lifespan(None):
                    pass

            asyncio.run(run_lifespan())
            mock_get_model.assert_called_once()

    @pytest.mark.integration
    def test_query_endpoint_integration(self, client):
        """Integration test for query endpoint"""
        # Mock the full pipeline
        mock_result = {
            "answer": "Integration test response",
            "sources": ["test_policy.txt"],
            "confidence": 0.75,
            "cache_hit": True,
            "latency_seconds": 0.12,
        }

        with patch("app.main.generate_answer", return_value=mock_result):
            response = client.post(
                "/query", json={"question": "What is the deductible amount?"}
            )

            assert response.status_code == 200
            data = response.json()

            # Verify all expected fields in response
            required_fields = [
                "answer",
                "sources",
                "confidence",
                "cache_hit",
                "latency_seconds",
            ]
            for field in required_fields:
                assert field in data

    @pytest.mark.unit
    def test_query_request_model(self):
        """Test QueryRequest pydantic model"""
        from app.main import QueryRequest

        # Valid request
        request = QueryRequest(question="What is covered?")
        assert request.question == "What is covered?"

        # Test validation
        with pytest.raises(ValueError):
            QueryRequest()  # Missing required field

    @pytest.mark.unit
    def test_cors_headers(self, client):
        """Test CORS headers are present (if configured)"""
        response = client.get("/")

        # Basic test - in production you might want to configure CORS
        assert response.status_code == 200
        # Could add CORS header checks if configured

    @pytest.mark.unit
    def test_query_endpoint_missing_index(self, client):
        """Test query endpoint when vector index is missing"""
        with patch(
            "app.main.generate_answer",
            side_effect=FileNotFoundError("Vector index not found"),
        ):
            response = client.post(
                "/query", json={"question": "What does the policy cover?"}
            )

            assert response.status_code == 503
            data = response.json()
            assert "detail" in data
