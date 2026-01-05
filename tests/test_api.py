"""Tests for the FastAPI application."""

import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient

# Add packages to path
sys.path.insert(0, str(Path(__file__).parent.parent / "packages"))


class TestHealthEndpoint:
    """Tests for the health endpoint."""

    def test_health_without_model(self):
        """Test health endpoint when model is not loaded."""
        with patch("services.api.main.model_loaded", False):
            from services.api.main import app

            client = TestClient(app)
            response = client.get("/health")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert data["model_loaded"] is False


class TestPredictEndpoint:
    """Tests for the predict endpoint."""

    def test_predict_without_model(self):
        """Test predict endpoint returns 503 when model not loaded."""
        with patch("services.api.main.model_loaded", False):
            with patch("services.api.main.classifier_instance", None):
                from services.api.main import app

                client = TestClient(app)
                response = client.post(
                    "/predict",
                    json={"text": "Test text"},
                )

                assert response.status_code == 503

    def test_predict_with_mock_model(self):
        """Test predict endpoint with mocked model."""
        mock_classifier = Mock()
        mock_classifier.predict.return_value = {
            "label": "Innovacion",
            "confidence": 0.95,
        }

        with patch("services.api.main.model_loaded", True):
            with patch("services.api.main.classifier_instance", mock_classifier):
                from services.api.main import app

                client = TestClient(app)
                response = client.post(
                    "/predict",
                    json={"text": "La empresa lanzó una nueva tecnología."},
                )

                assert response.status_code == 200
                data = response.json()
                assert data["label"] == "Innovacion"
                assert data["confidence"] == 0.95

    def test_predict_empty_text(self):
        """Test predict endpoint rejects empty text."""
        with patch("services.api.main.model_loaded", True):
            with patch("services.api.main.classifier_instance", Mock()):
                from services.api.main import app

                client = TestClient(app)
                response = client.post(
                    "/predict",
                    json={"text": ""},
                )

                assert response.status_code == 422


class TestBatchPredictEndpoint:
    """Tests for the batch predict endpoint."""

    def test_batch_predict_with_mock_model(self):
        """Test batch predict endpoint with mocked model."""
        mock_classifier = Mock()
        mock_classifier.predict_batch.return_value = [
            {"label": "Innovacion", "confidence": 0.95},
            {"label": "Alianzas", "confidence": 0.88},
        ]

        with patch("services.api.main.model_loaded", True):
            with patch("services.api.main.classifier_instance", mock_classifier):
                from services.api.main import app

                client = TestClient(app)
                response = client.post(
                    "/predict/batch",
                    json={"texts": ["Texto 1", "Texto 2"]},
                )

                assert response.status_code == 200
                data = response.json()
                assert len(data["predictions"]) == 2
