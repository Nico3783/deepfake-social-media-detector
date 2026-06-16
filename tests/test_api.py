"""Tests for FastAPI inference service."""

from __future__ import annotations

import io
import pytest
from fastapi.testclient import TestClient
from PIL import Image
import numpy as np

from src.api.app import create_app


@pytest.fixture
def client() -> TestClient:
    """Create a test client without a loaded model."""
    app = create_app()
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for GET /api/v1/health."""

    def test_health_returns_200(self, client: TestClient) -> None:
        """Health check returns 200 OK."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200

    def test_health_response_structure(self, client: TestClient) -> None:
        """Health response contains required fields."""
        data = client.get("/api/v1/health").json()
        assert "status" in data
        assert "model_loaded" in data
        assert "device" in data

    def test_health_model_not_loaded(self, client: TestClient) -> None:
        """Health check reports model_loaded=False when no model."""
        data = client.get("/api/v1/health").json()
        assert data["model_loaded"] is False


class TestImagePredictionEndpoint:
    """Tests for POST /api/v1/predict/image."""

    def test_no_model_returns_503(self, client: TestClient) -> None:
        """Image prediction returns 503 when no model loaded."""
        img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)

        response = client.post(
            "/api/v1/predict/image",
            files={"file": ("test.jpg", buf, "image/jpeg")},
        )
        assert response.status_code == 503

    def test_non_image_returns_400(self, client: TestClient) -> None:
        """Non-image file returns 400 Bad Request."""
        response = client.post(
            "/api/v1/predict/image",
            files={"file": ("test.txt", b"not an image", "text/plain")},
        )
        assert response.status_code == 400


class TestVideoPredictionEndpoint:
    """Tests for POST /api/v1/predict/video."""

    def test_no_model_returns_503(self, client: TestClient) -> None:
        """Video prediction returns 503 when no model loaded."""
        response = client.post(
            "/api/v1/predict/video",
            files={"file": ("test.mp4", b"fake video", "video/mp4")},
        )
        assert response.status_code == 503

    def test_non_video_returns_400(self, client: TestClient) -> None:
        """Non-video file returns 400 Bad Request."""
        response = client.post(
            "/api/v1/predict/video",
            files={"file": ("test.txt", b"not a video", "text/plain")},
        )
        assert response.status_code == 400


class TestOpenAPIDocs:
    """Tests for API documentation."""

    def test_docs_endpoint(self, client: TestClient) -> None:
        """OpenAPI docs are accessible."""
        response = client.get("/docs")
        assert response.status_code == 200

    def test_openapi_schema(self, client: TestClient) -> None:
        """OpenAPI schema is generated."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert "paths" in schema
        assert "/api/v1/health" in schema["paths"]
