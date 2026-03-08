"""API tests. Run with pytest -k 'not slow' to skip model-dependent tests."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))


@pytest.fixture
def api_client():
    """FastAPI test client. Model loading may take time on first request."""
    import os
    test_data = Path(__file__).parent / "test_data"
    test_data.mkdir(exist_ok=True)
    os.environ["DATA_DIR"] = str(test_data)
    from main import app
    from fastapi.testclient import TestClient
    return TestClient(app, timeout=10.0)


class TestHealth:
    """Health endpoint."""

    def test_health_returns_200_or_503(self, api_client) -> None:
        r = api_client.get("/health")
        assert r.status_code in (200, 503)
        data = r.json()
        assert "status" in data
        assert data["status"] in ("ready", "loading")


class TestAuth:
    """Auth endpoints."""

    def test_register(self, api_client) -> None:
        r = api_client.post(
            "/api/auth/register",
            json={"username": "testuser123", "password": "password123"},
        )
        assert r.status_code == 200
        data = r.json()
        assert "token" in data
        assert data.get("username") == "testuser123"

    def test_register_duplicate_fails(self, api_client) -> None:
        api_client.post(
            "/api/auth/register",
            json={"username": "dupuser", "password": "password123"},
        )
        r = api_client.post(
            "/api/auth/register",
            json={"username": "dupuser", "password": "otherpass"},
        )
        assert r.status_code == 400

    def test_login(self, api_client) -> None:
        api_client.post(
            "/api/auth/register",
            json={"username": "loginuser", "password": "mypass"},
        )
        r = api_client.post(
            "/api/auth/login",
            json={"username": "loginuser", "password": "mypass"},
        )
        assert r.status_code == 200
        assert "token" in r.json()


class TestGeneration:
    """Generation API endpoints."""

    @pytest.fixture
    def auth_headers(self, api_client) -> dict[str, str]:
        """Register, login, return Authorization headers."""
        api_client.post(
            "/api/auth/register",
            json={"username": "genuser", "password": "genpass123"},
        )
        r = api_client.post(
            "/api/auth/login",
            json={"username": "genuser", "password": "genpass123"},
        )
        token = r.json()["token"]
        return {"Authorization": f"Bearer {token}"}

    @pytest.fixture
    def session_id(self, api_client, auth_headers) -> str:
        """Create a session and return its ID."""
        r = api_client.post(
            "/api/sessions",
            json={"title": "Test session"},
            headers=auth_headers,
        )
        assert r.status_code == 200
        return r.json()["id"]

    def test_generate_without_auth_returns_401_or_404(self, api_client) -> None:
        """Generate without token: session lookup fails (404) or auth required (401)."""
        r = api_client.post(
            "/api/generate",
            json={
                "session_id": "nonexistent-session",
                "prompt": "a red apple",
            },
        )
        assert r.status_code in (401, 404)

    def test_generate_invalid_session_returns_404(
        self, api_client, auth_headers
    ) -> None:
        """Generate with valid auth but nonexistent session_id returns 404."""
        r = api_client.post(
            "/api/generate",
            json={
                "session_id": "nonexistent-session-id-12345",
                "prompt": "a red apple",
            },
            headers=auth_headers,
        )
        assert r.status_code == 404

    def test_generate_valid_request_returns_200_or_503(
        self, api_client, auth_headers, session_id
    ) -> None:
        """Generate with valid auth and session: 200 (success) or 503 (model loading)."""
        r = api_client.post(
            "/api/generate",
            json={
                "session_id": session_id,
                "prompt": "a red apple on white background",
                "width": 512,
                "height": 512,
                "num_images": 1,
            },
            headers=auth_headers,
            timeout=120.0,
        )
        assert r.status_code in (200, 503)
        data = r.json()
        if r.status_code == 200:
            assert data.get("status") == "success"
            assert "images" in data
            assert "message" in data
        else:
            assert "detail" in data

    def test_generate_accepts_optional_params(
        self, api_client, auth_headers, session_id
    ) -> None:
        """Generate accepts steps, guidance_scale, seed, style, lighting, etc."""
        r = api_client.post(
            "/api/generate",
            json={
                "session_id": session_id,
                "prompt": "a cat",
                "steps": 25,
                "guidance_scale": 7.5,
                "seed": 42,
                "style": "Photographic",
                "lighting": "Studio",
                "quality": "High",
            },
            headers=auth_headers,
            timeout=120.0,
        )
        assert r.status_code in (200, 503)
