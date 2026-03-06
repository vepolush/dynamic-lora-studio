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
