"""Session service — fetch/create sessions from backend or mock."""

from __future__ import annotations

from typing import Any

from api.client import APIClient, BackendError
from config import BACKEND_ENABLED
from state.session import MOCK_SESSIONS


def get_sessions() -> list[dict[str, Any]]:
    """Fetch sessions from backend or return mock data."""
    if not BACKEND_ENABLED:
        return MOCK_SESSIONS.copy()

    try:
        client = APIClient()
        return client.get_sessions()
    except BackendError:
        return MOCK_SESSIONS.copy()


def create_session(title: str = "New session") -> dict[str, Any] | None:
    """Create a new session via backend. Returns None if backend unavailable."""
    if not BACKEND_ENABLED:
        return None

    try:
        client = APIClient()
        return client.create_session(title=title)
    except BackendError:
        return None
