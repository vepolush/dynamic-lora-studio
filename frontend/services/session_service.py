"""Session service — fetch/create sessions from backend or mock."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from api.client import APIClient, BackendError
from config import BACKEND_ENABLED
from state.session import MOCK_SESSIONS


def _make_local_session(title: str) -> dict[str, Any]:
    """Create session dict for local/mock mode."""
    import uuid
    return {
        "id": f"sess_{uuid.uuid4().hex[:12]}",
        "title": title,
        "created_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S"),
        "prompt": "",
        "helper_specs": "",
        "images": 0,
        "favourite": False,
    }


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
    """Create a new session via backend or locally when backend disabled."""
    if not BACKEND_ENABLED:
        return _make_local_session(title)

    try:
        client = APIClient()
        return client.create_session(title=title)
    except BackendError:
        return None
