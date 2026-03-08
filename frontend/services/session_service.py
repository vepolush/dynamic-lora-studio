"""Session service — fetch/create/update sessions from backend."""

from __future__ import annotations

import base64
from typing import Any

from api.client import BackendError
from config import BACKEND_ENABLED
from services.auth_service import get_client


def get_session_image_base64(session_id: str, filename: str) -> str | None:
    """Fetch session image as base64 data URI. Returns None if not found."""
    if not BACKEND_ENABLED:
        return None
    try:
        client = get_client()
        data = client.get_session_image_bytes(session_id, filename)
        if not data:
            return None
        return f"data:image/png;base64,{base64.b64encode(data).decode()}"
    except BackendError:
        return None


def get_sessions() -> list[dict[str, Any]]:
    """Fetch sessions from backend. Returns empty list when backend disabled or on error."""
    if not BACKEND_ENABLED:
        return []
    try:
        client = get_client()
        return client.get_sessions()
    except BackendError:
        return []


def get_session(session_id: str) -> dict[str, Any] | None:
    """Fetch full session with messages from backend."""
    if not BACKEND_ENABLED:
        return None
    try:
        client = get_client()
        return client.get_session(session_id)
    except BackendError:
        return None


def create_session(title: str = "New session") -> dict[str, Any] | None:
    """Create a new session via backend. Returns None when backend disabled."""
    if not BACKEND_ENABLED:
        return None
    try:
        client = get_client()
        return client.create_session(title=title)
    except BackendError:
        return None


def update_session(
    session_id: str,
    *,
    title: str | None = None,
    favourite: bool | None = None,
    favourite_image_filenames: list[str] | None = None,
    archived: bool | None = None,
) -> dict[str, Any] | None:
    """Update session metadata on backend."""
    if not BACKEND_ENABLED:
        return None
    try:
        client = get_client()
        return client.update_session(
            session_id,
            title=title,
            favourite=favourite,
            favourite_image_filenames=favourite_image_filenames,
            archived=archived,
        )
    except BackendError:
        return None


def delete_session(session_id: str) -> bool:
    """Delete session on backend. Returns True on success."""
    if not BACKEND_ENABLED:
        return False
    try:
        client = get_client()
        client.delete_session(session_id)
        return True
    except BackendError:
        return False
