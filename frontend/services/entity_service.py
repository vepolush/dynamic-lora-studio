"""Entity service — fetch/upload/delete entities."""

from __future__ import annotations

import base64
from typing import Any

from api.client import APIClient, BackendError
from config import BACKEND_ENABLED
from state.session import MOCK_ENTITIES


def get_entity_preview_base64(entity_id: str) -> str | None:
    """Fetch entity preview image as base64 data URI. Returns None if not found."""
    if not BACKEND_ENABLED:
        return None
    try:
        client = APIClient()
        data = client.get_entity_preview_bytes(entity_id)
        if not data:
            return None
        return f"data:image/png;base64,{base64.b64encode(data).decode()}"
    except BackendError:
        return None


def get_entities() -> list[dict[str, Any]] | None:
    """Fetch entities from backend or return mock data. Returns None on error."""
    if not BACKEND_ENABLED:
        return MOCK_ENTITIES.copy()

    try:
        client = APIClient()
        return client.get_entities()
    except BackendError:
        return None


def upload_entity(
    name: str,
    trigger_word: str,
    zip_bytes: bytes,
    training_profile: str = "balanced",
    caption_mode: str = "auto",
    filename: str = "images.zip",
) -> dict[str, Any] | None:
    """Upload ZIP and train entity. Returns new entity on success, None on error."""
    if not BACKEND_ENABLED:
        return None

    try:
        client = APIClient()
        return client.upload_entity(
            name=name,
            trigger_word=trigger_word,
            zip_bytes=zip_bytes,
            training_profile=training_profile,
            caption_mode=caption_mode,
            filename=filename,
        )
    except BackendError as e:
        return {"status": "failed", "error": str(e)}


def delete_entity(entity_id: str) -> bool:
    """Delete entity. Returns True on success."""
    if not BACKEND_ENABLED:
        return False

    try:
        client = APIClient()
        client.delete_entity(entity_id)
        return True
    except BackendError:
        return False
