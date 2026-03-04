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


def update_entity(
    entity_id: str,
    *,
    name: str | None = None,
    trigger_word: str | None = None,
) -> dict[str, Any] | None:
    """Update entity name and/or trigger_word. Returns updated entity on success."""
    if not BACKEND_ENABLED:
        return None
    if name is None and trigger_word is None:
        return None

    try:
        client = APIClient()
        return client.update_entity(
            entity_id,
            name=name,
            trigger_word=trigger_word,
        )
    except BackendError:
        return None


def get_entity_dataset(entity_id: str) -> list[dict[str, Any]] | None:
    """Fetch dataset images for entity. Returns None on error."""
    if not BACKEND_ENABLED:
        return []
    try:
        client = APIClient()
        return client.get_entity_dataset(entity_id)
    except BackendError:
        return None


def get_dataset_image_base64(entity_id: str, filename: str) -> str | None:
    """Fetch dataset image as base64 data URI. Returns None if not found."""
    if not BACKEND_ENABLED:
        return None
    try:
        client = APIClient()
        data = client.get_dataset_image_bytes(entity_id, filename)
        if not data:
            return None
        return f"data:image/png;base64,{base64.b64encode(data).decode()}"
    except BackendError:
        return None


def remove_dataset_images(entity_id: str, filenames: list[str]) -> dict[str, Any] | None:
    """Remove images from entity dataset. Returns result dict or None on error."""
    if not BACKEND_ENABLED:
        return None
    try:
        client = APIClient()
        return client.remove_dataset_images(entity_id, filenames)
    except BackendError:
        return None


def retrain_entity(
    entity_id: str,
    *,
    zip_bytes: bytes | None = None,
    filename: str = "images.zip",
    remove_filenames: list[str] | None = None,
    training_profile: str = "balanced",
    caption_mode: str = "auto",
    use_custom: bool = False,
    steps: int = 1200,
    rank: int = 16,
    learning_rate: float = 1e-4,
    lr_scheduler: str = "polynomial",
    warmup_ratio: float = 0.06,
) -> dict[str, Any] | None:
    """Start retraining entity. Returns response with entity/job_id or None on error."""
    if not BACKEND_ENABLED:
        return None
    try:
        client = APIClient()
        return client.retrain_entity(
            entity_id,
            zip_bytes=zip_bytes,
            filename=filename,
            remove_filenames=remove_filenames,
            training_profile=training_profile,
            caption_mode=caption_mode,
            use_custom=use_custom,
            steps=steps,
            rank=rank,
            learning_rate=learning_rate,
            lr_scheduler=lr_scheduler,
            warmup_ratio=warmup_ratio,
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
