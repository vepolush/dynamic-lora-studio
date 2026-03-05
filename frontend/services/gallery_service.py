"""Gallery service — list, publish, like."""

from __future__ import annotations

from typing import Any

from api.client import APIClient, BackendError
from config import BACKEND_ENABLED
from services.auth_service import get_client


def get_gallery(sort: str = "newest", limit: int = 50, offset: int = 0) -> list[dict[str, Any]] | None:
    """Fetch gallery images. Returns None on error."""
    if not BACKEND_ENABLED:
        return []
    try:
        client = get_client()
        return client.get_gallery(sort=sort, limit=limit, offset=offset)
    except BackendError:
        return None


def get_gallery_image(image_id: str) -> dict[str, Any] | None:
    """Fetch single gallery image."""
    if not BACKEND_ENABLED:
        return None
    try:
        client = get_client()
        return client.get_gallery_image(image_id)
    except BackendError:
        return None


def get_gallery_image_base64(image_id: str) -> str | None:
    """Fetch gallery image as base64 data URI."""
    if not BACKEND_ENABLED:
        return None
    try:
        import base64
        client = get_client()
        data = client.get_gallery_image_bytes(image_id)
        if not data:
            return None
        return f"data:image/png;base64,{base64.b64encode(data).decode()}"
    except BackendError:
        return None


def publish_to_gallery(
    session_id: str,
    filename: str,
    prompt: str,
    settings: dict | None = None,
) -> dict[str, Any] | None:
    """Publish image to gallery. Requires auth. Returns gallery item or None."""
    if not BACKEND_ENABLED:
        return None
    try:
        client = get_client()
        return client.publish_to_gallery(session_id, filename, prompt, settings)
    except BackendError:
        return None


def like_gallery_image(image_id: str) -> dict[str, Any] | None:
    """Toggle like on gallery image. Requires auth."""
    if not BACKEND_ENABLED:
        return None
    try:
        client = get_client()
        return client.like_gallery_image(image_id)
    except BackendError:
        return None


# ---- Gallery LoRAs ----

def get_gallery_loras(
    sort: str = "newest",
    limit: int = 50,
    offset: int = 0,
) -> list[dict[str, Any]] | None:
    """Fetch gallery LoRAs. Returns None on error."""
    if not BACKEND_ENABLED:
        return []
    try:
        client = get_client()
        return client.get_gallery_loras(sort=sort, limit=limit, offset=offset)
    except BackendError:
        return None


def get_gallery_lora(lora_id: str) -> dict[str, Any] | None:
    """Fetch single gallery LoRA."""
    if not BACKEND_ENABLED:
        return None
    try:
        client = get_client()
        return client.get_gallery_lora(lora_id)
    except BackendError:
        return None


def get_gallery_lora_preview_base64(lora_id: str) -> str | None:
    """Fetch gallery LoRA preview as base64 data URI."""
    if not BACKEND_ENABLED:
        return None
    try:
        import base64
        client = get_client()
        data = client.get_gallery_lora_preview_bytes(lora_id)
        if not data:
            return None
        return f"data:image/png;base64,{base64.b64encode(data).decode()}"
    except BackendError:
        return None


def publish_lora_to_gallery(
    entity_id: str,
    name: str,
    trigger_word: str,
    description: str | None = None,
) -> dict[str, Any] | None:
    """Publish LoRA to gallery. Requires auth. Returns gallery LoRA or None."""
    if not BACKEND_ENABLED:
        return None
    try:
        client = get_client()
        return client.publish_lora_to_gallery(
            entity_id=entity_id,
            name=name,
            trigger_word=trigger_word,
            description=description,
        )
    except BackendError:
        return None


def add_gallery_lora(lora_id: str) -> dict[str, Any] | None:
    """Add (copy) gallery LoRA to user's entities. Requires auth."""
    if not BACKEND_ENABLED:
        return None
    try:
        client = get_client()
        return client.add_gallery_lora(lora_id)
    except BackendError:
        return None


def unpublish_gallery_lora(lora_id: str) -> bool:
    """Unpublish LoRA. Only creator can unpublish. Requires auth."""
    if not BACKEND_ENABLED:
        return False
    try:
        client = get_client()
        client.unpublish_gallery_lora(lora_id)
        return True
    except BackendError:
        return False
