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


def get_published_filenames(session_id: str) -> list[str]:
    """Get filenames from session that are already published to gallery."""
    if not BACKEND_ENABLED:
        return []
    try:
        client = get_client()
        return client.get_published_filenames(session_id)
    except BackendError:
        return []


def publish_to_gallery(
    session_id: str,
    filename: str,
    prompt: str,
    settings: dict | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    """Publish image to gallery. Requires auth. Returns (gallery_item, error_message)."""
    if not BACKEND_ENABLED:
        return None, "Backend not available"
    try:
        client = get_client()
        result = client.publish_to_gallery(session_id, filename, prompt, settings)
        return result, None
    except BackendError as e:
        import json
        msg = str(e)
        if "Backend error:" in msg:
            try:
                rest = msg.split("Backend error:", 1)[1].strip()
                data = json.loads(rest)
                return None, data.get("detail", msg)
            except (json.JSONDecodeError, IndexError, KeyError):
                pass
        return None, msg


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
) -> tuple[dict[str, Any] | None, str | None]:
    """Publish LoRA to gallery. Returns (gallery_lora, error_message)."""
    if not BACKEND_ENABLED:
        return None, "Backend not available"
    try:
        client = get_client()
        result = client.publish_lora_to_gallery(
            entity_id=entity_id,
            name=name,
            trigger_word=trigger_word,
            description=description,
        )
        return result, None
    except BackendError as e:
        import json
        msg = str(e)
        if "Backend error:" in msg:
            try:
                rest = msg.split("Backend error:", 1)[1].strip()
                data = json.loads(rest)
                return None, data.get("detail", msg)
            except (json.JSONDecodeError, IndexError, KeyError):
                pass
        return None, msg


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
