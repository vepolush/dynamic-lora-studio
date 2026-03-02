"""Generation service — request image generation from backend."""

from __future__ import annotations

from typing import Any

from api.client import APIClient, BackendError
from config import BACKEND_ENABLED


def generate(
    session_id: str,
    prompt: str,
    *,
    negative_prompt: str = "",
    steps: int | None = None,
    guidance_scale: float = 7.5,
    width: int = 512,
    height: int = 512,
    seed: int = -1,
    num_images: int = 1,
    scheduler: str | None = None,
    quality: str = "Normal",
    entity_id: str | None = None,
    entity_version: str | None = None,
    lora_strength: float = 0.8,
    style: str | None = None,
    lighting: str | None = None,
    color: str | None = None,
) -> dict[str, Any] | None:
    """Request image generation. Returns result dict or None on error."""
    if not BACKEND_ENABLED:
        return None

    try:
        client = APIClient()
        return client.generate(
            session_id=session_id,
            prompt=prompt,
            negative_prompt=negative_prompt,
            steps=steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            seed=seed,
            num_images=num_images,
            scheduler=scheduler,
            quality=quality,
            entity_id=entity_id,
            entity_version=entity_version,
            lora_strength=lora_strength,
            style=style if style and style != "None" else None,
            lighting=lighting if lighting and lighting != "None" else None,
            color=color if color and color != "Default" else None,
        )
    except BackendError:
        return None
