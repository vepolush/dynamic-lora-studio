"""App configuration — backend URL, feature flags."""

from __future__ import annotations

import os
from pathlib import Path

CONFIG_DIR: Path = Path.home() / ".config" / "dynamic-lora-studio"
CONFIG_FILE: Path = CONFIG_DIR / "settings.json"


def _load_effective_config() -> dict:
    """Load config from file; env overrides for deployment."""
    try:
        from services.config_service import load_config
        cfg = load_config()
    except Exception:
        cfg = {"backend_url": "http://localhost:8000", "backend_enabled": True}
    url = os.getenv("BACKEND_URL", cfg.get("backend_url", "http://localhost:8000"))
    enabled = os.getenv("BACKEND_ENABLED", str(cfg.get("backend_enabled", True)))
    return {"backend_url": url, "backend_enabled": enabled.lower() in ("1", "true", "yes")}


_CFG = _load_effective_config()

BACKEND_URL: str = _CFG["backend_url"]

BACKEND_ENABLED: bool = _CFG["backend_enabled"]
