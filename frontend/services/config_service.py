"""Local app config — load/save user preferences to disk."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

CONFIG_DIR: Path = Path.home() / ".config" / "dynamic-lora-studio"
CONFIG_FILE: Path = CONFIG_DIR / "settings.json"

DEFAULTS: dict[str, Any] = {
    "backend_url": "http://localhost:8000",
    "backend_enabled": True,
}


def load_config() -> dict[str, Any]:
    """Load config from file. Returns defaults if file missing or invalid."""
    if not CONFIG_FILE.exists():
        return DEFAULTS.copy()

    try:
        data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        out = DEFAULTS.copy()
        out.update({k: v for k, v in data.items() if k in DEFAULTS})
        return out
    except (json.JSONDecodeError, OSError):
        return DEFAULTS.copy()


def save_config(cfg: dict[str, Any]) -> None:
    """Save config to file."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
