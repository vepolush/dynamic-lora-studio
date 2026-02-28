"""File-based session storage — JSON per session, images on disk."""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

DATA_DIR = Path(os.getenv("DATA_DIR", "/workspace/data"))
SESSIONS_DIR = DATA_DIR / "sessions"
IMAGES_DIR = DATA_DIR / "images"


def _ensure_dirs() -> None:
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)


def _session_path(session_id: str) -> Path:
    return SESSIONS_DIR / f"{session_id}.json"


def _session_images_dir(session_id: str) -> Path:
    d = IMAGES_DIR / session_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def new_session_id() -> str:
    return f"sess_{uuid.uuid4().hex[:12]}"


def new_message_id() -> str:
    return f"msg_{uuid.uuid4().hex[:8]}"


def create_session(session_id: str | None = None, title: str = "New session") -> dict[str, Any]:
    """Create and persist a new session."""
    _ensure_dirs()
    sid = session_id or new_session_id()
    now = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")
    session: dict[str, Any] = {
        "id": sid,
        "title": title,
        "created_at": now,
        "messages": [],
        "favourite": False,
    }
    _write_json(_session_path(sid), session)
    return session


def get_session(session_id: str) -> dict[str, Any] | None:
    """Load a single session from disk."""
    path = _session_path(session_id)
    if not path.exists():
        return None
    return _read_json(path)


def list_sessions() -> list[dict[str, Any]]:
    """List all sessions (without full message history, for sidebar)."""
    _ensure_dirs()
    sessions: list[dict[str, Any]] = []
    for fp in sorted(SESSIONS_DIR.glob("sess_*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            data = _read_json(fp)
            summary = {
                "id": data["id"],
                "title": data["title"],
                "created_at": data["created_at"],
                "message_count": len(data.get("messages", [])),
                "favourite": data.get("favourite", False),
            }
            msgs = data.get("messages", [])
            if msgs:
                last = msgs[-1]
                summary["last_prompt"] = last.get("prompt", "")
            sessions.append(summary)
        except (json.JSONDecodeError, KeyError):
            continue
    return sessions


def update_session(session_id: str, updates: dict[str, Any]) -> dict[str, Any] | None:
    """Update session metadata (title, favourite). Does NOT touch messages."""
    path = _session_path(session_id)
    if not path.exists():
        return None
    data = _read_json(path)
    for key in ("title", "favourite"):
        if key in updates:
            data[key] = updates[key]
    _write_json(path, data)
    return data


def delete_session(session_id: str) -> bool:
    """Delete session file and its images."""
    path = _session_path(session_id)
    if not path.exists():
        return False
    path.unlink()
    img_dir = IMAGES_DIR / session_id
    if img_dir.exists():
        import shutil
        shutil.rmtree(img_dir, ignore_errors=True)
    return True


def add_message(session_id: str, message: dict[str, Any]) -> dict[str, Any] | None:
    """Append a message to a session and persist."""
    path = _session_path(session_id)
    if not path.exists():
        return None
    data = _read_json(path)
    if "id" not in message:
        message["id"] = new_message_id()
    if "timestamp" not in message:
        message["timestamp"] = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")
    data["messages"].append(message)
    _write_json(path, data)
    return message


def save_image(session_id: str, image_bytes: bytes, filename: str) -> str:
    """Save image bytes to disk, return relative path."""
    img_dir = _session_images_dir(session_id)
    filepath = img_dir / filename
    with open(filepath, "wb") as f:
        f.write(image_bytes)
    return str(filepath)


def load_image_bytes(session_id: str, filename: str) -> bytes | None:
    """Load image file bytes."""
    filepath = IMAGES_DIR / session_id / filename
    if not filepath.exists():
        return None
    with open(filepath, "rb") as f:
        return f.read()
