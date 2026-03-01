"""Persistent file-based storage for LoRA entities."""

from __future__ import annotations

import json
import os
import re
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

DATA_DIR = Path(os.getenv("DATA_DIR", "/workspace/data"))
ENTITIES_DIR = DATA_DIR / "storage" / "entities"


def _ensure_root() -> None:
    ENTITIES_DIR.mkdir(parents=True, exist_ok=True)


def _slugify(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower())
    return cleaned.strip("_") or "entity"


def _entity_dir(entity_id: str) -> Path:
    return ENTITIES_DIR / entity_id


def _metadata_path(entity_id: str) -> Path:
    return _entity_dir(entity_id) / "metadata.json"


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _version_sort_key(name: str) -> tuple[int, str]:
    match = re.match(r"v(\d+)", name.lower())
    if match:
        return (int(match.group(1)), name)
    return (0, name)


def _collect_versions(entity_id: str) -> list[str]:
    weights_dir = _entity_dir(entity_id) / "weights"
    if not weights_dir.exists():
        return []
    versions = [p.name for p in weights_dir.iterdir() if p.is_dir()]
    versions.sort(key=_version_sort_key)
    return versions


def _normalize_entity(metadata: dict[str, Any]) -> dict[str, Any]:
    entity_id = str(metadata["id"])
    versions = metadata.get("versions")
    if not isinstance(versions, list):
        versions = _collect_versions(entity_id)

    active_version = metadata.get("active_version")
    if active_version and active_version not in versions:
        active_version = None

    return {
        "id": entity_id,
        "name": str(metadata.get("name", "")),
        "trigger_word": str(metadata.get("trigger_word", "")),
        "status": str(metadata.get("status", "queued")),
        "versions": versions,
        "active_version": active_version,
        "preview_url": metadata.get("preview_url"),
        "created_at": str(metadata.get("created_at", "")),
        "has_lora": bool(versions),
        "image_count": int(metadata.get("image_count", 0)),
    }


def list_entities() -> list[dict[str, Any]]:
    _ensure_root()
    entities: list[dict[str, Any]] = []
    for entity_path in sorted(ENTITIES_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not entity_path.is_dir():
            continue
        metadata_path = entity_path / "metadata.json"
        if not metadata_path.exists():
            continue
        try:
            metadata = _read_json(metadata_path)
            entities.append(_normalize_entity(metadata))
        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            continue
    return entities


def get_entity(entity_id: str) -> dict[str, Any] | None:
    metadata_path = _metadata_path(entity_id)
    if not metadata_path.exists():
        return None
    try:
        metadata = _read_json(metadata_path)
    except (json.JSONDecodeError, ValueError, TypeError):
        return None
    return _normalize_entity(metadata)


def get_entity_metadata(entity_id: str) -> dict[str, Any] | None:
    metadata_path = _metadata_path(entity_id)
    if not metadata_path.exists():
        return None
    try:
        return _read_json(metadata_path)
    except (json.JSONDecodeError, ValueError, TypeError):
        return None


def update_entity_metadata(entity_id: str, updates: dict[str, Any]) -> dict[str, Any] | None:
    metadata_path = _metadata_path(entity_id)
    if not metadata_path.exists():
        return None
    try:
        metadata = _read_json(metadata_path)
    except (json.JSONDecodeError, ValueError, TypeError):
        return None

    metadata.update(updates)
    _write_json(metadata_path, metadata)
    return _normalize_entity(metadata)


def create_entity(
    *,
    name: str,
    trigger_word: str,
    uploaded_filename: str,
    temp_zip_path: Path,
) -> dict[str, Any]:
    _ensure_root()
    now = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")
    entity_id = f"entity_{uuid.uuid4().hex[:8]}_{_slugify(name)[:32]}"

    entity_dir = _entity_dir(entity_id)
    dataset_dir = entity_dir / "dataset"
    weights_dir = entity_dir / "weights"
    temp_dir = entity_dir / "temp"

    dataset_dir.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    saved_zip_path = temp_dir / f"upload_{uuid.uuid4().hex[:8]}_{Path(uploaded_filename).name}"
    shutil.copy2(temp_zip_path, saved_zip_path)

    metadata: dict[str, Any] = {
        "id": entity_id,
        "name": name,
        "trigger_word": trigger_word,
        "status": "queued",
        "created_at": now,
        "versions": [],
        "active_version": None,
        "preview_url": None,
        "image_count": 0,
        "uploaded_zip_path": str(saved_zip_path),
    }
    _write_json(_metadata_path(entity_id), metadata)
    return _normalize_entity(metadata)
