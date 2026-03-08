"""Entity storage — SQLAlchemy for metadata, files on disk for binaries."""

from __future__ import annotations

import json
import os
import re
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from db import EntityModel, init_db, session_scope

DATA_DIR = Path(os.getenv("DATA_DIR", "/workspace/data"))
ENTITIES_DIR = DATA_DIR / "storage" / "entities"


def _ensure_root() -> None:
    ENTITIES_DIR.mkdir(parents=True, exist_ok=True)


def _slugify(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower())
    return cleaned.strip("_") or "entity"


def _entity_dir(entity_id: str) -> Path:
    return ENTITIES_DIR / entity_id


def entity_dataset_dir(entity_id: str) -> Path:
    return _entity_dir(entity_id) / "dataset"


def entity_weights_dir(entity_id: str) -> Path:
    return _entity_dir(entity_id) / "weights"


def entity_preview_path(entity_id: str) -> Path:
    return _entity_dir(entity_id) / "preview.png"


def _version_sort_key(name: str) -> tuple[int, str]:
    match = re.match(r"v(\d+)", name.lower())
    if match:
        return (int(match.group(1)), name)
    return (0, name)


def _collect_versions(entity_id: str) -> list[str]:
    weights_dir = entity_weights_dir(entity_id)
    if not weights_dir.exists():
        return []
    versions = [p.name for p in weights_dir.iterdir() if p.is_dir()]
    versions.sort(key=_version_sort_key)
    return versions


def _entity_to_dict(row: EntityModel) -> dict[str, Any]:
    """Convert DB row to raw metadata dict (legacy format)."""
    return {
        "id": row.id,
        "name": row.name,
        "trigger_word": row.trigger_word,
        "status": row.status,
        "created_at": row.created_at,
        "image_count": row.image_count,
        "training_profile": row.training_profile,
        "training_params": json.loads(row.training_params) if row.training_params else None,
        "active_version": row.active_version,
        "caption_mode": row.caption_mode,
        "caption_stats": json.loads(row.caption_stats) if row.caption_stats else None,
        "error": row.error,
        "training_job_id": row.training_job_id,
        "preview_error": row.preview_error,
        "uploaded_zip_path": row.uploaded_zip_path,
        "preview_url": row.preview_url,
        "versions": [],
        "source_gallery_lora_id": getattr(row, "source_gallery_lora_id", None),
        "subject_type": getattr(row, "subject_type", None),
    }


def _normalize_entity(metadata: dict[str, Any]) -> dict[str, Any]:
    """Normalize entity for API response."""
    entity_id = str(metadata["id"])
    versions = metadata.get("versions")
    if not isinstance(versions, list):
        versions = _collect_versions(entity_id)

    active_version = metadata.get("active_version")
    if active_version and active_version not in versions:
        active_version = None

    preview_url = metadata.get("preview_url")
    if not preview_url and entity_preview_path(entity_id).exists():
        preview_url = f"/api/entities/{entity_id}/preview"

    return {
        "id": entity_id,
        "name": str(metadata.get("name", "")),
        "trigger_word": str(metadata.get("trigger_word", "")),
        "status": str(metadata.get("status", "queued")),
        "caption_mode": metadata.get("caption_mode"),
        "caption_stats": metadata.get("caption_stats"),
        "error": metadata.get("error"),
        "training_job_id": metadata.get("training_job_id"),
        "preview_error": metadata.get("preview_error"),
        "training_profile": metadata.get("training_profile"),
        "training_params": metadata.get("training_params"),
        "versions": versions,
        "active_version": active_version,
        "preview_url": preview_url,
        "created_at": str(metadata.get("created_at", "")),
        "has_lora": bool(versions),
        "image_count": int(metadata.get("image_count", 0)),
        "source_gallery_lora_id": metadata.get("source_gallery_lora_id"),
        "subject_type": metadata.get("subject_type"),
    }


def user_has_entity_from_gallery_lora(user_id: str, gallery_lora_id: str) -> bool:
    """Check if user already has an entity created from this gallery LoRA."""
    init_db()
    with session_scope() as session:
        from sqlalchemy import and_
        row = session.query(EntityModel).filter(
            and_(
                EntityModel.user_id == user_id,
                EntityModel.source_gallery_lora_id == gallery_lora_id,
            ),
        ).first()
        return row is not None


def list_entities(user_id: str | None = None) -> list[dict[str, Any]]:
    _ensure_root()
    init_db()
    entities: list[dict[str, Any]] = []
    with session_scope() as session:
        from sqlalchemy import or_
        q = session.query(EntityModel).order_by(EntityModel.created_at.desc())
        if user_id is not None:
            q = q.filter(or_(EntityModel.user_id == user_id, EntityModel.user_id.is_(None)))
        else:
            q = q.filter(EntityModel.user_id.is_(None))
        rows = q.all()
        for row in rows:
            if not _entity_dir(row.id).exists():
                continue
            meta = _entity_to_dict(row)
            meta["versions"] = _collect_versions(row.id)
            entities.append(_normalize_entity(meta))
    return entities


def get_entity(entity_id: str, user_id: str | None = None) -> dict[str, Any] | None:
    init_db()
    with session_scope() as session:
        row = session.get(EntityModel, entity_id)
        if row is None:
            return None
        if user_id is not None:
            if row.user_id is not None and row.user_id != user_id:
                return None
        else:
            if row.user_id is not None:
                return None
        if not _entity_dir(entity_id).exists():
            return None
        meta = _entity_to_dict(row)
        meta["versions"] = _collect_versions(entity_id)
        return _normalize_entity(meta)


def get_entity_metadata(entity_id: str) -> dict[str, Any] | None:
    init_db()
    with session_scope() as session:
        row = session.get(EntityModel, entity_id)
        if row is None:
            return None
        meta = _entity_to_dict(row)
        meta["versions"] = _collect_versions(entity_id)
        return meta


def load_entity_preview_bytes(entity_id: str) -> bytes | None:
    path = entity_preview_path(entity_id)
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return f.read()


def list_dataset_images(entity_id: str) -> list[dict[str, Any]]:
    """List dataset images. Prefer manifest when present; fallback to glob."""
    entity_dir = _entity_dir(entity_id)
    ds_dir = entity_dir / "dataset"
    if not ds_dir.exists():
        return []
    manifest_path = entity_dir / "dataset_manifest.json"
    if manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            images = manifest.get("images", [])
            result: list[dict[str, Any]] = []
            for img in images:
                fname = img.get("filename", "")
                if not fname or not isinstance(fname, str):
                    continue
                p = ds_dir / fname
                if p.exists() and p.is_file() and p.suffix.lower() == ".png":
                    try:
                        size = p.stat().st_size
                    except OSError:
                        size = 0
                    result.append({"filename": fname, "size": size})
            if result:
                return result
        except (OSError, json.JSONDecodeError, TypeError):
            pass
    result = []
    for p in sorted(ds_dir.glob("*.png")):
        if p.is_file():
            try:
                size = p.stat().st_size
            except OSError:
                size = 0
            result.append({"filename": p.name, "size": size})
    return result


def load_dataset_image_bytes(entity_id: str, filename: str) -> bytes | None:
    """Load dataset image. Handles URL decoding and path safety."""
    from urllib.parse import unquote

    ds_dir = entity_dataset_dir(entity_id)
    if not ds_dir.exists():
        return None
    filename = unquote(filename)
    if ".." in filename or "/" in filename or "\\" in filename:
        return None
    path = ds_dir / filename
    if path.exists() and path.is_file() and path.suffix.lower() == ".png":
        try:
            with open(path, "rb") as f:
                return f.read()
        except OSError:
            return None
    fname_lower = filename.lower()
    for p in ds_dir.iterdir():
        if p.is_file() and p.suffix.lower() == ".png" and p.name.lower() == fname_lower:
            try:
                with open(p, "rb") as f:
                    return f.read()
            except OSError:
                return None
    return None


def update_entity_metadata(
    entity_id: str,
    updates: dict[str, Any],
    user_id: str | None = None,
) -> dict[str, Any] | None:
    init_db()
    with session_scope() as session:
        row = session.get(EntityModel, entity_id)
        if row is None:
            return None
        if user_id is not None and row.user_id is not None and row.user_id != user_id:
            return None

        json_fields = ("training_params", "caption_stats")
        for key, value in updates.items():
            if hasattr(row, key):
                if key in json_fields and value is not None:
                    setattr(row, key, json.dumps(value, ensure_ascii=False))
                else:
                    setattr(row, key, value)

        session.flush()
        meta = _entity_to_dict(row)
        meta["versions"] = _collect_versions(entity_id)
        return _normalize_entity(meta)


def create_entity(
    *,
    name: str,
    trigger_word: str,
    uploaded_filename: str,
    temp_zip_path: Path,
    user_id: str | None = None,
    subject_type: str | None = None,
) -> dict[str, Any]:
    _ensure_root()
    init_db()
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

    with session_scope() as session:
        entity = EntityModel(
            id=entity_id,
            user_id=user_id,
            name=name,
            trigger_word=trigger_word,
            status="queued",
            created_at=now,
            image_count=0,
            uploaded_zip_path=str(saved_zip_path),
            subject_type=subject_type.strip() if subject_type and isinstance(subject_type, str) else None,
        )
        session.add(entity)

    meta = get_entity_metadata(entity_id)
    return _normalize_entity(meta) if meta else {"id": entity_id, "name": name, "trigger_word": trigger_word}


def create_entity_from_weights(
    *,
    name: str,
    trigger_word: str,
    source_weights_dir: Path,
    user_id: str | None = None,
    source_gallery_lora_id: str | None = None,
) -> dict[str, Any]:
    """Create entity from copied LoRA weights from gallery. No dataset."""
    _ensure_root()
    init_db()
    now = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")
    entity_id = f"entity_{uuid.uuid4().hex[:8]}_{_slugify(name)[:32]}"

    entity_dir = _entity_dir(entity_id)
    dataset_dir = entity_dir / "dataset"
    weights_dir = entity_dir / "weights"
    version_name = "v1_shared"

    dataset_dir.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(parents=True, exist_ok=True)
    version_dir = weights_dir / version_name
    version_dir.mkdir(parents=True, exist_ok=True)

    for f in source_weights_dir.iterdir():
        if f.is_file():
            shutil.copy2(f, version_dir / f.name)

    with session_scope() as session:
        entity = EntityModel(
            id=entity_id,
            user_id=user_id,
            name=name,
            trigger_word=trigger_word,
            status="ready",
            created_at=now,
            image_count=0,
            active_version=version_name,
            source_gallery_lora_id=source_gallery_lora_id,
        )
        session.add(entity)

    meta = get_entity_metadata(entity_id)
    return _normalize_entity(meta) if meta else {"id": entity_id, "name": name, "trigger_word": trigger_word}


def delete_entity(entity_id: str, user_id: str | None = None) -> bool:
    init_db()
    entity_dir = _entity_dir(entity_id)
    with session_scope() as session:
        row = session.get(EntityModel, entity_id)
        if row is None:
            return False
        if user_id is not None and row.user_id is not None and row.user_id != user_id:
            return False
        session.delete(row)
    if entity_dir.exists():
        shutil.rmtree(entity_dir, ignore_errors=True)
    return True
