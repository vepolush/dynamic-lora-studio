"""Gallery LoRA storage: publish, list, add (copy), unpublish."""

from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from db import GalleryLoraModel, UserModel, init_db, session_scope
from entity_store import (
    _ensure_root,
    create_entity_from_weights,
    entity_weights_dir,
    get_entity,
)

DATA_DIR = Path(__import__("os").environ.get("DATA_DIR", "/workspace/data"))
GALLERY_LORAS_DIR = DATA_DIR / "gallery" / "loras"


def _ensure_loras_dir() -> None:
    GALLERY_LORAS_DIR.mkdir(parents=True, exist_ok=True)


def _gallery_lora_dir(lora_id: str) -> Path:
    return GALLERY_LORAS_DIR / lora_id


def _copy_entity_weights_to_gallery(entity_id: str, gallery_lora_id: str, version: str) -> None:
    """Copy entity weights version to gallery storage."""
    src = entity_weights_dir(entity_id) / version
    if not src.exists():
        raise FileNotFoundError(f"Entity weights not found: {entity_id}/{version}")
    dst = _gallery_lora_dir(gallery_lora_id)
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.is_file():
            shutil.copy2(f, dst / f.name)




def publish_lora(
    user_id: str,
    entity_id: str,
    name: str,
    trigger_word: str,
    description: str | None = None,
) -> dict[str, Any]:
    """Publish entity LoRA to gallery. Copies weights to shared storage."""
    init_db()
    _ensure_loras_dir()

    entity = get_entity(entity_id)
    if not entity:
        raise FileNotFoundError(f"Entity not found: {entity_id}")
    if entity.get("status") != "ready":
        raise ValueError("Entity must be ready (trained) to publish")
    versions = entity.get("versions", [])
    if not versions:
        raise ValueError("Entity has no LoRA weights")
    active = entity.get("active_version") or versions[-1]

    lora_id = f"lor_{uuid.uuid4().hex[:12]}"
    _copy_entity_weights_to_gallery(entity_id, lora_id, active)

    now = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")
    with session_scope() as session:
        existing = session.query(GalleryLoraModel).filter(
            GalleryLoraModel.entity_id == entity_id,
            GalleryLoraModel.user_id == user_id,
        ).first()
        if existing:
            raise ValueError("This LoRA is already published")

        row = GalleryLoraModel(
            id=lora_id,
            user_id=user_id,
            entity_id=entity_id,
            name=name,
            trigger_word=trigger_word,
            description=description,
            add_count=0,
            published_at=now,
        )
        session.add(row)

    return _lora_to_dict(lora_id, user_id, entity_id, name, trigger_word, description, 0, now, is_mine=True)


def _lora_to_dict(
    lora_id: str,
    user_id: str,
    entity_id: str,
    name: str,
    trigger_word: str,
    description: str | None,
    add_count: int,
    published_at: str,
    author_email: str | None = None,
    is_mine: bool = False,
) -> dict[str, Any]:
    return {
        "id": lora_id,
        "user_id": user_id,
        "entity_id": entity_id,
        "name": name,
        "trigger_word": trigger_word,
        "description": description or "",
        "add_count": add_count,
        "published_at": published_at,
        "author_email": author_email or "",
        "is_mine": is_mine,
    }


def list_gallery_loras(
    sort: str = "newest",
    limit: int = 50,
    offset: int = 0,
    current_user_id: str | None = None,
) -> list[dict[str, Any]]:
    """List gallery LoRAs. sort: newest, oldest, popular (by add_count)."""
    init_db()
    with session_scope() as session:
        q = session.query(GalleryLoraModel)
        if sort == "popular":
            q = q.order_by(GalleryLoraModel.add_count.desc(), GalleryLoraModel.published_at.desc())
        elif sort == "oldest":
            q = q.order_by(GalleryLoraModel.published_at.asc())
        else:
            q = q.order_by(GalleryLoraModel.published_at.desc())

        rows = q.offset(offset).limit(limit).all()
        result = []
        for row in rows:
            user = session.get(UserModel, row.user_id)
            result.append(_lora_to_dict(
                row.id,
                row.user_id,
                row.entity_id,
                row.name,
                row.trigger_word,
                row.description,
                row.add_count,
                row.published_at,
                author_email=user.username if user else "",
                is_mine=current_user_id == row.user_id if current_user_id else False,
            ))
        return result


def get_gallery_lora(
    lora_id: str,
    current_user_id: str | None = None,
) -> dict[str, Any] | None:
    """Get single gallery LoRA."""
    init_db()
    with session_scope() as session:
        row = session.get(GalleryLoraModel, lora_id)
        if not row:
            return None
        user = session.get(UserModel, row.user_id)
        return _lora_to_dict(
            row.id,
            row.user_id,
            row.entity_id,
            row.name,
            row.trigger_word,
            row.description,
            row.add_count,
            row.published_at,
            author_email=user.username if user else "",
            is_mine=current_user_id == row.user_id if current_user_id else False,
        )


def add_gallery_lora(lora_id: str, user_id: str) -> dict[str, Any]:
    """Add (copy) gallery LoRA to user's entities. Returns new entity."""
    init_db()
    _ensure_root()
    with session_scope() as session:
        row = session.get(GalleryLoraModel, lora_id)
        if not row:
            raise FileNotFoundError("Gallery LoRA not found")

        new_entity = create_entity_from_weights(
            name=row.name,
            trigger_word=row.trigger_word,
            source_weights_dir=_gallery_lora_dir(lora_id),
            user_id=user_id,
        )
        row.add_count += 1

    return {
        "entity": new_entity,
        "add_count": row.add_count,
    }


def unpublish_lora(lora_id: str, user_id: str) -> bool:
    """Unpublish LoRA. Only creator can unpublish."""
    init_db()
    with session_scope() as session:
        row = session.get(GalleryLoraModel, lora_id)
        if not row:
            return False
        if row.user_id != user_id:
            return False
        session.delete(row)

    lora_dir = _gallery_lora_dir(lora_id)
    if lora_dir.exists():
        shutil.rmtree(lora_dir, ignore_errors=True)
    return True


def load_gallery_lora_preview_bytes(lora_id: str) -> bytes | None:
    """Load preview from source entity if available."""
    init_db()
    with session_scope() as session:
        row = session.get(GalleryLoraModel, lora_id)
        if not row:
            return None
        entity_id = row.entity_id
    from entity_store import entity_preview_path
    path = entity_preview_path(entity_id)
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return f.read()
