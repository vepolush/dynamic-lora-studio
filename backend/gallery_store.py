"""Gallery storage: publish, list, like."""

from __future__ import annotations

import json
import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from db import GalleryImageModel, GalleryLikeModel, init_db, session_scope

DATA_DIR = Path(os.getenv("DATA_DIR", "/workspace/data"))
IMAGES_DIR = DATA_DIR / "images"
GALLERY_DIR = DATA_DIR / "gallery"


def _ensure_gallery_dir() -> None:
    GALLERY_DIR.mkdir(parents=True, exist_ok=True)


def _gallery_image_path(image_id: str) -> Path:
    return GALLERY_DIR / f"{image_id}.png"


def _settings_without_lora(settings: dict | None) -> dict | None:
    """Remove LoRA-related keys from settings for public display."""
    if not settings:
        return None
    exclude = {"entity_id", "entity_version", "lora_strength"}
    return {k: v for k, v in settings.items() if k not in exclude}


def get_published_filenames_for_session(session_id: str) -> set[str]:
    """Return set of filenames from this session that are already published."""
    init_db()
    with session_scope() as session:
        rows = session.query(GalleryImageModel.filename).filter(
            GalleryImageModel.session_id == session_id,
        ).all()
        return {r[0] for r in rows}


def _is_image_already_published(session_id: str, filename: str) -> bool:
    """Check if image (session_id+filename) is already published to gallery."""
    init_db()
    with session_scope() as session:
        row = session.query(GalleryImageModel).filter(
            GalleryImageModel.session_id == session_id,
            GalleryImageModel.filename == filename,
        ).first()
        return row is not None


def publish_image(
    user_id: str,
    session_id: str,
    filename: str,
    prompt: str,
    settings: dict | None,
) -> dict[str, Any]:
    """Copy image to gallery and create record. Returns gallery image dict."""
    init_db()
    _ensure_gallery_dir()

    if _is_image_already_published(session_id, filename):
        raise ValueError("Image already published to gallery")

    src = IMAGES_DIR / session_id / filename
    if not src.exists():
        raise FileNotFoundError(f"Image not found: {session_id}/{filename}")

    image_id = f"gal_{uuid.uuid4().hex[:12]}"
    dst = _gallery_image_path(image_id)
    shutil.copy2(src, dst)

    now = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")
    clean_settings = _settings_without_lora(settings)

    with session_scope() as session:
        img = GalleryImageModel(
            id=image_id,
            user_id=user_id,
            session_id=session_id,
            filename=filename,
            prompt=prompt,
            settings=json.dumps(clean_settings, ensure_ascii=False) if clean_settings else None,
            likes_count=0,
            published_at=now,
        )
        session.add(img)

    return _gallery_image_to_dict(image_id, user_id, session_id, filename, prompt, clean_settings, 0, now)


def _gallery_image_to_dict(
    image_id: str,
    user_id: str,
    session_id: str,
    filename: str,
    prompt: str | None,
    settings: dict | None,
    likes_count: int,
    published_at: str,
    author_email: str | None = None,
    liked: bool = False,
) -> dict[str, Any]:
    return {
        "id": image_id,
        "user_id": user_id,
        "author_email": author_email or "",
        "session_id": session_id,
        "filename": filename,
        "prompt": prompt,
        "settings": settings,
        "likes_count": likes_count,
        "published_at": published_at,
        "liked": liked,
    }


def list_gallery(
    sort: str = "newest",
    limit: int = 50,
    offset: int = 0,
    current_user_id: str | None = None,
) -> list[dict[str, Any]]:
    """List gallery images. sort: newest, oldest, popular."""
    init_db()
    with session_scope() as session:
        q = session.query(GalleryImageModel)
        if sort == "popular":
            q = q.order_by(GalleryImageModel.likes_count.desc(), GalleryImageModel.published_at.desc())
        elif sort == "oldest":
            q = q.order_by(GalleryImageModel.published_at.asc())
        else:
            q = q.order_by(GalleryImageModel.published_at.desc())

        rows = q.offset(offset).limit(limit).all()
        liked_ids = set()
        if current_user_id:
            likes = session.query(GalleryLikeModel.image_id).filter(
                GalleryLikeModel.user_id == current_user_id,
            ).all()
            liked_ids = {r[0] for r in likes}

        from db import UserModel
        result = []
        for row in rows:
            user = session.get(UserModel, row.user_id)
            author = user.username if user else ""
            result.append(_gallery_image_to_dict(
                row.id,
                row.user_id,
                row.session_id,
                row.filename,
                row.prompt,
                json.loads(row.settings) if row.settings else None,
                row.likes_count,
                row.published_at,
                author_email=author,
                liked=row.id in liked_ids,
            ))
        return result


def get_gallery_image(
    image_id: str,
    current_user_id: str | None = None,
) -> dict[str, Any] | None:
    """Get single gallery image by id."""
    init_db()
    with session_scope() as session:
        row = session.get(GalleryImageModel, image_id)
        if not row:
            return None
        liked = False
        if current_user_id:
            like = session.query(GalleryLikeModel).filter(
                GalleryLikeModel.user_id == current_user_id,
                GalleryLikeModel.image_id == image_id,
            ).first()
            liked = like is not None
        from db import UserModel
        user = session.get(UserModel, row.user_id)
        return _gallery_image_to_dict(
            row.id,
            row.user_id,
            row.session_id,
            row.filename,
            row.prompt,
            json.loads(row.settings) if row.settings else None,
            row.likes_count,
            row.published_at,
            author_email=user.username if user else "",
            liked=liked,
        )


def load_gallery_image_bytes(image_id: str) -> bytes | None:
    """Load gallery image bytes from disk."""
    path = _gallery_image_path(image_id)
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return f.read()


def toggle_like(image_id: str, user_id: str) -> dict[str, Any] | None:
    """Toggle like. Returns updated image dict."""
    init_db()
    with session_scope() as session:
        row = session.get(GalleryImageModel, image_id)
        if not row:
            return None
        like = session.query(GalleryLikeModel).filter(
            GalleryLikeModel.user_id == user_id,
            GalleryLikeModel.image_id == image_id,
        ).first()
        if like:
            session.delete(like)
            row.likes_count = max(0, row.likes_count - 1)
            liked = False
        else:
            session.add(GalleryLikeModel(user_id=user_id, image_id=image_id))
            row.likes_count += 1
            liked = True
        session.flush()
        from db import UserModel
        user = session.get(UserModel, row.user_id)
        return _gallery_image_to_dict(
            row.id,
            row.user_id,
            row.session_id,
            row.filename,
            row.prompt,
            json.loads(row.settings) if row.settings else None,
            row.likes_count,
            row.published_at,
            author_email=user.username if user else "",
            liked=liked,
        )
