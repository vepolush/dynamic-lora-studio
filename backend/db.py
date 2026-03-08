"""SQLAlchemy database setup and models."""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator

from sqlalchemy import JSON, Boolean, Float, ForeignKey, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship

DATA_DIR = Path(os.getenv("DATA_DIR", "/workspace/data"))
DB_PATH = DATA_DIR / "studio.db"
DB_URL = f"sqlite:///{DB_PATH}"


class Base(DeclarativeBase):
    """Base class for all models."""

    type_annotation_map = {
        dict[str, Any]: JSON,
    }


class EntityModel(Base):
    """Entity metadata (LoRA training subject)."""

    __tablename__ = "entities"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    user_id: Mapped[str | None] = mapped_column(String(64), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    trigger_word: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="queued")
    created_at: Mapped[str] = mapped_column(String(32), nullable=False)
    image_count: Mapped[int] = mapped_column(Integer, default=0)
    training_profile: Mapped[str | None] = mapped_column(String(32), nullable=True)
    training_params: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON
    active_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    caption_mode: Mapped[str | None] = mapped_column(String(32), nullable=True)
    caption_stats: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    training_job_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    preview_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    uploaded_zip_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    preview_url: Mapped[str | None] = mapped_column(String(256), nullable=True)
    source_gallery_lora_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    subject_type: Mapped[str | None] = mapped_column(String(64), nullable=True)  # e.g. cat, person, dog

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict matching legacy metadata.json structure."""
        return {
            "id": self.id,
            "name": self.name,
            "trigger_word": self.trigger_word,
            "status": self.status,
            "created_at": self.created_at,
            "image_count": self.image_count,
            "training_profile": self.training_profile,
            "training_params": _parse_json(self.training_params),
            "active_version": self.active_version,
            "caption_mode": self.caption_mode,
            "caption_stats": _parse_json(self.caption_stats),
            "error": self.error,
            "training_job_id": self.training_job_id,
            "preview_error": self.preview_error,
            "uploaded_zip_path": self.uploaded_zip_path,
            "preview_url": self.preview_url,
        }


class SessionModel(Base):
    """Generation session."""

    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str | None] = mapped_column(String(64), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    created_at: Mapped[str] = mapped_column(String(32), nullable=False)
    favourite: Mapped[bool] = mapped_column(Boolean, default=False)
    favourite_image_filenames: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON array
    archived: Mapped[bool] = mapped_column(Boolean, default=False)

    messages: Mapped[list["MessageModel"]] = relationship(
        "MessageModel",
        back_populates="session",
        order_by="MessageModel.timestamp",
        cascade="all, delete-orphan",
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to full session dict with messages."""
        return {
            "id": self.id,
            "title": self.title,
            "created_at": self.created_at,
            "favourite": self.favourite,
            "favourite_image_filenames": _parse_json(self.favourite_image_filenames) or [],
            "archived": self.archived,
            "messages": [m.to_dict() for m in self.messages],
        }


class UserModel(Base):
    """User account for auth and gallery."""

    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    password_hash: Mapped[str] = mapped_column(String(256), nullable=False)
    created_at: Mapped[str] = mapped_column(String(32), nullable=False)


class GalleryImageModel(Base):
    """Published image in global gallery."""

    __tablename__ = "gallery_images"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(64), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    session_id: Mapped[str] = mapped_column(String(64), nullable=False)
    filename: Mapped[str] = mapped_column(String(256), nullable=False)
    prompt: Mapped[str | None] = mapped_column(Text, nullable=True)
    settings: Mapped[str | None] = mapped_column(Text, nullable=True)
    likes_count: Mapped[int] = mapped_column(Integer, default=0)
    published_at: Mapped[str] = mapped_column(String(32), nullable=False)


class GalleryLikeModel(Base):
    """User like on gallery image."""

    __tablename__ = "gallery_likes"

    user_id: Mapped[str] = mapped_column(String(64), ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    image_id: Mapped[str] = mapped_column(String(64), ForeignKey("gallery_images.id", ondelete="CASCADE"), primary_key=True)


class GalleryLoraModel(Base):
    """Published LoRA in gallery — others can add (copy) to their entities."""

    __tablename__ = "gallery_loras"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(64), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    entity_id: Mapped[str] = mapped_column(String(128), nullable=False)
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    trigger_word: Mapped[str] = mapped_column(String(128), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    add_count: Mapped[int] = mapped_column(Integer, default=0)
    published_at: Mapped[str] = mapped_column(String(32), nullable=False)


class MessageModel(Base):
    """Generation message (prompt + images)."""

    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    session_id: Mapped[str] = mapped_column(String(64), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False)
    timestamp: Mapped[str] = mapped_column(String(32), nullable=False)
    prompt: Mapped[str | None] = mapped_column(Text, nullable=True)
    enhanced_prompt: Mapped[str | None] = mapped_column(Text, nullable=True)
    negative_prompt: Mapped[str | None] = mapped_column(Text, nullable=True)
    settings: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON
    images: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON array
    generation_time: Mapped[float | None] = mapped_column(Float, nullable=True)

    session: Mapped["SessionModel"] = relationship("SessionModel", back_populates="messages")

    def to_dict(self) -> dict[str, Any]:
        """Convert to message dict."""
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "prompt": self.prompt,
            "enhanced_prompt": self.enhanced_prompt,
            "negative_prompt": self.negative_prompt,
            "settings": _parse_json(self.settings),
            "images": _parse_json(self.images) or [],
            "generation_time": self.generation_time,
        }


def _parse_json(s: str | None) -> Any:
    """Parse JSON string, return None if invalid."""
    if not s:
        return None
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return None


def _ensure_db_dir() -> None:
    """Ensure DATA_DIR exists for DB file."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def get_engine():
    """Create SQLite engine with check_same_thread=False for FastAPI."""
    _ensure_db_dir()
    return create_engine(
        DB_URL,
        connect_args={"check_same_thread": False},
        echo=False,
    )


_engine = None


def get_db_engine():
    """Get or create singleton engine."""
    global _engine
    if _engine is None:
        _engine = get_engine()
    return _engine


def _run_migrations(engine) -> None:
    """Run schema migrations for existing databases."""
    from sqlalchemy import text
    with engine.connect() as conn:
        try:
            conn.execute(text("ALTER TABLE sessions ADD COLUMN archived BOOLEAN DEFAULT 0"))
            conn.commit()
        except Exception:
            conn.rollback()
        try:
            conn.execute(text("ALTER TABLE entities ADD COLUMN source_gallery_lora_id VARCHAR(64)"))
            conn.commit()
        except Exception:
            conn.rollback()
        try:
            conn.execute(text("ALTER TABLE entities ADD COLUMN subject_type VARCHAR(64)"))
            conn.commit()
        except Exception:
            conn.rollback()


def init_db() -> None:
    """Create all tables if they don't exist."""
    engine = get_db_engine()
    Base.metadata.create_all(engine)
    _run_migrations(engine)


@contextmanager
def session_scope() -> Generator[Session, None, None]:
    """Context manager for database sessions."""
    engine = get_db_engine()
    session = Session(engine, expire_on_commit=False)
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
