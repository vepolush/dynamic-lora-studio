"""Session storage — SQLAlchemy for sessions/messages, images on disk."""

from __future__ import annotations

import json
import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from db import MessageModel, SessionModel, init_db, session_scope

DATA_DIR = Path(os.getenv("DATA_DIR", "/workspace/data"))
IMAGES_DIR = DATA_DIR / "images"


def _ensure_dirs() -> None:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)


def _session_images_dir(session_id: str) -> Path:
    d = IMAGES_DIR / session_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def new_session_id() -> str:
    return f"sess_{uuid.uuid4().hex[:12]}"


def new_message_id() -> str:
    return f"msg_{uuid.uuid4().hex[:8]}"


def create_session(
    session_id: str | None = None,
    title: str = "New session",
    user_id: str | None = None,
) -> dict[str, Any]:
    _ensure_dirs()
    init_db()
    sid = session_id or new_session_id()
    now = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")

    with session_scope() as session:
        s = SessionModel(
            id=sid,
            title=title,
            created_at=now,
            favourite=False,
            user_id=user_id,
        )
        session.add(s)

    return {
        "id": sid,
        "title": title,
        "created_at": now,
        "messages": [],
        "favourite": False,
        "archived": False,
    }


def get_session(session_id: str, user_id: str | None = None) -> dict[str, Any] | None:
    init_db()
    with session_scope() as session:
        row = session.get(SessionModel, session_id)
        if row is None:
            return None
        if user_id is not None:
            if row.user_id is not None and row.user_id != user_id:
                return None
        else:
            if row.user_id is not None:
                return None  # Anonymous: only legacy sessions
        return row.to_dict()


def list_sessions(user_id: str | None = None) -> list[dict[str, Any]]:
    _ensure_dirs()
    init_db()
    sessions: list[dict[str, Any]] = []
    with session_scope() as session:
        q = session.query(SessionModel).order_by(SessionModel.created_at.desc())
        from sqlalchemy import or_
        if user_id is not None:
            q = q.filter(or_(SessionModel.user_id == user_id, SessionModel.user_id.is_(None)))
        else:
            q = q.filter(SessionModel.user_id.is_(None))  # Anonymous: only legacy sessions
        rows = q.all()
        for row in rows:
            msg_count = len(row.messages)
            last_prompt = ""
            if row.messages:
                last_prompt = row.messages[-1].prompt or ""
            fav_filenames = json.loads(row.favourite_image_filenames) if row.favourite_image_filenames else []
            sessions.append({
                "id": row.id,
                "title": row.title,
                "created_at": row.created_at,
                "message_count": msg_count,
                "favourite": row.favourite,
                "favourite_image_filenames": fav_filenames,
                "archived": getattr(row, "archived", False),
                "last_prompt": last_prompt,
            })
    return sessions


def update_session(
    session_id: str,
    updates: dict[str, Any],
    user_id: str | None = None,
) -> dict[str, Any] | None:
    init_db()
    with session_scope() as session:
        row = session.get(SessionModel, session_id)
        if row is None:
            return None
        if user_id is not None and row.user_id is not None and row.user_id != user_id:
            return None
        if "title" in updates:
            row.title = updates["title"]
        if "favourite" in updates:
            row.favourite = updates["favourite"]
        if "favourite_image_filenames" in updates:
            row.favourite_image_filenames = json.dumps(
                updates["favourite_image_filenames"],
                ensure_ascii=False,
            )
        if "archived" in updates:
            row.archived = bool(updates["archived"])
        return row.to_dict()


def delete_session(session_id: str, user_id: str | None = None) -> bool:
    init_db()
    with session_scope() as session:
        row = session.get(SessionModel, session_id)
        if row is None:
            return False
        if user_id is not None and row.user_id is not None and row.user_id != user_id:
            return False
        session.delete(row)

    img_dir = IMAGES_DIR / session_id
    if img_dir.exists():
        shutil.rmtree(img_dir, ignore_errors=True)
    return True


def add_message(session_id: str, message: dict[str, Any]) -> dict[str, Any] | None:
    init_db()
    msg_id = message.get("id") or new_message_id()
    timestamp = message.get("timestamp") or datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")

    with session_scope() as session:
        row = session.get(SessionModel, session_id)
        if row is None:
            return None

        m = MessageModel(
            id=msg_id,
            session_id=session_id,
            timestamp=timestamp,
            prompt=message.get("prompt"),
            enhanced_prompt=message.get("enhanced_prompt"),
            negative_prompt=message.get("negative_prompt"),
            settings=json.dumps(message.get("settings", {}), ensure_ascii=False) if message.get("settings") else None,
            images=json.dumps(message.get("images", []), ensure_ascii=False) if message.get("images") else None,
            generation_time=message.get("generation_time"),
        )
        session.add(m)
        message["id"] = msg_id
        message["timestamp"] = timestamp
        return message


def save_image(session_id: str, image_bytes: bytes, filename: str) -> str:
    img_dir = _session_images_dir(session_id)
    filepath = img_dir / filename
    with open(filepath, "wb") as f:
        f.write(image_bytes)
    return str(filepath)


def load_image_bytes(session_id: str, filename: str) -> bytes | None:
    filepath = IMAGES_DIR / session_id / filename
    if not filepath.exists():
        return None
    with open(filepath, "rb") as f:
        return f.read()
