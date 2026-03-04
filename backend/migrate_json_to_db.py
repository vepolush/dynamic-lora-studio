#!/usr/bin/env python3
"""
Migrate existing JSON storage to SQLite.
Run once before switching to SQLAlchemy, or when upgrading.

Usage:
    cd backend
    python migrate_json_to_db.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

import os

from db import EntityModel, MessageModel, SessionModel, init_db, session_scope

DATA_DIR = Path(os.getenv("DATA_DIR", "/workspace/data"))
ENTITIES_DIR = DATA_DIR / "storage" / "entities"
SESSIONS_DIR = DATA_DIR / "sessions"


def migrate_entities() -> int:
    """Migrate entity metadata from JSON to DB."""
    count = 0
    if not ENTITIES_DIR.exists():
        return 0

    with session_scope() as session:
        for entity_path in ENTITIES_DIR.iterdir():
            if not entity_path.is_dir():
                continue
            meta_path = entity_path / "metadata.json"
            if not meta_path.exists():
                continue
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue

            entity_id = str(data.get("id", ""))
            if not entity_id:
                continue

            existing = session.get(EntityModel, entity_id)
            if existing:
                continue

            training_params = data.get("training_params")
            caption_stats = data.get("caption_stats")
            entity = EntityModel(
                id=entity_id,
                name=str(data.get("name", "")),
                trigger_word=str(data.get("trigger_word", "")),
                status=str(data.get("status", "queued")),
                created_at=str(data.get("created_at", "")),
                image_count=int(data.get("image_count", 0)),
                training_profile=data.get("training_profile"),
                training_params=json.dumps(training_params, ensure_ascii=False) if training_params else None,
                caption_stats=json.dumps(caption_stats, ensure_ascii=False) if caption_stats else None,
                active_version=data.get("active_version"),
                caption_mode=data.get("caption_mode"),
                error=data.get("error"),
                training_job_id=data.get("training_job_id"),
                preview_error=data.get("preview_error"),
                uploaded_zip_path=data.get("uploaded_zip_path"),
                preview_url=data.get("preview_url"),
            )
            session.add(entity)
            count += 1

    return count


def migrate_sessions() -> int:
    """Migrate sessions and messages from JSON to DB."""
    count = 0
    if not SESSIONS_DIR.exists():
        return 0

    for fp in SESSIONS_DIR.glob("*.json"):
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        session_id = str(data.get("id", ""))
        if not session_id:
            continue

        with session_scope() as session:
            existing = session.get(SessionModel, session_id)
            if existing:
                continue

            fav_filenames = data.get("favourite_image_filenames", [])
            s = SessionModel(
                id=session_id,
                title=str(data.get("title", "New session")),
                created_at=str(data.get("created_at", "")),
                favourite=bool(data.get("favourite", False)),
                favourite_image_filenames=json.dumps(fav_filenames, ensure_ascii=False) if fav_filenames else None,
            )
            session.add(s)
            session.flush()

            for msg in data.get("messages", []):
                msg_id = msg.get("id") or f"msg_{msg.get('timestamp', '')[:8]}"
                m = MessageModel(
                    id=msg_id,
                    session_id=session_id,
                    timestamp=str(msg.get("timestamp", "")),
                    prompt=msg.get("prompt"),
                    enhanced_prompt=msg.get("enhanced_prompt"),
                    negative_prompt=msg.get("negative_prompt"),
                    settings=json.dumps(msg.get("settings", {}), ensure_ascii=False) if msg.get("settings") else None,
                    images=json.dumps(msg.get("images", []), ensure_ascii=False) if msg.get("images") else None,
                    generation_time=msg.get("generation_time"),
                )
                session.add(m)

            count += 1

    return count


def main() -> None:
    init_db()
    entities = migrate_entities()
    sessions = migrate_sessions()
    print(f"Migrated {entities} entities, {sessions} sessions.")


if __name__ == "__main__":
    main()
