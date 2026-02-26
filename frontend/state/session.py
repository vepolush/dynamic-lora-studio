"""Session state initialization and mock data for the UI prototype."""

from __future__ import annotations

from datetime import datetime

import streamlit as st

MOCK_SESSIONS: list[dict] = [
    {
        "id": "sess_001",
        "title": "3D sphere",
        "created_at": "2026-02-26T12:00:00",
        "prompt": (
            "a high-texture pearl 3D sphere glowing softly in pastel purple, "
            "orange, and blue colors against a solid black background"
        ),
        "helper_specs": "::3 glowing::5 pearl::3 foil::3 --v 6.0",
        "images": 4,
        "favourite": False,
    },
    {
        "id": "sess_002",
        "title": "Portrait study",
        "created_at": "2026-02-25T18:30:00",
        "prompt": "cinematic portrait of a woman in neon lighting, cyberpunk style",
        "helper_specs": "::2 neon::4 cinematic::3 --v 6.0",
        "images": 2,
        "favourite": True,
    },
    {
        "id": "sess_003",
        "title": "Landscape",
        "created_at": "2026-02-24T09:15:00",
        "prompt": "vast alien landscape with crystal formations, purple sky, dramatic lighting",
        "helper_specs": "::3 crystal::4 dramatic::3 --v 6.0",
        "images": 4,
        "favourite": True,
    },
]

MOCK_ENTITIES: list[dict] = [
    {
        "id": "entity_01_my_cat",
        "name": "My Cat",
        "trigger_word": "<my_cat>",
        "has_lora": True,
        "versions": ["v1_rank8_steps500", "v2_rank16_steps800"],
        "active_version": "v2_rank16_steps800",
        "image_count": 12,
        "created_at": "2026-02-20",
    },
    {
        "id": "entity_02_cyber_helmet",
        "name": "Cyber Helmet",
        "trigger_word": "<cyber_helmet>",
        "has_lora": True,
        "versions": ["v1_rank8_steps600"],
        "active_version": "v1_rank8_steps600",
        "image_count": 8,
        "created_at": "2026-02-22",
    },
]

DEFAULT_SETTINGS: dict = {
    "steps": 25,
    "guidance_scale": 7.5,
    "image_size": "512x512",
    "seed": -1,
    "quality": "Normal",
    "style": "None",
    "lightning": "None",
    "color": "Default",
}


def init_session_state() -> None:
    """Populate st.session_state with defaults if keys are missing."""
    defaults = {
        "current_page": "generate",
        "active_session_id": MOCK_SESSIONS[0]["id"],
        "sessions": MOCK_SESSIONS,
        "generation_settings": {**DEFAULT_SETTINGS},
        "entities": MOCK_ENTITIES,
        "active_entity_id": None,
        "lora_strength": 0.8,
        "show_entity_form": False,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_active_session() -> dict | None:
    """Return the currently active session dict."""
    for s in st.session_state.get("sessions", []):
        if s["id"] == st.session_state.get("active_session_id"):
            return s
    return None


def get_favourite_count() -> int:
    return sum(1 for s in st.session_state.get("sessions", []) if s.get("favourite"))


def format_session_date(iso_str: str) -> str:
    dt = datetime.fromisoformat(iso_str)
    today = datetime.now().date()
    if dt.date() == today:
        return f"Today at {dt.strftime('%I:%M %p')}"
    return dt.strftime("%d %b, %I:%M %p")
