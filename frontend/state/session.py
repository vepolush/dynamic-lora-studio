"""Session state initialization and helpers."""

from __future__ import annotations

from datetime import datetime

import streamlit as st

MOCK_SESSIONS: list[dict] = [
    {
        "id": "sess_001",
        "title": "3D sphere",
        "created_at": "2026-02-26T12:00:00",
        "message_count": 2,
        "favourite": False,
    },
    {
        "id": "sess_002",
        "title": "Portrait study",
        "created_at": "2026-02-25T18:30:00",
        "message_count": 1,
        "favourite": True,
    },
    {
        "id": "sess_003",
        "title": "Landscape",
        "created_at": "2026-02-24T09:15:00",
        "message_count": 3,
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
    "steps": 0,
    "guidance_scale": 7.5,
    "image_size": "512x512",
    "seed": -1,
    "quality": "Normal",
    "style": "None",
    "lightning": "None",
    "color": "Default",
    "scheduler": "Auto",
    "num_images": 1,
    "negative_prompt": "",
}

SCHEDULERS = [
    "Auto",
    "DPM++ 2M Karras",
    "DPM++ 2M SDE Karras",
    "Euler",
    "Euler a",
    "DDIM",
    "PNDM",
]

SCHEDULER_MAP: dict[str, str] = {
    "Auto": "",
    "DPM++ 2M Karras": "dpm++_2m_karras",
    "DPM++ 2M SDE Karras": "dpm++_2m_sde_karras",
    "Euler": "euler",
    "Euler a": "euler_a",
    "DDIM": "ddim",
    "PNDM": "pndm",
}


def init_session_state() -> None:
    """Populate st.session_state with defaults if keys are missing."""
    if "sessions" not in st.session_state or "entities" not in st.session_state:
        with st.spinner("Loading..."):
            if "sessions" not in st.session_state:
                from services.session_service import get_sessions
                st.session_state["sessions"] = get_sessions()
            if "entities" not in st.session_state:
                from services.entity_service import get_entities
                st.session_state["entities"] = get_entities()

    sessions = st.session_state["sessions"]
    if not sessions:
        from services.session_service import create_session
        new_sess = create_session("New session")
        if new_sess:
            st.session_state["sessions"] = [new_sess]
            sessions = [new_sess]
    active_id = sessions[0]["id"] if sessions else None

    defaults = {
        "current_page": "generate",
        "active_session_id": active_id,
        "generation_settings": {**DEFAULT_SETTINGS},
        "active_entity_id": None,
        "lora_strength": 0.8,
        "show_entity_form": False,
        "chat_messages": {},
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


def get_session_messages(session_id: str) -> list[dict]:
    """Get messages for a session from local state, loading from backend if needed."""
    chat_messages = st.session_state.get("chat_messages", {})
    if session_id in chat_messages:
        return chat_messages[session_id]

    from services.session_service import get_session
    session_data = get_session(session_id)
    if session_data and "messages" in session_data:
        msgs = session_data["messages"]
        chat_messages[session_id] = msgs
        st.session_state["chat_messages"] = chat_messages
        return msgs

    chat_messages[session_id] = []
    st.session_state["chat_messages"] = chat_messages
    return []


def add_chat_message(session_id: str, message: dict) -> None:
    """Add a message to the local chat history for a session."""
    chat_messages = st.session_state.get("chat_messages", {})
    if session_id not in chat_messages:
        chat_messages[session_id] = []
    chat_messages[session_id].append(message)
    st.session_state["chat_messages"] = chat_messages


def get_favourite_count() -> int:
    return sum(1 for s in st.session_state.get("sessions", []) if s.get("favourite"))


def format_session_date(iso_str: str) -> str:
    dt = datetime.fromisoformat(iso_str)
    today = datetime.now().date()
    if dt.date() == today:
        return f"Today at {dt.strftime('%I:%M %p')}"
    return dt.strftime("%d %b, %I:%M %p")
