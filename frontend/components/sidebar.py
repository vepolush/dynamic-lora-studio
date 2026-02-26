"""Left sidebar: navigation, sessions list, profile."""

from __future__ import annotations

import streamlit as st

from state.session import get_favourite_count

NAV_ITEMS = [
    ("generate", "Generate", ":material/auto_awesome:"),
    ("my_images", "My Images", ":material/photo_library:"),
    ("settings", "Settings", ":material/settings:"),
]


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown(
            '<div class="sidebar-logo">✦ <span>Dynamic LoRA Studio</span></div>',
            unsafe_allow_html=True,
        )

        for page_id, label, icon in NAV_ITEMS:
            _nav_button(page_id, label, icon)

        st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

        sessions = st.session_state.get("sessions", [])
        st.markdown('<div class="sidebar-section">Recent Sessions</div>', unsafe_allow_html=True)
        for session in sessions:
            _session_item(session, len(sessions))

        st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

        st.markdown('<div class="sidebar-section">Chat List</div>', unsafe_allow_html=True)
        _session_category("All chats", len(sessions))
        _session_category("Favourite", get_favourite_count())
        _session_category("Archived", 0)

        st.markdown('<div class="new-chat-btn">', unsafe_allow_html=True)
        if st.button("＋ New Chat", key="new_chat_btn", use_container_width=True):
            _handle_new_chat()
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            '<div class="profile-block">'
            '<div class="profile-avatar">U</div>'
            '<div class="profile-name">User</div>'
            "</div>",
            unsafe_allow_html=True,
        )


def _nav_button(page_id: str, label: str, icon: str) -> None:
    current = st.session_state.get("current_page", "generate")
    is_active = current == page_id
    outer = "nav-active" if is_active else ""

    st.markdown(f'<div class="{outer}"><div class="nav-btn">', unsafe_allow_html=True)
    if st.button(label, key=f"nav_{page_id}", use_container_width=True, icon=icon):
        st.session_state["current_page"] = page_id
        st.rerun()
    st.markdown("</div></div>", unsafe_allow_html=True)


def _session_category(label: str, count: int) -> None:
    st.markdown(
        f'<div class="session-item">'
        f"<span>• {label}</span>"
        f'<span class="session-count">{count}</span>'
        f"</div>",
        unsafe_allow_html=True,
    )


def _handle_new_chat() -> None:
    from services.session_service import create_session

    with st.spinner("Creating session..."):
        new_sess = create_session("New session")
    if new_sess:
        st.session_state["sessions"] = [new_sess] + st.session_state.get("sessions", [])
        st.session_state["active_session_id"] = new_sess["id"]
        st.toast("New session created")
    else:
        st.error("Could not create session. Check backend in Settings.")
    st.rerun()


def _session_item(session: dict, total: int) -> None:
    is_active = st.session_state.get("active_session_id") == session["id"]
    if st.button(
        f"  {session['title']}",
        key=f"session_btn_{session['id']}",
        use_container_width=True,
        type="primary" if is_active else "secondary",
    ):
        st.session_state["active_session_id"] = session["id"]
        st.rerun()
