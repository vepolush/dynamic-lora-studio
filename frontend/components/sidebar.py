"""Left sidebar: navigation, sessions list, profile."""

from __future__ import annotations

import streamlit as st

from state.session import get_favourite_count

_ICON_GENERATE = (
    '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
    'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5"/>'
    '<path d="M2 12l10 5 10-5"/></svg>'
)
_ICON_IMAGES = (
    '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
    'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
    '<rect x="3" y="3" width="18" height="18" rx="2"/>'
    '<circle cx="8.5" cy="8.5" r="1.5"/>'
    '<path d="M21 15l-5-5L5 21"/></svg>'
)
_ICON_EXPLORE = (
    '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
    'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
    '<circle cx="12" cy="12" r="10"/>'
    '<polygon points="16.24 7.76 14.12 14.12 7.76 16.24 9.88 9.88 16.24 7.76"/></svg>'
)
_ICON_SETTINGS = (
    '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
    'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
    '<circle cx="12" cy="12" r="3"/>'
    '<path d="M19.4 15a1.65 1.65 0 00.33 1.82l.06.06a2 2 0 01-2.83 2.83l-.06-.06a1.65 1.65 0 00-1.82-.33 '
    '1.65 1.65 0 00-1 1.51V21a2 2 0 01-4 0v-.09A1.65 1.65 0 009 19.4a1.65 1.65 0 00-1.82.33l-.06.06a2 2 0 '
    '01-2.83-2.83l.06-.06A1.65 1.65 0 004.68 15a1.65 1.65 0 00-1.51-1H3a2 2 0 010-4h.09A1.65 1.65 0 004.6 '
    '9a1.65 1.65 0 00-.33-1.82l-.06-.06a2 2 0 012.83-2.83l.06.06A1.65 1.65 0 009 4.68a1.65 1.65 0 001-1.51V3a2 '
    '2 0 014 0v.09a1.65 1.65 0 001 1.51 1.65 1.65 0 001.82-.33l.06-.06a2 2 0 012.83 2.83l-.06.06A1.65 1.65 0 '
    '0019.4 9a1.65 1.65 0 001.51 1H21a2 2 0 010 4h-.09a1.65 1.65 0 00-1.51 1z"/></svg>'
)

NAV_ITEMS = [
    ("generate", "Generate", _ICON_GENERATE),
    ("my_images", "My Images", _ICON_IMAGES),
    ("explore", "Explore", _ICON_EXPLORE),
    ("settings", "Settings", _ICON_SETTINGS),
]


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown(
            '<div class="sidebar-logo">✦ <span>Dynamic LoRA Studio</span></div>',
            unsafe_allow_html=True,
        )

        for page_id, label, icon_svg in NAV_ITEMS:
            _nav_button(page_id, label, icon_svg)

        st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

        st.markdown('<div class="sidebar-section">Chat List</div>', unsafe_allow_html=True)

        sessions = st.session_state.get("sessions", [])
        _session_category("All chats", len(sessions))
        _session_category("Favourite", get_favourite_count())
        _session_category("Archived", 0)

        st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

        st.markdown('<div class="new-chat-btn">', unsafe_allow_html=True)
        st.button("＋ New Chat", key="new_chat_btn", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="sidebar-section">Recent Sessions</div>', unsafe_allow_html=True)

        for session in sessions:
            _session_item(session)

        st.markdown(
            '<div class="profile-block">'
            '<div class="profile-avatar">U</div>'
            '<div class="profile-name">User</div>'
            "</div>",
            unsafe_allow_html=True,
        )


def _nav_button(page_id: str, label: str, icon_svg: str) -> None:
    current = st.session_state.get("current_page", "generate")
    is_active = current == page_id
    outer = "nav-active" if is_active else ""

    st.markdown(f'<div class="{outer}"><div class="nav-btn">', unsafe_allow_html=True)
    if st.button(f"  {label}", key=f"nav_{page_id}", use_container_width=True, icon=None):
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


def _session_item(session: dict) -> None:
    is_active = st.session_state.get("active_session_id") == session["id"]
    weight = "600" if is_active else "400"
    indicator = "▸ " if is_active else "  "
    color = "var(--accent-light)" if is_active else "inherit"
    st.markdown(
        f'<div class="session-item" style="font-weight:{weight};color:{color}">'
        f"{indicator}{session['title']}"
        f"</div>",
        unsafe_allow_html=True,
    )
