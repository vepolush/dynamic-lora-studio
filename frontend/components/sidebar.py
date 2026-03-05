"""Left sidebar: navigation, sessions list, profile."""

from __future__ import annotations

import streamlit as st

from state.session import get_archived_count, get_favourite_count

NAV_ITEMS = [
    ("generate", "Generate", ":material/auto_awesome:"),
    ("my_images", "My Images", ":material/photo_library:"),
    ("gallery", "Gallery", ":material/image:"),
    ("settings", "Settings", ":material/settings:"),
]


def render_sidebar() -> None:
    from services.auth_service import is_logged_in

    with st.sidebar:
        st.markdown(
            '<div class="sidebar-logo">✦ <span>Dynamic LoRA Studio</span></div>',
            unsafe_allow_html=True,
        )

        if is_logged_in():
            for page_id, label, icon in NAV_ITEMS:
                _nav_button(page_id, label, icon)

            st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

            sessions = st.session_state.get("sessions", [])
            non_archived = [s for s in sessions if not s.get("archived")]
            recent = non_archived[:5]
            st.markdown('<div class="sidebar-section">Recent Sessions</div>', unsafe_allow_html=True)
            for session in recent:
                _session_item(session, len(recent))

            st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

            st.markdown('<div class="sidebar-section">Chat List</div>', unsafe_allow_html=True)
            _session_category_clickable("All chats", len(non_archived), "all")
            _session_category_clickable("Favourite", get_favourite_count(), "favourite")
            _session_category_clickable("Archived", get_archived_count(), "archived")

            st.markdown('<div class="new-chat-btn">', unsafe_allow_html=True)
            if st.button("＋ New Chat", key="new_chat_btn", use_container_width=True):
                _handle_new_chat()
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            _nav_button("gallery", "Gallery", ":material/image:")
            _nav_button("settings", "Settings", ":material/settings:")

        from components.auth import render_auth_sidebar
        st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
        render_auth_sidebar()


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


def _session_category_clickable(label: str, count: int, filter_key: str) -> None:
    """Clickable Chat List item - opens dialog with scrollable session list."""
    if st.button(
        f"• {label} ({count})",
        key=f"chat_list_{filter_key}",
        use_container_width=True,
    ):
        st.session_state["chat_list_filter"] = filter_key
        st.rerun()

    if st.session_state.get("chat_list_filter") == filter_key:
        _show_chat_list_dialog(filter_key)
        st.session_state.pop("chat_list_filter", None)


@st.dialog("Chat List")
def _show_chat_list_dialog(filter_key: str) -> None:
    """Show scrollable list of sessions based on filter: all, favourite, archived."""
    sessions = st.session_state.get("sessions", [])
    if filter_key == "all":
        filtered = [s for s in sessions if not s.get("archived")]
    elif filter_key == "favourite":
        filtered = [s for s in sessions if s.get("favourite") is True]
    else:
        filtered = [s for s in sessions if s.get("archived") is True]

    if not filtered:
        st.info("No sessions" if filter_key == "all" else f"No {filter_key} sessions")
    else:
        scroll_container = st.container(height=300)
        with scroll_container:
            for session in filtered:
                if st.button(
                    f"  {session.get('title', 'Session')}",
                    key=f"chat_list_sess_{session['id']}",
                    use_container_width=True,
                ):
                    st.session_state["active_session_id"] = session["id"]
                    st.session_state.pop("chat_list_filter", None)
                    st.rerun()


def _handle_new_chat() -> None:
    from services.session_service import create_session

    with st.spinner("Creating session..."):
        new_sess = create_session("New session")
    if new_sess:
        st.session_state["sessions"] = [new_sess] + st.session_state.get("sessions", [])
        st.session_state["active_session_id"] = new_sess["id"]
        st.toast("New session created")
    else:
        st.error("Could not create session. Login required or check backend in Settings.")
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
