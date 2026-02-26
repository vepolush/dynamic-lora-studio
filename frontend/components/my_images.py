"""My Images page — gallery of generated images."""

from __future__ import annotations

import random

import streamlit as st

from state.session import format_session_date, get_active_session, get_favourite_count

_STARS_HTML = "".join(
    f'<div class="star" style="left:{random.randint(0,100)}%;top:{random.randint(0,100)}%;--dur:{random.uniform(2,6):.1f}s"></div>'
    for _ in range(40)
)

_PLACEHOLDER_CARD = (
    '<div class="image-card">'
    '<div class="placeholder-orb"></div>'
    '<div class="image-card-actions">'
    "<button>♡ Variation</button>"
    "<button>⊕ Upscale</button>"
    "</div>"
    "</div>"
)


def render_my_images() -> None:
    st.markdown(f'<div class="stars-layer">{_STARS_HTML}</div>', unsafe_allow_html=True)

    sessions = st.session_state.get("sessions", [])
    filter_type = st.session_state.get("images_filter", "all")

    st.markdown(
        '<div class="session-header">'
        "<h3>My Images</h3>"
        "</div>",
        unsafe_allow_html=True,
    )

    col_all, col_fav, col_arch = st.columns(3)
    with col_all:
        if st.button("All", key="filter_all", use_container_width=True):
            st.session_state["images_filter"] = "all"
            st.rerun()
    with col_fav:
        if st.button(f"Favourite ({get_favourite_count()})", key="filter_fav", use_container_width=True):
            st.session_state["images_filter"] = "favourite"
            st.rerun()
    with col_arch:
        if st.button("Archived", key="filter_arch", use_container_width=True):
            st.session_state["images_filter"] = "archived"
            st.rerun()

    filtered = _filter_sessions(sessions, filter_type)

    if not filtered:
        st.info("No images yet. Generate some on the Generate page.")
        return

    for session in filtered:
        _render_session_block(session)


def _truncate(s: str | None, max_len: int) -> str:
    text = (s or "")[:max_len]
    return f"{text}..." if len(s or "") > max_len else text


def _filter_sessions(sessions: list[dict], filter_type: str) -> list[dict]:
    if filter_type == "favourite":
        return [s for s in sessions if s.get("favourite")]
    if filter_type == "archived":
        return [s for s in sessions if s.get("archived", False)]
    return sessions


def _render_session_block(session: dict) -> None:
    date_str = format_session_date(session["created_at"])
    st.markdown(
        f'<div class="prompt-info">'
        f'<div class="session-title"><strong>{session["title"]}</strong> • {date_str}</div>'
        f'<div class="helper-specs">{_truncate(session.get("prompt"), 120)}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )

    image_count = session.get("images", 4)
    cols = st.columns(min(4, image_count), gap="small")
    for i in range(image_count):
        with cols[i % len(cols)]:
            st.markdown(_PLACEHOLDER_CARD, unsafe_allow_html=True)
