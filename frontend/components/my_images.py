"""My Images page — gallery of generated images."""

from __future__ import annotations

import random

import streamlit as st

from services.session_service import get_session, get_session_image_base64, update_session
from state.session import format_session_date, get_favourite_images_count

_STARS_HTML = "".join(
    f'<div class="star" style="left:{random.randint(0,100)}%;top:{random.randint(0,100)}%;--dur:{random.uniform(2,6):.1f}s"></div>'
    for _ in range(40)
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

    col_all, col_fav = st.columns(2)
    with col_all:
        if st.button("All", key="filter_all", use_container_width=True):
            st.session_state["images_filter"] = "all"
            st.rerun()
    with col_fav:
        if st.button(f"Favourite ({get_favourite_images_count()})", key="filter_fav", use_container_width=True):
            st.session_state["images_filter"] = "favourite"
            st.rerun()

    if filter_type == "favourite":
        _render_favourite_images(sessions)
    else:
        filtered = _filter_sessions(sessions, "all")
        if not filtered:
            st.info("No images yet. Generate some on the Generate page.")
            return
        for session in filtered:
            _render_session_block(session, show_fav_btn=True)


def _truncate(s: str | None, max_len: int) -> str:
    text = (s or "")[:max_len]
    return f"{text}..." if len(s or "") > max_len else text


def _filter_sessions(sessions: list[dict], filter_type: str) -> list[dict]:
    if filter_type == "archived":
        return [s for s in sessions if s.get("archived", False)]
    return sessions


def _render_favourite_images(sessions: list[dict]) -> None:
    """Render only favourite images from all sessions."""
    fav_items: list[tuple[str, str, str, str]] = []
    for session in sessions:
        full = get_session(session.get("id", ""))
        if not full:
            continue
        fav_filenames = set(full.get("favourite_image_filenames", []))
        if not fav_filenames:
            continue
        for msg in full.get("messages", []):
            prompt = msg.get("prompt", "")
            for img in msg.get("images", []):
                fname = img.get("filename")
                if fname and fname in fav_filenames:
                    fav_items.append((
                        session["id"],
                        fname,
                        str(img.get("seed", "?")),
                        prompt,
                    ))
    if not fav_items:
        st.info("No favourite images. Click ♡ on images in the chat.")
        return
    cols_per_row = 4
    for i in range(0, len(fav_items), cols_per_row):
        cols = st.columns(cols_per_row, gap="small")
        for j, col in enumerate(cols):
            idx = i + j
            if idx >= len(fav_items):
                break
            session_id, fname, seed, prompt = fav_items[idx]
            with col:
                img_src = get_session_image_base64(session_id, fname)
                if img_src:
                    key_safe = "".join(c if c.isalnum() else "_" for c in f"{session_id}_{fname}")
                    st.markdown('<div class="like-on-image">', unsafe_allow_html=True)
                    st.image(img_src, use_container_width=True, caption=f"seed: {seed}")
                    full = get_session(session_id)
                    fav_filenames = list(full.get("favourite_image_filenames", []))
                    if st.button("♥", key=f"unfav_my_{key_safe}", help="Remove from favourites"):
                        fav_filenames.remove(fname)
                        if update_session(session_id, favourite_image_filenames=fav_filenames):
                            st.rerun()
                    st.markdown("</div>", unsafe_allow_html=True)
                else:
                    st.markdown(
                        '<div class="image-card"><div class="placeholder-orb"></div></div>',
                        unsafe_allow_html=True,
                    )


def _render_session_block(session: dict, show_fav_btn: bool = False) -> None:
    session_id = session.get("id")
    if not session_id:
        return

    full_session = get_session(session_id)
    if not full_session or "messages" not in full_session:
        return

    messages = full_session.get("messages", [])
    fav_filenames = set(full_session.get("favourite_image_filenames", []))
    all_images: list[tuple[str, str, str]] = []
    for msg in messages:
        prompt = msg.get("prompt", "")
        for img in msg.get("images", []):
            fname = img.get("filename")
            seed = str(img.get("seed", "?"))
            if fname:
                all_images.append((fname, seed, prompt))

    if not all_images:
        return

    date_str = format_session_date(session.get("created_at", ""))
    last_prompt = _truncate(session.get("last_prompt") or (all_images[-1][2] if all_images else ""), 120)
    st.markdown(
        f'<div class="prompt-info">'
        f'<div class="session-title"><strong>{session.get("title", "Session")}</strong> • {date_str} • {len(all_images)} image(s)</div>'
        f'<div class="helper-specs">{last_prompt}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )

    cols_per_row = 4
    for i in range(0, len(all_images), cols_per_row):
        cols = st.columns(cols_per_row, gap="small")
        for j, col in enumerate(cols):
            idx = i + j
            if idx >= len(all_images):
                break
            fname, seed, prompt = all_images[idx]
            with col:
                img_src = get_session_image_base64(session_id, fname)
                if img_src:
                    key_safe = "".join(c if c.isalnum() else "_" for c in f"{session_id}_{fname}")
                    st.markdown('<div class="like-on-image">', unsafe_allow_html=True)
                    st.image(img_src, use_container_width=True, caption=f"seed: {seed}")
                    if show_fav_btn:
                        is_fav = fname in fav_filenames
                        if st.button(
                            "♥" if is_fav else "♡",
                            key=f"fav_my_{key_safe}",
                            help="Toggle favourite",
                        ):
                            new_fav = list(fav_filenames)
                            if is_fav:
                                new_fav.remove(fname)
                            else:
                                new_fav.append(fname)
                            if update_session(session_id, favourite_image_filenames=new_fav):
                                st.rerun()
                    st.markdown("</div>", unsafe_allow_html=True)
                else:
                    st.markdown(
                        '<div class="image-card"><div class="placeholder-orb"></div></div>',
                        unsafe_allow_html=True,
                    )
