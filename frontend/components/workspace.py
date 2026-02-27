"""Center panel: session header, prompt info, image grid, chat input."""

from __future__ import annotations

import random

import streamlit as st

from state.session import format_session_date, get_active_session


def _render_no_session() -> None:
    """Show create-session prompt when no sessions exist."""
    if st.button("＋ New Session", key="workspace_new_session", type="primary"):
        from services.session_service import create_session

        with st.spinner("Creating session..."):
            new_sess = create_session("New session")
        if new_sess:
            st.session_state["sessions"] = [new_sess] + st.session_state.get("sessions", [])
            st.session_state["active_session_id"] = new_sess["id"]
            st.toast("Session created")
            st.rerun()
        else:
            st.error("Could not create session. Check backend in Settings.")

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


def render_workspace() -> None:
    st.markdown(f'<div class="stars-layer">{_STARS_HTML}</div>', unsafe_allow_html=True)

    session = get_active_session()
    if session is None:
        _render_no_session()
        return

    _render_session_header(session)
    _render_prompt_info(session)
    _render_image_grid(session)
    _render_prompt_input()


def _render_session_header(session: dict) -> None:
    date_str = format_session_date(session["created_at"])
    st.markdown(
        f'<div class="session-header">'
        f'<div class="session-title"><strong>{session["title"]}</strong> • {date_str}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )


def _render_prompt_info(session: dict) -> None:
    st.markdown(
        f'<div class="prompt-info">'
        f'<div class="bot-label">'
        f"✦ Dynamic LoRA Studio "
        f'<span class="bot-badge">BOT</span> '
        f"{format_session_date(session['created_at'])}"
        f"</div>"
        f'<div class="prompt-text">Prompt: {session["prompt"]}</div>'
        f'<div class="helper-specs">Helper specs applied: {session["helper_specs"]}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )


def _render_image_grid(session: dict) -> None:
    image_count = session.get("images", 4)
    cards = _PLACEHOLDER_CARD * image_count
    st.markdown(f'<div class="image-grid-wrapper">{cards}</div>', unsafe_allow_html=True)


def _render_prompt_input() -> None:
    from services.generation_service import generate

    prompt = st.chat_input("Describe the image you want to generate")
    if prompt:
        session = get_active_session()
        if not session:
            st.error("No active session.")
            return

        settings = st.session_state.get("generation_settings", {})
        size = settings.get("image_size", "512x512")
        w, h = (int(x) for x in size.split("x"))

        with st.spinner("Generating image..."):
            result = generate(
                session_id=session["id"],
                prompt=prompt,
                steps=settings.get("steps", 25),
                guidance_scale=settings.get("guidance_scale", 7.5),
                width=w,
                height=h,
                seed=settings.get("seed", -1),
                entity_id=st.session_state.get("active_entity_id"),
                lora_strength=st.session_state.get("lora_strength", 0.8),
                style=settings.get("style"),
                lightning=settings.get("lightning"),
                color=settings.get("color"),
            )
        if result:
            st.session_state["last_prompt"] = prompt
            st.toast("Generation started.")
            st.rerun()
        else:
            st.error("Backend unavailable. Check Settings.")
