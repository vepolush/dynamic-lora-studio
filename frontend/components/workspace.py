"""Center panel: session header, chat history with images, prompt input."""

from __future__ import annotations

import html
import json
import random
from datetime import datetime

import streamlit as st

from state.session import (
    add_chat_message,
    format_session_date,
    get_active_session,
    get_session_messages,
)


def _render_no_session() -> None:
    st.info("No sessions. Creating one...")
    if st.button("+ New Session", key="workspace_new_session", type="primary"):
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


def render_workspace() -> None:
    st.markdown(f'<div class="stars-layer">{_STARS_HTML}</div>', unsafe_allow_html=True)

    session = get_active_session()
    if session is None:
        _render_no_session()
        return

    _render_session_header(session)
    _render_chat_history(session)
    _render_prompt_input(session)


def _render_session_header(session: dict) -> None:
    date_str = format_session_date(session["created_at"])
    msg_count = session.get("message_count", 0)
    count_label = f" · {msg_count} generation(s)" if msg_count else ""
    st.markdown(
        f'<div class="session-header">'
        f'<div class="session-title"><strong>{session["title"]}</strong> · {date_str}{count_label}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )


def _render_chat_history(session: dict) -> None:
    """Render all messages in the session as a scrollable chat."""
    messages = get_session_messages(session["id"])

    if not messages:
        st.markdown(
            '<div class="chat-empty">'
            '<div class="chat-empty-icon">✦</div>'
            '<div class="chat-empty-text">Enter a prompt below to generate images</div>'
            "</div>",
            unsafe_allow_html=True,
        )
        return

    chat_height = st.session_state.get("chat_height", 520)
    heights = [400, 520, 700, 900, 1100]
    idx = heights.index(chat_height) if chat_height in heights else 1
    new_h = st.selectbox(
        "Chat height",
        options=heights,
        index=idx,
        key="chat_height_select",
    )
    if new_h != chat_height:
        st.session_state["chat_height"] = new_h
        st.rerun()

    chat_container = st.container(height=chat_height)
    with chat_container:
        for msg in messages:
            _render_message(msg, session["id"])


def _render_message(msg: dict, session_id: str) -> None:
    """Render a single chat message (user prompt + generated images)."""
    prompt = msg.get("prompt", "")
    enhanced = msg.get("enhanced_prompt", "")
    images = msg.get("images", [])
    gen_time = msg.get("generation_time", 0)
    timestamp = msg.get("timestamp", "")
    settings = msg.get("settings", {})

    time_str = ""
    if timestamp:
        try:
            dt = datetime.fromisoformat(timestamp)
            time_str = dt.strftime("%H:%M")
        except ValueError:
            time_str = ""

    specs_parts = []
    if settings.get("style") and settings["style"] != "None":
        specs_parts.append(settings["style"])
    if settings.get("lighting") and settings["lighting"] != "None":
        specs_parts.append(settings["lighting"])
    if settings.get("color") and settings["color"] != "Default":
        specs_parts.append(settings["color"])
    specs_line = " · ".join(specs_parts) if specs_parts else ""

    settings_str = json.dumps(settings, ensure_ascii=False, indent=2) if settings else "{}"
    prompt_esc = html.escape(prompt or "", quote=True)
    settings_esc = html.escape(settings_str, quote=True)

    st.markdown(
        f'<div class="msg-with-actions">'
        f'<div class="chat-msg user-msg">'
        f'<div class="msg-header">'
        f'<span class="msg-author">You</span>'
        f'<span class="msg-time">{time_str}</span>'
        f'<span class="msg-copy-btns">'
        f'<button class="copy-btn" data-text="{prompt_esc}" title="Copy prompt" onclick="navigator.clipboard.writeText(this.dataset.text);this.textContent=\'✓\';setTimeout(()=>this.textContent=\'📋\',800)">📋</button>'
        f'<button class="copy-btn" data-text="{settings_esc}" title="Copy settings" onclick="navigator.clipboard.writeText(this.dataset.text);this.textContent=\'✓\';setTimeout(()=>this.textContent=\'⚙️\',800)">⚙️</button>'
        f"</span></div>"
        f'<div class="msg-prompt">{html.escape(prompt or "")}</div>'
        f"</div></div>",
        unsafe_allow_html=True,
    )

    gen_info = f"{gen_time:.1f}s" if gen_time else ""
    num_imgs = len(images)
    meta = f"{num_imgs} image(s)"
    if gen_info:
        meta += f" · {gen_info}"
    if specs_line:
        meta += f" · {specs_line}"

    st.markdown(
        f'<div class="chat-msg bot-msg">'
        f'<div class="msg-header">'
        f'<span class="msg-author">✦ Studio <span class="bot-badge">BOT</span></span>'
        f'<span class="msg-time">{meta}</span>'
        f"</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    if images:
        _render_message_images(session_id, images)

    if enhanced and enhanced != prompt:
        with st.expander("Enhanced prompt", expanded=False):
            st.caption(enhanced)


def _render_message_images(session_id: str, images: list[dict]) -> None:
    """Render images from base64 data or by fetching from backend when missing."""
    from services.session_service import get_session_image_base64

    n = len(images)
    cols_count = min(n, 2) if n <= 4 else 2
    cols = st.columns(cols_count, gap="small")

    for i, img_data in enumerate(images):
        with cols[i % cols_count]:
            b64_uri = img_data.get("base64")
            if b64_uri and isinstance(b64_uri, str) and b64_uri.startswith("data:"):
                img_src = b64_uri
            elif b64_uri and isinstance(b64_uri, str):
                img_src = f"data:image/png;base64,{b64_uri}"
            else:
                filename = img_data.get("filename")
                if filename:
                    img_src = get_session_image_base64(session_id, filename)
                else:
                    img_src = None
            seed = img_data.get("seed", "?")
            if img_src:
                st.image(
                    img_src,
                    use_container_width=True,
                    caption=f"seed: {seed}",
                )
            else:
                st.markdown(
                    '<div class="image-card"><div class="placeholder-orb"></div></div>',
                    unsafe_allow_html=True,
                )


def _render_prompt_input(session: dict) -> None:
    from services.generation_service import generate
    from components.prompt_helper import (
        get_current_settings,
        build_helper_specs,
        build_effective_prompt,
    )

    prompt = st.chat_input("Describe the image you want to generate")
    if prompt:
        settings = get_current_settings()
        size = settings.get("image_size", "512x512")
        w, h = (int(x) for x in size.split("x"))
        effective_prompt = build_effective_prompt(prompt)
        helper_specs = build_helper_specs()

        steps_val = settings.get("steps", 0)

        with st.spinner("Generating image..."):
            result = generate(
                session_id=session["id"],
                prompt=effective_prompt,
                negative_prompt=settings.get("negative_prompt", ""),
                steps=steps_val if steps_val > 0 else None,
                guidance_scale=settings.get("guidance_scale", 7.5),
                width=w,
                height=h,
                seed=settings.get("seed", -1),
                num_images=settings.get("num_images", 1),
                scheduler=settings.get("scheduler"),
                quality=settings.get("quality", "Normal"),
                entity_id=st.session_state.get("active_entity_id"),
                entity_version=st.session_state.get("entity_version_select"),
                lora_strength=st.session_state.get("lora_strength_slider", st.session_state.get("lora_strength", 0.8)),
                style=settings.get("style"),
                lighting=settings.get("lightning"),
                color=settings.get("color"),
            )

        if result and result.get("status") == "success":
            message = result.get("message", {})
            if not message:
                message = {
                    "prompt": effective_prompt,
                    "enhanced_prompt": result.get("enhanced_prompt", effective_prompt),
                    "images": result.get("images", []),
                    "generation_time": result.get("generation_time", 0),
                    "settings": {
                        "style": settings.get("style"),
                        "lighting": settings.get("lightning"),
                        "color": settings.get("color"),
                    },
                }

            for img in result.get("images", []):
                if "base64" in img:
                    for m_img in message.get("images", []):
                        if m_img.get("filename") == img.get("filename") or m_img.get("seed") == img.get("seed"):
                            m_img["base64"] = img["base64"]
                            break

            add_chat_message(session["id"], message)

            for s in st.session_state.get("sessions", []):
                if s["id"] == session["id"]:
                    s["message_count"] = s.get("message_count", 0) + 1
                    s["last_prompt"] = effective_prompt
                    break

            st.toast("Generation complete!")
            st.rerun()
        else:
            st.error("Backend unavailable. Check Settings.")
