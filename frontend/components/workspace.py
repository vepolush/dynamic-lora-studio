"""Center panel: session header, chat history with images, prompt input."""

from __future__ import annotations

import base64
import html
import json
import random
from urllib.parse import unquote
from datetime import datetime

import streamlit as st

from state.session import (
    SCHEDULER_MAP,
    add_chat_message,
    format_session_date,
    get_active_session,
    get_session_messages,
)


def _render_login_required() -> None:
    """Show login prompt when user is not authenticated."""
    st.markdown(
        '<div class="login-required-box">'
        '<div class="login-required-icon">🔐</div>'
        '<div class="login-required-title">Log in to continue</div>'
        '<div class="login-required-text">Create sessions, generate images, and manage your LoRAs.</div>'
        '<div class="login-required-hint">Use the form in the sidebar to log in or register.</div>'
        "</div>",
        unsafe_allow_html=True,
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


def _scheduler_display_from_key(scheduler_key: str | None) -> str:
    for label, value in SCHEDULER_MAP.items():
        if value == (scheduler_key or ""):
            return label
    return "Auto"


def _apply_message_settings(settings: dict) -> None:
    """Apply message settings to Prompt Helper widgets."""
    if not settings:
        return

    width = settings.get("width")
    height = settings.get("height")
    if isinstance(width, int) and isinstance(height, int):
        size = f"{width}x{height}"
        if size in {"512x512", "640x640", "512x768", "512x896", "768x512", "896x512"}:
            st.session_state["size_select"] = size

    if settings.get("steps") is not None:
        st.session_state["steps_slider"] = int(settings.get("steps", 0))
    if settings.get("guidance_scale") is not None:
        st.session_state["guidance_slider"] = float(settings.get("guidance_scale", 7.5))
    if settings.get("seed") is not None:
        st.session_state["seed_input"] = int(settings.get("seed", -1))
    else:
        st.session_state["seed_input"] = -1
    if settings.get("num_images") is not None:
        st.session_state["num_images_input"] = int(settings.get("num_images", 1))
    if settings.get("quality") is not None:
        st.session_state["quality_slider"] = settings.get("quality", "Normal")

    st.session_state["scheduler_select"] = _scheduler_display_from_key(settings.get("scheduler"))
    st.session_state["style_select"] = settings.get("style") or "None"
    st.session_state["lighting_select"] = settings.get("lighting") or "None"
    st.session_state["color_select"] = settings.get("color") or "Default"

    st.session_state["generation_settings"] = {
        **st.session_state.get("generation_settings", {}),
        "steps": st.session_state.get("steps_slider", 0),
        "guidance_scale": st.session_state.get("guidance_slider", 7.5),
        "image_size": st.session_state.get("size_select", "512x512"),
        "seed": st.session_state.get("seed_input", -1),
        "num_images": st.session_state.get("num_images_input", 1),
        "quality": st.session_state.get("quality_slider", "Normal"),
        "style": st.session_state.get("style_select", "None"),
        "lighting": st.session_state.get("lighting_select", "None"),
        "color": st.session_state.get("color_select", "Default"),
        "scheduler": st.session_state.get("scheduler_select", "Auto"),
    }


def _apply_settings_from_query() -> None:
    payload = st.query_params.get("apply_settings_b64")
    if not payload:
        return
    if isinstance(payload, list):
        payload = payload[0] if payload else ""

    try:
        decoded = base64.b64decode(unquote(str(payload))).decode("utf-8")
        settings = json.loads(decoded)
        if isinstance(settings, dict):
            _apply_message_settings(settings)
            st.toast("Settings applied to Prompt Helper")
    except Exception:
        st.toast("Failed to apply settings")
    finally:
        if "apply_settings_b64" in st.query_params:
            del st.query_params["apply_settings_b64"]
        st.rerun()


def render_workspace() -> None:
    from services.auth_service import is_logged_in

    _apply_settings_from_query()
    st.markdown(f'<div class="stars-layer">{_STARS_HTML}</div>', unsafe_allow_html=True)

    if not is_logged_in():
        _render_login_required()
        return

    session = get_active_session()
    if session is None:
        _render_no_session()
        return

    _render_session_header(session)
    _render_chat_history(session)
    _render_prompt_input(session)


def _render_session_header(session: dict) -> None:
    from services.session_service import delete_session, get_sessions, update_session

    date_str = format_session_date(session["created_at"])
    msg_count = session.get("message_count", 0)
    count_label = f" · {msg_count} generation(s)" if msg_count else ""
    sid = session["id"]
    is_fav = session.get("favourite", False)
    is_archived = session.get("archived", False)

    col_title, col_fav, col_arch, col_del = st.columns([6, 1, 1, 1])
    with col_title:
        st.markdown(
            f'<div class="session-title"><strong>{session["title"]}</strong> · {date_str}{count_label}</div>',
            unsafe_allow_html=True,
        )
    with col_fav:
        if st.button("★" if is_fav else "☆", key=f"session_fav_{sid}", help="Favourite"):
            if update_session(sid, favourite=not is_fav):
                st.session_state["sessions"] = get_sessions()
                st.rerun()
    with col_arch:
        if st.button("📦" if is_archived else "📁", key=f"session_arch_{sid}", help="Archive"):
            if update_session(sid, archived=not is_archived):
                st.session_state["sessions"] = get_sessions()
                st.rerun()
    with col_del:
        if st.button("🗑", key=f"session_del_{sid}", help="Delete"):
            if delete_session(sid):
                st.session_state["sessions"] = [s for s in st.session_state.get("sessions", []) if s["id"] != sid]
                if st.session_state.get("active_session_id") == sid:
                    sess = st.session_state["sessions"]
                    st.session_state["active_session_id"] = sess[0]["id"] if sess else None
                st.toast("Session deleted")
                st.rerun()


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
    prompt_b64 = base64.b64encode((prompt or "").encode("utf-8")).decode("ascii")
    settings_b64 = base64.b64encode(settings_str.encode("utf-8")).decode("ascii")
    msg_id = f"msg-copy-{msg.get('id', abs(hash(prompt + time_str)) % 10**8)}".replace("'", "")

    copy_html = (
        f'<div class="msg-with-actions" id="{msg_id}">'
        '<div class="chat-msg user-msg">'
        '<div class="msg-header">'
        f'<span class="msg-author">You</span>'
        f'<span class="msg-time">{html.escape(time_str)}</span>'
        "</div>"
        '<div class="msg-copy-row">'
        f'<button class="copy-btn" data-b64="{prompt_b64}" data-icon="📋" title="Copy prompt">📋</button>'
        f'<button class="copy-btn apply-style-btn" data-b64="{settings_b64}" title="Apply settings">⚙️</button>'
        "</div>"
        f'<div class="msg-prompt">{html.escape(prompt or "")}</div>'
        "</div></div>"
        "<script>"
        "(function(){"
        f"var c=document.getElementById('{msg_id}');if(c){{"
        "var copyBtn=c.querySelector('.copy-btn:not(.apply-style-btn)');"
        "if(copyBtn){copyBtn.onclick=function(){var btn=this;var t=atob(btn.getAttribute('data-b64'));"
        "if(navigator.clipboard){navigator.clipboard.writeText(t).then(function(){btn.textContent='✓';setTimeout(function(){btn.textContent=btn.getAttribute('data-icon');},600);});}"
        "else{var ta=document.createElement('textarea');ta.value=t;ta.style.position='fixed';ta.style.opacity='0';document.body.appendChild(ta);ta.select();try{document.execCommand('copy');}catch(e){}document.body.removeChild(ta);btn.textContent='✓';setTimeout(function(){btn.textContent=btn.getAttribute('data-icon');},600);}"
        "};}"
        "var applyBtn=c.querySelector('.apply-style-btn');"
        "if(applyBtn){applyBtn.onclick=function(){var u=new URL(window.location.href);u.searchParams.set('apply_settings_b64', encodeURIComponent(this.getAttribute('data-b64')));window.location.assign(u.toString());};}"
        "}"
        "})();"
        "</script>"
    )
    st.html(copy_html, unsafe_allow_javascript=True)

    gen_info = f"{gen_time:.1f}s" if gen_time else ""
    num_imgs = len(images)
    meta = f"{num_imgs} image(s)"
    if gen_info:
        meta += f" · {gen_info}"
    if specs_line:
        meta += f" · {specs_line}"

    model_name = "Stable Diffusion 1.5"
    st.markdown(
        f'<div class="chat-msg bot-msg">'
        f'<div class="msg-header">'
        f'<span class="msg-author">✦ {model_name}</span>'
        f'<span class="msg-time">{meta}</span>'
        f"</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    if images:
        _render_message_images(session_id, images, prompt=prompt, settings=settings)

    if enhanced and enhanced != prompt:
        with st.expander("Enhanced prompt", expanded=False):
            st.caption(enhanced)


def _render_message_images(
    session_id: str,
    images: list[dict],
    prompt: str = "",
    settings: dict | None = None,
) -> None:
    """Render images from base64 data or by fetching from backend when missing."""
    from services.auth_service import is_logged_in
    from services.gallery_service import get_published_filenames, publish_to_gallery
    from services.session_service import get_session, get_session_image_base64, update_session

    full_session = get_session(session_id)
    fav_filenames = set(full_session.get("favourite_image_filenames", [])) if full_session else set()
    published_filenames = set(get_published_filenames(session_id))

    n = len(images)
    cols_count = min(n, 2) if n <= 4 else 2
    max_img_width = 280
    if n == 1:
        cols = st.columns([1, 2, 1])
        col = cols[1]
    else:
        cols = st.columns(cols_count, gap="small")
        col = None

    for i, img_data in enumerate(images):
        target_col = col if n == 1 else cols[i % cols_count]
        with target_col:
            filename = img_data.get("filename")
            b64_uri = img_data.get("base64")
            if b64_uri and isinstance(b64_uri, str) and b64_uri.startswith("data:"):
                img_src = b64_uri
            elif b64_uri and isinstance(b64_uri, str):
                img_src = f"data:image/png;base64,{b64_uri}"
            elif filename:
                img_src = get_session_image_base64(session_id, filename)
            else:
                img_src = None
            seed = img_data.get("seed", "?")
            if img_src:
                key_safe = "".join(c if c.isalnum() else "_" for c in f"{session_id}_{filename or i}")
                is_published = filename in published_filenames
                row_cols = st.columns([3, 1], gap="small")
                with row_cols[0]:
                    st.image(
                        img_src,
                        width=max_img_width,
                        caption=f"seed: {seed}",
                    )
                with row_cols[1]:
                    if filename:
                        is_fav = filename in fav_filenames
                        if st.button("♥" if is_fav else "♡", key=f"fav_img_{key_safe}", help="Toggle favourite"):
                            new_fav = list(fav_filenames)
                            if is_fav:
                                new_fav.remove(filename)
                            else:
                                new_fav.append(filename)
                            if update_session(session_id, favourite_image_filenames=new_fav):
                                st.rerun()
                        if not is_published:
                            if st.button("📤", key=f"pub_img_{key_safe}", help="Publish to gallery"):
                                if is_logged_in():
                                    result, err = publish_to_gallery(
                                        session_id,
                                        filename,
                                        prompt=prompt,
                                        settings=settings or {},
                                    )
                                    if result:
                                        st.toast("Published to gallery!")
                                        st.rerun()
                                    else:
                                        st.error(err or "Failed to publish")
                                else:
                                    st.warning("Log in to publish")
            else:
                st.markdown(
                    '<div class="image-card"><div class="placeholder-orb"></div></div>',
                    unsafe_allow_html=True,
                )


def _render_prompt_input(session: dict) -> None:
    from services.generation_service import generate
    from components.prompt_helper import (
        get_current_settings,
        build_effective_prompt,
    )

    prompt = st.chat_input("Describe the image you want to generate")
    if prompt:
        settings = get_current_settings()
        size = settings.get("image_size", "512x512")
        w, h = (int(x) for x in size.split("x"))
        effective_prompt = build_effective_prompt(prompt)
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
                lora_strength=st.session_state.get("lora_strength", 0.8),
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
                        "steps": steps_val if steps_val > 0 else 25,
                        "guidance_scale": settings.get("guidance_scale", 7.5),
                        "width": w,
                        "height": h,
                        "seed": None if settings.get("seed", -1) == -1 else settings.get("seed"),
                        "num_images": settings.get("num_images", 1),
                        "scheduler": settings.get("scheduler"),
                        "quality": settings.get("quality", "Normal"),
                        "style": settings.get("style"),
                        "lighting": settings.get("lightning"),
                        "color": settings.get("color"),
                    },
                }
            else:
                message["settings"] = {
                    **message.get("settings", {}),
                    "steps": steps_val if steps_val > 0 else 25,
                    "guidance_scale": settings.get("guidance_scale", 7.5),
                    "width": w,
                    "height": h,
                    "seed": None if settings.get("seed", -1) == -1 else settings.get("seed"),
                    "num_images": settings.get("num_images", 1),
                    "scheduler": settings.get("scheduler"),
                    "quality": settings.get("quality", "Normal"),
                    "style": message.get("settings", {}).get("style", settings.get("style")),
                    "lighting": message.get("settings", {}).get("lighting", settings.get("lightning")),
                    "color": message.get("settings", {}).get("color", settings.get("color")),
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
                    if s.get("message_count", 0) == 1:
                        first_five = " ".join(effective_prompt.split()[:5])
                        if first_five:
                            from services.session_service import update_session
                            updated = update_session(session["id"], title=first_five)
                            if updated:
                                s["title"] = first_five
                    break

            st.toast("Generation complete!")
            st.rerun()
        else:
            st.error("Backend unavailable. Check Settings.")
