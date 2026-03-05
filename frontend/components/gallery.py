"""Global gallery page — Images and LoRAs tabs, grid, overlay dialogs."""

from __future__ import annotations

import base64
import html
import json
import streamlit as st

from services.auth_service import is_logged_in
from services.gallery_service import (
    add_gallery_lora,
    get_gallery,
    get_gallery_image,
    get_gallery_image_base64,
    get_gallery_lora,
    get_gallery_loras,
    get_gallery_lora_preview_base64,
    like_gallery_image,
    unpublish_gallery_lora,
)


def render_gallery() -> None:
    """Render gallery page with Images | LoRAs tabs."""
    st.markdown(
        '<div class="session-header"><h3>Gallery</h3></div>',
        unsafe_allow_html=True,
    )

    tab_images, tab_loras = st.tabs(["Images", "LoRAs"])

    with tab_images:
        _render_images_tab()

    with tab_loras:
        _render_loras_tab()


def _render_images_tab() -> None:
    """Render images gallery tab."""
    sort = st.selectbox(
        "Sort by",
        ["newest", "oldest", "popular"],
        format_func=lambda x: {"newest": "Newest", "oldest": "Oldest", "popular": "Popular"}.get(x, x),
        key="gallery_sort",
    )
    sort_key = {"newest": "newest", "oldest": "oldest", "popular": "popular"}.get(sort, "newest")

    images = get_gallery(sort=sort_key, limit=50, offset=0)
    if not images:
        st.info("No images in gallery yet. Generate and publish to share!")
        return

    cols = st.columns(4, gap="small")
    for i, img in enumerate(images):
        with cols[i % 4]:
            _render_gallery_thumb(img)

    selected_id = st.session_state.get("gallery_selected_id")
    if selected_id:
        _render_gallery_popup(selected_id)


def _render_loras_tab() -> None:
    """Render LoRAs gallery tab."""
    sort = st.selectbox(
        "Sort by",
        ["newest", "oldest", "popular"],
        format_func=lambda x: {"newest": "Newest", "oldest": "Oldest", "popular": "Popular"}.get(x, x),
        key="gallery_loras_sort",
    )
    sort_key = {"newest": "newest", "oldest": "oldest", "popular": "popular"}.get(sort, "newest")

    loras = get_gallery_loras(sort=sort_key, limit=50, offset=0)
    if not loras:
        st.info("No LoRAs in gallery yet. Train an entity and publish it to share!")
        return

    cols = st.columns(4, gap="small")
    for i, lora in enumerate(loras):
        with cols[i % 4]:
            _render_lora_thumb(lora)

    selected_lora_id = st.session_state.get("gallery_selected_lora_id")
    if selected_lora_id:
        _render_lora_popup(selected_lora_id)


def _render_lora_thumb(lora: dict) -> None:
    """Render LoRA thumbnail with Add button."""
    lora_id = lora.get("id", "")
    name = lora.get("name", "")
    trigger = lora.get("trigger_word", "")
    add_count = lora.get("add_count", 0)
    is_mine = lora.get("is_mine", False)

    thumb_src = get_gallery_lora_preview_base64(lora_id)
    if not thumb_src:
        from config import BACKEND_URL
        thumb_src = f"{BACKEND_URL.rstrip('/')}/api/gallery/loras/{lora_id}/preview"

    if thumb_src:
        st.image(thumb_src, use_container_width=True)

    st.caption(f"**{name}** · `{trigger}`")
    st.caption(f"Added {add_count}x")

    col_v, col_a = st.columns(2, gap="small")
    with col_v:
        if st.button("View", key=f"view_lora_{lora_id}", use_container_width=True):
            st.session_state["gallery_selected_lora_id"] = lora_id
            st.rerun()
    with col_a:
        if is_mine:
            if st.button("Unpublish", key=f"unpublish_lora_{lora_id}", use_container_width=True):
                if unpublish_gallery_lora(lora_id):
                    st.toast("LoRA unpublished")
                    st.rerun()
        else:
            if st.button("Add", key=f"add_lora_{lora_id}", use_container_width=True):
                if is_logged_in():
                    result = add_gallery_lora(lora_id)
                    if result:
                        new_entity = result.get("entity")
                        if new_entity:
                            entities = st.session_state.get("entities", [])
                            st.session_state["entities"] = [new_entity] + entities
                            st.session_state["active_entity_id"] = new_entity.get("id")
                        st.toast(f"LoRA added: {name}")
                        st.rerun()
                else:
                    st.warning("Log in to add LoRA")


@st.dialog("Gallery LoRA")
def _show_lora_dialog(lora_id: str) -> None:
    """Show LoRA details as overlay dialog."""
    if not item:
        st.session_state.pop("gallery_selected_lora_id", None)
        st.rerun()
        return

    thumb_src = get_gallery_lora_preview_base64(lora_id)
    if not thumb_src:
        from config import BACKEND_URL
        thumb_src = f"{BACKEND_URL.rstrip('/')}/api/gallery/loras/{lora_id}/preview"

    col_img, col_info = st.columns([1.2, 1])
    with col_img:
        if thumb_src:
            st.image(thumb_src, use_container_width=True)

    with col_info:
        st.markdown(f"**Name:** {item.get('name', '')}")
        st.markdown(f"**Trigger:** `{item.get('trigger_word', '')}`")
        st.markdown(f"**Author:** {item.get('author_email', '')}")
        st.markdown(f"**Added:** {item.get('add_count', 0)}x")
        if item.get("description"):
            st.markdown(f"**Description:** {item.get('description', '')}")

        if is_logged_in():
            if st.button("Add to my entities", key="gallery_lora_popup_add", type="primary"):
                result = add_gallery_lora(lora_id)
                if result:
                    new_entity = result.get("entity")
                    if new_entity:
                        entities = st.session_state.get("entities", [])
                        st.session_state["entities"] = [new_entity] + entities
                        st.session_state["active_entity_id"] = new_entity.get("id")
                    st.toast(f"LoRA added: {item.get('name', '')}")
                    st.rerun()
            if item.get("is_mine"):
                st.divider()
                if st.button("Unpublish", key="gallery_lora_popup_unpublish"):
                    if unpublish_gallery_lora(lora_id):
                        st.session_state.pop("gallery_selected_lora_id", None)
                        st.toast("LoRA unpublished")
                        st.rerun()
                    else:
                        st.error("Failed to unpublish")
        else:
            st.caption("Log in to add LoRA")

        if st.button("✕ Close", key="gallery_lora_close_popup"):
            st.session_state.pop("gallery_selected_lora_id", None)
            st.rerun()


def _render_lora_popup(lora_id: str) -> None:
    """Render LoRA as overlay dialog."""
    _show_lora_dialog(lora_id)


def _render_gallery_thumb(img: dict) -> None:
    """Render thumbnail with View and Like buttons."""
    image_id = img.get("id", "")
    likes = img.get("likes_count", 0)
    liked = img.get("liked", False)

    thumb_src = get_gallery_image_base64(image_id)
    if not thumb_src:
        from config import BACKEND_URL
        thumb_src = f"{BACKEND_URL.rstrip('/')}/api/gallery/{image_id}/image"

    if thumb_src:
        st.image(thumb_src, use_container_width=True)

    col_v, col_l = st.columns(2, gap="small")
    with col_v:
        if st.button("View", key=f"view_{image_id}", use_container_width=True):
            st.session_state["gallery_selected_id"] = image_id
            st.rerun()
    with col_l:
        if st.button(
            f"❤ {likes}" if liked else f"♡ {likes}",
            key=f"like_{image_id}",
            use_container_width=True,
            help="Like" if not liked else "Unlike",
        ):
            if is_logged_in():
                updated = like_gallery_image(image_id)
                if updated:
                    st.rerun()
            else:
                st.warning("Log in to like")


@st.dialog("Gallery Image")
def _show_gallery_image_dialog(image_id: str) -> None:
    """Show gallery image as overlay dialog."""
    item = get_gallery_image(image_id)
    if not item:
        st.session_state.pop("gallery_selected_id", None)
        st.rerun()
        return

    img_src = get_gallery_image_base64(image_id)
    if not img_src:
        from config import BACKEND_URL
        img_src = f"{BACKEND_URL.rstrip('/')}/api/gallery/{image_id}/image"

    col_img, col_info = st.columns([1.2, 1])
    with col_img:
        if img_src:
            st.image(img_src, use_container_width=True)

    with col_info:
        st.markdown(f"**Author:** {item.get('author_email', '')}")
        st.markdown(
            f'<div class="gallery-prompt">**Prompt:** {html.escape(item.get("prompt", "") or "")}</div>',
            unsafe_allow_html=True,
        )
        settings = item.get("settings") or {}
        if settings:
            st.markdown("**Settings:**")
            st.json(settings)
        st.markdown(f"**Likes:** {item.get('likes_count', 0)}")

        if is_logged_in():
            liked = item.get("liked", False)
            if st.button("❤ Like" if liked else "♡ Like", key="gallery_popup_like"):
                updated = like_gallery_image(image_id)
                if updated:
                    st.rerun()

        if is_logged_in():
            st.divider()
            settings_for_apply = {"prompt": item.get("prompt", ""), **(settings or {})}
            settings_b64 = base64.b64encode(json.dumps(settings_for_apply, ensure_ascii=False).encode()).decode()
            if st.button("Create session with these settings", key="gallery_create_session_btn"):
                from services.session_service import create_session
                new_sess = create_session("New session")
                if new_sess:
                    st.session_state["sessions"] = [new_sess] + st.session_state.get("sessions", [])
                    st.session_state["active_session_id"] = new_sess["id"]
                    st.session_state["current_page"] = "generate"
                    st.session_state.pop("gallery_selected_id", None)
                    st.query_params["apply_settings_b64"] = settings_b64
                    st.toast("Session created with settings applied")
                    st.rerun()

        if st.button("✕ Close", key="gallery_close_popup"):
            st.session_state.pop("gallery_selected_id", None)
            st.rerun()


def _render_gallery_popup(image_id: str) -> None:
    """Render gallery image as overlay dialog."""
    _show_gallery_image_dialog(image_id)
