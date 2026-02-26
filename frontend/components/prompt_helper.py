"""Right panel: Prompt Helper with Entities, Technical, and Styling sections."""

from __future__ import annotations

import streamlit as st

IMAGE_SIZES = [
    "512x512", "512x768", "768x512", "768x768",
    "512x896", "896x512", "768x1024", "1024x768",
    "1024x1024", "640x640", "640x960", "960x640",
]

STYLES = [
    "None", "Realistic", "Photographic", "3D Render", "Anime",
    "Digital Art", "Oil Painting", "Watercolor", "Pixel Art",
    "Comic Book", "Cinematic", "Fantasy Art", "Line Art",
    "Isometric", "Low Poly", "Neon Punk", "Origami",
]

LIGHTINGS = [
    "None", "Backlight", "Glowing", "Direct Sunlight", "Neon Light",
    "Studio", "Soft Light", "Hard Light", "Rim Light", "Volumetric",
    "Golden Hour", "Blue Hour", "Moonlight", "Candlelight",
    "Dramatic", "Ambient Occlusion",
]

COLORS = [
    "Default", "Warm", "Cool", "Vibrant", "Muted", "Monochrome",
    "Pastel", "Neon", "Sepia", "High Contrast", "Desaturated",
    "Cyberpunk Palette", "Earthy Tones", "Sunset Gradient",
]


def render_prompt_helper() -> None:
    st.markdown(
        '<div class="prompt-helper-header">'
        "<h3>Prompt Helper</h3>"
        "</div>",
        unsafe_allow_html=True,
    )

    _render_entities_section()
    _render_technical_section()
    _render_styling_section()

    st.markdown('<div class="apply-btn">', unsafe_allow_html=True)
    st.button("Apply to Prompt", key="apply_to_prompt", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Section A: Entities & LoRA
# ---------------------------------------------------------------------------

def _render_entities_section() -> None:
    with st.expander("Entities & LoRA", expanded=True):
        entities = st.session_state.get("entities", [])
        active_id = st.session_state.get("active_entity_id")

        _render_entity_grid(entities, active_id)

        active_entity = _get_active_entity()
        if active_entity:
            versions = active_entity.get("versions", ["v1"])
            image_count = active_entity.get("image_count", 0)
            st.markdown(
                f'<div class="entity-detail">'
                f'<div class="entity-detail-name">{active_entity.get("name", "")}</div>'
                f'<div class="entity-detail-trigger">{active_entity.get("trigger_word", "")}</div>'
                f'<div class="entity-detail-meta">'
                f'{image_count} images · '
                f'{len(versions)} version(s)'
                f"</div></div>",
                unsafe_allow_html=True,
            )

            if len(versions) > 1:
                active_ver = active_entity.get("active_version", versions[0])
                idx = versions.index(active_ver) if active_ver in versions else 0
                st.selectbox(
                    "Version",
                    versions,
                    index=idx,
                    key="entity_version_select",
                )

        st.slider(
            "LoRA Strength",
            min_value=0.0,
            max_value=1.5,
            value=st.session_state.get("lora_strength", 0.8),
            step=0.05,
            key="lora_strength_slider",
        )

        if st.session_state.get("show_entity_form", False):
            _render_entity_form()


def _render_entity_grid(entities: list[dict], active_id: str | None) -> None:
    """Render entity tiles as a grid of real st.button elements."""
    n_entities = len(entities)
    cols_per_row = 3
    total_slots = n_entities + 1  # +1 for "Add new"
    rows = (total_slots + cols_per_row - 1) // cols_per_row

    slot = 0
    for _row in range(rows):
        cols = st.columns(cols_per_row, gap="small")
        for col_idx in range(cols_per_row):
            if slot >= total_slots:
                break
            with cols[col_idx]:
                if slot < n_entities:
                    entity = entities[slot]
                    is_active = entity["id"] == active_id
                    css_class = "entity-tile-active" if is_active else "entity-tile-btn"

                    st.markdown(f'<div class="{css_class}">', unsafe_allow_html=True)
                    label = f"✦\n{entity['name']}"
                    if st.button(label, key=f"entity_btn_{entity['id']}", use_container_width=True):
                        if is_active:
                            st.session_state["active_entity_id"] = None
                        else:
                            st.session_state["active_entity_id"] = entity["id"]
                        st.rerun()
                    st.markdown("</div>", unsafe_allow_html=True)
                else:
                    st.markdown('<div class="entity-tile-add">', unsafe_allow_html=True)
                    if st.button("+\nAdd new", key="add_entity_tile", use_container_width=True):
                        st.session_state["show_entity_form"] = not st.session_state.get(
                            "show_entity_form", False
                        )
                        st.rerun()
                    st.markdown("</div>", unsafe_allow_html=True)
            slot += 1


def _render_entity_form() -> None:
    st.markdown(
        '<div class="entity-form-wrapper">'
        '<div class="entity-form-title">✦ Create New Entity</div>'
        "</div>",
        unsafe_allow_html=True,
    )
    st.text_input("Entity name", key="new_entity_name", placeholder="e.g. My Dog")
    st.text_input("Trigger word", key="new_entity_trigger", placeholder="e.g. <my_dog>")
    st.file_uploader(
        "Training images (ZIP)",
        type=["zip"],
        key="entity_zip_upload",
        help="Upload a ZIP archive with 5-20 images of the subject",
    )
    col_train, col_cancel = st.columns(2, gap="small")
    with col_train:
        st.markdown('<div class="generate-btn">', unsafe_allow_html=True)
        if st.button("Upload & Train", key="upload_train_btn", use_container_width=True):
            _handle_entity_upload()
        st.markdown("</div>", unsafe_allow_html=True)
    with col_cancel:
        if st.button("Cancel", key="cancel_entity_form", use_container_width=True):
            st.session_state["show_entity_form"] = False
            st.rerun()


def _handle_entity_upload() -> None:
    from services.entity_service import upload_entity

    name = st.session_state.get("new_entity_name", "").strip()
    trigger = st.session_state.get("new_entity_trigger", "").strip()
    zip_file = st.session_state.get("entity_zip_upload")
    if not name or not trigger:
        st.error("Name and trigger word required")
    elif not zip_file:
        st.error("Upload a ZIP file")
    else:
        zip_bytes = zip_file.read()
        entity = upload_entity(name, trigger, zip_bytes, zip_file.name)
        if entity:
            st.session_state["entities"] = st.session_state.get("entities", []) + [entity]
            st.session_state["show_entity_form"] = False
            st.toast("Entity uploaded & training started")
            st.rerun()
        else:
            st.error("Upload failed. Check backend in Settings.")


# ---------------------------------------------------------------------------
# Section B: Technical
# ---------------------------------------------------------------------------

def _render_technical_section() -> None:
    with st.expander("Technical", expanded=True):
        settings = st.session_state.get("generation_settings", {})

        st.slider(
            "Steps",
            min_value=10,
            max_value=50,
            value=settings.get("steps", 25),
            key="steps_slider",
        )

        st.slider(
            "Guidance Scale",
            min_value=1.0,
            max_value=20.0,
            value=settings.get("guidance_scale", 7.5),
            step=0.5,
            key="guidance_slider",
        )

        st.selectbox(
            "Image Size",
            IMAGE_SIZES,
            index=IMAGE_SIZES.index(settings.get("image_size", "512x512")),
            key="size_select",
        )

        st.number_input(
            "Seed",
            min_value=-1,
            max_value=999999,
            value=settings.get("seed", -1),
            key="seed_input",
            help="-1 for random seed",
        )

        st.select_slider(
            "Quality",
            options=["Draft", "Normal", "High", "Ultra"],
            value=settings.get("quality", "Normal"),
            key="quality_slider",
        )


# ---------------------------------------------------------------------------
# Section C: Styling
# ---------------------------------------------------------------------------

def _render_styling_section() -> None:
    with st.expander("Styling", expanded=False):
        settings = st.session_state.get("generation_settings", {})

        st.selectbox(
            "Style",
            STYLES,
            index=STYLES.index(settings.get("style", "None")),
            key="style_select",
        )

        st.selectbox(
            "Lightning",
            LIGHTINGS,
            index=LIGHTINGS.index(settings.get("lightning", "None")),
            key="lightning_select",
        )

        st.selectbox(
            "Color",
            COLORS,
            index=COLORS.index(settings.get("color", "Default")),
            key="color_select",
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_active_entity() -> dict | None:
    active_id = st.session_state.get("active_entity_id")
    if not active_id:
        return None
    for e in st.session_state.get("entities", []):
        if e["id"] == active_id:
            return e
    return None
