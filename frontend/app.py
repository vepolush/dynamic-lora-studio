"""Dynamic LoRA Studio — Streamlit entry point."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="Dynamic LoRA Studio",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

_CSS_PATH = Path(__file__).parent / "styles" / "main.css"
if _CSS_PATH.exists():
    st.markdown(f"<style>{_CSS_PATH.read_text()}</style>", unsafe_allow_html=True)

from components.my_images import render_my_images  # noqa: E402
from components.prompt_helper import render_prompt_helper  # noqa: E402
from components.sidebar import render_sidebar  # noqa: E402
from components.settings import render_settings  # noqa: E402
from components.workspace import render_workspace  # noqa: E402
from state.session import init_session_state  # noqa: E402

init_session_state()

render_sidebar()

current_page = st.session_state.get("current_page", "generate")

if current_page == "my_images":
    render_my_images()
elif current_page == "settings":
    render_settings()
else:
    workspace_col, helper_col = st.columns([3, 1.3], gap="medium")
    with workspace_col:
        render_workspace()
    with helper_col:
        render_prompt_helper()
