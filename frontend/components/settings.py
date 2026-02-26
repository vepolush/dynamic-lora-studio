"""Settings page — app configuration."""

from __future__ import annotations

import os

import streamlit as st

from services.config_service import load_config, save_config


def render_settings() -> None:
    st.markdown(
        '<div class="session-header">'
        "<h3>App Settings</h3>"
        "</div>",
        unsafe_allow_html=True,
    )

    cfg = load_config()

    backend_url = st.text_input(
        "Backend URL",
        value=cfg.get("backend_url", "http://localhost:8000"),
        help="Base URL of the backend API (e.g. http://localhost:8000)",
        key="settings_backend_url",
    )

    backend_enabled = st.checkbox(
        "Use backend API",
        value=cfg.get("backend_enabled", True),
        help="When unchecked, app uses mock data only",
        key="settings_backend_enabled",
    )

    if st.button("Save", key="settings_save"):
        new_cfg = {
            "backend_url": backend_url,
            "backend_enabled": backend_enabled,
        }
        save_config(new_cfg)
        os.environ["BACKEND_URL"] = backend_url
        os.environ["BACKEND_ENABLED"] = "true" if backend_enabled else "false"
        st.toast("Settings saved. Restart the app to apply.")
        st.rerun()
