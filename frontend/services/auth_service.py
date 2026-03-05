"""Auth service — login, register, token storage."""

from __future__ import annotations

import json
from typing import Any

import streamlit as st

from api.client import APIClient, BackendError
from config import BACKEND_ENABLED


def _extract_detail(err: BackendError) -> str:
    """Extract detail from BackendError (backend returns JSON with 'detail')."""
    msg = str(err)
    if "Backend error:" in msg:
        try:
            rest = msg.split("Backend error:", 1)[1].strip()
            data = json.loads(rest)
            return data.get("detail", msg)
        except (json.JSONDecodeError, IndexError, KeyError):
            pass
    return msg


def get_token() -> str | None:
    """Get auth token from session state."""
    return st.session_state.get("auth_token")


def is_logged_in() -> bool:
    """Check if user is authenticated."""
    return bool(get_token())


def get_user_email() -> str | None:
    """Get current user display name (username) from session state."""
    return st.session_state.get("auth_user_username")


def get_user_username() -> str | None:
    """Get current user username from session state."""
    return st.session_state.get("auth_user_username")


def get_client() -> APIClient:
    """Get API client with auth token if logged in."""
    return APIClient(token=get_token())


def register(username: str, password: str) -> tuple[dict[str, Any] | None, str | None]:
    """Register and store token. Returns (user_dict, error_message)."""
    if not BACKEND_ENABLED:
        return None, "Backend not available"
    try:
        client = APIClient()
        data = client.register(username, password)
        st.session_state["auth_token"] = data.get("token")
        st.session_state["auth_user_username"] = data.get("username")
        st.session_state["auth_user_id"] = data.get("user_id")
        return data, None
    except BackendError as e:
        return None, _extract_detail(e)


def login(username: str, password: str) -> tuple[dict[str, Any] | None, str | None]:
    """Login and store token. Returns (user_dict, error_message)."""
    if not BACKEND_ENABLED:
        return None, "Backend not available"
    try:
        client = APIClient()
        data = client.login(username, password)
        st.session_state["auth_token"] = data.get("token")
        st.session_state["auth_user_username"] = data.get("username")
        st.session_state["auth_user_id"] = data.get("user_id")
        return data, None
    except BackendError as e:
        return None, _extract_detail(e)


def logout() -> None:
    """Clear auth state."""
    for key in ("auth_token", "auth_user_username", "auth_user_id"):
        if key in st.session_state:
            del st.session_state[key]
