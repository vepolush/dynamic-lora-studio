"""Auth service — login, register, token storage."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any

import streamlit as st

from api.client import APIClient, BackendError
from config import BACKEND_ENABLED

AUTH_COOKIE_NAME = "auth_token"
AUTH_COOKIE_DAYS = 1


def _get_cookie_manager():
    """Lazy init CookieManager to avoid import at module load."""
    try:
        import extra_streamlit_components as stx
        return stx.CookieManager()
    except ImportError:
        return None


def _restore_token_from_cookie() -> None:
    """Restore auth token from cookie to session_state on app load."""
    if "auth_token" in st.session_state:
        return
    cm = _get_cookie_manager()
    if not cm:
        return
    try:
        cookies = cm.get_all()
        if isinstance(cookies, dict) and AUTH_COOKIE_NAME in cookies:
            token = cookies.get(AUTH_COOKIE_NAME)
            if token and isinstance(token, str):
                st.session_state["auth_token"] = token
                if "auth_user_username" not in st.session_state and "auth_user_username" in cookies:
                    st.session_state["auth_user_username"] = cookies.get("auth_user_username")
    except Exception:
        pass


def _save_token_to_cookie(token: str, username: str | None = None) -> None:
    """Save auth token to cookie for ~1 day persistence."""
    cm = _get_cookie_manager()
    if not cm:
        return
    try:
        expires = datetime.utcnow() + timedelta(days=AUTH_COOKIE_DAYS)
        cm.set(AUTH_COOKIE_NAME, token, expires_at=expires)
        if username:
            cm.set("auth_user_username", username, expires_at=expires)
    except Exception:
        pass


def _clear_auth_cookie() -> None:
    """Clear auth cookie on logout."""
    cm = _get_cookie_manager()
    if not cm:
        return
    try:
        cm.delete(AUTH_COOKIE_NAME)
        cm.delete("auth_user_username")
    except Exception:
        pass


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
        token = data.get("token")
        username_val = data.get("username")
        if token:
            st.session_state["auth_token"] = token
            st.session_state["auth_user_username"] = username_val
            st.session_state["auth_user_id"] = data.get("user_id")
            _save_token_to_cookie(token, username_val)
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
        token = data.get("token")
        username_val = data.get("username")
        if token:
            st.session_state["auth_token"] = token
            st.session_state["auth_user_username"] = username_val
            st.session_state["auth_user_id"] = data.get("user_id")
            _save_token_to_cookie(token, username_val)
        return data, None
    except BackendError as e:
        return None, _extract_detail(e)


def logout() -> None:
    """Clear auth state and cookie."""
    _clear_auth_cookie()
    for key in ("auth_token", "auth_user_username", "auth_user_id"):
        if key in st.session_state:
            del st.session_state[key]
