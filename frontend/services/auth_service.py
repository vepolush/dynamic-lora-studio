"""Auth service — login, register, token storage."""

from __future__ import annotations

import json
from typing import Any

import streamlit as st

from api.client import APIClient, BackendError
from config import BACKEND_ENABLED

AUTH_STORAGE_TOKEN = "auth_token"
AUTH_STORAGE_USERNAME = "auth_user_username"

_AUTH_STORAGE_KEY = "_auth_local_storage"


def _get_local_storage():
    """Lazy init LocalStorage (singleton) for auth persistence in browser localStorage."""
    if _AUTH_STORAGE_KEY in st.session_state:
        return st.session_state[_AUTH_STORAGE_KEY]
    try:
        from streamlit_local_storage import LocalStorage
        ls = LocalStorage(key="auth_storage_init")
        st.session_state[_AUTH_STORAGE_KEY] = ls
        return ls
    except ImportError:
        return None
    except Exception:
        return None


def _restore_token_from_storage() -> None:
    """Restore auth token from localStorage to session_state on app load."""
    if "auth_token" in st.session_state:
        return
    ls = _get_local_storage()
    if not ls:
        return
    try:
        token = ls.getItem(AUTH_STORAGE_TOKEN)
        if token and isinstance(token, str) and token.strip():
            st.session_state["auth_token"] = token
            if "auth_user_username" not in st.session_state:
                username = ls.getItem(AUTH_STORAGE_USERNAME)
                if username and isinstance(username, str):
                    st.session_state["auth_user_username"] = username
    except Exception:
        pass


def _save_token_to_storage(token: str, username: str | None = None) -> None:
    """Save auth token to localStorage for persistence across page reloads."""
    ls = _get_local_storage()
    if not ls:
        return
    try:
        ls.setItem(AUTH_STORAGE_TOKEN, token, key="auth_set_token")
        if username:
            ls.setItem(AUTH_STORAGE_USERNAME, username, key="auth_set_username")
    except Exception:
        pass


def _schedule_save_to_storage(token: str, username: str | None = None) -> None:
    """Schedule save to localStorage on next run (avoids rerun killing the component before browser receives it)."""
    st.session_state["_auth_pending_sync"] = (token, username)


def _flush_pending_save_to_storage() -> None:
    """Run on each app load: persist pending auth to localStorage (in a run without immediate rerun)."""
    pending = st.session_state.pop("_auth_pending_sync", None)
    if pending:
        token, username = pending
        _save_token_to_storage(token, username)


def _clear_auth_storage() -> None:
    """Clear auth data from localStorage on logout."""
    ls = _get_local_storage()
    if not ls:
        return
    try:
        ls.deleteItem(AUTH_STORAGE_TOKEN, key="auth_del_token")
        ls.deleteItem(AUTH_STORAGE_USERNAME, key="auth_del_username")
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
            _schedule_save_to_storage(token, username_val)
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
            _schedule_save_to_storage(token, username_val)
        return data, None
    except BackendError as e:
        return None, _extract_detail(e)


def logout() -> None:
    """Clear auth state and localStorage."""
    _clear_auth_storage()
    for key in ("auth_token", "auth_user_username", "auth_user_id"):
        if key in st.session_state:
            del st.session_state[key]
