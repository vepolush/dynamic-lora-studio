"""Auth UI — login, register, logout."""

from __future__ import annotations

import streamlit as st

from services.auth_service import get_user_email, is_logged_in, login, logout, register


def render_auth_sidebar() -> None:
    """Render auth section in sidebar: login/register or user + logout."""
    if is_logged_in():
        username = get_user_email() or "User"  # get_user_email returns username for display
        st.markdown(f'<div class="auth-user">{username}</div>', unsafe_allow_html=True)
        if st.button("Logout", key="auth_logout", use_container_width=True):
            logout()
            st.rerun()
        return

    tab_login, tab_register = st.tabs(["Login", "Register"])
    with tab_login:
        with st.form("login_form"):
            login_username = st.text_input("Username", key="login_username", type="default")
            login_pass = st.text_input("Password", key="login_pass", type="password")
            if st.form_submit_button("Login"):
                if login_username and login_pass:
                    result, err = login(login_username.strip(), login_pass)
                    if result:
                        st.toast("Logged in")
                        st.rerun()
                    else:
                        st.error(err or "Invalid username or password")
                else:
                    st.warning("Enter username and password")

    with tab_register:
        with st.form("register_form"):
            reg_username = st.text_input("Username", key="reg_username", type="default", help="3-32 chars, letters, numbers, underscore")
            reg_pass = st.text_input("Password", key="reg_pass", type="password")
            if st.form_submit_button("Register"):
                if reg_username and reg_pass:
                    result, err = register(reg_username.strip(), reg_pass)
                    if result:
                        st.toast("Registered and logged in")
                        st.rerun()
                    else:
                        st.error(err or "Registration failed")
                else:
                    st.warning("Enter username and password")


@st.dialog("Login required")
def show_login_required() -> None:
    """Show login prompt when action requires auth."""
    st.info("Please log in to continue.")
    render_auth_sidebar()
