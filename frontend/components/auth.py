"""Auth UI — login, register, logout."""

from __future__ import annotations

import streamlit as st

from services.auth_service import get_user_email, is_logged_in, login, logout, register


def render_auth_sidebar() -> None:
    """Render auth section in sidebar: login/register or user + logout."""
    if is_logged_in():
        email = get_user_email() or "User"
        st.markdown(f'<div class="auth-user">{email}</div>', unsafe_allow_html=True)
        if st.button("Logout", key="auth_logout", use_container_width=True):
            logout()
            st.rerun()
        return

    tab_login, tab_register = st.tabs(["Login", "Register"])
    with tab_login:
        with st.form("login_form"):
            login_email = st.text_input("Email", key="login_email", type="default")
            login_pass = st.text_input("Password", key="login_pass", type="password")
            if st.form_submit_button("Login"):
                if login_email and login_pass:
                    result, err = login(login_email.strip(), login_pass)
                    if result:
                        st.toast("Logged in")
                        st.rerun()
                    else:
                        st.error(err or "Invalid email or password")
                else:
                    st.warning("Enter email and password")

    with tab_register:
        with st.form("register_form"):
            reg_email = st.text_input("Email", key="reg_email", type="default")
            reg_pass = st.text_input("Password", key="reg_pass", type="password")
            if st.form_submit_button("Register"):
                if reg_email and reg_pass:
                    result, err = register(reg_email.strip(), reg_pass)
                    if result:
                        st.toast("Registered and logged in")
                        st.rerun()
                    else:
                        st.error(err or "Registration failed")
                else:
                    st.warning("Enter email and password")


@st.dialog("Login required")
def show_login_required() -> None:
    """Show login prompt when action requires auth."""
    st.info("Please log in to continue.")
    render_auth_sidebar()
