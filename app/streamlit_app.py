"""Evidentum — entrypoint. Defines multi-page navigation with explicit Russian titles."""
import streamlit as st

st.set_page_config(
    page_title="Evidentum",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

pg = st.navigation(
    [
        st.Page("pages/main_page.py", title="Главная", icon="🏥"),
        st.Page("pages/Аналитика.py", title="Аналитика", icon="📊"),
    ],
    position="sidebar",
)
pg.run()
