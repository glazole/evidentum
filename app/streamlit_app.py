"""Evidentum — entrypoint. Defines multi-page navigation with explicit Russian titles."""
import streamlit as st

st.set_page_config(
    page_title="Evidentum",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

with st.sidebar:
    st.toggle(
        "Гибкая настройка",
        key="advanced_ui",
        help=(
            "Показывает выбор модели, температуру, Top-K, режим перевода запроса, "
            "порог релевантности и фильтр по нозологии (в режиме сравнения), загрузку PDF; "
            "в блоке «Используемые гайдлайны» — статусы, прогресс и действия с документами."
        ),
    )

pg = st.navigation(
    [
        st.Page("pages/main_page.py", title="Главная", icon="🏥"),
        st.Page("pages/Аналитика.py", title="Аналитика", icon="📊"),
    ],
    position="sidebar",
)
pg.run()
