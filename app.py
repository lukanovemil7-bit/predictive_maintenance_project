import streamlit as st

st.set_page_config(
    page_title="Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
)

pages = {
    "Проект": [
        st.Page("analysis_and_model.py", title="Анализ и модель", icon="📊"),
        st.Page("presentation.py", title="Презентация", icon="🎞️"),
    ]
}

current_page = st.navigation(pages, position="sidebar", expanded=True)
current_page.run()
