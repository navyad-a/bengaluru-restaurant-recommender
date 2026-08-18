# -*- coding: utf-8 -*-
"""
Main Application Entrypoint for Bengaluru Restaurant Intelligence
Streamlit UI Client Layer connecting to FastAPI Backend.
"""

import os
import sys

# Ensure root is in sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import streamlit as st
from streamlit_app.config import APP_TITLE, APP_ICON, APP_VERSION
from streamlit_app.state import init_session_state
from streamlit_app.components.sidebar import render_sidebar
from streamlit_app.pages.home import render_home_page
from streamlit_app.pages.recommendations import render_recommendations_page
from streamlit_app.pages.system_status import render_system_status_page


def inject_custom_css():
    """Injects custom CSS from assets/style.css."""
    css_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "style.css")
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            css_content = f.read()
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)


def main():
    """Main application loop."""
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=APP_ICON,
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # State & Styling
    init_session_state()
    inject_custom_css()

    # Sidebar
    sidebar_controls = render_sidebar()

    # Top-level Page Navigation
    nav_selection = st.radio(
        "Navigation",
        ["🏠 Restaurant Discovery", "📋 Active Recommendations", "⚙️ System Status & Telemetry"],
        horizontal=True,
        label_visibility="collapsed"
    )

    st.markdown("<div style='margin-bottom: 1.2rem;'></div>", unsafe_allow_html=True)

    # Route Views
    if nav_selection == "🏠 Restaurant Discovery":
        render_home_page()
    elif nav_selection == "📋 Active Recommendations":
        render_recommendations_page()
    elif nav_selection == "⚙️ System Status & Telemetry":
        render_system_status_page()

    # Footer
    st.markdown("---")
    footer_html = (
        f'<div style="text-align: center; font-size: 0.82rem; color: #5f6368; padding: 1rem 0;">'
        f'<strong>{APP_TITLE}</strong> v{APP_VERSION} &bull; '
        f'Powered by <strong>FastAPI Async Engine</strong> &bull; '
        f'12,481 Physical Bengaluru Restaurant Venues &bull; '
        f'Content TF-IDF &bull; Collaborative SVD &bull; BallTree Spatial Proximity &bull; MMR Diversification'
        f'</div>'
    )
    st.markdown(footer_html, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

