# -*- coding: utf-8 -*-
"""
Header Component: Hero Banner & Context
"""

import streamlit as st
from streamlit_app.config import APP_TITLE, APP_SUBTITLE


def render_header():
    """Renders the top banner for Bengaluru Restaurant Intelligence."""
    header_html = (
        f'<div class="hero-banner">'
        f'  <span class="hero-badge">⚡ 12,481 Authentic Bengaluru Venues &bull; Hybrid AI &bull; MMR &bull; Bayesian Priors</span>'
        f'  <h1>{APP_TITLE}</h1>'
        f'  <p>{APP_SUBTITLE}</p>'
        f'</div>'
    )
    st.markdown(header_html, unsafe_allow_html=True)

