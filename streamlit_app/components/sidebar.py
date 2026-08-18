# -*- coding: utf-8 -*-
"""
Sidebar Component: User Identity, MMR Controls & System Telemetry
"""

import streamlit as st
from typing import Dict, Any, Optional
from streamlit_app.api_client import api_client
from streamlit_app.config import (
    DEFAULT_TOP_K,
    DEFAULT_MMR_ENABLED,
    DEFAULT_MMR_LAMBDA,
    API_BASE_URL
)


def render_sidebar() -> Dict[str, Any]:
    """
    Renders sidebar navigation, user identity selectors, MMR controls, and system health status.
    Returns the active control settings dict.
    """
    with st.sidebar:
        st.markdown("### 👤 User Identity")
        user_mode = st.radio(
            "Select User Profile Mode",
            ["Anonymous / Preference User", "Known User ID (Collaborative SVD)"],
            index=0 if not st.session_state.get("is_known_user") else 1,
            help="Anonymous mode routes to cold-start preference engine. Known User ID loads historical SVD collaborative factors."
        )

        user_id: Optional[int] = None
        if user_mode == "Known User ID (Collaborative SVD)":
            st.session_state["is_known_user"] = True
            user_id = st.number_input(
                "Enter User ID (Benchmark 1–600)",
                min_value=1,
                max_value=600,
                value=int(st.session_state.get("user_id") or 2),
                step=1,
                help="Select a benchmark user to blend SVD collaborative signals with content & quality."
            )
            st.session_state["user_id"] = user_id
            st.caption(f"Active User ID: **{user_id}**")
        else:
            st.session_state["is_known_user"] = False
            st.session_state["user_id"] = None
            st.caption("Mode: **Cold-Start / Preferences**")

        st.divider()

        # Recommendation & MMR Controls
        st.markdown("### ⚙️ Recommendation Controls")
        
        top_k = st.select_slider(
            "Recommendations Count (Top-K)",
            options=[5, 10, 15, 20],
            value=st.session_state.get("top_k", DEFAULT_TOP_K)
        )
        st.session_state["top_k"] = top_k

        mmr_enabled = st.toggle(
            "Enable MMR Diversification",
            value=st.session_state.get("mmr_enabled", DEFAULT_MMR_ENABLED),
            help="Maximal Marginal Relevance reduces near-duplicates and balances relevance with cuisine variety."
        )
        st.session_state["mmr_enabled"] = mmr_enabled

        mmr_lambda = st.session_state.get("mmr_lambda", DEFAULT_MMR_LAMBDA)
        if mmr_enabled:
            mmr_lambda = st.slider(
                "MMR Lambda (λ)",
                min_value=0.50,
                max_value=1.00,
                value=float(mmr_lambda),
                step=0.05,
                help="Lower λ increases variety. Higher λ prioritizes pure relevance."
            )
            st.caption(f"λ = **{mmr_lambda:.2f}** (Relevance: {int(mmr_lambda*100)}%, Variety: {int((1-mmr_lambda)*100)}%)")
            st.session_state["mmr_lambda"] = mmr_lambda
        else:
            st.info("Diversification disabled. Ranking by pure hybrid relevance score.")

        st.divider()

        # Backend Health Status
        st.markdown("### 🔌 Backend Connectivity")
        st.caption(f"API Target: `{API_BASE_URL}`")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Check API", use_container_width=True):
                health, err = api_client.get_health()
                if health and health.get("status") == "healthy":
                    st.success("API Online")
                else:
                    st.error("API Offline")
        with col2:
            if st.button("Clear Cache", use_container_width=True):
                res, err = api_client.clear_cache()
                if res and res.get("status") == "success":
                    st.success("Cache Cleared")
                else:
                    st.warning("Clear Failed")

        st.divider()

        # Transparency Notice
        notice_html = (
            '<div style="font-size: 0.76rem; color: #5f6368; line-height: 1.4;">'
            '<strong>Data Transparency Notice:</strong><br>'
            '• Authentic catalog: 12,481 physical Bengaluru venues.<br>'
            '• Coordinates: Locality-level centroids.<br>'
            '• Collaborative SVD: Evaluated on synthetic benchmark.'
            '</div>'
        )
        st.markdown(notice_html, unsafe_allow_html=True)

    return {
        "user_id": user_id,
        "is_known_user": st.session_state.get("is_known_user", False),
        "top_k": top_k,
        "mmr_enabled": mmr_enabled,
        "mmr_lambda": mmr_lambda
    }

