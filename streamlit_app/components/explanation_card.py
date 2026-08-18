# -*- coding: utf-8 -*-
"""
Explanation Card Component: Recommendation Transparency & Signals
"""

import streamlit as st
from typing import Dict, Any, Optional


def render_explanation_card(
    explanation: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    item: Optional[Dict[str, Any]] = None
):
    """
    Renders user-friendly explainability information grounded in backend signals.
    """
    with st.expander("💡 Why this recommendation?", expanded=False):
        if explanation:
            exp_html = f'<div class="explanation-box"><div class="explanation-title">Recommendation Summary</div>{explanation}</div>'
            st.markdown(exp_html, unsafe_allow_html=True)

        if metadata:
            st.markdown("<div style='margin-top: 0.5rem;'></div>", unsafe_allow_html=True)
            reasons = metadata.get("reasons", [])
            if reasons:
                st.markdown("**Key Recommendation Factors:**")
                for r in reasons:
                    st.markdown(f"- {r}")

            matched_cuisines = metadata.get("matched_cuisines", [])
            if matched_cuisines:
                st.markdown(f"**Matched Cuisines:** `{', '.join(matched_cuisines)}`")

            div_reason = metadata.get("diversification_reason")
            if div_reason:
                st.markdown(f"**Diversity Enhancement:** _{div_reason}_")

        # Technical Score Breakdown for technical reviewers
        if item:
            with st.expander("🔬 Technical Signal Scores & Weights", expanded=False):
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Hybrid Score", f"{item.get('hybrid_score', 0.0):.4f}")
                with col2:
                    st.metric("Content Score", f"{item.get('content_score', 0.0):.4f}")
                with col3:
                    st.metric("Collab SVD", f"{item.get('collaborative_score', 0.0):.4f}")
                with col4:
                    st.metric("Bayesian Quality", f"{item.get('quality_score', 0.0):.4f}")

                st.caption(
                    f"**Location Source:** `{item.get('location_source', 'Bengaluru locality centroid')}` | "
                    f"**Model Source:** `{item.get('model_source', 'hybrid')}`"
                )

