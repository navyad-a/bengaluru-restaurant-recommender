# -*- coding: utf-8 -*-
"""
Recommendations Page: Dedicated Slate Inspection & Diversity Analytics
"""

import streamlit as st
from streamlit_app.components.recommendation_card import render_recommendation_card
from streamlit_app.components.diversity_panel import render_diversity_panel
from streamlit_app.components.metrics_panel import render_metrics_panel


def render_recommendations_page():
    """Renders dedicated recommendation view for the currently stored slate."""
    st.markdown("## 📋 Active Recommendation Slate")
    
    response = st.session_state.get("last_recommendation_response")
    if not response:
        st.info("No recommendations currently generated. Navigate to the **Home** tab to generate recommendations.")
        return

    # Render summary metrics and diversity panel
    render_metrics_panel(response)
    render_diversity_panel(response.get("diversification"))

    recs = response.get("recommendations", [])
    if not recs:
        st.warning("No matching restaurants found for the selected constraints. Try relaxing your budget or locality filters.")
        return

    st.markdown(f"### 🍽️ Recommended Venues ({len(recs)})")
    for idx, item in enumerate(recs, start=1):
        render_recommendation_card(item, rank=idx)

