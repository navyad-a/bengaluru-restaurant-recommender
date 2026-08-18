# -*- coding: utf-8 -*-
"""
Metrics Panel Component: Result Summary Analytics
"""

import streamlit as st
from typing import Dict, Any, List
from streamlit_app.config import CURRENCY_SYMBOL


def render_metrics_panel(response: Dict[str, Any]):
    """
    Renders top-level summary metrics about the active recommendation set.
    """
    items = response.get("recommendations", [])
    if not items:
        return

    count = len(items)
    ratings = [r.get("rating") for r in items if r.get("rating") is not None and r.get("rating") > 0]
    avg_rating = sum(ratings) / len(ratings) if ratings else 0.0
    costs = [r.get("cost_for_two_inr", 0) for r in items if r.get("cost_for_two_inr", 0) > 0]
    avg_cost = sum(costs) / len(costs) if costs else 0

    is_cold_start = response.get("is_cold_start", False)
    strategy = response.get("strategy", "Hybrid Ranking")
    model_source = response.get("model_source", "hybrid")

    strategy_label = f"🧊 Cold-Start ({strategy})" if is_cold_start else f"👤 Personalized ({strategy})"

    st.markdown("#### 📈 Slate Summary")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Recommendations", f"{count}")
    with col2:
        st.metric("Average Rating", f"★ {avg_rating:.2f}" if avg_rating > 0 else "N/A")
    with col3:
        st.metric(f"Avg Cost for Two", f"{CURRENCY_SYMBOL}{int(avg_cost):,}")
    with col4:
        st.metric("Routing Strategy", strategy_label)

    # Technical Details Expander
    with st.expander("🛠️ Technical Metadata & Pipeline Weights", expanded=False):
        tcol1, tcol2 = st.columns(2)
        with tcol1:
            st.markdown("**Effective Model Signal Weights:**")
            effective_weights = response.get("effective_weights", {})
            if effective_weights:
                for k, v in effective_weights.items():
                    st.progress(float(v), text=f"{k.capitalize()}: {v * 100:.1f}%")
            else:
                st.caption("Standard signal weights applied.")
        with tcol2:
            st.markdown("**Routing Details:**")
            st.markdown(f"- **User ID:** `{response.get('user_id', 'Anonymous')}`")
            st.markdown(f"- **Model Pipeline:** `{model_source}`")
            st.markdown(f"- **Strategy Mode:** `{strategy}`")
            st.caption("All models trained on 12,481 authentic Bengaluru restaurant venues.")

    st.markdown("<div style='margin-bottom: 1.5rem;'></div>", unsafe_allow_html=True)

