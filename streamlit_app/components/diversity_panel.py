# -*- coding: utf-8 -*-
"""
Diversity Panel Component: MMR Diversification Metrics & Variety Stats
"""

import streamlit as st
from typing import Dict, Any, Optional


def render_diversity_panel(diversification_meta: Optional[Dict[str, Any]]):
    """
    Renders the MMR diversification dashboard when MMR is active.
    """
    if not diversification_meta or not diversification_meta.get("enabled"):
        return

    metrics = diversification_meta.get("diversity_metrics")
    lambda_val = diversification_meta.get("lambda_param", 0.75)

    st.markdown("### 📊 Recommendation Diversity Dashboard")
    st.caption(
        f"**Algorithm:** Maximal Marginal Relevance (MMR) &bull; **λ Parameter:** `{lambda_val:.2f}` &bull; "
        "Balances personal relevance against list variety to eliminate repetitive chain recommendations."
    )

    if metrics:
        col1, col2, col3, col4, col5 = st.columns(5)
        
        ild = metrics.get("intra_list_diversity", 0.0)
        redundancy = metrics.get("redundancy_rate", 0.0)
        cuisine_ratio = metrics.get("unique_cuisine_ratio", 1.0)
        loc_ratio = metrics.get("unique_locality_ratio", 1.0)
        retention = metrics.get("relevance_retention_pct", 100.0)

        with col1:
            st.metric(
                label="Intra-List Diversity",
                value=f"{ild:.3f}",
                help="Higher ILD indicates greater feature variety among recommended restaurants."
            )
        with col2:
            st.metric(
                label="Redundancy Rate",
                value=f"{redundancy * 100:.1f}%",
                help="Percentage of near-duplicate or repeated chain outlets in recommendation slate."
            )
        with col3:
            st.metric(
                label="Cuisine Variety",
                value=f"{cuisine_ratio:.2f}",
                help="Ratio of distinct cuisines represented across recommendations."
            )
        with col4:
            st.metric(
                label="Locality Spread",
                value=f"{loc_ratio:.2f}",
                help="Ratio of unique Bengaluru localities represented."
            )
        with col5:
            st.metric(
                label="Relevance Retention",
                value=f"{retention:.1f}%",
                help="Percentage of original top relevance score retained while injecting diversity."
            )

    st.markdown("<div style='margin-bottom: 1.5rem;'></div>", unsafe_allow_html=True)

