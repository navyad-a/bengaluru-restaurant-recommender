# -*- coding: utf-8 -*-
"""
Phase 14 Test Suite: Streamlit UI Component Rendering & Page Flow Tests
"""

import pytest
from unittest.mock import patch, MagicMock
import streamlit as st

from streamlit_app.components.header import render_header
from streamlit_app.components.recommendation_card import render_recommendation_card
from streamlit_app.components.explanation_card import render_explanation_card
from streamlit_app.components.diversity_panel import render_diversity_panel
from streamlit_app.components.metrics_panel import render_metrics_panel
from streamlit_app.components.filters import render_preference_filters
from streamlit_app.components.sidebar import render_sidebar
from streamlit_app.pages.home import render_home_page
from streamlit_app.pages.recommendations import render_recommendations_page
from streamlit_app.pages.system_status import render_system_status_page
from streamlit_app.state import init_session_state


@pytest.fixture(autouse=True)
def init_st_state():
    init_session_state()


@pytest.mark.streamlit
def test_render_header():
    """Verifies render_header executes without error."""
    with patch("streamlit.markdown") as mock_md:
        render_header()
        assert mock_md.called


@pytest.mark.streamlit
def test_render_recommendation_card():
    """Verifies render_recommendation_card with various restaurant payloads."""
    item = {
        "restaurant_id": 1,
        "name": "Nagarjuna Restaurant",
        "rating": 4.3,
        "review_count": 1250,
        "cost_for_two_inr": 800,
        "price_tier": "Moderate",
        "cuisines": ["Andhra", "Biryani", "South Indian"],
        "area": "Indiranagar",
        "online_order": True,
        "book_table": True,
        "distance_km": 1.25,
        "explanation": "Recommended because you enjoy Andhra biryani in Indiranagar.",
        "explanation_metadata": {
            "primary_signal": "content_preference",
            "matched_cuisines": ["Andhra", "Biryani"],
            "contributing_signals": ["bayesian_quality", "spatial_proximity"],
            "scores": {"content": 0.88, "quality": 0.85, "location": 0.90}
        }
    }
    with patch("streamlit.markdown") as mock_md, patch("streamlit.expander") as mock_exp:
        render_recommendation_card(item, rank=1)
        assert mock_md.called


@pytest.mark.streamlit
def test_render_explanation_card():
    """Verifies render_explanation_card formats reasoning properly."""
    meta = {
        "primary_signal": "content_preference",
        "matched_cuisines": ["North Indian"],
        "contributing_signals": ["collaborative_svd"],
        "diversity_reason": "Selected by MMR for culinary variety",
        "scores": {"content": 0.75, "collaborative": 0.82}
    }
    with patch("streamlit.markdown") as mock_md, patch("streamlit.expander") as mock_exp:
        render_explanation_card("Great match for north indian food", meta)
        assert mock_md.called


@pytest.mark.streamlit
def test_render_diversity_panel():
    """Verifies render_diversity_panel calculates and renders diversity metrics."""
    div_data = {
        "enabled": True,
        "lambda_param": 0.75,
        "diversity_metrics": {
            "intra_list_diversity": 0.654,
            "redundancy_rate": 0.0,
            "unique_cuisine_ratio": 0.80,
            "unique_locality_ratio": 0.70,
            "relevance_retention": 0.95
        }
    }
    with patch("streamlit.metric") as mock_metric, patch("streamlit.expander") as mock_exp:
        render_diversity_panel(div_data)
        assert mock_metric.called

    # When None -> should handle gracefully
    render_diversity_panel(None)


@pytest.mark.streamlit
def test_render_metrics_panel():
    """Verifies render_metrics_panel summarizes slate metrics."""
    response = {
        "count": 5,
        "is_cold_start": False,
        "strategy": "warm_hybrid",
        "effective_weights": {"collaborative": 0.4, "content": 0.35, "location": 0.15, "quality": 0.10},
        "recommendations": [
            {"rating": 4.2, "cost_for_two_inr": 800},
            {"rating": 4.5, "cost_for_two_inr": 1200}
        ]
    }
    with patch("streamlit.metric") as mock_metric, patch("streamlit.progress") as mock_prog:
        render_metrics_panel(response)
        assert mock_metric.called


@pytest.mark.streamlit
def test_render_preference_filters():
    """Verifies filter rendering with default options."""
    with patch("streamlit.multiselect", return_value=["Biryani"]), \
         patch("streamlit.selectbox", return_value="Indiranagar"), \
         patch("streamlit.select_slider", return_value=1000), \
         patch("streamlit.checkbox", return_value=False):
        filters = render_preference_filters()
        assert filters["preferred_cuisines"] == ["Biryani"]
        assert filters["preferred_area"] == "Indiranagar"
        assert filters["max_cost_for_two"] == 1000


@pytest.mark.streamlit
def test_render_sidebar():
    """Verifies sidebar controls rendering."""
    with patch("streamlit.sidebar.radio", return_value="Anonymous / New User (Cold-Start)"), \
         patch("streamlit.sidebar.slider", return_value=10), \
         patch("streamlit.sidebar.toggle", return_value=True), \
         patch("streamlit.sidebar.button", return_value=False):
        controls = render_sidebar()
        assert controls["top_k"] == 10
        assert controls["mmr_enabled"] is True


@pytest.mark.streamlit
def test_render_recommendations_page_empty_and_populated():
    """Verifies recommendations view handles both empty and populated states."""
    # Empty
    st.session_state["last_recommendation_response"] = None
    with patch("streamlit.info") as mock_info:
        render_recommendations_page()
        assert mock_info.called

    # Populated
    st.session_state["last_recommendation_response"] = {
        "count": 1,
        "is_cold_start": True,
        "strategy": "global_popularity",
        "recommendations": [{
            "restaurant_id": 10,
            "name": "Empire",
            "rating": 4.1,
            "review_count": 500,
            "cost_for_two_inr": 600,
            "price_tier": "Moderate",
            "cuisines": ["Mughlai"],
            "area": "Frazer Town",
            "online_order": True,
            "book_table": False,
            "explanation": "Popular in Bengaluru"
        }]
    }
    with patch("streamlit.markdown") as mock_md:
        render_recommendations_page()
        assert mock_md.called

