# -*- coding: utf-8 -*-
"""
Streamlit Session State Initialization & Management
"""

import streamlit as st
from typing import Dict, Any, Optional, List
from streamlit_app.config import (
    DEFAULT_TOP_K,
    DEFAULT_MMR_ENABLED,
    DEFAULT_MMR_LAMBDA,
    DEFAULT_SEARCH_RADIUS_KM,
    DEFAULT_LATITUDE,
    DEFAULT_LONGITUDE
)


def init_session_state():
    """Initializes Streamlit session state keys with default values if not already present."""
    defaults: Dict[str, Any] = {
        # User Context
        "user_id": None,
        "is_known_user": False,
        
        # User Preference Filters
        "selected_cuisines": ["Biryani", "North Indian"],
        "selected_area": "All Localities",
        "max_budget_for_two": 1000,
        "selected_price_tier": "Any Tier",
        "selected_dining_types": [],
        "online_order_only": False,
        "book_table_only": False,
        "is_pure_veg": False,
        
        # Spatial Coordinates & Radius
        "latitude": DEFAULT_LATITUDE,
        "longitude": DEFAULT_LONGITUDE,
        "radius_km": DEFAULT_SEARCH_RADIUS_KM,
        
        # Recommendation Controls
        "top_k": DEFAULT_TOP_K,
        "mmr_enabled": DEFAULT_MMR_ENABLED,
        "mmr_lambda": DEFAULT_MMR_LAMBDA,
        
        # Recommendation Cache/Results in View
        "last_recommendation_response": None,
        "last_recommendation_error": None,
        "last_query_type": None,
        "is_loading": False,
        
        # Navigation
        "active_tab": "Personalized Discovery"
    }

    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def reset_filters():
    """Resets all recommendation filters to clean initial states."""
    st.session_state["selected_cuisines"] = []
    st.session_state["selected_area"] = "All Localities"
    st.session_state["max_budget_for_two"] = 1000
    st.session_state["selected_price_tier"] = "Any Tier"
    st.session_state["selected_dining_types"] = []
    st.session_state["online_order_only"] = False
    st.session_state["book_table_only"] = False
    st.session_state["is_pure_veg"] = False
    st.session_state["top_k"] = DEFAULT_TOP_K
    st.session_state["mmr_enabled"] = DEFAULT_MMR_ENABLED
    st.session_state["mmr_lambda"] = DEFAULT_MMR_LAMBDA


def clear_recommendation_results():
    """Clears stored recommendation results and error payloads."""
    st.session_state["last_recommendation_response"] = None
    st.session_state["last_recommendation_error"] = None
    st.session_state["last_query_type"] = None

