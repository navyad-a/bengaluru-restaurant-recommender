# -*- coding: utf-8 -*-
"""
Filters Component: Preference Selectors & Constraint Inputs
"""

import streamlit as st
from typing import Dict, Any, List
from streamlit_app.config import (
    BENGALURU_LOCALITIES,
    POPULAR_CUISINES,
    PRICE_TIERS,
    DINING_TYPES,
    BUDGET_OPTIONS,
    CURRENCY_SYMBOL
)
from streamlit_app.state import reset_filters


def render_preference_filters() -> Dict[str, Any]:
    """
    Renders input form for cuisines, locality, budget, price tiers, and hard filters.
    Returns the parsed user preference dictionary.
    """
    st.markdown("#### 🎯 Dining Preferences & Constraints")

    col1, col2 = st.columns(2)
    with col1:
        selected_cuisines = st.multiselect(
            "Preferred Cuisines",
            options=POPULAR_CUISINES,
            default=st.session_state.get("selected_cuisines", ["Biryani", "North Indian"]),
            help="Select one or more preferred cuisines."
        )
        st.session_state["selected_cuisines"] = selected_cuisines

        area_options = ["All Localities"] + sorted(BENGALURU_LOCALITIES)
        current_area = st.session_state.get("selected_area", "All Localities")
        area_idx = area_options.index(current_area) if current_area in area_options else 0
        selected_area = st.selectbox(
            "Preferred Bengaluru Locality",
            options=area_options,
            index=area_idx,
            help="Filter recommendations to a specific Bengaluru neighborhood."
        )
        st.session_state["selected_area"] = selected_area

    with col2:
        max_budget = st.select_slider(
            f"Maximum Budget for Two ({CURRENCY_SYMBOL})",
            options=[300, 500, 700, 1000, 1500, 2000, 3000, 5000],
            value=st.session_state.get("max_budget_for_two", 1000),
            format_func=lambda x: f"{CURRENCY_SYMBOL}{x:,}",
            help="Hard maximum cost for two."
        )
        st.session_state["max_budget_for_two"] = max_budget

        tier_options = ["Any Tier"] + PRICE_TIERS
        current_tier = st.session_state.get("selected_price_tier", "Any Tier")
        tier_idx = tier_options.index(current_tier) if current_tier in tier_options else 0
        selected_price_tier = st.selectbox(
            "Price Tier",
            options=tier_options,
            index=tier_idx,
            help="Budget (<=₹400), Moderate (₹400-₹800), Premium (₹800-₹1500), Luxury (>₹1500)"
        )
        st.session_state["selected_price_tier"] = selected_price_tier

    # Additional options
    with st.expander("➕ More Dining Formats & Hard Constraints", expanded=False):
        ecol1, ecol2 = st.columns(2)
        with ecol1:
            selected_dining_types = st.multiselect(
                "Dining Format",
                options=DINING_TYPES,
                default=st.session_state.get("selected_dining_types", []),
                help="E.g. Casual Dining, Quick Bites, Cafe, Fine Dining"
            )
            st.session_state["selected_dining_types"] = selected_dining_types
        with ecol2:
            st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
            online_order = st.checkbox(
                "🛵 Online Ordering Only",
                value=st.session_state.get("online_order_only", False)
            )
            book_table = st.checkbox(
                "📅 Table Booking Only",
                value=st.session_state.get("book_table_only", False)
            )
            is_pure_veg = st.checkbox(
                "🥗 Pure Veg Preferred",
                value=st.session_state.get("is_pure_veg", False)
            )
            st.session_state["online_order_only"] = online_order
            st.session_state["book_table_only"] = book_table
            st.session_state["is_pure_veg"] = is_pure_veg

    return {
        "preferred_cuisines": selected_cuisines if selected_cuisines else None,
        "preferred_area": selected_area if selected_area != "All Localities" else None,
        "max_cost_for_two": max_budget,
        "preferred_price_tier": selected_price_tier if selected_price_tier != "Any Tier" else None,
        "preferred_type": ", ".join(selected_dining_types) if selected_dining_types else None,
        "online_order_only": online_order,
        "book_table_only": book_table,
        "is_pure_veg": is_pure_veg
    }

