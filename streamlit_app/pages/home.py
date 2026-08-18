# -*- coding: utf-8 -*-
"""
Home Page: Multi-Tab Interactive Restaurant Recommendation Discovery
"""

import streamlit as st
from typing import Dict, Any, List
from streamlit_app.api_client import api_client
from streamlit_app.components.header import render_header
from streamlit_app.components.filters import render_preference_filters
from streamlit_app.components.recommendation_card import render_recommendation_card
from streamlit_app.components.diversity_panel import render_diversity_panel
from streamlit_app.components.metrics_panel import render_metrics_panel
from streamlit_app.config import (
    BENGALURU_LOCALITIES,
    POPULAR_CUISINES,
    PRICE_TIERS,
    DINING_TYPES,
    DEFAULT_LATITUDE,
    DEFAULT_LONGITUDE,
    DEFAULT_SEARCH_RADIUS_KM
)


def render_home_page():
    """Renders the main discovery interface with 4 interactive discovery tabs."""
    render_header()

    tab1, tab2, tab3, tab4 = st.tabs([
        "✨ Personalized Discovery",
        "🚀 Cold-Start Onboarding",
        "🔥 Popularity Rankings",
        "📍 Find Near Me (Spatial)"
    ])

    # =========================================================================
    # TAB 1: Personalized & Preference Hybrid Discovery
    # =========================================================================
    with tab1:
        st.markdown("### 🎯 Find Your Ideal Dining Experience")
        st.caption("Blends content preferences, SVD collaborative signals (for known users), quality priors, and MMR diversity.")

        prefs = render_preference_filters()

        col_btn1, col_btn2 = st.columns([3, 1])
        with col_btn1:
            submit = st.button("🍽️ Generate Hybrid Recommendations", type="primary", use_container_width=True)
        with col_btn2:
            if st.button("🔄 Reset Filters", use_container_width=True):
                st.rerun()

        if submit:
            with st.spinner("Finding restaurants that match your preferences..."):
                payload: Dict[str, Any] = {
                    "user_id": st.session_state.get("user_id"),
                    "preferred_cuisines": prefs.get("preferred_cuisines"),
                    "preferred_area": prefs.get("preferred_area"),
                    "preferred_price_tier": prefs.get("preferred_price_tier"),
                    "preferred_type": prefs.get("preferred_type"),
                    "max_cost_for_two": prefs.get("max_cost_for_two"),
                    "online_order_only": prefs.get("online_order_only", False),
                    "book_table_only": prefs.get("book_table_only", False),
                    "top_k": st.session_state.get("top_k", 10),
                    "mmr_enabled": st.session_state.get("mmr_enabled", True),
                    "mmr_lambda": st.session_state.get("mmr_lambda", 0.75)
                }

                res, err = api_client.get_hybrid_recommendations(payload)
                if err:
                    st.error(f"**Error [{err.get('error_code')}]:** {err.get('message')}")
                    if err.get("request_id") != "N/A":
                        st.caption(f"Request ID: `{err.get('request_id')}`")
                elif res:
                    st.session_state["last_recommendation_response"] = res
                    st.session_state["last_query_type"] = "hybrid"

        # Render Active Recommendations if available
        if st.session_state.get("last_query_type") == "hybrid" and st.session_state.get("last_recommendation_response"):
            resp = st.session_state["last_recommendation_response"]
            st.divider()
            render_metrics_panel(resp)
            render_diversity_panel(resp.get("diversification"))
            
            recs = resp.get("recommendations", [])
            st.markdown(f"### 📋 Recommended Restaurants ({len(recs)})")
            for idx, item in enumerate(recs, start=1):
                render_recommendation_card(item, rank=idx)

    # =========================================================================
    # TAB 2: Cold-Start Onboarding Wizard
    # =========================================================================
    with tab2:
        st.markdown("### 👋 New to Bengaluru Restaurant Intelligence?")
        st.caption("Tell us your dining tastes to boot an instant personalized profile without needing historical ratings.")

        with st.form("onboarding_form"):
            ob_cuisines = st.multiselect(
                "1. What are your favorite cuisines?",
                options=POPULAR_CUISINES,
                default=["Biryani", "South Indian", "North Indian"],
                help="Select at least 1 cuisine."
            )

            ob_col1, ob_col2 = st.columns(2)
            with ob_col1:
                ob_area = st.selectbox(
                    "2. Primary Bengaluru Locality",
                    options=["No Preference"] + sorted(BENGALURU_LOCALITIES)
                )
                ob_dining = st.multiselect(
                    "3. Preferred Dining Formats",
                    options=DINING_TYPES,
                    default=["Casual Dining", "Quick Bites"]
                )
            with ob_col2:
                ob_price_tier = st.selectbox(
                    "4. Preferred Price Tier",
                    options=PRICE_TIERS,
                    index=1
                )
                ob_budget = st.select_slider(
                    "5. Maximum Budget for Two (₹)",
                    options=[300, 500, 700, 1000, 1500, 2000, 3000],
                    value=1000,
                    format_func=lambda x: f"₹{x:,}"
                )

            ob_col3, ob_col4 = st.columns(2)
            with ob_col3:
                ob_pure_veg = st.checkbox("🥗 I prefer Pure Veg restaurants")
            with ob_col4:
                ob_online = st.checkbox("🛵 I frequently order food online")

            ob_submit = st.form_submit_button("🚀 Complete Onboarding & Get Recommendations", type="primary", use_container_width=True)

        if ob_submit:
            if not ob_cuisines:
                st.warning("Please select at least one favorite cuisine.")
            else:
                with st.spinner("Bootstrapping personalized cold-start recommendations..."):
                    ob_payload = {
                        "favorite_cuisines": ob_cuisines,
                        "preferred_dining_types": ob_dining if ob_dining else None,
                        "preferred_area": ob_area if ob_area != "No Preference" else None,
                        "price_tier": ob_price_tier,
                        "max_budget_for_two": ob_budget,
                        "is_pure_veg_preferred": ob_pure_veg,
                        "online_ordering": ob_online,
                        "top_k": st.session_state.get("top_k", 10),
                        "mmr_enabled": st.session_state.get("mmr_enabled", True),
                        "mmr_lambda": st.session_state.get("mmr_lambda", 0.75)
                    }

                    res, err = api_client.get_onboarding_recommendations(ob_payload)
                    if err:
                        st.error(f"**Error [{err.get('error_code')}]:** {err.get('message')}")
                    elif res:
                        st.session_state["last_recommendation_response"] = res
                        st.session_state["last_query_type"] = "onboarding"

        if st.session_state.get("last_query_type") == "onboarding" and st.session_state.get("last_recommendation_response"):
            resp = st.session_state["last_recommendation_response"]
            st.divider()
            render_metrics_panel(resp)
            render_diversity_panel(resp.get("diversification"))
            
            recs = resp.get("recommendations", [])
            st.markdown(f"### 📋 Onboarding Recommendations ({len(recs)})")
            for idx, item in enumerate(recs, start=1):
                render_recommendation_card(item, rank=idx)

    # =========================================================================
    # TAB 3: Locality & Cuisine Popularity (Bayesian Priors)
    # =========================================================================
    with tab3:
        st.markdown("### 🔥 Popularity Rankings (Bayesian Priors)")
        st.caption("Ranks Bengaluru venues with Bayesian shrinkage over review volume and star ratings.")

        pcol1, pcol2, pcol3 = st.columns(3)
        with pcol1:
            pop_scope = st.radio("Popularity Scope", ["Global Bengaluru", "By Locality", "By Cuisine"], horizontal=True)
        with pcol2:
            pop_area = st.selectbox(
                "Select Locality",
                options=sorted(BENGALURU_LOCALITIES),
                disabled=(pop_scope != "By Locality")
            )
        with pcol3:
            pop_cuisine = st.selectbox(
                "Select Cuisine",
                options=POPULAR_CUISINES,
                disabled=(pop_scope != "By Cuisine")
            )

        pop_submit = st.button("🔥 Fetch Popular Restaurants", type="primary", use_container_width=True)

        if pop_submit:
            with st.spinner("Fetching Bayesian popularity rankings..."):
                target_area = pop_area if pop_scope == "By Locality" else None
                target_cuisine = pop_cuisine if pop_scope == "By Cuisine" else None

                res, err = api_client.get_popular_restaurants(
                    area=target_area,
                    cuisine=target_cuisine,
                    top_k=st.session_state.get("top_k", 10)
                )

                if err:
                    st.error(f"**Error [{err.get('error_code')}]:** {err.get('message')}")
                elif res:
                    st.session_state["last_recommendation_response"] = res
                    st.session_state["last_query_type"] = "popular"

        if st.session_state.get("last_query_type") == "popular" and st.session_state.get("last_recommendation_response"):
            resp = st.session_state["last_recommendation_response"]
            st.divider()
            st.info(f"**Popularity Scope:** `{resp.get('scope', 'global')}` &bull; Total: `{resp.get('count', 0)}` venues")
            recs = resp.get("recommendations", [])
            for idx, item in enumerate(recs, start=1):
                render_recommendation_card(item, rank=idx)

    # =========================================================================
    # TAB 4: Spatial Radius Search ("Find Restaurants Near Me")
    # =========================================================================
    with tab4:
        st.markdown("### 📍 Find Restaurants Near Me (Spatial BallTree)")
        st.caption("Fast nearest-neighbor spatial search using great-circle Haversine distance over Bengaluru locality centroids.")

        st.info("ℹ️ **Locality Precision Notice:** Location estimates are computed from Bengaluru locality-level centroid coordinates.")

        scol1, scol2, scol3 = st.columns(3)
        with scol1:
            lat = st.number_input(
                "Latitude (°N)",
                min_value=12.0,
                max_value=14.0,
                value=float(st.session_state.get("latitude", DEFAULT_LATITUDE)),
                format="%.4f"
            )
        with scol2:
            lon = st.number_input(
                "Longitude (°E)",
                min_value=76.0,
                max_value=78.5,
                value=float(st.session_state.get("longitude", DEFAULT_LONGITUDE)),
                format="%.4f"
            )
        with scol3:
            radius_km = st.slider(
                "Search Radius (km)",
                min_value=0.5,
                max_value=15.0,
                value=float(st.session_state.get("radius_km", DEFAULT_SEARCH_RADIUS_KM)),
                step=0.5
            )

        # Quick preset selector
        st.markdown("**Quick Bengaluru Center Presets:**")
        preset_cols = st.columns(4)
        with preset_cols[0]:
            if st.button("📍 MG Road / Central", use_container_width=True):
                st.session_state["latitude"] = 12.9716
                st.session_state["longitude"] = 77.5946
                st.rerun()
        with preset_cols[1]:
            if st.button("📍 Indiranagar", use_container_width=True):
                st.session_state["latitude"] = 12.9784
                st.session_state["longitude"] = 77.6408
                st.rerun()
        with preset_cols[2]:
            if st.button("📍 Koramangala", use_container_width=True):
                st.session_state["latitude"] = 12.9352
                st.session_state["longitude"] = 77.6245
                st.rerun()
        with preset_cols[3]:
            if st.button("📍 Whitefield", use_container_width=True):
                st.session_state["latitude"] = 12.9698
                st.session_state["longitude"] = 77.7499
                st.rerun()

        spatial_submit = st.button("🔍 Search Nearby Outlets", type="primary", use_container_width=True)

        if spatial_submit:
            with st.spinner(f"Querying BallTree spatial index for restaurants within {radius_km} km..."):
                res, err = api_client.get_nearby_restaurants(
                    latitude=lat,
                    longitude=lon,
                    radius_km=radius_km,
                    top_k=st.session_state.get("top_k", 10)
                )

                if err:
                    st.error(f"**Error [{err.get('error_code')}]:** {err.get('message')}")
                elif res:
                    st.session_state["last_recommendation_response"] = res
                    st.session_state["last_query_type"] = "nearby"

        if st.session_state.get("last_query_type") == "nearby" and st.session_state.get("last_recommendation_response"):
            resp = st.session_state["last_recommendation_response"]
            st.divider()
            st.success(f"Found **{resp.get('count', 0)}** nearby restaurants within **{resp.get('radius_km', radius_km)} km** of `({resp.get('latitude'):.4f}, {resp.get('longitude'):.4f})`.")
            recs = resp.get("recommendations", [])
            for idx, item in enumerate(recs, start=1):
                render_recommendation_card(item, rank=idx)

