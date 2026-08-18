# -*- coding: utf-8 -*-
"""
System Status Page: Production Telemetry & Model Health
"""

import streamlit as st
from streamlit_app.api_client import api_client
from streamlit_app.config import API_BASE_URL


def render_system_status_page():
    """Renders real-time telemetry, model readiness checks, and recommendation cache statistics."""
    st.markdown("## ⚙️ Backend Telemetry & Model Readiness")
    st.caption(f"Connected to FastAPI Production Service at `{API_BASE_URL}`")

    # 1. Probe Backend
    with st.spinner("Fetching system telemetry and model health..."):
        ready_data, ready_err = api_client.get_ready()
        status_data, status_err = api_client.get_system_status()

    # 2. Readiness Overview
    st.markdown("### 🔍 Model Readiness Probes (`/ready`)")
    if ready_err:
        st.error(f"❌ Backend Unavailable: {ready_err.get('message')}")
    elif ready_data:
        is_ready = ready_data.get("status") == "ready"
        if is_ready:
            st.success("✅ **FastAPI Service Status: READY** — All recommendation engines loaded in memory.")
        else:
            st.warning("⚠️ **FastAPI Service Status: PARTIAL / NOT READY**")

        checks = ready_data.get("checks", {})
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Content TF-IDF", "✅ Ready" if checks.get("content_recommender") else "❌ Error")
        with col2:
            st.metric("Collab SVD", "✅ Ready" if checks.get("collaborative_recommender") else "❌ Error")
        with col3:
            st.metric("Spatial BallTree", "✅ Ready" if checks.get("spatial_search_engine") else "❌ Error")
        with col4:
            st.metric("Hybrid Engine", "✅ Ready" if checks.get("hybrid_recommender") else "❌ Error")
        with col5:
            st.metric("Database", "✅ Ready" if checks.get("database_configured") else "❌ Error")

    st.divider()

    # 3. Runtime Telemetry
    if status_data:
        st.markdown("### 📊 Runtime Telemetry (`/api/v1/system/status`)")
        
        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        with mcol1:
            uptime = status_data.get("uptime_seconds", 0.0)
            st.metric("Uptime", f"{uptime:.1f}s")
        with mcol2:
            mem = status_data.get("memory", {})
            st.metric("Process RSS Memory", f"{mem.get('rss_mb', 0.0):.1f} MB")
        with mcol3:
            catalog = status_data.get("catalog", {})
            st.metric("Authentic Catalog", f"{catalog.get('total_restaurants', 12481):,} Outlets")
        with mcol4:
            concurrency = status_data.get("concurrency", {})
            st.metric("Worker Threads", f"{concurrency.get('thread_pool_workers', 8)}")

        # Cache Telemetry
        st.markdown("### ⚡ Recommendation Cache Analytics")
        cache_stats = status_data.get("cache", {})
        
        ccol1, ccol2, ccol3, ccol4, ccol5 = st.columns(5)
        with ccol1:
            st.metric("Cache Enabled", "Yes" if cache_stats.get("enabled") else "No")
        with ccol2:
            st.metric("Cached Keys", f"{cache_stats.get('total_keys', 0)} / {cache_stats.get('max_size', 1000)}")
        with ccol3:
            st.metric("Cache Hits", f"{cache_stats.get('hits', 0)}")
        with ccol4:
            st.metric("Cache Misses", f"{cache_stats.get('misses', 0)}")
        with ccol5:
            st.metric("Hit Ratio", f"{cache_stats.get('hit_ratio_pct', 0.0):.1f}%")

        if st.button("🗑️ Purge Recommendation Cache", type="secondary"):
            clear_res, clear_err = api_client.clear_cache()
            if clear_res and clear_res.get("status") == "success":
                st.success("Cache cleared successfully.")
                st.rerun()
            else:
                st.error("Failed to clear cache.")

