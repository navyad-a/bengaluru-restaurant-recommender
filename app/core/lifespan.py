# -*- coding: utf-8 -*-
"""
FastAPI Application Lifespan Management
Handles startup singleton pre-warming and graceful shutdown.
"""

import time
from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.core.logging import logger
from app.core.cache import get_recommendation_cache
from app.services.recommendation_service import (
    get_content_recommender,
    get_collaborative_recommender,
    get_hybrid_recommender,
    get_spatial_search_engine
)

app_start_time = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI.
    Pre-warms all recommendation models, spatial indexes, and caches on startup.
    """
    global app_start_time
    app_start_time = time.time()
    logger.info("Starting application initialization and singleton pre-warming...")
    t0 = time.perf_counter()

    try:
        # Pre-warm Content Recommender & TF-IDF
        t_start = time.perf_counter()
        content_rec = get_content_recommender()
        logger.info(f"  [+] ContentRecommender pre-warmed ({len(content_rec.engine.restaurant_catalog):,} items) in {(time.perf_counter() - t_start)*1000:.1f} ms")

        # Pre-warm Collaborative Recommender & SVD
        t_start = time.perf_counter()
        collab_rec = get_collaborative_recommender()
        logger.info(f"  [+] CollaborativeRecommender pre-warmed ({len(collab_rec.user_rated_items):,} train users) in {(time.perf_counter() - t_start)*1000:.1f} ms")

        # Pre-warm Spatial Search Engine & BallTree
        t_start = time.perf_counter()
        spatial_engine = get_spatial_search_engine()
        logger.info(f"  [+] SpatialSearchEngine pre-warmed (BallTree initialized) in {(time.perf_counter() - t_start)*1000:.1f} ms")

        # Pre-warm Hybrid Recommender & MMR Engine
        t_start = time.perf_counter()
        hybrid_rec = get_hybrid_recommender()
        logger.info(f"  [+] HybridRecommender pre-warmed in {(time.perf_counter() - t_start)*1000:.1f} ms")

        # Pre-warm Cache
        cache = get_recommendation_cache()
        logger.info(f"  [+] RecommendationCache initialized (enabled={cache.enabled}, max_size={cache.max_size})")

        total_init_ms = (time.perf_counter() - t0) * 1000
        logger.info(f"All recommendation engines successfully pre-warmed in {total_init_ms:.1f} ms. System ready for traffic!")
    except Exception as e:
        logger.error(f"Error during startup pre-warming: {e}", exc_info=True)

    yield

    logger.info("Shutting down application. Cleaning up caches and singleton resources...")
    try:
        cache = get_recommendation_cache()
        cache.clear()
        logger.info("RecommendationCache cleared successfully.")
    except Exception as e:
        logger.warning(f"Error during shutdown cleanup: {e}")

