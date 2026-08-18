# -*- coding: utf-8 -*-
"""
System Health, Readiness, and Telemetry Endpoints
"""

import time
from typing import Dict, Any
from fastapi import APIRouter, status
from app.config import settings
from app.core.cache import get_recommendation_cache
from app.core.lifespan import app_start_time
from app.services.recommendation_service import (
    get_content_recommender,
    get_collaborative_recommender,
    get_hybrid_recommender,
    get_spatial_search_engine
)

router = APIRouter()


@router.get("/status", summary="Detailed system health and status telemetry")
async def get_system_status() -> Dict[str, Any]:
    """Returns system uptime, cache statistics, model readiness, and configuration."""
    cache_stats = get_recommendation_cache().get_stats()
    uptime_seconds = round(time.time() - app_start_time, 2)

    # Check model loading status
    models_ready = {}
    try:
        c_rec = get_content_recommender()
        models_ready["content_recommender"] = {
            "status": "ready",
            "catalog_size": len(c_rec.engine.restaurant_catalog)
        }
    except Exception as e:
        models_ready["content_recommender"] = {"status": "error", "error": str(e)}

    try:
        collab_rec = get_collaborative_recommender()
        models_ready["collaborative_recommender"] = {
            "status": "ready",
            "train_users": len(collab_rec.user_rated_items)
        }
    except Exception as e:
        models_ready["collaborative_recommender"] = {"status": "error", "error": str(e)}

    try:
        spatial_eng = get_spatial_search_engine()
        models_ready["spatial_search_engine"] = {"status": "ready"}
    except Exception as e:
        models_ready["spatial_search_engine"] = {"status": "error", "error": str(e)}

    try:
        hybrid_rec = get_hybrid_recommender()
        models_ready["hybrid_recommender"] = {"status": "ready"}
    except Exception as e:
        models_ready["hybrid_recommender"] = {"status": "error", "error": str(e)}

    return {
        "status": "healthy",
        "app_name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "environment": settings.APP_ENV,
        "uptime_seconds": uptime_seconds,
        "models": models_ready,
        "cache": cache_stats,
        "database": {
            "configured": "postgresql+asyncpg" in settings.DATABASE_URL
        },
        "concurrency": {
            "thread_pool_workers": settings.THREAD_POOL_WORKERS
        }
    }


@router.post("/cache/clear", summary="Purge in-memory recommendation cache")
async def clear_cache() -> Dict[str, Any]:
    """Clears all cached recommendation responses."""
    cache = get_recommendation_cache()
    cache.clear()
    return {"status": "success", "message": "Recommendation cache cleared successfully."}

