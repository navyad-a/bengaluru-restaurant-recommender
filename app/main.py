# -*- coding: utf-8 -*-
from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from app.config import settings
from app.api.v1.router import api_router
from app.core.lifespan import lifespan
from app.core.errors import (
    http_exception_handler,
    validation_exception_handler,
    global_exception_handler
)
from app.middleware.request_id import RequestIDMiddleware
from app.middleware.timing import TimingMiddleware
from app.middleware.rate_limit import RateLimitMiddleware
from app.services.recommendation_service import (
    get_content_recommender,
    get_collaborative_recommender,
    get_hybrid_recommender,
    get_spatial_search_engine
)

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Production-quality Hybrid Indian Restaurant Recommendation Engine combining TF-IDF Content Similarity, Location-Aware Proximity, Bayesian Quality Shrinkage, and MMR Diversification.",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# Exception Handlers
app.add_exception_handler(StarletteHTTPException, http_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(Exception, global_exception_handler)

# Middlewares (Executed in reverse order of addition)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(TimingMiddleware)
app.add_middleware(RequestIDMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API v1 Router
app.include_router(api_router, prefix=settings.API_V1_STR)


@app.get("/", tags=["Root"])
async def root():
    return {
        "message": f"Welcome to {settings.APP_NAME}",
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "status": "healthy"
    }


@app.get("/health", tags=["Health & Readiness"])
async def health_check():
    """Lightweight liveness probe - immediate return."""
    return {
        "status": "healthy",
        "app_name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "environment": settings.APP_ENV
    }


@app.get("/ready", tags=["Health & Readiness"])
async def readiness_check():
    """Readiness probe - verifies all recommendation engines and indexes are initialized."""
    readiness_report = {
        "status": "ready",
        "checks": {
            "content_recommender": False,
            "collaborative_recommender": False,
            "spatial_search_engine": False,
            "hybrid_recommender": False,
            "database_configured": "postgresql+asyncpg" in settings.DATABASE_URL
        }
    }

    try:
        c_rec = get_content_recommender()
        readiness_report["checks"]["content_recommender"] = len(c_rec.engine.restaurant_catalog) > 0
    except Exception:
        readiness_report["status"] = "not_ready"

    try:
        collab_rec = get_collaborative_recommender()
        readiness_report["checks"]["collaborative_recommender"] = (
            collab_rec.engine.model is not None or len(collab_rec.user_rated_items) > 0
        )
    except Exception:
        readiness_report["status"] = "not_ready"

    try:
        spatial_eng = get_spatial_search_engine()
        readiness_report["checks"]["spatial_search_engine"] = (
            spatial_eng.index is not None and spatial_eng.index.tree is not None
        )
    except Exception:
        readiness_report["status"] = "not_ready"

    try:
        hybrid_rec = get_hybrid_recommender()
        readiness_report["checks"]["hybrid_recommender"] = hybrid_rec is not None
    except Exception:
        readiness_report["status"] = "not_ready"

    return readiness_report


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)

