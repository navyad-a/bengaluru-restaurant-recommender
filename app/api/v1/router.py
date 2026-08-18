# -*- coding: utf-8 -*-
"""
API v1 Router Aggregator
"""

from fastapi import APIRouter
from app.api.v1.endpoints import recommendations, system

api_router = APIRouter()

api_router.include_router(
    recommendations.router,
    prefix="/recommendations",
    tags=["Recommendations"]
)

api_router.include_router(
    system.router,
    prefix="/system",
    tags=["System Telemetry"]
)
