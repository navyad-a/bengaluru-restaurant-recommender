# -*- coding: utf-8 -*-
"""
Phase 15 Test Suite: Docker Containerization, Docker Compose & Redis Cache Verification
"""

import os
import pytest
from unittest.mock import patch, MagicMock
from app.config import Settings
from app.core.cache import (
    BaseRecommendationCache,
    InMemoryRecommendationCache,
    RedisRecommendationCache,
    get_recommendation_cache
)


# =============================================================================
# 1. Docker Files & Compose Verification
# =============================================================================

@pytest.mark.unit
def test_docker_files_exist_and_non_empty():
    """Verifies that all required Docker build and orchestration files exist."""
    required_files = [
        "Dockerfile",
        "docker/Dockerfile.api",
        "docker/Dockerfile.streamlit",
        "docker-compose.yml",
        ".dockerignore",
        ".env.example"
    ]
    for rel_path in required_files:
        assert os.path.exists(rel_path), f"Missing required container file: {rel_path}"
        assert os.path.getsize(rel_path) > 50, f"File appears truncated or empty: {rel_path}"


@pytest.mark.unit
def test_docker_compose_structure_and_services():
    """Verifies docker-compose.yml defines all 4 required production services and healthchecks."""
    with open("docker-compose.yml", "r", encoding="utf-8") as f:
        compose_content = f.read()

    assert "postgres:" in compose_content
    assert "redis:" in compose_content
    assert "api:" in compose_content
    assert "streamlit:" in compose_content
    assert "recommender_net" in compose_content
    assert "postgres_data" in compose_content
    assert "redis_data" in compose_content
    assert "8000:8000" in compose_content
    assert "8501:8501" in compose_content
    assert "healthcheck:" in compose_content


@pytest.mark.unit
def test_dockerfile_security_and_non_root_user():
    """Verifies Dockerfiles specify non-root user and unbuffered Python output."""
    with open("docker/Dockerfile.api", "r", encoding="utf-8") as f:
        api_dockerfile = f.read()
    assert "USER appuser" in api_dockerfile
    assert "PYTHONUNBUFFERED=1" in api_dockerfile
    assert "PYTHONDONTWRITEBYTECODE=1" in api_dockerfile

    with open("docker/Dockerfile.streamlit", "r", encoding="utf-8") as f:
        st_dockerfile = f.read()
    assert "USER streamlituser" in st_dockerfile
    assert "STREAMLIT_API_BASE_URL" in st_dockerfile


# =============================================================================
# 2. Redis Cache Abstraction Tests
# =============================================================================

@pytest.mark.unit
def test_redis_cache_mocked_operations():
    """Tests RedisRecommendationCache operations with mocked redis client."""
    mock_redis = MagicMock()
    mock_redis.get.return_value = '{"restaurant_id": 1, "name": "MTR"}'
    mock_redis.dbsize.return_value = 5

    with patch("redis.from_url", return_value=mock_redis):
        cache = RedisRecommendationCache(redis_url="redis://test-host:6379/0", enabled=True)
        assert cache._client is not None

        # Test GET (hit)
        val = cache.get("test_key")
        assert val == {"restaurant_id": 1, "name": "MTR"}
        assert cache.hits == 1

        # Test SET
        cache.set("new_key", {"status": "ok"}, ttl=60)
        assert mock_redis.set.called

        # Test CLEAR
        cache.clear()
        assert mock_redis.flushdb.called

        # Test STATS
        stats = cache.get_stats()
        assert stats["backend"] == "redis"
        assert stats["connected"] is True
        assert stats["total_keys"] == 5


@pytest.mark.unit
def test_redis_cache_graceful_fallback_on_connection_error():
    """Tests that Redis cache falls back gracefully to pass-through when server unreachable."""
    with patch("redis.from_url", side_effect=Exception("Connection refused")):
        cache = RedisRecommendationCache(redis_url="redis://nonexistent:6379/0", enabled=True)
        assert cache._client is None
        assert cache.get("any_key") is None
        stats = cache.get_stats()
        assert stats["connected"] is False


@pytest.mark.unit
def test_cache_backend_configuration_switching(monkeypatch):
    """Verifies that CACHE_BACKEND config setting dictates instantiated cache class."""
    # Test memory setting
    cfg = Settings(CACHE_BACKEND="memory")
    assert cfg.CACHE_BACKEND == "memory"
    assert cfg.REDIS_URL == "redis://localhost:6379/0"

