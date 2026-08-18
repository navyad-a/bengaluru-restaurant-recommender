# -*- coding: utf-8 -*-
"""
Phase 14 Test Suite: Backend Unit Tests (Config, Middleware, Lifespan, Errors)
"""

import pytest
import os
import time
from unittest.mock import patch, MagicMock
from fastapi import Request, HTTPException, status
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient
from app.config import Settings
from app.core.errors import (
    http_exception_handler,
    validation_exception_handler,
    global_exception_handler
)
from app.core.lifespan import lifespan
from app.middleware.rate_limit import RateLimitMiddleware
from app.main import app


@pytest.mark.unit
def test_config_defaults_and_types():
    """Verifies Settings defaults and types."""
    cfg = Settings()
    assert cfg.APP_NAME != ""
    assert cfg.DEFAULT_WEIGHT_CF == 0.40
    assert cfg.DEFAULT_WEIGHT_CONTENT == 0.35
    assert cfg.DEFAULT_WEIGHT_LOCATION == 0.15
    assert cfg.DEFAULT_WEIGHT_QUALITY == 0.10
    assert cfg.DEFAULT_TOP_K == 10
    assert cfg.THREAD_POOL_WORKERS >= 1
    assert cfg.RECOMMENDATION_CACHE_ENABLED is True
    assert cfg.RECOMMENDATION_CACHE_TTL_SECONDS == 300


@pytest.mark.unit
def test_config_env_overrides(monkeypatch):
    """Verifies environment variables correctly override configuration defaults."""
    monkeypatch.setenv("DEFAULT_TOP_K", "25")
    monkeypatch.setenv("THREAD_POOL_WORKERS", "16")
    monkeypatch.setenv("RECOMMENDATION_CACHE_TTL_SECONDS", "600")
    
    cfg = Settings()
    assert cfg.DEFAULT_TOP_K == 25
    assert cfg.THREAD_POOL_WORKERS == 16
    assert cfg.RECOMMENDATION_CACHE_TTL_SECONDS == 600


@pytest.mark.unit
@pytest.mark.asyncio
async def test_rate_limiter_middleware_sliding_window():
    """Tests rate limiter allows allowed requests and blocks when threshold exceeded."""
    from app.config import settings
    
    async def dummy_app(request):
        from starlette.responses import PlainTextResponse
        return PlainTextResponse("OK")

    limiter = RateLimitMiddleware(app=dummy_app)
    limiter.max_requests = 2

    # Create mock request with client host
    mock_req = MagicMock(spec=Request)
    mock_req.url.path = "/api/v1/recommendations/hybrid"
    mock_req.client.host = "192.168.1.100"
    mock_req.state.request_id = "req-1"

    with patch.object(settings, "RATE_LIMIT_ENABLED", True):
        # Request 1: Allowed
        resp1 = await limiter.dispatch(mock_req, dummy_app)
        assert resp1.status_code == 200

        # Request 2: Allowed
        resp2 = await limiter.dispatch(mock_req, dummy_app)
        assert resp2.status_code == 200

        # Request 3: Blocked (429)
        resp3 = await limiter.dispatch(mock_req, dummy_app)
        assert resp3.status_code == 429
        import json
        body = json.loads(resp3.body.decode())
        assert body["error_code"] == "RATE_LIMIT_EXCEEDED"

        # Different IP allowed
        mock_req2 = MagicMock(spec=Request)
        mock_req2.url.path = "/api/v1/recommendations/hybrid"
        mock_req2.client.host = "192.168.1.101"
        mock_req2.state.request_id = "req-2"
        resp_diff_ip = await limiter.dispatch(mock_req2, dummy_app)
        assert resp_diff_ip.status_code == 200


@pytest.mark.unit
@pytest.mark.asyncio
async def test_structured_http_exception_envelope():
    """Tests http_exception_handler returns standardized error structure."""
    request = MagicMock(spec=Request)
    request.state.request_id = "test-req-abc"
    request.url.path = "/api/v1/recommendations/test"
    
    exc = HTTPException(status_code=404, detail="Requested item not found")
    resp = await http_exception_handler(request, exc)
    
    assert resp.status_code == 404
    import json
    body = json.loads(resp.body.decode())
    assert body["status"] == "error"
    assert body["error_code"] == "RESOURCE_NOT_FOUND"
    assert body["message"] == "Requested item not found"
    assert body["detail"] == "Requested item not found"
    assert body["request_id"] == "test-req-abc"
    assert body["path"] == "/api/v1/recommendations/test"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_structured_validation_error_envelope():
    """Tests validation_exception_handler formats 422 errors properly."""
    request = MagicMock(spec=Request)
    request.state.request_id = "test-val-xyz"
    request.url.path = "/api/v1/recommendations/nearby"
    
    exc = RequestValidationError(errors=[{"loc": ["query", "latitude"], "msg": "Input should be greater than or equal to -90", "type": "greater_than_equal"}])
    resp = await validation_exception_handler(request, exc)
    
    assert resp.status_code == 422
    import json
    body = json.loads(resp.body.decode())
    assert body["status"] == "error"
    assert body["error_code"] == "VALIDATION_ERROR"
    assert "details" in body
    assert body["request_id"] == "test-val-xyz"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_unhandled_exception_sanitization():
    """Tests global_exception_handler returns generic 500 without leaking stack traces."""
    request = MagicMock(spec=Request)
    request.state.request_id = "test-500-trace"
    request.url.path = "/api/v1/recommendations/hybrid"
    request.method = "POST"
    
    exc = RuntimeError("Secret DB Password or internal stack trace: /var/secrets/key.pem")
    resp = await global_exception_handler(request, exc)
    
    assert resp.status_code == 500
    import json
    body = json.loads(resp.body.decode())
    assert body["status"] == "error"
    assert body["error_code"] == "INTERNAL_SERVER_ERROR"
    assert "Secret DB Password" not in body["message"]
    assert body["request_id"] == "test-500-trace"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_lifespan_lifecycle():
    """Tests lifespan pre-warming and cleanup without errors."""
    mock_app = MagicMock()
    async with lifespan(mock_app):
        # Startup phase completed
        assert True
    # Shutdown phase completed
    assert True

