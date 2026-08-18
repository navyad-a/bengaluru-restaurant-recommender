# FastAPI REST API Reference & Integration Guide

## 1. Overview & Headers

The Bengaluru Restaurant Recommendation API is an asynchronous HTTP REST service built on FastAPI.

- **Base URL (Local)**: `http://127.0.0.1:8000`
- **Base URL (Docker Compose)**: `http://api:8000`
- **Interactive Swagger UI**: `http://127.0.0.1:8000/docs`
- **OpenAPI Schema (JSON)**: `http://127.0.0.1:8000/openapi.json`

### Standard Response Headers:
- `X-Request-ID`: Distributed correlation UUID (e.g. `c4b18f8e-32dc-4f81-9be3-6e3e5f2cf29e`).
- `X-Process-Time-Ms`: End-to-end backend processing duration in milliseconds (e.g. `3.42`).
- `Content-Type`: `application/json`

---

## 2. Standard Error Response Contracts

All errors return sanitized, structured JSON envelopes:

```json
{
  "error": "Validation Error",
  "detail": "Input validation failed",
  "status_code": 422,
  "request_id": "c4b18f8e-32dc-4f81-9be3-6e3e5f2cf29e",
  "errors": [
    {
      "field": "body.top_k",
      "message": "Input should be less than or equal to 50"
    }
  ]
}
```

---

## 3. System & Health Endpoints

### A. Liveness Probe (`GET /health`)
- **Status Codes**: `200 OK`
```json
{
  "status": "healthy",
  "timestamp": "2026-08-18T21:10:00.000000"
}
```

### B. Model Readiness Probe (`GET /ready`)
- **Status Codes**: `200 OK`, `503 Service Unavailable`
```json
{
  "status": "ready",
  "models_loaded": {
    "content_recommender": true,
    "collaborative_recommender": true,
    "spatial_search": true
  }
}
```

### C. System Telemetry (`GET /api/v1/system/status`)
- **Status Codes**: `200 OK`
```json
{
  "status": "operational",
  "environment": "production",
  "uptime_seconds": 3600.5,
  "models": {
    "content_recommender": { "catalog_size": 12481, "vector_dim": 2450 },
    "collaborative_recommender": { "benchmark_users": 600, "n_factors": 100 },
    "spatial_search": { "indexed_restaurants": 12481, "metric": "haversine" }
  },
  "cache": {
    "backend": "redis",
    "enabled": true,
    "total_keys": 42,
    "hit_ratio": 0.845
  },
  "concurrency": {
    "thread_pool_workers": 8
  }
}
```

### D. Clear Cache (`POST /api/v1/system/cache/clear`)
- **Status Codes**: `200 OK`
```json
{
  "status": "success",
  "message": "Recommendation cache cleared successfully."
}
```

---

## 4. Recommendation Endpoints

### 1. Similar Restaurants (`GET /api/v1/recommendations/similar/{restaurant_id}`)
- **Path Parameters**: `restaurant_id` (int, ge=1)
- **Query Parameters**: `top_k` (int, default=10, 1-50)
- **Example Request**: `GET /api/v1/recommendations/similar/1?top_k=3`
- **Response**:
```json
{
  "seed_restaurant_id": 1,
  "top_k": 3,
  "total_returned": 3,
  "recommendations": [
    {
      "restaurant_id": 482,
      "name": "Brahmins' Coffee Bar",
      "locality": "Basavanagudi",
      "cuisines": ["South Indian", "Quick Bites"],
      "cost_for_two": 150,
      "rating": 4.5,
      "review_count": 8200,
      "similarity_score": 0.9412,
      "explanation": "High content similarity based on South Indian cuisine and Quick Bites format."
    }
  ]
}
```

---

### 2. Spatial Nearby Search (`GET /api/v1/recommendations/nearby`)
- **Query Parameters**:
  - `latitude` (float, required, $-90.0 \dots 90.0$)
  - `longitude` (float, required, $-180.0 \dots 180.0$)
  - `radius_km` (float, default=5.0, $0.1 \dots 50.0$)
  - `top_k` (int, default=10, $1 \dots 50$)
- **Example Request**: `GET /api/v1/recommendations/nearby?latitude=12.9716&longitude=77.5946&radius_km=3.0&top_k=5`

---

### 3. Collaborative SVD Predictions (`GET /api/v1/recommendations/collaborative/{user_id}`)
- **Path Parameters**: `user_id` (int, ge=1)
- **Query Parameters**: `top_k` (int, default=10, $1 \dots 50$)

---

### 4. Hybrid Recommendations for User (`GET /api/v1/recommendations/hybrid/{user_id}`)
- **Path Parameters**: `user_id` (int, ge=1)
- **Query Parameters**:
  - `latitude` (float, optional)
  - `longitude` (float, optional)
  - `max_distance_km` (float, optional)
  - `use_mmr` (bool, default=true)
  - `mmr_lambda` (float, default=0.75, $0.0 \dots 1.0$)
  - `top_k` (int, default=10, $1 \dots 50$)

---

### 5. Custom Hybrid Search (`POST /api/v1/recommendations/hybrid`)
- **Request Body**:
```json
{
  "user_id": 2,
  "preferred_cuisines": ["North Indian", "Mughlai"],
  "max_budget_inr": 1200,
  "latitude": 12.9784,
  "longitude": 77.6408,
  "radius_km": 5.0,
  "use_mmr": true,
  "mmr_lambda": 0.75,
  "top_k": 5
}
```
- **Response**:
```json
{
  "user_id": 2,
  "routing_strategy": "HybridEnsemble",
  "top_k": 5,
  "mmr_applied": true,
  "mmr_lambda": 0.75,
  "recommendations": [
    {
      "restaurant_id": 218,
      "name": "Empire Restaurant",
      "locality": "Indiranagar",
      "cuisines": ["North Indian", "Mughlai", "Biryani"],
      "cost_for_two": 800,
      "rating": 4.3,
      "review_count": 5210,
      "hybrid_score": 0.8912,
      "content_score": 0.9200,
      "cf_score": 0.8400,
      "location_score": 0.9100,
      "quality_score": 0.8950,
      "distance_km": 1.2,
      "explanation": "Strong North Indian & Mughlai match (92%), popular Indiranagar landmark (4.3★), and well within your ₹1200 budget."
    }
  ]
}
```

---

### 6. Cold-Start Onboarding Wizard (`POST /api/v1/recommendations/onboarding`)
- **Request Body**:
```json
{
  "selected_cuisines": ["Cafe", "Continental"],
  "budget_tier": "moderate",
  "max_cost_for_two": 800,
  "locality": "Koramangala",
  "dietary_preference": "any",
  "top_k": 5
}
```

