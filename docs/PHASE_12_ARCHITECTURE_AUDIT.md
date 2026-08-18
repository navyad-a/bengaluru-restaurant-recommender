# Phase 12: FastAPI Async Architecture & Performance Audit

**Date**: 2026-08-18  
**Scope**: Comprehensive performance, concurrency, lifecycle, and async execution audit of the FastAPI backend in `restaurant-recommendation-system`.

---

## 1. Executive Summary

This architecture audit evaluates the performance characteristics, concurrency safety, resource lifecycle, and asynchronous execution patterns of the Bengaluru Restaurant Recommendation API prior to Phase 12 optimizations.

The existing system consists of:
- **12,481 authentic physical Bengaluru restaurant outlets**
- **TF-IDF Content-Based Recommender**
- **Surprise SVD Collaborative Filtering Engine** (trained on the Synthetic Collaborative Filtering Benchmark)
- **Spatial Search Engine** (BallTree + Haversine distance)
- **Bayesian Quality Scorer**
- **5-Tier Cold-Start Router**
- **Maximal Marginal Relevance (MMR) Diversification Engine**
- **Explainability Subsystem**

---

## 2. Workload Classification: CPU-Bound vs. I/O-Bound

| Operation | Component | Classification | Latency Profile | Current Execution | Risk / Bottleneck |
|---|---|:---:|:---:|:---:|---|
| **TF-IDF Cosine Similarity** | `ContentRecommender` | CPU-Bound | $5-15\text{ ms}$ | Direct in `async def` | Blocks asyncio event loop during user/item vector projection |
| **SVD Rating Estimation** | `CollaborativeRecommender` | CPU-Bound | $50-110\text{ ms}$ | Direct in `async def` | Blocks event loop during full-catalog candidate scoring |
| **BallTree Spatial Query** | `SpatialSearchEngine` | CPU-Bound | $3-10\text{ ms}$ | Direct in `async def` | Minor event loop blockage during k-NN tree traversal |
| **Bayesian Quality Shrinkage** | `BayesianQualityScorer` | CPU-Bound | $<1\text{ ms}$ | Direct in `async def` | Low CPU, safe vectorized dictionary lookup |
| **MMR Greedy Diversification** | `MMREngine` | CPU-Bound | $200-500\text{ ms}$ | Direct in `async def` | **Critical event loop bottleneck** due to iterative similarity evaluations |
| **Database Queries** | `app.database.session` | I/O-Bound | $5-25\text{ ms}$ | `asyncpg` async session | Non-blocking, properly managed via `AsyncSession` dependency |
| **Pydantic Serialization** | `app.schemas.recommendation` | CPU-Bound | $2-8\text{ ms}$ | Pydantic v2 `BaseModel` | Fast, but scales with response item count |

---

## 3. Resource Lifecycle & Global Object Analysis

### A. Globally Reusable Singleton Objects
The following objects are immutable/read-only during inference and safe to reuse globally across requests:
1. `ContentRecommender`: Holds pre-fitted TF-IDF vectorizer, sparse CSR feature matrix ($12,481 \times D$), and metadata mappings.
2. `CollaborativeRecommender`: Holds Surprise SVD model factors ($p_u, q_i$) and user-rated set cache.
3. `SpatialSearchEngine`: Holds pre-computed `sklearn.neighbors.BallTree` and radian coordinates array.
4. `BayesianQualityScorer`: Holds empirical prior constants ($\mu = 3.70, m = 10$) and pre-computed Bayesian scores.
5. `PopularityEngine`: Holds pre-computed global and locality-filtered restaurant rankings.
6. `SparseSimilarityEngine`: Holds pre-normalized TF-IDF CSR matrix for MMR pairwise similarity.
7. `RedundancyChecker`: Stateless string normalization and chain frequency tracking.

### B. Objects Expensive to Recreate Per Request
- Recreating `ContentRecommender` or reloading `tfidf_matrix.joblib` takes $\sim 250\text{ ms}$.
- Re-fitting `BallTree` on 12,481 coordinates takes $\sim 45\text{ ms}$.
- Reloading `svd_model.joblib` takes $\sim 180\text{ ms}$.
- **Requirement**: All model objects and indexes must be pre-warmed at application startup and maintained in memory.

---

## 4. Concurrency & Event Loop Audit

### Current Issue in `app/api/v1/endpoints/recommendations.py`:
- All endpoint handler functions are declared as `async def` (e.g., `async def get_hybrid_recommendations_for_user(...)`).
- In FastAPI / Starlette, `async def` handlers run directly on the single-threaded asyncio event loop.
- When an endpoint executes synchronous CPU-heavy code (such as `hybrid_engine.recommend(...)`), the single event loop thread is completely occupied.
- Under concurrent user traffic (concurrency $\ge 5$), incoming requests queue up, causing P95 latency to degrade significantly.

### Solution for Phase 12:
- Offload CPU-bound ML scoring calls to a managed worker thread pool using `starlette.concurrency.run_in_threadpool` or `asyncio.to_thread`.
- This ensures the asyncio event loop remains immediately responsive to incoming connections, I/O tasks, and lightweight queries.

---

## 5. Caching Opportunities

Deterministic recommendation requests can be cached to eliminate redundant ML computations:
1. **Popular Restaurants** (`/popular`): High read frequency, strictly deterministic Bayesian ranking.
2. **Similar Restaurants** (`/similar/{restaurant_id}`): High read frequency, static TF-IDF cosine similarity.
3. **Deterministic Hybrid Requests** (`/hybrid`): Identical user/preference queries with same filters and coordinates.
4. **Nearby Searches** (`/nearby`): Geographic searches with rounded coordinate buckets.

---

## 6. Audit Conclusions & Phase 12 Action Plan

1. **Lifespan Management**: Implement `app/core/lifespan.py` to pre-warm all singletons during startup.
2. **Threadpool Offloading**: Wrap CPU-bound ML execution with `run_in_threadpool`.
3. **Response Caching**: Implement in-memory TTL LRU cache in `app/core/cache.py`.
4. **Middleware**: Add Request ID tracking (`app/middleware/request_id.py`) and latency timing (`app/middleware/timing.py`).
5. **Health & Readiness**: Add `/health`, `/ready`, and `/api/v1/system/status`.
6. **Error Handling**: Standardize structured JSON error responses across all failure modes.
7. **Rate Limiting**: Add lightweight in-memory rate limiting middleware.
8. **Load Benchmarking**: Implement `scripts/benchmark_api_performance.py` to evaluate throughput and latency across concurrency levels.
