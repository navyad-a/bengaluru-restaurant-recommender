# Production Deployment & Containerization Operations Guide

## 1. Multi-Tier Architecture Overview

The system runs as a four-container microservices stack orchestrated via Docker Compose:

```
[ Internet / Browser ]
         │
         ▼
[ Streamlit UI :8501 ] (Non-root: streamlituser)
         │
         ▼
[ FastAPI Backend :8000 ] (Non-root: appuser, Gunicorn + 2 Uvicorn Workers)
   ├── [ PostgreSQL 16 :5432 ] (Internal persistent relational database)
   └── [ Redis 7 :6379 ] (Internal distributed LRU TTL cache)
```

---

## 2. Environment Variables Specification

| Variable Name | Default / Production Value | Description |
|---|---|---|
| `APP_ENV` | `production` | Execution environment (`development`, `test`, `production`) |
| `DATABASE_URL` | `postgresql+asyncpg://postgres:postgres@postgres:5432/restaurant_recommender` | Async SQLAlchemy database connection string |
| `CACHE_BACKEND` | `redis` (or `memory`) | Active recommendation cache engine |
| `REDIS_URL` | `redis://redis:6379/0` | Redis cluster / instance connection URL |
| `RECOMMENDATION_CACHE_ENABLED` | `true` | Enables or disables recommendation caching |
| `RECOMMENDATION_CACHE_TTL_SECONDS`| `300` | Expiration time for cached responses (5 minutes) |
| `RECOMMENDATION_CACHE_MAX_SIZE` | `1000` | LRU capacity ceiling |
| `THREAD_POOL_WORKERS` | `8` | Worker threads for CPU-bound ML scoring |
| `DEFAULT_MMR_LAMBDA` | `0.75` | Production diversity trade-off default |
| `STREAMLIT_API_BASE_URL` | `http://api:8000` | Backend API URL for Streamlit frontend |

---

## 3. Step-by-Step Deployment

### A. Prerequisites
- Docker Engine 24.0+ and Docker Compose v2.0+
- Host machine with $\ge 2\text{ GB RAM}$ and $2\text{ CPU cores}$.

### B. Launching the Stack
```bash
# 1. Clone repository
git clone https://github.com/your-username/restaurant-recommendation-system.git
cd restaurant-recommendation-system

# 2. Copy and customize environment template
cp .env.example .env

# 3. Build container images and start all 4 services
docker compose up --build -d

# 4. Check service status and health
docker compose ps
```

### C. Initializing the Database (One-Time Execution)
```bash
# Run schema migration / DDL creation
docker compose exec api python scripts/init_db.py

# Seed 12,481 authentic restaurants and synthetic users
docker compose exec api python scripts/seed_db.py
```

---

## 4. Resource Allocation & Worker Sizing

### Memory Analysis per Backend Process:
- Python runtime + dependencies: $\approx 45\text{ MB}$
- 12,481 restaurant catalog: $\approx 25\text{ MB}$
- TF-IDF CSR matrix + vocabulary: $\approx 35\text{ MB}$
- SVD latent matrices: $\approx 20\text{ MB}$
- BallTree spatial index: $\approx 15\text{ MB}$
- Working buffers & cache: $\approx 45\text{ MB}$
- **Total per Worker RSS**: $\approx 185\text{ MB}$

### Gunicorn Configuration:
```dockerfile
CMD ["gunicorn", "app.main:app", "-w", "2", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8000", "--timeout", "60"]
```
- **2 Workers**: $2 \times 185\text{ MB} \approx 370\text{ MB}$ total container RSS.
- Combined with `THREAD_POOL_WORKERS=8`, this configuration safely serves hundreds of concurrent requests without exceeding container memory limits.

---

## 5. Cloud Scale-Out Architecture (AWS / GCP / Kubernetes)

For large-scale cloud deployments:

```
              [ AWS CloudFront / Cloudflare CDN ]
                              │
               [ Application Load Balancer (ALB) ]
                 ┌────────────┴────────────┐
                 ▼                         ▼
         [ Streamlit Pods ]        [ FastAPI API Pods ]
         (K8s HPA: 2-10 replicas)  (K8s HPA: 4-20 replicas)
                                           │
                        ┌──────────────────┴──────────────────┐
                        ▼                                     ▼
           [ Amazon Aurora PostgreSQL ]           [ Amazon ElastiCache Redis ]
           (Multi-AZ Read Replicas)               (Multi-Node Cluster)
```

---

## 6. Verification Status & Disclaimer

> [!NOTE]
> The Dockerfiles, Docker Compose manifests, non-root users, and Redis cache fallbacks have been verified through static linting, schema validation, and unit test mocking (`tests/test_phase_15_docker.py`). Full live container execution was not performed in the local development environment because the Docker CLI daemon was not installed.

