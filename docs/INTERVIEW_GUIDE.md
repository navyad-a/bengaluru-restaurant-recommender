# Technical Interview Preparation Handbook & Project Defense Guide

## 1. Project Elevator Pitches

### A. 30-Second Elevator Pitch
"I built a production-oriented hybrid recommendation engine for Bengaluru restaurants. It ensembles namespaced TF-IDF content similarity, Surprise SVD collaborative filtering, empirical Bayes quality priors, and geospatial BallTree search into an async FastAPI backend. To eliminate chain redundancy, I implemented Maximal Marginal Relevance diversification, achieving a 49.9% intra-list diversity improvement with zero duplicate chains. The system serves sub-5ms cached recommendations via Redis and is thoroughly validated across 195 automated pytest tests."

### B. 1-Minute Elevator Pitch
"My project solves personalized restaurant discovery in dense metropolitan markets like Bengaluru. Standard recommenders suffer from filter bubbles, rating volatility, and cold-start failures. I engineered a 4-signal hybrid recommender combining TF-IDF content matching, collaborative SVD, empirical Bayes rating shrinkage, and spatial BallTree search. For new users, a 5-tier routing hierarchy falls back gracefully from interactive onboarding to Bayesian popularity. I implemented Maximal Marginal Relevance (MMR with $\lambda=0.75$), which eliminated 100% of chain redundancy while preserving 94.75% ranking relevance. The backend uses async FastAPI with an 8-worker thread pool to prevent CPU-bound ML math from blocking the event loop. The system is containerized via Docker Compose with Redis caching, PostgreSQL, and verified with 195 automated tests."

### C. 2-Minute Technical Pitch
"The architecture is designed as a high-concurrency microservices stack tailored for the Indian dining market.
1. **Machine Learning Layer**: We process 12,481 authentic Bengaluru restaurant venues. We use prefix-isolated TF-IDF namespaces to prevent token collision, SVD matrix factorization for latent affinity, empirical Bayes shrinkage to penalize sparse review counts, and a spatial BallTree for sub-15ms radius filtering.
2. **Diversification & Cold Start**: To prevent top slates from being dominated by multiple branches of the same chain, we apply MMR with $\lambda=0.75$, improving intra-list distance by 49.9%. A 5-tier router guarantees deterministic fallbacks for cold users.
3. **Backend Engineering**: Built on FastAPI, the service offloads linear algebra to an asynchronous worker thread pool to maintain event loop liveness. It includes request tracing, timing headers, sliding-window rate limiting, and a multi-backend cache supporting local in-memory LRU and distributed Redis with coordinate rounding.
4. **Quality & Reliability**: The entire codebase is verified with 195 automated pytest tests covering unit, integration, ML data integrity, security bounds, and performance benchmarks."

---

## 2. Top 30 Difficult "Defend Your Project" Questions

### 1. Why use a hybrid recommendation system instead of deep learning (e.g. Two-Tower or Transformer)?
- **Short Answer**: Interpretability, low compute overhead, cold-start resilience, and extreme low latency.
- **Strong Interview Answer**: "For tabular restaurant discovery across 12,481 items, deep models introduce substantial inference latency ($50\text{–}200\text{ ms}$), require millions of dense interaction logs to prevent overfitting, and operate as black boxes. A hybrid ensemble combining linear algebra (SVD), text similarity (TF-IDF), spatial indexing, and empirical Bayes delivers sub-5ms cached latency, full cold-start explainability, and deterministic parameter control without GPU infrastructure costs."

### 2. Why is the collaborative filtering benchmark synthetic?
- **Short Answer**: Authentic Bangalore catalogs lack publicly available explicit user rating matrix logs.
- **Strong Interview Answer**: "While our restaurant catalog of 12,481 physical venues is authentic, real user interaction matrices are proprietary to platforms like Zomato or Swiggy. To demonstrate collaborative filtering capabilities without fabricating user reviews, we created a strictly labeled synthetic benchmark (600 users, 11,920 ratings) following power-law distributions. We maintain strict provenance isolation so synthetic data is never represented as authentic customer behavior."

### 3. How did you arrive at the production MMR default $\lambda=0.75$?
- **Strong Interview Answer**: "In Phase 11, we ran an offline Pareto trade-off study across $\lambda \in [0.1, 1.0]$. At $\lambda=1.0$ (no MMR), redundancy was $8.60\%$ and ILD was $0.3822$. At $\lambda=0.75$, redundancy dropped to $0.00\%$ and ILD increased by $+49.9\%$ to $0.5730$, while retaining $94.75\%$ of ranking relevance. Lower values ($\lambda \le 0.50$) degraded top-k relevance below $80\%$, making $\lambda=0.75$ the optimal sweet spot."

### 4. Why does your FastAPI backend use an async thread pool if Python ML is CPU-bound?
- **Strong Interview Answer**: "FastAPI's event loop is single-threaded. If an async endpoint performs dense NumPy or SciPy matrix multiplication directly on the main thread, the event loop blocks, starving concurrent HTTP requests. By wrapping synchronous ML inference in `anyio.to_thread.run_sync()`, we offload heavy vector math to an 8-worker thread pool, keeping the async event loop responsive to incoming traffic."

### 5. Why did aggregate code coverage report 79% in Phase 15 despite $\ge 90\%$ on critical modules?
- **Strong Interview Answer**: "Our core ML engines, configuration, middleware, schemas, and cache backends maintain $\ge 90\text{–}100\%$ unit test coverage. The aggregate metric of 79% reflects untracked interactive Streamlit multi-page UI scripts and environment-specific Docker fallback branches that only trigger when external infrastructure is disconnected. We maintain strict quality gates on all algorithmic and backend modules."

---

## 3. Structured Category Q&A Matrix

### ML & Algorithms
1. **What is empirical Bayes shrinkage?** Smooths small-sample ratings toward the global population mean ($C=4.14$) with prior weight $m=10$.
2. **Why use BallTree over KD-Tree?** BallTree natively supports spherical Earth geometry using the Haversine metric on radian coordinates, whereas standard KD-Trees assume Euclidean flat space.
3. **What is Intra-List Distance (ILD)?** The average pairwise cosine distance among recommended items in a slate, measuring topic diversity.

### System Design & Backend
1. **How do you handle rate limiting?** An in-memory sliding window middleware tracking timestamps per client IP, returning 429 when exceeding 120 req/min.
2. **Why round coordinates to 4 decimal places in cache?** Coordinates are rounded to 4 decimals (~11m) so users searching within the same block hit the same warm cache entry.
3. **What happens if Redis fails in production?** `RedisRecommendationCache` catches the socket timeout, logs a warning, and operates in non-blocking pass-through mode without crashing recommendations.

