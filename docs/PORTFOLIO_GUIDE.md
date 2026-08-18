# AI Engineer Portfolio & Career Assets Guide

## 1. Resume Project Descriptions

### Version A — 1-Line Compact Resume Bullet
- **Bengaluru Restaurant Recommender**: Built a production hybrid recommender across 12,481 venues using TF-IDF, SVD, and BallTree spatial search on FastAPI with MMR diversification ($\lambda=0.75$), achieving 49.9% higher diversity and sub-5ms Redis cached latency across 195 automated tests.

---

### Version B — 2-Bullet Standard Engineering Resume
- **Bengaluru Restaurant Recommender (FastAPI, Scikit-Learn, Surprise SVD, Redis, Streamlit, Docker)**
  - Architected a 4-signal hybrid recommender (TF-IDF Content 40%, SVD 20%, Haversine Proximity 15%, Bayesian Quality 25%) across 12,481 authentic Bengaluru outlets, implementing a 5-tier cold-start router and Maximal Marginal Relevance (MMR $\lambda=0.75$) that eliminated 100% of chain redundancy.
  - Engineered an asynchronous FastAPI microservice with threadpool ML offloading, Redis multi-backend caching with 4-decimal spatial clustering, and automated test coverage across 195 pytest suites (100% pass rate).

---

### Version C — 4-Bullet Detailed AI Engineer Resume
- **Hybrid Restaurant Recommendation & Discovery Platform (FastAPI, Scikit-Learn, SVD, Redis, PostgreSQL, Docker)**
  - Engineered an end-to-end hybrid recommendation engine over 12,481 authentic Bengaluru restaurant venues, combining namespaced TF-IDF content similarity, Surprise SVD matrix factorization ($\text{RMSE}=0.6171$), empirical Bayes rating shrinkage, and BallTree spatial search.
  - Designed and deployed Maximal Marginal Relevance (MMR with $\lambda=0.75$) and a 5-tier cold-start hierarchy, expanding long-tail catalog coverage by $+55.3\%$ and boosting intra-list diversity by $+49.9\%$ while eliminating duplicate chain recommendations.
  - Developed an asynchronous FastAPI backend offloading CPU-bound vector algebra to an 8-worker thread pool, integrating sliding-window rate limiting, request tracing, and multi-backend Redis caching ($15.5\times$ latency speedup, $3.1\text{ ms}$ cached response).
  - Built an interactive Streamlit frontend with INR pricing and verified the full multi-tier Docker Compose stack through 195 automated pytest test suites.

---

## 2. Professional LinkedIn Project Post

```markdown
🚀 Excited to share my latest AI Engineering project: The Bengaluru Restaurant Recommendation Platform! 🍽️

Finding great dining in dense food hubs like Bengaluru often leads to filter bubbles, rating volatility, and repetitive chain recommendations. I engineered an enterprise-ready, production-grade hybrid recommendation system to solve this.

✨ Core Engineering Highlights:
🔹 Multi-Signal Hybrid Recommender: Ensembles 4 orthogonal signals across 12,481 authentic Bengaluru outlets:
   • 40% Namespaced TF-IDF Content Similarity
   • 20% Collaborative Matrix Factorization (Surprise SVD, RMSE=0.6171)
   • 15% Geospatial Intelligence (BallTree + Haversine Metric)
   • 25% Empirical Bayes Rating Shrinkage (m=10 prior)
🔹 MMR Diversification (λ=0.75): Balanced relevance vs novelty using Maximal Marginal Relevance, increasing Intra-List Distance (ILD) by +49.9% and completely eliminating duplicate chain redundancy (0.0%).
🔹 Deterministic Cold-Start Routing: 5-tier fallback hierarchy guiding diners from interactive onboarding to Bayesian popularity.
🔹 High-Concurrency Backend: FastAPI with an 8-worker threadpool for non-blocking ML inference, UUIDv4 request correlation, and multi-backend Redis caching with coordinate rounding (sub-5ms cached latency).
🔹 Interactive Streamlit UI: Customized for Indian dining (INR pricing, 90+ Bengaluru localities, live diversity tuning).
🔹 Reliability & Containerization: 195 automated pytest tests (100% pass rate) with multi-container Docker Compose orchestration.

GitHub Repository & Full Technical Documentation: [Link]

#MachineLearning #FastAPI #DataScience #AIEngineering #Python #RecommendationSystems #Redis #Docker #Streamlit #Portfolio
```

---

## 3. GitHub Metadata & Portfolio Screenshot Checklist

### GitHub Short Description (Under 160 Characters):
> "Production hybrid AI recommendation platform for Bengaluru restaurants with FastAPI, SVD, BallTree, MMR, Redis, and Streamlit."

### Suggested GitHub Topic Tags:
`recommendation-system`, `machine-learning`, `fastapi`, `streamlit`, `collaborative-filtering`, `matrix-factorization`, `mmr-diversification`, `redis`, `postgresql`, `docker-compose`, `python`

---

### Portfolio Screenshot Plan (10 Showcase Assets):
1. **Streamlit Hero & Filter View**: Demonstrating cuisine chips, INR budget slider, and locality dropdown.
2. **Recommendation Cards**: Showing structured restaurant cards with cuisine tags, cost for two, and rating stars.
3. **Natural Language Explanation Card**: Highlighting factual justification breakdown (Content Match %, Quality, Proximity).
4. **Interactive MMR Diversity Panel**: Showing real-time ILD and redundancy metrics adjusting as $\lambda$ moves.
5. **Cold-Start Onboarding Wizard**: Demonstrating new-user questionnaire and immediate profile matching.
6. **FastAPI OpenAPI Swagger Documentation**: Displaying all 12 public endpoints.
7. **System Status & Telemetry Endpoint**: Displaying live catalog size (12,481), cache stats, and threadpool workers.
8. **Automated Pytest Terminal Output**: Showing `195 passed in 15.52s`.
9. **Docker Compose Architecture**: Highlighting multi-tier container configuration.
10. **Pareto Diversity Trade-Off Curve**: Graphing ILD and relevance retention vs $\lambda$.

---

## 4. The AI Engineer Narrative: From Raw Data to Production

```
Raw Data Inspection (12,481 Bengaluru Venues)
      │
      ▼
Prefix-Isolated Feature Engineering (Namespaced TF-IDF)
      │
      ▼
Collaborative SVD & Empirical Bayes Shrinkage
      │
      ▼
Geospatial BallTree Radians Spatial Search
      │
      ▼
Maximal Marginal Relevance Diversification (λ=0.75)
      │
      ▼
Deterministic Rule-Based Grounded Explainability
      │
      ▼
Asynchronous FastAPI Gateway with Threadpool ML Offloading
      │
      ▼
Distributed Redis Caching with 4-Decimal Coordinate Rounding
      │
      ▼
Interactive Streamlit Frontend with INR Dining Experience
      │
      ▼
195 Automated Pytest Suites & Docker Compose Containerization
```

