# Technical Interview Cheatsheet & Rapid Architecture Reference

## 1. Core Algorithmic Decisions

| Question | Technical Interview Answer |
|---|---|
| **Why Hybrid Recommendation?** | Solves the sparsity and filter-bubble limitations of single-paradigm recommenders. Content provides cold-start taste matching, SVD captures latent collaborative affinity, BallTree enforces physical proximity, and empirical Bayes penalizes small-sample rating noise. |
| **Why TF-IDF with Namespaces?** | Prefix-isolated tokens (`cuisine:north_indian locality:indiranagar`) prevent lexical bleeding between restaurant brand names and dish categories while producing thread-safe sparse CSR vectors. |
| **Why Biased Regularized SVD?** | Decomposes sparse interaction matrices into user/item latent factors ($k=100$) while explicitly modeling baseline user rating optimism ($b_u$) and venue quality bias ($b_i$). |
| **Why Empirical Bayes Shrinkage?** | Smooths volatile review counts on unreviewed venues toward the citywide prior ($C=4.14, m=10$), preventing an outlet with a single 5.0-star rating from outranking proven landmark restaurants. |
| **Why BallTree over KD-Tree?** | Standard KD-Trees assume flat Euclidean geometry; BallTree natively supports spherical Earth geometry using the Haversine metric on radian coordinates for sub-15ms spatial radius filtering. |
| **Why MMR Diversification?** | Greedy ranking produces redundant slates dominated by multiple branches of the same chain. Maximal Marginal Relevance iteratively penalizes candidate similarity to already-selected items. |
| **Why Production Default λ=0.75?** | Pareto trade-off studies showed $\lambda=0.75$ completely eliminates duplicate chain redundancy ($0.0\%$) and boosts intra-list diversity by $+49.9\%$ while retaining $94.75\%$ of ranking relevance. |
| **How is Cold-Start Handled?** | Via a 5-tier routing hierarchy: Warm users get full hybrid; sparse users get content+quality; onboarding users get questionnaire matching; location-only users get spatial+popularity; unknown users get empirical Bayes popularity. |

---

## 2. Backend & Systems Architecture

| Question | Technical Interview Answer |
|---|---|
| **Why Asynchronous FastAPI?** | High-throughput asynchronous I/O allows thousands of concurrent connections with low memory footprint (~185MB RSS per worker). |
| **Why ThreadPool Offloading?** | CPU-bound NumPy/SciPy linear algebra and SVD dot products will block Python's single-threaded event loop. We offload vector math to an 8-worker thread pool via `anyio.to_thread.run_sync()`. |
| **How does Caching Work?** | Multi-backend cache abstraction supporting thread-safe in-memory LRU TTL and distributed Redis with 4-decimal coordinate rounding (~11m clustering), delivering $40	ext{–}60	imes$ speedups ($300	ext{ ms} ightarrow 4.3	ext{ ms}$). |
| **How were Tests Structured?** | 195 automated pytest tests across unit, integration, spatial bounding, ML data integrity, security injection, concurrency, and performance regressions with 100% pass rate. |
