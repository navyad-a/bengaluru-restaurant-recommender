# 5-Minute Live Project Presentation Script

## 🎙️ Timed Script for Hiring Managers & Technical Panels

### [0:00 – 0:30] The Problem
> "Good morning everyone. Today I'm presenting the Bengaluru Restaurant Recommendation Platform. Finding great food in dense metropolitan markets like Bengaluru presents three unique machine learning challenges:
> 1. **Filter Bubbles & Slate Redundancy**, where a single cafe chain captures entire top-10 recommendation slates.
> 2. **Rating Volatility**, where an obscure venue with one 5-star review outranks landmark culinary institutions.
> 3. **Cold-Start User Sparsity**, where new diners have zero historical interaction logs."

### [0:30 – 1:00] The Solution & Stack
> "To solve these, I engineered an end-to-end hybrid recommendation platform indexing 12,481 authentic Bengaluru restaurant venues. The architecture pairs an asynchronous FastAPI gateway with an interactive Streamlit UI, powered by Scikit-Learn, Surprise SVD, BallTree spatial indexing, Redis caching, and 195 automated pytest test suites."

### [1:00 – 2:00] The ML Pipeline
> "For any recommendation request, our engine computes four orthogonal signals:
> - **40% Content Similarity**: Namespaced TF-IDF vectors encoding cuisines, price tiers, and dining styles.
> - **20% Collaborative Filtering**: Biased Regularized SVD matrix factorization discovering latent cross-cuisine taste affinity.
> - **15% Geospatial Intelligence**: Continuous exponential distance decay over spherical Haversine coordinates retrieved via BallTree in under 15ms.
> - **25% Empirical Bayes Quality Shrinkage**: Shrinks small-sample ratings toward the citywide prior of 4.14 stars."

### [2:00 – 3:00] Live Recommendation Demo
> "Let's look at the live Streamlit application running on port 8501.
> - We select **Indiranagar**, choose **North Indian and Mughlai** cuisines, and set a budget of **INR 1,200 for two**.
> - Clicking Get Recommendations delivers top venues like *Empire Restaurant* and *BOX8*.
> - Notice the **Explanation Card**: It provides transparent, grounded natural language rationales reflecting exact content match percentages, community review volumes, and proximity."

### [3:00 – 4:00] MMR Diversification, Caching & Cold Start
> "Next, let's look at **Maximal Marginal Relevance (MMR)**:
> - When MMR is off (lambda = 1.0), duplicate chain branches dominate top slots.
> - When we set lambda to our production default of **0.75**, duplicate chains are eliminated (0.0% redundancy), expanding intra-list diversity by **+49.9%** while retaining 95.6% relevance.
> - Repeating the query demonstrates our multi-backend cache with 4-decimal spatial rounding: latency drops from **240 ms to 4.3 ms (a 55x speedup)**.
> - For new users, our **5-tier Cold-Start Router** navigates from onboarding questionnaires to Bayesian popularity priors."

### [4:00 – 4:40] Backend & API Architecture
> "Under the hood, FastAPI pre-warms all 12,481 items on startup. CPU-heavy linear algebra is offloaded to an 8-worker background threadpool to prevent blocking the async event loop. Every request carries UUIDv4 tracing and timing telemetry."

### [4:40 – 5:00] Summary & Results
> "The entire system is protected by 195 automated pytest tests with a 100% pass rate. Thank you, and I am excited to take your questions!"
