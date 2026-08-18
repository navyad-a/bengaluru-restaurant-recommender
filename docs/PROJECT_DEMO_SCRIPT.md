# 5-Minute Live Project Demonstration Script

## Overview
This script is designed for live technical presentations, hiring manager interviews, portfolio video recordings, or college project defense presentations.

---

## ⏱️ Minute-by-Minute Demo Breakdown

### 0:00 – 0:30 | Problem Statement & Context
**Spoken Audio:**
> "Hi everyone. Today I'm demonstrating the Bengaluru Restaurant Recommendation System. In dense metropolitan markets like Bengaluru, food discovery platforms struggle with three major challenges:
> 1. **Filter Bubbles & Chain Redundancy**, where single cafe chains dominate entire top-10 slates.
> 2. **Rating Volatility**, where unreviewed venues with a single 5-star rating outrank legendary landmark restaurants.
> 3. **Cold-Start Failures** for new diners who haven't logged interaction histories yet.
> Our system solves these with a production-grade 4-signal hybrid ensemble and MMR diversification."

---

### 0:30 – 1:00 | Architecture & The 4 Signals
**Action on Screen:** *Display the Mermaid System Architecture Diagram.*
**Spoken Audio:**
> "Our architecture is built on FastAPI and Streamlit. For any given diner, we compute four independent signals:
> - **40% Content-Based TF-IDF** over namespaced cuisine and metadata features.
> - **20% Collaborative Filtering** using Surprise SVD matrix factorization.
> - **15% Spatial Proximity** via a geospatial BallTree index using Haversine distance.
> - **25% Bayesian Quality Shrinkage** that smooths volatile review counts toward the citywide average of 4.14 stars.
> All linear algebra is offloaded to a background worker thread pool to prevent blocking FastAPI's async event loop."

---

### 1:00 – 2:00 | Streamlit Interactive Dashboard
**Action on Screen:** *Open `http://localhost:8501`. Show the Streamlit UI.*
**Spoken Audio:**
> "Here is our Streamlit dashboard, tailored specifically for the Indian dining market.
> - Notice all pricing is natively in **Indian Rupees (₹ INR)** with dynamic budget sliders.
> - We support 90+ Bengaluru localities with centroid coordinate lookups—let's select **Indiranagar** and filter for **North Indian & Mughlai** cuisines with a budget of **₹1,200 for two**.
> - Let's click **Get Recommendations**."

---

### 2:00 – 3:00 | Grounded Explainability Breakdown
**Action on Screen:** *Scroll through recommended restaurant cards and expand an Explanation Card.*
**Spoken Audio:**
> "In under 30 milliseconds, the system produces a tailored slate. Look at the top recommendation—**Empire Restaurant in Indiranagar**.
> - Notice the **Explanation Card**: It explains in clear natural language that it has a **92% content match**, fits comfortably within the **₹1,200 budget**, is just **1.2 km away**, and is a trusted landmark with **4.3 stars across 5,200+ reviews**.
> - Every single recommendation is grounded in active scoring features—no hallucinated or generic explanations."

---

### 3:00 – 4:00 | Live MMR Diversification Demonstration
**Action on Screen:** *Navigate to the Sidebar Diversity Slider. Change $\lambda$ from 1.0 to 0.75.*
**Spoken Audio:**
> "Now let's demonstrate **Maximal Marginal Relevance (MMR)**.
> - If we set $\lambda = 1.0$, the system optimizes purely for relevance. Notice how several branches of the same chain appear in the list.
> - Now, let's slide $\lambda$ to our production default of **0.75**.
> - Instantly, the **Diversity Panel** updates: Our Intra-List Distance increases by **+49.9%**, duplicate chain redundancy drops to **0.0%**, and unique local venues are surfaced without sacrificing relevance."

---

### 4:00 – 4:30 | FastAPI REST API & Telemetry
**Action on Screen:** *Switch browser tab to Swagger UI (`http://localhost:8000/docs`). Execute `POST /api/v1/recommendations/hybrid`.*
**Spoken Audio:**
> "Behind this UI is our high-performance FastAPI service.
> - Every API response includes an `X-Request-ID` for distributed correlation and an `X-Process-Time-Ms` telemetry header.
> - Our multi-backend cache with 4-decimal coordinate rounding delivers **$15.5\times$ speedups** on repeated queries, serving cached slates in just **3.1 ms**."

---

### 4:30 – 5:00 | Testing, Containerization & Summary
**Action on Screen:** *Show terminal with `pytest` execution showing 195 passed tests.*
**Spoken Audio:**
> "Finally, the entire repository is containerized with Docker Compose across PostgreSQL, Redis, FastAPI, and Streamlit, and protected by **195 automated pytest tests with a 100% pass rate**.
> Thank you! I'm happy to dive into any architectural or algorithmic details."

