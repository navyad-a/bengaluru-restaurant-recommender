# Final Live Project Demonstration Checklist

## 1. Before Demo Checklist (Preparation Phase)

- [x] **Environment Verification**: Confirm Python 3.10+ is active.
- [x] **Configuration**: Verify `.env` has `CACHE_BACKEND=memory` and `STREAMLIT_API_BASE_URL=http://127.0.0.1:8000`.
- [x] **Model Artifacts**: Confirm `saved_models/` has all required artifacts and `data/processed/restaurants_clean.csv` contains 12,481 records.
- [x] **Backend Server Startup**:
  ```bash
  uvicorn app.main:app --host 127.0.0.1 --port 8000
  ```
- [x] **Health Check**: Open `http://127.0.0.1:8000/health` and verify `status: healthy`.
- [x] **Readiness Check**: Open `http://127.0.0.1:8000/ready` and verify all checks report `true`.
- [x] **Frontend Server Startup**:
  ```bash
  streamlit run streamlit_app/app.py --server.port 8501
  ```
- [x] **Frontend Check**: Open `http://localhost:8501` and verify the home landing page loads.

---

## 2. During Demo Checklist (Live Demonstration Script)

### Step 1: High-Level Architecture (30 Seconds)
- Show the Mermaid System Architecture Diagram in `docs/PROJECT_ARCHITECTURE.md` or README.
- Explain the 4-signal hybrid ensemble: **40% Content-Based TF-IDF**, **20% Surprise SVD**, **15% Geospatial BallTree**, **25% Bayesian Quality Shrinkage**.

### Step 2: FastAPI OpenAPI Documentation (30 Seconds)
- Navigate to `http://127.0.0.1:8000/docs`.
- Highlight standard error envelopes, telemetry headers (`X-Request-ID`, `X-Process-Time-Ms`), and the 12 public endpoints.

### Step 3: Real-Time System Telemetry (30 Seconds)
- Execute `GET /api/v1/system/status`.
- Point out resident catalog count (**12,481 venues**), collaborative training benchmark (**600 users**), and **8 worker threads**.

### Step 4: Streamlit Personalized Recommendation Flow (1 Minute)
- In `http://localhost:8501`, set:
  - Locality: **Indiranagar**
  - Cuisines: **North Indian, Mughlai**
  - Budget for Two: **INR 1,200**
  - Minimum Rating: **4.0 stars**
- Click **Get Recommendations**.
- Show the returned recommendation cards (*BOX8 - Desi Meals*, *Empire Restaurant*, *The Kebab Room*).

### Step 5: Grounded Explainability Cards (30 Seconds)
- Expand an **Explanation Card** on the top recommendation.
- Point out the factual justification: exact percentage content match, budget match, proximity, and community review volume.

### Step 6: MMR Diversification Live Toggle (1 Minute)
- Move the MMR diversity slider from **lambda = 1.00** (pure relevance) to **lambda = 0.75** (production default).
- Demonstrate how duplicate chain outlets are replaced with unique independent brands while retaining 95.6% relevance.

### Step 7: Sub-5ms Caching Acceleration (30 Seconds)
- Click **Get Recommendations** a second time.
- Show the Metrics Panel: Cold request took ~240 ms; warm request returned in **4.3 ms (55x+ speedup)**.

### Step 8: Cold-Start Onboarding Wizard (30 Seconds)
- Navigate to the **Onboarding Questionnaire** in the sidebar.
- Enter favorite cuisines without selecting an existing user to demonstrate deterministic cold-start profile generation.

---

## 3. After Demo Checklist (Wrap-Up)

- [ ] **Capture Showcase Screenshots**: Capture recommendation cards, explainability dropdowns, and diversity panel for portfolio assets.
- [ ] **Verify Test Suite**: Run `pytest -v` to demonstrate 195 passing tests.
- [ ] **Stop Background Daemons**: Gracefully terminate uvicorn and streamlit processes when finished.
