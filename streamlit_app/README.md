# Streamlit UI: Bengaluru Restaurant Intelligence

## Overview
A production-grade, interactive Streamlit frontend for the Bengaluru Restaurant Recommendation System. It serves as a modern presentation layer that connects directly to the FastAPI recommendation backend via REST endpoints.

---

## Key Features

1. **Personalized Hybrid Discovery**: Multi-signal restaurant recommendation combining Content TF-IDF, Collaborative SVD (for known users), BallTree Spatial proximity, and Bayesian Quality shrinkage.
2. **Cold-Start Onboarding Wizard**: Guided multi-step questionnaire for new/unregistered users without historical ratings.
3. **Locality & Cuisine Popularity Rankings**: Bayesian popularity prior rankings across Bengaluru localities (Indiranagar, Koramangala, Whitefield, etc.) and cuisines.
4. **Spatial Radius Search ("Find Restaurants Near Me")**: BallTree accelerated nearest-neighbor search with adjustable search radius (0.5 km to 15.0 km).
5. **MMR Diversification Dashboard**: Visualizes Intra-List Diversity (ILD), Redundancy Rate, Unique Cuisine Ratio, and Relevance Retention with interactive $\lambda$ controls.
6. **Natural Language Explainability**: "Why this recommendation?" cards explaining why each restaurant was selected.
7. **System Status & Telemetry**: Real-time health, model readiness, uptime, memory, and cache hit/miss statistics.

---

## Project Structure

```
streamlit_app/
├── app.py                     # Main application entrypoint
├── config.py                  # Frontend configurations & constants
├── api_client.py              # Resilient FastAPI REST client
├── state.py                   # Session state management
├── components/
│   ├── header.py              # Hero banner
│   ├── sidebar.py             # User identity & MMR controls
│   ├── filters.py             # Preference filters & constraints
│   ├── recommendation_card.py # Restaurant discovery card
│   ├── explanation_card.py    # Explainability breakdown
│   ├── diversity_panel.py     # MMR diversification metrics
│   └── metrics_panel.py       # Slate summary analytics
├── pages/
│   ├── home.py                # 4-tab discovery interface
│   ├── recommendations.py     # Dedicated slate inspection
│   └── system_status.py       # Telemetry & model readiness
└── assets/
    └── style.css              # Custom CSS styling
```

---

## Running the Application

### 1. Start the FastAPI Backend (Terminal 1)
```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

### 2. Start the Streamlit Frontend (Terminal 2)
```bash
streamlit run streamlit_app/app.py
```

By default, the Streamlit app connects to `http://127.0.0.1:8000`. You can configure a custom URL using:
```bash
export STREAMLIT_API_BASE_URL="http://localhost:8000"
streamlit run streamlit_app/app.py
```

