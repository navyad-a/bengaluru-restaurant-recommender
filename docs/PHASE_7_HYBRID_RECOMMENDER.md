# Phase 7: Hybrid Recommendation Engine

## 1. Objective & Architectural Overview

Phase 7 implements a unified **Hybrid Recommendation Engine** that integrates four independent, heterogeneous scoring signals into a convex weighted ensemble:

1. **Content-Based Similarity ($S_{\text{content}}$)**: TF-IDF feature cosine similarity over cuisine, dining format, price tier, and locality tokens.
2. **Collaborative Filtering ($S_{\text{collab}}$)**: Personalized rating predictions from Surprise SVD matrix factorization trained on the synthetic benchmark.
3. **Location Proximity ($S_{\text{location}}$)**: Great-circle Haversine distance with exponential spatial decay towards locality centroids.
4. **Bayesian Quality Shrinkage ($S_{\text{quality}}$)**: IMDB-style Bayesian weighted rating regularizing raw review scores by vote reliability.

```mermaid
flowchart TD
    subgraph Input Request
        A[User ID / Preferences / Location / Filters]
    end

    subgraph Candidate Generation
        B1[Content-Based Retrieval]
        B2[Collaborative SVD Retrieval]
        B3[Locality Proximity Retrieval]
        B1 & B2 & B3 --> C[Candidate Pool Union]
        C --> D[Exclude Already Rated Items]
        D --> E[Enforce Hard Business Constraints]
    end

    subgraph Multi-Signal Scoring [Candidate Scoring]
        E --> F1[Content Cosine Similarity: S_content in 0,1]
        E --> F2[SVD Predicted Rating: S_collab in 0,1]
        E --> F3[Haversine Spatial Decay: S_location in 0,1]
        E --> F4[Bayesian Quality Shrinkage: S_quality in 0,1]
    end

    subgraph Dynamic Fusion
        G[Active Signals Detection] --> H[Dynamic Weight Normalization: Sum w_i = 1.0]
        F1 & F2 & F3 & F4 & H --> I[S_hybrid = Sum w_i * S_i]
    end

    subgraph Deterministic Ranking
        I --> J[Sort: S_hybrid DESC, Content DESC, Quality DESC, SVD DESC, Votes DESC, Rating DESC, ID ASC]
        J --> K[Top-K Recommendations with Explanations]
    end

    A --> Candidate Generation
```

---

## 2. Mathematical Formulation & Score Normalization

### A. Convex Combination Hybrid Score
$$S_{\text{hybrid}} = w_{\text{content}} S_{\text{content}} + w_{\text{collab}} S_{\text{collab}} + w_{\text{location}} S_{\text{location}} + w_{\text{quality}} S_{\text{quality}}$$

Subject to:
$$\sum_{s \in \text{Available}} w_s^{\text{eff}} = 1.0, \quad 0 \le S_s \le 1.0 \quad \forall s$$

### B. Base & Production Default Weights
| Signal | Base Weight ($w_s$) | Rationale |
|---|:---:|---|
| **Content-Based** ($S_{\text{content}}$) | **`0.40`** | Primary match against regional Indian cuisines and dining format. |
| **Collaborative SVD** ($S_{\text{collab}}$) | **`0.20`** | Personalized latent taste preferences (synthetic benchmark). |
| **Location Proximity** ($S_{\text{location}}$) | **`0.15`** | Geographic proximity penalty to minimize diner travel distance. |
| **Bayesian Quality** ($S_{\text{quality}}$) | **`0.25`** | Proven restaurant excellence and customer satisfaction. |

---

## 3. Component Modules

### A. Bayesian Quality Shrinkage (`ml/hybrid/quality.py`)
Prevents unproven outliers (e.g. 5.0★ with 2 reviews) from outranking proven institutions (e.g. 4.5★ with 10,000 reviews):

$$\text{WR} = \left(\frac{v}{v + m}\right) R + \left(\frac{m}{v + m}\right) C$$

Where:
- $R$: Raw restaurant rating $\in [1.0, 5.0]$.
- $v$: Review count (votes).
- $C$: Global catalog mean rating ($C = 3.626$).
- $m$: Minimum vote threshold ($m = 50.0$, 50th percentile of rated catalog).
- Normalized Quality Score: $S_{\text{quality}} = \text{clip}\left(\frac{\text{WR} - 1.0}{4.0}, 0.0, 1.0\right)$.

### B. Location Proximity (`ml/hybrid/location.py`)
Computes great-circle distance $d$ in kilometers via the Haversine formula and maps to an exponential decay proximity score:

$$S_{\text{location}} = \exp\left(-\frac{d}{\tau}\right)$$

Where $\tau = 3.0\text{ km}$ is the spatial decay constant.
- $0.0\text{ km} \rightarrow 1.000$
- $3.0\text{ km} \rightarrow 0.368$
- $10.0\text{ km} \rightarrow 0.036$

### C. Dynamic Weight Redistribution (`ml/hybrid/scoring.py`)
When a signal is absent or cold-start, its weight is set to 0.0 and remaining active weights are scaled proportionally to guarantee exact unit sum:

- **Known User + Location + Preferences**: `{content: 0.40, collab: 0.20, location: 0.15, quality: 0.25}` (Sum = 1.0)
- **Known User without Location**: `{content: 0.4706, collab: 0.2353, location: 0.0, quality: 0.2941}` (Sum = 1.0)
- **Cold-Start User with Location**: `{content: 0.5000, collab: 0.0, location: 0.1875, quality: 0.3125}` (Sum = 1.0)
- **Cold-Start User without Location**: `{content: 0.6154, collab: 0.0, location: 0.0, quality: 0.3846}` (Sum = 1.0)

---

## 4. Hard Constraints vs Soft Preferences

- **Hard Constraints** (`CandidateGenerator.apply_hard_filters`):
  1. `max_cost_for_two`: Strictly requires $\text{cost} \le \text{budget}$.
  2. `min_rating`: Strictly requires $\text{rating} \ge \text{threshold}$.
  3. `area`: Exact neighborhood match (e.g. *Jayanagar*, *Koramangala 5th Block*).
  4. `online_order_only`: Requires `online_order == True`.
  5. `book_table_only`: Requires `book_table == True`.
  *Violators are pruned before ranking.*
- **Soft Preferences**: Regional cuisine tokens, dining formats, and price tiers softly bias the TF-IDF content similarity vector.

---

## 5. Deterministic Ranking Hierarchy

To prevent unstable ordering and nondeterministic ties across API requests:
1. `hybrid_score` DESC
2. `content_score` DESC
3. `quality_score` DESC
4. `collaborative_score` DESC
5. `review_count` DESC
6. `rating` DESC
7. `restaurant_id` ASC

---

## 6. REST API Endpoints

### 1. `GET /api/v1/recommendations/hybrid/{user_id}`
- **Path Parameter**: `user_id` (integer)
- **Query Parameters**: `top_k`, `area`, `max_cost_for_two`, `min_rating`, `price_tier`, `online_order_only`, `book_table_only`, `latitude`, `longitude`.

### 2. `POST /api/v1/recommendations/hybrid`
- **Request Body**:
```json
{
  "user_id": 1,
  "preferred_cuisines": ["Biryani", "Mughlai"],
  "preferred_price_tier": "Moderate",
  "preferred_area": "Koramangala 5th Block",
  "latitude": 12.9352,
  "longitude": 77.6245,
  "max_cost_for_two": 800,
  "top_k": 3
}
```

- **Sample Response**:
```json
{
  "status": "success",
  "user_id": 1,
  "is_cold_start": false,
  "model_source": "hybrid",
  "effective_weights": {
    "content": 0.4,
    "collaborative": 0.2,
    "location": 0.15,
    "quality": 0.25
  },
  "count": 3,
  "recommendations": [
    {
      "restaurant_id": 4821,
      "name": "Biryani Foodies",
      "hybrid_score": 0.7482,
      "content_score": 0.82,
      "collaborative_score": 0.76,
      "location_score": 1.0,
      "quality_score": 0.66,
      "distance_km": 0.0,
      "rating": 3.9,
      "review_count": 142,
      "cuisines": "Biryani, Mughlai",
      "restaurant_type": "Casual Dining",
      "area": "Koramangala 5th Block",
      "address": "Koramangala 5th Block, Bengaluru",
      "price_tier": "Moderate",
      "cost_for_two_inr": 500,
      "online_order": true,
      "book_table": false,
      "location_source": "Bengaluru locality centroid",
      "location_precision": "locality-level",
      "model_source": "hybrid",
      "explanation": "Recommended for strong cuisine & style match (Biryani, Mughlai), high taste alignment with your dining history, close proximity (0.0 km away in Koramangala 5th Block)."
    }
  ]
}
```

---

## 7. Performance & Latency Benchmarks

- **Service Initialization (Catalog + TF-IDF + SVD)**: **`162 ms`**
- **Single User Hybrid Scoring (Candidates Scoring + Pruning + Tie-Breaking)**: **`42 ms – 58 ms`**
- **Memory Footprint**: Reuses singleton model references across Content and Collaborative engines without dataset duplication.

---

## 8. Test Verification

Total test suite across Phases 4–7: **`59 passed in 4.49s`**
- Database tests (Phase 4): `7 passed`
- Content-Based tests (Phase 5): `14 passed`
- Collaborative SVD tests (Phase 6): `12 passed`
- Hybrid Engine tests (Phase 7): **`26 passed`**
