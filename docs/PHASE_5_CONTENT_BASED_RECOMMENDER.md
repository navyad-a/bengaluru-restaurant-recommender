# Phase 5: Content-Based Recommendation Engine

## 1. Objective & Overview

Phase 5 implements a high-performance, production-grade **Content-Based Recommendation Engine** tailored for the authentic **12,481 physical Bengaluru restaurant catalog**. 

The engine operates on two core modes:
1. **Mode A — Restaurant-to-Restaurant Recommendation**: Given a specific restaurant ID, retrieves the top-$K$ most similar restaurants based on TF-IDF metadata cosine similarity.
2. **Mode B — Preference-to-Restaurant Recommendation**: Given user dietary preferences (cuisines, dining format, price tier, maximum budget, locality), constructs a matched query vector, evaluates hard constraints, and ranks candidate restaurants.

---

## 2. Recommendation Pipeline Architecture

```mermaid
flowchart TD
    A[Authoritative Catalog: 12,481 Clean Restaurants] --> B[Feature Engineering & Token Normalization]
    B -->|Prefix Isolation: cuisine_, type_, area_, price_| C[Normalized Feature Documents]
    C --> D[TF-IDF Vectorizer with Sublinear TF & min_df=2]
    D --> E[Sparse TF-IDF Matrix: 12,481 x 1,673 CSR]
    E --> F[Artifact Storage: saved_models/content_model/]
    
    subgraph Online Inference Engine
        G1[Query Restaurant ID] --> H1[Retrieve Sparse Row Vector]
        G2[User Preference Query] --> H2[Transform via TF-IDF Vectorizer]
        
        H1 --> I[Sparse Dot Product Cosine Similarity: S = X . q^T]
        H2 --> I
        
        I --> J[Hard Constraint Filtering: Max Cost, Min Rating, Area]
        J --> K[Deterministic Tie-Breaking & Ranking]
        K --> L[Top-K Recommended Restaurants with Content Scores]
    end
    
    F -.-> Online Inference Engine
```

---

## 3. Authentic Features Used

The content model strictly utilizes legitimate fields from `data/processed/restaurants_clean.csv`:

| Feature Name | Type | Importance | Replicated Weight | Purpose |
|---|---|---|:---:|---|
| **`cuisines`** | Text (Comma-separated) | High | **3x** | Primary culinary taste profile (e.g. South Indian, North Indian, Chettinad, Mughlai). |
| **`rest_type`** | Text (Comma-separated) | High | **2x** | Dining format & atmosphere (e.g. Quick Bites, Casual Dining, Cafe, Microbrewery). |
| **`area`** | Text | Medium | **2x** | Neighborhood locality affinity across 93 Bengaluru areas. |
| **`price_tier`** | Categorical string | Medium | **1x** | Budget, Moderate, Premium, Luxury brackets. |
| **`rating`** | Categorical bucket | Supporting | **1x** | Binned into `rating_exceptional`, `rating_high`, `rating_medium`, `rating_low`, `rating_unrated`. |
| **`cost_for_two_inr`** | Categorical bucket | Supporting | **1x** | Binned into 5 granular INR brackets (`cost_under_300`, `cost_300_to_600`, etc.). |
| **`online_order`** | Boolean flag | Supporting | **1x** | `online_order_yes` / `online_order_no`. |
| **`book_table`** | Boolean flag | Supporting | **1x** | `book_table_yes` / `book_table_no`. |
| **`dish_liked`** | Text (Comma-separated) | Supporting | **1x** | Signature menu items (e.g. `dish_masala_dosa`, `dish_dum_biryani`). |

---

## 4. Token Collision Prevention & Prefixing

To eliminate false semantic collisions (e.g., preventing a restaurant in locality *"Indiranagar"* from matching a cuisine named *"Indiranagar"*, or *"Cafe"* locality vs *"Cafe"* restaurant type), every token is strictly prefixed:
- `cuisine_{cleaned_name}` (e.g. `cuisine_south_indian`, `cuisine_karnataka`, `cuisine_mughlai`)
- `type_{cleaned_type}` (e.g. `type_casual_dining`, `type_quick_bites`, `type_microbrewery`)
- `area_{cleaned_area}` (e.g. `area_indiranagar`, `area_koramangala_5th_block`, `area_jayanagar`)
- `price_{tier}` (e.g. `price_budget`, `price_moderate`, `price_premium`, `price_luxury`)
- `dish_{cleaned_dish}` (e.g. `dish_filter_coffee`, `dish_crispy_dosa`, `dish_craft_beer`)

---

## 5. TF-IDF Vectorizer Parameters & Justification

```python
TfidfVectorizer(
    token_pattern=r"(?u)\b\w+\b",
    min_df=2,
    sublinear_tf=True,
    ngram_range=(1, 1),
    norm="l2",
    lowercase=False
)
```

### Parameter Rationale:
1. **`token_pattern=r'(?u)\b\w+\b'`**: Captures multi-word tokens joined by underscores (e.g. `cuisine_south_indian`).
2. **`min_df=2`**: Prunes one-off typographical noise in scraped dish names while preserving all 107 authentic cuisines, 93 localities, and dining types.
3. **`sublinear_tf=True`**: Scales term frequency as $1 + \log(\text{tf})$, preventing restaurants with 20 listed dishes from artificially drowning out focused specialty eateries.
4. **`norm='l2'`**: Normalizes all output vectors to unit Euclidean length ($\|\mathbf{v}\|_2 = 1.0$). Consequently, the cosine similarity between candidate $\mathbf{x}_i$ and query $\mathbf{q}$ reduces to a direct sparse dot product:
   $$\text{Sim}(\mathbf{x}_i, \mathbf{q}) = \mathbf{x}_i \cdot \mathbf{q}^T$$

---

## 6. Matrix & Artifact Statistics

| Metric | Measured Value |
|---|---|
| **Authoritative Catalog Size** | `12,481` physical restaurant outlets |
| **Vocabulary Dimension** | `1,673` unique clean tokens |
| **TF-IDF Matrix Shape** | `(12481, 1673)` |
| **Non-Zero Entries** | `140,390` non-zero values |
| **Matrix Storage Format** | `scipy.sparse.csr_matrix` (~1.7 MB on disk) |
| **Model Build / Fit Time** | **`0.458 seconds`** |
| **Average Query Latency** | **`7.5 ms – 18.5 ms`** |

---

## 7. Model Artifact Management

Artifacts are serialized in `saved_models/content_model/`:
- `tfidf_vectorizer.joblib` (52 KB): Fitted `TfidfVectorizer` instance.
- `tfidf_matrix.joblib` (1.69 MB): Compressed Sparse Row (`csr_matrix`) feature matrix.
- `restaurant_catalog.joblib` (4.16 MB): In-memory DataFrame for metadata lookups.
- `restaurant_mappings.joblib` (145 KB): Bi-directional mapping between `restaurant_id` and matrix row indices.
- `feature_metadata.json` (0.3 KB): Model provenance and dimensional metadata.

---

## 8. REST API Endpoints

### 1. `GET /api/v1/recommendations/similar/{restaurant_id}?top_k=10`
* **Purpose**: Restaurant-to-Restaurant recommendation (Mode A).

### 2. `POST /api/v1/recommendations/content`
* **Purpose**: Preference-to-Restaurant matching with hard filtering (Mode B).

---

## 9. Test Suite Verification

Execution of complete Pytest suite (`pytest -v`):
- `tests/test_content_recommender.py`: 14 passed
- `tests/test_database.py`: 7 passed
- **Total: 21 passed (100%)**

---

## 10. Documented Limitations & Future Integration

1. **Metadata Cosine Similarity**: Measures similarity between restaurant descriptive attributes and user stated preferences; it does not model collaborative community wisdom (which will be addressed by Surprise SVD in Phase 6).
2. **Locality Coordinates**: Distances and localities reflect Bengaluru locality centroids (`location_precision = "locality-level"`).
3. **Pure Content Model**: Does not use synthetic ratings for training or evaluation, preserving complete data integrity.
