# Phase 6: Collaborative Filtering — Surprise SVD Matrix Factorization

## 1. Objective & Overview

Phase 6 implements, trains, validates, and serves a **Collaborative Filtering (CF) Recommendation Engine** utilizing **Surprise SVD Matrix Factorization**.

> [!IMPORTANT]
> **Data Integrity & Benchmark Transparency Notice**:
> The authentic Zomato Bengaluru dataset is restaurant-centric and lacks genuine user IDs and customer interaction matrices. To evaluate Collaborative Filtering algorithms, an explicitly separated **Synthetic Collaborative Filtering Benchmark** ($N=600$ simulated users, 11,920 interactions, `SEED=42`, 80/20 per-user holdout) was constructed.
> 
> **All metrics in this phase evaluate SVD algorithm mechanics on the synthetic benchmark and must NOT be cited as real-world customer prediction accuracy.**

---

## 2. Mathematical Formulation

Surprise SVD predicts the rating $\hat{r}_{u,i}$ of user $u$ for restaurant $i$ as:

$$\hat{r}_{u,i} = \mu + b_u + b_i + q_i^T p_u$$

Where:
- $\mu$: Global baseline rating mean across all observed training interactions.
- $b_u \in \mathbb{R}$: User baseline bias (captures users who systematically rate higher or lower than average).
- $b_i \in \mathbb{R}$: Restaurant item baseline bias (captures inherently popular vs underperforming outlets).
- $p_u \in \mathbb{R}^k$: User latent factor vector ($k = \text{n\_factors}$).
- $q_i \in \mathbb{R}^k$: Restaurant latent factor vector ($k = \text{n\_factors}$).

### Optimization Objective:
The parameters are learned via Stochastic Gradient Descent (SGD) minimizing regularized squared error:

$$\min_{P, Q, b_u, b_i} \sum_{(u, i) \in \mathcal{K}} \left( r_{u,i} - (\mu + b_u + b_i + q_i^T p_u) \right)^2 + \lambda \left( \|p_u\|_2^2 + \|q_i\|_2^2 + b_u^2 + b_i^2 \right)$$

where $\lambda = \text{reg\_all}$ and $\gamma = \text{lr\_all}$.

---

## 3. Data Integrity & Benchmark Validation

Before training, automated integrity validation confirmed:
- **Simulated Users Count**: `600` (100% present in `synthetic_users.csv`)
- **Authentic Catalog Outlets**: `12,481` physical restaurant branches in Bengaluru
- **Represented Outlets in SVD**: `6,675` outlets with training interactions
- **Training Ratings Partition (80%)**: `9,777` ratings
- **Test Ratings Partition (20% holdout)**: `2,143` ratings
- **Matrix Sparsity**: `99.8408%`
- **Train/Test Pair Overlap**: `0` collisions (strict leakage-free holdout)
- **Rating Bounds**: All ratings bounded in $[1.0, 5.0]$.

---

## 4. Hyperparameter Experimentation (3-Fold CV on Training Data)

A 3-fold cross-validation grid search was executed strictly on the training partition:
* **`n_factors`**: `[50, 100, 150]`
* **`n_epochs`**: `[10, 20, 30]`
* **`reg_all`**: `[0.02, 0.05, 0.10]`
* **`lr_all`**: `0.005`
* **`random_state`**: `42`

### Results:
* **Total Configurations Tested**: `27` configurations (completed in 6.51s).
* **Optimal Configuration**:
  - `n_factors`: **50**
  - `n_epochs`: **30**
  - `reg_all`: **0.10**
  - `lr_all`: **0.005**
* **Validation RMSE**: **`0.6304`**

---

## 5. Final Offline Evaluation on Held-Out Test Set

The optimal SVD model was trained on the complete 9,777 training ratings and evaluated once against the 2,143 test ratings:

### A. Rating Prediction Error Metrics:
| Metric | Measured Score |
|---|:---:|
| **Test RMSE** | **`0.6171`** |
| **Test MAE** | **`0.5081`** |

### B. Top-K Ranking Metrics (Evaluated across 598 eligible test users against all 12,481 catalog items):
| Metric | K = 5 | K = 10 |
|---|:---:|:---:|
| **Precision@K** | `0.0003` | `0.0003` |
| **Recall@K** | `0.0006` | `0.0010` |
| **Hit Rate@K** | `0.0017` | `0.0033` |

*Note: In an unconstrained retrieval pool of 12,481 items with only 2–5 positive test items per user, precision@K reflects standard extreme catalog ranking dynamics.*

---

## 6. Cold-Start Handling & Unknown Users

1. **Known User**: Generates personalized SVD predicted ratings for all unseen authentic restaurants, ranking by predicted score with deterministic tie-breaking.
2. **Cold-Start / Unknown User**: When queried with an unseen `user_id`, the recommender raises an explicit `KeyError` / returns HTTP 404 with message:
   `"User ID {id} is unknown to Collaborative SVD (cold-start user). Collaborative Filtering requires historical interaction ratings."`
   *(In Phase 9, cold-start users will be seamlessly routed to Content-Based + Popularity routing).*
3. **Cold-Start Restaurant**: Unrated outlets fall back to global mean $\mu + b_u$.

---

## 7. Synthetic Benchmark Bias Analysis

1. **Persona Generator Priors**: Because synthetic interactions were generated from 7 parameterized dining personas with Gaussian noise ($\sigma=0.4$), the SVD model learns latent representations corresponding to those simulated taste clusters.
2. **Benchmark Utility**: Demonstrates end-to-end matrix factorization engineering, cross-validation tuning, sparse matrix evaluation, and REST API serving on an authentic Indian restaurant catalog.

---

## 8. Artifact Storage & Persistence

Stored in `saved_models/collaborative_model/`:
- `svd_model.joblib` (3.25 MB): Serialized Surprise SVD model instance.
- `known_entities.joblib` (135 KB): Sets of known user and item IDs, global rating mean.
- `model_metadata.json` (1.8 KB): Hyperparameters, metrics, and dataset disclaimer.

---

## 9. REST API Integration

### Endpoint:
`GET /api/v1/recommendations/collaborative/{user_id}?top_k=10`

### Sample Response:
```json
{
  "status": "success",
  "user_id": 1,
  "model_source": "collaborative_svd_synthetic_benchmark",
  "count": 5,
  "disclaimer": "Predictions generated by Surprise SVD trained on the Synthetic Collaborative Filtering Benchmark.",
  "recommendations": [
    {
      "restaurant_id": 1420,
      "name": "Balle-Licious Kitchen",
      "predicted_rating": 4.52,
      "rating": 4.0,
      "review_count": 85,
      "cuisines": "North Indian",
      "restaurant_type": "Quick Bites",
      "area": "HSR",
      "address": "HSR Layout, Sector 7",
      "price_tier": "Moderate",
      "cost_for_two_inr": 400,
      "online_order": true,
      "book_table": false,
      "location_source": "Bengaluru locality centroid",
      "location_precision": "locality-level",
      "model_source": "collaborative_svd_synthetic_benchmark"
    }
  ]
}
```

---

## 10. Performance Benchmarks

* **Cross-Validation Search (27 configs)**: `6.51 seconds`
* **Final Model Training (9,777 ratings)**: **`0.073 seconds`**
* **Model Serialization Time**: **`0.045 seconds`**
* **Service Loading Time**: **`141 ms`**
* **Single User Recommendation Latency (Scoring 12,481 outlets)**: **`48 ms – 53 ms`**
