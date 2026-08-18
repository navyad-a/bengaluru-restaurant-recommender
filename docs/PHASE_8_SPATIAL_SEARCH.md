# Phase 8: Location-Aware Proximity Scoring & Spatial Search Optimization

## 1. Objective & Architectural Overview

Phase 8 elevates the spatial retrieval and geographic proximity subsystem into a high-performance, modular geospatial engine supporting:

1. **Robust Coordinate Validation**: Bounds checking ($[-90, 90]$, $[-180, 180]$), non-finite (NaN/inf) rejection, and clean error handling.
2. **Numerically Stable Haversine Engine**: Scalar and vectorized spherical great-circle distance computation ($R = 6371.0088\text{ km}$).
3. **Bounding-Box Pre-Filtering**: Fast geometric spatial pruning prior to exact trigonometric distance evaluation.
4. **BallTree Spherical Spatial Indexing**: $O(\log N)$ nearest-neighbor and radius search on the 12,481 authentic Bengaluru restaurant catalog.
5. **Deterministic Spatial Ranking**: Tie-breaking hierarchy ensuring stable, reproducible ordering.
6. **Locality Spatial Analytics**: City-wide cluster summaries, pricing metrics, and inter-locality distance matrices.
7. **FastAPI Endpoints**: Dedicated `GET /api/v1/recommendations/nearby` and radius-constrained `POST /api/v1/recommendations/hybrid`.

```mermaid
flowchart TD
    subgraph Input Request
        A[User Latitude & Longitude / Radius]
    end

    subgraph Spatial Validation & Indexing
        A --> B[Coordinate Validation: Range [-90,90], [-180,180], No NaN/Inf]
        B --> C{Search Type}
        C -->|Radius Search| D1[BallTree query_radius OR Bounding Box Filter]
        C -->|k-NN Search| D2[BallTree query_nearest: Top K]
    end

    subgraph Distance & Proximity Computation
        D1 & D2 --> E[Calculate Exact Haversine Distance in km]
        E --> F[Compute Spatial Decay Score: S_location = exp(-d / tau)]
    end

    subgraph Constraint Filtering & Ranking
        F --> G[Enforce Hard Business Constraints: Cost, Rating, Type, Online Order]
        G --> H[Deterministic Tie-Breaking: Distance ASC, Rating DESC, Votes DESC, ID ASC]
        H --> I[Structured Response with Provenance Metadata]
    end
```

---

## 2. Mathematical Formulation & Distance Calculations

### A. Great-Circle Haversine Formula
For user coordinates $(\phi_1, \lambda_1)$ and restaurant coordinates $(\phi_2, \lambda_2)$ in radians:

$$\Delta\phi = \phi_2 - \phi_1, \quad \Delta\lambda = \lambda_2 - \lambda_1$$

$$a = \sin^2\left(\frac{\Delta\phi}{2}\right) + \cos(\phi_1)\cos(\phi_2)\sin^2\left(\frac{\Delta\lambda}{2}\right)$$

$$c = 2 \cdot \text{atan2}\left(\sqrt{a}, \sqrt{1 - a}\right)$$

$$d = R \cdot c \quad (R = 6371.0088\text{ km})$$

### B. Bounding-Box Angular Deltas
For a circular search radius $r_{\text{km}}$ around $(\text{lat}, \text{lon})$:

$$\Delta\text{lat} = \left(\frac{r_{\text{km}}}{R}\right) \cdot \left(\frac{180}{\pi}\right)$$

$$\Delta\text{lon} = \left(\frac{r_{\text{km}}}{R \cdot \cos(\text{lat}_{\text{rad}})}\right) \cdot \left(\frac{180}{\pi}\right)$$

$$\text{Bounding Box} = [\text{lat} - \Delta\text{lat}, \text{lat} + \Delta\text{lat}] \times [\text{lon} - \Delta\text{lon}, \text{lon} + \Delta\text{lon}]$$

### C. Exponential Spatial Proximity Score
$$S_{\text{location}} = \exp\left(-\frac{d}{\tau}\right) \quad (\text{Default } \tau = 3.0\text{ km})$$

- **$d = 0.0\text{ km}$**: $S_{\text{location}} = 1.0000$ (Direct proximity)
- **$d = 3.0\text{ km}$**: $S_{\text{location}} = 0.3679$ (1 decay constant)
- **$d = 10.0\text{ km}$**: $S_{\text{location}} = 0.0357$ (Substantial distance penalty)
- **Monotonicity**: Guaranteed strictly non-increasing over $d \ge 0$.

---

## 3. Coordinate Coverage & Provenance Metadata

| Parameter | Value / Status |
|---|---|
| **Catalog Outlets Total** | **`12,481` physical restaurant venues in Bengaluru** |
| **Geographic Coordinate Coverage** | **`100.00%` (12,481 / 12,481 non-null coordinates)** |
| **Latitude Range** | `12.8399° N` to `13.1007° N` |
| **Longitude Range** | `77.4838° E` to `77.7500° E` |
| **Data Source Attribution** | `location_source = "Bengaluru locality centroid"` |
| **Precision Level** | `location_precision = "locality-level"` |

> [!IMPORTANT]
> **Locality-Level Precision Transparency**: Coordinates represent Bengaluru neighborhood locality centroids. Exact street-level meter-precision GPS is not fabricated.

---

## 4. Locality Spatial Analytics (Top Bengaluru Dining Hubs)

| Locality / Hub | Outlets Count | Mean Rating | Median Cost (₹) | Centroid Coordinates |
|---|:---:|:---:|:---:|:---:|
| **Whitefield** | 882 | 3.60★ | ₹400 | (12.9700° N, 77.7500° E) |
| **Electronic City** | 729 | 3.47★ | ₹400 | (12.8400° N, 77.6800° E) |
| **BTM** | 728 | 3.57★ | ₹300 | (12.9200° N, 77.6100° E) |
| **HSR Layout** | 705 | 3.64★ | ₹400 | (12.9100° N, 77.6400° E) |
| **Marathahalli** | 684 | 3.53★ | ₹400 | (12.9600° N, 77.7000° E) |
| **Indiranagar** | 528 | 3.79★ | ₹500 | (12.9800° N, 77.6400° E) |
| **JP Nagar** | 523 | 3.63★ | ₹400 | (12.9100° N, 77.5900° E) |
| **Bannerghatta Road** | 474 | 3.48★ | ₹400 | (12.8900° N, 77.6000° E) |
| **Jayanagar** | 366 | 3.76★ | ₹400 | (12.9300° N, 77.5800° E) |
| **Bellandur** | 362 | 3.50★ | ₹400 | (12.9300° N, 77.6800° E) |

### Inter-Hub Haversine Distance Matrix (km):
```text
                 Whitefield  Electronic City    BTM    HSR  Marathahalli  Indiranagar
Whitefield             0.00            16.33  16.16  13.66          5.53        11.97
Electronic City       16.33             0.00  11.69   8.91         13.52        16.16
BTM                   16.16            11.69   0.00   3.44         10.72         7.42
HSR                   13.66             8.91   3.44   0.00          8.56         7.78
Marathahalli           5.53            13.52  10.72   8.56          0.00         6.87
Indiranagar           11.97            16.16   7.42   7.78          6.87         0.00
```

---

## 5. Performance Benchmark Results

Evaluated across $N=1,000$ spatial queries over the 12,481-outlet catalog (Search Radius = 3.0 km):

| Spatial Retrieval Method | Mean Latency | Median Latency | P95 Latency | Max Latency | Speedup |
|---|:---:|:---:|:---:|:---:|:---:|
| **A. Naive Haversine Scan (12,481 items)** | `0.3263 ms` | `0.3041 ms` | `0.4990 ms` | `1.3117 ms` | **`1.00x` (Baseline)** |
| **B. Bounding Box + Haversine** | `0.0896 ms` | `0.0774 ms` | `0.1881 ms` | `0.4703 ms` | **`3.64x`** |
| **C. BallTree Spherical Index** | `0.0924 ms` | `0.0814 ms` | `0.1539 ms` | `0.4277 ms` | **`3.53x`** |

---

## 6. REST API Endpoints

### 1. `GET /api/v1/recommendations/nearby`
- **Query Parameters**:
  - `latitude` (float, required): e.g. `12.9352`
  - `longitude` (float, required): e.g. `77.6245`
  - `radius_km` (float, optional): e.g. `3.0`
  - `top_k` (int, default 10): e.g. `5`

```json
{
  "status": "success",
  "latitude": 12.9352,
  "longitude": 77.6245,
  "radius_km": 3.0,
  "count": 3,
  "recommendations": [
    {
      "restaurant_id": 4821,
      "name": "Biryani Foodies",
      "distance_km": 0.0,
      "rating": 3.9,
      "review_count": 142,
      "cuisines": "Biryani, Mughlai",
      "restaurant_type": "Casual Dining",
      "area": "Koramangala 5th Block",
      "address": "Koramangala 5th Block, Bengaluru",
      "cost_for_two_inr": 500,
      "price_tier": "Moderate",
      "online_order": true,
      "book_table": false,
      "latitude": 12.9352,
      "longitude": 77.6245,
      "location_source": "Bengaluru locality centroid",
      "location_precision": "locality-level"
    }
  ]
}
```

### 2. `POST /api/v1/recommendations/hybrid` (With Spatial Radius Constraints)
- **Request Body**:
```json
{
  "latitude": 12.9784,
  "longitude": 77.6408,
  "radius_km": 3.0,
  "preferred_cuisines": ["Cafe", "Continental"],
  "top_k": 5
}
```

---

## 7. Test Verification & Regression

- **Total Test Suite**: **`83 passed in 6.20s`**
  - Database Tests (Phase 4): `7 passed`
  - Content-Based Tests (Phase 5): `14 passed`
  - Collaborative SVD Tests (Phase 6): `12 passed`
  - Hybrid Recommender Tests (Phase 7): `26 passed`
  - Spatial Engine Tests (Phase 8): **`24 passed`**
