# GitHub Release & Repository Quality Checklist

## 1. Repository Cleanliness & Security
- [x] **No Secrets Committed**: `.env` is ignored by `.gitignore`; only `.env.example` is committed.
- [x] **No Machine Paths**: No hardcoded absolute local paths in application code.
- [x] **No Cache Clutter**: `__pycache__`, `.pytest_cache`, `.coverage`, `htmlcov` ignored.
- [x] **Lightweight Artifacts**: Processed dataset and model weights are under GitHub 100MB limit.

---

## 2. GitHub Metadata
- **Repository Name**: `bengaluru-restaurant-recommender` (or `restaurant-recommendation-system`)
- **Description** (Under 160 chars):
  > "Production hybrid AI recommendation platform for Bengaluru restaurants with FastAPI, SVD, BallTree, MMR, Redis, and Streamlit."
- **Topics & Tags**:
  `recommendation-system`, `fastapi`, `streamlit`, `machine-learning`, `collaborative-filtering`, `svd`, `mmr-diversification`, `balltree`, `redis`, `docker-compose`, `python`

---

## 3. Recommended Git Commit & Push Workflow
```bash
# 1. Review status
git status

# 2. Stage all polished files
git add .

# 3. Commit with semantic release message
git commit -m "feat(release): Phase 18 complete project polish, documentation and live verification"

# 4. Push to remote main branch
git push origin main
```
