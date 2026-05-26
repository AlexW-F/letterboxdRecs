# Phase 3 at-scale evaluation

- Dataset: `ml-32m`
- Model: SVD `svd_full_slim.pkl` + ALS `als_full.pkl`
- Users sampled: **200** (min 50 ratings each)
- Groups sampled: **30** × 3 members
- Bootstrap: 1000 resamples · 95% percentile CI
- Runtime: 319.3s

## Individual recs — by mode

| mode | NDCG@10 (95% CI) | Recall@50 | Intra-list diversity@10 |
|---|---|---|---|
| `balanced` | 0.1572  [0.1344, 0.1814]  n=200 | 0.1066  [0.0900, 0.1231]  n=200 | 0.8332  [0.8249, 0.8408]  n=200 |
| `calibrated` | 0.1269  [0.1103, 0.1469]  n=200 | 0.1320  [0.1141, 0.1509]  n=200 | 0.7456  [0.7361, 0.7549]  n=200 |

## Group recs — strategy × mode

| strategy | mode | avg per-member NDCG@10 (95% CI) | fairness CV | intra-list div |
|---|---|---|---|---|
| `average` | `balanced` | 0.0768  [0.0507, 0.1032]  n=30 | 0.6764  [0.5011, 0.8676]  n=30 | 0.6973  [0.6711, 0.7227]  n=30 |
| `average` | `calibrated` | 0.0757  [0.0500, 0.1018]  n=30 | 0.6552  [0.4963, 0.8418]  n=30 | 0.6993  [0.6738, 0.7256]  n=30 |
| `group_taste_vector` | `balanced` | 0.0439  [0.0235, 0.0661]  n=30 | 0.7129  [0.4966, 0.9478]  n=30 | 0.8477  [0.8291, 0.8637]  n=30 |
| `group_taste_vector` | `calibrated` | 0.0488  [0.0365, 0.0637]  n=30 | 1.0355  [0.8541, 1.1967]  n=30 | 0.7846  [0.7677, 0.8015]  n=30 |

## How to read this

- **NDCG@10**: ranking quality on held-out high-rated items (higher = better).
- **Recall@50**: fraction of relevant holdout items appearing in the top-50.
- **Fairness CV**: coefficient of variation of per-member NDCG@10 within a group (lower = more equitable across members).
- **Intra-list diversity@10**: avg pairwise genre dissimilarity inside one top-10 (higher = more varied).
- **CI overlap**: if two rows' CIs overlap, the difference between them is NOT statistically meaningful at α=0.05.
