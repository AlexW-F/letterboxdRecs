# Phase 3 at-scale evaluation

- Dataset: `ml-32m`
- Model: SVD `svd_full_slim.pkl` + ALS `als_full.pkl`
- Users sampled: **200** (min 50 ratings each)
- Groups sampled: **50** × 3 members
- Bootstrap: 1000 resamples · 95% percentile CI
- Runtime: 1150.4s

## Individual recs — by mode

| mode | NDCG@10 (95% CI) | Recall@50 | Intra-list diversity@10 |
|---|---|---|---|
| `balanced` | 0.1572  [0.1344, 0.1814]  n=200 | 0.1066  [0.0900, 0.1231]  n=200 | 0.8332  [0.8249, 0.8408]  n=200 |
| `niche` | 0.0247  [0.0157, 0.0340]  n=200 | 0.0440  [0.0335, 0.0551]  n=200 | 0.8359  [0.8272, 0.8451]  n=200 |
| `popular` | 0.2664  [0.2379, 0.2965]  n=200 | 0.2195  [0.1988, 0.2424]  n=200 | 0.7714  [0.7625, 0.7807]  n=200 |
| `serendipitous` | 0.1497  [0.1288, 0.1730]  n=200 | 0.1092  [0.0933, 0.1262]  n=200 | 0.9197  [0.9139, 0.9250]  n=200 |

## Group recs — strategy × mode

| strategy | mode | avg per-member NDCG@10 (95% CI) | fairness CV | intra-list div |
|---|---|---|---|---|
| `average` | `balanced` | 0.0616  [0.0414, 0.0803]  n=50 | 0.5390  [0.3780, 0.6959]  n=50 | 0.7013  [0.6801, 0.7230]  n=50 |
| `least_misery` | `balanced` | 0.0066  [0.0014, 0.0144]  n=50 | 0.2298  [0.0884, 0.3747]  n=50 | 0.7051  [0.6815, 0.7282]  n=50 |
| `most_pleasure` | `balanced` | 0.0768  [0.0602, 0.0947]  n=50 | 0.8896  [0.7589, 1.0169]  n=50 | 0.7273  [0.7113, 0.7437]  n=50 |
| `hybrid` | `balanced` | 0.0573  [0.0387, 0.0753]  n=50 | 0.5214  [0.3633, 0.6695]  n=50 | 0.7111  [0.6883, 0.7335]  n=50 |
| `group_taste_vector` | `balanced` | 0.0407  [0.0278, 0.0539]  n=50 | 0.7976  [0.6158, 0.9692]  n=50 | 0.8556  [0.8431, 0.8670]  n=50 |

## How to read this

- **NDCG@10**: ranking quality on held-out high-rated items (higher = better).
- **Recall@50**: fraction of relevant holdout items appearing in the top-50.
- **Fairness CV**: coefficient of variation of per-member NDCG@10 within a group (lower = more equitable across members).
- **Intra-list diversity@10**: avg pairwise genre dissimilarity inside one top-10 (higher = more varied).
- **CI overlap**: if two rows' CIs overlap, the difference between them is NOT statistically meaningful at α=0.05.
