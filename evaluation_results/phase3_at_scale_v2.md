# Phase 3 at-scale evaluation

- Dataset: `ml-32m`
- Model: SVD `svd_full_slim.pkl` + ALS `als_full.pkl`
- Users sampled: **200** (min 50 ratings each)
- Groups sampled: **50** × 3 members
- Bootstrap: 1000 resamples · 95% percentile CI
- Runtime: 1228.4s

## Individual recs — by mode

| mode | NDCG@10 | Recall@50 | List diversity@10 | Catalog coverage@50 | Gini popularity@50 |
|---|---|---|---|---|---|
| `balanced` | 0.1572  [0.1344, 0.1814]  n=200 | 0.1066  [0.0900, 0.1231]  n=200 | 0.8332  [0.8249, 0.8408]  n=200 | 0.0351  [0.0246, 0.0273] | 0.8102  [0.7997, 0.8193] |
| `niche` | 0.0247  [0.0157, 0.0340]  n=200 | 0.0440  [0.0335, 0.0551]  n=200 | 0.8359  [0.8272, 0.8451]  n=200 | 0.0371  [0.0258, 0.0287] | 0.9075  [0.8983, 0.9148] |
| `popular` | 0.2664  [0.2379, 0.2965]  n=200 | 0.2195  [0.1988, 0.2424]  n=200 | 0.7714  [0.7625, 0.7807]  n=200 | 0.0191  [0.0144, 0.0159] | 0.4914  [0.4772, 0.5053] |
| `serendipitous` | 0.1497  [0.1288, 0.1730]  n=200 | 0.1092  [0.0933, 0.1262]  n=200 | 0.9197  [0.9139, 0.9250]  n=200 | 0.0350  [0.0244, 0.0271] | 0.8102  [0.7994, 0.8195] |
| `calibrated` | 0.1269  [0.1103, 0.1469]  n=200 | 0.1320  [0.1141, 0.1509]  n=200 | 0.7456  [0.7361, 0.7549]  n=200 | 0.0349  [0.0245, 0.0273] | 0.7755  [0.7631, 0.7868] |

## Group recs — strategy × mode

| strategy | mode | avg per-member NDCG@10 | fairness CV | list div | catalog cov | Gini pop |
|---|---|---|---|---|---|---|
| `average` | `balanced` | 0.0616  [0.0414, 0.0803]  n=50 | 0.5390  [0.3780, 0.6959]  n=50 | 0.7013  [0.6801, 0.7230]  n=50 | 0.0105  [0.0067, 0.0084] | 0.8419  [0.8088, 0.8686] |
| `least_misery` | `balanced` | 0.0066  [0.0014, 0.0144]  n=50 | 0.2298  [0.0884, 0.3747]  n=50 | 0.7051  [0.6815, 0.7282]  n=50 | 0.0112  [0.0071, 0.0089] | 0.9085  [0.8828, 0.9199] |
| `most_pleasure` | `balanced` | 0.0775  [0.0607, 0.0959]  n=50 | 0.8860  [0.7630, 1.0051]  n=50 | 0.7277  [0.7114, 0.7442]  n=50 | 0.0110  [0.0070, 0.0088] | 0.7116  [0.6790, 0.7417] |
| `hybrid` | `balanced` | 0.0573  [0.0387, 0.0753]  n=50 | 0.5214  [0.3633, 0.6695]  n=50 | 0.7111  [0.6883, 0.7335]  n=50 | 0.0106  [0.0068, 0.0085] | 0.8545  [0.8259, 0.8806] |
| `group_taste_vector` | `balanced` | 0.0407  [0.0278, 0.0539]  n=50 | 0.7976  [0.6158, 0.9692]  n=50 | 0.8556  [0.8431, 0.8670]  n=50 | 0.0115  [0.0071, 0.0090] | 0.8097  [0.7893, 0.8236] |

## How to read this

- **NDCG@10**: ranking quality on held-out high-rated items (higher = better).
- **Recall@50**: fraction of relevant holdout items appearing in the top-50.
- **Fairness CV**: coefficient of variation of per-member NDCG@10 within a group (lower = more equitable across members).
- **List diversity@10**: avg pairwise genre dissimilarity inside one top-10 (higher = more varied).
- **Catalog coverage@50**: fraction of the full catalog (~88k films) that surfaces across all users'/groups' top-50 — high coverage means the system explores; low means it recycles the same shortlist.
- **Gini popularity@50**: inequality of popularity within the recommended pool. 0 = uniform across all titles, 1 = all recs concentrated on one mega-hit. High Gini = the system is funneling toward blockbusters.
- **CI overlap**: if two rows' CIs overlap, the difference between them is NOT statistically meaningful at α=0.05.
