# Phase 0 Findings

## Rating scale investigation

**What I tested:** built two tiny Surprise datasets — one with `Reader(rating_scale=(1, 5))` (the old default) and one with `Reader(rating_scale=(0.5, 5.0))` — containing a 0.5-star rating. Then loaded every existing `models/*.pkl` and recorded each model's baked-in `trainset.rating_scale`.

**What I found:**
- Surprise does **not** reject or clip out-of-range ratings during training. A 0.5★ rating fed to `Reader(rating_scale=(1, 5))` is retained as 0.5 and used in SGD.
- The scale **is** used for prediction clipping — `est = clip(est, scale[0], scale[1])` happens in `predict()`. With `(1, 5)`, predictions <1.0 get clamped to 1.0. Doesn't affect ranking, but the cosmetic "predicted star" in user-facing UI loses fidelity.
- Cause and fix: `src/data_processing.py:205` was using `(1, 5)`; `load_movielens_data` was using the default which is also `(1, 5)`. Both now set `(0.5, 5.0)` to match `config.MIN_RATING/MAX_RATING`, ml-32m, and Letterboxd exports.
- **Existing pickled models cannot be retroactively fixed** — their `trainset.rating_scale` is baked into the pickle. To get accurate user-facing star predictions, models must be retrained with the new Reader.

## Surprise findings: what's actually in models/

Loaded each pickle and inspected `trainset.n_users` / `trainset.n_items`:

| File | Size | Rating scale | Users × Items | Verdict |
|---|---|---|---|---|
| `svdpp.pkl` | 10.5 MB | (1, 5) | 610 × 9724 | **ml-latest-small** — *not* ml-32m as claimed |
| `svd_best.pkl` | 0.1 MB | (1, 5) | **1 × 150** | Trained on a single user's Letterboxd data — useless |
| `svdpp_best.pkl` | 0.4 MB | (1, 5) | **1 × 150** | Same — useless |
| `knnbasic_best.pkl` | 0.2 MB | (1, 5) | **1 × 150** | Same — useless |
| `knnbaseline_best.pkl` | 0.2 MB | (1, 5) | **1 × 150** | Same — useless |
| `knnwithmeans_best.pkl` | 0.2 MB | (1, 5) | **1 × 150** | Same — useless |
| `knn_knnbaseline.pkl` | 724.3 MB | (not opened — too big for trace) | — | Most likely the only model on ml-32m |

**Implications for the plan:**
1. `svdpp.pkl` is fine as a Phase 0/1 baseline — 9724 items is plenty to validate the re-ranking pipeline. Retrain on ml-32m as part of Phase 1 once ALS is in place (Phase 1.5 lite, or fold into Phase 1).
2. The 5 `*_best.pkl` files should be considered artifacts of a broken hyperparameter-search run and *not* used. Delete or move to `archives/` once we have legitimate replacements.
3. The 724 MB KNN file should be loaded once to verify shape; if it really is ml-32m, keep it but don't load it in the API (per plan, drop KNN from serving).

## Other Phase 0 cleanups
- Corrupted module docstring in `src/recommendations.py:1-28` (merge-conflict splice) — fixed.
- Hardcoded `k=40` and `reg=0.1` in `recommendations.py` — replaced with `DEFAULT_K_NEIGHBORS` and `DEFAULT_FOLD_IN_REG` from `src/config.py`.
- Added `DEFAULT_FOLD_IN_REG`, `DEFAULT_CANDIDATE_POOL_SIZE`, `COLD_START_MIN_OVERLAP` to `config.py` (forward-looking for Phase 1).

## Bug discovered during smoke test (queued for Phase 1)

**`load_user_data_with_tmdb` merge was broken.** Both `alex_data/ratings_with_tmdb.csv` and `ml-latest-small/links.csv` store `tmdbId` as float64 with a `.0` suffix. The old code did `user.astype(int).astype(str)` on one side but `links.astype(str)` on the other — yielding `"635731"` vs `"862.0"`, so zero rows merged. Fixed in `recommendations.py` by normalizing both sides through `pd.to_numeric` → `int` → `str`. With the fix, 116 of 250 alex_data ratings (~46%) map to MovieLens; the rest are post-2018 films missing from ml-latest-small.

**Group engine treats missing candidates as score=0.0.** When a movie appears in some members' top-30 individual recs but not others, the group aggregator inserts 0.0 for the missing members. This poisons `average` and `least_misery` strategies — e.g., a movie scored 7.77 for one member and "not predicted" for the other two ends up with min=0.0, avg=2.6, even though SVD++ could have predicted a real score in the 3-5 range for the other two if asked. **Fix in Phase 1:** when implementing the candidate-gen → re-rank refactor, ensure the per-member score lookup either (a) generates a real prediction via fold-in for all candidates, or (b) excludes the member from that movie's aggregation entirely. The current "zero-fill" semantics are pathological.

## Baseline smoke-test output snapshot

Synthetic 3-friend group from alex_data:
- `alex_real` (116 ratings, real)
- `optimistic_friend` (38 ratings, alex's mid-third with +0.5 bias)
- `selective_friend` (39 ratings, alex's high-rated bottom-third, all set to 5.0)

Top-3 group recs by strategy (truncated):
- average: Shawshank, Fight Club, Fifth Element (driven by selective_friend with zeros from others)
- least_misery: same (because of the zero-fill bug above)
- most_pleasure: Pulp Fiction, Shawshank, Chinatown
- consensus: Pulp Fiction, Chinatown, Kill Bill — most plausible group output
- hybrid: Pulp Fiction, Reservoir Dogs, Spider-Man

The recs are all canonical 90s/00s classics — exactly the popularity-bias signal the Phase 1 re-ranker needs to break.
