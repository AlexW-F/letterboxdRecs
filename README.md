# letterboxdRecs

**movienight** — a group movie-recommendation web app built on Letterboxd exports + the MovieLens 32M dataset. Friends each contribute their watch history (Letterboxd username via RSS, or full CSV export); the backend folds everyone into the same ALS latent space, blends in Tag Genome relevance and TMDB plot embeddings, and produces a re-ranked group recommendation list with per-member fairness controls. Includes a shareable join link with QR code, real-time voting on the shortlist, a "shared watchlist" overlap view, and a 3D UMAP visualization of your taste in latent space.

Live: [movienight.alex-wf.com](https://movienight.alex-wf.com) · Repo: [github.com/AlexW-F/letterboxdRecs](https://github.com/AlexW-F/letterboxdRecs)

## What you can do

- **Group movie night.** One person clicks *Start a shared group* → gets a URL + QR code → friends join from their phones with a Letterboxd username (no upload needed) or a CSV export. The group recs page shows top picks plus three lenses: high-disagreement "argue about this" films, films *no one* has seen yet, and a shared-watchlist overlap. Each member can thumbs-up / veto from their device; the highest net vote surfaces as the group pick.
- **Solo recs.** Upload your Letterboxd export once, get personalized recommendations across five modes (`balanced`, `niche`, `popular`, `serendipitous`, `calibrated`) with per-rec explanations showing which signal (SVD / ALS / Tag Genome / popularity-debias / diversity) drove the pick.
- **Compatibility report.** Pairwise Pearson correlation on shared ratings + cosine similarity of TF-IDF taste vectors, plus strict consensus picks (mean ≥ 3.5★, σ ≤ 0.75) vs disagreement picks (σ ≥ 1.0★) — no longer the same films appearing on both lists.
- **3D movie space.** Personalized UMAP projection of the ALS latent space with your rated films highlighted and your folded-in taste vector plotted as a marker.

## Architecture

```
letterboxdRecs/
├── src/
│   ├── reranking.py         # hybrid CF (SVD + ALS) + content + MMR / calibrated diversity
│   ├── group_reranker.py    # 6 strategies including the latent-space group_taste_vector
│   ├── content_features.py  # Tag Genome 2021 + IMDb directors + MiniLM overview embeddings
│   ├── evaluation.py        # NDCG/recall/coverage/Gini/fairness with bootstrap CIs
│   ├── letterboxd_rss.py    # username → recent ratings via the public RSS feed
│   ├── viz.py               # 3D UMAP renderer for the explore page
│   └── api/                 # FastAPI app (main, schemas, routers, state)
├── web/                     # SvelteKit + Tailwind frontend (served by nginx in prod)
├── scripts/
│   ├── train_svd.py / train_als.py            # train the underlying CF models
│   ├── build_genome_features.py               # build the primary content scorer
│   ├── build_movie_space_index.py             # 3D viz index
│   ├── run_eval.py                            # phase 1/2 single-config eval
│   └── run_eval_at_scale.py                   # phase 3: N≥200 users + bootstrap CIs
├── models/                  # trained pickles (gitignored — built locally)
├── ml-32m/                  # MovieLens 32M dataset (gitignored — download separately)
├── data/                    # processed content features (gitignored)
└── evaluation_results/      # JSON + markdown reports
```

## Quick start — full stack via docker compose

The simplest way to run everything locally — FastAPI + SvelteKit + nginx, frontend proxying `/api/*` to the backend over the internal docker network:

```bash
VITE_TMDB_KEY=your_tmdb_v3_key docker compose up --build
# open http://localhost:5173/
```

The `VITE_TMDB_KEY` is optional — without it, posters and streaming-availability badges silently degrade. Get one at https://www.themoviedb.org/settings/api.

What the stack expects on disk (read-only mounted into the api container):
- `models/svd_full_slim.pkl` — Surprise SVD on ml-32m (`scripts/train_svd.py` then `scripts/slim_svd_pickle.py`)
- `models/als_full.pkl` — implicit ALS on ml-32m (`scripts/train_als.py --ratings ml-32m/ratings.csv`)
- `data/content_genome.{npz,json}` — Tag Genome 2021 (download `genome_2021.zip` from grouplens.org into `archives/`, unzip, then `scripts/build_genome_features.py`)
- `ml-32m/` — the MovieLens 32M dataset
- (optional) `data/movie_space_index.pkl` — 3D viz precomputed index (`scripts/build_movie_space_index.py`)

See `.env.example` for tunable paths.

## REST API

OpenAPI docs at `http://localhost:8000/docs`.

| Endpoint | Purpose |
|---|---|
| `GET /health` | service + model-loaded status |
| `GET /modes` | the five recommendation modes (`balanced`, `niche`, `popular`, `serendipitous`, `calibrated`) with weight breakdown |
| `GET /strategies` | the six group strategies including `group_taste_vector` |
| `POST /upload-letterboxd` | upload `ratings.csv` (+ optional `watched.csv`, `watchlist.csv`). A raw export maps to the catalog via local title+year matching — no TMDB key required; a configured key only adds a TMDB-search fallback for titles not found locally. Returns a SHA-256 content hash that doubles as a session id. |
| `POST /upload-letterboxd-username` | RSS-based ingest: `{username}` → the most-recent ~50 ratings, TMDB IDs included inline |
| `POST /recommend/individual` | `{hash, mode, top_n}` → top-N with per-rec breakdown |
| `POST /recommend/group` | `{hashes, member_names?, strategy, mode, top_n, exclude_seen_by_any?}` — `exclude_seen_by_any` powers the strict "Nobody's seen" tab |
| `POST /recommend/group/disagreement` | high-variance picks where members disagree |
| `POST /group/analyze` | pairwise compatibility report + threshold-gated consensus / disagreement lists |
| `POST /group/watchlist-overlap` | films multiple members already want to see |
| `POST /group` | create a shareable group → returns `group_id` |
| `POST /group/demo` | spin up a shareable group seeded with 3 cached Letterboxd users (handy for API smoke tests) |
| `POST /group/{id}/join` | add a member (existing upload hash + display name) |
| `POST /group/{id}/vote` | cast `up` / `veto` / `clear` on a film |
| `GET  /group/{id}` | fetch current group state (members + votes) |
| `GET  /explore/background` | catalog coordinates for the 3D viz |
| `GET  /explore/personalized?hash=…` | personalized 3D HTML for a specific upload |

Caching is content-addressable — re-uploading the same export hits the same diskcache entry, no session lifecycle.

## Modes + strategies

**Individual rec modes** (verified at scale across N=200 users, 95% bootstrap CIs — see `evaluation_results/phase3_at_scale_v2.md`):

| mode | NDCG@10 | Catalog coverage | Gini | When to use |
|---|---|---|---|---|
| `popular` | 0.27 | 0.019 | 0.49 | Wins NDCG by recycling the same ~1.7k canon films across all users |
| `balanced` | 0.16 | 0.035 | 0.81 | Default. ~2x coverage of `popular`, sane mid-popularity floor |
| `serendipitous` | 0.15 | 0.035 | 0.81 | Heaviest diversity weight, slightly wider net |
| `calibrated` | 0.13 | 0.035 | 0.78 | Steck-style: matches the user's historical genre distribution. Trades top-K precision for breadth |
| `niche` | 0.025 | 0.037 | 0.91 | Aggressive popularity penalty — hidden gems, poor NDCG by design |

`popular` winning NDCG@10 is partly an artifact of catalog concentration: bootstrap CIs on `catalog_coverage_at_50` show it surfaces ~half as many unique films as the other modes.

**Group aggregation strategies** (NDCG and fairness CV across N=50 groups, balanced mode):

| strategy | NDCG | Fairness CV | Notes |
|---|---|---|---|
| `most_pleasure` | 0.077 | 0.89 (worst) | Highest NDCG, but actively hurts the unhappy member |
| `average` | 0.062 | 0.54 | Principled middle |
| `hybrid` | 0.057 | 0.52 | Average + worst-score bonus |
| `group_taste_vector` | 0.041 | 0.80 | The interesting one: latent-space fusion (max-merge ratings → single ALS fold-in). Lower NDCG but distinct picks no member would rank top-K individually. |
| `least_misery` | 0.007 | 0.23 (best) | Broken at scale — the require-all-members filter is too aggressive |

The `group_taste_vector` strategy is the genuinely uncommon move in this codebase: academic group-recsys literature (Masthoff, Felfernig, Kaya & Bridge, Serbos) focuses on aggregating *scores* after each member's individual recs are computed. Fusing in latent space *before* fold-in can surface films no individual member would rank top-K.

## Eval harness

Per-call evaluation (`scripts/run_eval.py`): ranking quality on holdout, no CIs.

At-scale evaluation (`scripts/run_eval_at_scale.py`): streams ml-32m, samples N users + M groups, runs 1000 bootstrap resamples for percentile CIs on NDCG/Recall/intra-list-diversity/fairness-CV, plus list-aggregated CIs on catalog coverage + Gini-popularity.

```bash
# the full at-scale run (~20 min on ml-32m via docker exec)
docker exec letterboxd-recs-api python scripts/run_eval_at_scale.py \
  --n-users 200 --n-groups 50 \
  --modes balanced,niche,popular,serendipitous,calibrated \
  --strategies average,least_misery,most_pleasure,hybrid,group_taste_vector \
  --bootstrap-n 1000 \
  --output evaluation_results/phase3_at_scale_v2.json \
  --report evaluation_results/phase3_at_scale_v2.md
```

Reports land in `evaluation_results/phase3_at_scale_v2.md`.

## Library API

- `src.reranking` — `Reranker` (hybrid CF + content + diversity), `ALSScorer`, `SVDScorer`, `MODE_WEIGHTS`
- `src.group_reranker` — `GroupReranker` with 6 strategies, including the latent-space `group_taste_vector`
- `src.content_features` — Tag Genome 2021 + IMDb directors + sentence-transformer overview embeddings, weight-averaged
- `src.evaluation` — NDCG/Recall/intra-list-diversity/fairness-CV with bootstrap CIs + list-aggregated catalog coverage + Gini
- `src.letterboxd_rss` — username → recent rated films via the public RSS feed (HTML is Cloudflare-gated)
- `src.viz` — 3D UMAP renderer for the explore page
- `src.api` — FastAPI app + state singleton + routers (recommendations, groups, explore, meta)
- `web/` — SvelteKit frontend, Tailwind, Playfair Display for headers / Inter for body

## Datasets + credits

- **MovieLens 32M** — community ratings (200,948 users × 84,432 items). Required.
- **Tag Genome 2021** — 1,084 curated content tags × 9,734 films. Required.
- **TMDB** — posters, streaming-availability, and overview embeddings via their public API. Optional but recommended. *This product uses the TMDB API but is not endorsed or certified by TMDB.*
- **Letterboxd** — personal ratings + watchlist via the public RSS feed or CSV export.
- **IMDb bulk dumps** — director one-hots for the content scorer. Optional.

## Status

- ✅ Modular `src/` layout, train/val/test splits with overfitting detection
- ✅ Hybrid candidate generation (SVD + ALS fold-in) → re-rank pipeline with 5 modes
- ✅ Group recommendations: 6 strategies (5 traditional + `group_taste_vector` latent-space fusion)
- ✅ Tag Genome + IMDb directors + MiniLM TMDB-overview embeddings as content scorers (weighted)
- ✅ Calibrated diversity (Steck 2018) as an opt-in mode — eval-validated but not the default
- ✅ FastAPI backend + SvelteKit frontend (docker compose)
- ✅ Shareable groups with QR-code join, real-time voting, watchlist overlap
- ✅ At-scale eval harness with bootstrap CIs + catalog coverage + Gini popularity
- ⏳ External-baseline comparison (vs. SLIM / popularity-only / Sam Learner's letterboxd-recommendations)
- ⏳ Second dataset replication (Amazon Movies or anime)
- ⏳ Per-user-cluster eval (cinephile vs mainstream NDCG breakdown)

## Contributing

Issues and PRs welcome — especially around the eval harness (more baselines, second-dataset replication) and the group-aggregation strategies. There's no formal contribution process; open an issue first if you're planning a non-trivial change so we can talk through it.

## License

[MIT](./LICENSE).
