"""Phase 3 eval: scale up to N≥200 users / M≥50 groups with bootstrap CIs.

The Phase 1/2 ``run_eval.py`` used ~20 users → metrics had no statistical
power to distinguish modes/strategies. This driver:

- Samples ``--n-users`` MovieLens users with ≥ ``--min-ratings`` ratings.
- For groups: samples ``--n-groups`` synthetic groups of ``--group-size``
  members drawn from the same user pool.
- Runs the chosen modes × strategies through the reranker / group reranker.
- Computes 95% bootstrap CIs on every metric so callers can tell whether
  observed differences are signal or noise.
- Writes a structured JSON + a markdown summary report.

Defaults are deliberately conservative on runtime (~10 min on the full
ml-32m models). Bump ``--n-users`` / ``--n-groups`` for tighter CIs.

Usage:
    python scripts/run_eval_at_scale.py
    python scripts/run_eval_at_scale.py --n-users 400 --n-groups 100
    python scripts/run_eval_at_scale.py --modes balanced,niche --strategies average,group_taste_vector
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.content_features import ContentFeatures
from src.evaluation import (
    bootstrap_ci,
    bootstrap_set_metric,
    build_genre_features,
    catalog_coverage,
    evaluate_group_per_group,
    evaluate_individual_per_user,
    gini_popularity,
)
from src.group_reranker import GroupReranker, GROUP_STRATEGIES
from src.reranking import (
    ALSScorer,
    MODE_WEIGHTS,
    PopularityModel,
    Reranker,
    SVDScorer,
)


def build_popularity_streaming(ratings_path: Path, chunksize: int = 1_000_000) -> PopularityModel:
    """Stream-construct PopularityModel without loading full ratings.csv."""
    from collections import Counter
    counts: Counter = Counter()
    for chunk in pd.read_csv(ratings_path, usecols=["movieId"], chunksize=chunksize):
        counts.update(chunk["movieId"].astype(str).tolist())
    # Build an empty PopularityModel and inject counts (cheaper than synthesizing a DataFrame)
    pm = PopularityModel.__new__(PopularityModel)
    pm.counts = {str(k): int(v) for k, v in counts.items()}
    pm.max_count = max(pm.counts.values()) if pm.counts else 1
    return pm


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SVDPP_PATH_DEFAULT = PROJECT_ROOT / "models" / "svd_full_slim.pkl"
ALS_PATH_DEFAULT = PROJECT_ROOT / "models" / "als_full.pkl"
RATINGS_PATH = PROJECT_ROOT / "ml-32m" / "ratings.csv"
MOVIES_PATH = PROJECT_ROOT / "ml-32m" / "movies.csv"


# ---------------------------------------------------------------------------
# User / group sampling
# ---------------------------------------------------------------------------

def sample_users_streaming(
    ratings_path: Path, n_users: int, min_ratings: int = 50, seed: int = 42,
    chunksize: int = 1_000_000,
) -> Dict[str, Dict[str, float]]:
    """Two-pass streaming sampler for ml-32m's 32M-row ratings file.

    Pass 1: count ratings per userId via chunked read (~int dict, ~1MB).
    Pass 2: pick userIds with ≥ min_ratings, then re-stream and keep only
    matching rows.

    Avoids materializing the full 32M-row DataFrame in memory (~3GB).
    """
    # Pass 1: counts per userId
    from collections import Counter
    counts: Counter = Counter()
    for chunk in pd.read_csv(ratings_path, usecols=["userId"], chunksize=chunksize):
        counts.update(chunk["userId"].astype(int).tolist())

    eligible = [uid for uid, n in counts.items() if n >= min_ratings]
    rng = random.Random(seed)
    picked = set(rng.sample(eligible, min(n_users, len(eligible))))

    # Pass 2: stream again and collect rows for picked users
    out: Dict[str, Dict[str, float]] = {f"ml_user_{u}": {} for u in picked}
    for chunk in pd.read_csv(
        ratings_path, usecols=["userId", "movieId", "rating"], chunksize=chunksize,
    ):
        chunk = chunk[chunk["userId"].isin(picked)]
        if chunk.empty:
            continue
        for uid, mid, rating in zip(
            chunk["userId"].astype(int),
            chunk["movieId"].astype(str),
            chunk["rating"].astype(float),
        ):
            out[f"ml_user_{uid}"][mid] = rating
    return out


def sample_groups(
    users: Dict[str, Dict[str, float]],
    n_groups: int,
    group_size: int = 3,
    seed: int = 42,
) -> List[List[Dict[str, float]]]:
    keys = list(users.keys())
    rng = random.Random(seed)
    out: List[List[Dict[str, float]]] = []
    for _ in range(n_groups):
        members = rng.sample(keys, group_size)
        out.append([users[k] for k in members])
    return out


# ---------------------------------------------------------------------------
# Recommender wrappers
# ---------------------------------------------------------------------------

def reranker_individual_fn(rr: Reranker, mode: str):
    def fn(user_ratings, top_n):
        recs = rr.recommend(user_ratings, mode=mode, top_n=top_n)
        return [(c.movie_id, c.score) for c in recs]
    return fn


def reranker_group_fn(gr: GroupReranker, strategy: str, mode: str):
    def fn(members_ratings, top_n):
        members = [(f"m{i}", r, None) for i, r in enumerate(members_ratings)]
        recs = gr.recommend(members, strategy=strategy, mode=mode, top_n=top_n)
        return [(c.movie_id, c.score) for c in recs]
    return fn


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def fmt_ci(ci: Dict[str, float]) -> str:
    return f"{ci['mean']:.4f}  [{ci['lower']:.4f}, {ci['upper']:.4f}]  n={ci['n']}"


def fmt_set_ci(ci: Dict[str, float]) -> str:
    """bootstrap_set_metric uses 'point' (no mean since it's not averageable)."""
    return f"{ci['point']:.4f}  [{ci['lower']:.4f}, {ci['upper']:.4f}]"


def render_markdown_report(payload: Dict) -> str:
    lines: List[str] = []
    lines.append(f"# Phase 3 at-scale evaluation\n")
    lines.append(f"- Dataset: `{payload['dataset']}`")
    lines.append(f"- Model: SVD `{payload['model']['svd']}` + ALS `{payload['model']['als']}`")
    lines.append(f"- Users sampled: **{payload['config']['n_users']}** "
                 f"(min {payload['config']['min_ratings']} ratings each)")
    lines.append(f"- Groups sampled: **{payload['config']['n_groups']}** × "
                 f"{payload['config']['group_size']} members")
    lines.append(f"- Bootstrap: {payload['config']['bootstrap_n']} resamples · 95% percentile CI")
    lines.append(f"- Runtime: {payload['runtime_seconds']:.1f}s\n")

    # --- Individual table ---
    lines.append("## Individual recs — by mode\n")
    lines.append("| mode | NDCG@10 | Recall@50 | List diversity@10 | Catalog coverage@50 | Gini popularity@50 |")
    lines.append("|---|---|---|---|---|---|")
    for mode, m in payload["individual"].items():
        lines.append(
            f"| `{mode}` | {fmt_ci(m['ndcg_at_10'])} | "
            f"{fmt_ci(m['recall_at_50'])} | "
            f"{fmt_ci(m['intra_list_diversity_at_10'])} | "
            f"{fmt_set_ci(m['catalog_coverage_at_50'])} | "
            f"{fmt_set_ci(m['gini_popularity_at_50'])} |"
        )
    lines.append("")

    # --- Group table ---
    lines.append("## Group recs — strategy × mode\n")
    lines.append("| strategy | mode | avg per-member NDCG@10 | fairness CV | list div | catalog cov | Gini pop |")
    lines.append("|---|---|---|---|---|---|---|")
    for strategy, by_mode in payload["group"].items():
        for mode, m in by_mode.items():
            lines.append(
                f"| `{strategy}` | `{mode}` | "
                f"{fmt_ci(m['avg_per_member_ndcg_at_10'])} | "
                f"{fmt_ci(m['fairness_cv_at_10'])} | "
                f"{fmt_ci(m['intra_list_diversity_at_10'])} | "
                f"{fmt_set_ci(m['catalog_coverage_at_50'])} | "
                f"{fmt_set_ci(m['gini_popularity_at_50'])} |"
            )
    lines.append("")

    # --- Reading guide ---
    lines.append("## How to read this\n")
    lines.append(
        "- **NDCG@10**: ranking quality on held-out high-rated items (higher = better).\n"
        "- **Recall@50**: fraction of relevant holdout items appearing in the top-50.\n"
        "- **Fairness CV**: coefficient of variation of per-member NDCG@10 within a group "
        "(lower = more equitable across members).\n"
        "- **List diversity@10**: avg pairwise genre dissimilarity inside one top-10 (higher = more varied).\n"
        "- **Catalog coverage@50**: fraction of the full catalog (~88k films) that surfaces "
        "across all users'/groups' top-50 — high coverage means the system explores; low means "
        "it recycles the same shortlist.\n"
        "- **Gini popularity@50**: inequality of popularity within the recommended pool. "
        "0 = uniform across all titles, 1 = all recs concentrated on one mega-hit. High Gini = "
        "the system is funneling toward blockbusters.\n"
        "- **CI overlap**: if two rows' CIs overlap, the difference between them is NOT statistically "
        "meaningful at α=0.05.\n"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n-users", type=int, default=200,
                   help="Individual-eval user pool size (default 200)")
    p.add_argument("--n-groups", type=int, default=50,
                   help="Synthetic groups to evaluate (default 50)")
    p.add_argument("--group-size", type=int, default=3,
                   help="Members per synthetic group (default 3)")
    p.add_argument("--min-ratings", type=int, default=50,
                   help="Drop users with fewer than this many ratings")
    p.add_argument("--top-n", type=int, default=50)
    p.add_argument("--holdout-frac", type=float, default=0.2)
    p.add_argument("--bootstrap-n", type=int, default=1000)
    p.add_argument("--modes", type=str, default="balanced,niche,popular,serendipitous",
                   help="Comma-separated modes to evaluate")
    p.add_argument("--strategies", type=str, default=",".join(GROUP_STRATEGIES),
                   help="Comma-separated group strategies")
    p.add_argument("--group-modes", type=str, default="balanced,niche",
                   help="Modes to evaluate group strategies under")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path,
                   default=PROJECT_ROOT / "evaluation_results" / "phase3_at_scale.json")
    p.add_argument("--report", type=Path,
                   default=PROJECT_ROOT / "evaluation_results" / "phase3_at_scale.md")
    p.add_argument("--content", type=Path, default=None)
    p.add_argument("--content-extra", type=Path, action="append", default=[])
    p.add_argument("--svd-path", type=Path, default=SVDPP_PATH_DEFAULT,
                   help="Path to SVD pickle (defaults to slim variant for memory)")
    p.add_argument("--als-path", type=Path, default=ALS_PATH_DEFAULT)
    args = p.parse_args()
    svd_path: Path = args.svd_path
    als_path: Path = args.als_path

    args.output.parent.mkdir(parents=True, exist_ok=True)

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    group_modes = [m.strip() for m in args.group_modes.split(",") if m.strip()]

    # Sanity-check requested modes/strategies
    bad_modes = [m for m in modes + group_modes if m not in MODE_WEIGHTS]
    if bad_modes:
        print(f"ERROR: unknown modes {bad_modes}; choose from {list(MODE_WEIGHTS)}", file=sys.stderr)
        return 2
    bad_strategies = [s for s in strategies if s not in GROUP_STRATEGIES]
    if bad_strategies:
        print(f"ERROR: unknown strategies {bad_strategies}; choose from {GROUP_STRATEGIES}", file=sys.stderr)
        return 2

    t0 = time.time()
    print(f"Loading models from {svd_path.name}, {als_path.name}…")
    svd_scorer = SVDScorer.from_path(svd_path)
    als_scorer = ALSScorer.from_path(als_path)
    movies_df = pd.read_csv(MOVIES_PATH)
    print(f"Streaming {RATINGS_PATH.name} to build PopularityModel…")
    popularity = build_popularity_streaming(RATINGS_PATH)
    print(f"  popularity: {len(popularity.counts):,} unique movies")
    genre_features = build_genre_features(movies_df)

    content_path = args.content or (PROJECT_ROOT / "data" / "content_genome")
    content_features = (
        ContentFeatures.load(content_path) if content_path.with_suffix(".npz").exists() else None
    )
    print(f"  content_genome loaded: {content_features is not None}")
    extra_content: list[ContentFeatures] = []
    for pe in args.content_extra:
        if pe.with_suffix(".npz").exists():
            extra_content.append(ContentFeatures.load(pe))
            print(f"  + extra content from {pe.name}")

    rr = Reranker(svd_scorer, als_scorer, popularity, movies_df, genre_features,
                  content_features=content_features, content_features_extra=extra_content)
    gr = GroupReranker(rr)

    # ---- Sample users (streaming) ----
    print(f"\nSampling {args.n_users} users (min {args.min_ratings} ratings)…")
    users = sample_users_streaming(RATINGS_PATH, args.n_users, args.min_ratings, seed=args.seed)
    print(f"  → {len(users)} users sampled")

    # ---- Sample groups ----
    print(f"Sampling {args.n_groups} groups of {args.group_size}…")
    groups = sample_groups(users, args.n_groups, args.group_size, seed=args.seed)

    # Streaming item-popularity over ratings (for Gini). Same trick as PopularityModel.
    print(f"Building item-popularity index for Gini…")
    item_pop_pm = build_popularity_streaming(RATINGS_PATH)
    item_popularity = item_pop_pm.counts
    catalog_size = movies_df["movieId"].nunique()

    def _coverage_fn(lists: List[List[str]]) -> float:
        return catalog_coverage(lists, catalog_size)

    def _gini_fn(lists: List[List[str]]) -> float:
        return gini_popularity(lists, item_popularity)

    # ---- Individual eval ----
    print(f"\n=== Individual eval over {len(users)} users × {len(modes)} modes ===")
    individual: Dict[str, Dict[str, Dict[str, float]]] = {}
    for mode in modes:
        t_mode = time.time()
        raw, top_lists = evaluate_individual_per_user(
            reranker_individual_fn(rr, mode), users,
            item_features=genre_features, holdout_frac=args.holdout_frac, top_n=args.top_n,
        )
        per_metric_ci = {k: bootstrap_ci(v, n_resamples=args.bootstrap_n, seed=args.seed)
                        for k, v in raw.items()}
        per_metric_ci["catalog_coverage_at_50"] = bootstrap_set_metric(
            top_lists, _coverage_fn, n_resamples=args.bootstrap_n, seed=args.seed,
        )
        per_metric_ci["gini_popularity_at_50"] = bootstrap_set_metric(
            top_lists, _gini_fn, n_resamples=args.bootstrap_n, seed=args.seed,
        )
        individual[mode] = per_metric_ci
        ndcg = per_metric_ci["ndcg_at_10"]
        cov = per_metric_ci["catalog_coverage_at_50"]
        gini = per_metric_ci["gini_popularity_at_50"]
        print(f"  {mode:14s}  NDCG@10 {ndcg['mean']:.4f} [{ndcg['lower']:.4f}, {ndcg['upper']:.4f}]"
              f"  cov {cov['point']:.4f}  gini {gini['point']:.3f}"
              f"  n={ndcg['n']}  ({time.time()-t_mode:.1f}s)")

    # ---- Group eval ----
    print(f"\n=== Group eval over {len(groups)} groups × "
          f"{len(strategies)} strategies × {len(group_modes)} modes ===")
    group_out: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    for strategy in strategies:
        group_out[strategy] = {}
        for mode in group_modes:
            t_cell = time.time()
            raw, top_lists = evaluate_group_per_group(
                reranker_group_fn(gr, strategy, mode), groups,
                item_features=genre_features, holdout_frac=args.holdout_frac, top_n=args.top_n,
            )
            per_metric_ci = {k: bootstrap_ci(v, n_resamples=args.bootstrap_n, seed=args.seed)
                            for k, v in raw.items()}
            per_metric_ci["catalog_coverage_at_50"] = bootstrap_set_metric(
                top_lists, _coverage_fn, n_resamples=args.bootstrap_n, seed=args.seed,
            )
            per_metric_ci["gini_popularity_at_50"] = bootstrap_set_metric(
                top_lists, _gini_fn, n_resamples=args.bootstrap_n, seed=args.seed,
            )
            group_out[strategy][mode] = per_metric_ci
            ndcg = per_metric_ci["avg_per_member_ndcg_at_10"]
            fair = per_metric_ci["fairness_cv_at_10"]
            cov = per_metric_ci["catalog_coverage_at_50"]
            gini = per_metric_ci["gini_popularity_at_50"]
            print(f"  {strategy:20s}/{mode:14s}  "
                  f"NDCG {ndcg['mean']:.4f} [{ndcg['lower']:.4f}, {ndcg['upper']:.4f}]  "
                  f"fairCV {fair['mean']:.3f}  cov {cov['point']:.4f}  gini {gini['point']:.3f}  "
                  f"({time.time()-t_cell:.1f}s)")

    runtime = time.time() - t0
    payload = {
        "phase": "3_at_scale",
        "dataset": RATINGS_PATH.parent.name,
        "model": {"svd": svd_path.name, "als": als_path.name},
        "config": {
            "n_users": len(users),
            "n_groups": len(groups),
            "group_size": args.group_size,
            "min_ratings": args.min_ratings,
            "top_n": args.top_n,
            "holdout_frac": args.holdout_frac,
            "bootstrap_n": args.bootstrap_n,
            "modes": modes,
            "strategies": strategies,
            "group_modes": group_modes,
            "seed": args.seed,
        },
        "individual": individual,
        "group": group_out,
        "runtime_seconds": runtime,
    }
    args.output.write_text(json.dumps(payload, indent=2))
    args.report.write_text(render_markdown_report(payload))
    print(f"\nWrote:")
    print(f"  {args.output}")
    print(f"  {args.report}")
    print(f"  total runtime: {runtime/60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
