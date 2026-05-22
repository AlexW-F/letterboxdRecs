"""Offline evaluation driver.

Phase 0: baseline (SVD++ top-N, GroupRecommendationEngine).
Phase 1: reranker (Reranker per mode, GroupReranker per strategy x mode).

Default behavior runs everything and writes a single JSON report.

Usage:
    python scripts/run_eval.py
    python scripts/run_eval.py --skip-reranker
    python scripts/run_eval.py --output evaluation_results/custom.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import (
    build_genre_features,
    build_item_popularity,
    evaluate_group,
    evaluate_individual,
)
from src.group_reranker import GroupReranker, GROUP_STRATEGIES
from src.group_recommendations import GroupRecommendationEngine
from src.recommendations import RecommendationEngine, load_user_data_with_tmdb
from src.reranking import (
    ALSScorer,
    MODE_WEIGHTS,
    PopularityModel,
    Reranker,
    SVDScorer,
)


SVDPP_PATH = PROJECT_ROOT / "models" / "svdpp.pkl"
ALS_PATH = PROJECT_ROOT / "models" / "als_small.pkl"
RATINGS_PATH = PROJECT_ROOT / "ml-latest-small" / "ratings.csv"
MOVIES_PATH = PROJECT_ROOT / "ml-latest-small" / "movies.csv"
LINKS_PATH = PROJECT_ROOT / "ml-latest-small" / "links.csv"
RATINGS_TMDB_PATH = PROJECT_ROOT / "alex_data" / "ratings_with_tmdb.csv"


def load_eval_users(
    real_user: Dict[str, float],
    ratings_df: pd.DataFrame,
    n_synthetic: int = 20,
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    ratings_df = ratings_df.copy()
    ratings_df["userId"] = ratings_df["userId"].astype(int)
    ratings_df["movieId"] = ratings_df["movieId"].astype(str)
    counts = ratings_df.groupby("userId").size()
    sampled = counts[counts >= 50].sample(n=n_synthetic, random_state=seed).index.tolist()
    out: Dict[str, Dict[str, float]] = {"alex_real": real_user}
    for uid in sampled:
        r = (
            ratings_df[ratings_df["userId"] == uid]
            .set_index("movieId")["rating"]
            .to_dict()
        )
        out[f"ml_user_{uid}"] = {str(k): float(v) for k, v in r.items()}
    return out


# -- Baseline wrappers (Phase 0) -------------------------------------------------


def baseline_individual_fn(engine: RecommendationEngine, movies_df: pd.DataFrame, title_to_id):
    def fn(user_ratings, top_n):
        recs = engine.get_user_recommendations(user_ratings, movies_df, top_n=top_n)
        out: List[Tuple[str, float]] = []
        for title, score in recs:
            mid = title_to_id.get(title)
            if mid is not None:
                out.append((mid, float(score)))
        return out
    return fn


def baseline_group_fn(engine: GroupRecommendationEngine, movies_df: pd.DataFrame, title_to_id, strategy: str):
    def fn(members_ratings, top_n):
        group = {f"m{i}": r for i, r in enumerate(members_ratings)}
        recs = engine.get_group_recommendations(group, movies_df, strategy=strategy, top_n=top_n)
        out: List[Tuple[str, float]] = []
        for entry in recs:
            title, score, _ = entry
            mid = title_to_id.get(title)
            if mid is not None:
                out.append((mid, float(score)))
        return out
    return fn


# -- Reranker wrappers (Phase 1) -------------------------------------------------


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


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--top-n", type=int, default=50)
    p.add_argument("--holdout-frac", type=float, default=0.2)
    p.add_argument("--output", type=Path,
                   default=PROJECT_ROOT / "evaluation_results" / "phase1_eval.json")
    p.add_argument("--skip-baseline", action="store_true")
    p.add_argument("--skip-reranker", action="store_true")
    args = p.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    print("Loading models...")
    svd_scorer = SVDScorer.from_path(SVDPP_PATH)
    als_scorer = ALSScorer.from_path(ALS_PATH)
    movies_df = pd.read_csv(MOVIES_PATH)
    ratings_df = pd.read_csv(RATINGS_PATH)
    popularity = PopularityModel(ratings_df)
    genre_features = build_genre_features(movies_df)
    catalog_size = movies_df["movieId"].nunique()

    title_to_id: Dict[str, str] = {}
    for mid, title in zip(movies_df["movieId"].astype(str), movies_df["title"]):
        title_to_id.setdefault(title, mid)

    real_user = load_user_data_with_tmdb(str(RATINGS_TMDB_PATH), str(LINKS_PATH))
    print(f"  real user has {len(real_user)} ratings mapped")

    eval_users = load_eval_users(real_user, ratings_df, n_synthetic=20)
    print(f"  evaluating {len(eval_users)} users")

    item_popularity = build_item_popularity(ratings_df)

    # Build the group: real user + 2 distinct ml users
    ml_user_keys = [k for k in eval_users if k.startswith("ml_user_")][:2]
    group_members = [real_user] + [eval_users[k] for k in ml_user_keys]

    out: Dict = {
        "phase": "1_reranker",
        "model": {"svd": str(SVDPP_PATH.name), "als": str(ALS_PATH.name)},
        "dataset": "ml-latest-small",
        "config": {
            "top_n": args.top_n,
            "holdout_frac": args.holdout_frac,
            "n_individual_users": len(eval_users),
            "n_group_members": len(group_members),
        },
    }

    # ---- Baseline (Phase 0) ----
    if not args.skip_baseline:
        print("\n--- Baseline: SVD++ top-N, no reranker ---")
        engine = RecommendationEngine(model_path=str(SVDPP_PATH))
        group_engine = GroupRecommendationEngine(model_path=str(SVDPP_PATH))

        b_indiv = evaluate_individual(
            baseline_individual_fn(engine, movies_df, title_to_id),
            eval_users,
            item_features=genre_features,
            item_popularity=item_popularity,
            catalog_size=catalog_size,
            holdout_frac=args.holdout_frac,
            top_n=args.top_n,
        )
        print(f"  individual: NDCG@10={b_indiv.ndcg_at_10:.4f}  recall@50={b_indiv.recall_at_50:.4f}"
              f"  coverage={b_indiv.catalog_coverage_at_50:.4f}  gini={b_indiv.gini_popularity_at_50:.4f}"
              f"  diversity={b_indiv.intra_list_diversity_at_10:.4f}")

        baseline_group: Dict[str, Dict] = {}
        for strategy in ("average", "least_misery", "most_pleasure", "consensus", "hybrid"):
            rpt = evaluate_group(
                baseline_group_fn(group_engine, movies_df, title_to_id, strategy),
                group_members,
                strategy=strategy,
                item_features=genre_features,
                item_popularity=item_popularity,
                catalog_size=catalog_size,
                holdout_frac=args.holdout_frac,
                top_n=args.top_n,
            )
            baseline_group[strategy] = rpt.to_dict()
            print(f"  group/{strategy:14s} NDCG@10={rpt.avg_per_member_ndcg_at_10:.4f}"
                  f"  recall@50={rpt.avg_per_member_recall_at_50:.4f}"
                  f"  fair_cv={rpt.fairness_cv_at_10:.4f}"
                  f"  div={rpt.intra_list_diversity_at_10:.4f}")
        out["baseline"] = {"individual": b_indiv.to_dict(), "group": baseline_group}

    # ---- Reranker (Phase 1) ----
    if not args.skip_reranker:
        print("\n--- Reranker: SVD++ + ALS + popularity + diversity ---")
        rr = Reranker(svd_scorer, als_scorer, popularity, movies_df, genre_features)
        gr = GroupReranker(rr)

        rerank_indiv: Dict[str, Dict] = {}
        for mode in MODE_WEIGHTS.keys():
            rpt = evaluate_individual(
                reranker_individual_fn(rr, mode),
                eval_users,
                item_features=genre_features,
                item_popularity=item_popularity,
                catalog_size=catalog_size,
                holdout_frac=args.holdout_frac,
                top_n=args.top_n,
            )
            rerank_indiv[mode] = rpt.to_dict()
            print(f"  individual/{mode:14s}  NDCG@10={rpt.ndcg_at_10:.4f}"
                  f"  recall@50={rpt.recall_at_50:.4f}"
                  f"  coverage={rpt.catalog_coverage_at_50:.4f}"
                  f"  gini={rpt.gini_popularity_at_50:.4f}"
                  f"  diversity={rpt.intra_list_diversity_at_10:.4f}")

        rerank_group: Dict[str, Dict[str, Dict]] = {}
        for strategy in GROUP_STRATEGIES:
            for mode in ("balanced", "niche"):  # measure two modes to compare
                rpt = evaluate_group(
                    reranker_group_fn(gr, strategy, mode),
                    group_members,
                    strategy=strategy,
                    item_features=genre_features,
                    item_popularity=item_popularity,
                    catalog_size=catalog_size,
                    holdout_frac=args.holdout_frac,
                    top_n=args.top_n,
                )
                rerank_group.setdefault(strategy, {})[mode] = rpt.to_dict()
                print(f"  group/{strategy:20s}/{mode:10s}  NDCG@10={rpt.avg_per_member_ndcg_at_10:.4f}"
                      f"  recall@50={rpt.avg_per_member_recall_at_50:.4f}"
                      f"  fair_cv={rpt.fairness_cv_at_10:.4f}"
                      f"  div={rpt.intra_list_diversity_at_10:.4f}")
        out["reranker"] = {"individual": rerank_indiv, "group": rerank_group}

    args.output.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
