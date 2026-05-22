"""Offline evaluation driver.

Captures the Phase 0 baseline for the existing SVD++ + group pipeline.
Writes a JSON report to evaluation_results/baseline_phase0.json.

Usage:
    python scripts/run_eval.py
    python scripts/run_eval.py --top-n 50 --output evaluation_results/custom.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import (
    GroupEvalReport,
    IndividualEvalReport,
    build_genre_features,
    build_item_popularity,
    evaluate_group,
    evaluate_individual,
)
from src.group_recommendations import GroupRecommendationEngine
from src.recommendations import RecommendationEngine, load_user_data_with_tmdb


SVDPP_PATH = PROJECT_ROOT / "models" / "svdpp.pkl"
RATINGS_PATH = PROJECT_ROOT / "ml-latest-small" / "ratings.csv"
MOVIES_PATH = PROJECT_ROOT / "ml-latest-small" / "movies.csv"
LINKS_PATH = PROJECT_ROOT / "ml-latest-small" / "links.csv"
RATINGS_TMDB_PATH = PROJECT_ROOT / "alex_data" / "ratings_with_tmdb.csv"


def make_individual_recommender(engine: RecommendationEngine, movies_df: pd.DataFrame):
    """Wrap the engine into a (ratings, top_n) -> [(movieId, score)] callable.

    The engine returns titles; we reverse-map to movieIds via movies_df.
    Tolerates dropped titles (the eval harness just sees a shorter list).
    """
    title_to_id: Dict[str, str] = {}
    for mid, title in zip(movies_df["movieId"].astype(str), movies_df["title"]):
        title_to_id.setdefault(title, mid)

    def recommender(user_ratings: Dict[str, float], top_n: int) -> List[Tuple[str, float]]:
        recs = engine.get_user_recommendations(user_ratings, movies_df, top_n=top_n)
        out: List[Tuple[str, float]] = []
        for title, score in recs:
            mid = title_to_id.get(title)
            if mid is not None:
                out.append((mid, float(score)))
        return out

    return recommender


def make_group_recommender(engine: GroupRecommendationEngine, movies_df: pd.DataFrame, strategy: str):
    title_to_id: Dict[str, str] = {}
    for mid, title in zip(movies_df["movieId"].astype(str), movies_df["title"]):
        title_to_id.setdefault(title, mid)

    def recommender(members_ratings: List[Dict[str, float]], top_n: int) -> List[Tuple[str, float]]:
        group_dict = {f"m{i}": r for i, r in enumerate(members_ratings)}
        recs = engine.get_group_recommendations(
            group_ratings=group_dict,
            movies_df=movies_df,
            strategy=strategy,
            top_n=top_n,
            random_seed=42,
        )
        out: List[Tuple[str, float]] = []
        for title, score, _per_member in recs:
            mid = title_to_id.get(title)
            if mid is not None:
                out.append((mid, float(score)))
        return out

    return recommender


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "evaluation_results" / "baseline_phase0.json",
    )
    parser.add_argument("--holdout-frac", type=float, default=0.2)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading SVD++ model: {SVDPP_PATH}")
    engine = RecommendationEngine(model_path=str(SVDPP_PATH))
    group_engine = GroupRecommendationEngine(model_path=str(SVDPP_PATH))
    movies_df = pd.read_csv(MOVIES_PATH)
    ratings_df = pd.read_csv(RATINGS_PATH)

    item_popularity = build_item_popularity(ratings_df)
    item_features = build_genre_features(movies_df)
    catalog_size = movies_df["movieId"].nunique()

    real_user = load_user_data_with_tmdb(str(RATINGS_TMDB_PATH), str(LINKS_PATH))
    print(f"  real user has {len(real_user)} ratings mapped to MovieLens")

    # Build a small set of synthetic users from MovieLens itself: pick a
    # handful of real users with >=50 ratings to evaluate against. This
    # gives the eval more statistical weight than a single user.
    ratings_df["userId"] = ratings_df["userId"].astype(int)
    ratings_df["movieId"] = ratings_df["movieId"].astype(str)
    user_counts = ratings_df.groupby("userId").size()
    sampled_user_ids = (
        user_counts[user_counts >= 50].sample(n=20, random_state=42).index.tolist()
    )
    eval_users: Dict[str, Dict[str, float]] = {"alex_real": real_user}
    for uid in sampled_user_ids:
        u_ratings = ratings_df[ratings_df["userId"] == uid].set_index("movieId")["rating"].to_dict()
        eval_users[f"ml_user_{uid}"] = {str(k): float(v) for k, v in u_ratings.items()}
    print(f"  evaluating {len(eval_users)} users in total")

    # -- Individual evaluation -------------------------------------------------
    print("\nEvaluating individual SVD++ recommendations...")
    rec_fn = make_individual_recommender(engine, movies_df)
    indiv_report = evaluate_individual(
        rec_fn,
        eval_users,
        item_features=item_features,
        item_popularity=item_popularity,
        catalog_size=catalog_size,
        holdout_frac=args.holdout_frac,
        top_n=args.top_n,
    )
    print(f"  NDCG@10={indiv_report.ndcg_at_10:.4f}  Recall@50={indiv_report.recall_at_50:.4f}"
          f"  coverage@50={indiv_report.catalog_coverage_at_50:.4f}"
          f"  gini={indiv_report.gini_popularity_at_50:.4f}"
          f"  diversity@10={indiv_report.intra_list_diversity_at_10:.4f}")

    # -- Group evaluation ------------------------------------------------------
    print("\nEvaluating group strategies (3 ml users + alex_real)...")
    # Pick 3 users for the group (one of them is alex_real, the others are
    # MovieLens users with deliberately different rating-count profiles).
    group_members: List[Dict[str, float]] = [
        real_user,
        eval_users[f"ml_user_{sampled_user_ids[0]}"],
        eval_users[f"ml_user_{sampled_user_ids[1]}"],
    ]

    group_reports: Dict[str, GroupEvalReport] = {}
    for strategy in ("average", "least_misery", "most_pleasure", "consensus", "hybrid"):
        group_fn = make_group_recommender(group_engine, movies_df, strategy)
        rpt = evaluate_group(
            group_fn,
            group_members,
            strategy=strategy,
            item_features=item_features,
            item_popularity=item_popularity,
            catalog_size=catalog_size,
            holdout_frac=args.holdout_frac,
            top_n=args.top_n,
        )
        group_reports[strategy] = rpt
        print(f"  {strategy:14s}  NDCG@10={rpt.avg_per_member_ndcg_at_10:.4f}"
              f"  fairness_cv={rpt.fairness_cv_at_10:.4f}"
              f"  recall@50={rpt.avg_per_member_recall_at_50:.4f}"
              f"  div@10={rpt.intra_list_diversity_at_10:.4f}")

    out = {
        "phase": "0_baseline",
        "model": "svdpp.pkl (ml-latest-small)",
        "dataset": "ml-latest-small",
        "individual": indiv_report.to_dict(),
        "group": {k: r.to_dict() for k, r in group_reports.items()},
        "config": {
            "top_n": args.top_n,
            "holdout_frac": args.holdout_frac,
            "n_individual_users": len(eval_users),
            "n_group_members": len(group_members),
        },
    }
    args.output.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
