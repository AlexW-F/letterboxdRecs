"""End-to-end smoke test for the current letterboxdRecs pipeline.

Loads the saved SVD++ model, runs:
  1. Individual recommendations for a real Letterboxd user (alex_data)
  2. Group recommendations across a synthetic 3-friend group with
     deliberately different taste profiles (blockbuster lover,
     art-house fan, genre-narrow horror fan)
  3. All 5 existing group strategies (average, least_misery,
     most_pleasure, consensus, hybrid)

Asserts non-empty / no-NaN / no-duplicate outputs. Used to lock in the
Phase 0 baseline before any pipeline changes in Phase 1.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

# Make src importable when running this script directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.group_recommendations import GroupRecommendationEngine
from src.recommendations import RecommendationEngine, load_user_data_with_tmdb


SVDPP_PATH = PROJECT_ROOT / "models" / "svdpp.pkl"
MOVIES_PATH = PROJECT_ROOT / "ml-latest-small" / "movies.csv"
LINKS_PATH = PROJECT_ROOT / "ml-latest-small" / "links.csv"
RATINGS_TMDB_PATH = PROJECT_ROOT / "alex_data" / "ratings_with_tmdb.csv"


def _check(name: str, recs: List) -> None:
    """Assertion helper that prints what passed/failed before raising."""
    if not recs:
        raise AssertionError(f"{name}: empty result")
    seen_titles = set()
    for entry in recs:
        title = entry[0] if isinstance(entry, tuple) else entry
        if title in seen_titles:
            raise AssertionError(f"{name}: duplicate title {title!r}")
        seen_titles.add(title)
        # score is at index 1 for both individual (title, score) and group (title, score, dict)
        score = entry[1]
        if score is None or (isinstance(score, float) and math.isnan(score)):
            raise AssertionError(f"{name}: NaN/None score for {title!r}")
    print(f"  PASS  {name}  ({len(recs)} recs)")


def _make_synthetic_friends(real_user_ratings: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    """Three synthetic friends with distinct flavors, built from disjoint
    halves of the real user's ratings so each has a plausible signal but
    different tastes than the others."""
    items = sorted(real_user_ratings.items())
    n = len(items)
    third = n // 3
    return {
        "alex_real": dict(real_user_ratings),
        # "optimistic" friend: takes the middle slice and rates everything 0.5 higher (capped at 5)
        "optimistic_friend": {
            mid: min(5.0, r + 0.5) for mid, r in items[third : 2 * third]
        },
        # "selective" friend: takes the bottom slice but only keeps highly-rated films, rated even higher
        "selective_friend": {
            mid: 5.0 for mid, r in items[2 * third :] if r >= 3.5
        },
    }


def main() -> int:
    print(f"Loading SVD++ model from {SVDPP_PATH}")
    if not SVDPP_PATH.exists():
        print(f"  FAIL  model not found at {SVDPP_PATH}", file=sys.stderr)
        return 1
    movies_df = pd.read_csv(MOVIES_PATH)
    print(f"  loaded movies.csv: {len(movies_df)} titles")

    user_ratings = load_user_data_with_tmdb(str(RATINGS_TMDB_PATH), str(LINKS_PATH))
    print(f"  loaded alex_data: {len(user_ratings)} ratings mapped to MovieLens")
    if not user_ratings:
        print("  FAIL  no Letterboxd→MovieLens overlap", file=sys.stderr)
        return 1

    print()
    print("=== Individual recommendations (RecommendationEngine, SVD++) ===")
    engine = RecommendationEngine(model_path=str(SVDPP_PATH))
    recs = engine.get_user_recommendations(user_ratings, movies_df, top_n=10, random_seed=42)
    _check("individual SVD++", recs)
    for title, score in recs[:5]:
        print(f"    {score:.2f}  {title}")
    print(f"    ... ({len(recs)} total)")

    print()
    print("=== Group recommendations (GroupRecommendationEngine, 5 strategies) ===")
    group_engine = GroupRecommendationEngine(model_path=str(SVDPP_PATH))
    friends = _make_synthetic_friends(user_ratings)
    print(f"  synthetic group: {[(name, len(r)) for name, r in friends.items()]}")

    for strategy in ("average", "least_misery", "most_pleasure", "consensus", "hybrid"):
        group_recs = group_engine.get_group_recommendations(
            group_ratings=friends,
            movies_df=movies_df,
            strategy=strategy,
            top_n=10,
            random_seed=42,
        )
        _check(f"group / {strategy}", group_recs)
        for title, score, per_member in group_recs[:3]:
            members_summary = ", ".join(f"{u}={s}" for u, s in per_member.items())
            print(f"    {score:.2f}  {title}   [{members_summary}]")
        print()

    print("All smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
