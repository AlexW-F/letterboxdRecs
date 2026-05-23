"""Pre-compute the UMAP background for the per-user 3D viz.

Run once after training. Saves the fitted UMAP reducer + the projected
background coordinates to ``data/movie_space_index.pkl``. The API
deserializes this once at startup; per-user requests just call
``project_user()`` which is ~10ms.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import build_genre_features  # noqa: F401  (matches import order)
from src.reranking import ALSScorer, PopularityModel
from src.viz import MovieSpaceIndex


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--als", type=Path, default=PROJECT_ROOT / "models" / "als_full.pkl")
    p.add_argument("--movies", type=Path, default=PROJECT_ROOT / "ml-32m" / "movies.csv")
    p.add_argument("--ratings", type=Path, default=PROJECT_ROOT / "ml-32m" / "ratings.csv")
    p.add_argument("--output", type=Path,
                   default=PROJECT_ROOT / "data" / "movie_space_index.pkl")
    p.add_argument("--max-points", type=int, default=8000)
    p.add_argument("--min-rating-count", type=int, default=100)
    args = p.parse_args()

    print(f"Loading ALS from {args.als}...")
    als = ALSScorer.from_path(args.als)
    print(f"  item factors: {als.model.item_factors.shape}")

    print(f"Loading {args.movies}...")
    movies_df = pd.read_csv(args.movies)

    print(f"Loading {args.ratings} (popularity counts)...")
    ratings_df = pd.read_csv(args.ratings, usecols=["movieId"])
    pop = PopularityModel(ratings_df)

    t0 = time.time()
    print(f"Fitting UMAP background (max_points={args.max_points}, "
          f"min_rating_count={args.min_rating_count})...")
    index = MovieSpaceIndex.build(
        als, movies_df, pop,
        max_points=args.max_points,
        min_rating_count=args.min_rating_count,
    )
    print(f"  background ready in {time.time()-t0:.1f}s; "
          f"{index.background_coords.shape[0]} projected films")

    index.save(args.output)
    size_mb = args.output.stat().st_size / 1024 / 1024
    print(f"  wrote {args.output} ({size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
