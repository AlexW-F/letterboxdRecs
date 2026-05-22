"""One-shot builder for the Phase 2 content-feature matrix.

Reads ml-32m's movies.csv and tags.csv (2M tags), constructs a per-movie
tag+genre document, fits TF-IDF, and writes:

    data/content_features.npz   (sparse tfidf matrix)
    data/content_features.json  (movie_ids array + vocabulary list)

The script is idempotent — rerun any time tags/movies are updated.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.content_features import ContentFeatures


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--movies", type=Path,
                   default=PROJECT_ROOT / "ml-32m" / "movies.csv")
    p.add_argument("--tags", type=Path,
                   default=PROJECT_ROOT / "ml-32m" / "tags.csv")
    p.add_argument("--output", type=Path,
                   default=PROJECT_ROOT / "data" / "content_features")
    p.add_argument("--max-features", type=int, default=20000)
    p.add_argument("--min-df", type=int, default=2)
    args = p.parse_args()

    if not args.movies.exists():
        print(f"movies.csv not found at {args.movies}", file=sys.stderr)
        return 1

    print(f"Loading {args.movies}...")
    movies_df = pd.read_csv(args.movies)
    print(f"  {len(movies_df):,} movies")

    tags_df = None
    if args.tags.exists():
        print(f"Loading {args.tags}...")
        t0 = time.time()
        tags_df = pd.read_csv(args.tags, usecols=["movieId", "tag"])
        print(f"  {len(tags_df):,} tag applications loaded in {time.time()-t0:.1f}s")
    else:
        print(f"  no tags.csv at {args.tags} — using genres only")

    print(f"Building TF-IDF (max_features={args.max_features}, min_df={args.min_df})...")
    t0 = time.time()
    cf = ContentFeatures.from_dataframes(
        movies_df, tags_df,
        max_features=args.max_features,
        min_df=args.min_df,
    )
    elapsed = time.time() - t0
    print(f"  matrix shape: {cf.tfidf.shape}  nnz: {cf.tfidf.nnz:,}  "
          f"vocab: {len(cf.vocabulary):,}  ({elapsed:.1f}s)")

    cf.save(args.output)
    npz_size = args.output.with_suffix(".npz").stat().st_size / 1024 / 1024
    json_size = args.output.with_suffix(".json").stat().st_size / 1024 / 1024
    print(f"  wrote {args.output.with_suffix('.npz')}  ({npz_size:.1f} MB)")
    print(f"  wrote {args.output.with_suffix('.json')}  ({json_size:.1f} MB)")

    # Sample for verification
    print("\nSample tokens (first 20 vocabulary entries):")
    print("  ", ", ".join(cf.vocabulary[:20]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
