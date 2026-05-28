#!/usr/bin/env python3
"""Precompute per-movie rating counts.

The API uses these counts for the popularity-debias term + tier labels.
Building them from the full ~32M-row ``ratings.csv`` at every startup
dominates both memory and cold-start time, so precompute once to a small
JSON the server can load instantly (``state._load_popularity``).

Usage:
    python scripts/build_popularity.py \
        --ratings ml-32m/ratings.csv \
        --out data/popularity.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ratings", default="ml-32m/ratings.csv",
                    help="MovieLens ratings.csv (needs a movieId column)")
    ap.add_argument("--out", default="data/popularity.json",
                    help="Where to write the {movieId: count} JSON")
    args = ap.parse_args()

    ratings_path = Path(args.ratings)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"reading {ratings_path} ...")
    df = pd.read_csv(ratings_path, usecols=["movieId"])
    counts = {str(k): int(v) for k, v in df["movieId"].astype(str).value_counts().items()}

    with open(out_path, "w") as f:
        json.dump(counts, f)
    print(f"wrote {len(counts)} movie counts -> {out_path} "
          f"({out_path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
