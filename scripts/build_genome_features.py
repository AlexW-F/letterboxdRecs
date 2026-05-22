"""One-shot builder for the Tag Genome 2021 content features.

The Tag Genome is GroupLens's curated tag-relevance dataset: 1,084 tags
× ~9,734 movies × ~10.5M relevance scores in [0, 1]. Drop-in replacement
for the noisy user-tag TF-IDF: same ContentFeatures API, much higher
signal-to-noise.

Reads ``genome-scores.csv`` and ``genome-tags.csv`` (from the standalone
genome-2021 zip on grouplens.org) and writes ``data/content_genome.{npz,json}``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.content_features import ContentFeatures


def main() -> int:
    p = argparse.ArgumentParser()
    # tagdl.csv is the DL-fitted relevance scores (preferred over glmer.csv per
    # the dataset README: deeper coverage of long-tail tags). Format is
    # (tag, item_id, score).
    p.add_argument(
        "--scores",
        type=Path,
        default=PROJECT_ROOT / "archives" / "movie_dataset_public_final" / "scores" / "tagdl.csv",
    )
    p.add_argument("--output", type=Path, default=PROJECT_ROOT / "data" / "content_genome")
    p.add_argument(
        "--min-score",
        type=float,
        default=0.05,
        help="drop scores below this to keep the sparse matrix lean (most are near-zero noise)",
    )
    args = p.parse_args()

    if not args.scores.exists():
        print(f"tagdl.csv missing at {args.scores}", file=sys.stderr)
        return 1

    print(f"Loading {args.scores}...")
    t0 = time.time()
    scores = pd.read_csv(args.scores)
    print(f"  {len(scores):,} (tag, item_id, score) rows in {time.time()-t0:.1f}s")

    if args.min_score > 0:
        before = len(scores)
        scores = scores[scores["score"] >= args.min_score]
        print(f"  filtered score >= {args.min_score}: {before:,} -> {len(scores):,} rows")

    print("Building genome features...")
    t0 = time.time()
    cf = ContentFeatures.from_genome(scores)
    print(f"  matrix shape: {cf.tfidf.shape}  nnz: {cf.tfidf.nnz:,}  "
          f"vocab: {len(cf.vocabulary):,}  ({time.time()-t0:.1f}s)")

    cf.save(args.output)
    npz_mb = args.output.with_suffix(".npz").stat().st_size / 1024 / 1024
    json_mb = args.output.with_suffix(".json").stat().st_size / 1024 / 1024
    print(f"  wrote {args.output.with_suffix('.npz')}  ({npz_mb:.1f} MB)")
    print(f"  wrote {args.output.with_suffix('.json')}  ({json_mb:.1f} MB)")

    print("\nSample tags (first 30):")
    print("  " + ", ".join(cf.vocabulary[:30]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
