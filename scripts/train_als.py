"""Train an implicit-feedback ALS model on ml-32m.

ALS is the primary candidate generator for the Phase 1 re-ranker.
SVD++ (small) is retained as a secondary scoring signal for items it
covers; ALS covers the full ml-32m catalog.

Treats rating >= RATING_THRESHOLD as positive feedback with a confidence
weight `1 + alpha * (rating - threshold)`. This is the standard Hu/Koren
implicit-feedback formulation: confidence grows with rating, but every
positive rating contributes at least 1.0.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ratings",
        type=Path,
        default=PROJECT_ROOT / "ml-32m" / "ratings.csv",
        help="Path to ratings.csv (ml-32m or ml-latest-small)",
    )
    parser.add_argument("--factors", type=int, default=64)
    parser.add_argument("--regularization", type=float, default=0.01)
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--alpha", type=float, default=40.0,
                        help="Confidence-scale factor; conf = 1 + alpha*(rating-threshold)")
    parser.add_argument("--threshold", type=float, default=3.5,
                        help="Min rating treated as positive feedback")
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT / "models" / "als.npz")
    parser.add_argument("--card", type=Path,
                        default=PROJECT_ROOT / "models" / "als.model_card.json")
    args = parser.parse_args()

    if not args.ratings.exists():
        print(f"ratings not found at {args.ratings}", file=sys.stderr)
        return 1

    print(f"Loading {args.ratings}...")
    t0 = time.time()
    df = pd.read_csv(args.ratings, usecols=["userId", "movieId", "rating"])
    print(f"  {len(df):,} total ratings ({time.time()-t0:.1f}s)")

    pos = df[df["rating"] >= args.threshold]
    print(f"  {len(pos):,} positive ratings (>= {args.threshold})")

    user_codes, user_uniques = pd.factorize(pos["userId"])
    item_codes, item_uniques = pd.factorize(pos["movieId"])
    n_users = len(user_uniques)
    n_items = len(item_uniques)
    print(f"  {n_users:,} users x {n_items:,} items")

    confidence = (1.0 + args.alpha * (pos["rating"].values - args.threshold)).astype(np.float32)
    matrix = csr_matrix(
        (confidence, (user_codes, item_codes)),
        shape=(n_users, n_items),
        dtype=np.float32,
    )
    print(f"  matrix nnz = {matrix.nnz:,}  density = {matrix.nnz / (n_users * n_items):.4%}")

    # Lazy import — implicit prints a BLAS / threading message on import.
    from implicit.als import AlternatingLeastSquares

    print(f"Fitting ALS: factors={args.factors} reg={args.regularization} "
          f"iters={args.iterations} alpha={args.alpha}")
    t0 = time.time()
    model = AlternatingLeastSquares(
        factors=args.factors,
        regularization=args.regularization,
        iterations=args.iterations,
        use_gpu=False,
    )
    model.fit(matrix)
    elapsed = time.time() - t0
    print(f"  trained in {elapsed:.1f}s")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        user_factors=np.asarray(model.user_factors, dtype=np.float32),
        item_factors=np.asarray(model.item_factors, dtype=np.float32),
        item_ids=item_uniques.astype(np.int64),
        user_ids=user_uniques.astype(np.int64),
    )
    print(f"  wrote {args.output} ({args.output.stat().st_size/1024/1024:.1f} MB)")

    # Also persist the model object — needed for proper fold-in via
    # implicit.AlternatingLeastSquares.recommend(recalculate_user=True),
    # which solves the Hu/Koren weighted least-squares for new users.
    import pickle
    pickle_path = args.output.with_suffix(".pkl")
    with open(pickle_path, "wb") as f:
        pickle.dump({"model": model, "item_ids": item_uniques.astype(np.int64),
                     "user_ids": user_uniques.astype(np.int64),
                     "alpha": args.alpha, "threshold": args.threshold}, f)
    print(f"  wrote {pickle_path} ({pickle_path.stat().st_size/1024/1024:.1f} MB)")

    # Path may be relative or absolute — resolve and try to make it
    # project-relative for portability, otherwise fall back to the basename.
    rated_resolved = args.ratings.resolve()
    try:
        ratings_label = str(rated_resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        ratings_label = str(args.ratings)

    card = {
        "model": "implicit.AlternatingLeastSquares",
        "version": "0.7.3",
        "training_dataset": ratings_label,
        "factors": args.factors,
        "regularization": args.regularization,
        "iterations": args.iterations,
        "alpha": args.alpha,
        "rating_threshold": args.threshold,
        "confidence_formula": "1 + alpha * (rating - threshold)",
        "n_users": int(n_users),
        "n_items": int(n_items),
        "n_positive_ratings": int(len(pos)),
        "n_total_ratings": int(len(df)),
        "training_time_seconds": round(elapsed, 1),
        "user_factors_shape": list(model.user_factors.shape),
        "item_factors_shape": list(model.item_factors.shape),
    }
    args.card.write_text(json.dumps(card, indent=2))
    print(f"  wrote {args.card}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
