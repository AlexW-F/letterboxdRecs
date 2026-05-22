"""Train a Surprise SVD model on ml-32m (or any ratings CSV).

This replaces the existing models/svdpp.pkl (which is ml-latest-small only,
~9.7k items) with an SVD model trained on the full ml-32m catalog (~87k
items). We use SVD rather than SVD++ at this scale because SVD++ on 32M
ratings is prohibitively slow without a GPU; SVD captures most of the
quality and trains in minutes.

Saves the trained Surprise model as a pickle that the existing
``SVDScorer.from_path`` can load without changes.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import pandas as pd
from surprise import Dataset, Reader, SVD

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ratings", type=Path,
                   default=PROJECT_ROOT / "ml-32m" / "ratings.csv")
    p.add_argument("--n-factors", type=int, default=64)
    p.add_argument("--n-epochs", type=int, default=20)
    p.add_argument("--lr-all", type=float, default=0.005)
    p.add_argument("--reg-all", type=float, default=0.02)
    p.add_argument("--output", type=Path,
                   default=PROJECT_ROOT / "models" / "svd_full.pkl")
    p.add_argument("--card", type=Path,
                   default=PROJECT_ROOT / "models" / "svd_full.model_card.json")
    args = p.parse_args()

    if not args.ratings.exists():
        print(f"ratings not found at {args.ratings}", file=sys.stderr)
        return 1

    print(f"Loading {args.ratings}...")
    t0 = time.time()
    reader = Reader(line_format="user item rating timestamp", sep=",",
                    skip_lines=1, rating_scale=(0.5, 5.0))
    data = Dataset.load_from_file(str(args.ratings), reader=reader)
    print(f"  surprise dataset built in {time.time()-t0:.1f}s")

    print("Building trainset...")
    t0 = time.time()
    trainset = data.build_full_trainset()
    print(f"  n_users={trainset.n_users:,}  n_items={trainset.n_items:,}  "
          f"global_mean={trainset.global_mean:.4f}  ({time.time()-t0:.1f}s)")

    print(f"Fitting SVD: factors={args.n_factors} epochs={args.n_epochs} "
          f"lr={args.lr_all} reg={args.reg_all}")
    model = SVD(
        n_factors=args.n_factors,
        n_epochs=args.n_epochs,
        lr_all=args.lr_all,
        reg_all=args.reg_all,
        verbose=True,
    )
    t0 = time.time()
    model.fit(trainset)
    elapsed = time.time() - t0
    print(f"  trained in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(model, f)
    size_mb = args.output.stat().st_size / 1024 / 1024
    print(f"  wrote {args.output} ({size_mb:.1f} MB)")

    rated_resolved = args.ratings.resolve()
    try:
        ratings_label = str(rated_resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        ratings_label = str(args.ratings)

    card = {
        "model": "surprise.SVD",
        "training_dataset": ratings_label,
        "n_users": int(trainset.n_users),
        "n_items": int(trainset.n_items),
        "global_mean": float(trainset.global_mean),
        "rating_scale": [0.5, 5.0],
        "n_factors": args.n_factors,
        "n_epochs": args.n_epochs,
        "lr_all": args.lr_all,
        "reg_all": args.reg_all,
        "training_time_seconds": round(elapsed, 1),
        "size_mb": round(size_mb, 1),
        "note": ("Replaces models/svdpp.pkl (ml-latest-small only). "
                 "Plain SVD chosen over SVD++ to keep training tractable on "
                 "ml-32m without a GPU; SVD captures the bulk of the quality."),
    }
    args.card.write_text(json.dumps(card, indent=2))
    print(f"  wrote {args.card}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
