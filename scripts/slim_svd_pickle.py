"""Slim a Surprise SVD pickle for inference-only deployment.

Surprise embeds the entire training set in the pickled model object
(ur/ir dicts holding all 32M ratings in our case), which inflates the
ml-32m SVD pickle to ~1 GB. For inference we only need:

- ``trainset.global_mean``
- ``trainset.rating_scale``
- ``trainset._raw2inner_id_items`` and ``trainset._raw2inner_id_users``
  (or the reverse maps; we only use ``to_inner_iid`` / ``to_raw_iid``)
- ``model.qi`` (item factor matrix)
- ``model.bi`` (item biases)
- ``model.bu`` (user biases — unused at inference for new users, but cheap)
- ``model.pu`` (user factors — unused at inference for new users)
- ``model.n_factors``

This script loads the heavy pickle, throws away the ratings dicts, and
re-pickles. Drops the model file from ~1GB to ~50-200MB, fits in the
default 8GB Docker Desktop memory limit during load.
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def slim(model):
    """Mutate the model's trainset in place to drop heavy ratings dicts."""
    ts = model.trainset
    # Surprise stores raw -> inner id mappings as private dicts. Keep them.
    # Drop the rating storage (ur, ir): they hold all (item, rating) pairs
    # per user (and vice versa) — multiple GB for ml-32m.
    for attr in ("ur", "ir"):
        if hasattr(ts, attr):
            try:
                setattr(ts, attr, {})
            except Exception:
                # Some Surprise versions back this with a defaultdict whose
                # default_factory is a closure; recreate the same shape but empty.
                try:
                    cls = type(getattr(ts, attr))
                    setattr(ts, attr, cls())
                except Exception:
                    pass
    # If there's a numpy view (rare), drop it too.
    if hasattr(ts, "raw_ratings"):
        try:
            ts.raw_ratings = []
        except Exception:
            pass
    return model


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path,
                   default=PROJECT_ROOT / "models" / "svd_full.pkl")
    p.add_argument("--output", type=Path,
                   default=PROJECT_ROOT / "models" / "svd_full_slim.pkl")
    args = p.parse_args()

    if not args.input.exists():
        print(f"input missing: {args.input}", file=sys.stderr)
        return 1

    print(f"Loading {args.input} ({args.input.stat().st_size/1024/1024:.1f} MB)...")
    t0 = time.time()
    with open(args.input, "rb") as f:
        model = pickle.load(f)
    print(f"  loaded in {time.time()-t0:.1f}s; trainset has "
          f"{getattr(model.trainset, 'n_users', '?')} users x "
          f"{getattr(model.trainset, 'n_items', '?')} items")

    print("Slimming trainset (dropping ur/ir ratings dicts)...")
    model = slim(model)

    t0 = time.time()
    with open(args.output, "wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = args.output.stat().st_size / 1024 / 1024
    print(f"  wrote {args.output} ({size_mb:.1f} MB) in {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
