"""Embed TMDB overview text with sentence-transformers.

Reads the JSONL produced by ``fetch_tmdb_overviews.py``, concatenates
title + tagline + overview into a single document per movie, embeds
each with ``all-MiniLM-L6-v2`` (384-d), L2-normalizes, and saves into
the standard ContentFeatures contract so the reranker can stack it
alongside the genome and director scorers.

Output: ``data/content_overviews.{npz,json}``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize as sk_normalize

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.content_features import ContentFeatures


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--overviews", type=Path,
                   default=PROJECT_ROOT / "data" / "tmdb_overviews.jsonl")
    p.add_argument("--output", type=Path,
                   default=PROJECT_ROOT / "data" / "content_overviews")
    p.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--min-text-length", type=int, default=40,
                   help="skip films whose combined text is shorter than this; "
                        "1-word overviews aren't worth embedding")
    args = p.parse_args()

    if not args.overviews.exists():
        print(f"overviews JSONL missing at {args.overviews}", file=sys.stderr)
        print("Run scripts/fetch_tmdb_overviews.py first.", file=sys.stderr)
        return 1

    print(f"Reading {args.overviews}...")
    rows: list[dict] = []
    with args.overviews.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    print(f"  {len(rows):,} records")

    # Build per-movie document: title (1x) + tagline (1x) + overview (1x).
    # Tagline is short and often theme-laden ("In space, no one can hear you
    # scream"), so include it explicitly even though it duplicates style.
    texts: list[str] = []
    kept_movie_ids: list[int] = []
    for rec in rows:
        parts = []
        for k in ("title", "tagline", "overview"):
            v = rec.get(k) or ""
            v = v.strip()
            if v:
                parts.append(v)
        text = " — ".join(parts).strip()
        if len(text) < args.min_text_length:
            continue
        texts.append(text)
        kept_movie_ids.append(int(rec["movieId"]))
    print(f"  {len(texts):,} films pass the {args.min_text_length}-char filter")

    print(f"Loading model {args.model}... (first run downloads ~80MB)")
    t0 = time.time()
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.model)
    print(f"  loaded in {time.time()-t0:.1f}s; embedding dim {model.get_sentence_embedding_dimension()}")

    print(f"Encoding {len(texts):,} texts in batches of {args.batch_size}...")
    t0 = time.time()
    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,   # already L2 normalized
        convert_to_numpy=True,
    ).astype(np.float32)
    print(f"  encoded in {time.time()-t0:.1f}s; shape {embeddings.shape}")

    # Wrap as CSR so the ContentFeatures contract holds. The matrix is dense
    # in practice — every entry is non-zero — but the size is fine:
    # 87k × 384 × 4 bytes ≈ 134MB sparse, ≈ same dense.
    matrix = csr_matrix(embeddings)

    # Stable sort by movieId so the index_of map is reproducible.
    order = np.argsort(kept_movie_ids)
    matrix = matrix[order]
    movie_ids = np.array([kept_movie_ids[i] for i in order], dtype=np.int64)
    index_of = {str(int(m)): i for i, m in enumerate(movie_ids)}
    # Vocabulary is just dim indices — these are dense, not human-readable.
    vocabulary = [f"emb_{i}" for i in range(matrix.shape[1])]

    cf = ContentFeatures(
        tfidf=matrix,
        movie_ids=movie_ids,
        vocabulary=vocabulary,
        index_of=index_of,
        n_features=matrix.shape[1],
    )
    cf.save(args.output)
    npz_mb = args.output.with_suffix(".npz").stat().st_size / 1024 / 1024
    json_mb = args.output.with_suffix(".json").stat().st_size / 1024 / 1024
    print(f"  wrote {args.output.with_suffix('.npz')}  ({npz_mb:.1f} MB)")
    print(f"  wrote {args.output.with_suffix('.json')}  ({json_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
