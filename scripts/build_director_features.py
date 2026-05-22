"""Build per-movie director features from IMDb bulk dumps.

For cinephiles, director affinity is often a stronger signal than genre.
Collaborative filtering loses this for long-tail films because the
director vector is sparse and few users overlap on it. A simple one-hot
director feature surfaces this exactly.

Inputs (download from datasets.imdbws.com):
- archives/title.crew.tsv.gz   — (tconst, directors, writers)
- archives/name.basics.tsv.gz  — (nconst, primaryName, ...) for the vocab
- ml-32m/links.csv             — (movieId, imdbId, tmdbId) for joins

Output: ``data/content_directors.{npz,json}`` in the same ContentFeatures
contract used by the genome and TF-IDF scorers.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize as sk_normalize

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.content_features import ContentFeatures


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--crew", type=Path,
                   default=PROJECT_ROOT / "archives" / "title.crew.tsv.gz")
    p.add_argument("--names", type=Path,
                   default=PROJECT_ROOT / "archives" / "name.basics.tsv.gz")
    p.add_argument("--links", type=Path,
                   default=PROJECT_ROOT / "ml-32m" / "links.csv")
    p.add_argument("--output", type=Path,
                   default=PROJECT_ROOT / "data" / "content_directors")
    p.add_argument("--min-director-films", type=int, default=2,
                   help="drop directors with fewer than this many films in our catalog")
    args = p.parse_args()

    for path in (args.crew, args.names, args.links):
        if not path.exists():
            print(f"missing: {path}", file=sys.stderr)
            return 1

    print(f"Loading {args.links}...")
    links = pd.read_csv(args.links)
    # imdbId in links.csv is the raw number (no 'tt' prefix); make it match.
    links["tconst"] = "tt" + links["imdbId"].astype("Int64").astype(str).str.zfill(7)
    links["movieId"] = links["movieId"].astype(np.int64)
    valid_tconsts = set(links["tconst"].dropna())
    print(f"  {len(links):,} link rows, {len(valid_tconsts):,} unique imdb ids")

    print(f"Loading {args.crew}...")
    t0 = time.time()
    crew = pd.read_csv(args.crew, sep="\t", na_values=r"\N",
                       usecols=["tconst", "directors"])
    print(f"  {len(crew):,} crew rows in {time.time()-t0:.1f}s")
    # Filter to our catalog.
    crew = crew[crew["tconst"].isin(valid_tconsts)].dropna(subset=["directors"])
    print(f"  {len(crew):,} after joining to ml-32m catalog")

    # Explode comma-separated director nconsts.
    crew["directors"] = crew["directors"].str.split(",")
    long = crew.explode("directors").rename(columns={"directors": "nconst"})
    long = long.dropna(subset=["nconst"])
    long = long[long["nconst"].str.startswith("nm")]
    print(f"  {len(long):,} (movie, director) edges")

    # Drop directors with too few films to be useful (single-film noise).
    if args.min_director_films > 1:
        nconst_counts = long["nconst"].value_counts()
        keep_nconsts = set(nconst_counts[nconst_counts >= args.min_director_films].index)
        before = len(long)
        long = long[long["nconst"].isin(keep_nconsts)]
        print(f"  filter directors with >= {args.min_director_films} films: "
              f"{before:,} -> {len(long):,} edges ({len(keep_nconsts):,} unique directors)")

    print(f"Loading {args.names} for vocabulary...")
    t0 = time.time()
    needed_nconsts = set(long["nconst"].unique())
    # name.basics is huge; read in chunks to keep memory in check.
    parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(args.names, sep="\t", na_values=r"\N",
                             usecols=["nconst", "primaryName"], chunksize=200_000):
        parts.append(chunk[chunk["nconst"].isin(needed_nconsts)])
    names = pd.concat(parts, ignore_index=True)
    name_of = dict(zip(names["nconst"], names["primaryName"].fillna(names["nconst"])))
    print(f"  resolved {len(name_of):,} of {len(needed_nconsts):,} director names "
          f"({time.time()-t0:.1f}s)")

    # Join the director edges with movieIds.
    long = long.merge(links[["movieId", "tconst"]], on="tconst", how="inner")
    print(f"  {len(long):,} (movieId, director) edges after final join")

    # Build sparse matrix.
    movie_ids = np.array(sorted(long["movieId"].unique()), dtype=np.int64)
    director_nconsts = sorted(long["nconst"].unique())
    mid_to_row = {int(m): i for i, m in enumerate(movie_ids)}
    nconst_to_col = {n: i for i, n in enumerate(director_nconsts)}

    rows = long["movieId"].astype(np.int64).map(mid_to_row).values
    cols = long["nconst"].map(nconst_to_col).values
    data = np.ones(len(long), dtype=np.float32)

    n_movies = len(movie_ids)
    n_directors = len(director_nconsts)
    matrix = csr_matrix((data, (rows, cols)), shape=(n_movies, n_directors), dtype=np.float32)
    # L2-normalize rows so a co-directed film contributes 1/sqrt(k) per director
    # — keeps the cosine contract with the other ContentFeatures.
    matrix = sk_normalize(matrix, norm="l2", axis=1)
    print(f"  matrix shape: {matrix.shape}  nnz: {matrix.nnz:,}")

    vocabulary = [name_of.get(n, n) for n in director_nconsts]

    cf = ContentFeatures(
        tfidf=matrix,
        movie_ids=movie_ids,
        vocabulary=list(vocabulary),
        index_of={str(int(m)): i for i, m in enumerate(movie_ids)},
        n_features=n_directors,
    )
    cf.save(args.output)
    print(f"  wrote {args.output.with_suffix('.npz')} "
          f"({args.output.with_suffix('.npz').stat().st_size/1024/1024:.1f} MB)")
    print(f"  wrote {args.output.with_suffix('.json')} "
          f"({args.output.with_suffix('.json').stat().st_size/1024/1024:.1f} MB)")

    print("\nSample directors (alphabetical, first 30):")
    print("  " + ", ".join(vocabulary[:30]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
