"""Fetch TMDB overview text for every film in ml-32m's links.csv.

Streaming write to a JSONL so the fetch is resumable — re-running picks
up where it left off. Each record is {movieId, tmdbId, overview, title}.

Without a TMDB API key this script can't run; with one it takes ~45 min
to fetch 87k films at ~30 req/s. Output ~50 MB.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--links", type=Path, default=PROJECT_ROOT / "ml-32m" / "links.csv")
    p.add_argument("--output", type=Path,
                   default=PROJECT_ROOT / "data" / "tmdb_overviews.jsonl")
    p.add_argument("--api-key", default=os.getenv("TMDB_API_KEY"))
    p.add_argument("--rate-limit", type=float, default=0.0,
                   help="sleep between calls (seconds); TMDB allows ~40/s")
    p.add_argument("--workers", type=int, default=16,
                   help="parallel request workers")
    p.add_argument("--max", type=int, default=None,
                   help="cap number of new fetches (default unlimited)")
    args = p.parse_args()

    if not args.api_key:
        # Last-ditch: scripts/add_tmdb_ids.py has historically hardcoded one.
        try:
            scaffold = (PROJECT_ROOT / "scripts" / "add_tmdb_ids.py").read_text()
            import re
            m = re.search(r'TMDB_API_KEY\s*=\s*"([a-f0-9]{20,})"', scaffold)
            if m:
                args.api_key = m.group(1)
                print("Using TMDB key from scripts/add_tmdb_ids.py")
        except Exception:
            pass
    if not args.api_key:
        print("No TMDB API key (set TMDB_API_KEY env var or --api-key)", file=sys.stderr)
        return 1

    links = pd.read_csv(args.links)
    links = links.dropna(subset=["tmdbId"]).copy()
    links["tmdbId"] = pd.to_numeric(links["tmdbId"], errors="coerce")
    links = links.dropna(subset=["tmdbId"])
    links["tmdbId"] = links["tmdbId"].astype(int)
    links["movieId"] = links["movieId"].astype(int)
    print(f"{len(links):,} films with TMDB IDs")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    done: set[int] = set()
    if args.output.exists():
        with args.output.open() as f:
            for line in f:
                try:
                    done.add(int(json.loads(line)["movieId"]))
                except Exception:
                    pass
    print(f"already cached: {len(done):,}")

    remaining = links[~links["movieId"].isin(done)]
    if args.max is not None:
        remaining = remaining.head(args.max)
    print(f"fetching {len(remaining):,}")

    from concurrent.futures import ThreadPoolExecutor, as_completed
    from threading import Lock

    session = requests.Session()
    n_ok = n_404 = n_err = 0
    t_start = time.time()
    write_lock = Lock()
    counters_lock = Lock()

    def fetch_one(movie_id: int, tmdb_id: int):
        try:
            r = session.get(
                f"https://api.themoviedb.org/3/movie/{tmdb_id}",
                params={"api_key": args.api_key, "language": "en-US"},
                timeout=10,
            )
        except Exception:
            return ("err", None)
        if r.status_code == 200:
            data = r.json()
            return ("ok", {
                "movieId": movie_id,
                "tmdbId": tmdb_id,
                "title": data.get("title") or data.get("original_title") or "",
                "overview": data.get("overview") or "",
                "tagline": data.get("tagline") or "",
            })
        if r.status_code == 404:
            return ("404", None)
        if r.status_code == 429:
            time.sleep(2.0)
            return ("err", None)
        return ("err", None)

    with args.output.open("a") as out, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {}
        for _, row in remaining.iterrows():
            mid = int(row["movieId"])
            tid = int(row["tmdbId"])
            futures[pool.submit(fetch_one, mid, tid)] = mid
            if args.rate_limit > 0:
                time.sleep(args.rate_limit)

        for i, fut in enumerate(as_completed(futures), 1):
            status, payload = fut.result()
            if status == "ok":
                with write_lock:
                    out.write(json.dumps(payload, ensure_ascii=False))
                    out.write("\n")
                with counters_lock:
                    n_ok += 1
            elif status == "404":
                with counters_lock:
                    n_404 += 1
            else:
                with counters_lock:
                    n_err += 1

            if i % 1000 == 0:
                elapsed = time.time() - t_start
                rate = i / elapsed
                eta_min = (len(remaining) - i) / rate / 60
                with write_lock:
                    out.flush()
                print(f"  {i:6d}/{len(remaining):,}  ok={n_ok} 404={n_404} err={n_err}  "
                      f"{rate:.1f}/s  eta {eta_min:.1f}m")

    print(f"done: {n_ok} ok, {n_404} 404, {n_err} err in {(time.time()-t_start)/60:.1f}m")
    print(f"wrote {args.output} ({args.output.stat().st_size/1024/1024:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
