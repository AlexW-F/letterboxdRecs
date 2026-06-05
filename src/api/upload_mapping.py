"""Map uploaded Letterboxd CSVs to ml-32m movieIds.

Split out from the FastAPI app module so the resolution logic — the bit
that decides which catalog movie each of a user's films is — can be unit
tested with just pandas, without importing the model stack.

Per-row resolution order (see ``title_matching.resolve_movie_id``):

  1. ``tmdb_id`` column → movieId (authoritative; same id space as links)
  2. local ``(title, year)`` match against the catalog — keyless, instant
  3. TMDB *search* enrichment — last resort, only for rows that carried no
     id and didn't match locally, and only when a TMDB key is configured.

Step 2 is why a raw export (``Name``/``Year`` only) maps without a
``TMDB_API_KEY``; the key, when present, only adds recall on top via step 3.
"""

from __future__ import annotations

import io
import os
from typing import TYPE_CHECKING, Optional

import pandas as pd
from fastapi import HTTPException

from ..title_matching import coerce_int, resolve_movie_id

if TYPE_CHECKING:
    from diskcache import Cache

    from ..title_matching import LocalTitleIndex

# Public-endpoint DoS guard: how many titles we'll TMDB-search per request
# (search calls run on our key). Local matching resolves most rows for free,
# so only genuine local-misses count against this.
MAX_ENRICH_ROWS = int(os.getenv("MAX_ENRICH_ROWS", "5000"))


def _name_col(df: pd.DataFrame) -> Optional[str]:
    return "Name" if "Name" in df.columns else ("name" if "name" in df.columns else None)


def _tmdb_col(df: pd.DataFrame) -> Optional[str]:
    return "tmdb_id" if "tmdb_id" in df.columns else ("tmdbId" if "tmdbId" in df.columns else None)


def _coerce_rating(val) -> Optional[float]:
    """CSV cell → rating float, or ``None`` for missing/NaN/garbage."""
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None
    return None if f != f else f  # drop NaN


def enrich_with_tmdb(
    df: pd.DataFrame, cache: "Cache", api_key: str, max_workers: int = 16
) -> pd.DataFrame:
    """For each row missing a tmdb_id, search TMDB and fill it in.

    Uses a content-addressable per-(name,year) cache so previous lookups
    are reused — re-uploads of overlapping watchlists are nearly free.
    """
    from concurrent.futures import ThreadPoolExecutor

    from ..data_enrichment import extract_year_from_title, search_tmdb_for_movie

    if "tmdb_id" not in df.columns:
        df = df.copy()
        df["tmdb_id"] = pd.NA

    def lookup(name: str, year: Optional[int]) -> Optional[int]:
        key = f"tmdb_search::{name}::{year or ''}"
        if key in cache:
            return cache[key]
        tid = search_tmdb_for_movie(name, year=year, api_key=api_key)
        cache[key] = tid
        return tid

    to_fetch: list[int] = []
    args: list[tuple[str, Optional[int]]] = []
    for idx, row in df.iterrows():
        if pd.notna(row.get("tmdb_id")):
            continue
        name = str(row.get("Name") or row.get("name") or "").strip()
        if not name:
            continue
        year = row.get("Year")
        try:
            year_int: Optional[int] = int(year) if pd.notna(year) else None
        except (TypeError, ValueError):
            year_int = extract_year_from_title(name)
        to_fetch.append(idx)
        args.append((name, year_int))

    if not to_fetch:
        return df
    if len(to_fetch) > MAX_ENRICH_ROWS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"too many titles need TMDB enrichment ({len(to_fetch)} > {MAX_ENRICH_ROWS}). "
                "Pre-enrich with scripts/add_tmdb_ids.py, or use the Letterboxd-username "
                "flow (TMDB IDs arrive inline via RSS)."
            ),
        )

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        results = list(pool.map(lambda a: lookup(*a), args))

    for idx, tid in zip(to_fetch, results):
        if tid is not None:
            df.at[idx, "tmdb_id"] = tid
    return df


def ratings_csv_to_movielens(
    payload: bytes,
    tmdb_to_movie: dict,
    title_index: "LocalTitleIndex",
    cache: "Cache",
    tmdb_key: Optional[str],
) -> tuple[dict[str, float], int, int, int]:
    """Map a Letterboxd ratings CSV to ``{movieId: rating}``.

    A raw export (``Name``/``Year`` only, no key) maps via local matching
    instead of being rejected. Returns ``(mapping, n_input, n_local,
    n_with_tmdb)``; the two counts sum to ``len(mapping)``.
    """
    df = pd.read_csv(io.BytesIO(payload))
    n_input = len(df)
    rating_col = "Rating" if "Rating" in df.columns else "rating"
    if rating_col not in df.columns:
        return {}, n_input, 0, 0
    name_col = _name_col(df)
    tmdb_col = _tmdb_col(df)

    mapping: dict[str, float] = {}
    n_local = n_tmdb = 0
    needs_search: list[tuple] = []  # (df_index, rating) — no id and no local hit

    for idx, row in df.iterrows():
        rating = _coerce_rating(row.get(rating_col))
        if rating is None:
            continue
        name = str(row.get(name_col) or "").strip() if name_col else ""
        year = coerce_int(row.get("Year"))
        tmdb_id = coerce_int(row.get(tmdb_col)) if tmdb_col else None
        mid, how = resolve_movie_id(name, year, tmdb_id, tmdb_to_movie, title_index)
        if mid is None:
            if tmdb_id is None and name:
                needs_search.append((idx, rating))
            continue
        if mid not in mapping:
            mapping[mid] = rating
            if how == "local":
                n_local += 1
            else:
                n_tmdb += 1

    if tmdb_key and needs_search:
        sub = df.loc[[i for i, _ in needs_search]].copy()
        enriched = enrich_with_tmdb(sub, cache, tmdb_key)
        rating_by_idx = dict(needs_search)
        for idx, raw_tid in enriched["tmdb_id"].items():
            tid = coerce_int(raw_tid)
            mid = tmdb_to_movie.get(str(tid)) if tid is not None else None
            if mid and mid not in mapping:
                mapping[mid] = rating_by_idx[idx]
                n_tmdb += 1

    return mapping, n_input, n_local, n_tmdb


def watched_csv_to_movielens(
    payload: bytes,
    tmdb_to_movie: dict,
    title_index: "LocalTitleIndex",
    cache: "Cache",
    tmdb_key: Optional[str],
) -> set[str]:
    """Map a watched/watchlist CSV to a set of movieIds.

    Same resolution order as the ratings path (tmdb_id → local title → TMDB
    search) but there are no ratings to carry — just the id set. Keyless
    local matching means side files map even without a TMDB_API_KEY."""
    df = pd.read_csv(io.BytesIO(payload))
    name_col = _name_col(df)
    tmdb_col = _tmdb_col(df)

    ids: set[str] = set()
    needs_search_idx: list = []
    for idx, row in df.iterrows():
        name = str(row.get(name_col) or "").strip() if name_col else ""
        year = coerce_int(row.get("Year"))
        tmdb_id = coerce_int(row.get(tmdb_col)) if tmdb_col else None
        mid, _ = resolve_movie_id(name, year, tmdb_id, tmdb_to_movie, title_index)
        if mid:
            ids.add(mid)
        elif tmdb_id is None and name:
            needs_search_idx.append(idx)

    if tmdb_key and needs_search_idx:
        sub = df.loc[needs_search_idx].copy()
        enriched = enrich_with_tmdb(sub, cache, tmdb_key)
        for raw_tid in enriched["tmdb_id"]:
            tid = coerce_int(raw_tid)
            mid = tmdb_to_movie.get(str(tid)) if tid is not None else None
            if mid:
                ids.add(mid)
    return ids
