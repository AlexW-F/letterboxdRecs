"""FastAPI backend for letterboxdRecs.

Endpoints:
- ``GET /health``                    — service + model status
- ``GET /modes``                     — available recommendation modes
- ``GET /strategies``                — available group strategies
- ``POST /upload-letterboxd``        — upload ratings.csv (+ watched.csv); returns content hash
- ``POST /recommend/individual``     — body: {hash, mode, top_n, exclude_*} → recs
- ``POST /recommend/group``          — body: {hashes, strategy, mode, top_n, ...} → group recs
- ``POST /group/analyze``            — body: {hashes} → similarity + consensus + disagreement

Caching is content-addressable: SHA-256 of the sorted ratings.csv lines is
both the lookup key and the de-facto "session id". Re-uploads of the same
file are idempotent. No session lifecycle, no PII at rest beyond the
enriched ratings dict.
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from diskcache import Cache

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .routers import (
    explore as explore_router,
    groups as groups_router,
    meta,
    recommendations as rec_router,
)
from . import limits
from .schemas import HealthResponse, UploadOut, UploadUsernameRequest
from .state import load_state
from ..letterboxd_rss import LetterboxdScraperError, fetch_user_blobs, validate_username

logger = logging.getLogger(__name__)

# Public-endpoint DoS guards: cap the raw upload body and how many unmatched
# titles we'll enrich against TMDB per request (TMDB calls run on our key).
MAX_UPLOAD_BYTES = limits.MAX_BODY_BYTES
MAX_ENRICH_ROWS = int(os.getenv("MAX_ENRICH_ROWS", "5000"))

# Allowed CORS origins, comma-separated. Defaults to local dev servers; set
# CORS_ALLOW_ORIGINS to the deployed frontend origin(s) in production.
_DEFAULT_ORIGINS = "http://localhost:5173,http://localhost:4173"
CORS_ALLOW_ORIGINS = [
    o.strip() for o in os.getenv("CORS_ALLOW_ORIGINS", _DEFAULT_ORIGINS).split(",") if o.strip()
]


# Three popular Letterboxd accounts seeded at startup so /group/demo has real
# data on a fresh deploy. Picked for taste variety; if any 404s the demo just
# shows fewer members. Display name = handle.
DEMO_USERNAMES: list[tuple[str, str]] = [
    ("dave", "dave"),
    ("karsten", "karsten"),
    ("davidehrlich", "david ehrlich"),
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Deserialize models once at startup — too slow to pay per request.
    app.state.models = load_state()
    # Seed the /group/demo cache from 3 Letterboxd users via RSS in a daemon
    # thread so the 3 fetches don't add 30-45s to cold-start. The endpoint
    # returns a friendly 503 until seeding finishes.
    import threading
    threading.Thread(
        target=_seed_demo_users, args=(app.state.models,), daemon=True
    ).start()
    yield


app = FastAPI(
    title="letterboxdRecs API",
    description=(
        "Upload Letterboxd exports and get personalized + group movie "
        "recommendations powered by SVD + ALS + tag-content TF-IDF re-ranking."
    ),
    version="0.3.0",
    lifespan=lifespan,
)

# Inner: per-IP rate limiting + body-size guard. Added before CORS so CORS is
# the outermost layer and 429/413 responses still carry CORS headers.
app.add_middleware(limits.RateLimitMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

app.include_router(meta.router)
app.include_router(rec_router.router)
app.include_router(explore_router.router)
app.include_router(groups_router.router)


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    state = getattr(request.app.state, "models", None)
    if state is None:
        return HealthResponse(
            status="warming_up",
            svd_loaded=False,
            als_loaded=False,
            content_loaded=False,
            catalog_size=0,
            model_name="",
        )
    return HealthResponse(
        status="ok",
        svd_loaded=state.reranker.svd is not None,
        als_loaded=state.reranker.als is not None,
        content_loaded=state.content is not None,
        catalog_size=len(state.movies_df),
        model_name=state.svd_path.name + " + " + state.als_path.name,
        movie_space_loaded=state.movie_space_index is not None,
    )


def _hash_csv(payload: bytes) -> str:
    """Sorted-line normalization makes the hash insensitive to row order."""
    text = payload.decode("utf-8", errors="replace")
    lines = sorted(line for line in text.splitlines() if line.strip())
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def _enrich_with_tmdb(
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


def _ratings_csv_to_movielens(
    payload: bytes, links_df: pd.DataFrame, cache: "Cache", tmdb_key: Optional[str]
) -> tuple[dict[str, float], int, int]:
    df = pd.read_csv(io.BytesIO(payload))
    n_input = len(df)
    rating_col = "Rating" if "Rating" in df.columns else "rating"
    tmdb_col = "tmdb_id" if "tmdb_id" in df.columns else "tmdbId"
    if tmdb_col not in df.columns:
        if not tmdb_key:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Uploaded CSV is missing a tmdb_id column and the API has no "
                    "TMDB_API_KEY configured for server-side enrichment. Either set "
                    "TMDB_API_KEY or pre-enrich the CSV with scripts/add_tmdb_ids.py."
                ),
            )
        df = _enrich_with_tmdb(df, cache, tmdb_key)
        tmdb_col = "tmdb_id"
    df[tmdb_col] = pd.to_numeric(df[tmdb_col], errors="coerce")
    df = df.dropna(subset=[tmdb_col, rating_col])
    n_with_tmdb = len(df)
    df[tmdb_col] = df[tmdb_col].astype(int).astype(str)
    merged = df.merge(links_df, left_on=tmdb_col, right_on="tmdbId")
    return (
        dict(zip(merged["movieId"], merged[rating_col].astype(float))),
        n_input,
        n_with_tmdb,
    )


def _watched_csv_to_movielens(
    payload: bytes, links_df: pd.DataFrame, cache: "Cache", tmdb_key: Optional[str]
) -> set[str]:
    df = pd.read_csv(io.BytesIO(payload))
    tmdb_col = "tmdb_id" if "tmdb_id" in df.columns else "tmdbId"
    if tmdb_col not in df.columns:
        if not tmdb_key:
            return set()
        df = _enrich_with_tmdb(df, cache, tmdb_key)
        tmdb_col = "tmdb_id"
    df[tmdb_col] = pd.to_numeric(df[tmdb_col], errors="coerce")
    df = df.dropna(subset=[tmdb_col])
    df[tmdb_col] = df[tmdb_col].astype(int).astype(str)
    merged = df.merge(links_df, left_on=tmdb_col, right_on="tmdbId")
    return set(merged["movieId"].tolist())


@app.post("/upload-letterboxd", response_model=UploadOut)
async def upload_letterboxd(
    request: Request,
    ratings: UploadFile = File(..., description="ratings_with_tmdb.csv from a Letterboxd export"),
    watched: Optional[UploadFile] = File(
        None,
        description="watched_with_tmdb.csv — used for implicit feedback and watched-exclusion",
    ),
    watchlist: Optional[UploadFile] = File(
        None,
        description="watchlist.csv — films the user wants to watch; powers /group/watchlist-overlap",
    ),
) -> UploadOut:
    state = request.app.state.models
    ratings_bytes = await ratings.read()
    if not ratings_bytes:
        raise HTTPException(status_code=400, detail="Empty ratings file")
    if len(ratings_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"ratings file too large (max {MAX_UPLOAD_BYTES} bytes)",
        )
    payload_hash = _hash_csv(ratings_bytes)

    # Hit-or-miss on ratings; watched/watchlist are merged in even on a hit
    # so re-uploading with previously-omitted side files still updates state.
    cached = state.cache.get(payload_hash)
    if cached is None:
        mapping, n_input, n_with_tmdb = _ratings_csv_to_movielens(
            ratings_bytes, state.links_df, state.cache, state.tmdb_api_key,
        )
        cached = {
            "ratings": mapping,
            "watched_movie_ids": [],
            "watchlist_movie_ids": [],
            "n_input": n_input,
            "n_with_tmdb": n_with_tmdb,
        }
    else:
        # Older cache entries may pre-date the watchlist field
        cached.setdefault("watchlist_movie_ids", [])

    if watched is not None:
        watched_bytes = await watched.read()
        if watched_bytes:
            try:
                cached["watched_movie_ids"] = list(_watched_csv_to_movielens(
                    watched_bytes, state.links_df, state.cache, state.tmdb_api_key,
                ))
            except (ValueError, KeyError, UnicodeDecodeError,
                    pd.errors.ParserError, pd.errors.EmptyDataError, HTTPException) as e:
                logger.warning("watched.csv parse failed: %s — proceeding without it", e)

    if watchlist is not None:
        watchlist_bytes = await watchlist.read()
        if watchlist_bytes:
            try:
                # Same shape as watched.csv (Name, Year, Letterboxd URI) — reuse the parser.
                cached["watchlist_movie_ids"] = list(_watched_csv_to_movielens(
                    watchlist_bytes, state.links_df, state.cache, state.tmdb_api_key,
                ))
            except (ValueError, KeyError, UnicodeDecodeError,
                    pd.errors.ParserError, pd.errors.EmptyDataError, HTTPException) as e:
                logger.warning("watchlist.csv parse failed: %s — proceeding without it", e)

    state.cache[payload_hash] = cached
    return UploadOut(
        hash=payload_hash,
        n_ratings_in=cached["n_input"],
        n_ratings_mapped=len(cached["ratings"]),
        n_with_tmdb=cached["n_with_tmdb"],
        n_watchlist=len(cached["watchlist_movie_ids"]),
        source=cached.get("source", "csv"),
        letterboxd_username=cached.get("letterboxd_username"),
    )


def _seed_demo_users(state) -> None:
    """Populate the cache with DEMO_USERNAMES so /group/demo works on a fresh
    deploy. Mirrors the upload-letterboxd-username flow per user. Runs in a
    daemon thread from lifespan; failures are logged + skipped so the API
    boots even if Letterboxd is unreachable. Persists the resulting
    ``[(display, hash), ...]`` list under cache key ``demo::members`` for
    the /group/demo endpoint to read.
    """
    seeded: list[tuple[str, str]] = []
    for handle, display in DEMO_USERNAMES:
        try:
            username = validate_username(handle)
            ratings_blob, watched_blob, _n_rated, _n_watched = fetch_user_blobs(username)
            payload_hash = _hash_csv(ratings_blob)
            if payload_hash not in state.cache:
                mapping, n_input, n_with_tmdb = _ratings_csv_to_movielens(
                    ratings_blob, state.links_df, state.cache, state.tmdb_api_key,
                )
                watched_ids: list = []
                if watched_blob:
                    try:
                        watched_ids = list(_watched_csv_to_movielens(
                            watched_blob, state.links_df, state.cache, state.tmdb_api_key,
                        ))
                    except (ValueError, KeyError, UnicodeDecodeError,
                            pd.errors.ParserError, pd.errors.EmptyDataError, HTTPException) as e:
                        logger.warning("demo seed: watched parse failed for %s: %s", username, e)
                state.cache[payload_hash] = {
                    "ratings": mapping,
                    "watched_movie_ids": watched_ids,
                    "watchlist_movie_ids": [],
                    "n_input": n_input,
                    "n_with_tmdb": n_with_tmdb,
                    "source": "letterboxd_rss",
                    "letterboxd_username": username,
                }
            seeded.append((display, payload_hash))
            logger.info("demo seed: cached %s -> %s", username, payload_hash[:10])
        except Exception as e:  # noqa: BLE001 - background task, swallow all
            logger.warning("demo seed: skipping %s: %s", handle, e)
    if seeded:
        state.cache["demo::members"] = seeded
        logger.info("demo seed: %d/%d ready", len(seeded), len(DEMO_USERNAMES))


@app.post("/upload-letterboxd-username", response_model=UploadOut)
async def upload_letterboxd_username(
    payload: UploadUsernameRequest, request: Request,
) -> UploadOut:
    """Ingest the most-recent ~50 ratings from a public Letterboxd profile's
    RSS feed. Letterboxd's HTML is Cloudflare-gated so this is the only
    no-login route to recent ratings; the CSV upload path remains for users
    who want their full history."""
    state = request.app.state.models
    try:
        username = validate_username(payload.username)
        ratings_blob, watched_blob, _n_rated, _n_watched = fetch_user_blobs(username)
    except LetterboxdScraperError as e:
        raise HTTPException(status_code=400, detail=str(e))

    payload_hash = _hash_csv(ratings_blob)
    cached = state.cache.get(payload_hash)
    if cached is None:
        mapping, n_input, n_with_tmdb = _ratings_csv_to_movielens(
            ratings_blob, state.links_df, state.cache, state.tmdb_api_key,
        )
        watched_ids: list = []
        if watched_blob:
            try:
                watched_ids = list(_watched_csv_to_movielens(
                    watched_blob, state.links_df, state.cache, state.tmdb_api_key,
                ))
            except (ValueError, KeyError, UnicodeDecodeError,
                    pd.errors.ParserError, pd.errors.EmptyDataError, HTTPException) as e:
                logger.warning("synthesized watched parse failed: %s", e)
        cached = {
            "ratings": mapping,
            "watched_movie_ids": watched_ids,
            "watchlist_movie_ids": [],
            "n_input": n_input,
            "n_with_tmdb": n_with_tmdb,
            "source": "letterboxd_rss",
            "letterboxd_username": username,
        }
    else:
        # Same RSS content already cached. Tag it with the username for the
        # UI if it came from a CSV originally (rare but harmless).
        cached.setdefault("source", "letterboxd_rss")
        cached.setdefault("letterboxd_username", username)
        cached.setdefault("watchlist_movie_ids", [])

    state.cache[payload_hash] = cached
    return UploadOut(
        hash=payload_hash,
        n_ratings_in=cached["n_input"],
        n_ratings_mapped=len(cached["ratings"]),
        n_with_tmdb=cached["n_with_tmdb"],
        n_watchlist=len(cached["watchlist_movie_ids"]),
        source=cached.get("source", "letterboxd_rss"),
        letterboxd_username=cached.get("letterboxd_username"),
    )
