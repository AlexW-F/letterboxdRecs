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
from typing import Optional

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .routers import meta, recommendations as rec_router
from .schemas import HealthResponse, UploadOut
from .state import load_state

logger = logging.getLogger(__name__)

app = FastAPI(
    title="letterboxdRecs API",
    description=(
        "Upload Letterboxd exports and get personalized + group movie "
        "recommendations powered by SVD + ALS + tag-content TF-IDF re-ranking."
    ),
    version="0.3.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:4173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(meta.router)
app.include_router(rec_router.router)


@app.on_event("startup")
def _load_models() -> None:
    app.state.models = load_state()


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
            cache_dir="",
        )
    return HealthResponse(
        status="ok",
        svd_loaded=state.reranker.svd is not None,
        als_loaded=state.reranker.als is not None,
        content_loaded=state.content is not None,
        catalog_size=len(state.movies_df),
        model_name=state.svd_path.name + " + " + state.als_path.name,
        cache_dir=str(state.cache.directory),
    )


def _hash_csv(payload: bytes) -> str:
    """Sorted-line normalization makes the hash insensitive to row order."""
    text = payload.decode("utf-8", errors="replace")
    lines = sorted(line for line in text.splitlines() if line.strip())
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def _ratings_csv_to_movielens(
    payload: bytes, links_df: pd.DataFrame
) -> tuple[dict[str, float], int, int]:
    df = pd.read_csv(io.BytesIO(payload))
    n_input = len(df)
    rating_col = "Rating" if "Rating" in df.columns else "rating"
    tmdb_col = "tmdb_id" if "tmdb_id" in df.columns else "tmdbId"
    if tmdb_col not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=(
                "Uploaded CSV is missing a tmdb_id column. The API expects "
                "Letterboxd exports already enriched with TMDB IDs."
            ),
        )
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


def _watched_csv_to_movielens(payload: bytes, links_df: pd.DataFrame) -> set[str]:
    df = pd.read_csv(io.BytesIO(payload))
    tmdb_col = "tmdb_id" if "tmdb_id" in df.columns else "tmdbId"
    if tmdb_col not in df.columns:
        return set()
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
) -> UploadOut:
    state = request.app.state.models
    ratings_bytes = await ratings.read()
    if not ratings_bytes:
        raise HTTPException(status_code=400, detail="Empty ratings file")
    payload_hash = _hash_csv(ratings_bytes)
    if payload_hash in state.cache:
        cached = state.cache[payload_hash]
        return UploadOut(
            hash=payload_hash,
            n_ratings_in=cached["n_input"],
            n_ratings_mapped=len(cached["ratings"]),
            n_with_tmdb=cached["n_with_tmdb"],
        )

    mapping, n_input, n_with_tmdb = _ratings_csv_to_movielens(ratings_bytes, state.links_df)
    watched_ids: set[str] = set()
    if watched is not None:
        watched_bytes = await watched.read()
        if watched_bytes:
            try:
                watched_ids = _watched_csv_to_movielens(watched_bytes, state.links_df)
            except Exception as e:  # noqa: BLE001
                logger.warning("watched.csv parse failed: %s — proceeding without it", e)

    state.cache[payload_hash] = {
        "ratings": mapping,
        "watched_movie_ids": list(watched_ids),
        "n_input": n_input,
        "n_with_tmdb": n_with_tmdb,
    }
    return UploadOut(
        hash=payload_hash,
        n_ratings_in=n_input,
        n_ratings_mapped=len(mapping),
        n_with_tmdb=n_with_tmdb,
    )
