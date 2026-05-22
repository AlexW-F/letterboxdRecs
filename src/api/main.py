"""FastAPI backend for letterboxdRecs.

Phase 0 skeleton: health + upload only. Recommendation routes land in
Phase 3 once the re-ranking pipeline (Phase 1) and content scorer
(Phase 2) are in place.

Stateless caching: the upload endpoint computes a SHA-256 of the
ratings.csv contents and caches the TMDB-enriched, MovieLens-mapped
ratings dict against that hash. Future calls just need the hash —
no session IDs, no expiry logic, idempotent re-uploads.
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
import pickle
from pathlib import Path
from typing import Optional

import pandas as pd
from diskcache import Cache
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = Path(os.getenv("MODELS_DIR", PROJECT_ROOT / "models"))
CACHE_DIR = Path(os.getenv("CACHE_DIR", PROJECT_ROOT / ".api_cache"))
LINKS_PATH = Path(os.getenv("LINKS_PATH", PROJECT_ROOT / "ml-latest-small" / "links.csv"))
DEFAULT_MODEL_FILE = os.getenv("DEFAULT_MODEL_FILE", "svdpp.pkl")

app = FastAPI(
    title="letterboxdRecs API",
    description="Upload Letterboxd exports and get personalized + group movie recommendations.",
    version="0.1.0",
)

# Svelte dev server runs on 5173 by default. Add prod origins later.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:4173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

cache = Cache(str(CACHE_DIR))


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str
    cache_dir: str


class UploadResponse(BaseModel):
    hash: str
    n_ratings_in: int
    n_ratings_mapped: int
    n_with_tmdb: int


@app.on_event("startup")
def _load_model() -> None:
    """Eager-load the default model so /health can report status quickly."""
    model_path = MODELS_DIR / DEFAULT_MODEL_FILE
    if not model_path.exists():
        logger.warning("default model %s not found", model_path)
        app.state.model = None
        app.state.model_name = ""
        return
    with open(model_path, "rb") as f:
        app.state.model = pickle.load(f)
    app.state.model_name = DEFAULT_MODEL_FILE
    logger.info("loaded model %s", model_path)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=getattr(app.state, "model", None) is not None,
        model_name=getattr(app.state, "model_name", ""),
        cache_dir=str(CACHE_DIR),
    )


def _hash_csv(payload: bytes) -> str:
    """Content-addressable hash. Sorted-line normalization makes the hash
    insensitive to row ordering — re-exporting Letterboxd in a different
    order still hits the same cache entry."""
    text = payload.decode("utf-8", errors="replace")
    lines = sorted(line for line in text.splitlines() if line.strip())
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def _ratings_to_movielens(ratings_csv: bytes) -> tuple[dict[str, float], int, int]:
    """Map a Letterboxd ratings.csv (with `tmdb_id` column already populated)
    to {movieId: rating}. Returns (mapping, n_input, n_with_tmdb).

    For Phase 0 we require the user to upload an already-enriched CSV;
    Phase 3 will add server-side TMDB lookup for raw exports.
    """
    df = pd.read_csv(io.BytesIO(ratings_csv))
    n_input = len(df)
    rating_col = "Rating" if "Rating" in df.columns else "rating"
    tmdb_col = "tmdb_id" if "tmdb_id" in df.columns else "tmdbId"
    if tmdb_col not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=(
                "Uploaded CSV is missing a tmdb_id column. Phase 0 accepts "
                "Letterboxd exports already enriched with TMDB IDs (see "
                "scripts/add_tmdb_ids.py)."
            ),
        )
    df[tmdb_col] = pd.to_numeric(df[tmdb_col], errors="coerce")
    df = df.dropna(subset=[tmdb_col, rating_col])
    n_with_tmdb = len(df)
    df[tmdb_col] = df[tmdb_col].astype(int).astype(str)

    links_df = pd.read_csv(LINKS_PATH)
    links_df["tmdbId"] = pd.to_numeric(links_df["tmdbId"], errors="coerce")
    links_df = links_df.dropna(subset=["tmdbId"])
    links_df["tmdbId"] = links_df["tmdbId"].astype(int).astype(str)
    links_df["movieId"] = links_df["movieId"].astype(str)

    merged = df.merge(links_df, left_on=tmdb_col, right_on="tmdbId")
    mapping = dict(zip(merged["movieId"], merged[rating_col].astype(float)))
    return mapping, n_input, n_with_tmdb


@app.post("/upload-letterboxd", response_model=UploadResponse)
async def upload_letterboxd(
    ratings: UploadFile = File(..., description="ratings_with_tmdb.csv from a Letterboxd export"),
    watched: Optional[UploadFile] = File(
        None,
        description="watched_with_tmdb.csv — currently filter-only; Phase 1 will use it for implicit feedback",
    ),
) -> UploadResponse:
    ratings_bytes = await ratings.read()
    if not ratings_bytes:
        raise HTTPException(status_code=400, detail="Empty ratings file")
    payload_hash = _hash_csv(ratings_bytes)
    if payload_hash in cache:
        cached = cache[payload_hash]
        return UploadResponse(
            hash=payload_hash,
            n_ratings_in=cached["n_input"],
            n_ratings_mapped=len(cached["ratings"]),
            n_with_tmdb=cached["n_with_tmdb"],
        )

    mapping, n_input, n_with_tmdb = _ratings_to_movielens(ratings_bytes)
    watched_bytes = await watched.read() if watched else b""

    cache[payload_hash] = {
        "ratings": mapping,
        "n_input": n_input,
        "n_with_tmdb": n_with_tmdb,
        "watched_bytes": watched_bytes,
    }
    return UploadResponse(
        hash=payload_hash,
        n_ratings_in=n_input,
        n_ratings_mapped=len(mapping),
        n_with_tmdb=n_with_tmdb,
    )
