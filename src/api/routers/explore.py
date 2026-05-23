"""GET /explore/personalized — render the 3D movie-space viz for a user.

Uses the precomputed ``MovieSpaceIndex`` (background UMAP fit + ALS item
factor matrix) to project the user's folded-in latent vector into the
same 3D space. Returns a self-contained Plotly HTML page that the
frontend embeds in an iframe.

Cached aggressively: the rendered HTML is keyed by the upload hash, so
repeated visits / refreshes are instant. First-time render is ~1s.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response

from ..routers.recommendations import _load_member  # reuse hash -> ratings lookup
from ...viz import GENRE_COLOR, render_personalized_html

logger = logging.getLogger(__name__)
router = APIRouter(tags=["explore"])


def _state(request: Request):
    state = getattr(request.app.state, "models", None)
    if state is None:
        raise HTTPException(status_code=503, detail="API still warming up")
    return state


@router.get("/explore/background")
def background_coords(
    request: Request,
    limit: int = Query(3000, ge=100, le=8000),
) -> JSONResponse:
    """Return the precomputed UMAP scatter as JSON so the frontend can
    render it inline with plotly.js. Sorted by popularity descending so
    the densest, most-recognizable films land first.

    Used by the landing-page 3D hero.
    """
    state = _state(request)
    if state.movie_space_index is None:
        raise HTTPException(status_code=503, detail="movie-space index not built")
    cache_key = f"bg_json::{limit}"
    if cache_key in state.cache:
        return JSONResponse(state.cache[cache_key])

    msi = state.movie_space_index
    df = msi.background_meta.copy()
    df["x"] = msi.background_coords[:, 0]
    df["y"] = msi.background_coords[:, 1]
    df["z"] = msi.background_coords[:, 2]
    df = df.sort_values("popularity", ascending=False).head(int(limit))

    payload = {
        "n": int(len(df)),
        "coords": df[["x", "y", "z"]].astype(float).values.tolist(),
        "titles": df["title"].astype(str).tolist(),
        "genres": df["genre"].astype(str).tolist(),
        "popularity": df["popularity"].astype(int).tolist(),
        "genre_colors": GENRE_COLOR,
    }
    state.cache[cache_key] = payload
    return JSONResponse(payload)


@router.get("/explore/personalized", response_class=HTMLResponse)
def personalized(
    request: Request,
    hash: str = Query(..., description="upload hash from /upload-letterboxd"),
    label: Optional[str] = Query(None, description="display name for the marker"),
) -> Response:
    state = _state(request)
    if state.movie_space_index is None:
        raise HTTPException(
            status_code=503,
            detail=("The movie-space index isn't built. "
                    "Run scripts/build_movie_space_index.py once."),
        )

    cache_key = f"viz_html::{hash}::{label or ''}"
    if cache_key in state.cache:
        html = state.cache[cache_key]
        return HTMLResponse(html)

    user_ratings, watched = _load_member(state, hash)
    if not user_ratings:
        raise HTTPException(status_code=400, detail="no ratings mapped for that hash")

    msi = state.movie_space_index
    user_pos = msi.project_user(state.als_scorer, user_ratings, watched)
    if user_pos is None:
        raise HTTPException(
            status_code=400,
            detail="no overlap with the ALS catalog — too few rated films were known",
        )

    rated_coords = msi.project_movies(user_ratings.keys())
    watched_only_coords = (
        msi.project_movies(watched - set(user_ratings.keys())) if watched else {}
    )

    html = render_personalized_html(
        msi,
        user_ratings=user_ratings,
        user_position=user_pos,
        rated_coords=rated_coords,
        watched_only_coords=watched_only_coords,
        user_label=label or "you",
    )
    state.cache[cache_key] = html
    logger.info("rendered personalized viz for hash %s (%d rated, %d watched)",
                hash[:10], len(rated_coords), len(watched_only_coords))
    return HTMLResponse(html)
