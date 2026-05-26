"""Recommendation routers: individual, group, and group/analyze."""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from fastapi import APIRouter, HTTPException, Request

from ..schemas import (
    ExplanationOut,
    GroupAnalyzeRequest,
    GroupAnalyzeResponse,
    GroupRecOut,
    GroupRecRequest,
    GroupRecResponse,
    IndividualRecRequest,
    IndividualRecResponse,
    PairwiseSimilarity,
    RecOut,
    WatchlistOverlapItem,
    WatchlistOverlapRequest,
    WatchlistOverlapResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["recommendations"])


def _state(request: Request):
    state = getattr(request.app.state, "models", None)
    if state is None:
        raise HTTPException(status_code=503, detail="API still warming up")
    return state


def _explanation_to_out(exp) -> ExplanationOut:
    if exp is None:
        return ExplanationOut()
    return ExplanationOut(
        top_contributing_rated_movies=[
            [str(t), float(r)] for t, r in (exp.top_contributing_rated_movies or [])
        ],
        dominant_genre_overlap=str(exp.dominant_genre_overlap or ""),
        popularity_tier=str(exp.popularity_tier or ""),
        source=str(exp.source or ""),
    )


def _load_member(state, hash_: str) -> Tuple[Dict[str, float], Optional[set]]:
    """Pull a cached uploaded user from diskcache. Returns (ratings, watched_set)."""
    if hash_ not in state.cache:
        raise HTTPException(status_code=404, detail=f"unknown hash {hash_!r}")
    entry = state.cache[hash_]
    ratings = entry.get("ratings") or {}
    watched_ids = entry.get("watched_movie_ids")
    watched = set(watched_ids) if watched_ids else None
    return ratings, watched


# ---------------------------------------------------------------------------
# Individual
# ---------------------------------------------------------------------------


@router.post("/recommend/individual", response_model=IndividualRecResponse)
def recommend_individual(req: IndividualRecRequest, request: Request) -> IndividualRecResponse:
    state = _state(request)
    if req.mode not in state.reranker_modes_set:
        raise HTTPException(status_code=400, detail=f"unknown mode {req.mode!r}")
    ratings, watched = _load_member(state, req.hash)
    if not ratings:
        raise HTTPException(status_code=400, detail="no ratings mapped for that hash")

    recs = state.reranker.recommend(
        user_ratings=ratings,
        watched_movies=watched,
        mode=req.mode,
        top_n=req.top_n,
        exclude_rated=req.exclude_rated,
        exclude_watched=req.exclude_watched,
    )
    return IndividualRecResponse(
        hash=req.hash,
        mode=req.mode,
        n_ratings_used=len(ratings),
        n_watched_excluded=len(watched) if watched else 0,
        recommendations=[
            RecOut(
                movie_id=str(c.movie_id),
                title=str(c.title),
                score=float(c.score),
                breakdown={k: float(v) for k, v in (c.breakdown or {}).items()},
                explanation=_explanation_to_out(c.explanation),
            )
            for c in recs
        ],
    )


# ---------------------------------------------------------------------------
# Group
# ---------------------------------------------------------------------------


def _resolve_members(state, hashes: List[str], names: Optional[List[str]]):
    if names is not None and len(names) != len(hashes):
        raise HTTPException(status_code=400,
                            detail="member_names length must match hashes")
    member_names = list(names) if names else [f"member_{i+1}" for i in range(len(hashes))]
    members = []
    for name, h in zip(member_names, hashes):
        ratings, watched = _load_member(state, h)
        members.append((name, ratings, watched))
    return member_names, members


def _group_recs_to_response(
    names: List[str],
    strategy: str,
    mode: str,
    recs,
) -> GroupRecResponse:
    return GroupRecResponse(
        member_names=names,
        strategy=strategy,
        mode=mode,
        recommendations=[
            GroupRecOut(
                movie_id=str(c.movie_id),
                title=str(c.title),
                score=float(c.score),
                per_member_score={k: float(v) for k, v in c.per_member_score.items()},
                fairness=float(c.fairness),
                explanation=_explanation_to_out(c.explanation),
                breakdown=({k: float(v) for k, v in c.breakdown.items()} if c.breakdown else None),
            )
            for c in recs
        ],
    )


@router.post("/recommend/group", response_model=GroupRecResponse)
def recommend_group(req: GroupRecRequest, request: Request) -> GroupRecResponse:
    state = _state(request)
    if req.mode not in state.reranker_modes_set:
        raise HTTPException(status_code=400, detail=f"unknown mode {req.mode!r}")
    if req.strategy not in state.group_strategies_set:
        raise HTTPException(status_code=400, detail=f"unknown strategy {req.strategy!r}")
    names, members = _resolve_members(state, req.hashes, req.member_names)

    recs = state.group_reranker.recommend(
        members=members,
        strategy=req.strategy,
        mode=req.mode,
        top_n=req.top_n,
        exclude_rated=req.exclude_rated,
        exclude_watched=req.exclude_watched,
        exclude_seen_by_any=req.exclude_seen_by_any,
    )
    return _group_recs_to_response(names, req.strategy, req.mode, recs)


@router.post("/recommend/group/disagreement", response_model=GroupRecResponse)
def recommend_group_disagreement(req: GroupRecRequest, request: Request) -> GroupRecResponse:
    """High-variance group picks — surfaces films one member loves and another
    would skip. Same request shape as /recommend/group; the ``strategy`` field
    is ignored (disagreement has its own sort)."""
    state = _state(request)
    if req.mode not in state.reranker_modes_set:
        raise HTTPException(status_code=400, detail=f"unknown mode {req.mode!r}")
    names, members = _resolve_members(state, req.hashes, req.member_names)

    recs = state.group_reranker.recommend_disagreement(
        members=members,
        mode=req.mode,
        top_n=req.top_n,
        exclude_rated=req.exclude_rated,
        exclude_watched=req.exclude_watched,
    )
    return _group_recs_to_response(names, "disagreement", req.mode, recs)


# ---------------------------------------------------------------------------
# Group analyze — compatibility report
# ---------------------------------------------------------------------------


def _shared_ratings_correlation(a: Dict[str, float], b: Dict[str, float]):
    common = set(a) & set(b)
    if len(common) < 5:
        return len(common), None
    xa = np.array([a[m] for m in common], dtype=np.float32)
    xb = np.array([b[m] for m in common], dtype=np.float32)
    if xa.std() < 1e-6 or xb.std() < 1e-6:
        return len(common), None
    return len(common), float(np.corrcoef(xa, xb)[0, 1])


@router.post("/group/analyze", response_model=GroupAnalyzeResponse)
def analyze_group(req: GroupAnalyzeRequest, request: Request) -> GroupAnalyzeResponse:
    state = _state(request)
    names, members = _resolve_members(state, req.hashes, req.member_names)
    rating_dicts = [r for _, r, _ in members]

    # Pairwise: rating correlation on shared films + cosine of content taste vectors.
    pairwise: List[PairwiseSimilarity] = []
    taste_vecs: List[Optional[np.ndarray]] = []
    if state.content is not None:
        for r in rating_dicts:
            taste_vecs.append(state.content.taste_vector(r))
    else:
        taste_vecs = [None] * len(rating_dicts)

    for i in range(len(members)):
        for j in range(i + 1, len(members)):
            n_shared, pearson = _shared_ratings_correlation(rating_dicts[i], rating_dicts[j])
            cosine = None
            if taste_vecs[i] is not None and taste_vecs[j] is not None:
                cosine = float(np.dot(taste_vecs[i], taste_vecs[j]))
            pairwise.append(PairwiseSimilarity(
                pair=[names[i], names[j]],
                n_shared_movies=n_shared,
                pearson_on_shared=pearson,
                cosine_on_taste=cosine,
            ))

    # Consensus / disagreement: only consider movies rated by all members
    common_ids = set(rating_dicts[0].keys())
    for r in rating_dicts[1:]:
        common_ids &= set(r.keys())

    rows = []
    for mid in common_ids:
        rs = [r[mid] for r in rating_dicts]
        rows.append({
            "movie_id": str(mid),
            "title": state.title_of.get(str(mid), str(mid)),
            "member_ratings": {names[k]: float(rs[k]) for k in range(len(rs))},
            "mean": float(np.mean(rs)),
            "std": float(np.std(rs)),
        })

    # Two meaningful categories with hard thresholds so the lists actually
    # mean what they say (vs. the old behavior where a tightly-rated group
    # got the same films in both lists, just re-sorted).
    #
    # - Consensus = "universally liked": positive mean (>=3.5) AND tight
    #   agreement (std <= 0.75 stars on the 0.5–5.0 Letterboxd scale).
    # - Disagreement = "you'd argue about this": real divergence
    #   (std >= 1.0 stars). Items in the consensus set are excluded so the
    #   two surfaces are disjoint.
    CONSENSUS_MIN_MEAN = 3.5
    CONSENSUS_MAX_STD = 0.75
    DISAGREEMENT_MIN_STD = 1.0

    consensus_pool = [r for r in rows
                      if r["mean"] >= CONSENSUS_MIN_MEAN and r["std"] <= CONSENSUS_MAX_STD]
    consensus = sorted(consensus_pool, key=lambda r: r["mean"] - r["std"], reverse=True)[: req.top_overlap]

    consensus_ids = {r["movie_id"] for r in consensus}
    disagreement_pool = [r for r in rows
                         if r["std"] >= DISAGREEMENT_MIN_STD and r["movie_id"] not in consensus_ids]
    disagreement = sorted(disagreement_pool, key=lambda r: r["std"], reverse=True)[: req.top_overlap]

    return GroupAnalyzeResponse(
        member_names=names,
        pairwise=pairwise,
        consensus_movies=consensus,
        disagreement_movies=disagreement,
    )


# ---------------------------------------------------------------------------
# Group watchlist overlap — films multiple members want to see
# ---------------------------------------------------------------------------


@router.post("/group/watchlist-overlap", response_model=WatchlistOverlapResponse)
def group_watchlist_overlap(
    req: WatchlistOverlapRequest, request: Request,
) -> WatchlistOverlapResponse:
    """Films appearing on >= ``min_members`` members' watchlists. No CF needed —
    a conscious "want to watch" signal from each friend is a strong direct
    group recommendation on its own."""
    state = _state(request)
    if req.member_names is not None and len(req.member_names) != len(req.hashes):
        raise HTTPException(status_code=400,
                            detail="member_names length must match hashes")
    names = (
        list(req.member_names) if req.member_names
        else [f"member_{i+1}" for i in range(len(req.hashes))]
    )

    # Build per-member watchlist sets; track who actually uploaded one.
    per_member: List[set] = []
    n_with_watchlist = 0
    for h in req.hashes:
        if h not in state.cache:
            raise HTTPException(status_code=404, detail=f"unknown hash {h!r}")
        entry = state.cache[h]
        wl = set(entry.get("watchlist_movie_ids") or [])
        if wl:
            n_with_watchlist += 1
        per_member.append(wl)

    # Tally per-movie occurrence across members.
    movie_to_members: Dict[str, List[str]] = {}
    for name, wl in zip(names, per_member):
        for mid in wl:
            movie_to_members.setdefault(str(mid), []).append(name)

    items: List[WatchlistOverlapItem] = []
    for mid, member_list in movie_to_members.items():
        if len(member_list) < req.min_members:
            continue
        items.append(WatchlistOverlapItem(
            movie_id=mid,
            title=state.title_of.get(mid, mid),
            members=member_list,
            count=len(member_list),
        ))
    # Most-shared first; ties broken alphabetically by title for determinism.
    items.sort(key=lambda it: (-it.count, it.title.lower()))
    return WatchlistOverlapResponse(
        member_names=names,
        n_with_watchlist=n_with_watchlist,
        items=items[: req.top_n],
    )
