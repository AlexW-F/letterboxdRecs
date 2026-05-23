"""Pydantic request/response schemas for the letterboxdRecs API."""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared types
# ---------------------------------------------------------------------------


class ExplanationOut(BaseModel):
    """Why this rec — designed to be UI-renderable as a tooltip."""
    top_contributing_rated_movies: List[List] = Field(
        default_factory=list,
        description="List of [title, your_rating] pairs that pulled this rec toward you.",
    )
    dominant_genre_overlap: str = ""
    popularity_tier: str = ""
    source: str = ""


class RecOut(BaseModel):
    movie_id: str
    title: str
    score: float
    breakdown: Optional[Dict[str, float]] = None
    explanation: Optional[ExplanationOut] = None


class GroupRecOut(BaseModel):
    movie_id: str
    title: str
    score: float
    per_member_score: Dict[str, float]
    fairness: float
    explanation: Optional[ExplanationOut] = None


# ---------------------------------------------------------------------------
# /upload-letterboxd
# ---------------------------------------------------------------------------


class UploadOut(BaseModel):
    hash: str
    n_ratings_in: int
    n_ratings_mapped: int
    n_with_tmdb: int


# ---------------------------------------------------------------------------
# /recommend/individual
# ---------------------------------------------------------------------------


class IndividualRecRequest(BaseModel):
    hash: str = Field(..., description="Content hash returned by /upload-letterboxd")
    mode: str = Field("balanced", description="One of /modes (e.g. balanced / niche / popular / serendipitous)")
    top_n: int = Field(10, ge=1, le=100)
    exclude_rated: bool = True
    exclude_watched: bool = True


class IndividualRecResponse(BaseModel):
    hash: str
    mode: str
    n_ratings_used: int
    n_watched_excluded: int
    recommendations: List[RecOut]


# ---------------------------------------------------------------------------
# /recommend/group
# ---------------------------------------------------------------------------


class GroupRecRequest(BaseModel):
    hashes: List[str] = Field(..., min_length=2, max_length=20)
    strategy: str = Field("average", description="One of /strategies (e.g. average, group_taste_vector)")
    mode: str = Field("balanced", description="Mode applied to each member before aggregation")
    top_n: int = Field(10, ge=1, le=100)
    exclude_rated: bool = True
    exclude_watched: bool = True
    member_names: Optional[List[str]] = Field(
        None,
        description="Optional display labels for each hash (same length as hashes).",
    )


class GroupRecResponse(BaseModel):
    member_names: List[str]
    strategy: str
    mode: str
    recommendations: List[GroupRecOut]


# ---------------------------------------------------------------------------
# /group/analyze — compatibility report
# ---------------------------------------------------------------------------


class GroupAnalyzeRequest(BaseModel):
    hashes: List[str] = Field(..., min_length=2, max_length=20)
    member_names: Optional[List[str]] = None
    top_overlap: int = Field(10, ge=1, le=50,
                             description="How many consensus/disagreement movies to surface")


class PairwiseSimilarity(BaseModel):
    pair: List[str]                     # [name_a, name_b]
    n_shared_movies: int                # how many films both rated
    pearson_on_shared: Optional[float]  # correlation across shared ratings (-1..1)
    cosine_on_taste: Optional[float]    # cosine of content taste vectors (-1..1)


class GroupAnalyzeResponse(BaseModel):
    member_names: List[str]
    pairwise: List[PairwiseSimilarity]
    consensus_movies: List[Dict]   # [{title, member_ratings: {name: rating}, mean, std}]
    disagreement_movies: List[Dict]   # same shape, sorted by std desc


# ---------------------------------------------------------------------------
# /modes and /strategies
# ---------------------------------------------------------------------------


class ModeInfo(BaseModel):
    name: str
    description: str
    weights: Dict[str, float]


class ModesResponse(BaseModel):
    modes: List[ModeInfo]


class StrategyInfo(BaseModel):
    name: str
    description: str


class StrategiesResponse(BaseModel):
    strategies: List[StrategyInfo]


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    status: str
    svd_loaded: bool
    als_loaded: bool
    content_loaded: bool
    catalog_size: int
    model_name: str
    cache_dir: str
    movie_space_loaded: bool = False
