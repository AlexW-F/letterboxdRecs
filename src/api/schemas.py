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
    breakdown: Optional[Dict[str, float]] = None


# ---------------------------------------------------------------------------
# /upload-letterboxd
# ---------------------------------------------------------------------------


class UploadOut(BaseModel):
    hash: str
    n_ratings_in: int
    n_ratings_mapped: int
    n_with_tmdb: int
    n_watchlist: int = 0
    source: str = "csv"               # "csv" or "letterboxd_rss"
    letterboxd_username: Optional[str] = None


# ---------------------------------------------------------------------------
# /upload-letterboxd-username — RSS-based ingest
# ---------------------------------------------------------------------------


class UploadUsernameRequest(BaseModel):
    username: str = Field(..., description="A public Letterboxd username (2–15 chars).")


# ---------------------------------------------------------------------------
# Shared groups (multi-device join link)
# ---------------------------------------------------------------------------


class GroupMemberOut(BaseModel):
    name: str
    hash: str
    n_ratings_mapped: int = 0
    n_watchlist: int = 0
    source: str = "csv"
    letterboxd_username: Optional[str] = None
    joined_at: float  # unix timestamp


class GroupStateOut(BaseModel):
    group_id: str
    created_at: float
    members: List[GroupMemberOut]
    votes: Dict[str, Dict[str, str]] = Field(
        default_factory=dict,
        description="movie_id -> {member_name: 'up'|'veto'}",
    )


class GroupCreateRequest(BaseModel):
    name: Optional[str] = Field(None, description="Optional group label, e.g. 'tuesday-movie-club'")


class GroupJoinRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=40,
                      description="Display name for this member within the group")
    hash: str = Field(..., description="Upload hash from /upload-letterboxd or /upload-letterboxd-username")


class GroupVoteRequest(BaseModel):
    member_name: str = Field(..., min_length=1, max_length=40)
    movie_id: str
    vote: str = Field(..., pattern="^(up|veto|clear)$",
                      description="'up' / 'veto' to cast, 'clear' to remove the previous vote")


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
    exclude_seen_by_any: bool = Field(
        False,
        description=(
            "If true, drop films any member has rated or watched (union across all members). "
            "Default per-member exclusion lets a film through if SOME member hasn't seen it; "
            "this strict mode requires NO member to have seen it. Powers the 'Nobody's seen' tab."
        ),
    )
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
# /group/watchlist-overlap — films multiple members want to watch
# ---------------------------------------------------------------------------


class WatchlistOverlapRequest(BaseModel):
    hashes: List[str] = Field(..., min_length=2, max_length=20)
    member_names: Optional[List[str]] = None
    min_members: int = Field(
        2, ge=2, le=20,
        description="Only return films at least this many members have on their watchlist.",
    )
    top_n: int = Field(50, ge=1, le=500)


class WatchlistOverlapItem(BaseModel):
    movie_id: str
    title: str
    members: List[str]   # names of members who have it on their watchlist
    count: int


class WatchlistOverlapResponse(BaseModel):
    member_names: List[str]
    n_with_watchlist: int  # how many members actually uploaded a watchlist
    items: List[WatchlistOverlapItem]


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
