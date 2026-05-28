"""Shared model state for the FastAPI app.

Loaded once at startup so endpoints don't pay deserialization cost per
request. Exposed via ``app.state.models``.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
from diskcache import Cache

from ..content_features import ContentFeatures
from ..evaluation import build_genre_features
from ..group_reranker import GROUP_STRATEGIES, GroupReranker
from ..reranking import ALSScorer, MODE_WEIGHTS, PopularityModel, Reranker, SVDScorer
from ..viz import MovieSpaceIndex

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class ModelState:
    movies_df: pd.DataFrame
    links_df: pd.DataFrame  # normalized TMDB→movieId table for upload-time enrichment
    title_of: dict
    popularity: PopularityModel
    genre_features: dict
    content: Optional[ContentFeatures]

    reranker: Reranker
    group_reranker: GroupReranker
    cache: Cache

    svd_path: Path
    als_path: Path
    als_scorer: ALSScorer
    movie_space_index: Optional[MovieSpaceIndex] = None
    tmdb_api_key: Optional[str] = None
    reranker_modes_set: frozenset = frozenset(MODE_WEIGHTS.keys())
    group_strategies_set: frozenset = frozenset(GROUP_STRATEGIES)


def _load_popularity(ml_dir: Path) -> PopularityModel:
    """Per-movie rating counts. Prefer a precomputed JSON (built by
    ``scripts/build_popularity.py``) so we don't load the full ~32M-row
    ``ratings.csv`` at startup — that read dominates both memory and
    cold-start time on a constrained host. Falls back to streaming the
    raw ratings file when the precomputed table is absent.
    """
    pop_path = Path(os.getenv("POPULARITY_FILE", PROJECT_ROOT / "data" / "popularity.json"))
    if pop_path.exists():
        with open(pop_path) as f:
            counts = json.load(f)
        logger.info("loaded precomputed popularity from %s (%d movies)", pop_path, len(counts))
        return PopularityModel.from_counts(counts)
    ratings_csv = ml_dir / "ratings.csv"
    logger.warning("no precomputed popularity at %s — reading %s (slow: ~32M rows). "
                   "Run scripts/build_popularity.py to precompute.", pop_path, ratings_csv)
    ratings_df = pd.read_csv(ratings_csv, usecols=["movieId"])
    return PopularityModel(ratings_df)


def load_state() -> ModelState:
    models_dir = Path(os.getenv("MODELS_DIR", PROJECT_ROOT / "models"))
    ml_dir = Path(os.getenv("ML_DATA_DIR", PROJECT_ROOT / "ml-32m"))
    cache_dir = Path(os.getenv("CACHE_DIR", PROJECT_ROOT / ".api_cache"))
    content_path = Path(os.getenv("CONTENT_FEATURES", PROJECT_ROOT / "data" / "content_features"))

    svd_path = models_dir / os.getenv("SVD_FILE", "svd_full.pkl")
    als_path = models_dir / os.getenv("ALS_FILE", "als_full.pkl")

    logger.info("loading SVD from %s", svd_path)
    svd = SVDScorer.from_path(svd_path)
    logger.info("loading ALS from %s", als_path)
    als = ALSScorer.from_path(als_path)

    movies_df = pd.read_csv(ml_dir / "movies.csv")
    popularity = _load_popularity(ml_dir)
    genre_features = build_genre_features(movies_df)
    title_of = {
        str(mid): title
        for mid, title in zip(movies_df["movieId"].astype(str), movies_df["title"])
    }

    # Pre-normalized links table for upload-time TMDB→movieId joins.
    links_df = pd.read_csv(ml_dir / "links.csv")
    links_df["tmdbId"] = pd.to_numeric(links_df["tmdbId"], errors="coerce")
    links_df = links_df.dropna(subset=["tmdbId"]).copy()
    links_df["tmdbId"] = links_df["tmdbId"].astype(int).astype(str)
    links_df["movieId"] = links_df["movieId"].astype(str)
    # Dedup so a single tmdbId can't fan a rating out to multiple movieIds.
    links_df = links_df.drop_duplicates(subset="tmdbId")

    content = None
    if content_path.with_suffix(".npz").exists():
        content = ContentFeatures.load(content_path)
        logger.info("loaded primary content features: %d movies, %d features",
                    content.tfidf.shape[0], content.n_features)
    else:
        logger.warning("no content features at %s — running CF-only", content_path)

    # Optional extra content scorers, comma-separated paths in env. The
    # reranker averages all of them. Default: stack genome (primary) +
    # directors if both exist on disk.
    extra: list[ContentFeatures] = []
    extra_env = os.getenv("CONTENT_FEATURES_EXTRA", "")
    if extra_env.strip():
        for path_str in extra_env.split(","):
            p = Path(path_str.strip())
            if p.with_suffix(".npz").exists():
                cf = ContentFeatures.load(p)
                extra.append(cf)
                logger.info("loaded extra content features %s: %d movies, %d features",
                            p, cf.tfidf.shape[0], cf.n_features)
    # Auto-discover content_overviews if it's there. Sentence-transformer
    # embeddings on TMDB plot text capture thematic taste (war docs, period
    # drama, anime, etc.) — orthogonal to genome's "prestige/aesthetic" axis.
    # Weight 0.3 nudges niche mode toward semantically-similar deep cuts
    # without dominating the genome's clean popularity-debiased ranking.
    auto_weights: list[float] = []
    weight_env = os.getenv("CONTENT_FEATURES_WEIGHTS", "").strip()
    if not extra:
        overviews_path = PROJECT_ROOT / "data" / "content_overviews"
        if (content_path != overviews_path
                and overviews_path.with_suffix(".npz").exists()):
            cf = ContentFeatures.load(overviews_path)
            extra.append(cf)
            auto_weights.append(float(os.getenv("OVERVIEWS_WEIGHT", "0.3")))
            logger.info("auto-loaded TMDB-overview embeddings (weight=%.2f): "
                        "%d movies, %d-d", auto_weights[-1],
                        cf.tfidf.shape[0], cf.n_features)

    if weight_env:
        parsed = [float(x.strip()) for x in weight_env.split(",") if x.strip()]
        # First weight is for the primary; remainder for extras.
        content_weights = parsed
    elif content is not None:
        content_weights = [1.0] + auto_weights + [1.0] * (len(extra) - len(auto_weights))
    else:
        content_weights = ([1.0] * len(extra)) if extra else []

    reranker = Reranker(svd, als, popularity, movies_df, genre_features,
                        content_features=content,
                        content_features_extra=extra,
                        content_features_weights=content_weights or None)
    group_reranker = GroupReranker(reranker)

    cache = Cache(str(cache_dir))

    # Pre-computed UMAP background for /explore personalization. Optional —
    # if missing, /explore just falls back to the static snapshot.
    msi_path = Path(os.getenv("MOVIE_SPACE_INDEX", PROJECT_ROOT / "data" / "movie_space_index.pkl"))
    movie_space_index: Optional[MovieSpaceIndex] = None
    if msi_path.exists():
        movie_space_index = MovieSpaceIndex.load(msi_path)
        logger.info("loaded movie-space index from %s: %d background films",
                    msi_path, movie_space_index.background_coords.shape[0])
    else:
        logger.warning("no movie-space index at %s — /explore is static", msi_path)

    # TMDB key for server-side enrichment of raw Letterboxd uploads. Env only.
    tmdb_key = os.getenv("TMDB_API_KEY")
    if not tmdb_key:
        logger.warning("no TMDB_API_KEY available — uploads of raw Letterboxd CSVs "
                       "(without a tmdb_id column) will be rejected")

    logger.info("API ready: %d movies, %d users in cache",
                len(movies_df), len(cache))
    return ModelState(
        movies_df=movies_df,
        links_df=links_df,
        title_of=title_of,
        popularity=popularity,
        genre_features=genre_features,
        content=content,
        reranker=reranker,
        group_reranker=group_reranker,
        cache=cache,
        svd_path=svd_path,
        als_path=als_path,
        als_scorer=als,
        movie_space_index=movie_space_index,
        tmdb_api_key=tmdb_key,
    )
