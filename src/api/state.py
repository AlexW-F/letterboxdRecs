"""Shared model state for the FastAPI app.

Loaded once at startup so endpoints don't pay deserialization cost per
request. Exposed via ``app.state.models``.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
from diskcache import Cache

import re

from ..content_features import ContentFeatures
from ..evaluation import build_genre_features
from ..group_reranker import GROUP_STRATEGIES, GroupReranker
from ..reranking import ALSScorer, MODE_WEIGHTS, PopularityModel, Reranker, SVDScorer

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
    tmdb_api_key: Optional[str] = None
    reranker_modes_set: frozenset = frozenset(MODE_WEIGHTS.keys())
    group_strategies_set: frozenset = frozenset(GROUP_STRATEGIES)


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
    ratings_df = pd.read_csv(ml_dir / "ratings.csv", usecols=["movieId"])
    popularity = PopularityModel(ratings_df)
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
    # Per-scorer weights via env: CONTENT_FEATURES_WEIGHTS="1.0,0.4" parallel
    # to primary then each extra. Default to 1.0 across the board.
    extra_weights: list[float] = []
    weight_env = os.getenv("CONTENT_FEATURES_WEIGHTS", "").strip()
    if weight_env:
        parsed = [float(x.strip()) for x in weight_env.split(",") if x.strip()]
        # First weight is for the primary; remainder for extras.
        content_weights = parsed
    elif content is not None:
        content_weights = [1.0] + [1.0] * len(extra)
    else:
        content_weights = [1.0] * len(extra)

    reranker = Reranker(svd, als, popularity, movies_df, genre_features,
                        content_features=content,
                        content_features_extra=extra,
                        content_features_weights=content_weights or None)
    group_reranker = GroupReranker(reranker)

    cache = Cache(str(cache_dir))

    # TMDB key for server-side enrichment of raw Letterboxd uploads.
    tmdb_key = os.getenv("TMDB_API_KEY")
    if not tmdb_key:
        # Fallback: existing scripts/add_tmdb_ids.py used to hardcode one.
        scaffold = PROJECT_ROOT / "scripts" / "add_tmdb_ids.py"
        if scaffold.exists():
            m = re.search(r'TMDB_API_KEY\s*=\s*"([a-f0-9]{20,})"', scaffold.read_text())
            if m:
                tmdb_key = m.group(1)
                logger.info("loaded TMDB key from scripts/add_tmdb_ids.py")
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
        tmdb_api_key=tmdb_key,
    )
