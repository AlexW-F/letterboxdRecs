"""Phase 1 re-ranking pipeline.

Architecture: ``candidate generation -> hybrid scoring -> re-ranking``.

Two candidate generators feed the pool:
- **SVDScorer**: wraps the trained Surprise SVD/SVD++ model. Uses the
  existing ridge-regression fold-in for new users.
- **ALSScorer**: wraps an `implicit.AlternatingLeastSquares` model
  trained on the same dataset. Uses the library's
  ``recommend(recalculate_user=True)`` path, which solves the Hu/Koren
  weighted least squares for the new user. Naturally incorporates the
  *watched but unrated* list as low-confidence positive signal.

The re-ranker then computes a hybrid score:

.. code-block::

    final = w_svd * svd_norm
          + w_als * als_norm
          - w_pop * log(1 + popularity) / log(1 + max_pop)
          + w_implicit * implicit_overlap_norm
          - w_diversity * mmr_redundancy(item, already_picked)

``MODE_WEIGHTS`` chooses a preset for ``niche`` / ``balanced`` /
``popular`` / ``serendipitous`` — productizing the four niche-discovery
strategies from notebooks/01_knn_collaborative_filtering.ipynb.

A cold-start gate: if a user has fewer than
``config.COLD_START_MIN_OVERLAP`` ratings in the SVD trainset, fall back
to a popularity-by-genre list rather than emit low-confidence SVD scores.

Each returned ``ScoredCandidate`` carries an ``Explanation`` so the UI
can render "why this rec" without re-running the scorer.
"""

from __future__ import annotations

import math
import pickle
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from .config import COLD_START_MIN_OVERLAP, DEFAULT_CANDIDATE_POOL_SIZE
from .content_features import ContentFeatures


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Explanation:
    top_contributing_rated_movies: List[Tuple[str, float]] = field(default_factory=list)
    dominant_genre_overlap: str = ""
    popularity_tier: str = ""  # "blockbuster" / "popular" / "niche" / "obscure"
    source: str = ""  # which scorer drove the rec ("svd+als" / "als" / "svd" / "fallback")

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ScoredCandidate:
    movie_id: str
    title: str
    score: float
    breakdown: Dict[str, float]
    explanation: Explanation


@dataclass
class ModeWeights:
    """Weights for the hybrid re-rank formula. Higher = more influence.

    ``min_rating_count`` is a hard filter (not a weight): candidates with
    fewer than N ratings in the trainset are dropped before scoring. This
    guards against statistical noise from long-tail items where the CF
    model has fit a single user's enthusiasm into a high item bias.
    """
    svd: float = 1.0
    als: float = 1.0
    popularity_penalty: float = 0.3
    implicit_bonus: float = 0.5
    diversity_penalty: float = 0.4
    content: float = 0.0  # Phase 2 plugs into this
    min_rating_count: int = 20


MODE_WEIGHTS: Dict[str, ModeWeights] = {
    # "balanced_discovery" — mid-popularity, decent diversity, sane floor.
    # popularity_penalty pulls noticeably away from blockbusters so this
    # diverges from `popular` in the top-K. Content term blends semantic
    # tag/genre similarity into the score.
    "balanced": ModeWeights(svd=1.0, als=1.0, popularity_penalty=0.7,
                            implicit_bonus=0.5, diversity_penalty=0.4,
                            content=0.6, min_rating_count=50),
    # "hidden_gems" / "niche": strong popularity penalty + minimum rating
    # count, plus a heavier content weight so picks are at least
    # semantically aligned with what the user likes.
    "niche": ModeWeights(svd=0.7, als=1.0, popularity_penalty=1.5,
                         implicit_bonus=0.6, diversity_penalty=0.3,
                         content=0.9, min_rating_count=30),
    # Mainstream-friendly: popularity bonus, SVD-heavy, less content
    # influence (popular picks already overlap obvious tag clusters).
    "popular": ModeWeights(svd=1.4, als=1.0, popularity_penalty=-0.8,
                           implicit_bonus=0.3, diversity_penalty=0.15,
                           content=0.3, min_rating_count=200),
    # Serendipitous: highest diversity weight, real content signal so
    # "surprise" stays *adjacent* to taste rather than fully random.
    "serendipitous": ModeWeights(svd=0.6, als=0.9, popularity_penalty=0.7,
                                 implicit_bonus=0.7, diversity_penalty=1.0,
                                 content=0.7, min_rating_count=25),
}


# ---------------------------------------------------------------------------
# Popularity
# ---------------------------------------------------------------------------

class PopularityModel:
    """Per-movie popularity counts derived from the training ratings."""

    def __init__(self, ratings_df: pd.DataFrame, movie_col: str = "movieId") -> None:
        counts = ratings_df[movie_col].astype(str).value_counts().to_dict()
        self.counts: Dict[str, int] = {str(k): int(v) for k, v in counts.items()}
        self.max_count = max(self.counts.values()) if self.counts else 1

    def log_norm(self, movie_id: str) -> float:
        """log(1 + popularity) / log(1 + max). In [0, 1]."""
        c = self.counts.get(str(movie_id), 0)
        return math.log1p(c) / math.log1p(self.max_count) if self.max_count > 0 else 0.0

    def tier(self, movie_id: str) -> str:
        c = self.counts.get(str(movie_id), 0)
        ratio = c / self.max_count if self.max_count > 0 else 0.0
        if ratio >= 0.5:
            return "blockbuster"
        if ratio >= 0.1:
            return "popular"
        if ratio >= 0.02:
            return "niche"
        return "obscure"


# ---------------------------------------------------------------------------
# Scorers (one per candidate-generation source)
# ---------------------------------------------------------------------------

class SVDScorer:
    """Wraps a Surprise SVD/SVD++ model + new-user fold-in.

    Exposes ``score_all(user_ratings) -> Dict[movieId, raw_score]`` for items
    in the model's trainset.
    """

    def __init__(self, model) -> None:
        self.model = model
        self.trainset = model.trainset

    @classmethod
    def from_path(cls, path: Path) -> "SVDScorer":
        with open(path, "rb") as f:
            return cls(pickle.load(f))

    @property
    def n_items(self) -> int:
        return self.trainset.n_items

    def known_items(self) -> Set[str]:
        return {str(self.trainset.to_raw_iid(i)) for i in range(self.trainset.n_items)}

    def overlap_count(self, user_ratings: Dict[str, float]) -> int:
        n = 0
        for mid in user_ratings:
            try:
                self.trainset.to_inner_iid(str(mid))
                n += 1
            except ValueError:
                continue
        return n

    def fold_in(self, user_ratings: Dict[str, float], reg: float = 0.1) -> Optional[np.ndarray]:
        """Solve ridge regression for new-user latent vector. Returns None if
        no overlap with the trainset (caller should fall back)."""
        mu = self.trainset.global_mean
        bi_arr = self.model.bi
        Q_rows: List[np.ndarray] = []
        y: List[float] = []
        for mid, rating in user_ratings.items():
            try:
                inner = self.trainset.to_inner_iid(str(mid))
            except ValueError:
                continue
            q_i = self.model.qi[inner]
            target = rating - mu - bi_arr[inner]
            Q_rows.append(q_i)
            y.append(target)
        if not Q_rows:
            return None
        Q = np.vstack(Q_rows)
        ya = np.array(y, dtype=np.float32)
        A = Q.T.dot(Q) + reg * np.eye(self.model.n_factors)
        return np.linalg.solve(A, Q.T.dot(ya))

    def score_all(self, user_ratings: Dict[str, float]) -> Dict[str, float]:
        p_u = self.fold_in(user_ratings)
        if p_u is None:
            return {}
        mu = self.trainset.global_mean
        bi = self.model.bi
        qi = self.model.qi
        scores = mu + bi + qi.dot(p_u)
        out: Dict[str, float] = {}
        for inner_i in range(qi.shape[0]):
            out[str(self.trainset.to_raw_iid(inner_i))] = float(scores[inner_i])
        return out


class ALSScorer:
    """Wraps an `implicit.AlternatingLeastSquares` model and provides
    new-user fold-in via ``recommend(recalculate_user=True)``.

    Folds in BOTH explicit positive ratings (rating >= threshold, weighted
    by confidence) AND watched-but-unrated items (confidence = 1.0). This
    is the principled way to incorporate the implicit signal.
    """

    def __init__(self, model, item_ids: np.ndarray, alpha: float, threshold: float) -> None:
        self.model = model
        self.item_ids = np.asarray(item_ids)
        self.index_of: Dict[str, int] = {str(int(mid)): i for i, mid in enumerate(self.item_ids)}
        self.alpha = float(alpha)
        self.threshold = float(threshold)

    @classmethod
    def from_path(cls, path: Path) -> "ALSScorer":
        with open(path, "rb") as f:
            blob = pickle.load(f)
        return cls(blob["model"], blob["item_ids"], blob["alpha"], blob["threshold"])

    def known_items(self) -> Set[str]:
        return set(self.index_of.keys())

    def _build_user_vector(
        self,
        user_ratings: Dict[str, float],
        watched_movies: Optional[Iterable[str]] = None,
    ) -> Optional[csr_matrix]:
        cols: List[int] = []
        confs: List[float] = []
        seen: Set[int] = set()
        for mid, rating in user_ratings.items():
            if rating < self.threshold:
                continue
            idx = self.index_of.get(str(mid))
            if idx is None or idx in seen:
                continue
            seen.add(idx)
            cols.append(idx)
            confs.append(1.0 + self.alpha * (float(rating) - self.threshold))
        if watched_movies:
            for mid in watched_movies:
                idx = self.index_of.get(str(mid))
                if idx is None or idx in seen:
                    continue
                seen.add(idx)
                cols.append(idx)
                confs.append(1.0)  # implicit positive: minimum confidence
        if not cols:
            return None
        n_items = len(self.item_ids)
        return csr_matrix(
            (np.asarray(confs, dtype=np.float32), ([0] * len(cols), cols)),
            shape=(1, n_items),
        )

    def score_all(
        self,
        user_ratings: Dict[str, float],
        watched_movies: Optional[Iterable[str]] = None,
        n_return: int = 1000,
    ) -> Dict[str, float]:
        """Returns top-``n_return`` scored items as {movieId: score}. ALS
        recommend() can return all items, but in practice 1000 is plenty
        for downstream union + re-rank."""
        user_vec = self._build_user_vector(user_ratings, watched_movies)
        if user_vec is None:
            return {}
        ids, scores = self.model.recommend(
            userid=0,
            user_items=user_vec,
            N=n_return,
            recalculate_user=True,
            filter_already_liked_items=True,
        )
        out: Dict[str, float] = {}
        for idx, sc in zip(ids, scores):
            out[str(int(self.item_ids[int(idx)]))] = float(sc)
        return out


# ---------------------------------------------------------------------------
# Diversity (Jaccard on genres)
# ---------------------------------------------------------------------------

def _genre_jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b) or 1
    return inter / union


# ---------------------------------------------------------------------------
# Score normalization helpers
# ---------------------------------------------------------------------------

def _minmax(values: Dict[str, float]) -> Dict[str, float]:
    if not values:
        return {}
    vmin = min(values.values())
    vmax = max(values.values())
    span = vmax - vmin
    if span <= 1e-9:
        return {k: 0.5 for k in values}
    return {k: (v - vmin) / span for k, v in values.items()}


# ---------------------------------------------------------------------------
# Reranker — orchestrator
# ---------------------------------------------------------------------------

class Reranker:
    """Two-pass recommender: union candidates, then hybrid-score + re-rank.

    The class is stateful in the sense that ``svd_scorer``, ``als_scorer``,
    ``popularity``, ``movies_df`` and ``genre_features`` are wired at
    construction time. ``recommend()`` is the single user-facing entry point.
    """

    def __init__(
        self,
        svd_scorer: Optional[SVDScorer],
        als_scorer: Optional[ALSScorer],
        popularity: PopularityModel,
        movies_df: pd.DataFrame,
        genre_features: Dict[str, Set[str]],
        content_features: Optional[ContentFeatures] = None,
        content_features_extra: Optional[List[ContentFeatures]] = None,
    ) -> None:
        if svd_scorer is None and als_scorer is None:
            raise ValueError("At least one of svd_scorer / als_scorer is required")
        self.svd = svd_scorer
        self.als = als_scorer
        self.popularity = popularity
        self.genre_features = genre_features
        # All content scorers as a flat list; primary first if both provided.
        # Each operates in its own embedding space (e.g. tag genome vs director
        # one-hot); the reranker averages their per-candidate normalized scores.
        self.content_scorers: List[ContentFeatures] = []
        if content_features is not None:
            self.content_scorers.append(content_features)
        if content_features_extra:
            self.content_scorers.extend(content_features_extra)
        # Back-compat alias for code that still reads .content (e.g. analyze endpoint).
        self.content = self.content_scorers[0] if self.content_scorers else None
        # Title lookup: movieId -> title
        self.titles: Dict[str, str] = {
            str(mid): title
            for mid, title in zip(movies_df["movieId"].astype(str), movies_df["title"])
        }

    def _content_norms(self, user_ratings: Dict[str, float], candidate_ids) -> Dict[str, float]:
        """Average per-candidate normalized score across all content scorers
        that can score the user. Empty scorers / no-overlap users contribute
        nothing; movies not in any scorer's catalog get the neutral default
        in the caller (0.5)."""
        if not self.content_scorers:
            return {}
        per_scorer: List[Dict[str, float]] = []
        for cf in self.content_scorers:
            taste = cf.taste_vector(user_ratings)
            if taste is None:
                continue
            raw = cf.score(taste, candidate_ids)
            if raw:
                per_scorer.append(_minmax(raw))
        if not per_scorer:
            return {}
        # Average over scorers that have a value for the candidate. Use 0.5
        # for missing-from-scorer so a candidate covered by one scorer doesn't
        # get unfairly penalized vs one covered by multiple.
        all_keys = set().union(*(d.keys() for d in per_scorer))
        out: Dict[str, float] = {}
        for k in all_keys:
            vals = [d.get(k, 0.5) for d in per_scorer]
            out[k] = sum(vals) / len(vals)
        return out

    # ------------------------------------------------------------------
    # Candidate generation
    # ------------------------------------------------------------------

    def generate_candidates(
        self,
        user_ratings: Dict[str, float],
        watched_movies: Optional[Iterable[str]] = None,
        n_per_source: int = DEFAULT_CANDIDATE_POOL_SIZE,
        min_rating_count: int = 0,
        exclude_rated: bool = True,
        exclude_watched: bool = True,
    ) -> Tuple[Set[str], Dict[str, float], Dict[str, float]]:
        """Union top-N candidates from each scorer; return (candidates, svd_scores, als_scores).

        ``exclude_rated`` (default True) drops movies in ``user_ratings`` from
        the candidate pool — the usual "don't recommend movies I've already
        scored" semantic.

        ``exclude_watched`` (default True) drops movies in ``watched_movies``
        from candidates. Set to False when you want recs from the
        watched-but-unrated set ("films I've seen but never rated — surface
        them so I can decide if I liked them").

        Either way, ``watched_movies`` is still fed as positive implicit
        feedback to the ALS scorer.

        ``min_rating_count`` > 0 drops candidates with fewer than that
        many ratings in the trainset — guard against single-rater-noise
        long-tail items.
        """
        exclude: Set[str] = set()
        if exclude_rated:
            exclude |= {str(m) for m in user_ratings}
        if exclude_watched and watched_movies:
            exclude |= {str(m) for m in watched_movies}

        svd_scores = self.svd.score_all(user_ratings) if self.svd else {}
        als_scores = self.als.score_all(user_ratings, watched_movies, n_return=n_per_source * 2) if self.als else {}

        # Trim each to top-N per source by raw score; union the IDs.
        top_svd = sorted(svd_scores.items(), key=lambda kv: kv[1], reverse=True)[: n_per_source]
        top_als = sorted(als_scores.items(), key=lambda kv: kv[1], reverse=True)[: n_per_source]

        def _passes(mid: str) -> bool:
            if mid in exclude:
                return False
            if min_rating_count > 0 and self.popularity.counts.get(mid, 0) < min_rating_count:
                return False
            return True

        candidates = {mid for mid, _ in top_svd if _passes(mid)}
        candidates |= {mid for mid, _ in top_als if _passes(mid)}
        return candidates, svd_scores, als_scores

    # ------------------------------------------------------------------
    # Cold-start fallback
    # ------------------------------------------------------------------

    def _popularity_by_genre_fallback(
        self,
        user_ratings: Dict[str, float],
        top_n: int,
    ) -> List[ScoredCandidate]:
        """When the user has too few overlapping ratings, recommend popular
        items in genres the user has rated highly. Better than emitting
        confidently-wrong SVD scores."""
        liked_genres: Dict[str, float] = {}
        for mid, rating in user_ratings.items():
            if rating < 3.5:
                continue
            for g in self.genre_features.get(str(mid), set()):
                liked_genres[g] = liked_genres.get(g, 0.0) + (rating - 3.0)
        if not liked_genres:
            # Total cold start: just return globally popular
            top = sorted(self.popularity.counts.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
            return [self._make_fallback(mid, score) for mid, score in top]

        scored: List[Tuple[str, float]] = []
        excluded = {str(m) for m in user_ratings}
        for mid, count in self.popularity.counts.items():
            if mid in excluded:
                continue
            genres = self.genre_features.get(mid, set())
            if not genres:
                continue
            relevance = sum(liked_genres.get(g, 0.0) for g in genres)
            if relevance <= 0:
                continue
            scored.append((mid, relevance * math.log1p(count)))
        scored.sort(key=lambda kv: kv[1], reverse=True)
        return [self._make_fallback(mid, sc) for mid, sc in scored[:top_n]]

    def _make_fallback(self, mid: str, score: float) -> ScoredCandidate:
        return ScoredCandidate(
            movie_id=mid,
            title=self.titles.get(mid, str(mid)),
            score=float(score),
            breakdown={"fallback_genre_popularity": float(score)},
            explanation=Explanation(
                popularity_tier=self.popularity.tier(mid),
                dominant_genre_overlap="(cold-start fallback)",
                source="fallback",
            ),
        )

    # ------------------------------------------------------------------
    # Explicit-candidate scoring (used by GroupReranker)
    # ------------------------------------------------------------------

    def score_specific(
        self,
        user_ratings: Dict[str, float],
        watched_movies: Optional[Iterable[str]],
        candidate_ids: Iterable[str],
        mode: str = "balanced",
    ) -> Dict[str, float]:
        """Score an explicit list of candidates for this user, bypassing
        candidate generation. Used by group aggregation to compute the
        same member's score for every movie in the *union* candidate set,
        not just the ones that made the member's own top-K.

        Returns {movieId: final_score} only for candidates the member can
        actually score (item in at least one of the underlying CF models).
        """
        weights = MODE_WEIGHTS.get(mode)
        if weights is None:
            raise ValueError(f"Unknown mode {mode!r}")

        candidate_ids = {str(c) for c in candidate_ids}
        if weights.min_rating_count > 0:
            candidate_ids = {
                c for c in candidate_ids
                if self.popularity.counts.get(c, 0) >= weights.min_rating_count
            }
        svd_all = self.svd.score_all(user_ratings) if self.svd else {}
        als_all = self.als.score_all(user_ratings, watched_movies, n_return=max(2000, len(candidate_ids))) if self.als else {}

        svd_norm = _minmax({mid: svd_all[mid] for mid in candidate_ids if mid in svd_all})
        als_norm = _minmax({mid: als_all[mid] for mid in candidate_ids if mid in als_all})

        liked_genres: Dict[str, float] = {}
        for mid, rating in user_ratings.items():
            if rating >= 3.5:
                for g in self.genre_features.get(str(mid), set()):
                    liked_genres[g] = liked_genres.get(g, 0.0) + (rating - 3.0)
        total_genre_weight = sum(liked_genres.values()) or 1.0

        # Content scoring — averaged across however many scorers are wired
        # (genome + directors today; sentence-transformer embeddings later).
        content_norm: Dict[str, float] = {}
        if self.content_scorers and weights.content > 0:
            content_norm = self._content_norms(user_ratings, candidate_ids)

        out: Dict[str, float] = {}
        for mid in candidate_ids:
            in_svd = mid in svd_all
            in_als = mid in als_all
            if not in_svd and not in_als:
                continue
            svd_term = weights.svd * (svd_norm.get(mid, 0.5) if in_svd else 0.5)
            als_term = weights.als * (als_norm.get(mid, 0.5) if in_als else 0.5)
            pop_term = weights.popularity_penalty * self.popularity.log_norm(mid)
            cand_genres = self.genre_features.get(mid, set())
            if liked_genres and cand_genres:
                overlap = sum(liked_genres.get(g, 0.0) for g in cand_genres)
                overlap_norm = overlap / total_genre_weight
            else:
                overlap_norm = 0.0
            imp_term = weights.implicit_bonus * overlap_norm
            content_term = weights.content * content_norm.get(mid, 0.5)
            out[mid] = svd_term + als_term - pop_term + imp_term + content_term
        return out

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def recommend(
        self,
        user_ratings: Dict[str, float],
        watched_movies: Optional[Iterable[str]] = None,
        mode: str = "balanced",
        top_n: int = 10,
        n_candidates: int = DEFAULT_CANDIDATE_POOL_SIZE,
        exclude_rated: bool = True,
        exclude_watched: bool = True,
    ) -> List[ScoredCandidate]:
        weights = MODE_WEIGHTS.get(mode)
        if weights is None:
            raise ValueError(f"Unknown mode {mode!r}; choose from {list(MODE_WEIGHTS)}")

        # Cold-start gate
        if self.svd and self.svd.overlap_count(user_ratings) < COLD_START_MIN_OVERLAP:
            return self._popularity_by_genre_fallback(user_ratings, top_n)

        candidates, svd_scores, als_scores = self.generate_candidates(
            user_ratings, watched_movies, n_per_source=n_candidates,
            min_rating_count=weights.min_rating_count,
            exclude_rated=exclude_rated,
            exclude_watched=exclude_watched,
        )
        if not candidates:
            return []

        # Normalize the candidate-set portion of each scorer's outputs
        svd_norm = _minmax({mid: svd_scores[mid] for mid in candidates if mid in svd_scores})
        als_norm = _minmax({mid: als_scores[mid] for mid in candidates if mid in als_scores})

        # Build the user's "taste genre" set for explanation + implicit bonus
        liked_genres: Dict[str, float] = {}
        top_rated_movies: List[Tuple[str, float]] = []
        for mid, rating in user_ratings.items():
            if rating >= 3.5:
                top_rated_movies.append((str(mid), float(rating)))
                for g in self.genre_features.get(str(mid), set()):
                    liked_genres[g] = liked_genres.get(g, 0.0) + (rating - 3.0)
        top_rated_movies.sort(key=lambda kv: kv[1], reverse=True)

        # Content scoring — averaged across however many scorers are wired
        # (genome + directors today; sentence-transformer embeddings later).
        content_norm: Dict[str, float] = {}
        if self.content_scorers and weights.content > 0:
            content_norm = self._content_norms(user_ratings, candidates)

        # Score each candidate (without diversity yet — diversity is applied iteratively)
        base_scores: Dict[str, Dict[str, float]] = {}
        for mid in candidates:
            svd_term = weights.svd * svd_norm.get(mid, 0.5)
            als_term = weights.als * als_norm.get(mid, 0.5)
            pop_term = weights.popularity_penalty * self.popularity.log_norm(mid)
            # implicit bonus: genre overlap with watched + rated set; cheap proxy
            cand_genres = self.genre_features.get(mid, set())
            if liked_genres and cand_genres:
                overlap = sum(liked_genres.get(g, 0.0) for g in cand_genres)
                overlap_norm = overlap / (sum(liked_genres.values()) or 1.0)
            else:
                overlap_norm = 0.0
            imp_term = weights.implicit_bonus * overlap_norm
            content_term = weights.content * content_norm.get(mid, 0.5)
            base_scores[mid] = {
                "svd": svd_term,
                "als": als_term,
                "popularity_penalty": -pop_term,
                "implicit_bonus": imp_term,
                "content": content_term,
                "diversity_penalty": 0.0,  # filled below
                "base_total": svd_term + als_term - pop_term + imp_term + content_term,
            }

        # Greedy MMR: pick highest base score, then iteratively penalize remaining
        # candidates by their max genre-Jaccard with the already-picked set.
        remaining = dict(base_scores)
        picked: List[str] = []
        results: List[ScoredCandidate] = []
        while remaining and len(results) < top_n:
            best_mid: Optional[str] = None
            best_total = -math.inf
            for mid, parts in remaining.items():
                if picked:
                    cand_genres = self.genre_features.get(mid, set())
                    redundancy = max(
                        _genre_jaccard(cand_genres, self.genre_features.get(p, set()))
                        for p in picked
                    )
                else:
                    redundancy = 0.0
                div_term = weights.diversity_penalty * redundancy
                total = parts["base_total"] - div_term
                parts["diversity_penalty"] = -div_term
                if total > best_total:
                    best_total = total
                    best_mid = mid
                    parts["final_total"] = total
            if best_mid is None:
                break

            # Build explanation
            cand_genres = self.genre_features.get(best_mid, set())
            shared_genres = sorted(set(liked_genres) & cand_genres,
                                   key=lambda g: -liked_genres.get(g, 0.0))
            sources = []
            if best_mid in svd_scores:
                sources.append("svd")
            if best_mid in als_scores:
                sources.append("als")
            explanation = Explanation(
                top_contributing_rated_movies=[
                    (self.titles.get(m, m), r) for m, r in top_rated_movies[:3]
                ],
                dominant_genre_overlap=", ".join(shared_genres[:3]),
                popularity_tier=self.popularity.tier(best_mid),
                source="+".join(sources) if sources else "unknown",
            )
            results.append(ScoredCandidate(
                movie_id=best_mid,
                title=self.titles.get(best_mid, best_mid),
                score=best_total,
                breakdown=dict(remaining[best_mid]),
                explanation=explanation,
            ))
            picked.append(best_mid)
            del remaining[best_mid]
        return results
