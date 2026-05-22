"""Group recommender built on top of the Phase 1 ``Reranker``.

Two big changes from the legacy ``GroupRecommendationEngine``:

1. **No zero-fill.** The old engine took each member's top-N individual
   recommendations and unioned them, then aggregated. When a candidate
   was missing from a member's top-N, it was assigned score 0.0 —
   poisoning ``average``, ``least_misery``, ``consensus``, and ``hybrid``.
   The new engine computes a real predicted score for every member ×
   every candidate. Members who genuinely cannot score a candidate (item
   not in their CF model's catalog) are omitted from that candidate's
   aggregation rather than dragging the score to zero.

2. **A sixth strategy: ``group_taste_vector``.** Instead of aggregating
   per-member predictions, this fuses all members' positive ratings into
   one virtual super-user, runs ALS fold-in once, and re-ranks against
   that fused signal. This is the only strategy that can surface movies
   none of the individuals would have predicted highly on their own but
   that the *combined* taste vector strongly recommends.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from .reranking import (
    Explanation,
    ModeWeights,
    MODE_WEIGHTS,
    PopularityModel,
    Reranker,
    ScoredCandidate,
    _genre_jaccard,
    _minmax,
)

GROUP_STRATEGIES = [
    "average",
    "least_misery",
    "most_pleasure",
    "consensus",
    "hybrid",
    "group_taste_vector",
]


@dataclass
class GroupScoredCandidate:
    movie_id: str
    title: str
    score: float
    per_member_score: Dict[str, float]   # name -> normalized rerank score (only for members who could score it)
    fairness: float                       # CV across per_member_score (lower = more uniform)
    explanation: Explanation


class GroupReranker:
    """Wraps a single-user ``Reranker`` and orchestrates group aggregation."""

    def __init__(self, reranker: Reranker) -> None:
        self.rr = reranker

    # ------------------------------------------------------------------
    # Per-member scoring on a union of candidates
    # ------------------------------------------------------------------

    def _score_member_over_union(
        self,
        user_ratings: Dict[str, float],
        watched: Optional[Iterable[str]],
        candidates: set,
        mode: str,
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Score an explicit candidate set for this member.

        Calls ``Reranker.score_specific`` which bypasses candidate
        generation and scores every item in the union. Movies the member
        genuinely cannot score (neither CF model knows them, even though
        another member's model does) are omitted — *not* zero-filled.
        """
        scores = self.rr.score_specific(
            user_ratings=user_ratings,
            watched_movies=watched,
            candidate_ids=candidates,
            mode=mode,
        )
        return scores, {}

    # ------------------------------------------------------------------
    # Five member-aggregation strategies (original five, fixed)
    # ------------------------------------------------------------------

    @staticmethod
    def _aggregate(member_scores: List[float], strategy: str, fairness_weight: float = 0.5) -> float:
        if not member_scores:
            return float("-inf")
        arr = np.asarray(member_scores, dtype=np.float64)
        if strategy == "average":
            return float(arr.mean())
        if strategy == "least_misery":
            return float(arr.min())
        if strategy == "most_pleasure":
            return float(arr.max())
        if strategy == "consensus":
            # Reward agreement: mean minus variance
            return float(arr.mean() - arr.var())
        if strategy == "hybrid":
            # mean + fairness_weight * min  (favors broadly-liked picks)
            return float(arr.mean() + fairness_weight * arr.min())
        raise ValueError(f"unknown member-agg strategy {strategy!r}")

    @staticmethod
    def _fairness(member_scores: List[float]) -> float:
        if not member_scores:
            return 0.0
        arr = np.asarray(member_scores, dtype=np.float64)
        mean = float(arr.mean())
        if abs(mean) < 1e-9:
            return 0.0
        return float(arr.std() / abs(mean))

    # ------------------------------------------------------------------
    # The 6th strategy — fused taste vector
    # ------------------------------------------------------------------

    def _group_taste_vector_recommend(
        self,
        members: List[Tuple[str, Dict[str, float], Optional[Iterable[str]]]],
        mode: str,
        top_n: int,
    ) -> List[GroupScoredCandidate]:
        """Merge all members' positive ratings into one super-user and run
        the reranker against that. Aggregate per-member scores afterwards
        for the explanation/UX, but the *ranking* comes from the fused
        signal alone.

        Conflict handling: when multiple members rate the same movie, take
        the *max* — preserves the strongest preference rather than averaging
        away outlier enthusiasm.
        """
        fused_ratings: Dict[str, float] = {}
        fused_watched: set = set()
        for _, ratings, watched in members:
            for mid, r in ratings.items():
                prev = fused_ratings.get(str(mid))
                if prev is None or r > prev:
                    fused_ratings[str(mid)] = float(r)
            if watched:
                fused_watched |= {str(m) for m in watched}

        if not fused_ratings:
            return []

        # Single rerank call against the super-user
        fused_recs = self.rr.recommend(
            user_ratings=fused_ratings,
            watched_movies=fused_watched,
            mode=mode,
            top_n=top_n * 3,  # over-request so we can still compute fairness
        )
        if not fused_recs:
            return []

        candidate_ids = {c.movie_id for c in fused_recs[: top_n * 3]}

        # Score each candidate per-member (for the fairness/per-member breakdown)
        per_member_scores: Dict[str, Dict[str, float]] = {}
        for name, ratings, watched in members:
            scores, _ = self._score_member_over_union(ratings, watched, candidate_ids, mode)
            per_member_scores[name] = scores

        out: List[GroupScoredCandidate] = []
        for cand in fused_recs[: top_n * 3]:
            member_view = {
                name: per_member_scores[name][cand.movie_id]
                for name, _, _ in members
                if cand.movie_id in per_member_scores[name]
            }
            fairness = self._fairness(list(member_view.values()))
            out.append(GroupScoredCandidate(
                movie_id=cand.movie_id,
                title=cand.title,
                score=cand.score,
                per_member_score={k: round(v, 4) for k, v in member_view.items()},
                fairness=round(fairness, 4),
                explanation=cand.explanation,
            ))
        return out[:top_n]

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def recommend(
        self,
        members: List[Tuple[str, Dict[str, float], Optional[Iterable[str]]]],
        strategy: str = "average",
        mode: str = "balanced",
        top_n: int = 10,
        fairness_weight: float = 0.5,
    ) -> List[GroupScoredCandidate]:
        """``members`` is a list of (name, ratings_dict, watched_iter_or_none)."""
        if strategy not in GROUP_STRATEGIES:
            raise ValueError(
                f"unknown strategy {strategy!r}; pick from {GROUP_STRATEGIES}"
            )
        if not members:
            return []

        if strategy == "group_taste_vector":
            return self._group_taste_vector_recommend(members, mode=mode, top_n=top_n)

        # 1) Each member proposes candidates via the reranker (post-rerank top-N).
        per_member_top: Dict[str, List[ScoredCandidate]] = {}
        for name, ratings, watched in members:
            per_member_top[name] = self.rr.recommend(
                user_ratings=ratings, watched_movies=watched,
                mode=mode, top_n=top_n * 5,
            )

        # 2) Candidate union — anything any member ranked into their top-(5N).
        union_ids: set = set()
        for recs in per_member_top.values():
            union_ids.update(c.movie_id for c in recs)

        # 3) For each member, expand to score every candidate (no zero-fill).
        per_member_scores: Dict[str, Dict[str, float]] = {}
        for name, ratings, watched in members:
            scores, _ = self._score_member_over_union(ratings, watched, union_ids, mode)
            per_member_scores[name] = scores

        # 4) Normalize per-member scores into [0, 1] so different members'
        #    score scales don't dominate aggregation.
        for name in per_member_scores:
            per_member_scores[name] = _minmax(per_member_scores[name])

        # 5) Aggregate.
        results: List[GroupScoredCandidate] = []
        for mid in union_ids:
            contributing = {
                name: per_member_scores[name][mid]
                for name in per_member_scores
                if mid in per_member_scores[name]
            }
            if not contributing:
                continue
            # For least_misery, require all members to have scored it;
            # otherwise the "minimum" is misleading.
            if strategy == "least_misery" and len(contributing) < len(members):
                continue
            agg = self._aggregate(list(contributing.values()), strategy,
                                  fairness_weight=fairness_weight)
            fairness = self._fairness(list(contributing.values()))
            # Build an explanation from the first member who has a rerank
            # entry for this candidate (any will do — explanations describe
            # the *movie*, not the member).
            explanation = Explanation(popularity_tier=self.rr.popularity.tier(mid))
            for name, recs in per_member_top.items():
                for c in recs:
                    if c.movie_id == mid:
                        explanation = c.explanation
                        break
                if explanation.source:
                    break
            results.append(GroupScoredCandidate(
                movie_id=mid,
                title=self.rr.titles.get(mid, mid),
                score=float(agg),
                per_member_score={k: round(v, 4) for k, v in contributing.items()},
                fairness=round(fairness, 4),
                explanation=explanation,
            ))

        results.sort(key=lambda c: c.score, reverse=True)
        return results[:top_n]
