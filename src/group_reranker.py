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
    breakdown: Dict[str, float] = field(default_factory=dict)  # signal-level contributions: svd/als/content/popularity_penalty/implicit_bonus/diversity_penalty


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
        exclude_rated: bool = True,
        exclude_watched: bool = True,
        extra_exclude: Optional[set] = None,
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Score an explicit candidate set for this member.

        Calls ``Reranker.score_specific`` which bypasses candidate
        generation and scores every item in the union. Movies the member
        genuinely cannot score (neither CF model knows them, even though
        another member's model does) are omitted — *not* zero-filled.
        """
        # score_specific scores whatever explicit candidates are given.
        # If the caller wants exclusion semantics, they pre-filter the
        # candidate set — that happens at the union step in `recommend`.
        kept_candidates = set(candidates)
        if exclude_rated:
            kept_candidates -= {str(m) for m in user_ratings}
        if exclude_watched and watched:
            kept_candidates -= {str(m) for m in watched}
        if extra_exclude:
            kept_candidates -= {str(m) for m in extra_exclude}
        # When the union intentionally includes seen films, ALS must score
        # them too — otherwise this member's contribution to a known-loved
        # film is silently dropped from the aggregate.
        als_filter_seen = exclude_rated or exclude_watched
        scores = self.rr.score_specific(
            user_ratings=user_ratings,
            watched_movies=watched,
            candidate_ids=kept_candidates,
            mode=mode,
            filter_seen=als_filter_seen,
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
        exclude_rated: bool = True,
        exclude_watched: bool = True,
        exclude_seen_by_any: bool = False,
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
            exclude_rated=exclude_rated,
            exclude_watched=exclude_watched,
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
                breakdown={k: round(float(v), 4) for k, v in cand.breakdown.items()},
            ))
        return out[:top_n]

    # ------------------------------------------------------------------
    # Shared candidate-construction (used by both recommend and
    # recommend_disagreement). Returns GroupScoredCandidates with
    # score=mean(per_member). Callers re-score and sort per their needs.
    # ------------------------------------------------------------------

    def _build_union_candidates(
        self,
        members: List[Tuple[str, Dict[str, float], Optional[Iterable[str]]]],
        mode: str,
        candidate_top_n_per_member: int,
        exclude_rated: bool = True,
        exclude_watched: bool = True,
        exclude_seen_by_any: bool = False,
    ) -> List[GroupScoredCandidate]:
        # If strict-union exclusion is on, build the global seen set once.
        global_seen: Optional[set] = None
        if exclude_seen_by_any:
            global_seen = set()
            for _name, ratings, watched in members:
                global_seen |= {str(m) for m in ratings}
                if watched:
                    global_seen |= {str(m) for m in watched}

        # 1) Per-member candidate proposals (post-rerank top-N).
        per_member_top: Dict[str, List[ScoredCandidate]] = {}
        for name, ratings, watched in members:
            per_member_top[name] = self.rr.recommend(
                user_ratings=ratings, watched_movies=watched,
                mode=mode, top_n=candidate_top_n_per_member,
                exclude_rated=exclude_rated,
                exclude_watched=exclude_watched,
                extra_exclude=global_seen,
            )

        # 2) Candidate union.
        union_ids: set = set()
        for recs in per_member_top.values():
            union_ids.update(c.movie_id for c in recs)

        # 3) Score each member over the full union (no zero-fill).
        per_member_scores: Dict[str, Dict[str, float]] = {}
        for name, ratings, watched in members:
            scores, _ = self._score_member_over_union(
                ratings, watched, union_ids, mode,
                exclude_rated=exclude_rated, exclude_watched=exclude_watched,
                extra_exclude=global_seen,
            )
            per_member_scores[name] = scores

        # 4) Per-member min-max normalization — different members' score
        #    scales otherwise dominate aggregation/std calculations.
        for name in per_member_scores:
            per_member_scores[name] = _minmax(per_member_scores[name])

        # 5) Assemble candidate records.
        out: List[GroupScoredCandidate] = []
        for mid in union_ids:
            contributing = {
                name: per_member_scores[name][mid]
                for name in per_member_scores
                if mid in per_member_scores[name]
            }
            if not contributing:
                continue
            # Explanation + breakdown sourced from the first per-member-top
            # entry that has this candidate (movie-level facts).
            explanation = Explanation(popularity_tier=self.rr.popularity.tier(mid))
            breakdown: Dict[str, float] = {}
            for name, recs in per_member_top.items():
                for c in recs:
                    if c.movie_id == mid:
                        explanation = c.explanation
                        breakdown = dict(c.breakdown)
                        break
                if explanation.source:
                    break
            arr = np.asarray(list(contributing.values()), dtype=np.float64)
            out.append(GroupScoredCandidate(
                movie_id=mid,
                title=self.rr.titles.get(mid, mid),
                score=float(arr.mean()),  # default; callers may overwrite
                per_member_score={k: round(v, 4) for k, v in contributing.items()},
                fairness=round(self._fairness(list(contributing.values())), 4),
                explanation=explanation,
                breakdown={k: round(float(v), 4) for k, v in breakdown.items()},
            ))
        return out

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def recommend(
        self,
        members: List[Tuple[str, Dict[str, float], Optional[Iterable[str]]]],
        strategy: str = "average",
        mode: str = "balanced",
        top_n: int = 10,
        fairness_weight: float = 0.5,
        exclude_rated: bool = True,
        exclude_watched: bool = True,
        exclude_seen_by_any: bool = False,
    ) -> List[GroupScoredCandidate]:
        """``members`` is a list of (name, ratings_dict, watched_iter_or_none)."""
        if strategy not in GROUP_STRATEGIES:
            raise ValueError(
                f"unknown strategy {strategy!r}; pick from {GROUP_STRATEGIES}"
            )
        if not members:
            return []

        if strategy == "group_taste_vector":
            return self._group_taste_vector_recommend(
                members, mode=mode, top_n=top_n,
                exclude_rated=exclude_rated, exclude_watched=exclude_watched,
                exclude_seen_by_any=exclude_seen_by_any,
            )

        candidates = self._build_union_candidates(
            members, mode=mode,
            candidate_top_n_per_member=top_n * 5,
            exclude_rated=exclude_rated, exclude_watched=exclude_watched,
            exclude_seen_by_any=exclude_seen_by_any,
        )

        # Re-score via the chosen aggregation strategy. `least_misery`
        # requires every member to have a score for the candidate; the
        # minimum is misleading otherwise.
        results: List[GroupScoredCandidate] = []
        for c in candidates:
            if strategy == "least_misery" and len(c.per_member_score) < len(members):
                continue
            agg = self._aggregate(
                list(c.per_member_score.values()),
                strategy,
                fairness_weight=fairness_weight,
            )
            c.score = float(agg)
            results.append(c)

        results.sort(key=lambda c: c.score, reverse=True)
        return results[:top_n]

    def recommend_disagreement(
        self,
        members: List[Tuple[str, Dict[str, float], Optional[Iterable[str]]]],
        mode: str = "balanced",
        top_n: int = 10,
        exclude_rated: bool = True,
        exclude_watched: bool = True,
    ) -> List[GroupScoredCandidate]:
        """High-variance picks — items where one member loves it and another
        would skip. Requires every member to have a score for the item; std
        across one or two members is meaningless. Score stays as mean (so
        the card still conveys "is this any good overall?") and the
        per-member breakdown carries the disagreement."""
        if not members:
            return []
        # Larger candidate pool than `recommend` — disagreement picks often
        # live further down each member's individual ranking than the
        # consensus picks do.
        candidates = self._build_union_candidates(
            members, mode=mode,
            candidate_top_n_per_member=top_n * 8,
            exclude_rated=exclude_rated, exclude_watched=exclude_watched,
        )
        full_only = [c for c in candidates if len(c.per_member_score) == len(members)]
        if not full_only:
            return []

        def _std(c: GroupScoredCandidate) -> float:
            arr = np.asarray(list(c.per_member_score.values()), dtype=np.float64)
            return float(arr.std())

        full_only.sort(key=_std, reverse=True)
        return full_only[:top_n]
