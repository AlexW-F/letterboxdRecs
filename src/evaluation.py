"""
Offline evaluation harness for letterboxdRecs.

Designed to be model-agnostic: any callable that takes a dict of
{movie_id: rating} and returns a ranked list of (movie_id, score) tuples
can be evaluated. This lets us measure SVD++ baselines, the new re-ranker,
ALS candidates, and content-hybrid variants through the same lens.

Metrics computed (per phase plan):
- NDCG@10, NDCG@50  (ranking quality with graded relevance)
- Recall@10, Recall@50  (held-out hits)
- Catalog coverage  (fraction of all catalog items that surface across users)
- Gini popularity index  (inequality of popularity in recs; lower=more diverse)
- Intra-list diversity  (avg pairwise dissimilarity within one user's top-k)

Group evaluation extends the same metrics per-member and adds a fairness
coefficient-of-variation across members.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field, asdict
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd


RecommenderFn = Callable[[Dict[str, float], int], List[Tuple[str, float]]]
GroupRecommenderFn = Callable[[List[Dict[str, float]], int], List[Tuple[str, float]]]


@dataclass
class IndividualEvalReport:
    n_users: int
    n_holdout_total: int
    top_n: int
    ndcg_at_10: float
    ndcg_at_50: float
    recall_at_10: float
    recall_at_50: float
    catalog_coverage_at_50: float
    gini_popularity_at_50: float
    intra_list_diversity_at_10: float
    extras: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class GroupEvalReport:
    strategy: str
    n_members: int
    n_holdout_total: int
    top_n: int
    avg_per_member_ndcg_at_10: float
    fairness_cv_at_10: float  # coefficient of variation of per-member NDCG@10
    avg_per_member_recall_at_50: float
    intra_list_diversity_at_10: float
    catalog_coverage_at_50: float
    gini_popularity_at_50: float
    extras: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Holdout split
# ---------------------------------------------------------------------------

def holdout_split(
    ratings: Dict[str, float],
    holdout_frac: float = 0.2,
    seed: int = 42,
    min_train: int = 5,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Randomly split a user's ratings into (train, holdout) dicts.

    Ensures at least ``min_train`` items remain in the train fold so that
    fold-in stays meaningful. If the user has too few ratings overall, the
    holdout will be empty — caller should skip the user.
    """
    if len(ratings) <= min_train:
        return dict(ratings), {}
    items = list(ratings.items())
    rng = random.Random(seed)
    rng.shuffle(items)
    holdout_n = max(1, int(round(len(items) * holdout_frac)))
    holdout_n = min(holdout_n, len(items) - min_train)
    holdout = dict(items[:holdout_n])
    train = dict(items[holdout_n:])
    return train, holdout


# ---------------------------------------------------------------------------
# Ranking metrics
# ---------------------------------------------------------------------------

def _dcg(rels: Sequence[float]) -> float:
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(rels))


def ndcg_at_k(ranked_ids: Sequence[str], relevance: Dict[str, float], k: int) -> float:
    """NDCG@k using graded relevance from the holdout ratings.

    Relevance is shifted so 3.5-rated items get gain 1, 4.0 → 1.5, etc.
    Items not in the holdout contribute 0.
    """
    if not relevance:
        return 0.0
    rels = [max(0.0, relevance.get(str(mid), 0.0) - 3.0) for mid in ranked_ids[:k]]
    dcg = _dcg(rels)
    ideal = sorted((max(0.0, r - 3.0) for r in relevance.values()), reverse=True)[:k]
    idcg = _dcg(ideal)
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(ranked_ids: Sequence[str], relevant_items: Set[str], k: int) -> float:
    if not relevant_items:
        return 0.0
    top = set(str(mid) for mid in ranked_ids[:k])
    hits = top & relevant_items
    return len(hits) / len(relevant_items)


# ---------------------------------------------------------------------------
# Diversity / coverage / popularity
# ---------------------------------------------------------------------------

def catalog_coverage(per_user_lists: List[List[str]], catalog_size: int) -> float:
    if catalog_size <= 0:
        return 0.0
    seen: Set[str] = set()
    for lst in per_user_lists:
        seen.update(str(mid) for mid in lst)
    return len(seen) / catalog_size


def gini_popularity(per_user_lists: List[List[str]], popularity: Dict[str, int]) -> float:
    """Gini coefficient of recommended-item popularity counts.

    0.0 = recs spread evenly across popularity, 1.0 = all recs land on the
    same hyper-popular item. Useful as a popularity-bias diagnostic.
    """
    counts: List[float] = []
    for lst in per_user_lists:
        for mid in lst:
            counts.append(float(popularity.get(str(mid), 0)))
    if not counts:
        return 0.0
    arr = np.sort(np.array(counts))
    n = arr.size
    if arr.sum() == 0:
        return 0.0
    cum = np.cumsum(arr)
    return (n + 1 - 2.0 * cum.sum() / arr.sum()) / n


def intra_list_diversity(
    ranked_ids: Sequence[str],
    item_features: Dict[str, Set[str]],
    k: int = 10,
) -> float:
    """Average pairwise Jaccard *distance* (1 - similarity) within top-k.

    ``item_features`` maps movieId -> set of genre/tag tokens. Phase 2 will
    replace this with TF-IDF or embedding cosine; the contract stays the same.
    """
    top = [str(mid) for mid in ranked_ids[:k] if str(mid) in item_features]
    if len(top) < 2:
        return 0.0
    total = 0.0
    pairs = 0
    for i in range(len(top)):
        for j in range(i + 1, len(top)):
            a = item_features[top[i]]
            b = item_features[top[j]]
            if not a and not b:
                continue
            inter = len(a & b)
            union = len(a | b) or 1
            total += 1.0 - inter / union
            pairs += 1
    return total / pairs if pairs > 0 else 0.0


# ---------------------------------------------------------------------------
# Individual evaluation
# ---------------------------------------------------------------------------

def evaluate_individual(
    recommender_fn: RecommenderFn,
    users: Dict[str, Dict[str, float]],
    *,
    item_features: Dict[str, Set[str]],
    item_popularity: Dict[str, int],
    catalog_size: int,
    holdout_frac: float = 0.2,
    top_n: int = 50,
    seed: int = 42,
) -> IndividualEvalReport:
    """Evaluate a recommender across multiple synthetic/real Letterboxd users.

    ``users`` maps a user label to their full {movieId: rating} dict. We
    split per user, fold the train portion into the recommender, and measure
    against the holdout.
    """
    ndcgs_10: List[float] = []
    ndcgs_50: List[float] = []
    recalls_10: List[float] = []
    recalls_50: List[float] = []
    per_user_top_lists: List[List[str]] = []
    per_user_diversities: List[float] = []
    total_holdout = 0
    evaluated = 0

    for user_idx, (label, ratings) in enumerate(users.items()):
        train, holdout = holdout_split(
            ratings, holdout_frac=holdout_frac, seed=seed + user_idx
        )
        if not holdout:
            continue
        try:
            ranked = recommender_fn(train, top_n)
        except Exception:
            continue
        ranked_ids = [str(mid) for mid, _ in ranked]
        relevance = {str(k): float(v) for k, v in holdout.items()}
        relevant_set = {k for k, v in relevance.items() if v >= 3.5}

        ndcgs_10.append(ndcg_at_k(ranked_ids, relevance, 10))
        ndcgs_50.append(ndcg_at_k(ranked_ids, relevance, 50))
        recalls_10.append(recall_at_k(ranked_ids, relevant_set, 10))
        recalls_50.append(recall_at_k(ranked_ids, relevant_set, 50))
        per_user_top_lists.append(ranked_ids[:50])
        per_user_diversities.append(intra_list_diversity(ranked_ids, item_features, k=10))
        total_holdout += len(holdout)
        evaluated += 1

    def _mean(xs: List[float]) -> float:
        return float(np.mean(xs)) if xs else 0.0

    return IndividualEvalReport(
        n_users=evaluated,
        n_holdout_total=total_holdout,
        top_n=top_n,
        ndcg_at_10=_mean(ndcgs_10),
        ndcg_at_50=_mean(ndcgs_50),
        recall_at_10=_mean(recalls_10),
        recall_at_50=_mean(recalls_50),
        catalog_coverage_at_50=catalog_coverage(per_user_top_lists, catalog_size),
        gini_popularity_at_50=gini_popularity(per_user_top_lists, item_popularity),
        intra_list_diversity_at_10=_mean(per_user_diversities),
    )


# ---------------------------------------------------------------------------
# Group evaluation
# ---------------------------------------------------------------------------

def evaluate_group(
    group_recommender_fn: GroupRecommenderFn,
    members_ratings: List[Dict[str, float]],
    *,
    strategy: str,
    item_features: Dict[str, Set[str]],
    item_popularity: Dict[str, int],
    catalog_size: int,
    holdout_frac: float = 0.2,
    top_n: int = 50,
    seed: int = 42,
) -> GroupEvalReport:
    """Hold out ratings per member, generate group recs from the trains,
    score per-member NDCG/Recall against each member's holdout."""
    trains: List[Dict[str, float]] = []
    holdouts: List[Dict[str, float]] = []
    for i, ratings in enumerate(members_ratings):
        train, holdout = holdout_split(ratings, holdout_frac=holdout_frac, seed=seed + i)
        trains.append(train)
        holdouts.append(holdout)

    try:
        ranked = group_recommender_fn(trains, top_n)
    except Exception:
        return GroupEvalReport(
            strategy=strategy,
            n_members=len(members_ratings),
            n_holdout_total=0,
            top_n=top_n,
            avg_per_member_ndcg_at_10=0.0,
            fairness_cv_at_10=0.0,
            avg_per_member_recall_at_50=0.0,
            intra_list_diversity_at_10=0.0,
            catalog_coverage_at_50=0.0,
            gini_popularity_at_50=0.0,
            extras={"error": 1.0},
        )

    ranked_ids = [str(mid) for mid, _ in ranked]
    per_member_ndcg10: List[float] = []
    per_member_recall50: List[float] = []
    total_holdout = 0
    for h in holdouts:
        if not h:
            continue
        relevance = {str(k): float(v) for k, v in h.items()}
        relevant_set = {k for k, v in relevance.items() if v >= 3.5}
        per_member_ndcg10.append(ndcg_at_k(ranked_ids, relevance, 10))
        per_member_recall50.append(recall_at_k(ranked_ids, relevant_set, 50))
        total_holdout += len(h)

    ndcg_mean = float(np.mean(per_member_ndcg10)) if per_member_ndcg10 else 0.0
    ndcg_std = float(np.std(per_member_ndcg10)) if per_member_ndcg10 else 0.0
    fairness_cv = ndcg_std / ndcg_mean if ndcg_mean > 0 else 0.0

    return GroupEvalReport(
        strategy=strategy,
        n_members=len(members_ratings),
        n_holdout_total=total_holdout,
        top_n=top_n,
        avg_per_member_ndcg_at_10=ndcg_mean,
        fairness_cv_at_10=fairness_cv,
        avg_per_member_recall_at_50=float(np.mean(per_member_recall50)) if per_member_recall50 else 0.0,
        intra_list_diversity_at_10=intra_list_diversity(ranked_ids, item_features, k=10),
        catalog_coverage_at_50=catalog_coverage([ranked_ids[:50]], catalog_size),
        gini_popularity_at_50=gini_popularity([ranked_ids[:50]], item_popularity),
    )


# ---------------------------------------------------------------------------
# Helpers for loading the catalog signals used by the metrics
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Per-unit evaluation (returns raw lists for bootstrap CIs)
# ---------------------------------------------------------------------------


def evaluate_individual_per_user(
    recommender_fn: RecommenderFn,
    users: Dict[str, Dict[str, float]],
    *,
    item_features: Dict[str, Set[str]],
    holdout_frac: float = 0.2,
    top_n: int = 50,
    seed: int = 42,
) -> Tuple[Dict[str, List[float]], List[List[str]]]:
    """Same evaluation as ``evaluate_individual`` but returns per-user metric
    arrays instead of aggregate means — feed to ``bootstrap_ci`` for CIs.

    Also returns the per-user top-50 rec lists so the caller can compute
    list-aggregated metrics (catalog coverage, Gini popularity) and bootstrap
    them by resampling users.
    """
    out: Dict[str, List[float]] = {
        "ndcg_at_10": [], "ndcg_at_50": [],
        "recall_at_10": [], "recall_at_50": [],
        "intra_list_diversity_at_10": [],
    }
    per_user_top_lists: List[List[str]] = []
    for user_idx, (_label, ratings) in enumerate(users.items()):
        train, holdout = holdout_split(ratings, holdout_frac=holdout_frac, seed=seed + user_idx)
        if not holdout:
            continue
        try:
            ranked = recommender_fn(train, top_n)
        except Exception:
            continue
        ranked_ids = [str(mid) for mid, _ in ranked]
        relevance = {str(k): float(v) for k, v in holdout.items()}
        relevant_set = {k for k, v in relevance.items() if v >= 3.5}
        out["ndcg_at_10"].append(ndcg_at_k(ranked_ids, relevance, 10))
        out["ndcg_at_50"].append(ndcg_at_k(ranked_ids, relevance, 50))
        out["recall_at_10"].append(recall_at_k(ranked_ids, relevant_set, 10))
        out["recall_at_50"].append(recall_at_k(ranked_ids, relevant_set, 50))
        out["intra_list_diversity_at_10"].append(
            intra_list_diversity(ranked_ids, item_features, k=10)
        )
        per_user_top_lists.append(ranked_ids[:50])
    return out, per_user_top_lists


def evaluate_group_per_group(
    group_recommender_fn: GroupRecommenderFn,
    groups: List[List[Dict[str, float]]],
    *,
    item_features: Dict[str, Set[str]],
    holdout_frac: float = 0.2,
    top_n: int = 50,
    seed: int = 42,
) -> Tuple[Dict[str, List[float]], List[List[str]]]:
    """Per-group metric arrays for bootstrap CIs.

    For each synthetic group: hold out each member's ratings, run the group
    rec on the train portions, score each member's NDCG@10 against their
    holdout, return per-group averages + fairness CV across members.

    Also returns per-group top-50 lists so the caller can compute catalog
    coverage / Gini-popularity across the strategy's recommended pool.
    """
    out: Dict[str, List[float]] = {
        "avg_per_member_ndcg_at_10": [],
        "avg_per_member_recall_at_50": [],
        "fairness_cv_at_10": [],
        "intra_list_diversity_at_10": [],
    }
    per_group_top_lists: List[List[str]] = []
    for grp_idx, members in enumerate(groups):
        trains: List[Dict[str, float]] = []
        holdouts: List[Dict[str, float]] = []
        for i, ratings in enumerate(members):
            train, h = holdout_split(ratings, holdout_frac=holdout_frac, seed=seed + grp_idx * 100 + i)
            trains.append(train)
            holdouts.append(h)
        try:
            ranked = group_recommender_fn(trains, top_n)
        except Exception:
            continue
        ranked_ids = [str(mid) for mid, _ in ranked]
        per_member_ndcg10: List[float] = []
        per_member_recall50: List[float] = []
        for h in holdouts:
            if not h:
                continue
            relevance = {str(k): float(v) for k, v in h.items()}
            relevant_set = {k for k, v in relevance.items() if v >= 3.5}
            per_member_ndcg10.append(ndcg_at_k(ranked_ids, relevance, 10))
            per_member_recall50.append(recall_at_k(ranked_ids, relevant_set, 50))
        if not per_member_ndcg10:
            continue
        mean = float(np.mean(per_member_ndcg10))
        std = float(np.std(per_member_ndcg10))
        out["avg_per_member_ndcg_at_10"].append(mean)
        out["avg_per_member_recall_at_50"].append(
            float(np.mean(per_member_recall50)) if per_member_recall50 else 0.0
        )
        out["fairness_cv_at_10"].append(std / mean if mean > 0 else 0.0)
        out["intra_list_diversity_at_10"].append(
            intra_list_diversity(ranked_ids, item_features, k=10)
        )
        per_group_top_lists.append(ranked_ids[:50])
    return out, per_group_top_lists


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------


def bootstrap_set_metric(
    per_unit_lists: List[List[str]],
    metric_fn: Callable[[List[List[str]]], float],
    *,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    """Bootstrap CI for a list-aggregated metric (catalog coverage, Gini).

    The metric is a function ``f(list_of_rec_lists) -> float`` — i.e. it needs
    the full collection of users'/groups' top-N lists to compute. We resample
    the *unit* (one rec list per user/group) with replacement.
    """
    n = len(per_unit_lists)
    if n == 0:
        return {"point": 0.0, "lower": 0.0, "upper": 0.0, "n": 0}
    point = float(metric_fn(per_unit_lists))
    if n < 2:
        return {"point": point, "lower": point, "upper": point, "n": n}
    rng = np.random.default_rng(seed)
    samples = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        resampled = [per_unit_lists[j] for j in idx]
        samples[i] = metric_fn(resampled)
    alpha = (1.0 - confidence) / 2.0
    return {
        "point": point,
        "lower": float(np.percentile(samples, alpha * 100)),
        "upper": float(np.percentile(samples, (1 - alpha) * 100)),
        "n": n,
    }


def bootstrap_ci(
    values: Sequence[float],
    *,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    """Percentile bootstrap CI for the mean of ``values``.

    Returns ``{"mean", "lower", "upper", "n"}``. If ``len(values) < 2``,
    returns the mean with the same value for lower/upper (no CI computable).
    """
    if not values:
        return {"mean": 0.0, "lower": 0.0, "upper": 0.0, "n": 0}
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    if n < 2:
        v = float(arr[0])
        return {"mean": v, "lower": v, "upper": v, "n": n}
    rng = np.random.default_rng(seed)
    # Vectorized: one (n_resamples, n) sample-matrix; mean along axis=1.
    idx = rng.integers(0, n, size=(n_resamples, n))
    sampled_means = arr[idx].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    lower = float(np.percentile(sampled_means, alpha * 100))
    upper = float(np.percentile(sampled_means, (1 - alpha) * 100))
    return {"mean": float(arr.mean()), "lower": lower, "upper": upper, "n": n}


def build_item_popularity(ratings_df: pd.DataFrame, movie_col: str = "movieId") -> Dict[str, int]:
    counts = ratings_df[movie_col].astype(str).value_counts().to_dict()
    return {str(k): int(v) for k, v in counts.items()}


def build_genre_features(movies_df: pd.DataFrame) -> Dict[str, Set[str]]:
    """movies.csv has a pipe-separated genres column. Returns {movieId: {genre, ...}}."""
    out: Dict[str, Set[str]] = {}
    for _, row in movies_df.iterrows():
        genres = str(row.get("genres", "") or "")
        tokens = {g for g in genres.split("|") if g and g != "(no genres listed)"}
        out[str(row["movieId"])] = tokens
    return out
