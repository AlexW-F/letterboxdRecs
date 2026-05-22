"""Generate evaluation_results/human_eval_phase1.md.

Walk the user through 10 individual recs per mode + 10 group recs per
strategy for a synthetic 3-friend group, plus before/after metric tables
to substantiate the claim that Phase 1 actually improved things.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import build_genre_features
from src.group_reranker import GROUP_STRATEGIES, GroupReranker
from src.recommendations import load_user_data_with_tmdb
from src.reranking import (
    ALSScorer,
    MODE_WEIGHTS,
    PopularityModel,
    Reranker,
    SVDScorer,
)

OUTPUT = PROJECT_ROOT / "evaluation_results" / "human_eval_phase1_full.md"
METRICS = PROJECT_ROOT / "evaluation_results" / "phase1_eval_full.json"

SVD_PATH = PROJECT_ROOT / "models" / "svd_full.pkl"
ALS_PATH = PROJECT_ROOT / "models" / "als_full.pkl"
MOVIES_PATH = PROJECT_ROOT / "ml-32m" / "movies.csv"
RATINGS_PATH = PROJECT_ROOT / "ml-32m" / "ratings.csv"
LINKS_PATH = PROJECT_ROOT / "ml-32m" / "links.csv"


def main() -> int:
    svd = SVDScorer.from_path(SVD_PATH)
    als = ALSScorer.from_path(ALS_PATH)
    movies = pd.read_csv(MOVIES_PATH)
    ratings = pd.read_csv(RATINGS_PATH)
    pop = PopularityModel(ratings)
    gf = build_genre_features(movies)
    rr = Reranker(svd, als, pop, movies, gf)
    gr = GroupReranker(rr)

    real_user = load_user_data_with_tmdb(
        str(PROJECT_ROOT / "alex_data" / "ratings_with_tmdb.csv"),
        str(LINKS_PATH),
    )
    ratings["userId"] = ratings["userId"].astype(int)
    ratings["movieId"] = ratings["movieId"].astype(str)
    counts = ratings.groupby("userId").size()
    others = counts[counts >= 100].sample(n=2, random_state=42).index.tolist()
    friend_a = {
        str(k): float(v)
        for k, v in ratings[ratings["userId"] == others[0]]
        .set_index("movieId")["rating"]
        .to_dict()
        .items()
    }
    friend_b = {
        str(k): float(v)
        for k, v in ratings[ratings["userId"] == others[1]]
        .set_index("movieId")["rating"]
        .to_dict()
        .items()
    }
    members = [("alex", real_user, None), ("friend_A", friend_a, None), ("friend_B", friend_b, None)]

    out: list[str] = []
    out.append("# Phase 1 Human-Eval Check-in — ml-32m (full catalog)")
    out.append("")
    out.append("**Purpose:** scan a sample of individual and group recommendations and flag anything that looks off. The offline metrics say the re-ranker improved (table at the bottom), but only you can say if the *vibes* are right.")
    out.append("")
    out.append("**Models:** `models/svd_full.pkl` + `models/als_full.pkl` — both trained on **ml-32m** (87k items, 200k users). This replaces the Phase 1 initial run on ml-latest-small.")
    out.append(f"**User:** `alex_data/ratings_with_tmdb.csv` ({len(real_user)} ratings mapped to MovieLens 32m).")
    out.append(f"**Synthetic friends:** two random MovieLens users with {len(friend_a)} and {len(friend_b)} ratings respectively.")
    out.append("")
    out.append("---")
    out.append("")

    # Individual recs per mode
    out.append("## Individual recommendations — by mode")
    out.append("")
    for mode in MODE_WEIGHTS:
        out.append(f"### Mode: `{mode}`")
        out.append("")
        out.append("| # | Score | Pop | Title | Genre overlap | Source |")
        out.append("|--:|------:|:---:|:------|:--------------|:------:|")
        recs = rr.recommend(real_user, mode=mode, top_n=10)
        for i, c in enumerate(recs, 1):
            out.append(
                f"| {i} | {c.score:.3f} | {c.explanation.popularity_tier} | "
                f"{c.title} | {c.explanation.dominant_genre_overlap} | "
                f"{c.explanation.source} |"
            )
        out.append("")

    out.append("---")
    out.append("")

    # Group recs per strategy
    out.append("## Group recommendations — 3-friend group, mode=`balanced`")
    out.append("")
    out.append("Each row shows the predicted score per member (normalized 0-1) and the fairness coefficient of variation (lower = more uniform agreement).")
    out.append("")
    for strategy in GROUP_STRATEGIES:
        out.append(f"### Strategy: `{strategy}`")
        out.append("")
        out.append("| # | Score | Fair | Title | Pop | Per-member scores |")
        out.append("|--:|------:|-----:|:------|:---:|:------------------|")
        recs = gr.recommend(members, strategy=strategy, mode="balanced", top_n=10)
        for i, c in enumerate(recs, 1):
            per = ", ".join(f"{n}={v:.2f}" for n, v in c.per_member_score.items())
            out.append(
                f"| {i} | {c.score:.3f} | {c.fairness:.2f} | "
                f"{c.title} | {c.explanation.popularity_tier} | {per} |"
            )
        out.append("")

    out.append("---")
    out.append("")

    # Metrics summary
    if METRICS.exists():
        data = json.loads(METRICS.read_text())
        out.append("## Offline metrics — baseline vs Phase 1 re-ranker")
        out.append("")
        out.append("### Individual (averaged across alex + 20 sampled MovieLens users)")
        out.append("")
        out.append("| Variant | NDCG@10 | Recall@50 | Coverage | Gini popularity (lower = less popular bias) | Diversity@10 |")
        out.append("|:--------|--------:|----------:|---------:|--------------------------------------------:|-------------:|")
        if "baseline" in data:
            b = data["baseline"]["individual"]
            out.append(f"| baseline SVD++ | {b['ndcg_at_10']:.4f} | {b['recall_at_50']:.4f} | {b['catalog_coverage_at_50']:.4f} | {b['gini_popularity_at_50']:.4f} | {b['intra_list_diversity_at_10']:.4f} |")
        if "reranker" in data:
            for mode, m in data["reranker"]["individual"].items():
                out.append(f"| reranker / {mode} | {m['ndcg_at_10']:.4f} | {m['recall_at_50']:.4f} | {m['catalog_coverage_at_50']:.4f} | {m['gini_popularity_at_50']:.4f} | {m['intra_list_diversity_at_10']:.4f} |")
        out.append("")

        out.append("### Group (3 members: alex + 2 random MovieLens users)")
        out.append("")
        out.append("| Variant | Strategy | Per-member NDCG@10 | Per-member Recall@50 | Fairness CV | Diversity@10 |")
        out.append("|:--------|:---------|------------------:|---------------------:|------------:|-------------:|")
        if "baseline" in data:
            for s, m in data["baseline"]["group"].items():
                out.append(f"| baseline | {s} | {m['avg_per_member_ndcg_at_10']:.4f} | {m['avg_per_member_recall_at_50']:.4f} | {m['fairness_cv_at_10']:.4f} | {m['intra_list_diversity_at_10']:.4f} |")
        if "reranker" in data:
            for s, modes in data["reranker"]["group"].items():
                for mode, m in modes.items():
                    out.append(f"| reranker | {s} / {mode} | {m['avg_per_member_ndcg_at_10']:.4f} | {m['avg_per_member_recall_at_50']:.4f} | {m['fairness_cv_at_10']:.4f} | {m['intra_list_diversity_at_10']:.4f} |")
        out.append("")

    out.append("---")
    out.append("")
    out.append("## What I want your eye on")
    out.append("")
    out.append("1. **Modes:** does `niche` feel niche (would you describe these as 'hidden gems' you don't already know)? Does `popular` lean too mainstream, or about right? Are `serendipitous` picks delightful-surprising or random-noise?")
    out.append("2. **Group strategies:** `group_taste_vector` is the new 6th — finding movies the fused taste vector predicts highly, not just averaging the individual predictions. Does it surface movies that feel like *the group's* picks?")
    out.append("3. **Explanations:** the 'Genre overlap' column shows the top genres in your rated set that match the recommendation. Does this 'why' make sense, or feel post-hoc?")
    out.append("4. **The friends in this demo are random MovieLens users with mismatched taste.** A real friend group would have far more overlap. If something looks weird, ask whether it's the algorithm or the mismatched test friends.")
    out.append("5. **Anything missing?** The point of the Phase 1 check-in is to surface things the offline metrics can't catch. Examples of valid feedback: 'all the niche picks are 70s/80s, I want more recent', 'the explanations should mention specific movies not just genres', 'cold-start fallback fires too aggressively'.")
    out.append("")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("\n".join(out))
    print(f"Wrote {OUTPUT}  ({OUTPUT.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
