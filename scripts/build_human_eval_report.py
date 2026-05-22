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

from src.content_features import ContentFeatures
from src.evaluation import build_genre_features
from src.group_reranker import GROUP_STRATEGIES, GroupReranker
from src.recommendations import load_user_data_with_tmdb, load_watched_movies_with_tmdb
from src.reranking import (
    ALSScorer,
    MODE_WEIGHTS,
    PopularityModel,
    Reranker,
    SVDScorer,
)

import os

OUTPUT = PROJECT_ROOT / "evaluation_results" / os.getenv("REPORT_NAME", "human_eval_phase2_5.md")
METRICS = PROJECT_ROOT / "evaluation_results" / os.getenv("METRICS_NAME", "phase2_5_genome_eval.json")

SVD_PATH = PROJECT_ROOT / "models" / "svd_full.pkl"
ALS_PATH = PROJECT_ROOT / "models" / "als_full.pkl"
MOVIES_PATH = PROJECT_ROOT / "ml-32m" / "movies.csv"
RATINGS_PATH = PROJECT_ROOT / "ml-32m" / "ratings.csv"
LINKS_PATH = PROJECT_ROOT / "ml-32m" / "links.csv"
CONTENT_PATH = PROJECT_ROOT / "data" / os.getenv("CONTENT_NAME", "content_genome")


def main() -> int:
    svd = SVDScorer.from_path(SVD_PATH)
    als = ALSScorer.from_path(ALS_PATH)
    movies = pd.read_csv(MOVIES_PATH)
    ratings = pd.read_csv(RATINGS_PATH)
    pop = PopularityModel(ratings)
    gf = build_genre_features(movies)
    content = ContentFeatures.load(CONTENT_PATH) if CONTENT_PATH.with_suffix(".npz").exists() else None
    rr = Reranker(svd, als, pop, movies, gf, content_features=content)
    gr = GroupReranker(rr)

    real_user = load_user_data_with_tmdb(
        str(PROJECT_ROOT / "alex_data" / "ratings_with_tmdb.csv"),
        str(LINKS_PATH),
    )
    real_watched = load_watched_movies_with_tmdb(
        str(PROJECT_ROOT / "alex_data" / "watched_with_tmdb.csv"),
        str(LINKS_PATH),
    )
    # Watched-only set (watched but never rated) — these are the ones that
    # need explicit exclusion; rated films are already excluded via user_ratings.
    watched_only = real_watched - set(real_user.keys())
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
    members = [
        ("alex", real_user, watched_only),
        ("friend_A", friend_a, None),
        ("friend_B", friend_b, None),
    ]

    out: list[str] = []
    out.append("# Phase 2.5 Human-Eval Check-in — Tag Genome 2021 content features")
    out.append("")
    out.append("**Purpose:** the content scorer is now Tag Genome 2021 — GroupLens's curated, deep-learning-fitted tag-relevance dataset (1,084 tags × 9,734 movies, 10.5M relevance scores in [0, 1]). Drop-in replacement for the noisy user-tag TF-IDF. Offline metrics show clear gains on group strategies (the headline use case) and mixed-to-flat on individual modes; qualitative output is the better signal here.")
    out.append("")
    out.append("**Models:** `models/svd_full_slim.pkl` (SVD on ml-32m, trainset slimmed for inference) + `models/als_full.pkl` (implicit ALS on ml-32m) + `data/content_genome.npz` (Tag Genome 2021).")
    out.append(f"**User:** `alex_data/ratings_with_tmdb.csv` ({len(real_user)} ratings mapped to MovieLens 32m, plus {len(watched_only)} watched-but-unrated films also excluded from recs).")
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
        recs = rr.recommend(real_user, watched_movies=watched_only, mode=mode, top_n=10)
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
    out.append("## What I want your eye on (Phase 2)")
    out.append("")
    out.append("1. **Content alignment:** the new content term uses 2M user-generated tags from ml-32m, so each rec is now scored partly on tag/genre similarity to your rated films. Do the modes feel more 'yours' than the Phase 1 version (e.g. world cinema, Ghibli, anime, classic noir clusters showing up if your taste pulls there)?")
    out.append("2. **Modes:** `niche` should now be hidden-gems-with-substance rather than long-tail noise. `serendipitous` should be delightful jumps that still rhyme with your taste. Are they?")
    out.append("3. **Group strategies:** `group_taste_vector` fuses everyone's taste tags into one signal. Does it find movies your group would *actually* watch together, not just compromise picks?")
    out.append("4. **Explanations / 'why' column:** does showing 'Genre overlap' help you trust the rec, or is it too coarse — would you want top-3 contributing rated films named?")
    out.append("5. **The friends in this demo are random MovieLens users with mismatched taste.** A real friend group would have far more overlap; if something looks weird, ask whether it's the algorithm or the mismatched test friends.")
    out.append("6. **Anything missing?** Examples of valid feedback: 'I want the content scorer weighted higher in balanced', 'niche picks too many 70s/80s — would be nice to filter by decade', 'tag noise still slipping through (e.g. tags like \"01 10\")'.")
    out.append("")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("\n".join(out))
    print(f"Wrote {OUTPUT}  ({OUTPUT.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
