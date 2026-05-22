"""Build a 3D UMAP projection of the ALS item-factor space.

What you get: an interactive HTML at
``evaluation_results/movie_space_alex.html`` where each point is a movie,
positioned by ALS latent factors projected to 3D. Color encodes primary
genre. Alex's rated films are highlighted; her ALS fold-in vector is
plotted as a large star so you can see which clusters of movies she sits
near in the latent space.

Useful for spot-checking why the recommender clumps things the way it
does — if Pulp Fiction and Reservoir Dogs sit next to each other, that
visually justifies why one rec implies the other. Also useful for
diagnosing weird neighbors ("why is anime sitting in the romcom cluster?").
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.recommendations import (
    load_user_data_with_tmdb,
    load_watched_movies_with_tmdb,
)
from src.reranking import ALSScorer


# Order matters — first-match wins for "primary genre" tagging.
GENRE_PRIORITY = [
    "Documentary",
    "Animation",
    "Horror",
    "Sci-Fi",
    "Fantasy",
    "Thriller",
    "Crime",
    "Mystery",
    "Action",
    "Adventure",
    "Western",
    "War",
    "Musical",
    "Film-Noir",
    "Romance",
    "Comedy",
    "Drama",
    "Children",
]

GENRE_COLOR = {
    "Documentary": "#7f7f7f",
    "Animation": "#17becf",
    "Horror": "#8c564b",
    "Sci-Fi": "#1f77b4",
    "Fantasy": "#9467bd",
    "Thriller": "#d62728",
    "Crime": "#5f4f4b",
    "Mystery": "#5b3f7a",
    "Action": "#ff7f0e",
    "Adventure": "#ffbb78",
    "Western": "#bcbd22",
    "War": "#8d553b",
    "Musical": "#e377c2",
    "Film-Noir": "#1a1a1a",
    "Romance": "#f7c6c7",
    "Comedy": "#2ca02c",
    "Drama": "#aec7e8",
    "Children": "#98df8a",
    "Other": "#cccccc",
}


def primary_genre(genres_str: str) -> str:
    parts = {g for g in str(genres_str or "").split("|") if g and g != "(no genres listed)"}
    for g in GENRE_PRIORITY:
        if g in parts:
            return g
    return "Other"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--als", type=Path, default=PROJECT_ROOT / "models" / "als_full.pkl")
    p.add_argument("--movies", type=Path, default=PROJECT_ROOT / "ml-32m" / "movies.csv")
    p.add_argument("--ratings", type=Path, default=PROJECT_ROOT / "ml-32m" / "ratings.csv")
    p.add_argument("--links", type=Path, default=PROJECT_ROOT / "ml-32m" / "links.csv")
    p.add_argument("--user-ratings", type=Path,
                   default=PROJECT_ROOT / "alex_data" / "ratings_with_tmdb.csv")
    p.add_argument("--user-watched", type=Path,
                   default=PROJECT_ROOT / "alex_data" / "watched_with_tmdb.csv")
    p.add_argument("--output", type=Path,
                   default=PROJECT_ROOT / "evaluation_results" / "movie_space_alex.html")
    p.add_argument("--max-points", type=int, default=8000,
                   help="cap the number of catalog points to keep the HTML lean")
    p.add_argument("--min-rating-count", type=int, default=100,
                   help="only include films with >= N ratings in the background sample")
    p.add_argument("--user-label", default="alex",
                   help="display name for the user's marker / hover tooltip")
    p.add_argument("--n-neighbors", type=int, default=25)
    p.add_argument("--min-dist", type=float, default=0.15)
    p.add_argument("--random-seed", type=int, default=42)
    args = p.parse_args()

    print(f"Loading ALS model {args.als}...")
    als = ALSScorer.from_path(args.als)
    item_factors = np.asarray(als.model.item_factors, dtype=np.float32)
    n_items = item_factors.shape[0]
    print(f"  item factors: {item_factors.shape}")

    print(f"Loading {args.movies}...")
    movies_df = pd.read_csv(args.movies)
    movies_df["movieId"] = movies_df["movieId"].astype(np.int64)
    title_of: Dict[int, str] = dict(zip(movies_df["movieId"], movies_df["title"]))
    genre_of: Dict[int, str] = {
        int(mid): primary_genre(g)
        for mid, g in zip(movies_df["movieId"], movies_df["genres"])
    }

    print(f"Loading {args.ratings} (popularity counts only)...")
    ratings_df = pd.read_csv(args.ratings, usecols=["movieId"])
    pop_counts = ratings_df["movieId"].value_counts().to_dict()
    del ratings_df

    print(f"Loading user data...")
    user_ratings = load_user_data_with_tmdb(str(args.user_ratings), str(args.links))
    user_watched = (
        load_watched_movies_with_tmdb(str(args.user_watched), str(args.links))
        if args.user_watched.exists() else set()
    )
    user_rated_ids = {int(mid) for mid in user_ratings.keys()}
    user_watched_only = {int(mid) for mid in user_watched} - user_rated_ids

    # ----- select which items to project -----
    # All ALS-known items, mapped back to MovieLens IDs.
    als_ids = als.item_ids.astype(np.int64)
    in_als = set(int(m) for m in als_ids)
    eligible = [
        m for m in als_ids
        if int(m) in genre_of and pop_counts.get(int(m), 0) >= args.min_rating_count
    ]
    # Always include user's universe regardless of popularity.
    must_include = {int(m) for m in (user_rated_ids | user_watched_only) if int(m) in in_als}
    print(f"  eligible by popularity floor: {len(eligible):,}, "
          f"must-include from user: {len(must_include):,}")

    rng = np.random.default_rng(args.random_seed)
    eligible_set = set(int(m) for m in eligible)
    # Sample background; merge with must-include.
    background_pool = list(eligible_set - must_include)
    cap = max(0, args.max_points - len(must_include))
    if len(background_pool) > cap:
        background_pool = rng.choice(background_pool, size=cap, replace=False).tolist()
    selected_ids = sorted(set(int(m) for m in background_pool) | must_include)
    print(f"  selected {len(selected_ids):,} movies for projection")

    selected_idx = np.array(
        [als.index_of[str(m)] for m in selected_ids if str(m) in als.index_of],
        dtype=np.int64,
    )
    selected_ids = [
        int(als.item_ids[i]) for i in selected_idx
    ]
    factors = item_factors[selected_idx]

    # ----- fold in the user (ALS recommend with recalculate_user) -----
    # The user vector lives in the same factor space as items, so we can
    # project it through the same UMAP.transform() and place it as a star.
    print("Folding in user vector...")
    user_vec = als._build_user_vector(user_ratings, user_watched_only)
    if user_vec is None:
        print("  WARN: user has no ALS-mapped ratings/watched; skipping user marker")
        user_factors_3d = None
    else:
        # implicit ALS exposes recalculate_user via recommend(); the model
        # also has a `partial_fit_users` API we could use, but the simplest
        # path is to solve the implicit-ALS user update analytically with
        # the factor matrix we already have.
        # Use the *full* item factors so the user vector reflects the full
        # catalog, then project just selected items + user.
        full_Y = item_factors
        cols = user_vec.indices
        confs = user_vec.data.astype(np.float32)
        # Hu/Koren weighted least-squares: p = (Y^T Y + Y^T (C-I) Y + reg I)^-1 Y^T C 1
        reg = 0.01
        f = full_Y.shape[1]
        YtY = full_Y.T @ full_Y
        Yi = full_Y[cols]                           # (k, f)
        # (C - I) is diagonal with entries confs - 1
        weighted = (Yi * (confs[:, None] - 1.0)).T @ Yi
        A = YtY + weighted + reg * np.eye(f, dtype=np.float32)
        # Y^T * C * 1  = sum over rated items of confs[i] * Y[i]
        b = (Yi * confs[:, None]).sum(axis=0)
        user_p = np.linalg.solve(A, b).astype(np.float32)
        print(f"  user vector built (|.|={np.linalg.norm(user_p):.3f})")

    # ----- UMAP -----
    print(f"Fitting UMAP (n_neighbors={args.n_neighbors}, min_dist={args.min_dist})...")
    t0 = time.time()
    import umap
    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric="cosine",
        random_state=args.random_seed,
    )
    coords = reducer.fit_transform(factors)
    print(f"  UMAP fit in {time.time()-t0:.1f}s; coords shape {coords.shape}")
    if user_vec is not None:
        user_coords = reducer.transform(user_p.reshape(1, -1))[0]
    else:
        user_coords = None

    # ----- assemble dataframe for plotly -----
    print("Building figure...")
    rows: List[Dict] = []
    for mid, coord in zip(selected_ids, coords):
        title = title_of.get(mid, f"#{mid}")
        genre = genre_of.get(mid, "Other")
        pop = pop_counts.get(mid, 0)
        if mid in user_rated_ids:
            kind = "rated"
            rating = float(user_ratings[str(mid)])
            label = f"★{rating} — {title}"
        elif mid in user_watched_only:
            kind = "watched"
            rating = float("nan")
            label = f"(watched, unrated) {title}"
        else:
            kind = "catalog"
            rating = float("nan")
            label = title
        rows.append({
            "movieId": mid, "title": title, "genre": genre, "popularity": pop,
            "x": float(coord[0]), "y": float(coord[1]), "z": float(coord[2]),
            "kind": kind, "rating": rating, "label": label,
        })
    df = pd.DataFrame(rows)

    import plotly.graph_objects as go

    fig = go.Figure()
    # Background catalog (faint, small)
    cat = df[df["kind"] == "catalog"]
    for genre, g in cat.groupby("genre"):
        fig.add_trace(go.Scatter3d(
            x=g["x"], y=g["y"], z=g["z"],
            mode="markers",
            marker=dict(size=2.5, color=GENRE_COLOR.get(genre, "#cccccc"),
                        opacity=0.45, line=dict(width=0)),
            name=genre,
            legendgroup="catalog",
            hovertext=g["label"],
            hoverinfo="text",
            showlegend=True,
        ))

    # Watched-but-unrated (slightly bigger, white outline)
    watched_df = df[df["kind"] == "watched"]
    if not watched_df.empty:
        fig.add_trace(go.Scatter3d(
            x=watched_df["x"], y=watched_df["y"], z=watched_df["z"],
            mode="markers",
            marker=dict(size=4.5, color="#ffffff", opacity=0.6,
                        line=dict(color="#444", width=0.5)),
            name=f"{args.user_label}: watched (unrated)",
            hovertext=watched_df["label"], hoverinfo="text",
        ))

    # Rated by user — color by rating tier
    rated_df = df[df["kind"] == "rated"]
    if not rated_df.empty:
        # Map rating to size and a green→yellow→red continuous color
        sizes = 4.0 + (rated_df["rating"].fillna(3.0) - 1.0) * 1.4   # 5★ -> ~9.6, 1★ -> 4.0
        fig.add_trace(go.Scatter3d(
            x=rated_df["x"], y=rated_df["y"], z=rated_df["z"],
            mode="markers",
            marker=dict(
                size=sizes,
                color=rated_df["rating"], cmin=1.0, cmax=5.0,
                colorscale=[[0.0, "#a83232"], [0.5, "#e8c547"], [1.0, "#3aa84a"]],
                colorbar=dict(title=f"{args.user_label}'s rating", x=1.02, len=0.5),
                line=dict(color="#222", width=0.5), opacity=0.95,
            ),
            name=f"{args.user_label}: rated",
            hovertext=rated_df["label"], hoverinfo="text",
        ))

    # User taste vector position
    if user_coords is not None:
        fig.add_trace(go.Scatter3d(
            x=[float(user_coords[0])], y=[float(user_coords[1])], z=[float(user_coords[2])],
            mode="markers+text",
            marker=dict(size=14, color="#000000", symbol="diamond",
                        line=dict(color="#fff", width=2)),
            text=[args.user_label.upper()],
            textposition="top center",
            name=f"{args.user_label}: taste vector",
            hovertext=f"{args.user_label}'s folded-in ALS position",
            hoverinfo="text",
        ))

    fig.update_layout(
        title=f"3D ALS factor space (UMAP) — {len(df):,} movies",
        scene=dict(xaxis_title="", yaxis_title="", zaxis_title="",
                   xaxis=dict(showticklabels=False),
                   yaxis=dict(showticklabels=False),
                   zaxis=dict(showticklabels=False)),
        legend=dict(itemsizing="constant", y=0.95),
        height=900,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(args.output), include_plotlyjs="cdn")
    size = args.output.stat().st_size / 1024 / 1024
    print(f"Wrote {args.output} ({size:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
