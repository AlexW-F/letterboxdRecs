"""Reusable 3D-projection viz logic for the movie-space page.

Splits the original ``scripts/build_movie_space_viz.py`` into two pieces:

1. **Index build** — fit UMAP on a popularity-thresholded sample of ALS
   item factors once, persist the projection + the fitted reducer.
2. **Per-user render** — fold the user in via the ALS scorer, project
   their latent vector through the same reducer, render a Plotly HTML
   that overlays them on the precomputed background.

The split lets the API serve personalized visualizations in ~1s after
the 30s background fit happens once at startup (or via build script).
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from .reranking import ALSScorer, PopularityModel

GENRE_PRIORITY = [
    "Documentary", "Animation", "Horror", "Sci-Fi", "Fantasy",
    "Thriller", "Crime", "Mystery", "Action", "Adventure",
    "Western", "War", "Musical", "Film-Noir", "Romance",
    "Comedy", "Drama", "Children",
]
GENRE_COLOR = {
    "Documentary": "#7f7f7f", "Animation": "#17becf", "Horror": "#8c564b",
    "Sci-Fi": "#1f77b4", "Fantasy": "#9467bd", "Thriller": "#d62728",
    "Crime": "#5f4f4b", "Mystery": "#5b3f7a", "Action": "#ff7f0e",
    "Adventure": "#ffbb78", "Western": "#bcbd22", "War": "#8d553b",
    "Musical": "#e377c2", "Film-Noir": "#1a1a1a", "Romance": "#f7c6c7",
    "Comedy": "#2ca02c", "Drama": "#aec7e8", "Children": "#98df8a",
    "Other": "#cccccc",
}


def primary_genre(genres_str: str) -> str:
    parts = {g for g in str(genres_str or "").split("|") if g and g != "(no genres listed)"}
    for g in GENRE_PRIORITY:
        if g in parts:
            return g
    return "Other"


@dataclass
class MovieSpaceIndex:
    """Precomputed UMAP background of the ALS item-factor space."""

    reducer: object                # umap.UMAP fitted on item_factors[selected_idx]
    item_factors: np.ndarray       # (n_total_items, n_factors) — full matrix for transform
    item_ids: np.ndarray           # (n_total_items,) — MovieLens movieIds aligned to item_factors
    selected_ids: np.ndarray       # (n_selected,) — movieIds we projected (subset)
    background_coords: np.ndarray  # (n_selected, 3) — UMAP coords for the selected sample
    background_meta: pd.DataFrame  # (n_selected, ...) — title / genre / popularity per row

    @classmethod
    def build(
        cls,
        als: ALSScorer,
        movies_df: pd.DataFrame,
        popularity: PopularityModel,
        *,
        max_points: int = 8000,
        min_rating_count: int = 100,
        n_neighbors: int = 25,
        min_dist: float = 0.15,
        random_seed: int = 42,
    ) -> "MovieSpaceIndex":
        import umap  # heavy import, kept local
        item_factors = np.asarray(als.model.item_factors, dtype=np.float32)
        als_ids = als.item_ids.astype(np.int64)

        # movieId -> primary genre + title lookups
        genre_of = {
            int(mid): primary_genre(g)
            for mid, g in zip(movies_df["movieId"], movies_df["genres"])
        }
        title_of = {
            int(mid): t for mid, t in zip(movies_df["movieId"], movies_df["title"])
        }

        eligible = [int(m) for m in als_ids if popularity.counts.get(str(int(m)), 0) >= min_rating_count
                    and int(m) in genre_of]
        rng = np.random.default_rng(random_seed)
        if len(eligible) > max_points:
            chosen = set(rng.choice(eligible, size=max_points, replace=False).tolist())
        else:
            chosen = set(eligible)
        selected_ids = np.array(sorted(chosen), dtype=np.int64)

        selected_idx = np.array(
            [als.index_of[str(int(m))] for m in selected_ids if str(int(m)) in als.index_of],
            dtype=np.int64,
        )
        # selected_ids may have entries not in als.index_of (shouldn't, but
        # cover the path). Recompute to align.
        selected_ids = np.array(
            [int(als.item_ids[i]) for i in selected_idx], dtype=np.int64,
        )
        sample_factors = item_factors[selected_idx]

        reducer = umap.UMAP(
            n_components=3,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric="cosine",
            random_state=random_seed,
        )
        coords = reducer.fit_transform(sample_factors)

        meta_rows = [
            {
                "movieId": int(m),
                "title": title_of.get(int(m), str(int(m))),
                "genre": genre_of.get(int(m), "Other"),
                "popularity": int(popularity.counts.get(str(int(m)), 0)),
            }
            for m in selected_ids
        ]
        meta = pd.DataFrame(meta_rows)

        return cls(
            reducer=reducer,
            item_factors=item_factors,
            item_ids=als_ids,
            selected_ids=selected_ids,
            background_coords=np.asarray(coords, dtype=np.float32),
            background_meta=meta,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path) -> "MovieSpaceIndex":
        with path.open("rb") as f:
            return pickle.load(f)

    # ------------------------------------------------------------------
    # Per-user projection
    # ------------------------------------------------------------------

    def project_user(
        self,
        als: ALSScorer,
        user_ratings: Dict[str, float],
        watched: Optional[Iterable[str]] = None,
        reg: float = 0.01,
    ) -> Optional[np.ndarray]:
        """Hu/Koren weighted-least-squares fold-in against the full item
        matrix, then UMAP.transform to project into the background space."""
        user_vec = als._build_user_vector(user_ratings, watched)
        if user_vec is None:
            return None
        Y = self.item_factors
        cols = user_vec.indices
        confs = user_vec.data.astype(np.float32)
        Yi = Y[cols]
        YtY = Y.T @ Y
        weighted = (Yi * (confs[:, None] - 1.0)).T @ Yi
        A = YtY + weighted + reg * np.eye(Y.shape[1], dtype=np.float32)
        b = (Yi * confs[:, None]).sum(axis=0)
        user_p = np.linalg.solve(A, b).astype(np.float32)
        return self.reducer.transform(user_p.reshape(1, -1))[0]

    def project_movies(self, movie_ids: Iterable[str]) -> Dict[str, np.ndarray]:
        """Project a list of movies that may or may not be in the background
        sample. Uses ALS factors and the fitted reducer (the items have to
        exist in als.item_ids though — guaranteed for the user's rated set
        if they were folded in successfully)."""
        out: Dict[str, np.ndarray] = {}
        # Map from movieId -> row in item_factors
        id_to_idx = {int(mid): i for i, mid in enumerate(self.item_ids)}
        idxs: List[int] = []
        keep_ids: List[str] = []
        for mid in movie_ids:
            try:
                idx = id_to_idx.get(int(mid))
            except (TypeError, ValueError):
                continue
            if idx is None:
                continue
            idxs.append(idx)
            keep_ids.append(str(mid))
        if not idxs:
            return out
        factors = self.item_factors[np.array(idxs, dtype=np.int64)]
        coords = self.reducer.transform(factors)
        for mid, c in zip(keep_ids, coords):
            out[mid] = np.asarray(c, dtype=np.float32)
        return out


# ---------------------------------------------------------------------------
# Plotly HTML rendering
# ---------------------------------------------------------------------------


def render_personalized_html(
    index: MovieSpaceIndex,
    user_ratings: Dict[str, float],
    user_position: Optional[np.ndarray],
    rated_coords: Dict[str, np.ndarray],
    watched_only_coords: Dict[str, np.ndarray],
    *,
    user_label: str = "you",
) -> str:
    """Render the personalized 3D figure as a self-contained HTML string."""
    import plotly.graph_objects as go

    bg = index.background_meta.copy()
    bg["x"] = index.background_coords[:, 0]
    bg["y"] = index.background_coords[:, 1]
    bg["z"] = index.background_coords[:, 2]
    bg["label"] = bg["title"]

    fig = go.Figure()
    for genre, g in bg.groupby("genre"):
        fig.add_trace(go.Scatter3d(
            x=g["x"], y=g["y"], z=g["z"],
            mode="markers",
            marker=dict(size=2.5, color=GENRE_COLOR.get(genre, "#cccccc"),
                        opacity=0.45, line=dict(width=0)),
            name=genre,
            hovertext=g["label"], hoverinfo="text",
        ))

    # Watched but unrated — white-outlined
    if watched_only_coords:
        xs = [c[0] for c in watched_only_coords.values()]
        ys = [c[1] for c in watched_only_coords.values()]
        zs = [c[2] for c in watched_only_coords.values()]
        hovers = [
            f"(watched, unrated) {index.background_meta[index.background_meta['movieId']==int(mid)]['title'].squeeze() if int(mid) in set(index.background_meta['movieId']) else mid}"
            for mid in watched_only_coords.keys()
        ]
        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs, mode="markers",
            marker=dict(size=4.5, color="#ffffff", opacity=0.6,
                        line=dict(color="#444", width=0.5)),
            name=f"{user_label}: watched (unrated)",
            hovertext=hovers, hoverinfo="text",
        ))

    # Rated by user — color by rating
    if rated_coords:
        title_by_id = dict(zip(index.background_meta["movieId"].astype(int),
                               index.background_meta["title"]))
        xs, ys, zs, ratings_arr, hovers, sizes = [], [], [], [], [], []
        for mid, coord in rated_coords.items():
            r = float(user_ratings.get(str(mid), user_ratings.get(int(mid), 3.0)))
            xs.append(float(coord[0]))
            ys.append(float(coord[1]))
            zs.append(float(coord[2]))
            ratings_arr.append(r)
            sizes.append(4.0 + (r - 1.0) * 1.4)
            t = title_by_id.get(int(mid), str(mid))
            hovers.append(f"★{r:.1f} — {t}")
        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs, mode="markers",
            marker=dict(
                size=sizes,
                color=ratings_arr, cmin=1.0, cmax=5.0,
                colorscale=[[0.0, "#a83232"], [0.5, "#e8c547"], [1.0, "#3aa84a"]],
                colorbar=dict(
                    title=dict(text=f"{user_label}'s rating", side="top"),
                    orientation="h",
                    x=0.5, xanchor="center",
                    y=-0.05, yanchor="top",
                    len=0.5, thickness=12,
                ),
                line=dict(color="#222", width=0.5),
                opacity=0.95,
            ),
            name=f"{user_label}: rated",
            hovertext=hovers, hoverinfo="text",
        ))

    if user_position is not None:
        fig.add_trace(go.Scatter3d(
            x=[float(user_position[0])],
            y=[float(user_position[1])],
            z=[float(user_position[2])],
            mode="markers+text",
            marker=dict(size=14, color="#000000", symbol="diamond",
                        line=dict(color="#fff", width=2)),
            text=[user_label.upper()],
            textposition="top center",
            name=f"{user_label}: taste vector",
            hovertext=f"{user_label}'s folded-in ALS position",
            hoverinfo="text",
        ))

    fig.update_layout(
        title=f"Your taste in 3D — {len(bg):,} movie background + {len(rated_coords)} rated + {len(watched_only_coords)} watched",
        scene=dict(
            xaxis_title="", yaxis_title="", zaxis_title="",
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False),
            zaxis=dict(showticklabels=False),
        ),
        legend=dict(
            itemsizing="constant",
            x=1.02, xanchor="left",
            y=1.0, yanchor="top",
            bgcolor="rgba(11,13,17,0.5)",
            font=dict(size=11),
        ),
        margin=dict(l=10, r=160, t=50, b=90),
        paper_bgcolor="#0b0d11",
        font=dict(color="#e6e8ec"),
        autosize=True,
        height=820,
    )
    return fig.to_html(include_plotlyjs="cdn", full_html=True)
