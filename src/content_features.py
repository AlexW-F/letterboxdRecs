"""Phase 2 content features.

Builds a TF-IDF representation of each movie from two free signals
that ml-32m already ships with:
- ``movies.csv`` ``genres`` column (pipe-separated)
- ``tags.csv`` user-generated tags (2M rows in ml-32m)

No TMDB API calls required for the baseline — if offline eval shows
meaningful headroom after this, ``upgrade_to_embeddings()`` can swap in
sentence-transformer overview vectors (Phase 2.5) using the same
``ContentFeatures`` API.

Persistence: a single ``data/content_features.npz`` with the sparse
TF-IDF matrix, the vocabulary, and the movieId -> row-index mapping.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, load_npz, save_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize as sk_normalize


_NON_WORD = re.compile(r"[^a-z0-9 ]+")


def _tokenize(text: str) -> List[str]:
    """Lower-case, strip non-word, split on whitespace. Keeps tags like
    'kevin kline' together as two tokens but normalizes 'film noir!' to
    'film noir'."""
    if not isinstance(text, str):
        return []
    s = _NON_WORD.sub(" ", text.lower())
    return [t for t in s.split() if t]


def build_per_movie_documents(
    movies_df: pd.DataFrame,
    tags_df: Optional[pd.DataFrame] = None,
    *,
    movie_col: str = "movieId",
    genre_col: str = "genres",
    tag_col: str = "tag",
) -> Dict[str, str]:
    """For each movie, produce a single text document combining its
    pipe-separated genres and any user-generated tags. Tags repeated by
    many users naturally get higher TF (term frequency), which is exactly
    what TF-IDF wants — popular tags for a movie weigh more.
    """
    docs: Dict[str, List[str]] = {}
    for _, row in movies_df.iterrows():
        mid = str(row[movie_col])
        tokens: List[str] = []
        genres = str(row.get(genre_col, "") or "")
        for g in genres.split("|"):
            if g and g != "(no genres listed)":
                tokens.extend(_tokenize(g))
        docs[mid] = tokens

    if tags_df is not None:
        # Group tags by movieId to avoid Python-level row iteration on 2M rows.
        # The tag column has many forms; treat each tag application as one
        # occurrence to let TF reflect popularity.
        tags_df = tags_df.dropna(subset=[tag_col])
        for mid_raw, tags in tags_df.groupby(movie_col)[tag_col]:
            mid = str(mid_raw)
            if mid not in docs:
                docs[mid] = []
            for t in tags:
                docs[mid].extend(_tokenize(str(t)))

    return {mid: " ".join(toks) for mid, toks in docs.items()}


@dataclass
class ContentFeatures:
    """Sparse TF-IDF features over movies, with helpers for new-user
    taste vectors and candidate scoring."""

    tfidf: csr_matrix             # shape (n_movies, n_features), L2-normalized
    movie_ids: np.ndarray         # int64 movie ids, length n_movies
    vocabulary: List[str]
    index_of: Dict[str, int]      # str(movieId) -> row index
    n_features: int

    @classmethod
    def from_genome(
        cls,
        genome_scores_df: pd.DataFrame,
        *,
        movie_col: str = "item_id",
        tag_col: str = "tag",
        score_col: str = "score",
    ) -> "ContentFeatures":
        """Build a ContentFeatures from the Tag Genome 2021 long-form scores.

        The standalone genome-2021 dataset (``movie_dataset_public_final/scores/tagdl.csv``)
        is laid out as ``tag, item_id, score`` triples where ``tag`` is the
        literal tag string and ``score`` is the deep-learning-fitted relevance
        in [0, 1]. We treat each unique tag as its own vocabulary entry.

        Output matrix is sparse-CSR, L2-normalized rows so cosine collapses
        to dot product (matching the existing TF-IDF contract).
        """
        scores = genome_scores_df.dropna(subset=[movie_col, tag_col, score_col])
        # Stable orderings so the index_of map is reproducible.
        movie_ids = np.array(sorted(scores[movie_col].astype(np.int64).unique()), dtype=np.int64)
        unique_tags = sorted(scores[tag_col].astype(str).unique())
        mid_to_row = {int(m): i for i, m in enumerate(movie_ids)}
        tag_to_col = {t: i for i, t in enumerate(unique_tags)}

        rows = scores[movie_col].astype(np.int64).map(mid_to_row).values
        cols = scores[tag_col].astype(str).map(tag_to_col).values
        data = scores[score_col].astype(np.float32).values
        n_movies = len(movie_ids)
        n_tags = len(unique_tags)
        matrix = csr_matrix((data, (rows, cols)), shape=(n_movies, n_tags), dtype=np.float32)
        matrix = sk_normalize(matrix, norm="l2", axis=1)

        return cls(
            tfidf=matrix,
            movie_ids=movie_ids,
            vocabulary=list(unique_tags),
            index_of={str(int(m)): i for i, m in enumerate(movie_ids)},
            n_features=n_tags,
        )

    @classmethod
    def from_dataframes(
        cls,
        movies_df: pd.DataFrame,
        tags_df: Optional[pd.DataFrame] = None,
        *,
        max_features: int = 20000,
        min_df: int = 2,
    ) -> "ContentFeatures":
        docs = build_per_movie_documents(movies_df, tags_df)
        # Stable ordering by movieId so the row index is reproducible.
        ordered_mids = sorted(docs.keys(), key=lambda x: int(x))
        texts = [docs[mid] for mid in ordered_mids]

        vec = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            ngram_range=(1, 2),
            sublinear_tf=True,
            norm="l2",
        )
        matrix = vec.fit_transform(texts).astype(np.float32)
        return cls(
            tfidf=matrix,
            movie_ids=np.array([int(m) for m in ordered_mids], dtype=np.int64),
            vocabulary=list(vec.get_feature_names_out()),
            index_of={mid: i for i, mid in enumerate(ordered_mids)},
            n_features=matrix.shape[1],
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, base_path: Path) -> None:
        base_path = Path(base_path)
        base_path.parent.mkdir(parents=True, exist_ok=True)
        save_npz(base_path.with_suffix(".npz"), self.tfidf)
        meta = {
            "movie_ids": self.movie_ids.tolist(),
            "vocabulary": self.vocabulary,
        }
        base_path.with_suffix(".json").write_text(json.dumps(meta))

    @classmethod
    def load(cls, base_path: Path) -> "ContentFeatures":
        base_path = Path(base_path)
        matrix = load_npz(base_path.with_suffix(".npz")).astype(np.float32).tocsr()
        meta = json.loads(base_path.with_suffix(".json").read_text())
        movie_ids = np.asarray(meta["movie_ids"], dtype=np.int64)
        vocabulary = list(meta["vocabulary"])
        index_of = {str(int(m)): i for i, m in enumerate(movie_ids)}
        return cls(
            tfidf=matrix,
            movie_ids=movie_ids,
            vocabulary=vocabulary,
            index_of=index_of,
            n_features=matrix.shape[1],
        )

    # ------------------------------------------------------------------
    # User taste vector
    # ------------------------------------------------------------------

    def taste_vector(self, user_ratings: Dict[str, float]) -> Optional[np.ndarray]:
        """Build a length-n_features taste vector from a user's ratings.

        Each rated movie contributes its TF-IDF row, weighted by
        ``rating - user_mean`` so that *preference* (above your own mean)
        drives the vector — not mere exposure. The result is L2-normalized.

        Returns None if no rated movies are in the catalog or all
        weights end up near zero.
        """
        if not user_ratings:
            return None
        weights: List[float] = []
        rows: List[csr_matrix] = []
        mean_rating = float(np.mean(list(user_ratings.values())))
        for mid, rating in user_ratings.items():
            idx = self.index_of.get(str(mid))
            if idx is None:
                continue
            w = float(rating) - mean_rating
            if abs(w) < 1e-9:
                continue
            weights.append(w)
            rows.append(self.tfidf.getrow(idx))
        if not rows:
            return None
        weighted = sum(w * row for w, row in zip(weights, rows))
        dense = np.asarray(weighted.todense()).ravel().astype(np.float32)
        norm = np.linalg.norm(dense)
        if norm < 1e-9:
            return None
        return dense / norm

    # ------------------------------------------------------------------
    # Candidate scoring (cosine similarity)
    # ------------------------------------------------------------------

    def score(
        self,
        taste_vec: np.ndarray,
        candidate_ids: Iterable[str],
    ) -> Dict[str, float]:
        """Cosine similarity between the (L2-normalized) taste vector and
        each candidate's TF-IDF row. Returns ``{movieId: cosine_in [-1,1]}``.

        Candidates not in the catalog are silently omitted — callers
        should treat missing as "no content signal" rather than 0.
        """
        out: Dict[str, float] = {}
        for mid in candidate_ids:
            idx = self.index_of.get(str(mid))
            if idx is None:
                continue
            row = self.tfidf.getrow(idx).toarray().ravel()
            # Since both taste_vec and row are L2-normalized, dot is cosine.
            out[str(mid)] = float(np.dot(taste_vec, row))
        return out

    def group_taste_vector(
        self,
        members_ratings: Sequence[Dict[str, float]],
        combination: str = "mean",
    ) -> Optional[np.ndarray]:
        """Fuse multiple member taste vectors. ``combination``:
        - "mean"          — average (default)
        - "min"           — element-wise min (least-misery semantics)
        - "weighted_mean" — weighted by sqrt(n_ratings)
        """
        per_member: List[np.ndarray] = []
        weights: List[float] = []
        for ratings in members_ratings:
            v = self.taste_vector(ratings)
            if v is not None:
                per_member.append(v)
                weights.append(float(len(ratings) ** 0.5))
        if not per_member:
            return None
        stacked = np.stack(per_member, axis=0)
        if combination == "mean":
            fused = stacked.mean(axis=0)
        elif combination == "min":
            fused = stacked.min(axis=0)
        elif combination == "weighted_mean":
            w = np.asarray(weights, dtype=np.float32)
            w = w / (w.sum() or 1.0)
            fused = (stacked * w[:, None]).sum(axis=0)
        else:
            raise ValueError(f"unknown combination {combination!r}")
        norm = np.linalg.norm(fused)
        if norm < 1e-9:
            return None
        return fused / norm
