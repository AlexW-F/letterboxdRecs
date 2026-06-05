"""Integration tests for the upload → movieId mapping pipeline.

Exercises ``src/api/upload_mapping.py`` end-to-end on in-memory CSV bytes
with a synthetic catalog. Needs pandas (and fastapi, which upload_mapping
imports) but NOT the model stack. The headline assertion is that a *raw*
Letterboxd export (Name/Year/Rating, no tmdb_id, no API key) now maps to
movieIds instead of being rejected with a 400.

Also runnable as a script: ``python tests/test_upload_mapping.py``.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.api.upload_mapping import (  # noqa: E402
    ratings_csv_to_movielens,
    watched_csv_to_movielens,
)
from src.title_matching import build_local_title_index  # noqa: E402

CATALOG = [
    ("1", "Toy Story (1995)"),
    ("2", "American President, The (1995)"),
    ("3", "Heat (1995)"),
    ("10", "Fast & Furious (2009)"),
]
INDEX = build_local_title_index(CATALOG)
TMDB_TO_MOVIE = {"862": "1", "777": "3"}  # 862→Toy Story, 777→Heat


def _csv(text: str) -> bytes:
    return text.strip().encode("utf-8")


# --------------------------------------------------------------------------
# Keyless raw export — the behavior that used to 400
# --------------------------------------------------------------------------

def test_raw_csv_maps_locally_without_key():
    payload = _csv(
        """
Date,Name,Year,Letterboxd URI,Rating
2020-01-01,Toy Story,1995,uri,5.0
2020-01-02,The American President,1995,uri,4.0
2020-01-03,Some Unknown Film,2050,uri,3.0
"""
    )
    mapping, n_input, n_local, n_tmdb = ratings_csv_to_movielens(
        payload, TMDB_TO_MOVIE, INDEX, cache=None, tmdb_key=None
    )
    assert mapping == {"1": 5.0, "2": 4.0}     # unknown film simply dropped
    assert n_input == 3
    assert n_local == 2
    assert n_tmdb == 0
    assert n_local + n_tmdb == len(mapping)


def test_raw_csv_no_match_returns_empty_not_error():
    # No key, nothing matches → empty mapping, NOT an exception/400.
    payload = _csv(
        """
Name,Year,Rating
Totally Made Up Movie,2099,5.0
"""
    )
    mapping, n_input, n_local, n_tmdb = ratings_csv_to_movielens(
        payload, TMDB_TO_MOVIE, INDEX, cache=None, tmdb_key=None
    )
    assert mapping == {}
    assert (n_input, n_local, n_tmdb) == (1, 0, 0)


# --------------------------------------------------------------------------
# tmdb_id column (authoritative) + fallthrough to local
# --------------------------------------------------------------------------

def test_tmdb_id_column_takes_precedence():
    payload = _csv(
        """
Name,Year,Rating,tmdb_id
Toy Story,1995,5.0,862
"""
    )
    mapping, _, n_local, n_tmdb = ratings_csv_to_movielens(
        payload, TMDB_TO_MOVIE, INDEX, cache=None, tmdb_key=None
    )
    assert mapping == {"1": 5.0}
    assert (n_local, n_tmdb) == (0, 1)   # resolved via id, not title


def test_mixed_resolution_paths():
    payload = _csv(
        """
Name,Year,Rating,tmdb_id
Toy Story,1995,5.0,862
Heat,1995,4.5,
Fast and Furious,2009,3.0,999999
"""
    )
    # Row 2: blank id → local "heat". Row 3: id 999999 not in table → local.
    mapping, n_input, n_local, n_tmdb = ratings_csv_to_movielens(
        payload, TMDB_TO_MOVIE, INDEX, cache=None, tmdb_key=None
    )
    assert mapping == {"1": 5.0, "3": 4.5, "10": 3.0}
    assert n_input == 3
    assert (n_local, n_tmdb) == (2, 1)
    assert n_local + n_tmdb == len(mapping)


# --------------------------------------------------------------------------
# Rating hygiene
# --------------------------------------------------------------------------

def test_missing_ratings_are_dropped():
    payload = _csv(
        """
Name,Year,Rating
Toy Story,1995,
Heat,1995,4.0
"""
    )
    mapping, n_input, n_local, _ = ratings_csv_to_movielens(
        payload, TMDB_TO_MOVIE, INDEX, cache=None, tmdb_key=None
    )
    assert mapping == {"3": 4.0}
    assert n_input == 2 and n_local == 1


def test_csv_without_rating_column_maps_nothing():
    payload = _csv("Name,Year\nToy Story,1995\n")
    mapping, n_input, n_local, n_tmdb = ratings_csv_to_movielens(
        payload, TMDB_TO_MOVIE, INDEX, cache=None, tmdb_key=None
    )
    assert mapping == {} and (n_input, n_local, n_tmdb) == (1, 0, 0)


# --------------------------------------------------------------------------
# watched / watchlist side files
# --------------------------------------------------------------------------

def test_watched_csv_maps_locally_without_key():
    payload = _csv(
        """
Date,Name,Year,Letterboxd URI
2020,Toy Story,1995,uri
2020,Heat,1995,uri
2020,Unknown,2050,uri
"""
    )
    ids = watched_csv_to_movielens(
        payload, TMDB_TO_MOVIE, INDEX, cache=None, tmdb_key=None
    )
    assert ids == {"1", "3"}


# --------------------------------------------------------------------------
# TMDB-search fallback wiring (no network — search is monkeypatched)
# --------------------------------------------------------------------------

def test_search_fallback_only_hits_local_misses():
    import src.data_enrichment as de

    calls = []

    def fake_search(name, year=None, api_key=None):
        calls.append((name, year))
        return 777  # → movieId "3" via TMDB_TO_MOVIE

    orig = de.search_tmdb_for_movie
    de.search_tmdb_for_movie = fake_search
    try:
        payload = _csv(
            """
Name,Year,Rating
Toy Story,1995,5.0
Mystery Film,2001,4.0
"""
        )
        mapping, _, n_local, n_tmdb = ratings_csv_to_movielens(
            payload, TMDB_TO_MOVIE, INDEX, cache={}, tmdb_key="fake-key"
        )
    finally:
        de.search_tmdb_for_movie = orig

    assert mapping == {"1": 5.0, "3": 4.0}
    assert (n_local, n_tmdb) == (1, 1)
    # Toy Story resolved locally → never searched; only the miss is searched.
    assert calls == [("Mystery Film", 2001)]


if __name__ == "__main__":
    fns = sorted(
        (name, fn)
        for name, fn in globals().items()
        if name.startswith("test_") and callable(fn)
    )
    failures = 0
    for name, fn in fns:
        try:
            fn()
            print(f"PASS {name}")
        except AssertionError as e:
            failures += 1
            print(f"FAIL {name}: {e!r}")
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"ERROR {name}: {e!r}")
    print(f"\n{len(fns) - failures}/{len(fns)} passed")
    sys.exit(1 if failures else 0)
