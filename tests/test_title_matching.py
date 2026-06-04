"""Unit tests for keyless title+year → movieId matching.

Dependency-free (no pandas / no model load) so they run anywhere. Also
runnable as a plain script: ``python tests/test_title_matching.py``.

These validate the matcher *logic* on a synthetic catalog that mirrors the
real MovieLens quirks (trailing articles, alternate titles, diacritics,
remakes). They can't measure real-world recall against the full ml-32m
catalog — that's surfaced live by the "mapped to ml-32m / coverage %" stat
after an upload.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.title_matching import (  # noqa: E402
    build_local_title_index,
    coerce_int,
    match_local,
    normalize_title,
    parse_movielens_title,
    resolve_movie_id,
)

# A synthetic catalog mirroring real MovieLens title quirks.
CATALOG = [
    ("1", "Toy Story (1995)"),
    ("2", "American President, The (1995)"),          # trailing article
    ("3", "Cidade de Deus (City of God) (2002)"),     # alternate title
    ("4", "Seven (a.k.a. Se7en) (1995)"),             # a.k.a. alternate
    ("5", "Amelie (Fabuleux destin d'Amélie Poulain, Le) (2001)"),  # diacritics
    ("6", "Star Wars: Episode IV - A New Hope (1977)"),  # punctuation
    ("7", "(500) Days of Summer (2009)"),             # leading parenthetical
    ("8", "King Kong (1933)"),                        # remake — ambiguous title
    ("9", "King Kong (2005)"),                        # remake — ambiguous title
    ("10", "Fast & Furious (2009)"),                  # ampersand
    ("11", "Léon: The Professional (Léon) (1994)"),   # diacritics + alt
]

INDEX = build_local_title_index(CATALOG)
# tmdb→movieId table for resolve_movie_id precedence tests.
# 862 (Toy Story) maps to catalog id "1"; 999999 is absent on purpose.
TMDB_TO_MOVIE = {"862": "1"}


# --------------------------------------------------------------------------
# normalize_title
# --------------------------------------------------------------------------

def test_normalize_moves_trailing_article():
    assert normalize_title("American President, The") == "the american president"
    assert normalize_title("The American President") == "the american president"


def test_normalize_strips_diacritics_and_lowercases():
    assert normalize_title("Amélie") == "amelie"
    assert normalize_title("LÉON") == "leon"


def test_normalize_collapses_punctuation():
    assert normalize_title("Star Wars: Episode IV - A New Hope") == (
        "star wars episode iv a new hope"
    )


def test_normalize_ampersand_to_and():
    assert normalize_title("Fast & Furious") == normalize_title("Fast and Furious")


def test_normalize_empty_and_none():
    assert normalize_title("") == ""
    assert normalize_title(None) == ""


def test_normalize_non_article_comma_not_moved():
    # "Crazy, Stupid, Love" must not treat "love" as an article.
    assert normalize_title("Crazy, Stupid, Love") == "crazy stupid love"


# --------------------------------------------------------------------------
# parse_movielens_title
# --------------------------------------------------------------------------

def test_parse_extracts_year():
    variants, year = parse_movielens_title("Toy Story (1995)")
    assert year == 1995
    assert "toy story" in variants


def test_parse_alternate_title_variants():
    variants, year = parse_movielens_title("Cidade de Deus (City of God) (2002)")
    assert year == 2002
    assert "cidade de deus" in variants
    assert "city of god" in variants


def test_parse_strips_aka_prefix():
    variants, _ = parse_movielens_title("Seven (a.k.a. Se7en) (1995)")
    assert "seven" in variants
    assert "se7en" in variants


def test_parse_no_year():
    variants, year = parse_movielens_title("Some Obscure Short")
    assert year is None
    assert "some obscure short" in variants


# --------------------------------------------------------------------------
# match_local
# --------------------------------------------------------------------------

def test_match_exact_title_year():
    assert match_local("Toy Story", 1995, INDEX) == "1"


def test_match_natural_article_form():
    assert match_local("The American President", 1995, INDEX) == "2"


def test_match_alternate_title():
    assert match_local("City of God", 2002, INDEX) == "3"
    assert match_local("Cidade de Deus", 2002, INDEX) == "3"


def test_match_aka_form():
    assert match_local("Se7en", 1995, INDEX) == "4"
    assert match_local("Seven", 1995, INDEX) == "4"


def test_match_diacritics_insensitive():
    assert match_local("Amélie", 2001, INDEX) == "5"
    assert match_local("Amelie", 2001, INDEX) == "5"


def test_match_leading_parenthetical_full_form():
    assert match_local("(500) Days of Summer", 2009, INDEX) == "7"


def test_match_ampersand_variant():
    assert match_local("Fast and Furious", 2009, INDEX) == "10"


def test_remake_requires_year():
    # Ambiguous title across two years — must use the year to disambiguate.
    assert match_local("King Kong", 1933, INDEX) == "8"
    assert match_local("King Kong", 2005, INDEX) == "9"


def test_remake_without_year_refuses_to_guess():
    # No year + ambiguous title → refuse rather than mis-map.
    assert match_local("King Kong", None, INDEX) is None


def test_unique_title_matches_without_year():
    # Unambiguous title resolves even when the year is missing/wrong.
    assert match_local("Toy Story", None, INDEX) == "1"
    assert match_local("Toy Story", 1900, INDEX) == "1"


def test_unknown_title_returns_none():
    assert match_local("This Film Does Not Exist", 2020, INDEX) is None


def test_empty_name_returns_none():
    assert match_local("", 1995, INDEX) is None


# --------------------------------------------------------------------------
# resolve_movie_id — precedence
# --------------------------------------------------------------------------

def test_resolve_prefers_tmdb_id():
    mid, how = resolve_movie_id("Toy Story", 1995, 862, TMDB_TO_MOVIE, INDEX)
    assert (mid, how) == ("1", "tmdb")


def test_resolve_falls_back_to_local_when_tmdb_misses_catalog():
    # tmdb_id 999999 isn't in the table → fall through to the local match.
    mid, how = resolve_movie_id("Toy Story", 1995, 999999, TMDB_TO_MOVIE, INDEX)
    assert (mid, how) == ("1", "local")


def test_resolve_local_when_no_tmdb_id():
    mid, how = resolve_movie_id("City of God", 2002, None, TMDB_TO_MOVIE, INDEX)
    assert (mid, how) == ("3", "local")


def test_resolve_returns_none_for_unresolvable():
    mid, how = resolve_movie_id("Nonexistent", 2020, None, TMDB_TO_MOVIE, INDEX)
    assert (mid, how) == (None, None)


# --------------------------------------------------------------------------
# coerce_int
# --------------------------------------------------------------------------

def test_coerce_int_variants():
    assert coerce_int("1995") == 1995
    assert coerce_int(1995.0) == 1995
    assert coerce_int(550) == 550
    assert coerce_int(None) is None
    assert coerce_int("") is None
    assert coerce_int("not a number") is None
    assert coerce_int(float("nan")) is None


if __name__ == "__main__":
    # Run without pytest: execute every test_* function and report.
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
