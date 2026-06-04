"""Keyless title+year → MovieLens movieId matching.

The upload pipeline maps a user's films to MovieLens ``movieId`` so the
recommender can score them. The authoritative join is ``tmdb_id`` →
``links.csv`` → ``movieId``, but a raw Letterboxd ``ratings.csv`` export
carries only ``Name`` + ``Year`` (no TMDB id). Historically that meant the
upload was *rejected* unless a server-side ``TMDB_API_KEY`` was configured
to search TMDB for each title.

This module removes that hard dependency: it matches ``(Name, Year)``
directly against the ml-32m catalog titles, which are already loaded in
memory. It is:

- **keyless** — no third-party API, no network, no rate limits;
- **instant** — a dict lookup per row vs. a TMDB search round-trip;
- **high precision** — exact normalized title + year, and it refuses to
  guess when a title is ambiguous (e.g. remakes) without a year.

It does *not* replace the TMDB path. ``resolve_movie_id`` prefers an
authoritative ``tmdb_id`` when present, falls back to local matching, and
leaves TMDB *search* enrichment to the caller as a last resort for rows
that have neither. With a key set you get the union of both; without one,
uploads still work for everything in the catalog.

The catalog title format (MovieLens ``movies.csv``) is quirky:

    "Toy Story (1995)"
    "American President, The (1995)"          # trailing article
    "Cidade de Deus (City of God) (2002)"     # alternate title in parens
    "Seven (a.k.a. Se7en) (1995)"             # a.k.a. alternate
    "Amelie (Fabuleux destin d'Amélie..., Le) (2001)"  # diacritics + alt

``normalize_title`` / ``parse_movielens_title`` fold all of these into a
small set of comparable keys so a natural Letterboxd title
("The American President", "Se7en", "Amélie") lines up with the catalog.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple

# Leading articles MovieLens moves to the end of a title ("Matrix, The").
# Includes the major non-English articles the catalog also reorders so
# foreign-language titles round-trip. Matched case-insensitively after
# diacritic folding, so accented forms ("l'") are already ASCII here.
_ARTICLES = frozenset({
    "the", "a", "an",
    "la", "le", "les", "l'", "un", "une",          # French
    "el", "los", "las", "lo",                        # Spanish
    "il", "lo", "gli", "i",                          # Italian
    "der", "die", "das", "ein", "eine",              # German
    "de", "het", "een",                              # Dutch
    "o", "os", "as", "um", "uma",                    # Portuguese
})

# Trailing ", <article>" on a catalog title piece, e.g. "American President, The".
_TRAILING_ARTICLE_RE = re.compile(r"^(.*),\s+([a-z']+)$")
# A leading "a.k.a." / "aka" label inside an alternate-title parenthetical.
_AKA_PREFIX_RE = re.compile(r"^a\.?k\.?a\.?\s+", re.IGNORECASE)
# A trailing release year in parentheses: "... (1995)".
_TRAILING_YEAR_RE = re.compile(r"\((\d{4})\)\s*$")
# Any (non-nested) parenthetical group.
_PAREN_GROUP_RE = re.compile(r"\(([^()]*)\)")


def coerce_int(val) -> Optional[int]:
    """Best-effort ``val`` → ``int``; ``None`` for missing/NaN/garbage.

    Handles the messy types a CSV cell arrives as: ``"1995"``, ``1995.0``,
    ``float('nan')``, ``pandas.NA``, ``None``. Kept pandas-free so this
    module imports without the heavy data stack.
    """
    if val is None:
        return None
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None
    if f != f:  # NaN — including numpy/pandas NaN
        return None
    return int(f)


def _strip_diacritics(s: str) -> str:
    """Fold accented characters to ASCII ("Léon" → "Leon")."""
    return "".join(
        c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)
    )


def normalize_title(title: str) -> str:
    """Fold a single title into a comparison key.

    Lowercases, strips diacritics, moves a trailing article to the front
    ("American President, The" → "the american president"), normalizes
    ``&`` → ``and``, and reduces everything else to space-separated
    alphanumerics. Applied to *both* sides of the match so the catalog and
    a natural Letterboxd title meet in the middle.
    """
    if not title:
        return ""
    s = _strip_diacritics(str(title)).lower().strip()
    s = s.replace("&", " and ")
    m = _TRAILING_ARTICLE_RE.match(s)
    if m and m.group(2) in _ARTICLES:
        s = f"{m.group(2)} {m.group(1)}"
    # Collapse any remaining punctuation to spaces; keep alphanumerics.
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return s.strip()


def parse_movielens_title(raw: str) -> Tuple[List[str], Optional[int]]:
    """Split a MovieLens catalog title into ``(normalized_variants, year)``.

    Generates several comparison keys per title so a user's natural title
    can match whichever form the catalog stored:

    - the full string (parentheticals kept as inline text), so an inline
      alternate like "(500) Days of Summer" still lines up;
    - the primary title with parentheticals removed;
    - each parenthetical group on its own (alternate / original titles),
      with a leading "a.k.a." label stripped.
    """
    s = str(raw).strip()
    year: Optional[int] = None
    m = _TRAILING_YEAR_RE.search(s)
    if m:
        year = int(m.group(1))
        s = s[: m.start()].strip()

    pieces: List[str] = [s]                                   # full, parens inline
    pieces.append(_PAREN_GROUP_RE.sub(" ", s))                # primary only
    pieces.extend(_PAREN_GROUP_RE.findall(s))                 # alternates

    variants: List[str] = []
    seen: Set[str] = set()
    for piece in pieces:
        piece = _AKA_PREFIX_RE.sub("", piece.strip())
        norm = normalize_title(piece)
        if norm and norm not in seen:
            seen.add(norm)
            variants.append(norm)
    return variants, year


@dataclass
class LocalTitleIndex:
    """In-memory title lookup built once from the catalog at startup.

    ``by_title_year`` keys on the precise (normalized title, year) pair;
    ``by_title`` keys on title alone. Values are *sets* of movieIds so the
    matcher can detect ambiguity (remakes, shared alternate titles) and
    refuse to guess rather than return a wrong id.
    """

    by_title_year: Dict[Tuple[str, int], Set[str]]
    by_title: Dict[str, Set[str]]

    @property
    def n_titles(self) -> int:
        return len(self.by_title)


def build_local_title_index(
    movie_id_title_pairs: Iterable[Tuple[object, str]],
) -> LocalTitleIndex:
    """Build a :class:`LocalTitleIndex` from ``(movieId, title)`` pairs.

    Takes a plain iterable (not a DataFrame) so this module stays
    dependency-free and unit-testable. The caller passes, e.g.,
    ``zip(movies_df["movieId"].astype(str), movies_df["title"])``.
    """
    by_title_year: Dict[Tuple[str, int], Set[str]] = {}
    by_title: Dict[str, Set[str]] = {}
    for movie_id, title in movie_id_title_pairs:
        mid = str(movie_id)
        variants, year = parse_movielens_title(title)
        for v in variants:
            by_title.setdefault(v, set()).add(mid)
            if year is not None:
                by_title_year.setdefault((v, year), set()).add(mid)
    return LocalTitleIndex(by_title_year, by_title)


def match_local(
    name: str, year: Optional[int], index: LocalTitleIndex
) -> Optional[str]:
    """Resolve ``(name, year)`` to a movieId via the local catalog index.

    Precision-first: an exact (title, year) hit wins; otherwise a
    title-only hit is accepted *only if the title is unique* across the
    whole catalog. Ambiguous titles (remakes, shared names) with no
    matching year resolve to ``None`` — we'd rather miss than mis-map.
    """
    norm = normalize_title(name)
    if not norm:
        return None
    if year is not None:
        ids = index.by_title_year.get((norm, int(year)))
        if ids and len(ids) == 1:
            return next(iter(ids))
    ids = index.by_title.get(norm)
    if ids and len(ids) == 1:
        return next(iter(ids))
    return None


def resolve_movie_id(
    name: str,
    year: Optional[int],
    tmdb_id: Optional[int],
    tmdb_to_movie: Dict[str, str],
    index: LocalTitleIndex,
) -> Tuple[Optional[str], Optional[str]]:
    """Resolve one film to ``(movieId, method)`` without any network call.

    Order of preference:

    1. ``tmdb_id`` → ``movieId`` (authoritative; same id space as the
       catalog's ``links.csv``). Method ``"tmdb"``.
    2. local title+year match. Method ``"local"``.

    Returns ``(None, None)`` when neither resolves — the caller may then
    fall back to a TMDB *search* (network) if a key is configured.
    """
    if tmdb_id is not None:
        mid = tmdb_to_movie.get(str(tmdb_id))
        if mid is not None:
            return mid, "tmdb"
    mid = match_local(name, year, index)
    if mid is not None:
        return mid, "local"
    return None, None
