"""Letterboxd username → recent ratings via the public RSS feed.

Letterboxd's HTML pages (``/{user}/films/``) are gated by Cloudflare's
"Just a moment..." challenge — a plain ``requests.get()`` returns the
challenge page, not the content. The RSS feed at ``/{user}/rss/`` is not
gated, but it only exposes the most-recent ~50 diary entries.

Each ``<item>`` in the RSS gives us:
- ``<letterboxd:filmTitle>`` / ``<letterboxd:filmYear>`` — display
- ``<letterboxd:memberRating>`` — 0.5–5.0 (matches the Letterboxd CSV export)
- ``<tmdb:movieId>`` — TMDB ID inline, so no extra TMDB-search round-trip
- ``<letterboxd:rewatch>`` — Yes/No (we don't filter rewatches today)

We map RSS entries to the same in-memory representation that
``_ratings_csv_to_movielens`` produces for CSV uploads, so the rest of
the pipeline is unchanged.
"""

from __future__ import annotations

import io
import logging
import re
from typing import Dict, List, Optional, Set, Tuple

import requests
from lxml import etree


logger = logging.getLogger(__name__)

# Letterboxd usernames: 2–15 chars, alnum + underscore + hyphen
USERNAME_RE = re.compile(r"^[a-zA-Z0-9_-]{2,15}$")

# Realistic UA is required — Cloudflare 524s default `python-requests`.
_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

_NS = {
    "lb": "https://letterboxd.com",
    "tmdb": "https://themoviedb.org",
}


class LetterboxdScraperError(Exception):
    """Raised when we can't get usable ratings for a username."""


def validate_username(username: str) -> str:
    """Return the lowercased username if valid; raise otherwise."""
    cleaned = (username or "").strip().lstrip("@")
    if not USERNAME_RE.match(cleaned):
        raise LetterboxdScraperError(
            f"invalid Letterboxd username {username!r}; expected 2–15 chars, "
            "alnum + underscore + hyphen"
        )
    return cleaned.lower()


def fetch_rss(username: str, timeout: float = 15.0) -> bytes:
    """Fetch the raw RSS bytes for a Letterboxd user. Raises on HTTP error."""
    username = validate_username(username)
    url = f"https://letterboxd.com/{username}/rss/"
    try:
        resp = requests.get(
            url,
            headers={"User-Agent": _UA, "Accept": "application/rss+xml,*/*"},
            timeout=timeout,
        )
    except requests.RequestException as e:
        raise LetterboxdScraperError(
            f"network error fetching {url}: {e}"
        ) from e
    if resp.status_code == 404:
        raise LetterboxdScraperError(
            f"no Letterboxd user named {username!r} (404). "
            "Check the spelling, or the account may be private."
        )
    if resp.status_code != 200:
        raise LetterboxdScraperError(
            f"Letterboxd returned HTTP {resp.status_code} for {url}"
        )
    return resp.content


def parse_rss(xml_bytes: bytes) -> Tuple[List[Dict], Set[int]]:
    """Parse RSS bytes into:

    - ``rated``: list of {tmdb_id, name, year, rating} dicts (rating present)
    - ``watched_tmdb_ids``: TMDB IDs the user diary'd without a rating
      (analog of CSV watched.csv)

    Entries missing both a rating *and* a TMDB ID are dropped silently.
    """
    try:
        root = etree.fromstring(xml_bytes)
    except etree.XMLSyntaxError as e:
        raise LetterboxdScraperError(f"could not parse RSS XML: {e}") from e

    rated: List[Dict] = []
    watched: Set[int] = set()
    for item in root.iter("item"):
        rating_el = item.find("lb:memberRating", _NS)
        tmdb_el = item.find("tmdb:movieId", _NS)
        if tmdb_el is None or not tmdb_el.text:
            continue
        try:
            tmdb_id = int(tmdb_el.text.strip())
        except ValueError:
            continue

        if rating_el is not None and rating_el.text:
            try:
                rating = float(rating_el.text.strip())
            except ValueError:
                rating = None
        else:
            rating = None

        if rating is None:
            # Watched-but-unrated diary entry — implicit-positive signal only.
            watched.add(tmdb_id)
            continue

        title_el = item.find("lb:filmTitle", _NS)
        year_el = item.find("lb:filmYear", _NS)
        name = (title_el.text or "").strip() if title_el is not None else ""
        try:
            year = int(year_el.text) if year_el is not None and year_el.text else None
        except ValueError:
            year = None

        rated.append({
            "tmdb_id": tmdb_id,
            "name": name,
            "year": year,
            "rating": rating,
        })

    return rated, watched


def synthesize_ratings_csv(rated: List[Dict]) -> bytes:
    """Build a CSV identical in shape to a Letterboxd ratings export so it
    can be fed to the existing ``_ratings_csv_to_movielens`` pipeline.

    Sort by tmdb_id so the same RSS content always produces the same bytes
    (and therefore the same content-hash → same cache entry)."""
    rows = sorted(rated, key=lambda r: r["tmdb_id"])
    buf = io.StringIO()
    buf.write("Name,Year,Rating,tmdb_id\n")
    for r in rows:
        name = (r.get("name") or "").replace('"', '""')
        year = r.get("year") or ""
        buf.write(f'"{name}",{year},{r["rating"]},{r["tmdb_id"]}\n')
    return buf.getvalue().encode("utf-8")


def synthesize_watched_csv(tmdb_ids: Set[int]) -> bytes:
    """Build a watched.csv-shaped blob from the watched-but-unrated set."""
    buf = io.StringIO()
    buf.write("Name,Year,tmdb_id\n")
    for tid in sorted(tmdb_ids):
        buf.write(f',,"{tid}"\n')
    return buf.getvalue().encode("utf-8")


def fetch_user_blobs(username: str) -> Tuple[bytes, Optional[bytes], int, int]:
    """End-to-end: username → (ratings_csv_bytes, watched_csv_bytes_or_none,
    n_rated, n_watched).

    Raises LetterboxdScraperError on any failure mode the API endpoint
    should surface as a 4xx (bad username, private profile, no data).
    """
    xml = fetch_rss(username)
    rated, watched = parse_rss(xml)
    if not rated and not watched:
        raise LetterboxdScraperError(
            f"no diary entries found for {username!r}. "
            "The profile may be empty, private, or fresh."
        )
    if not rated:
        raise LetterboxdScraperError(
            f"{username!r} has diary entries but no ratings in their recent ~50. "
            "Recommendation requires at least a few rated films."
        )
    ratings_blob = synthesize_ratings_csv(rated)
    watched_blob = synthesize_watched_csv(watched) if watched else None
    return ratings_blob, watched_blob, len(rated), len(watched)
