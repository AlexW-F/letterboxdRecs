# add_tmdb_ids.py
import os
import time
import requests
import pandas as pd

from typing import Optional

TMDB_API_KEY = os.getenv("TMDB_API_KEY")

if not TMDB_API_KEY:
    raise RuntimeError("Please set your TMDB_API_KEY environment variable")

SEARCH_URL = "https://api.themoviedb.org/3/search/movie"

def fetch_tmdb_id(title: str, year: Optional[int]) -> Optional[int]:
    """
    Query TMDB search/movie for a given title (and year, if available).
    Returns the first result's 'id', or None if no match.
    """
    params = {
        "api_key": TMDB_API_KEY,
        "query": title,
        "page": 1,
    }
    if year:
        params["year"] = int(year)

    resp = requests.get(SEARCH_URL, params=params, timeout=5)
    resp.raise_for_status()
    data = resp.json()

    results = data.get("results", [])
    if not results:
        return None
    return results[0]["id"]


def enrich_letterboxd_csv(
    input_csv: str,
    output_csv: str,
    sleep_between: float = 0.25,
) -> None:
    """
    Reads the input CSV, fetches TMDB IDs for each row, and writes out
    a new CSV with an added 'tmdb_id' column.
    """
    df = pd.read_csv(input_csv)
    tmdb_ids = []

    for idx, row in df.iterrows():
        title = row["Name"]
        year  = row.get("year", None)
        try:
            tmdb_id = fetch_tmdb_id(title, year)
        except Exception as e:
            print(f"[row {idx}] Error fetching '{title}' ({year}): {e}")
            tmdb_id = None
        tmdb_ids.append(tmdb_id)

        # be kind to the API
        time.sleep(sleep_between)

    df["tmdb_id"] = tmdb_ids
    df.to_csv(output_csv, index=False)
    print(f"Enriched CSV written to {output_csv}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="Add TMDB IDs to Letterboxd export CSV"
    )
    p.add_argument("input_csv",  help="Path to your letterboxd_export.csv")
    p.add_argument("output_csv", help="Where to write the enriched CSV")
    p.add_argument(
        "--delay", type=float, default=0.25,
        help="Seconds to wait between API calls (default: 0.25)"
    )
    args = p.parse_args()

    enrich_letterboxd_csv(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        sleep_between=args.delay,
    )
