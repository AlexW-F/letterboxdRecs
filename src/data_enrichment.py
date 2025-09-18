"""
Utilities for enriching Letterboxd data with TMDB IDs and other metadata.

This module contains functions for:
- Adding TMDB IDs to Letterboxd exports
- Validating and cleaning movie data
- Handling data mismatches and edge cases
"""

import pandas as pd
import requests
import time
from typing import Optional, Dict, List
import logging

logger = logging.getLogger(__name__)


def enrich_letterboxd_csv(input_csv: str, output_csv: str, sleep_between: float = 0.1) -> None:
    """
    Enrich Letterboxd CSV with TMDB IDs by searching TMDB API.
    
    Args:
        input_csv: Path to input Letterboxd CSV file
        output_csv: Path to output enriched CSV file  
        sleep_between: Sleep time between API calls to respect rate limits
    """
    logger.info(f"Enriching {input_csv} with TMDB IDs...")
    
    df = pd.read_csv(input_csv)
    
    # Add tmdb_id column if it doesn't exist
    if 'tmdb_id' not in df.columns:
        df['tmdb_id'] = None
    
    # Iterate through rows and lookup TMDB IDs
    for idx, row in df.iterrows():
        if pd.isna(row.get('tmdb_id')) and 'Name' in row:
            movie_name = row['Name']
            year = extract_year_from_title(movie_name)
            
            tmdb_id = search_tmdb_for_movie(movie_name, year)
            if tmdb_id:
                df.at[idx, 'tmdb_id'] = tmdb_id
                logger.debug(f"Found TMDB ID {tmdb_id} for '{movie_name}'")
            
            time.sleep(sleep_between)
    
    # Save enriched data
    df.to_csv(output_csv, index=False)
    logger.info(f"Saved enriched data to {output_csv}")


def extract_year_from_title(title: str) -> Optional[int]:
    """
    Extract year from movie title if present.
    
    Args:
        title: Movie title string
        
    Returns:
        Year as integer if found, None otherwise
    """
    import re
    
    # Look for year in parentheses at the end
    match = re.search(r'\((\d{4})\)$', title)
    if match:
        return int(match.group(1))
    
    # Look for 4-digit year anywhere in title
    match = re.search(r'\b(19|20)\d{2}\b', title)
    if match:
        return int(match.group(0))
    
    return None


def search_tmdb_for_movie(movie_name: str, year: Optional[int] = None, 
                         api_key: Optional[str] = None) -> Optional[int]:
    """
    Search TMDB API for a movie and return its ID.
    
    Args:
        movie_name: Name of the movie to search for
        year: Optional release year to improve search accuracy
        api_key: TMDB API key (if not provided, will try to use environment variable)
        
    Returns:
        TMDB ID if found, None otherwise
    """
    if not api_key:
        import os
        api_key = os.getenv('TMDB_API_KEY')
        
    if not api_key:
        logger.warning("No TMDB API key provided, skipping TMDB lookup")
        return None
    
    base_url = "https://api.themoviedb.org/3/search/movie"
    params = {
        'api_key': api_key,
        'query': movie_name
    }
    
    if year:
        params['year'] = year
    
    try:
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        results = data.get('results', [])
        
        if results:
            # Return the first (most relevant) result
            return results[0]['id']
        else:
            logger.debug(f"No TMDB results found for '{movie_name}'")
            return None
            
    except Exception as e:
        logger.error(f"Error searching TMDB for '{movie_name}': {e}")
        return None


def validate_tmdb_links(df: pd.DataFrame, links_df: pd.DataFrame) -> Dict[str, int]:
    """
    Validate TMDB ID links between user data and MovieLens links.
    
    Args:
        df: DataFrame with TMDB IDs 
        links_df: MovieLens links DataFrame
        
    Returns:
        Dictionary with validation statistics
    """
    logger.info("Validating TMDB links...")
    
    # Clean and prepare data
    df_clean = df.dropna(subset=['tmdb_id']).copy()
    df_clean['tmdb_id'] = df_clean['tmdb_id'].astype(float).astype(int).astype(str)
    
    links_clean = links_df.dropna(subset=['tmdbId']).copy()
    links_clean['tmdbId'] = links_clean['tmdbId'].astype(float).astype(int).astype(str)
    
    # Find matches
    matched_ids = set(df_clean['tmdb_id']) & set(links_clean['tmdbId'])
    
    stats = {
        'total_user_movies': len(df),
        'user_movies_with_tmdb': len(df_clean),
        'total_movielens_links': len(links_clean),
        'matched_tmdb_ids': len(matched_ids),
        'match_rate': len(matched_ids) / len(df_clean) if len(df_clean) > 0 else 0
    }
    
    logger.info(f"TMDB validation: {stats['matched_tmdb_ids']}/{stats['user_movies_with_tmdb']} movies matched ({stats['match_rate']:.1%})")
    
    return stats


def create_movielens_mapping(user_csv: str, links_csv: str, output_csv: str) -> pd.DataFrame:
    """
    Create a mapping from user data to MovieLens IDs via TMDB.
    
    Args:
        user_csv: Path to user's CSV file with TMDB IDs
        links_csv: Path to MovieLens links.csv file
        output_csv: Path to save the mapped data
        
    Returns:
        DataFrame with MovieLens mapping
    """
    logger.info("Creating MovieLens mapping...")
    
    # Load data
    user_df = pd.read_csv(user_csv)
    links_df = pd.read_csv(links_csv)
    
    # Clean TMDB IDs
    user_df = user_df.dropna(subset=['tmdb_id'])
    user_df['tmdb_id'] = user_df['tmdb_id'].astype(float).astype(int).astype(str)
    
    links_df['tmdbId'] = links_df['tmdbId'].astype(str)
    links_df['movieId'] = links_df['movieId'].astype(str)
    
    # Merge on TMDB ID
    merged = user_df.merge(
        links_df, 
        left_on='tmdb_id', 
        right_on='tmdbId',
        how='inner'
    )
    
    # Save mapped data
    merged.to_csv(output_csv, index=False)
    logger.info(f"Created MovieLens mapping with {len(merged)} movies, saved to {output_csv}")
    
    return merged


def clean_rating_data(ratings_df: pd.DataFrame, min_rating: float = 0.5, max_rating: float = 5.0) -> pd.DataFrame:
    """
    Clean and validate rating data.
    
    Args:
        ratings_df: DataFrame with rating data
        min_rating: Minimum valid rating value
        max_rating: Maximum valid rating value
        
    Returns:
        Cleaned DataFrame
    """
    logger.info("Cleaning rating data...")
    
    initial_count = len(ratings_df)
    
    # Handle different rating column names
    rating_col = None
    for col in ['Rating', 'rating', 'score']:
        if col in ratings_df.columns:
            rating_col = col
            break
    
    if not rating_col:
        raise ValueError("No rating column found in DataFrame")
    
    # Clean ratings
    df_clean = ratings_df.copy()
    
    # Remove null ratings
    df_clean = df_clean.dropna(subset=[rating_col])
    
    # Remove invalid ratings
    df_clean = df_clean[
        (df_clean[rating_col] >= min_rating) & 
        (df_clean[rating_col] <= max_rating)
    ]
    
    # Remove duplicates if movieId exists
    if 'movieId' in df_clean.columns:
        df_clean = df_clean.drop_duplicates(subset=['movieId'])
    elif 'tmdb_id' in df_clean.columns:
        df_clean = df_clean.drop_duplicates(subset=['tmdb_id'])
    
    final_count = len(df_clean)
    removed_count = initial_count - final_count
    
    logger.info(f"Cleaned rating data: {removed_count} invalid entries removed, {final_count} valid ratings remaining")
    
    return df_clean
