"""
Data processing utilities for letterboxdRecs.

This module contains functions for:
- Loading and preprocessing MovieLens data
- Enriching Letterboxd data with TMDB IDs
- Data validation and cleaning
"""

import pandas as pd
import numpy as np
from surprise import Dataset, Reader
from typing import Dict, List, Tuple, Optional
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_movielens_data(ratings_path: str, movies_path: str, links_path: str = None) -> Tuple[Dataset, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load MovieLens dataset for use with Surprise library.
    
    Args:
        ratings_path: Path to ratings.csv file
        movies_path: Path to movies.csv file  
        links_path: Optional path to links.csv file
        
    Returns:
        Tuple of (surprise_dataset, ratings_df, movies_df, links_df)
    """
    logger.info("Loading MovieLens data...")
    
    # Load for Surprise. ml-32m uses 0.5-step ratings (0.5..5.0); Letterboxd
    # exports are also 0.5-step. Surprise's default rating_scale=(1,5) would
    # clip *predictions* to >=1.0 (training values pass through fine), which
    # is harmless for ranking but cosmetic in user-facing "predicted star"
    # output. We set the true scale here and in create_surprise_dataset.
    reader = Reader(line_format="user item rating timestamp", sep=",",
                    skip_lines=1, rating_scale=(0.5, 5.0))
    surprise_data = Dataset.load_from_file(ratings_path, reader=reader)
    
    # Load as pandas DataFrames
    ratings_df = pd.read_csv(ratings_path)
    movies_df = pd.read_csv(movies_path)
    links_df = pd.read_csv(links_path) if links_path else None
    
    logger.info(f"Loaded {len(ratings_df):,} ratings for {ratings_df['userId'].nunique():,} users and {ratings_df['movieId'].nunique():,} movies")
    
    return surprise_data, ratings_df, movies_df, links_df


def validate_data_quality(ratings_df: pd.DataFrame, movies_df: pd.DataFrame) -> Dict[str, any]:
    """
    Validate data quality and return statistics.
    
    Args:
        ratings_df: Ratings DataFrame
        movies_df: Movies DataFrame
        
    Returns:
        Dictionary with data quality metrics
    """
    logger.info("Validating data quality...")
    
    quality_metrics = {
        'ratings_count': len(ratings_df),
        'users_count': ratings_df['userId'].nunique(),
        'movies_count': ratings_df['movieId'].nunique(),
        'rating_range': (ratings_df['rating'].min(), ratings_df['rating'].max()),
        'avg_rating': ratings_df['rating'].mean(),
        'missing_ratings': ratings_df.isnull().sum().sum(),
        'duplicate_ratings': ratings_df.duplicated().sum(),
        'movies_without_ratings': len(movies_df) - len(set(movies_df['movieId']) & set(ratings_df['movieId'])),
        'sparsity': 1 - (len(ratings_df) / (ratings_df['userId'].nunique() * ratings_df['movieId'].nunique()))
    }
    
    return quality_metrics


def preprocess_letterboxd_data(ratings_path: str, watched_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and preprocess Letterboxd user data.
    
    Args:
        ratings_path: Path to user's ratings.csv
        watched_path: Path to user's watched.csv
        
    Returns:
        Tuple of (ratings_df, watched_df)
    """
    logger.info("Loading Letterboxd user data...")
    
    ratings_df = pd.read_csv(ratings_path)
    watched_df = pd.read_csv(watched_path)
    
    # Basic cleaning
    if 'Rating' in ratings_df.columns:
        ratings_df = ratings_df.dropna(subset=['Rating'])
        ratings_df = ratings_df[ratings_df['Rating'] > 0]
    
    logger.info(f"Loaded {len(ratings_df)} ratings and {len(watched_df)} watched movies")
    
    return ratings_df, watched_df


def create_user_movie_matrix(ratings_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create user-movie rating matrix.
    
    Args:
        ratings_df: Ratings DataFrame with userId, movieId, rating columns
        
    Returns:
        User-movie matrix with users as rows and movies as columns
    """
    logger.info("Creating user-movie matrix...")
    
    matrix = ratings_df.pivot_table(
        index='userId', 
        columns='movieId', 
        values='rating',
        fill_value=0
    )
    
    logger.info(f"Created matrix of shape {matrix.shape}")
    return matrix


def load_letterboxd_data(data_path: str) -> pd.DataFrame:
    """
    Load Letterboxd data from the specified directory.
    
    Args:
        data_path: Path to directory containing Letterboxd CSV files
        
    Returns:
        DataFrame with ratings data
    """
    logger.info(f"Loading Letterboxd data from {data_path}...")
    
    ratings_path = os.path.join(data_path, "ratings.csv")
    watched_path = os.path.join(data_path, "watched.csv")
    
    if os.path.exists(ratings_path):
        # Load ratings data
        ratings_df = pd.read_csv(ratings_path)
        logger.info(f"Loaded {len(ratings_df)} ratings from {ratings_path}")
        return ratings_df
    elif os.path.exists(watched_path):
        # Load watched data and convert to ratings
        watched_df = pd.read_csv(watched_path)
        # Assign default rating of 3.5 to watched movies without ratings
        ratings_df = watched_df[['Name', 'Year']].copy()
        ratings_df['rating'] = 3.5
        ratings_df['userId'] = 1  # Single user
        ratings_df['movieId'] = range(1, len(ratings_df) + 1)
        logger.info(f"Loaded {len(ratings_df)} watched movies from {watched_path}")
        return ratings_df
    else:
        raise FileNotFoundError(f"No ratings.csv or watched.csv found in {data_path}")


def create_surprise_dataset(ratings_df: pd.DataFrame) -> Dataset:
    """
    Create a Surprise Dataset from a ratings DataFrame.
    
    Args:
        ratings_df: DataFrame with columns ['userId', 'movieId', 'rating']
        
    Returns:
        Surprise Dataset object
    """
    from surprise import Dataset, Reader
    
    logger.info("Creating Surprise dataset...")
    
    # Ensure required columns exist
    required_columns = ['userId', 'movieId', 'rating']
    
    # Check if we need to create the required columns from Letterboxd format
    if not all(col in ratings_df.columns for col in required_columns):
        if 'Name' in ratings_df.columns and 'Rating' in ratings_df.columns:
            # Letterboxd format - create movie IDs and user IDs
            processed_df = ratings_df.copy()
            processed_df['userId'] = 1  # Single user
            processed_df['movieId'] = range(1, len(processed_df) + 1)
            processed_df['rating'] = processed_df['Rating']
            
            # Select only the required columns
            processed_df = processed_df[['userId', 'movieId', 'rating']]
        elif 'Name' in ratings_df.columns:
            # Watched movies without ratings - assign default rating
            processed_df = ratings_df.copy()
            processed_df['userId'] = 1
            processed_df['movieId'] = range(1, len(processed_df) + 1)
            processed_df['rating'] = 3.5  # Default rating
            
            # Select only the required columns
            processed_df = processed_df[['userId', 'movieId', 'rating']]
        else:
            raise ValueError(f"DataFrame must contain columns: {required_columns} or Letterboxd format columns")
    else:
        # DataFrame already has required columns
        processed_df = ratings_df[required_columns].copy()
    
    # 0.5-step rating scale matches MovieLens 32M and Letterboxd exports;
    # config.MIN_RATING / MAX_RATING agree. Surprise will clip predictions
    # to this range; training data passes through unchanged.
    reader = Reader(rating_scale=(0.5, 5.0))
    
    # Create dataset
    dataset = Dataset.load_from_df(processed_df, reader)
    
    logger.info(f"Created Surprise dataset with {len(processed_df)} ratings")
    return dataset
