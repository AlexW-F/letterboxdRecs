"""
Recommendation generation utilities for letterboxdRecs.

This module contains:
- RecommendationEngine: top-N recommendations for individual users using a
  trained SVD/SVD++ or KNN model, with ridge-regression fold-in for users
  whose ratings were not in the original training set.
- load_user_data_with_tmdb: helper that joins a Letterboxd CSV (enriched
  with TMDB IDs) to MovieLens via links.csv, producing a {movieId: rating}
  dict consumable by the engine.

Group recommendations live in src/group_recommendations.py.
"""

import numpy as np
import pandas as pd
import pickle
from typing import Dict, List, Tuple, Optional
import logging

from .config import DEFAULT_K_NEIGHBORS, DEFAULT_FOLD_IN_REG

logger = logging.getLogger(__name__)


class RecommendationEngine:
    """Main class for generating recommendations."""
    
    def __init__(self, model_path: str = None, model = None):
        """
        Initialize the recommendation engine.
        
        Args:
            model_path: Path to saved model file
            model: Pre-loaded model object
        """
        if model_path:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
        elif model:
            self.model = model
        else:
            raise ValueError("Either model_path or model must be provided")
            
        self.trainset = self.model.trainset
        
    def get_user_recommendations(self, user_ratings: Dict[str, float], 
                               movies_df: pd.DataFrame,
                               top_n: int = 10,
                               exclude_watched: bool = True,
                               random_seed: Optional[int] = None) -> List[Tuple[str, float]]:
        """
        Generate recommendations for a user based on their ratings.
        
        Args:
            user_ratings: Dictionary of {movieId: rating}
            movies_df: DataFrame with movie information
            top_n: Number of recommendations to return
            exclude_watched: Whether to exclude movies the user has already rated
            random_seed: Random seed for reproducible recommendations
            
        Returns:
            List of (movie_title, predicted_rating) tuples
        """
        if random_seed is not None:
            np.random.seed(random_seed)
            
        # Handle different model types
        if hasattr(self.model, 'qi'):  # SVD-based model
            return self._svd_recommendations(user_ratings, movies_df, top_n, exclude_watched)
        else:  # KNN-based model
            return self._knn_recommendations(user_ratings, movies_df, top_n, exclude_watched)
    
    def _svd_recommendations(self, user_ratings: Dict[str, float], 
                           movies_df: pd.DataFrame,
                           top_n: int = 10,
                           exclude_watched: bool = True) -> List[Tuple[str, float]]:
        """Generate recommendations using SVD-based model with fold-in."""
        
        # Fold in the new user
        p_u = self._fold_in_user(user_ratings)
        
        # Generate predictions for all movies
        results = []
        mu = self.trainset.global_mean
        
        for inner_i, q_i in enumerate(self.model.qi):
            raw_i = self.trainset.to_raw_iid(inner_i)
            
            # Skip if user has already rated this movie
            if exclude_watched and str(raw_i) in [str(mid) for mid in user_ratings.keys()]:
                continue
                
            # Calculate prediction
            b_i = self.model.bi[inner_i]
            est = mu + b_i + q_i.dot(p_u)
            results.append((raw_i, est))
        
        # Sort and get top N
        top_results = sorted(results, key=lambda x: x[1], reverse=True)[:top_n]
        
        # Map to movie titles
        recommendations = []
        for movie_id, score in top_results:
            try:
                movie_id_str = str(movie_id)
                title = movies_df.loc[movies_df['movieId'].astype(str) == movie_id_str, 'title'].iloc[0]
                recommendations.append((title, round(score, 2)))
            except (IndexError, KeyError):
                continue
                
        return recommendations
    
    def _knn_recommendations(self, user_ratings: Dict[str, float],
                           movies_df: pd.DataFrame,
                           top_n: int = 10,
                           exclude_watched: bool = True,
                           k: int = DEFAULT_K_NEIGHBORS) -> List[Tuple[str, float]]:
        """Generate recommendations using KNN-based model."""
        
        sim_matrix = self.model.sim
        raw2inner = self.trainset._raw2inner_id_items
        inner2raw = self.trainset._inner2raw_id_items
        
        scores = {}
        sim_sums = {}
        
        # For each movie the user rated, find similar movies
        for movie_id, rating in user_ratings.items():
            movie_id_str = str(movie_id)
            if movie_id_str not in raw2inner:
                continue
                
            inner_j = raw2inner[movie_id_str]
            neighbors = self.model.get_neighbors(inner_j, k=k)
            
            for inner_i in neighbors:
                raw_i = inner2raw[inner_i]
                
                # Skip if user has already rated this movie
                if exclude_watched and str(raw_i) in [str(mid) for mid in user_ratings.keys()]:
                    continue
                    
                sim_ij = sim_matrix[inner_j, inner_i]
                scores.setdefault(raw_i, 0.0)
                sim_sums.setdefault(raw_i, 0.0)
                
                scores[raw_i] += sim_ij * rating
                sim_sums[raw_i] += abs(sim_ij)
        
        # Calculate final predictions
        predictions = []
        for movie_id, score_sum in scores.items():
            sim_sum = sim_sums[movie_id]
            if sim_sum > 0:
                est = score_sum / sim_sum
                predictions.append((movie_id, est))
        
        # Sort and get top N
        top_results = sorted(predictions, key=lambda x: x[1], reverse=True)[:top_n]
        
        # Map to movie titles
        recommendations = []
        for movie_id, score in top_results:
            try:
                movie_id_str = str(movie_id)
                title = movies_df.loc[movies_df['movieId'].astype(str) == movie_id_str, 'title'].iloc[0]
                recommendations.append((title, round(score, 2)))
            except (IndexError, KeyError):
                continue
                
        return recommendations
    
    def _fold_in_user(self, user_ratings: Dict[str, float], reg: float = DEFAULT_FOLD_IN_REG) -> np.ndarray:
        """
        Fold in a new user for SVD-based models using ridge regression.
        
        Args:
            user_ratings: Dictionary of {movieId: rating}
            reg: Regularization parameter
            
        Returns:
            User's latent factor vector
        """
        mu = self.trainset.global_mean
        
        # Build item bias map
        bi = {
            self.trainset.to_raw_iid(inner_i): bias
            for inner_i, bias in enumerate(self.model.bi)
        }
        
        # Collect movies the user rated that are in training set
        Q_rows = []
        y = []
        
        for movie_id, rating in user_ratings.items():
            # Try both original ID and string conversion
            valid_id = None
            inner_i = None
            
            # First try with the original movie_id type
            try:
                inner_i = self.trainset.to_inner_iid(movie_id)
                valid_id = movie_id
            except ValueError:
                # Then try with string conversion
                try:
                    movie_id_str = str(movie_id)
                    inner_i = self.trainset.to_inner_iid(movie_id_str)
                    valid_id = movie_id_str
                except ValueError:
                    continue  # Movie not in training set
            
            if inner_i is not None and valid_id is not None:
                q_i = self.model.qi[inner_i]
                # Get the raw ID from the trainset to ensure we have the right key for bi
                raw_id = self.trainset.to_raw_iid(inner_i)
                target = rating - mu - bi[raw_id]
                
                Q_rows.append(q_i)
                y.append(target)
        
        if not Q_rows:
            raise ValueError("No overlap between user ratings and trained items")
            
        Q = np.vstack(Q_rows)
        y = np.array(y)
        
        # Solve ridge regression: (Q^T Q + reg*I) p_u = Q^T y
        A = Q.T.dot(Q) + reg * np.eye(self.model.n_factors)
        b = Q.T.dot(y)
        p_u = np.linalg.solve(A, b)
        
        return p_u
    
    def get_similar_movies(self, movie_id: str, movies_df: pd.DataFrame, 
                          top_n: int = 10) -> List[Tuple[str, float]]:
        """
        Find movies similar to a given movie.
        
        Args:
            movie_id: ID of the movie to find similarities for
            movies_df: DataFrame with movie information
            top_n: Number of similar movies to return
            
        Returns:
            List of (movie_title, similarity_score) tuples
        """
        if not hasattr(self.model, 'sim'):
            raise ValueError("Similarity matrix not available for this model type")
            
        movie_id_str = str(movie_id)
        raw2inner = self.trainset._raw2inner_id_items
        inner2raw = self.trainset._inner2raw_id_items
        
        if movie_id_str not in raw2inner:
            raise ValueError(f"Movie {movie_id} not found in training data")
            
        inner_i = raw2inner[movie_id_str]
        sim_matrix = self.model.sim
        
        # Get similarities for this movie
        similarities = []
        for inner_j in range(len(inner2raw)):
            if inner_j != inner_i:
                raw_j = inner2raw[inner_j]
                sim_score = sim_matrix[inner_i, inner_j]
                similarities.append((raw_j, sim_score))
        
        # Sort by similarity and get top N
        top_similar = sorted(similarities, key=lambda x: x[1], reverse=True)[:top_n]
        
        # Map to movie titles
        similar_movies = []
        for movie_id, sim_score in top_similar:
            try:
                movie_id_str = str(movie_id)
                title = movies_df.loc[movies_df['movieId'].astype(str) == movie_id_str, 'title'].iloc[0]
                similar_movies.append((title, round(sim_score, 3)))
            except (IndexError, KeyError):
                continue
                
        return similar_movies


def load_user_data_with_tmdb(ratings_path: str, links_path: str) -> Dict[str, float]:
    """
    Load user ratings and map them to MovieLens IDs via TMDB.
    
    Args:
        ratings_path: Path to user's ratings file with TMDB IDs
        links_path: Path to MovieLens links.csv file
        
    Returns:
        Dictionary mapping MovieLens movieId to rating
    """
    user_df = pd.read_csv(ratings_path)
    links_df = pd.read_csv(links_path)
    
    # Handle different column naming conventions
    rating_col = 'Rating' if 'Rating' in user_df.columns else 'rating'
    tmdb_col = 'tmdb_id' if 'tmdb_id' in user_df.columns else 'tmdbId'
    
    # Normalize TMDB IDs identically on both sides. Both CSVs store the id
    # as a float (with ".0" suffix); naive astype(str) on one side and
    # int->str on the other produces "862.0" vs "862" and the merge yields
    # zero rows. Drop NaNs on both sides, then coerce through int->str.
    user_df[tmdb_col] = pd.to_numeric(user_df[tmdb_col], errors="coerce")
    user_df = user_df.dropna(subset=[tmdb_col])
    user_df[tmdb_col] = user_df[tmdb_col].astype(int).astype(str)

    links_df['tmdbId'] = pd.to_numeric(links_df['tmdbId'], errors="coerce")
    links_df = links_df.dropna(subset=['tmdbId'])
    links_df['tmdbId'] = links_df['tmdbId'].astype(int).astype(str)
    links_df['movieId'] = links_df['movieId'].astype(str)
    
    # Merge to get MovieLens IDs
    merged = user_df.merge(links_df, left_on=tmdb_col, right_on='tmdbId')
    user_ratings = dict(zip(merged['movieId'], merged[rating_col]))
    
    logger.info(f"Loaded {len(user_ratings)} user ratings with MovieLens mapping")
    return user_ratings
