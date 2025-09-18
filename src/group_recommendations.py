"""
Group recommendation utilities for letterboxdRecs.

This module contains functions for:
- Aggregating individual user preferences into group preferences
- Generating recommendations for groups using different strategies
- Handling consensus, satisfaction, and fairness in group recommendations
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from collections import defaultdict, Counter
import logging
from .recommendations import RecommendationEngine

logger = logging.getLogger(__name__)


class GroupRecommendationEngine(RecommendationEngine):
    """Extended recommendation engine for group recommendations."""
    
    def __init__(self, model_path: str = None, model = None):
        """Initialize the group recommendation engine."""
        super().__init__(model_path, model)
    
    def get_group_recommendations(self, 
                                group_ratings: Dict[str, Dict[str, float]], 
                                movies_df: pd.DataFrame,
                                strategy: str = "average",
                                top_n: int = 10,
                                exclude_watched: bool = True,
                                fairness_weight: float = 0.5,
                                random_seed: Optional[int] = None) -> List[Tuple[str, float, Dict]]:
        """
        Generate recommendations for a group of users.
        
        Args:
            group_ratings: Dictionary of {user_id: {movieId: rating}}
            movies_df: DataFrame with movie information
            strategy: Group aggregation strategy ('average', 'least_misery', 'most_pleasure', 'consensus', 'hybrid')
            top_n: Number of recommendations to return
            exclude_watched: Whether to exclude movies already watched by any group member
            fairness_weight: Weight for fairness component in hybrid strategy (0-1)
            random_seed: Random seed for reproducible recommendations
            
        Returns:
            List of (movie_title, group_score, individual_scores) tuples
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        logger.info(f"Generating group recommendations for {len(group_ratings)} users using {strategy} strategy")
        
        # Get individual recommendations for each user
        individual_recs = {}
        all_watched_movies = set()
        
        for user_id, user_ratings in group_ratings.items():
            user_recs = self.get_user_recommendations(
                user_ratings, movies_df, top_n=top_n*3,  # Get more to have variety
                exclude_watched=exclude_watched, random_seed=random_seed
            )
            individual_recs[user_id] = {title: score for title, score in user_recs}
            
            if exclude_watched:
                all_watched_movies.update(user_ratings.keys())
        
        # Get all candidate movies (union of all individual recommendations)
        all_candidate_movies = set()
        for user_recs in individual_recs.values():
            all_candidate_movies.update(user_recs.keys())
        
        # Calculate group scores using the specified strategy
        group_scores = self._calculate_group_scores(
            individual_recs, all_candidate_movies, strategy, fairness_weight
        )
        
        # Sort by group score and return top N
        sorted_recs = sorted(group_scores.items(), key=lambda x: x[1]['group_score'], reverse=True)
        
        result = []
        for movie_title, scores_dict in sorted_recs[:top_n]:
            result.append((
                movie_title, 
                round(scores_dict['group_score'], 2),
                {user: round(score, 2) for user, score in scores_dict['individual_scores'].items()}
            ))
        
        return result
    
    def _calculate_group_scores(self, 
                              individual_recs: Dict[str, Dict[str, float]], 
                              candidate_movies: set, 
                              strategy: str,
                              fairness_weight: float) -> Dict[str, Dict]:
        """
        Calculate group scores for candidate movies using different aggregation strategies.
        
        Args:
            individual_recs: Dictionary of {user_id: {movie: score}}
            candidate_movies: Set of all candidate movie titles
            strategy: Aggregation strategy
            fairness_weight: Weight for fairness in hybrid strategy
            
        Returns:
            Dictionary of {movie: {'group_score': float, 'individual_scores': dict}}
        """
        group_scores = {}
        
        for movie in candidate_movies:
            # Get individual scores for this movie (0 if user didn't have it in their recs)
            individual_scores = {}
            scores_list = []
            
            for user_id in individual_recs.keys():
                score = individual_recs[user_id].get(movie, 0.0)
                individual_scores[user_id] = score
                if score > 0:  # Only include non-zero scores for aggregation
                    scores_list.append(score)
            
            if not scores_list:  # No user had this movie in their recommendations
                continue
            
            # Calculate group score based on strategy
            if strategy == "average":
                group_score = np.mean(scores_list)
            elif strategy == "least_misery":
                group_score = np.min(scores_list)
            elif strategy == "most_pleasure":
                group_score = np.max(scores_list)
            elif strategy == "consensus":
                # Higher score if more users have it and scores are similar
                coverage = len(scores_list) / len(individual_recs)
                agreement = 1 - (np.std(scores_list) / np.mean(scores_list)) if len(scores_list) > 1 else 1
                group_score = np.mean(scores_list) * coverage * agreement
            elif strategy == "hybrid":
                # Combine average satisfaction with fairness
                avg_score = np.mean(scores_list)
                fairness_score = self._calculate_fairness_score(scores_list)
                group_score = (1 - fairness_weight) * avg_score + fairness_weight * fairness_score
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            group_scores[movie] = {
                'group_score': group_score,
                'individual_scores': individual_scores
            }
        
        return group_scores
    
    def _calculate_fairness_score(self, scores: List[float]) -> float:
        """
        Calculate fairness score based on how evenly distributed the scores are.
        Higher fairness = lower variance in individual satisfaction.
        
        Args:
            scores: List of individual scores for a movie
            
        Returns:
            Fairness score (higher = more fair)
        """
        if len(scores) <= 1:
            return np.mean(scores) if scores else 0.0
        
        # Use negative coefficient of variation as fairness measure
        # Lower CV = more fair distribution
        mean_score = np.mean(scores)
        if mean_score == 0:
            return 0.0
        
        cv = np.std(scores) / mean_score
        fairness = mean_score * (2 - cv)  # Penalize high variance
        
        return max(0, fairness)
    
    def analyze_group_preferences(self, 
                                group_ratings: Dict[str, Dict[str, float]],
                                movies_df: pd.DataFrame) -> Dict[str, any]:
        """
        Analyze group preferences and compatibility.
        
        Args:
            group_ratings: Dictionary of {user_id: {movieId: rating}}
            movies_df: DataFrame with movie information
            
        Returns:
            Dictionary with group analysis metrics
        """
        logger.info("Analyzing group preferences...")
        
        # Find common movies rated by multiple users
        all_users = list(group_ratings.keys())
        movie_ratings = defaultdict(dict)
        
        for user_id, ratings in group_ratings.items():
            for movie_id, rating in ratings.items():
                movie_ratings[movie_id][user_id] = rating
        
        # Calculate similarities between users
        user_similarities = {}
        for i, user1 in enumerate(all_users):
            for user2 in all_users[i+1:]:
                similarity = self._calculate_user_similarity(
                    group_ratings[user1], group_ratings[user2]
                )
                user_similarities[f"{user1}-{user2}"] = similarity
        
        # Find consensus and disagreement movies
        consensus_movies = []
        disagreement_movies = []
        
        for movie_id, user_ratings in movie_ratings.items():
            if len(user_ratings) >= 2:  # At least 2 users rated it
                ratings_list = list(user_ratings.values())
                std_dev = np.std(ratings_list)
                mean_rating = np.mean(ratings_list)
                
                try:
                    movie_title = movies_df.loc[movies_df['movieId'].astype(str) == str(movie_id), 'title'].iloc[0]
                except (IndexError, KeyError):
                    movie_title = f"Movie {movie_id}"
                
                if std_dev < 0.5:  # Low disagreement
                    consensus_movies.append((movie_title, mean_rating, std_dev))
                elif std_dev > 1.5:  # High disagreement
                    disagreement_movies.append((movie_title, mean_rating, std_dev))
        
        # Sort by rating
        consensus_movies.sort(key=lambda x: x[1], reverse=True)
        disagreement_movies.sort(key=lambda x: x[2], reverse=True)
        
        analysis = {
            'num_users': len(all_users),
            'total_ratings': sum(len(ratings) for ratings in group_ratings.values()),
            'avg_ratings_per_user': np.mean([len(ratings) for ratings in group_ratings.values()]),
            'user_similarities': user_similarities,
            'avg_group_similarity': np.mean(list(user_similarities.values())) if user_similarities else 0,
            'consensus_movies': consensus_movies[:10],
            'disagreement_movies': disagreement_movies[:10],
            'common_movies_count': len([m for m, ratings in movie_ratings.items() if len(ratings) >= 2])
        }
        
        return analysis
    
    def _calculate_user_similarity(self, 
                                 ratings1: Dict[str, float], 
                                 ratings2: Dict[str, float]) -> float:
        """
        Calculate similarity between two users based on their common ratings.
        
        Args:
            ratings1: First user's ratings
            ratings2: Second user's ratings
            
        Returns:
            Pearson correlation coefficient (-1 to 1)
        """
        # Find common movies
        common_movies = set(ratings1.keys()) & set(ratings2.keys())
        
        if len(common_movies) < 2:
            return 0.0
        
        # Extract ratings for common movies
        r1 = [ratings1[movie] for movie in common_movies]
        r2 = [ratings2[movie] for movie in common_movies]
        
        # Calculate Pearson correlation
        if np.std(r1) == 0 or np.std(r2) == 0:
            return 0.0
        
        correlation = np.corrcoef(r1, r2)[0, 1]
        return correlation if not np.isnan(correlation) else 0.0
    
    def get_group_recommendations_with_explanation(self, 
                                                 group_ratings: Dict[str, Dict[str, float]], 
                                                 movies_df: pd.DataFrame,
                                                 strategy: str = "average",
                                                 top_n: int = 10) -> Dict[str, any]:
        """
        Generate group recommendations with detailed explanations.
        
        Args:
            group_ratings: Dictionary of {user_id: {movieId: rating}}
            movies_df: DataFrame with movie information
            strategy: Group aggregation strategy
            top_n: Number of recommendations to return
            
        Returns:
            Dictionary with recommendations and explanations
        """
        # Get recommendations
        recommendations = self.get_group_recommendations(
            group_ratings, movies_df, strategy, top_n
        )
        
        # Get group analysis
        analysis = self.analyze_group_preferences(group_ratings, movies_df)
        
        # Generate explanations
        explanations = []
        for movie_title, group_score, individual_scores in recommendations:
            explanation = self._generate_recommendation_explanation(
                movie_title, group_score, individual_scores, strategy
            )
            explanations.append(explanation)
        
        return {
            'recommendations': recommendations,
            'explanations': explanations,
            'group_analysis': analysis,
            'strategy_used': strategy
        }
    
    def _generate_recommendation_explanation(self, 
                                           movie_title: str, 
                                           group_score: float, 
                                           individual_scores: Dict[str, float], 
                                           strategy: str) -> str:
        """Generate a human-readable explanation for a recommendation."""
        
        non_zero_scores = {user: score for user, score in individual_scores.items() if score > 0}
        
        if strategy == "average":
            explanation = f"Average predicted rating: {group_score}★. "
            if len(non_zero_scores) == len(individual_scores):
                explanation += "All group members would likely enjoy this movie."
            else:
                explanation += f"{len(non_zero_scores)}/{len(individual_scores)} group members would likely enjoy this movie."
        
        elif strategy == "least_misery":
            min_user = min(non_zero_scores.items(), key=lambda x: x[1])[0]
            explanation = f"Minimum satisfaction guaranteed: {group_score}★ (for {min_user}). This ensures no one will be disappointed."
        
        elif strategy == "most_pleasure":
            max_user = max(non_zero_scores.items(), key=lambda x: x[1])[0]
            explanation = f"Maximum satisfaction potential: {group_score}★ (for {max_user}). At least one person will love this movie."
        
        elif strategy == "consensus":
            explanation = f"Consensus score: {group_score}★. This movie balances broad appeal with agreement among group members."
        
        elif strategy == "hybrid":
            explanation = f"Balanced score: {group_score}★. This movie optimizes both overall satisfaction and fairness among group members."
        
        else:
            explanation = f"Group score: {group_score}★"
        
        return explanation


def compare_group_strategies(group_ratings: Dict[str, Dict[str, float]], 
                           movies_df: pd.DataFrame,
                           model_path: str,
                           top_n: int = 10) -> pd.DataFrame:
    """
    Compare different group recommendation strategies.
    
    Args:
        group_ratings: Dictionary of {user_id: {movieId: rating}}
        movies_df: DataFrame with movie information
        model_path: Path to the trained model
        top_n: Number of recommendations to compare
        
    Returns:
        DataFrame comparing strategies
    """
    engine = GroupRecommendationEngine(model_path)
    strategies = ["average", "least_misery", "most_pleasure", "consensus", "hybrid"]
    
    comparison_data = []
    
    for strategy in strategies:
        recs = engine.get_group_recommendations(
            group_ratings, movies_df, strategy=strategy, top_n=top_n
        )
        
        # Calculate strategy metrics
        group_scores = [score for _, score, _ in recs]
        individual_score_vars = []
        
        for _, _, individual_scores in recs:
            non_zero_scores = [s for s in individual_scores.values() if s > 0]
            if len(non_zero_scores) > 1:
                individual_score_vars.append(np.var(non_zero_scores))
        
        comparison_data.append({
            'Strategy': strategy,
            'Avg_Group_Score': np.mean(group_scores),
            'Min_Group_Score': np.min(group_scores),
            'Max_Group_Score': np.max(group_scores),
            'Avg_Individual_Variance': np.mean(individual_score_vars) if individual_score_vars else 0,
            'Top_Movie': recs[0][0] if recs else "None"
        })
    
    return pd.DataFrame(comparison_data)
