"""
letterboxdRecs: A collaborative filtering recommendation system

This package provides tools for:
- Training KNN and SVD-based recommendation models
- Generating individual and group recommendations
- Integrating Letterboxd data with MovieLens datasets
- Analyzing recommendation quality and group preferences
"""

from .config import *

# Convenience reexports for interactive / notebook use. Wrapped in try/except
# so slim runtime envs (e.g. the deployed API image without matplotlib /
# seaborn) can `import src.api` without pulling in dev-only deps. Direct
# imports (e.g. `from src.model_training import ModelTrainer`) still work.
try:
    from .data_processing import load_movielens_data, validate_data_quality
    from .model_training import ModelTrainer, EnhancedModelTrainer, compare_models
    from .recommendations import RecommendationEngine, load_user_data_with_tmdb
    from .group_recommendations import GroupRecommendationEngine, compare_group_strategies
    from .data_enrichment import enrich_letterboxd_csv, create_movielens_mapping
except ImportError:
    pass

__version__ = "1.0.0"
__author__ = "letterboxdRecs Team"

__all__ = [
    # Data processing
    'load_movielens_data',
    'validate_data_quality',
    
    # Model training
    'ModelTrainer',
    'compare_models',
    
    # Individual recommendations
    'RecommendationEngine',
    'load_user_data_with_tmdb',
    
    # Group recommendations
    'GroupRecommendationEngine',
    'compare_group_strategies',
    
    # Data enrichment
    'enrich_letterboxd_csv',
    'create_movielens_mapping',
]