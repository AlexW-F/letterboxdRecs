"""
Configuration settings and constants for letterboxdRecs.
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
ALEX_DATA_DIR = PROJECT_ROOT / "alex_data"

# MovieLens data paths
ML_SMALL_DIR = PROJECT_ROOT / "ml-latest-small"
ML_32M_DIR = PROJECT_ROOT / "ml-32m"

# Default model parameters
DEFAULT_KNN_PARAMS = {
    'k': 40,
    'sim_options': {'name': 'cosine', 'user_based': False},
    'verbose': False
}

DEFAULT_SVD_PARAMS = {
    'n_factors': 50,
    'lr_all': 0.005,
    'reg_all': 0.02,
    'verbose': False
}

DEFAULT_SVDPP_PARAMS = {
    'n_factors': 50,
    'lr_all': 0.005,
    'reg_all': 0.02,
    'verbose': False,
    'n_jobs': -1
}

# Hyperparameter grids for tuning
KNN_PARAM_GRID = {
    'k': [20, 30, 40, 50],
    'sim_options': {
        'name': ['cosine', 'pearson'],
        'user_based': [False, True]
    }
}

SVD_PARAM_GRID = {
    'n_factors': [50, 100, 150],
    'lr_all': [0.002, 0.005, 0.01],
    'reg_all': [0.01, 0.02, 0.05]
}

# API settings
TMDB_API_KEY = os.getenv('TMDB_API_KEY')
TMDB_BASE_URL = "https://api.themoviedb.org/3"

# Logging configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
}

# Data validation settings
MIN_RATING = 0.5
MAX_RATING = 5.0
MIN_USERS_PER_MOVIE = 5
MIN_MOVIES_PER_USER = 5

# Recommendation settings
DEFAULT_TOP_N = 10
DEFAULT_K_NEIGHBORS = 40
DEFAULT_RANDOM_SEED = 42
