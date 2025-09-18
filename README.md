# letterboxdRecs

A collaborative filtering recommendation system that combines MovieLens community data with personal Letterboxd ratings to generate personalized movie recommendations.

## Features

- **Multiple Algorithm Support**: KNN (Basic, WithMeans, Baseline) and SVD-based (SVD, SVD++) collaborative filtering
- **Letterboxd Integration**: Import your Letterboxd ratings and watched history
- **TMDB Enrichment**: Automatically map Letterboxd movies to MovieLens dataset via TMDB IDs
- **Hyperparameter Tuning**: Automated grid search for optimal model parameters
- **Cross-Validation**: Robust model evaluation with k-fold cross-validation
- **Seeding System**: Generate reproducible and diverse recommendation sets
- **Group Recommendations**: Support for generating recommendations for multiple users

## Project Structure

```
letterboxdRecs/
├── src/                          # Core library code
│   ├── __init__.py
│   ├── config.py                 # Configuration and constants
│   ├── data_processing.py        # Data loading and preprocessing
│   ├── data_enrichment.py        # TMDB integration and data enrichment
│   ├── model_training.py         # Model training and evaluation
│   └── recommendations.py        # Recommendation generation
├── notebooks/                    # Jupyter notebooks for experiments
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_recommendations.ipynb
├── models/                       # Saved trained models
├── data/                         # Processed datasets
├── alex_data/                    # User's Letterboxd export data
├── ml-latest-small/              # MovieLens small dataset
├── ml-32m/                       # MovieLens 32M dataset
└── requirements.txt              # Python dependencies
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd letterboxdRecs
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download MovieLens datasets:
   - [MovieLens Latest Small](https://grouplens.org/datasets/movielens/latest/) → `ml-latest-small/`
   - [MovieLens 32M](https://grouplens.org/datasets/movielens/32m/) → `ml-32m/`

5. (Optional) Set up TMDB API for data enrichment:
```bash
export TMDB_API_KEY="your_api_key_here"
```

## Quick Start

### 1. Train Models

```python
from src.data_processing import load_movielens_data
from src.model_training import ModelTrainer

# Load data
data, ratings_df, movies_df, links_df = load_movielens_data(
    "ml-latest-small/ratings.csv",
    "ml-latest-small/movies.csv", 
    "ml-latest-small/links.csv"
)

# Train models
trainer = ModelTrainer(data)
knn_results = trainer.train_knn_models()
svd_results = trainer.train_svd_models()

# Save models
trainer.save_models()
```

### 2. Generate Recommendations

```python
from src.recommendations import RecommendationEngine, load_user_data_with_tmdb

# Load your Letterboxd data
user_ratings = load_user_data_with_tmdb(
    "alex_data/ratings_with_tmdb.csv",
    "ml-latest-small/links.csv"
)

# Generate recommendations
engine = RecommendationEngine("models/svdpp.pkl")
recommendations = engine.get_user_recommendations(
    user_ratings, 
    movies_df, 
    top_n=10,
    random_seed=42
)

for title, score in recommendations:
    print(f"• {title} — {score}★")
```

### 3. Enrich Letterboxd Data

```python
from src.data_enrichment import enrich_letterboxd_csv

# Add TMDB IDs to your Letterboxd export
enrich_letterboxd_csv(
    "alex_data/ratings.csv",
    "alex_data/ratings_with_tmdb.csv",
    sleep_between=0.1
)
```

## Advanced Features

### Hyperparameter Tuning

```python
# Train with hyperparameter optimization
trainer = ModelTrainer(data)
knn_results = trainer.train_knn_models(tune_hyperparams=True)
```

### Seeded Recommendations

```python
# Generate different recommendation sets with different seeds
recs_1 = engine.get_user_recommendations(user_ratings, movies_df, random_seed=42)
recs_2 = engine.get_user_recommendations(user_ratings, movies_df, random_seed=123)
recs_3 = engine.get_user_recommendations(user_ratings, movies_df, random_seed=999)
```

### Cross-Validation

```python
# Evaluate model performance
cv_results = trainer.cross_validate_models(cv=5)
for model, metrics in cv_results.items():
    print(f"{model}: RMSE = {metrics['rmse_mean']:.4f} ± {metrics['rmse_std']:.4f}")
```

## Datasets

- **MovieLens**: Community ratings dataset with 100k+ users and 60k+ movies
- **Letterboxd**: Personal movie ratings and watch history export
- **TMDB**: Movie metadata and ID mapping service

## Models

- **KNNBasic**: Simple k-nearest neighbors collaborative filtering
- **KNNWithMeans**: KNN with user/item mean normalization
- **KNNBaseline**: KNN with baseline estimates
- **SVD**: Singular Value Decomposition matrix factorization
- **SVD++**: Enhanced SVD with implicit feedback

## API Documentation

See the individual module docstrings for detailed API documentation:
- `src.data_processing`: Data loading and preprocessing utilities
- `src.model_training`: Model training and evaluation
- `src.recommendations`: Recommendation generation engine
- `src.data_enrichment`: TMDB integration and data enrichment

## TODO

- ✅ **Organize!** - Create proper project structure and modular code
- Create group-prediction capabilities for both SVD and KNN methods
- Create train/valid/test split and check errors and overfitting over random hyper-parameter search
- Train on large dataset
- Organize again!
- Create Svelte web app wrapper to allow for letterboxd data upload and hook it up to backend prediction

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

MIT License - see LICENSE file for details
