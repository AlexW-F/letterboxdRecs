# letterboxdRecs

## COLLABORATIVE FILTERING RECOMMENDATION ENGINE
• Implemented multiple algorithm support including KNN (Basic, WithMeans, Baseline) and SVD-based (SVD, SVD++) collaborative filtering
• Built comprehensive hyperparameter search with early stopping and overfitting detection across 50+ parameter combinations  
• Architected train/validation/test splits with robust evaluation metrics for model comparison
• Optimized recommendation generation using matrix factorization fold-in techniques for new users

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

## REST API

Phase 3 added a FastAPI backend. After training the models (`scripts/train_svd.py`, `scripts/train_als.py`) and building content features (`scripts/build_content_features.py`):

```bash
uvicorn src.api.main:app --reload --port 8000
```

OpenAPI docs at `http://localhost:8000/docs`. Endpoints:

| Endpoint | Purpose |
|---|---|
| `GET /health` | service + model-loaded status |
| `GET /modes` | list of recommendation modes (`balanced`, `niche`, `popular`, `serendipitous`) with descriptions + weight breakdown |
| `GET /strategies` | list of group strategies (`average`, `least_misery`, `most_pleasure`, `consensus`, `hybrid`, `group_taste_vector`) |
| `POST /upload-letterboxd` | upload `ratings_with_tmdb.csv` (+ optional `watched_with_tmdb.csv`). Returns a SHA-256 content hash that doubles as a session id. |
| `POST /recommend/individual` | `{hash, mode, top_n, exclude_rated?, exclude_watched?}` → top-N recs with per-rec explanations |
| `POST /recommend/group` | `{hashes, member_names?, strategy, mode, top_n}` → group recs with per-member scores + fairness |
| `POST /group/analyze` | `{hashes, member_names?}` → pairwise similarity (Pearson on shared ratings + cosine on TF-IDF taste vectors), consensus and disagreement films |

Caching is content-addressable — re-uploading the same export hits the same diskcache entry, no session lifecycle.

### Full stack via docker compose

The fastest way to run everything locally — FastAPI + SvelteKit + nginx, with the frontend proxying `/api/*` to the backend over the internal docker network:

```bash
docker compose up --build
# then open http://localhost:5173/
```

What the stack expects on disk (read-only mounted into the api container):
- `models/svd_full_slim.pkl` — Surprise SVD on ml-32m (run `scripts/train_svd.py` then `scripts/slim_svd_pickle.py`)
- `models/als_full.pkl` — implicit ALS on ml-32m (run `scripts/train_als.py --ratings ml-32m/ratings.csv`)
- `data/content_genome.{npz,json}` — Tag Genome 2021 content features (download `genome_2021.zip` from grouplens.org into `archives/`, unzip, then `scripts/build_genome_features.py`)
- `ml-32m/` — the MovieLens 32M dataset

Optional TMDB poster fetching in the UI: set `VITE_TMDB_KEY` before `docker compose build` to bake a read-only TMDB API key into the SvelteKit bundle. Without it the cards show a placeholder.

See `.env.example` for tunable paths (`MODELS_DIR`, `ML_DATA_DIR`, `CONTENT_FEATURES`, `CACHE_DIR`, `SVD_FILE`, `ALS_FILE`).

### Backend only

```bash
docker build -t letterboxd-recs-api .
docker run -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/ml-32m:/app/ml-32m \
  -v $(pwd)/data:/app/data \
  letterboxd-recs-api
```

Or natively for development:

```bash
uvicorn src.api.main:app --reload --port 8000
```

## Library API

See the individual module docstrings for detailed API documentation:
- `src.data_processing`: Data loading and preprocessing utilities
- `src.model_training`: Model training and evaluation
- `src.recommendations`: Recommendation generation engine
- `src.data_enrichment`: TMDB integration and data enrichment
- `src.reranking`: Phase 1 candidate-gen → re-rank pipeline (`Reranker` is the new public surface)
- `src.group_reranker`: Group recommendations across 6 strategies
- `src.content_features`: Content scorer — Tag Genome 2021 by default, TF-IDF fallback (Phase 2 / 2.5)
- `src.evaluation`: Offline eval harness (NDCG@k, recall@k, coverage, Gini, intra-list diversity, group fairness)
- `src.api`: FastAPI app + state singleton + routers (Phase 3)
- `web/`: SvelteKit frontend (Phase 4)

## Visualization

`scripts/build_movie_space_viz.py` writes `evaluation_results/movie_space_alex.html` — a 3D UMAP projection of the ALS latent space with the user's rated films highlighted and their folded-in taste vector plotted in the same space. Useful for spot-checking why the recommender clusters things the way it does.

## TODO

- ✅ **Organize!** — modular `src/` layout
- ✅ Group-prediction capabilities for both SVD and KNN methods — `GroupReranker` with 6 strategies including `group_taste_vector`
- ✅ Train/valid/test split + overfitting detection — `src.model_training.EnhancedModelTrainer`
- ✅ Train on large dataset — SVD + ALS now on ml-32m (200,948 users × 84,432 items)
- ✅ Create Svelte web app — SvelteKit in `web/`, deployed via docker compose

Queued upgrades (see `evaluation_results/phase0_findings.md` and the subagent research):
- Sentence-transformer embeddings on TMDB overview text as a second content scorer
- Director-id features from IMDb bulk dumps (cinephile-critical signal)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

MIT License - see LICENSE file for details
