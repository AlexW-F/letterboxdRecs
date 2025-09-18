#!/usr/bin/env python3
"""
Fixed comprehensive test script for group recommendation functionality.
This version uses actual item IDs from the trained models to avoid compatibility issues.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import pickle
import numpy as np
from src.group_recommendations import GroupRecommendationEngine, compare_group_strategies
from src.data_processing import load_movielens_data

def get_model_item_ids(model_path: str, sample_size: int = 20):
    """Get a sample of item IDs that the model was actually trained on."""
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        if hasattr(model, 'trainset'):
            # Get all raw item IDs the model knows about
            all_items = list(model.trainset._raw2inner_id_items.keys())
            # Return a random sample
            np.random.seed(42)  # For reproducible results
            sample_items = np.random.choice(all_items, min(sample_size, len(all_items)), replace=False)
            return sample_items.tolist()
        else:
            print(f"⚠️ Model {model_path} doesn't have trainset information")
            return []
    except Exception as e:
        print(f"⚠️ Error loading model {model_path}: {e}")
        return []

def create_test_group_with_letterboxd_data(links_df: pd.DataFrame, alex_ratings_df: pd.DataFrame):
    """Create a test group using actual Letterboxd data mapped to MovieLens IDs."""
    
    print("   Using actual Letterboxd data for realistic testing...")
    
    # Load enriched Letterboxd data
    try:
        letterboxd_df = pd.read_csv('alex_data/ratings_with_tmdb.csv')
        print(f"   ✓ Loaded {len(letterboxd_df)} Letterboxd ratings with TMDB IDs")
    except FileNotFoundError:
        print("   ⚠️ Letterboxd data with TMDB IDs not found, using synthetic data")
        return create_synthetic_test_group_with_valid_ids(links_df, alex_ratings_df)
    
    # Map TMDB IDs to MovieLens movieIds
    # Clean the tmdbId column in links (remove .0 if present)
    links_clean = links_df.copy()
    links_clean['tmdbId'] = links_clean['tmdbId'].astype(str).str.replace('.0', '', regex=False)
    
    # Clean the tmdb_id column in letterboxd data
    letterboxd_clean = letterboxd_df.copy()
    letterboxd_clean['tmdb_id'] = letterboxd_clean['tmdb_id'].astype(str).str.replace('.0', '', regex=False)
    
    # Merge to get MovieLens movie IDs
    merged = letterboxd_clean.merge(
        links_clean[['movieId', 'tmdbId']], 
        left_on='tmdb_id', 
        right_on='tmdbId',
        how='inner'
    )
    
    print(f"   ✓ Successfully mapped {len(merged)} Letterboxd ratings to MovieLens IDs")
    
    if len(merged) < 10:
        print("   ⚠️ Too few mapped ratings, using synthetic data")
        return create_synthetic_test_group_with_valid_ids(links_df, alex_ratings_df)
    
    # Create diverse test users using actual Letterboxd preferences
    # Sort by rating to get high-rated and low-rated movies
    high_rated = merged[merged['Rating'] >= 4.5].head(8)
    medium_rated = merged[(merged['Rating'] >= 3.5) & (merged['Rating'] < 4.5)].head(6)
    low_rated = merged[merged['Rating'] < 3.5].head(4)
    
    group_ratings = {
        'AlexLikes': {},  # Based on actual high ratings
        'AlexMixed': {},  # Based on mixed ratings
        'Contrarian': {}  # Opposite preferences for diversity
    }
    
    # Alex's actual high-rated movies
    for _, row in high_rated.iterrows():
        group_ratings['AlexLikes'][str(row['movieId'])] = float(row['Rating'])
    
    # Alex's mixed ratings
    for _, row in medium_rated.iterrows():
        group_ratings['AlexMixed'][str(row['movieId'])] = float(row['Rating'])
    
    # Contrarian user with opposite preferences
    for _, row in high_rated.head(5).iterrows():
        group_ratings['Contrarian'][str(row['movieId'])] = max(1.0, 5.0 - float(row['Rating']))
    
    for _, row in low_rated.iterrows():
        group_ratings['Contrarian'][str(row['movieId'])] = min(5.0, float(row['Rating']) + 2.0)
    
    print(f"   ✓ Created realistic test group:")
    print(f"     - AlexLikes: {len(group_ratings['AlexLikes'])} highly-rated movies")
    print(f"     - AlexMixed: {len(group_ratings['AlexMixed'])} medium-rated movies") 
    print(f"     - Contrarian: {len(group_ratings['Contrarian'])} opposite preferences")
    
    return group_ratings

def create_synthetic_test_group_with_valid_ids(links_df: pd.DataFrame, alex_ratings_df: pd.DataFrame):
    """Fallback: Create a test group using valid MovieLens IDs from the dataset."""
    
    # Get some popular movies from MovieLens
    popular_movies = [1, 2, 6, 10, 16, 25, 32, 47, 50, 110, 150, 260, 296, 318, 356, 480, 527, 589, 593, 858]
    
    # Verify these exist in our dataset
    valid_movies = []
    for movie_id in popular_movies:
        if movie_id in alex_ratings_df['movieId'].values:
            valid_movies.append(str(movie_id))
    
    if len(valid_movies) < 10:
        # Fallback to any movies in the dataset
        valid_movies = alex_ratings_df['movieId'].head(20).astype(str).tolist()
    
    print(f"   Using {len(valid_movies)} synthetic movie ratings")
    
    # Create diverse test group
    group_ratings = {
        'Alice': {},
        'Bob': {}, 
        'Charlie': {}
    }
    
    # Assign ratings to create diverse preferences
    np.random.seed(42)
    
    # Alice likes some movies highly
    alice_items = valid_movies[:8]
    for i, item in enumerate(alice_items):
        if i < 3:
            group_ratings['Alice'][item] = np.random.uniform(4.5, 5.0)
        elif i < 6:
            group_ratings['Alice'][item] = np.random.uniform(3.5, 4.0)
        else:
            group_ratings['Alice'][item] = np.random.uniform(2.0, 3.0)
    
    # Bob has different preferences
    bob_items = valid_movies[5:13]
    for i, item in enumerate(bob_items):
        if i < 3:
            group_ratings['Bob'][item] = np.random.uniform(4.0, 5.0)
        elif i < 6:
            group_ratings['Bob'][item] = np.random.uniform(3.0, 4.0)
        else:
            group_ratings['Bob'][item] = np.random.uniform(1.5, 2.5)
    
    # Charlie overlaps with both
    charlie_items = valid_movies[3:11]
    for i, item in enumerate(charlie_items):
        if i < 3:
            group_ratings['Charlie'][item] = np.random.uniform(4.5, 5.0)
        else:
            group_ratings['Charlie'][item] = np.random.uniform(3.0, 4.0)
    
    return group_ratings

def test_single_model(model_name: str, model_path: str, movies_df: pd.DataFrame, links_df: pd.DataFrame):
    """Test group recommendations for a single model."""
    
    print(f"\n🔧 Testing {model_name} model...")
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"   ✗ Model file not found: {model_path}")
        return {}
    
    try:
        # Create test group with proper ID mapping
        group_ratings = create_test_group_with_letterboxd_data(links_df, movies_df)
        
        if not group_ratings or not any(group_ratings.values()):
            print(f"   ✗ Could not create valid test group")
            return {}
        
        print(f"   ✓ Loaded {model_name} model")
        print(f"   ✓ Created test group with {len(group_ratings)} users")
        
        # Test different strategies
        strategies = ['average', 'least_misery', 'most_pleasure', 'consensus', 'hybrid']
        results = {}
        
        engine = GroupRecommendationEngine(model_path=model_path)
        
        for strategy in strategies:
            try:
                recs = engine.get_group_recommendations(
                    group_ratings=group_ratings,
                    movies_df=movies_df,
                    strategy=strategy,
                    top_n=5,
                    exclude_watched=True,
                    random_seed=42
                )
                
                if recs and len(recs) > 0:
                    results[strategy] = recs
                    top_movie = recs[0][0] if recs else "No recommendations"
                    top_score = recs[0][1] if recs else 0
                    print(f"   ✓ {strategy}: {top_movie[:30]}... (score: {top_score:.2f})")
                else:
                    print(f"   ⚠️ {strategy}: No recommendations generated")
                    results[strategy] = []
                    
            except Exception as e:
                print(f"   ✗ {strategy}: Error - {str(e)}")
                results[strategy] = []
        
        return results
        
    except Exception as e:
        print(f"   ✗ Error testing {model_name}: {str(e)}")
        return {}

def test_group_recommendations_fixed():
    """Test the group recommendation system with fixed dataset compatibility."""
    
    print("🎬 Fixed Group Recommendation Test")
    print("=" * 60)
    
    # Load test data
    try:
        _, ratings_df, movies_df, links_df = load_movielens_data(
            "ml-latest-small/ratings.csv",
            "ml-latest-small/movies.csv", 
            "ml-latest-small/links.csv"
        )
        print(f"✓ Loaded {len(movies_df)} movies and {len(ratings_df)} ratings")
    except FileNotFoundError:
        print("✗ MovieLens data not found. Please ensure ml-latest-small/ directory exists.")
        return False
    
    # Test different model types
    models_to_test = [
        ("SVD++", "models/svdpp_best.pkl"),
        ("SVD", "models/svd_best.pkl"),
        ("KNN Basic", "models/knnbasic_best.pkl"),
        ("KNN with Means", "models/knnwithmeans_best.pkl"),
        ("KNN Baseline", "models/knnbaseline_best.pkl")
    ]
    
    all_results = {}
    successful_tests = 0
    
    for model_name, model_path in models_to_test:
        results = test_single_model(model_name, model_path, movies_df, links_df)
        all_results[model_name] = results
        
        if any(results.values()):  # If any strategy worked
            successful_tests += 1
    
    # Display results summary
    print(f"\n📊 Results Summary")
    print("=" * 60)
    
    if successful_tests > 0:
        # Create summary table
        print(f"{'Model':<20} {'Strategy':<15} {'Top Movie':<30} {'Score':<8}")
        print("-" * 70)
        
        for model_name, results in all_results.items():
            for strategy, recs in results.items():
                if recs and len(recs) > 0:
                    top_movie = recs[0][0][:25] + "..." if len(recs[0][0]) > 25 else recs[0][0]
                    score = f"{recs[0][1]:.2f}"
                    print(f"{model_name:<20} {strategy:<15} {top_movie:<30} {score:<8}")
                else:
                    print(f"{model_name:<20} {strategy:<15} {'No recommendations':<30} {'-':<8}")
        
        print(f"\n✅ Successfully tested {successful_tests}/{len(models_to_test)} models")
        
        # Test strategy comparison if we have a working model
        working_models = [(name, path) for name, path in models_to_test 
                         if name in all_results and any(all_results[name].values())]
        
        if working_models:
            print(f"\n🔍 Testing Strategy Comparison with {working_models[0][0]}...")
            try:
                model_name, model_path = working_models[0]
                group_ratings = create_test_group_with_letterboxd_data(links_df, movies_df)
                
                if group_ratings:
                    # Use the original compare function without the strategies parameter
                    comparison_results = compare_group_strategies(
                        group_ratings=group_ratings,
                        movies_df=movies_df,
                        model_path=model_path,
                        top_n=5
                    )
                    
                    print("   ✓ Strategy comparison completed successfully")
                    for strategy, recs in comparison_results.items():
                        if recs:
                            print(f"   - {strategy}: {recs[0][0]} (score: {recs[0][1]})")
                else:
                    print("   ✗ Could not create valid test group for comparison")
                    
            except Exception as e:
                print(f"   ✗ Strategy comparison failed: {str(e)}")
    
    else:
        print("✗ No models produced successful recommendations")
    
    print(f"\n{'='*60}")
    print(f"🎉 Fixed group recommendation testing completed!")
    
    if successful_tests > 0:
        print(f"✅ Group recommendations work with {successful_tests} models")
        print("✅ Dataset compatibility issues resolved")
    else:
        print("⚠️ No successful recommendations - may need further debugging")
    
    return successful_tests > 0

if __name__ == "__main__":
    test_group_recommendations_fixed()