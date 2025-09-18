#!/usr/bin/env python3
"""
Simple test to verify the group recommendations work with proper ID mapping.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.data_processing import load_movielens_data
from src.group_recommendations import GroupRecommendationEngine

def test_basic_group_recs():
    """Basic test of group recommendations with known good data."""
    
    print("🎬 Basic Group Recommendation Test")
    print("=" * 50)
    
    # Load data
    print("Loading data...")
    try:
        _, ratings_df, movies_df, links_df = load_movielens_data(
            "ml-latest-small/ratings.csv",
            "ml-latest-small/movies.csv", 
            "ml-latest-small/links.csv"
        )
        print(f"✓ Loaded {len(movies_df)} movies, {len(ratings_df)} ratings")
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return False
    
    # Create simple test group with known MovieLens IDs (as integers!)
    print("\nCreating test group...")
    group_ratings = {
        'User1': {
            1: 5.0,    # Toy Story 
            2: 4.0,    # Jumanji
            6: 4.5,    # Heat
            10: 3.5,   # GoldenEye
        },
        'User2': {
            1: 3.0,    # Toy Story (different preference)
            3: 5.0,    # Grumpier Old Men  
            6: 2.0,    # Heat (dislikes action)
            16: 4.5,   # Casino
        }
    }
    
    print(f"✓ Created group with {len(group_ratings)} users")
    
    # Test with SVD++ model
    model_path = "models/svdpp_best.pkl"
    if not os.path.exists(model_path):
        print(f"✗ Model not found: {model_path}")
        return False
    
    print(f"\nTesting {model_path}...")
    try:
        engine = GroupRecommendationEngine(model_path=model_path)
        
        # Test average strategy
        recs = engine.get_group_recommendations(
            group_ratings=group_ratings,
            movies_df=movies_df,
            strategy='average',
            top_n=5,
            exclude_watched=True,
            random_seed=42
        )
        
        if recs and len(recs) > 0:
            print("✅ Group recommendations successful!")
            print("Top 5 recommendations:")
            for i, (title, score) in enumerate(recs[:5], 1):
                print(f"  {i}. {title} (score: {score:.2f})")
            return True
        else:
            print("⚠️ No recommendations generated")
            return False
            
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_basic_group_recs()
    if success:
        print("\n🎉 Basic test passed! Group recommendations are working.")
    else:
        print("\n❌ Basic test failed. Need to debug further.")
