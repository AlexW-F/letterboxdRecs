#!/usr/bin/env python3
"""
Simple test to verify group recommendations work.
"""
import os
import sys
import pandas as pd

# Add project root to path
sys.path.insert(0, '/Users/awf/Desktop/Projects/letterboxdRecs')

try:
    from src.data_processing import load_movielens_data
    from src.group_recommendations import GroupRecommendationEngine
    print("✓ Imports successful")
except Exception as e:
    print(f"✗ Import error: {e}")
    exit(1)

def main():
    print("🎬 Simple Group Recommendation Test")
    print("=" * 40)
    
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
    
    # Test with simple integer IDs (as expected by MovieLens models)
    group_ratings = {
        'User1': {1: 5.0, 2: 4.0, 6: 4.5},  # Integer IDs
        'User2': {1: 3.0, 3: 5.0, 6: 2.0}
    }
    
    # Test SVD++ model
    model_path = "models/svdpp_best.pkl"
    if os.path.exists(model_path):
        print(f"\nTesting {model_path}...")
        try:
            engine = GroupRecommendationEngine(model_path=model_path)
            
            recs = engine.get_group_recommendations(
                group_ratings=group_ratings,
                movies_df=movies_df,
                strategy='average',
                top_n=3
            )
            
            if recs:
                print("✅ SUCCESS! Recommendations generated:")
                for i, (title, score) in enumerate(recs[:3], 1):
                    print(f"  {i}. {title} (score: {score:.2f})")
                return True
            else:
                print("⚠️ No recommendations returned")
                return False
                
        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()
            return False
    else:
        print(f"✗ Model not found: {model_path}")
        return False

if __name__ == "__main__":
    success = main()
    print("\n" + ("🎉 Test PASSED!" if success else "❌ Test FAILED!"))
