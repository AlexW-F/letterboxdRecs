#!/usr/bin/env python3
import sys
import os
sys.path.append('/Users/awf/Desktop/Projects/letterboxdRecs')
os.chdir('/Users/awf/Desktop/Projects/letterboxdRecs')

print("🔍 Quick test of group recommendation system...")

# Test basic imports
try:
    from src.data_processing import load_movielens_data
    from src.group_recommendations import GroupRecommendationEngine
    import pickle
    print("✓ All imports successful")
except Exception as e:
    print(f"✗ Import error: {e}")
    exit(1)

# Test model loading
try:
    with open('models/svdpp_best.pkl', 'rb') as f:
        model = pickle.load(f)
    
    if hasattr(model, 'trainset'):
        all_items = list(model.trainset._raw2inner_id_items.keys())
        print(f"✓ Model loaded: {len(all_items)} items")
        print(f"   Sample item IDs: {all_items[:5]}")
    else:
        print("✗ Model has no trainset")
        exit(1)
        
except Exception as e:
    print(f"✗ Model loading error: {e}")
    exit(1)

# Test group recommendations with one simple case
try:
    # Create simple test ratings using actual item IDs
    test_group = {
        'User1': {
            str(all_items[0]): 5.0,
            str(all_items[1]): 4.0,
            str(all_items[2]): 3.0
        }
    }
    
    # Load movies data
    _, _, movies_df, _ = load_movielens_data(
        "ml-latest-small/ratings.csv",
        "ml-latest-small/movies.csv", 
        "ml-latest-small/links.csv"
    )
    
    # Test group recommendation
    engine = GroupRecommendationEngine(model_path='models/svdpp_best.pkl')
    recs = engine.get_group_recommendations(
        group_ratings=test_group,
        movies_df=movies_df,
        strategy='average',
        top_n=3
    )
    
    if recs:
        print(f"✅ Group recommendations working!")
        print(f"   Top recommendation: {recs[0][0]} (score: {recs[0][1]})")
    else:
        print("⚠️ No recommendations generated")
    
except Exception as e:
    print(f"✗ Group recommendation error: {e}")
    import traceback
    traceback.print_exc()

print("🎉 Quick test completed!")
