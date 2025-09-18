#!/usr/bin/env python3
"""
Comprehensive hyperparameter search and overfitting analysis script.

This script demonstrates the enhanced model training capabilities with:
- Train/validation/test splits
- Random hyperparameter search
- Overfitting detection and analysis
- Performance visualization
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import pandas as pd
import numpy as np
from surprise import Dataset, Reader

from src.data_processing import load_letterboxd_data, create_surprise_dataset
from src.model_training import EnhancedModelTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_comprehensive_hyperparameter_search(data_path: str = "alex_data", 
                                           n_iter: int = 30,
                                           use_small_dataset: bool = False):
    """
    Run comprehensive hyperparameter search and overfitting analysis.
    
    Args:
        data_path: Path to the data directory (can be 'ml-latest-small', 'ml-32m', or custom path)
        n_iter: Number of hyperparameter combinations to try
        use_small_dataset: Whether to use a smaller subset for faster testing
    """
    logger.info("Starting comprehensive hyperparameter search...")
    
    try:
        # Load data
        logger.info(f"Loading data from {data_path}...")
        
        if data_path == "ml-latest-small":
            # Load MovieLens small dataset
            logger.info("Using MovieLens ml-latest-small dataset...")
            from surprise import Dataset, Reader
            reader = Reader(line_format="user item rating timestamp", sep=",", skip_lines=1)
            surprise_data = Dataset.load_from_file("ml-latest-small/ratings.csv", reader=reader)
        elif data_path == "ml-32m":
            # Load MovieLens large dataset
            logger.info("Using MovieLens ml-32m dataset...")
            from surprise import Dataset, Reader
            reader = Reader(line_format="user item rating timestamp", sep=",", skip_lines=1)
            surprise_data = Dataset.load_from_file("ml-32m/ratings.csv", reader=reader)
        elif os.path.exists(data_path):
            # Load Letterboxd data
            data = load_letterboxd_data(data_path)
            surprise_data = create_surprise_dataset(data)
        else:
            # Fallback to MovieLens builtin for testing
            logger.info("Using MovieLens builtin dataset for testing...")
            surprise_data = Dataset.load_builtin('ml-100k')
        
        # Use smaller dataset for faster testing if requested
        if use_small_dataset:
            logger.info("Using smaller dataset for faster testing...")
            # For quick testing, we'll just use the original dataset
            # but reduce the number of hyperparameter iterations
            pass
        
        # Initialize enhanced trainer
        logger.info("Initializing enhanced model trainer...")
        trainer = EnhancedModelTrainer(
            surprise_data, 
            val_size=0.2, 
            test_size=0.2, 
            random_state=42
        )
        
        # Run hyperparameter search
        logger.info(f"Running hyperparameter search with {n_iter} iterations...")
        search_results = trainer.random_hyperparameter_search(
            model_type='both',
            n_iter=n_iter,
            patience=10,
            save_results=True
        )
        
        # Evaluate final models on test set
        logger.info("Evaluating final models on test set...")
        final_results = trainer.evaluate_final_models()
        
        # Generate and display overfitting report
        logger.info("Generating overfitting analysis report...")
        overfitting_report = trainer.generate_overfitting_report()
        
        print("\n" + "="*80)
        print("HYPERPARAMETER SEARCH RESULTS")
        print("="*80)
        
        # Display best hyperparameters for each model
        for model_name, results in search_results.items():
            print(f"\n{model_name}:")
            print(f"  Best Validation RMSE: {results['best_val_rmse']:.4f}")
            print(f"  Best Parameters: {results['best_params']}")
            print(f"  Total Iterations: {results['total_iterations']}")
        
        print("\n" + "="*80)
        print("FINAL TEST SET EVALUATION")
        print("="*80)
        
        # Create comparison DataFrame
        comparison_data = []
        for model_name, results in final_results.items():
            comparison_data.append({
                'Model': model_name,
                'Train_RMSE': f"{results['train_rmse']:.4f}",
                'Val_RMSE': f"{results['val_rmse']:.4f}",
                'Test_RMSE': f"{results['test_rmse']:.4f}",
                'Generalization_Gap': f"{results['generalization_gap']:.4f}",
                'Val/Train_Ratio': f"{results['val_to_train_ratio']:.3f}",
                'Test/Train_Ratio': f"{results['test_to_train_ratio']:.3f}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.to_string(index=False))
        
        print("\n" + "="*80)
        print("OVERFITTING ANALYSIS")
        print("="*80)
        print(overfitting_report.to_string(index=False))
        
        # Analyze results
        print("\n" + "="*80)
        print("ANALYSIS & RECOMMENDATIONS")
        print("="*80)
        
        # Find best model overall
        best_model = min(final_results.items(), key=lambda x: x[1]['test_rmse'])
        print(f"\nBest performing model: {best_model[0]}")
        print(f"Test RMSE: {best_model[1]['test_rmse']:.4f}")
        
        # Analyze overfitting
        print(f"\nOverfitting Analysis:")
        for _, row in overfitting_report.iterrows():
            severity = row['Overfitting_Severity']
            model = row['Model']
            gap = row['Best_Overfitting_Score']
            
            if severity in ['High', 'Severe']:
                print(f"  ⚠️  {model}: {severity} overfitting (gap: {gap:.4f})")
                print(f"      Consider regularization or simpler parameters")
            elif severity == 'Moderate':
                print(f"  ⚡ {model}: {severity} overfitting (gap: {gap:.4f})")
                print(f"      Monitor performance, consider slight regularization")
            else:
                print(f"  ✅ {model}: {severity} overfitting (gap: {gap:.4f})")
        
        # Performance recommendations
        print(f"\nPerformance Recommendations:")
        sorted_models = sorted(final_results.items(), key=lambda x: x[1]['test_rmse'])
        
        for i, (model_name, results) in enumerate(sorted_models):
            rank = i + 1
            test_rmse = results['test_rmse']
            gap = results['generalization_gap']
            
            if rank == 1:
                print(f"  🥇 Use {model_name} for best accuracy (RMSE: {test_rmse:.4f})")
            elif gap < 0.05:  # Low overfitting
                print(f"  👍 {model_name} is well-generalized (RMSE: {test_rmse:.4f}, gap: {gap:.4f})")
            elif gap > 0.15:  # High overfitting
                print(f"  ⚠️  Avoid {model_name} due to overfitting (gap: {gap:.4f})")
        
        # Save models
        logger.info("Saving best models...")
        trainer.save_models()
        
        logger.info("Hyperparameter search completed successfully!")
        return trainer, search_results, final_results, overfitting_report
        
    except Exception as e:
        logger.error(f"Error during hyperparameter search: {e}")
        raise


def quick_test(data_path: str = "alex_data"):
    """Run a quick test with a small number of iterations."""
    logger.info("Running quick test with reduced iterations...")
    return run_comprehensive_hyperparameter_search(
        data_path=data_path,
        n_iter=10,
        use_small_dataset=True
    )


def full_search(data_path: str = "alex_data"):
    """Run full hyperparameter search."""
    logger.info("Running full hyperparameter search...")
    return run_comprehensive_hyperparameter_search(
        data_path=data_path,
        n_iter=50,
        use_small_dataset=False
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Hyperparameter search and overfitting analysis")
    parser.add_argument("--mode", choices=["quick", "full"], default="quick",
                       help="Run mode: quick test or full search")
    parser.add_argument("--iterations", type=int, default=None,
                       help="Number of hyperparameter combinations to try")
    parser.add_argument("--data-path", default="alex_data",
                       help="Path to data directory")
    
    args = parser.parse_args()
    
    try:
        if args.mode == "quick":
            quick_test(args.data_path)
        else:
            n_iter = args.iterations if args.iterations else 50
            run_comprehensive_hyperparameter_search(
                data_path=args.data_path,
                n_iter=n_iter,
                use_small_dataset=False
            )
    except KeyboardInterrupt:
        logger.info("Search interrupted by user")
    except Exception as e:
        logger.error(f"Search failed: {e}")
        sys.exit(1)
