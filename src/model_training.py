"""
Model training and evaluation utilities for letterboxdRecs.

This module contains functions for:
- Training KNN and SVD models
- Hyperparameter tuning
- Model evaluation and validation
- Model persistence
"""

import pickle
import numpy as np
import pandas as pd
from surprise import KNNBasic, KNNWithMeans, KNNBaseline, SVDpp, SVD
from surprise.model_selection import train_test_split, GridSearchCV, cross_validate
from surprise import accuracy
from typing import Dict, List, Tuple, Any, Optional
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import ParameterSampler
import warnings
import time
from datetime import datetime
import json
import os

logger = logging.getLogger(__name__)


class EnhancedModelTrainer:
    """Enhanced model trainer with train/validation/test splits and overfitting detection."""
    
    def __init__(self, data, val_size: float = 0.2, test_size: float = 0.2, random_state: int = 42):
        """
        Initialize the enhanced model trainer with train/validation/test splits.
        
        Args:
            data: Surprise Dataset object
            val_size: Proportion of data to use for validation
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
        """
        self.data = data
        self.val_size = val_size
        self.test_size = test_size
        self.random_state = random_state
        
        # Create train/validation/test splits
        self._create_splits()
        
        self.models = {}
        self.validation_history = {}
        self.hyperparameter_results = {}
        self.best_models = {}
        
    def _create_splits(self):
        """Create train/validation/test splits from the data."""
        # First split: separate test set
        trainset_full, self.testset = train_test_split(
            self.data, 
            test_size=self.test_size, 
            random_state=self.random_state
        )
        
        # Second split: separate validation set from remaining training data
        # We need to create a new dataset from the remaining data for further splitting
        remaining_ratings = []
        for uid, iid, rating in trainset_full.all_ratings():
            remaining_ratings.append((uid, iid, rating))
        
        # Create temporary dataset from remaining ratings
        from surprise import Dataset, Reader
        reader = Reader(rating_scale=(1, 5))
        temp_df = pd.DataFrame(remaining_ratings, columns=['userID', 'itemID', 'rating'])
        temp_dataset = Dataset.load_from_df(temp_df, reader)
        
        # Split into train and validation
        adjusted_val_size = self.val_size / (1 - self.test_size)
        self.trainset, self.validset = train_test_split(
            temp_dataset, 
            test_size=adjusted_val_size,
            random_state=self.random_state
        )
        
        logger.info(f"Data splits created:")
        logger.info(f"  Training set: {self.trainset.n_ratings} ratings")
        logger.info(f"  Validation set: {len(self.validset)} ratings")
        logger.info(f"  Test set: {len(self.testset)} ratings")
    
    def random_hyperparameter_search(self, 
                                   model_type: str = 'both',
                                   n_iter: int = 50,
                                   patience: int = 10,
                                   save_results: bool = True) -> Dict[str, Any]:
        """
        Perform random hyperparameter search with overfitting detection.
        
        Args:
            model_type: Type of models to search ('knn', 'svd', or 'both')
            n_iter: Number of random hyperparameter combinations to try
            patience: Number of iterations without improvement before early stopping
            save_results: Whether to save detailed results to disk
            
        Returns:
            Dictionary with best hyperparameters and performance metrics
        """
        logger.info(f"Starting random hyperparameter search with {n_iter} iterations...")
        
        results = {}
        
        if model_type in ['knn', 'both']:
            results.update(self._search_knn_hyperparams(n_iter, patience))
            
        if model_type in ['svd', 'both']:
            results.update(self._search_svd_hyperparams(n_iter, patience))
        
        if save_results:
            self._save_hyperparameter_results(results)
            
        return results
    
    def _search_knn_hyperparams(self, n_iter: int, patience: int) -> Dict[str, Any]:
        """Search hyperparameters for KNN models."""
        knn_models = {
            'KNNBasic': KNNBasic,
            'KNNWithMeans': KNNWithMeans,
            'KNNBaseline': KNNBaseline
        }
        
        # Define hyperparameter distributions
        param_distributions = {
            'k': range(10, 101, 5),  # [10, 15, 20, ..., 100]
            'sim_options': [
                {'name': 'cosine', 'user_based': False},
                {'name': 'cosine', 'user_based': True},
                {'name': 'pearson', 'user_based': False},
                {'name': 'pearson', 'user_based': True},
                {'name': 'msd', 'user_based': False},
                {'name': 'msd', 'user_based': True}
            ]
        }
        
        results = {}
        
        for model_name, model_class in knn_models.items():
            logger.info(f"Searching hyperparameters for {model_name}...")
            
            best_val_score = float('inf')
            best_params = None
            best_model = None
            no_improvement_count = 0
            
            search_history = []
            
            # Generate random parameter combinations
            param_sampler = ParameterSampler(param_distributions, n_iter=n_iter, random_state=self.random_state)
            
            for i, params in enumerate(param_sampler):
                start_time = time.time()
                
                try:
                    # Create and train model
                    model = model_class(**params, verbose=False)
                    model.fit(self.trainset)
                    
                    # Evaluate on validation set
                    val_predictions = model.test(self.validset)
                    val_rmse = accuracy.rmse(val_predictions, verbose=False)
                    val_mae = accuracy.mae(val_predictions, verbose=False)
                    
                    # Evaluate on training set (to detect overfitting)
                    # Build training test data from trainset
                    train_test_data = [(uid, iid, rating) for (uid, iid, rating) in self.trainset.all_ratings()]
                    train_predictions = model.test(train_test_data)
                    train_rmse = accuracy.rmse(train_predictions, verbose=False)
                    train_mae = accuracy.mae(train_predictions, verbose=False)
                    
                    train_time = time.time() - start_time
                    
                    # Calculate overfitting score (difference between train and validation RMSE)
                    overfitting_score = val_rmse - train_rmse
                    
                    search_history.append({
                        'iteration': i + 1,
                        'params': params.copy(),
                        'train_rmse': train_rmse,
                        'train_mae': train_mae,
                        'val_rmse': val_rmse,
                        'val_mae': val_mae,
                        'overfitting_score': overfitting_score,
                        'train_time': train_time
                    })
                    
                    # Check if this is the best model so far
                    if val_rmse < best_val_score:
                        best_val_score = val_rmse
                        best_params = params.copy()
                        best_model = model
                        no_improvement_count = 0
                        logger.info(f"  New best {model_name} - Val RMSE: {val_rmse:.4f}, Overfitting: {overfitting_score:.4f}")
                    else:
                        no_improvement_count += 1
                    
                    # Early stopping
                    if no_improvement_count >= patience:
                        logger.info(f"  Early stopping for {model_name} after {i+1} iterations")
                        break
                        
                except Exception as e:
                    logger.warning(f"  Error with params {params}: {e}")
                    continue
            
            # Store results
            results[model_name] = {
                'best_params': best_params,
                'best_val_rmse': best_val_score,
                'best_model': best_model,
                'search_history': search_history,
                'total_iterations': len(search_history)
            }
            
            self.models[model_name] = best_model
            self.validation_history[model_name] = search_history
            self.best_models[model_name] = best_model  # Keep track of best models
            
        return results
    
    def _search_svd_hyperparams(self, n_iter: int, patience: int) -> Dict[str, Any]:
        """Search hyperparameters for SVD models."""
        svd_models = {
            'SVD': SVD,
            'SVDpp': SVDpp
        }
        
        # Define hyperparameter distributions
        param_distributions = {
            'n_factors': [20, 50, 100, 150, 200],
            'lr_all': [0.001, 0.002, 0.005, 0.01, 0.02],
            'reg_all': [0.005, 0.01, 0.02, 0.05, 0.1],
            'n_epochs': [10, 20, 30, 50]
        }
        
        results = {}
        
        for model_name, model_class in svd_models.items():
            logger.info(f"Searching hyperparameters for {model_name}...")
            
            best_val_score = float('inf')
            best_params = None
            best_model = None
            no_improvement_count = 0
            
            search_history = []
            
            # Generate random parameter combinations
            param_sampler = ParameterSampler(param_distributions, n_iter=n_iter, random_state=self.random_state)
            
            for i, params in enumerate(param_sampler):
                start_time = time.time()
                
                try:
                    # Create and train model
                    model_params = params.copy()
                    model_params['verbose'] = False
                    
                    if model_name == 'SVDpp':
                        # SVD++ doesn't support n_jobs parameter
                        pass
                    else:
                        # Regular SVD can use n_jobs for faster training
                        pass
                    
                    model = model_class(**model_params)
                    model.fit(self.trainset)
                    
                    # Evaluate on validation set
                    val_predictions = model.test(self.validset)
                    val_rmse = accuracy.rmse(val_predictions, verbose=False)
                    val_mae = accuracy.mae(val_predictions, verbose=False)
                    
                    # Evaluate on training set (to detect overfitting)
                    # Build training test data from trainset
                    train_test_data = [(uid, iid, rating) for (uid, iid, rating) in self.trainset.all_ratings()]
                    train_predictions = model.test(train_test_data)
                    train_rmse = accuracy.rmse(train_predictions, verbose=False)
                    train_mae = accuracy.mae(train_predictions, verbose=False)
                    
                    train_time = time.time() - start_time
                    
                    # Calculate overfitting score
                    overfitting_score = val_rmse - train_rmse
                    
                    search_history.append({
                        'iteration': i + 1,
                        'params': params.copy(),
                        'train_rmse': train_rmse,
                        'train_mae': train_mae,
                        'val_rmse': val_rmse,
                        'val_mae': val_mae,
                        'overfitting_score': overfitting_score,
                        'train_time': train_time
                    })
                    
                    # Check if this is the best model so far
                    if val_rmse < best_val_score:
                        best_val_score = val_rmse
                        best_params = params.copy()
                        best_model = model
                        no_improvement_count = 0
                        logger.info(f"  New best {model_name} - Val RMSE: {val_rmse:.4f}, Overfitting: {overfitting_score:.4f}")
                    else:
                        no_improvement_count += 1
                    
                    # Early stopping
                    if no_improvement_count >= patience:
                        logger.info(f"  Early stopping for {model_name} after {i+1} iterations")
                        break
                        
                except Exception as e:
                    logger.warning(f"  Error with params {params}: {e}")
                    continue
            
            # Store results
            results[model_name] = {
                'best_params': best_params,
                'best_val_rmse': best_val_score,
                'best_model': best_model,
                'search_history': search_history,
                'total_iterations': len(search_history)
            }
            
            self.models[model_name] = best_model
            self.validation_history[model_name] = search_history
            self.best_models[model_name] = best_model  # Keep track of best models
            
        return results
    
    def evaluate_final_models(self) -> Dict[str, Dict]:
        """
        Evaluate the best models on the test set.
        
        Returns:
            Dictionary with final test performance for each model
        """
        logger.info("Evaluating final models on test set...")
        
        final_results = {}
        
        for model_name, model in self.models.items():
            if model is None:
                continue
                
            # Test set evaluation
            test_predictions = model.test(self.testset)
            test_rmse = accuracy.rmse(test_predictions, verbose=False)
            test_mae = accuracy.mae(test_predictions, verbose=False)
            
            # Validation set evaluation (for comparison)
            val_predictions = model.test(self.validset)
            val_rmse = accuracy.rmse(val_predictions, verbose=False)
            val_mae = accuracy.mae(val_predictions, verbose=False)
            
            # Training set evaluation (for overfitting analysis)
            # Build training test data from trainset
            train_test_data = [(uid, iid, rating) for (uid, iid, rating) in self.trainset.all_ratings()]
            train_predictions = model.test(train_test_data)
            train_rmse = accuracy.rmse(train_predictions, verbose=False)
            train_mae = accuracy.mae(train_predictions, verbose=False)
            
            final_results[model_name] = {
                'train_rmse': train_rmse,
                'train_mae': train_mae,
                'val_rmse': val_rmse,
                'val_mae': val_mae,
                'test_rmse': test_rmse,
                'test_mae': test_mae,
                'val_to_train_ratio': val_rmse / train_rmse,
                'test_to_train_ratio': test_rmse / train_rmse,
                'generalization_gap': test_rmse - train_rmse
            }
            
            logger.info(f"{model_name} Final Results:")
            logger.info(f"  Train RMSE: {train_rmse:.4f}")
            logger.info(f"  Val RMSE: {val_rmse:.4f}")
            logger.info(f"  Test RMSE: {test_rmse:.4f}")
            logger.info(f"  Generalization Gap: {test_rmse - train_rmse:.4f}")
            
        return final_results
    
    def plot_hyperparameter_search_results(self, save_path: Optional[str] = None):
        """
        Plot the results of hyperparameter search to visualize overfitting.
        
        Args:
            save_path: Optional path to save the plots
        """
        if not self.validation_history:
            logger.warning("No validation history found. Run hyperparameter search first.")
            return
        
        n_models = len(self.validation_history)
        fig, axes = plt.subplots(2, n_models, figsize=(5*n_models, 10))
        
        if n_models == 1:
            axes = axes.reshape(-1, 1)
        
        for i, (model_name, history) in enumerate(self.validation_history.items()):
            if not history:
                continue
                
            iterations = [h['iteration'] for h in history]
            train_rmse = [h['train_rmse'] for h in history]
            val_rmse = [h['val_rmse'] for h in history]
            overfitting_scores = [h['overfitting_score'] for h in history]
            
            # Plot 1: Training vs Validation RMSE
            axes[0, i].plot(iterations, train_rmse, label='Train RMSE', alpha=0.7)
            axes[0, i].plot(iterations, val_rmse, label='Validation RMSE', alpha=0.7)
            axes[0, i].set_title(f'{model_name} - Training vs Validation')
            axes[0, i].set_xlabel('Iteration')
            axes[0, i].set_ylabel('RMSE')
            axes[0, i].legend()
            axes[0, i].grid(True, alpha=0.3)
            
            # Plot 2: Overfitting Score
            axes[1, i].plot(iterations, overfitting_scores, color='red', alpha=0.7)
            axes[1, i].axhline(y=0, color='black', linestyle='--', alpha=0.5)
            axes[1, i].set_title(f'{model_name} - Overfitting Score')
            axes[1, i].set_xlabel('Iteration')
            axes[1, i].set_ylabel('Overfitting Score (Val - Train RMSE)')
            axes[1, i].grid(True, alpha=0.3)
            
            # Highlight the best iteration
            best_idx = np.argmin(val_rmse)
            axes[0, i].scatter(iterations[best_idx], val_rmse[best_idx], 
                             color='red', s=100, zorder=5, label='Best')
            axes[1, i].scatter(iterations[best_idx], overfitting_scores[best_idx], 
                             color='red', s=100, zorder=5)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Hyperparameter search plots saved to {save_path}")
        else:
            plt.show()
    
    def generate_overfitting_report(self) -> pd.DataFrame:
        """
        Generate a comprehensive overfitting analysis report.
        
        Returns:
            DataFrame with overfitting metrics for each model
        """
        if not self.validation_history:
            logger.warning("No validation history found. Run hyperparameter search first.")
            return pd.DataFrame()
        
        report_data = []
        
        for model_name, history in self.validation_history.items():
            if not history:
                continue
                
            # Get best iteration metrics
            val_rmse_list = [h['val_rmse'] for h in history]
            best_idx = np.argmin(val_rmse_list)
            best_result = history[best_idx]
            
            # Calculate overfitting statistics
            overfitting_scores = [h['overfitting_score'] for h in history]
            
            report_data.append({
                'Model': model_name,
                'Best_Iteration': best_result['iteration'],
                'Total_Iterations': len(history),
                'Best_Train_RMSE': best_result['train_rmse'],
                'Best_Val_RMSE': best_result['val_rmse'],
                'Best_Overfitting_Score': best_result['overfitting_score'],
                'Mean_Overfitting_Score': np.mean(overfitting_scores),
                'Std_Overfitting_Score': np.std(overfitting_scores),
                'Max_Overfitting_Score': np.max(overfitting_scores),
                'Min_Overfitting_Score': np.min(overfitting_scores),
                'Overfitting_Severity': self._categorize_overfitting(best_result['overfitting_score'])
            })
        
        return pd.DataFrame(report_data)
    
    def _categorize_overfitting(self, overfitting_score: float) -> str:
        """Categorize the severity of overfitting based on the overfitting score."""
        if overfitting_score < 0.01:
            return 'None/Underfitting'
        elif overfitting_score < 0.05:
            return 'Low'
        elif overfitting_score < 0.1:
            return 'Moderate'
        elif overfitting_score < 0.2:
            return 'High'
        else:
            return 'Severe'
    
    def _save_hyperparameter_results(self, results: Dict[str, Any]):
        """Save hyperparameter search results to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = "hyperparameter_search_results"
        os.makedirs(results_dir, exist_ok=True)
        
        # Save detailed results
        results_file = os.path.join(results_dir, f"search_results_{timestamp}.json")
        
        # Convert results to JSON-serializable format
        json_results = {}
        for model_name, model_results in results.items():
            json_results[model_name] = {
                'best_params': model_results['best_params'],
                'best_val_rmse': model_results['best_val_rmse'],
                'total_iterations': model_results['total_iterations'],
                'search_history': model_results['search_history']
            }
        
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2, default=str)
        
        logger.info(f"Hyperparameter search results saved to {results_file}")
        
        # Save overfitting report
        overfitting_report = self.generate_overfitting_report()
        if not overfitting_report.empty:
            report_file = os.path.join(results_dir, f"overfitting_report_{timestamp}.csv")
            overfitting_report.to_csv(report_file, index=False)
            logger.info(f"Overfitting report saved to {report_file}")
        
        # Save plots
        plot_file = os.path.join(results_dir, f"hyperparameter_plots_{timestamp}.png")
        self.plot_hyperparameter_search_results(save_path=plot_file)
    
    def save_models(self, models_dir: str = "models"):
        """
        Save the best trained models to disk.
        
        Args:
            models_dir: Directory to save models in
        """
        os.makedirs(models_dir, exist_ok=True)
        
        if not self.best_models:
            logger.warning("No trained models found. Run hyperparameter search first.")
            return
        
        for model_name, model in self.best_models.items():
            model_path = os.path.join(models_dir, f"{model_name.lower()}_best.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            logger.info(f"Saved {model_name} model to {model_path}")

        # Save best hyperparameters as well
        params_file = os.path.join(models_dir, "best_hyperparameters.json")
        if hasattr(self, 'best_params') and self.best_params:
            with open(params_file, 'w') as f:
                json.dump(self.best_params, f, indent=2, default=str)
            logger.info(f"Saved best hyperparameters to {params_file}")


class ModelTrainer:
    """Original model trainer class for backward compatibility."""
    
    def __init__(self, data, test_size: float = 0.2, random_state: int = 42):
        """
        Initialize the model trainer.
        
        Args:
            data: Surprise Dataset object
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
        """
        self.data = data
        self.test_size = test_size
        self.random_state = random_state
        self.trainset, self.testset = train_test_split(data, test_size=test_size, random_state=random_state)
        self.models = {}
        
    def train_knn_models(self, tune_hyperparams: bool = False) -> Dict[str, Any]:
        """
        Train KNN-based models (Basic, WithMeans, Baseline).
        
        Args:
            tune_hyperparams: Whether to perform hyperparameter tuning
            
        Returns:
            Dictionary with trained models and their performance metrics
        """
        logger.info("Training KNN models...")
        
        knn_models = {
            'KNNBasic': KNNBasic,
            'KNNWithMeans': KNNWithMeans, 
            'KNNBaseline': KNNBaseline
        }
        
        results = {}
        
        for name, model_class in knn_models.items():
            logger.info(f"Training {name}...")
            
            if tune_hyperparams:
                # Define parameter grid
                param_grid = {
                    'k': [20, 30, 40, 50],
                    'sim_options': {
                        'name': ['cosine', 'pearson'],
                        'user_based': [False, True]  # False = item-based, True = user-based
                    }
                }
                
                # Perform grid search
                gs = GridSearchCV(model_class, param_grid, measures=['rmse', 'mae'], cv=3, n_jobs=-1)
                gs.fit(self.data)
                
                # Best model
                model = gs.best_estimator['rmse']
                best_params = gs.best_params['rmse']
                
                logger.info(f"Best params for {name}: {best_params}")
                
            else:
                # Use default parameters
                model = model_class(
                    k=40,
                    sim_options={'name': 'cosine', 'user_based': False},
                    verbose=False
                )
                
            # Train on full trainset
            model.fit(self.trainset)
            
            # Evaluate
            predictions = model.test(self.testset)
            rmse = accuracy.rmse(predictions, verbose=False)
            mae = accuracy.mae(predictions, verbose=False)
            
            results[name] = {
                'model': model,
                'rmse': rmse,
                'mae': mae,
                'params': model.get_params() if hasattr(model, 'get_params') else None
            }
            
            self.models[name] = model
            logger.info(f"{name} - RMSE: {rmse:.4f}, MAE: {mae:.4f}")
            
        return results
    
    def train_svd_models(self, tune_hyperparams: bool = False) -> Dict[str, Any]:
        """
        Train SVD-based models (SVD, SVD++).
        
        Args:
            tune_hyperparams: Whether to perform hyperparameter tuning
            
        Returns:
            Dictionary with trained models and their performance metrics
        """
        logger.info("Training SVD models...")
        
        svd_models = {
            'SVD': SVD,
            'SVDpp': SVDpp
        }
        
        results = {}
        
        for name, model_class in svd_models.items():
            logger.info(f"Training {name}...")
            
            if tune_hyperparams:
                # Define parameter grid
                param_grid = {
                    'n_factors': [50, 100, 150],
                    'lr_all': [0.002, 0.005, 0.01],
                    'reg_all': [0.01, 0.02, 0.05]
                }
                
                # Perform grid search
                gs = GridSearchCV(model_class, param_grid, measures=['rmse', 'mae'], cv=3, n_jobs=-1)
                gs.fit(self.data)
                
                # Best model
                model = gs.best_estimator['rmse']
                best_params = gs.best_params['rmse']
                
                logger.info(f"Best params for {name}: {best_params}")
                
            else:
                # Use default parameters
                if name == 'SVDpp':
                    model = model_class(
                        n_factors=50,
                        lr_all=0.005,
                        reg_all=0.02,
                        verbose=False,
                        n_jobs=-1
                    )
                else:
                    model = model_class(
                        n_factors=50,
                        lr_all=0.005,
                        reg_all=0.02,
                        verbose=False
                    )
            
            # Train on full trainset
            model.fit(self.trainset)
            
            # Evaluate
            predictions = model.test(self.testset)
            rmse = accuracy.rmse(predictions, verbose=False)
            mae = accuracy.mae(predictions, verbose=False)
            
            results[name] = {
                'model': model,
                'rmse': rmse,
                'mae': mae,
                'params': model.get_params() if hasattr(model, 'get_params') else None
            }
            
            self.models[name] = model
            logger.info(f"{name} - RMSE: {rmse:.4f}, MAE: {mae:.4f}")
            
        return results
    
    def cross_validate_models(self, cv: int = 5) -> Dict[str, Dict]:
        """
        Perform cross-validation on trained models.
        
        Args:
            cv: Number of cross-validation folds
            
        Returns:
            Dictionary with cross-validation results for each model
        """
        logger.info(f"Performing {cv}-fold cross-validation...")
        
        cv_results = {}
        
        for name, model in self.models.items():
            logger.info(f"Cross-validating {name}...")
            
            # Perform cross-validation
            cv_scores = cross_validate(
                model, 
                self.data, 
                measures=['RMSE', 'MAE'], 
                cv=cv, 
                verbose=False,
                n_jobs=-1
            )
            
            cv_results[name] = {
                'rmse_mean': np.mean(cv_scores['test_rmse']),
                'rmse_std': np.std(cv_scores['test_rmse']),
                'mae_mean': np.mean(cv_scores['test_mae']),
                'mae_std': np.std(cv_scores['test_mae']),
                'fit_time_mean': np.mean(cv_scores['fit_time']),
                'test_time_mean': np.mean(cv_scores['test_time'])
            }
            
            logger.info(f"{name} CV - RMSE: {cv_results[name]['rmse_mean']:.4f} (±{cv_results[name]['rmse_std']:.4f})")
            
        return cv_results
    
    def save_models(self, models_dir: str = "models"):
        """
        Save trained models to disk.
        
        Args:
            models_dir: Directory to save models in
        """
        os.makedirs(models_dir, exist_ok=True)
        
        for name, model in self.models.items():
            model_path = os.path.join(models_dir, f"{name.lower()}.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            logger.info(f"Saved {name} model to {model_path}")
    
    def load_model(self, model_path: str):
        """
        Load a trained model from disk.
        
        Args:
            model_path: Path to the saved model file
            
        Returns:
            Loaded model object
        """
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Loaded model from {model_path}")
        return model


def compare_models(results: Dict[str, Dict]) -> pd.DataFrame:
    """
    Compare model performance and return a summary DataFrame.
    
    Args:
        results: Dictionary with model results from training
        
    Returns:
        DataFrame with model comparison
    """
    comparison_data = []
    
    for model_name, metrics in results.items():
        comparison_data.append({
            'Model': model_name,
            'RMSE': metrics['rmse'],
            'MAE': metrics['mae']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('RMSE')
    
    return comparison_df
