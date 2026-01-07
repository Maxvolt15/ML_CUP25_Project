"""
Advanced Hyperparameter Search for ML-CUP 2025
Target: MEE < 2.5 (Silver Medal level)

Uses the improved NeuralNetworkV2 with:
- Mini-batch Adam optimizer
- Learning rate scheduling
- Proper initialization
- Gradient clipping
"""

import numpy as np
import pandas as pd
import csv
import random
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from scipy.stats import uniform, loguniform
from src.data_loader import load_cup_data
from src.neural_network_v2 import NeuralNetworkV2


def run_advanced_search(n_iterations=200, n_splits=5, results_file='search_results_v3.csv'):
    """
    Advanced randomized hyperparameter search with k-fold CV.
    """
    X, y = load_cup_data('data/ML-CUP25-TR.csv')
    print(f"Loaded ML-CUP data: X={X.shape}, y={y.shape}")
    print(f"Target: MEE < 2.5 (Silver Medal: 2.430)")
    
    # Expanded hyperparameter search space
    param_dist = {
        # Learning rate - wider range for Adam
        'learning_rate': loguniform(1e-4, 1e-1),
        
        # L2 regularization
        'l2_lambda': loguniform(1e-7, 1e-2),
        
        # Batch size options
        'batch_size': [16, 32, 64, 128, 256],
        
        # Network architectures - much deeper and wider
        'hidden_layers': [
            # Single layer (wide)
            [64], [128], [256], [512],
            # Two layers
            [128, 64], [256, 128], [512, 256], [256, 64],
            [128, 32], [256, 64], [512, 128],
            # Three layers
            [256, 128, 64], [512, 256, 128], [256, 128, 32],
            [128, 64, 32], [512, 256, 64], [256, 64, 32],
            # Four layers
            [512, 256, 128, 64], [256, 128, 64, 32],
            [128, 128, 64, 32], [256, 256, 128, 64],
        ],
        
        # Activation functions
        'hidden_activation': ['relu', 'leaky_relu', 'elu', 'tanh'],
        
        # Weight initialization
        'weight_init': ['he', 'xavier'],
        
        # Optimizer
        'optimizer': ['adam', 'sgd'],
        
        # Momentum for SGD
        'momentum': uniform(loc=0.85, scale=0.14),  # [0.85, 0.99]
    }
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    best_mee = float('inf')
    best_params = {}
    all_results = []
    
    # Write CSV header
    with open(results_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['iteration', 'avg_mee', 'std_mee', 'params'])
    
    print(f"\n{'='*60}")
    print(f"Starting Advanced Random Search ({n_iterations} iterations)")
    print(f"{'='*60}\n")
    
    for i in range(n_iterations):
        # Sample random parameters
        params = {
            'learning_rate': float(param_dist['learning_rate'].rvs()),
            'l2_lambda': float(param_dist['l2_lambda'].rvs()),
            'batch_size': random.choice(param_dist['batch_size']),
            'hidden_layers': random.choice(param_dist['hidden_layers']),
            'hidden_activation': random.choice(param_dist['hidden_activation']),
            'weight_init': random.choice(param_dist['weight_init']),
            'optimizer': random.choice(param_dist['optimizer']),
            'momentum': float(param_dist['momentum'].rvs()),
        }
        
        # Match initialization to activation
        if params['hidden_activation'] in ['relu', 'leaky_relu', 'elu']:
            params['weight_init'] = 'he'
        else:
            params['weight_init'] = 'xavier'
        
        print(f"\n[{i+1}/{n_iterations}] Testing:")
        print(f"  Layers: {params['hidden_layers']}, Act: {params['hidden_activation']}")
        print(f"  LR: {params['learning_rate']:.6f}, L2: {params['l2_lambda']:.2e}")
        print(f"  Batch: {params['batch_size']}, Opt: {params['optimizer']}")
        
        fold_mees = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Standardize data
            x_scaler = StandardScaler().fit(X_train)
            y_scaler = StandardScaler().fit(y_train)
            
            X_train_scaled = x_scaler.transform(X_train)
            y_train_scaled = y_scaler.transform(y_train)
            X_val_scaled = x_scaler.transform(X_val)
            
            # Build network
            layer_sizes = [X_train.shape[1]] + params['hidden_layers'] + [y_train.shape[1]]
            
            nn = NeuralNetworkV2(
                layer_sizes=layer_sizes,
                hidden_activation=params['hidden_activation'],
                output_activation='linear',
                weight_init=params['weight_init'],
                random_state=42 + fold
            )
            
            # Train with mini-batch
            nn.train(
                X_train_scaled.T, y_train_scaled.T,
                X_val_scaled.T, y_val, y_scaler,
                epochs=3000,
                batch_size=params['batch_size'],
                learning_rate=params['learning_rate'],
                optimizer=params['optimizer'],
                momentum=params['momentum'],
                l2_lambda=params['l2_lambda'],
                patience=100,
                verbose=False
            )
            
            # Evaluate
            y_pred_scaled = nn.predict(X_val_scaled.T)
            y_pred = y_scaler.inverse_transform(y_pred_scaled.T)
            
            from src.utils import mee
            fold_mee = mee(y_val, y_pred)
            fold_mees.append(fold_mee)
        
        avg_mee = np.mean(fold_mees)
        std_mee = np.std(fold_mees)
        
        print(f"  -> MEE: {avg_mee:.4f} (+/- {std_mee:.4f})")
        
        # Save result
        all_results.append({
            'iteration': i + 1,
            'avg_mee': avg_mee,
            'std_mee': std_mee,
            'params': params
        })
        
        with open(results_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([i + 1, f"{avg_mee:.4f}", f"{std_mee:.4f}", str(params)])
        
        if avg_mee < best_mee:
            best_mee = avg_mee
            best_params = params.copy()
            print(f"  *** NEW BEST! MEE: {best_mee:.4f} ***")
    
    print(f"\n{'='*60}")
    print(f"Search Complete!")
    print(f"{'='*60}")
    print(f"\nBest MEE: {best_mee:.4f}")
    print(f"Best Parameters:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")
    
    return best_params, best_mee, all_results


def fine_tune_around_best(best_params, n_iterations=50, n_splits=5):
    """
    Fine-tune hyperparameters around the best found configuration.
    """
    X, y = load_cup_data('data/ML-CUP25-TR.csv')
    
    print(f"\n{'='*60}")
    print(f"Fine-tuning around best parameters")
    print(f"{'='*60}")
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    best_mee = float('inf')
    best_fine_params = best_params.copy()
    
    base_lr = best_params['learning_rate']
    base_l2 = best_params['l2_lambda']
    
    for i in range(n_iterations):
        # Small perturbations around best params
        params = best_params.copy()
        params['learning_rate'] = base_lr * np.random.uniform(0.5, 2.0)
        params['l2_lambda'] = base_l2 * np.random.uniform(0.1, 10.0)
        params['batch_size'] = random.choice([16, 32, 64, params['batch_size']])
        
        fold_mees = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            x_scaler = StandardScaler().fit(X_train)
            y_scaler = StandardScaler().fit(y_train)
            
            X_train_scaled = x_scaler.transform(X_train)
            y_train_scaled = y_scaler.transform(y_train)
            X_val_scaled = x_scaler.transform(X_val)
            
            layer_sizes = [X_train.shape[1]] + params['hidden_layers'] + [y_train.shape[1]]
            
            nn = NeuralNetworkV2(
                layer_sizes=layer_sizes,
                hidden_activation=params['hidden_activation'],
                output_activation='linear',
                weight_init=params['weight_init'],
                random_state=42 + fold
            )
            
            nn.train(
                X_train_scaled.T, y_train_scaled.T,
                X_val_scaled.T, y_val, y_scaler,
                epochs=5000,
                batch_size=params['batch_size'],
                learning_rate=params['learning_rate'],
                optimizer=params['optimizer'],
                momentum=params['momentum'],
                l2_lambda=params['l2_lambda'],
                patience=150,
                verbose=False
            )
            
            y_pred_scaled = nn.predict(X_val_scaled.T)
            y_pred = y_scaler.inverse_transform(y_pred_scaled.T)
            
            from src.utils import mee
            fold_mees.append(mee(y_val, y_pred))
        
        avg_mee = np.mean(fold_mees)
        
        if avg_mee < best_mee:
            best_mee = avg_mee
            best_fine_params = params.copy()
            print(f"[{i+1}/{n_iterations}] New best: {best_mee:.4f}")
    
    return best_fine_params, best_mee


if __name__ == '__main__':
    # Run the main search
    best_params, best_mee, results = run_advanced_search(n_iterations=100)
    
    # Fine-tune if we found a good result
    if best_mee < 15.0:
        print("\nStarting fine-tuning phase...")
        best_params, best_mee = fine_tune_around_best(best_params, n_iterations=30)
    
    print(f"\n\nFinal Best MEE: {best_mee:.4f}")
    print(f"Final Best Parameters: {best_params}")
