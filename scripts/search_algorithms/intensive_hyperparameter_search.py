"""
Intensive and Continuous Hyperparameter Search for ML-CUP 2025

This script implements an indefinite randomized search to find optimal
hyperparameters for NeuralNetworkV2. It explores a wide range of options,
including feature engineering (polynomial, PCA) and advanced regularization
(Batch Norm, Dropout).
"""

import numpy as np
import pandas as pd
import csv
import random
import itertools
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from scipy.stats import uniform, loguniform
from src.data_loader import load_cup_data
from src.neural_network_v2 import NeuralNetworkV2
from src.utils import mee


def run_continuous_search(results_file='intensive_search_results.csv', n_splits=5):
    """
    Runs a continuous, randomized hyperparameter search with k-fold CV.
    """
    print(f"{'='*60}")
    print("Starting Continuous Hyperparameter Search")
    print("Press Ctrl+C to stop the search.")
    print(f"{'='*60}\n")
    
    # Expanded hyperparameter search space including new features
    param_dist = {
        'learning_rate': loguniform(1e-4, 1e-1),
        'l2_lambda': loguniform(1e-7, 1e-2),
        'batch_size': [16, 32, 64],
        'hidden_layers': [
            [64], [128], [256], [128, 64], [256, 128], [512, 256],
            [256, 128, 64], [512, 256, 128], [128, 64, 32]
        ],
        'hidden_activation': ['relu', 'leaky_relu', 'tanh'],
        'optimizer': ['adam', 'sgd'],
        'momentum': uniform(loc=0.85, scale=0.14),
        
        # New regularization and feature options
        'use_batch_norm': [True, False],
        'dropout_rate': uniform(0.0, 0.5),
        'use_polynomial_features': [True, False],
        'poly_degree': [2], # Degree 2 is a good start
        'use_pca_features': [False], # Disabled for now to reduce complexity
        'pca_components': [5]
    }
    
    # File setup
    try:
        with open(results_file, 'x', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['iteration', 'avg_mee', 'std_mee', 'params'])
    except FileExistsError:
        pass # File already exists, append to it

    iteration = 0
    while True:
        iteration += 1
        
        # --- 1. Sample Hyperparameters ---
        params = {key: dist.rvs() if hasattr(dist, 'rvs') else random.choice(dist) 
                  for key, dist in param_dist.items()}

        # Conditional logic
        if not params['use_polynomial_features']:
            params['poly_degree'] = 0
        if not params['use_pca_features']:
            params['pca_components'] = 0
        if params['hidden_activation'] in ['relu', 'leaky_relu', 'elu']:
            params['weight_init'] = 'he'
        else:
            params['weight_init'] = 'xavier'

        print(f"\n[{iteration}] Testing:")
        print(f"  Features: Poly={params['poly_degree']}, PCA={params['pca_components']}")
        print(f"  Regularization: BN={params['use_batch_norm']}, Dropout={params['dropout_rate']:.2f}")
        print(f"  Architecture: {params['hidden_layers']}, Act: {params['hidden_activation']}")
        print(f"  Training: LR={params['learning_rate']:.6f}, L2={params['l2_lambda']:.2e}, Batch={params['batch_size']}, Opt={params['optimizer']}")

        # --- 2. Load and Prepare Data For This Iteration ---
        X, y = load_cup_data('data/ML-CUP25-TR.csv', 
                               use_polynomial_features=params['use_polynomial_features'],
                               poly_degree=params['poly_degree'],
                               use_pca_features=params['use_pca_features'],
                               pca_components=params['pca_components'])
        
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=iteration) # Change seed each time
        fold_mees = []
        
        # --- 3. Run Cross-Validation ---
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            x_scaler = StandardScaler().fit(X_train)
            y_scaler = StandardScaler().fit(y_train)
            
            X_train_scaled = x_scaler.transform(X_train)
            y_train_scaled = y_scaler.transform(y_train)
            X_val_scaled = x_scaler.transform(X_val)
            
            layer_sizes = [X_train_scaled.shape[1]] + params['hidden_layers'] + [y_train.shape[1]]
            
            nn = NeuralNetworkV2(
                layer_sizes=layer_sizes,
                hidden_activation=params['hidden_activation'],
                weight_init=params['weight_init'],
                use_batch_norm=params['use_batch_norm'],
                dropout_rate=params['dropout_rate'],
                random_state=42 + fold
            )
            
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
            
            y_pred_scaled = nn.predict(X_val_scaled.T)
            y_pred = y_scaler.inverse_transform(y_pred_scaled.T)
            fold_mees.append(mee(y_val, y_pred))
        
        avg_mee = np.mean(fold_mees)
        std_mee = np.std(fold_mees)
        
        print(f"  -> MEE: {avg_mee:.4f} (+/- {std_mee:.4f})")
        
        # --- 4. Save Result ---
        with open(results_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([iteration, f"{avg_mee:.4f}", f"{std_mee:.4f}", str(params)])
        
        # Check for new best
        # This is just for printing, the file is the source of truth
        try:
            results_df = pd.read_csv(results_file)
            if avg_mee < results_df['avg_mee'].min():
                print(f"  *** NEW BEST OVERALL! MEE: {avg_mee:.4f} ***")
        except (FileNotFoundError, pd.errors.EmptyDataError):
            pass


if __name__ == '__main__':
    try:
        run_continuous_search()
    except KeyboardInterrupt:
        print("\n\nSearch stopped by user. Exiting.")
