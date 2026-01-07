import numpy as np
import pandas as pd
import itertools
import random
import csv
from sklearn.model_selection import KFold
from scipy.stats import uniform, loguniform
from src.data_loader import load_cup_data
from src.neural_network import NeuralNetwork
from src.utils import mee, mse, tanh, sigmoid, linear

def run_random_search(n_iterations=500):
    """
    Performs a randomized search with k-fold cross-validation to find the best hyperparameters.
    """
    X, y = load_cup_data('data/ML-CUP25-TR.csv')
    print(f"Loaded ML-CUP data with shapes: X={X.shape}, y={y.shape}")

    # 1. Define the expanded hyperparameter search space
    param_dist = {
        'learning_rate': loguniform(1e-3, 3e-1),
        'l2_lambda': loguniform(1e-6, 1e-1),
        'momentum': uniform(loc=0.85, scale=0.14),      # Range [0.85, 0.99]
        'hidden_layers': [
            # One Layer
            [20], [30], [40], [50], [60], [80],
            # Two Layers
            [40, 20], [50, 25], [60, 30], [80, 40], [100, 50],
            # Three Layers
            [50, 30, 15], [60, 40, 20], [80, 50, 20]
        ],
        'hidden_activation': [tanh, sigmoid]
    }

    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    best_mee = float('inf')
    best_params = {}
    
    results_file = 'search_results_v2.csv'
    with open(results_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['iteration', 'avg_mee', 'params'])

    print(f"\n--- Starting Randomized Search ({n_iterations} iterations) ---")

    # 2. Loop for n_iterations
    for i in range(n_iterations):
        # 3. Randomly sample a set of parameters
        params = {
            'learning_rate': param_dist['learning_rate'].rvs(),
            'l2_lambda': param_dist['l2_lambda'].rvs(),
            'momentum': param_dist['momentum'].rvs(),
            'hidden_layers': random.choice(param_dist['hidden_layers']),
            'hidden_activation': random.choice(param_dist['hidden_activation'])
        }
        
        # Format for printing
        printable_params = params.copy()
        printable_params['hidden_activation'] = params['hidden_activation'].__name__
        print(f"\n[{i+1}/{n_iterations}] Testing params: {printable_params}")
        
        fold_mees = []
        
        # 4. K-Fold Cross-Validation
        for fold, (train_index, val_index) in enumerate(kf.split(X)):
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]

            from sklearn.preprocessing import StandardScaler
            x_scaler = StandardScaler().fit(X_train)
            y_scaler = StandardScaler().fit(y_train)

            X_train_scaled = x_scaler.transform(X_train)
            y_train_scaled = y_scaler.transform(y_train)
            X_val_scaled = x_scaler.transform(X_val)

            layer_sizes = [X_train.shape[1]] + params['hidden_layers'] + [y_train.shape[1]]
            
            nn = NeuralNetwork(
                layer_sizes=layer_sizes,
                hidden_activation=params['hidden_activation'],
                output_activation=linear,
                loss=mse,
                random_state=42
            )
            
            nn.train(
                X_train_scaled.T, y_train_scaled.T,
                X_val_scaled.T, y_val, y_scaler,
                epochs=2000, 
                learning_rate=params['learning_rate'],
                momentum=params['momentum'],
                l2_lambda=params['l2_lambda'],
                patience=15
            )
            
            y_pred_scaled = nn.predict(X_val_scaled.T)
            y_pred = y_scaler.inverse_transform(y_pred_scaled.T)
            
            fold_mee = mee(y_val, y_pred)
            fold_mees.append(fold_mee)

        avg_mee = np.mean(fold_mees)
        print(f"-> Average MEE for params: {avg_mee:.4f}")
        
        # 5. Log results and update best score
        with open(results_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([i+1, f"{avg_mee:.4f}", printable_params])

        if avg_mee < best_mee:
            best_mee = avg_mee
            best_params = printable_params
            print(f"!!! New best MEE found: {best_mee:.4f} with params {best_params}")

    print("\n--- Randomized Search Complete ---")
    print(f"Best MEE: {best_mee:.4f}")
    print(f"Best Hyperparameters: {best_params}")
    print(f"Full results logged to {results_file}")

if __name__ == '__main__':
    run_random_search()
