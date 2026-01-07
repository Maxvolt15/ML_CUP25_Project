#!/usr/bin/env python3
"""
Quick baseline comparison: Run hall of fame single model on 5-fold CV
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import time

from src.data_loader import load_cup_data
from src.neural_network_v2 import NeuralNetworkV2
from src.utils import mee

# Hall of fame configuration
BEST_CONFIG = {
    'hidden_layers': [128, 84, 65],
    'learning_rate': 0.011984920164083283,
    'l2_lambda': 0.0006003417127125739,
    'dropout_rate': 0.13106988250978824,
    'momentum': 0.9838483610904715,
    'batch_size': 64,
    'hidden_activation': 'tanh',
    'optimizer': 'adam',
    'use_batch_norm': False,
    'weight_init': 'he',
}

N_SPLITS = 5
RANDOM_STATE = 42
MAX_EPOCHS = 2500
PATIENCE = 100

def run_baseline():
    print("=" * 70)
    print("HALL OF FAME BASELINE: Single Model on 5-Fold CV")
    print("=" * 70)
    
    # Load data with polynomial features
    X, y = load_cup_data('data/ML-CUP25-TR.csv', use_polynomial_features=True, poly_degree=2)
    print(f"Data loaded: X shape {X.shape}, y shape {y.shape}")
    
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    fold_mees = []
    
    print(f"\nRunning 5-Fold CV with Hall of Fame config...")
    print("-" * 70)
    
    start_time = time.time()
    
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Scale
        x_scaler = StandardScaler().fit(X_train)
        y_scaler = StandardScaler().fit(y_train)
        
        X_train_scaled = x_scaler.transform(X_train).T
        X_val_scaled = x_scaler.transform(X_val).T
        y_train_scaled = y_scaler.transform(y_train).T
        y_val_scaled = y_scaler.transform(y_val).T
        
        # Create and train model
        nn = NeuralNetworkV2(
            layer_sizes=[X_train.shape[1]] + BEST_CONFIG['hidden_layers'] + [y_train.shape[1]],
            hidden_activation=BEST_CONFIG['hidden_activation'],
            weight_init=BEST_CONFIG['weight_init'],
            use_batch_norm=BEST_CONFIG['use_batch_norm'],
            dropout_rate=BEST_CONFIG['dropout_rate'],
            random_state=RANDOM_STATE + fold_idx
        )
        
        nn.train(
            X_train_scaled, y_train_scaled,
            X_val_scaled, y_val_scaled,
            epochs=MAX_EPOCHS,
            batch_size=BEST_CONFIG['batch_size'],
            learning_rate=BEST_CONFIG['learning_rate'],
            l2_lambda=BEST_CONFIG['l2_lambda'],
            momentum=BEST_CONFIG['momentum'],
            optimizer=BEST_CONFIG['optimizer'],
            patience=PATIENCE,
            verbose=False
        )
        
        # Evaluate
        y_pred_scaled = nn.predict(X_val_scaled).T
        y_pred = y_scaler.inverse_transform(y_pred_scaled)
        
        fold_mee = mee(y_val, y_pred)
        fold_mees.append(fold_mee)
        
        print(f"Fold {fold_idx+1}/5: MEE = {fold_mee:.4f}")
    
    elapsed = time.time() - start_time
    
    mean_mee = np.mean(fold_mees)
    std_mee = np.std(fold_mees)
    
    print("-" * 70)
    print(f"\nHALL OF FAME RESULTS:")
    print(f"Mean MEE: {mean_mee:.4f} (± {std_mee:.4f})")
    print(f"Per-fold: {[f'{m:.4f}' for m in fold_mees]}")
    print(f"Total Time: {elapsed:.1f}s")
    print("=" * 70)
    
    return mean_mee, std_mee

if __name__ == "__main__":
    run_baseline()
