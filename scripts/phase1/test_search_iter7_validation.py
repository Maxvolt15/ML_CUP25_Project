#!/usr/bin/env python3
'''Validate search_results_v3 iter 7 on proper 5-fold CV'''

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import time

from src.data_loader import load_cup_data
from src.neural_network_v2 import NeuralNetworkV2
from src.utils import mee

# Load configuration from search_results_v3.csv
search_results = pd.read_csv('experiments/search_results_v3.csv')
iter7_row = search_results[search_results['iteration'] == 7].iloc[0]

# Parse config (it's a string representation of dict)
import ast
config_str = iter7_row['params']
# Clean up numpy types
config_str = config_str.replace("np.float64(", "").replace(")", "")
config = ast.literal_eval(config_str)

print("="*70)
print("VALIDATING SEARCH RESULTS V3 ITERATION 7")
print("="*70)
print(f"Original (90/10): MEE = {iter7_row['avg_mee']:.4f} ± {iter7_row['std_mee']:.4f}")
print(f"Config: {config}")
print()

# Load data
X, y = load_cup_data('data/ML-CUP25-TR.csv', use_polynomial_features=True, poly_degree=2)
print(f"Data loaded: X shape {X.shape}, y shape {y.shape}")

# 5-fold CV
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_mees = []

print("Running 5-Fold Cross-Validation...")
for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X), 1):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    xs = StandardScaler().fit(X_train)
    ys = StandardScaler().fit(y_train)
    
    X_train_scaled = xs.transform(X_train).T
    X_val_scaled = xs.transform(X_val).T
    y_train_scaled = ys.transform(y_train).T
    y_val_scaled = ys.transform(y_val).T
    
    # Create and train network
    layer_sizes = [X_train_scaled.shape[0]] + config['hidden_layers'] + [4]
    
    nn = NeuralNetworkV2(
        layer_sizes=layer_sizes,
        hidden_activation=config['hidden_activation'],
        weight_init=config['weight_init'],
        dropout_rate=config.get('dropout_rate', 0.0)
    )
    
    history = nn.train(
        X_train_scaled, y_train_scaled,
        X_val_scaled, y_val_scaled,
        ys,
        epochs=2500,
        batch_size=config['batch_size'],
        learning_rate=config['learning_rate'],
        optimizer=config['optimizer'],
        momentum=config['momentum'],
        l2_lambda=config['l2_lambda'],
        patience=100,
        verbose=False
    )
    
    # Get validation MEE
    # Predict returns (outputs, samples)
    y_pred = nn.predict(X_val_scaled)
    # Transpose to (samples, outputs) for inverse_transform
    y_pred_unscaled = ys.inverse_transform(y_pred.T)
    # y_val is (samples, outputs)
    fold_mee = mee(y_val, y_pred_unscaled)
    fold_mees.append(fold_mee)
    
    print(f"  Fold {fold_idx}/5: MEE = {fold_mee:.4f}")

print()
print("="*70)
print(f"5-Fold CV Result: {np.mean(fold_mees):.4f} ± {np.std(fold_mees):.4f}")
print(f"Per-fold: {[f'{m:.4f}' for m in fold_mees]}")
print("="*70)
print()

# Comparison
baseline = 22.30
if np.mean(fold_mees) < baseline:
    improvement = baseline - np.mean(fold_mees)
    print(f"✅ VALIDATED: Iter 7 is BETTER than baseline!")
    print(f"   Improvement: {improvement:.2f} MEE over 22.30 baseline")
    print(f"   → Use this as NEW baseline for Phase 2")
else:
    print(f"⚠️  NOT better than baseline (22.30 MEE)")
    print(f"    Difference: {np.mean(fold_mees) - baseline:.2f} MEE")