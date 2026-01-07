#!/usr/bin/env python3
"""
Test if iter 25 config (leaky_relu [256,128,64]) works better WITH polynomial features.
This is a critical test: iter 25 achieved 18.04 on raw features, but what if poly2 helps?
"""

import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
import sys
import time

# Add src to path
sys.path.insert(0, r'C:\Users\maxvo\OneDrive\Desktop\ML\ML_CUP25_Project')

from src.neural_network_v2 import NeuralNetworkV2
from src.data_loader import load_cup_data

def mee(y_true, y_pred):
    """Mean Euclidean Error"""
    euclidean_errors = np.sqrt(np.sum((y_true - y_pred) ** 2, axis=1))
    return np.mean(euclidean_errors)

# Load raw data (no polynomial features applied yet)
X, y = load_cup_data(
    r'C:\Users\maxvo\OneDrive\Desktop\ML\ML_CUP25_Project\data\ML-CUP25-TR.csv',
    use_polynomial_features=False,  # Load raw features
    poly_degree=2
)

print(f"Data shape: X={X.shape}, y={y.shape}")
print("=" * 60)

# ITER 25 Configuration (from search_results_v3.csv)
# Architecture: [256, 128, 64], Activation: leaky_relu, LR: 0.00554, L2: 0.0034
# Momentum: 0.911, batch_size: 16

iter25_config = {
    'hidden_layers': [256, 128, 64],
    'hidden_activation': 'leaky_relu',
    'weight_init': 'he',
    'learning_rate': 0.00554,
    'l2_lambda': 0.0034,
    'dropout_rate': 0.0,  # No dropout in iter 25
    'optimizer': 'adam',
    'momentum': 0.911,
    'batch_size': 16,
    'patience': 100,
}

print("TESTING: Iter 25 Config + Polynomial Features Degree 2")
print(f"Config: {iter25_config}")
print("=" * 60)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
mee_scores = []

for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X), 1):
    fold_start = time.time()
    
    # Split data
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    # Apply polynomial features on TRAIN data, then transform validation
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_val_poly = poly.transform(X_val)
    
    print(f"Fold {fold_idx}/5:")
    print(f"  Raw features: {X_train.shape[1]} -> Poly features: {X_train_poly.shape[1]}")
    
    # Normalize
    xs = StandardScaler().fit(X_train_poly)
    ys = StandardScaler().fit(y_train)
    
    X_train_norm = xs.transform(X_train_poly)
    X_val_norm = xs.transform(X_val_poly)
    y_train_norm = ys.transform(y_train)
    y_val_norm = ys.transform(y_val)
    
    # Transpose for NN (features, samples) format
    X_train_nt = X_train_norm.T
    X_val_nt = X_val_norm.T
    y_train_nt = y_train_norm.T
    y_val_nt = y_val_norm.T
    
    # Initialize and train NN
    layer_sizes = [X_train_nt.shape[0]] + iter25_config['hidden_layers'] + [y_train_nt.shape[0]]
    
    nn = NeuralNetworkV2(
        layer_sizes=layer_sizes,
        hidden_activation=iter25_config['hidden_activation'],
        weight_init=iter25_config['weight_init'],
        dropout_rate=iter25_config['dropout_rate']
    )
    
    history = nn.train(
        X_train_nt, y_train_nt,
        X_val_nt, y_val_nt,
        ys,  # Scaler for inverse_transform
        epochs=2500,
        batch_size=iter25_config['batch_size'],
        learning_rate=iter25_config['learning_rate'],
        optimizer=iter25_config['optimizer'],
        momentum=iter25_config['momentum'],
        l2_lambda=iter25_config['l2_lambda'],
        patience=iter25_config['patience'],
        verbose=False
    )
    
    fold_mee = min(history['val_mee'])
    mee_scores.append(fold_mee)
    
    fold_time = time.time() - fold_start
    print(f"  MEE: {fold_mee:.4f} ({fold_time:.1f}s)")

print("=" * 60)
print(f"Fold results: {[f'{m:.4f}' for m in mee_scores]}")
print(f"Mean MEE:    {np.mean(mee_scores):.4f}")
print(f"Std Dev:     {np.std(mee_scores):.4f}")
print(f"Mean ± Std:  {np.mean(mee_scores):.4f} ± {np.std(mee_scores):.4f}")
print("=" * 60)

# Compare to baselines
print("\nComparison to Known Results:")
print(f"  Iter 25 (raw features):        18.04 MEE (on 90/10 split)")
print(f"  Hall of Fame (poly2 features): 22.30 MEE (on 5-fold CV)")
print(f"  Iter 25 + Poly2 (this test):   {np.mean(mee_scores):.4f} MEE (on 5-fold CV)")

if np.mean(mee_scores) < 22.30:
    print(f"  ✅ BETTER than hall of fame!")
elif np.mean(mee_scores) < 25.0:
    print(f"  ⚠️  Similar to hall of fame")
else:
    print(f"  ❌ Worse than hall of fame")
