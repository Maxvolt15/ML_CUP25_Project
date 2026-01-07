#!/usr/bin/env python3
"""
Test the PREVIOUS YEAR'S STRATEGY on current ML-CUP25 dataset.
Previous year (ML-CUP24) used: 1 hidden layer, 30 neurons, tanh activation.
This test adds polynomial features degree 2 to see if simple + features works.
"""

import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
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

# PREVIOUS YEAR STRATEGY (ML-CUP24)
# Architecture: [30] (1 hidden layer, 30 neurons)
# Activation: tanh
# Standard hyperparameters

prev_year_config = {
    'hidden_layers': [30],
    'hidden_activation': 'tanh',
    'weight_init': 'xavier',  # Standard for tanh
    'learning_rate': 0.01,
    'l2_lambda': 0.001,
    'dropout_rate': 0.0,
    'optimizer': 'adam',
    'momentum': 0.9,
    'batch_size': 32,
    'patience': 100,
}

print("TESTING: Previous Year Strategy (30 neurons tanh) + Polynomial Features Degree 2")
print(f"Config: {prev_year_config}")
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
    layer_sizes = [X_train_nt.shape[0]] + prev_year_config['hidden_layers'] + [y_train_nt.shape[0]]
    
    nn = NeuralNetworkV2(
        layer_sizes=layer_sizes,
        hidden_activation=prev_year_config['hidden_activation'],
        weight_init=prev_year_config['weight_init'],
        dropout_rate=prev_year_config['dropout_rate']
    )
    
    history = nn.train(
        X_train_nt, y_train_nt,
        X_val_nt, y_val_nt,
        ys,  # Scaler for inverse_transform
        epochs=2500,
        batch_size=prev_year_config['batch_size'],
        learning_rate=prev_year_config['learning_rate'],
        optimizer=prev_year_config['optimizer'],
        momentum=prev_year_config['momentum'],
        l2_lambda=prev_year_config['l2_lambda'],
        patience=prev_year_config['patience'],
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
print(f"  Hall of Fame (277 neurons):    22.30 MEE (on 5-fold CV)")
print(f"  Prev Year (30 neurons):        {np.mean(mee_scores):.4f} MEE (on 5-fold CV)")

if np.mean(mee_scores) < 22.30:
    print(f"  ✅ BETTER than hall of fame! Simpler is better!")
    print(f"  Improvement: {22.30 - np.mean(mee_scores):.2f} MEE")
elif np.mean(mee_scores) < 25.0:
    print(f"  ⚠️  Similar to hall of fame")
else:
    print(f"  ❌ Worse than hall of fame")

print("\nConclusion:")
print(f"  Previous year's simple approach (30 neurons + tanh + poly2)")
print(f"  achieved {np.mean(mee_scores):.2f} MEE.")
if np.mean(mee_scores) < 22.30:
    print(f"  This is better than the deep architecture!")
    print(f"  Recommendation: Use previous year's strategy as baseline.")
