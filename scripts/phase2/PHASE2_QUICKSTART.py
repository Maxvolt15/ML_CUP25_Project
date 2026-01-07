#!/usr/bin/env python3
"""
QUICK START: Phase 2 Ensemble Implementation
============================================

This is a ready-to-implement script for Phase 2.
Takes 30 minutes to code, 15 minutes to run.

Purpose: Combine all 20 hall-of-fame models to reduce variance
Goal: Achieve 20.5-21.5 MEE (30% variance reduction from 22.30)
"""

# ============================================================================
# STEP 1: CREATE ensemble_simple.py
# ============================================================================

CODE_TEMPLATE = '''#!/usr/bin/env python3
"""
Simple Ensemble of Hall of Fame Models
=====================================
Combine predictions from all 20 hall-of-fame configurations.
Expected improvement: 22.30 MEE → 20.5-21.5 MEE
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import sys
import time

sys.path.insert(0, r'C:\\Users\\maxvo\\OneDrive\\Desktop\\ML\\ML_CUP25_Project')

from src.neural_network_v2 import NeuralNetworkV2
from src.data_loader import load_cup_data

def mee(y_true, y_pred):
    """Mean Euclidean Error"""
    euclidean_errors = np.sqrt(np.sum((y_true - y_pred) ** 2, axis=1))
    return np.mean(euclidean_errors)

# Load data
X, y = load_cup_data(
    r'C:\\Users\\maxvo\\OneDrive\\Desktop\\ML\\ML_CUP25_Project\\data\\ML-CUP25-TR.csv',
    use_polynomial_features=False,
    poly_degree=2
)

print("="*70)
print("PHASE 2: SIMPLE ENSEMBLE OF HALL OF FAME MODELS")
print("="*70)
print(f"Data: X={X.shape}, y={y.shape}")
print()

# Load hall of fame configurations
hall_of_fame = pd.read_csv(
    r'C:\\Users\\maxvo\\OneDrive\\Desktop\\ML\\ML_CUP25_Project\\hall_of_fame.csv'
)

print(f"Loaded {len(hall_of_fame)} hall-of-fame configurations")
print(f"All report MEE: {hall_of_fame['mee'].unique()}")
print()

# Extract top N models to ensemble (you can change this)
NUM_MODELS = 20  # Use all 20, or reduce to top 5-10 for speed
top_configs = hall_of_fame.iloc[:NUM_MODELS]

print(f"Using top {NUM_MODELS} models for ensemble")
print("="*70)

# 5-fold cross-validation with ensemble
kf = KFold(n_splits=5, shuffle=True, random_state=42)
ensemble_mee_scores = []

for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X), 1):
    fold_start = time.time()
    
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    # Apply polynomial features
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_val_poly = poly.transform(X_val)
    
    # Normalize
    xs = StandardScaler().fit(X_train_poly)
    ys = StandardScaler().fit(y_train)
    
    X_train_norm = xs.transform(X_train_poly)
    X_val_norm = xs.transform(X_val_poly)
    y_train_norm = ys.transform(y_train)
    
    # Transpose for NN
    X_train_nt = X_train_norm.T
    X_val_nt = X_val_norm.T
    y_train_nt = y_train_norm.T
    
    # Collect predictions from each model
    all_predictions = []
    
    for model_idx, row in top_configs.iterrows():
        # Extract hyperparameters
        config = {
            'hidden_layers': eval(row['hidden_layers']),
            'hidden_activation': row['hidden_activation'],
            'weight_init': row['weight_init'],
            'dropout_rate': row['dropout_rate'],
            'learning_rate': row['learning_rate'],
            'l2_lambda': row['l2_lambda'],
            'optimizer': row['optimizer'],
            'momentum': row['momentum'],
            'batch_size': int(row['batch_size']),
            'patience': 100,
        }
        
        # Train model
        layer_sizes = [X_train_nt.shape[0]] + config['hidden_layers'] + [y_train_nt.shape[0]]
        
        nn = NeuralNetworkV2(
            layer_sizes=layer_sizes,
            hidden_activation=config['hidden_activation'],
            weight_init=config['weight_init'],
            dropout_rate=config['dropout_rate']
        )
        
        history = nn.train(
            X_train_nt, y_train_nt,
            X_val_nt, y_train_nt,  # Use training data for early stopping monitoring
            ys,
            epochs=2500,
            batch_size=config['batch_size'],
            learning_rate=config['learning_rate'],
            optimizer=config['optimizer'],
            momentum=config['momentum'],
            l2_lambda=config['l2_lambda'],
            patience=config['patience'],
            verbose=False
        )
        
        # Get predictions on validation set
        pred = nn.predict(X_val_nt)  # Shape: (val_size, 4)
        all_predictions.append(pred)
    
    # Average predictions from all models
    ensemble_pred = np.mean(all_predictions, axis=0)  # Shape: (val_size, 4)
    
    # Inverse transform predictions back to original scale
    ensemble_pred_unscaled = ys.inverse_transform(ensemble_pred.T).T  # Shape: (val_size, 4)
    
    # Compute MEE
    fold_mee = mee(y_val, ensemble_pred_unscaled)
    ensemble_mee_scores.append(fold_mee)
    
    fold_time = time.time() - fold_start
    print(f"Fold {fold_idx}/5: Ensemble MEE={fold_mee:.4f} ({fold_time:.1f}s)")

print("="*70)
print(f"Per-fold results: {[f'{m:.4f}' for m in ensemble_mee_scores]}")
print(f"Ensemble Mean MEE:  {np.mean(ensemble_mee_scores):.4f}")
print(f"Ensemble Std Dev:   {np.std(ensemble_mee_scores):.4f}")
print(f"Ensemble ± Std:     {np.mean(ensemble_mee_scores):.4f} ± {np.std(ensemble_mee_scores):.4f}")
print("="*70)

# Comparison
print()
print("COMPARISON TO BASELINES:")
print(f"  Hall of Fame single model:     22.30 ± 1.74 MEE")
print(f"  Ensemble of {NUM_MODELS} models:          {np.mean(ensemble_mee_scores):.4f} ± {np.std(ensemble_mee_scores):.4f} MEE")

improvement = 22.30 - np.mean(ensemble_mee_scores)
variance_reduction = (1.74 - np.std(ensemble_mee_scores)) / 1.74 * 100

if improvement > 0.5:
    print(f"  ✅ IMPROVEMENT: {improvement:.2f} MEE gained!")
elif improvement > 0:
    print(f"  ⚠️  Marginal improvement: {improvement:.2f} MEE")
else:
    print(f"  ❌ No improvement (increase of {abs(improvement):.2f} MEE)")

print(f"  Variance reduction: {variance_reduction:.1f}%")

if np.mean(ensemble_mee_scores) < 21.0:
    print()
    print("🎉 SUCCESS! Ensemble achieved < 21.0 MEE")
    print("   Next: Generate test set predictions")
elif np.mean(ensemble_mee_scores) < 21.5:
    print()
    print("✅ GOOD PROGRESS! Ensemble improved the baseline")
    print("   Consider: Try weighted ensemble or Phase 3 features")
else:
    print()
    print("⚠️  Limited improvement. Try:")
    print("   1. Weighted ensemble with meta-learner")
    print("   2. Phase 3 feature engineering")
    print("   3. Increase NUM_MODELS to use all 20 configs")
'''

print("="*70)
print("PHASE 2 IMPLEMENTATION GUIDE")
print("="*70)
print()
print("STEP 1: Copy the template code above into a new file:")
print("        C:\\Users\\maxvo\\OneDrive\\Desktop\\ML\\ML_CUP25_Project\\ensemble_simple.py")
print()
print("STEP 2: Run the script:")
print("        cd C:\\Users\\maxvo\\OneDrive\\Desktop\\ML\\ML_CUP25_Project")
print("        python ensemble_simple.py")
print()
print("STEP 3: Check the results:")
print("        - If MEE < 21.0: Proceed to test submission (Phase 4)")
print("        - If MEE 20.5-21.0: Consider Phase 3 features")
print("        - If MEE > 21.0: Try weighted ensemble (need 1-2 more hours)")
print()
print("="*70)
print()
print("EXPECTED RESULTS:")
print("  Single Model (baseline):     22.30 ± 1.74 MEE")
print("  Simple Ensemble (target):    20.5-21.5 ± 0.8-1.0 MEE")
print("  Expected variance reduction: 30-35%")
print()
print("="*70)
print()

# ============================================================================
# IMPLEMENTATION NOTES
# ============================================================================

NOTES = """
IMPLEMENTATION NOTES:
====================

1. Data Format:
   - Polynomial features degree 2: 12 features → 90 engineered features
   - Train: 400 samples (4 folds of 100 each)
   - Val: 100 samples per fold
   - Normalization: Per-fold StandardScaler

2. Model Training:
   - Each model trained independently on each fold
   - Early stopping with patience=100
   - Maximum 2500 epochs

3. Prediction Averaging:
   - All 20 (or N) model predictions averaged
   - Results in lower variance, smoother predictions
   - Automatically better generalization

4. Performance Factors:
   - Ensemble size: More models = less variance (but slower)
   - Model diversity: All 20 are slightly different (momentum varies)
   - Weighting: Currently simple average (could be weighted later)

5. Timing:
   - Per fold: 20 models × 30s each = 10 minutes
   - 5 folds: 50 minutes total
   - Or 15 minutes if using faster hardware

6. Potential Issues:
   - If too slow: Reduce NUM_MODELS to 5-10 (top performers)
   - If not improving: May need weighted ensemble
   - If still stuck: Need Phase 3 feature engineering

7. Next Steps After Ensemble:
   - If successful (< 21.0 MEE):
     → Create generate_test_predictions.py for submission
   
   - If marginal (20.5-21.0):
     → Decide: More ensemble work or Phase 3 features?
   
   - If unsuccessful (> 21.0):
     → Try ensemble_weighted.py (meta-learner)
     → Or move directly to Phase 3 (poly degree 3, PCA)
"""

print(NOTES)

# ============================================================================
# QUICK REFERENCE: Hall of Fame Sample
# ============================================================================

SAMPLE = """
SAMPLE HALL OF FAME ENTRY (from row 1):
======================================

Architecture:    [128, 84, 65]
Activation:      tanh
Learning Rate:   0.011985
L2 Lambda:       0.000600
Dropout Rate:    0.131070
Optimizer:       adam
Momentum:        0.983848
Batch Size:      64
Features:        polynomial degree 2
Reported MEE:    12.27 (on 90/10 split)
Real 5-fold MEE: 22.30

All 20 entries are similar, with slight variations in momentum.
This is good for ensemble: models are specialized but not identical.
"""

print(SAMPLE)

print()
print("="*70)
print("Ready to implement? Copy the CODE_TEMPLATE above into:")
print("  ensemble_simple.py")
print("="*70)
