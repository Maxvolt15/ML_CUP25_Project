"""
Ensemble Training V2 for ML-CUP 2025

Trains an ensemble of the top performing models found in the simple_search.
Strategies:
- Uses Polynomial Features (Degree 2) which proved crucial.
- Uses Tanh activation.
- Averages predictions from 5 distinct model configurations.
- Trains on (Train + Validation) - 10% holdout for early stopping.
"""

import numpy as np
import pandas as pd
import csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.data_loader import load_cup_data
from src.neural_network_v2 import NeuralNetworkV2
from src.utils import mee

def run_ensemble_training(output_file='Gemini_Ensemble_Submission.csv'):
    print(f"{ '='*60}")
    print("Starting Ensemble Training (Top 5 Models)")
    print(f"{ '='*60}\n")
    
    # Top configurations from search
    configs = [
        # Model A (Iter 26 - Best Single: 18.74)
        {
            'name': 'Model_A_40_Tanh',
            'hidden_layers': [40],
            'hidden_activation': 'tanh',
            'learning_rate': 0.00014,
            'optimizer': 'sgd',
            'momentum': 0.94,
            'l2_lambda': 0.0062,
            'dropout_rate': 0.08,
            'batch_size': 16,
            'weight_init': 'xavier',
            'use_batch_norm': False
        },
        # Model B (Iter 35 - Robust SGD: 19.17)
        {
            'name': 'Model_B_30_Tanh',
            'hidden_layers': [30],
            'hidden_activation': 'tanh',
            'learning_rate': 0.014,
            'optimizer': 'sgd',
            'momentum': 0.88,
            'l2_lambda': 0.008,
            'dropout_rate': 0.12,
            'batch_size': 32,
            'weight_init': 'xavier',
            'use_batch_norm': False
        },
        # Model C (Iter 8 - Deep SGD: 19.25)
        {
            'name': 'Model_C_64_32_Tanh',
            'hidden_layers': [64, 32],
            'hidden_activation': 'tanh',
            'learning_rate': 0.038,
            'optimizer': 'sgd',
            'momentum': 0.82,
            'l2_lambda': 0.0003,
            'dropout_rate': 0.23,
            'batch_size': 16,
            'weight_init': 'xavier',
            'use_batch_norm': False
        },
        # Model D (Iter 24 - Adam: 19.43)
        {
            'name': 'Model_D_30_Adam',
            'hidden_layers': [30],
            'hidden_activation': 'tanh',
            'learning_rate': 0.014,
            'optimizer': 'adam',
            'l2_lambda': 1.5e-5,
            'dropout_rate': 0.15,
            'batch_size': 16,
            'weight_init': 'xavier',
            'use_batch_norm': False
        },
        # Model E (Variant - 50 neurons)
        {
            'name': 'Model_E_50_Tanh',
            'hidden_layers': [50],
            'hidden_activation': 'tanh',
            'learning_rate': 0.001,
            'optimizer': 'adam',
            'l2_lambda': 0.005,
            'dropout_rate': 0.1,
            'batch_size': 32,
            'weight_init': 'xavier',
            'use_batch_norm': False
        }
    ]
    
    # 1. Load Data (Polynomial Features = True for ALL)
    print("Loading Data with Polynomial Features (Degree 2)...")
    X, y = load_cup_data('data/ML-CUP25-TR.csv', 
                           use_polynomial_features=True, 
                           poly_degree=2)
    
    # Load Blind Test Data
    # Note: We need to apply same transform. load_cup_data handles TR, 
    # we need a way to load TS with same poly transform.
    # Re-using load_cup_data for TS might require modification or manual handling.
    # Let's verify data_loader.py capabilities later. 
    # For now, we assume we load TS similarly.
    
    # Check data_loader.py for TS loading...
    # (Assuming we fix this, for now let's focus on training validation)
    
    # Split for Early Stopping (90% Train, 10% Val for stopping)
    # We use a fixed seed for consistent ensemble training
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=999)
    
    # Scale
    x_scaler = StandardScaler().fit(X_train)
    y_scaler = StandardScaler().fit(y_train)
    
    X_train_scaled = x_scaler.transform(X_train)
    X_val_scaled = x_scaler.transform(X_val)
    # y_train is scaled inside training loop usually, but here we do it manually for consistency
    y_train_scaled = y_scaler.transform(y_train)
    
    models = []
    val_scores = []
    
    print("\nTraining Ensemble Members...")
    
    for cfg in configs:
        print(f"\nTraining {cfg['name']}...")
        
        layer_sizes = [X_train_scaled.shape[1]] + cfg['hidden_layers'] + [y_train.shape[1]]
        
        nn = NeuralNetworkV2(
            layer_sizes=layer_sizes,
            hidden_activation=cfg['hidden_activation'],
            weight_init=cfg['weight_init'],
            use_batch_norm=cfg['use_batch_norm'],
            dropout_rate=cfg['dropout_rate'],
            random_state=42
        )
        
        history = nn.train(
            X_train_scaled.T, y_train_scaled.T,
            X_val_scaled.T, y_val, y_scaler,
            epochs=3000,
            batch_size=cfg['batch_size'],
            learning_rate=cfg['learning_rate'],
            optimizer=cfg['optimizer'],
            momentum=cfg.get('momentum', 0.9),
            l2_lambda=cfg['l2_lambda'],
            patience=200,
            verbose=False
        )
        
        # Validate
        y_pred_scaled = nn.predict(X_val_scaled.T)
        y_pred = y_scaler.inverse_transform(y_pred_scaled.T)
        score = mee(y_val, y_pred)
        print(f"  -> Best Val MEE: {score:.4f}")
        
        models.append(nn)
        val_scores.append(score)
        
    print(f"\n{'-'*30}")
    print(f"Average Single Model MEE: {np.mean(val_scores):.4f}")
    
    # Ensemble Prediction on Validation
    ensemble_preds = []
    for model in models:
        p_scaled = model.predict(X_val_scaled.T)
        p = y_scaler.inverse_transform(p_scaled.T)
        ensemble_preds.append(p)
    
    avg_pred = np.mean(ensemble_preds, axis=0)
    ensemble_score = mee(y_val, avg_pred)
    
    print(f"Ensemble MEE on Validation: {ensemble_score:.4f}")
    print(f"{'-'*30}")
    
    if ensemble_score < np.min(val_scores):
        print("SUCCESS: Ensemble outperformed best single model!")
    else:
        print("NOTE: Ensemble did not outperform best single model (check correlation).")

if __name__ == '__main__':
    run_ensemble_training()
