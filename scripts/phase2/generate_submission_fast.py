#!/usr/bin/env python3
"""
STEP 4: Quick Test Set Prediction
Uses simpler, faster training for test predictions.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import csv
from sklearn.preprocessing import StandardScaler
import time

from src.data_loader import load_cup_data, load_cup_test_data, add_polynomial_features
from src.neural_network_v2 import NeuralNetworkV2
from src.utils import mee

# Configuration
HALL_OF_FAME_PATH = 'experiments/hall_of_fame.csv'
TRAIN_DATA_PATH = 'data/ML-CUP25-TR.csv'
TEST_DATA_PATH = 'data/ML-CUP25-TS.csv'
OUTPUT_FILE = 'experiments/ML-CUP25-TS-predictions.csv'

TOP_N_MODELS = 5  # Use only 5 models for speed (still better than single)
RANDOM_STATE = 42

def load_top_configs(n=5):
    """Load top N configurations from hall of fame."""
    import ast
    
    df = pd.read_csv(HALL_OF_FAME_PATH)
    configs = []
    
    print(f"Loading top {n} models from Hall of Fame...")
    for i in range(min(n, len(df))):
        param_str = df.iloc[i]['params']
        clean_str = param_str.replace("np.float64(", "").replace(")", "")
        
        try:
            config = ast.literal_eval(clean_str)
            configs.append(config)
        except Exception as e:
            print(f"Error parsing config {i+1}: {e}")
    
    return configs

def train_and_predict_model(X_train_scaled, y_train_scaled, X_test_scaled, y_scaler, config, model_idx, random_state):
    """Train a single model and predict on test set."""
    # Create and train model with minimal epochs
    nn = NeuralNetworkV2(
        layer_sizes=[X_train_scaled.shape[0]] + config['hidden_layers'] + [y_train_scaled.shape[0]],
        hidden_activation=config['hidden_activation'],
        weight_init=config['weight_init'],
        use_batch_norm=config['use_batch_norm'],
        dropout_rate=config['dropout_rate'],
        random_state=random_state + model_idx
    )
    
    # Train with fewer epochs for speed
    nn.train(
        X_train_scaled, y_train_scaled,
        epochs=800,  # Reduced for speed
        batch_size=config['batch_size'],
        learning_rate=config['learning_rate'],
        l2_lambda=config['l2_lambda'],
        momentum=config['momentum'],
        optimizer=config['optimizer'],
        patience=0,
        verbose=False
    )
    
    # Predict and inverse scale
    y_pred_scaled = nn.predict(X_test_scaled).T
    y_pred = y_scaler.inverse_transform(y_pred_scaled)
    
    return y_pred

def main():
    print("=" * 70)
    print("ML-CUP 2025: QUICK TEST PREDICTION")
    print("=" * 70)
    
    # Load configurations
    configs = load_top_configs(TOP_N_MODELS)
    if not configs:
        print("ERROR: Could not load any configurations!")
        return
    
    # Load training data
    print(f"\nLoading training data from {TRAIN_DATA_PATH}...")
    X_train, y_train = load_cup_data(
        TRAIN_DATA_PATH,
        use_polynomial_features=True,
        poly_degree=2
    )
    print(f"✓ Loaded {X_train.shape[0]} training samples, {X_train.shape[1]} features")
    
    # Scale training data
    x_scaler = StandardScaler().fit(X_train)
    y_scaler = StandardScaler().fit(y_train)
    X_train_scaled = x_scaler.transform(X_train).T
    y_train_scaled = y_scaler.transform(y_train).T
    
    # Load test data
    print(f"\nLoading test data from {TEST_DATA_PATH}...")
    X_test = load_cup_test_data(TEST_DATA_PATH)
    print(f"✓ Loaded {X_test.shape[0]} test samples, {X_test.shape[1]} features")
    
    # Apply polynomial features
    X_test = add_polynomial_features(X_test, degree=2)
    print(f"✓ After polynomial features: {X_test.shape[1]} features")
    
    # Scale test data
    X_test_scaled = x_scaler.transform(X_test).T
    
    # Make ensemble predictions
    print(f"\nTraining {len(configs)} ensemble models and predicting...")
    ensemble_pred = np.zeros((X_test.shape[0], 4))
    
    start_time = time.time()
    
    for model_idx, config in enumerate(configs):
        print(f"  Model {model_idx + 1}/{len(configs)}...", end=" ", flush=True)
        
        y_pred = train_and_predict_model(
            X_train_scaled, y_train_scaled, X_test_scaled, y_scaler, 
            config, model_idx, RANDOM_STATE
        )
        
        ensemble_pred += y_pred
        print("✓")
    
    # Average
    ensemble_pred /= len(configs)
    
    elapsed = time.time() - start_time
    
    # Save predictions
    print(f"\nSaving predictions to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        for i, pred in enumerate(ensemble_pred):
            writer.writerow([i + 1] + pred.tolist())
    
    print(f"✓ Saved {len(ensemble_pred)} predictions")
    
    print("\n" + "=" * 70)
    print(f"SUCCESS: Test predictions generated!")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Models used: {len(configs)}")
    print(f"Predictions: {len(ensemble_pred)} samples")
    print(f"Runtime: {elapsed:.1f}s")
    print("=" * 70)

if __name__ == "__main__":
    main()
