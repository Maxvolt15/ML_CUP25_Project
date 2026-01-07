#!/usr/bin/env python3
"""
Phase 3: MONK's Benchmark Suite
Objective: Validate the simulator (NeuralNetworkV2) on standard classification tasks.
Deliverables: Learning curves (MSE & Accuracy) for MONK 1, 2, and 3.
"""

import numpy as np
import pandas as pd
import time
import os
from sklearn.metrics import accuracy_score

from src.data_loader import load_monk_data
from src.neural_network_v2 import NeuralNetworkV2
from src.utils import mse

# ============================================================================
# CONFIGURATION
# ============================================================================

MONK_CONFIGS = {
    'monks-1': {
        'hidden_layers': [4], # Standard small architecture for Monk 1
        'learning_rate': 0.2,
        'momentum': 0.7,
        'epochs': 500,
        'batch_size': 16 # Mini-batch
    },
    'monks-2': {
        'hidden_layers': [4],
        'learning_rate': 0.2,
        'momentum': 0.7,
        'epochs': 500,
        'batch_size': 16
    },
    'monks-3': {
        'hidden_layers': [5], # Monk 3 is noisy, might need slightly more capacity or regularization
        'learning_rate': 0.1,
        'momentum': 0.7,
        'l2_lambda': 0.001, # Regularization for noisy task
        'epochs': 500,
        'batch_size': 16
    }
}

OUTPUT_DIR = 'ML_CUP25_Project/experiments/monk_results'

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def train_and_evaluate(problem_name, config):
    print(f"\nRunning Benchmark: {problem_name}")
    print("-" * 60)
    
    # 1. Load Data
    # load_monk_data returns data with shape (samples, features)
    X_train, y_train, X_test, y_test = load_monk_data(problem_name)
    
    # Transpose for NN (features, samples)
    X_train_T = X_train.T
    y_train_T = y_train.T
    X_test_T = X_test.T
    y_test_T = y_test.T
    
    print(f"Train samples: {X_train.shape[0]}, Features: {X_train.shape[1]}")
    print(f"Test samples:  {X_test.shape[0]}")
    
    # 2. Initialize Network
    # Binary classification: Output activation 'sigmoid'
    nn = NeuralNetworkV2(
        layer_sizes=[X_train.shape[1]] + config['hidden_layers'] + [1],
        hidden_activation='tanh',
        output_activation='sigmoid',
        weight_init='xavier',
        random_state=42
    )
    
    # 3. Custom Training Loop for Logging
    # We need to log Train/Test Accuracy at every epoch, which standard .train() might not do fully
    
    history = {
        'epoch': [],
        'train_mse': [],
        'test_mse': [],
        'train_acc': [],
        'test_acc': []
    }
    
    epochs = config['epochs']
    batch_size = config['batch_size']
    lr = config['learning_rate']
    mom = config['momentum']
    l2 = config.get('l2_lambda', 0.0)
    
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        # Train one epoch (using internal train logic would be cleaner, but we want granular control)
        # Actually, let's use the internal train for 1 epoch at a time to leverage its batching/optimizer
        nn.train(
            X_train_T, y_train_T,
            epochs=1,
            batch_size=batch_size,
            learning_rate=lr,
            momentum=mom,
            l2_lambda=l2,
            optimizer='sgd', # Standard for Monk is usually SGD+Momentum
            verbose=False
        )
        
        # Evaluate
        # Predictions
        pred_train_raw = nn.predict(X_train_T)
        pred_test_raw = nn.predict(X_test_T)
        
        # MSE
        mse_train = mse(y_train_T, pred_train_raw)
        mse_test = mse(y_test_T, pred_test_raw)
        
        # Accuracy (Threshold 0.5)
        pred_train_cls = (pred_train_raw > 0.5).astype(int)
        pred_test_cls = (pred_test_raw > 0.5).astype(int)
        
        acc_train = accuracy_score(y_train, pred_train_cls.T)
        acc_test = accuracy_score(y_test, pred_test_cls.T)
        
        # Log
        history['epoch'].append(epoch)
        history['train_mse'].append(mse_train)
        history['test_mse'].append(mse_test)
        history['train_acc'].append(acc_train)
        history['test_acc'].append(acc_test)
        
        if epoch % 50 == 0 or epoch == epochs:
            print(f"Epoch {epoch:03d} | Train MSE: {mse_train:.4f} Acc: {acc_train:.1%} | Test MSE: {mse_test:.4f} Acc: {acc_test:.1%}")
            
elapsed = time.time() - start_time
    print(f"Completed in {elapsed:.2f}s")
    
    return history

def save_results(problem_name, history):
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    df = pd.DataFrame(history)
    filename = f"{OUTPUT_DIR}/{problem_name}_metrics.csv"
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("STARTING MONK BENCHMARKS")
    
    for problem, config in MONK_CONFIGS.items():
        try:
            history = train_and_evaluate(problem, config)
            save_results(problem, history)
        except Exception as e:
            print(f"Error running {problem}: {e}")
            import traceback
            traceback.print_exc()
            
    print("\nBENCHMARK COMPLETE")
