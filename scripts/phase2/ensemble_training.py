"""
Ensemble Training for ML-CUP 2025
Train multiple models and average predictions for better generalization.

Key techniques:
1. Multiple random seeds for initialization diversity
2. Different architectures in the ensemble
3. K-fold cross-validation for each model
4. Prediction averaging/weighted averaging
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from src.data_loader import load_cup_data, load_cup_test_data
from src.neural_network_v2 import NeuralNetworkV2
from src.utils import mee


def train_single_model(X_train, y_train, X_val, y_val, params, seed=42):
    """Train a single model with given parameters."""
    x_scaler = StandardScaler().fit(X_train)
    y_scaler = StandardScaler().fit(y_train)
    
    X_train_scaled = x_scaler.transform(X_train)
    y_train_scaled = y_scaler.transform(y_train)
    X_val_scaled = x_scaler.transform(X_val)
    
    layer_sizes = [X_train.shape[1]] + params['hidden_layers'] + [y_train.shape[1]]
    
    nn = NeuralNetworkV2(
        layer_sizes=layer_sizes,
        hidden_activation=params['hidden_activation'],
        output_activation='linear',
        weight_init=params['weight_init'],
        random_state=seed
    )
    
    nn.train(
        X_train_scaled.T, y_train_scaled.T,
        X_val_scaled.T, y_val, y_scaler,
        epochs=params.get('epochs', 5000),
        batch_size=params.get('batch_size', 32),
        learning_rate=params['learning_rate'],
        optimizer=params.get('optimizer', 'adam'),
        momentum=params.get('momentum', 0.9),
        l2_lambda=params['l2_lambda'],
        patience=params.get('patience', 150),
        verbose=False
    )
    
    return nn, x_scaler, y_scaler


def train_ensemble(configs, n_models_per_config=5, n_splits=5):
    """
    Train an ensemble of models.
    
    Args:
        configs: List of hyperparameter configurations
        n_models_per_config: Number of models to train per config (different seeds)
        n_splits: Number of CV folds for evaluation
    
    Returns:
        List of trained models with their scalers
    """
    X, y = load_cup_data('data/ML-CUP25-TR.csv')
    print(f"Training ensemble on data: X={X.shape}, y={y.shape}")
    
    ensemble = []
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    for config_idx, params in enumerate(configs):
        print(f"\n{'='*50}")
        print(f"Config {config_idx + 1}/{len(configs)}: {params['hidden_layers']}")
        print(f"{'='*50}")
        
        for model_idx in range(n_models_per_config):
            seed = 42 + config_idx * 100 + model_idx
            
            # Use CV to get validation MEE estimate
            fold_mees = []
            
            for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                nn, x_scaler, y_scaler = train_single_model(
                    X_train, y_train, X_val, y_val, params, seed + fold
                )
                
                y_pred_scaled = nn.predict(x_scaler.transform(X_val).T)
                y_pred = y_scaler.inverse_transform(y_pred_scaled.T)
                fold_mees.append(mee(y_val, y_pred))
            
            avg_mee = np.mean(fold_mees)
            print(f"  Model {model_idx + 1}: CV MEE = {avg_mee:.4f}")
            
            # Train final model on full data
            x_scaler_full = StandardScaler().fit(X)
            y_scaler_full = StandardScaler().fit(y)
            
            X_scaled = x_scaler_full.transform(X)
            y_scaled = y_scaler_full.transform(y)
            
            layer_sizes = [X.shape[1]] + params['hidden_layers'] + [y.shape[1]]
            
            nn_full = NeuralNetworkV2(
                layer_sizes=layer_sizes,
                hidden_activation=params['hidden_activation'],
                output_activation='linear',
                weight_init=params['weight_init'],
                random_state=seed
            )
            
            # Train without validation (full data)
            nn_full.train(
                X_scaled.T, y_scaled.T,
                epochs=params.get('epochs', 5000),
                batch_size=params.get('batch_size', 32),
                learning_rate=params['learning_rate'],
                optimizer=params.get('optimizer', 'adam'),
                momentum=params.get('momentum', 0.9),
                l2_lambda=params['l2_lambda'],
                patience=params.get('patience', 150),
                verbose=False
            )
            
            ensemble.append({
                'model': nn_full,
                'x_scaler': x_scaler_full,
                'y_scaler': y_scaler_full,
                'params': params,
                'cv_mee': avg_mee
            })
    
    return ensemble


def predict_ensemble(ensemble, X_test, method='mean'):
    """
    Make predictions using the ensemble.
    
    Args:
        ensemble: List of model dictionaries
        X_test: Test features (n_samples, n_features)
        method: 'mean', 'median', or 'weighted'
    
    Returns:
        Averaged predictions
    """
    all_predictions = []
    weights = []
    
    for item in ensemble:
        nn = item['model']
        x_scaler = item['x_scaler']
        y_scaler = item['y_scaler']
        
        X_scaled = x_scaler.transform(X_test)
        y_pred_scaled = nn.predict(X_scaled.T)
        y_pred = y_scaler.inverse_transform(y_pred_scaled.T)
        
        all_predictions.append(y_pred)
        # Weight inversely proportional to CV MEE
        weights.append(1.0 / (item['cv_mee'] + 0.1))
    
    all_predictions = np.array(all_predictions)  # (n_models, n_samples, n_outputs)
    
    if method == 'mean':
        return np.mean(all_predictions, axis=0)
    elif method == 'median':
        return np.median(all_predictions, axis=0)
    elif method == 'weighted':
        weights = np.array(weights) / np.sum(weights)
        return np.average(all_predictions, axis=0, weights=weights)


def evaluate_ensemble(ensemble, X, y, method='mean'):
    """Evaluate ensemble on validation data."""
    y_pred = predict_ensemble(ensemble, X, method)
    return mee(y, y_pred)


def generate_submission(ensemble, output_file='Gemini-Agent_ML-CUP25-TS.csv'):
    """Generate submission file for the blind test set."""
    X_test = load_cup_test_data('data/ML-CUP25-TS.csv')
    print(f"Loaded test data: {X_test.shape}")
    
    # Try different averaging methods
    pred_mean = predict_ensemble(ensemble, X_test, 'mean')
    pred_weighted = predict_ensemble(ensemble, X_test, 'weighted')
    
    # Use weighted average as final prediction
    predictions = pred_weighted
    
    # Create submission file
    header = """# Suranjan Kumar Ghosh, Max Verstappen
# Gemini-Agent
# ML-CUP25
# Submission Date: December 2025
"""
    
    with open(output_file, 'w') as f:
        f.write(header)
        for i, pred in enumerate(predictions, 1):
            line = f"{i},{pred[0]},{pred[1]},{pred[2]},{pred[3]}\n"
            f.write(line)
    
    print(f"Submission saved to {output_file}")
    return predictions


# Best configurations found from hyperparameter search
# Update these after running hyperparameter_search_v2.py
BEST_CONFIGS = [
    {
        'hidden_layers': [256, 128, 64],
        'hidden_activation': 'relu',
        'weight_init': 'he',
        'learning_rate': 0.001,
        'l2_lambda': 1e-5,
        'batch_size': 32,
        'optimizer': 'adam',
        'momentum': 0.9,
        'epochs': 5000,
        'patience': 150
    },
    {
        'hidden_layers': [512, 256, 128],
        'hidden_activation': 'relu',
        'weight_init': 'he',
        'learning_rate': 0.0005,
        'l2_lambda': 1e-5,
        'batch_size': 64,
        'optimizer': 'adam',
        'momentum': 0.9,
        'epochs': 5000,
        'patience': 150
    },
    {
        'hidden_layers': [256, 128, 64, 32],
        'hidden_activation': 'elu',
        'weight_init': 'he',
        'learning_rate': 0.001,
        'l2_lambda': 1e-4,
        'batch_size': 32,
        'optimizer': 'adam',
        'momentum': 0.9,
        'epochs': 5000,
        'patience': 150
    },
    {
        'hidden_layers': [128, 64, 32],
        'hidden_activation': 'leaky_relu',
        'weight_init': 'he',
        'learning_rate': 0.002,
        'l2_lambda': 1e-5,
        'batch_size': 32,
        'optimizer': 'adam',
        'momentum': 0.9,
        'epochs': 5000,
        'patience': 150
    },
]


if __name__ == '__main__':
    print("="*60)
    print("ML-CUP 2025 - Ensemble Training")
    print("Target: MEE < 2.5 (Silver Medal: 2.430)")
    print("="*60)
    
    # Train ensemble
    ensemble = train_ensemble(BEST_CONFIGS, n_models_per_config=3, n_splits=5)
    
    # Evaluate on training data (as sanity check)
    X, y = load_cup_data('data/ML-CUP25-TR.csv')
    
    print("\n" + "="*60)
    print("Ensemble Evaluation (on training data - for reference)")
    print("="*60)
    
    for method in ['mean', 'median', 'weighted']:
        ensemble_mee = evaluate_ensemble(ensemble, X, y, method)
        print(f"Ensemble ({method}): MEE = {ensemble_mee:.4f}")
    
    # Print individual model performance
    print("\nIndividual model CV MEEs:")
    for i, item in enumerate(ensemble):
        print(f"  Model {i+1}: {item['cv_mee']:.4f} ({item['params']['hidden_layers']})")
    
    # Generate submission
    print("\n" + "="*60)
    print("Generating Submission")
    print("="*60)
    generate_submission(ensemble)
