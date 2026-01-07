import numpy as np
import itertools
from sklearn.model_selection import KFold
from src.data_loader import load_cup_data
from src.neural_network import NeuralNetwork
from src.utils import mee, mse, tanh, sigmoid, linear

def grid_search_cv():
    """
    Performs an extensive grid search with k-fold cross-validation.
    """
    X, y = load_cup_data('data/ML-CUP25-TR.csv')
    print(f"Loaded ML-CUP data with shapes: X={X.shape}, y={y.shape}")

    param_grid = {
        'learning_rate': [0.01, 0.05],
        'l2_lambda': [0.0001, 0.00001],
        'momentum': [0.9, 0.95],
        'hidden_layers': [[30], [40], [20, 10], [30, 15]],
        'hidden_activation': [tanh, sigmoid]
    }

    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    best_mee = float('inf')
    best_params = {}

    print("\n--- Starting Extensive Grid Search with K-Fold CV and Early Stopping ---")

    # Create all combinations of hyperparameters
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    for i, params in enumerate(param_combinations):
        print(f"\n[{i+1}/{len(param_combinations)}] Testing params: {params}")
        
        fold_mees = []
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
                X_val_scaled.T, y_val,
                y_scaler,
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
        
        if avg_mee < best_mee:
            best_mee = avg_mee
            best_params = params
            print(f"!!! New best MEE found: {best_mee:.4f} with params {best_params}")

    print("\n--- Grid Search Complete ---")
    print(f"Best MEE: {best_mee:.4f}")
    print(f"Best Hyperparameters: {best_params}")

if __name__ == '__main__':
    grid_search_cv()
