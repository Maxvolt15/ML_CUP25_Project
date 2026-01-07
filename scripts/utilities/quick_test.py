"""Quick test to verify the improved neural network works."""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.data_loader import load_cup_data
from src.neural_network_v2 import NeuralNetworkV2
from src.utils import mee

def quick_test():
    # Load data
    X, y = load_cup_data('data/ML-CUP25-TR.csv')
    print(f"Data loaded: X={X.shape}, y={y.shape}")
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize
    x_scaler = StandardScaler().fit(X_train)
    y_scaler = StandardScaler().fit(y_train)
    
    X_train_scaled = x_scaler.transform(X_train)
    y_train_scaled = y_scaler.transform(y_train)
    X_val_scaled = x_scaler.transform(X_val)
    
    # Test different configurations
    configs = [
        {'layers': [128, 64], 'activation': 'relu', 'lr': 0.001},
        {'layers': [256, 128, 64], 'activation': 'relu', 'lr': 0.001},
        {'layers': [128, 64, 32], 'activation': 'elu', 'lr': 0.002},
        {'layers': [64, 32], 'activation': 'tanh', 'lr': 0.01},
    ]
    
    print("\n" + "="*60)
    print("Quick Test - Training Different Architectures")
    print("="*60)
    
    for config in configs:
        layer_sizes = [X_train.shape[1]] + config['layers'] + [y_train.shape[1]]
        
        nn = NeuralNetworkV2(
            layer_sizes=layer_sizes,
            hidden_activation=config['activation'],
            output_activation='linear',
            weight_init='he' if config['activation'] in ['relu', 'elu', 'leaky_relu'] else 'xavier',
            random_state=42
        )
        
        print(f"\nTesting: {config['layers']} with {config['activation']}")
        
        history = nn.train(
            X_train_scaled.T, y_train_scaled.T,
            X_val_scaled.T, y_val, y_scaler,
            epochs=1000,
            batch_size=32,
            learning_rate=config['lr'],
            optimizer='adam',
            l2_lambda=1e-5,
            patience=50,
            verbose=False
        )
        
        # Evaluate
        y_pred_scaled = nn.predict(X_val_scaled.T)
        y_pred = y_scaler.inverse_transform(y_pred_scaled.T)
        val_mee = mee(y_val, y_pred)
        
        print(f"  Final Validation MEE: {val_mee:.4f}")
        print(f"  Min Val MEE during training: {min(history['val_mee']):.4f}")

if __name__ == '__main__':
    quick_test()
