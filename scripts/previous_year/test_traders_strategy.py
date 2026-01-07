import sys
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import time

# Add the FINAL PROJECT paths so we can import their modules
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), 'FINAL PROJECT/monk')))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), 'ML_CUP25_Project')))

# Import their classes
# Note: We need to handle potential import errors if their code relies on relative imports
try:
    import NN as NN_module # Import the module to patch it
    from NN import NN
    from activation_functions import Tanh, Identity
    from utility import mean_square_error, accuracy
except ImportError as e:
    print(f"Error importing Traders modules: {e}")
    sys.exit(1)

# MONKEY PATCH: Disable accuracy calculation for regression
# The Traders' NN.py unconditionaly calculates accuracy using confusion matrix, which fails for regression
NN_module.accuracy = lambda t, p: 0

# Import our data loader
from src.data_loader import load_cup_data

# --- Override NN class to fix the 3-output hardcoding ---
class AdaptiveNN(NN):
    def output_layer(self, act_function=None, units=1):
        """
        Overridden method to allow flexible output units for CUP.
        The original method hardcoded units=3 if type_d=='cup'.
        """
        # We ignore the type_d check for units and trust the passed value
        # But we keep their logic for activation function default
        if self.type_d == 'cup':
            act_function = Identity
            # We DO NOT force units=3 here. We use the passed 'units'.
        
        self.add_layer(act_function, units)

def mee(y_true, y_pred):
    """Mean Euclidean Error for evaluation"""
    euclidean_errors = np.sqrt(np.sum((y_true - y_pred) ** 2, axis=1))
    return np.mean(euclidean_errors)

def run_test():
    print("="*60)
    print("TESTING TRADERS STRATEGY (ML-CUP24 WINNER) ON ML-CUP25")
    print("Architecture: [30] Tanh")
    print("Optimizer: SGD + Momentum")
    print("Regularization: L2 (lambda=0.006)")
    print("="*60)

    # 1. Load Data
    # Traders used simple Z-score normalization, no poly features mentioned in abstract (but let's check results)
    # The user asked to test "their code", implying their methodology.
    # Their main.py used raw features. So we use raw features (poly_degree=0/1).
    X, y = load_cup_data('ML_CUP25_Project/data/ML-CUP25-TR.csv', use_polynomial_features=False)
    
    print(f"Data Loaded: X={X.shape}, y={y.shape}")
    
    # 2. Setup 5-Fold CV
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    fold_mees = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\nFold {fold+1}/5")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Normalize (like they did in main.py)
        scaler_x = StandardScaler()
        scaler_y = StandardScaler()
        
        X_train_norm = scaler_x.fit_transform(X_train)
        X_val_norm = scaler_x.transform(X_val)
        y_train_norm = scaler_y.fit_transform(y_train)
        # y_val is kept raw for MEE calc, but we need normalized for training
        
        # 3. Initialize Model
        # They used: units=Input, 1 hidden layer (30 units, Tanh), Output (Identity)
        # Their NN constructor takes 'units' as input size
        # momentum=0.6 (from their hyperparams table in slides? or 0.9? Abstract says "Momentum technique")
        # main.py had learning_rate=0.01. Slides say eta=0.1, alpha=0.6.
        # Let's use slide values: eta=0.1, alpha=0.6, lambda=0.006
        
        input_size = X_train.shape[1]
        output_size = y_train.shape[1] # Should be 4
        
        model = AdaptiveNN(units=input_size, momentum=True, type_d='cup', regularization=True)
        model.add_layer(act_function=Tanh, units=30)
        model.output_layer(units=output_size) # This calls our overridden method
        model.output_l = True # PREVENT train() from adding another output layer!
        
        # Transpose data for Traders' NN (expects features x samples)
        X_train_T = X_train_norm.T
        y_train_T = y_train_norm.T
        X_val_T = X_val_norm.T
        
        # 4. Train
        # train(epochs, error_rate, eta, X, y, alpha, lambda1)
        # Slides: epochs=fixed (or early stop), eta=0.1, alpha=0.6, lambda=0.006
        # main.py: epochs=500. 
        # We'll use 1000 epochs with early stopping to be safe.
        
        start_time = time.time()
        # Note: Their train method expects X and y as lists or arrays?
        # Their code: forward_pass(X) -> iterates.
        # They seem to implement Batch GD in NN.py (forward_pass takes whole X).
        
        # Their train signature: train(self, epochs, error_rate, eta, X, y, alpha, lambda1=0)
        errors, epochs_run, _, _ = model.train(
            epochs=2000,
            error_rate=1e-6, # effectively disable unless strictly flat
            eta=0.1,         # Learning rate
            X=X_train_T,     # Transposed
            y=y_train_T,     # Transposed
            alpha=0.6,       # Momentum
            lambda1=0.006    # L2 Regularization
        )
        
        train_time = time.time() - start_time
        
        # 5. Evaluate
        # Predict returns normalized values?
        # In NN.py: predict calls forward_pass. 
        # So it returns whatever the output layer outputs. Since we trained on norm targets, it outputs norm.
        # Predict expects (features, samples) and returns (outputs, samples) or (samples, outputs)?
        # forward_pass returns same shape as y.
        # Let's assume it returns (outputs, samples).
        y_pred_norm_T = np.array(model.predict(X_val_T))
        
        # Transpose back to (samples, outputs) for scaler
        y_pred_norm = y_pred_norm_T.T

        
        # Inverse transform
        y_pred = scaler_y.inverse_transform(y_pred_norm)
        
        # Calculate MEE
        val_mee = mee(y_val, y_pred)
        fold_mees.append(val_mee)
        
        print(f"  MEE: {val_mee:.4f} (Time: {train_time:.2f}s, Epochs: {epochs_run})")
        
    print("\n" + "="*60)
    print(f"TRADERS STRATEGY RESULTS (5-Fold CV)")
    print(f"Mean MEE: {np.mean(fold_mees):.4f}")
    print(f"Std Dev:  {np.std(fold_mees):.4f}")
    print("="*60)

if __name__ == "__main__":
    run_test()
