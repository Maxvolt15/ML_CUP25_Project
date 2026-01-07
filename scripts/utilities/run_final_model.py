"""Reproduce the best-found configuration (iteration 25) with flexible orchestration.

Features:
  * Single holdout split (default) or shuffled K-fold cross-validation (--cv)
  * Multiple seeds per run (--seeds 0 1 2)
  * Quick mode to reduce epochs/patience for sanity checks (--quick)
  * Automatic logging of every run to experiments/results_run_final.csv
  * Model parameters saved under experiments/ for each (seed, fold) pair
"""

from __future__ import annotations

import argparse
import os
import time
from typing import List, Optional

import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler

from src.data_loader import load_cup_data
from src.exp_utils import append_result, save_model_params
from src.neural_network_v2 import NeuralNetworkV2
from src.utils import mee


BEST_CONFIG = {
    'learning_rate': 0.005541145648003442,
    'l2_lambda': 0.003404989000472383,
    'batch_size': 16,
    'hidden_layers': [256, 128, 64],
    'hidden_activation': 'leaky_relu',
    'weight_init': 'he',
    'optimizer': 'adam',
    'momentum': 0.9114767773338317,
}


def _train_single_split(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    seed: int,
    quick: bool,
    mode: str,
    fold: Optional[int],
) -> dict:
    """Scale data, train NeuralNetworkV2, log metrics, and persist model params."""

    x_scaler = StandardScaler().fit(X_train)
    y_scaler = StandardScaler().fit(y_train)

    X_train_scaled = x_scaler.transform(X_train)
    y_train_scaled = y_scaler.transform(y_train)
    X_val_scaled = x_scaler.transform(X_val)

    layer_sizes = [X_train.shape[1]] + BEST_CONFIG['hidden_layers'] + [y_train.shape[1]]
    nn = NeuralNetworkV2(
        layer_sizes=layer_sizes,
        hidden_activation=BEST_CONFIG['hidden_activation'],
        output_activation='linear',
        weight_init=BEST_CONFIG['weight_init'],
        random_state=seed,
    )

    epochs = 3000
    patience = 100
    if quick:
        epochs = 200
        patience = 20

    t0 = time.time()
    nn.train(
        X_train_scaled.T,
        y_train_scaled.T,
        X_val_scaled.T,
        y_val,
        y_scaler,
        epochs=epochs,
        batch_size=BEST_CONFIG['batch_size'],
        learning_rate=BEST_CONFIG['learning_rate'],
        optimizer=BEST_CONFIG['optimizer'],
        momentum=BEST_CONFIG['momentum'],
        l2_lambda=BEST_CONFIG['l2_lambda'],
        patience=patience,
        verbose=False,
    )
    train_time = time.time() - t0

    y_pred_scaled = nn.predict(X_val_scaled.T)
    y_pred = y_scaler.inverse_transform(y_pred_scaled.T)
    val_mee = mee(y_val, y_pred)

    os.makedirs('experiments', exist_ok=True)
    suffix = f"seed{seed}"
    if fold is not None:
        suffix += f"_fold{fold}"
    model_path = os.path.join('experiments', f'model_iter25_{suffix}.npz')
    save_model_params(nn.get_params(), model_path)

    result = {
        'time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'seed': seed,
        'fold': fold,
        'mode': mode,
        'quick': bool(quick),
        'val_mee': float(val_mee),
        'train_time_s': float(train_time),
        'train_samples': int(len(X_train)),
        'val_samples': int(len(X_val)),
        'params': BEST_CONFIG,
    }

    append_result(os.path.join('experiments', 'results_run_final.csv'), result)
    fold_msg = f", fold={fold}" if fold is not None else ''
    print(
        f"mode={mode}{fold_msg} seed={seed} quick={quick} val_mee={val_mee:.4f} "
        f"time={train_time:.1f}s",
    )
    return result


def run_holdout(X: np.ndarray, y: np.ndarray, seeds: List[int], quick: bool) -> List[dict]:
    results = []
    for seed in seeds:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=seed
        )
        results.append(
            _train_single_split(
                X_train,
                y_train,
                X_val,
                y_val,
                seed=seed,
                quick=quick,
                mode='holdout',
                fold=None,
            )
        )
    return results


def run_cv(
    X: np.ndarray,
    y: np.ndarray,
    seeds: List[int],
    n_splits: int,
    quick: bool,
) -> List[dict]:
    results = []
    for seed in seeds:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]
            results.append(
                _train_single_split(
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    seed=seed,
                    quick=quick,
                    mode='cv',
                    fold=fold_idx,
                )
            )
    return results


def summarize(results: List[dict]) -> None:
    if not results:
        return
    val_mees = [r['val_mee'] for r in results]
    mean_mee = float(np.mean(val_mees))
    std_mee = float(np.std(val_mees))
    print('-' * 60)
    print(f"Completed {len(results)} runs | mean MEE = {mean_mee:.4f} ± {std_mee:.4f}")
    per_mode = {}
    for r in results:
        per_mode.setdefault(r['mode'], []).append(r['val_mee'])
    for mode, values in per_mode.items():
        print(f"  {mode}: n={len(values)} mean={np.mean(values):.4f} std={np.std(values):.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quick', action='store_true', help='Run a short verification run')
    parser.add_argument('--cv', action='store_true', help='Enable shuffled K-fold cross-validation')
    parser.add_argument('--n-splits', type=int, default=5, help='Number of folds when using --cv')
    parser.add_argument(
        '--seeds',
        type=int,
        nargs='+',
        help='List of random seeds to use. Defaults to [42] if omitted.',
    )
    parser.add_argument('--seed', type=int, default=42, help='Convenience single-seed flag')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = args.seeds if args.seeds else [args.seed]

    X, y = load_cup_data(os.path.join('data', 'ML-CUP25-TR.csv'))
    print(f"Loaded training data: X={X.shape}, y={y.shape}")
    mode = 'cv' if args.cv else 'holdout'
    print(f"Running mode={mode}, seeds={seeds}, quick={args.quick}")

    if args.cv:
        results = run_cv(X, y, seeds, args.n_splits, args.quick)
    else:
        results = run_holdout(X, y, seeds, args.quick)

    summarize(results)


if __name__ == '__main__':
    main()
