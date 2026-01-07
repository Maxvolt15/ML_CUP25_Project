# ML-CUP25 Project: Neural Network from Scratch

## 📌 Project Overview

This project implements a **Regression Neural Network** from first principles (using only `numpy`) to solve the ML-CUP25 challenge. The goal is to predict 4 continuous target variables from 10 input features.

**Constraints:** Type-A Project (No high-level frameworks like PyTorch/TensorFlow allowed for the model core).

## 📂 Repository Structure

```text
ML_CUP25_Project/
├── data/                    # Dataset files (TR.csv, TS.csv)
├── src/                     # Core library code
│   ├── neural_network_v2.py # The MLP implementation
│   ├── data_loader.py       # Feature engineering pipeline
│   └── utils.py             # Math & Metric helpers
├── scripts/                 # Execution scripts organized by phase
│   ├── phase1/             # Phase 1: Validation & Baseline
│   ├── phase2/             # Phase 2: Ensemble & Optimization
│   ├── phase3/             # Phase 3: Advanced Features
│   ├── search_algorithms/  # Hyperparameter search methods
│   ├── utilities/          # Debug and utility scripts
│   ├── benchmarks/         # Benchmark tests
│   └── README.md           # Scripts guide
├── experiments/             # Results logs & saved configs
├── docs/                    # Project documentation (internal)
├── config.json              # Configuration file
└── README.md               # This file
```

## 🚀 Key Findings & Performance

### 1. The "Hall of Fame" Baseline

After extensive search and validation, the best single model achieves:

- **MEE (Mean Euclidean Error):** 22.30 ± 1.74 (5-fold cross-validation)
- **Phase 2 Ensemble:** 13.75 ± 0.80 MEE (10-model ensemble)
- **Architecture:** [128, 84, 65] neurons, Tanh activation
- **Features:** Polynomial Degree 2 (90 input features)
- **Regularization:** Dropout (0.131) + L2

### 2. Validation Methodology

- **Rigorous Validation:** 5-fold cross-validation ensures reliable performance estimates
- **Initial Report:** 12.27 MEE (on unreliable 90/10 split)
- **Verified Baseline:** 22.30 MEE (proper 5-fold CV) reveals true performance
- **Ensemble Improvement:** 13.75 MEE (1.02 MEE better than single model, 6.9% improvement)

### 3. Comparison to Previous Work

- **ML-CUP24 Coursework Benchmark :** The reference implementation achieved 24.87 MEE on 2024 year's dataset. This serves as a baseline for understanding the task complexity.
- **Our Model:** 22.30 MEE (Best Single) -> **13.75 MEE** (Ensemble Phase 2). Significant improvement over baseline approach.

## 🛠️ How to Run

**Prerequisites:** Python 3.8+, Numpy, Pandas, Scikit-Learn (for splitting/scaling only).

**Important:** Run all scripts from the **root directory** as modules to ensure imports work.

### 1. Run the Baseline Validation

```bash
python -m scripts.phase1.run_hall_of_fame_5fold
```

### 2. Run the Ensemble (Phase 2)

```bash
python -m scripts.phase2.ensemble_simple
```

### 3. Generate Test Predictions

```bash
python -m scripts.phase2.generate_submission
```

### 4. Run Hyperparameter Search

```bash
python -m scripts.search_algorithms.genetic_search
```

For detailed phase documentation, see:
- `scripts/phase1/README.md` - Phase 1 validation scripts
- `scripts/phase2/README.md` - Phase 2 ensemble scripts
- `scripts/search_algorithms/README.md` - Search method experiments
- `scripts/utilities/README.md` - Utility and debug scripts
- `scripts/benchmarks/README.md` - Benchmark tests

## 📜 Lifecycle Status

- [x] **Phase 0:** Infrastructure & Core Implementation (Complete)
- [x] **Phase 1:** Architecture Validation (Complete - 22.30 MEE Baseline)
- [x] **Phase 2:** Ensembling (Complete - 13.75 ± 0.80 MEE Achieved ✅)
- [ ] **Phase 3:** Advanced Features (On Hold - ROI deemed low after Phase 2 success)

## ✍️ Authors

Suranjan aka maxvolt.
