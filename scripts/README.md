# Scripts Organization Guide

## Directory Structure

```
scripts/
├── phase1/              # Validation & Baseline Testing
├── phase2/              # Ensemble & Optimization
├── phase3/              # Feature Engineering & Advanced Techniques
├── benchmarks/          # Benchmark tests (Monks, etc)
├── previous_year/       # Previous year strategies (reference only)
├── utilities/           # Debug, testing, and utility scripts
└── README.md            # This file
```

---

## Phase 1: Validation & Baseline

**Purpose:** Establish reliable baseline performance and validate assumptions.

### Scripts in `phase1/`:
- `run_hall_of_fame_5fold.py` - Run baseline single model on proper 5-fold CV
- `test_search_iter7_validation.py` - Validate search results on 5-fold CV
- `baseline_test.py` - Quick baseline testing utilities

### Expected Output:
- Baseline MEE: ~14.77 MEE on 5-fold CV (verified Hall of Fame single model)
- Phase 1 reference: 22.30 MEE (earlier evaluation, Hall of Fame architecture test)
- Identification of misleading 90/10 split results
- Validated configuration for ensemble

### How to Run:
```bash
python scripts/phase1/run_hall_of_fame_5fold.py
python scripts/phase1/test_search_iter7_validation.py
```

---

## Phase 2: Ensemble & Optimization

**Purpose:** Combine models to reduce variance and improve generalization.

### Scripts in `phase2/`:
- `ensemble_simple.py` - Simple averaging ensemble of hall of fame models
- `ensemble_training.py` - Alternative ensemble training approach
- `ensemble_training_v2.py` - Enhanced ensemble with weighted averaging
- `generate_submission.py` - Generate test set predictions using ensemble
- `PHASE2_QUICKSTART.py` - Quick start guide and code templates

### Expected Output:
- Ensemble MEE: 13.75 MEE on 5-fold CV (1.02 MEE better than 14.77 baseline)
- Test predictions: `experiments/ML-CUP25-TS-predictions.csv`
- Performance improvement: 6.9% variance reduction through ensemble

### How to Run:
```bash
# Run ensemble validation on 5-fold CV
python scripts/phase2/ensemble_simple.py

# Generate test set predictions
python scripts/phase2/generate_submission.py
```

---

## Phase 3: Feature Engineering & Advanced Techniques

**Purpose:** Explore advanced optimization and feature engineering strategies.

### Scripts in `phase3/`:
- `genetic_search.py` - Genetic algorithm for hyperparameter search
- `hyperparameter_search.py` - Basic grid/random search
- `hyperparameter_search_v2.py` - Enhanced hyperparameter search
- `intensive_hyperparameter_search.py` - Exhaustive search (slow, thorough)
- `intensive_training.py` - Extended training runs
- `run_phase3_advanced_features.py` - Advanced feature engineering

### Expected Output:
- Improved hyperparameters
- Feature engineering results (PCA, polynomial degree 3, etc)
- Performance: 12-13 MEE (marginal improvement)

### How to Run:
```bash
# Run genetic algorithm search
python scripts/phase3/genetic_search.py

# Try advanced feature engineering
python scripts/phase3/run_phase3_advanced_features.py
```

---

## Benchmarks

**Purpose:** Test on smaller datasets and validate implementation.

### Scripts in `benchmarks/`:
- `run_monk_benchmark.py` - Monks dataset benchmark (3 datasets)

### How to Run:
```bash
python scripts/benchmarks/run_monk_benchmark.py
```

---

## Previous Year Reference

**Purpose:** Reference implementations from ML-CUP 2024 (read-only).

### Scripts in `previous_year/`:
- `test_previous_year_strategy.py` - 2024 ML-CUP strategy
- `test_traders_strategy.py` - Alternative 2024 approach

### Note:
These are reference implementations only. If using, adapt paths and configurations to current project structure.

---

## Utilities & Testing

**Purpose:** Debugging, testing, and utility functions.

### Scripts in `utilities/`:
- `debug_training.py` - Debug training process
- `debug_training_v2.py` - Enhanced debugging
- `quick_test.py` - Quick validation tests
- `simple_search.py` - Simple hyperparameter search
- `main.py` - Alternative entry point
- `test_iter25_poly2.py` - Test iteration 25 with poly2 features
- `run_final_model.py` - Final model training

### How to Run:
```bash
# Quick test
python scripts/utilities/quick_test.py

# Debug training
python scripts/utilities/debug_training.py
```

---

## Typical Workflow

### Complete Pipeline (New Run):
```bash
# Step 1: Validate baseline
python scripts/phase1/run_hall_of_fame_5fold.py

# Step 2: Run ensemble
python scripts/phase2/ensemble_simple.py

# Step 3: Generate predictions
python scripts/phase2/generate_submission.py

# Step 4: (Optional) Try advanced features
python scripts/phase3/genetic_search.py
```

### Quick Testing:
```bash
python scripts/utilities/quick_test.py
```

### Benchmarking:
```bash
python scripts/benchmarks/run_monk_benchmark.py
```

---

## Import Paths

After reorganization, update imports in scripts:

### Old (Before):
```python
from src.neural_network_v2 import NeuralNetworkV2
from src.data_loader import load_cup_data
```

### New (After):
Since scripts are in subfolders, relative imports need to go up one level:
```python
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.neural_network_v2 import NeuralNetworkV2
from src.data_loader import load_cup_data
```

Or use absolute imports from project root:
```bash
# Run from project root
python -m scripts.phase1.run_hall_of_fame_5fold
python -m scripts.phase2.ensemble_simple
```

---

## Root-Level Scripts

Certain scripts remain in the root for convenience:
- `generate_submission_fast.py` - Fast test prediction (5 models, 800 epochs)
- `run_baseline_comparison.py` - Quick baseline verification
- `run_hall_of_fame_5fold.py` - Copy in root for easy access (alternative: remove and use scripts/phase1/)
- `test_previous_year_strategy.py` - Copy in root for reference (alternative: remove and use scripts/previous_year/)

**Recommendation:** Remove root copies once scripts/ structure is verified to work properly.

---

## Phase Completion Status

| Phase | Status | Key Script | Output |
|---|---|---|---|
| **Phase 1** | ✅ Complete | `run_hall_of_fame_5fold.py` | Baseline: 22.30 MEE |
| **Phase 2** | ✅ Complete | `ensemble_simple.py` | Ensemble: 13.75 MEE |
| **Phase 3** | ⏳ Optional | `genetic_search.py` | Advanced optimization |

---

## Troubleshooting

### Import Errors
If you get "ModuleNotFoundError: No module named 'src'":
```bash
# Run from project root
cd /path/to/ML_CUP25_Project
python -m scripts.phase1.run_hall_of_fame_5fold
```

### Path Issues
Scripts reference relative paths like `data/`, `experiments/`:
- Always run scripts from project root: `/path/to/ML_CUP25_Project/`
- Or update script to adjust working directory

### Missing Dependencies
```bash
# Install required packages
pip install numpy scikit-learn pandas
```

---

## Best Practices

1. **Always run from project root:**
   ```bash
   cd /path/to/ML_CUP25_Project
   python -m scripts.phase1.run_hall_of_fame_5fold
   ```

2. **Use consistent random seeds:**
   ```python
   RANDOM_STATE = 42  # Keep consistent across runs
   ```

3. **Document results:**
   - Save results to `experiments/` folder
   - Update `docs/` with findings

4. **Version control:**
   - Commit docs/ updates
   - Don't commit large experiment outputs
   - Use `.gitignore` for model weights

---

## Next Steps

1. **Update imports** in all scripts to handle new subfolder structure
2. **Test phase1 scripts** to verify they work from new locations
3. **Remove root duplicates** once verified working
4. **Update CI/CD** or run scripts if applicable
5. **Create phase-specific runners** if needed for batch execution

---

## Questions?

See `docs/` for detailed analysis and methodology.
- `docs/ML_CUP_2025_COMPLETE_ANALYSIS.md` - Full project analysis
- `docs/PHASE2_COMPLETE_ANALYSIS.md` - Ensemble analysis
- `docs/SESSION_SUMMARY.md` - Latest session work
