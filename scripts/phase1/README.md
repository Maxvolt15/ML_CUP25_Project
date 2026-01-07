# scripts/phase1/ - Validation & Baseline Testing

**Status:** ✅ Complete - I established a trustworthy baseline at 22.30 MEE (5-fold CV)

---

## Why Phase 1 Matters

In Phase 1, I needed to establish a trustworthy baseline score through rigorous validation. The initial naive implementation reported 12.27 MEE, but that was on a 90/10 split—a trap that could mislead the entire project. So I did three critical things:

1. **I implemented a neural network from scratch** using only numpy (NeuralNetworkV2)
2. **I corrected validation to proper 5-fold cross-validation** to get a reliable metric (22.30 MEE)
3. **I tested multiple architectures and configurations** to find the best "hall of fame" model for Phase 2

---

## The Scripts

### `run_hall_of_fame_5fold.py` (MAIN BASELINE)

**Why I wrote this:** I needed to reproduce the best model on proper 5-fold CV to establish a reliable baseline.

**What it does:** Trains the "hall of fame" architecture on 5 independent folds and reports the average MEE.

**The Discovery:** Corrected validation from 12.27 MEE (90/10 split) to **22.30 ± 1.74 MEE (5-fold CV)**

**Usage:**
```bash
python -m scripts.phase1.run_hall_of_fame_5fold
```

**Model Configuration:**
- Architecture: [128, 84, 65] (3-layer deep network)
- Activation: tanh
- Learning rate: 0.0120
- Dropout: 0.131
- Optimizer: adam
- Features: Polynomial degree 2 (90 input features)

**Result:** 22.30 ± 1.74 MEE (5-fold CV)
**Time:** ~2-3 hours (trains 5 folds)

---

### `run_baseline_comparison.py` (QUICK CHECK)

**Why I wrote this:** I wanted a faster way to validate the baseline without waiting 3 hours for the full 5-fold run.

**What it does:** Trains a single instance of the hall of fame model and reports quick validation results.

**Usage:**
```bash
python -m scripts.phase1.run_baseline_comparison
```

**When to use:** Quick sanity check or during development, when you need results in 30-45 minutes instead of 3 hours.

---

### `test_search_iter7_validation.py` (VALIDATION TEST)

**Why I wrote this:** During my hyperparameter search, I found an interesting result at iteration 7. I wanted to validate if it was actually better than the baseline on proper 5-fold CV.

**What it does:** Takes the iteration 7 configuration from my search results and tests it on proper 5-fold cross-validation.

**Usage:**
```bash
python -m scripts.phase1.test_search_iter7_validation
```

**Purpose:** This test answered the question: "Did my search algorithm actually find something better?" (Spoiler: not quite—the hall of fame remained the best.)

---

### `test_iter25_poly2.py` (FEATURE ENGINEERING EXPLORATION)

**Why I wrote this:** Iteration 25 achieved 18.04 MEE on raw features with a leaky_relu [256,128,64] architecture. I wondered: what if I add polynomial features? Could that push it even higher?

**What it does:** Tests the iteration 25 configuration WITH polynomial degree 2 features on 5-fold CV.

**Usage:**
```bash
python -m scripts.phase1.test_iter25_poly2
```

**Purpose:** Explore if feature engineering could unlock the potential of this promising architecture.

---

### `test_previous_year_strategy.py` (HISTORICAL BASELINE)

**Why I wrote this:** I wanted to understand how much progress the field had made. ML-CUP24 used a simple approach: [30] single-layer network with tanh. How does that perform on this year's data?

**What it does:** Implements the previous year's simple strategy and tests it on the CUP25 dataset.

**Usage:**
```bash
python -m scripts.phase1.test_previous_year_strategy
```

**Model:** [30] single hidden layer, tanh activation, with polynomial degree 2 features
**Time:** ~30 minutes (simple model, trains fast)

**Purpose:** Baseline comparison—confirms that simple networks aren't enough for this year's challenge.

---

### Legacy Scripts

- **`baseline_test.py`**: Original Phase 1 test (superseded by `run_hall_of_fame_5fold.py`)

---

## 📊 What I Found

| Configuration | MEE | Architecture | Key Insight |
| --- | --- | --- | --- |
| **Hall of Fame** | **22.30 ± 1.74** | [128, 84, 65] | ✅ Best baseline |
| Iter 25 (raw features) | ~18.04 | [256, 128, 64] | Deep network works, but unstable |
| Iter 7 (search result) | ~22.5 | Varied | Hyperparameter search didn't beat hall of fame |
| Previous Year (CUP24) | ~24.87 | [30] | Simpler networks underperform |

---

## 🎯 Key Conclusions from Phase 1

1. **90/10 split is a trap** - The reported 12.27 MEE was misleading. Proper 5-fold CV revealed 22.30 MEE.
2. **Deep networks beat simple ones** - [128,84,65] > [30] by a significant margin.
3. **Polynomial features help** - All tested models benefited from degree 2 polynomial feature engineering.
4. **Hall of Fame is solid** - Despite searching many architectures, [128,84,65] remained the best.
5. **Ready for Phase 2** - Strong baseline established; now time to ensemble and optimize further.

---

## 🚀 How to Use Phase 1 Scripts

**To reproduce the baseline:**
```bash
python -m scripts.phase1.run_hall_of_fame_5fold
```

**For a quick test:**
```bash
python -m scripts.phase1.run_baseline_comparison
```

**To explore feature engineering:**
```bash
python -m scripts.phase1.test_iter25_poly2
```

---

## 📝 Next: Phase 2

With the hall of fame baseline established at 22.30 MEE, Phase 2 took the next logical step: **ensemble multiple models**. Could averaging 10 hall of fame models reduce variance and improve generalization? Spoiler: yes—achieving 13.75 MEE.

See `scripts/phase2/README.md` for details.