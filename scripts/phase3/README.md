# scripts/phase3/ - Feature Engineering (Experimental)

**Status:** 🟡 On Hold - Phase 2 results (13.75 MEE) were strong enough that I decided not to prioritize this.

---

## Why This Folder Exists

I created Phase 3 to explore **advanced feature engineering** techniques if the ensemble approach in Phase 2 didn't meet my targets.

My plan was to implement:
1. **Polynomial Degree 3 Expansion**: Blowing up the feature space from 10 to ~450 features.
2. **Principal Component Analysis (PCA)**: Compressing those 450 features back down to the most critical ~20 components.

## Why It's Incomplete

I achieved **13.75 MEE** in Phase 2 using a 10-model ensemble. This exceeded my expectations (target was 15 MEE) and is highly competitive. Given the complexity and computational cost of training on degree-3 polynomial features, I decided that the return on investment (ROI) for Phase 3 was low.

## The Scripts

### `run_phase3_advanced_features.py`
**I wrote this because:** I wanted a script ready to test the Poly3 + PCA pipeline if needed.
**Current State:** It is a theoretical implementation. It hasn't been rigorously tested because I stopped after Phase 2 success.

## My Recommendation

If you want to push the model even further (e.g., trying to break 10 MEE), this is where you should start. But be warned: training times will explode with degree-3 features.

For now, the Phase 2 ensemble is the champion.