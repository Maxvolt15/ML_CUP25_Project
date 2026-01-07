# scripts/benchmarks/ - Reality Checks

I use these benchmarks to make sure my neural network implementation is actually correct and not just "lucky" on the project data.

## Why I wrote these
It's easy to fool yourself into thinking your model works. I wanted to test my `NeuralNetworkV2` class on standard, well-known problems (like the MONK datasets) where the answer is known.

## The Benchmarks

### 1. `run_monk_benchmark.py`
**I wrote this because:** The MONK datasets are the standard sanity check for this course. If my network can't learn MONK-2, it definitely won't learn the CUP.
**What it does:** Trains on MONK-1, 2, and 3 and reports accuracy.
**My Results:** I achieved ~100% accuracy on MONK-1 and MONK-3, and acceptable error on MONK-2, proving my backprop works.

Usage:
```bash
python -m scripts.benchmarks.run_monk_benchmark
```

### 2. `test_traders_strategy.py`
**I wrote this because:** I wanted to see how the previous year's winner ("Traders") would perform on *this year's* data.
**My Finding:** Their simple model wasn't enough for 2025's complexity, which confirmed I needed a deeper network.

## My Takeaway
These scripts gave me the confidence to proceed to the main challenge.