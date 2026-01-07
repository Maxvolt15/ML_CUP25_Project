# scripts/utilities/ - My Toolbox

I keep my helper scripts here. These aren't part of the main pipeline, but they were essential for me to build and debug the project.

## Why I have this folder
When building a neural network from scratch, things break. Gradients vanish, loss explodes, or dimensions mismatch. I needed a place for scripts that help me look "under the hood."

## The Tools

### 1. `debug_training.py`
**I wrote this because:** I needed to see exactly what was happening during training—step by step—to catch bugs in my backpropagation implementation.

### 2. `quick_test.py`
**I wrote this because:** I wanted a fast way (~5 mins) to check if the code runs without crashing, before committing to a full 4-hour training run.
Usage:
```bash
python -m scripts.utilities.quick_test
```

### 3. `run_final_model.py`
**I wrote this because:** Once I had my best model, I needed a clean, standalone script to train it and save the final weights.

## My Takeaway
A clean codebase needs a messy "workbench" where you can test things. This is my workbench.