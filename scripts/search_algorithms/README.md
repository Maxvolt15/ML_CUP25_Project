# scripts/search_algorithms/ - My Hyperparameter Hunt

I used these scripts to explore the vast space of hyperparameters and find the configuration that eventually became my "Hall of Fame" model.

## Why I wrote these

Finding the right number of neurons, learning rate, and regularization is impossible by hand. I wrote these algorithms to automate that discovery process.

## The Scripts

### 1. `genetic_search.py` (The Heavy Lifter)
**I wrote this because:** I wanted an evolutionary approach that could "breed" better models over time.
**How it works:** It creates a population of random models, trains them, and then combines the genes (hyperparameters) of the best ones.
**Status:** This was my primary tool for Phase 2 exploration.

Usage:
```bash
python -m scripts.search_algorithms.genetic_search
```

### 2. `hyperparameter_search.py` & `_v2.py`
**I wrote these because:** Sometimes a simple random search is enough to find a starting point.
**How it works:** It randomly samples parameters from distributions I defined and evaluates them.

### 3. `intensive_hyperparameter_search.py`
**I wrote this because:** I wanted to be absolutely sure I wasn't missing a global optimum, so I set up a very thorough (and slow) search grid.

## My Takeaway
These scripts were crucial for finding the `[128, 84, 65]` architecture. Without them, I would have been guessing in the dark.