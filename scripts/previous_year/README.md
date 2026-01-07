# scripts/previous_year/ - Learning from the Past

I created this folder to keep reference implementations from previous ML-CUP challenges.

## Why I kept this
I believe in not reinventing the wheel—at least, not without looking at the old wheel first. I wanted to test if simple strategies from 2024 would work on the 2025 dataset.

## The Scripts

### `test_previous_year_strategy.py`
**I wrote this because:** I needed a baseline. Before building a complex deep network, I tested a simple shallow network (like the one that won last year).
**The Result:** It performed poorly (MEE > 24), which proved to me that this year's function is much more complex and non-linear. This justified my decision to move to deeper architectures (3+ layers).

## My Takeaway
History is a good teacher, but this year's challenge required a new approach.
