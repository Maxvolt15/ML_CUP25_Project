# Changelog# Changelog



All notable changes to this project will be documented in this file.All notable changes to this project will be documented in this file.



## [0.2.0] - 2026-01-07The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),

and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

### Changed

- Reorganized scripts folder with phase-based structure (phase1/, phase2/, phase3/)## [Unreleased]

- Added separate folders for search_algorithms/, utilities/, and benchmarks/

- Updated .gitignore to exclude docs/ folder and auto-generated files### Added

- Updated README with accurate baseline metrics and proper attribution

- Clarified Phase 1 baseline: 22.30 MEE (5-fold CV)- `LICENSE` file (MIT).

- Clarified Phase 2 ensemble result: 13.75 MEE (6.9% improvement)- `CONTRIBUTING.md` guide.

- Fixed benchmark reference: ML-CUP24 coursework silver medal (24.87 MEE baseline)- `CODE_OF_CONDUCT.md`.

- `docs/diagrams/` directory for architectural visuals.

### Removed- `tests/` directory structure.

- Auto-generated documentation files from root (ARCHITECTURE_COMPARISON.md, PHASE1_SUMMARY.md, etc.)

- FINAL PROJECT reference that was not applicable### Changed

- Untracked helper files (GIT_COMMIT_SUMMARY.md, GITHUB_DESCRIPTION.md, etc.)

- Refactored root directory: Moved organization/temporary files to `docs/refactoring_history`.

### Added- Updated `README.md` to include contribution guidelines and improved structure.

- Phase 2 README with detailed documentation

- Search algorithms README explaining experimental scripts## [0.2.0] - 2026-01-05

- Utilities README for debug and testing scripts

- Benchmarks README for validation tests### Added

- Proper folder structure for better organization

- Ensemble implementation (`scripts/ensemble_simple.py`).

## [0.1.0] - 2025-12-20- Phase 2 ensemble validation achieving 13.75 MEE target.



### Added### Changed

- Initial project setup with ML-CUP25 dataset

- Neural network implementation from scratch (NeuralNetworkV2)- Updated Lifecycle Status: Phase 2 marked as Complete.

- Phase 1: Baseline validation with 22.30 MEE- Clarified Traders: Top achievers from CUP24 (not previous winner), used for performance comparison.

- Phase 2: Ensemble implementation achieving 13.75 MEE

- Data preprocessing with polynomial features## [0.1.0] - 2026-01-05

- Cross-validation methodology for rigorous evaluation

- Hall of fame model tracking### Added

- Documentation and analysis scripts

- Initial implementation of Neural Network from scratch (`src/neural_network_v2.py`).
- Data loading and preprocessing pipeline (`src/data_loader.py`).
- Base validation scripts (`scripts/run_hall_of_fame_5fold.py`).
- Hyperparameter search scripts (`scripts/intensive_hyperparameter_search.py`).
- Documentation structure.
