# Change Log
All notable changes to this project will be documented in this file.

This file continues the upstream CAFA-evaluator change log. Fork-specific
modifications (the speedup phases and the partial-knowledge coverage fix)
are tracked separately, with dates, in `CHANGES.md` as required by GPLv3.

## [current] - unreleased
- Quality sweep (2026-06-10): conservative, parity-preserving cleanup of
  the source (dead code and unused imports removed, ambiguous identifier
  renamed, package docstring and `__version__` added). No change to any
  metric value; the upstream parity harness stays green.
- Added `ruff`, `mypy` and `pytest` configuration under `pyproject.toml`,
  plus `test` and `dev` optional-dependency extras.
- Added `ci` (ruff + mypy + pytest + parity) and `docs` (Sphinx build,
  warnings as errors) GitHub Actions workflows.
- Expanded the Sphinx API reference and aligned the documentation source
  branch with the fork trunk.
- Upstream goal carried forward: bootstrap confidence intervals.

## [1.2.1] - 2024-03-26
- Minor bugfix affecting multi-thread calculation. 

## [1.2.0] - 2024-02-01 
- We included micro-average metrics. Now precision, recall and F-score 
in addition to previously reported metrics are calculated as micro-averages 
by averaging the confusion matrices over targets before calculating aggregated metrics
(precision, recall, ect.).
- We include the calculation of the average precision score (APS) in the plot notebook.

### Changed
- plot.ipynb, added calculation of average precision score (APS) in the plot notebook.
- evaluation.py, micro-average calculation, some refactoring of the core functions.
- parser.py, minor fixes and improvements.

## [1.1.0] - 2024-01-24
  
- We changed the way alternative identifiers in ontology files are considered.
Now alternative identifiers are recognized in both the ground truth and prediction files
and mapped to the "canonical" term.
 
### Added
- CHANGELOG.md, this file!
 
### Changed
- graph.py, changed the Graph class.
- parser.py, changed the ground truth and prediction parsers in order 
to replace alternative identifiers with canonical terms.
- plot.ipynb, cleaned up.

 
## [1.0.0] - 2023-08-04
 
First release after CAFA5 challenge closed. The version used in Kaggle is
provided under the 'Kaggle' branch. The 'main' branch includes instead a
packed version of the code. While usage and performance has been improved 
the calculation is exactly the same.