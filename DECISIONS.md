# Decisions log

Append-only. Each entry is prefixed with the agent that made it.

- **orchestrator**: Python 3.14.7 venv; latest numpy/pandas/scikit-learn/matplotlib/pytest pinned in requirements.txt.
- **orchestrator**: Added `pyproject.toml` (package installed editable into `.venv`) and pytest `pythonpath = ["src", "."]` so tests can `import segmentation` and `import run_all` without path hacks.
- **orchestrator**: Fixed `END_DATE = 2026-09-30` in config so generated dates do not depend on "today" (reproducibility).
- **orchestrator**: Every customer gets ≥ 1 transaction (simplest way to make all features well defined); features.py still guards against 0/0.
- **orchestrator**: Clustering uses `features.MODEL_FEATURES` (a subset of FEATURE_COLUMNS the Model Agent may tune) so feature selection never breaks the mean≈0/std≈1 contract.
- **orchestrator**: `run_all.run_pipeline(...)` accepts data/output dirs so tests can run the whole pipeline in a temp folder; this is also what lets the mutation check (shuffled rows) be caught by the ARI test.
- **orchestrator**: metrics.json is written once at the end by `evaluate.save_metrics` (includes the per-k silhouette scores from `choose_k`) instead of being written by two modules.
