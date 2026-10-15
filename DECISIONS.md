# Decisions log

Append-only. Each entry is prefixed with the agent that made it.

- **orchestrator**: Python 3.14.7 venv; latest numpy/pandas/scikit-learn/matplotlib/pytest pinned in requirements.txt.
- **orchestrator**: Added `pyproject.toml` (package installed editable into `.venv`) and pytest `pythonpath = ["src", "."]` so tests can `import segmentation` and `import run_all` without path hacks.
- **orchestrator**: Fixed `END_DATE = 2026-09-30` in config so generated dates do not depend on "today" (reproducibility).
- **orchestrator**: Every customer gets ≥ 1 transaction (simplest way to make all features well defined); features.py still guards against 0/0.
- **orchestrator**: Clustering uses `features.MODEL_FEATURES` (a subset of FEATURE_COLUMNS the Model Agent may tune) so feature selection never breaks the mean≈0/std≈1 contract.
- **orchestrator**: `run_all.run_pipeline(...)` accepts data/output dirs so tests can run the whole pipeline in a temp folder; this is also what lets the mutation check (shuffled rows) be caught by the ARI test.
- **orchestrator**: metrics.json is written once at the end by `evaluate.save_metrics` (includes the per-k silhouette scores from `choose_k`) instead of being written by two modules.
- **data_agent**: Segment sizes are exact (24/22/12/24/18%, rounding remainder to segment 0) and then shuffled, so no segment can drop below 8% by chance.
- **data_agent**: Per-customer params: orders ~ NegBin (min 1), online prob ~ Beta, basket ~ lognormal (customer sigma 0.5, order sigma 0.45), discount ~ Beta, category mix ~ Dirichlet, app/web ~ Gamma(shape 2), email ~ Beta, loyalty ~ Bernoulli, last-purchase gap ~ Exponential (one order pinned at that gap).
- **data_agent**: `days_since_last_visit` = min(last-purchase gap, an exponential session gap with mean 30/(app+web+0.5)), so a "visit" covers a purchase or a digital session.
- **data_agent**: Tuning: Browsers at mean 1.5 orders made k=2 (browsers vs buyers) win silhouette; raised to 2.5 orders and 60-day gap. VIPs merged with Loyalists at k=4; raised VIP basket to $220 and orders to 50.
- **data_agent**: Feasibility (agents/logs/data_sanity.py, 15 simple features, log1p money/counts, StandardScaler, KMeans n_init=10): seed 42 silhouette by k {2:.269,3:.260,4:.274,5:.285,6:.278,7:.267,8:.246}, best k=5, ARI(k=5)=0.923. Seeds 0,1,3,7,11,123,2024 all give best k=5, ARI 0.918–0.935, silhouette 0.285–0.294.
- **test_agent**: Date-window test accepts dates in [END_DATE - 365 days, END_DATE] so either reading of "365 days ending END_DATE" passes.
- **test_agent**: conftest imports project modules lazily inside fixtures so one missing module does not break collection of unrelated test files.
- **test_agent**: `name_segments` is tested by passing the full profiles frame (PROFILE_COLUMNS + size + share + name); implementations should only rely on PROFILE_COLUMNS/size/share.
- **test_agent**: test_pipeline's A12 subprocess run writes into the real `data/` and `outputs/` folders (that is what `run_all.py` does by default).
