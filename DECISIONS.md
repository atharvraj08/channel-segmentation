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
- **model_agent**: MODEL_FEATURES = all 15 FEATURE_COLUMNS (no hand-picking); LOG_COLUMNS = money/count/skewed columns (total_spend, order_count, avg_basket, online_orders, store_orders, recency_days, app_sessions, web_sessions) because they are right-skewed; shares/rates/loyalty stay linear.
- **model_agent**: Plain KMeans (n_init=10) chosen by silhouette; first attempt already gave seed 42 silhouette by k {2:.261,3:.278,4:.284,5:.303,6:.273,7:.279,8:.254} → k=5, so GaussianMixture was not tried. Throwaway check (agents/logs/model_check.py) ARI 0.941 (seed 42), 0.941 (7), 0.944 (123); ARI was only looked at, never used to choose anything.
- **model_agent**: name_segments uses ordered if/else rules on profile means (grocery_share>0.5 → Grocery; order_count<6 → Browsers; total_spend>5000 & 0.35≤online_share≤0.65 → VIPs; online_share>0.65 & discount_rate>0.4 → Deal-Seekers; online_share<0.35 & loyalty>0.5 → Loyalists; else channel-based fallback), duplicates get " 2", " 3". Gives the 5 intended names on seeds 42, 7, 123, 2024.
- **model_agent**: Created temporary root run_all.py (marked TEMPORARY) implementing run_pipeline without charts so tests/test_evaluate.py can run; viz_agent replaces it.
- **viz_agent**: `plot_pca_segments` takes an extra optional keyword `names=None` (segment id -> name) so the legend can show segment names; the SPEC positional signature is unchanged. `make_all_charts` passes `profiles["name"]`.
- **viz_agent**: `runtime_seconds` covers the whole run including charts (metrics.json is written last); profile heatmap colours are z-scores across segments with raw means as annotations; Okabe-Ito colour-blind-safe palette.
