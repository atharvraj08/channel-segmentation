# REVIEW: Omnichannel Customer Segmentation

Reviewer: `reviewer` agent, 2026-10-15. Audited against `SPEC.md`. Read-only except this file.

## Verdict

**No HIGH-severity FAILs. No MEDIUM issues.** All 14 acceptance criteria PASS and all 4 mutations
are caught by the test suite. Only LOW-severity notes remain (listed at the end).

Headline numbers (seed 42): k = 5, silhouette 0.303, ARI 0.941, `run_all.py` 3.4 s wall,
`pytest -q` 68 passed in 9.27 s.

---

## 1. Acceptance criteria

| # | Requirement | Result | Evidence (command → key output) |
|---|---|---|---|
| A1 | ~5,000 customers, columns, no NaN, amounts > 0, rates in [0,1] | **PASS** | `generate_data()` check → `customers 5000 tx 136447 na 0 min amt 1.1 email 0.0011 0.9965`; `test_exact_columns`, `test_no_missing_values`, `test_engagement_value_ranges` pass |
| A2 | Same seed twice → identical data | **PASS** | two `generate_data()` calls → `A2 identical True`; `test_same_seed_gives_identical_data` passes |
| A3 | Labels absent from transactions/engagement | **PASS** | `'true_segment' in tx/eng columns` → `False`; `test_no_segment_names_in_inputs` scans every text value |
| A4 | One row/customer, finite, scaled mean≈0 std≈1 | **PASS** | `build_features`+`scale_features` → `rows 5000, unique True, finite True, max|mean| 4.8e-16, max|std-1| 2.2e-16` |
| A5 | `features.py`/`cluster.py` never contain `ground_truth`/`true_segment` | **PASS** | `grep -nE "ground_truth\|true_segment"` → no hits in either file; `test_model_code_never_mentions_labels` passes |
| A6 | Chosen k = 5 ± 1 | **PASS** | `.venv/bin/python run_all.py` → `chosen k  : 5` |
| A7 | ARI ≥ 0.70 | **PASS** | `run_all.py` → `ARI       : 0.941` |
| A8 | Silhouette ≥ 0.15 | **PASS** | `run_all.py` → `silhouette: 0.303` |
| A9 | Every segment ≥ 3% | **PASS** | sizes `597/1180/890/1225/1108` → smallest share `0.1194` |
| A10 | Different seed → ARI ≥ 0.80 vs first run | **PASS** | `fit_segments(X,5,seed)` → `ARI 42v7 1.0, 42v123 1.0` |
| A11 | Some online_share > 0.65 and some < 0.35 | **PASS** | `outputs/segment_profiles.csv` → max `0.832` (Digital Deal-Seekers), min `0.159` (In-Store Loyalists) |
| A12 | `run_all.py` < 120 s, exit 0, all outputs | **PASS** | `time -p .venv/bin/python run_all.py` → `real 3.42`, `exit=0`; metrics.json, segment_profiles.csv, customer_segments.csv and all 4 charts exist and are non-empty |
| A13 | `pytest -q` passes in < 60 s | **PASS** | `time -p .venv/bin/python -m pytest -q` → `68 passed in 9.27s`, `real 9.57` |
| A14 | Tests catch the 4 mutations | **PASS** | see mutation table below: 4/4 caught |

Extra checks:
- **Reproducibility:** ran `run_all.py` twice and compared `metrics.json` without `runtime_seconds` → `identical: True`.
- **Docstrings:** an AST scan of `src/segmentation/*.py`, `run_all.py` and `tests/*.py` found **0** functions without a docstring.
- **Spec conformance:** signatures, metrics keys, profile columns (`PROFILE_COLUMNS` + size/share/name), file names and `matplotlib.use("Agg")` + `plt.close` all match the spec.

## 2. Mutation check

Setup: copied the project (without `.venv`, `.git`, `data`, `outputs`, `__pycache__`) to `/tmp/seg_mutation_test` and ran it with the original interpreter. A temporary probe test printed
`IMPORTED /private/tmp/seg_mutation_test/src/segmentation/cluster.py /private/tmp/seg_mutation_test/run_all.py`.
So pytest's `pythonpath = ["src", "."]` does take precedence over the editable install. Unmutated copy: `68 passed`.
Each mutation was applied on its own, and the file was restored and diffed against the original afterwards. `/tmp/seg_mutation_test` was deleted at the end.

| Mutation | Change made | Result | First failing test(s) |
|---|---|---|---|
| (a) shuffled feature rows | `run_pipeline`: `X = X[rng(0).permutation(len(X))]` before `choose_k`/`fit_segments`, ids unchanged | **CAUGHT** | `test_evaluate.py::test_pipeline_ari_at_least_0_70` (ARI ≈ -0.00) |
| (b) hard-coded k = 2 | `choose_k` returns `2, scores` | **CAUGHT** | `test_cluster.py::test_chosen_k_is_five_plus_minus_one` |
| (c) scaling removed | `scale_features` returns raw `df[MODEL_FEATURES]` values | **CAUGHT** | `test_cluster.py::test_chosen_k_is_five_plus_minus_one`; when `test_features.py` runs alone, `test_scaled_mean_zero_std_one` and `test_scaling_matches_log_then_standardize` also fail |
| (d) generator ignores seed | `np.random.default_rng()` (no seed) | **CAUGHT** | `test_generate_data.py::test_same_seed_gives_identical_data` |

## 3. Test-quality audit

I read all 6 test files (68 tests). None is trivial: there are no `assert True` lines and no tests that only check types. Every threshold matches or is stricter than the spec: ARI ≥ 0.70, silhouette ≥ 0.15, k ∈ {4,5,6}, shares ≥ 3%, seed-stability ARI ≥ 0.80, online_share > 0.65 / < 0.35, runtime < 120 s. Each test name matches what it checks. Strong points:
- the hand-built 3-customer feature example (`test_tiny_hand_checked_values`)
- the ARI test merges the written CSVs on `customer_id`
- tests that reverse the row order in scaling and clustering

## 4. Leakage grep

`grep -nE "ground_truth|true_segment|<5 segment names>" src/segmentation/*.py run_all.py`, excluding `evaluate.py` and `generate_data.py`, gives 2 hits. Both are harmless:
- `run_all.py:31`: `evaluate.compute_ari(segments, evaluate.load_ground_truth(data_dir))` runs only after the labels are final. It produces the score and has no influence on the clustering.
- `src/segmentation/visualize.py:39`: a docstring example (`"0: Omnichannel VIPs"`). The real names come from `name_segments` rules on profile means.

`run_all.py` also unpacks the generator's third return value (`truth`), but only to pass it to `save_data`. It never reaches features or clustering.

## 5. Issues (ranked by severity)

HIGH: none.
MEDIUM: none.

1. **LOW** (`tests/test_pipeline.py`): `test_run_all_script_end_to_end` runs the real script, so every pytest run overwrites the project's `data/` and `outputs/`. This is documented in DECISIONS.md and expected for A12, but it can surprise people.
2. **LOW** (`tests/test_pipeline.py`): the 4 chart functions are only checked for "file exists and is non-empty". Nothing checks that a figure is closed or that a chart reflects the data. That is acceptable for purely visual output.
3. **LOW** (`tests/test_generate_data.py`): `test_dates_within_history_window` allows `END_DATE - 365 days`, one day wider than the strict window. This is documented in DECISIONS.md. The generator stays within 364 days (`days_ago ≤ N_DAYS - 1`), so a strict test would also pass.
4. **LOW** (`tests/test_cluster.py`): A10 is tested with one alternative seed only (7). I also checked seed 123 by hand and got ARI 1.0.
5. **LOW** (`run_all.py`): `print_summary` always prints `config.OUTPUT_DIR`, not the `output_dir` actually used. It is only called from `main()` with the defaults, so the output is correct today.
6. **LOW** (`src/segmentation/visualize.py`): `plot_pca_segments` takes an extra optional `names=None` keyword that the spec does not list. It is backward compatible and documented in DECISIONS.md.
7. **LOW** (`DECISIONS.md`): I could not confirm whether tests were edited after phase 1, because reviewers may not run git. DECISIONS.md and the status files mention no test changes beyond the test_agent's own documented choices.

Readability: the modules are short and clearly commented, with "why" comments throughout. The per-segment recipe table in `generate_data.py` and the named thresholds in `evaluate.py` are especially easy for beginners to follow.
