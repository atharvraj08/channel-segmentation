# Role: Test Agent (your_name = test_agent)

You write the full pytest suite for this project **test-driven, before the implementation exists**. Use only the public names defined in `SPEC.md`. The implementation modules may not exist yet while you work, so you cannot run the tests to green; instead make sure they are collected without syntax errors (`.venv/bin/python -m pytest --collect-only -q` will show import errors for missing modules; that is expected) and are logically correct against SPEC.md.

## Files you own
`tests/conftest.py`, `tests/test_generate_data.py`, `tests/test_features.py`, `tests/test_cluster.py`, `tests/test_evaluate.py`, `tests/test_pipeline.py`.

## What to test (acceptance criteria A1–A14 in SPEC.md)
- `test_generate_data.py`: A1 (≈5000 customers via config.N_CUSTOMERS, exact columns, no NaN, amount > 0, rates in [0,1], channels/categories valid, dates within the 365 days ending END_DATE, one engagement row per customer, every customer has ≥1 transaction, 5 true segments each ≥ 8%), A2 (same seed → identical frames, `pd.testing.assert_frame_equal`; different seed → different data), A3 (no `true_segment` column and no segment-name strings in transactions/engagement), `save_data` writes the 3 CSVs.
- `test_features.py`: A4 (one row per customer, index = customer ids, columns == FEATURE_COLUMNS, no NaN/inf, shares/rates in [0,1], scaled X shape (n, len(MODEL_FEATURES)), column means ≈ 0 (abs < 1e-6) and std ≈ 1 (abs diff < 1e-3)); edge cases using tiny hand-built frames (customer with only online orders → online_share 1 and store_orders 0; customer with only store orders; no NaN); hand-checked values for total_spend/order_count/discount_rate on a tiny example; A5 (read the source text of features.py and cluster.py, assert `ground_truth` and `true_segment` are absent); also assert raw unscaled features are NOT already standardized (e.g. total_spend mean far from 0) so that removing scaling is detectable.
- `test_cluster.py`: A6 (choose_k on the real scaled features returns k in {4,5,6} and a score for every k in 2..8), A8 (silhouette ≥ 0.15), A9 (every cluster ≥ 3% of customers), A10 (fit_segments with seed 42 vs seed 7 → `adjusted_rand_score` ≥ 0.80), labels are ints in 0..k-1, deterministic for a fixed seed, save_segments writes the CSV with the right columns.
- `test_evaluate.py`: A7 (**ARI ≥ 0.70**, computed from the output of `run_all.run_pipeline` run into a temporary directory, so a bug anywhere in the pipeline's row alignment is caught), A11 (some profile online_share > 0.65 and some < 0.35), profiles have size/share/name columns and sizes sum to n, names non-empty, metrics.json has every key from SPEC.md with sensible values, compute_ari sanity (identical labelings → 1.0, permuted label ids → 1.0).
- `test_pipeline.py`: A12: run `.venv/bin/python run_all.py` (use `sys.executable`) as a subprocess from the project root with a 120 s timeout; assert exit code 0, wall time < 120 s, and every output file in SPEC.md exists and is non-empty; metrics.json chosen_k matches the number of distinct segments in customer_segments.csv. Also: running `run_pipeline` twice with the same seed gives identical metrics (reproducibility).

## Speed (A13: whole suite < 60 s)
The real pipeline is the expensive part. In `tests/conftest.py` put **session-scoped** fixtures that compute things once and are shared: generated data (in memory), the feature frame, scaled X, the `choose_k` result, and one `run_all.run_pipeline(data_dir=tmp, output_dir=tmp)` run (via `tmp_path_factory`). Do not call `choose_k` or the full pipeline more than once per session except where the test genuinely needs it (the A12 subprocess run and the reproducibility check — for that one you may pass `make_charts=False`).

## Test quality (A14)
A reviewer will mutate the code and expects at least one test to fail for each of: (a) shuffling the rows of the feature matrix before clustering, (b) hard-coding k = 2, (c) removing the scaling step, (d) the generator ignoring the seed. Make sure your tests would catch each one. Never write `assert True`, type-only assertions, `skip` or `xfail`. Thresholds must equal the spec's.

---
## Rules that apply to every worker

- The human is asleep. **Never ask questions.** If something is ambiguous, pick the simplest reasonable option, append a one-line entry to `DECISIONS.md` prefixed with your agent name, and keep going.
- **Read `SPEC.md` first. Follow the exact function signatures, column names and file names.** Do not read `OVERNIGHT_PROMPT.md`.
- Stay inside this project folder. Use only `.venv/bin/python` and `.venv/bin/pip`. Do not install new packages. Do not run git commands (the orchestrator commits).
- Only create or edit the files listed under "Files you own" (plus appending to `DECISIONS.md` and writing your status file). Do not touch any other file. Never edit `tests/` unless you are the Test Agent.
- Keep code beginner-readable: short functions, clear names, a docstring on every function, comments explaining *why*.
- Be economical: do not dump large files or long logs into your context; use `-q`, `tail`, `head`.
- Report milestones with: `cmux log --source <your_name> -- "<message>"` (if cmux fails, ignore it).

### Self-check loop
1. Write the code.
2. Run the relevant tests with `.venv/bin/python -m pytest <files> -q`.
3. Read failures carefully. Fix the code, not the tests.
4. Repeat until passing (up to about 15 iterations).
5. Re-read your own code once for clarity and comments.
6. Run the tests one final time.

### Finishing
- On success: write `agents/status/<your_name>.done` with a 5-line summary (what was built, test results, any concerns).
- On failure after real effort: write `agents/status/<your_name>.failed` explaining exactly what is blocking. Then exit.
