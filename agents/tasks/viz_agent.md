# Role: Viz & Pipeline Agent (your_name = viz_agent)

You build the charts and the end-to-end runner.

## Files you own
`src/segmentation/visualize.py`, `run_all.py` (replace the temporary stub entirely, keeping `run_pipeline`'s behavior identical to SPEC.md). Read-only: everything else.

## Acceptance criteria for you
A12, A13 in SPEC.md. Relevant tests: `tests/test_pipeline.py`, then the whole suite `.venv/bin/python -m pytest -q` (must pass in < 60 s).

## Details
- Charts per SPEC.md: `silhouette_by_k.png` (line+markers, chosen k highlighted), `pca_segments.png` (2-D PCA scatter coloured by predicted segment, legend with segment names if available, small alpha markers), `segment_profiles.png` (heatmap of per-segment means standardized across segments, annotated, readable feature labels), `channel_mix.png` (stacked horizontal bar of online vs store spend share per segment, labelled with segment names). Use `matplotlib.use("Agg")`, a tight layout, dpi ~120, and close every figure.
- `run_all.py`: put `src/` on `sys.path`, implement `run_pipeline` and `main` per SPEC.md, time the run and store `runtime_seconds` in the metrics, and print a short readable summary (k, silhouette, ARI, segment names + sizes, where outputs are). Exit code 0.
- Run `.venv/bin/python run_all.py` yourself, check it finishes well under 120 s, and look at one or two of the PNGs (you can read images) to make sure they look right.
- If a non-owned module is buggy, do not edit it: describe the problem in your .failed/.done file.

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
