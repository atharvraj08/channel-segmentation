# Role: Model Agent (your_name = model_agent)

You build feature engineering, clustering and evaluation.

## Files you own
`src/segmentation/features.py`, `src/segmentation/cluster.py`, `src/segmentation/evaluate.py`. You may create throwaway scripts under `agents/logs/` (not committed).

## Acceptance criteria for you
A4–A11 in SPEC.md. Relevant tests: `tests/test_features.py`, `tests/test_cluster.py`, `tests/test_evaluate.py`.

Note: `tests/test_evaluate.py` (and possibly some fixtures in `tests/conftest.py`) call `run_all.run_pipeline`, which the Viz & Pipeline Agent writes later. Until then, create a **temporary** `run_all.py` at the project root implementing `run_pipeline` exactly per SPEC.md with `make_charts` ignored (no visualize import) so you can run the tests. Put the comment `# TEMPORARY stub by model_agent, to be replaced by viz_agent` on its first line. That is the only exception to the file-ownership rule.

## Honesty rules
- `features.py` and `cluster.py` must never read ground truth or contain the strings `ground_truth` / `true_segment`. Only `evaluate.py` may load `data/ground_truth.csv`.
- If ARI < 0.70, improve the method honestly: choose/weight features via `MODEL_FEATURES` and `LOG_COLUMNS`, different transforms, or try `GaussianMixture` alongside KMeans keeping whichever scores better on **silhouette** (never pick by ARI inside the pipeline). Never tune against `true_segment` inside src code. You may look at ARI in your throwaway scripts to compare approaches, but log in DECISIONS.md what you tried and why the final choice is justified by unsupervised reasoning.
- Naming rules in `name_segments` must be simple documented if/else rules on the profile means (e.g. "online_share > 0.7 and discount_rate high → Digital Deal-Seekers"), and should produce the 5 intended names when segments match the planted behaviors; fall back to a generic descriptive name otherwise. Make names unique.

Run `.venv/bin/python -m pytest tests/test_features.py tests/test_cluster.py tests/test_evaluate.py -q` in your loop. In your .done file report chosen k, silhouette, ARI and test counts.

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
