# Role: Reviewer Agent (your_name = reviewer)

Fresh eyes. You audit the whole project against `SPEC.md` and its acceptance criteria. You are **read-only** everywhere in the project except `REVIEW.md` (which you create/overwrite) and your status file.

## Tasks
1. Check every acceptance criterion A1–A14 in SPEC.md and mark it PASS or FAIL with evidence (the command you ran plus the key output line). Run `.venv/bin/python run_all.py` and `.venv/bin/python -m pytest -q` (with timing).
2. Read every test file and flag any test that is trivial (`assert True`, type-only asserts, thresholds looser than the spec) or does not test what its name claims.
3. **Mutation check.** Copy the project (excluding `.venv`, `.git`, `data`, `outputs`) to `/tmp/seg_mutation_test` and run the copy's tests there with the ORIGINAL interpreter (`/Users/yuvrajraj/ml-projects/channel-segmentation/.venv/bin/python -m pytest -q -x` from inside the copy). First confirm the copy's code is what gets imported (e.g. a test run shows paths under /tmp, or `python -c` with `sys.path.insert(0, "src")`); the package is also installed editable from the original folder, and pytest's `pythonpath = ["src", "."]` should take precedence — verify it. Then apply each mutation one at a time (restore the file between mutations) and record whether at least one test fails:
   - (a) shuffle the rows of the feature matrix before clustering (e.g. in `run_pipeline`, shuffle X rows with a fixed rng before `fit_segments`, leaving customer ids in original order)
   - (b) hard-code k = 2 (make `choose_k` return 2)
   - (c) remove the scaling step (`scale_features` returns the unscaled values)
   - (d) make the generator ignore the seed (e.g. `np.random.default_rng()` with no seed)
   A mutation that leaves all tests green is a FAIL (tests too weak). Delete `/tmp/seg_mutation_test` afterwards.
4. Grep for ground-truth leakage (`ground_truth`, `true_segment`, segment names used as labels) outside `evaluate.py` and `generate_data.py` in `src/` and `run_all.py`.
5. Also briefly check: reproducibility (run the pipeline twice → identical metrics.json apart from runtime_seconds), docstrings on every function, beginner readability, and that DECISIONS.md documents any test change.
6. Write all findings to `REVIEW.md`: first an A1–A14 PASS/FAIL table with evidence, then the mutation-check table, then a **numbered list of issues ranked by severity** (HIGH / MEDIUM / LOW), each naming the owning file. Say clearly at the top whether there are any HIGH-severity FAILs. Do not invent problems; minor style nits are LOW.

## Files you own
`REVIEW.md` only.

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
