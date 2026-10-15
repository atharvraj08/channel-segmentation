# Overnight Build: Omnichannel Customer Segmentation (ML mini-project)

You are the ORCHESTRATOR. The human is asleep and cannot answer questions or approve anything until morning. Work fully autonomously until the project is complete and verified. Never stop to ask a question; if something is ambiguous, pick the simplest reasonable option, write the decision in `DECISIONS.md`, and keep going.

You are running inside cmux. Use the cmux CLI to split panes, launch worker agents, and report progress in the sidebar (commands are listed in section 7).

---

## 1. The goal

Build a small, clean, well-tested Python project that segments retail customers by how they shop across **digital** (website/app) and **in-store** channels, using purchase and engagement data.

Because no real customer data is available, the project generates a **synthetic dataset with hidden "true" segments planted in it**. The model never sees the true labels. They are used only to *grade* the model. That makes it possible to test objectively whether the ML actually works.

Keep it simple. Use numpy, pandas, scikit-learn, matplotlib, and pytest. No PyTorch, no deep learning, no Jupyter, no web apps.

## 2. Ground rules

1. **Stay inside the project folder.** It is the current working directory. Never read, modify, or delete anything outside it, except for installing Python packages into the project's own virtual environment (`.venv`). Never use `sudo`. Never touch `~/.claude`, `~/.ssh`, or other dotfiles.
2. **Use a virtual environment.** Run `python3 -m venv .venv`, then use `.venv/bin/python` and `.venv/bin/pip` for everything. Pin installed versions in `requirements.txt`.
3. **Use git, on a single `main` branch.** Run `git init` if needed. Commit after every phase with a clear message, like `phase 2: features + clustering, 18/18 tests passing`. Never force-push and never rewrite history.
4. **Never weaken a test to make it pass.** Only the Test Agent and the Orchestrator may edit files in `tests/`. If a test is genuinely wrong, the person changing it must explain why in `DECISIONS.md`. Deleting a test, loosening a threshold without justification, or adding `skip`/`xfail` to dodge a failure is forbidden.
5. **Everything must be reproducible.** All randomness uses a fixed seed (default `42`). Running the pipeline twice must give identical results.
6. **Keep the code beginner-readable.** The human is a first-year Stat/ML student who knows basic Python. Use short functions, clear names, a docstring on every function, and comments explaining *why*, not just *what*.
7. **Be time-efficient.** The full pipeline should run in under 2 minutes on a MacBook Air M4, and the full test suite in under 60 seconds.

## 3. Project spec (the contract every agent follows)

### Folder layout

```
channel-segmentation/
├── OVERNIGHT_PROMPT.md       (this file, do not edit)
├── SPEC.md                   (Orchestrator writes from this section)
├── DECISIONS.md              (log of every judgment call)
├── PROGRESS.md               (live status, updated by the Orchestrator)
├── requirements.txt
├── run_all.py                (one command runs the whole pipeline)
├── src/segmentation/
│   ├── __init__.py
│   ├── config.py             (seed, n_customers, paths, thresholds)
│   ├── generate_data.py      (synthetic data with planted segments)
│   ├── features.py           (feature engineering + scaling)
│   ├── cluster.py            (choose k, fit KMeans, assign segments)
│   ├── evaluate.py           (metrics + segment profiles + naming)
│   └── visualize.py          (charts)
├── tests/
│   ├── test_generate_data.py
│   ├── test_features.py
│   ├── test_cluster.py
│   ├── test_evaluate.py
│   └── test_pipeline.py      (end-to-end)
├── data/                     (generated CSVs; git-ignore the large ones)
├── outputs/                  (metrics.json, segment_profiles.csv, charts/*.png)
├── agents/                   (task files for workers, logs, status flags)
│   ├── tasks/  logs/  status/
├── REVIEW.md                 (written by the Reviewer Agent)
├── REPORT.md                 (results in plain English)
└── LEARNING_GUIDE.md         (walkthrough for the human to study in the morning)
```

### Data, in `generate_data.py`

- Create about 5,000 customers in two tables:
  - `data/transactions.csv`: one row per purchase, with `customer_id`, `date` (over the past 12 months), `channel` (`online` or `store`), `amount`, `category` (e.g. electronics, apparel, grocery, home, beauty), and `used_discount` (bool).
  - `data/engagement.csv`: one row per customer, with `customer_id`, `email_open_rate` (0–1), `app_sessions_per_month`, `web_sessions_per_month`, `loyalty_member` (bool), and `days_since_last_visit`.
- Plant **5 hidden segments** with distinct, realistic behavior. For example:
  1. *Digital Deal-Seekers*: mostly online, frequent small orders, heavy discount use, high email opens.
  2. *In-Store Loyalists*: mostly store, loyalty members, steady medium baskets, low app use.
  3. *Omnichannel VIPs*: both channels, high spend, high engagement everywhere.
  4. *Occasional Browsers*: many web sessions, few purchases, long gaps between visits.
  5. *Convenience Grocery Shoppers*: store-heavy, grocery-dominant, frequent small trips.
- Segment sizes should be unequal but none smaller than 8%. Add realistic noise so the segments overlap somewhat; the problem should not be trivially easy.
- Save the true labels **separately**, in `data/ground_truth.csv` (`customer_id`, `true_segment`). They must never appear in the transactions or engagement files.
- Everything is driven by `config.SEED`.

### Features, in `features.py`

- Build one row per customer, including at least: total spend, order count, average basket size, **online share of spend** (0–1), online and store order counts, discount rate, recency, number of distinct categories, top-category share, grocery share, email open rate, app and web sessions, and loyalty flag.
- Handle edge cases: customers with zero orders in a channel, and division by zero. The result must contain no NaN or inf.
- Apply log transforms to skewed money and count features, then use `StandardScaler`.
- **Never** read `ground_truth.csv`.

### Clustering, in `cluster.py`

- Try k = 2 to 8 with KMeans (`n_init=10`, fixed seed). Choose k using the silhouette score. Record every k's score in `outputs/metrics.json`.
- Fit the final model and save each customer's assigned segment to `outputs/customer_segments.csv`.
- **Never** read `ground_truth.csv`.

### Evaluation, in `evaluate.py`

- Report the silhouette score of the chosen model.
- **Validation against hidden truth**, the only place `ground_truth.csv` may be read: compute the Adjusted Rand Index (ARI) between the predicted segments and the true segments.
- Build a segment profile table: the mean of each key feature per segment plus each segment's size. Save it as `outputs/segment_profiles.csv`.
- Give each segment a human-readable name using simple, documented rules on the profile, like "online share > 0.7 and discount rate high → Digital Deal-Seekers."
- Save all metrics to `outputs/metrics.json`.

### Charts, in `visualize.py` (saved to `outputs/charts/`)

- `silhouette_by_k.png`
- `pca_segments.png`: a 2D PCA scatter colored by predicted segment
- `segment_profiles.png`: a heatmap or grouped bar chart of standardized feature means per segment
- `channel_mix.png`: online vs. store spend share per segment

### `run_all.py`

A single command, `.venv/bin/python run_all.py`, runs generate → features → cluster → evaluate → visualize, prints a short summary, and exits with code 0.

## 4. Acceptance criteria (the project is NOT done until all of these pass)

| # | Requirement | How it is checked |
|---|---|---|
| A1 | Data has about 5,000 customers, required columns, no missing values, sensible ranges (amounts > 0, rates in [0, 1]) | `test_generate_data.py` |
| A2 | Running the generator twice with the same seed gives identical files | test |
| A3 | Ground-truth labels are absent from transactions and engagement files | test |
| A4 | Feature matrix has one row per customer, no NaN/inf, and scaled columns have mean ≈ 0 and std ≈ 1 | `test_features.py` |
| A5 | `features.py` and `cluster.py` never reference `ground_truth` or `true_segment` | test that reads those source files and checks for the strings |
| A6 | Chosen k is 5 ± 1 | `test_cluster.py` |
| A7 | **ARI vs. hidden truth ≥ 0.70** (the core "the ML actually works" check) | `test_evaluate.py` |
| A8 | Silhouette of chosen model ≥ 0.15 | test |
| A9 | Every predicted segment holds ≥ 3% of customers | test |
| A10 | Stability: re-running clustering with a different KMeans seed gives ARI ≥ 0.80 against the first run | test |
| A11 | At least one predicted segment has online share > 0.65 and at least one has online share < 0.35 (it captures the channel split) | test |
| A12 | `run_all.py` completes in under 120 s with exit code 0 and creates every file in `outputs/` | `test_pipeline.py` |
| A13 | Full test suite passes, in under 60 s total | `pytest -q` |
| A14 | Tests are meaningful, not trivially passing | Reviewer's mutation check (section 6, phase 3) |

If A7 fails, the fix is to improve feature engineering or the clustering approach, for example by adding or removing features, transforms, or weighting, or by trying `GaussianMixture` alongside KMeans and keeping whichever scores better on silhouette. Never fix it by changing the test or by leaking ground truth. If the data generator makes segments so overlapping that no honest method passes, the Orchestrator may adjust the generator's noise level and must document that in `DECISIONS.md`.

## 5. Team structure

You (Orchestrator) stay in the original pane. Each worker is a separate headless Claude Code process launched in its own cmux split. Each worker gets a task file, and each one owns specific files and must not edit anyone else's.

| Agent | Owns these files | Job |
|---|---|---|
| **Test Agent** | `tests/*` | Writes the full test suite from SPEC.md and the acceptance criteria *before* seeing the implementation (test-driven). Uses only the public function names defined in SPEC.md. |
| **Data Agent** | `config.py`, `generate_data.py`, `requirements.txt` | Builds the generator and gets `test_generate_data.py` passing. |
| **Model Agent** | `features.py`, `cluster.py`, `evaluate.py` | Builds features, clustering, and evaluation. Iterates until the A4–A11 tests pass honestly. |
| **Viz & Pipeline Agent** | `visualize.py`, `run_all.py` | Builds the charts and the end-to-end runner. Gets `test_pipeline.py` passing. |
| **Reviewer Agent** | `REVIEW.md` only (read-only everywhere else) | Fresh eyes. Audits everything against the spec, runs a mutation check, and lists problems. |
| **Docs Agent** | `REPORT.md`, `LEARNING_GUIDE.md` | Writes the human-friendly explanations at the end. |

Before launching anyone, write `SPEC.md`. It must pin down the **exact public function signatures** for every module (names, arguments, and return types) so the Test Agent and the builders agree without talking to each other. For example: `generate_data(n_customers: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]`, `build_features(transactions, engagement) -> pd.DataFrame`, `scale_features(df) -> tuple[np.ndarray, StandardScaler]`, `choose_k(X, k_range, seed) -> tuple[int, dict]`, `fit_segments(X, k, seed) -> np.ndarray`, and so on.

## 6. Phases

Update `PROGRESS.md`, the cmux sidebar status, and the progress bar at every step.

**Phase 0, setup (you, about 10 min):**
1. Run `cmux identify --json` and record your own surface ID.
2. Check that `python3 --version` is 3.10 or higher. Create `.venv`, install `numpy pandas scikit-learn matplotlib pytest`, and write `requirements.txt`.
3. Create the folder layout. Write `SPEC.md`, the `.gitignore`, and the launcher script below. Then `git init` and commit.
4. Create `agents/run_agent.sh`:
   ```bash
   #!/bin/bash
   # Usage: bash agents/run_agent.sh <agent_name>
   NAME="$1"
   cd "$(dirname "$0")/.." || exit 1
   echo "=== $NAME started $(date) ===" | tee "agents/logs/$NAME.log"
   claude -p "$(cat agents/tasks/$NAME.md)" --dangerously-skip-permissions 2>&1 | tee -a "agents/logs/$NAME.log"
   echo "=== $NAME exited code ${PIPESTATUS[0]} at $(date) ===" | tee -a "agents/logs/$NAME.log"
   touch "agents/status/$NAME.finished"
   ```
5. Smoke-test it with a trivial agent task, like a task that writes `agents/status/smoke.done` and does nothing else. This confirms headless workers work before the real run. If `claude -p` fails, read the log, fix the cause, and retry. If it cannot be made to work, fall back to doing the workers' jobs yourself, sequentially, following the same rules.

**Phase 1, tests and data in parallel:**
- Write `agents/tasks/test_agent.md` and `agents/tasks/data_agent.md`, using the worker task template below.
- Launch each in its own split:
  `cmux new-split right --command "bash agents/run_agent.sh test_agent"`
  `cmux new-split down --command "bash agents/run_agent.sh data_agent"`
- Wait for both (see "Waiting," below). Then run `pytest tests/test_generate_data.py -q` yourself to confirm. Commit.

**Phase 2, model, then viz and pipeline:**
- Launch the Model Agent. When it finishes, run its tests yourself and commit.
- Launch the Viz & Pipeline Agent. When it finishes, run `.venv/bin/python run_all.py` and the full `pytest -q` yourself. Commit.

**Phase 3, independent review (the self-checking loop):**
- Launch the Reviewer Agent. Its task file must instruct it to:
  1. Check every acceptance criterion A1–A14 and mark it PASS or FAIL with evidence (command plus output).
  2. Read every test and flag any that are trivial (`assert True`, asserting only types, thresholds far looser than the spec) or that don't actually test what their names claim.
  3. **Mutation check.** In a temporary copy of the project (`/tmp/seg_mutation_test`, deleted afterward), deliberately break things one at a time and confirm at least one test fails each time:
     - (a) shuffle the rows of the feature matrix before clustering
     - (b) hard-code k = 2
     - (c) remove the scaling step
     - (d) make the generator ignore the seed

     Any mutation that leaves all tests green means the tests are too weak. Report that as a FAIL.
  4. Grep for ground-truth leakage outside `evaluate.py`.
  5. Write all findings to `REVIEW.md` as a numbered list, ranked by severity.
- **Fix loop.** For each FAIL in `REVIEW.md`, write a focused fix task for the owning agent and relaunch that agent. Test weaknesses go to the Test Agent. Then re-run the Reviewer. Repeat up to **3 rounds**. If issues remain after 3 rounds, document them honestly in `REPORT.md` under "Known issues." Do not hide them.

**Phase 4, docs:**
- Launch the Docs Agent with the final results. It writes:
  - `REPORT.md`: what was built, the final metrics (k, silhouette, ARI), a table of segment names and sizes with what makes each distinct, the charts embedded with relative image links, and known limitations (synthetic data, the ARI only being possible because the truth is known, etc.).
  - `LEARNING_GUIDE.md`, written for a first-year student who will study this over breakfast:
    1. The big picture in 5 sentences.
    2. How to run it, with exact commands.
    3. A file-by-file walkthrough in pipeline order: what each file does and the key lines to read.
    4. Each ML concept explained simply, with where it appears in the code: feature engineering, log transforms, standardization, KMeans, silhouette score, choosing k, PCA, ARI, and why synthetic ground truth lets us grade an unsupervised model.
    5. How the tests work, and what the mutation check proved.
    6. Five "try this next" exercises, such as changing the number of planted segments, adding a feature, or swapping in GaussianMixture.
    7. A glossary.

**Phase 5, final verification (you):**
1. Start from a clean state: delete `data/` and `outputs/`, then run `.venv/bin/python run_all.py` and `.venv/bin/python -m pytest -q`. Both must succeed.
2. Confirm that every file in the folder layout exists.
3. Confirm the final `REVIEW.md` shows no unresolved high-severity FAILs, or that they are disclosed in `REPORT.md`.
4. Make the final commit: `final: all acceptance criteria verified`.
5. Close the worker panes. Leave yours open.
6. Run `cmux markdown open LEARNING_GUIDE.md` so it's on screen in the morning.
7. Run `cmux notify --title "Overnight build finished" --body "<k, ARI, tests passed>"` and `cmux set-progress 1.0 --label "Done"`.
8. Write a 10-line morning summary at the top of `PROGRESS.md`: what got done, the final metrics, what (if anything) is still broken, and where to start reading.

### Worker task template

Every `agents/tasks/<name>.md` must include:
- A role statement and the human-asleep rule: never ask questions, decide and log the decision in `DECISIONS.md`.
- "Read `SPEC.md` first. Follow the exact function signatures."
- The exact list of files this agent may create or edit, and the instruction that it must not touch any others.
- The acceptance criteria relevant to this agent.
- **The self-check loop:**
  1. Write the code.
  2. Run the relevant tests with `.venv/bin/python -m pytest <files> -q`.
  3. Read failures carefully. Fix the code, not the tests.
  4. Repeat until passing (up to about 15 iterations).
  5. Re-read your own code once for clarity and comments.
  6. Run the tests one final time.
- Progress reporting through `cmux log --source <name> "<message>"` at each milestone.
- On success: write `agents/status/<name>.done` containing a 5-line summary (what was built, test results, any concerns). On failure after real effort: write `agents/status/<name>.failed` explaining exactly what is blocking, then exit.

### Waiting for workers

Poll in bounded chunks so no single shell command runs longer than about 9 minutes, for example:
```bash
for i in $(seq 1 50); do [ -f agents/status/data_agent.finished ] && break; sleep 10; done; ls agents/status/
```
Repeat as needed. While waiting, check `tail -20 agents/logs/<name>.log` occasionally, but do not read entire logs (keep your context small). Treat `.finished` without `.done` as a crash or failure: read the end of the log, rewrite the task file with what went wrong, and relaunch, at most 2 retries per agent. If a worker shows no new log output for 30 minutes, assume it is stuck: close its pane and relaunch it. Note that `claude -p` only prints at the end, so judge activity by file changes (`git status`, file timestamps) as well.

## 7. cmux commands you will use

```bash
cmux identify --json                                   # who/where am I
cmux new-split right --command "bash agents/run_agent.sh NAME"
cmux new-split down  --command "bash agents/run_agent.sh NAME"
cmux list-panels --json                                # find surface IDs
cmux set-status phase "Phase 2: model" --icon hammer --priority 90
cmux set-progress 0.4 --label "Model agent running"
cmux log --level success --source orchestrator -- "Phase 1 done: 12/12 tests"
cmux log --level error --source orchestrator -- "Model agent failed, retrying"
cmux notify --title "Title" --body "Body"              # only at the very end, or if completely blocked
cmux markdown open LEARNING_GUIDE.md
```

Close finished worker panes when they're done (use the cmux skill for the exact close command) so the layout stays tidy. If any cmux command fails, for example a socket error, the build must not stop. cmux is only for visibility. Carry on and log the problem in `DECISIONS.md`.

## 8. If things go wrong

- **Package install fails:** retry once. Then try installing without version pins. Log it.
- **A test seems impossible to pass honestly:** re-read the spec, try at least 3 genuinely different approaches, and if it's still failing, document it as a known issue and move on. Never cheat.
- **You hit a usage or rate limit:** wait in 5-minute chunks (sleep plus retry) for up to 1 hour, then continue.
- **Something is truly unrecoverable:** commit whatever works, write an honest status in `PROGRESS.md`, send one `cmux notify` describing it, and stop.

Start now with Phase 0.
