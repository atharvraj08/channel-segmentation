
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
