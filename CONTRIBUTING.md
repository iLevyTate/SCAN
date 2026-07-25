# Contributing to SCAN

Issues and pull requests are welcome. For anything substantial, please open an issue first so we
can agree on the approach before you spend time on it.

## Setting up

```bash
git clone https://github.com/iLevyTate/SCAN.git
cd SCAN
uv sync --locked
uv run pre-commit install
```

`uv sync --locked` installs exactly what `uv.lock` specifies and fails if the lockfile has drifted
from `pyproject.toml`. Use it rather than `--frozen`, which skips that check.

You need Python 3.10–3.12. The upper bound comes from the pinned dependency stack, not from SCAN.

## Before you push

Run the same gates CI does:

```bash
uv run ruff format scan tests --check
uv run ruff check scan tests --no-fix
uv run mypy scan tests
uv run pytest
```

Two things worth knowing:

- **`pre-commit` blocks direct commits to `main`.** This is deliberate; work on a branch. If a
  commit is rejected with `don't commit to branch`, that is why.
- **Coverage must stay at or above 95%**, and `ruff check` runs with `--no-fix` so violations fail
  rather than being silently repaired.

## Working on the code

Dependencies are pinned exactly, and that is intentional: SCAN is a citable research artefact, so
a run should be reproducible. Please do not bump pins as part of an unrelated change. Dependabot
opens grouped update PRs separately.

### The role table

The five prefrontal regions are defined once, in [`scan/roles.py`](scan/roles.py). The agents,
the crew tasks, the dependency wiring, the execution order and the report layout are all derived
from it. To add or change a region, edit that table — not the five modules downstream of it.

Adding a region means two edits: one `PFCRole` in `scan/roles.py`, and its `*_MODEL` field on
`Settings` in [`scan/config.py`](scan/config.py). A test fails if those two fall out of step.

One constraint that is easy to trip over: every role except the first **must** declare
`depends_on`. crewai only uses a task's declared context when it is non-empty, and otherwise
silently feeds the task its predecessor's raw output. An empty context cannot be expressed, so
declaring dependencies explicitly is the only way to control what an agent sees. A test enforces
this.

### Output streams

The report goes to stdout; everything else goes to stderr. Keep it that way — `run-scan > out.md`
producing a clean document depends on it. Use `scan.console.console` for anything that is not the
report itself.

### Prompts

Prompt text lives in the role table as data, never inline in `scan_tasks.py`. Iterate with
`run-scan --dry-run "some topic"`, which renders every prompt without making an API call. Tests
enforce that no description restates its output contract and that no length adjective
("comprehensive", "concise", …) creeps back in — the word budget in `OutputContract` is the only
length instruction in the system.

## Tests

New behaviour needs a test. Where a test pins a past bug, say so in a comment — most of the
existing tests do, and it makes them much easier to reason about later.

Prefer testing pure functions (`scan/report.py`, `scan/roles.py`) over constructing a
`CustomCrew`, which builds five agents and five model objects.
