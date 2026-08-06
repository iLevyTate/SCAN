# Changelog

All notable changes to SCAN are recorded here. This project follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Each released version is archived on Zenodo under the concept DOI
[10.5281/zenodo.14052884](https://doi.org/10.5281/zenodo.14052884), which always resolves to the
most recent release.

## [1.0.0] - 2026-08-06

First stable release. Supersedes `1.0.0-alpha`, the version archived as
[10.5281/zenodo.14052885](https://doi.org/10.5281/zenodo.14052885) and cited by the SCAN paper.

The headline change is that the five prefrontal-cortex agents now actually execute their own
tasks. In `1.0.0-alpha` they did not.

### Fixed

- **The five PFC agents were bypassed at runtime.** The crew ran under `Process.hierarchical`,
  where crewai routes every task to an auto-created generic "Crew Manager" agent and ignores the
  agent assigned to each task. All five report sections were produced by that one manager, the
  per-region `*_MODEL` settings had no effect, and the search tool never reached the executing
  agent. The crew now runs `Process.sequential`, so each task executes on its own agent with its
  own model and tools.
- **Agents received each other's output without being told.** crewai uses a task's declared
  context only when non-empty, and otherwise silently substitutes the preceding task's raw output
  with no attribution. Every role after the first now declares its inputs explicitly, and each
  prompt carries a generated legend naming them. The integrating agent (DLPFC) now receives the
  conflict analysis it is meant to integrate, which it never previously saw.
- **Report content was silently altered.** Rich markup parsing deleted any bracketed span from
  model output, and an unmatched `[/]` raised an error from inside the error handlers themselves,
  losing a completed run. Rich's `soft_wrap` also truncated long block content. The report now
  bypasses Rich entirely.
- **Failed runs discarded completed work.** A failure on the final analysis threw away the four
  that had already finished. Completed analyses are now emitted as a partial report.
- **Every failure path exited 0**, so scripts could not detect a failed run.
- **A single unrelated key in `.env` crashed the application and the test suite** at import time.
- `SERPER_API_KEY` fed a SerpAPI client; renamed to `SERPAPI_API_KEY` with the old name accepted
  as an alias.
- No request timeout ever reached the model client, so a hung connection blocked a run
  indefinitely. Added `REQUEST_TIMEOUT` (default 180s).
- `ruff check` ran with `fix = true` in CI, repairing violations and exiting 0, so the lint gate
  could never fail.
- `uv sync --frozen` in CI skipped the lockfile consistency check; now `--locked`.

### Added

- A command-line interface: `--output`, `--model` (globally or per region), `--max-tokens`,
  `--timeout`, `--no-search`, `--log-level`, `--dry-run`, `--version`. Bare `run-scan` still
  prompts interactively, unchanged.
- Report on stdout, all progress and diagnostics on stderr, so `run-scan "..." > report.md`
  produces a clean document.
- Per-analysis progress reporting during a run.
- An exception hierarchy (`ScanError` → `ConfigurationError` / `ProviderError`) with distinct exit
  codes, and translation of provider errors into actionable messages.
- `scan/roles.py`, a single source of truth defining each region's persona, task, output contract
  and dependencies.
- `CONTRIBUTING.md`, `SECURITY.md`, `CITATION.cff`, and `py.typed`.

### Changed

- **Prompts are now region-specific.** All five previously shared one template and differed only
  in nouns, so they produced interchangeable output. Each region is now asked for the kind of
  reasoning it models, with an explicit output contract and word budget.
- **The report leads with the recommendation.** DLPFC's synthesis was previously rendered as a
  peer section below the others.
- Default model changed from `gpt-4` to `gpt-4o`; crewai's function-calling capability check
  returns false for `gpt-4`, which disables its structured-output paths.
- Tests 26 → 195; coverage 84% → 98%, with branch coverage and a 95% floor enforced in CI.
- `mypy` raised to `strict`.

### Removed

- `MANAGER_MODEL`, which only configured the manager agent that `Process.sequential` no longer
  creates. Unknown keys in `.env` are now ignored, so existing files keep working.
- The GitHub Pages deployment workflow, which published the repository root as a site with no
  `index.html`.

### Dependencies

Unchanged. crewai remains pinned at 0.76.9 and `uv.lock` is untouched, so this release is
directly comparable with the archived `1.0.0-alpha`.

[1.0.0]: https://github.com/iLevyTate/SCAN/releases/tag/v1.0.0
