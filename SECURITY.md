# Security Policy

## Supported versions

SCAN is a research project. Only the latest commit on `main` is supported.

## Reporting a vulnerability

Please report vulnerabilities privately via
[GitHub's private vulnerability reporting](https://github.com/iLevyTate/SCAN/security/advisories/new)
rather than opening a public issue.

Please do not include real API keys in a report. A redacted example is enough.

## Scope

SCAN is a command-line tool that runs locally and talks to the OpenAI API (and, if configured,
SerpAPI). It exposes no network service and stores no credentials of its own.

Worth knowing when assessing a report:

- **Credentials** are read from the environment or a local `.env`, which is gitignored. They are
  passed to the model client and are not logged. `.env` is never read from anywhere but the
  working directory.
- **Model output is untrusted input.** It is rendered with Rich markup disabled and written to
  stdout without interpretation. Reports are not executed, and no output is ever passed to a
  shell.
- **Dependencies are pinned exactly** for reproducibility, which means the tree can lag upstream
  security releases. Dependabot is configured for both the `uv` and `github-actions` ecosystems.
  If you are reporting a vulnerability in a transitive dependency, please say which package and
  version, since a pin bump also requires regenerating `uv.lock`.
- **Telemetry** is disabled by default (`DISABLE_TELEMETRY=true`). With it enabled, crewai sends
  anonymous traces to `telemetry.crewai.com`.
