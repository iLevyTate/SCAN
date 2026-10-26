# Dependency maintenance — 2026-10-25

CrewAI 1.15.30 removes the constraints that blocked patched MCP and uv
releases. The lockfile now resolves MCP 1.30.0 and uv 0.13.5, with compatible
Pillow and filelock updates for each supported Python version.

## Compatibility bounds

CrewAI still requires OpenAI <3 and Pydantic <2.13. Its instructor dependency
requires Rich <15. Dependabot excludes only those incompatible version ranges;
updates within the supported ranges remain enabled. Reevaluate and remove each
exclusion when upgrading CrewAI or instructor to a version that supports it.
The `mypy*` group pattern includes `mypy[faster-cache]` in development updates.

## Unresolved upstream advisories

CrewAI requires `chromadb~=1.5.9`. The dependency audit reports four distinct
advisories (one appears twice in the audit feed), with no listed fixed version:

- GHSA-f4j7-r4q5-qw2c: server code injection through remote model loading.
- GHSA-36p7-vc44-83pf: server code injection during collection updates.
- GHSA-xph7-9rjv-w5fr: authorization scope enforcement.
- GHSA-2wm9-hf6c-p5cr: collection authorization validation.

SCAN does not start a Chroma HTTP server and explicitly sets Crew memory to
false. That limits exposure to the described server endpoints; it does not
remove the vulnerable installed dependency or establish a clean audit. Do not
expose a Chroma server from this environment. Recheck upstream fixes and
CrewAI's dependency requirement before enabling memory or server usage.
No advisory is suppressed in the audit configuration.

## Verification

- `uv lock --check` and `uv sync --locked`
- Ruff lint/format, mypy, and the existing pytest coverage gate
- CI tests Python 3.10, 3.11, and 3.12
- Export the locked production requirements and run `pip-audit` to reassess
  advisories; retain the Python marker-specific versions in the export.
