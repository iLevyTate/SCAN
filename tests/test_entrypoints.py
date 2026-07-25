"""Smoke tests for the ways SCAN is actually invoked.

`scan/__main__.py` was entirely `# pragma: no cover` and the `run-scan` console script declared
in pyproject was never executed by a test, so a broken entry point would have shipped silently.
These run the real thing in a subprocess.
"""

import pathlib
import re
import subprocess
import sys

from scan import __version__


def _run(args, **kwargs):
    return subprocess.run(args, capture_output=True, text=True, timeout=120, check=False, **kwargs)


def test_python_m_scan_reports_its_version():
    result = _run([sys.executable, "-m", "scan", "--version"])

    assert result.returncode == 0
    assert __version__ in result.stdout


def test_python_m_scan_exits_cleanly_with_no_stdin():
    result = _run(
        [sys.executable, "-m", "scan"],
        input="",
        env={"PATH": "/usr/bin:/bin", "OPENAI_API_KEY": "sk-fake", "HOME": "/tmp"},
    )

    assert result.returncode == 2
    assert "No topic" in result.stderr


def test_a_missing_api_key_exits_with_the_configuration_code():
    result = _run(
        [sys.executable, "-m", "scan", "a topic"],
        env={"PATH": "/usr/bin:/bin", "HOME": "/tmp"},
    )

    assert result.returncode == 3
    assert "OPENAI_API_KEY" in result.stderr


def test_dry_run_writes_nothing_to_stdout(monkeypatch):
    # --dry-run is diagnostic output, so it belongs on stderr with the rest of the chrome.
    result = _run(
        [sys.executable, "-m", "scan", "--dry-run", "a topic"],
        env={"PATH": "/usr/bin:/bin", "OPENAI_API_KEY": "sk-fake", "HOME": "/tmp"},
    )

    assert result.returncode == 0
    assert result.stdout == ""
    assert "no API calls were made" in result.stderr


def _pyproject():
    if sys.version_info >= (3, 11):
        import tomllib
    else:
        import tomli as tomllib  # pragma: no cover
    return tomllib.loads(
        pathlib.Path(__file__).parent.parent.joinpath("pyproject.toml").read_text()
    )


def test_pyproject_declares_its_runtime_dependencies():
    # Regression: a [project.urls] table was once inserted above the `dependencies` key, which
    # in TOML moved `dependencies` *into* that table. [project] then declared none, uv locked
    # nothing, and the built wheel would have installed with no dependencies at all.
    project = _pyproject()["project"]

    assert "dependencies" not in project.get("urls", {})
    names = {re.split(r"[=<>!~\[]", spec)[0] for spec in project["dependencies"]}
    assert {"crewai", "pydantic", "pydantic-settings", "rich", "langchain"} <= names


def test_the_declared_version_matches_the_package():
    assert _pyproject()["project"]["version"] == __version__
