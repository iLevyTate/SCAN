import pytest

from scan import __version__, cli
from scan.config import settings
from scan.roles import ROLES

TOPIC = "adopting a rescue dog"


def _parse(argv):
    parser = cli.build_parser()
    return parser, parser.parse_args(argv)


def _overrides(argv):
    parser, args = _parse(argv)
    return cli.settings_overrides(args, parser)


class StubCrew:
    """Stands in for CustomCrew so no model is ever called."""

    report = "## the report"

    def __init__(self, topic, settings=None):
        self.topic = topic
        self.settings = settings

    def run(self):
        return self.report


@pytest.fixture
def stub_crew(monkeypatch):
    monkeypatch.setattr(cli, "CustomCrew", StubCrew)
    return StubCrew


def test_bare_invocation_still_prompts_interactively(monkeypatch, stub_crew, capsys):
    # The one hard compatibility constraint: `run-scan` with no arguments must behave exactly
    # as it always has.
    calls = []

    def record_prompt():
        calls.append(True)
        return TOPIC

    monkeypatch.setattr("builtins.input", record_prompt)

    cli.main([])

    assert calls, "bare run-scan did not prompt for a topic"
    assert capsys.readouterr().out == "## the report\n"


def test_positional_topic_skips_the_prompt(monkeypatch, stub_crew, capsys):
    def fail():
        raise AssertionError("should not have prompted")

    monkeypatch.setattr("builtins.input", fail)

    cli.main([TOPIC])

    captured = capsys.readouterr()
    assert captured.out == "## the report\n"
    assert f"You entered: {TOPIC}" in captured.err


def test_a_blank_positional_topic_is_rejected(stub_crew, capsys):
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["   "])

    assert excinfo.value.code == cli.EXIT_NO_TOPIC
    assert "No topic was provided" in capsys.readouterr().err


def test_output_writes_to_a_file_and_keeps_stdout_clean(tmp_path, stub_crew, capsys):
    destination = tmp_path / "report.md"

    cli.main([TOPIC, "--output", str(destination)])

    assert destination.read_text(encoding="utf-8") == "## the report\n"
    captured = capsys.readouterr()
    assert captured.out == ""
    assert str(destination) in captured.err


def test_version_reports_the_package_version(capsys):
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--version"])

    assert excinfo.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_help_lists_the_flags(capsys):
    with pytest.raises(SystemExit):
        cli.main(["--help"])

    out = capsys.readouterr().out
    for flag in ("--output", "--model", "--timeout", "--dry-run", "--no-search"):
        assert flag in out


def test_model_sets_every_region():
    assert _overrides(["--model", "gpt-4o-mini"]) == {
        role.model_setting: "gpt-4o-mini" for role in ROLES
    }


def test_model_can_target_one_region():
    assert _overrides(["--model", "DLPFC=gpt-4o"]) == {"DLPFC_MODEL": "gpt-4o"}


def test_model_is_case_insensitive_for_the_region():
    assert _overrides(["--model", "dlpfc=gpt-4o"]) == {"DLPFC_MODEL": "gpt-4o"}


def test_later_model_flags_win():
    overrides = _overrides(["--model", "gpt-4o-mini", "--model", "DLPFC=gpt-4o"])

    assert overrides["DLPFC_MODEL"] == "gpt-4o"
    assert overrides["VMPFC_MODEL"] == "gpt-4o-mini"


def test_model_rejects_an_unknown_region(capsys):
    with pytest.raises(SystemExit):
        _overrides(["--model", "HIPPOCAMPUS=gpt-4o"])

    assert "unknown region" in capsys.readouterr().err


def test_model_rejects_a_missing_model_name(capsys):
    with pytest.raises(SystemExit):
        _overrides(["--model", "DLPFC="])

    assert "no model name" in capsys.readouterr().err


def test_scalar_overrides():
    overrides = _overrides(["--max-tokens", "50", "--timeout", "2.5", "--no-search"])

    assert overrides == {"MAX_TOKENS": 50, "REQUEST_TIMEOUT": 2.5, "SERPAPI_API_KEY": None}


def test_no_flags_means_no_overrides():
    assert _overrides([TOPIC]) == {}


def test_log_level_is_normalised():
    assert _overrides(["--log-level", "debug"]) == {"LOG_LEVEL": "DEBUG"}


def test_log_level_rejects_an_invalid_value(capsys):
    # model_copy(update=...) skips validation, so argparse has to be the gate: an unvalidated
    # value would reach _LOG_LEVELS[...] and raise KeyError instead of falling back.
    with pytest.raises(SystemExit):
        _overrides(["--log-level", "verbose"])

    assert "invalid choice" in capsys.readouterr().err


def test_resolve_settings_leaves_the_singleton_untouched():
    resolved = cli.resolve_settings({"MAX_TOKENS": 11})

    assert resolved.MAX_TOKENS == 11
    assert settings.MAX_TOKENS != 11
    assert resolved is not settings


def test_resolve_settings_without_overrides_returns_the_singleton():
    assert cli.resolve_settings({}) is settings


def test_overrides_reach_the_crew(monkeypatch, capsys):
    captured = {}

    class RecordingCrew(StubCrew):
        def __init__(self, topic, settings=None):
            super().__init__(topic, settings)
            captured["settings"] = settings

    monkeypatch.setattr(cli, "CustomCrew", RecordingCrew)

    cli.main([TOPIC, "--model", "DLPFC=gpt-4o-injected", "--max-tokens", "77"])

    assert captured["settings"].DLPFC_MODEL == "gpt-4o-injected"
    assert captured["settings"].MAX_TOKENS == 77


def test_dry_run_makes_no_api_calls_and_shows_every_prompt(capsys):
    cli.main([TOPIC, "--dry-run"])

    err = capsys.readouterr().err
    assert "no API calls were made" in err
    for role in ROLES:
        assert role.name in err
    assert "Expected output:" in err


def test_dry_run_does_not_require_an_api_key(monkeypatch, capsys):
    monkeypatch.setattr(settings, "OPENAI_API_KEY", None)

    cli.main([TOPIC, "--dry-run"])

    assert "no API calls were made" in capsys.readouterr().err


def test_missing_api_key_still_fails_a_real_run(monkeypatch, stub_crew, capsys):
    monkeypatch.setattr(settings, "OPENAI_API_KEY", None)

    with pytest.raises(SystemExit) as excinfo:
        cli.main([TOPIC])

    assert excinfo.value.code == cli.EXIT_ERROR
    assert "OPENAI_API_KEY" in capsys.readouterr().err


def test_crew_failure_exits_non_zero(monkeypatch, capsys):
    class ExplodingCrew(StubCrew):
        def run(self):
            raise RuntimeError("boom")

    monkeypatch.setattr(cli, "CustomCrew", ExplodingCrew)

    with pytest.raises(SystemExit) as excinfo:
        cli.main([TOPIC])

    assert excinfo.value.code == cli.EXIT_ERROR
    assert "An unexpected error occurred" in capsys.readouterr().err


def test_no_stdin_is_reported_clearly(monkeypatch, stub_crew, capsys):
    def raise_eof():
        raise EOFError

    monkeypatch.setattr("builtins.input", raise_eof)

    with pytest.raises(SystemExit) as excinfo:
        cli.main([])

    assert excinfo.value.code == cli.EXIT_NO_TOPIC
    assert "No topic supplied on stdin" in capsys.readouterr().err


def test_interruption_uses_the_conventional_exit_code(monkeypatch, stub_crew, capsys):
    def raise_interrupt():
        raise KeyboardInterrupt

    monkeypatch.setattr("builtins.input", raise_interrupt)

    with pytest.raises(SystemExit) as excinfo:
        cli.main([])

    assert excinfo.value.code == cli.EXIT_INTERRUPTED
    assert "interrupted by user" in capsys.readouterr().err
