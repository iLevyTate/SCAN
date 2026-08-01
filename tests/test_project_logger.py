import logging

from scan.project_logger import NOISY_LOGGERS, configure_logging, get_logger


def test_get_logger_preserves_the_module_name():
    # Regression: every module imported one shared logger, so all log lines were labelled
    # "scan.project_logger" and you could not tell which module emitted them.
    assert get_logger("scan.scan_agents").name == "scan.scan_agents"


def test_configure_logging_sets_the_package_level_only():
    # Regression: basicConfig ran at import and set the level on the *root* logger, so
    # LOG_LEVEL=INFO also turned on httpx/LiteLLM/otel and buried SCAN's own output.
    root_level_before = logging.getLogger().level

    configure_logging(logging.INFO)

    assert logging.getLogger("scan").level == logging.INFO
    assert logging.getLogger().level == root_level_before


def test_configure_logging_keeps_noisy_libraries_quiet():
    configure_logging(logging.DEBUG)

    for name in NOISY_LOGGERS:
        assert logging.getLogger(name).level == logging.WARNING


def test_configure_logging_is_idempotent():
    configure_logging(logging.INFO)
    configure_logging(logging.INFO)

    assert len(logging.getLogger("scan").handlers) == 1
