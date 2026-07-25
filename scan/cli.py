"""Command-line interface for SCAN.

Bare ``run-scan`` still drops into the interactive prompt exactly as it always did; every flag
is additive.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from scan import __version__, report
from scan.config import Settings
from scan.config import settings as default_settings
from scan.console import console
from scan.errors import MissingEnvironmentVariableError
from scan.main import CustomCrew
from scan.project_logger import configure_logging, get_logger
from scan.roles import BY_NAME, BY_TASK_NAME, ROLES

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from crewai import Task
    from crewai.tasks.task_output import TaskOutput

logger = get_logger(__name__)

__all__ = ["build_parser", "main", "settings_overrides"]

LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_NO_TOPIC = 2
EXIT_INTERRUPTED = 130


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        prog="run-scan",
        description=(
            "Analyse a topic with five agents modelled on prefrontal-cortex regions. "
            "The report goes to stdout; progress and errors go to stderr."
        ),
        epilog=(
            "Run with no arguments to be prompted for a topic. "
            f"Exit codes: {EXIT_OK} success, {EXIT_ERROR} error, "
            f"{EXIT_NO_TOPIC} no topic supplied, {EXIT_INTERRUPTED} interrupted."
        ),
    )
    parser.add_argument(
        "topic",
        nargs="?",
        help="Topic to analyse. Omit to be prompted interactively.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        metavar="PATH",
        help="Write the report to PATH instead of stdout.",
    )
    parser.add_argument(
        "--model",
        action="append",
        metavar="SPEC",
        help=(
            "Model to use. 'gpt-4o-mini' sets every region; 'DLPFC=gpt-4o' sets one. "
            "Repeatable; later values win."
        ),
    )
    parser.add_argument("--max-tokens", type=int, metavar="N", help="Override MAX_TOKENS.")
    parser.add_argument(
        "--timeout",
        type=float,
        metavar="SECONDS",
        help="Per-request timeout. Bounds a hung connection, not the whole run.",
    )
    parser.add_argument(
        "--no-search",
        action="store_true",
        help="Disable the web search tool even if SERPAPI_API_KEY is set.",
    )
    parser.add_argument(
        "--log-level",
        type=str.upper,
        choices=LOG_LEVELS,
        help="Logging verbosity. Defaults to LOG_LEVEL, or WARNING.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved configuration and every task prompt, and make no API calls.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def parse_model_specs(specs: Sequence[str], parser: argparse.ArgumentParser) -> dict[str, str]:
    """Turn --model values into Settings overrides.

    A bare model name sets every region; ``ROLE=MODEL`` sets one. Applied in order, so
    ``--model gpt-4o-mini --model DLPFC=gpt-4o`` does what it looks like.
    """
    overrides: dict[str, str] = {}
    for spec in specs:
        role_name, separator, model = spec.partition("=")
        if not separator:
            if not spec.strip():
                parser.error("--model requires a model name")
            overrides.update({role.model_setting: spec.strip() for role in ROLES})
            continue
        role_name = role_name.strip().upper()
        if role_name not in BY_NAME:
            known = ", ".join(role.name for role in ROLES)
            parser.error(
                f"unknown region {role_name!r} in --model {spec!r}; expected one of: {known}"
            )
        if not model.strip():
            parser.error(f"--model {spec!r} has no model name")
        overrides[BY_NAME[role_name].model_setting] = model.strip()  # type: ignore[index]
    return overrides


def settings_overrides(args: argparse.Namespace, parser: argparse.ArgumentParser) -> dict[str, Any]:
    """Map parsed arguments onto Settings field overrides."""
    overrides: dict[str, Any] = {}
    if args.model:
        overrides.update(parse_model_specs(args.model, parser))
    if args.max_tokens is not None:
        overrides["MAX_TOKENS"] = args.max_tokens
    if args.timeout is not None:
        overrides["REQUEST_TIMEOUT"] = args.timeout
    if args.no_search:
        overrides["SERPAPI_API_KEY"] = None
    if args.log_level is not None:
        overrides["LOG_LEVEL"] = args.log_level
    return overrides


def resolve_settings(overrides: dict[str, Any]) -> Settings:
    """Apply CLI overrides on top of the configured settings.

    ``model_copy(update=...)`` skips validation, so anything that reaches it must already have
    been validated by argparse -- which is why ``--log-level`` uses ``choices``.
    """
    return default_settings.model_copy(update=overrides) if overrides else default_settings


def prompt_for_topic() -> str:
    """Ask for the topic interactively.

    The prompt goes to the stderr console rather than to ``input()``, which writes its prompt
    argument to *stdout* and would put it inside a redirected report.
    """
    console.print("Please enter the topic you need help with: ", end="")
    return input().strip()


def describe_plan(topic: str, settings: Settings) -> None:
    """Print the resolved configuration and every prompt, without calling any model."""
    console.print(f"Topic: {topic}")
    console.print(f"Max tokens: {settings.MAX_TOKENS}   Timeout: {settings.REQUEST_TIMEOUT}s")
    console.print(f"Search tool: {'enabled' if settings.SERPAPI_API_KEY else 'disabled'}")
    crew = CustomCrew(topic=topic, settings=settings)
    for index, task in enumerate(crew.build_tasks(), start=1):
        role = task.agent.role
        console.print(
            f"\n--- [{index}/{len(ROLES)}] {role} "
            f"({getattr(settings, BY_NAME[role].model_setting)}) ---"  # type: ignore[index]
        )
        console.print(task.description)
        console.print(f"Expected output: {task.expected_output}")


class ProgressReporter:
    """Reports each analysis to stderr as it lands.

    Keyed off the task list in execution order, never off the report order -- the two differ,
    because the synthesis runs last but is reported first.
    """

    def __init__(self, tasks: Sequence[Task], clock: Callable[[], float] = time.monotonic):
        self._labels = [task.name or "task" for task in tasks]
        self._clock = clock
        self._started = clock()
        self._done = 0

    @property
    def next_label(self) -> str:
        role = BY_TASK_NAME.get(self._labels[self._done]) if self._done < self.total else None
        return f"{role.section_title} ({role.name})" if role else "Working"

    @property
    def total(self) -> int:
        return len(self._labels)

    def status_line(self) -> str:
        return f"[{self._done + 1}/{self.total}] {self.next_label}..."

    def on_task_complete(self, output: TaskOutput) -> None:
        role = BY_TASK_NAME.get(output.name or "")
        label = f"{role.section_title} ({role.name})" if role else (output.name or "task")
        self._done += 1
        elapsed = self._clock() - self._started
        console.print(f"[{self._done}/{self.total}] {label} - done ({elapsed:.0f}s)")


def deliver(text: str, output: Path | None) -> None:
    """Send the finished report to its destination."""
    if output is None:
        report.emit(text)
        return
    output.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")
    console.print(f"Report written to {output}")


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point for the ``run-scan`` console script."""
    parser = build_parser()
    args = parser.parse_args(argv)

    overrides = settings_overrides(args, parser)
    settings = resolve_settings(overrides)
    configure_logging(settings.COMPUTED_LOG_LEVEL)

    console.print("## Welcome to the SCAN System")
    console.print("---------------------------------------------------------------")
    try:
        # --dry-run makes no API calls, so it must not require a key.
        if not args.dry_run and not settings.OPENAI_API_KEY:
            raise MissingEnvironmentVariableError("OPENAI_API_KEY")

        topic = args.topic.strip() if args.topic else prompt_for_topic()
        if not topic:
            console.print("No topic was provided; nothing to analyse.")
            sys.exit(EXIT_NO_TOPIC)
        console.print(f"You entered: {topic}")

        if args.dry_run:
            describe_plan(topic, settings)
            console.print("\nDry run: no API calls were made.")
            return

        custom_crew = CustomCrew(topic=topic, settings=settings)
        progress = ProgressReporter(custom_crew.build_tasks())
        try:
            with console.status(progress.status_line()) as status:

                def advance(output: TaskOutput) -> None:
                    progress.on_task_complete(output)
                    status.update(progress.status_line())

                final_report = custom_crew.run(on_task_complete=advance)
        except Exception:
            # Emit whatever finished before re-raising: the user has already paid for it,
            # and until now a failure on the last analysis discarded all of the earlier ones.
            if custom_crew.completed:
                console.print(
                    f"Run failed after {len(custom_crew.completed)} of "
                    f"{len(ROLES)} analyses; writing a partial report."
                )
                deliver(custom_crew.partial_report(), args.output)
            raise
        deliver(final_report, args.output)
    except MissingEnvironmentVariableError as e:
        logger.error(e)
        console.print(str(e))
        sys.exit(EXIT_ERROR)
    except EOFError:
        console.print("No topic supplied on stdin; run SCAN interactively or pipe a topic in.")
        sys.exit(EXIT_NO_TOPIC)
    except KeyboardInterrupt:
        logger.info("Execution interrupted by user.")
        console.print("Execution interrupted by user.")
        sys.exit(EXIT_INTERRUPTED)
    except Exception as e:
        logger.exception("An unexpected error occurred")
        console.print(f"An unexpected error occurred: {e}")
        sys.exit(EXIT_ERROR)
    console.print("Thank you for using the SCAN System. Have a great day!")


if __name__ == "__main__":  # pragma: no cover
    main()
