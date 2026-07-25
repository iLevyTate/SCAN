"""Building and emitting the final report.

Pure string handling, deliberately free of crewai and of settings, so it can be tested without
constructing five agents and five language models.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from scan.project_logger import get_logger
from scan.roles import REPORT_ORDER, SUMMARY_ROLE

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import TextIO

logger = get_logger(__name__)

__all__ = ["MISSING_SECTION", "REPORT_SECTIONS", "build", "emit"]

MISSING_SECTION = "_No output was produced for this section._"

#: Report heading -> the task whose output fills it. Generated from the role table so a heading
#: can never name a region other than the one that actually produced the section.
REPORT_SECTIONS: tuple[tuple[str, str], ...] = tuple(
    (f"{role.section_title} ({role.name})", role.task_name) for role in REPORT_ORDER
)


def build(topic: str, outputs: Mapping[str, str], *, partial: bool = False) -> str:
    """Assemble the markdown report from the per-task outputs.

    The integrating region leads as a summary rather than sitting among the others as an equal
    peer: someone who reads only the first screen should come away with the recommendation.

    ``partial`` marks the document itself as incomplete. A banner in the text survives being
    redirected to a file, where an exit code does not.
    """
    missing = [name for _, name in REPORT_SECTIONS if not outputs.get(name)]
    if missing:
        logger.warning(f"Report is missing output for: {', '.join(missing)}")

    lines = [f"# SCAN report: {topic}", ""]
    if partial or missing:
        finished = len(REPORT_SECTIONS) - len(missing)
        lines += [
            f"> **Incomplete report.** {finished} of {len(REPORT_SECTIONS)} analyses "
            "finished; the rest are marked below.",
            "",
        ]

    summary = SUMMARY_ROLE
    lines += [
        f"## {summary.section_title} ({summary.name})",
        "",
        outputs.get(summary.task_name) or MISSING_SECTION,
        "",
        "## How we got here",
        "",
    ]
    for role in REPORT_ORDER:
        if role.is_summary:
            continue
        lines += [
            f"### {role.section_title} ({role.name})",
            "",
            outputs.get(role.task_name) or MISSING_SECTION,
            "",
        ]
    return "\n".join(lines)


def emit(report: str, stream: TextIO | None = None) -> None:
    """Write the report to stdout, and nothing else to stdout.

    Plain ``sys.stdout.write`` rather than the Rich console: Rich's ``soft_wrap`` silently
    truncates block renderables, and the report must reach a pipe or a file byte for byte.
    """
    target = stream if stream is not None else sys.stdout
    target.write(report if report.endswith("\n") else report + "\n")
    target.flush()
