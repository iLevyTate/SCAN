"""Report building and emission.

Pure functions, so these need no agents and no language models -- which is most of why the
suite got faster.
"""

import io

import pytest

from scan import report
from scan.report import MISSING_SECTION, REPORT_SECTIONS
from scan.roles import BY_TASK_NAME, ROLES, SUMMARY_ROLE

ALL_SECTIONS = {name: f"Body of {name}" for _, name in REPORT_SECTIONS}


def test_build_renders_every_section_in_report_order():
    result = report.build("Some topic", ALL_SECTIONS)

    assert result.startswith("# SCAN report: Some topic\n")
    positions = [result.index(title) for title, _ in REPORT_SECTIONS]
    assert positions == sorted(positions)
    for _, task_name in REPORT_SECTIONS:
        assert ALL_SECTIONS[task_name] in result


def test_the_integrating_region_leads_the_report():
    # Regression: DLPFC synthesises the other four but was rendered as just another peer
    # section, so a reader had to get through four analyses to reach the recommendation.
    result = report.build("Some topic", ALL_SECTIONS)

    assert f"## {SUMMARY_ROLE.section_title} ({SUMMARY_ROLE.name})" in result
    assert result.index(ALL_SECTIONS[SUMMARY_ROLE.task_name]) < result.index("## How we got here")
    for role in ROLES:
        if not role.is_summary:
            assert result.index("## How we got here") < result.index(ALL_SECTIONS[role.task_name])


def test_supporting_sections_sit_below_the_summary():
    result = report.build("Some topic", ALL_SECTIONS)

    for role in ROLES:
        heading = f"{role.section_title} ({role.name})"
        expected = "## " if role.is_summary else "### "
        assert f"{expected}{heading}" in result


def test_build_flags_missing_sections(caplog):
    # Regression: an empty dict produced a complete-looking report with every section blank
    # and no warning, so a failed run was indistinguishable from a successful one.
    result = report.build("Some topic", {})

    assert result.count(MISSING_SECTION) == len(REPORT_SECTIONS)
    assert "Report is missing output for" in caplog.text


def test_build_marks_only_the_absent_sections():
    outputs = dict(ALL_SECTIONS)
    dropped = REPORT_SECTIONS[-1][1]
    del outputs[dropped]

    result = report.build("Some topic", outputs)

    assert result.count(MISSING_SECTION) == 1
    assert BY_TASK_NAME[dropped].section_title in result


@pytest.mark.parametrize("role", ROLES, ids=lambda role: role.name)
def test_every_role_has_exactly_one_section(role):
    titles = [title for title, name in REPORT_SECTIONS if name == role.task_name]

    assert titles == [f"{role.section_title} ({role.name})"]


def test_section_titles_are_unique():
    titles = [title for title, _ in REPORT_SECTIONS]

    assert len(set(titles)) == len(titles)


def test_emit_writes_the_report_verbatim():
    stream = io.StringIO()

    report.emit("## title\n\nbody", stream=stream)

    assert stream.getvalue() == "## title\n\nbody\n"


def test_emit_does_not_double_the_trailing_newline():
    stream = io.StringIO()

    report.emit("body\n", stream=stream)

    assert stream.getvalue() == "body\n"


@pytest.mark.parametrize(
    "text",
    ["Cost estimate [see appendix], risk [high].", "Checklist: [/] done", "word " * 60],
)
def test_emit_never_transforms_the_payload(text):
    # Regression, twice over: Rich's markup parsing deleted bracketed spans and raised on a
    # stray "[/]", and soft-wrapping injected real newlines into a redirected report. The
    # report now bypasses Rich entirely.
    stream = io.StringIO()

    report.emit(text, stream=stream)

    assert stream.getvalue() == text + "\n"
