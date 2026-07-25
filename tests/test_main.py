import pytest
from crewai import Process
from crewai.crews.crew_output import CrewOutput, TaskOutput

from scan import main as main_module
from scan.config import settings
from scan.main import REPORT_SECTIONS, CustomCrew
from scan.roles import BY_TASK_NAME, execution_order

ALL_SECTIONS = {
    "complex_decision_making_task": "Something complex",
    "emotional_risk_assessment_task": "Something emotional",
    "reward_evaluation_task": "Something rewarding",
    "conflict_resolution_task": "Something conflicting",
    "social_cognition_task": "Something social",
}


def _crew_output(outputs):
    return CrewOutput(
        tasks_output=[
            TaskOutput(description=f"desc {name}", name=name, raw=raw, agent="some_agent")
            for name, raw in outputs.items()
        ]
    )


def test_custom_crew_combine_outputs():
    crew = CustomCrew("Some topic")
    result = crew.combine_outputs(ALL_SECTIONS)

    assert (
        result
        == "## SCAN AI Final Report on: Some topic\n\n### Decision-making analysis (DLPFC)\nSomething complex\n\n### Emotional analysis (VMPFC)\nSomething emotional\n\n### Reward evaluation (OFC)\nSomething rewarding\n\n### Conflict resolution (ACC)\nSomething conflicting\n\n### Social insights (MPFC)\nSomething social\n\n"
    )


def test_combine_outputs_flags_missing_sections(caplog):
    # Regression: an empty dict produced a complete-looking report with every section blank
    # and no warning, so a failed run was indistinguishable from a successful one.
    crew = CustomCrew("Some topic")

    result = crew.combine_outputs({})

    assert "No output was produced for this section." in result
    assert "Report is missing output for" in caplog.text


def test_get_task_outputs():
    crew = CustomCrew("Some topic")
    result = crew.get_task_outputs(_crew_output({"task 1": "raw task 1", "task 2": "raw task 2"}))

    assert result == {"task 1": "raw task 1", "task 2": "raw task 2"}


def test_get_task_outputs_skips_unnamed_tasks(caplog):
    # An unnamed TaskOutput used to be stored under the key None, so several of them would
    # overwrite each other and none could ever be matched to a report section.
    crew = CustomCrew("Some topic")
    output = CrewOutput(
        tasks_output=[TaskOutput(description="d", raw="orphan", agent="a")],
    )

    assert crew.get_task_outputs(output) == {}
    assert "no task name" in caplog.text


def test_report_sections_match_the_task_names():
    crew = CustomCrew("Some topic")
    task_names = {task.name for task in crew.build_tasks()}

    assert {name for _, name in REPORT_SECTIONS} == task_names


def test_report_headings_name_the_agent_that_produced_them():
    # Regression: REPORT_SECTIONS used to embed the region as a naked literal, so reassigning
    # a task to another agent would leave the heading quietly lying and every test still green.
    tasks = {task.name: task for task in CustomCrew("Some topic").build_tasks()}

    for title, task_name in REPORT_SECTIONS:
        assert title.endswith(f"({tasks[task_name].agent.role})")
        assert title.startswith(BY_TASK_NAME[task_name].section_title)


def test_build_tasks_wires_context_and_orders_dependencies():
    # The graph itself is asserted against the role table in test_scan_tasks.py; here we only
    # check that CustomCrew hands crewai a correctly ordered list.
    crew = CustomCrew("Some topic")
    tasks = crew.build_tasks()
    order = [task.name for task in tasks]

    assert order == [role.task_name for role in execution_order()]
    for task in tasks:
        for dependency in task.context:
            assert order.index(dependency.name) < order.index(task.name)


def test_the_first_task_is_the_only_one_without_dependencies():
    tasks = CustomCrew("Some topic").build_tasks()

    assert tasks[0].context == []
    assert all(task.context for task in tasks[1:])


def test_run_uses_sequential_process_so_tasks_keep_their_own_agent(monkeypatch):
    # Regression: with Process.hierarchical crewai routes every task to an auto-created
    # generic "Crew Manager" agent and ignores task.agent, which bypassed all five PFC
    # agents, their per-role models and their tools.
    captured = {}

    class FakeCrew:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def kickoff(self):
            return _crew_output(ALL_SECTIONS)

    monkeypatch.setattr(main_module, "Crew", FakeCrew)
    crew = CustomCrew("Some topic")

    report = crew.run()

    assert captured["process"] == Process.sequential
    assert captured["memory"] is False
    assert [task.agent.role for task in captured["tasks"]] == [
        "VMPFC",
        "OFC",
        "MPFC",
        "ACC",
        "DLPFC",
    ]
    assert "Something complex" in report


def test_run_propagates_failures(monkeypatch):
    # Regression: run() caught every exception and printed it, so a completely failed run
    # still returned normally and the process exited 0.
    class ExplodingCrew:
        def __init__(self, **kwargs):
            pass

        def kickoff(self):
            raise RuntimeError("kickoff exploded")

    monkeypatch.setattr(main_module, "Crew", ExplodingCrew)

    with pytest.raises(RuntimeError, match="kickoff exploded"):
        CustomCrew("Some topic").run()


def test_main_reports_missing_openai_key(monkeypatch, capsys):
    # Regression: the friendly MissingEnvironmentVariableError path was dead because the
    # error was caught but never raised.
    monkeypatch.setattr(settings, "OPENAI_API_KEY", None)

    with pytest.raises(SystemExit) as excinfo:
        main_module.main()

    assert excinfo.value.code == 1
    assert "OPENAI_API_KEY" in capsys.readouterr().out


def test_main_exits_non_zero_when_the_crew_fails(monkeypatch, capsys):
    monkeypatch.setattr("builtins.input", lambda _: "a topic")

    class ExplodingCrew:
        def __init__(self, topic):
            pass

        def run(self):
            raise RuntimeError("boom")

    monkeypatch.setattr(main_module, "CustomCrew", ExplodingCrew)

    with pytest.raises(SystemExit) as excinfo:
        main_module.main()

    assert excinfo.value.code == 1
    assert "An unexpected error occurred" in capsys.readouterr().out


def test_main_rejects_an_empty_topic(monkeypatch, capsys):
    monkeypatch.setattr("builtins.input", lambda _: "   ")

    with pytest.raises(SystemExit) as excinfo:
        main_module.main()

    assert excinfo.value.code == 2
    assert "No topic was provided" in capsys.readouterr().out


def test_main_handles_no_stdin(monkeypatch, capsys):
    def raise_eof(_):
        raise EOFError

    monkeypatch.setattr("builtins.input", raise_eof)

    with pytest.raises(SystemExit) as excinfo:
        main_module.main()

    assert excinfo.value.code == 2
    assert "No topic supplied on stdin" in capsys.readouterr().out


def test_main_reports_interruption(monkeypatch, capsys):
    def raise_interrupt(_):
        raise KeyboardInterrupt

    monkeypatch.setattr("builtins.input", raise_interrupt)

    with pytest.raises(SystemExit) as excinfo:
        main_module.main()

    assert excinfo.value.code == 130
    assert "interrupted by user" in capsys.readouterr().out


def test_main_prints_the_report_on_success(monkeypatch, capsys):
    monkeypatch.setattr("builtins.input", lambda _: "a topic")

    class StubCrew:
        def __init__(self, topic):
            self.topic = topic

        def run(self):
            return "## report body"

    monkeypatch.setattr(main_module, "CustomCrew", StubCrew)

    main_module.main()

    out = capsys.readouterr().out
    assert "## report body" in out
    assert "Thank you for using the SCAN System" in out
