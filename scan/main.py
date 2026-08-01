"""Crew orchestration: turning a topic into a finished report.

The command-line interface lives in :mod:`scan.cli`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from crewai import Crew, Process

from scan import report
from scan.config import apply_environment
from scan.config import settings as default_settings
from scan.project_logger import get_logger
from scan.provider_errors import translate
from scan.report import REPORT_SECTIONS
from scan.scan_agents import PFCAgents
from scan.scan_tasks import PFCTasks

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from crewai import Task
    from crewai.crews.crew_output import CrewOutput
    from crewai.tasks.task_output import TaskOutput

    from scan.config import Settings

logger = get_logger(__name__)

__all__ = ["REPORT_SECTIONS", "CustomCrew"]


class CustomCrew:
    """Manages PFC agents and tasks for the SCAN system."""

    def __init__(self, topic: str, settings: Settings | None = None) -> None:
        self.topic = topic
        self.settings = settings if settings is not None else default_settings
        apply_environment(self.settings)
        self.agents = PFCAgents(topic=self.topic, settings=self.settings)
        self.tasks = PFCTasks(agents=self.agents)
        #: Outputs of the tasks that have finished, accumulated as they land so a failure
        #: part-way through still leaves the completed analyses recoverable.
        self.completed: dict[str, str] = {}

    def build_tasks(self) -> list[Task]:
        """Build the task list, wiring dependencies through crewai's `context`."""
        return self.tasks.build_all(self.topic)

    def partial_report(self) -> str:
        """Render whatever finished before the run stopped."""
        return report.build(self.topic, self.completed, partial=True)

    def _on_task_complete(
        self, on_task_complete: Callable[[TaskOutput], None] | None
    ) -> Callable[[TaskOutput], None]:
        """Wrap the caller's progress callback so display can never fail a paid run."""

        def callback(output: TaskOutput) -> None:
            if output.name and output.raw:
                self.completed[output.name] = output.raw
            if on_task_complete is None:
                return
            try:
                # crewai invokes this inline (crewai/task.py), so an exception raised here
                # aborts the whole crew after minutes of work.
                on_task_complete(output)
            except Exception:
                logger.debug("Progress callback failed", exc_info=True)

        return callback

    def run(self, on_task_complete: Callable[[TaskOutput], None] | None = None) -> str:
        """Execute all tasks and return the final report.

        Raises whatever the crew raises: the caller decides how to report it and what to exit
        with. Swallowing exceptions here meant a totally failed run still exited 0. Anything
        that finished before the failure is left in :attr:`completed`.
        """
        crew = Crew(
            agents=self.agents.get_all_agents(),
            tasks=self.build_tasks(),
            task_callback=self._on_task_complete(on_task_complete),
            # Sequential, not hierarchical. Under Process.hierarchical crewai routes *every*
            # task to an auto-created generic "Crew Manager" agent (Crew._get_agent_to_use
            # ignores task.agent), so the five PFC agents, their goals/backstories, their
            # per-role models and their search tool were all bypassed.
            process=Process.sequential,
            # Crew memory is off: nothing in SCAN reads it back, but it adds an evaluation
            # completion plus embedding calls per task, and its failures print directly to
            # stdout from inside crewai.
            memory=False,
        )

        logger.info("Starting crew execution...")
        try:
            crew_output = crew.kickoff()
        except Exception as error:
            # Translate at the boundary so a mistyped model or a rejected key reads as
            # something the user can act on, not as a raw provider message.
            translated = translate(error, self.settings)
            if translated is error:
                raise
            raise translated from error
        logger.info("Crew execution completed.")

        # crewai declares token_usage as UsageMetrics but defaults it to a bare dict, so this
        # has to tolerate both. Reported from the crew because the langchain callback handler
        # SCAN used to carry never reaches the model crewai actually calls.
        total = getattr(crew_output.token_usage, "total_tokens", None)
        if total is not None:
            usage = crew_output.token_usage
            logger.info(
                f"Token usage: {total} total "
                f"({usage.prompt_tokens} prompt, {usage.completion_tokens} completion)"
            )

        return self.combine_outputs(self.get_task_outputs(crew_output))

    def get_task_outputs(self, crew_output: CrewOutput) -> dict[str, str]:
        """Retrieve and process outputs from each task in the crew."""
        task_outputs = {}
        for task_result in crew_output.tasks_output:
            task_name = task_result.name
            output = task_result.raw
            if not task_name:
                logger.warning("Skipping a task output with no task name; cannot place it.")
                continue
            if output:
                task_outputs[task_name] = output
            else:
                logger.warning(f"No output found for task: {task_name}")
        return task_outputs

    def combine_outputs(self, task_outputs: Mapping[str, str]) -> str:
        """Combine outputs from all tasks into a final report."""
        return report.build(self.topic, task_outputs)
