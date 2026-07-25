from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

from crewai import Crew, Process

from scan import report
from scan.config import apply_environment
from scan.config import settings as default_settings
from scan.console import console
from scan.errors import MissingEnvironmentVariableError
from scan.project_logger import configure_logging, get_logger
from scan.report import REPORT_SECTIONS
from scan.scan_agents import PFCAgents
from scan.scan_tasks import PFCTasks

if TYPE_CHECKING:
    from collections.abc import Mapping

    from crewai import Task
    from crewai.crews.crew_output import CrewOutput

    from scan.config import Settings

logger = get_logger(__name__)

__all__ = ["REPORT_SECTIONS", "CustomCrew", "main"]


class CustomCrew:
    """Manages PFC agents and tasks for the SCAN system."""

    def __init__(self, topic: str, settings: Settings | None = None) -> None:
        self.topic = topic
        self.settings = settings if settings is not None else default_settings
        apply_environment(self.settings)
        self.agents = PFCAgents(topic=self.topic, settings=self.settings)
        self.tasks = PFCTasks(agents=self.agents)

    def build_tasks(self) -> list[Task]:
        """Build the task list, wiring dependencies through crewai's `context`."""
        return self.tasks.build_all(self.topic)

    def run(self) -> str:
        """Execute all tasks and return the final report.

        Raises whatever the crew raises: the caller decides how to report it and what to exit
        with. Swallowing exceptions here meant a totally failed run still exited 0.
        """
        crew = Crew(
            agents=self.agents.get_all_agents(),
            tasks=self.build_tasks(),
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
        crew_output = crew.kickoff()
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

    def get_task_outputs(self, crew_output: CrewOutput) -> dict[str, Any]:
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


def prompt_for_topic() -> str:
    """Ask for the topic interactively.

    The prompt is printed to stderr rather than passed to ``input()``, which writes its prompt
    argument to *stdout* -- that put "Please enter the topic..." inside any redirected report.
    """
    console.print("Please enter the topic you need help with: ", end="")
    return input().strip()


def main() -> None:
    """Main entry point for the SCAN system."""
    configure_logging()
    console.print("## Welcome to the SCAN System")
    console.print("---------------------------------------------------------------")
    try:
        if not default_settings.OPENAI_API_KEY:
            raise MissingEnvironmentVariableError("OPENAI_API_KEY")
        topic = prompt_for_topic()
        if not topic:
            console.print("No topic was provided; nothing to analyse.")
            sys.exit(2)
        console.print(f"You entered: {topic}")

        custom_crew = CustomCrew(topic=topic)
        with console.status("Thinking..."):
            final_report = custom_crew.run()

        report.emit(final_report)
    except MissingEnvironmentVariableError as e:
        logger.error(e)
        console.print(str(e))
        sys.exit(1)
    except EOFError:
        console.print("No topic supplied on stdin; run SCAN interactively or pipe a topic in.")
        sys.exit(2)
    except KeyboardInterrupt:
        logger.info("Execution interrupted by user.")
        console.print("Execution interrupted by user.")
        sys.exit(130)
    except Exception as e:
        logger.exception("An unexpected error occurred")
        console.print(f"An unexpected error occurred: {e}")
        sys.exit(1)
    console.print("Thank you for using the SCAN System. Have a great day!")


if __name__ == "__main__":
    main()
