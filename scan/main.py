from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

from crewai import Crew, Process
from dotenv import load_dotenv

from scan.errors import MissingEnvironmentVariableError
from scan.openai_llm import OpenAIWrapper
from scan.scan_agents import PFCAgents
from scan.scan_tasks import PFCTasks

if TYPE_CHECKING:
    from crewai.crews.crew_output import CrewOutput

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set to INFO for general usage
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_api_key() -> str:
    """Load the OpenAI API key from environment variables."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY environment variable is missing.")
        raise MissingEnvironmentVariableError("OPENAI_API_KEY")
    logger.info("OpenAI API Key Loaded Successfully.")
    return api_key


class CustomCrew:
    """Manages PFC agents and tasks for the SCAN system."""

    def __init__(self, topic: str, api_key: str) -> None:
        self.topic = topic
        self.api_key = api_key
        self.manager_llm = OpenAIWrapper(api_key=self.api_key, model_name="gpt-4").llm
        self.agents = PFCAgents(api_key=self.api_key, topic=self.topic)
        self.tasks = PFCTasks(agents=self.agents)

    def run(self) -> None:
        """Execute all tasks and generate the final output."""
        try:
            # Initialize tasks
            tasks = [
                self.tasks.complex_decision_making_task(self.topic),
                # self.tasks.emotional_risk_assessment_task(self.topic),
                # self.tasks.reward_evaluation_task(self.topic),
                # self.tasks.conflict_resolution_task(self.topic),
                # self.tasks.social_cognition_task(self.topic),
            ]

            # Create the crew
            crew = Crew(
                agents=self.agents.get_all_agents(),
                tasks=tasks,
                manager_llm=self.manager_llm,
                process=Process.hierarchical,  # Updated to match latest crewai constants
                memory=True,
            )

            logger.info("Starting crew execution...")
            crew_output = crew.kickoff()
            logger.info("Crew execution completed.")

            # Get task outputs
            task_outputs = self.get_task_outputs(crew_output)
            # Combine outputs into final report
            final_report = self.combine_outputs(task_outputs)
            print("\n\n########################")
            print("## SCAN AI Operation Result:")
            print("########################\n")
            print(final_report)

        except Exception as e:
            logger.exception("An error occurred during crew execution")
            print(f"An error occurred during crew execution: {e}")

    def get_task_outputs(self, crew_output: CrewOutput) -> dict[str, Any]:
        """Retrieve and process outputs from each task in the crew."""
        task_outputs = {}
        for task_result in crew_output.tasks_output:
            task_name = task_result.name
            output = task_result.raw
            if output:
                task_outputs[task_name] = output
            else:
                logger.warning(f"No output found for task: {task_name}")
        return task_outputs

    def combine_outputs(self, task_outputs: dict[str, str]) -> str:
        """Combine outputs from all tasks into a final report."""
        sections = [
            (
                "Decision-making analysis (DLPFC)",
                task_outputs.get("complex_decision_making_task", ""),
            ),
            ("Emotional analysis (VMPFC)", task_outputs.get("emotional_risk_assessment_task", "")),
            ("Reward evaluation (OFC)", task_outputs.get("reward_evaluation_task", "")),
            ("Conflict resolution (ACC)", task_outputs.get("conflict_resolution_task", "")),
            ("Social insights (MPFC)", task_outputs.get("social_cognition_task", "")),
        ]
        final_report = f"## SCAN AI Final Report on: {self.topic}\n\n"
        for title, content in sections:
            final_report += f"### {title}\n{content}\n\n"
        return final_report


def main() -> None:
    """Main entry point for the SCAN system."""
    load_dotenv()
    api_key = load_api_key()
    print("## Welcome to the SCAN System")
    print("---------------------------------------------------------------")
    try:
        topic = input("Please enter the topic you need help with: ")
        print(f"You entered: {topic}")
        custom_crew = CustomCrew(topic=topic, api_key=api_key)
        custom_crew.run()
    except MissingEnvironmentVariableError as e:
        logger.error(e)
        print(e)
    except KeyboardInterrupt:
        logger.info("Execution interrupted by user.")
        print("Execution interrupted by user.")
    except Exception as e:
        logger.exception("An unexpected error occurred")
        print(f"An unexpected error occurred: {e}")
    print("Thank you for using the SCAN System. Have a great day!")


if __name__ == "__main__":
    main()
