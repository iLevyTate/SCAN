# scan/main.py

import os
import logging

from dotenv import load_dotenv

from scan.openai_llm import OpenAIWrapper
from scan.scan_agents import PFCAgents
from scan.scan_tasks import PFCTasks
from scan.errors import MissingEnvironmentVariableError
from crewai import Crew, Process

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Get the OpenAI API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise MissingEnvironmentVariableError("OPENAI_API_KEY")


class CustomCrew:
    """Custom crew to manage PFC agents and tasks."""

    def __init__(self, topic: str):
        self.topic = topic
        # Initialize manager LLM
        self.manager_llm = OpenAIWrapper(api_key=openai_api_key, model_name='gpt-4').llm
        # Initialize agents
        logger.info("Initializing agents...")
        self.agents = PFCAgents(api_key=openai_api_key, topic=self.topic)
        # Initialize tasks
        logger.info("Initializing tasks...")
        self.tasks = PFCTasks(agents=self.agents)

    def run(self) -> str:
        """Run the crew to execute all tasks and generate the final output."""
        # Initialize tasks
        decision_task = self.tasks.complex_decision_making_task(self.topic)
        emotional_task = self.tasks.emotional_risk_assessment_task(self.topic)
        reward_task = self.tasks.reward_evaluation_task(self.topic)
        conflict_task = self.tasks.conflict_resolution_task(self.topic)
        social_task = self.tasks.social_cognition_task(self.topic)

        # Create a Crew
        crew = Crew(
            agents=[
                self.agents.dlpfc_agent(),
                self.agents.vmpfc_agent(),
                self.agents.ofc_agent(),
                self.agents.acc_agent(),
                self.agents.mpfc_agent(),
            ],
            tasks=[
                decision_task,
                emotional_task,
                reward_task,
                conflict_task,
                social_task,
            ],
            manager_llm=self.manager_llm,  # Use manager LLM
            process=Process.hierarchical,
            memory=True,
        )

        logger.info("Starting crew execution...")
        # Kick off the crew and get the structured output
        crew_output = crew.kickoff()
        logger.info("Crew execution completed.")

        # Combine results
        final_output = self.combine_outputs(crew_output)
        return final_output

    def combine_outputs(self, crew_output: dict) -> str:
        """Combine the outputs from all agents into a final report."""
        # Extract task outputs from the crew output
        task_results = crew_output.get('tasks', {})

        # Retrieve outputs for each task
        decision_output = task_results.get('complex_decision_making_task', {}).get('output', 'No output')
        emotional_output = task_results.get('emotional_risk_assessment_task', {}).get('output', 'No output')
        reward_output = task_results.get('reward_evaluation_task', {}).get('output', 'No output')
        conflict_output = task_results.get('conflict_resolution_task', {}).get('output', 'No output')
        social_output = task_results.get('social_cognition_task', {}).get('output', 'No output')

        # Format the final report
        final_report = f"""
Final analysis on '{self.topic}':

1. Decision-making report (DLPFC):
{decision_output}

2. Emotional analysis (VMPFC):
{emotional_output}

3. Reward evaluation (OFC):
{reward_output}

4. Conflict resolution (ACC):
{conflict_output}

5. Social insights (MPFC):
{social_output}
"""
        return final_report


def main() -> int:
    """Main entry point for the SCAN system."""
    print("## Welcome to the SCAN System")
    print("---------------------------------------------------------------")
    try:
        topic = input("Please enter the topic you need help with: ")
        print("You entered:", topic)
        custom_crew = CustomCrew(topic)
        result = custom_crew.run()
        print("\n\n########################")
        print("## SCAN AI Operation Result:")
        print("########################\n")
        print(result)
    except MissingEnvironmentVariableError as e:
        logger.error(e.message)
        print(e.message)
    except Exception as e:
        logger.exception("An unexpected error occurred: %s", e)
        print(f"An unexpected error occurred: {e}")

    print("Thank you for using the SCAN System. Have a great day!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
