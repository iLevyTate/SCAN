import sys
import os

# Ensure the project root is in the PYTHONPATH
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
from openai_llm import OpenAIWrapper  # Import the updated wrapper

# Load environment variables from .env file
load_dotenv()

# Get the OpenAI API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("The OPENAI_API_KEY environment variable is not set.")

from crewai import Crew, Process
from scan.errors import MissingEnvironmentVairableError
from scan.scan_agents import PFCAgents
from scan.scan_tasks import PFCTasks


class CustomCrew:
    def __init__(self, topic):
        self.topic = topic
        self.llm = OpenAIWrapper(api_key=openai_api_key)  # Initialize the LLM wrapper
        self.agents = PFCAgents(llm=self.llm.llm)  # Initialize agents with OpenAI LLM
        self.tasks = PFCTasks(self.agents)  # Initialize tasks with agents

    def run(self):
        decision_task = self.tasks.complex_decision_making_task(self.topic)
        emotional_task = self.tasks.emotional_risk_assessment_task(self.topic)
        reward_task = self.tasks.reward_evaluation_task(self.topic)
        conflict_task = self.tasks.conflict_resolution_task(self.topic)
        social_task = self.tasks.social_cognition_task(self.topic)

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
            manager_llm=self.llm.llm,  # Use the OpenAI LLM wrapper as the manager
            process=Process.hierarchical,  # Specifies the hierarchical management approach
            memory=True,  # Enable memory usage for enhanced task execution
        )

        result = crew.kickoff()
        return result


def main() -> int:
    print("## Welcome to the SCAN")
    print("---------------------------------------------------------------")
    continue_analysis = True
    while continue_analysis:
        topic = input("Please enter the topic you want to analyze: ")
        print("You entered:", topic)
        custom_crew = CustomCrew(topic)
        result = custom_crew.run()
        print("\n\n########################")
        print("## Crew AI Operation Result:")
        print("########################\n")
        print(result)
        response = input("Do you want to analyze another topic? (yes/no): ")
        if response.lower() != "yes":
            continue_analysis = False

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
