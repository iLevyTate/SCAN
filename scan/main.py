import os
import logging

from crewai import Crew, Process
from dotenv import load_dotenv
from openai import RateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from scan.openai_llm import OpenAIWrapper  # Import the updated wrapper
from scan.scan_agents import PFCAgents
from scan.scan_tasks import PFCTasks

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Get the OpenAI API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("The OPENAI_API_KEY environment variable is not set.")
else:
    logger.debug("OpenAI API Key loaded successfully.")
    # Optionally log part of the key for verification
    logger.debug(f"OpenAI API Key starts with: {openai_api_key[:5]}")

class CustomCrew:
    def __init__(self, topic):
        self.topic = topic
        self.llm = OpenAIWrapper(api_key=openai_api_key)  # Initialize the LLM wrapper
        self.agents = PFCAgents(llm=self.llm.llm, topic=self.topic)  # Pass topic to agents
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

        results = _run_crew(crew)
        logger.info("Raw results: %s", results)

        # Directly use the results without converting to or from JSON
        final_output = self.combine_outputs(results)
        return final_output

    def combine_outputs(self, results):
        # Assuming the results are concatenated strings
        combined_output = f"""
        Based on the analysis of the topic '{self.topic}', here are the recommended steps and solutions:

        {results}

        By following these steps and integrating these strategies, you can effectively address the challenges related to '{self.topic}' and achieve your goals with greater clarity and focus.
        """
        return combined_output


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, max=60),
    retry=retry_if_exception_type(RateLimitError),
)
def _run_crew(crew):
    try:
        return crew.kickoff()
    except RateLimitError as e:
        if e.code == "insufficient_quota":
            raise RuntimeError(
                "Insufficient quota. Please check your OpenAI plan and billing details."
            ) from e
        else:
            raise


def main() -> int:
    print("## Welcome to the SCAN System")
    print("---------------------------------------------------------------")
    continue_analysis = True
    while continue_analysis:
        topic = input("Please enter the topic you need help with: ")
        print("You entered:", topic)
        custom_crew = CustomCrew(topic)
        result = custom_crew.run()
        print("\n\n########################")
        print("## SCAN AI Operation Result:")
        print("########################\n")
        print(result)

        continue_analysis = False

    print("Thank you for using the SCAN System. Have a great day!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
