import os

from crewai import Crew, Process
from dotenv import load_dotenv
from openai import RateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from scan.openai_llm import OpenAIWrapper
from scan.scan_agents import PFCAgents
from scan.scan_tasks import PFCTasks

# Load environment variables from .env file
load_dotenv()

# Get the OpenAI API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("The OPENAI_API_KEY environment variable is not set.")


class CustomCrew:
    def __init__(self, topic):
        self.topic = topic
        self.llm = OpenAIWrapper(api_key=openai_api_key)
        self.agents = PFCAgents(llm=self.llm.llm, topic=self.topic)
        self.tasks = PFCTasks(self.agents)

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
            manager_llm=self.llm.llm,
            process=Process.hierarchical,
            memory=True,
        )

        result = _run_crew(crew)
        return result


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
        response = input("Would you like to analyze another topic? (yes/no): ")
        if response.lower() != "yes":
            continue_analysis = False

    print("Thank you for using the SCAN System. Have a great day!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
