# main.py

import os
from crewai import Crew, Process
from scan_agents import PFCAgents
from scan_tasks import PFCTasks
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up the language model
llm = ChatGoogleGenerativeAI(
    model="gemini-pro",
    verbose=True,
    temperature=0.6,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

class CustomCrew:
    def __init__(self, topic):
        self.topic = topic
        self.agents = PFCAgents(llm)
        self.tasks = PFCTasks(self.agents)

    def run(self):
        decision_task = self.tasks.complex_decision_making_task(self.topic)
        emotional_task = self.tasks.emotional_risk_assessment_task(self.topic)
        reward_task = self.tasks.reward_evaluation_task(self.topic)
        conflict_task = self.tasks.conflict_resolution_task(self.topic)
        social_task = self.tasks.social_cognition_task(self.topic)

        crew = Crew(
            agents=[
                self.agents.dlpfc_agent(), self.agents.vmpfc_agent(), 
                self.agents.ofc_agent(), self.agents.acc_agent(), self.agents.mpfc_agent()
            ],
            tasks=[
                decision_task, emotional_task, reward_task, conflict_task, social_task
            ],
            manager_llm=llm,  # Use the previously defined Google AI LLM as the manager
            process=Process.hierarchical,  # Specifies the hierarchical management approach
            memory=True  # Enable memory usage for enhanced task execution
        )

        result = crew.kickoff()
        return result

if __name__ == "__main__":
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
        if response.lower() != 'yes':
            continue_analysis = False
