import logging
import os

from crewai import Crew, Process
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from openai import OpenAI, RateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from scan.openai_llm import OpenAIWrapper
from scan.scan_agents import PFCAgents
from scan.scan_tasks import PFCTasks

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='app.log'
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Get the OpenAI API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("The OPENAI_API_KEY environment variable is not set.")

client = OpenAI(api_key=openai_api_key)

# Define a prompt template for general queries
GENERAL_PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["query"],
    template="You are a helpful AI assistant. Provide a friendly and informative response to the following query: {query}"
)

class CustomCrew:
    def __init__(self, topic):
        self.topic = topic
        self.llm = OpenAIWrapper(api_key=openai_api_key)
        self.agents = PFCAgents(llm=self.llm.llm, topic=self.topic)
        self.tasks = PFCTasks(self.agents)

    def run(self, conversation_history):
        context = "\n".join(conversation_history[-5:])  # Use last 5 exchanges for context
        
        decision_task = self.tasks.complex_decision_making_task(self.topic, context)
        emotional_task = self.tasks.emotional_risk_assessment_task(self.topic, context)
        reward_task = self.tasks.reward_evaluation_task(self.topic, context)
        conflict_task = self.tasks.conflict_resolution_task(self.topic, context)
        social_task = self.tasks.social_cognition_task(self.topic, context)

        crew = Crew(
            agents=[
                self.agents.dlpfc_agent(context),
                self.agents.vmpfc_agent(context),
                self.agents.ofc_agent(context),
                self.agents.acc_agent(context),
                self.agents.mpfc_agent(context),
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

        results = _run_crew(crew)
        logger.info("Raw results: %s", results)

        final_output = self.combine_outputs(results)
        return final_output

    def combine_outputs(self, results):
        combined_output = f"""
        Based on our analysis of '{self.topic}', here are some recommendations and insights:

        {results}

        By following these suggestions, you can tackle the challenges related to '{self.topic}' more effectively and achieve your goals with greater focus and clarity.
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
        logger.error("Whoops! We hit a rate limit: %s", e)
        raise
    except Exception as e:
        logger.error(f"Oh no! Something unexpected happened: {e}")
        raise

def classify_input(input_text):
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an AI assistant. Classify the following input as either 'PFC task' or 'General task' based on its content. For PFC tasks, provide a brief description of which PFC functions might be involved (e.g., decision-making, emotional regulation, executive functions)."},
                {"role": "user", "content": f"Classify this input: {input_text}"}
            ],
            max_tokens=100,
            temperature=0.3
        )
        if response.choices and response.choices[0].message and response.choices[0].message.content:
            classification = response.choices[0].message.content.strip()
            return classification
        else:
            logger.error("Received an unexpected response format from OpenAI API.")
            return "General task"
    except Exception as e:
        logger.error("Error in classifying input: %s", e)
        return "General task"  # Default to general task if something goes wrong

def handle_input(user_input, conversation_history):
    try:
        classification = classify_input(user_input)
        if "PFC task" in classification:
            custom_crew = CustomCrew(topic=user_input)
            response = custom_crew.run(conversation_history)
        else:
            prompt = GENERAL_PROMPT_TEMPLATE.format(query=user_input)
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7
            ).choices[0].message.content.strip()
        if response is not None:
            conversation_history.append(f"Bot: {response}")
            return response
        else:
            logger.error("Hmm, looks like we didn't get a response. Let's try that again!")
            return "I'm sorry, I couldn't generate a response. Could you please rephrase your question?"
    except Exception as e:
        logger.exception(f"Whoops! Something unexpected happened: {e}")
        return "I encountered an error while processing your request. Let's give it another shot!"

def main() -> int:
    print("Hey there! Welcome to the SCAN System. How can I help you today?")
    print("---------------------------------------------------------------")
    conversation_history = []
    combined_output = ""
    
    while True:
        try:
            user_input = input("What would you like to know about or analyze? (Type 'exit' to end) ")
            if user_input.lower() == 'exit':
                break
            
            print(f"Got it! You're interested in: {user_input}")
            
            # Add the user's input to the conversation history
            conversation_history.append(f"User: {user_input}")
            
            response = handle_input(user_input, conversation_history)
            combined_output += "\n" + response + "\n"
            print("\n" + combined_output + "\n")
            
            while True:
                follow_up = input("Do you have a follow-up question? (yes/no): ").lower()
                if follow_up.startswith('y'):
                    follow_up_question = input("What's your follow-up question? ")
                    conversation_history.append(f"User: {follow_up_question}")
                    follow_up_response = handle_input(follow_up_question, conversation_history)
                    combined_output += "\n" + follow_up_response + "\n"
                    print("\n" + combined_output + "\n")
                else:
                    break
            
        except Exception as e:
            logger.exception(f"Oops! We hit an unexpected bump: %s", e)
            print("I ran into a little trouble there. Let's try again, shall we?")

    print("Thanks for using the SCAN System! I hope you found it helpful. Take care and come back anytime!")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
