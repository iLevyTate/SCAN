from langchain_openai import ChatOpenAI
import os
import time
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import openai  # Ensure the openai library is correctly imported

class OpenAIWrapper:
    def __init__(self, api_key):
        self.api_key = api_key
        os.environ["OPENAI_API_KEY"] = self.api_key
        os.environ["OPENAI_MODEL_NAME"] = "gpt-3.5-turbo"  # Update the model name here
        self.llm = ChatOpenAI(model="gpt-3.5-turbo")

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, max=60), retry=retry_if_exception_type(openai.RateLimitError))
    def generate(self, prompt, max_tokens=150):
        response = self.llm.generate(prompt=prompt, max_tokens=max_tokens)
        return response.choices[0].text.strip()
