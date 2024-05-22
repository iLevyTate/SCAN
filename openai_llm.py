from langchain_openai import ChatOpenAI
import os
import time

class OpenAIWrapper:
    def __init__(self, api_key):
        self.api_key = api_key
        os.environ["OPENAI_API_KEY"] = self.api_key
        os.environ["OPENAI_MODEL_NAME"] = "gpt-3.5-turbo"  # Update the model name here
        self.llm = ChatOpenAI(model="gpt-3.5-turbo")

    def generate(self, prompt, max_tokens=150, max_retries=5):
        for i in range(max_retries):
            try:
                response = self.llm.generate(prompt=prompt, max_tokens=max_tokens)
                return response.choices[0].text.strip()
            except openai.RateLimitError as e:
                wait_time = int(e.headers.get("Retry-After", 60))  # Use the Retry-After header if available
                print(f"Rate limit exceeded. Retrying after {wait_time} seconds.")
                time.sleep(wait_time)
            except openai.OpenAIError as e:
                print(f"OpenAI error: {e}")
                return None
        return None  # Return None if all
