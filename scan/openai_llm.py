import os
from langchain_openai import ChatOpenAI

class OpenAIWrapper:
    def __init__(self, api_key):
        self.api_key = api_key
        os.environ["OPENAI_API_KEY"] = self.api_key
        os.environ["OPENAI_MODEL_NAME"] = "gpt-3.5-turbo"  # You can update the model name here if needed
        self.llm = ChatOpenAI(model="gpt-3.5-turbo")

    def generate(self, prompt, max_tokens=150):
        response = self.llm.generate(prompt=prompt, max_tokens=max_tokens)
        if response and response.choices:
            return response.choices[0].text.strip()
        else:
            raise ValueError("Received an unexpected response format from the LLM.")
