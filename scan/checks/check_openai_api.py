import os

from openai import OpenAI

from scan.config import settings
from scan.console import console

# Initialize the OpenAI client with the API key
client = OpenAI(api_key=settings.OPENAI_API_KEY)

# Get model name
model_name = os.getenv("TEST_MODEL", "gpt-3.5-turbo")

# Test OpenAI API call using the chat endpoint
try:
    response = client.chat.completions.create(
        model=model_name, messages=[{"role": "user", "content": "This is a test."}], max_tokens=5
    )
    console.print("OpenAI API call successful.")
    if response.choices and response.choices[0].message.content:
        console.print(response.choices[0].message.content.strip())
except Exception as e:
    console.print(f"OpenAI API call failed: {e}")
