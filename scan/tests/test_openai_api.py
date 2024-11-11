import os

from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Set your OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set.")

# Initialize the OpenAI client with the API key
client = OpenAI(api_key=api_key)

# Get model name
model_name = os.getenv("TEST_MODEL", "gpt-3.5-turbo")

# Test OpenAI API call using the chat endpoint
try:
    response = client.chat.completions.create(
        model=model_name, messages=[{"role": "user", "content": "This is a test."}], max_tokens=5
    )
    print("OpenAI API call successful.")
    if response.choices and response.choices[0].message.content:
        print(response.choices[0].message.content.strip())
except Exception as e:
    print(f"OpenAI API call failed: {e}")
