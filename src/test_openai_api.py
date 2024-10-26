import os
from dotenv import load_dotenv
import openai

# Load environment variables
load_dotenv()

# Set your OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set.")

openai.api_key = api_key

# Get model name
model_name = os.getenv("TEST_MODEL", "gpt-3.5-turbo")  # Ensure this is a valid chat model

# Test OpenAI API call using the chat endpoint
try:
    response = openai.ChatCompletion.create(
        model=model_name,  # Use the chat model
        messages=[{"role": "user", "content": "This is a test."}],  # Use messages format
        max_tokens=5
    )
    print("OpenAI API call successful.")
    print(response.choices[0].message.content.strip())  # Print the response text
except Exception as e:
    print(f"OpenAI API call failed: {e}")
