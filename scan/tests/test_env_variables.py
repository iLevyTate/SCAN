# test_env_variables.py

import os
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Test OpenAI API Key
openai_api_key = os.getenv("OPENAI_API_KEY")
if openai_api_key:
    logger.info("OpenAI API Key Loaded Successfully.")
else:
    logger.error("OpenAI API Key NOT Loaded.")

# Test Model IDs
agent_models = {
    "DLPFC": os.getenv("DLPFC_MODEL", "gpt-4"),
    "VMPFC": os.getenv("VMPFC_MODEL", "gpt-4"),
    "OFC": os.getenv("OFC_MODEL", "gpt-4"),
    "ACC": os.getenv("ACC_MODEL", "gpt-4"),
    "MPFC": os.getenv("MPFC_MODEL", "gpt-4"),
}

for role, model in agent_models.items():
    if model != "gpt-4":
        logger.info(f"{role} Model Loaded: {model}")
    else:
        logger.warning(f"{role} Model NOT Loaded. Using default 'gpt-4'")