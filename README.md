# SCAN Project
The SCAN (Synthetic Cognitive Augmentation Network) project aims to develop a collaborative multi-agent system that replicates the functions of the prefrontal cortex (PFC) using AI agents to enhance problem-solving, planning, decision-making, and other PFC-related activities.

## Features

- Simulates complex cognitive functions of the PFC
- Utilizes multiple specialized AI agents for different cognitive tasks
- Integrates generative AI for decision making and analysis
- Tracks task progress and updates details dynamically
- Provides detailed reports on cognitive task simulations

## Tech Stack

- **CrewAI**: Framework for orchestrating role-playing, autonomous AI agents
- **LangChain**: Framework for building language model applications
- **LLM Integrations**:
 - OpenAI GPT API (used in primary code but can be replaced with various LLMs)
- **Serper**: API for integrating search capabilities

## Solution

The SCAN project offers a multi-agent system designed to simulate the functions of the prefrontal cortex (PFC), enabling detailed analysis and decision-making simulations. By employing collaborative agents that emulate different components of the PFC, SCAN aims to enhance the capabilities of LLMs through customization. This system aids in understanding cognitive processes, developing advanced AI applications that mimic human cognitive functions, and assisting end users in making better decisions, solving problems, planning, and other PFC-related activities.

## Configuration

To configure the project, create a `.env` file in the root directory with the following variables:

    OpenAI API Key
    OPENAI_API_KEY=your_openai_api_key_here
    
    Serper API Key
    SERPER_API_KEY=your_serper_api_key_here

## Usage

1.  **Clone the repository**:
    
    `git clone https://github.com/iLevyTate/SCAN.git`

    `cd SCAN`
    
3.  **Install the required packages**:
    
    `poetry install` 
    
4.  **Set up your environment variables** in the `.env` file as shown above.
    
5.  **Run the main script**:
    
    `poetry run run-scan` 
    
## Contributors
Feel free to contribute by submitting issues or pull requests. For major changes, please open an issue first to discuss what you would like to change.
