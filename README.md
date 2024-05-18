# SCAN Project

| Link                                      | Description |
| :---------------------------------------- | :---------- |
| [SCAN repo](https://github.com/iLevyTate/SCAN) | Codebase    |

---

The SCAN (Synthetic Cognitive Augmentation Network) project aims to develop a collaborative multi-agent system that replicates the functions of the prefrontal cortex (PFC) using AI agents to enhance problem-solving, planning, decision-making, and other PFC-related activities.

---

## Features

- Simulates complex cognitive functions of the PFC
- Utilizes multiple specialized AI agents for different cognitive tasks

---

## Tech Stack

- **CrewAI**: Framework for orchestrating role-playing, autonomous AI agents
- **LangChain**: Framework for building language model applications
- **LLM Integrations**:
  - OpenAI GPT API
  - Google Generative AI
  - Ollama API
- **Serper**: API for integrating search capabilities

---

## Configuration

To configure the project, create a `.env` file in the root directory with the following variables:

```env
# .env file

# OpenAI API Key
OPENAI_API_KEY=your_openai_api_key_here

# Google API Key (for Google Generative AI)
GOOGLE_API_KEY=your_google_api_key_here

# Ollama API Key
OLLAMA_API_KEY=your_ollama_api_key_here

# SERPER API Key
SERPER_API_KEY=your_serper_api_key_here

```

---

## Usage

1. **Clone the repository**:
   ```bash
   git clone https://github.com/iLevyTate/SCAN.git
   cd SCAN
   ```

2. **Install the required packages**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up your environment variables** in the `.env` file as shown above.

4. **Run the main script**:
   ```bash
   python main.py
   ```

---
