# SCAN

[![DOI](https://zenodo.org/badge/802333022.svg)](https://doi.org/10.5281/zenodo.14052884)

![SCANGITHUBLOGO](https://github.com/user-attachments/assets/eca90256-2974-44e3-b7b3-e44798d2b482)

The SCAN (Synthetic Cognitive Augmentation Network) project aims to develop a collaborative multi-agent system that replicates the functions of the prefrontal cortex (PFC) using AI agents to enhance problem-solving, planning, decision-making, and other PFC-related activities.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Solution](#solution)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Clone the Repository](#clone-the-repository)
  - [Install the Required Packages](#install-the-required-packages)
  - [Set up Environment Variables](#set-up-environment-variables)
  - [Run the Main Script](#run-the-main-script)
- [Contributors](#contributors)
- [License](#license)
- [Acknowledgements](#acknowledgements)
- [uv Instructions](#uv-instruction)

## Overview

The SCAN project offers a multi-agent system designed to simulate the functions of the prefrontal cortex (PFC), enabling detailed analysis and decision-making simulations. By employing collaborative agents that emulate different components of the PFC, SCAN aims to enhance the capabilities of LLMs through customization. This system aids in understanding cognitive processes, developing advanced AI applications that mimic human cognitive functions, and assisting end users in making better decisions, solving problems, planning, and other PFC-related activities.

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

## Solution

The SCAN project offers a multi-agent system designed to simulate the functions of the prefrontal cortex (PFC), enabling detailed analysis and decision-making simulations. By employing collaborative agents that emulate different components of the PFC, SCAN aims to enhance the capabilities of LLMs through customization. This system aids in understanding cognitive processes, developing advanced AI applications that mimic human cognitive functions, and assisting end users in making better decisions, solving problems, planning, and other PFC-related activities.

## Configuration

To configure the project, create a `.env` file in the root directory. `OPENAI_API_KEY` is the
only required variable; see [`example.env`](example.env) for the full list.

```env
# Required
OPENAI_API_KEY=your_openai_api_key_here

# Optional: enables the web search tool. This is a SerpAPI (serpapi.com) key --
# serper.dev is a different service and its keys will not work here.
SERPAPI_API_KEY=your_serpapi_api_key_here
```

| Variable | Default | Purpose |
| --- | --- | --- |
| `OPENAI_API_KEY` | — | **Required.** OpenAI credentials. |
| `SERPAPI_API_KEY` | unset | Enables the search tool. Accepts the legacy name `SERPER_API_KEY`. |
| `DLPFC_MODEL`, `VMPFC_MODEL`, `OFC_MODEL`, `ACC_MODEL`, `MPFC_MODEL` | `gpt-4o` | Model per PFC agent. |
| `MAX_TOKENS` | `4000` | Maximum tokens per agent response. |
| `LOG_LEVEL` | `WARNING` | `DEBUG`, `INFO`, `WARNING`, `ERROR` or `CRITICAL`. |
| `DISABLE_TELEMETRY` | `true` | crewai ships anonymous traces to `telemetry.crewai.com`; disabled by default. |

Unrecognised variables in `.env` are ignored, so SCAN can share a `.env` with other tools.

## Usage

### Clone the Repository

```bash
git clone https://github.com/iLevyTate/SCAN.git
cd SCAN
```

### Install the Required Packages

```bash
uv sync --frozen
```

### Set up Environment Variables

Set up your environment variables in the `.env` file as shown above. Real environment variables
take precedence over `.env`, so you can also export them directly:

```bash
export OPENAI_API_KEY=your_openai_api_key_here
```

### Run the Main Script

```bash
uv run run-scan
```

`run-scan` exits non-zero if the run fails (`1` on error, `2` on missing input, `130` if
interrupted), so it is safe to use in scripts.

## Contributors

Feel free to contribute by submitting issues or pull requests. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- **CrewAI**: For providing the framework to create autonomous AI agents.
- **LangChain**: For the framework to build language model applications.
- **OpenAI**: For the language processing capabilities.
- **Contributors**: Thanks to all contributors who have helped in building and improving the SCAN project.

## uv Instructions

For the uv package manager:

- **Installation**: Install uv via the official instructions at [https://docs.astral.sh/uv/getting-started/installation/](https://docs.astral.sh/uv/getting-started/installation/).
- **Add Dependencies**: Use `uv add <package-name>` to add dependencies.
- **Run Scripts**: Use `uv run <command>` to run scripts defined in `pyproject.toml`.
