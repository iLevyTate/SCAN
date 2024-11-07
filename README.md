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
- [Poetry Instructions](#poetry-instructions)

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

To configure the project, create a `.env` file in the root directory with the following variables:

```env
# OpenAI API Key
OPENAI_API_KEY=your_openai_api_key_here
```

## Usage

### Clone the Repository

```bash
git clone https://github.com/iLevyTate/SCAN.git
cd SCAN
```

### Install the Required Packages

```bash
poetry install
```

### Set up Environment Variables

Set up your environment variables in the `.env` file as shown above.

### (Optional) Export the API Key in the Terminal

If the API key from the `.env` file is not being picked up, you can export it directly in the terminal:

```bash
export OPENAI_API_KEY=your_openai_api_key_here
```

### Run the Main Script

```bash
poetry run run-scan
```

## Contributors

Feel free to contribute by submitting issues or pull requests. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- **CrewAI**: For providing the framework to create autonomous AI agents.
- **LangChain**: For the framework to build language model applications.
- **OpenAI**: For the language processing capabilities.
- **Contributors**: Thanks to all contributors who have helped in building and improving the SCAN project.

## Poetry Instructions

For the Poetry package manager:

- **Installation**: Install Poetry via the official instructions at [https://python-poetry.org/docs/#installation](https://python-poetry.org/docs/#installation).
- **Add Dependencies**: Use `poetry add <package-name>` to add dependencies.
- **Run Scripts**: Use `poetry run <command>` to run scripts defined in `pyproject.toml`.
