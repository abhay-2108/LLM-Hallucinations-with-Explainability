# LLM Hallucination 

Welcome to the LLM Hallucination project, powered by [crewAI](https://crewai.com). This template is designed to help you set up a multi-agent AI system with ease, leveraging the powerful and flexible framework provided by crewAI. Our goal is to enable your agents to collaborate effectively on complex tasks, maximizing their collective intelligence and capabilities.

## Installation

Ensure you have Python >=3.10 <3.14 installed on your system. This project uses [UV](https://docs.astral.sh/uv/) for dependency management and package handling, offering a seamless setup and execution experience.

First, if you haven't already, install uv:

```bash
pip install -r requirements.txt
```

### Customizing

**Add your `OPENAI_API_KEY` into the `.env` file**

- Modify `src/llm_hallucination/config/agents.yaml` to define your agents
- Modify `src/llm_hallucination/config/tasks.yaml` to define your tasks
- Modify `src/llm_hallucination/crew.py` to add your own logic, tools and specific args
- Modify `src/llm_hallucination/main.py` to add custom inputs for your agents and tasks

## Running the Project

To kickstart your crew of AI agents and begin task execution, run this from the root folder of your project:

```bash
python src/llm_hallucination/main.py
```

This command initializes the llm_hallucination Crew, allowing you to **interactively enter a question/topic**. The agents will assemble and execute the task.

### Features
- **Interactive Input**: You will be prompted to enter a question at runtime.
- **Trace Storage**: Execution traces (including model activations and analysis) are automatically saved to the `src/runs/` directory as JSON files.

This example, unmodified, will run the analysis on your input topic and save the detailed trace.

## Understanding Your Crew

The llm_hallucination Crew is composed of multiple AI agents, each with unique roles, goals, and tools. These agents collaborate on a series of tasks, defined in `config/tasks.yaml`, leveraging their collective skills to achieve complex objectives. The `config/agents.yaml` file outlines the capabilities and configurations of each agent in your crew.

