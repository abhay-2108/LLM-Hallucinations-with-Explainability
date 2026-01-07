from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List

# Manual imports
from crewai import LLM

llm = LLM(
        model="ollama/qwen3:8b",
        base_url="http://localhost:11434"
    )

@CrewBase
class LlmHallucination:
    """
    Multi-Agent Explainable AI Framework
    Agent-1: Hallucination-Prone Generator
    """

    agents: List[BaseAgent]
    tasks: List[Task]

    @agent
    def hallucinated_generator(self) -> Agent:
        from llm_hallucination.tools.qwen_trace_tool import QwenTraceTool
        return Agent(
            config=self.agents_config["hallucinated_generator"],
            tools=[QwenTraceTool()],
            llm=llm,
            allow_delegation=False
        )

    @task
    def hallucination_generation_task(self) -> Task:
        return Task(
            config=self.tasks_config["hallucination_generation_task"],
            agent=self.hallucinated_generator()
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            tracing = True
        )
