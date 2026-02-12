from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
from llm_hallucination.qwen_instrumented import agent1_trace

class QwenTraceToolInput(BaseModel):
    """Input schema for QwenTraceTool."""
    question: str = Field(..., description="The user question to answer and trace.")
    temperature: float = Field(0.9, description="The temperature for generation.")
    top_p: float = Field(0.9, description="The top_p for generation.")

class QwenTraceTool(BaseTool):
    name: str = "qwen_trace_tool"
    description: str = (
        "Generates a detailed answer using an instrumented Qwen model, "
        "extracting activation traces, attention maps, and reasoning analysis. "
        "Use this tool to answer the user's question."
    )
    args_schema: Type[BaseModel] = QwenTraceToolInput

    def _run(self, question: str, temperature: float = 0.9, top_p: float = 0.9) -> dict:
        try:
            return agent1_trace(question, temperature=temperature, top_p=top_p)
        except Exception as e:
            return {"error": str(e)}
