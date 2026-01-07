from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
from llm_hallucination.qwen_instrumented import agent1_trace

class QwenTraceToolInput(BaseModel):
    """Input schema for QwenTraceTool."""
    question: str = Field(..., description="The user question to answer and trace.")

class QwenTraceTool(BaseTool):
    name: str = "qwen_trace_tool"
    description: str = (
        "Generates a detailed answer using an instrumented Qwen model, "
        "extracting activation traces, attention maps, and reasoning analysis. "
        "Use this tool to answer the user's question."
    )
    args_schema: Type[BaseModel] = QwenTraceToolInput

    def _run(self, question: str) -> dict:
        try:
            return agent1_trace(question)
        except Exception as e:
            return {"error": str(e)}
