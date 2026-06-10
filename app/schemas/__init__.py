"""Schemas for the QA agent API and internal state."""

import operator
from typing import Annotated, Any, Dict, List, Optional

from pydantic import BaseModel, Field

from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class ChatRequest(BaseModel):
    """Request body for the chat endpoint."""

    message: str = Field(..., min_length=1, description="The user's question or message")


class Source(BaseModel):
    """A web source cited in an answer."""

    name: str = Field(..., description="Short display name of the source, e.g. 'BBC News' or 'Wikipedia'")
    url: str = Field(..., description="Full URL of the source, e.g. 'https://www.bbc.com/news/article-123'")


class QAResponse(BaseModel):
    """Structured output for the QA agent's final answer with separated answer text and source citations."""

    answer: str = Field(
        ...,
        description="The plain-text answer to the user's question. Do NOT include URLs, markdown links, or a sources section here. Only the factual response text.",
    )
    sources: List[Source] = Field(
        default_factory=list,
        description="Every external source referenced in the answer. Each entry must have a 'name' (short display name) and 'url' (full URL). Extract all URLs mentioned in the answer into this list.",
    )


class ChatResponse(BaseModel):
    """Response from the chat endpoint."""

    answer: str = Field(..., description="The agent's answer to the user's question")
    sources: List[Source] = Field(default_factory=list, description="Sources cited in the answer")
    input_tokens: int = Field(default=0, description="Total input tokens consumed")
    output_tokens: int = Field(default=0, description="Total output tokens consumed")
    total_tokens: int = Field(default=0, description="Total tokens consumed")
    total_duration_ms: float = Field(default=0, description="Total wall-clock duration in milliseconds")


class ChatStreamChunk(BaseModel):
    """A single chunk in the SSE stream."""

    type: str = Field(..., description="Chunk type: token, tool_call, metadata, error, or done")
    content: str = Field(default="", description="The text content of this chunk")
    tool_name: Optional[str] = Field(None, description="Name of the tool being called (for tool_call type)")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata (for metadata type)")


class StepTiming(BaseModel):
    """Timing record for a single agent step."""

    step: str = Field(..., description="Name of the graph node that executed")
    duration_ms: float = Field(..., description="Wall-clock duration in milliseconds")


class AgentState(BaseModel):
    """Internal state for the QA agent graph."""

    messages: Annotated[List[BaseMessage], add_messages] = Field(default_factory=list)
    tool_call_count: Annotated[int, operator.add] = Field(default=0)
    current_query: str = Field(default="")
    input_tokens: Annotated[int, operator.add] = Field(default=0)
    output_tokens: Annotated[int, operator.add] = Field(default=0)
    total_tokens: Annotated[int, operator.add] = Field(default=0)
    agent_model: str = Field(default="")
    step_timings: Annotated[List[Dict[str, Any]], operator.add] = Field(default_factory=list)


class AgentInputState(BaseModel):
    """Input state for the QA agent graph."""

    messages: List[BaseMessage] = Field(default_factory=list)
    current_query: str = Field(default="")


class AgentOutputState(BaseModel):
    """Output state for the QA agent graph."""

    messages: List[BaseMessage] = Field(default_factory=list)